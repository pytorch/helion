"""End-to-end checks of the CuTe metadata-guarded fast relaunch on real launches."""

from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import patch

import torch

import helion
from helion._testing import DEVICE
from helion._testing import RefEagerTestDisabled
from helion._testing import TestCase
from helion._testing import onlyBackends
import helion.language as hl
from helion.runtime.cute import launcher as cute_launcher


def _make_add_pair() -> helion.Kernel:
    @helion.kernel(static_shapes=True, config=helion.Config(block_sizes=[64]))
    def add_pair(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        for tile in hl.tile(x.size(0)):
            out[tile] = x[tile] + y[tile]
        return out

    return add_pair


def _cute_kernel(bound: helion.runtime.kernel.BoundKernel) -> object:
    """The ``@cute.kernel`` object of a compiled bound kernel."""
    run = bound._run
    assert run is not None
    return run.__globals__[f"_helion_{bound.kernel.name}"]


@onlyBackends(["cute"])
class TestCuteFastRelaunch(RefEagerTestDisabled, TestCase):
    def test_fresh_compile_builds_packed_fast_relaunch(self) -> None:
        """A fresh ``cute.compile`` yields a TVM-FFI function; the relaunch
        must still build (through the lazily JIT-ed C-API twin) and launch
        fresh allocations through the pre-packed argument array."""
        add_pair = _make_add_pair()
        x = torch.randn(256, device=DEVICE)
        y = torch.randn_like(x)
        with (
            tempfile.TemporaryDirectory() as cache_dir,
            patch.dict(os.environ, {"CUTE_DSL_CACHE_DIR": cache_dir}),
        ):
            torch.testing.assert_close(add_pair(x, y), x + y)
            cute_kernel = _cute_kernel(add_pair.bind((x, y)))
            launchers = list(cute_kernel._helion_cute_compiled_launchers.values())  # type: ignore[attr-defined]
            self.assertEqual(len(launchers), 1)
            compiled = launchers[0]
            self.assertTrue(
                cute_launcher._is_tvm_ffi_compiled_function(compiled._compiled)
            )
            fastpath = cute_kernel._helion_cute_fastpath  # type: ignore[attr-defined]
            self.assertIsInstance(fastpath, cute_launcher._CuteFastRelaunch)
            self.assertIsNotNone(fastpath.packed_args)
            twin = compiled.capi_function()
            self.assertIsNot(twin, compiled._compiled)
            self.assertIs(compiled.capi_function(), twin)
            self.assertIs(
                fastpath.executor.jit_module.execution_args, twin.execution_args
            )
            with patch.object(
                fastpath.executor,
                "run_compiled_program",
                side_effect=AssertionError("packed relaunch fell back"),
            ):
                for _ in range(3):
                    a = torch.randn(256, device=DEVICE)
                    b = torch.randn_like(a)
                    torch.testing.assert_close(add_pair(a, b), a + b)
                stream = torch.cuda.Stream()
                with torch.cuda.stream(stream):
                    a = torch.randn(256, device=DEVICE)
                    b = torch.randn_like(a)
                    out = add_pair(a, b)
                stream.synchronize()
                torch.testing.assert_close(out, a + b)
                self.assertEqual(fastpath.last_raw, stream.cuda_stream)
            # Mismatched metadata misses the guard before any slot is patched.
            self.assertEqual(
                fastpath.try_launch(
                    (x[::2], y[::2], torch.empty_like(x[::2])),
                    fastpath.grid,
                    fastpath.block,
                    fastpath.compile_options,
                ),
                cute_launcher._CUTE_FASTPATH_MISS,
            )

    def test_fast_relaunch_under_cuda_graph_capture(self) -> None:
        add_pair = _make_add_pair()
        x = torch.randn(256, device=DEVICE)
        y = torch.randn_like(x)
        torch.testing.assert_close(add_pair(x, y), x + y)
        fastpath = _cute_kernel(add_pair.bind((x, y)))._helion_cute_fastpath  # type: ignore[attr-defined]
        self.assertIsInstance(fastpath, cute_launcher._CuteFastRelaunch)
        static_x = torch.randn(256, device=DEVICE)
        static_y = torch.randn_like(static_x)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                add_pair(static_x, static_y)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with (
            patch.object(
                fastpath.executor,
                "run_compiled_program",
                side_effect=AssertionError("packed relaunch fell back"),
            ),
            torch.cuda.graph(graph),
        ):
            static_out = add_pair(static_x, static_y)
        static_x.copy_(torch.randn(256, device=DEVICE))
        static_y.copy_(torch.randn(256, device=DEVICE))
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(static_out, static_x + static_y)

    def test_first_launch_inside_capture_defers_the_relaunch_build(self) -> None:
        """A fresh compile's first launch inside CUDA-graph capture neither
        JITs the C-API twin nor builds the relaunch; the captured marshalled
        launch replays, and the next eager launch builds both."""
        add_pair = _make_add_pair()
        x = torch.randn(256, device=DEVICE)
        y = torch.randn_like(x)
        # A fresh cache dir keeps this a fresh TVM-FFI compile even when an
        # earlier test in the same process compiled an identical kernel (a
        # disk-cache reload needs no twin, which the assertions below pin).
        with (
            tempfile.TemporaryDirectory() as cache_dir,
            patch.dict(os.environ, {"CUTE_DSL_CACHE_DIR": cache_dir}),
        ):
            torch.testing.assert_close(add_pair(x, y), x + y)
        cute_kernel = _cute_kernel(add_pair.bind((x, y)))
        (compiled,) = cute_kernel._helion_cute_compiled_launchers.values()  # type: ignore[attr-defined]
        self.assertTrue(cute_launcher._is_tvm_ffi_compiled_function(compiled._compiled))
        self.assertIsInstance(
            cute_kernel._helion_cute_fastpath,  # type: ignore[attr-defined]
            cute_launcher._CuteFastRelaunch,
        )
        # Rewind to the state right after a fresh compile: no twin, no relaunch.
        compiled._capi_compiled = None
        cute_kernel._helion_cute_fastpath = None  # type: ignore[attr-defined]
        # Fresh allocations miss the pointer-keyed launch cache, so the capture
        # takes the build path that would otherwise construct the relaunch.
        static_x = torch.randn(256, device=DEVICE)
        static_y = torch.randn_like(static_x)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=side):
            static_out = add_pair(static_x, static_y)
        self.assertIsNone(compiled._capi_compiled)
        self.assertIsNone(cute_kernel._helion_cute_fastpath)  # type: ignore[attr-defined]
        static_x.copy_(torch.randn(256, device=DEVICE))
        static_y.copy_(torch.randn(256, device=DEVICE))
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(static_out, static_x + static_y)
        a = torch.randn(256, device=DEVICE)
        b = torch.randn_like(a)
        torch.testing.assert_close(add_pair(a, b), a + b)
        self.assertIsInstance(
            cute_kernel._helion_cute_fastpath,  # type: ignore[attr-defined]
            cute_launcher._CuteFastRelaunch,
        )
        self.assertIsNotNone(compiled._capi_compiled)


if __name__ == "__main__":
    unittest.main()
