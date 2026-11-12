"""End-to-end checks of the CuTe launcher on real launches: the metadata-guarded
fast relaunch, and the CUDA context of a thread that launches."""

from __future__ import annotations

import os
import tempfile
from typing import Any
from typing import cast
import unittest
from unittest.mock import patch

# pyrefly: ignore [missing-import]
import torch

import helion
from helion._testing import DEVICE
from helion._testing import RefEagerTestDisabled
from helion._testing import TestCase
from helion._testing import onlyBackends
from helion._testing.cuda_context import clear_current_context
from helion._testing.cuda_context import current_context
from helion._testing.cuda_context import on_new_thread
import helion.language as hl
from helion.runtime.cute import launcher as cute_launcher
from helion.runtime.cute.compile_cache import CompiledCuteLauncher
from helion.runtime.cute.compile_cache import is_tvm_ffi_compiled_function
from helion.runtime.cute.fast_relaunch import FASTPATH_MISS


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
            self.assertTrue(is_tvm_ffi_compiled_function(compiled._compiled))
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
                FASTPATH_MISS,
            )

    def test_host_code_runs_with_a_cuda_context_on_a_new_thread(self) -> None:
        """Launches that set function attributes need a current context; the
        first launch on a new thread, here of a module reloaded from the disk
        cache, runs its host code with one."""
        add_pair = _make_add_pair()
        x = torch.randn(256, device=DEVICE)
        y = torch.randn_like(x)
        contexts: list[int] = []
        call = CompiledCuteLauncher.__call__

        def recording_call(launcher: CompiledCuteLauncher, *args: object) -> object:
            contexts.append(current_context())
            return call(launcher, *args)

        with (
            tempfile.TemporaryDirectory() as cache_dir,
            patch.dict(os.environ, {"CUTE_DSL_CACHE_DIR": cache_dir}),
        ):
            torch.testing.assert_close(add_pair(x, y), x + y)
            # Forget the compiled launcher, the fast relaunch and the last
            # launch: the next launch reloads the persisted module and runs
            # its host code, as in a later process.
            cute_kernel = cast("Any", _cute_kernel(add_pair.bind((x, y))))
            cute_kernel._helion_cute_compiled_launchers.clear()
            cute_kernel._helion_cute_fastpath = None
            cute_kernel._helion_cute_last_launch_cache = None
            with patch.object(CompiledCuteLauncher, "__call__", recording_call):
                start, out = on_new_thread(lambda: (current_context(), add_pair(x, y)))
        self.assertEqual(start, 0)
        self.assertEqual(len(contexts), 1)
        self.assertNotEqual(contexts[0], 0)
        torch.testing.assert_close(out, x + y)

    def test_fast_relaunch_finds_a_cuda_context_on_every_launch(self) -> None:
        """A kernel launched on the main thread and relaunched on a new thread
        (a backward on the autograd device thread), which starts without a
        current context, finds the device's primary context current before
        the fast relaunch runs, and again on the thread's next launch after
        other code cleared its context (a check once per thread would skip
        it, and a launch that sets function attributes would abort)."""
        add_pair = _make_add_pair()
        x = torch.randn(256, device=DEVICE)
        y = torch.randn_like(x)
        for _ in range(2):
            torch.testing.assert_close(add_pair(x, y), x + y)
        relaunch_type = cute_launcher._CuteFastRelaunch
        self.assertIsInstance(
            cast("Any", _cute_kernel(add_pair.bind((x, y))))._helion_cute_fastpath,
            relaunch_type,
        )
        seen: list[tuple[int, bool]] = []
        try_launch_raw = relaunch_type.try_launch_raw

        def recording_try_launch_raw(
            relaunch: object, *args: object
        ) -> tuple[bool, object]:
            context = current_context()
            hit, result = try_launch_raw(cast("Any", relaunch), *args)
            seen.append((context, hit))
            return hit, result

        def launch_clear_launch() -> tuple[int, torch.Tensor, int, torch.Tensor]:
            start = current_context()
            first = add_pair(x, y)
            clear_current_context()
            between = current_context()
            return start, first, between, add_pair(x, y)

        with patch.object(relaunch_type, "try_launch_raw", recording_try_launch_raw):
            start, first, between, second = on_new_thread(launch_clear_launch)
        self.assertEqual((start, between), (0, 0))
        self.assertEqual(len(seen), 2)
        self.assertTrue(all(hit and context for context, hit in seen), seen)
        torch.testing.assert_close(first, x + y)
        torch.testing.assert_close(second, x + y)

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
        self.assertTrue(is_tvm_ffi_compiled_function(compiled._compiled))
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
