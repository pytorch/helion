from __future__ import annotations

import operator
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from test.test_generated_code_binding import _runtime_scale

import helion
from helion._testing import TestCase
from helion._testing import skipIfRefEager
from helion.autotuner.base_search import BaseAutotuner
import helion.language as hl
from helion.runtime import generated_code_cache as cache
from helion.runtime.cached_kernel import _CachedBoundKernel
from helion.runtime.kernel import Kernel
from helion.runtime.kernel import KernelCompiler
from helion.runtime.kernel import PyCodeCache

_CONFIG = helion.Config(block_sizes=[32])


def _specialized_rows(x: torch.Tensor, scale: int) -> torch.Tensor:
    hl.specialize(x.size(1))
    scale = hl.specialize(scale)
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile, :] = x[tile, :] * scale
    return out


def _add_one_2d(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile_m, tile_n in hl.tile(x.size()):
        out[tile_m, tile_n] = x[tile_m, tile_n] + 1
    return out


class _PresetSearch(BaseAutotuner):
    def autotune(self, *, skip_cache: bool = False) -> helion.Config:
        return helion.Config(block_sizes=[16, 32], indexing="pointer")


def _select(bound: object, args: object) -> _PresetSearch:
    return _PresetSearch()


def _dynamic_2d_kernel(**settings: object) -> Kernel:
    return helion.kernel(
        backend="triton",
        static_shapes=False,
        generated_code_cache=True,
        autotuner_fn=_select,
        **settings,
    )(_add_one_2d)


@skipIfRefEager("generated source caching requires frontend compilation")
class TestGeneratedCodeDispatch(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.root = Path(self._test_stack.enter_context(tempfile.TemporaryDirectory()))
        self._test_stack.enter_context(
            patch.dict(
                "os.environ",
                {"HELION_CACHE_DIR": str(self.root), "HELION_SKIP_CACHE": "0"},
            )
        )
        for name in ("helion_key", "torch_key_wrapper", "triton_key_wrapper"):
            self._test_stack.enter_context(patch.object(cache, name, return_value=name))
        self._test_stack.enter_context(
            patch.object(cache, "supports_torch_compile_fusion", return_value=True)
        )
        self.loads = 0
        self._test_stack.enter_context(
            patch.object(PyCodeCache, "load", side_effect=self._load)
        )

    def _load(self, source: str, *, extra: str = "") -> SimpleNamespace:
        self.loads += 1
        path = self.root / f"module_{self.loads}.py"
        path.write_text(source)
        # The CPU tests cover dispatch; the CUDA test launches real source.
        return SimpleNamespace(
            __file__=str(path),
            _runtime_scale=operator.mul,
            _specialized_rows=operator.mul,
        )

    def _kernel(self, fn: object = _runtime_scale, **settings: object) -> Kernel:
        settings.setdefault("config", _CONFIG)
        settings.setdefault("generated_code_cache", True)
        return helion.kernel(backend="triton", static_shapes=False, **settings)(fn)

    def _frontend_count(self):
        return patch.object(
            KernelCompiler, "compile", autospec=True, side_effect=KernelCompiler.compile
        )

    def test_cache_hit_publishes_fast_dispatch(self) -> None:
        x = torch.ones(16)
        self._kernel()(x, 2.0)
        kernel = self._kernel()
        with patch.object(
            KernelCompiler, "compile", side_effect=AssertionError("frontend ran")
        ):
            torch.testing.assert_close(kernel(x, 2.0), x * 2)
        runner = kernel.bind((x, 2.0))
        self.assertIsInstance(runner, _CachedBoundKernel)
        prepared = kernel._prepared_call
        self.assertIsNotNone(prepared)
        assert prepared is not None
        self.assertIs(prepared.bound, runner)
        # Steady-state launches use the prepared call, as native bindings do:
        # no binding, dispatch-key construction or runner facade per launch.
        with (
            patch.object(Kernel, "_bind", side_effect=AssertionError("bind")),
            patch.object(
                Kernel,
                "_fast_dispatch_key_and_guards",
                side_effect=AssertionError("dispatch key"),
            ),
            patch.object(
                _CachedBoundKernel, "__call__", side_effect=AssertionError("facade")
            ),
        ):
            for scale in (2.0, 3.0, 4.0):
                torch.testing.assert_close(kernel(x, scale), x * scale)
            torch.testing.assert_close(prepared.bound._run(x, 5.0), x * 5)
        # Direct calls may validate once, then reuse their own guard.
        torch.testing.assert_close(runner(x, 6.0), x * 6)
        with patch.object(Kernel, "_bind", side_effect=AssertionError("bind")):
            torch.testing.assert_close(runner(x, 7.0), x * 7)

    def test_specialized_runner_uses_prepared_guards(self) -> None:
        x = torch.ones(8, 32)
        self._kernel(_specialized_rows)(x, 2)
        kernel = self._kernel(_specialized_rows)
        with patch.object(
            KernelCompiler, "compile", side_effect=AssertionError("frontend ran")
        ):
            torch.testing.assert_close(kernel(x, 2), x * 2)
            torch.testing.assert_close(kernel(x, 2), x * 2)
        runner = kernel.bind((x, 2))
        self.assertIsInstance(runner, _CachedBoundKernel)
        with (
            patch.object(Kernel, "_bind", side_effect=AssertionError("bind")),
            patch.object(
                _CachedBoundKernel, "__call__", side_effect=AssertionError("facade")
            ),
        ):
            torch.testing.assert_close(kernel(x, 2), x * 2)
        # Direct calls revalidate value guards once, then use their own guard.
        torch.testing.assert_close(runner(x, 2), x * 2)
        with patch.object(Kernel, "_bind", side_effect=AssertionError("bind")):
            torch.testing.assert_close(runner(x, 2), x * 2)

    def test_warm_dynamic_binding_serves_unseen_shapes(self) -> None:
        self._kernel()(torch.ones(16), 2.0)
        kernel = self._kernel()
        with patch.object(
            KernelCompiler, "compile", side_effect=AssertionError("frontend ran")
        ):
            for size in (16, 64, 1000, 3):
                y = torch.ones(size)
                torch.testing.assert_close(kernel(y, 2.0), y * 2)
        self.assertEqual(
            len({id(bound) for bound in kernel._bound_kernels.values()}), 1
        )

    def test_saved_guards_select_bindings_before_frontend(self) -> None:
        self._kernel(_specialized_rows)(torch.ones(8, 32), 2)
        kernel = self._kernel(_specialized_rows)
        with self._frontend_count() as compile_fn:
            # A different dynamic row count matches the saved guards.
            y = torch.ones(100, 32)
            torch.testing.assert_close(kernel(y, 2), y * 2)
            self.assertEqual(compile_fn.call_count, 0)
            # New specialized values miss the saved guards and compile once.
            y = torch.ones(8, 64)
            torch.testing.assert_close(kernel(y, 2), y * 2)
            torch.testing.assert_close(
                kernel(torch.ones(4, 64), 2), torch.ones(4, 64) * 2
            )
            self.assertEqual(compile_fn.call_count, 1)
            torch.testing.assert_close(kernel(y, 3), y * 3)
            self.assertEqual(compile_fn.call_count, 2)
        # Each specialization saved its own binding for later processes.
        warm = self._kernel(_specialized_rows)
        with patch.object(
            KernelCompiler, "compile", side_effect=AssertionError("frontend ran")
        ):
            for shape, scale in (((7, 32), 2), ((9, 64), 2), ((5, 64), 3)):
                y = torch.ones(shape)
                torch.testing.assert_close(warm(y, scale), y * scale)

    def test_runner_guards_match_frontend_extractors(self) -> None:
        x = torch.ones(8, 32)
        self._kernel(_specialized_rows)(x, 2)
        runner = self._kernel(_specialized_rows).bind((x, 2))
        self.assertIsInstance(runner, _CachedBoundKernel)
        native = self._kernel(_specialized_rows, generated_code_cache=False).bind(
            (x, 2)
        )
        runner_extractors = runner._specialize_extra()
        native_extractors = native._specialize_extra()
        self.assertEqual(
            [type(extractor) for extractor in runner_extractors],
            [type(extractor) for extractor in native_extractors],
        )
        for args in ((x, 2), (torch.ones(3, 64), 5), (torch.ones(8, 32)[:, ::2], 2)):
            self.assertEqual(
                [extractor(args) for extractor in runner_extractors],
                [extractor(args) for extractor in native_extractors],
            )

    def test_unknown_attributes_do_not_compile(self) -> None:
        x = torch.ones(16)
        self._kernel()(x, 2.0)
        runner = self._kernel().bind((x, 2.0))
        with self._frontend_count() as compile_fn:
            with self.assertRaises(AttributeError):
                runner.not_a_binding_attribute  # noqa: B018
            self.assertFalse(hasattr(runner, "_matches_inputs"))
            self.assertEqual(compile_fn.call_count, 0)
            with self.assertLogs("helion.runtime.cached_kernel", level="DEBUG") as logs:
                self.assertIsNotNone(runner.config_spec)
            self.assertEqual(compile_fn.call_count, 1)
        self.assertIn("config_spec requires the Helion frontend", logs.output[0])

    def test_source_key_ignores_shapes_within_dynamic_bucket(self) -> None:
        def key(x: torch.Tensor) -> str | None:
            bound = self._kernel().bind((x, 2.0))
            return cache.generated_code_cache_key(
                bound, bound._normalized_config_copy(_CONFIG)
            )

        self.assertIsNotNone(key(torch.ones(16)))
        self.assertEqual(key(torch.ones(16)), key(torch.ones(64)[::2]))
        self.assertNotEqual(
            key(torch.ones(16)), key(torch.ones(16, dtype=torch.float64))
        )


@skipIfRefEager("generated source caching requires frontend compilation")
class TestGeneratedCodeDispatchCUDA(TestCase):
    @unittest.skipIf(not torch.cuda.is_available(), reason="requires CUDA and Triton")
    def test_warm_process_serves_unseen_shapes_without_frontend(self) -> None:
        root = self._test_stack.enter_context(tempfile.TemporaryDirectory())
        self._test_stack.enter_context(
            patch.dict(os.environ, {"HELION_CACHE_DIR": root, "HELION_SKIP_CACHE": "0"})
        )
        # A custom selector keeps every descriptor guard active, so each
        # power-of-two row class is its own binding, as without the cache.
        kernel = _dynamic_2d_kernel()
        for m in (4, 16):
            x = torch.randn(m, 64, device=torch.device("cuda"))
            torch.testing.assert_close(kernel(x), x + 1)
        code = """
import importlib.util
import sys
from unittest.mock import patch
import torch

spec = importlib.util.spec_from_file_location(sys.argv[2], sys.argv[1])
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
kernel = module._dynamic_2d_kernel()
with patch.object(module.KernelCompiler, "compile", side_effect=AssertionError("frontend ran")):
    # Unseen row counts within the saved descriptor extent classes.
    for m in (5, 7, 17, 31):
        x = torch.randn(m, 64, device=torch.device("cuda"))
        torch.testing.assert_close(kernel(x), x + 1)
        torch.testing.assert_close(kernel(x), x + 1)
        assert kernel._prepared_call is not None
        assert isinstance(kernel._prepared_call.bound, module._CachedBoundKernel)
assert len(kernel._bound_kernels) == 2
bound = kernel.bind((x,))
torch.testing.assert_close(bound(x), x + 1)
with patch.object(module.Kernel, "_bind", side_effect=AssertionError("bind")):
    torch.testing.assert_close(kernel(x), x + 1)
    torch.testing.assert_close(bound(x), x + 1)
"""
        subprocess.run(
            [sys.executable, "-c", code, __file__, _add_one_2d.__module__],
            check=True,
            timeout=60,
        )


if __name__ == "__main__":
    unittest.main()
