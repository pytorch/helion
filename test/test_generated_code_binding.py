from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
from unittest.mock import patch

import torch

from test.test_generated_code_cache import _bound
from test.test_generated_code_cache import _source_loader

import helion
from helion._testing import TestCase
from helion._testing import skipIfRefEager
import helion.language as hl
from helion.runtime import generated_code_cache as cache
from helion.runtime.kernel import BoundKernel
from helion.runtime.kernel import Kernel
from helion.runtime.kernel import KernelCompiler
from helion.runtime.kernel import PyCodeCache


def _runtime_scale(x: torch.Tensor, scale: float) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] * scale
    return out


@skipIfRefEager("generated source caching requires frontend compilation")
class _GeneratedCodeBindingTestCase(TestCase):
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
        self.sources: list[str] = []
        self._test_stack.enter_context(
            patch.object(
                PyCodeCache, "load", side_effect=_source_loader(self.root, self.sources)
            )
        )


class TestGeneratedCodeBinding(_GeneratedCodeBindingTestCase):
    def test_dynamic_bindings_and_runtime_keys_match_cache_disabled(self) -> None:
        x, y = torch.ones(16), torch.ones(64)
        kernels = [
            helion.kernel(
                backend="triton",
                static_shapes=False,
                config=helion.Config(block_sizes=[32]),
                disable_autotuner_heuristics=True,
                generated_code_cache=enabled,
            )(_runtime_scale)
            for enabled in (False, True)
        ]
        keys = []
        for kernel in kernels:
            first = kernel.bind((x, 1.0))
            self.assertIs(first, kernel.bind((y, 3.0)))
            first.set_config(helion.Config(block_sizes=[32]))
            keys.append(
                kernel._create_bound_kernel_cache_key(
                    first, (y, 3.0), kernel._base_specialization_key((y, 3.0))
                )
            )
            with patch.object(
                cache, "_dependency_key", side_effect=AssertionError("hot-path hashing")
            ):
                torch.testing.assert_close(kernel(y, 3.0), y * 3)
                torch.testing.assert_close(kernel(y, 4.0), y * 4)
        self.assertEqual(keys[0], keys[1])

    def test_warm_runner_rebinds_specialized_scalars_without_stale_source(self) -> None:
        x = torch.ones(16)
        config = helion.Config(block_sizes=[32])
        _bound(x, static_shapes=False).set_config(config)
        warm = _bound(x, static_shapes=False)
        warm.set_config(config)
        original = self.sources[-1]
        with patch.object(
            KernelCompiler, "compile", autospec=True, side_effect=KernelCompiler.compile
        ) as compile_fn:
            warm.kernel(x, 2)
            warm.kernel(x, 3)
            changed = warm.kernel.bind((x, 3))
            self.assertIs(changed, warm.kernel.bind((torch.ones(64), 3)))
        self.assertEqual(compile_fn.call_count, 1)
        self.assertNotEqual(original, self.sources[-1])
        self.assertIsNot(changed, warm)

    def test_failed_materialization_keeps_runner_and_retry_publishes_once(self) -> None:
        x = torch.ones(16)
        config = helion.Config(block_sizes=[32])
        other = helion.Config(block_sizes=[64])
        _bound(x).set_config(config)
        warm = _bound(x)
        warm.set_config(config)
        with (
            patch.object(
                KernelCompiler, "compile", side_effect=RuntimeError("frontend failed")
            ),
            self.assertRaisesRegex(RuntimeError, "frontend failed"),
        ):
            warm.set_config(other)
        self.assertIsNone(warm(x, 2))
        with (
            patch.object(
                KernelCompiler,
                "compile",
                autospec=True,
                side_effect=KernelCompiler.compile,
            ) as compile_fn,
            ThreadPoolExecutor(max_workers=2) as pool,
        ):
            sources = list(pool.map(lambda _: warm.to_code(other), range(2)))
        self.assertEqual(compile_fn.call_count, 1)
        self.assertEqual(sources[0], sources[1])
        self.assertIsNot(warm.kernel.bind((x, 2)), warm)


class TestGeneratedCodeLegacyCapture(_GeneratedCodeBindingTestCase):
    def _kernel(self) -> Kernel:
        return helion.kernel(
            backend="triton",
            static_shapes=True,
            config=helion.Config(block_sizes=[32]),
            generated_code_cache=True,
            torch_compile_fusion=False,
        )(_runtime_scale)

    def test_supported_integration_keeps_frontend_reuse_with_fusion_disabled(
        self,
    ) -> None:
        x = torch.ones(16)
        self._kernel().bind((x, 2.0)).set_config(helion.Config(block_sizes=[32]))
        with patch.object(
            KernelCompiler, "compile", side_effect=AssertionError("frontend ran")
        ):
            warm = self._kernel().bind((x, 2.0))
            self.assertNotIsInstance(warm, BoundKernel)
            torch.testing.assert_close(warm(x, 2.0), x * 2)

    def test_legacy_integration_reuses_source_with_native_dispatch(self) -> None:
        x = torch.ones(16)
        with patch.object(cache, "supports_torch_compile_fusion", return_value=False):
            self._kernel().bind((x, 2.0)).set_config(helion.Config(block_sizes=[32]))
            with (
                patch.object(
                    KernelCompiler,
                    "compile",
                    autospec=True,
                    side_effect=KernelCompiler.compile,
                ) as compile_fn,
                patch.object(
                    BoundKernel, "to_code", side_effect=AssertionError("codegen ran")
                ),
            ):
                kernel = self._kernel()
                bound = kernel.bind((x, 2.0))
                self.assertIsInstance(bound, BoundKernel)
                torch.testing.assert_close(kernel(x, 2.0), x * 2)
            self.assertEqual(compile_fn.call_count, 1)
            self.assertTrue(kernel._dispatch_cache)
            with patch.object(torch.compiler, "is_compiling", return_value=True):
                torch.testing.assert_close(kernel(x, 3.0), x * 3)

    def test_legacy_source_load_failure_propagates_and_retry_succeeds(self) -> None:
        x = torch.ones(16)
        config = helion.Config(block_sizes=[32])
        with patch.object(cache, "supports_torch_compile_fusion", return_value=False):
            self._kernel().bind((x, 2.0)).set_config(config)
            warm = self._kernel().bind((x, 2.0))
            with (
                patch.object(
                    PyCodeCache,
                    "load",
                    side_effect=RuntimeError("source import failed"),
                ),
                self.assertRaisesRegex(RuntimeError, "source import failed"),
            ):
                warm.set_config(config)
            warm.set_config(config)
            torch.testing.assert_close(warm(x, 2.0), x * 2)
