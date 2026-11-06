from __future__ import annotations

from builtins import __import__ as _import_module
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

import torch

import helion
from helion._testing import DEVICE
from helion._testing import RefEagerTestDisabled
from helion._testing import TestCase
import helion.language as hl
from helion.runtime import generated_code_cache as cache
from helion.runtime.ref_mode import RefMode


def _import_scale() -> float:
    from _helion_cache_test_constant import SCALE

    return SCALE


def _lazy_scale(x: torch.Tensor) -> torch.Tensor:
    scale = _import_scale()
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] * scale
    return out


def _builtin_import_scale() -> float:
    return __import__("_helion_cache_test_constant").SCALE


def _lazy_builtin_scale(x: torch.Tensor) -> torch.Tensor:
    scale = _builtin_import_scale()
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] * scale
    return out


def _nested_import(x: torch.Tensor) -> torch.Tensor:
    def get_scale() -> float:
        from _helion_cache_test_constant import SCALE

        return SCALE

    return x * get_scale()


def _divide_by_len(x: torch.Tensor, shape: hl.constexpr) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] / shape[0]
    return out


def _increment(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] + 1
    return out


class TestGeneratedCodeKeyRegressions(RefEagerTestDisabled, TestCase):
    def setUp(self) -> None:
        super().setUp()
        root = self._test_stack.enter_context(tempfile.TemporaryDirectory())
        self.root = Path(root)
        self._test_stack.enter_context(
            patch.dict(
                "os.environ", {"HELION_CACHE_DIR": root, "HELION_SKIP_CACHE": "0"}
            )
        )
        for name in ("helion_key", "torch_key_wrapper", "triton_key_wrapper"):
            self._test_stack.enter_context(
                patch.object(cache, name, return_value=f"{name}-test")
            )
        self._test_stack.enter_context(
            patch.object(cache, "supports_torch_compile_fusion", return_value=True)
        )
        self.x = torch.ones(16)

    def _kernel(self, fn=_increment, **kwargs):
        return helion.kernel(
            backend="triton",
            cache_level="code",
            ref_mode=RefMode.OFF,
            autotune_effort="none",
            **kwargs,
        )(fn)

    def _key(self, kernel):
        return cache.compiled_kernel_cache_key(kernel, (self.x,), ())

    def test_local_and_nested_imports_do_not_execute_during_lookup(self) -> None:
        # Even an import in an untaken branch cannot be executed merely to
        # discover dependencies. Such functions use the normal frontend.
        self.assertNotIn("_helion_cache_test_constant", sys.modules)
        for fn in (_lazy_scale, _nested_import):
            with self.subTest(fn=fn.__name__):
                self.assertIsNone(self._key(self._kernel(fn)))
        self.assertNotIn("_helion_cache_test_constant", sys.modules)
        self.assertFalse((self.root / "generated_code").exists())

    def test_builtin_imports_do_not_execute_during_lookup(self) -> None:
        def aliased_import(x: torch.Tensor) -> torch.Tensor:
            return x * _import_module("_helion_cache_test_constant").SCALE

        def nested_import(x: torch.Tensor) -> torch.Tensor:
            def get_scale() -> float:
                return __import__("_helion_cache_test_constant").SCALE

            return x * get_scale()

        self.assertNotIn("_helion_cache_test_constant", sys.modules)
        for fn in (_lazy_builtin_scale, aliased_import, nested_import):
            with self.subTest(fn=fn.__name__):
                kernel = self._kernel(fn)
                self.assertIsNone(cache.binding_input_key(kernel, (self.x,)))
                self.assertIsNone(self._key(kernel))
        self.assertNotIn("_helion_cache_test_constant", sys.modules)
        self.assertFalse((self.root / "generated_code").exists())

    def test_unsupported_constexpr_does_not_produce_a_key(self) -> None:
        kernel = self._kernel(_divide_by_len)
        for value in (torch.Size([16]), object(), (torch.Size([16]),)):
            with self.subTest(value=type(value).__name__):
                args = (self.x, hl.constexpr(value))
                self.assertIsNone(cache.binding_input_key(kernel, args))
                self.assertIsNone(cache.compiled_kernel_cache_key(kernel, args, ()))
        self.assertIsNotNone(
            cache.binding_input_key(kernel, (self.x, hl.constexpr((16,))))
        )
        self.assertFalse((self.root / "generated_code").exists())

    def test_removing_or_changing_pinned_config_invalidates_manifest(self) -> None:
        config = helion.Config(block_sizes=[32])
        pinned = self._key(self._kernel(config=config))
        equivalent = self._key(self._kernel(config=helion.Config(block_sizes=[32])))
        changed = self._key(self._kernel(config=helion.Config(block_sizes=[64])))
        default = self._key(self._kernel())
        explicit_empty = self._key(self._kernel(config=helion.Config()))
        self.assertIsNotNone(pinned)
        self.assertEqual(equivalent, pinned)
        self.assertEqual(len({pinned, changed, default, explicit_empty}), 4)

    def test_autocast_keys_the_argument_device(self) -> None:
        kernel = self._kernel()
        with (
            patch.object(
                cache, "_find_argument_device", return_value=torch.device("xpu")
            ),
            patch.object(cache, "get_device_name", return_value="test XPU"),
            patch.object(torch, "is_autocast_enabled", return_value=False) as enabled,
            patch.object(
                torch, "get_autocast_dtype", return_value=torch.float16
            ) as dtype,
        ):
            disabled = self._key(kernel)
            enabled.assert_called_once_with("xpu")
            dtype.assert_called_once_with("xpu")
            enabled.return_value = True
            active = self._key(kernel)
            dtype.return_value = torch.bfloat16
            other_dtype = self._key(kernel)
        self.assertNotIn(None, (disabled, active, other_dtype))
        self.assertEqual(len({disabled, active, other_dtype}), 3)

    @unittest.skipIf(not torch.cuda.is_available(), "requires CUDA and Triton")
    def test_lazy_import_change_recompiles_instead_of_serving_stale_code(self) -> None:
        module = ModuleType("_helion_cache_test_constant")
        x = torch.ones(128, device=DEVICE)
        with patch.dict(sys.modules, {module.__name__: module}):
            for scale in (2.0, 3.0):
                with self.subTest(scale=scale):
                    module.SCALE = scale
                    kernel = self._kernel(
                        _lazy_scale, config=helion.Config(block_sizes=[128])
                    )
                    torch.testing.assert_close(kernel(x), x * scale)
        self.assertFalse((self.root / "generated_code").exists())

    @unittest.skipIf(not torch.cuda.is_available(), "requires CUDA and Triton")
    def test_builtin_import_change_recompiles_instead_of_serving_stale_code(
        self,
    ) -> None:
        module = ModuleType("_helion_cache_test_constant")
        x = torch.ones(128, device=DEVICE)
        with patch.dict(sys.modules, {module.__name__: module}):
            for scale in (2.0, 3.25):
                with self.subTest(scale=scale):
                    module.SCALE = scale
                    kernel = self._kernel(
                        _lazy_builtin_scale, config=helion.Config(block_sizes=[128])
                    )
                    torch.testing.assert_close(kernel(x), x * scale)
        self.assertFalse((self.root / "generated_code").exists())

    @unittest.skipIf(not torch.cuda.is_available(), "requires CUDA and Triton")
    def test_removed_config_uses_frontend_default(self) -> None:
        x = torch.ones(128, device=DEVICE)
        pinned = self._kernel(config=helion.Config(block_sizes=[16]))
        torch.testing.assert_close(pinned(x), x + 1)
        default = self._kernel()
        torch.testing.assert_close(default(x), x + 1)
        baseline = helion.kernel(
            backend="triton", cache_level="config", autotune_effort="none"
        )(_increment)
        torch.testing.assert_close(baseline(x), x + 1)
        self.assertEqual(default.bind((x,))._config, baseline.bind((x,))._config)
        self.assertNotEqual(default.bind((x,))._config, pinned.bind((x,))._config)

    @unittest.skipIf(not torch.cuda.is_available(), "requires CUDA and Triton")
    def test_unsupported_constexpr_executes_without_writing_cache(self) -> None:
        x = torch.ones(128, device=DEVICE)
        for _ in range(2):
            kernel = self._kernel(
                _divide_by_len, config=helion.Config(block_sizes=[128])
            )
            torch.testing.assert_close(kernel(x, hl.constexpr(x.shape)), x / 128)
        self.assertFalse((self.root / "generated_code").exists())
