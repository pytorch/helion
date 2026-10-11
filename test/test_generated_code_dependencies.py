from __future__ import annotations

from pathlib import Path
import tempfile
from typing import TYPE_CHECKING
from unittest.mock import patch

import torch

import helion
from helion._testing import TestCase
from helion.runtime import generated_code_cache as cache
from helion.runtime.ref_mode import RefMode

if TYPE_CHECKING:
    from helion.runtime.kernel import Kernel

_NESTED_SHIFT = 1
_NESTED_VALUES = {"values": [(1, {"shift": 2})]}
_MISSING = 1


class _OpaqueState:
    def __repr__(self) -> str:
        raise AssertionError("opaque dependencies must not be represented")


_OPAQUE = _OpaqueState()


def _nested_global(x: torch.Tensor) -> torch.Tensor:
    def shift() -> int:
        return _NESTED_SHIFT

    return x + shift()


def _nested_values(x: torch.Tensor) -> torch.Tensor:
    def outer() -> int:
        def inner() -> int:
            return sum(value[1]["shift"] for value in _NESTED_VALUES["values"][:1])

        return inner()

    return x + outer()


def _nested_opaque(x: torch.Tensor) -> torch.Tensor:
    def state() -> object:
        return _OPAQUE

    state()
    return x


def _nested_dynamic(x: torch.Tensor) -> torch.Tensor:
    def shift() -> int:
        return globals()["_NESTED_SHIFT"]

    return x + shift()


def _nested_missing(x: torch.Tensor) -> torch.Tensor:
    def shift() -> int:
        return _MISSING

    return x + shift()


class TestGeneratedCodeDependencies(TestCase):
    def setUp(self) -> None:
        super().setUp()
        root = self._test_stack.enter_context(tempfile.TemporaryDirectory())
        self._test_stack.enter_context(
            patch.dict(
                "os.environ", {"HELION_CACHE_DIR": root, "HELION_SKIP_CACHE": "0"}
            )
        )
        self._test_stack.enter_context(
            patch.object(cache, "helion_key", return_value="helion-v1")
        )
        self._test_stack.enter_context(
            patch.object(cache, "torch_key_wrapper", return_value="torch-v1")
        )
        self._test_stack.enter_context(
            patch.object(cache, "triton_key_wrapper", return_value="triton-v1")
        )
        self._test_stack.enter_context(
            patch.object(cache, "supports_torch_compile_fusion", return_value=True)
        )
        self.cache_root = Path(root)
        self.x = torch.ones(16)

    def _key(self, kernel: Kernel) -> str | None:
        return cache.compiled_kernel_cache_key(kernel, (self.x,), ())

    def test_nested_global_change_invalidates_saved_binding(self) -> None:
        kernel = helion.kernel(
            backend="triton",
            ref_mode=RefMode.OFF,
            generated_code_cache=True,
            autotune_effort="none",
        )(_nested_global)
        key = self._key(kernel)
        self.assertIsNotNone(key)
        assert key is not None
        config = helion.Config(block_sizes=[32])
        source = "def nested_global():\n    return 1\n"
        source_key = "a" * 64
        cache.save_generated_code(source_key, source)
        cache.save_compiled_kernel(key, config, config, source_key)
        self.assertEqual(cache.load_compiled_kernel(key), (config, config, source))
        with patch.dict(_nested_global.__globals__, {"_NESTED_SHIFT": 2}):
            changed = self._key(kernel)
            self.assertIsNotNone(changed)
            self.assertNotEqual(changed, key)
            assert changed is not None
            self.assertIsNone(cache.load_compiled_kernel(changed))
        self.assertEqual(self._key(kernel), key)

    def test_recursive_supported_values_preserve_binding_identity(self) -> None:
        kernel = helion.kernel(
            backend="triton",
            ref_mode=RefMode.OFF,
            generated_code_cache=True,
            autotune_effort="none",
        )(_nested_values)
        key = self._key(kernel)
        self.assertIsNotNone(key)
        equivalent = {"values": [(1, {"shift": 2})]}
        with patch.dict(_nested_values.__globals__, {"_NESTED_VALUES": equivalent}):
            self.assertEqual(self._key(kernel), key)
            equivalent["values"][0][1]["shift"] = 4
            self.assertNotEqual(self._key(kernel), key)

    def test_unsupported_dependencies_and_settings_skip_binding_cache(self) -> None:
        opaque = helion.kernel(
            backend="triton",
            ref_mode=RefMode.OFF,
            generated_code_cache=True,
            autotune_effort="none",
        )(_nested_opaque)
        self.assertIsNone(self._key(opaque))
        dynamic = helion.kernel(
            backend="triton",
            ref_mode=RefMode.OFF,
            generated_code_cache=True,
            autotune_effort="none",
        )(_nested_dynamic)
        self.assertIsNone(self._key(dynamic))
        missing = helion.kernel(
            backend="triton",
            ref_mode=RefMode.OFF,
            generated_code_cache=True,
            autotune_effort="none",
        )(_nested_missing)
        with patch.dict(_nested_missing.__globals__):
            _nested_missing.__globals__.pop("_MISSING")
            self.assertIsNone(self._key(missing))
        supported = helion.kernel(
            backend="triton",
            ref_mode=RefMode.OFF,
            generated_code_cache=True,
            autotune_effort="none",
        )(_nested_global)
        with patch.object(supported.settings, "fast_math", _OPAQUE):
            self.assertIsNone(self._key(supported))
        self.assertFalse((self.cache_root / "generated_code").exists())
