from __future__ import annotations

import enum
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch

import helion
from helion._compiler.cache_dependencies import DependencyCapture
from helion._compiler.cache_dependencies import validate_dependencies
from helion._testing import TestCase
import helion.language as hl
from helion.runtime import generated_code_cache as cache
from helion.runtime.cached_kernel import _CachedBoundKernel
from helion.runtime.kernel import BoundKernel
from helion.runtime.kernel import KernelCompiler
from helion.runtime.kernel import PyCodeCache
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

    value = shift()
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] + value
    return out


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


def _direct_values(x: torch.Tensor) -> torch.Tensor:
    value = _NESTED_VALUES["values"][0][1]["shift"]
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] + value
    return out


_DEPENDENCY_HELPER = int


def _dependency_call_kernel(x: torch.Tensor) -> torch.Tensor:
    value = _DEPENDENCY_HELPER()
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] + value
    return out


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
        self.sources = []
        self._test_stack.enter_context(
            patch.object(PyCodeCache, "load", side_effect=self._load)
        )

    def _load(self, source, *, extra=""):
        self.sources.append(source)
        path = self.cache_root / f"module_{len(self.sources)}.py"
        path.write_text(source)
        return SimpleNamespace(
            __file__=str(path),
            _nested_global=lambda x: None,
            _direct_values=lambda x: None,
            _dependency_call_kernel=lambda x: None,
        )

    def _compile(self, fn):
        config = helion.Config(block_sizes=[32])
        kernel = helion.kernel(
            backend="triton", cache_level="code", config=config, autotune_effort="none"
        )(fn)
        with kernel._bind_lock:
            bound = kernel._bind((self.x,))
        bound.set_config(config)
        return bound

    def _key(self, kernel: Kernel) -> str | None:
        return cache.compiled_kernel_cache_key(kernel, (self.x,), ())

    def test_nested_global_change_invalidates_saved_binding(self) -> None:
        bound = self._compile(_nested_global)
        snapshot = bound.env.cache_dependencies.snapshot()
        self.assertIsNotNone(snapshot)
        with patch.object(
            KernelCompiler,
            "compile",
            side_effect=AssertionError("unchanged dependency recompiled"),
        ):
            warm = self._compile(_nested_global)
        self.assertIsInstance(warm, _CachedBoundKernel)
        with patch.dict(_nested_global.__globals__, {"_NESTED_SHIFT": 2}):
            self.assertFalse(
                validate_dependencies(
                    snapshot, _nested_global, cache.dependency_fingerprint
                )
            )
            with patch.object(
                KernelCompiler,
                "compile",
                autospec=True,
                side_effect=KernelCompiler.compile,
            ) as compile_fn:
                changed = self._compile(_nested_global)
            self.assertIsInstance(changed, BoundKernel)
            self.assertGreater(compile_fn.call_count, 0)
        self.assertTrue(
            validate_dependencies(
                snapshot, _nested_global, cache.dependency_fingerprint
            )
        )
        self.assertEqual(self.sources[0], self.sources[1])

    def test_recursive_supported_values_preserve_binding_identity(self) -> None:
        bound = self._compile(_direct_values)
        snapshot = bound.env.cache_dependencies.snapshot()
        self.assertIsNotNone(snapshot)
        equivalent = {"values": [(1, {"shift": 2})]}
        with patch.dict(_direct_values.__globals__, {"_NESTED_VALUES": equivalent}):
            self.assertTrue(
                validate_dependencies(
                    snapshot, _direct_values, cache.dependency_fingerprint
                )
            )
            with patch.object(
                KernelCompiler,
                "compile",
                side_effect=AssertionError("unchanged dependency recompiled"),
            ):
                warm = self._compile(_direct_values)
            self.assertIsInstance(warm, _CachedBoundKernel)
            equivalent["values"][0][1]["shift"] = 4
            self.assertFalse(
                validate_dependencies(
                    snapshot, _direct_values, cache.dependency_fingerprint
                )
            )
            changed = self._compile(_direct_values)
            self.assertIsInstance(changed, BoundKernel)
        self.assertEqual(self.sources[0], self.sources[1])
        self.assertTrue(
            validate_dependencies(
                snapshot, _direct_values, cache.dependency_fingerprint
            )
        )
        # Nested user execution remains unsupported even when it reads the
        # same containers as the admitted direct compiler expression.
        with patch.dict(
            _dependency_call_kernel.__globals__, {"_DEPENDENCY_HELPER": _nested_values}
        ):
            capture = DependencyCapture(
                _dependency_call_kernel, cache.dependency_fingerprint
            )
            capture.record_global("_DEPENDENCY_HELPER", _nested_values)
            self.assertIsNone(capture.snapshot())

    def test_global_and_nonlocal_with_the_same_name_are_both_keyed(self) -> None:
        scale = 2

        def helper():
            def read_global():
                global scale
                return scale

            return scale + read_global()

        with patch.dict(
            _dependency_call_kernel.__globals__, {"_DEPENDENCY_HELPER": helper}
        ):
            for value in (3, 4):
                with patch.dict(helper.__globals__, {"scale": value}):
                    self.assertEqual(helper(), 2 + value)
                    bound = self._compile(_dependency_call_kernel)
                    self.assertIsInstance(bound, BoundKernel)
                    self.assertIsNone(bound.env.cache_dependencies.snapshot())
                    self.assertIsNone(bound._generated_code)
        self.assertNotEqual(self.sources[0], self.sources[1])
        self.assertFalse((self.cache_root / "generated_code").exists())

    def test_user_enum_with_mutable_attributes_bypasses_cache(self) -> None:
        class Scale(enum.Enum):
            X = 1

        def helper():
            return Scale.X.factor

        Scale.X.factor = 2.0
        self.assertIs(cache._dependency_key(Scale.X), cache._UNCACHEABLE)
        self.assertIs(cache._dependency_key(helper), cache._UNCACHEABLE)

    def test_function_attributes_through_local_alias_bypass_cache(self) -> None:
        def helper():
            alias = helper
            return alias.scale

        helper.scale = 2.0
        self.assertEqual(helper(), 2.0)
        self.assertIs(cache._dependency_key(helper), cache._UNCACHEABLE)
        helper.scale = 3.0
        self.assertEqual(helper(), 3.0)
        self.assertIs(cache._dependency_key(helper), cache._UNCACHEABLE)

    @pytest.mark.slow
    def test_replaced_intrinsic_before_first_lookup_is_not_trusted(self) -> None:
        script = """
import helion.language as hl
from helion.runtime import generated_code_cache as cache

shift = 2
def replacement():
    return shift
hl.tile = replacement
def helper():
    return hl.tile()

from helion._compiler.cache_dependencies import DependencyCapture
capture = DependencyCapture(helper, cache.dependency_fingerprint)
capture.record_global("hl", hl)
capture.record_attribute(hl, "tile", hl.tile)
assert capture.snapshot() is None
assert not cache.cacheable_call(replacement)
first = cache._dependency_key(replacement)
assert helper() == 2
shift = 3
second = cache._dependency_key(replacement)
assert helper() == 3
assert first is cache._UNCACHEABLE or second is cache._UNCACHEABLE or first != second
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_unsupported_dependencies_and_settings_skip_binding_cache(self) -> None:
        for name, value in (
            ("_OPAQUE", _OPAQUE),
            ("_DEPENDENCY_HELPER", _nested_dynamic),
        ):
            with patch.dict(_dependency_call_kernel.__globals__, {name: value}):
                capture = DependencyCapture(
                    _dependency_call_kernel, cache.dependency_fingerprint
                )
                capture.record_global(name, value)
                self.assertIsNone(capture.snapshot())
                self.assertFalse(
                    validate_dependencies(
                        capture.snapshot(),
                        _dependency_call_kernel,
                        cache.dependency_fingerprint,
                    )
                )
        with patch.dict(_nested_missing.__globals__):
            _nested_missing.__globals__.pop("_MISSING")
            capture = DependencyCapture(_nested_missing, cache.dependency_fingerprint)
            capture.record_global("_MISSING", 1)
            self.assertIsNone(capture.snapshot())
        supported = helion.kernel(
            backend="triton",
            ref_mode=RefMode.OFF,
            cache_level="code",
            autotune_effort="none",
        )(_direct_values)
        with patch.object(supported.settings, "fast_math", _OPAQUE):
            self.assertIsNone(self._key(supported))
        self.assertFalse((self.cache_root / "generated_code").exists())
