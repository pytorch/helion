from __future__ import annotations

from builtins import __import__ as _import_module
import json
import operator
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import Mock
from unittest.mock import patch

import pytest
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


def _module_scale() -> float:
    return torch._helion_cache_test_scale


def _read_module_scale(x: torch.Tensor) -> torch.Tensor:
    return x * _module_scale()


def _named_triton_helper(x: torch.Tensor) -> torch.Tensor:
    return hl.triton_kernel("combine", args=(x, x), output_like=x)


_triton_call = hl.triton_kernel
_scale_of = {"x": 2.0}.get


def _aliased_triton_helper(x: torch.Tensor) -> torch.Tensor:
    return _triton_call("combine", args=(x, x), output_like=x)


def _bound_builtin_scale(x: torch.Tensor) -> torch.Tensor:
    return x * _scale_of("x")


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

    def test_module_attribute_change_invalidates_or_bypasses_cache(self) -> None:
        module_alias = torch

        def closure_alias() -> float:
            return module_alias._helion_cache_test_scale

        for fn in (_module_scale, closure_alias):
            with self.subTest(fn=fn.__name__):
                with patch.dict(_read_module_scale.__globals__, {"_module_scale": fn}):
                    kernel = self._kernel(_read_module_scale)
                    with patch.object(
                        torch, "_helion_cache_test_scale", 2.0, create=True
                    ):
                        cold = self._key(kernel)
                        self.assertEqual(self._key(kernel), cold)
                    with patch.object(
                        torch, "_helion_cache_test_scale", 3.0, create=True
                    ):
                        changed = self._key(kernel)
                self.assertTrue(cold is None or changed is None or cold != changed)
        self.assertIsNotNone(self._key(self._kernel()))

    def test_bound_builtin_receiver_change_invalidates_or_bypasses_cache(self) -> None:
        scales = {"x": 2.0}
        with patch.dict(_bound_builtin_scale.__globals__, {"_scale_of": scales.get}):
            kernel = self._kernel(_bound_builtin_scale)
            cold = self._key(kernel)
            self.assertEqual(self._key(kernel), cold)
            scales["x"] = 3.0
            changed = self._key(kernel)
        self.assertTrue(cold is None or changed is None or cold != changed)

    def test_named_triton_helper_change_invalidates_or_bypasses_cache(self) -> None:
        kernel = self._kernel(_named_triton_helper)
        with patch.dict(_named_triton_helper.__globals__, {"combine": operator.add}):
            cold = self._key(kernel)
        with patch.dict(_named_triton_helper.__globals__, {"combine": operator.sub}):
            changed = self._key(kernel)
        self.assertTrue(cold is None or changed is None or cold != changed)
        self.assertIsNone(self._key(self._kernel(_aliased_triton_helper)))

    def test_untracked_module_and_state_access_bypasses_cache(self) -> None:
        def consume_module(module: ModuleType) -> float:
            return module._helion_cache_test_scale

        def pass_module() -> float:
            return consume_module(torch)

        def return_module() -> ModuleType:
            return torch

        def inspect_returned_module() -> float:
            return return_module()._helion_cache_test_scale

        read_attribute = getattr

        def aliased_getattr() -> float:
            return read_attribute(torch, "_helion_cache_test_scale")

        def read_thread_count() -> int:
            return torch.get_num_threads()

        ordinary = self._kernel()
        ordinary_key = self._key(ordinary)
        self.assertIsNotNone(ordinary_key)
        for fn in (
            pass_module,
            inspect_returned_module,
            aliased_getattr,
            read_thread_count,
        ):
            with (
                self.subTest(fn=fn.__name__),
                patch.dict(_read_module_scale.__globals__, {"_module_scale": fn}),
            ):
                self.assertIsNone(self._key(self._kernel(_read_module_scale)))
        self.assertEqual(self._key(ordinary), ordinary_key)

    def test_module_getattr_is_not_executed_during_lookup(self) -> None:
        module = ModuleType("_helion_lazy_cache_test")
        get_attribute = Mock(side_effect=AssertionError("lazy attribute executed"))
        module.__getattr__ = get_attribute

        def lazy_attribute() -> float:
            return module.scale

        with patch.dict(
            _read_module_scale.__globals__, {"_module_scale": lazy_attribute}
        ):
            self.assertIsNone(self._key(self._kernel(_read_module_scale)))
        get_attribute.assert_not_called()

    @pytest.mark.slow
    @pytest.mark.timeout(300)
    @unittest.skipIf(not torch.cuda.is_available(), "requires CUDA and Triton")
    def test_changed_dependencies_in_fresh_processes(self) -> None:
        script = self.root / "dependency_repro.py"
        script.write_text(
            """
import json
import os

import torch
import triton

import helion
import helion.language as hl
from helion._utils import counters

changed = os.environ["CHANGED"] == "1"
if changed:
    @triton.jit
    def combine(a, b):
        return a - b
else:
    @triton.jit
    def combine(a, b):
        return a + b

torch._helion_cache_test_scale = 3.0 if changed else 2.0
SCALES = {"x": 3.0 if changed else 2.0}
scale_of = SCALES.get

def get_scale():
    return torch._helion_cache_test_scale

@helion.kernel(backend="triton", config=helion.Config(block_sizes=[128]))
def named(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = hl.triton_kernel(
            "combine", args=(x[tile], y[tile]), output_like=x[tile]
        )
    return out

@helion.kernel(backend="triton", config=helion.Config(block_sizes=[128]))
def module_attribute(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    scale = get_scale()
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] * scale
    return out

@helion.kernel(backend="triton", config=helion.Config(block_sizes=[128]))
def bound_method(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    scale = scale_of("x")
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] * scale
    return out

@helion.kernel(backend="triton", config=helion.Config(block_sizes=[128]))
def ordinary(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] + 1
    return out

x = torch.ones(128, device="cuda")
results = [named(x * 5, x * 2), module_attribute(x), bound_method(x), ordinary(x)]
expected = [3.0 if changed else 7.0, 3.0 if changed else 2.0,
            3.0 if changed else 2.0, 2.0]
for result, value in zip(results, expected, strict=True):
    torch.testing.assert_close(result, torch.full_like(x, value))
print(json.dumps({"values": [result[0].item() for result in results],
                  "hits": counters["generated_code_cache"]["frontend_hit"]}))
"""
        )
        results = {}
        for level in ("config", "code"):
            results[level] = []
            for changed in ("0", "0", "1"):
                with self.subTest(cache_level=level, changed=changed):
                    process = subprocess.run(
                        [sys.executable, str(script)],
                        env={
                            **os.environ,
                            "HELION_CACHE_LEVEL": level,
                            "HELION_CACHE_DIR": str(self.root / level),
                            "HELION_BACKEND": "triton",
                            "ENABLE_TILE": "0",
                            "CHANGED": changed,
                        },
                        capture_output=True,
                        text=True,
                        timeout=120,
                    )
                    self.assertEqual(process.returncode, 0, process.stderr)
                    results[level].append(json.loads(process.stdout.splitlines()[-1]))
        self.assertEqual(
            [result["values"] for result in results["code"]],
            [result["values"] for result in results["config"]],
        )
        # The ordinary kernel must still use the warm frontend bypass.
        self.assertGreater(results["code"][1]["hits"], 0)

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
