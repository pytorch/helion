from __future__ import annotations

import ast
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
from unittest.mock import patch

import pytest
import torch
from torch.overrides import TorchFunctionMode
from torch.utils._python_dispatch import TorchDispatchMode

import helion
from helion._testing import RefEagerTestDisabled
from helion._testing import TestCase
from helion._testing import onlyBackends
from helion._utils import counters
from helion.autotuner import BooleanFragment
from helion.autotuner import PowerOfTwoFragment
import helion.language as hl
from helion.language import creation_ops
from helion.runtime import generated_code_cache as cache
from helion.runtime.cached_kernel import _CachedBoundKernel
from helion.runtime.kernel import BoundKernel
from helion.runtime.kernel import KernelCompiler
from helion.runtime.kernel import PyCodeCache

pytestmark = pytest.mark.cpu_only

_SCALE_MODULE = ModuleType("_helion_admission_scale")
_SCALE_MODULE.scale = 2.0
_SCALE_ALIAS = _SCALE_MODULE
_ORIGINAL_TENSOR_ABS = torch.Tensor.abs
_ABS_FACTOR = 2.0


def _scaled_abs(value):
    return _ORIGINAL_TENSOR_ABS(value) * _ABS_FACTOR


class _FunctionScale(TorchFunctionMode):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def __torch_function__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        return result * self.factor if func is torch.add else result


class _DispatchScale(TorchDispatchMode):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        return result * self.factor if func is torch.ops.aten.add.Tensor else result


def _make_nested_helper(scale: float):
    def helper() -> float:
        class Constants:
            value = scale

        return Constants.value

    return helper


_NESTED_HELPER = _make_nested_helper(2.0)


def _direct_module_scale(x: torch.Tensor) -> torch.Tensor:
    scale = _SCALE_MODULE.scale
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] * scale
    return out


def _nested_class_scale(x: torch.Tensor) -> torch.Tensor:
    scale = _NESTED_HELPER()
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] * scale
    return out


def _module_string(x: torch.Tensor) -> torch.Tensor:
    value = len(str(_SCALE_MODULE))
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] + value
    return out


def _module_identity(x: torch.Tensor) -> torch.Tensor:
    value = 2.0 if _SCALE_MODULE is _SCALE_ALIAS else 3.0
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] + value
    return out


def _untracked_tensor_metadata(x: torch.Tensor) -> torch.Tensor:
    scale = 2.0 if x.is_contiguous() else 3.0
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] * scale
    return out


def _tensor_abs(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile].abs()
    return out


def _torch_add(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = torch.add(x[tile], 1.0)
    return out


def _full_kernel(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = hl.full([tile], 1.0)
    return out


def _full_handler_with_mutable_closure():
    original = hl.full._codegen["common"]
    factor = [2.0]

    def modified(state):
        expression = original(state)
        expression.args[1] = ast.Constant(value=factor[0])
        return expression

    return modified, factor


# The function below is copied from vLLM, retaining its complete kernel body.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# https://github.com/vllm-project/vllm/blob/d88feb483b9c4cd8f507ebbe88f0d785a15dfb35/vllm/kernels/helion/ops/scaled_mm.py
def scaled_mm(
    out: torch.Tensor,  # [M, N]
    a: torch.Tensor,  # [M, K]
    b: torch.Tensor,  # [K, N]
    a_scales: torch.Tensor,  # [1]/[1, 1]/[M]/[M, 1]
    b_scales: torch.Tensor,  # [1]/[1, 1]/[N]/[N, 1]
    bias: torch.Tensor | None = None,  # [N]
) -> None:
    M, K = a.shape
    N = b.shape[1]
    hl.specialize(K)
    hl.specialize(N)

    assert N > 0 and K > 0 and M > 0
    assert b.shape[0] == K
    assert a.dtype == b.dtype
    assert a.stride(1) == 1
    assert b.stride(0) == 1

    a_scales = a_scales.reshape(-1, 1) if a_scales.dim() <= 1 else a_scales
    b_scales = b_scales.reshape(-1, 1) if b_scales.dim() <= 1 else b_scales

    assert a_scales.dtype == b_scales.dtype and a_scales.is_floating_point()
    assert a_scales.shape[1] == 1 and (a_scales.shape[0] == 1 or a_scales.shape[0] == M)
    assert b_scales.shape[1] == 1 and (b_scales.shape[0] == 1 or b_scales.shape[0] == N)
    hl.specialize(b_scales.shape[1])

    out_dtype = out.dtype
    assert out_dtype.is_floating_point
    if bias is not None:
        assert bias.numel() == N and bias.dtype == out_dtype

    acc_dtype = torch.float32 if a.is_floating_point() else torch.int32

    split_k = hl.register_tunable("split_k", PowerOfTwoFragment(1, 256))
    k_block_size = helion.next_power_of_2(helion.cdiv(K, split_k))
    if split_k > 1:
        out.zero_()

    swap_ab = hl.register_tunable("swap_ab", BooleanFragment())

    for tile_m, tile_n, outer_k in hl.tile(
        [M, N, K], block_size=[None, None, k_block_size]
    ):
        acc = hl.zeros([tile_m, tile_n], acc_dtype)
        acc_t = acc.t()
        for tile_k in hl.tile(outer_k.begin, outer_k.end):
            if swap_ab:
                a_blk = hl.load(a, [tile_m.index[None, :], tile_k.index[:, None]])
                b_blk = hl.load(b, [tile_k.index[None, :], tile_n.index[:, None]])
                acc_t = hl.dot(
                    b_blk,
                    a_blk,
                    acc=acc_t,
                    out_dtype=acc_dtype,
                )
            else:
                acc = hl.dot(
                    a[tile_m, tile_k],
                    b[tile_k, tile_n],
                    acc=acc,
                    out_dtype=acc_dtype,
                )

        if swap_ab:
            acc = acc_t.t().to(torch.float32)
        else:
            acc = acc.to(torch.float32)

        if a_scales.shape[0] == M:
            a_scales_blk = a_scales[tile_m, :]
        else:
            a_scales_blk = a_scales[0, 0].expand(tile_m.block_size, 1)
        acc = a_scales_blk * acc

        if b_scales.shape[0] == N:
            b_scales_blk = b_scales[tile_n, :]
        else:
            b_scales_blk = b_scales[0, 0].expand(tile_n.block_size, 1)
        acc = b_scales_blk.T * acc

        out_blk = acc.to(out_dtype)

        if bias is not None:
            if outer_k.begin == 0:
                out_blk += bias[tile_n]

        if split_k == 1:
            out[tile_m, tile_n] = out_blk
        else:
            hl.atomic_add(out, [tile_m, tile_n], out_blk)


def _kernel(fn, *, cache_level="code", config=None):
    return helion.kernel(
        backend="triton",
        cache_level=cache_level,
        config=config or helion.Config(block_sizes=[32]),
        static_shapes=False,
        autotune_effort="none",
    )(fn)


@onlyBackends(["triton"])
class TestGeneratedCodeAdmission(RefEagerTestDisabled, TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.root = Path(self._test_stack.enter_context(tempfile.TemporaryDirectory()))
        self._test_stack.enter_context(
            patch.dict(
                os.environ,
                {"HELION_CACHE_DIR": str(self.root), "HELION_SKIP_CACHE": "0"},
            )
        )
        for name in ("helion_key", "torch_key_wrapper", "triton_key_wrapper"):
            self._test_stack.enter_context(patch.object(cache, name, return_value=name))
        self._test_stack.enter_context(
            patch.object(cache, "supports_torch_compile_fusion", return_value=True)
        )
        self._test_stack.enter_context(
            patch.object(PyCodeCache, "load", side_effect=self._load)
        )
        self.sources = []
        counters["generated_code_cache"].clear()
        self.x = torch.ones(16)

    def _load(self, source: str, *, extra: str = "") -> SimpleNamespace:
        self.sources.append(source)
        path = self.root / f"module_{len(self.sources)}.py"
        path.write_text(source)
        # CPU tests exercise real binding/codegen and artifact admission.
        # GPU subprocess tests below validate the emitted kernels numerically.
        return SimpleNamespace(
            __file__=str(path),
            _direct_module_scale=lambda x: None,
            _nested_class_scale=lambda x: None,
            _module_string=lambda x: None,
            _module_identity=lambda x: None,
            _untracked_tensor_metadata=lambda x: None,
            _tensor_abs=lambda x: None,
            _torch_add=lambda x: None,
            _full_kernel=lambda x: None,
        )

    def _bind_and_compile(self, fn, *, cache_level="code"):
        kernel = _kernel(fn, cache_level=cache_level)
        with kernel._bind_lock:
            bound = kernel._bind((self.x,))
        bound.set_config(helion.Config(block_sizes=[32]))
        return bound

    def test_nested_class_helper_never_loads_a_frontend_artifact(self) -> None:
        for scale in (2.0, 2.0, 3.0):
            with (
                self.subTest(scale=scale),
                patch.dict(
                    _nested_class_scale.__globals__,
                    {"_NESTED_HELPER": _make_nested_helper(scale)},
                ),
                patch.object(
                    KernelCompiler,
                    "compile",
                    autospec=True,
                    side_effect=KernelCompiler.compile,
                ) as compile_fn,
            ):
                bound = self._bind_and_compile(_nested_class_scale)
                self.assertIsInstance(bound, BoundKernel)
                self.assertGreater(compile_fn.call_count, 0)
        self.assertEqual(self.sources[0], self.sources[1])
        self.assertNotEqual(self.sources[0], self.sources[2])
        self.assertEqual(counters["generated_code_cache"]["frontend_hit"], 0)

    def test_module_attribute_snapshot_is_checked_before_reuse(self) -> None:
        with patch.object(_SCALE_MODULE, "scale", 2.0):
            self._bind_and_compile(_direct_module_scale)
            with patch.object(
                KernelCompiler,
                "compile",
                side_effect=AssertionError("warm frontend ran"),
            ):
                warm = self._bind_and_compile(_direct_module_scale)
            self.assertIsInstance(warm, _CachedBoundKernel)
        with (
            patch.object(_SCALE_MODULE, "scale", 3.0),
            patch.object(
                KernelCompiler,
                "compile",
                autospec=True,
                side_effect=KernelCompiler.compile,
            ) as compile_fn,
        ):
            changed = self._bind_and_compile(_direct_module_scale)
            self.assertIsInstance(changed, BoundKernel)
            self.assertGreater(compile_fn.call_count, 0)
        self.assertEqual(self.sources[0], self.sources[1])
        self.assertNotEqual(self.sources[0], self.sources[2])

    def test_changed_codegen_handler_invalidates_saved_artifact(self) -> None:
        cold = self._bind_and_compile(_direct_module_scale)
        snapshot = cache._dependency_snapshot(cold)
        self.assertIsNotNone(snapshot)
        with patch.object(
            KernelCompiler, "compile", side_effect=AssertionError("warm frontend ran")
        ):
            warm = self._bind_and_compile(_direct_module_scale)
        self.assertIsInstance(warm, _CachedBoundKernel)

        original = hl.load._codegen["triton"]

        def shifted_load(state):
            return ast.BinOp(
                left=original(state), op=ast.Add(), right=ast.Constant(1.0)
            )

        with patch.dict(hl.load._codegen, {"triton": shifted_load}):
            self.assertFalse(
                cache.validate_kernel_dependencies(snapshot, _direct_module_scale)
            )
            for _ in range(2):
                with patch.object(
                    KernelCompiler,
                    "compile",
                    autospec=True,
                    side_effect=KernelCompiler.compile,
                ) as compile_fn:
                    changed = self._bind_and_compile(_direct_module_scale)
                self.assertIsInstance(changed, BoundKernel)
                self.assertGreater(compile_fn.call_count, 0)
                self.assertIsNone(cache._dependency_snapshot(changed))
                self.assertIsNone(changed._generated_code)
        self.assertEqual(self.sources[0], self.sources[1])
        self.assertNotEqual(self.sources[0], self.sources[2])
        self.assertEqual(self.sources[2], self.sources[3])
        self.assertIn(" + 1.0", self.sources[2])

    def test_codegen_closure_before_cold_compile_bypasses_cache(self) -> None:
        modified, factor = _full_handler_with_mutable_closure()
        with patch.dict(hl.full._codegen, {"common": modified}):
            for value in (2.0, 3.0):
                factor[0] = value
                with patch.object(
                    KernelCompiler,
                    "compile",
                    autospec=True,
                    side_effect=KernelCompiler.compile,
                ) as compile_fn:
                    bound = self._bind_and_compile(_full_kernel)
                self.assertIsInstance(bound, BoundKernel)
                self.assertGreater(compile_fn.call_count, 0)
                self.assertIsNone(cache._dependency_snapshot(bound))
                self.assertIsNone(bound._generated_code)
        self.assertNotEqual(self.sources[0], self.sources[1])
        self.assertIn("2.0", self.sources[0])
        self.assertIn("3.0", self.sources[1])
        self.assertFalse((self.root / "generated_code").exists())

    def test_codegen_closure_after_warm_hit_invalidates_artifact(self) -> None:
        cold = self._bind_and_compile(_full_kernel)
        snapshot = cache._dependency_snapshot(cold)
        self.assertIsNotNone(snapshot)
        with patch.object(
            KernelCompiler, "compile", side_effect=AssertionError("warm frontend ran")
        ):
            warm = self._bind_and_compile(_full_kernel)
        self.assertIsInstance(warm, _CachedBoundKernel)

        modified, factor = _full_handler_with_mutable_closure()
        with patch.dict(hl.full._codegen, {"common": modified}):
            self.assertFalse(cache.validate_kernel_dependencies(snapshot, _full_kernel))
            for value in (2.0, 3.0):
                factor[0] = value
                with patch.object(
                    KernelCompiler,
                    "compile",
                    autospec=True,
                    side_effect=KernelCompiler.compile,
                ) as compile_fn:
                    changed = self._bind_and_compile(_full_kernel)
                self.assertIsInstance(changed, BoundKernel)
                self.assertGreater(compile_fn.call_count, 0)
                self.assertIsNone(cache._dependency_snapshot(changed))
                self.assertIsNone(changed._generated_code)
        self.assertEqual(self.sources[0], self.sources[1])
        self.assertNotEqual(self.sources[0], self.sources[2])
        self.assertNotEqual(self.sources[2], self.sources[3])
        self.assertIn("2.0", self.sources[2])
        self.assertIn("3.0", self.sources[3])

    def test_operation_change_after_frontend_prevents_cache_publication(self) -> None:
        bound = _kernel(_direct_module_scale).bind((self.x,))
        self.assertIsNotNone(cache._dependency_snapshot(bound))
        original = hl.load._codegen["triton"]

        def shifted_load(state):
            return ast.BinOp(
                left=original(state), op=ast.Add(), right=ast.Constant(1.0)
            )

        with patch.dict(hl.load._codegen, {"triton": shifted_load}):
            bound.set_config(helion.Config(block_sizes=[32]))
            self.assertIsNone(cache._dependency_snapshot(bound))
        self.assertIn(" + 1.0", self.sources[0])
        self.assertFalse((self.root / "generated_code").exists())
        self.assertIsNone(bound.env.cache_dependencies.snapshot())

    def test_untracked_tensor_metadata_accessor_uses_normal_compilation(self) -> None:
        for _ in range(2):
            with patch.object(
                KernelCompiler,
                "compile",
                autospec=True,
                side_effect=KernelCompiler.compile,
            ) as compile_fn:
                bound = self._bind_and_compile(_untracked_tensor_metadata)
            self.assertIsInstance(bound, BoundKernel)
            self.assertGreater(compile_fn.call_count, 0)
            self.assertIsNone(bound.env.cache_dependencies.snapshot())
            self.assertIsNone(bound._generated_code)
        self.assertFalse((self.root / "generated_code").exists())

    def test_tensor_method_replaced_before_cold_compile_bypasses_cache(self) -> None:
        with patch.object(torch.Tensor, "abs", _scaled_abs):
            for factor in (2.0, 3.0):
                with (
                    patch.dict(_scaled_abs.__globals__, {"_ABS_FACTOR": factor}),
                    patch.object(
                        KernelCompiler,
                        "compile",
                        autospec=True,
                        side_effect=KernelCompiler.compile,
                    ) as compile_fn,
                ):
                    bound = self._bind_and_compile(_tensor_abs)
                self.assertIsInstance(bound, BoundKernel)
                self.assertGreater(compile_fn.call_count, 0)
                self.assertIsNone(bound.env.cache_dependencies.snapshot())
                self.assertIsNone(bound._generated_code)
        self.assertNotEqual(self.sources[0], self.sources[1])
        self.assertIn("2.0", self.sources[0])
        self.assertIn("3.0", self.sources[1])
        self.assertFalse((self.root / "generated_code").exists())

    def test_tensor_method_replaced_after_warm_hit_invalidates_artifact(self) -> None:
        cold = self._bind_and_compile(_tensor_abs)
        snapshot = cache._dependency_snapshot(cold)
        self.assertIsNotNone(snapshot)
        with patch.object(
            KernelCompiler, "compile", side_effect=AssertionError("warm frontend ran")
        ):
            warm = self._bind_and_compile(_tensor_abs)
        self.assertIsInstance(warm, _CachedBoundKernel)
        with patch.object(torch.Tensor, "abs", _scaled_abs):
            self.assertFalse(cache.validate_kernel_dependencies(snapshot, _tensor_abs))
            for factor in (2.0, 3.0):
                with (
                    patch.dict(_scaled_abs.__globals__, {"_ABS_FACTOR": factor}),
                    patch.object(
                        KernelCompiler,
                        "compile",
                        autospec=True,
                        side_effect=KernelCompiler.compile,
                    ) as compile_fn,
                ):
                    changed = self._bind_and_compile(_tensor_abs)
                self.assertIsInstance(changed, BoundKernel)
                self.assertGreater(compile_fn.call_count, 0)
                self.assertIsNone(changed.env.cache_dependencies.snapshot())
                self.assertIsNone(changed._generated_code)
        self.assertEqual(self.sources[0], self.sources[1])
        self.assertNotEqual(self.sources[0], self.sources[2])
        self.assertNotEqual(self.sources[2], self.sources[3])
        self.assertIn("2.0", self.sources[2])
        self.assertIn("3.0", self.sources[3])

    def test_implicit_tensor_operator_replacement_invalidates_saved_artifact(
        self,
    ) -> None:
        cold = self._bind_and_compile(_direct_module_scale)
        snapshot = cache._dependency_snapshot(cold)
        self.assertIsNotNone(snapshot)
        with patch.object(
            KernelCompiler, "compile", side_effect=AssertionError("warm frontend ran")
        ):
            warm = self._bind_and_compile(_direct_module_scale)
        self.assertIsInstance(warm, _CachedBoundKernel)
        original = torch.Tensor.__mul__

        def shifted_multiply(left, right):
            return original(left, right) + 7

        with patch.object(torch.Tensor, "__mul__", shifted_multiply):
            self.assertFalse(
                cache.validate_kernel_dependencies(snapshot, _direct_module_scale)
            )
            self._bind_and_compile(_direct_module_scale, cache_level="config")
            with patch.object(
                KernelCompiler,
                "compile",
                autospec=True,
                side_effect=KernelCompiler.compile,
            ) as compile_fn:
                changed = self._bind_and_compile(_direct_module_scale)
            self.assertIsInstance(changed, BoundKernel)
            self.assertGreater(compile_fn.call_count, 0)
            self.assertIsNone(cache._dependency_snapshot(changed))
            self.assertIsNone(changed._generated_code)
        self.assertEqual(self.sources[0], self.sources[1])
        self.assertNotEqual(self.sources[0], self.sources[2])
        self.assertEqual(self.sources[2], self.sources[3])

    def test_native_codegen_helper_replacement_invalidates_saved_artifact(self) -> None:
        cold = self._bind_and_compile(_full_kernel)
        snapshot = cache._dependency_snapshot(cold)
        self.assertIsNotNone(snapshot)
        with patch.object(
            KernelCompiler, "compile", side_effect=AssertionError("warm frontend ran")
        ):
            warm = self._bind_and_compile(_full_kernel)
        self.assertIsInstance(warm, _CachedBoundKernel)
        original = creation_ops.expr_from_string

        def changed_expression(*args, **kwargs):
            expression = original(*args, **kwargs)
            if (
                isinstance(expression, ast.Call)
                and isinstance(expression.func, ast.Attribute)
                and expression.func.attr == "full"
            ):
                expression.args[1] = ast.Constant(value=2.0)
            return expression

        with patch.object(creation_ops, "expr_from_string", changed_expression):
            self.assertFalse(cache.validate_kernel_dependencies(snapshot, _full_kernel))
            self._bind_and_compile(_full_kernel, cache_level="config")
            with patch.object(
                KernelCompiler,
                "compile",
                autospec=True,
                side_effect=KernelCompiler.compile,
            ) as compile_fn:
                changed = self._bind_and_compile(_full_kernel)
            self.assertIsInstance(changed, BoundKernel)
            self.assertGreater(compile_fn.call_count, 0)
            self.assertIsNone(cache._dependency_snapshot(changed))
            self.assertIsNone(changed._generated_code)
        self.assertEqual(self.sources[0], self.sources[1])
        self.assertNotEqual(self.sources[0], self.sources[2])
        self.assertEqual(self.sources[2], self.sources[3])
        self.assertIn("1.0", self.sources[0])
        self.assertIn("2.0", self.sources[2])

    def test_mask_value_handler_replacement_rejects_saved_certificate(self) -> None:
        cold = self._bind_and_compile(_full_kernel)
        snapshot = cache._dependency_snapshot(cold)
        self.assertTrue(cache.validate_kernel_dependencies(snapshot, _full_kernel))
        with patch.object(hl.full, "_get_masked_value", lambda node: 42.0):
            self.assertFalse(cache.validate_kernel_dependencies(snapshot, _full_kernel))
            with (
                patch.object(
                    KernelCompiler,
                    "compile",
                    side_effect=RuntimeError("frontend fallback"),
                ),
                self.assertRaisesRegex(RuntimeError, "frontend fallback"),
            ):
                self._bind_and_compile(_full_kernel)

    def test_external_torch_modes_before_cold_compile_bypass_cache(self) -> None:
        for mode_type in (_FunctionScale, _DispatchScale):
            first_source = len(self.sources)
            for factor in (2.0, 3.0):
                with (
                    self.subTest(mode=mode_type.__name__, factor=factor),
                    mode_type(factor),
                    patch.object(
                        KernelCompiler,
                        "compile",
                        autospec=True,
                        side_effect=KernelCompiler.compile,
                    ) as compile_fn,
                ):
                    bound = self._bind_and_compile(_torch_add)
                self.assertIsInstance(bound, BoundKernel)
                self.assertGreater(compile_fn.call_count, 0)
                self.assertIsNone(bound.env.cache_dependencies.snapshot())
                self.assertIsNone(bound._generated_code)
            self.assertNotEqual(
                self.sources[first_source], self.sources[first_source + 1]
            )
            self.assertIn("2.0", self.sources[first_source])
            self.assertIn("3.0", self.sources[first_source + 1])
        self.assertFalse((self.root / "generated_code").exists())

    def test_external_torch_modes_reject_an_ordinary_saved_artifact(self) -> None:
        cold = self._bind_and_compile(_torch_add)
        snapshot = cache._dependency_snapshot(cold)
        self.assertIsNotNone(snapshot)
        with patch.object(
            KernelCompiler, "compile", side_effect=AssertionError("warm frontend ran")
        ):
            warm = self._bind_and_compile(_torch_add)
        self.assertIsInstance(warm, _CachedBoundKernel)
        self.assertEqual(self.sources[0], self.sources[1])

        for mode_type in (_FunctionScale, _DispatchScale):
            first_source = len(self.sources)
            for factor in (2.0, 3.0):
                with (
                    self.subTest(mode=mode_type.__name__, factor=factor),
                    mode_type(factor),
                    patch.object(
                        KernelCompiler,
                        "compile",
                        autospec=True,
                        side_effect=KernelCompiler.compile,
                    ) as compile_fn,
                ):
                    self.assertFalse(
                        cache.validate_kernel_dependencies(snapshot, _torch_add)
                    )
                    changed = self._bind_and_compile(_torch_add)
                self.assertIsInstance(changed, BoundKernel)
                self.assertGreater(compile_fn.call_count, 0)
                self.assertIsNone(changed.env.cache_dependencies.snapshot())
                self.assertIsNone(changed._generated_code)
            self.assertNotEqual(self.sources[0], self.sources[first_source])
            self.assertNotEqual(
                self.sources[first_source], self.sources[first_source + 1]
            )
            self.assertIn("2.0", self.sources[first_source])
            self.assertIn("3.0", self.sources[first_source + 1])

    def test_dependency_replay_does_not_invoke_dynamic_attributes(self) -> None:
        with patch.object(_SCALE_MODULE, "scale", 2.0):
            self._bind_and_compile(_direct_module_scale)

        read_attribute = Mock(
            side_effect=AssertionError("dependency replay invoked user code")
        )
        missing = ModuleType("_helion_admission_scale")
        missing.__getattr__ = read_attribute

        class DescriptorModule(ModuleType):
            @property
            def scale(self):
                return read_attribute()

        for replacement in (missing, DescriptorModule("_helion_admission_scale")):
            with (
                self.subTest(module_type=type(replacement).__name__),
                patch.dict(
                    _direct_module_scale.__globals__, {"_SCALE_MODULE": replacement}
                ),
                patch.object(
                    KernelCompiler,
                    "compile",
                    side_effect=RuntimeError("frontend fallback"),
                ),
                self.assertRaisesRegex(RuntimeError, "frontend fallback"),
            ):
                self._bind_and_compile(_direct_module_scale)
            read_attribute.assert_not_called()

    def test_module_namespace_values_are_not_admitted_as_constants(self) -> None:
        for fn in (_module_string, _module_identity):
            with self.subTest(fn=fn.__name__):
                bound = self._bind_and_compile(fn)
                self.assertIsNone(bound.env.cache_dependencies.snapshot())
                self.assertIsNone(bound._generated_code)
        with patch.dict(
            _module_identity.__globals__,
            {"_SCALE_ALIAS": ModuleType(_SCALE_MODULE.__name__)},
        ):
            changed = self._bind_and_compile(_module_identity)
            self.assertIsInstance(changed, BoundKernel)
            self.assertIsNone(changed.env.cache_dependencies.snapshot())
        self.assertNotEqual(self.sources[1], self.sources[2])
        self.assertFalse((self.root / "generated_code").exists())


@unittest.skipIf(not torch.cuda.is_available(), "requires CUDA and Triton")
@pytest.mark.cuda
@onlyBackends(["triton"])
class TestGeneratedCodeAdmissionCUDA(RefEagerTestDisabled, TestCase):
    @pytest.mark.slow
    def test_fresh_process_admission_and_exact_vllm_scaled_mm(self) -> None:
        root = Path(self._test_stack.enter_context(tempfile.TemporaryDirectory()))
        code = """
import importlib.util
import sys
from unittest.mock import patch
import torch

spec = importlib.util.spec_from_file_location("_helion_admission_repro", sys.argv[1])
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
case, phase, level = sys.argv[2:]
if case == "nested":
    scale = 3.0 if phase == "changed" else 2.0
    module._NESTED_HELPER = module._make_nested_helper(scale)
    kernel = module._kernel(module._nested_class_scale, cache_level=level)
    x = torch.ones(128, device="cuda")
    torch.testing.assert_close(kernel(x), x * scale)
    assert module.counters["generated_code_cache"]["frontend_hit"] == 0
    module._SCALE_MODULE.scale = scale
    direct = module._kernel(module._direct_module_scale, cache_level=level)
    if phase == "warm" and level == "code":
        with patch.object(module.KernelCompiler, "compile", side_effect=AssertionError("direct module warm frontend ran")):
            torch.testing.assert_close(direct(x), x * scale)
        assert module.counters["generated_code_cache"]["frontend_hit"] > 0
    else:
        with patch.object(module.KernelCompiler, "compile", autospec=True, side_effect=module.KernelCompiler.compile) as compile_fn:
            torch.testing.assert_close(direct(x), x * scale)
        assert compile_fn.call_count > 0
else:
    torch.manual_seed(0)
    a = torch.randint(-3, 4, (16, 64), device="cuda", dtype=torch.int8)
    b = torch.randint(-3, 4, (64, 64), device="cuda", dtype=torch.int8).t()
    a_scales = torch.full((16, 1), 0.5, device="cuda")
    b_scales = torch.full((64, 1), 0.25, device="cuda")
    out = torch.empty((16, 64), device="cuda", dtype=torch.bfloat16)
    expected = (a.float() @ b.float()) * a_scales * b_scales.T
    config = module.helion.Config(block_sizes=[16, 32, 32], split_k=1, swap_ab=False)
    kernel = module._kernel(module.scaled_mm, cache_level=level, config=config)
    if phase == "warm" and level == "code":
        with patch.object(module.KernelCompiler, "compile", side_effect=AssertionError("exact vLLM warm frontend ran")):
            kernel(out, a, b, a_scales, b_scales)
        assert module.counters["generated_code_cache"]["frontend_hit"] > 0
    else:
        kernel(out, a, b, a_scales, b_scales)
    torch.testing.assert_close(out, expected.to(out.dtype), atol=0, rtol=0)
"""
        for case in ("nested", "scaled_mm"):
            for level in ("config", "code"):
                phases = (
                    ("cold", "warm", "changed")
                    if case == "nested"
                    else ("cold", "warm")
                )
                for phase in phases:
                    with self.subTest(case=case, cache_level=level, phase=phase):
                        process = subprocess.run(
                            [sys.executable, "-c", code, __file__, case, phase, level],
                            env={
                                **os.environ,
                                "HELION_CACHE_DIR": str(root / case / level),
                                "HELION_SKIP_CACHE": "0",
                                "HELION_BACKEND": "triton",
                                "ENABLE_TILE": "0",
                            },
                            capture_output=True,
                            text=True,
                            timeout=120,
                        )
                        self.assertEqual(
                            process.returncode, 0, process.stdout + process.stderr
                        )
