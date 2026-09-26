from __future__ import annotations

import ast
from collections import Counter
import os
import re
import subprocess
import sys
from typing import Any
from unittest.mock import patch

import pytest
import torch

from test.test_cute_chained_scan_export import _scan_export
from test.test_cute_chained_scan_export import _scan_export_args
from test.test_cute_chained_scan_export import _scan_export_config
from test.test_cute_chained_scan_export import _scan_export_cpu_codegen
from test.test_cute_chained_tcgen05 import _tcgen_compile

import helion
from helion import exc
from helion._compiler.autotuner_heuristics.cute import CuteChainedMatmulHeuristic
from helion._compiler.cute.chained_pointwise_inplace import PointwiseInplace
from helion._compiler.cute.chained_pointwise_inplace import sw128_ownership
from helion._compiler.cute.chained_pointwise_unroll import PointwiseUnroll
from helion._compiler.cute.chained_tcgen05 import _VectorLeaf
from helion._compiler.cute.mma_support import get_cute_mma_support
from helion._compiler.cute.tcgen05_config import CuteTcgen05Config
from helion._testing import DEVICE
from helion._testing import patch_cute_mma_support
from helion._testing import skipUnlessBackends
from helion.autotuner.config_fragment import EnumFragment
from helion.autotuner.config_generation import ConfigGeneration
import helion.language as hl

pytestmark = skipUnlessBackends(["cute"])


# Pointwise.


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _pointwise_dot(
    a: torch.Tensor,
    b: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor,
    transpose: hl.constexpr,
) -> torch.Tensor:
    batch, rows, physical_reduction = a.shape
    reduction = scale.shape[1]
    columns = b.shape[1] if transpose else b.shape[2]
    out = torch.empty((batch, rows, columns), device=a.device, dtype=a.dtype)
    for bt, row, col in hl.tile([batch, rows, columns], block_size=[1, None, None]):
        bi = bt.begin
        kk = hl.arange(reduction)
        if transpose:
            right = b[bi, col, kk].T
        else:
            right = b[bi, kk, col]
        factor = torch.exp(scale[bi, kk].float()) * bias[bi].float()
        right = (right.float() * factor[:, None] + 1.0).to(a.dtype)
        left = (a[bi, row, kk].float() + 1.0).to(a.dtype)
        out[bi, row, col] = hl.dot(left, right).to(a.dtype)
    return out


def _pointwise_args(
    device: Any, kind: str, dtype: torch.dtype = torch.bfloat16
) -> tuple[Any, ...]:
    reduction = 49 if kind in ("tail", "padded") else 128
    physical = 64 if kind == "padded" else reduction
    generator = torch.Generator(device=device).manual_seed(936)
    a = (
        torch.randn((2, 128, physical), generator=generator, device=device, dtype=dtype)
        * 0.1
    )
    transpose = kind == "transpose"
    shape = (2, 64, physical) if transpose else (2, physical, 64)
    if kind == "stride":
        b = torch.randn(
            (*shape[:-1], shape[-1] * 2),
            generator=generator,
            device=device,
            dtype=dtype,
        )[..., ::2]
    elif kind == "offset":
        storage = torch.randn(
            (2 * reduction * 64 + 1,), generator=generator, device=device, dtype=dtype
        )
        b = storage[1:].view(shape)
    else:
        b = torch.randn(shape, generator=generator, device=device, dtype=dtype)
    scale = (
        torch.randn((2, reduction), generator=generator, device=device, dtype=dtype)
        * 0.05
    )
    bias = torch.randn((2,), generator=generator, device=device, dtype=dtype) * 0.1
    return a, b, scale, bias, transpose


def _pointwise_config(enabled: bool = True) -> helion.Config:
    return helion.Config(
        block_sizes=[128, 64],
        num_warps=4,
        cute_chained_mma_schedule="tcgen05_tmem",
        cute_chained_pointwise_vectorize=enabled,
    )


def _pointwise_code(
    args: tuple[Any, ...], enabled: bool = True, kernel: Any = _pointwise_dot
) -> str:
    with (
        patch_cute_mma_support(),
        patch("torch.cuda.is_available", return_value=False),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA forbidden")),
        patch(
            "helion._compiler.compile_environment.target_device_capability",
            return_value=(10, 3),
        ),
        patch("helion.runtime.get_num_sm", return_value=148),
        patch.object(
            CuteTcgen05Config, "per_cta_smem_capacity_bytes", return_value=232448
        ),
    ):
        return kernel._bind_isolated(args).to_code(_pointwise_config(enabled))


@pytest.mark.parametrize(
    "kind", ["dense", "transpose", "offset", "stride", "tail", "padded"]
)
def test_pointwise_codegen(kind: str) -> None:
    code = _pointwise_code(_pointwise_args("cpu", kind))
    assert ("chain_0_b_pointwise_copy =" in code) == (kind != "stride")
    if kind != "stride":
        assert "_pointwise_leaf_0_pointer.toint() % 16 == 0" in code
        assert ".layout.stride[" in code
        assert "_pointwise_element in cutlass.range_constexpr(8)" in code
        assert "for chain_0_b_step in cutlass.range" in code  # Masked fallback.
    if kind in ("tail", "padded"):
        assert "< 49" in code


def test_pointwise_default_retains_scalar_staging() -> None:
    assert "_pointwise_copy =" not in _pointwise_code(
        _pointwise_args("cpu", "dense"), False
    )


def test_pointwise_broadcast_factor_is_outside_vector_loop() -> None:
    code = _pointwise_code(_pointwise_args("cpu", "dense"))
    body = code.split("for chain_0_b_pointwise_step", 1)[1].split("else:", 1)[0]
    outside, inside = body.split("for chain_0_b_pointwise_element", 1)
    assert "cute.math.exp2" in outside
    assert "cute.math.exp2" not in inside


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("kind", ["dense", "transpose", "padded"])
def test_pointwise_real_cpu_compile(dtype: torch.dtype, kind: str) -> None:
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "CUTE_DSL_ARCH": "sm_103a",
        "CUTE_DSL_KEEP": "ptx",
    }
    result = subprocess.run(
        [sys.executable, "-m", __name__, "pointwise", kind, str(dtype).split(".")[-1]],
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "kind", ["dense", "transpose", "offset", "stride", "tail", "padded"]
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_pointwise_runtime(kind: str, dtype: torch.dtype) -> None:
    args = _pointwise_args(DEVICE, kind, dtype)
    a, b, scale, bias, transpose = args
    frozen = tuple(value.clone() for value in args[:-1])
    reduction = scale.shape[1]
    right = (b.transpose(-2, -1) if transpose else b)[:, :reduction]
    right = (
        right.float() * (scale.float().exp() * bias.float()[:, None])[:, :, None] + 1
    ).to(dtype)
    left = (a[:, :, :reduction].float() + 1.0).to(dtype)
    expected = (left.float() @ right.float()).to(dtype)
    run = _pointwise_dot._bind_isolated(args).compile_config(_pointwise_config())
    actual = run(*args)
    torch.testing.assert_close(actual, expected, atol=0.015, rtol=0.015)
    repeated = run(*args)
    assert actual.data_ptr() != repeated.data_ptr()
    torch.testing.assert_close(repeated, actual, atol=0, rtol=0)
    for before, value in zip(frozen, args[:-1], strict=True):
        torch.testing.assert_close(before, value, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_pointwise_reused_callable_checks_layout_and_alignment() -> None:
    args = _pointwise_args(DEVICE, "dense")
    run = _pointwise_dot._bind_isolated(args).compile_config(_pointwise_config())
    for kind in ("dense", "stride", "offset"):
        current = _pointwise_args(DEVICE, kind)
        a, b, scale, bias, _ = current
        right = (
            b.float() * (scale.float().exp() * bias.float()[:, None])[:, :, None] + 1
        ).to(a.dtype)
        expected = ((a.float() + 1).to(a.dtype).float() @ right.float()).to(a.dtype)
        torch.testing.assert_close(run(*current), expected, atol=0.015, rtol=0.015)


def _broadcast_args(args: tuple[Any, ...]) -> tuple[Any, ...]:
    a, b, scale, bias, transpose = args
    return a, b[:, :1, :].expand_as(b), scale, bias, transpose


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_pointwise_zero_outer_stride_codegen(dtype: torch.dtype) -> None:
    args = _broadcast_args(_pointwise_args("cpu", "dense", dtype))
    assert args[1].stride()[-2:] == (0, 1)
    code = _pointwise_code(args)
    assert "chain_0_b_pointwise_copy =" in code
    assert "stride=(0, 1)" in code
    assert "_pointwise_leaf_0_pointer.toint() % 16 == 0" in code
    assert ".layout.stride[1] == 0" in code


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("compile_broadcast", [False, True])
def test_pointwise_zero_outer_stride_runtime(
    dtype: torch.dtype, compile_broadcast: bool
) -> None:
    a, b, scale, bias, transpose = _pointwise_args(DEVICE, "dense", dtype)
    # A unit factor keeps an incorrect B row visible despite the +1 epilogue.
    dense = a, b, torch.zeros_like(scale), torch.ones_like(bias), transpose
    broadcast = _broadcast_args(dense)
    compile_args = broadcast if compile_broadcast else dense
    run = _pointwise_dot._bind_isolated(compile_args).compile_config(
        _pointwise_config()
    )
    for args in (broadcast, dense, broadcast):
        a, b, scale, bias, _ = args
        frozen = tuple(value.clone() for value in args[:-1])
        left = (a.float() + 1.0).to(dtype)
        right = (
            b.float() * (scale.float().exp() * bias.float()[:, None])[:, :, None] + 1.0
        ).to(dtype)
        expected = (left.float() @ right.float()).to(dtype)
        actual = run(*args)
        torch.testing.assert_close(actual, expected, atol=0.015, rtol=0.015)
        repeated = run(*args)
        assert actual.data_ptr() != repeated.data_ptr()
        torch.testing.assert_close(repeated, actual, atol=0, rtol=0)
        for before, value in zip(frozen, args[:-1], strict=True):
            torch.testing.assert_close(before, value, atol=0, rtol=0)


# Pointwise fp32.


def _mixed_args(
    device: Any, kind: str, dtype: torch.dtype = torch.bfloat16, seed: int = 0
) -> tuple[Any, ...]:
    a, b, scale, bias, transpose = _pointwise_args(device, kind, dtype)
    # Keep fractional values that would be lost by an early BF16 conversion.
    b = b.float() + 0.0038 + seed * 0.00013
    if kind == "offset":
        storage = torch.empty(b.numel() + 1, device=device, dtype=torch.float32)
        offset = storage[1:].view(b.shape)
        offset.copy_(b)
        b = offset
    elif kind == "stride":
        storage = torch.full(
            (*b.shape[:-1], b.shape[-1] * 2),
            float("nan"),
            device=device,
            dtype=torch.float32,
        )
        strided = storage[..., ::2]
        strided.copy_(b)
        b = strided
    elif kind == "broadcast":
        b = b[:, :1].expand_as(b)
    return a, b, scale.float(), bias.float(), transpose


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "kind", ["dense", "transpose", "offset", "stride", "tail", "padded", "broadcast"]
)
def test_fp32_pointwise_codegen(dtype: torch.dtype, kind: str) -> None:
    code = _pointwise_code(_mixed_args("cpu", kind, dtype))
    prefix = "chain_0_b_pointwise_leaf_0"
    assert (f"{prefix}_copy =" in code) == (kind != "stride")
    if kind != "stride":
        assert (
            f"{prefix}_copy = cute.make_tiled_copy_tv(cute.make_copy_atom("
            "cute.nvgpu.CopyUniversalOp(), cutlass.Float32, num_bits_per_copy=128)"
        ) in code
        assert f"{prefix}_thread.partition_S({prefix}_source)" in code
        assert (
            f"{prefix}_values = cute.make_rmem_tensor("
            f"{prefix}_partition[None, 0, 0].shape, cutlass.Float32)"
        ) in code
        assert f"cute.copy({prefix}_copy, {prefix}_partition" in code
        assert f"{prefix}_pointer.toint() % 16 == 0" in code
        assert "for chain_0_b_step in cutlass.range" in code
    if kind in ("tail", "padded"):
        assert "< 49" in code
    if kind == "broadcast":
        assert "stride=(0, 1)" in code


def test_fp32_pointwise_disabled_preserves_scalar_loads() -> None:
    assert "_pointwise_copy =" not in _pointwise_code(
        _mixed_args("cpu", "dense"), False
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("kind", ["dense", "transpose", "padded", "broadcast"])
def test_fp32_pointwise_cpu_compile(dtype: torch.dtype, kind: str) -> None:
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "CUTE_DSL_ARCH": "sm_103a",
        "CUTE_DSL_KEEP": "ptx",
    }
    result = subprocess.run(
        [sys.executable, "-m", __name__, "fp32", kind, str(dtype).split(".")[-1]],
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "kind", ["dense", "transpose", "offset", "stride", "padded", "broadcast"]
)
def test_fp32_pointwise_runtime(dtype: torch.dtype, kind: str) -> None:
    run = None
    for seed in range(5):
        args = _mixed_args(DEVICE, kind, dtype, seed)
        a, b, scale, bias, transpose = args
        before = tuple(value.clone() for value in args[:-1])
        if run is None:
            run = _pointwise_dot._bind_isolated(args).compile_config(
                _pointwise_config()
            )
        reduction = scale.shape[1]
        raw = (b.transpose(-2, -1) if transpose else b)[:, :reduction]
        factor = (scale.exp() * bias[:, None])[:, :, None]
        right = (raw * factor + 1.0).to(dtype)
        left = (a[:, :, :reduction].float() + 1.0).to(dtype)
        expected = (left.double() @ right.double()).to(dtype)
        actual = run(*args)
        torch.testing.assert_close(actual, expected, atol=0.015, rtol=0.015)
        repeated = run(*args)
        assert repeated.data_ptr() != actual.data_ptr()
        torch.testing.assert_close(repeated, actual, atol=0, rtol=0)
        torch.testing.assert_close(args[:-1], before, atol=0, rtol=0)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _weighted_identity_dot(
    left: torch.Tensor,
    values: torch.Tensor,
    addend: torch.Tensor,
    factor: torch.Tensor,
    transpose: hl.constexpr,
) -> torch.Tensor:
    batch, rows, reduction = left.shape
    columns = addend.shape[2]
    out = torch.empty((batch, rows, columns), device=left.device, dtype=left.dtype)
    for bt, row, col in hl.tile([batch, rows, columns], block_size=[1, None, None]):
        bi = bt.begin
        kk = hl.arange(reduction)
        if transpose:
            raw = values[bi, col, kk].T
        else:
            raw = values[bi, kk, col]
        weighted = (raw * factor[bi, kk][:, None] + addend[bi, kk, col].float()).to(
            left.dtype
        )
        out[bi, row, col] = hl.dot(left[bi, row, kk], weighted).to(left.dtype)
    return out


def _exact_args(
    kind: str, dtype: torch.dtype, seed: int = 0, device: Any = "cpu"
) -> tuple[Any, ...]:
    """Dyadic arithmetic makes FMA/non-FMA agree; only the final cast rounds."""
    batch, reduction, columns = 2, 128, 64
    left = torch.eye(reduction, dtype=dtype).expand(batch, -1, -1).clone()
    bi = torch.arange(batch, dtype=torch.float32)[:, None, None]
    row = torch.arange(reduction, dtype=torch.float32)[None, :, None]
    col = torch.arange(columns, dtype=torch.float32)[None, None, :]
    epsilon = 2**-10 if dtype == torch.bfloat16 else 2**-13
    logical = 1.0 + bi / 4 + row / 128 + ((col + seed) % 64) / 64 + 3 * epsilon
    addend = ((row % 8) / 16 + col / 128 + bi / 8).to(dtype)
    factor = (1.5 + (torch.arange(reduction) % 2).float() / 2).expand(batch, -1)
    transpose = kind == "transpose"
    raw = logical.transpose(1, 2).contiguous() if transpose else logical
    raw = raw.to(device)
    if kind == "offset":
        storage = torch.full(
            (raw.numel() + 1,), float("nan"), dtype=torch.float32, device=device
        )
        values = storage[1:].view(raw.shape)
        values.copy_(raw)
    elif kind == "stride":
        storage = torch.full(
            (*raw.shape[:-1], raw.shape[-1] * 2),
            float("nan"),
            dtype=torch.float32,
            device=device,
        )
        values = storage[..., ::2]
        values.copy_(raw)
    elif kind == "broadcast":
        values = raw[:, :1].expand_as(raw)
    else:
        assert kind in ("dense", "transpose")
        values = raw
    return left.to(device), values, addend.to(device), factor.to(device), transpose


def _expected(args: tuple[Any, ...], *, premature: bool = False) -> torch.Tensor:
    left, raw, addend, factor, transpose = args
    values = raw.transpose(1, 2) if transpose else raw
    if premature:
        values = values.to(left.dtype).float()
    # Identity-left contraction copies this operand exactly. No approximate
    # matrix multiplication or TF32-dependent oracle is needed.
    return (values * factor[:, :, None] + addend.float()).to(left.dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "kind", ["dense", "transpose", "offset", "stride", "broadcast"]
)
def test_exact_oracle_detects_rounding_and_four_eight_column_errors(
    dtype: torch.dtype, kind: str
) -> None:
    for seed in range(5):
        args = _exact_args(kind, dtype, seed)
        expected = _expected(args)
        assert torch.equal(args[0].double() @ expected.double(), expected.double())
        assert not torch.equal(expected, _expected(args, premature=True))
        groups = expected.reshape(2, 128, 8, 8)
        assert not torch.equal(expected, groups.roll(4, -1).reshape_as(expected))
        assert not torch.equal(expected, groups.roll(1, -2).reshape_as(expected))
        if kind == "offset":
            assert args[1].storage_offset() == 1 and args[1].data_ptr() % 16 == 4
        elif kind == "stride":
            assert args[1].stride(-1) == 2
        elif kind == "broadcast":
            assert args[1].stride(1) == 0


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("width", [32, 64, 128])
def test_fp32_and_narrow_copy_partitions_match_every_logical_slot(
    dtype: torch.dtype, width: int
) -> None:
    """Query actual CuTe TV algebra, not an assumed 4/8-element lane rule."""
    import cutlass
    from cutlass._mlir import ir
    import cutlass.cute as cute

    narrow = cutlass.BFloat16 if dtype == torch.bfloat16 else cutlass.Float16
    height = 128
    columns = width // 8
    rows = 128 // columns
    # MLIR exports these pybind types dynamically; its Python stub omits them.
    mlir: Any = ir
    with mlir.Context(), mlir.Location.unknown():
        module = mlir.Module.create()
        with mlir.InsertionPoint(module.body):
            copies = [
                cute.make_tiled_copy_tv(
                    cute.make_copy_atom(
                        cute.nvgpu.CopyUniversalOp(), element, num_bits_per_copy=128
                    ),
                    cute.make_layout((rows, columns), stride=(columns, 1)),
                    cute.make_layout((1, 8)),
                )
                for element in (cutlass.Float32, narrow)
            ]
            identity = cute.make_identity_tensor((height, width))
            visited: set[tuple[int, int]] = set()
            for thread in range(128):
                source = copies[0].get_slice(thread).partition_S(identity)
                target = copies[1].get_slice(thread).partition_D(identity)
                assert int(cute.size(source[None, 0, 0])) == 8
                assert int(cute.size(target[None, 0, 0])) == 8
                for step in range(height // rows):
                    source_values = source[None, step, 0]
                    target_values = target[None, step, 0]
                    for element in range(8):
                        source_row, source_col = source_values[element]
                        actual = (int(source_row), int(source_col))
                        assert actual == tuple(map(int, target_values[element]))
                        assert actual == (
                            thread // columns + step * rows,
                            thread % columns * 8 + element,
                        )
                        assert actual not in visited
                        visited.add(actual)
            assert len(visited) == height * width


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "kind", ["dense", "transpose", "offset", "stride", "broadcast"]
)
def test_mixed_leaf_codegen_uses_typed_partitions_and_retains_fallback(
    dtype: torch.dtype, kind: str
) -> None:
    code = _pointwise_code(_exact_args(kind, dtype), kernel=_weighted_identity_dot)
    tree = ast.parse(code)
    assignments = {
        node.targets[0].id: ast.unparse(node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
    }
    prefix = "chain_0_b_pointwise"
    float_copies = {
        name.removesuffix("_copy")
        for name, value in assignments.items()
        if name.startswith(prefix + "_leaf_")
        and name.endswith("_copy")
        and "cutlass.Float32" in value
    }
    assert bool(float_copies) == (kind != "stride")
    for name in float_copies:
        assert "num_bits_per_copy=128" in assignments[name + "_copy"]
        assert "cute.make_layout((1, 8))" in assignments[name + "_copy"]
        assert assignments[name + "_partition"] == (
            f"{name}_thread.partition_S({name}_source)"
        )
        assert assignments[name + "_values"] == (
            f"cute.make_rmem_tensor({name}_partition[None, 0, 0].shape, cutlass.Float32)"
        )
        assert f"{name}_pointer.toint() % 16 == 0" in code
    narrow = "cutlass.BFloat16" if dtype == torch.bfloat16 else "cutlass.Float16"
    assert narrow in assignments[prefix + "_copy"]
    narrow_leaves = [
        name.removesuffix("_values")
        for name, value in assignments.items()
        if name.startswith(prefix + "_leaf_")
        and name.endswith("_values")
        and value.endswith(f", {narrow})")
    ]
    # Transposed FP32 values choose the K-contiguous staging orientation;
    # the N-contiguous narrow addend then intentionally stays scalar.
    assert len(narrow_leaves) == (0 if kind == "transpose" else 1)
    for name in narrow_leaves:
        assert assignments[name + "_partition"].startswith(
            prefix + "_thread.partition_S("
        )
    # Host strides/alignment guard fast loads, and scalar staging remains.
    assert ".layout.stride[" in code
    assert "for chain_0_b_step in cutlass.range" in code
    if kind == "broadcast":
        assert "stride=(0, 1)" in code


def test_disabled_vectorization_keeps_the_same_explicit_operand_cast() -> None:
    code = _pointwise_code(
        _exact_args("dense", torch.bfloat16), False, _weighted_identity_dot
    )
    assert "_pointwise_copy =" not in code
    assert "for chain_0_b_step in cutlass.range" in code
    assert "cutlass.BFloat16(" in code


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("kind", ["dense", "transpose", "broadcast"])
def test_exact_mapping_runtime(dtype: torch.dtype, kind: str) -> None:
    if not get_cute_mma_support().tcgen05_f16bf16:
        pytest.skip("requires TCgen05 BF16/FP16 support")
    run = None
    for seed in range(5):
        args = _exact_args(kind, dtype, seed, DEVICE)
        expected = _expected(args)
        before = tuple(value.clone() for value in args[:-1])
        if run is None:
            run = _weighted_identity_dot._bind_isolated(args).compile_config(
                _pointwise_config()
            )
        actual = run(*args)
        repeated = run(*args)
        assert torch.equal(actual, expected)
        assert torch.equal(repeated, actual)
        assert actual.data_ptr() != repeated.data_ptr()
        saved = actual.clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = run(*args)
        for _ in range(3):
            captured.fill_(float("nan"))
            graph.replay()
            assert torch.equal(captured, expected)
        assert captured.data_ptr() not in (actual.data_ptr(), repeated.data_ptr())
        assert torch.equal(actual, saved)
        for value, original in zip(args[:-1], before, strict=True):
            assert torch.equal(value, original)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_exact_mapping_reused_callable_offset_stride_fallback(
    dtype: torch.dtype,
) -> None:
    if not get_cute_mma_support().tcgen05_f16bf16:
        pytest.skip("requires TCgen05 BF16/FP16 support")
    initial = _exact_args("dense", dtype, device=DEVICE)
    run = _weighted_identity_dot._bind_isolated(initial).compile_config(
        _pointwise_config()
    )
    for seed, kind in enumerate(("dense", "offset", "stride", "broadcast", "dense")):
        args = _exact_args(kind, dtype, seed, DEVICE)
        before = tuple(value.clone() for value in args[:-1])
        actual = run(*args)
        repeated = run(*args)
        assert torch.equal(actual, _expected(args))
        assert torch.equal(repeated, actual)
        assert actual.data_ptr() != repeated.data_ptr()
        for value, original in zip(args[:-1], before, strict=True):
            assert torch.equal(value, original)


# Pointwise unroll.

POINTWISE_UNROLL_KEY = "cute_chained_pointwise_unroll"

LOOP = re.compile(
    r"(for chain_\d+_[ab]_pointwise_step in cutlass.range\(\d+, unroll=)2(\):)"
)


def _config_unroll(factor: object = 1, enabled: bool = True) -> helion.Config:
    return helion.Config.from_dict(
        _pointwise_config(enabled).config | {POINTWISE_UNROLL_KEY: factor}
    )


def _unrolled_code(args: tuple, factor: object = 1, enabled: bool = True) -> str:
    with patch(
        "test.test_cute_chained_pointwise._pointwise_config",
        return_value=_config_unroll(factor, enabled),
    ):
        return _pointwise_code(args, enabled)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "kind", ["dense", "transpose", "offset", "stride", "tail", "padded", "broadcast"]
)
def test_only_existing_vector_loop_digit_changes(dtype: torch.dtype, kind: str) -> None:
    args = _mixed_args("cpu", kind, dtype)
    old = _pointwise_code(args)
    assert _unrolled_code(args) == old
    candidate = _unrolled_code(args, 2)
    restored, count = LOOP.subn(r"\g<1>1\2", candidate)
    assert count >= 1
    assert restored == old  # Complete source, including host, comments and fallback.
    assert "cutlass.Float32, num_bits_per_copy=128" in candidate or kind == "stride"


@pytest.mark.parametrize(
    "value", [True, False, 0, 3, 16, -1, 1.0, 2.0, 4.0, 8.0, "2", None]
)
def test_invalid_values_rejected(value: object) -> None:
    with pytest.raises(exc.InvalidConfig, match="must be 1, 2, 4 or 8"):
        _unrolled_code(_pointwise_args("cpu", "dense"), value)


def test_disabled_vectorization_canonicalizes_to_default() -> None:
    args = _pointwise_args("cpu", "dense")
    assert _unrolled_code(args, 2, False) == _pointwise_code(args, False)


def test_no_exact_vector_leaf_rejects_factor_two() -> None:
    a, b, scale, bias, transpose = _pointwise_args("cpu", "stride")
    # Leave scalar coefficient inputs intact so the structural discovery is a
    # superset; neither dense operand satisfies the actual coordinate proof.
    a = torch.empty((*a.shape[:-1], a.shape[-1] * 2), dtype=a.dtype)[..., ::2]
    args = (a, b, scale, bias, transpose)
    assert "_pointwise_copy =" not in _unrolled_code(args)
    with pytest.raises(exc.BackendUnsupported, match="admitted vector staging loop"):
        _unrolled_code(args, 2)


def test_activation_is_private_and_single_trip_is_inactive() -> None:
    first, second = PointwiseUnroll(2), PointwiseUnroll(2)
    assert first.loop_factor(1) == 1
    with pytest.raises(exc.BackendUnsupported):
        first.validate()
    assert first.loop_factor(2) == 2
    first.validate()
    with pytest.raises(exc.BackendUnsupported):
        second.validate()
    PointwiseUnroll(1).validate()


def test_direct_and_dot_derived_operands_do_not_advertise_unroll() -> None:
    from test.test_cute_chained_tcgen05 import _config_chain
    from test.test_cute_chained_tcgen05 import _tcgen_config_inputs as _inputs

    with patch_cute_mma_support():
        bound = _config_chain._bind_isolated((*_inputs(), None, False))
    spec = bound.config_spec
    assert not spec.cute_chained_pointwise_unroll_search_enabled
    assert POINTWISE_UNROLL_KEY not in spec._flat_fields()
    assert all(
        POINTWISE_UNROLL_KEY not in seed.config for seed in spec.compiler_seed_configs
    )
    with pytest.raises(exc.InvalidConfig, match="computed TCgen05 vector operands"):
        spec.normalized_config(_config_unroll(2))
    fixed = _config_unroll(2)
    spec.normalize(fixed, _fix_invalid=True)
    assert POINTWISE_UNROLL_KEY not in fixed.config


def test_search_preserves_old_order_and_reaches_both_factors() -> None:
    with patch_cute_mma_support():
        bound = _pointwise_dot._bind_isolated(_pointwise_args("cpu", "dense"))
    spec = bound.config_spec
    assert spec.cute_chained_pointwise_unroll_search_enabled
    fragment = spec._flat_fields()[POINTWISE_UNROLL_KEY]
    assert isinstance(fragment, EnumFragment)
    assert fragment.search_values() == [1, 2, 4, 8]
    assert bound.host_function is not None
    with bound.env:
        new = CuteChainedMatmulHeuristic.get_seed_configs(
            bound.env, bound.host_function.device_ir
        )
        spec.cute_chained_pointwise_unroll_search_enabled = False
        try:
            old = CuteChainedMatmulHeuristic.get_seed_configs(
                bound.env, bound.host_function.device_ir
            )
        finally:
            spec.cute_chained_pointwise_unroll_search_enabled = True
        assert old is not None and new is not None
        assert [
            seed for seed in new if seed.config.get(POINTWISE_UNROLL_KEY, 1) == 1
        ] == old
        assert new[0] == old[0]
        generation = ConfigGeneration(spec)
        population = [
            generation.unflatten(flat)
            for flat in generation.random_population_flat(100)
        ]
        factor2 = [
            seed for seed in population if seed.config.get(POINTWISE_UNROLL_KEY) == 2
        ]
        assert factor2
        for seed in factor2:
            assert seed.config["cute_chained_mma_schedule"] == "tcgen05_tmem"
            assert seed.config["cute_chained_pointwise_vectorize"] is True
            canonical = generation.unflatten(generation.flatten(seed))
            assert canonical.config[POINTWISE_UNROLL_KEY] == 2
        assert population[0].config[POINTWISE_UNROLL_KEY] == 1
    with patch(
        "test.test_cute_chained_pointwise._pointwise_config", return_value=factor2[0]
    ):
        assert LOOP.search(_pointwise_code(_pointwise_args("cpu", "dense"))) is not None


@pytest.mark.parametrize("schedule", ["coalesced", "cp_async_register"])
def test_other_schedules_canonicalize_factor_two(schedule: str) -> None:
    with patch_cute_mma_support():
        bound = _pointwise_dot._bind_isolated(_pointwise_args("cpu", "dense"))
    config = _config_unroll(2)
    config.config["cute_chained_mma_schedule"] = schedule
    normalized = bound.config_spec.normalized_config(config)
    assert normalized.config[POINTWISE_UNROLL_KEY] == 1
    assert normalized.config["cute_chained_pointwise_vectorize"] is False


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_original_operand_dtypes_keep_exact_source(dtype: torch.dtype) -> None:
    args = _pointwise_args("cpu", "dense", dtype)
    candidate = _unrolled_code(args, 2)
    restored, count = LOOP.subn(r"\g<1>1\2", candidate)
    assert count == 2
    assert restored == _pointwise_code(args)


@pytest.mark.parametrize("factor", [4, 8])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "kind", ["dense", "transpose", "offset", "stride", "broadcast"]
)
def test_larger_factors_change_only_the_existing_loop_digit(
    factor: int, dtype: torch.dtype, kind: str
) -> None:
    args = _mixed_args("cpu", kind, dtype)
    original = _unrolled_code(args, 1)
    candidate = _unrolled_code(args, factor)
    pattern = re.compile(
        rf"(for chain_\d+_[ab]_pointwise_step in cutlass.range\(\d+, unroll=){factor}(\):)"
    )
    restored, count = pattern.subn(r"\g<1>1\2", candidate)
    assert count >= 1 and restored == original


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("kind", ["tail", "padded"])
def test_shorter_real_staging_loop_accepts_four_but_rejects_eight(
    dtype: torch.dtype, kind: str
) -> None:
    args = _mixed_args("cpu", kind, dtype)
    source = _unrolled_code(args, 4)
    assert "cutlass.range(4, unroll=4)" in source
    assert "< 49" in source  # Keep all scalar fallback/tail predicates.
    with pytest.raises(exc.BackendUnsupported, match="whole number of unroll groups"):
        _unrolled_code(args, 8)


@pytest.mark.parametrize("factor", [4, 8])
@pytest.mark.parametrize("trips", [2, 3, 4, 6, 8, 12, 16])
def test_larger_factor_requires_complete_groups(factor: int, trips: int) -> None:
    tracker = PointwiseUnroll(factor)
    if trips < factor or trips % factor:
        with pytest.raises(
            exc.BackendUnsupported, match="whole number of unroll groups"
        ):
            tracker.loop_factor(trips)
        assert not tracker.activated
    else:
        assert tracker.loop_factor(trips) == factor
        tracker.validate()


@pytest.mark.parametrize("factor", [4, 8])
def test_larger_factor_cannot_hide_a_short_loop_after_activation(factor: int) -> None:
    tracker = PointwiseUnroll(factor)
    assert tracker.loop_factor(1) == 1
    with pytest.raises(exc.BackendUnsupported, match="admitted vector staging loop"):
        tracker.validate()
    assert tracker.loop_factor(16) == factor
    with pytest.raises(exc.BackendUnsupported, match="whole number of unroll groups"):
        tracker.loop_factor(2)


@pytest.mark.parametrize("factor", [1, 2])
@pytest.mark.parametrize("trips", [1, 2, 3, 5, 8, 16])
def test_existing_factor_activation_behavior_unchanged(factor: int, trips: int) -> None:
    tracker = PointwiseUnroll(factor)
    assert tracker.loop_factor(trips) == (1 if trips == 1 else factor)
    assert tracker.activated == (trips >= 2)


@pytest.mark.parametrize("factor", [4, 8])
def test_larger_inactive_configs_preserve_canonical_default(factor: int) -> None:
    args = _pointwise_args("cpu", "dense")
    assert _unrolled_code(args, factor, False) == _pointwise_code(args, False)
    with patch_cute_mma_support():
        bound = _pointwise_dot._bind_isolated(args)
    for schedule in ("coalesced", "cp_async_register"):
        config = _config_unroll(factor)
        config.config["cute_chained_mma_schedule"] = schedule
        normalized = bound.config_spec.normalized_config(config)
        assert normalized.config[POINTWISE_UNROLL_KEY] == 1
        assert normalized.config["cute_chained_pointwise_vectorize"] is False


def test_larger_seed_factors_preserve_old_pool_order_and_reach_initial_population() -> (
    None
):
    with patch_cute_mma_support():
        bound = _pointwise_dot._bind_isolated(_pointwise_args("cpu", "dense"))
    assert bound.host_function is not None
    reached_configs = []
    with bound.env:
        with patch(
            "helion._compiler.autotuner_heuristics.cute.VALID_CUTE_CHAINED_POINTWISE_UNROLLS",
            (1, 2),
        ):
            old = CuteChainedMatmulHeuristic.get_seed_configs(
                bound.env, bound.host_function.device_ir
            )
        new = CuteChainedMatmulHeuristic.get_seed_configs(
            bound.env, bound.host_function.device_ir
        )
        assert old is not None and new is not None
        assert [
            seed for seed in new if seed.config.get(POINTWISE_UNROLL_KEY, 1) in (1, 2)
        ] == old
        assert new[0] == old[0]
        old_two = [seed for seed in old if seed.config.get(POINTWISE_UNROLL_KEY) == 2]
        assert len(new) == len(old) + 2 * len(old_two)
        for factor in (4, 8):
            assert Counter(
                helion.Config.from_dict(seed.config | {POINTWISE_UNROLL_KEY: 2})
                for seed in new
                if seed.config.get(POINTWISE_UNROLL_KEY) == factor
            ) == Counter(old_two)
        generation = ConfigGeneration(bound.config_spec)
        population = [
            generation.unflatten(flat)
            for flat in generation.random_population_flat(100)
        ]
        assert (
            len(population) == 100 and population[0].config[POINTWISE_UNROLL_KEY] == 1
        )
        for factor in (2, 4, 8):
            reached = [
                candidate
                for candidate in population
                if candidate.config.get(POINTWISE_UNROLL_KEY) == factor
                and candidate.block_sizes == [128, 64]
            ]
            assert reached, factor
            assert all(
                seed.config["cute_chained_pointwise_vectorize"]
                and seed.config["cute_chained_mma_schedule"] == "tcgen05_tmem"
                for seed in reached
            )
            config = generation.unflatten(generation.flatten(reached[0]))
            assert config.config[POINTWISE_UNROLL_KEY] == factor
            reached_configs.append(config)
    for config in reached_configs:
        with patch(
            "test.test_cute_chained_pointwise._pointwise_config", return_value=config
        ):
            source = _pointwise_code(_pointwise_args("cpu", "dense"))
        assert f"unroll={config.config[POINTWISE_UNROLL_KEY]}" in source


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("kind", ["dense", "offset"])
def test_pointwise_unroll_runtime(dtype: torch.dtype, kind: str) -> None:
    if not get_cute_mma_support().tcgen05_f16bf16:
        pytest.skip("requires TCgen05")
    run = None
    for seed in range(5):
        args = _mixed_args(DEVICE, kind, dtype, seed)
        a, b, scale, bias, transpose = args
        before = tuple(value.clone() for value in args[:-1])
        if run is None:
            run = _pointwise_dot._bind_isolated(args).compile_config(_config_unroll(2))
        raw = b.transpose(-2, -1) if transpose else b
        right = (raw * (scale.exp() * bias[:, None])[:, :, None] + 1.0).to(dtype)
        left = (a.float() + 1.0).to(dtype)
        expected = (left.double() @ right.double()).to(dtype)
        actual, repeated = run(*args), run(*args)
        torch.testing.assert_close(actual, expected, atol=0.015, rtol=0.015)
        assert repeated.data_ptr() != actual.data_ptr()
        torch.testing.assert_close(repeated, actual, atol=0, rtol=0)
        torch.testing.assert_close(args[:-1], before, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("factor", [4, 8])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("kind", ["dense", "offset"])
def test_larger_pointwise_unroll_runtime(
    factor: int, dtype: torch.dtype, kind: str
) -> None:
    if not get_cute_mma_support().tcgen05_f16bf16:
        pytest.skip("requires TCgen05")
    run = None
    for seed in range(5):
        args = _mixed_args(DEVICE, kind, dtype, seed)
        a, b, scale, bias, transpose = args
        before = tuple(value.clone() for value in args[:-1])
        if run is None:
            run = _pointwise_dot._bind_isolated(args).compile_config(
                _config_unroll(factor)
            )
        raw = b.transpose(-2, -1) if transpose else b
        right = (raw * (scale.exp() * bias[:, None])[:, :, None] + 1.0).to(dtype)
        left = (a.float() + 1.0).to(dtype)
        expected = (left.double() @ right.double()).to(dtype)
        actual, repeated = run(*args), run(*args)
        torch.testing.assert_close(actual, expected, atol=0.015, rtol=0.015)
        assert repeated.data_ptr() != actual.data_ptr()
        torch.testing.assert_close(repeated, actual, atol=0, rtol=0)
        torch.testing.assert_close(args[:-1], before, atol=0, rtol=0)


# Pointwise inplace.

POINTWISE_INPLACE_KEY = "cute_chained_pointwise_inplace_async"


def code(args: tuple, value: object = False, *, enabled: bool = True) -> str:
    config = helion.Config.from_dict(
        _pointwise_config(enabled).config | {POINTWISE_INPLACE_KEY: value}
    )
    with patch(
        "test.test_cute_chained_pointwise._pointwise_config", return_value=config
    ):
        return _pointwise_code(args, enabled)


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize(
    "kind", ("dense", "transpose", "offset", "stride", "tail", "padded")
)
def test_source_default_identity_and_exact_math(dtype: torch.dtype, kind: str) -> None:
    args = _pointwise_args("cpu", kind, dtype)
    old = _pointwise_code(args)
    assert code(args) == old
    new = code(args, True)
    assert "_raw_copy =" in new
    a, b = ast.parse(old), ast.parse(new)
    assert ast.dump(a.body[-1]) == ast.dump(b.body[-1])

    def branches(tree: ast.AST) -> list[ast.If]:
        return [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.If) and "pointwise_leaf_" in ast.unparse(n.test)
        ]

    before, after = branches(a), branches(b)
    assert len(before) == len(after)
    for first, second in zip(before, after, strict=True):
        assert ast.dump(first.test) == ast.dump(second.test)
        assert ast.dump(ast.Module(first.orelse, type_ignores=[])) == ast.dump(
            ast.Module(second.orelse, type_ignores=[])
        )
        old_loop = next(n for n in first.body if isinstance(n, ast.For))
        new_loop = next(n for n in second.body if isinstance(n, ast.For))
        assert ast.dump(old_loop.iter) == ast.dump(new_loop.iter)

        def math_nodes(loop):
            return [
                ast.dump(n)
                for n in loop.body
                if not (
                    isinstance(n, ast.Expr)
                    and isinstance(n.value, ast.Call)
                    and ast.unparse(n.value.func) == "cute.copy"
                )
            ]

        assert math_nodes(old_loop) == math_nodes(new_loop)
        if "_raw_copy" in ast.unparse(second):
            prefix = [
                ast.unparse(n) for n in second.body[: second.body.index(new_loop)]
            ]
            pos = next(
                i
                for i, text in enumerate(prefix)
                if text.startswith("cute.copy(") and "_raw_copy" in text
            )
            assert prefix[pos + 1 : pos + 4] == [
                "cute.arch.cp_async_commit_group()",
                "cute.arch.cp_async_wait_group(0)",
                "cute.arch.sync_threads()",
            ]
    assert [line for line in old.splitlines() if "alloc_smem(" in line] == [
        line for line in new.splitlines() if "alloc_smem(" in line
    ]


@pytest.mark.parametrize("value", (0, 1, "true", None, 2))
def test_invalid_bool_rejected(value: object) -> None:
    with pytest.raises(exc.InvalidConfig, match="must be bool"):
        code(_pointwise_args("cpu", "dense"), value)


def test_disabled_vector_path_canonicalizes() -> None:
    args = _pointwise_args("cpu", "dense")
    assert code(args, True, enabled=False) == _pointwise_code(args, False)


@pytest.mark.parametrize(
    "shape,inner,expected",
    [
        ((128, 128), 0, True),
        ((128, 64), 1, True),
        ((64, 128), 0, True),
        ((128, 32), 1, False),
        ((12, 64), 1, False),
        ((128, 96), 1, False),
        ((128, 2048), 1, False),
    ],
)
def test_sw128_layout_envelope(shape, inner, expected) -> None:
    assert sw128_ownership(shape, inner) is expected


def test_typed_selection_and_private_activation() -> None:
    def leaf(dtype: torch.dtype, stride: int = 128) -> _VectorLeaf:
        return _VectorLeaf(
            torch.fx.Graph().placeholder("input"),
            ("row", "col"),
            "input",
            "0",
            stride,
            (),
            dtype,
        )

    first = PointwiseInplace(True)
    leaves = [leaf(torch.float32), leaf(torch.bfloat16)]
    assert first.select(leaves, (128, 128), 1, torch.bfloat16) == 1
    first.validate()
    other = PointwiseInplace(True)
    assert other.select(leaves[:1], (128, 128), 1, torch.bfloat16) is None
    with pytest.raises(exc.BackendUnsupported, match="same-dtype"):
        other.validate()
    PointwiseInplace(False).validate()
    assert (
        PointwiseInplace(True).select(
            [leaf(torch.float16)], (128, 128), 1, torch.bfloat16
        )
        is None
    )
    assert (
        PointwiseInplace(True).select(
            [leaf(torch.bfloat16, 0)], (128, 128), 1, torch.bfloat16
        )
        is None
    )


def test_actual_initial100_and_filtered_seed_order() -> None:
    with patch_cute_mma_support():
        bound = _pointwise_dot._bind_isolated(_pointwise_args("cpu", "dense"))
    spec = bound.config_spec
    assert bound.host_function is not None
    assert spec.cute_chained_pointwise_inplace_search_enabled
    with bound.env:
        new = CuteChainedMatmulHeuristic.get_seed_configs(
            bound.env, bound.host_function.device_ir
        )
        spec.cute_chained_pointwise_inplace_search_enabled = False
        try:
            old = CuteChainedMatmulHeuristic.get_seed_configs(
                bound.env, bound.host_function.device_ir
            )
        finally:
            spec.cute_chained_pointwise_inplace_search_enabled = True
        assert new is not None and old is not None
        assert [
            seed for seed in new if not seed.config.get(POINTWISE_INPLACE_KEY)
        ] == old
        generation = ConfigGeneration(spec)
        population = [
            generation.unflatten(value)
            for value in generation.random_population_flat(100)
        ]
        candidates = [
            seed for seed in population if seed.config.get(POINTWISE_INPLACE_KEY)
        ]
        assert candidates
        assert not population[0].config.get(POINTWISE_INPLACE_KEY)
        for candidate in candidates:
            assert (
                generation.unflatten(generation.flatten(candidate)).config[
                    POINTWISE_INPLACE_KEY
                ]
                is True
            )
    with patch(
        "test.test_cute_chained_pointwise._pointwise_config", return_value=candidates[0]
    ):
        assert "_raw_copy =" in _pointwise_code(_pointwise_args("cpu", "dense"))


def test_fp32_leaf_keeps_original_typed_load_path() -> None:
    args = _mixed_args("cpu", "dense")
    before, after = _pointwise_code(args), code(args, True)
    assert "chain_0_a_raw_copy" in after
    assert "chain_0_b_raw_copy" not in after
    begin, end = "    chain_0_b_ptr =", "    cute.arch.cp_async_commit_group()"
    assert before.split(begin)[1].split(end)[0] == after.split(begin)[1].split(end)[0]


def test_no_dense_16bit_leaf_rejects_explicit_request() -> None:
    a, b, scale, bias, transpose = _mixed_args("cpu", "dense")
    a = torch.empty((*a.shape[:-1], a.shape[-1] * 2), dtype=a.dtype)[..., ::2]
    with pytest.raises(exc.InvalidConfig, match="computed same-dtype TCgen05 operands"):
        code((a, b, scale, bias, transpose), True)


@pytest.mark.parametrize("factor", (1, 2, 4, 8))
def test_scan_export_cache_and_unroll_interaction(factor: int) -> None:
    config = _scan_export_config().config | {
        POINTWISE_INPLACE_KEY: True,
        "cute_chained_pointwise_vectorize": True,
        "cute_chained_pointwise_unroll": factor,
        "cute_chained_pointwise_read_cache": True,
        "cute_chained_auxiliary_cache": True,
    }
    with _scan_export_cpu_codegen():
        bound = _scan_export._bind_isolated((*_scan_export_args(), "normal"))
        old = bound.to_code(
            helion.Config.from_dict(config | {POINTWISE_INPLACE_KEY: False})
        )
        new = bound.to_code(helion.Config.from_dict(config))
    assert "_raw_copy =" in new and "_read_cache_" in new
    marker = "    if chain_origin_1 == 0 and chain_origin_2 == 0"
    assert marker in new and new.split(marker)[1] == old.split(marker)[1]


def test_direct_family_does_not_advertise_inplace() -> None:
    from test.test_cute_chained_tcgen05 import _config_chain
    from test.test_cute_chained_tcgen05 import _tcgen_config_inputs as _inputs

    with patch_cute_mma_support():
        bound = _config_chain._bind_isolated((*_inputs(), None, False))
    assert not bound.config_spec.cute_chained_pointwise_inplace_search_enabled
    assert POINTWISE_INPLACE_KEY not in bound.config_spec._flat_fields()
    with pytest.raises(exc.InvalidConfig, match="same-dtype TCgen05"):
        bound.config_spec.normalized_config(
            helion.Config.from_dict(
                _pointwise_config().config | {POINTWISE_INPLACE_KEY: True}
            )
        )


if __name__ == "__main__":
    command = sys.argv.pop(1)
    if command == "pointwise":
        assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
        assert not torch.cuda.is_initialized()
        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[sys.argv[2]]
        args = _pointwise_args("cpu", sys.argv[1], dtype)
        with patch.object(
            torch.cuda, "_lazy_init", side_effect=AssertionError("CUDA forbidden")
        ):
            ptx = _tcgen_compile(
                _pointwise_code(args), args, None, entry="_pointwise_dot"
            )
        assert "ld.global.v4.b32" in ptx
        assert "st.shared.v4.b32" in ptx
    elif command == "fp32":
        assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
        assert not torch.cuda.is_initialized()
        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[sys.argv[2]]
        args = _mixed_args("cpu", sys.argv[1], dtype)
        with patch.object(
            torch.cuda, "_lazy_init", side_effect=AssertionError("CUDA forbidden")
        ):
            ptx = _tcgen_compile(
                _pointwise_code(args), args, None, entry="_pointwise_dot"
            )
        assert "ld.global.v4.b32" in ptx
        assert "st.shared.v4.b32" in ptx
    else:
        raise AssertionError(f"Unknown test driver: {command}")
