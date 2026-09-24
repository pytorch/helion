from __future__ import annotations

import ast
import os
import subprocess
import sys
from typing import Any
from unittest.mock import patch

import pytest
import torch

from .test_cute_chained_pointwise import _args
from .test_cute_chained_pointwise import _code
from .test_cute_chained_pointwise import _config
from .test_cute_chained_pointwise import _pointwise_dot
from .test_cute_chained_tcgen05 import _compile
import helion
from helion._compiler.cute.mma_support import get_cute_mma_support
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends
import helion.language as hl

pytestmark = skipUnlessBackends(["cute"])


def _mixed_args(
    device: Any, kind: str, dtype: torch.dtype = torch.bfloat16, seed: int = 0
) -> tuple[Any, ...]:
    a, b, scale, bias, transpose = _args(device, kind, dtype)
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
    code = _code(_mixed_args("cpu", kind, dtype))
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
    assert "_pointwise_copy =" not in _code(_mixed_args("cpu", "dense"), False)


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
        [sys.executable, "-m", __name__, kind, str(dtype).split(".")[-1]],
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
            run = _pointwise_dot._bind_isolated(args).compile_config(_config())
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
    code = _code(_exact_args(kind, dtype), kernel=_weighted_identity_dot)
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
    code = _code(_exact_args("dense", torch.bfloat16), False, _weighted_identity_dot)
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
            run = _weighted_identity_dot._bind_isolated(args).compile_config(_config())
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
    run = _weighted_identity_dot._bind_isolated(initial).compile_config(_config())
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


if __name__ == "__main__":
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    assert not torch.cuda.is_initialized()
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[sys.argv[2]]
    args = _mixed_args("cpu", sys.argv[1], dtype)
    with patch.object(
        torch.cuda, "_lazy_init", side_effect=AssertionError("CUDA forbidden")
    ):
        ptx = _compile(_code(args), args, None, entry="_pointwise_dot")
    assert "ld.global.v4.b32" in ptx
    assert "st.shared.v4.b32" in ptx
