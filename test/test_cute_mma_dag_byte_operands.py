"""CPU source-generation coverage for typed byte MMA operand staging."""

from __future__ import annotations

import ast
from unittest.mock import patch

import pytest
import torch

import helion
from helion._testing import default_cute_mma_support
from helion._testing import pairwise
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import cpu_target
from helion._testing.cute_mma_dag_bytes import byte_matmul
from helion._testing.cute_mma_dag_bytes import byte_rhs_tiled_matmul
from helion._testing.cute_mma_dag_bytes import require_byte_mma_mode

pytestmark = pytest.mark.cpu_only


# Pairwise coverage spans both emitters, signedness, operand orientation, and
# aligned packet versus unaligned scalar loading without a full product.
@pytest.mark.parametrize(
    "mode,dtype,byte_rhs,transpose_byte,offset,output_dtype",
    pairwise(
        mode=("warp", "tcgen05"),
        dtype=(torch.int8, torch.uint8),
        byte_rhs=(False, True),
        transpose_byte=(False, True),
        offset=(0, 1),
        output_dtype=(torch.bfloat16, torch.float16),
    ),
)
def test_byte_operand_source(
    mode: str,
    dtype: torch.dtype,
    byte_rhs: bool,
    transpose_byte: bool,
    offset: int,
    output_dtype: torch.dtype,
) -> None:
    byte = torch.empty((64, 80), dtype=dtype)[:, offset : offset + 64]
    other = torch.empty((64, 64), dtype=output_dtype)
    args = (other, byte) if byte_rhs else (byte, other)
    with cpu_target():
        bound = cpu_bind(
            helion.kernel(byte_matmul, backend="cute", static_shapes=True),
            (*args, byte_rhs, transpose_byte, False),
        )
        with bound.env:
            config = bound.config_spec.normalized_config(
                helion.Config(
                    block_sizes=[64],
                    cute_mma_dag=mode,
                    cute_mma_dag_warps=4,
                    cute_mma_dag_stages=1,
                )
            )
        source = bound.to_code(config)
    assert "_helion_mma_dag" in source
    assert "md_byte" in source
    if offset:
        op = "ldg_s8_pred" if dtype == torch.int8 else "ldg_u8_pred"
        assert f".{op}(" in source
    else:
        assert ".lds_v4_b32(" in source
        assert ".sts_v4_b32(" in source


def test_byte_to_fp32_keeps_the_generic_fallback() -> None:
    # The new packet protocol writes 16-bit operands. A tf32 destination
    # has a separate transpose/scatter contract and must not enter it.
    args = (
        torch.empty((64, 64), dtype=torch.int8),
        torch.empty((64, 64), dtype=torch.float32),
        False,
        False,
        False,
    )
    with cpu_target():
        bound = cpu_bind(
            helion.kernel(byte_matmul, backend="cute", static_shapes=True), args
        )
        assert bound.config_spec.cute_mma_dag is None
        bound.to_code(helion.Config(block_sizes=[64]))


def test_byte_arithmetic_precedes_widening() -> None:
    # One source render witnesses wrapping byte arithmetic in the CPU lane;
    # the native module covers the overflow values for both signednesses.
    args = (
        torch.empty((64, 64), dtype=torch.int8),
        torch.empty((64, 64), dtype=torch.bfloat16),
        False,
        False,
        True,
    )
    with cpu_target():
        bound = cpu_bind(
            helion.kernel(byte_matmul, backend="cute", static_shapes=True), args
        )
        source = bound.to_code(
            helion.Config(
                block_sizes=[64],
                cute_mma_dag="warp",
                cute_mma_dag_warps=4,
                cute_mma_dag_stages=1,
            )
        )
    assignments = {
        node.targets[0].id: node.value
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
    }
    byte_ones = {
        name
        for name, value in assignments.items()
        if ast.unparse(value) == "cutlass.Int8(1)"
    }
    assert byte_ones
    assert any(
        isinstance(value, ast.BinOp)
        and isinstance(value.op, ast.Add)
        and isinstance(value.right, ast.Name)
        and value.right.id in byte_ones
        and isinstance(value.left, ast.Name)
        and value.left.id.startswith("md_byte")
        for value in assignments.values()
    )


@pytest.mark.parametrize("columns", [8, 32])
def test_aligned_short_byte_rows_use_scalar_conversion(columns: int) -> None:
    # The address and pitch are packet aligned, but each RHS row is only
    # eight bytes. It has no 16-byte packet to stage in the raw tile.
    other = torch.empty((64, 64), dtype=torch.bfloat16)
    byte = torch.empty((64, max(16, columns)), dtype=torch.int8)[:, :columns]
    # The shared dot frontend currently requires N >= 16. Target the
    # warp owner's actual 16x8x16 atom to exercise its smaller legal tile,
    # including an 8-column tile of a wider tensor.
    with (
        cpu_target(),
        patch("helion._compat._min_dot_size", return_value=(16, 8, 16)),
    ):
        bound = cpu_bind(
            helion.kernel(byte_rhs_tiled_matmul, backend="cute", static_shapes=True),
            (other, byte),
        )
        source = bound.to_code(
            helion.Config(
                block_sizes=[64, 8],
                cute_mma_dag="warp",
                cute_mma_dag_warps=4,
                cute_mma_dag_stages=1,
            )
        )
    assert "_helion_mma_dag" in source
    assert ".ldg_s8_pred(" in source


@pytest.mark.parametrize(
    "mode,capability,supported,skip",
    [
        ("warp", (8, 0), False, False),
        ("tcgen05", (9, 0), True, True),
        ("tcgen05", (10, 0), True, False),
        ("tcgen05", (10, 0), False, True),
        ("tcgen05", (12, 0), True, True),
    ],
)
def test_native_byte_mma_mode_guard(
    mode: str, capability: tuple[int, int], supported: bool, skip: bool
) -> None:
    with (
        patch("torch.cuda.get_device_capability", return_value=capability),
        patch(
            "helion._testing.cute_mma_dag_bytes.get_cute_mma_support",
            return_value=default_cute_mma_support(tcgen05_f16bf16=supported),
        ),
    ):
        if skip:
            with pytest.raises(pytest.skip.Exception, match="SM100-family"):
                require_byte_mma_mode(mode)
        else:
            require_byte_mma_mode(mode)
