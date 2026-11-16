"""CPU generation for a grouped physical axis with logical operand shape."""

from __future__ import annotations

import ast

import pytest
import torch

import helion
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import cpu_target
from helion._testing.cute_mma_dag_grouped import grouped_carry_matmul
from helion._testing.cute_mma_dag_grouped import grouped_loop_matmul
from helion._testing.cute_mma_dag_grouped import grouped_matmul
from helion._testing.cute_mma_dag_grouped import grouped_vector_matmul

pytestmark = pytest.mark.cpu_only


def _grouped_loads(
    source: str, group: int, *, axis_base: bool = False
) -> list[ast.Call]:
    """Match grouped pointer coordinates emitted by ``plain_address``."""
    loads = []
    for call in ast.walk(ast.parse(source)):
        if not (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr.startswith("ldg_")
            and call.args
        ):
            continue
        for coordinate in ast.walk(call.args[0]):
            if not (
                isinstance(coordinate, ast.BinOp)
                and isinstance(coordinate.op, ast.FloorDiv)
                and isinstance(coordinate.right, ast.Constant)
                and coordinate.right.value == group
                and isinstance(coordinate.left, ast.BinOp)
                and isinstance(coordinate.left.op, ast.Add)
            ):
                continue
            base = coordinate.left.left
            if (
                isinstance(base, ast.Name)
                and base.id.startswith("md_a")
                and base.id[4:].isdigit()
            ) or (not axis_base and isinstance(base, ast.Constant) and base.value == 0):
                loads.append(call)
                break
    assert loads, f"no scalar load pointer divides its global coordinate by {group}"
    return loads


@pytest.mark.parametrize("mode", ["warp", "tcgen05"])
@pytest.mark.parametrize("row_group,group", [(False, 32), (True, 3)])
def test_grouped_byte_scale_source(mode: str, row_group: bool, group: int) -> None:
    m, k = 128, 64
    shape = (
        ((m + group - 1) // group, k) if row_group else (m, (k + group - 1) // group)
    )
    args = (
        torch.empty((m, k), dtype=torch.int8),
        torch.empty((k, 64), dtype=torch.bfloat16),
        torch.empty(shape, dtype=torch.uint8),
        group,
        row_group,
        False,
    )
    with cpu_target():
        bound = cpu_bind(
            helion.kernel(grouped_matmul, backend="cute", static_shapes=True), args
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
    assert ".ldg_u8_pred(" in source
    _grouped_loads(source, group, axis_base=row_group)


@pytest.mark.parametrize("mode", ["warp", "tcgen05"])
def test_grouped_loop_scale_source(mode: str) -> None:
    args = (
        torch.empty((64, 128), dtype=torch.int8),
        torch.empty((128, 64), dtype=torch.bfloat16),
        torch.empty((64, 43), dtype=torch.uint8),
        3,
    )
    with cpu_target():
        bound = cpu_bind(
            helion.kernel(grouped_loop_matmul, backend="cute", static_shapes=True), args
        )
        with bound.env:
            config = bound.config_spec.normalized_config(
                helion.Config(
                    block_sizes=[64, 64],
                    cute_mma_dag=mode,
                    cute_mma_dag_warps=4,
                    cute_mma_dag_stages=2,
                )
            )
        source = bound.to_code(config)
    assert "_helion_mma_dag" in source
    assert ".ldg_u8_pred(" in source
    _grouped_loads(source, 3, axis_base=True)


@pytest.mark.parametrize("dtype", [torch.float16, torch.int32])
def test_grouped_nonbyte_auxiliary_source(dtype: torch.dtype) -> None:
    args = (
        torch.empty((128, 64), dtype=torch.int8),
        torch.empty((64, 64), dtype=torch.bfloat16),
        torch.empty((128, 2), dtype=dtype),
        32,
        False,
        False,
    )
    with cpu_target():
        bound = cpu_bind(
            helion.kernel(grouped_matmul, backend="cute", static_shapes=True), args
        )
        source = bound.to_code(helion.Config(block_sizes=[64], cute_mma_dag="warp"))
    assert "_helion_mma_dag" in source
    loads = _grouped_loads(source, 32)
    assert all(
        any(
            keyword.arg == "nc"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is True
            for keyword in call.keywords
        )
        for call in loads
    )


@pytest.mark.parametrize(
    "columns,group,index_offset", [(1, 32, 0), (64, -1, 0), (3, 32, 1)]
)
def test_unsupported_grouped_index_keeps_generic_fallback(
    columns: int, group: int, index_offset: int
) -> None:
    args = (
        torch.empty((128, 64), dtype=torch.int8),
        torch.empty((64, 64), dtype=torch.bfloat16),
        torch.empty((128, columns), dtype=torch.uint8),
        group,
        False,
        False,
        index_offset,
    )
    with cpu_target():
        bound = cpu_bind(
            helion.kernel(grouped_matmul, backend="cute", static_shapes=True), args
        )
        assert bound.config_spec.cute_mma_dag is None
        bound.to_code(helion.Config(block_sizes=[64]))


@pytest.mark.parametrize(
    "dtype,grouped", [(torch.float32, True), (torch.uint8, True), (torch.uint8, False)]
)
def test_vector_auxiliary_protocols_are_not_extended(
    dtype: torch.dtype, grouped: bool
) -> None:
    args = (
        torch.empty((64, 128), dtype=torch.int8),
        torch.empty((128, 64), dtype=torch.bfloat16),
        torch.empty((43 if grouped else 128,), dtype=dtype),
        grouped,
    )
    with cpu_target():
        bound = cpu_bind(
            helion.kernel(grouped_vector_matmul, backend="cute", static_shapes=True),
            args,
        )
        assert bound.config_spec.cute_mma_dag is None
        bound.to_code(helion.Config(block_sizes=[64, 64]))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("grouped", [False, True])
def test_grouped_raw_carry_seed_keeps_generic_fallback(
    dtype: torch.dtype, grouped: bool
) -> None:
    args = (
        torch.empty((64, 128), dtype=torch.bfloat16),
        torch.empty((128, 64), dtype=torch.bfloat16),
        torch.empty((64, 32 if grouped else 64), dtype=dtype),
        grouped,
    )
    with cpu_target():
        bound = cpu_bind(
            helion.kernel(grouped_carry_matmul, backend="cute", static_shapes=True),
            args,
        )
        if grouped:
            assert bound.config_spec.cute_mma_dag is None
            bound.to_code(helion.Config(block_sizes=[64, 64]))
        else:
            assert bound.config_spec.cute_mma_dag is not None
            source = bound.to_code(
                helion.Config(block_sizes=[64, 64], cute_mma_dag="warp")
            )
            assert "_helion_mma_dag" in source
