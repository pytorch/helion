"""CPU proofs for gathered row coordinates on rolled vector packets."""

from __future__ import annotations

import ast
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import cast

import pytest

from helion._compiler.cute.memory_ops import _cute_looped_unroll_uniform_definitions
from helion._compiler.cute.memory_ops import _cute_relocate_above_vloop

if TYPE_CHECKING:
    from helion._compiler.inductor_lowering import CodegenState
    from helion._compiler.reduction_strategy import LoopedReductionStrategy


def _scope(source: str):
    body = ast.parse(source).body
    loop = ast.parse("for vector_lane in cutlass.range_constexpr(8):\n    pass").body[0]
    assert isinstance(loop, ast.For)
    loop.body = body
    lane_body = [loop]
    state = SimpleNamespace(codegen=SimpleNamespace(statements_stack=[body]))
    strategy = SimpleNamespace(
        _cute_lane_vloop=loop,
        _cute_lane_body=lane_body,
        _mask_var="mask",
        block_index=1,
        index_var=lambda block_index: "rindex",
    )
    return state, strategy


def _proof(source: str, mask: str = "row_copy < rows and mask"):
    state, strategy = _scope(source)
    return _cute_looped_unroll_uniform_definitions(
        cast("CodegenState", state),
        cast("LoopedReductionStrategy", strategy),
        ["cutlass.Int32(row_copy)", "rindex"],
        mask,
    )


def test_outer_row_copy_moves_above_packet() -> None:
    result = _proof("rindex = lane_base + cutlass.Int32(vector_lane)\nrow_copy = row\n")
    assert result is not None
    scope, proof = result
    assert [ast.unparse(statement) for statement in proof.moved] == ["row_copy = row"]
    _cute_relocate_above_vloop(scope, proof.moved)
    assert ast.unparse(scope.lane_body[0]) == "row_copy = row"
    assert all("row_copy =" not in ast.unparse(statement) for statement in scope.body)


def test_transitive_outer_aliases_keep_row_guard() -> None:
    result = _proof(
        "rindex = lane_base + cutlass.Int32(vector_lane)\n"
        "row_index_copy = logical_row\n"
        "sequence_copy = sequence\n"
        "row_copy = row_index_copy + sequence_copy\n"
    )
    assert result is not None
    _scope, proof = result
    assert len(proof.moved) == 3
    assert proof.uniform_mask == "(row_copy < rows)"


@pytest.mark.parametrize(
    "suffix",
    [
        "row_copy = row_copy + 1",
        "if active:\n    row_copy = row_copy + 1",
        "for row_copy in range(2):\n    pass",
        "(matrix.iterator + row).store(value)",
        "cute.arch.sync_threads()",
    ],
)
def test_rebinding_or_effects_decline(suffix: str) -> None:
    assert (
        _proof(
            "rindex = lane_base + cutlass.Int32(vector_lane)\n"
            "row_copy = row\n" + suffix + "\n"
        )
        is None
    )


def test_lane_dependent_row_or_mask_stays_scalar() -> None:
    assert (
        _proof(
            "rindex = lane_base + cutlass.Int32(vector_lane)\nrow_copy = rindex // 32\n"
        )
        is None
    )
    assert (
        _proof(
            "rindex = lane_base + cutlass.Int32(vector_lane)\n"
            "row_copy = row\n"
            "even = rindex % 2 == 0\n",
            "row_copy < rows and mask and even",
        )
        is None
    )
