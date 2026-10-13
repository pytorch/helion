"""Candidate analysis must scale without sharing mutable resolution state."""

from __future__ import annotations

import ast
from unittest.mock import patch

import pytest

from helion._compiler.cute.hoist_lane_invariant_reductions import (
    _build_vector_candidates,
)
from helion._compiler.cute.hoist_lane_invariant_reductions import _constexpr_loop_info
from helion._compiler.cute.hoist_lane_invariant_reductions import _statement_effects
from helion._compiler.cute.hoist_lane_invariant_reductions import _VectorCandidate


def _loop(count: int) -> ast.For:
    source = "for row_lane in range(16):\n    base = cutlass.Float32(scale)\n"
    for index in range(count):
        source += (
            f"    for lane_{index} in cutlass.range_constexpr(8):\n"
            f"        value_{index} = cute.math.exp2(base + cutlass.Float32(lane_{index}))\n"
            f"        result_{index} = value_{index} + cutlass.Float32(1)\n"
            f"        (out.iterator + row_lane).store(result_{index})\n"
        )
    node = ast.parse(source).body[0]
    assert isinstance(node, ast.For)
    return node


def _candidates(loop: ast.For) -> list[_VectorCandidate]:
    infos = [
        info
        for index, stmt in enumerate(loop.body)
        if (info := _constexpr_loop_info(stmt, index)) is not None
    ]
    return _build_vector_candidates(
        loop, infos, {}, {}, {"out": "cutlass.Float32"}, {"out"}, {"scale"}, {}
    )


@pytest.mark.parametrize("count", (2, 32))
def test_outer_effect_analysis_is_independent_of_candidate_count(count: int) -> None:
    loop = _loop(count)
    before = ast.dump(loop, include_attributes=True)
    with patch(
        "helion._compiler.cute.hoist_lane_invariant_reductions._statement_effects",
        wraps=_statement_effects,
    ) as effects:
        candidates = _candidates(loop)
    assert len(candidates) == count * 2
    assert sum(call.args[0] is loop for call in effects.call_args_list) == 1
    assert ast.dump(loop, include_attributes=True) == before
    # A later resolver cannot append its selected statements to an earlier one.
    first_ids = candidates[0].selected_statement_ids
    assert len(first_ids) == 2
    assert all(
        candidate.selected_statement_ids is not first_ids
        for candidate in candidates[1:]
    )


def test_loop_facts_are_recomputed_after_mutation_between_collections() -> None:
    loop = _loop(2)
    assert len(_candidates(loop)) == 4
    loop.body.append(ast.parse("scale = cutlass.Float32(0)").body[0])
    assert not _candidates(loop)
    loop.body.pop()
    assert len(_candidates(loop)) == 4


def test_transitive_alias_mutation_rejects_every_dependent_candidate() -> None:
    loop = _loop(2)
    loop.body[:0] = ast.parse("alias_2 = alias_1\nalias_1 = scale").body
    loop.body.append(ast.parse("alias_2[0] = cutlass.Float32(0)").body[0])
    assert not _candidates(loop)


def test_no_effect_analysis_when_no_assignment_is_used_later() -> None:
    loop = _loop(2)
    for node in loop.body:
        if isinstance(node, ast.For):
            node.body[:] = node.body[:1]
    with patch(
        "helion._compiler.cute.hoist_lane_invariant_reductions._statement_effects",
        wraps=_statement_effects,
    ) as effects:
        assert not _candidates(loop)
    assert all(call.args[0] is not loop for call in effects.call_args_list)
