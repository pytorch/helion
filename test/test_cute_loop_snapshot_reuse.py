from __future__ import annotations

import ast
from copy import deepcopy
from unittest.mock import patch

import pytest

from helion._compiler.cute import persistent_branch_vec as branch


def _loop(mutation: str = "", *, budget: bool = False) -> ast.For:
    prefix = ""
    if budget:
        prefix = "term_0 = base\n" + "\n".join(
            f"term_{i} = term_{i - 1} + term_{i - 1}" for i in range(1, 13)
        )
    lines = f"""
{prefix}
lane_index = base + cutlass.Int32(synthetic_lane_7)
a = _helion_persistent_branch_vec_load(7, 4, 'cutlass.Float32', '', x.iterator + lane_index, (x.iterator + lane_index).load())
b = _helion_persistent_branch_vec_load(7, 4, 'cutlass.Float32', '', y.iterator + lane_index, (y.iterator + lane_index).load())
{mutation}
_helion_persistent_branch_vec_store(7, 4, 'cutlass.Float32', x.iterator + lane_index, a + b, None)
"""
    body = "\n".join("    " + line for line in lines.splitlines() if line)
    node = ast.parse(
        "for synthetic_lane_7 in cutlass.range_constexpr(4):\n" + body
    ).body[0]
    assert isinstance(node, ast.For)
    return node


def _transform(loop: ast.For, *, reuse: bool) -> tuple[str, list[str]]:
    original_snapshots = branch._definition_snapshots
    seen = []

    def snapshot(body, **kwargs):
        seen.append(ast.dump(ast.Module(body=body, type_ignores=[])))
        return original_snapshots(body, **kwargs)

    load_check = branch._BranchLocalPersistentVectorizer._has_aliasing_write_in_loop
    store_check = (
        branch._BranchLocalPersistentVectorizer._has_other_aliasing_access_in_loop
    )

    def check_load(*args, **kwargs):
        if not reuse:
            kwargs.pop("get_definition_snapshots", None)
        return load_check(*args, **kwargs)

    def check_store(*args, **kwargs):
        if not reuse:
            kwargs.pop("get_definition_snapshots", None)
        return store_check(*args, **kwargs)

    with (
        patch.object(branch, "_definition_snapshots", snapshot),
        patch.object(
            branch._BranchLocalPersistentVectorizer,
            "_has_aliasing_write_in_loop",
            check_load,
        ),
        patch.object(
            branch._BranchLocalPersistentVectorizer,
            "_has_other_aliasing_access_in_loop",
            check_store,
        ),
    ):
        result = branch.vectorize_branch_local_persistent_fragments(
            [deepcopy(loop)], proven_disjoint_tensor_pairs={frozenset(("x", "y"))}
        )
    return ast.unparse(ast.Module(body=result, type_ignores=[])), seen


@pytest.mark.parametrize("budget", [False, True])
@pytest.mark.parametrize(
    "mutation",
    [
        "",
        "lane_index = base",
        "lane_index += 1",
        "alias = lane_index\nalias.value = 0",
        "lane_index = lane_index + 1",
        "(x.iterator + lane_index + 1).store(a)",
        "cute.arch.atomic_add(x.iterator + lane_index, b)",
        "(unknown.iterator + lane_index).store(a)",
    ],
)
def test_local_reuse_preserves_original_alias_effect_and_budget_decisions(
    mutation, budget
):
    loop = _loop(mutation, budget=budget)
    original, original_calls = _transform(loop, reuse=False)
    candidate, candidate_calls = _transform(loop, reuse=True)
    assert candidate == original
    assert len(candidate_calls) <= len(original_calls)
    if budget:
        assert len(candidate_calls) == 1
        assert 2 <= len(original_calls) <= 3
        assert "_persistent_branch_vec" not in candidate
        assert "_persistent_branch_store_values" not in candidate


def test_successful_in_place_load_rewrite_invalidates_snapshot():
    loop = _loop()
    original, original_calls = _transform(loop, reuse=False)
    candidate, candidate_calls = _transform(loop, reuse=True)
    assert candidate == original
    assert "cute.arch.load" in candidate
    assert len(candidate_calls) == 3
    assert candidate_calls == original_calls
    # Two successful load replacements change the reaching expressions. The
    # store check must see their register extracts, not the original loads.
    assert len(set(candidate_calls)) == 3
    assert "_persistent_branch_vec" in candidate_calls[-1]


def test_snapshot_reuse_does_not_escape_the_current_loop_call():
    loop = _loop()
    first, first_calls = _transform(loop, reuse=True)
    assert "cute.arch.load" in first
    loop.body.insert(-1, ast.parse("lane_index = base").body[0])
    second, second_calls = _transform(loop, reuse=True)
    fresh, _ = _transform(loop, reuse=False)
    assert second == fresh and second != first
    assert second_calls and first_calls[0] != second_calls[0]
