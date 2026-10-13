"""Raw statement effects are reusable only inside one unchanged collection."""

from __future__ import annotations

import ast
from unittest.mock import patch

import pytest

from test.test_cute_vector_loop_analysis import _candidates
from test.test_cute_vector_loop_analysis import _loop

from helion._compiler.cute import hoist_lane_invariant_reductions as hoist


def _uncached(self: object, stmt: ast.stmt) -> frozenset[str]:
    return hoist._statement_effects(stmt).writes | hoist._mutation_roots(stmt)


def test_statement_summary_is_shared_between_resolvers_only() -> None:
    loop = _loop(8)
    facts = hoist._vector_loop_facts(loop)
    stmt = loop.body[1]
    before = ast.dump(loop, include_attributes=True)
    expected = _uncached(facts, stmt)
    with (
        patch.object(
            hoist, "_statement_effects", wraps=hoist._statement_effects
        ) as effects,
        patch.object(
            hoist, "_mutation_roots", wraps=hoist._mutation_roots
        ) as mutations,
    ):
        for _ in range(20):
            assert facts.writes_and_mutations(stmt) == expected
    assert sum(call.args[0] is stmt for call in effects.call_args_list) == 1
    assert sum(call.args[0] is stmt for call in mutations.call_args_list) == 1
    assert ast.dump(loop, include_attributes=True) == before
    assert hoist._vector_loop_facts(loop).statement_writes == {}


@pytest.mark.parametrize(
    "mutation",
    [
        "scale = cutlass.Float32(0)",
        "scale[0] = cutlass.Float32(0)",
        "scale.value = cutlass.Float32(0)",
        "alias = scale\nalias[0] = cutlass.Float32(0)",
        "if cond:\n    scale = cutlass.Float32(0)",
        "for nested in range(2):\n    scale += cutlass.Float32(1)",
    ],
)
def test_new_collection_rechecks_mutated_statements(mutation: str) -> None:
    loop = _loop(4)
    assert len(_candidates(loop)) == 8
    original = loop.body.copy()
    loop.body.extend(ast.parse(mutation).body)
    assert not _candidates(loop)
    loop.body[:] = original
    assert len(_candidates(loop)) == 8


@pytest.mark.parametrize("count", [2, 8, 32])
def test_cached_and_original_effect_queries_have_identical_provenance(
    count: int,
) -> None:
    loop = _loop(count)
    before = ast.dump(loop, include_attributes=True)
    with patch.object(hoist._VectorLoopFacts, "writes_and_mutations", _uncached):
        original = _candidates(loop)
    current = _candidates(loop)
    # Dataclass equality includes canonical expressions, actual statement IDs,
    # cache-access IDs, context/indices and per-candidate selected sets.
    assert current == original
    assert ast.dump(loop, include_attributes=True) == before
    for candidate, reference in zip(current, original, strict=True):
        assert candidate.selected_statement_ids is not reference.selected_statement_ids


def test_raw_summary_preserves_current_rename_mapping() -> None:
    loop = ast.parse(
        "for row in range(4):\n"
        "    value = cutlass.Float32(scale)\n"
        "    shadow = cutlass.Float32(0)\n"
        "    output = value\n"
    ).body[0]
    assert isinstance(loop, ast.For)
    facts = hoist._vector_loop_facts(loop)
    renames: dict[str, str] = {}
    resolver = hoist._VectorExpressionResolver(
        outer_loop=loop,
        loop_facts=facts,
        infos=[],
        cache_stores={},
        rmem={},
        rename_groups=renames,
        tensor_dtypes={},
        tensor_names=set(),
        float_scalar_names={"scale"},
    )
    context = hoist._ExprContext(2)
    assert isinstance(
        resolver.normalize(ast.Name(id="value", ctx=ast.Load()), context), ast.Call
    )
    assert id(loop.body[1]) in facts.statement_writes
    renames["shadow"] = "value"
    with pytest.raises(hoist._VectorCacheReject):
        resolver.normalize(ast.Name(id="value", ctx=ast.Load()), context)


def test_intervening_rebind_is_checked_at_each_reaching_context() -> None:
    loop = ast.parse(
        "for row in range(4):\n"
        "    value = cutlass.Float32(scale)\n"
        "    before = value\n"
        "    if cond:\n"
        "        value = cutlass.Float32(row)\n"
        "    after = value\n"
    ).body[0]
    assert isinstance(loop, ast.For)
    resolver = hoist._VectorExpressionResolver(
        outer_loop=loop,
        loop_facts=hoist._vector_loop_facts(loop),
        infos=[],
        cache_stores={},
        rmem={},
        rename_groups={},
        tensor_dtypes={},
        tensor_names=set(),
        float_scalar_names={"scale"},
    )
    resolver.normalize(ast.Name(id="value", ctx=ast.Load()), hoist._ExprContext(1))
    with pytest.raises(hoist._VectorCacheReject):
        resolver.normalize(ast.Name(id="value", ctx=ast.Load()), hoist._ExprContext(3))
