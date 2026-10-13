"""Allocation is an effect barrier only where its private lifetime requires it."""

from __future__ import annotations

import ast

import pytest

from test.test_cute_hoist_lane_invariant_reductions import _rewrite

from helion._compiler.cute import hoist_lane_invariant_reductions as h

BODY = """
for row_lane in range(4):
    coeff = cute.arch.load(key.iterator + base, vector_type)
    cache = cute.make_rmem_tensor(4, cutlass.Float32)
    cache[0] = cutlass.Float32(row_lane)
    value = cutlass.Float32(coeff[0]) + cache[0]
    (out.iterator + row_lane).store(value)
"""


def source(body: list[ast.stmt]) -> str:
    return ast.unparse(ast.Module(body=body, type_ignores=[]))


def test_allocation_stays_in_row_and_disjoint_load_moves() -> None:
    result = _rewrite(BODY)
    assert len(result) == 2
    assert isinstance(result[0], ast.Assign)
    assert ast.unparse(result[0].targets[0]) == "coeff"
    assert isinstance(result[1], ast.For)
    assert (
        ast.unparse(result[1].body[0])
        == "cache = cute.make_rmem_tensor(4, cutlass.Float32)"
    )
    assert source(result[1].body) == source(ast.parse(BODY).body[0].body[1:])
    allocation = result[1].body[0].value
    assert not h._is_allowed_pure_call(allocation)
    assert not h._is_hoistable_statement(result[1].body[0], {"key", "out"})


@pytest.mark.parametrize("dtype", ["Float16", "BFloat16", "Float32", "Int32"])
def test_supported_register_cache_dtypes_are_effect_only(dtype: str) -> None:
    code = source(
        _rewrite(
            BODY.replace(
                "make_rmem_tensor(4, cutlass.Float32)",
                f"make_rmem_tensor(4, cutlass.{dtype})",
            )
        )
    )
    assert code.index("coeff =") < code.index("for row_lane") < code.index("cache =")


@pytest.mark.parametrize(
    "constructor",
    [
        "cute.make_rmem_tensor(dynamic, cutlass.Float32)",
        "cute.make_rmem_tensor(0, cutlass.Float32)",
        "cute.make_rmem_tensor(-1, cutlass.Float32)",
        "cute.make_rmem_tensor(True, cutlass.Float32)",
        "cute.make_rmem_tensor(4, unknown_dtype)",
        "cute.make_rmem_tensor(4, cutlass.Float32, layout=layout)",
        "other.make_rmem_tensor(4, cutlass.Float32)",
    ],
)
def test_unknown_allocation_still_blocks(constructor: str) -> None:
    result = _rewrite(
        BODY.replace("cute.make_rmem_tensor(4, cutlass.Float32)", constructor)
    )
    assert len(result) == 1 and isinstance(result[0], ast.For)


@pytest.mark.parametrize(
    "extra",
    [
        "unknown(cache)",
        "alias = cache\n    unknown(alias)",
        "cute.arch.sync_threads()",
    ],
)
def test_escape_or_synchronization_stays_barrier(extra: str) -> None:
    result = _rewrite(BODY.replace("    value =", f"    {extra}\n    value ="))
    assert len(result) == 1 and isinstance(result[0], ast.For)


def test_no_disjoint_authority_does_not_move_load() -> None:
    result = _rewrite(BODY, disjoint=set())
    assert len(result) == 1 and isinstance(result[0], ast.For)


def test_same_owner_and_row_varying_load_do_not_move() -> None:
    for replacement in ("out.iterator + base", "key.iterator + row_lane"):
        result = _rewrite(BODY.replace("key.iterator + base", replacement))
        assert len(result) == 1 and isinstance(result[0], ast.For)


def test_cache_result_cannot_escape_its_allocation_scope() -> None:
    changed = BODY.replace(
        "    value =",
        "    earlier = cache[0]\n    cache[0] = cutlass.Float32(9)\n    value =",
    ).replace(" + cache[0]", " + earlier + cache[0]")
    code = source(_rewrite(changed))
    row = code.index("for row_lane")
    assert code.index("earlier = cache[0]") > row
    assert code.index("cache[0] = cutlass.Float32(9)") > code.index("earlier =")


def test_renamed_alias_mutation_prevents_hoist() -> None:
    changed = BODY.replace(
        "    value =", "    shadow = cutlass.Float32(row_lane)\n    value ="
    )
    result = _rewrite(changed, rename_groups={"shadow": "coeff"})
    assert len(result) == 1 and isinstance(result[0], ast.For)


def test_unknown_nested_constructor_is_not_granted_authority() -> None:
    node = ast.parse("cache = wrap(cute.make_rmem_tensor(4, cutlass.Float32))")
    assert not h._has_only_known_effects(node)


def test_prefix_accepts_local_cache_but_never_moves_allocation() -> None:
    from helion._compiler.cute.lane_prefix_snapshots import _known_loop_effects
    from helion._compiler.cute.unroll_lane_loads import _candidate_target

    loop = ast.parse(BODY).body[0]
    assert _known_loop_effects(loop, frozenset())
    assert _candidate_target(loop.body[1]) is None


@pytest.mark.parametrize(
    "extra",
    [
        "cache = other",
        "alias = cache\n    alias[0] = cutlass.Float32(2)",
        "cache.value = other",
        "unknown(cache)",
    ],
)
def test_prefix_rejects_local_register_rebind_alias_and_escape(extra: str) -> None:
    from helion._compiler.cute.lane_prefix_snapshots import _known_loop_effects

    loop = ast.parse(BODY.replace("    value =", f"    {extra}\n    value =")).body[0]
    assert not _known_loop_effects(loop, frozenset())
