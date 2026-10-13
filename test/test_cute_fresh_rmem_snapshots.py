"""Fresh register ownership must not turn content writes into namespace writes."""

from __future__ import annotations

import ast

import pytest

from helion._compiler.cute.persistent_branch_vec import _definition_snapshots

INDEX = "index = base + cutlass.Int32(lane)\n"
ALLOC = "cute.make_rmem_tensor(4, cutlass.Float32)"


def final(source: str, required: set[str] | None = None) -> dict[str, ast.expr]:
    result = _definition_snapshots(
        ast.parse(source + "\nsentinel = 0\n").body, required_names=required
    )
    assert result is not None
    return result[-1]


def test_exact_fresh_storage_preserves_unrelated_index() -> None:
    result = final(INDEX + f"cache = {ALLOC}\ncache[0] = cutlass.Float32(1)\n")
    assert "index" in result
    assert "cache" not in result


def test_identical_constructors_still_have_distinct_storage() -> None:
    result = final(
        INDEX + f"a = {ALLOC}\nb = {ALLOC}\nleft = a[0]\nright = b[0]\na[0] = 1\n"
    )
    assert {"index", "b", "right"} <= result.keys()
    assert not {"a", "left"} & result.keys()


def test_content_write_invalidates_all_aliases_and_derived_values() -> None:
    result = final(
        INDEX
        + f"cache = {ALLOC}\nalias = cache\nother = alias\nvalue = cache[0]\nderived = value + 1\nother[0] = 2\n"
    )
    assert "index" in result
    assert not {"cache", "alias", "other", "value", "derived"} & result.keys()


def test_ownership_survives_repeated_content_writes() -> None:
    result = final(
        INDEX
        + f"cache = {ALLOC}\nalias = cache\ncache[0] = 1\nvalue = alias[0]\ncache[0] = 2\n"
    )
    assert "index" in result
    assert "value" not in result


def test_surviving_alias_keeps_old_owner_after_binding_replacement() -> None:
    result = final(
        INDEX
        + f"cache = {ALLOC}\nalias = cache\nold = cache[0]\ncache = {ALLOC}\nnew = cache[0]\nalias[0] = 2\n"
    )
    assert {"index", "cache", "new"} <= result.keys()
    assert not {"alias", "old"} & result.keys()


def test_rebound_fresh_name_to_external_value_is_not_fresh() -> None:
    result = final(
        "index = external[0]\n" + f"cache = {ALLOC}\ncache = external\ncache[0] = 2\n"
    )
    assert "index" not in result


@pytest.mark.parametrize(
    "constructor",
    [
        "cute.make_rmem_tensor(dynamic, cutlass.Float32)",
        "cute.make_rmem_tensor(True, cutlass.Float32)",
        "cute.make_rmem_tensor(0, cutlass.Float32)",
        "cute.make_rmem_tensor(4, unknown(cutlass))",
        "cute.make_rmem_tensor(4, cutlass.Float32, layout=layout)",
        "unknown(cutlass)",
    ],
)
def test_nonfresh_constructor_keeps_original_conservative_invalidation(
    constructor: str,
) -> None:
    result = final(INDEX + f"cache = {constructor}\ncache[0] = 1\n")
    assert "index" not in result


def test_unknown_alias_call_keeps_namespace_and_owner_invalidation() -> None:
    result = final(
        INDEX
        + f"cache = {ALLOC}\nvalue = cache[0]\nalias = unknown(cache)\nalias[0] = 1\n"
    )
    assert not {"index", "cache", "value"} & result.keys()


def test_binding_projection_retains_fresh_alias_provenance() -> None:
    source = INDEX + f"cache = {ALLOC}\nalias = cache\nvalue = cache[0]\nalias[0] = 1\n"
    all_facts = final(source)
    selected = final(source, {"index", "value"})
    assert "index" in selected and "value" not in selected
    assert ast.dump(selected["index"]) == ast.dump(all_facts["index"])
