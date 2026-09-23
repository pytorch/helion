from __future__ import annotations

import ast

import pytest

from helion._compiler.cute.persistent_branch_vec import _BranchLocalPersistentVectorizer
from helion._compiler.cute.persistent_branch_vec import _definition_snapshots
from helion._compiler.cute.persistent_branch_vec import (
    _memory_address_definition_snapshots,
)
from helion._compiler.tile_strategy import _lane_split_reorders_aliasing_memory
from helion._compiler.tile_strategy import _LaneReduceMarker


def _repeated_expression(count: int) -> list[ast.stmt]:
    return ast.parse(
        "value_0 = seed\n"
        + "\n".join(
            f"value_{index} = (value_{index - 1} * 3) ^ value_{index - 1}"
            for index in range(1, count + 1)
        )
        + "\npass"
    ).body


def test_small_assignment_time_definitions_preserve_values() -> None:
    statements = _repeated_expression(4)
    snapshots = _definition_snapshots(statements)
    assert snapshots is not None
    values = {"seed": 17}
    for statement, definitions in zip(statements, snapshots, strict=True):
        for name, expression in definitions.items():
            assert sum(1 for node in ast.walk(expression)) <= 512
            code = compile(ast.Expression(body=expression), "<frozen>", "eval")
            assert eval(code, {}, values) == values[name]
        exec(
            compile(
                ast.Module(body=[statement], type_ignores=[]), "<original>", "exec"
            ),
            {},
            values,
        )


def test_oversized_definition_declines_without_modifying_source() -> None:
    statements = _repeated_expression(80)
    before = ast.dump(ast.Module(body=statements, type_ignores=[]))
    assert _definition_snapshots(statements) is None
    assert ast.dump(ast.Module(body=statements, type_ignores=[])) == before


def test_cached_dependencies_invalidate_aliases_on_reassignment() -> None:
    statements = ast.parse("alias = source\naddress = alias + 4\nsource = 0\npass").body
    snapshots = _definition_snapshots(statements)
    assert snapshots is not None
    assert ast.unparse(snapshots[-2]["alias"]) == "source"
    assert "alias" not in snapshots[-1]
    assert "address" not in snapshots[-1]
    assert ast.literal_eval(snapshots[-1]["source"]) == 0


def test_oversized_lane_dependency_cannot_justify_load_or_store_motion() -> None:
    body = ast.parse(
        "term_0 = lane\n"
        + "\n".join(
            f"term_{index} = term_{index - 1} + term_{index - 1}"
            for index in range(1, 13)
        )
        + "\nhidden = (term_12 - term_12) - lane\n"
        + "value = (x.iterator + lane + hidden).load()\n"
        + "(x.iterator + lane + hidden).store(value)\n"
    ).body
    # Every iteration reads/writes x[0]. Forgetting `hidden` would incorrectly
    # make the pointer's visible `lane` term appear to prove independence.
    assert _definition_snapshots(body) is None
    load = ast.parse(
        "marker(0, 0, 0, 0, x.iterator + lane + hidden, True)", mode="eval"
    ).body
    store = ast.parse(
        "marker(0, 0, 0, x.iterator + lane + hidden, value, True)", mode="eval"
    ).body
    assert isinstance(load, ast.Call) and isinstance(store, ast.Call)
    vectorizer = _BranchLocalPersistentVectorizer(body, set(), {})
    assert vectorizer._has_aliasing_write_in_loop(body, load, len(body) - 2, "lane", 4)
    assert vectorizer._has_other_aliasing_access_in_loop(
        body, store, len(body) - 1, "lane", 4
    )
    body.extend(ast.parse("reduced = value").body)
    marker = _LaneReduceMarker(
        "reduced", "value", "sum", "0.0", 32, "__HELION_FINALIZED__"
    )
    assert _lane_split_reorders_aliasing_memory(
        list(body), [(len(body) - 1, marker)], set(), "lane", 4, {}
    )


def test_disjoint_memory_does_not_require_expanding_unrelated_arithmetic() -> None:
    body = ast.parse("value = (x.iterator + lane).load()").body
    body.extend(_repeated_expression(80))
    body.extend(
        ast.parse(
            "result = value + value_80\n"
            "(out.iterator + lane).store(result)\n"
            "reduced = result\n"
        ).body
    )
    marker = _LaneReduceMarker(
        "reduced", "result", "sum", "0.0", 32, "__HELION_FINALIZED__"
    )
    assert _definition_snapshots(body) is None
    assert not _lane_split_reorders_aliasing_memory(
        list(body),
        [(len(body) - 1, marker)],
        {frozenset(("x", "out"))},
        "lane",
        4,
        {},
    )
    # Without the allocation proof the same source must remain conservative.
    assert _lane_split_reorders_aliasing_memory(
        list(body), [(len(body) - 1, marker)], set(), "lane", 4, {}
    )


@pytest.mark.parametrize(
    "store_offset, reorders", [("lane", False), ("lane + 1", True)]
)
def test_address_proof_ignores_unrelated_large_values(
    store_offset: str, reorders: bool
) -> None:
    body = ast.parse("pointer = x.iterator + lane\nvalue = pointer.load()").body
    body.extend(_repeated_expression(80))
    body.extend(
        ast.parse(
            "result = value + value_80\n"
            f"(x.iterator + {store_offset}).store(result)\n"
            "reduced = result\n"
        ).body
    )
    before = ast.dump(ast.Module(body=body, type_ignores=[]))
    assert _definition_snapshots(body) is None
    snapshots = _memory_address_definition_snapshots(body)
    assert snapshots is not None
    assert ast.unparse(snapshots[1]["pointer"]) == "x.iterator + lane"
    assert all("value_80" not in definitions for definitions in snapshots)
    marker = _LaneReduceMarker(
        "reduced", "result", "sum", "0.0", 32, "__HELION_FINALIZED__"
    )
    assert (
        _lane_split_reorders_aliasing_memory(
            list(body), [(len(body) - 1, marker)], set(), "lane", 4, {}
        )
        is reorders
    )
    assert ast.dump(ast.Module(body=body, type_ignores=[])) == before


@pytest.mark.parametrize("overwrite", ["", "hidden = 0\n"])
def test_address_slice_keeps_every_reaching_hidden_lane_definition(
    overwrite: str,
) -> None:
    body = ast.parse(
        "term_0 = lane\n"
        + "\n".join(
            f"term_{index} = term_{index - 1} + term_{index - 1}"
            for index in range(1, 13)
        )
        + "\nhidden = (term_12 - term_12) - lane\n"
        + "value = (x.iterator + lane + hidden).load()\n"
        + overwrite
        + "(x.iterator + lane + hidden).store(value)\n"
    ).body
    assert _memory_address_definition_snapshots(body) is None


@pytest.mark.parametrize(
    "mutation",
    [
        "alias[0] = 9",
        "alias.field = 9",
        "source += 1",
        "source: int = 9",
        "for source in range(2):\n    pass",
        "del source",
    ],
)
def test_address_slice_retains_all_write_invalidation(mutation: str) -> None:
    body = ast.parse(
        "alias = source\n"
        "address = source[0] + lane\n"
        "unused = other * 2\n" + mutation + "\npass\n"
    ).body
    full = _definition_snapshots(body)
    selected = _definition_snapshots(body, required_names={"address"})
    assert full is not None and selected is not None
    assert "address" in selected[-2]
    assert "address" not in selected[-1]
    for all_definitions, selected_definitions in zip(full, selected, strict=True):
        for name, expression in selected_definitions.items():
            assert ast.dump(expression) == ast.dump(all_definitions[name])
        assert "unused" not in selected_definitions


def test_address_slice_does_not_expand_store_value_in_vector_marker() -> None:
    body = ast.parse("address = x.iterator + synthetic_lane_0\n").body
    body.extend(_repeated_expression(80))
    body.extend(
        ast.parse(
            "value = _helion_persistent_branch_vec_load("
            "0, 4, 'cutlass.Float32', 0.0, address, True)\n"
            "_helion_persistent_branch_vec_store("
            "0, 4, 'cutlass.Float32', address, value + value_80, True)\n"
        ).body
    )
    assert _definition_snapshots(body) is None
    snapshots = _memory_address_definition_snapshots(body)
    assert snapshots is not None
    assert ast.unparse(snapshots[-1]["address"]) == "x.iterator + synthetic_lane_0"
    assert "value_80" not in snapshots[-1]
