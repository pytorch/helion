from __future__ import annotations

import ast

import pytest

from helion._compiler.cute.fuse_two_pass_loads import _canonical_load_text
from helion._compiler.cute.fuse_two_pass_loads import _CuteFuseTwoPassLoads
from helion._compiler.cute.fuse_two_pass_loads import fuse_two_pass_loads


def _dump(node: ast.AST) -> str:
    return ast.dump(node, include_attributes=True)


@pytest.mark.parametrize(
    "expression",
    [
        "base + 4",
        "base - (other - third)",
        "base / (other / third)",
        "(base and other) and third",
        "base if active else other",
        "(x.iterator + base).load() if active else cutlass.Float32(0)",
        "values[base:other:2]",
        "fn(base, scale=other)",
    ],
)
def test_frozen_input_is_owned_and_normalization_is_stable(expression: str) -> None:
    owner = _CuteFuseTwoPassLoads()
    first = ast.parse(f"saved = {expression}").body[0]
    assert isinstance(first, ast.Assign)
    frozen = owner._freeze_binding(first, first.value, {})
    snapshot = _dump(frozen)
    use = ast.parse("result = saved + saved").body[0]
    assert isinstance(use, ast.Assign)
    composed = owner._freeze_binding(use, use.value, {"saved": frozen})
    assert isinstance(composed, ast.BinOp)
    assert composed.left is frozen and composed.right is frozen
    assert ast.dump(composed) == ast.dump(
        ast.parse(ast.unparse(composed), mode="eval").body
    )
    # Mutating the source expression must not alter the frozen analysis DAG.
    for node in ast.walk(first):
        if isinstance(node, ast.Name):
            node.id = "changed_source"
    assert _dump(frozen) == snapshot
    assert ast.unparse(composed) != ast.unparse(first.value)


def test_rebinding_and_cycles_keep_earlier_snapshot_values() -> None:
    owner = _CuteFuseTwoPassLoads()
    body = ast.parse("a = head + 1\nb = a\na = a + 1\na = b + 2").body
    definitions = owner._pure_definitions_before(body, 2)
    old = definitions["a"]
    assert definitions["b"] is old
    sibling = dict(definitions)
    owner._extend_definitions(body, 2, 4, definitions)
    assert ast.unparse(definitions["a"]) == "head + 1 + 2"
    assert ast.unparse(definitions["b"]) == "head + 1"
    assert sibling["a"] is old and sibling["b"] is old
    assert definitions["a"] is not old
    assert len(list(ast.walk(definitions["a"]))) < 20


def test_canonicalization_copies_before_alias_and_location_rewrites() -> None:
    owner = _CuteFuseTwoPassLoads()
    body = ast.parse("base = lane + 4\nrepeated = base + base").body
    bindings = owner._pure_definitions_before(body, len(body))
    snapshots = {name: _dump(value) for name, value in bindings.items()}
    load = ast.parse("(x.iterator + repeated).load()", mode="eval").body
    original = _dump(load)
    text = _canonical_load_text(load, {"lane": "other_lane"}, bindings)
    assert "other_lane" in text and "repeated" not in text
    assert _dump(load) == original
    assert snapshots == {name: _dump(value) for name, value in bindings.items()}


def test_budget_counts_expanded_occurrences_not_unique_nodes() -> None:
    owner = _CuteFuseTwoPassLoads()
    definitions: dict[str, ast.expr] = {}
    for index in range(9):
        expr = "input_value" if index == 0 else f"x{index - 1} + x{index - 1}"
        statement = ast.parse(f"x{index} = {expr}").body[0]
        owner._extend_definitions([statement], 0, 1, definitions)
        value = definitions[f"x{index}"]
        if index < 8:
            assert sum(1 for _ in ast.walk(value)) == 2 ** (index + 2) - 2
            assert not isinstance(value, ast.Name) or value.id == "input_value"
        else:
            assert isinstance(value, ast.Name)
            assert value.id.startswith("_fuser_binding_")


@pytest.mark.parametrize("mode", ["register", "computed"])
@pytest.mark.parametrize("effect", ["", "(x.iterator + 0).store(7)", "unknown(x)"])
def test_actual_transforms_do_not_mutate_or_publish_frozen_nodes(
    monkeypatch: pytest.MonkeyPatch, mode: str, effect: str
) -> None:
    original = _CuteFuseTwoPassLoads._freeze_binding
    retained: list[tuple[ast.expr, str]] = []

    def observe(
        self: _CuteFuseTwoPassLoads,
        statement: ast.stmt,
        value: ast.expr,
        previous: dict[str, ast.expr],
    ) -> ast.expr:
        before_source = _dump(value)
        before_previous = {name: _dump(node) for name, node in previous.items()}
        result = original(self, statement, value, previous)
        assert _dump(value) == before_source
        assert before_previous == {name: _dump(node) for name, node in previous.items()}
        retained.append((result, _dump(result)))
        return result

    monkeypatch.setattr(_CuteFuseTwoPassLoads, "_freeze_binding", observe)
    source = f"""
base = start + 1
offset = base + base
total = cutlass.Float32(0)
for synthetic_lane_7 in range(8):
    address = offset + synthetic_lane_7
    raw = (x.iterator + address).load()
    current = cutlass.Float32(raw) * cutlass.Float32(2)
    total = total + current
{effect}
for synthetic_lane_7 in range(8):
    address = offset + synthetic_lane_7
    raw = (x.iterator + address).load()
    current = cutlass.Float32(raw) * cutlass.Float32(2)
    (out.iterator + synthetic_lane_7).store(current + total)
"""
    output = fuse_two_pass_loads(
        ast.parse(source).body,
        tensor_dtypes={"x": "cutlass.Float32", "out": "cutlass.Float32"},
        reload_modes={7: mode},
        proven_disjoint_tensor_pairs={frozenset(("x", "out"))},
    )
    module = ast.Module(body=output, type_ignores=[])
    code = ast.unparse(module)
    assert retained
    assert all(_dump(value) == snapshot for value, snapshot in retained)
    frozen_ids = {
        id(node)
        for value, _ in retained
        for node in ast.walk(value)
        if isinstance(node, ast.expr)
    }
    assert not frozen_ids.intersection(
        id(node) for node in ast.walk(module) if isinstance(node, ast.expr)
    )
    if not effect:
        assert code.count(".load()") == 1
        assert "cute.make_rmem_tensor" in code
    elif effect != "unknown(x)" or mode == "computed":
        assert code.count(".load()") == 2
    else:
        # The existing raw register pass accepts this call; test only snapshot
        # ownership here. Computed-mode unknown-effect rejection is above.
        assert "unknown(x)" in code
