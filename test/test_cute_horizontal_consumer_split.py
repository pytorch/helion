from __future__ import annotations

import ast

import pytest

from helion._compiler.cute.horizontal_state_codegen import _split_consumer_transport
from helion._compiler.cute.horizontal_state_update import InapplicableStateUpdate

BODY = """
for stage in range(stages):
    barrier('full-arrive', stage)
    barrier('full-wait', stage, 0)
    if row < rows:
        for item in range(0, 16, 2):
            column = (member * 16 + item + 2 * (row_group // 2)) % 32
            state_values = [0, 0]
            if eligible:
                state_values = load_shared(row, stage, column)
            coefficient = load_coefficient(stage, column)
            new_state_values = []
            for component in range(2):
                feature = stage * 32 + column + component
                old = state_values[component] if eligible else load_scalar(slot, row, feature) if slot < slots and feature < features else 0
                current = old * decay + product * coefficient[component]
                acc = acc + current * weight
                new_state_values.append(current)
                if not eligible:
                    if slot < slots and feature < features:
                        store_scalar(slot, row, feature, current)
            if eligible:
                store_shared(row, stage, column, new_state_values)
    barrier('empty-arrive', stage)
"""


def execute(module, *, eligible, row, rows, slot, features, member):
    events = []

    def emit(kind, *args):
        events.append((kind, *args))

    def shared(row, stage, column):
        emit("load_shared", row, stage, column)
        return [stage * 32 + column + 1, stage * 32 + column + 2]

    def coefficient(stage, column):
        emit("coefficient", stage, column)
        return [3, 5]

    def scalar(slot, row, feature):
        emit("load_scalar", slot, row, feature)
        return feature + 1

    env = {
        "stages": features // 32,
        "eligible": eligible,
        "row": row,
        "rows": rows,
        "row_group": row % 16,
        "member": member,
        "slot": slot,
        "slots": 3,
        "features": features,
        "decay": 2,
        "product": 7,
        "weight": 11,
        "acc": 0,
        "barrier": lambda *args: emit("barrier", *args),
        "load_shared": shared,
        "load_coefficient": coefficient,
        "load_scalar": scalar,
        "store_shared": lambda *args: emit("store_shared", *args),
        "store_scalar": lambda *args: emit("store_scalar", *args),
    }
    exec(compile(ast.fix_missing_locations(module), "transport-trace", "exec"), env)
    return events, env["acc"]


@pytest.mark.parametrize(
    "eligible,slot", [(True, 0), (False, -1), (False, 3), (False, 2)]
)
@pytest.mark.parametrize("rows,features", [(16, 32), (65, 64), (128, 128)])
def test_same_ordered_events_and_arithmetic(eligible, slot, rows, features):
    original = ast.parse(BODY)
    before = ast.dump(original)
    loop = original.body[0]
    assert isinstance(loop, ast.For)
    split = _split_consumer_transport(loop, "eligible")
    assert ast.dump(original) == before
    for row in (0, rows - 1, rows, rows + 7):
        for member in (0, 1):
            kw = {
                "eligible": eligible,
                "row": row,
                "rows": rows,
                "slot": slot,
                "features": features,
                "member": member,
            }
            a = execute(original, **kw)
            b = execute(ast.Module(body=[split], type_ignores=[]), **kw)
            assert a == b
            barriers = [event for event in b[0] if event[0] == "barrier"]
            assert barriers == [
                event
                for stage in range(features // 32)
                for event in (
                    ("barrier", "full-arrive", stage),
                    ("barrier", "full-wait", stage, 0),
                    ("barrier", "empty-arrive", stage),
                )
            ]
    for arm in (split.body, split.orelse):
        assert not any(
            isinstance(node, ast.Name) and node.id == "eligible"
            for statement in arm
            for node in ast.walk(statement)
        )


@pytest.mark.parametrize(
    "line",
    [
        "eligible = False",
        "del eligible",
        "foreign(eligible)",
        "x = int(eligible)",
        "x = eligible & other",
    ],
)
def test_unclassified_or_changed_predicate_rejects(line):
    loop = ast.parse("for stage in range(4):\n    " + line).body[0]
    assert isinstance(loop, ast.For)
    with pytest.raises(InapplicableStateUpdate):
        _split_consumer_transport(loop, "eligible")
