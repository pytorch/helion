from __future__ import annotations

import ast
import itertools
import textwrap
from types import SimpleNamespace

import pytest

from helion._compiler.ast_read_writes import HELION_LANE_LOOP_VAR_ATTR
from helion._compiler.cute.unroll_lane_loads import unroll_lane_loads

BASE = """
for lane in range(2):
    index = lane * 8
    ptr = state.iterator + index
    loaded = ptr.load()
    total = loaded + 3
    loaded = total + 1
    (state.iterator + lane * 8).store(loaded)
"""


def transform(
    text: str,
    disjoint: tuple[tuple[str, str], ...] | list[tuple[str, str]] = (),
    rename: dict[str, str] | None = None,
) -> tuple[str, bool]:
    body = ast.parse(textwrap.dedent(text)).body
    loop = body[0]
    setattr(loop, HELION_LANE_LOOP_VAR_ATTR, "lane")
    counter = itertools.count()
    occupied = {n.id for n in ast.walk(loop) if isinstance(n, ast.Name)}

    def fresh(prefix: str) -> str:
        while (name := f"{prefix}_{next(counter)}") in occupied:
            pass
        occupied.add(name)
        return name

    out, fired = unroll_lane_loads(
        body,
        2,
        fresh,
        proven_disjoint_tensor_pairs={frozenset(pair) for pair in disjoint},
        proven_tensor_stride_values={("state", 0): 1},
        rename_groups=rename,
    )
    return ast.unparse(ast.Module(body=out, type_ignores=[])), fired


class Pointer:
    def __init__(self, tensor: Tensor, offset: int = 0) -> None:
        self.tensor, self.offset = tensor, offset

    def __add__(self, offset: int) -> Pointer:
        return Pointer(self.tensor, self.offset + offset)

    def load(self) -> int:
        self.tensor.events.append(("load", self.offset))
        return self.tensor.data[self.offset]

    def store(self, value: int) -> None:
        self.tensor.events.append(("store", self.offset, value))
        self.tensor.data[self.offset] = value


class Tensor:
    def __init__(self, values: range) -> None:
        self.data = list(values)
        self.events: list[tuple[str, int] | tuple[str, int, int]] = []
        self.iterator = Pointer(self)


def execute(
    text: str,
) -> tuple[list[int], list[tuple[str, int] | tuple[str, int, int]]]:
    state = Tensor(range(24))
    namespace = {"state": state, "cutlass": SimpleNamespace(range_constexpr=range)}
    exec(compile(text, "<test>", "exec"), namespace)
    return state.data, state.events


def test_redefinitions_snapshot_load_and_original_store_order() -> None:
    transformed, fired = transform(BASE)
    assert fired
    expected, old_events = execute(BASE)
    actual, events = execute(transformed)
    assert actual == expected
    assert events[:2] == [("load", 0), ("load", 8)]
    assert [e for e in events if e[0] == "store"] == [
        e for e in old_events if e[0] == "store"
    ]
    assert "_lane_unroll_prefix" in transformed


def test_address_redefinition_keeps_assignment_time_value() -> None:
    source = BASE.replace(
        "    total = loaded + 3",
        "    index = 9\n    ptr = state.iterator + index\n    total = loaded + 3",
    )
    transformed, fired = transform(source)
    assert fired and execute(transformed)[0] == execute(source)[0]


@pytest.mark.parametrize(
    "source",
    [
        BASE.replace("lane * 8).store", "(lane + 1) * 8).store"),
        BASE.replace("index = lane * 8", "index = 0").replace(
            "lane * 8).store", "0).store"
        ),
        BASE.replace("index = lane * 8", "index = index + 8"),
        BASE.replace("index = lane * 8", "index += 8"),
        BASE.replace(
            "    index = lane * 8", "    unknown(state)\n    index = lane * 8"
        ),
        BASE.replace(
            "    total = loaded + 3", "    unknown(state)\n    total = loaded + 3"
        ),
        BASE.replace(
            "    total = loaded + 3", "    cute.arch.barrier()\n    total = loaded + 3"
        ),
        BASE.replace(
            "    total = loaded + 3",
            "    state.iterator = other\n    total = loaded + 3",
        ),
        BASE.replace(
            "    total = loaded + 3", "    other[0] = 1\n    total = loaded + 3"
        ),
        BASE.replace(
            "    total = loaded + 3",
            "    alias = state\n    alias.value = 1\n    total = loaded + 3",
        ),
        BASE.replace("    total = loaded + 3", "    break\n    total = loaded + 3"),
        BASE.replace(
            "    loaded = ptr.load()", "    if flag:\n        loaded = ptr.load()"
        ),
    ],
)
def test_unsafe_prefix_declines(source: str) -> None:
    _out, fired = transform(source)
    assert not fired


def test_disjoint_external_owner_is_required() -> None:
    source = BASE.replace(
        "(state.iterator + lane * 8).store", "(output.iterator + lane * 8).store"
    )
    assert not transform(source)[1]
    assert transform(source, [("state", "output")])[1]


MASKED = """
for lane in range(2):
    slot = (indices.iterator + 0).load()
    row = lane * 8
    loaded = cute.arch.load(state.iterator + (slot * 128 + row if slot < 2 else 0), ir.VectorType.get([8], cutlass.Uint16.mlir_type))
    total = loaded[0] + 1
    loaded = total
    if slot < 2:
        _cute_store_u16_vec(state.iterator + slot * 128 + row, values)
    slot = (indices.iterator + 0).load()
"""


def test_masked_lane_invariant_predicate_keeps_mask_text() -> None:
    out, fired = transform(MASKED, [("indices", "state")])
    assert fired
    assert "if _lane_unroll_prefix_" in out and "else 0" in out
    assert "if slot < 2:" in out


@pytest.mark.parametrize(
    "source,pairs",
    [
        (MASKED, []),
        (MASKED.replace("if slot < 2:\n", "if slot < 3:\n"), [("indices", "state")]),
        (MASKED.replace("slot < 2", "lane < 1"), [("indices", "state")]),
        (
            MASKED.replace("    if slot < 2:", "    slot = 1\n    if slot < 2:"),
            [("indices", "state")],
        ),
        (
            MASKED.replace("slot * 128 + row, values", "slot * 128 + row + 8, values"),
            [("indices", "state")],
        ),
    ],
)
def test_mask_or_alias_change_declines(
    source: str, pairs: list[tuple[str, str]]
) -> None:
    assert not transform(source, pairs)[1]


def test_rename_alias_mutation_declines() -> None:
    source = BASE.replace("    index = lane * 8", "    index = old + 8").replace(
        "    total = loaded + 3", "    new = 0\n    total = loaded + 3"
    )
    assert not transform(source, rename={"old": "shared", "new": "shared"})[1]


def test_deleted_or_loop_target_dependency_declines() -> None:
    for tail in ["del state", "for state in items:\n        pass"]:
        source = BASE + "    " + tail + "\n"
        assert not transform(source)[1]
