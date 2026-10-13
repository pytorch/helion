from __future__ import annotations

import ast
import importlib
from itertools import combinations
from typing import TYPE_CHECKING
from typing import Callable

import pytest

if TYPE_CHECKING:
    from helion._compiler.cute.hoist_lane_invariant_reductions import _Effects

impl = importlib.import_module("helion._compiler.cute.hoist_lane_invariant_reductions")


def reference() -> Callable[[list[ast.stmt], set[int]], set[str]]:
    def original(body: list[ast.stmt], selected: set[int]) -> set[str]:
        needed: set[str] = set()
        for index in sorted(selected):
            effects = impl._statement_effects(body[index])
            needed.update(
                set(effects.live_in)
                - {
                    name
                    for prior in selected
                    if prior < index
                    for name in impl._statement_effects(body[prior]).writes
                }
            )
        return needed

    return original


@pytest.mark.parametrize(
    "code",
    [
        "first = source\nsecond = first + bias\nthird = second",
        "first = first + source\nfirst += delta\nsecond = first",
        "left, right = right, left\nresult = left + right",
        "value = source\nvalue = other\nresult = value\nlast = result + tail",
        "if active:\n    value = left\nelse:\n    value = right\nresult = value",
        "if active:\n    value = source\nresult = value",
        "for lane in range(extent):\n    value = source + lane\nresult = value",
        "value = source\nfor lane in range(extent):\n    value += lane\nresult = value",
        "buffer[index] = source\nresult = buffer[next_index]\nindex += delta",
        "value: scalar_type = source\nresult = value\nobj.field = result",
        "while predicate:\n    value = source\nresult = value",
    ],
)
def test_every_subset_matches_original_structured_effects(code: str) -> None:
    body = ast.parse(code).body
    old = reference()
    for size in range(len(body) + 1):
        for indices in combinations(range(len(body)), size):
            selected = set(indices)
            assert impl._slice_external_reads(body, selected) == old(body, selected)


def test_reads_precede_writes_in_same_statement() -> None:
    body = ast.parse("value = value + delta").body
    assert impl._slice_external_reads(body, {0}) == {"value", "delta"}


def test_selected_prior_writes_only() -> None:
    body = ast.parse("value = source\nskipped = value\nresult = value + skipped").body
    assert impl._slice_external_reads(body, {0, 2}) == {"source", "skipped"}
    assert impl._slice_external_reads(body, {2}) == {"value", "skipped"}


def test_empty_selection() -> None:
    assert impl._slice_external_reads(ast.parse("value = source").body, set()) == set()


def test_one_effect_evaluation_per_selected_statement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = ast.parse("\n".join(f"v{i} = source + {i}" for i in range(128))).body
    selected = set(range(len(body)))
    original_effects = impl._statement_effects
    calls = 0

    def counted(stmt: ast.stmt) -> _Effects:
        nonlocal calls
        calls += 1
        return original_effects(stmt)

    monkeypatch.setattr(impl, "_statement_effects", counted)
    expected = reference()(body, selected)
    old_calls = calls
    calls = 0
    assert impl._slice_external_reads(body, selected) == expected == {"source"}
    assert old_calls == 128 * 129 // 2
    assert calls == 128
