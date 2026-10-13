from __future__ import annotations

import ast

import pytest

from test.test_cute_hoist_lane_invariant_reductions import _rewrite

SOURCE = """
for row_lane in range(4):
    row = row_base + row_lane
    acc = cutlass.Float32(0)
    for element in cutlass.range_constexpr(4):
        gate_value = (gate.iterator + base).load()
        gain = cute.math.exp2(cutlass.Float32(gate_value))
        norm = cute.math.rsqrt(seed)
        value = (state.iterator + row * 128 + element).load()
        acc = acc + cutlass.Float32(value) * gain * norm
    (out.iterator + row).store(acc)
"""


def scopes(source=SOURCE, **kwargs):
    result = _rewrite(source, **kwargs)
    row = next(stmt for stmt in result if isinstance(stmt, ast.For))
    before = ast.unparse(ast.Module(body=result[: result.index(row)], type_ignores=[]))
    return before, ast.unparse(row), result


def test_ordinary_nested_scalar_chain_hoists_without_branch():
    before, row, _ = scopes()
    assert "gate_value = (gate.iterator + base).load()" in before
    assert "gain = cute.math.exp2(cutlass.Float32(gate_value))" in before
    assert "norm = cute.math.rsqrt(seed)" in before
    assert "gain =" not in row and "gate_value =" not in row
    assert "acc = acc + cutlass.Float32(value) * gain * norm" in row
    assert "state.iterator + row * 128 + element" in row


def test_unproven_store_alias_prevents_cross_row_load_motion():
    before, row, _ = scopes(disjoint=set())
    assert "gate_value =" not in before and "gate_value =" in row


@pytest.mark.parametrize(
    "replacement",
    [
        "(gate.iterator + element).load()",
        "(gate.iterator + base).load() if element < 2 else cutlass.BFloat16(0)",
        "unknown(gate)",
    ],
)
def test_varying_mask_or_unknown_call_keeps_chain_in_feature_loop(replacement):
    source = SOURCE.replace("(gate.iterator + base).load()", replacement)
    before, row, _ = scopes(source)
    assert "gate_value =" not in before
    assert row.index("gate_value =") > row.index("for element")


@pytest.mark.parametrize(
    "extra",
    [
        "gate_value.value = element",
        "operator.setitem(alias, 0, element)",
        "cute.arch.barrier()",
        "(gate.iterator + base).store(value)",
    ],
)
def test_mutation_or_unknown_effect_keeps_gate_local(extra):
    source = SOURCE.replace(
        "        acc = acc +", "        " + extra + "\n        acc = acc +"
    )
    before, row, _ = scopes(source)
    assert "gate_value =" not in before
    assert row.index("gate_value =") > row.index("for element")


def test_zero_trip_loop_does_not_introduce_load_or_math():
    before, row, _ = scopes(SOURCE.replace("range_constexpr(4)", "range_constexpr(0)"))
    assert "gate_value =" not in before and "gate_value =" in row


def test_read_before_definition_retains_order():
    source = SOURCE.replace(
        "        gate_value =",
        "        previous = gain + element\n        gate_value =",
    )
    before, row, _ = scopes(source)
    assert "gain =" not in before
    assert row.index("previous =") < row.index("gain =")


def test_rename_collision_still_rejects_scalar_motion():
    source = SOURCE.replace(
        "    (out.iterator", "    alias = row_lane\n    (out.iterator"
    )
    before, row, _ = scopes(source, rename_groups={"gain": "alias"})
    assert "gain =" not in before and "gain =" in row
