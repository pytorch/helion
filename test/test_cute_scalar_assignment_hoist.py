from __future__ import annotations

import ast

import pytest

from helion._compiler.ast_read_writes import HELION_LANE_LOOP_VAR_ATTR
from helion._compiler.cute.hoist_lane_invariant_reductions import (
    hoist_lane_invariant_reductions,
)

TENSORS = {"key", "state", "out"}
PAIRS = {frozenset((a, b)) for a in TENSORS for b in TENSORS if a != b}


def fixture(address="key.iterator + base", suffix=""):
    module = ast.parse(f"""
for row_lane in range(4):
    row = row_base + row_lane
    coefficients = cute.arch.load({address}, ir.VectorType.get([4], cutlass.Uint32.mlir_type))
    state_packet = cute.arch.load(state.iterator + row * 128 + base, ir.VectorType.get([4], cutlass.Uint16.mlir_type))
    value = cutlass.Float32(state_packet[0]) * cutlass.Float32(coefficients[0])
    (out.iterator + row).store(value)
    {suffix or "pass"}
""")
    setattr(module.body[0], HELION_LANE_LOOP_VAR_ATTR, "row_lane")
    return module.body


def transform(body, pairs=PAIRS):
    return hoist_lane_invariant_reductions(
        body,
        tensor_names=TENSORS,
        proven_disjoint_tensor_pairs=pairs,
        thread_block_dims=(32, 4, 1),
    )


def outside(body, target):
    return any(
        isinstance(stmt, ast.Assign)
        and isinstance(stmt.targets[0], ast.Name)
        and stmt.targets[0].id == target
        for stmt in body
    )


def test_assignment_load_hoist_without_invariant_reduction():
    original = fixture()
    coefficient = ast.dump(original[0].body[1], include_attributes=False)
    result = transform(original)
    assert outside(result, "coefficients")
    assert not outside(result, "state_packet")
    assert ast.dump(result[0], include_attributes=False) == coefficient
    loop = next(node for node in result if isinstance(node, ast.For))
    assert isinstance(loop.body[-2], ast.Expr)
    assert ast.unparse(loop.body[-2]) == "(out.iterator + row).store(value)"


@pytest.mark.parametrize(
    "address",
    [
        "key.iterator + base + row_lane",
        "key.iterator + (base if row_lane < 3 else 0)",
    ],
)
def test_row_varying_and_masked_addresses_remain(address):
    assert not outside(transform(fixture(address)), "coefficients")


def test_no_owner_proof_keeps_all_loads():
    result = transform(fixture(), set())
    assert not outside(result, "coefficients")
    assert not outside(result, "state_packet")


@pytest.mark.parametrize(
    "suffix",
    [
        "(key.iterator + row).store(value)",
        "unknown(coefficients)",
        "base = row_lane",
    ],
)
def test_written_owner_unknown_effect_and_loop_carried_pointer_decline(suffix):
    assert not outside(transform(fixture(suffix=suffix)), "coefficients")
