from __future__ import annotations

import ast
from unittest.mock import patch

import pytest
import torch

from . import test_cute_chained_accumulator as accumulator
from ._cute_prepared_source import expand_prepared_root_source

pytestmark = accumulator.pytestmark


@pytest.fixture(scope="module")
def original_sources():
    args = accumulator._initialized_args(torch.bfloat16, 32)
    return {
        "initialized": accumulator._initialized_code(args),
        "initialized_old": accumulator._initialized_code(args, False),
        "major": accumulator._major_code(accumulator._major_args()),
        "late_old": accumulator._late_rhs_code(accumulator._late_rhs_args(), False),
        "late_new": accumulator._late_rhs_code(accumulator._late_rhs_args()),
    }


def test_expansion_preserves_wrong_initialized_bit(original_sources):
    tree = ast.parse(original_sources["initialized"])
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and ast.unparse(node.func) == "execute_prepared_continuation"
        and ast.unparse(node.args[1].elts[0].elts[3]) == "chain_1_mma"
    ]
    assert len(calls) == 1
    program = list(ast.literal_eval(calls[0].args[0]))
    assert program[0][0] == 4 and program[0][4] is True
    instruction = list(program[0])
    instruction[4] = False
    program[0] = tuple(instruction)
    calls[0].args[0] = ast.parse(repr(tuple(program)), mode="eval").body
    broken = ast.unparse(tree)
    assert (
        "chain_1_mma.set(tcgen05.Field.ACCUMULATE, False)"
        in expand_prepared_root_source(broken)
    )

    def code(args, enabled=True, **extra):
        return broken if enabled is True else original_sources["initialized_old"]

    with (
        patch.object(accumulator, "_initialized_code", side_effect=code),
        pytest.raises(AssertionError, match="chain_1_mma.set"),
    ):
        accumulator.test_typed_emitted_seed_and_join(torch.bfloat16, 32)


def test_expansion_preserves_wrong_operand_major(original_sources):
    tree = ast.parse(original_sources["major"])
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and ast.unparse(node.func) == "chain_sm100.make_trivial_tiled_mma"
    ]
    assert len(calls) == 2
    assert ast.unparse(calls[0].args[2]) == "cute.nvgpu.OperandMajorMode.K"
    calls[0].args[2] = ast.parse("cute.nvgpu.OperandMajorMode.MN", mode="eval").body
    with pytest.raises(AssertionError):
        accumulator._assert_majors(ast.unparse(tree), ("K", "MN"), ("K", "MN"))


def test_expansion_preserves_wrong_seed_store_value(original_sources):
    tree = ast.parse(original_sources["late_new"])
    stores = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and ast.unparse(node.func) == "prepared_tcgen_edge.execute_prepared_store"
    ]
    assert len(stores) == 1
    assert ast.unparse(stores[0].args[0]) == "chain_0_values"
    stores[0].args[0] = ast.Name(id="wrong_seed_values", ctx=ast.Load())
    broken = ast.unparse(tree)
    assert "wrong_seed_values" in expand_prepared_root_source(broken)
    with pytest.raises(AssertionError):
        accumulator._without_schedule_delta(original_sources["late_old"], broken)
