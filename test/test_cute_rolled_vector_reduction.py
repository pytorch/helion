from __future__ import annotations

import ast
from types import SimpleNamespace

import numpy as np

from helion._compiler.cute.rolled_vector_reduction import split_rolled_vector_sums

PROGRAM = """
accumulator = cutlass.Float32(0)
for offset in range(3):
    for lane in range(2):
        base = offset * 16 + lane * 8
        for element in cutlass.range_constexpr(8):
            index = base + element
            value = values[index]
            accumulator = accumulator + value
answer = accumulator
"""


def test_independent_streams_cover_the_original_traversal():
    original = ast.parse(PROGRAM)
    changed = ast.Module(body=split_rolled_vector_sums(original.body), type_ignores=[])
    ast.fix_missing_locations(changed)
    namespace = {
        "cutlass": SimpleNamespace(Float32=np.float32, range_constexpr=range),
        "values": np.arange(48, dtype=np.float32),
    }
    exec(compile(changed, "<independent-sums>", "exec"), namespace)
    assert namespace["answer"] == np.arange(48, dtype=np.float32).sum()
    initializers = [
        statement
        for statement in changed.body
        if isinstance(statement, ast.Assign)
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id.startswith("_cute_rolled_sum_")
    ]
    assert len(initializers) == 8


def test_other_loop_carried_state_keeps_the_original_order():
    program = PROGRAM.replace("value = values[index]", "value = value + values[index]")
    original = ast.parse(program)
    before = ast.unparse(original)
    changed = ast.Module(body=split_rolled_vector_sums(original.body), type_ignores=[])
    assert ast.unparse(changed) == before
