from __future__ import annotations

import ast
from types import SimpleNamespace

import pytest

from helion._compiler.ast_read_writes import HELION_LANE_LOOP_VAR_ATTR
from helion._compiler.cute.hoist_lane_invariant_reductions import (
    hoist_lane_invariant_reductions,
)
from helion._compiler.cute.proven_loop_bounds import simplify_proven_loop_bounds


def rewrite(source: str, excluded: frozenset[str] = frozenset()) -> str:
    body = simplify_proven_loop_bounds(
        ast.parse(source).body,
        enabled=True,
        thread_block_dims=(32, 2, 1),
        constexpr_values={"BLOCK": 32},
        excluded_names=excluded,
    )
    return ast.unparse(ast.Module(body=body, type_ignores=[]))


SOURCE = """
pid = cutlass.Int32(cute.arch.block_idx()[0]) % 4
offset = pid * BLOCK
for lane in range(16):
    index = offset + cutlass.Int32(cute.arch.thread_idx()[1]) * 16 + cutlass.Int32(lane)
    record(index - offset < 32)
"""


def execute(source: str, block: int, row_thread: int) -> list[bool]:
    observed = []
    namespace = {
        "BLOCK": 32,
        "cutlass": SimpleNamespace(Int32=int),
        "cute": SimpleNamespace(
            arch=SimpleNamespace(
                block_idx=lambda: (block, 0, 0),
                thread_idx=lambda: (0, row_thread, 0),
            )
        ),
        "record": observed.append,
    }
    exec(source, namespace)
    return observed


def test_correlated_difference_uses_current_immutable_values() -> None:
    output = rewrite(SOURCE)
    assert "record(True)" in output
    for block in (0, 1, 2, 3, 1023, 2147483646):
        for row_thread in range(2):
            assert execute(output, block, row_thread) == execute(
                SOURCE, block, row_thread
            )


@pytest.mark.parametrize("limit", [1, 7, 16, 31])
def test_partial_lane_domains_retain_predicate(limit: int) -> None:
    source = SOURCE.replace("< 32", f"< {limit}")
    output = rewrite(source)
    assert "index - offset" in output
    for block in range(4):
        for row_thread in range(2):
            assert execute(output, block, row_thread) == execute(
                source, block, row_thread
            )


@pytest.mark.parametrize(
    "change",
    [
        "offset = runtime",
        "offset = cutlass.Int32(2147483640)",
        "offset = cutlass.Int32(-1)",
    ],
)
def test_unknown_or_overflowing_operand_is_not_cancelled(change: str) -> None:
    source = SOURCE.replace("offset = pid * BLOCK", change)
    assert "record(index - offset < 32)" in rewrite(source)


@pytest.mark.parametrize("name", ["offset", "index", "lane", "pid"])
def test_pending_rename_names_cannot_supply_proof(name: str) -> None:
    assert "record(index - offset < 32)" in rewrite(SOURCE, frozenset({name}))


def test_rebinding_does_not_cancel_a_different_value() -> None:
    source = SOURCE.replace("    record(", "    offset = 128\n    record(")
    assert "record(index - offset < 32)" in rewrite(source)


def test_branch_rebinding_invalidates_the_reaching_value() -> None:
    source = SOURCE.replace(
        "    record(", "    if unknown:\n        offset = runtime\n    record("
    )
    assert "record(index - offset < 32)" in rewrite(source)


def test_overflowing_inner_sum_stays_unproved_after_outer_cancellation() -> None:
    source = """
x = cutlass.Int32(cute.arch.thread_idx()[0])
bad = cutlass.Int32(2147483640) + x
record(bad - x < 2147483647)
"""
    assert "record(bad - x < 2147483647)" in rewrite(source)


@pytest.mark.parametrize("disjoint", [False, True])
def test_bounds_enable_unchanged_norm_hoisting_only_with_storage_proof(
    disjoint: bool,
) -> None:
    source = SOURCE.replace(
        "    record(index - offset < 32)",
        """    acc = cutlass.Float32(0)
    for k in cutlass.range_constexpr(4):
        loaded = (key.iterator + k).load() if index - offset < 32 else cutlass.Float32(0)
        value = cutlass.Float32(loaded)
        acc = acc + value * value
    norm = cutlass.Float32(cute.arch.warp_reduction_sum(acc, threads_in_group=32))
    state_value = (state.iterator + index).load()
    (out.iterator + index).store(norm + state_value)""",
    )
    body = ast.parse(source).body
    loop = body[-1]
    assert isinstance(loop, ast.For)
    setattr(loop, HELION_LANE_LOOP_VAR_ATTR, "lane")
    body = simplify_proven_loop_bounds(
        body,
        enabled=True,
        thread_block_dims=(32, 2, 1),
        constexpr_values={"BLOCK": 32},
    )
    before = next(
        n
        for n in ast.walk(loop)
        if isinstance(n, ast.Assign)
        and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id == "norm"
    )
    result = hoist_lane_invariant_reductions(
        body,
        tensor_names={"key", "state", "out"},
        proven_disjoint_tensor_pairs={frozenset({"key", "out"})} if disjoint else set(),
        thread_block_dims=(32, 2, 1),
    )
    moved = [
        n
        for n in result
        if isinstance(n, ast.Assign)
        and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id == "norm"
    ]
    assert bool(moved) is disjoint
    if moved:
        assert ast.dump(moved[0]) == ast.dump(before)
