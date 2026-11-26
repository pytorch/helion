"""GPU-free coverage for what a lane split reruns under register reloads.

``split_lane_loop_reductions`` lowers a lane loop holding a
``_helion_lane_reduce`` marker into an accumulate pass, the cross-thread
combine and a consume pass.  Under ``cute_reduction_reloads`` "register", the
producers of the reduction's input that read no lane run once, ahead of the
accumulate pass, instead of once per lane inside it (and again between the
passes), and the consume pass of a lane loop unrolled at trace time reads the
values it needs of the accumulate pass from per-lane trace-time lists instead
of rerunning their producers: a softmax backward's exp and division.  Other
modes keep both passes as they were (bit for bit).
"""

from __future__ import annotations

import ast
from unittest.mock import patch

import pytest

from helion._compiler import tile_strategy
from helion._compiler.cute.lane_loops import split
from helion._testing import skipUnlessBackends

pytestmark = [pytest.mark.cpu_only, skipUnlessBackends(["cute"])]

_MARKER = (
    "_helion_lane_reduce(v_3, 'sum', cutlass.Float32(0), threads_in_group=32, "
    "group_span=32, group_lane_expr='cutlass.Int32(cute.arch.thread_idx()[0])', "
    "group_count=4, owner_lane='lane')"
)
# A softmax backward's row: p = exp(x - max) / sum, the sum of g * p, and
# g * p - p * that sum stored per lane.
_PROGRAM = f"""
for lane in cutlass.range_constexpr(4):
    col = base + cutlass.Int32(lane)
    x = cutlass.Float32((xs.iterator + row * 128 + col).load())
    row_max = (maxes.iterator + row).load()
    row_sum = (sums.iterator + row).load()
    v_1 = cute.math.exp(cutlass.Float32(x - row_max))
    v_2 = v_1 / row_sum
    g = cutlass.Float32((gs.iterator + row * 128 + col).load())
    v_3 = g * v_2
    dot = cutlass.Float32({_MARKER})
    v_4 = v_3 - v_2 * dot
    (out.iterator + row * 128 + col).store(cutlass.BFloat16(v_4))
"""
_TENSORS = {"xs", "maxes", "sums", "gs", "out"}


def _split(*, registers: bool) -> list[ast.AST]:
    loop = ast.parse(_PROGRAM).body[0]
    assert isinstance(loop, ast.For)
    setattr(loop, tile_strategy.HELION_LANE_LOOP_VAR_ATTR, "lane")
    with patch.object(split, "_reloads_in_registers", return_value=registers):
        return split.split_lane_loop_reductions(
            [loop],
            uniform_names={*_TENSORS, "row"},
            # Each thread of the reduction's group holds 4 columns of its own.
            thread_axis_names={"base": frozenset({0})},
            proven_disjoint_tensor_pairs={
                frozenset({"out", name}) for name in _TENSORS - {"out"}
            },
        )


def _loops(body: list[ast.AST]) -> list[ast.For]:
    return [statement for statement in body if isinstance(statement, ast.For)]


def test_lane_invariant_producers_run_once_ahead_of_the_accumulate_pass() -> None:
    """The row's max and sum read no lane: they are loaded once, ahead of
    the accumulate pass, where they were loaded for each of its 4 lanes, and
    not again between the passes."""
    body = _split(registers=True)
    accumulate, consume = _loops(body)
    ahead = [ast.unparse(s) for s in body[: body.index(accumulate)]]
    assert any(line.startswith("row_max = ") for line in ahead), ahead
    assert any(line.startswith("row_sum = ") for line in ahead), ahead
    between = body[body.index(accumulate) + 1 : body.index(consume)]
    for statement in [*between, *accumulate.body]:
        assert not ast.unparse(statement).startswith(("row_max", "row_sum"))


def test_a_producer_reading_one_that_stays_stays_too() -> None:
    """``y`` reads no lane, but ``doc``, which it reads, is not proven
    relocatable and stays in the loop: hoisted ahead of the loop, ``y``
    would read ``doc`` before the loop assigns it. ``w``, which reads
    neither, is hoisted."""
    body = ast.parse(
        "doc = _cute_scalar_policy_load(seeds.iterator + base, cutlass.Int64)\n"
        "y = doc + cutlass.Int64(1)\n"
        "v = cute.arch.load(x.iterator + lane) * cutlass.Float32(y)\n"
        "w = cutlass.Float32(base) * cutlass.Float32(2.0)\n"
    ).body
    assert split._lane_invariant_producers(body, [0, 1, 2, 3], "lane", {}) == {3}


def test_a_producer_writing_what_one_that_stays_writes_stays_too() -> None:
    """``s``'s reset reads no lane, but the serial loop that sums into ``s``
    stays in the lane loop: hoisted, the reset would run once and each lane
    would add its sums to the last lane's. A producer that writes what one
    before it reads (``t`` reads the last lane's ``u``) stays too."""
    body = ast.parse(
        "s = cutlass.Float32(0.0)\n"
        "for j in cutlass.range(k):\n"
        "    s = s + cutlass.Float32((w.iterator + j).load())\n"
        "v = cute.arch.load(x.iterator + lane) * s\n"
    ).body
    assert split._lane_invariant_producers(body, [0, 1, 2], "lane", {}) == set()
    body = ast.parse(
        "t = u * cutlass.Float32(2.0)\n"
        "u = cutlass.Float32(base)\n"
        "v = cute.arch.load(x.iterator + lane) * t\n"
    ).body
    assert split._lane_invariant_producers(body, [0, 1, 2], "lane", {}) == set()


def test_other_reload_modes_keep_both_passes() -> None:
    """Without register reloads the split keeps its passes: the row's max is
    loaded in the accumulate pass and again between the passes, and the
    consume pass reruns the exp."""
    body = _split(registers=False)
    accumulate, consume = _loops(body)
    assert "row_max = " in ast.unparse(accumulate)
    assert "cute.math.exp" in ast.unparse(consume)
    assert "_lane_stash" not in "\n".join(ast.unparse(s) for s in body)


def test_register_reloads_read_the_accumulate_pass_values() -> None:
    """Under register reloads, the accumulate pass keeps p and g * p per lane
    and the consume pass reads them, with no exp, division or load of x."""
    body = _split(registers=True)
    accumulate, consume = _loops(body)
    first, second = ast.unparse(accumulate), ast.unparse(consume)
    assert "v_2_lane_stash.append(v_2)" in first, first
    assert "v_3_lane_stash.append(v_3)" in first, first
    assert "v_2 = v_2_lane_stash[lane]" in second, second
    assert "v_3 = v_3_lane_stash[lane]" in second, second
    for recomputed in ("cute.math.exp", "/ row_sum", "xs.iterator"):
        assert recomputed not in second, second
    ahead = [ast.unparse(s) for s in body[: body.index(accumulate)]]
    assert "v_2_lane_stash = []" in ahead and "v_3_lane_stash = []" in ahead


def test_a_frontier_name_assigned_twice_gets_no_stash() -> None:
    """``v = exp(x); v = v * 2`` would append to v's stash twice per lane and
    the consume pass would read other lanes' values: no stash then."""
    loop = ast.parse("for lane in cutlass.range_constexpr(4):\n    pass").body[0]
    assert isinstance(loop, ast.For)

    def frontier(source: str) -> object:
        body = ast.parse(source).body
        reruns = [True] * (len(body) - 1) + [False]
        return split._stash_lane_frontier(
            loop, "lane", body, reruns, list(range(len(body))), body, body[:-1], {}
        )

    assert frontier("v = cute.math.exp(x)\nw = v * 2.0\nout = w + 1.0") is not None
    assert frontier("v = cute.math.exp(x)\nv = v * 2.0\nout = v + 1.0") is None
