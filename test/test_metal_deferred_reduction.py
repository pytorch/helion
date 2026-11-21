"""A K loop's cross-thread sum moves after the loop, rendered on CPU without a
Metal device."""

from __future__ import annotations

import ast
import re
import textwrap

import pytest
import torch

import helion
from helion._compiler.metal.deferred_reduction import defer_group_sums
from helion._testing import skipUnlessBackends
from helion._testing.metal import render_msl
import helion.language as hl

pytestmark = [pytest.mark.cpu_only, skipUnlessBackends(["metal"])]

#: A GEMV on the scalar contraction: K's tile spread over 32 threads.
GEMV_CONFIG = helion.Config(
    block_sizes=[1, 8, 256], metal_matmul="scalar", num_threads=[1, 8, 32]
)


@helion.kernel(backend="metal", static_shapes=True)
def gemv(x: torch.Tensor, w: torch.Tensor, decay: hl.constexpr) -> torch.Tensor:
    m, k = x.size()
    n = w.size(0)
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            if decay:
                acc = acc * 0.5
            acc = torch.addmm(acc, x[tile_m, tile_k], w[tile_n, tile_k].T)
        out[tile_m, tile_n] = acc.to(x.dtype)
    return out


@helion.kernel(backend="metal", static_shapes=True)
def gemv_two_halves(
    x: torch.Tensor, w: torch.Tensor, x2: torch.Tensor, w2: torch.Tensor
) -> torch.Tensor:
    """Two dots into one accumulator per K tile, as a packed-weight GEMV's
    low and high codes."""
    m, k = x.size()
    n = w.size(0)
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = hl.dot(x[tile_m, tile_k], w[tile_n, tile_k].T, acc=acc)
            acc = hl.dot(x2[tile_m, tile_k], w2[tile_n, tile_k].T, acc=acc)
        out[tile_m, tile_n] = acc.to(x.dtype)
    return out


@helion.kernel(backend="metal", static_shapes=True)
def gemv_side_value(
    x: torch.Tensor, w: torch.Tensor, w2: torch.Tensor, snapshot: hl.constexpr
) -> torch.Tensor:
    """A K loop whose accumulator chain has a side value read after the
    loop: with ``snapshot``, a copy of the accumulator before each tile's
    dot; without, a second dot's sum into a value off the chain."""
    m, k = x.size()
    n = w.size(0)
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        side = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            if snapshot:
                side = acc
            else:
                side = hl.dot(x[tile_m, tile_k], w2[tile_n, tile_k].T, acc=acc)
            acc = hl.dot(x[tile_m, tile_k], w[tile_n, tile_k].T, acc=acc)
        out[tile_m, tile_n] = (acc + side).to(x.dtype)
    return out


@helion.kernel(backend="metal", static_shapes=True)
def row_sums(x: torch.Tensor) -> torch.Tensor:
    """Each row's sum over a K loop: a ``.sum()``, whose result reaches the
    add through a cast."""
    m, k = x.size()
    out = torch.empty([m], dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = acc + x[tile_m, tile_k].to(torch.float32).sum(-1)
        out[tile_m] = acc
    return out


@helion.kernel(backend="metal", static_shapes=True)
def row_sums_named(x: torch.Tensor, lane_sum_carry: torch.Tensor) -> torch.Tensor:
    """:func:`row_sums` plus a parameter and a kernel variable named as the
    pass would name its carries (``<sum>_carry``)."""
    m, k = x.size()
    out = torch.empty([m], dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(m):
        acc = lane_sum_carry[tile_m]
        lane_sum_1_carry = hl.full([tile_m], 1.0, dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = acc + x[tile_m, tile_k].to(torch.float32).sum(-1)
        out[tile_m] = acc + lane_sum_1_carry
    return out


def _sums_in_k_loop(msl: str) -> tuple[int, int]:
    """(group sums inside the K loop, group sums after it)."""
    lines = msl.splitlines()
    start = next(i for i, line in enumerate(lines) if "for (int offset_" in line)
    indent = len(lines[start]) - len(lines[start].lstrip())
    end = next(
        i
        for i in range(start + 1, len(lines))
        if lines[i].startswith(" " * indent + "}")
        and len(lines[i]) - len(lines[i].lstrip()) == indent
    )
    inside = sum("group_sum(" in line for line in lines[start:end])
    after = sum("group_sum(" in line for line in lines[end:])
    return inside, after


def _args(decay: bool) -> tuple[torch.Tensor, torch.Tensor, bool]:
    return (
        torch.zeros(1, 1024, dtype=torch.float16),
        torch.zeros(64, 1024, dtype=torch.float16),
        decay,
    )


def test_gemv_sums_threads_once_after_k_loop():
    msl = render_msl(gemv, _args(False), GEMV_CONFIG)
    inside, after = _sums_in_k_loop(msl)
    assert inside == 0 and after > 0


def test_accumulator_read_in_loop_keeps_its_sum():
    """``acc * 0.5`` reads the accumulator every iteration, so each tile's
    sum must land in it before the next."""
    msl = render_msl(gemv, _args(True), GEMV_CONFIG)
    inside, after = _sums_in_k_loop(msl)
    assert inside > 0
    assert after == 0


def test_chained_dots_sum_threads_once_after_k_loop():
    """Both dots' tile sums leave the loop: the first add's result is only an
    operand of the second, which assigns the accumulator."""
    args = (*_args(False)[:2], *_args(False)[:2])
    msl = render_msl(gemv_two_halves, args, GEMV_CONFIG)
    inside, after = _sums_in_k_loop(msl)
    _, one_dot = _sums_in_k_loop(render_msl(gemv, _args(False), GEMV_CONFIG))
    assert (inside, after) == (0, 2 * one_dot)


@pytest.mark.parametrize("snapshot", [True, False])
def test_side_value_off_the_chain_keeps_its_sums(snapshot: bool):
    """A value read after the loop that copies the accumulator, or takes a
    tile's sum off the chain back to it, holds a per-tile value, which
    sums moved after the loop would not give it."""
    x, w, _ = _args(False)
    msl = render_msl(gemv_side_value, (x, w, w, snapshot), GEMV_CONFIG)
    inside, after = _sums_in_k_loop(msl)
    assert inside > 0
    assert after == 0


def test_cast_sum_threads_once_after_k_loop():
    """A ``.sum()`` reaches its add through a cast of its result, which the
    rewrite drops with the add."""
    x = torch.zeros(4, 1024, dtype=torch.float16)
    config = helion.Config(block_sizes=[1, 256], num_threads=[1, 32])
    inside, after = _sums_in_k_loop(render_msl(row_sums, (x,), config))
    assert inside == 0 and after > 0


def test_carry_names_avoid_kernel_names():
    """A carry takes a name no parameter or statement of the kernel has."""
    args = (torch.zeros(4, 1024, dtype=torch.float16), torch.zeros(4))
    config = helion.Config(block_sizes=[1, 256], num_threads=[1, 32])
    msl = render_msl(row_sums_named, args, config)
    inside, after = _sums_in_k_loop(msl)
    assert inside == 0 and after > 0
    carries = set(re.findall(r"auto (\w+) = float\(0\);", msl)) - {"acc"}
    assert carries
    assert not {"lane_sum_carry", "lane_sum_1_carry"} & carries


gemv_dynamic = helion.kernel(gemv.fn, backend="metal", static_shapes=False)


def test_sums_after_a_loop_that_may_not_run_are_guarded():
    """With K not a constant, the loop may run no iteration: the sums after
    it are added only when it ran, so a ``-0.0`` accumulator stays."""
    msl = render_msl(gemv_dynamic, _args(False), GEMV_CONFIG)
    inside, after = _sums_in_k_loop(msl)
    assert inside == 0 and after > 0
    lines = msl.splitlines()
    first = next(
        i
        for i, line in enumerate(lines)
        if "group_sum(" in line and i > msl[: msl.index("for (int offset_")].count("\n")
    )
    assert lines[first - 1].strip().startswith("if ((0 < "), lines[first - 1]
    static = render_msl(gemv, _args(False), GEMV_CONFIG)
    assert "if ((0 < " not in static


def _deferred(loop_head: str) -> str:
    """The GEMV-shaped loop under ``loop_head`` after the pass, as source."""
    source = textwrap.dedent(
        f"""\
        {loop_head}
            acc_copy = acc
            part = float(0)
            tile_sum = helion_red.group_sum(site, scratch, static_cast(float, part), tid)
            acc = static_cast(float, acc_copy + tile_sum)
        """
    )
    stmts = defer_group_sums(ast.parse(source).body, ())
    return "\n".join(ast.unparse(s) for s in stmts)


def test_only_ascending_range_loops_defer():
    """An ascending range defers its sum; a descending one (a negative literal
    step, whose loop tests ``offset > end``, not the guard's ``begin < end``)
    and a loop that is no range (a fragment run's) keep theirs."""
    ascending = _deferred("for offset_1 in tl.range(0, k, _BLOCK_SIZE_1):")
    assert ascending.index("group_sum") > ascending.index("tile_sum_carry + ")
    for head in (
        "for offset_1 in range(3, -1, -1):",
        "for _fs in _helion_mpp_fragments(acc, 4, [], 0, None, None):",
    ):
        kept = _deferred(head)
        assert "_carry" not in kept, kept
