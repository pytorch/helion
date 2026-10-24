"""Tests for ``helion/_compiler/cute/fold_lane_reductions.py``.

A device-loop tile wider than ``threads x V`` gives each thread several
packets per trip (a vectorized lane loop).  A reduction over that tile used
to be emitted once per lane, reducing one lane's packet, with the consumers
run once per lane: an online fold (softmax) still computed the program, any
other consumer squared or stored per-lane partials, and every lane paid a
cross-thread round.  The pass unrolls such lane loops, folds the per-lane
partials across the lanes and runs each cross-thread reduction once per
trip.
"""

from __future__ import annotations

import ast
import os
import re
from unittest.mock import patch

import pytest
import torch

from test._cute_binding import _cpu_bind
from test._cute_binding import _mock_cuda_unavailable
from test.cute_population_contracts import _target

import helion
from helion import exc
from helion._compiler.cute.fold_lane_reductions import fold_lane_reductions
from helion._compiler.cute.fold_lane_reductions import fold_site_key
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
from helion._testing import skipIfRefEager
from helion._testing import skipUnlessBackends
import helion.language as hl

REDUCE = "_cute_grouped_reduce_shared_two_stage("


def squared_tile_sums(x: torch.Tensor) -> torch.Tensor:
    """``acc += sum(tile) ** 2``: squares per-lane partials unless the tile
    is reduced whole."""
    m, n = x.size()
    out = torch.empty([m], dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m], dtype=torch.float32)
        for tile_n in hl.tile(n):
            s = torch.sum(x[tile_m, tile_n].to(torch.float32), dim=-1)
            acc = acc + s * s
        out[tile_m] = acc
    return out


def online_softmax_rows(x: torch.Tensor) -> torch.Tensor:
    """The online max / rescaled sum of ``examples/grpo_loss.py``'s row loop."""
    m, n = x.size()
    lse = torch.empty([m], dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(m):
        running_max = hl.full([tile_m], float("-inf"), dtype=torch.float32)
        running_sum = hl.zeros([tile_m], dtype=torch.float32)
        for tile_n in hl.tile(n):
            values = x[tile_m, tile_n].to(torch.float32)
            new_max = torch.maximum(running_max, torch.amax(values, dim=-1))
            alpha = torch.exp(running_max - new_max)
            running_sum = running_sum * alpha + torch.sum(
                torch.exp(values - new_max[:, None]), dim=-1
            )
            running_max = new_max
        lse[tile_m] = running_max + torch.log(running_sum)
    return lse


def shifted_sum_rows(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """``sum(exp(y - max(x)))`` as two separate one-trip sweeps over a
    register block size: the lane loop's combine is already moved after the
    loop by the two-pass passes (the cluster online-pair forms build on it)."""
    m, n = x.size()
    sums = torch.empty([m], dtype=torch.float32, device=x.device)
    block_size_m = hl.register_block_size(m)
    block_size_n = hl.register_block_size(n)
    for tile_m in hl.tile(m, block_size=block_size_m):
        mi = hl.full([tile_m], float("-inf"), dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=block_size_n):
            mi = torch.maximum(mi, torch.amax(x[tile_m, tile_n], dim=1))
        di = hl.zeros([tile_m], dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=block_size_n):
            di = di + torch.exp(y[tile_m, tile_n] - mi[:, None]).sum(dim=1)
        sums[tile_m] = di
    return sums


def softmax_shifted_by_max_of_other_rows(
    x: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    """``sum(exp(y - running max of x))`` in one device loop: the summand's
    minuend is not the value the max folded, so the lanes keep two folds."""
    m, n = x.size()
    sums = torch.empty([m], dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(m):
        running_max = hl.full([tile_m], float("-inf"), dtype=torch.float32)
        running_sum = hl.zeros([tile_m], dtype=torch.float32)
        for tile_n in hl.tile(n):
            values = x[tile_m, tile_n].to(torch.float32)
            other = y[tile_m, tile_n].to(torch.float32)
            new_max = torch.maximum(running_max, torch.amax(values, dim=-1))
            running_sum = running_sum * torch.exp(running_max - new_max) + torch.sum(
                torch.exp(other - new_max[:, None]), dim=-1
            )
            running_max = new_max
        sums[tile_m] = running_sum
    return sums


def weighted_softmax_rows(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """``sum(w * exp(x - running max))``: a factor beside the exponential keeps
    the two folds."""
    m, n = x.size()
    sums = torch.empty([m], dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(m):
        running_max = hl.full([tile_m], float("-inf"), dtype=torch.float32)
        running_sum = hl.zeros([tile_m], dtype=torch.float32)
        for tile_n in hl.tile(n):
            values = x[tile_m, tile_n].to(torch.float32)
            weights = w[tile_m, tile_n].to(torch.float32)
            new_max = torch.maximum(running_max, torch.amax(values, dim=-1))
            running_sum = running_sum * torch.exp(running_max - new_max) + torch.sum(
                weights * torch.exp(values - new_max[:, None]), dim=-1
            )
            running_max = new_max
        sums[tile_m] = running_sum
    return sums


def two_same_op_loops(x: torch.Tensor) -> torch.Tensor:
    """Two sequential device loops with the same ``sum`` reduction (a
    two-pass statistic): their grouped warp reduces have identical calls."""
    m, n = x.size()
    out = torch.empty([m], dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m], dtype=torch.float32)
        for tile_n in hl.tile(n):
            s = torch.sum(x[tile_m, tile_n].to(torch.float32), dim=-1)
            acc = acc + s * s
        acc2 = hl.zeros([tile_m], dtype=torch.float32)
        for tile_n2 in hl.tile(n):
            s2 = torch.sum(x[tile_m, tile_n2].to(torch.float32), dim=-1)
            acc2 = acc2 + s2 * s2
        out[tile_m] = acc + acc2
    return out


def two_pass_softmax(x: torch.Tensor) -> torch.Tensor:
    """``examples/softmax.py``'s two-pass softmax: the first device loop's
    lane body also fills the register cache the second loop reads."""
    m, n = x.size()
    out = torch.empty_like(x)
    block_size_m = hl.register_block_size(m)
    block_size_n = hl.register_block_size(n)
    for tile_m in hl.tile(m, block_size=block_size_m):
        mi = hl.full([tile_m], float("-inf"), dtype=torch.float32)
        di = hl.zeros([tile_m], dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=block_size_n):
            values = x[tile_m, tile_n]
            local_amax = torch.amax(values, dim=1)
            mi_next = torch.maximum(mi, local_amax)
            di = di * torch.exp(mi - mi_next) + torch.exp(
                values - mi_next[:, None]
            ).sum(dim=1)
            mi = mi_next
        for tile_n in hl.tile(n, block_size=block_size_n):
            values = x[tile_m, tile_n]
            out[tile_m, tile_n] = torch.exp(values - mi[:, None]) / di[:, None]
    return out


def running_int_sums(x: torch.Tensor) -> torch.Tensor:
    """An integer reduction: ``hoist_warp_reduce`` leaves it per element
    inside the V-loop, a form the lane fold does not model."""
    m, n = x.size()
    out = torch.empty([m], dtype=torch.int64, device=x.device)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m], dtype=torch.int64)
        for tile_n in hl.tile(n):
            acc = acc + torch.sum(x[tile_m, tile_n], dim=-1)
        out[tile_m] = acc
    return out


def _config(block_n: int, threads: int, vec: int, **extra: object) -> helion.Config:
    return helion.Config(
        block_sizes=[1, block_n],
        num_threads=[1, threads],
        cute_vector_widths=[1, vec],
        cute_lane_layouts=["blocked", "strided"],
        **extra,
    )


def _device_loop_body(code: str) -> str:
    """The text of the ``hl.tile(n)`` device loop."""
    tree = ast.parse(code)
    kernel = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("_helion_")
    )
    loops = [
        node
        for node in ast.walk(kernel)
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id.startswith("tile_offset_")
    ]
    assert len(loops) == 1, [ast.unparse(loop.target) for loop in loops]
    return ast.unparse(loops[0])


# --------------------------------------------------------------------------
# CPU code generation


@pytest.fixture
def _cpu_only():
    with (
        _target(),
        _mock_cuda_unavailable(),
        patch(
            "helion._compiler.reduction_strategy._cute_shared_memory_budget_bytes",
            return_value=232448,
        ),
    ):
        yield


def _bind(fn, n: int = 16384, dtype: torch.dtype = torch.bfloat16):
    kernel = helion.kernel(
        fn, backend="cute", static_shapes=True, autotune_effort="full"
    )
    return _cpu_bind(kernel, (torch.empty((8, n), dtype=dtype),))


@skipIfRefEager("generates code")
@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("unroll", [1, 2])
def test_two_lanes_fold_into_one_reduction_per_trip(_cpu_only, unroll: int) -> None:
    """128 threads x V16 over a 4096-wide tile: two packets per thread.  Each
    reduction runs once per trip over a tree of per-lane partials, whether
    or not ``cute_lane_unroll`` asked for the loads-first unroll."""
    bound = _bind(online_softmax_rows)
    extra = {"cute_lane_unroll": unroll} if unroll > 1 else {}
    code = bound.to_code(_config(4096, 128, 16, **extra))
    loop = _device_loop_body(code)
    assert loop.count(REDUCE) == 2
    assert loop.count("_fold_parts_") >= 4  # two lists: init + append (+ tree reads)
    assert "cutlass.range_constexpr(2)" in loop
    assert "for lane_1 in range(" not in loop
    # Each fold keeps a per-lane accumulator and combines the two partials once.
    assert "cute.arch.fmax(_fold_parts_" in loop
    assert "] + _fold_parts_" in loop
    # The online update runs once per trip, after the folded reductions.
    tree = ast.parse(loop).body[0]
    assert isinstance(tree, ast.For)
    top = [ast.unparse(stmt).split("\n")[0] for stmt in tree.body]
    reduce_positions = [i for i, text in enumerate(top) if REDUCE in text]
    assert len(reduce_positions) == 2
    assert not any(
        text.startswith("for lane_1") for text in top[reduce_positions[-1] :]
    )


@skipIfRefEager("generates code")
@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("fast_math", [False, True])
def test_online_softmax_lanes_fuse_into_one_pass(_cpu_only, fast_math: bool) -> None:
    """max then sum of exp(x - max): one lane loop over the packets (per-lane
    max, per-lane sum against the lane's frame), the lane sums rescaled to the
    global frame after the cross-thread max.  Both exponential spellings:
    ``exp2((x - m) * c)`` and fast-math ``exp2(fma(x, c, -m * c))``."""
    kernel = helion.kernel(
        online_softmax_rows,
        backend="cute",
        static_shapes=True,
        autotune_effort="full",
        fast_math=fast_math,
    )
    bound = _cpu_bind(kernel, (torch.empty((8, 16384), dtype=torch.bfloat16),))
    loop = _device_loop_body(bound.to_code(_config(16384, 128, 16)))
    compute_loops = [
        line
        for line in loop.splitlines()
        if line.strip() == "for lane_1 in cutlass.range_constexpr(8):"
    ]
    assert len(compute_loops) == 2  # the load phase and ONE fused fold
    assert loop.count(REDUCE) == 2
    assert "_pair_root" in loop and "_pair_frame" in loop
    assert re.search(
        r"_pair_root\w* = cutlass\.Float32\(0\) if _helion_vfold_acc_0 == cutlass\.Float32\(float\('-inf'\)\) else _helion_vfold_acc_0",
        loop,
    )
    assert loop.count("* cute.math.exp2(") == 8  # one rescale per lane
    if fast_math:
        assert "fastmath=True)" in loop.split("_pair_frame")[2]
    # the per-lane frame feeds the lane's exponentials, the global one the online update
    assert re.search(
        r"v_\d+_lane = cute\.math\.max\(cutlass\.Float32\(running_max\)", loop
    )


@skipIfRefEager("generates code")
@skipUnlessBackends(["cute"])
@pytest.mark.parametrize(
    "fn", [softmax_shifted_by_max_of_other_rows, weighted_softmax_rows]
)
def test_other_minuend_or_weighted_exponential_keeps_two_folds(_cpu_only, fn) -> None:
    kernel = helion.kernel(
        fn, backend="cute", static_shapes=True, autotune_effort="full"
    )
    bound = _cpu_bind(
        kernel,
        (
            torch.empty((8, 16384), dtype=torch.bfloat16),
            torch.empty((8, 16384), dtype=torch.bfloat16),
        ),
    )
    loop = _device_loop_body(bound.to_code(_config(16384, 128, 16)))
    assert "_pair_root" not in loop
    assert loop.count(REDUCE) == 2
    assert loop.count("_fold_parts_") >= 4


@skipIfRefEager("generates code")
@skipUnlessBackends(["cute"])
def test_single_lane_code_is_unchanged(_cpu_only) -> None:
    """One packet per thread per trip reduces the whole tile already: the
    recorded site needs no fold and the knob-off code is kept (no lists,
    the rolled ``range(1)`` lane loop the prefetch pipeline matches)."""
    bound = _bind(online_softmax_rows)
    loop = _device_loop_body(bound.to_code(_config(2048, 128, 16)))
    assert loop.count(REDUCE) == 2
    assert "_fold_parts_" not in loop
    assert "for lane_1 in range(1):" in loop


@skipIfRefEager("generates code")
@skipUnlessBackends(["cute"])
def test_non_fold_consumer_reduces_the_whole_tile(_cpu_only) -> None:
    """``acc += sum(tile) ** 2`` with four packets per thread: one reduction
    per trip over the four per-lane partials, the square once."""
    bound = _bind(squared_tile_sums)
    loop = _device_loop_body(bound.to_code(_config(8192, 128, 16)))
    assert loop.count(REDUCE) == 1
    assert "cutlass.range_constexpr(4)" in loop
    assert loop.count("_fold_tree_") >= 3  # 4 partials -> 2 -> 1


@skipIfRefEager("generates code")
@skipUnlessBackends(["cute"])
def test_fold_under_a_row_lane_loop(_cpu_only) -> None:
    """Two rows per CTA walked by a rolled row lane loop (one thread on the
    rows) around the vectorized reduction axis: the knob unrolls nests from
    their outermost loop only, but the reduction axis still unrolls and folds
    inside the row loop."""
    bound = _bind(online_softmax_rows)
    code = bound.to_code(
        helion.Config(
            block_sizes=[2, 4096],
            num_threads=[1, 128],
            cute_vector_widths=[1, 16],
            cute_lane_layouts=["blocked", "strided"],
        )
    )
    assert "for lane_0 in range(2):" in code  # the row lane loop wraps the device loop
    loop = _device_loop_body(code)
    assert loop.count(REDUCE) == 2
    assert "_fold_parts_" in loop
    assert "cutlass.range_constexpr(2)" in loop


@skipIfRefEager("generates code")
@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("cluster", [1, 2])
def test_lane_loop_already_reduced_after_it_keeps_its_form(
    _cpu_only, cluster: int
) -> None:
    """A one-trip sweep whose lane loop the two-pass passes already
    accumulate across its lanes (one combine after the loop) is not a form
    this pass models: no forced unroll, no fold, the rolled lane loop and the
    cluster exchange form stay as before (the cluster online-pair rewrite
    reads that form and must still decline this minuend)."""
    kernel = helion.kernel(
        shifted_sum_rows, backend="cute", static_shapes=True, autotune_effort="full"
    )
    bound = _cpu_bind(kernel, (torch.empty((4, 32768)), torch.empty((4, 32768))))
    code = bound.to_code(
        helion.Config(
            block_sizes=[1, 32768],
            num_threads=[0, 256],
            cute_vector_widths=[1, 4],
            cute_lane_layouts=["blocked", "strided"],
            cute_cluster_n=cluster,
        )
    )
    assert "_lane_unroll_loads_" not in code
    assert "_fold_parts_" not in code
    lanes = 32 // cluster  # 32768 / (256 threads x V4), split across the cluster
    assert re.search(rf"for lane_\d+ in range\({lanes}\):", code)
    assert f"range_constexpr({lanes})" not in code
    assert "_cute_grouped_reduce_cluster_online_pair(" not in code
    expected_reduce = "_cute_grouped_reduce_cluster(" if cluster > 1 else REDUCE
    assert expected_reduce in code


@skipIfRefEager("generates code")
@skipUnlessBackends(["cute"])
def test_scalar_lanes_keep_the_two_pass_marker_path(_cpu_only) -> None:
    """A scalar lane loop (V = 1) is the two-pass lane marker's shape; the
    fold records no site there."""
    bound = _bind(squared_tile_sums, dtype=torch.float32)
    code = bound.to_code(_config(4096, 128, 1))
    assert "_fold_parts_" not in code


@skipIfRefEager("generates code")
@skipUnlessBackends(["cute"])
def test_two_loops_with_identical_reduce_calls_both_fold(_cpu_only) -> None:
    """Two device loops reducing with the same op on the grouped warp path
    (32 threads per row, two rows per CTA) emit identical reduce calls;
    sites are recorded per lane loop, so both loops fold."""
    bound = _bind(two_same_op_loops)
    code = bound.to_code(
        helion.Config(
            block_sizes=[2, 1024, 1024],
            num_threads=[2, 32, 32],
            cute_vector_widths=[1, 16, 16],
            cute_lane_layouts=["blocked", "strided", "strided"],
        )
    )
    assert code.count("_cute_grouped_reduce_warp(") == 2
    assert not re.search(r"for lane_\d+ in range\((?!1\))", code)
    assert code.count("cutlass.range_constexpr(2)") >= 2
    assert len(re.findall(r"^\s*_fold_tree_\w* = ", code, re.MULTILINE)) == 2


@skipIfRefEager("generates code")
@skipUnlessBackends(["cute"])
def test_single_warp_direct_reduce_is_folded(_cpu_only) -> None:
    """A one-warp CTA takes the direct ``cute.arch.warp_reduction_*`` path;
    inside a vectorized lane loop it is recorded and folded like the
    strided forms (the online pair included)."""
    loop = _device_loop_body(_bind(squared_tile_sums).to_code(_config(1024, 32, 16)))
    assert loop.count("cute.arch.warp_reduction_sum(") == 1
    assert "_fold_parts_" in loop
    assert "cutlass.range_constexpr(2)" in loop
    assert "for lane_1 in range(" not in loop
    loop = _device_loop_body(_bind(online_softmax_rows).to_code(_config(1024, 32, 16)))
    assert loop.count("cute.arch.warp_reduction_max(") == 1
    assert loop.count("cute.arch.warp_reduction_sum(") == 1
    assert "_pair_root" in loop
    assert "for lane_1 in range(" not in loop


@skipIfRefEager("generates code")
@skipUnlessBackends(["cute"])
def test_register_cache_fill_runs_once_per_lane_after_the_folds(
    _cpu_only, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A one-warp two-pass softmax with two packets per thread: the first
    loop's lane body also fills the row cache the second loop reads
    (``fuse_two_pass_loads``).  The fill feeds no reduction, so it runs in
    a trailing lane loop after the fused online pair.

    The second loop divides by the row sum; the plain division keeps that
    loop a single copy (the ``redo`` form of ``cute_exact_div_form`` renders
    a second, guarded copy of it for the div.rn re-run), so the device loops
    of the code are the two of the source."""
    monkeypatch.setenv("HELION_DISABLE_ONLINE_TO_3PASS", "1")
    kernel = helion.kernel(
        two_pass_softmax, backend="cute", static_shapes=True, autotune_effort="full"
    )
    bound = _cpu_bind(kernel, (torch.empty((64, 2048), dtype=torch.float16),))
    code = bound.to_code(
        helion.Config(
            block_sizes=[1, 256],
            num_threads=[0, 32],
            cute_vector_widths=[1, 4],
            cute_exact_div_form="plain",
        )
    )
    assert "_pair_root" in code
    match = re.search(r"(_fuse_cache_\d+) = cute\.make_rmem_tensor\(", code)
    assert match is not None
    cache = match.group(1)
    loops = [
        node
        for node in ast.walk(ast.parse(code))
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id.startswith("tile_offset_")
    ]
    assert len(loops) == 2
    top = [ast.unparse(stmt).split("\n")[0] for stmt in loops[0].body]
    lane_loops = [i for i, line in enumerate(top) if "range_constexpr(2)" in line]
    assert len(lane_loops) == 3  # loads, the fused pair fold, the cache fill
    fills = [
        ast.unparse(stmt)
        for stmt in loops[0].body[lane_loops[-1]].body  # type: ignore[attr-defined]
    ]
    assert sum(f"{cache}[" in text for text in fills) == 4
    assert all(
        f"{cache}[" not in ast.unparse(loops[0].body[i]) for i in lane_loops[:-1]
    )
    assert not re.search(r"for lane_\d+ in range\(", ast.unparse(loops[0]))


@skipIfRefEager("generates code")
@skipUnlessBackends(["cute"])
def test_online_pair_zeroes_the_rescaled_term_of_an_empty_lane(_cpu_only) -> None:
    """A lane whose max is -inf has an exact zero sum but a guarded frame
    F(0); its rescale factor exp(F(0) - F(m)) overflows for rows far below
    zero, so the term is selected to zero rather than multiplied."""
    loop = _device_loop_body(
        _bind(online_softmax_rows).to_code(_config(16384, 128, 16))
    )
    guarded = re.findall(
        r"cutlass\.Float32\(0\) if (_fold_parts_\w*)\[(\d+)\] == "
        r"cutlass\.Float32\(float\('-inf'\)\) else (_fold_parts_\w*)\[(\d+)\] \* "
        r"cute\.math\.exp2\(",
        loop,
    )
    assert len(guarded) == 8
    assert [int(index) for _, index, _, _ in guarded] == list(range(8))
    assert all(lane == other for _, lane, _, other in guarded)
    maxima = {name for name, _, _, _ in guarded}
    sums = {name for _, _, name, _ in guarded}
    assert len(maxima) == 1 and len(sums) == 1 and maxima != sums


@skipIfRefEager("generates code")
@skipUnlessBackends(["cute"])
def test_more_lanes_than_the_unroll_bound_are_rejected(_cpu_only) -> None:
    """128 packets per thread would be unrolled in place 128 ways; the
    forced unroll is bounded like the knob's."""
    bound = _bind(online_softmax_rows, n=131072)
    with pytest.raises(exc.BackendUnsupported, match="unrolls at most 64"):
        bound.to_code(_config(131072, 64, 16))


@skipIfRefEager("generates code")
@skipUnlessBackends(["cute"])
def test_integer_reduction_left_per_element_is_rejected(_cpu_only) -> None:
    """``hoist_warp_reduce`` does not V-fold integer inputs, so the reduce
    stays inside the ``range_constexpr(V)`` loop of every lane: not a form
    the fold models (the per-lane form would be kept), so a multi-packet
    config is rejected while the one-packet config compiles."""
    bound = _bind(running_int_sums, dtype=torch.int32)
    with pytest.raises(
        exc.BackendUnsupported, match="not a plain assignment of the lane body"
    ):
        bound.to_code(_config(2048, 128, 4))
    code = bound.to_code(_config(512, 128, 4))
    assert "_fold_parts_" not in code
    assert "for lane_1 in range(1):" in code


@skipIfRefEager("generates code")
@skipUnlessBackends(["cute"])
def test_folded_cluster_sites_keep_two_exchanges(_cpu_only) -> None:
    """A single-trip full-row online softmax split across a cluster whose
    per-lane consumers the collapse cannot move: the fold runs each cluster
    exchange once per kernel (the placement validator's requirement) and the
    cluster online-pair rewrite leaves the two exchanges alone."""
    bound = _bind(online_softmax_rows)
    code = bound.to_code(
        helion.Config(
            block_sizes=[1, 16384],
            num_threads=[0, 256],
            cute_vector_widths=[1, 16],
            cute_lane_layouts=["blocked", "strided"],
            cute_cluster_n=2,
        )
    )
    loop = _device_loop_body(code)
    assert "_pair_root" in loop
    assert loop.count("_cute_grouped_reduce_cluster(") == 2
    assert "_cute_grouped_reduce_cluster_online_pair(" not in code
    assert "cutlass.range_constexpr(2)" in loop
    assert "for lane_1 in range(" not in loop


# --------------------------------------------------------------------------
# The pass on hand-written bodies


def _lane_body(extra: str = "", consumer: str = "m = v_2") -> list[ast.stmt]:
    source = f"""
for tile_offset_1 in range(0, 16384, 4096):
    _lane_unroll_loads_0 = []
    for lane_1 in cutlass.range_constexpr(2):
        lane_base_1 = tile_offset_1 + (cutlass.Int32(lane_1) * 128 + tid) * 16
        vec = cute.arch.load(ptr + lane_base_1, ir.VectorType.get([16], cutlass.Uint16.mlir_type))
        _lane_unroll_loads_0.append(vec)
    for lane_1 in cutlass.range_constexpr(2):
        lane_base_1 = tile_offset_1 + (cutlass.Int32(lane_1) * 128 + tid) * 16
        vec = _lane_unroll_loads_0[lane_1]
        strided_lane = cutlass.Int32(cute.arch.thread_idx()[0])
        acc = cutlass.Float32(float('-inf'))
        for vec_lane_1 in cutlass.range_constexpr(16):
            v_0 = cutlass.Float32(cutlass.Uint16(vec[vec_lane_1]).bitcast(cutlass.BFloat16))
            acc = cute.arch.fmax(acc, v_0, nan=True)
        red = _cute_grouped_reduce_shared_two_stage(acc, 'max', cutlass.Float32(float('-inf')), strided_lane, strided_lane % 128, 0, pre=1, group_span=128, group_count=1)
        amax = cutlass.Float32(red)
        v_2 = cute.math.max(cutlass.Float32(m), cutlass.Float32(amax), propagate_nan=True)
        {extra}
        {consumer}
"""
    return ast.parse(source).body


def _sites(body: list[ast.stmt], lane: str = "lane_1") -> dict[str, set[str]]:
    return {
        lane: {
            fold_site_key(node)
            for stmt in body
            for node in ast.walk(stmt)
            if isinstance(node, ast.Call)
            and ast.unparse(node.func)
            in ("_cute_grouped_reduce_shared_two_stage", "_cute_grouped_reduce_warp")
        }
    }


def _new_var_factory():
    counter = [0]

    def new_var(name: str) -> str:
        counter[0] += 1
        return f"{name}{counter[0]}"

    return new_var


def test_pass_rewrites_the_compute_loop_only() -> None:
    body = _lane_body()
    out = fold_lane_reductions(body, _sites(body), _new_var_factory())
    text = ast.unparse(ast.Module(body=out, type_ignores=[]))
    device_loop = out[0]
    assert isinstance(device_loop, ast.For)
    top = [ast.unparse(stmt).split("\n")[0] for stmt in device_loop.body]
    # load phase untouched, then: invariant setup, parts list, fold loop,
    # tree, reduce, consumers.
    assert top[0] == "_lane_unroll_loads_0 = []"
    assert top[1].startswith("for lane_1 in cutlass.range_constexpr(2):")
    assert top[2] == "strided_lane = cutlass.Int32(cute.arch.thread_idx()[0])"
    assert top[3] == "_fold_parts_1 = []"
    assert top[4].startswith("for lane_1 in cutlass.range_constexpr(2):")
    assert (
        "_fold_tree_2 = cute.arch.fmax(_fold_parts_1[0], _fold_parts_1[1], nan=True)"
        in top
    )
    assert "acc = _fold_tree_2" in top
    assert text.count("_cute_grouped_reduce_shared_two_stage(") == 1
    assert top[-1] == "m = v_2"
    fold_loop = device_loop.body[4]
    assert isinstance(fold_loop, ast.For)
    inner = [ast.unparse(stmt).split("\n")[0] for stmt in fold_loop.body]
    # The per-lane slice: packet fetch, per-lane init, the V fold, the append;
    # the unused lane base is not re-derived.
    assert inner == [
        "vec = _lane_unroll_loads_0[lane_1]",
        "acc = cutlass.Float32(float('-inf'))",
        "for vec_lane_1 in cutlass.range_constexpr(16):",
        "_fold_parts_1.append(acc)",
    ]


def test_pass_keeps_lane_base_when_a_mask_reads_it() -> None:
    body = _lane_body()
    device_loop = body[0]
    assert isinstance(device_loop, ast.For)
    compute = device_loop.body[2]
    assert isinstance(compute, ast.For)
    vloop = next(stmt for stmt in compute.body if isinstance(stmt, ast.For))
    vloop.body[-1] = ast.parse(
        "acc = cute.arch.fmax(acc, v_0 if lane_base_1 < 16384 else cutlass.Float32(float('-inf')), nan=True)"
    ).body[0]
    out = fold_lane_reductions(body, _sites(body), _new_var_factory())
    fold_loop = next(
        stmt
        for stmt in out[0].body  # type: ignore[attr-defined]
        if isinstance(stmt, ast.For) and "_fold_parts_" in ast.unparse(stmt)
    )
    assert ast.unparse(fold_loop.body[0]).startswith("lane_base_1 = ")


def test_pass_leaves_single_lane_and_unrelated_loops() -> None:
    body = _lane_body()
    rolled = ast.parse(
        ast.unparse(ast.Module(body=body, type_ignores=[])).replace(
            "cutlass.range_constexpr(2)", "range(1)"
        )
    ).body
    out = fold_lane_reductions(rolled, _sites(rolled), _new_var_factory())
    assert ast.dump(ast.Module(body=out, type_ignores=[])) == ast.dump(
        ast.Module(body=rolled, type_ignores=[])
    )
    other = fold_lane_reductions(body, _sites(body, "lane_9"), _new_var_factory())
    assert ast.dump(ast.Module(body=other, type_ignores=[])) == ast.dump(
        ast.Module(body=body, type_ignores=[])
    )


@pytest.mark.parametrize(
    "extra, consumer, reason",
    [
        ("", "out[lane_base_1] = v_2", "depends on lane_1"),
        ("tail = acc + 1.0", "m = tail", "read outside the fold"),
        ("(ptr + lane_base_1).store(amax)", "m = v_2", "depends on lane_1"),
        ("(ptr + 1).store(amax)", "m = v_2", "has effects"),
        ("done = (ptr + 1).store(amax)", "m = v_2", "stores inside lane_1"),
        ("cute.arch.atomic_add(ptr, amax)", "m = v_2", "has effects"),
    ],
)
def test_pass_rejects_bodies_it_cannot_fold(
    extra: str, consumer: str, reason: str
) -> None:
    body = _lane_body(extra, consumer)
    with pytest.raises(exc.BackendUnsupported, match=reason):
        fold_lane_reductions(body, _sites(body), _new_var_factory())


def test_pass_rejects_rolled_multi_lane_loops_and_nested_sites() -> None:
    body = _lane_body()
    rolled = ast.parse(
        ast.unparse(ast.Module(body=body, type_ignores=[])).replace(
            "for lane_1 in cutlass.range_constexpr(2):\n        lane_base_1 = tile_offset_1 + (cutlass.Int32(lane_1) * 128 + tid) * 16\n        vec = _lane_unroll_loads_0[lane_1]",
            "for lane_1 in range(2):\n        lane_base_1 = tile_offset_1 + (cutlass.Int32(lane_1) * 128 + tid) * 16\n        vec = _lane_unroll_loads_0[lane_1]",
        )
    ).body
    with pytest.raises(exc.BackendUnsupported, match="not unrolled"):
        fold_lane_reductions(rolled, _sites(rolled), _new_var_factory())
    nested = _lane_body(
        "if m > 0.0:\n            red2 = _cute_grouped_reduce_shared_two_stage(acc, 'sum', cutlass.Float32(0), strided_lane, strided_lane % 128, 0, pre=1, group_span=128, group_count=1)"
    )
    with pytest.raises(exc.BackendUnsupported, match="plain assignment"):
        fold_lane_reductions(nested, _sites(nested), _new_var_factory())


def test_pass_rejects_an_accumulator_carried_across_lanes() -> None:
    """``acc`` initialized before the lane loop and updated per lane already
    accumulates across the lanes (``hoist_warp_reduce``'s running-sum form);
    folding its per-lane snapshots would count lane 0 twice."""
    source = ast.unparse(ast.Module(body=_lane_body(), type_ignores=[]))
    carried = source.replace(
        "        acc = cutlass.Float32(float('-inf'))\n", ""
    ).replace(
        "    for lane_1 in cutlass.range_constexpr(2):\n        lane_base_1 = tile_offset_1 + (cutlass.Int32(lane_1) * 128 + tid) * 16\n        vec = _lane_unroll_loads_0[lane_1]",
        "    acc = cutlass.Float32(float('-inf'))\n    for lane_1 in cutlass.range_constexpr(2):\n        lane_base_1 = tile_offset_1 + (cutlass.Int32(lane_1) * 128 + tid) * 16\n        vec = _lane_unroll_loads_0[lane_1]",
    )
    assert carried != source
    body = ast.parse(carried).body
    with pytest.raises(exc.BackendUnsupported, match="not initialized to the identity"):
        fold_lane_reductions(body, _sites(body), _new_var_factory())


def test_pass_rejects_an_init_other_than_the_identity() -> None:
    source = ast.unparse(ast.Module(body=_lane_body(), type_ignores=[]))
    body = ast.parse(
        source.replace(
            "acc = cutlass.Float32(float('-inf'))", "acc = cutlass.Float32(0)"
        )
    ).body
    with pytest.raises(exc.BackendUnsupported, match="not the identity of 'max'"):
        fold_lane_reductions(body, _sites(body), _new_var_factory())


def test_pass_recomputes_a_lane_value_read_outside_the_device_loop() -> None:
    """The dead lane base is dropped only while nothing reads it; a read
    after the enclosing device loop sees the last lane's value, so the base
    is recomputed per lane in a trailing lane loop."""
    body = _lane_body()
    body.append(ast.parse("tail = lane_base_1 + 1").body[0])
    out = fold_lane_reductions(body, _sites(body), _new_var_factory())
    top = _top_lines(out)
    assert top[-1].startswith("for lane_1 in cutlass.range_constexpr(2):")
    trailing = out[0].body[-1]  # type: ignore[attr-defined]
    assert [ast.unparse(stmt) for stmt in trailing.body] == [
        "lane_base_1 = tile_offset_1 + (cutlass.Int32(lane_1) * 128 + tid) * 16"
    ]
    assert ast.unparse(out[1]) == "tail = lane_base_1 + 1"


def test_pass_rejects_a_recomputed_lane_value_whose_input_is_redefined() -> None:
    """``keep`` reads ``m`` before the consumer redefines it; the trailing
    loop would read the new ``m``."""
    body = _lane_body("keep = lane_base_1 + m")
    body.append(ast.parse("tail = keep").body[0])
    with pytest.raises(exc.BackendUnsupported, match="redefines after it"):
        fold_lane_reductions(body, _sites(body), _new_var_factory())


def test_pass_rejects_subscript_stores_outside_register_caches() -> None:
    body = _lane_body("smem_buf[lane_1] = v_2")
    with pytest.raises(exc.BackendUnsupported, match="not a pure register computation"):
        fold_lane_reductions(body, _sites(body), _new_var_factory())
    # A per-thread register cache filled by the V-loop stays pure register work.
    source = (
        ast.unparse(ast.Module(body=_lane_body(), type_ignores=[]))
        .replace(
            "            acc = cute.arch.fmax(acc, v_0, nan=True)",
            "            _helion_vmerge_cache_0[vec_lane_1] = v_0\n"
            "            acc = cute.arch.fmax(acc, v_0, nan=True)",
        )
        .replace(
            "        acc = cutlass.Float32(float('-inf'))\n",
            "        _helion_vmerge_cache_0 = cute.make_rmem_tensor(16, cutlass.Float32)\n"
            "        acc = cutlass.Float32(float('-inf'))\n",
        )
    )
    body = ast.parse(source).body
    out = fold_lane_reductions(body, _sites(body), _new_var_factory())
    fold_loop = next(
        stmt
        for stmt in out[0].body  # type: ignore[attr-defined]
        if isinstance(stmt, ast.For) and "_fold_parts_" in ast.unparse(stmt)
    )
    assert "cute.make_rmem_tensor(16, cutlass.Float32)" in ast.unparse(fold_loop)


def _pair_body(between: str, frame: str = "v_1") -> list[ast.stmt]:
    """An online-softmax step after ``unroll_lane_loads``: a max fold, the
    invariants ``between`` the sites, and a sum fold of ``exp(x - frame)``
    whose V-loop also reads ``w``."""
    source = f"""
for tile_offset_1 in range(0, 16384, 4096):
    _lane_unroll_loads_0 = []
    for lane_1 in cutlass.range_constexpr(2):
        lane_base_1 = tile_offset_1 + (cutlass.Int32(lane_1) * 128 + tid) * 16
        vec = cute.arch.load(ptr + lane_base_1, ir.VectorType.get([16], cutlass.Uint16.mlir_type))
        _lane_unroll_loads_0.append(vec)
    for lane_1 in cutlass.range_constexpr(2):
        vec = _lane_unroll_loads_0[lane_1]
        strided_lane = cutlass.Int32(cute.arch.thread_idx()[0])
        acc = cutlass.Float32(float('-inf'))
        for vec_lane_1 in cutlass.range_constexpr(16):
            v_0 = cutlass.Float32(cutlass.Uint16(vec[vec_lane_1]).bitcast(cutlass.BFloat16))
            acc = cute.arch.fmax(acc, v_0, nan=True)
        red = _cute_grouped_reduce_shared_two_stage(acc, 'max', cutlass.Float32(float('-inf')), strided_lane, strided_lane % 128, 0, pre=1, group_span=128, group_count=1)
        amax = cutlass.Float32(red)
        v_1 = cute.math.max(cutlass.Float32(running_max), cutlass.Float32(amax), propagate_nan=True)
{between}
        acc_1 = cutlass.Float32(0)
        for vec_lane_1 in cutlass.range_constexpr(16):
            v_0 = cutlass.Float32(cutlass.Uint16(vec[vec_lane_1]).bitcast(cutlass.BFloat16))
            v_5 = v_0 - {frame}
            v_7 = v_0 * w
            acc_1 = acc_1 + cutlass.Float32(cute.math.exp2(cutlass.Float32(v_5) * 1.4426950408889634))
        s = _cute_grouped_reduce_shared_two_stage(acc_1, 'sum', cutlass.Float32(0), strided_lane, strided_lane % 128, 0, pre=1, group_span=128, group_count=1)
        running_sum = running_sum * w + s
        running_max = v_1
"""
    return ast.parse(source).body


def _top_lines(out: list[ast.stmt]) -> list[str]:
    device_loop = out[0]
    assert isinstance(device_loop, ast.For)
    return [ast.unparse(stmt).split("\n")[0] for stmt in device_loop.body]


def test_pair_plan_hoists_invariants_with_their_definitions() -> None:
    """``w`` (read by the sum fold) derives from ``q``, which does not
    depend on the max: both move above the fused loop, ``q`` first."""
    body = _pair_body("        q = running_max * 2.0\n        w = q + 1.0")
    out = fold_lane_reductions(body, _sites(body), _new_var_factory())
    assert "_pair_root" in ast.unparse(ast.Module(body=out, type_ignores=[]))
    top = _top_lines(out)
    fused = next(i for i, text in enumerate(top) if text.startswith("_fold_parts_"))
    assert top.index("q = running_max * 2.0") < top.index("w = q + 1.0") < fused


@pytest.mark.parametrize(
    "between, frame",
    [
        # ``w`` depends on the max through ``q``: hoisting it above the
        # fused loop would read ``q`` before its definition.
        ("        q = v_1 * 2.0\n        w = q + 1.0", "v_1"),
        # The frame decreases in the max: a lane's frame could exceed the
        # global one and its rescale factor overflow.
        ("        w = 1.0\n        neg = 10.0 - v_1", "neg"),
    ],
)
def test_pair_plan_declines_transitive_max_dependence_and_non_monotone_frames(
    between: str, frame: str
) -> None:
    body = _pair_body(between, frame)
    out = fold_lane_reductions(body, _sites(body), _new_var_factory())
    text = ast.unparse(ast.Module(body=out, type_ignores=[]))
    assert "_pair_root" not in text
    assert text.count("_cute_grouped_reduce_shared_two_stage(") == 2
    top = _top_lines(out)
    reduce_max = next(i for i, line in enumerate(top) if "'max'" in line)
    # Every invariant between the sites keeps its place after the max reduce.
    for line in between.strip().splitlines():
        assert top.index(line.strip()) > reduce_max


def test_fold_site_key_ignores_the_input() -> None:
    before = "_cute_grouped_reduce_shared_two_stage(v_1, 'max', cutlass.Float32(float('-inf')), a, b, c, pre=1, group_span=128, group_count=1)"
    after = before.replace("v_1", "cutlass.Float32(_helion_vfold_acc_0)")
    assert fold_site_key(before) == fold_site_key(after)
    assert fold_site_key(before) != fold_site_key(before.replace("'max'", "'sum'"))
    # ``hoist_warp_reduce`` re-wraps a grouped helper's identity in the
    # accumulator dtype; the key keeps one wrapper.
    integer = "_cute_grouped_reduce_shared_two_stage(v_0, 'sum', cutlass.Int64(0), a, b, c, pre=1, group_span=128, group_count=1)"
    assert fold_site_key(integer) == fold_site_key(
        integer.replace("cutlass.Int64(0)", "cutlass.Int64(cutlass.Int64(0))")
    )
    direct = "cute.arch.warp_reduction_max(v_1, op=_cute_nan_max, threads_in_group=32)"
    assert fold_site_key(direct) == fold_site_key(direct.replace("v_1", "acc"))
    with pytest.raises(exc.BackendUnsupported, match="not a reduce call"):
        fold_site_key("v_1")


# --------------------------------------------------------------------------
# GPU


@onlyBackends(["cute"])
class TestCuteLaneFoldReductionsCuda(TestCase):
    @skipIfRefEager("per-config lane counts and generated code")
    def test_squared_tile_sums_match_with_two_packets_per_thread(self) -> None:
        x = torch.randn(8, 16384, device=DEVICE, dtype=torch.bfloat16)
        expected = (x.float().view(8, 4, 4096).sum(-1) ** 2).sum(-1)
        for threads, unroll in ((128, 1), (128, 2), (64, 1)):
            extra = {"cute_lane_unroll": unroll} if unroll > 1 else {}
            code, out = code_and_output(
                helion.kernel(squared_tile_sums, backend="cute", static_shapes=True),
                (x,),
                **_config(4096, threads, 16, **extra).config,
            )
            torch.testing.assert_close(out, expected, rtol=1e-3, atol=1e-2)
            self.assertIn("_fold_parts_", code)
            self.assertEqual(_device_loop_body(code).count(REDUCE), 1)

    @skipIfRefEager("per-config lane counts and generated code")
    def test_online_pair_special_values_match_the_single_packet_form(self) -> None:
        """NaN, -inf lanes, all--inf rows and +inf behave exactly as the
        one-packet-per-thread lowering (no fold, no pair): the lane frames
        are -inf-guarded and NaN propagates through max and exp alike."""
        torch.manual_seed(0)
        x = torch.randn(8, 16384, device=DEVICE, dtype=torch.bfloat16)
        x[1, 3] = torch.nan
        # The first lane of every thread but the first is all -inf (the lane
        # frame guard), while the reference's first 2048-wide trip keeps a
        # finite max: a trip that is entirely -inf makes the example's own
        # online update exp(-inf - -inf) = NaN in every lowering.
        x[2, 16:2048] = -torch.inf
        x[3] = -torch.inf
        x[4, 100] = torch.inf
        x[5, 2048:] = -torch.inf
        # A log-domain row far below zero with whole lanes of -inf: the
        # empty lanes' rescale factor exp(F(0) - F(m)) overflows and must
        # not reach their zero sums.
        x[6, :] = -1e4
        x[6, 2048:4096] = -torch.inf
        x[7, :] = -1e4
        kernel = helion.kernel(online_softmax_rows, backend="cute", static_shapes=True)
        _, expected = code_and_output(kernel, (x,), **_config(2048, 128, 16).config)
        torch.testing.assert_close(
            expected[6:], torch.logsumexp(x[6:].float(), dim=-1), rtol=1e-5, atol=1e-3
        )
        for block_n, threads in ((16384, 128), (4096, 128), (8192, 64), (1024, 32)):
            code, out = code_and_output(
                helion.kernel(online_softmax_rows, backend="cute", static_shapes=True),
                (x,),
                **_config(block_n, threads, 16).config,
            )
            self.assertIn("_pair_root", code)
            torch.testing.assert_close(
                out, expected, rtol=1e-4, atol=1e-4, equal_nan=True
            )

    @skipIfRefEager("per-config lane counts and generated code")
    def test_two_loops_with_identical_reduce_calls_match(self) -> None:
        """Both device loops fold on the grouped warp path (32 threads per
        row, two rows per CTA) and on the two-stage path."""
        x = torch.randn(8, 16384, device=DEVICE, dtype=torch.bfloat16)
        for block_n, rows, threads in ((1024, 2, 32), (4096, 1, 128)):
            expected = 2 * (x.float().view(8, -1, block_n).sum(-1) ** 2).sum(-1)
            code, out = code_and_output(
                helion.kernel(two_same_op_loops, backend="cute", static_shapes=True),
                (x,),
                block_sizes=[rows, block_n, block_n],
                num_threads=[rows, threads, threads],
                cute_vector_widths=[1, 16, 16],
                cute_lane_layouts=["blocked", "strided", "strided"],
            )
            torch.testing.assert_close(out, expected, rtol=1e-3, atol=1e-2)
            self.assertEqual(
                len(re.findall(r"^\s*_fold_tree_\w* = ", code, re.MULTILINE)),
                2 * (block_n // (threads * 16) - 1),
            )
            self.assertIsNone(re.search(r"for lane_\d+ in range\((?!1\))", code))

    @skipIfRefEager("per-config lane counts and generated code")
    def test_single_warp_direct_reduces_match(self) -> None:
        """A one-warp CTA with two packets per thread: the direct warp
        reduce is folded for the squared sums and the online pair."""
        x = torch.randn(8, 16384, device=DEVICE, dtype=torch.bfloat16)
        code, out = code_and_output(
            helion.kernel(squared_tile_sums, backend="cute", static_shapes=True),
            (x,),
            **_config(1024, 32, 16).config,
        )
        expected = (x.float().view(8, 16, 1024).sum(-1) ** 2).sum(-1)
        torch.testing.assert_close(out, expected, rtol=1e-3, atol=1e-2)
        self.assertIn("cute.arch.warp_reduction_sum(", code)
        self.assertIn("_fold_parts_", code)
        code, out = code_and_output(
            helion.kernel(online_softmax_rows, backend="cute", static_shapes=True),
            (x,),
            **_config(1024, 32, 16).config,
        )
        torch.testing.assert_close(
            out, torch.logsumexp(x.float(), dim=-1), rtol=1e-3, atol=1e-3
        )
        self.assertIn("cute.arch.warp_reduction_max(", code)
        self.assertIn("_pair_root", code)

    @skipIfRefEager("per-config lane counts and generated code")
    def test_two_pass_softmax_register_cache_matches(self) -> None:
        """The second device loop reads the row cache the folded first loop
        filled once per lane (one warp, two and four packets per thread)."""
        x = torch.randn(64, 2048, device=DEVICE, dtype=torch.float16)
        expected = torch.nn.functional.softmax(x.float(), dim=1).to(x.dtype)
        with patch.dict(os.environ, {"HELION_DISABLE_ONLINE_TO_3PASS": "1"}):
            for block_n in (256, 512):
                code, out = code_and_output(
                    helion.kernel(two_pass_softmax, backend="cute", static_shapes=True),
                    (x,),
                    block_sizes=[1, block_n],
                    num_threads=[0, 32],
                    cute_vector_widths=[1, 4],
                )
                torch.testing.assert_close(out, expected, rtol=1e-2, atol=1e-2)
                self.assertIn("_pair_root", code)
                self.assertRegex(code, r"_fuse_cache_\d+ = cute\.make_rmem_tensor\(")
                self.assertNotRegex(code, r"for lane_\d+ in range\(")

    @skipIfRefEager("per-config lane counts and generated code")
    def test_folded_cluster_sites_match(self) -> None:
        """The single-trip online softmax across a cluster of two CTAs with
        two packets per thread each: both cluster exchanges run once."""
        torch.manual_seed(0)
        x = torch.randn(8, 16384, device=DEVICE, dtype=torch.bfloat16)
        x[2, 16:2048] = -torch.inf
        x[6, :] = -1e4
        x[6, 2048:4096] = -torch.inf
        code, out = code_and_output(
            helion.kernel(online_softmax_rows, backend="cute", static_shapes=True),
            (x,),
            block_sizes=[1, 16384],
            num_threads=[0, 256],
            cute_vector_widths=[1, 16],
            cute_lane_layouts=["blocked", "strided"],
            cute_cluster_n=2,
        )
        self.assertEqual(code.count("_cute_grouped_reduce_cluster("), 2)
        self.assertIn("_pair_root", code)
        torch.testing.assert_close(
            out, torch.logsumexp(x.float(), dim=-1), rtol=1e-4, atol=1e-3
        )

    def test_online_softmax_rows_match(self) -> None:
        x = torch.randn(16, 16384, device=DEVICE, dtype=torch.bfloat16)
        expected = torch.logsumexp(x.float(), dim=-1)
        for block_n, threads in ((4096, 128), (8192, 128), (8192, 64), (2048, 128)):
            code, out = code_and_output(
                helion.kernel(online_softmax_rows, backend="cute", static_shapes=True),
                (x,),
                **_config(block_n, threads, 16).config,
            )
            torch.testing.assert_close(out, expected, rtol=1e-3, atol=1e-3)
            loop = _device_loop_body(code)
            self.assertEqual(loop.count(REDUCE), 2)
            self.assertEqual("_fold_parts_" in loop, block_n > 2048)
