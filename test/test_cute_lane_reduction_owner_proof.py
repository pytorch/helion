"""CPU tests: the lane-loop split's owner proof for the stores after a
reduce over a second thread axis (``owner_guards._grouped_shared_reduce``)
and for discarded pure calls (``_validated_owner_store``)."""

from __future__ import annotations

import ast
import re

import pytest
import torch

import helion
from helion._compiler.cute.lane_loops.owner_guards import _store_thread_axes
from helion._compiler.cute.lane_loops.owner_guards import _update_thread_axis_names
from helion._testing import cpu_render_target
from helion._testing.cute_codegen import cpu_bind
import helion.language as hl

pytestmark = pytest.mark.cpu_only


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _run_sums(x: torch.Tensor, out: torch.Tensor):
    for tile_x, tile_q in hl.tile([x.size(0), 4], block_size=[None, 4]):
        partial = torch.sum(x[tile_x, tile_q, :].to(torch.float32), -1)
        out[tile_x] = torch.sum(partial, -1)
    return out


@pytest.mark.parametrize("width", [300, 1000])
def test_a_store_after_a_reduce_over_a_second_thread_axis_has_one_writer(width):
    """HelionKernel's sum over the 4 runs of [4, 130, W] read as a tile: a
    lane loop reduces each run over thread_idx[0], then the one-group shared
    reduce over the runs on thread_idx[1] (pre=32, group_span=128) leaves a
    value uniform across the CTA, which the default config of runs past 256
    stores from one thread instead of failing with "lane reduction store
    aliases a value-varying thread axis"."""
    x = torch.empty(4, 130, width).permute(1, 0, 2)
    out = torch.empty(130)
    with cpu_render_target(reduction_smem_budget_bytes=232448):
        bound = cpu_bind(_run_sums, (x, out))
        code = bound.to_code(bound.config_spec.default_config())
    assert "_cute_grouped_reduce_shared_two_stage(" in code
    assert "synthetic_lane" in code
    tree = ast.parse(code)
    guards = [
        ast.unparse(node.test)
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and any(
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "store"
            and "out.iterator" in ast.unparse(call.func.value)
            for stmt in node.body
            for call in ast.walk(stmt)
        )
    ]
    assert len(guards) == 1
    # One writer: thread_idx[1] == 0 and lane 0 of the warp along axis 0.
    assert "cutlass.Int32(cute.arch.thread_idx()[1]) == 0" in guards[0]
    assert re.search(
        r"cutlass\.Int32\(cute\.arch\.thread_idx\(\)\[0\]\)[^\n]*% 32 < 1", guards[0]
    )


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _broadcast_run_sums(in0, in1, out0, out1):
    # helion.inductor's where and sum over 8 runs of [8, 48, 364] inputs
    # broadcast over the runs (stride 0): the store of the where repeats
    # along tile_q, the sum counts each of the 8 copies.
    for tile_x, tile_q in hl.tile([48, 8], block_size=[None, 8]):
        rdim = hl.zeros([tile_q, 364]).size(1)
        tmp0 = in0[tile_x, :]
        tmp1 = in1[tile_x, :]
        tmp2 = hl.full([], 0.5, dtype=torch.float64)
        tmp3 = torch.where(tmp0, tmp2, tmp1)
        out0[tile_x, tile_q, :] = tmp3.reshape([tile_x, 1, rdim])
        tmp4 = tmp3.reshape([tile_x, 1, rdim]).expand(tile_x, tile_q, rdim)
        tmp5 = torch.sum(torch.sum(tmp4, -1), -1)
        out1[tile_x] = tmp5.to(torch.float32)


def test_a_discarded_pure_call_needs_no_owner():
    """The default config of _broadcast_run_sums leaves a dead
    ``cutlass.Float32(0.0)`` beside the stores that consume the reduction:
    a call whose calls are all proven pure has no effect to guard, where it
    failed with "cannot infer whether a standalone call has side effects"."""
    args = (
        torch.empty(48, 364, dtype=torch.bool),
        torch.empty(48, 364, dtype=torch.float64),
        torch.empty(48, 8, 364, dtype=torch.float64),
        torch.empty(48),
    )
    with cpu_render_target(reduction_smem_budget_bytes=232448):
        bound = cpu_bind(_broadcast_run_sums, args)
        code = bound.to_code(bound.config_spec.default_config())
    assert "_cute_grouped_reduce_shared_two_stage(" in code
    assert "cutlass.Int32(cute.arch.thread_idx()[1]) == 0" in code
    stmt = ast.parse("if c:\n    cutlass.Float32(0.0)\n    unknown_effect(x)\n").body[0]
    with pytest.raises(helion.exc.BackendUnsupported, match="standalone call"):
        _store_thread_axes(stmt, {}, {})


_LANE = {
    32: "cutlass.Int32(cute.arch.thread_idx()[0])"
    " + cutlass.Int32(cute.arch.thread_idx()[1]) * cutlass.Int32(32)",
    48: "cutlass.Int32(cute.arch.thread_idx()[0])"
    " + cutlass.Int32(cute.arch.thread_idx()[1]) * cutlass.Int32(48)",
    -32: "cutlass.Int32(cute.arch.thread_idx()[0])"
    " - cutlass.Int32(cute.arch.thread_idx()[1]) * cutlass.Int32(32)",
}


@pytest.mark.parametrize(
    "stride,group_count,value_axes,expected",
    [
        # reduced over thread_idx[1]: uniform
        (32, 1, {1}, set()),
        # thread_idx[0] is the sibling coordinate below pre: kept
        (32, 1, {0, 1}, {0}),
        # several groups: the group index varies, every axis read is kept
        (32, 2, {1}, {0, 1}),
        # a stride that is neither a multiple nor a divisor of pre mixes the
        # residues of both axes: the conservative provenance, every axis read
        (48, 1, {1}, {0, 1}),
        # a negative stride: the conservative provenance too
        (-32, 1, {1}, {0, 1}),
        # the lane named through an alias of an alias: followed, uniform
        ("alias", 1, {1}, set()),
    ],
)
def test_a_one_group_shared_reduce_drops_the_thread_axes_it_reduces(
    stride, group_count, value_axes, expected
):
    names = {
        "v": frozenset(value_axes),
        "lane": frozenset({0, 1}),
        "lane_in_group": frozenset({0, 1}),
        "lane_mod_pre": frozenset({0, 1}),
    }
    if stride == "alias":
        definitions = {
            "lane": ast.Name(id="lane2", ctx=ast.Load()),
            "lane2": ast.parse(_LANE[32], mode="eval").body,
        }
    else:
        definitions = {"lane": ast.parse(_LANE[stride], mode="eval").body}
    stmt = ast.parse(
        "r = _cute_grouped_reduce_shared_two_stage(v, 'sum', cutlass.Float32(0), lane,"
        f" lane_in_group, lane_mod_pre, pre=32, group_span=128,"
        f" group_count={group_count})"
    ).body[0]
    _update_thread_axis_names(stmt, names, definitions)
    assert names["r"] == frozenset(expected)
