"""CPU tests: ``cute_defer_tile_reduce`` keeps a carried tile sum's per-thread
accumulators through the device loop and combines the threads once after it;
the pass declines every other use of the accumulators, the results or the
carries, and a tile with one thread, which has nothing to combine; the
autotuner offers the knob where the pass can move every tile reduction of
such a loop, in a kernel without a matmul, with witnesses on seeds that walk
several tiles with several threads of several rows each."""

from __future__ import annotations

import ast
import textwrap
from typing import TYPE_CHECKING

import pytest
import torch

import helion
from helion import exc
from helion._compiler.ast_read_writes import HELION_LANE_LOOP_VAR_ATTR
from helion._compiler.cute.autotune.heuristics import walks_a_deferrable_tile_sum
from helion._compiler.cute.lane_loops.defer_tile_reductions import defer_tile_reductions
from helion._compiler.cute.lane_loops.markers import HELION_FINALIZED
from helion._compiler.cute.lane_loops.markers import HELION_LANE_SPLIT_ATTR
from helion._compiler.cute.lane_loops.markers import LaneSplitRecord
from helion._compiler.cute.lane_loops.markers import _LaneReduceMarker
from helion._testing import cpu_render_target
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_vloop_sink_kernels import _sink_config
from helion._testing.cute_vloop_sink_kernels import col_reduce_max_dynamic
from helion._testing.cute_vloop_sink_kernels import col_reduce_sum_dynamic
from helion._testing.cute_vloop_sink_kernels import col_reduce_sum_static
from helion._testing.cute_vloop_sink_kernels import col_weighted_mean_static
from helion._testing.split_reduction_kernels import cross_thread_combines
from helion._testing.split_reduction_kernels import split_column_sums
import helion.language as hl

if TYPE_CHECKING:
    from typing import Any

    from helion.autotuner.config_spec import ConfigSpec
    from helion.runtime.kernel import Kernel

pytestmark = pytest.mark.cpu_only

_KNOB = "cute_defer_tile_reduce"
# 128-row tiles on one warp (16 row threads beside 2 column threads), eight
# bf16 columns per thread: the layout of the split column reduction's seed.
_ONE_WARP = _sink_config(
    block_sizes=[128, 16], num_threads=[16, 2], vec=[1, 8], unroll=8
)
# 512-row tiles on sixteen warps (128 row threads beside 4 column threads):
# the row threads combine through shared memory.
_SIXTEEN_WARPS = _sink_config(
    block_sizes=[512, 16], num_threads=[128, 4], vec=[1, 4], unroll=4
)


def _code(kernel: Kernel[Any], args: tuple[object, ...], **config: object) -> str:
    with cpu_render_target(reduction_smem_budget_bytes=232448):
        bound = cpu_bind(kernel, args)
        return bound.to_code(
            bound.config_spec.normalized_config(helion.Config.from_dict(dict(config)))
        )


def test_a_sunk_column_sum_combines_once_after_the_row_loop() -> None:
    x = torch.empty(4096, 64, dtype=torch.bfloat16)
    off = _code(col_reduce_sum_static, (x,), **_ONE_WARP)
    on = _code(col_reduce_sum_static, (x,), **_ONE_WARP, **{_KNOB: True})
    assert cross_thread_combines(off) == (1, 1)
    assert cross_thread_combines(on) == (0, 1)
    before, loop = on.split("for tile_offset_0 in range(", 1)
    loop, after = loop.split("_cute_grouped_reduce", 1)
    # The per-thread accumulators are zeroed once, ahead of the row loop,
    # which only adds into them; the combine and the carried add follow it.
    reset = "sum_1_lane_acc_frag[vec_lane_1] = cutlass.Float32(0)"
    assert reset in before and reset not in loop
    assert "sum_1_lane_acc_frag[vec_lane_1] = sum_1_lane_acc_frag[vec_lane_1] +" in loop
    assert "col_acc_frag[" not in loop
    assert "col_acc_frag[vec_lane_1] = col_acc_copy + sum_1" in after
    # An explicit False is the code without the knob.
    assert _code(col_reduce_sum_static, (x,), **_ONE_WARP, **{_KNOB: False}) == off


@pytest.mark.parametrize(
    ("kernel", "config", "combines"),
    [
        (col_reduce_sum_static, _SIXTEEN_WARPS, 1),
        (col_reduce_sum_static, {**_SIXTEEN_WARPS, "cute_vloop_sink": False}, 1),
        (
            col_reduce_sum_static,
            _sink_config(
                block_sizes=[256, 16], num_threads=[32, 16], vec=[1, 1], sink=False
            ),
            1,
        ),
        (col_reduce_sum_dynamic, _SIXTEEN_WARPS, 1),
        # A second sum of the loop (the weights' own) moves with the first.
        (col_weighted_mean_static, _SIXTEEN_WARPS, 2),
    ],
    ids=["shared_memory", "unsunk", "scalar_lanes", "dynamic_rows", "two_sums"],
)
def test_the_combine_leaves_the_loop_in_every_lane_layout(
    kernel: Kernel[Any], config: dict[str, object], combines: int
) -> None:
    x = torch.empty(4096, 64, dtype=torch.bfloat16)
    args = (x, torch.empty(4096)) if kernel is col_weighted_mean_static else (x,)
    assert cross_thread_combines(_code(kernel, args, **config)) == (combines, combines)
    assert cross_thread_combines(_code(kernel, args, **config, **{_KNOB: True})) == (
        0,
        combines,
    )


@pytest.mark.parametrize(
    "config",
    [
        _sink_config(block_sizes=[32, 16], num_threads=[1, 16], vec=[1, 1], sink=False),
        _sink_config(block_sizes=[32, 16], num_threads=[1, 2], vec=[1, 8], unroll=8),
    ],
    ids=["scalar_lanes", "sunk"],
)
def test_one_thread_on_the_tile_keeps_its_sums_per_tile(
    config: dict[str, object],
) -> None:
    # The one row thread folds its 32 rows of a tile in a lane loop, and the
    # split finalizes the sum as that thread's accumulator: no combine to
    # defer, so the knob must not reorder the sum either.
    x = torch.empty(4096, 64, dtype=torch.bfloat16)
    off = _code(col_reduce_sum_static, (x,), **config)
    assert cross_thread_combines(off) == (0, 0)
    assert "sum_1_lane_acc" in off
    assert _code(col_reduce_sum_static, (x,), **config, **{_KNOB: True}) == off


# The lane split's output for ``dot = dot + sum(v)`` over a row loop, as the
# pass receives it (``v_16`` is the loop's spelling of the carried ``dot``).
_NEST = """
dot = cutlass.Float32(0.0)
for tile_offset_2 in range(cutlass.Int32(0), cutlass.Int32(2384), cutlass.Int32(128)):
    sum_1_lane_acc = cutlass.Float32(0)
    for lane_2 in range(8):
        indices_2 = tile_offset_2 + cutlass.Int32(cute.arch.thread_idx()[1]) + lane_2 * 16
        v = cutlass.Float32(x[indices_2])
        sum_1_lane_acc = sum_1_lane_acc + cutlass.Float32(v)
    sum_1 = cutlass.Float32(cute.arch.warp_reduction_sum(sum_1_lane_acc, threads_in_group=32))
    scale = cutlass.Float32(2.0)
    dot_copy = dot
    dot_copy_0 = dot_copy
    v_16 = dot_copy_0 + sum_1
out[0] = dot
"""
_RENAMES = {"dot": "dot", "v_16": "dot"}


def _nest(
    text: str = _NEST, *, reduction_type: str = "sum", wrap: str = "cutlass.Float32"
) -> list[ast.AST]:
    """``text`` parsed, its lane loop marked and carrying the split's record."""
    body: list[ast.AST] = list(ast.parse(textwrap.dedent(text)).body)
    (loop,) = [stmt for stmt in body if isinstance(stmt, ast.For)]
    (lane_loop,) = [
        stmt
        for stmt in loop.body
        if isinstance(stmt, ast.For) and ast.unparse(stmt.target) == "lane_2"
    ]
    setattr(lane_loop, HELION_LANE_LOOP_VAR_ATTR, "lane_2")
    at = loop.body.index(lane_loop)
    setattr(
        lane_loop,
        HELION_LANE_SPLIT_ATTR,
        LaneSplitRecord(
            markers=(
                _LaneReduceMarker(
                    result_var="sum_1",
                    input_name="v",
                    reduction_type=reduction_type,
                    identity_expr="cutlass.Float32(0)",
                    threads_in_group=32,
                    wrap_template=f"{wrap}({HELION_FINALIZED})",
                ),
            ),
            accumulators=("sum_1_lane_acc",),
            resets=(loop.body[at - 1],),
            finalize=(loop.body[at + 1],),
        ),
    )
    return body


def _deferred(body: list[ast.AST]) -> str:
    return "\n".join(
        ast.unparse(stmt) for stmt in defer_tile_reductions(body, _RENAMES)
    )


def test_the_pass_moves_the_resets_the_combine_and_the_carried_add() -> None:
    assert (
        _deferred(_nest())
        == textwrap.dedent(
            """
        dot = cutlass.Float32(0.0)
        sum_1_lane_acc = cutlass.Float32(0)
        for tile_offset_2 in range(cutlass.Int32(0), cutlass.Int32(2384), cutlass.Int32(128)):
            for lane_2 in range(8):
                indices_2 = tile_offset_2 + cutlass.Int32(cute.arch.thread_idx()[1]) + lane_2 * 16
                v = cutlass.Float32(x[indices_2])
                sum_1_lane_acc = sum_1_lane_acc + cutlass.Float32(v)
            scale = cutlass.Float32(2.0)
        sum_1 = cutlass.Float32(cute.arch.warp_reduction_sum(sum_1_lane_acc, threads_in_group=32))
        dot_copy = dot
        dot_copy_0 = dot_copy
        v_16 = dot_copy_0 + sum_1
        out[0] = dot
        """
        ).strip()
    )


# One mutation of the nest per rule of the pass: each alone must keep the
# combine in the loop (the last one the finalizer of a tile with one thread,
# which combines nothing).
@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("v_16 = dot_copy_0 + sum_1", "v_16 = dot_copy_0 + sum_1 * scale"),
        ("v_16 = dot_copy_0 + sum_1", "v_16 = dot_copy_0 + sum_1\n    last = sum_1"),
        ("out[0] = dot", "out[0] = dot + sum_1"),
        ("out[0] = dot", "out[0] = dot + dot_copy"),
        ("out[0] = dot", "out[0] = dot + sum_1_lane_acc"),
        ("scale = cutlass.Float32(2.0)", "scale = sum_1_lane_acc"),
        (
            "v = cutlass.Float32(x[indices_2])",
            "v = cutlass.Float32(x[indices_2]) + dot",
        ),
        (
            "sum_1_lane_acc = sum_1_lane_acc + cutlass.Float32(v)",
            (
                "sum_1_lane_acc = sum_1_lane_acc + cutlass.Float32(v)\n"
                "        sum_1_lane_acc = cutlass.Float32(v)"
            ),
        ),
        (
            "sum_1_lane_acc = sum_1_lane_acc + cutlass.Float32(v)",
            "sum_1_lane_acc = cutlass.Float32(v)",
        ),
        (
            "sum_1_lane_acc = sum_1_lane_acc + cutlass.Float32(v)",
            "sum_1_lane_acc = sum_1_lane_acc * cutlass.Float32(v)",
        ),
        (
            "sum_1_lane_acc = sum_1_lane_acc + cutlass.Float32(v)",
            "sum_1_lane_acc = sum_1_lane_acc + sum_1_lane_acc * cutlass.Float32(v)",
        ),
        ("threads_in_group=32", "threads_in_group=32 + tile_offset_2"),
        ("dot_copy = dot", "dot_copy = other"),
        (
            "dot_copy = dot\n    dot_copy_0 = dot_copy",
            "dot_copy_0 = dot_copy\n    dot_copy = dot",
        ),
        ("v_16 = dot_copy_0 + sum_1", "v_16 = cutlass.Float64(dot_copy_0) + sum_1"),
        ("scale = cutlass.Float32(2.0)", "if v > 0:\n        break"),
        ("out[0] = dot", "else:\n    pass\nout[0] = dot"),
        (
            "cute.arch.warp_reduction_sum(sum_1_lane_acc, threads_in_group=32)",
            "sum_1_lane_acc",
        ),
    ],
    ids=[
        "scaled_result",
        "result_read_twice",
        "result_read_after_the_loop",
        "copy_read_after_the_loop",
        "accumulator_read_after_the_loop",
        "accumulator_read_in_the_loop",
        "carry_read_in_the_lane_loop",
        "accumulator_overwritten",
        "accumulator_assigned_not_added_to",
        "accumulator_scaled",
        "addend_reads_the_accumulator",
        "combine_reads_the_loop_index",
        "copy_of_another_value",
        "copy_made_after_its_reader",
        "carry_in_another_dtype",
        "break",
        "else",
        "one_thread_on_the_tile",
    ],
)
def test_the_pass_declines_any_other_use(old: str, new: str) -> None:
    assert old in _NEST
    text = _NEST.replace(old, new)
    assert _deferred(_nest(text)) == ast.unparse(ast.parse(textwrap.dedent(text)))


@pytest.mark.parametrize(
    "marker",
    [{"reduction_type": "max"}, {"wrap": "cutlass.Int32"}],
    ids=["max", "result_in_another_dtype"],
)
def test_the_pass_declines_a_marker_that_is_no_plain_sum(
    marker: dict[str, str],
) -> None:
    assert _deferred(_nest(**marker)) == ast.unparse(ast.parse(textwrap.dedent(_NEST)))


def _spec(kernel: Kernel[Any], *args: object) -> ConfigSpec:
    with cpu_render_target(reduction_smem_budget_bytes=232448):
        return cpu_bind(kernel, args).config_spec


def test_the_knob_is_searched_where_a_loop_adds_its_tile_sum_to_a_carry() -> None:
    # Structure alone opens it: the same search space whether the row tiles
    # divide the rows, cover them, or neither.
    specs = [
        _spec(col_reduce_sum_static, torch.empty(rows, 64, dtype=torch.bfloat16))
        for rows in (4096, 1000, 64)
    ]
    assert all(_KNOB in spec._flat_fields() for spec in specs)
    assert len({spec.structural_fingerprint() for spec in specs}) == 1
    # A carried max is not offered it and rejects an explicit True; an
    # explicit False, as a config tuned where the knob applies carries it, is
    # dropped.
    spec = _spec(col_reduce_max_dynamic, torch.empty(4096, 64, dtype=torch.bfloat16))
    assert _KNOB not in spec._flat_fields()
    with pytest.raises(exc.InvalidConfig, match=f"{_KNOB} requires"):
        spec.normalized_config(helion.Config.from_dict({_KNOB: True}))
    off = spec.normalized_config(
        helion.Config.from_dict({**spec.default_config().config, _KNOB: False})
    )
    assert off == spec.default_config()


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _carried_max(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty(n, dtype=torch.float32, device=x.device)
    for tile_n in hl.tile(n):
        acc = hl.zeros([tile_n], dtype=torch.float32)
        for tile_m in hl.tile(m):
            acc = acc + torch.amax(x[tile_m, tile_n].to(torch.float32), dim=0)
        out[tile_n] = acc
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _last_tile_sum(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty(n, dtype=torch.float32, device=x.device)
    for tile_n in hl.tile(n):
        acc = hl.zeros([tile_n], dtype=torch.float32)
        base = hl.full([tile_n], 1.0, dtype=torch.float32)
        for tile_m in hl.tile(m):
            acc = base + torch.sum(x[tile_m, tile_n].to(torch.float32), dim=0)
        out[tile_n] = acc
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _sum_over_another_dim(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty(n, dtype=torch.float32, device=x.device)
    for tile_n in hl.tile(n):
        acc = hl.zeros([tile_n], dtype=torch.float32)
        for _tile_m in hl.tile(m):
            acc = acc + torch.sum(w[tile_n, :], dim=1)
        out[tile_n] = acc
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _carried_product(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty(n, dtype=torch.float32, device=x.device)
    for tile_n in hl.tile(n):
        acc = hl.full([tile_n], 1.0, dtype=torch.float32)
        for tile_m in hl.tile(m):
            acc = acc * torch.sum(x[tile_m, tile_n].to(torch.float32), dim=0)
        out[tile_n] = acc
    return out


# One kernel per rule of the structural match: each has a reduction in its
# row loop that the knob could not move.
@pytest.mark.parametrize(
    "kernel",
    [_carried_max, _last_tile_sum, _sum_over_another_dim, _carried_product],
    ids=["carried_max", "last_tile_sum", "sum_over_another_dim", "carried_product"],
)
def test_a_loop_without_a_carried_sum_of_its_tile_does_not_offer_the_knob(
    kernel: Kernel[Any],
) -> None:
    spec = _spec(
        kernel, torch.empty(4096, 64, dtype=torch.bfloat16), torch.empty(64, 32)
    )
    assert _KNOB not in spec._flat_fields()


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _carried_sum_beside_a_matmul(
    x: torch.Tensor, w: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    m, n = x.size()
    k = w.size(1)
    out = torch.empty(n, dtype=torch.float32, device=x.device)
    prod = torch.empty([m, k], dtype=torch.float32, device=x.device)
    for tile_n in hl.tile(n):
        acc = hl.zeros([tile_n], dtype=torch.float32)
        for tile_m in hl.tile(m):
            acc = acc + torch.sum(x[tile_m, tile_n], dim=0)
        out[tile_n] = acc
    for tile_i, tile_k in hl.tile([m, k]):
        dot = hl.zeros([tile_i, tile_k], dtype=torch.float32)
        for tile_j in hl.tile(n):
            dot = torch.addmm(dot, x[tile_i, tile_j], w[tile_j, tile_k])
        prod[tile_i, tile_k] = dot
    return out, prod


def test_a_kernel_with_a_matmul_does_not_offer_the_knob() -> None:
    # The first loop alone is the carried fp32 sum the knob is offered for
    # (``_carried_sum_in_its_own_dtype``); the matmul beside it withholds it.
    spec = _spec(
        _carried_sum_beside_a_matmul, torch.empty(4096, 64), torch.empty(64, 32)
    )
    assert spec.matmul_facts
    assert not spec.cute_carried_tile_sum_blocks
    assert _KNOB not in spec._flat_fields()
    with pytest.raises(exc.InvalidConfig, match="in a kernel without a matmul"):
        spec.normalized_config(helion.Config.from_dict({_KNOB: True}))


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _tile_sum_read_twice(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    m, n = x.size()
    out = torch.empty(n, dtype=torch.float32, device=x.device)
    squares = torch.empty(n, dtype=torch.float32, device=x.device)
    for tile_n in hl.tile(n):
        acc = hl.zeros([tile_n], dtype=torch.float32)
        sq = hl.zeros([tile_n], dtype=torch.float32)
        for tile_m in hl.tile(m):
            part = torch.sum(x[tile_m, tile_n].to(torch.float32), dim=0)
            acc = acc + part
            sq = sq + part * part
        out[tile_n] = acc
        squares[tile_n] = sq
    return out, squares


def test_a_loop_whose_only_tile_sum_has_another_use_does_not_offer_the_knob() -> None:
    spec = _spec(_tile_sum_read_twice, torch.empty(4096, 64, dtype=torch.bfloat16))
    assert _KNOB not in spec._flat_fields()


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _carried_sum_in_its_own_dtype(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty(n, dtype=x.dtype, device=x.device)
    # The rows' block first, as in the column reductions ``_sink_config`` fits.
    block_m = hl.register_block_size(m)
    block_n = hl.register_block_size(n)
    for tile_n in hl.tile(n, block_size=block_n):
        acc = hl.zeros([tile_n], dtype=x.dtype)
        for tile_m in hl.tile(m, block_size=block_m):
            acc = acc + torch.sum(x[tile_m, tile_n], dim=0)
        out[tile_n] = acc
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _half_sum_into_a_float_carry(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty(n, dtype=torch.float32, device=x.device)
    for tile_n in hl.tile(n):
        acc = hl.zeros([tile_n], dtype=torch.float32)
        for tile_m in hl.tile(m):
            acc = acc + torch.sum(x[tile_m, tile_n], dim=0)
        out[tile_n] = acc
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _float_sum_into_a_double_carry(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty(n, dtype=torch.float64, device=x.device)
    for tile_n in hl.tile(n):
        acc = hl.zeros([tile_n], dtype=torch.float64)
        for tile_m in hl.tile(m):
            acc = acc + torch.sum(x[tile_m, tile_n].to(torch.float32), dim=0)
        out[tile_n] = acc
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _carry_read_before_its_add(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    m, n = x.size()
    out = torch.empty(n, dtype=torch.float32, device=x.device)
    prefix = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tile_n in hl.tile(n):
        acc = hl.zeros([tile_n], dtype=torch.float32)
        for tile_m in hl.tile(m):
            prefix[tile_m, tile_n] = x[tile_m, tile_n].to(torch.float32) + acc[None, :]
            acc = acc + torch.sum(x[tile_m, tile_n].to(torch.float32), dim=0)
        out[tile_n] = acc
    return out, prefix


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _carry_read_after_its_add(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    m, n = x.size()
    out = torch.empty(n, dtype=torch.float32, device=x.device)
    prefix = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tile_n in hl.tile(n):
        acc = hl.zeros([tile_n], dtype=torch.float32)
        for tile_m in hl.tile(m):
            acc = acc + torch.sum(x[tile_m, tile_n].to(torch.float32), dim=0)
            prefix[tile_m, tile_n] = x[tile_m, tile_n].to(torch.float32) + acc[None, :]
        out[tile_n] = acc
    return out, prefix


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _sum_and_squared_tile_sums(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    m, n = x.size()
    out = torch.empty(n, dtype=torch.float32, device=x.device)
    squares = torch.empty(n, dtype=torch.float32, device=x.device)
    for tile_n in hl.tile(n):
        acc = hl.zeros([tile_n], dtype=torch.float32)
        sq = hl.zeros([tile_n], dtype=torch.float32)
        for tile_m in hl.tile(m):
            acc = acc + torch.sum(x[tile_m, tile_n].to(torch.float32), dim=0)
            part = torch.sum(x[tile_m, tile_n].to(torch.float32) * 2.0, dim=0)
            sq = sq + part * part
        out[tile_n] = acc
        squares[tile_n] = sq
    return out, squares


# One kernel per rule that keeps a carried tile sum per tile: each adds a
# sum over its row tile to a carry, in a form the pass declines at every
# config (a cast between the sum and the carry, another read of the carry,
# a second tile sum that has to stay), so the search is not offered a knob
# that could not change the code.
@pytest.mark.parametrize(
    ("kernel", "dtype"),
    [
        (_carried_sum_in_its_own_dtype, torch.int64),
        (_carried_sum_in_its_own_dtype, torch.bfloat16),
        (_half_sum_into_a_float_carry, torch.bfloat16),
        (_float_sum_into_a_double_carry, torch.float32),
        (_carry_read_before_its_add, torch.bfloat16),
        (_carry_read_after_its_add, torch.bfloat16),
        (_sum_and_squared_tile_sums, torch.bfloat16),
    ],
    ids=[
        "integer_sum",
        "half_sum",
        "half_sum_into_a_float_carry",
        "float_sum_into_a_double_carry",
        "carry_read_before_its_add",
        "carry_read_after_its_add",
        "another_tile_sum_stays",
    ],
)
def test_a_carried_tile_sum_the_pass_cannot_move_does_not_offer_the_knob(
    kernel: Kernel[Any], dtype: torch.dtype
) -> None:
    spec = _spec(kernel, torch.empty(4096, 64, dtype=dtype))
    assert _KNOB not in spec._flat_fields()
    with pytest.raises(exc.InvalidConfig, match=f"{_KNOB} requires"):
        spec.normalized_config(helion.Config.from_dict({_KNOB: True}))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_a_float_or_double_sum_into_a_carry_of_its_dtype_moves(
    dtype: torch.dtype,
) -> None:
    x = torch.empty(4096, 64, dtype=dtype)
    assert _KNOB in _spec(_carried_sum_in_its_own_dtype, x)._flat_fields()
    config = _sink_config(
        block_sizes=[128, 16], num_threads=[16, 16], vec=[1, 1], sink=False
    )
    assert cross_thread_combines(
        _code(_carried_sum_in_its_own_dtype, (x,), **config)
    ) == (1, 1)
    assert cross_thread_combines(
        _code(_carried_sum_in_its_own_dtype, (x,), **config, **{_KNOB: True})
    ) == (0, 1)


def test_seeds_with_several_threads_over_several_tiles_carry_the_witnesses() -> None:
    def witnesses(spec: ConfigSpec) -> list[dict[str, Any]]:
        return [
            witness.carrier.config
            for group in spec.compiler_coverage_groups
            if group.key == _KNOB
            for witness in group.witnesses
            if witness.value is True
        ]

    # Splits of 600 rows: the split column reduction's 128-row seeds walk
    # five tiles each, its 1024-row seed one.
    x = torch.empty(4700, 64, dtype=torch.bfloat16)
    carriers = witnesses(_spec(split_column_sums, x, x, 600))
    assert [(c["block_sizes"], c["num_threads"]) for c in carriers] == [
        ([1, 16, 128], [1, 2, 16]),
        ([1, 32, 128], [1, 4, 8]),
    ]
    assert all(c["cute_vector_widths"][2] == 1 and c[_KNOB] is False for c in carriers)
    # Each witness is code its carrier does not render.
    for carrier in carriers:
        off = _code(split_column_sums, (x, x, 600), **carrier)
        on = _code(split_column_sums, (x, x, 600), **{**carrier, _KNOB: True})
        assert cross_thread_combines(off) == (2, 2)
        assert cross_thread_combines(on) == (0, 2)
    # One rule of the carrier at a time: several tiles, several threads on
    # the row tile, several rows per thread, scalar lanes along the tile.
    with cpu_render_target(reduction_smem_budget_bytes=232448):
        env = cpu_bind(split_column_sums, (x, x, 600)).env

    def carries(**changed: object) -> bool:
        return walks_a_deferrable_tile_sum(
            env, helion.Config.from_dict({**carriers[0], **changed})
        )

    assert carries()
    assert not carries(block_sizes=[1, 16, 1024])
    assert not carries(num_threads=[1, 2, 1])
    assert not carries(cute_vector_widths=[1, 8, 8])
    # Neither is code the knob changes: one thread on the row tile combines
    # nothing, and vector lanes along it are folded by another lane split.
    for changed in ({"num_threads": [1, 2, 1]}, {"cute_vector_widths": [1, 8, 8]}):
        same = {**carriers[0], **changed}
        assert _code(split_column_sums, (x, x, 600), **same) == _code(
            split_column_sums, (x, x, 600), **{**same, _KNOB: True}
        )
    # One row per thread: no lane loop folds the rows, so the split records
    # nothing and the knob renders the carrier's own code.
    one_row = {**carriers[0], "block_sizes": [1, 16, 16]}
    assert not carries(block_sizes=[1, 16, 16])
    assert _code(split_column_sums, (x, x, 600), **one_row) == _code(
        split_column_sums, (x, x, 600), **{**one_row, _KNOB: True}
    )
    # Two rows per thread are enough.
    assert carries(block_sizes=[1, 16, 32])
    # Seeds whose row tile covers the rows combine once as it is, and seeds
    # with vector lanes along the rows fold them in another lane split.
    for shape in ((1 << 16, 64), (4096, 4096)):
        spec = _spec(col_reduce_sum_static, torch.empty(shape, dtype=torch.bfloat16))
        assert _KNOB in spec._flat_fields()
        assert not witnesses(spec)
