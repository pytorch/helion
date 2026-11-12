"""Grid vector-loop sinking for serial reductions (``cute_vloop_sink``).

A column sum (``acc += torch.sum(x[tile_m, tile_n], dim=0)`` with the vector
along the grid axis ``tile_n``) used to run its constexpr V-loop OUTSIDE the
serial row loops: every thread walked its row strip V times with 2-byte
scalar loads and paid V cross-thread combines per row tile.  With
``cute_vloop_sink`` the V-loop is distributed over the body and interchanged
with the loops: one V-wide vector load per row, V register accumulators, one
fragment grouped combine per row tile, strength-reduced row addressing and an
optional load-first unroll of the row lane loop (``cute_lane_unroll``).

CPU code generation first; the GPU tests at the end run the kernels.
"""

from __future__ import annotations

import ast
import builtins
from contextlib import ExitStack
from contextlib import contextmanager
import functools
import json
import re
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import cast
from unittest.mock import patch

import pytest
import torch

from test._cute_vloop_sink_kernels import FRAGMENT_REDUCE
from test._cute_vloop_sink_kernels import TWO_STAGE_REDUCE
from test._cute_vloop_sink_kernels import _sink_config
from test._cute_vloop_sink_kernels import col_reduce_max_dynamic
from test._cute_vloop_sink_kernels import col_reduce_sum_atomic_then_load_static
from test._cute_vloop_sink_kernels import col_reduce_sum_dynamic
from test._cute_vloop_sink_kernels import col_reduce_sum_from8_dynamic
from test._cute_vloop_sink_kernels import col_reduce_sum_from8_static
from test._cute_vloop_sink_kernels import col_reduce_sum_gather_then_atomic_static
from test._cute_vloop_sink_kernels import col_reduce_sum_lower_triangle_static
from test._cute_vloop_sink_kernels import col_reduce_sum_pair_dynamic
from test._cute_vloop_sink_kernels import col_reduce_sum_pair_static
from test._cute_vloop_sink_kernels import col_reduce_sum_rescaled_const_static
from test._cute_vloop_sink_kernels import col_reduce_sum_rescaled_dynamic
from test._cute_vloop_sink_kernels import col_reduce_sum_rescaled_static
from test._cute_vloop_sink_kernels import col_reduce_sum_static
from test._cute_vloop_sink_kernels import col_weighted_mean_dynamic
from test._cute_vloop_sink_kernels import col_weighted_mean_static

import helion
from helion._compiler.ast_read_writes import ReadWrites
from helion._compiler.cute import device_state
from helion._compiler.cute import kernel_generators
from helion._compiler.cute import sink_vector_loops
from helion._compiler.cute.autotune.heuristics.cute import CuteColumnReductionHeuristic
from helion._compiler.cute.autotune.heuristics.cute import (
    CuteSplitColumnReductionHeuristic,
)
from helion._compiler.cute.fuse_fma import fuse_fma
from helion._compiler.cute.lane_loops import root_passes
from helion._compiler.cute.sink_vector_loops import VloopSinkNotApplied
from helion._compiler.cute.state import cute_env_state
from helion._compiler.reduction_strategy import LoopedReductionStrategy
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import cpu_render_target
from helion._testing import import_example
from helion._testing import onlyBackends
from helion._testing import patch_target_facts
from helion._testing import skipIfNotCUDA
from helion._testing.cute_codegen import cpu_bind
import helion.language as hl

pytestmark = pytest.mark.cpu_only

rms_norm_fwd = import_example("rms_norm").rms_norm_fwd
softmax = import_example("softmax").softmax

if TYPE_CHECKING:
    from collections.abc import Iterator

    from helion._compiler.generate_ast import GenerateAST
    from helion.autotuner.config_spec import ConfigSpec
    from helion.runtime.kernel import BoundKernel

cutlass = pytest.importorskip("cutlass")


@pytest.fixture
def cpu_only() -> Iterator[None]:
    with cpu_render_target(reduction_smem_budget_bytes=232448):
        yield


def _cpu_code(
    kernel: helion.Kernel, args: tuple[torch.Tensor, ...], **config: object
) -> str:
    bound = cpu_bind(kernel, args)
    return bound.to_code(
        bound.config_spec.normalized_config(helion.Config.from_dict(dict(config)))
    )


def _loop(code: str, target: str) -> ast.For:
    """The (unique) ``for <target> in ...`` loop of the generated kernel."""
    loops = [
        node
        for node in ast.walk(ast.parse(code))
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == target
    ]
    assert len(loops) == 1, f"{len(loops)} loops over {target}"
    return loops[0]


def _vector_loads(body: list[ast.stmt]) -> list[ast.stmt]:
    """The packet loads of ``body``: plain ``cute.arch.load`` or, under a
    guard, the predicated ``_cute_load_v*b_if`` helpers."""
    return [
        stmt
        for stmt in body
        if isinstance(stmt, ast.Assign)
        and (
            "cute.arch.load(" in ast.unparse(stmt)
            or "_cute_load_v" in ast.unparse(stmt)
        )
        and "ir.VectorType.get(" in ast.unparse(stmt)
    ]


def _kernel_def(code: str) -> ast.FunctionDef:
    module = ast.parse(code)
    (kernel,) = [
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("_helion_")
    ]
    return kernel


def _assert_defined_before_use(code: str) -> None:
    """Every name the kernel reads was assigned earlier in program order (each
    loop body walked once): no hoisted statement outran its operands."""
    module = ast.parse(code)
    kernel = _kernel_def(code)
    defined = set(dir(builtins)) | {arg.arg for arg in kernel.args.args}
    for node in module.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            defined |= {alias.asname or alias.name for alias in node.names}
        elif isinstance(node, ast.Assign):
            defined |= set(ReadWrites.from_ast(node).writes)

    def visit(stmts: list[ast.stmt]) -> None:
        for stmt in stmts:
            if isinstance(stmt, ast.For):
                assert not set(ReadWrites.from_ast(stmt.iter).reads) - defined
                assert isinstance(stmt.target, ast.Name)
                defined.add(stmt.target.id)
                visit(stmt.body)
            elif isinstance(stmt, ast.If):
                assert not set(ReadWrites.from_ast(stmt.test).reads) - defined
                visit(stmt.body)
                visit(stmt.orelse)
            else:
                rw = ReadWrites.from_ast(stmt)
                undefined = set(rw.reads) - defined
                assert not undefined, (
                    f"{sorted(undefined)} read before assignment in {ast.unparse(stmt)}"
                )
                defined.update(rw.writes)

    visit(kernel.body)


def _scalar_carried_updates_in_vloops(code: str) -> list[str]:
    """Plain-name assignments inside a constexpr V-loop that read their own
    target: a V-invariant accumulator that would be updated V times."""
    found: list[str] = []
    for loop in ast.walk(ast.parse(code)):
        if not (
            isinstance(loop, ast.For)
            and "cutlass.range_constexpr" in ast.unparse(loop.iter)
        ):
            continue
        for stmt in loop.body:
            if (
                isinstance(stmt, ast.Assign)
                and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name)
                and stmt.targets[0].id in ReadWrites.from_ast(stmt.value).reads
            ):
                found.append(ast.unparse(stmt))
    return found


def _both(
    kernel: helion.Kernel, args: tuple[torch.Tensor, ...], **config: object
) -> str:
    """The knob-on code, asserted byte-identical to the knob-off code."""
    on = _cpu_code(kernel, args, **{**config, "cute_vloop_sink": True})
    off = _cpu_code(kernel, args, **{**config, "cute_vloop_sink": False})
    assert on == off
    assert "_vsink" not in on
    return on


def test_column_sum_sinks_the_vector_loop_into_the_row_loops(cpu_only: None) -> None:
    x = torch.empty(4096, 4096, dtype=torch.bfloat16)
    code = _cpu_code(
        col_reduce_sum_static,
        (x,),
        **_sink_config(
            block_sizes=[4096, 16], num_threads=[128, 4], vec=[1, 4], unroll=8
        ),
    )
    # Per-lane state lives in V-element register fragments: the carried
    # column accumulator, the per-row-tile lane accumulator and the reduced
    # values.
    assert code.count("cute.make_rmem_tensor(4, cutlass.Float32)") == 3
    # The row lane loop is the outermost loop over the strip: 32 rows per
    # thread unrolled by 8 -> 4 trips, each issuing all 8 vector loads
    # before any constexpr V-loop consumes them.
    lane_loop = _loop(code, "lane_0")
    assert ast.unparse(lane_loop.iter) == "range(4)"
    loads = _vector_loads(lane_loop.body)
    assert len(loads) == 8
    assert all(
        "ir.VectorType.get([4], cutlass.Uint16.mlir_type)" in ast.unparse(stmt)
        for stmt in loads
    )
    first_consume = next(
        index for index, stmt in enumerate(lane_loop.body) if isinstance(stmt, ast.For)
    )
    assert all(lane_loop.body.index(stmt) < first_consume for stmt in loads)
    # Strength-reduced addressing: base pointer + lane * row step, both
    # hoisted before the lane loop, instead of ``row * stride + col`` per row.
    assert "_vsink_ptr = x.iterator + " in code
    assert (
        "_vsink_step = cutlass.Int32(128) * cutlass.Int32(x.layout.stride[0])" in code
    )
    assert "_vsink_ptr + cutlass.Int32(lane_0 * 8 + 7) * _vsink_step" in code
    # The lane loop only accumulates; no per-element cross-thread combine.
    lane_body = ast.unparse(lane_loop)
    assert "_cute_grouped_reduce" not in lane_body
    assert "warp_reduction" not in lane_body
    assert ".load()" not in code
    # One fragment grouped combine per row tile replaces V two-stage reduces.
    assert code.count(FRAGMENT_REDUCE) == 1
    assert TWO_STAGE_REDUCE not in code
    assert "count=4, pre=4, group_span=512, group_count=1" in code
    # Sinking keeps the vectorized grid axis on thread x so adjacent column
    # chunks sit in adjacent threads (the warp-per-row swap is not applied).
    assert "block=(4, 128, 1)" in code
    # The vectorized store of the finished columns is unchanged.
    assert "_cute_store_u16_vec(out.iterator" in code


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _column_sum(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty([x.size(1)], dtype=torch.float32, device=x.device)
    for tile_c in hl.tile(x.size(1)):
        out[tile_c] = torch.sum(x[:, tile_c].float(), 0)
    return out


_ONE_TILE_COLUMN_SUM = {
    "block_sizes": [4],
    "num_threads": [1, 128],
    "cute_vector_widths": [1, 4],
    "reduction_loops": [1024],
    "cute_lane_unroll": 8,
}


def test_one_block_column_sum_sinks_its_vector_loop(cpu_only: None) -> None:
    """A column sum over a lone ``hl.tile(cols)`` takes the flattened tile
    strategy, whose V-loop the sink never saw: every thread walked its rows
    V times with scalar loads (better-benchmark's RMSNorm weight gradient
    over ``[36000, 512]`` fp32 partials). One packet per row now."""
    x = torch.empty(36000, 512, dtype=torch.float32)
    code = _cpu_code(_column_sum, (x,), **_ONE_TILE_COLUMN_SUM, cute_vloop_sink=True)
    rows = _loop(code, "roffset_1")
    loads = [
        stmt
        for stmt in ast.walk(rows)
        if isinstance(stmt, ast.Assign) and "_cute_load_v16b_if(" in ast.unparse(stmt)
    ]
    assert len(loads) == 8
    assert ".load()" not in code
    assert code.count("cute.make_rmem_tensor(4, cutlass.Float32)") == 2
    assert code.count(FRAGMENT_REDUCE) == 1
    # Resized for the launched block like the knob-off reduce: one group of
    # four warps, which combines in the serial form's order.
    assert "count=4, pre=1, group_span=128, group_count=1)" in code
    assert TWO_STAGE_REDUCE not in code
    assert "_cute_store_u32_vec(out.iterator" in code
    # Without the knob the per-element index keeps its ``offsets`` spelling
    # and the V-loop stays outside the row loop.
    off = _cpu_code(_column_sum, (x,), **_ONE_TILE_COLUMN_SUM, cute_vloop_sink=False)
    assert "indices_0 = offsets_0" in off
    assert "_vsink" not in off


@pytest.mark.parametrize(
    ("columns", "block", "threads"),
    [(520, 16, [1, 128]), (516, 8, [1, 128]), (516, 32, [8, 128])],
)
def test_a_masked_one_block_tile_sinks_its_vector_loop(
    cpu_only: None, columns: int, block: int, threads: list[int]
) -> None:
    """The last tile is partial, but its width is a multiple of V, so every
    chunk lies wholly inside or outside the columns: the tile's mask, spelled
    on the element index under the sink, is evaluated once per chunk."""
    x = torch.empty(3000, columns, dtype=torch.float32)
    config = {**_ONE_TILE_COLUMN_SUM, "block_sizes": [block], "num_threads": threads}
    code = _cpu_code(_column_sum, (x,), **config, cute_vloop_sink=True)
    assert f"mask_0 = lane_base_0 < {columns}" in code
    loads = [
        stmt
        for stmt in ast.walk(_loop(code, "roffset_1"))
        if isinstance(stmt, ast.Assign) and "_cute_load_v16b_if(" in ast.unparse(stmt)
    ]
    assert len(loads) == 8
    assert ".load()" not in code
    assert code.count(FRAGMENT_REDUCE) == 1


def test_a_one_block_tile_that_sinks_nothing_keeps_the_knob_off_code(
    cpu_only: None,
) -> None:
    """518 columns are not a multiple of V: a chunk can straddle the last
    column, nothing sinks, and the index the knob respelled for the sink is
    regenerated with the knob off."""
    x = torch.empty(3000, 518, dtype=torch.float32)
    code = _both(_column_sum, (x,), **{**_ONE_TILE_COLUMN_SUM, "block_sizes": [8]})
    assert "indices_0 = offsets_0" in code


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _doubled(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] * 2.0
    return out


def test_a_one_block_tile_without_a_loop_keeps_its_lane_unroll(
    cpu_only: None,
) -> None:
    """A pointwise tile has no loop to sink into: the knob-on code is the
    code of the knob-off config with the lane unroll it asks for, whose
    loads-first unroll of the grid lane loop stays (the knob-off retry used
    to drop the unroll as well)."""
    x = torch.empty(1 << 20, dtype=torch.float32)
    code = _both(
        _doubled,
        (x,),
        block_sizes=[2048],
        num_threads=[128],
        cute_vector_widths=[4],
        cute_lane_unroll=4,
    )
    assert "_lane_unroll_loads" in code


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _scaled_atomic_add(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    for tile in hl.tile(x.size(0)):
        hl.atomic_add(out, [tile], x[tile] * 2.0)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _column_sum_and_count(x: torch.Tensor, count: torch.Tensor) -> torch.Tensor:
    """``_column_sum`` that also bumps a per-column counter."""
    out = torch.empty([x.size(1)], dtype=torch.float32, device=x.device)
    for tile_c in hl.tile(x.size(1)):
        out[tile_c] = torch.sum(x[:, tile_c].float(), 0)
        hl.atomic_add(count, [tile_c], hl.full([tile_c], 1.0, dtype=torch.float32))
    return out


def test_a_one_block_atomic_flushes_as_a_vector_only_beside_a_sink(
    cpu_only: None,
) -> None:
    """The wrapper the knob records for the sink also lets a one-block
    tile's atomic flush its V lanes as one vector atomic, as an N-D tile's
    always may.  Without a reduction to sink the knob-on code is the
    knob-off code, scalar atomics included; beside a sunk column sum the
    counter's atomic becomes a vector atomic."""
    with patch_target_facts(capability=(10, 0)):
        x = torch.empty(4096, dtype=torch.float32)
        code = _both(
            _scaled_atomic_add,
            (x, torch.empty_like(x)),
            block_sizes=[512],
            num_threads=[128],
            cute_vector_widths=[4],
        )
        assert "_cute_red_add_f32_vec(" not in code
        columns = (torch.empty(3000, 512, dtype=torch.float32), torch.empty(512))
        code = _cpu_code(
            _column_sum_and_count,
            columns,
            **_ONE_TILE_COLUMN_SUM,
            cute_vloop_sink=True,
        )
        assert "_vsink_vec" in code
        assert "_cute_red_add_f32_vec((count.iterator" in code
        off = _cpu_code(
            _column_sum_and_count,
            columns,
            **_ONE_TILE_COLUMN_SUM,
            cute_vloop_sink=False,
        )
        assert "_cute_red_add_f32_vec(" not in off


def test_lane_unroll_one_keeps_the_rolled_lane_loop(cpu_only: None) -> None:
    x = torch.empty(4096, 4096, dtype=torch.bfloat16)
    code = _cpu_code(
        col_reduce_sum_static,
        (x,),
        **_sink_config(
            block_sizes=[4096, 16], num_threads=[256, 2], vec=[1, 8], unroll=1
        ),
    )
    lane_loop = _loop(code, "lane_0")
    assert ast.unparse(lane_loop.iter) == "range(16)"
    loads = _vector_loads(lane_loop.body)
    assert len(loads) == 1
    assert "ir.VectorType.get([8], cutlass.Uint16.mlir_type)" in ast.unparse(loads[0])
    assert "_vsink_ptr + cutlass.Int32(lane_0) * _vsink_step" in code
    assert "count=8, pre=2, group_span=512, group_count=1" in code
    assert "block=(2, 256, 1)" in code


def test_sunk_loads_keep_the_cache_policy(cpu_only: None) -> None:
    """A load's eviction policy applies to the vector form the sinking pass
    emits (the L2 policy helpers exist for 16-byte packets)."""
    x = torch.empty(4096, 4096, dtype=torch.bfloat16)
    code = _cpu_code(
        col_reduce_sum_static,
        (x,),
        load_eviction_policies=["l2_last"],
        **_sink_config(
            block_sizes=[4096, 16], num_threads=[256, 2], vec=[1, 8], unroll=1
        ),
    )
    assert "_vsink_vec = _cute_load_l2_evict_last(_vsink_ptr + " in code
    assert "load_v16b_l2_evict_last as _cute_load_l2_evict_last" in code
    assert "cute.arch.load(" not in code


def test_rows_above_columns_uses_the_interleaved_fragment_reduce(
    cpu_only: None,
) -> None:
    """Row threads above the column threads (pre=8, span=128): the fragment
    reduce keeps the interleaved sibling coordinate apart."""
    x = torch.empty(2048, 64, dtype=torch.bfloat16)
    code = _cpu_code(
        col_reduce_sum_static,
        (x,),
        **_sink_config(
            block_sizes=[2048, 16],
            num_threads=[16, 8],
            vec=[1, 2],
            layouts=["blocked", "blocked"],
            unroll=4,
        ),
    )
    assert code.count(FRAGMENT_REDUCE) == 1
    assert "count=2, pre=8, group_span=128, group_count=1" in code
    assert "_vsink_step = cutlass.Int32(x.layout.stride[0])" in code
    assert len(_vector_loads(_loop(code, "lane_0").body)) == 4


def test_dynamic_shapes_guard_whole_vectors_with_a_uniform_mask(
    cpu_only: None,
) -> None:
    """With ``n % V == 0`` the column mask is the same for all V lanes: it is
    evaluated once at the chunk base, guards the predicated vector transaction
    (masked threads load nothing and see zeros) and still gates every value."""
    x = torch.empty(4000, 4000, dtype=torch.bfloat16)
    code = _cpu_code(
        col_reduce_sum_dynamic,
        (x,),
        **_sink_config(
            block_sizes=[4096, 16], num_threads=[128, 4], vec=[1, 4], unroll=2
        ),
    )
    assert (
        "mask_1 = cutlass.Int32(cute.arch.thread_idx()[0]) < 4 and lane_base_1 < n"
        in code
    )
    assert "indices_1 < n" not in code
    lane_loop = _loop(code, "lane_0")
    loads = _vector_loads(lane_loop.body)
    assert len(loads) == 2
    # Four bf16 lanes: an 8-byte predicated packet at the strength-reduced
    # row pointer, guarded by the row mask and the chunk-base column mask.
    assert "_cute_load_v8b_if(_vsink_ptr + " in ast.unparse(loads[0])
    assert ", mask_0_u0 and mask_1, ir.VectorType" in ast.unparse(loads[0])
    assert "mask_0_u1 = indices_0_u1 < m" in ast.unparse(lane_loop)
    # The predicated load left zeros in the masked lanes; the value needs no
    # select of its own.
    assert "_vsink_vec_u1[vec_lane_1]).bitcast(cutlass.BFloat16)" in ast.unparse(
        lane_loop
    )
    assert "if mask_0_u1 and mask_1 else cutlass.BFloat16(0)" not in ast.unparse(
        lane_loop
    )
    # The finished columns are stored under the same uniform mask.
    assert "if mask_1:" in code
    assert code.count(FRAGMENT_REDUCE) == 1


def test_extent_not_a_multiple_of_the_vector_width_stays_scalar(
    cpu_only: None,
) -> None:
    """4002 columns cannot be covered by whole 4-wide chunks: no load fact is
    recorded, nothing sinks, and the knob-on code (thread layout included)
    is exactly the knob-off code."""
    x = torch.empty(4000, 4002, dtype=torch.bfloat16)
    code = _both(
        col_reduce_sum_dynamic,
        (x,),
        **_sink_config(
            block_sizes=[4096, 16], num_threads=[128, 4], vec=[1, 4], unroll=8
        ),
    )
    assert FRAGMENT_REDUCE not in code
    assert ".load()" in ast.unparse(_loop(code, "lane_0"))
    assert code.count(TWO_STAGE_REDUCE) == 1
    assert "block=(128, 4, 1)" in code


def test_nonzero_grid_origin_keeps_per_element_masks_and_scalar_loads(
    cpu_only: None,
) -> None:
    """``hl.tile(8, n)``: the chunks are not V-aligned and the column mask
    differs per element, so no load fact is recorded and the vector loop
    stays put.  Knob-on regenerates the knob-off code byte for byte."""
    x = torch.empty(1000, 1008, dtype=torch.bfloat16)
    code = _both(
        col_reduce_sum_from8_static,
        (x,),
        **_sink_config(
            block_sizes=[4096, 16], num_threads=[128, 4], vec=[1, 4], unroll=8
        ),
    )
    assert "indices_1 = lane_base_1 + cutlass.Int32(vec_lane_1)" in code
    assert "and indices_1 < 1008" in code
    assert "cute.arch.load(" not in code
    assert ".load() if mask_0 and mask_1 else" in code
    assert "block=(128, 4, 1)" in code


def _strided_view(view: str) -> torch.Tensor:
    if view == "offset_2":
        # Base four bytes into the allocation: misaligned for the 8-byte packet.
        return torch.empty(4096, 4112, dtype=torch.bfloat16)[:, 2:4098]
    if view == "row_stride_4098":
        # 8196-byte rows: every odd row starts four bytes into a packet.
        return torch.empty(4096, 4098, dtype=torch.bfloat16)[:, :4096]
    assert view == "row_stride_4100"
    # 8200-byte rows are 8-byte aligned, so the packet is legal.
    return torch.empty(4096, 4100, dtype=torch.bfloat16)[:, :4096]


@pytest.mark.parametrize("view", ["offset_2", "row_stride_4098"])
def test_misaligned_input_views_stay_scalar_with_the_knob_on(
    cpu_only: None, view: str
) -> None:
    """The sinkable-load fact shares the tile hoist's base and stride proof
    (``cute_reduction_vector_layout_aligned``): a view whose base or row
    stride is not a multiple of the packet records no fact, nothing sinks and
    the knob-on code is the knob-off code, which the hoist had already left
    on scalar loads."""
    code = _both(
        col_reduce_sum_static,
        (_strided_view(view),),
        **_sink_config(
            block_sizes=[4096, 16], num_threads=[128, 4], vec=[1, 4], unroll=8
        ),
    )
    assert "cute.arch.load(" not in code
    assert ".load()" in ast.unparse(_loop(code, "lane_0"))
    assert FRAGMENT_REDUCE not in code
    assert code.count(TWO_STAGE_REDUCE) == 1


def test_an_aligned_row_stride_still_sinks(cpu_only: None) -> None:
    code = _cpu_code(
        col_reduce_sum_static,
        (_strided_view("row_stride_4100"),),
        **_sink_config(
            block_sizes=[4096, 16], num_threads=[128, 4], vec=[1, 4], unroll=8
        ),
    )
    assert len(_vector_loads(_loop(code, "lane_0").body)) == 8
    assert ".load()" not in code


def test_per_element_load_conditions_are_unsupported(cpu_only: None) -> None:
    """A load under a condition that differs per V lane (an ``extra_mask``
    over both axes) has no whole-chunk transaction: the pass raises
    ``_Unsupported`` and the knob-off code is generated instead."""
    x = torch.empty(1024, 1024, dtype=torch.bfloat16)
    declined: list[bool] = []
    original = sink_vector_loops._Sinker.run

    def run(self: sink_vector_loops._Sinker) -> list[ast.stmt] | None:
        try:
            return original(self)
        except sink_vector_loops._Unsupported:
            declined.append(True)
            raise

    # The knob-off code puts the row threads on the z axis, whose launch limit
    # is 64; 128 of them were an unlaunchable ``block=(2, 4, 128)``.
    with patch.object(sink_vector_loops._Sinker, "run", run):
        code = _both(
            col_reduce_sum_lower_triangle_static,
            (x,),
            **_sink_config(
                block_sizes=[1024, 16], num_threads=[64, 4], vec=[1, 4], unroll=8
            ),
        )
    assert declined == [True]
    assert "cute.arch.load(" not in code
    assert "operator.ge(indices_0, indices_1)" in code


def test_row_only_loads_and_accumulators_become_per_lane(cpu_only: None) -> None:
    """``acc += sum(x * w[:, None]); wsum += sum(w)``: the row weight is
    loaded once per row before the V-loop (with its convert), and the
    row-only accumulator lives in a fragment like the column accumulator, so
    its update runs once per lane instead of V times per row."""
    x = torch.empty(1024, 1024, dtype=torch.bfloat16)
    w = torch.empty(1024, dtype=torch.bfloat16)
    code = _cpu_code(
        col_weighted_mean_static,
        (x, w),
        **_sink_config(block_sizes=[1024, 16], num_threads=[128, 4], vec=[1, 4]),
    )
    _assert_defined_before_use(code)
    assert _scalar_carried_updates_in_vloops(code) == []
    lane_loop = _loop(code, "lane_0")
    statements = [ast.unparse(stmt) for stmt in lane_loop.body]
    (row_load,) = [text for text in statements if ".load()" in text]
    assert row_load.startswith("load = (w.iterator + cutlass.Int32(indices_0)")
    assert statements.index("v_0 = cutlass.Float32(load)") > statements.index(row_load)
    assert statements.index("v_0 = cutlass.Float32(load)") < next(
        index for index, stmt in enumerate(lane_loop.body) if isinstance(stmt, ast.For)
    )
    assert len(_vector_loads(lane_loop.body)) == 1
    # acc, wsum, both lane accumulators and both reduced values are per lane.
    assert code.count("cute.make_rmem_tensor(4, cutlass.Float32)") == 6
    assert "wsum_frag[vec_lane_1] = wsum_copy + sum_2" in code
    assert "sum_2_lane_acc_frag[vec_lane_1] = sum_2_lane_acc_frag[vec_lane_1] +" in code
    assert code.count(FRAGMENT_REDUCE) == 2
    assert TWO_STAGE_REDUCE not in code
    assert "v_5 = acc_frag[vec_lane_1] / wsum_frag[vec_lane_1]" in code


def test_sibling_row_loops_sink_independently(cpu_only: None) -> None:
    """Two row loops over the same rows block define the same row index and
    mask names; each loop hoists its own definitions ahead of its vector
    load."""
    x = torch.empty(1000, 1024, dtype=torch.bfloat16)
    y = torch.empty(1000, 1024, dtype=torch.bfloat16)
    code = _cpu_code(
        col_reduce_sum_pair_static,
        (x, y),
        **_sink_config(block_sizes=[4096, 16], num_threads=[128, 4], vec=[1, 4]),
    )
    _assert_defined_before_use(code)
    assert _scalar_carried_updates_in_vloops(code) == []
    lane_loops = [
        node
        for node in ast.walk(_kernel_def(code))
        if isinstance(node, ast.For) and _vector_loads(node.body)
    ]
    assert len(lane_loops) == 2
    for loop, tensor in zip(lane_loops, ("x", "y"), strict=True):
        statements = [ast.unparse(stmt) for stmt in loop.body]
        (load,) = [text for text in statements if "_cute_load_v8b_if(" in text]
        # The strength-reduced row pointer of this loop's loads starts at
        # this loop's tensor.
        (base,) = re.findall(r"_cute_load_v8b_if\((\w+) \+ ", load)
        assert f"{base} = {tensor}.iterator + " in code
        # The row index and the row mask of this loop come first, in the
        # same lane iteration, ahead of the vector transaction they address
        # and guard.
        (row_index,) = [text for text in statements if text.startswith("indices_")]
        (row_mask,) = [text for text in statements if text.startswith("mask_")]
        assert f", {row_mask.split(' = ')[0]}, ir.VectorType" in load
        assert statements.index(row_index) < statements.index(row_mask)
        assert statements.index(row_mask) < statements.index(load)
    assert code.count(FRAGMENT_REDUCE) == 2
    assert ".load()" not in code


@pytest.mark.parametrize(
    ("kernel", "extra_args", "definition"),
    [
        (
            col_reduce_sum_rescaled_static,
            (torch.empty(1, dtype=torch.float32),),
            (
                "s = (scale.iterator + cutlass.Int32(0) * "
                "cutlass.Int32(scale.layout.stride[0])).load()"
            ),
        ),
        (col_reduce_sum_rescaled_const_static, (), "s = cutlass.Float32(1.5)"),
    ],
    ids=["load", "constant"],
)
def test_a_lane_invariant_scalar_redefined_inside_the_segment_stays_per_lane(
    cpu_only: None,
    kernel: helion.Kernel,
    extra_args: tuple[torch.Tensor, ...],
    definition: str,
) -> None:
    """``s = scale[0]; out1 = acc * s; if flag[0]: s = s * 2; out2 = acc * s``:
    ``s`` is lane-invariant but defined twice in the epilogue segment.  Its
    first definition must not run once before the V-loop (lanes after the
    first would see the doubled value in BOTH stores): it stays in the
    V-loop with the ``if``, while the row loop still sinks."""
    x = torch.empty(1024, 1024, dtype=torch.bfloat16)
    flag = torch.empty(1, dtype=torch.int32)
    code = _cpu_code(
        kernel,
        (x, *extra_args, flag),
        **_sink_config(block_sizes=[1024, 16], num_threads=[128, 4], vec=[1, 4]),
    )
    _assert_defined_before_use(code)
    assert "_vsink_vec" in code
    assert code.count(FRAGMENT_REDUCE) == 1
    # The epilogue V-loop holds both definitions of ``s`` and both uses.
    (epilogue,) = [
        loop
        for loop in ast.walk(_kernel_def(code))
        if isinstance(loop, ast.For)
        and "range_constexpr" in ast.unparse(loop.iter)
        and any(isinstance(stmt, ast.If) for stmt in loop.body)
    ]
    statements = [ast.unparse(stmt) for stmt in epilogue.body]
    assert statements[0] == definition
    assert "v_2 = acc_frag[vec_lane_1] * s" in statements
    (conditional,) = [stmt for stmt in epilogue.body if isinstance(stmt, ast.If)]
    assert "s = _cute_join_cast(v_6, s_before_if)" in ast.unparse(conditional)
    assert "v_8 = acc_frag[vec_lane_1] * s" in statements
    assert (
        statements.index("v_2 = acc_frag[vec_lane_1] * s")
        < statements.index("s_before_if = s")
        < epilogue.body.index(conditional)
        < statements.index("v_8 = acc_frag[vec_lane_1] * s")
    )
    # The pure single-definition constant of the ``!= 0`` test is hoisted.
    assert "v_3 = 0" not in statements
    assert "v_3 = 0" in code


def test_a_load_after_a_bound_atomic_rejects_the_config(cpu_only: None) -> None:
    """``old = hl.atomic_add(count, [tile_n], 1); first = count[tile.id * B]``:
    threads 1-3 read the element thread 0's atomic bumps, a race no barrier
    can order in the body the reduction split rewrites, so the config is
    refused."""
    x = torch.empty(1024, 1024, dtype=torch.bfloat16)
    count = torch.empty(1024, dtype=torch.float32)
    with pytest.raises(
        helion.exc.BackendUnsupported, match="a body a later pass rewrites"
    ):
        _cpu_code(
            col_reduce_sum_atomic_then_load_static,
            (x, count),
            **_sink_config(block_sizes=[1024, 16], num_threads=[1, 4], vec=[1, 4]),
        )


def test_a_gathered_value_is_not_recomputed_after_the_atomic(cpu_only: None) -> None:
    """``before = gather(count[tile_n], tile_n.index)`` then
    ``hl.atomic_add(count, [tile_n], 1)``: the gather lowers to
    ``count[i] if mask else 0``, a subscript read with no load call.  Reading
    a kernel tensor argument is a load, so the value is not recomputed after
    the row loop (past the atomic): it is read once per lane before that
    lane's atomic and carried in a register fragment, while the row loop
    still sinks."""
    x = torch.empty(1024, 1024, dtype=torch.bfloat16)
    count = torch.empty(1024, dtype=torch.float32)
    config = _sink_config(block_sizes=[1024, 16], num_threads=[1, 4], vec=[1, 4])
    # The vector atomic flush is an sm_90+ form (``cute/atomic_ops.py``).
    with (
        patch("helion.runtime.kernel.target_device_capability", return_value=(10, 0)),
        patch(
            "helion._compiler.compile_environment.target_device_capability",
            return_value=(10, 0),
        ),
    ):
        code = _cpu_code(col_reduce_sum_gather_then_atomic_static, (x, count), **config)
    _assert_defined_before_use(code)
    assert "_vsink_vec" in code
    kernel = _kernel_def(code)

    def reads_count(stmt: ast.AST) -> bool:
        return any(
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id == "count"
            for node in ast.walk(stmt)
        )

    # ``count[...]`` is read by exactly one statement of the kernel, ...
    (gather,) = [
        stmt
        for stmt in ast.walk(kernel)
        if isinstance(stmt, ast.Assign) and reads_count(stmt)
    ]
    gather_text = ast.unparse(gather)
    assert gather_text.startswith("before_frag[vec_lane_1] = count[")
    # ... in the first V-loop of the lane body, ahead of the atomic: the
    # per-lane adds of ``count`` are collected in that V-loop and flushed as
    # one vector atomic after it (``cute/atomic_ops.py``).
    (lane_loop,) = [
        loop
        for loop in ast.walk(kernel)
        if isinstance(loop, ast.For) and ast.unparse(loop.iter) == "range(1)"
    ]
    statements = [ast.unparse(stmt) for stmt in lane_loop.body]
    prologue = next(text for text in statements if text.startswith("for vec_lane_1 in"))
    assert gather_text in prologue
    assert "_tile_atomic_vals_1_0.append(" in prologue
    (atomic,) = [
        text for text in statements if text.startswith("_cute_red_add_f32_vec(")
    ]
    assert statements.index(prologue) < statements.index(atomic)
    assert "cute.arch.atomic_add(" not in ast.unparse(kernel)
    # ... and the value read there is what the epilogue adds.
    every_statement = [
        ast.unparse(stmt) for stmt in ast.walk(kernel) if isinstance(stmt, ast.stmt)
    ]
    assert "v_3 = acc_frag[vec_lane_1] + before_frag[vec_lane_1]" in every_statement
    # The knob-off code reads ``count[...]`` once as well, before the atomic.
    off = _cpu_code(
        col_reduce_sum_gather_then_atomic_static,
        (x, count),
        **{**config, "cute_vloop_sink": False},
    )
    (off_gather,) = [
        stmt
        for stmt in ast.walk(_kernel_def(off))
        if isinstance(stmt, ast.Assign) and reads_count(stmt)
    ]
    assert ast.unparse(off_gather).startswith("before = count[")


def test_fission_stages_regenerate_the_knob_off_code_when_nothing_sinks(
    cpu_only: None,
) -> None:
    """Materialized fission generates every stage from graphs it builds
    itself.  When the knob shaped a stage's thread layout but nothing was
    sunk, the driver rebuilds that stage from knob-off graphs, so the knob-on
    code is exactly the knob-off code here as well."""

    def pointwise_pair(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rows, columns = x.shape
        middle = torch.empty_like(x)
        out = torch.empty_like(x)
        for row in hl.tile(rows):
            for column in hl.tile(columns):
                middle[row, column] = x[row, column] + 1
            for column in hl.tile(columns):
                out[row, column] = middle[row, column] * 2
        return out, middle

    with ExitStack() as stack:
        for name, value in (
            ("helion.runtime.kernel.target_device_capability", (10, 0)),
            ("helion._compiler.compile_environment.target_device_capability", (10, 0)),
            ("helion.runtime.get_num_sm", 148),
            ("helion._compat._is_hip", False),
        ):
            stack.enter_context(patch(name, return_value=value))
        kernel = helion.kernel(
            pointwise_pair,
            backend="cute",
            static_shapes=True,
            autotune_effort="none",
            cute_region_fission=True,
            cute_full_slice_matmul_tiling=False,
        )
        bound = kernel._bind_isolated((torch.empty(64, 256, dtype=torch.bfloat16),))
        assert cute_env_state(bound.env).fission_plan is not None
        spec = bound.config_spec
        base = spec.default_config().config

        def code(sink: bool) -> str:
            return bound.to_code(
                spec.normalized_config(
                    helion.Config.from_dict({**base, "cute_vloop_sink": sink})
                )
            )

        off = code(False)
        assert off.count("_helion_cute_region_") > 0
        # Nothing to sink: the knob-on code is the knob-off code as it is.
        assert code(True) == off
        # Each stage that reports an unsunk knob-shaped layout is regenerated
        # with the knob off.
        restarts: list[bool] = []

        def not_applied(codegen: GenerateAST) -> None:
            restarts.append(True)
            raise VloopSinkNotApplied

        with patch.object(root_passes, "sink_root_vector_loops", not_applied):
            retried = code(True)
        assert restarts == [True, True]
        assert retried == off


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _scale_columns(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile_r, tile_c in hl.tile(x.size()):
        out[tile_r, tile_c] = (
            x[tile_r, tile_c].to(torch.float32) * w[tile_c].to(torch.float32)
        ).to(x.dtype)
    return out


def test_a_guarded_row_store_lets_the_row_loads_sink(cpu_only: None) -> None:
    """500 rows end the 32-row tiles ragged, so each row's store is guarded
    by its row mask.  A plain ``if mask: ptr.store(v)`` is a store into
    ``out`` (``_segment_stored_tensors``), so the vector loop sinks: each
    row's ``x`` loads as one predicated packet.  The guarded store used to be
    a statement substituted for a template line, wrapped in a second
    ``Expr``: an effect the pass could not name, which kept the knob-off code
    (a 2-byte load per element of each row)."""
    args = (
        torch.empty(500, 384, dtype=torch.bfloat16),
        torch.empty(384, dtype=torch.bfloat16),
    )
    code = _cpu_code(
        _scale_columns,
        args,
        block_sizes=[32, 32],
        num_threads=[8, 4],
        loop_orders=[[1, 0]],
        cute_vector_widths=[1, 8],
        cute_lane_layouts=["blocked", "strided"],
        cute_vloop_sink=True,
    )
    rows = _loop(code, "lane_0")
    assert len(_vector_loads(rows.body)) == 1, code


def test_knob_off_keeps_the_original_nest(cpu_only: None) -> None:
    x = torch.empty(4096, 4096, dtype=torch.bfloat16)
    code = _cpu_code(
        col_reduce_sum_static,
        (x,),
        **_sink_config(
            block_sizes=[4096, 16], num_threads=[128, 4], vec=[1, 4], sink=False
        ),
    )
    assert "_vsink" not in code
    assert ".load()" in ast.unparse(_loop(code, "lane_0"))
    assert code.count(TWO_STAGE_REDUCE) == 1
    assert FRAGMENT_REDUCE not in code


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _spatial_mean(x: torch.Tensor) -> torch.Tensor:
    """The squeeze-excite pool of a channels-last activation, as its
    ``[N, H*W, C]`` view: each output reduces a column ``C`` elements apart."""
    batch, _, channels = x.shape
    out = torch.empty([batch, channels], dtype=x.dtype, device=x.device)
    for tile_n, tile_c in hl.tile([batch, channels]):
        out[tile_n, tile_c] = torch.mean(
            x[tile_n, :, tile_c].to(torch.float32), dim=1
        ).to(x.dtype)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _relu_and_spatial_mean(
    x: torch.Tensor, shift: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """A BatchNorm-ReLU epilogue's shape: every element is stored inside the
    rolled reduction that pools it."""
    batch, rows, channels = x.shape
    act = torch.empty_like(x)
    pooled = torch.empty([batch, channels], dtype=x.dtype, device=x.device)
    for tile_n, tile_c in hl.tile([batch, channels]):
        y = torch.relu(
            x[tile_n, :, tile_c].to(torch.float32) - shift[tile_c][None, None, :]
        )
        act[tile_n, :, tile_c] = y.to(act.dtype)
        pooled[tile_n, tile_c] = torch.mean(y, dim=1).to(pooled.dtype)
    return act, pooled


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _relu_then_zero_even_channels(
    x: torch.Tensor, shift: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """``_relu_and_spatial_mean`` with a later masked store of the same
    elements: the even channels of ``act`` end up zero."""
    batch, rows, channels = x.shape
    act = torch.empty_like(x)
    pooled = torch.empty([batch, channels], dtype=x.dtype, device=x.device)
    for tile_n, tile_c in hl.tile([batch, channels]):
        y = torch.relu(
            x[tile_n, :, tile_c].to(torch.float32) - shift[tile_c][None, None, :]
        )
        act[tile_n, :, tile_c] = y.to(act.dtype)
        even = tile_c.index % 2 == 0
        hl.store(
            act,
            [tile_n, slice(None), tile_c],
            torch.zeros_like(y).to(act.dtype),
            extra_mask=even[None, None, :],
        )
        pooled[tile_n, tile_c] = torch.mean(y, dim=1).to(pooled.dtype)
    return act, pooled


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _spatial_argmax(x: torch.Tensor) -> torch.Tensor:
    """``_spatial_mean`` as an argmax: the rolled argreduce's finalize folds
    consecutive lanes, so it cannot combine across interleaved threads."""
    batch, _, channels = x.shape
    out = torch.empty([batch, channels], dtype=torch.int64, device=x.device)
    for tile_n, tile_c in hl.tile([batch, channels]):
        out[tile_n, tile_c] = torch.argmax(x[tile_n, :, tile_c], dim=1)
    return out


def _strided_reduction_config(
    bound: BoundKernel,
    *,
    sink: bool,
    block_sizes: tuple[int, int] = (1, 64),
    grid_threads: tuple[int, int] = (1, 8),
    reduction_threads: int = 16,
    rows: int | None = 32,
    loop_orders: list[list[int]] | None = None,
) -> helion.Config:
    """By default a ``[1, 64]`` tile of rows: eight vectors of eight channels
    on one axis and sixteen reduction threads over 32-row chunks (``rows``
    None: the persistent reduction)."""
    spec = bound.config_spec
    (reduction,) = spec.reduction_loops.valid_block_ids()
    widths = {reduction: 1, 0: 1, 1: 8}
    threads = {0: grid_threads[0], 1: grid_threads[1], reduction: reduction_threads}
    config = helion.Config(
        block_sizes=list(block_sizes),
        num_threads=[threads[b] for b in spec.num_threads.valid_block_ids()],
        cute_vector_widths=[
            widths[b] for b in spec.cute_vector_widths.valid_block_ids()
        ],
        reduction_loops=[rows],
        cute_vloop_sink=sink,
    )
    if loop_orders is not None:
        config.config["loop_orders"] = loop_orders
    return spec.normalized_config(config)


def _thread_axis_of(code: str, name: str) -> int | None:
    match = re.search(rf"{name} = .*?thread_idx\(\)\[(\d)\]", code)
    return None if match is None else int(match.group(1))


def test_strided_reduction_lays_the_vectorized_tile_on_thread_x(
    cpu_only: None,
) -> None:
    """Sinking the channel vector into the rolled spatial reduction puts the
    channel vectors on ``thread_idx[0]``, so a warp's lanes load adjacent
    16-byte chunks of one row instead of 16 rows ``C`` elements apart; the
    reduction threads move to ``thread_idx[1]`` and combine across the
    interleaved channel threads in shared memory, all eight channels of a
    vector in one fragment combine (one pair of barriers, not eight)."""
    x = torch.empty(8, 784, 120, dtype=torch.bfloat16)
    bound = cpu_bind(_spatial_mean, (x,))
    code = bound.to_code(_strided_reduction_config(bound, sink=True))
    assert _thread_axis_of(code, "lane_base_1") == 0, code
    assert _thread_axis_of(code, r"rindex_\d") == 1, code
    assert "_cute_load_v16b_if(" in code
    assert code.count(FRAGMENT_REDUCE) == 1, code
    assert TWO_STAGE_REDUCE not in code
    assert "block=(8, 16, 1)" in code


def test_strided_reduction_keeps_its_layout_with_the_knob_off(cpu_only: None) -> None:
    x = torch.empty(8, 784, 120, dtype=torch.bfloat16)
    bound = cpu_bind(_spatial_mean, (x,))
    code = bound.to_code(_strided_reduction_config(bound, sink=False))
    assert _thread_axis_of(code, r"rindex_\d") == 0, code
    assert "_vsink" not in code


@pytest.mark.parametrize(
    ("loop_orders", "launch", "vector_axis"),
    [
        # ``tile_n`` first: its two threads would sit below the channel
        # vectors, so the default layout stays.
        (None, "block=(16, 2, 8)", 2),
        # ``tile_c`` first: the vectors take thread_idx[0].
        ([[1, 0]], "block=(8, 2, 16)", 0),
    ],
)
def test_threads_on_both_grid_dims_lay_the_vectors_first_only_in_loop_order(
    cpu_only: None,
    loop_orders: list[list[int]] | None,
    launch: str,
    vector_axis: int,
) -> None:
    """The plan orders whole strategies, and the grid strategy numbers its
    own threaded blocks in loop order, so with threads on both grid dims the
    channel vectors come first only when the loop order reaches ``tile_c``
    first; otherwise the default layout stays.  The V-loop sinks either way,
    and no layout is regenerated."""
    x = torch.empty(8, 784, 120, dtype=torch.bfloat16)
    bound = cpu_bind(_spatial_mean, (x,))
    config = _strided_reduction_config(
        bound,
        sink=True,
        block_sizes=(2, 64),
        grid_threads=(2, 8),
        loop_orders=loop_orders,
    )
    with _regenerations() as signals:
        code = bound.to_code(config)
    assert signals == []
    assert launch in code, code
    assert _thread_axis_of(code, "lane_base_1") == vector_axis, code
    assert "_vsink_vec" in code, code


def test_a_persistent_strided_reduction_keeps_the_default_layout(
    cpu_only: None,
) -> None:
    """Only a rolled reduction takes the tile-first layout: a persistent one
    (64 rows on 16 threads) keeps its threads on thread_idx[0] and its warp
    shuffles rather than a shared-memory combine across the channel
    threads."""
    x = torch.empty(8, 64, 120, dtype=torch.bfloat16)
    bound = cpu_bind(_spatial_mean, (x,))
    code = bound.to_code(_strided_reduction_config(bound, sink=True, rows=None))
    assert "block=(16, 8, 1)" in code, code
    assert _thread_axis_of(code, "lane_base_1") == 1, code
    assert TWO_STAGE_REDUCE not in code
    assert FRAGMENT_REDUCE not in code


def test_a_tile_store_inside_the_sunk_reduction_flushes_as_one_packet(
    cpu_only: None,
) -> None:
    """The ReLU output of each row is stored inside the reduction loop; its
    eight lanes leave as one 16-byte store after the sunk V-loop instead of
    eight 2-byte stores."""
    x = torch.empty(8, 784, 120, dtype=torch.bfloat16)
    shift = torch.empty(120, dtype=torch.float32)
    bound = cpu_bind(_relu_and_spatial_mean, (x, shift))
    code = bound.to_code(_strided_reduction_config(bound, sink=True))
    assert "_cute_store_u16_vec(act.iterator" in code, code
    assert not re.search(r"\(act\.iterator[^\n]*\)\.store\(", code), code


def test_a_later_masked_store_of_the_same_elements_keeps_both_stores_in_order(
    cpu_only: None,
) -> None:
    """The even channels are zeroed by a masked store after the ReLU store.
    Its per-lane mask cannot predicate a packet, and flushing the ReLU store
    as one after the V-loop would overwrite the zeros the masked store wrote
    inside it: both stay per lane, in order."""
    x = torch.empty(8, 784, 120, dtype=torch.bfloat16)
    shift = torch.empty(120, dtype=torch.float32)
    bound = cpu_bind(_relu_then_zero_even_channels, (x, shift))
    code = bound.to_code(_strided_reduction_config(bound, sink=True))
    assert "_vsink_vec" in code, code
    assert "_cute_store_u16_vec(act.iterator" not in code, code
    assert len(re.findall(r"\(act\.iterator[^\n]*\)\.store\(", code)) == 2, code


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _store_repeating_the_vector_index(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``out[c, r, c]`` addresses dim 0 with the vectorized channel index too,
    beside a channel sum over the rows."""
    rows, channels = x.shape
    out = torch.empty([channels, rows, channels], dtype=x.dtype, device=x.device)
    pooled = torch.empty([channels], dtype=x.dtype, device=x.device)
    for c in hl.tile(channels):
        acc = hl.full([c], 0, dtype=torch.float32)
        for r in hl.tile(rows):
            y = x[r, c]
            out[c, r, c] = y[None, :, :]
            acc += y.float().sum(0)
        pooled[c] = acc.to(x.dtype)
    return out, pooled


def test_a_store_addressing_another_dim_by_the_vector_index_stays_per_lane(
    cpu_only: None,
) -> None:
    """Consecutive lanes of ``out[c, r, c]`` lie ``out.stride(0) + 1`` elements
    apart: under the sink the store keeps its per-lane form, as with the knob
    off, rather than one packet along the last dim.  ``check_repeated_block_ids``
    refuses the store first (per lane it reaches only the tile's diagonal), so
    the test turns that refusal off to reach the sink's own check."""
    x = torch.empty(784, 64, dtype=torch.bfloat16)
    with patch(
        "helion._compiler.cute.repeated_block_ids.check_repeated_block_ids",
        return_value=None,
    ):
        code = _cpu_code(
            _store_repeating_the_vector_index,
            (x,),
            block_sizes=[64, 32],
            num_threads=[8, 16],
            cute_vector_widths=[8, 1],
            cute_vloop_sink=True,
        )
    assert "_vsink" in code, code
    assert "_cute_store_u16_vec(out.iterator" not in code, code
    assert re.search(r"\(out\.iterator[^\n]*\)\.store\(", code), code


@contextmanager
def _regenerations() -> Iterator[list[str]]:
    """The signals ``CuteRegeneration`` repairs while the block runs."""
    signals: list[str] = []
    repair = kernel_generators.CuteRegeneration.repair

    def recording(
        self: kernel_generators.CuteRegeneration,
        signal: BaseException,
        config: helion.Config,
        *,
        caller_graphs: bool,
    ) -> helion.Config | None:
        signals.append(type(signal).__name__)
        return repair(self, signal, config, caller_graphs=caller_graphs)

    with patch.object(kernel_generators.CuteRegeneration, "repair", recording):
        yield signals


def test_a_strided_argmax_regenerates_with_the_default_layout(cpu_only: None) -> None:
    """The rolled argmax refuses the tile-first layout, so the kernel is
    generated again with the default one, where nothing sinks: the knob-on
    code is the knob-off code."""
    x = torch.empty(8, 784, 120, dtype=torch.bfloat16)
    bound = cpu_bind(_spatial_argmax, (x,))
    with _regenerations() as signals:
        code = bound.to_code(_strided_reduction_config(bound, sink=True))
    assert signals == ["VloopSinkLayoutRefused"]
    assert code == bound.to_code(_strided_reduction_config(bound, sink=False))


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _spatial_argmax_then_copy(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """``_spatial_argmax``, then an ``hl.barrier()`` and a copy: ordered
    phases, generated as materialized fission stages."""
    batch, _, channels = x.shape
    out = torch.empty([batch, channels], dtype=torch.int64, device=x.device)
    copy = torch.empty_like(y)
    for tile_n, tile_c in hl.tile([batch, channels]):
        out[tile_n, tile_c] = torch.argmax(x[tile_n, :, tile_c], dim=1)
    hl.barrier()
    for tile in hl.tile(y.size(0)):
        copy[tile] = y[tile]
    return out, copy


def test_an_ordered_phase_argmax_regenerates_with_the_default_layout(
    cpu_only: None,
) -> None:
    """Each phase is a stage generated from graphs materialized fission
    supplies, which carry the layout plan: the stage's refusal of the
    tile-first layout reaches the fission driver, which rebuilds the stage's
    graphs with the default layout."""
    x = torch.empty(8, 784, 120, dtype=torch.bfloat16)
    y = torch.empty(1024, dtype=torch.bfloat16)
    bound = cpu_bind(_spatial_argmax_then_copy, (x, y))
    spec = bound.config_spec

    def config(sink: bool) -> helion.Config:
        (reduction,) = spec.reduction_loops.valid_block_ids()
        threads = {0: 1, 1: 8, reduction: 16}
        widths = {1: 8}
        return spec.normalized_config(
            helion.Config(
                block_sizes=[1, 64, 256],
                num_threads=[
                    threads.get(b, 0) for b in spec.num_threads.valid_block_ids()
                ],
                cute_vector_widths=[
                    widths.get(b, 1) for b in spec.cute_vector_widths.valid_block_ids()
                ],
                reduction_loops=[32],
                pid_type="flat",
                cute_vloop_sink=sink,
            )
        )

    with _regenerations() as signals:
        code = bound.to_code(config(True))
    assert signals == ["VloopSinkLayoutRefused"]
    assert code == bound.to_code(config(False))


def test_a_refused_sibling_lane_combine_regenerates_with_the_default_layout(
    cpu_only: None,
) -> None:
    """Under the tile-first layout the rolled sum combines its lanes across
    the vector tile's interleaved threads and refuses where their group is
    not provable under the launch (one wider on x than the tile, as other
    sections sharing it make).  The refusal regenerates the kernel with the
    default layout, like the argmax's, instead of failing it; the planned
    launch is widened on x here to stand in for such sections."""
    x = torch.empty(8, 784, 120, dtype=torch.bfloat16)
    bound = cpu_bind(_spatial_mean, (x,))
    planned = LoopedReductionStrategy._planned_thread_dims

    def wider_on_x(self: LoopedReductionStrategy) -> tuple[int, int, int]:
        dims = planned(self)
        return (dims[0] * 2, dims[1], dims[2])

    with (
        patch.object(LoopedReductionStrategy, "_planned_thread_dims", wider_on_x),
        _regenerations() as signals,
    ):
        code = bound.to_code(_strided_reduction_config(bound, sink=True))
    assert signals == ["VloopSinkLayoutRefused"]
    # The default layout: sixteen reduction threads on x, the vectors on y.
    assert "block=(16, 8, 1)" in code, code
    assert "_vsink_vec" in code, code


def test_a_capacity_decline_passes_the_layout_refusal_through(cpu_only: None) -> None:
    """Once the computed-fragment owner declines a root for its shared
    memory, a refusal of the generic lowering becomes the config's
    InvalidConfig, but the planned layout's refusal is a regeneration signal
    and passes through: the kernel is generated again with the default
    layout."""
    bound = cpu_bind(_spatial_mean, (torch.empty(8, 784, 120, dtype=torch.bfloat16),))
    root_plans = SimpleNamespace(fragment_capacity_decline=(300000, 232448))
    codegen = cast(
        "GenerateAST",
        SimpleNamespace(
            device_function=SimpleNamespace(
                cute_state=SimpleNamespace(root_plans=root_plans)
            )
        ),
    )
    explain = bound.env.backend.explain_root_failure
    refusal = kernel_generators.VloopSinkLayoutRefused("cute", "rolled argmax")
    assert explain(codegen, refusal) is refusal
    other = helion.exc.BackendUnsupported("cute", "a lane restore")
    assert isinstance(explain(codegen, other), helion.exc.InvalidConfig)


def test_a_strided_reduction_past_the_z_launch_limit_keeps_the_default_layout(
    cpu_only: None,
) -> None:
    """Two threads on each grid axis and 128 reduction threads: the
    tile-first order would launch ``(2, 2, 128)``, past the 64 threads the z
    axis allows, so the default ``(128, 2, 2)`` stays and the V-loop still
    sinks."""
    x = torch.empty(8, 784, 120, dtype=torch.bfloat16)
    bound = cpu_bind(_spatial_mean, (x,))
    config = _strided_reduction_config(
        bound,
        sink=True,
        block_sizes=(2, 16),
        grid_threads=(2, 2),
        reduction_threads=128,
        rows=256,
    )
    with _regenerations() as signals:
        code = bound.to_code(config)
    assert signals == []
    assert "block=(128, 2, 2)" in code, code
    assert _thread_axis_of(code, r"rindex_\d") == 0, code
    assert "_vsink_vec" in code, code


@pytest.mark.parametrize(
    ("unroll", "retries"),
    [
        (1, ["VloopSinkNotApplied"]),
        # The knob-off retry keeps the unroll, which no lane loop of the
        # default layout takes either: that retry turns it off.
        (8, ["VloopSinkNotApplied", "LaneUnrollNotApplied"]),
    ],
)
def test_a_strided_reduction_that_sinks_nothing_regenerates_the_knob_off_code(
    cpu_only: None, unroll: int, retries: list[str]
) -> None:
    """248-byte rows put every other row's chunks off the 16-byte packet
    alignment: no load fact is recorded and nothing sinks, so the layout the
    knob shaped goes with the knob and the knob-on code is the knob-off
    code at the same ``cute_lane_unroll``."""
    x = torch.empty(8, 784, 124, dtype=torch.bfloat16)[..., :120]
    bound = cpu_bind(_spatial_mean, (x,))

    def config(sink: bool) -> helion.Config:
        config = _strided_reduction_config(bound, sink=sink)
        config.config["cute_lane_unroll"] = unroll
        return bound.config_spec.normalized_config(config)

    with _regenerations() as signals:
        code = bound.to_code(config(True))
    assert signals == retries
    assert code == bound.to_code(config(False))


_OUT_POINTER = (
    "out.iterator + cutlass.Int32(indices_1) * cutlass.Int32(out.layout.stride[0])"
)


def _packs_the_store(segment: list[str], *, disjoint: bool = True) -> bool:
    """Whether the sink flushes the per-lane store into ``out`` that ends a
    V-loop segment (one statement per string) as one packet; ``x`` is proven
    disjoint from ``out`` or not."""
    sinker = sink_vector_loops._Sinker(
        cast("ast.For", ast.parse("for vec_lane_1 in range(4):\n    pass").body[0]),
        device_state.CuteVloopWrapperFact(
            block_id=1,
            vec_width=4,
            lane_var="lane_1",
            vec_lane_var="vec_lane_1",
            base_index_var="lane_base_1",
            index_var="indices_1",
            mask_var=None,
            uniform_vector_mask=True,
        ),
        loads={},
        rename_groups={},
        new_var=lambda name, **_: name,
        lane_unroll=1,
        tensor_dtypes={"out": torch.float32, "x": torch.float32},
        proven_disjoint_tensor_pairs=[frozenset(("out", "x"))] if disjoint else [],
        stores={
            _OUT_POINTER: device_state.CuteVloopStoreFact(
                "vec_lane_1", "out", 4, torch.float32
            )
        },
    )
    sinker.analysis = sink_vector_loops._Analysis(
        varying=set(),
        expanded={},
        fragment_dtypes={},
        rematerialized={},
        definition_counts={},
    )
    statements = [
        ast.parse(text).body[0]
        for text in [*segment, f"({_OUT_POINTER}).store(cutlass.Float32(v))"]
    ]
    return bool(sinker._packable_store_statements(statements, frozenset({"out"}), True))


def test_a_store_read_back_through_a_gather_stays_per_lane() -> None:
    """A gather lowers to a bare subscript (``t[i] if mask else 0``) with no
    load call.  Reading the stored tensor that way in the store's segment
    keeps the store per lane: a later lane must see the earlier lanes'
    stores, which the packet flush would only write after the V-loop.  A
    tensor proven disjoint from it may be gathered."""
    load = "v = (x.iterator + cutlass.Int32(indices_1)).load()"
    assert _packs_the_store([load])
    assert not _packs_the_store([load], disjoint=False)
    assert not _packs_the_store([load, "back = out[indices_1]"])
    assert _packs_the_store(["v = x[indices_1]"])
    assert not _packs_the_store(["v = x[indices_1]"], disjoint=False)


@pytest.mark.parametrize(
    "load",
    [
        # A predicated packet of the sink itself and an L2-policy helper.
        "v = _cute_load_v16b_if(x.iterator + cutlass.Int32(indices_1), mask_1)",
        "v = _cute_load_l1_l2_evict_first(x.iterator + cutlass.Int32(indices_1))",
    ],
    ids=["packet", "policy_helper"],
)
def test_a_store_beside_an_unresolved_load_stays_per_lane(load: str) -> None:
    """The segment analysis does not resolve the tensor of a packet or
    helper load, so a segment holding one packs no store."""
    assert not _packs_the_store([load])


def test_strided_reduction_heuristic_seeds_the_tile_first_layout(
    cpu_only: None,
) -> None:
    """The reduction over the strided middle dim gets coalesced seeds: the
    120 channels on 16 vector threads (128 columns), 8 reduction threads over
    128-row chunks and an 8-deep lane unroll, and the seeded layout sinks."""
    x = torch.empty(8, 784, 120, dtype=torch.bfloat16)
    shift = torch.empty(120, dtype=torch.float32)
    bound = cpu_bind(_relu_and_spatial_mean, (x, shift))
    spec = bound.config_spec
    assert "cute_strided_reduction" in spec.autotuner_heuristics
    seeds = [
        seed.config
        for seed in spec.compiler_seed_configs
        if seed.config.get("cute_vloop_sink")
    ]
    assert seeds
    (reduction,) = spec.reduction_loops.valid_block_ids()
    threads = dict(
        zip(spec.num_threads.valid_block_ids(), seeds[0]["num_threads"], strict=True)
    )
    widths = dict(
        zip(
            spec.cute_vector_widths.valid_block_ids(),
            seeds[0]["cute_vector_widths"],
            strict=True,
        )
    )
    assert seeds[0]["block_sizes"] == [1, 128]
    assert threads == {0: 1, 1: 16, reduction: 8}
    assert widths[1] == 8
    assert seeds[0]["reduction_loops"] == [128]
    assert seeds[0]["cute_lane_unroll"] == 8
    code = bound.to_code(spec.normalized_config(helion.Config(**seeds[0])))
    assert _thread_axis_of(code, "lane_base_1") == 0, code
    assert "_cute_store_u16_vec(act.iterator" in code
    assert re.search(r"_vsink_store_vals\w*\.append\(", code)


def test_strided_reduction_heuristic_leaves_short_reductions_to_the_register_tile(
    cpu_only: None,
) -> None:
    """512 rows fit the register tile's persistent layout, whose seeds beat
    the strided ones on such column sums; 2048 rows do not."""
    for rows, seeded, skipped in (
        (512, "cute_register_tile", "cute_strided_reduction"),
        (2048, "cute_strided_reduction", "cute_register_tile"),
    ):
        x = torch.empty(rows, 1024, dtype=torch.bfloat16)
        heuristics = cpu_bind(_column_sum, (x,)).config_spec.autotuner_heuristics
        assert seeded in heuristics and skipped not in heuristics, (rows, heuristics)


def _sunk_seeds(spec: ConfigSpec, *fields: str) -> list[tuple[Any, ...]]:
    """``fields`` of each compiler seed of ``spec`` that sinks its V-loop
    (the strided-reduction seeds), ``reduction_threads`` naming the
    ``num_threads`` entry of the rolled reduction."""
    (reduction,) = spec.reduction_loops.valid_block_ids()

    def value(seed: helion.Config, field: str) -> object:
        if field == "reduction_threads":
            return spec.num_threads.config_get(seed.config["num_threads"], reduction)
        return seed.config[field]

    return [
        tuple(value(seed, field) for field in fields)
        for seed in spec.compiler_seed_configs
        if seed.config.get("cute_vloop_sink")
    ]


def test_strided_reduction_heuristic_seeds_narrow_tiles_for_a_small_grid(
    cpu_only: None,
) -> None:
    """128 images of 144 channels: the 128-column seeds launch 256 CTAs that
    each walk 3136 rows, so a 16-column seed (1152 CTAs) with 64 reduction
    threads joins them."""
    x = torch.empty(128, 3136, 144, dtype=torch.bfloat16)
    shift = torch.empty(144, dtype=torch.float32)
    bound = cpu_bind(_relu_and_spatial_mean, (x, shift))
    spec = bound.config_spec
    seeds = _sunk_seeds(spec, "block_sizes", "reduction_threads", "reduction_loops")
    assert ([1, 16], 64, [512]) in seeds, seeds
    assert ([1, 128], 8, [128]) in seeds, seeds


@pytest.mark.parametrize(
    ("rows", "threads", "chunk"),
    [(2048, 128, 1024), (3000, 256, 2048), (4096, 256, 2048), (36000, 512, 4096)],
)
def test_strided_reduction_heuristic_widens_the_narrow_seed_on_a_tiny_grid(
    cpu_only: None, rows: int, threads: int, chunk: int
) -> None:
    """512 fp32 columns over a lone grid axis: even one 4-wide vector per CTA
    launches only 128 CTAs, so the narrow seed takes up to 512 reduction
    threads (the thread count of the 4 CTAs per SM it cannot launch), as many
    as keep its chunk of 8 rows a thread shorter than the reduction."""
    x = torch.empty(rows, 512, dtype=torch.float32)
    bound = cpu_bind(_column_sum, (x,))
    spec = bound.config_spec
    seeds = _sunk_seeds(spec, "block_sizes", "reduction_threads", "reduction_loops")
    assert ([4], threads, [chunk]) in seeds, seeds


def test_strided_reduction_heuristic_seeds_a_short_reduction_in_one_chunk(
    cpu_only: None,
) -> None:
    """129 to 256 rows fit one 256-row chunk: beside the 128-row chunks, a
    seed gives each of 16 reduction threads 16 rows in one pass with the
    deepest lane unroll.  It is the only seed whose chunk covers the rows:
    another seed whose chunk would reach a short reduction's rows is dropped,
    not clamped into an unmeasured single pass, so 40 and 64 rows get no seed
    and 100 and 128 rows only the 64-row chunks; 3136 rows get no one-chunk
    seed."""
    for rows in (40, 64, 100, 128, 196, 256, 3136):
        expected = 128 < rows <= 256
        x = torch.empty(8, rows, 120, dtype=torch.bfloat16)
        shift = torch.empty(120, dtype=torch.float32)
        spec = cpu_bind(_relu_and_spatial_mean, (x, shift)).config_spec
        seeds = _sunk_seeds(
            spec, "reduction_loops", "cute_lane_unroll", "reduction_threads"
        )
        covering = [seed for seed in seeds if seed[0][0] >= rows]
        assert covering == ([([256], 16, 16)] if expected else []), (rows, seeds)
        if rows <= 64:
            assert seeds == [], (rows, seeds)
        elif rows <= 128:
            assert [seed[:2] for seed in seeds] == [([64], 8)], (rows, seeds)
        assert (([256], 16, 16) in seeds) is expected, (rows, seeds)


@pytest.mark.parametrize(
    ("width", "vec", "columns", "tile_threads", "packet_load"),
    (
        # Rows of 1026 bf16 elements are only 4-byte aligned (the BERT
        # vocabulary gradient's 30522 too): 2-element vectors, four times
        # the tile threads of a 16-byte vector, the wider tile first.
        (1026, 2, 256, 128, "_cute_load_v4b_if("),
        # 8-byte aligned rows of 1028: 4-element vectors, twice the tile
        # threads of a 16-byte vector, the wider tile first.
        (1028, 4, 256, 64, "_cute_load_v8b_if("),
        # 16-byte aligned rows: the measured shape.
        (1032, 8, 128, 16, "_cute_load_v16b_if("),
    ),
)
def test_strided_reduction_heuristic_seeds_the_widest_aligned_vector(
    cpu_only: None,
    width: int,
    vec: int,
    columns: int,
    tile_threads: int,
    packet_load: str,
) -> None:
    """A column sum over a lone grid axis: the first seed's tile leaves a
    jagged last tile, whose bounds mask on the vector fact's index the sink
    evaluates at the vector's base, and the seeded config sinks predicated
    packets with the vector threads on ``thread_idx[0]``."""
    x = torch.empty(2048, width, dtype=torch.bfloat16)
    bound = cpu_bind(_column_sum, (x,))
    spec = bound.config_spec
    seeds = [
        seed.config
        for seed in spec.compiler_seed_configs
        if seed.config.get("cute_vloop_sink")
    ]
    assert seeds
    (reduction,) = spec.reduction_loops.valid_block_ids()
    (tile,) = spec.block_sizes.valid_block_ids()
    threads = dict(
        zip(spec.num_threads.valid_block_ids(), seeds[0]["num_threads"], strict=True)
    )
    widths = dict(
        zip(
            spec.cute_vector_widths.valid_block_ids(),
            seeds[0]["cute_vector_widths"],
            strict=True,
        )
    )
    assert seeds[0]["block_sizes"] == [columns]
    assert threads == {tile: tile_threads, reduction: 8}
    assert widths[tile] == vec
    code = bound.to_code(spec.normalized_config(helion.Config(**seeds[0])))
    assert code.count(packet_load) == 8, code
    assert _thread_axis_of(code, "offsets_base_0") == 0, code
    assert _thread_axis_of(code, "_vsink_ptr") == 1, code


def test_strided_reduction_heuristic_skips_rows_without_a_vector(
    cpu_only: None,
) -> None:
    """Rows of an odd number of bf16 elements align no 2-element vector."""
    bound = cpu_bind(_column_sum, (torch.empty(2048, 1025, dtype=torch.bfloat16),))
    assert "cute_strided_reduction" not in bound.config_spec.autotuner_heuristics


def test_strided_reduction_heuristic_skips_a_contiguous_row(cpu_only: None) -> None:
    """A row reduction over the stride-1 dim is already coalesced on the
    reduction threads: no strided-reduction seed."""
    bound = cpu_bind(
        rms_norm_fwd,
        (
            torch.empty(256, 4096, dtype=torch.bfloat16),
            torch.empty(4096, dtype=torch.bfloat16),
        ),
    )
    assert "cute_strided_reduction" not in bound.config_spec.autotuner_heuristics


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _channel_affine_relu(
    x: torch.Tensor,
    mean: torch.Tensor,
    var: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """A BatchNorm-inference epilogue on a channels-last ``[rows, C]`` view:
    every element reads its channel's statistics and affine parameters."""
    out = torch.empty_like(x)
    for tile_r, tile_c in hl.tile(x.size()):
        y = x[tile_r, tile_c].to(torch.float32) - mean[tile_c].to(torch.float32)
        y = y * torch.rsqrt(var[tile_c].to(torch.float32) + 1e-05)
        y = y * weight[tile_c].to(torch.float32) + bias[tile_c].to(torch.float32)
        out[tile_r, tile_c] = torch.relu(y).to(out.dtype)
    return out


def _channel_affine_args(
    device: str = "cpu", rows: int = 512
) -> tuple[torch.Tensor, ...]:
    x = torch.randn(rows, 384, dtype=torch.bfloat16, device=device)
    mean, weight, bias = (
        torch.randn(384, dtype=torch.bfloat16, device=device) for _ in range(3)
    )
    var = torch.rand(384, dtype=torch.bfloat16, device=device) + 0.5
    return x, mean, var, weight, bias


def _channel_affine_config(*, sink: bool, vector: int = 8) -> helion.Config:
    """Four rows per thread on 8 row threads, one ``vector``-channel vector
    per thread on 4 channel threads."""
    return helion.Config(
        block_sizes=[32, 32],
        num_threads=[8, 4],
        loop_orders=[[1, 0]],
        flatten_loops=[False],
        cute_vector_widths=[1, vector],
        cute_lane_layouts=["blocked", "strided"],
        cute_vloop_sink=sink,
    )


@functools.cache
def _channel_affine_bound(rows: int) -> BoundKernel:
    return cpu_bind(_channel_affine_relu, _channel_affine_args(rows=rows))


@functools.cache
def _channel_affine_render(rows: int, config: str) -> str:
    bound = _channel_affine_bound(rows)
    return bound.to_code(
        bound.config_spec.normalized_config(helion.Config(**json.loads(config)))
    )


def _channel_affine_code(rows: int, config: dict[str, Any]) -> str:
    """``_channel_affine_relu``'s code for ``config`` on ``rows`` rows,
    rendered once per module: several tests read the same configs' code."""
    return _channel_affine_render(rows, json.dumps(config, sort_keys=True))


def _value_statements(code: str) -> set[str]:
    """The kernel's ``v_N = ...`` statements, each element of a sunk fragment
    read as the per-lane value it holds and unrolled copies merged: two
    renders with equal sets do the same arithmetic, contractions included."""
    statements = set()
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, ast.Assign):
            text = re.sub(r"_frag\[vec_lane_\d+\]", "", ast.unparse(node))
            text = re.sub(r"\b(\w+?)_u\d+\b", r"\1", text)
            if re.match(r"v_\d+ = ", text):
                statements.add(text)
    return statements


def test_channel_parameters_load_once_per_thread_beside_the_row_loop(
    cpu_only: None,
) -> None:
    """Sinking the channel vector into the rows' lane loop computes each
    channel's statistics (and its ``rsqrt``) once per thread into register
    fragments; each of the thread's four rows then loads one 16-byte chunk of
    ``x`` and stores one 16-byte packet."""
    code = _channel_affine_code(512, dict(_channel_affine_config(sink=True).config))
    rows = _loop(code, "lane_0")
    assert "rsqrt" not in ast.unparse(rows), code
    assert code.count("cute.math.rsqrt(") == 1
    assert code.count("ir.VectorType.get([8], cutlass.Uint16.mlir_type)") == 5
    assert "_cute_store_u16_vec(out.iterator" in ast.unparse(rows)
    assert ".load()" not in code


def test_channel_parameters_keep_the_per_element_form_with_the_knob_off(
    cpu_only: None,
) -> None:
    code = _channel_affine_code(512, dict(_channel_affine_config(sink=False).config))
    assert "_vsink" not in code
    assert "rsqrt" in ast.unparse(_loop(code, "vec_lane_1"))


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _add_rows(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile_r, tile_c in hl.tile(x.size()):
        out[tile_r, tile_c] = x[tile_r, tile_c] + y[tile_r, tile_c]
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _gather_rows(idx: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Embedding: each output row gathers one ``weight`` row."""
    out = torch.empty(
        [idx.size(0), weight.size(1)], dtype=weight.dtype, device=weight.device
    )
    for tile_b, tile_e in hl.tile(out.size()):
        out[tile_b, tile_e] = weight[idx[tile_b], tile_e]
    return out


def _hoist_seeds(bound: Any) -> list[dict[str, Any]]:
    return [
        seed.config
        for seed in bound.config_spec.compiler_seed_configs
        if seed.config.get("cute_vloop_sink")
        and not seed.config.get("cute_proven_bounds")
    ]


def test_pointwise_heuristic_seeds_rows_per_thread_for_channel_operands(
    cpu_only: None,
) -> None:
    """Per-channel operands get a seed giving each of 128 threads 8 rows of
    one 8-channel vector at three CTAs per SM, with the vector loop sunk and
    eight rows' loads issued together, plus its proven-bounds copy (a packet
    prefetch twin would render the same); the seeded config computes the
    channel terms once per thread and issues all of a thread's row loads at
    once."""
    bound = _channel_affine_bound(4096)
    seeds = _hoist_seeds(bound)
    assert [
        (seed["block_sizes"], seed["num_threads"], seed.get("cute_min_blocks_per_mp"))
        for seed in seeds
    ] == [([64, 128], [8, 16], 3)]
    (seed,) = seeds
    assert seed["cute_vector_widths"] == [1, 8]
    assert seed["cute_lane_layouts"] == ["blocked", "strided"]
    assert seed["loop_orders"] == [[1, 0]]
    assert seed["cute_lane_unroll"] == 8
    sunk = [
        seed.config
        for seed in bound.config_spec.compiler_seed_configs
        if seed.config.get("cute_vloop_sink")
    ]
    assert len(sunk) == 2
    assert not any(config.get("cute_packet_prefetch") for config in sunk)
    code = _channel_affine_code(4096, seeds[0])
    rows = _loop(code, "lane_0")
    assert code.count("cute.math.rsqrt(") == 1
    assert "rsqrt" not in ast.unparse(rows)
    # One rolled trip: the thread's eight rows load before any is consumed.
    assert ast.unparse(rows.iter) == "range(1)", code
    assert re.search(r"_vsink_vec_\d+_u7", ast.unparse(rows)), code


def test_pointwise_heuristic_skips_the_hoist_without_channel_operands(
    cpu_only: None,
) -> None:
    rows = torch.empty(4096, 384, dtype=torch.bfloat16)
    assert _hoist_seeds(cpu_bind(_add_rows, (rows, rows))) == []


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _bias_add(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile_r, tile_c in hl.tile(x.size()):
        out[tile_r, tile_c] = x[tile_r, tile_c] + bias[tile_c]
    return out


def test_pointwise_heuristic_skips_the_hoist_for_one_packet_rows(
    cpu_only: None,
) -> None:
    """Eight bf16 channels are one 16-byte packet: a thread's row has no
    vector loop to sink, so no seed; sixteen channels are two packets."""
    for channels, expected in ((8, 0), (16, 1)):
        x = torch.empty(4096, channels, dtype=torch.bfloat16)
        bias = torch.empty(channels, dtype=torch.bfloat16)
        assert len(_hoist_seeds(cpu_bind(_bias_add, (x, bias)))) == expected


def test_pointwise_heuristic_skips_the_hoist_for_a_gathered_row(
    cpu_only: None,
) -> None:
    """``weight[idx[tile_b], tile_e]`` is indexed by the contiguous axis and a
    gather, not by the contiguous axis alone: the sinking pass leaves such a
    nest unsunk, so the seeds would only cost search time."""
    idx = torch.zeros(4096, dtype=torch.int32)
    weight = torch.empty(4096, 384, dtype=torch.bfloat16)
    assert _hoist_seeds(cpu_bind(_gather_rows, (idx, weight))) == []


def test_hoist_seed_with_a_ragged_row_tile_masks_each_rows_packets(
    cpu_only: None,
) -> None:
    """500 rows end the seed's 64-row tiles ragged: the channel operands still
    load as one 16-byte packet each per thread before the row loop, and each
    of the thread's eight rows loads ``x`` and stores ``out`` as one 16-byte
    packet under that row's mask, its loads issued with the others'.  (The
    sinking pass packs a row's masked store, ``if mask: store``, as it packs
    an unmasked one.)"""
    seed = _hoist_seeds(_channel_affine_bound(500))[0]
    assert seed["block_sizes"] == [64, 128]
    code = _channel_affine_code(500, seed)
    rows = _loop(code, "lane_0")
    packet = "ir.VectorType.get([8], cutlass.Uint16.mlir_type)"
    assert ast.unparse(rows.iter) == "range(1)", code
    assert code.count(packet) == 12
    assert ast.unparse(rows).count(packet) == 8
    assert "rsqrt" not in ast.unparse(rows)
    stores = [
        node
        for node in ast.walk(rows)
        if isinstance(node, ast.If)
        and any("_cute_store_u16_vec(" in ast.unparse(stmt) for stmt in node.body)
    ]
    assert [ast.unparse(store.test) for store in stores] == [
        f"mask_0_u{row}" for row in range(8)
    ], code


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _channel_scale_shift(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile_r, tile_c in hl.tile(x.size()):
        out[tile_r, tile_c] = (
            torch.sqrt(x[tile_r, tile_c]) * weight[tile_c] + bias[tile_c]
        )
    return out


def test_sunk_float32_channel_operands_contract_like_the_unsunk_kernel(
    cpu_only: None,
) -> None:
    """With float32 operands neither form contracts ``sqrt(x) * weight +
    bias``: a plain fp32 load is not proven fp32, whether a per-lane
    ``.load()`` (knob off) or the packet lane that fills a fragment (knob
    on), so the knob cannot change the rounding here either."""
    args = (torch.empty(512, 384), torch.empty(384), torch.empty(384))
    bound = cpu_bind(_channel_scale_shift, args)
    sunk, unsunk = (
        bound.to_code(
            bound.config_spec.normalized_config(
                _channel_affine_config(sink=sink, vector=4)
            )
        )
        for sink in (True, False)
    )
    assert "_vsink_vec" in sunk
    assert "_vsink" not in unsunk
    assert "cute.math.fma(" not in sunk, sunk
    assert _value_statements(sunk) == _value_statements(unsunk)


@pytest.mark.parametrize(
    ("rows", "config"),
    [(512, "test"), (4096, "seed"), (500, "seed")],
    ids=["test_config", "seed_unrolled_rows", "seed_ragged_rows"],
)
def test_sunk_channel_operands_contract_like_the_unsunk_kernel(
    cpu_only: None, rows: int, config: str
) -> None:
    """``y * weight + bias`` contracts into one FMA whether ``weight`` and
    ``bias`` are per-lane names (knob off) or elements of the fragments the
    sinking pass fills before the row loop (knob on), so the knob cannot
    change the rounding: the 32x32 test config, and the first broadcast-hoist
    seed with its eight rows unrolled and with a ragged row tile."""
    values = (
        dict(_channel_affine_config(sink=True).config)
        if config == "test"
        else _hoist_seeds(_channel_affine_bound(rows))[0]
    )
    sunk, unsunk = (
        _channel_affine_code(rows, values | {"cute_vloop_sink": sink})
        for sink in (True, False)
    )
    assert "_vsink_vec" in sunk
    assert "_vsink" not in unsunk
    assert "cute.math.fma(" in ast.unparse(_loop(sunk, "lane_0")), sunk
    assert _value_statements(sunk) == _value_statements(unsunk)


_FILLED_FRAGMENTS = """
w_frag = cute.make_rmem_tensor(8, cutlass.Float32)
b_frag = cute.make_rmem_tensor(8, cutlass.Float32)
for vec_lane in cutlass.range_constexpr(8):
    w_frag[vec_lane] = cutlass.Float32(w)
    b_frag[vec_lane] = cutlass.Float32(b)
"""


@pytest.mark.parametrize(
    ("source", "contracted"),
    [
        (
            _FILLED_FRAGMENTS
            + """
for row in range(4):
    for vec_lane in cutlass.range_constexpr(8):
        y = cutlass.Float32(x)
        t = y * w_frag[vec_lane]
        out = t + b_frag[vec_lane]
""",
            "out = cute.math.fma(y, w_frag[vec_lane], b_frag[vec_lane])",
        ),
        (
            """
shift_frag = cute.make_rmem_tensor(8, cutlass.Float32)
for vec_lane in cutlass.range_constexpr(8):
    m = cutlass.Float32(mean)
    s = cutlass.Float32(scale)
    b = cutlass.Float32(bias)
    t = m * s
    shift_frag[vec_lane] = b - t
""",
            "shift_frag[vec_lane] = cute.math.fma(-m, s, b)",
        ),
        (
            """
acc_frag = cute.make_rmem_tensor(8, cutlass.Float32)
for vec_lane in cutlass.range_constexpr(8):
    acc_frag[vec_lane] = 0.0
for row in range(4):
    for vec_lane in cutlass.range_constexpr(8):
        y = cutlass.Float32(x)
        t = y * y
        acc_frag[vec_lane] = acc_frag[vec_lane] + t
""",
            None,
        ),
        (
            """
w_frag = cute.make_rmem_tensor(8, cutlass.Float32)
for vec_lane in cutlass.range_constexpr(8):
    y = cutlass.Float32(x)
    t = y * w_frag[vec_lane]
    w_frag[vec_lane] = cutlass.Float32(w)
    out = t + 1.0
""",
            None,
        ),
        (
            _FILLED_FRAGMENTS
            + """
for i in range(4):
    y = cutlass.Float32(x)
    t = y * w_frag[i]
    for i in range(2):
        pass
    out = t + b_frag[i]
""",
            None,
        ),
        (
            """
w_frag = cute.make_rmem_tensor(8, cutlass.Float32)
w_frag.fill(2.0)
for vec_lane in cutlass.range_constexpr(8):
    y = cutlass.Float32(x)
    t = y * w_frag[vec_lane]
    out = t + 1.0
""",
            None,
        ),
        (
            _FILLED_FRAGMENTS
            + """
for vec_lane in cutlass.range_constexpr(8):
    y = cutlass.Float32(x)
    t = y * w_frag[vec_lane]
    cute.autovec_copy(src, w_frag[None])
    out = t + 1.0
""",
            None,
        ),
        (
            _FILLED_FRAGMENTS
            + """
view_index = None
for vec_lane in cutlass.range_constexpr(8):
    y = cutlass.Float32(x)
    t = y * w_frag[vec_lane]
    cute.autovec_copy(src, w_frag[view_index])
    out = t + 1.0
""",
            None,
        ),
        (
            """
w_frag = cute.make_rmem_tensor(8, cutlass.BFloat16)
for vec_lane in cutlass.range_constexpr(8):
    w_frag[vec_lane] = cutlass.BFloat16(w)
for vec_lane in cutlass.range_constexpr(8):
    y = cutlass.Float32(x)
    t = y * w_frag[vec_lane]
    out = t + 1.0
""",
            None,
        ),
        (
            """
w_frag = cute.make_rmem_tensor(8, cutlass.Float32)
w_vec = cute.arch.load(w_ptr, ir.VectorType.get([8], cutlass.Uint32.mlir_type))
for vec_lane in cutlass.range_constexpr(8):
    w_frag[vec_lane] = cutlass.Uint32(w_vec[vec_lane]).bitcast(cutlass.Float32)
for vec_lane in cutlass.range_constexpr(8):
    y = cutlass.Float32(x)
    t = y * w_frag[vec_lane]
    out = t + 1.0
""",
            None,
        ),
    ],
    ids=[
        "filled_before_the_row_loop",
        "fragment_element_consumer",
        "accumulator",
        "stored_before_the_consumer",
        "index_rebound_before_the_consumer",
        "escaped",
        "escaped_through_a_view",
        "escaped_through_a_named_index",
        "bf16_fragment",
        "filled_with_a_value_not_proven_fp32",
    ],
)
def test_fma_contraction_reads_single_store_fragments_like_names(
    source: str, contracted: str | None
) -> None:
    """An element of an fp32 fragment that one statement fills with a value
    proven fp32, used only through element subscripts (an int literal or a
    ``range`` loop's variable), contracts like the single-assignment name the
    sinking pass moved into it; an accumulator, a store or index change
    between the product and its consumer, a fragment that escapes (also
    through a view, ``f[None]``, or an index that may hold one), a non-fp32
    one or one filled with a packet lane's bitcast (a plain load is not
    proven fp32 either) do not.  Generated loops do not always mark their
    target as a store, so every case also runs with each loop target a
    load."""
    for load_targets in (False, True):
        body = ast.parse(source).body
        if load_targets:
            for node in ast.walk(ast.Module(body=body, type_ignores=[])):
                if isinstance(node, ast.For):
                    node.target.ctx = ast.Load()
        text = ast.unparse(ast.Module(body=fuse_fma(body), type_ignores=[]))
        if contracted is None:
            assert "cute.math.fma" not in text, text
        else:
            assert contracted in text, text
            assert text.count("cute.math.fma") == 1


def test_kernels_without_the_pattern_generate_identical_code(cpu_only: None) -> None:
    def example(fn: Callable[..., Any]) -> helion.Kernel:
        return helion.kernel(
            fn,
            backend="cute",
            static_shapes=True,
            autotune_effort="none",
            ignore_warnings=[helion.exc.TensorOperationInWrapper],
        )

    def both(
        kernel: helion.Kernel, args: tuple[torch.Tensor, ...], **config: object
    ) -> None:
        on = _cpu_code(kernel, args, cute_vloop_sink=True, **config)
        off = _cpu_code(kernel, args, cute_vloop_sink=False, **config)
        assert on == off
        assert "_vsink" not in on

    def row_config(
        kernel: helion.Kernel, args: tuple[torch.Tensor, ...]
    ) -> dict[str, object]:
        """One 1024-wide row per CTA on 128 threads with V=8 (the persistent
        one-vector-per-thread layout of a row reduction)."""
        bound = cpu_bind(kernel, args)
        spec = bound.config_spec
        (reduction,) = [
            block.block_id for block in bound.env.block_sizes if block.reduction
        ]
        return {
            "block_sizes": [1 for _ in spec.block_sizes.valid_block_ids()],
            "num_threads": [
                128 if block_id == reduction else 0
                for block_id in spec.num_threads.valid_block_ids()
            ],
            "cute_vector_widths": [
                8 if block_id == reduction else 1
                for block_id in spec.cute_vector_widths.valid_block_ids()
            ],
        }

    rows = torch.empty(256, 1024, dtype=torch.bfloat16)
    weight = torch.empty(1024, dtype=torch.bfloat16)
    rms_norm = example(rms_norm_fwd.fn)
    both(rms_norm, (rows, weight), **row_config(rms_norm, (rows, weight)))
    softmax_kernel = example(softmax.fn)
    both(softmax_kernel, (rows,), **row_config(softmax_kernel, (rows,)))
    x = torch.empty(4096, 4096, dtype=torch.bfloat16)
    # A scalar grid axis has no vector loop to sink.
    both(
        col_reduce_sum_static,
        (x,),
        block_sizes=[4096, 4],
        num_threads=[32, 4],
        cute_vector_widths=[1, 1],
        cute_lane_layouts=["strided", "blocked"],
    )
    # A vector along the reduced (device-loop) axis is not a grid wrapper.
    both(
        col_reduce_sum_static,
        (x,),
        block_sizes=[4096, 4],
        num_threads=[32, 4],
        cute_vector_widths=[8, 1],
        cute_lane_layouts=["blocked", "blocked"],
    )


def test_lane_unroll_requires_sinking_or_a_grid_lane_loop(cpu_only: None) -> None:
    x = torch.empty(4096, 4096, dtype=torch.bfloat16)
    bound = cpu_bind(col_reduce_sum_static, (x,))

    def normalized_unroll(block_sizes: list[int], num_threads: list[int]) -> int:
        config = bound.config_spec.normalized_config(
            helion.Config(
                block_sizes=block_sizes,
                num_threads=num_threads,
                cute_vector_widths=[1, 4],
                cute_vloop_sink=False,
                cute_lane_unroll=8,
            )
        )
        return config.config["cute_lane_unroll"]

    # Without sinking the unroll belongs to the loads-first lane unroll
    # (``cute/unroll_lane_loads.py``), which needs a lane loop on some tile
    # axis, of the grid or of a device loop: four threads over 16 columns
    # form one, as do 128 threads over 4096 rows; one thread per element on
    # every axis does not.
    assert normalized_unroll([4096, 16], [128, 4]) == 8
    assert normalized_unroll([4096, 16], [128, 16]) == 8
    assert normalized_unroll([128, 16], [128, 16]) == 1
    with pytest.raises(helion.exc.InvalidConfig):
        bound.config_spec.normalize(
            helion.Config(
                block_sizes=[4096, 16],
                num_threads=[128, 4],
                cute_vector_widths=[1, 4],
                cute_vloop_sink=True,
                cute_lane_unroll=3,
            )
        )


def test_column_reduction_heuristic_seeds_the_coalesced_layout(
    cpu_only: None,
) -> None:
    x = torch.empty(4096, 4096, dtype=torch.bfloat16)
    bound = cpu_bind(col_reduce_sum_dynamic, (x,))
    spec = bound.config_spec
    assert "cute_column_reduction" in spec.autotuner_heuristics
    seeds = [
        seed
        for seed in spec.compiler_seed_configs
        if seed.config.get("cute_vloop_sink")
    ]
    assert seeds
    primary = seeds[0].config
    assert primary["block_sizes"] == [4096, 16]
    assert primary["num_threads"] == [256, 2]
    assert primary["cute_vector_widths"] == [1, 8]
    assert primary["cute_lane_layouts"] == ["strided", "blocked"]
    assert primary["cute_lane_unroll"] == 8
    assert {tuple(seed.config["num_threads"]) for seed in seeds} == {
        (256, 2),
        (128, 4),
        (128, 2),
    }
    # The half-width vector seed: four grid threads owning 8-byte vectors.
    assert any(
        seed.config["block_sizes"] == [4096, 16]
        and seed.config["num_threads"] == [128, 4]
        and seed.config["cute_vector_widths"] == [1, 4]
        for seed in seeds
    )
    # Every coalesced seed overlaps its launch with the output's zero fill.
    assert all(seed.config["cute_pdl"] is True for seed in seeds)
    # The seeded knobs are searchable and normalize unchanged.
    normalized = spec.normalized_config(seeds[0])
    assert normalized.config["cute_vloop_sink"] is True
    assert normalized.config["cute_lane_unroll"] == 8
    assert normalized.config["cute_pdl"] is True
    # The seeded layout actually sinks.
    assert FRAGMENT_REDUCE in bound.to_code(normalized)
    # Dynamic extents are proven through the exact input metadata; static
    # ones need no such specialization.
    assert CuteColumnReductionHeuristic.register_facts(
        bound.env, bound.host_function.device_ir
    ) == {"input_tensor_metadata"}
    static_bound = cpu_bind(col_reduce_sum_static, (x,))
    assert "cute_column_reduction" in static_bound.config_spec.autotuner_heuristics
    assert (
        CuteColumnReductionHeuristic.register_facts(
            static_bound.env, static_bound.host_function.device_ir
        )
        == frozenset()
    )
    # The seed only enables sinking where the pass can fire: a zero-origin
    # grid tile whose extent is a multiple of the vector width.
    for kernel, shape in (
        (col_reduce_sum_dynamic, (4000, 4002)),
        (col_reduce_sum_from8_static, (1000, 1008)),
        (col_reduce_sum_from8_dynamic, (1000, 1008)),
    ):
        declined = cpu_bind(kernel, (torch.empty(*shape, dtype=torch.bfloat16),))
        assert "cute_column_reduction" not in declined.config_spec.autotuner_heuristics
        assert not any(
            seed.config.get("cute_vloop_sink")
            for seed in declined.config_spec.compiler_seed_configs
        )
    # What no metadata could change (a nonzero grid origin, a grid dim that
    # is not contiguous) registers no exact-metadata specialization, so such
    # dynamic kernels keep one binding across shapes.
    for kernel, x in (
        (col_reduce_sum_from8_dynamic, torch.empty(1000, 1008, dtype=torch.bfloat16)),
        (col_reduce_sum_dynamic, torch.empty(4096, 4096, dtype=torch.bfloat16).t()),
    ):
        declined = cpu_bind(kernel, (x,))
        assert (
            CuteColumnReductionHeuristic.register_facts(
                declined.env, declined.host_function.device_ir
            )
            == frozenset()
        )
    # Row reductions are not column reductions.
    rows_bound = cpu_bind(
        helion.kernel(
            rms_norm_fwd.fn,
            backend="cute",
            static_shapes=True,
            autotune_effort="none",
            ignore_warnings=[helion.exc.TensorOperationInWrapper],
        ),
        (
            torch.empty(256, 1024, dtype=torch.bfloat16),
            torch.empty(1024, dtype=torch.bfloat16),
        ),
    )
    assert "cute_column_reduction" not in rows_bound.config_spec.autotuner_heuristics


def _split_column_sum_fn(x: torch.Tensor, rows_per_split: hl.constexpr) -> torch.Tensor:
    """Column sums of ``x``'s blocks of ``rows_per_split`` rows, the last one
    partial: the serial loop walks a split's rows with an explicit tile and
    masked loads, as Inductor's looped split reductions do."""
    rows, n = x.size()
    splits = (rows + rows_per_split - 1) // rows_per_split
    out = torch.empty([splits, n], dtype=torch.float32, device=x.device)
    for tile_s, tile_n in hl.tile([splits, n]):
        acc = hl.zeros([tile_s, tile_n], dtype=torch.float32)
        for tile_r in hl.tile(rows_per_split):
            row = tile_s.index[:, None] * rows_per_split + tile_r.index[None, :]
            valid = (row < rows) & (tile_r.index[None, :] < rows_per_split)
            vals = hl.load(x, [row, tile_n], extra_mask=valid[:, :, None])
            acc = acc + torch.sum(vals.to(torch.float32), dim=1)
        out[tile_s, tile_n] = acc
    return out


split_column_sum_static = helion.kernel(
    _split_column_sum_fn, backend="cute", static_shapes=True, autotune_effort="none"
)
split_column_sum_dynamic = helion.kernel(
    _split_column_sum_fn, backend="cute", static_shapes=False, autotune_effort="none"
)


def test_split_column_reduction_heuristic_seeds_the_coalesced_layout(
    cpu_only: None,
) -> None:
    """Splits of 600 rows of 64 contiguous bf16 columns, walked by an explicit
    row loop: warp-sized CTAs with a shuffle combine per chunk, and
    whole-split chunks with one cross-warp combine, both sinking the column
    vector."""
    x = torch.empty(4700, 64, dtype=torch.bfloat16)
    bound = cpu_bind(split_column_sum_static, (x, 600))
    spec = bound.config_spec
    assert "cute_split_column_reduction" in spec.autotuner_heuristics
    seeds = [seed.config for seed in spec.compiler_seed_configs]
    assert [(seed["block_sizes"], seed["num_threads"]) for seed in seeds] == [
        ([1, 16, 128], [1, 2, 16]),
        ([1, 32, 1024], [1, 4, 128]),
        ([1, 32, 128], [1, 4, 8]),
    ]
    for seed in seeds:
        assert seed["cute_vector_widths"] == [1, 8, 1]
        assert seed["cute_lane_layouts"] == ["blocked", "blocked", "strided"]
        assert seed["cute_vloop_sink"] is True
        assert seed["cute_lane_unroll"] == 8
    # One seed of each family sinks: the per-chunk shuffle combine and the
    # cross-warp fragment combine.
    chunked = bound.to_code(spec.normalized_config(spec.compiler_seed_configs[0]))
    assert "_vsink_vec" in chunked
    assert "_cute_grouped_reduce_warp(" in chunked
    assert FRAGMENT_REDUCE not in chunked
    whole = bound.to_code(spec.normalized_config(spec.compiler_seed_configs[1]))
    assert "_vsink_vec" in whole
    assert FRAGMENT_REDUCE in whole
    # Static extents and strides need no specialization; dynamic ones are
    # proven through the exact input metadata.
    heuristic = CuteSplitColumnReductionHeuristic
    assert (
        heuristic.register_facts(bound.env, bound.host_function.device_ir)
        == frozenset()
    )
    dynamic = cpu_bind(split_column_sum_dynamic, (x, 600))
    assert "cute_split_column_reduction" in dynamic.config_spec.autotuner_heuristics
    assert heuristic.register_facts(dynamic.env, dynamic.host_function.device_ir) == {
        "input_tensor_metadata"
    }
    # A column dim that is not contiguous, or a column extent that is not a
    # multiple of the vector width (60 of 64 contiguous columns), gets no
    # seed; a one-axis grid is the column heuristic's.
    for kernel, args in (
        (
            split_column_sum_static,
            (torch.empty(64, 4700, dtype=torch.bfloat16).t(), 600),
        ),
        (
            split_column_sum_static,
            (torch.empty(4700, 64, dtype=torch.bfloat16)[:, :60], 600),
        ),
        (col_reduce_sum_static, (torch.empty(4096, 4096, dtype=torch.bfloat16),)),
    ):
        declined = cpu_bind(kernel, args)
        assert (
            "cute_split_column_reduction"
            not in declined.config_spec.autotuner_heuristics
        )


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _batched_column_sum(x: torch.Tensor) -> torch.Tensor:
    """Column sums of each of ``x``'s ``[rows, C]`` planes."""
    batches, rows, n = x.size()
    out = torch.empty([batches, n], dtype=torch.float32, device=x.device)
    for tile_b, tile_c in hl.tile([batches, n]):
        acc = hl.zeros([tile_b, tile_c], dtype=torch.float32)
        for tile_r in hl.tile(rows):
            acc = acc + torch.sum(x[tile_b, tile_r, tile_c].to(torch.float32), dim=1)
        out[tile_b, tile_c] = acc
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _broadcast_sum_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``a @ b`` as a broadcast product summed over ``k``."""
    m, k = a.size()
    n = b.size(1)
    out = torch.empty([m, n], dtype=torch.float32, device=a.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = acc + torch.sum(
                a[tile_m, tile_k][:, :, None].to(torch.float32)
                * b[tile_k, tile_n][None, :, :].to(torch.float32),
                dim=1,
            )
        out[tile_m, tile_n] = acc
    return out


def test_split_column_reduction_heuristic_takes_batched_and_broadcast_sums(
    cpu_only: None,
) -> None:
    """The match takes any grid of two or more axes whose row loop reduces
    the column tile's rows: a batched column sum and a broadcast-sum
    contraction get the three seeds too, and a seed sinks the column
    vector."""
    a = torch.empty(64, 256, dtype=torch.bfloat16)
    b = torch.empty(256, 64, dtype=torch.bfloat16)
    for kernel, args in (
        (_batched_column_sum, (torch.empty(8, 600, 64, dtype=torch.bfloat16),)),
        (_broadcast_sum_matmul, (a, b)),
    ):
        bound = cpu_bind(kernel, args)
        spec = bound.config_spec
        assert "cute_split_column_reduction" in spec.autotuner_heuristics
        sunk = [
            seed
            for seed in spec.compiler_seed_configs
            if seed.config.get("cute_vloop_sink")
        ]
        assert len(sunk) == 3
        assert "_vsink_vec" in bound.to_code(spec.normalized_config(sunk[0]))


# GPU numerics for grid vector-loop sinking (``cute_vloop_sink``).
#
# Partial tiles on both axes, the interleaved (rows above columns) thread
# layout, static shapes with several unroll depths, a max reduction and an
# fp32 input; see ``test_cute_vloop_sink.py`` for the code-shape assertions.


def _col_sum(t: torch.Tensor) -> torch.Tensor:
    return t.float().sum(0).to(t.dtype)


def _col_max(t: torch.Tensor) -> torch.Tensor:
    return t.float().amax(0).to(t.dtype)


@pytest.mark.cuda
@skipIfNotCUDA()
@onlyBackends(["cute"])
class TestCuteVloopSink(TestCase):
    def _check(
        self,
        kernel: helion.Kernel,
        ref: Callable[[torch.Tensor], torch.Tensor],
        x: torch.Tensor,
        **config: object,
    ) -> str:
        code, out = code_and_output(kernel, (x,), **config)  # pyrefly: ignore
        torch.testing.assert_close(out, ref(x), atol=2e-2, rtol=2e-2)
        # A second, different input must produce a different (correct) output.
        y = torch.randn_like(x) * 3
        _, out_y = code_and_output(kernel, (y,), **config)  # pyrefly: ignore
        torch.testing.assert_close(out_y, ref(y), atol=5e-2, rtol=2e-2)
        self.assertFalse(torch.allclose(out.float(), out_y.float()))
        self.assertIn("_vsink_vec", code)
        self.assertIn(FRAGMENT_REDUCE, code)
        return code

    def test_one_block_column_sum_matches_the_knob_off_bits(self) -> None:
        """The sunk one-block column sum adds each column's rows, and then the
        threads' sums, in the knob-off form's order; 3000 rows leave a partial
        row chunk, and 520 columns in 16-column blocks a partial tile."""
        for columns, block in ((512, 4), (520, 16)):
            with self.subTest(columns=columns):
                x = torch.randn(3000, columns, device=DEVICE, dtype=torch.float32)
                config = {**_ONE_TILE_COLUMN_SUM, "block_sizes": [block]}
                code, sunk = code_and_output(
                    _column_sum, (x,), **config, cute_vloop_sink=True
                )
                self.assertIn("_vsink_vec", code)
                _, plain = code_and_output(
                    _column_sum, (x,), **config, cute_vloop_sink=False
                )
                torch.testing.assert_close(sunk, x.sum(0), atol=1e-3, rtol=1e-4)
                self.assertTrue(torch.equal(sunk, plain))

    def test_a_vector_atomic_beside_a_sunk_column_sum(self) -> None:
        """Each column's counter is bumped once per call, from sm_90 on by the
        vector atomic that flushes the one-block tile's V lanes."""
        x = torch.randn(3000, 512, device=DEVICE, dtype=torch.float32)
        count = torch.zeros(512, device=DEVICE, dtype=torch.float32)
        code, out = code_and_output(
            _column_sum_and_count,
            (x, count),
            **_ONE_TILE_COLUMN_SUM,
            cute_vloop_sink=True,
        )
        self.assertIn("_vsink_vec", code)
        if torch.cuda.get_device_capability() >= (9, 0):
            self.assertIn("_cute_red_add_f32_vec(", code)
        torch.testing.assert_close(out, x.sum(0), atol=1e-3, rtol=1e-4)
        torch.testing.assert_close(count, torch.ones_like(count))

    def test_split_column_sum_seeds(self) -> None:
        """A chunked seed and a whole-split one over 4700 rows in splits of
        600: the last split and the last chunk of each split are partial."""
        x = torch.randn(4700, 64, device=DEVICE, dtype=torch.bfloat16)
        expected = torch.stack(
            [x[row : row + 600].float().sum(0) for row in range(0, 4700, 600)]
        )
        bound = split_column_sum_static.bind((x, 600))
        # This heuristic's chunked seed and its whole-split one, picked by
        # shape: other heuristics may seed this kernel on a real device too.
        wanted = [([1, 16, 128], [1, 2, 16]), ([1, 32, 1024], [1, 4, 128])]
        seeds = [
            seed
            for seed in bound.config_spec.compiler_seed_configs
            if (seed.config["block_sizes"], seed.config["num_threads"]) in wanted
        ]
        self.assertEqual(len(seeds), 2)
        for seed in seeds:
            with self.subTest(block_sizes=seed.config["block_sizes"]):
                out = bound.compile_config(seed)(x, 600)
                torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-3)

    def test_partial_tiles_on_both_axes(self) -> None:
        """1000 rows against a 4096-row block and 1000 columns against
        16-column blocks: row and column masks, V=4 and V=8."""
        x = torch.randn(1000, 1000, device=DEVICE, dtype=torch.bfloat16)
        self._check(
            col_reduce_sum_dynamic,
            _col_sum,
            x,
            **_sink_config(
                block_sizes=[4096, 16], num_threads=[128, 4], vec=[1, 4], unroll=8
            ),
        )
        self._check(
            col_reduce_sum_dynamic,
            _col_sum,
            x,
            **_sink_config(
                block_sizes=[4096, 16], num_threads=[256, 2], vec=[1, 8], unroll=4
            ),
        )

    def test_rows_above_columns_partial_tiles(self) -> None:
        x = torch.randn(1000, 56, device=DEVICE, dtype=torch.bfloat16)
        code = self._check(
            col_reduce_sum_dynamic,
            _col_sum,
            x,
            **_sink_config(
                block_sizes=[2048, 16],
                num_threads=[16, 8],
                vec=[1, 2],
                layouts=["blocked", "blocked"],
                unroll=4,
            ),
        )
        self.assertIn("pre=8, group_span=128", code)

    def test_interleaved_fragment_combine_matches_the_knob_off_bits(self) -> None:
        """The fragment combine of eight interleaved column threads over four
        warps adds the threads' fp32 sums in the order of the per-lane reduces
        it replaces."""
        x = torch.randn(1000, 56, device=DEVICE, dtype=torch.float32)
        layout = {
            "block_sizes": [2048, 16],
            "num_threads": [16, 8],
            "vec": [1, 2],
            "layouts": ["blocked", "blocked"],
        }
        code, sunk = code_and_output(
            col_reduce_sum_dynamic, (x,), **_sink_config(**layout, unroll=4)
        )
        self.assertIn("count=2, pre=8, group_span=128", code)
        _, plain = code_and_output(
            col_reduce_sum_dynamic, (x,), **_sink_config(**layout, sink=False)
        )
        self.assertTrue(torch.equal(sunk, plain))

    def test_static_shapes_and_unroll_depths(self) -> None:
        x = torch.randn(2048, 512, device=DEVICE, dtype=torch.bfloat16)
        for unroll in (1, 2, 16):
            self._check(
                col_reduce_sum_static,
                _col_sum,
                x,
                **_sink_config(
                    block_sizes=[2048, 16],
                    num_threads=[128, 2],
                    vec=[1, 8],
                    unroll=unroll,
                ),
            )

    def test_max_reduction(self) -> None:
        x = torch.randn(1000, 1000, device=DEVICE, dtype=torch.bfloat16)
        code = self._check(
            col_reduce_max_dynamic,
            _col_max,
            x,
            **_sink_config(
                block_sizes=[4096, 16], num_threads=[128, 4], vec=[1, 4], unroll=8
            ),
        )
        self.assertIn("'max', cutlass.Float32(float('-inf'))", code)
        # The sunk per-lane combine keeps torch.amax's NaN propagation.
        x[17, 5] = float("nan")
        x[900, 640] = float("nan")
        _, out = code_and_output(
            col_reduce_max_dynamic,
            (x,),
            **_sink_config(
                block_sizes=[4096, 16], num_threads=[128, 4], vec=[1, 4], unroll=8
            ),
        )
        torch.testing.assert_close(out, _col_max(x), equal_nan=True)

    def test_float32_input(self) -> None:
        x = torch.randn(1000, 1000, device=DEVICE, dtype=torch.float32)
        code = self._check(
            col_reduce_sum_dynamic,
            _col_sum,
            x,
            **_sink_config(
                block_sizes=[4096, 8], num_threads=[128, 2], vec=[1, 4], unroll=4
            ),
        )
        self.assertIn("ir.VectorType.get([4], cutlass.Uint32.mlir_type)", code)

    def test_a_guarded_row_store_sinks_bitwise(self) -> None:
        """``test_a_guarded_row_store_lets_the_row_loads_sink``'s ragged row
        tiles: the sunk kernel's output is bitwise the knob-off kernel's."""
        x = torch.randn(500, 384, device=DEVICE, dtype=torch.bfloat16)
        w = torch.randn(384, device=DEVICE, dtype=torch.bfloat16)
        outputs = {}
        for sink in (True, False):
            code, outputs[sink] = code_and_output(
                _scale_columns,
                (x, w),
                block_sizes=[32, 32],
                num_threads=[8, 4],
                loop_orders=[[1, 0]],
                cute_vector_widths=[1, 8],
                cute_lane_layouts=["blocked", "strided"],
                cute_vloop_sink=sink,
            )
            self.assertEqual("_vsink" in code, sink)
        torch.testing.assert_close(outputs[True], outputs[False], atol=0, rtol=0)
        expected = (x.float() * w.float()).to(x.dtype)
        torch.testing.assert_close(outputs[True], expected, atol=0, rtol=0)

    def test_nonzero_grid_origin_declines_and_stays_correct(self) -> None:
        """``hl.tile(8, n)`` keeps per-element column masks: the pass declines
        (no out-of-bounds vector transaction) and the knob-off code runs."""
        x = torch.randn(1000, 1008, device=DEVICE, dtype=torch.bfloat16)
        code, out = code_and_output(
            col_reduce_sum_from8_dynamic,
            (x,),
            **_sink_config(
                block_sizes=[4096, 16], num_threads=[128, 4], vec=[1, 4], unroll=8
            ),
        )
        expected = torch.zeros_like(out)
        expected[8:] = _col_sum(x[:, 8:])
        torch.testing.assert_close(out, expected, atol=2e-2, rtol=2e-2)
        self.assertNotIn("_vsink", code)
        self.assertNotIn(FRAGMENT_REDUCE, code)

    def test_a_misaligned_input_view_declines_and_stays_correct(self) -> None:
        """A view whose base is four bytes into its allocation cannot take the
        8-byte sunk packet: the pass declines (it shares the tile hoist's base
        proof) and the knob-off code runs instead of faulting on a misaligned
        address."""
        x = torch.randn(4096, 4112, device=DEVICE, dtype=torch.bfloat16)[:, 2:4098]
        self.assertEqual(x.data_ptr() % 8, 4)
        config = _sink_config(
            block_sizes=[4096, 16], num_threads=[128, 4], vec=[1, 4], unroll=8
        )
        code, out = code_and_output(col_reduce_sum_static, (x,), **config)
        code_off, out_off = code_and_output(
            col_reduce_sum_static, (x,), **{**config, "cute_vloop_sink": False}
        )
        torch.testing.assert_close(out, _col_sum(x), atol=2e-2, rtol=2e-2)
        self.assertEqual(code, code_off)
        self.assertTrue(torch.equal(out, out_off))
        self.assertNotIn("_vsink", code)

    def test_row_weighted_mean(self) -> None:
        """A row-only weight load and a row-only sum next to the column sum:
        both become per-lane state; partial tiles on both axes."""
        x = torch.randn(1000, 1000, device=DEVICE, dtype=torch.bfloat16)
        w = torch.rand(1000, device=DEVICE, dtype=torch.bfloat16) + 0.5
        code, out = code_and_output(
            col_weighted_mean_dynamic,
            (x, w),
            **_sink_config(
                block_sizes=[4096, 16], num_threads=[128, 4], vec=[1, 4], unroll=8
            ),
        )
        expected = (x.float() * w.float()[:, None]).sum(0) / w.float().sum()
        torch.testing.assert_close(out, expected, atol=2e-2, rtol=2e-2)
        self.assertIn("_vsink_vec", code)
        self.assertEqual(code.count(FRAGMENT_REDUCE), 2)
        self.assertIn("wsum_frag", code)

    def test_lane_invariant_scalar_redefined_by_an_if(self) -> None:
        """``s = scale[0]`` feeds one store, an ``if flag[0]`` doubles ``s``
        and a second store uses it: with the flag set, every column of both
        outputs must see one ``scale`` (not a value compounded over the V
        lanes), and with it clear the second output equals the first."""
        x = torch.randn(1000, 1000, device=DEVICE, dtype=torch.bfloat16)
        scale = torch.full((1,), 1.5, device=DEVICE, dtype=torch.float32)
        column_sum = x.float().sum(0)
        for flag_value, factor in ((1, 2.0), (0, 1.0)):
            flag = torch.full((1,), flag_value, device=DEVICE, dtype=torch.int32)
            code, (out1, out2) = code_and_output(
                col_reduce_sum_rescaled_dynamic,
                (x, scale, flag),
                **_sink_config(
                    block_sizes=[4096, 16], num_threads=[128, 4], vec=[1, 4], unroll=8
                ),
            )
            torch.testing.assert_close(out1, column_sum * 1.5, atol=2e-2, rtol=2e-2)
            torch.testing.assert_close(
                out2, column_sum * (1.5 * factor), atol=2e-2, rtol=2e-2
            )
            self.assertIn("_vsink_vec", code)
            self.assertIn(FRAGMENT_REDUCE, code)

    def test_a_gathered_value_is_read_before_the_atomic(self) -> None:
        """Each lane gathers its own element of ``count`` and then bumps it,
        unguarded (static shapes, one thread along the rows), so one thread
        alone touches that element: the gathered value is the initial 2.0,
        never the 3.0 the atomic leaves behind, with the knob on as with it
        off.  Recomputing the gather after the row loop read 3.0."""
        x = torch.randn(1024, 1024, device=DEVICE, dtype=torch.bfloat16)
        outputs: dict[bool, torch.Tensor] = {}
        for sink in (False, True):
            count = torch.full((1024,), 2.0, device=DEVICE, dtype=torch.float32)
            code, out = code_and_output(
                col_reduce_sum_gather_then_atomic_static,
                (x, count),
                **_sink_config(
                    block_sizes=[1024, 16],
                    num_threads=[1, 4],
                    vec=[1, 4],
                    unroll=8,
                    sink=sink,
                ),
            )
            self.assertEqual("_vsink_vec" in code, sink)
            torch.testing.assert_close(count, torch.full_like(count, 3.0))
            before = out - x.float().sum(0)
            torch.testing.assert_close(
                before, torch.full_like(before, 2.0), atol=5e-2, rtol=0
            )
            outputs[sink] = out
        torch.testing.assert_close(outputs[True], outputs[False])

    def test_sibling_row_loops(self) -> None:
        x = torch.randn(1000, 1024, device=DEVICE, dtype=torch.bfloat16)
        y = torch.randn(1000, 1024, device=DEVICE, dtype=torch.bfloat16) * 2
        code, out = code_and_output(
            col_reduce_sum_pair_dynamic,
            (x, y),
            **_sink_config(
                block_sizes=[4096, 16], num_threads=[128, 4], vec=[1, 4], unroll=8
            ),
        )
        expected = (x.float().sum(0) + y.float().sum(0)).to(torch.bfloat16)
        torch.testing.assert_close(out, expected, atol=5e-2, rtol=2e-2)
        # Each row loop strength-reduces and vectorizes its own tensor.
        self.assertIn("_vsink_ptr = x.iterator", code)
        self.assertIn("_vsink_ptr_1 = y.iterator", code)
        self.assertEqual(code.count(FRAGMENT_REDUCE), 2)

    def test_a_later_masked_store_of_the_same_elements_lands_last(self) -> None:
        """The even channels the masked store zeroes after the ReLU store
        stay zero: the ReLU store is not flushed after the V-loop."""
        x = torch.randn(4, 784, 120, device=DEVICE, dtype=torch.bfloat16)
        shift = torch.randn(120, device=DEVICE, dtype=torch.float32)
        bound = _relu_then_zero_even_channels.bind((x, shift))
        config = _strided_reduction_config(bound, sink=True)
        bound.set_config(config)
        self.assertIn("_vsink_vec", bound.to_code(config))
        act, pooled = cast("tuple[torch.Tensor, torch.Tensor]", bound(x, shift))
        y = torch.relu(x.float() - shift)
        torch.testing.assert_close(pooled, y.mean(1).to(x.dtype), atol=1e-2, rtol=1e-2)
        y[..., ::2] = 0
        torch.testing.assert_close(act, y.to(x.dtype), atol=0, rtol=0)

    def test_a_strided_reduction_past_the_z_launch_limit_sinks_in_place(
        self,
    ) -> None:
        """Two threads on each grid axis and 128 reduction threads keep the
        default ``(128, 2, 2)`` layout, where the V-loop still sinks."""
        x = torch.randn(8, 784, 120, device=DEVICE, dtype=torch.bfloat16)
        bound = _spatial_mean.bind((x,))
        config = _strided_reduction_config(
            bound,
            sink=True,
            block_sizes=(2, 16),
            grid_threads=(2, 2),
            reduction_threads=128,
            rows=256,
        )
        bound.set_config(config)
        code = bound.to_code(config)
        self.assertIn("_vsink_vec", code)
        self.assertIn("block=(128, 2, 2)", code)
        torch.testing.assert_close(
            bound(x), x.float().mean(1).to(x.dtype), atol=1e-2, rtol=1e-2
        )

    def test_strided_reductions_with_threads_on_both_grid_dims(self) -> None:
        """Threads on both grid dims in both loop orders (the default layout
        with ``tile_n`` first, the tile-first one with ``tile_c`` first),
        then a persistent reduction, against eager."""
        for rows, loop_orders in ((784, None), (784, [[1, 0]]), (64, None)):
            x = torch.randn(8, rows, 120, device=DEVICE, dtype=torch.bfloat16)
            bound = _spatial_mean.bind((x,))
            config = (
                _strided_reduction_config(
                    bound,
                    sink=True,
                    block_sizes=(2, 64),
                    grid_threads=(2, 8),
                    loop_orders=loop_orders,
                )
                if rows == 784
                else _strided_reduction_config(bound, sink=True, rows=None)
            )
            bound.set_config(config)
            torch.testing.assert_close(
                bound(x), x.float().mean(1).to(x.dtype), atol=1e-2, rtol=1e-2
            )

    def test_strided_reductions_with_the_tile_on_thread_x(self) -> None:
        """Partial channel vectors (120 of a 64-wide tile's second block) and
        a partial last row chunk (784 rows in 32-row chunks of 16 threads),
        with the ReLU output stored inside the reduction as packets; then the
        heuristic's first seed and its 64-row seed (16 vector threads, 8
        reduction threads over 128- and 64-row chunks, the row loop unrolled
        8 deep with a packet flush per copy)."""
        x = torch.randn(4, 784, 120, device=DEVICE, dtype=torch.bfloat16)
        shift = torch.randn(120, device=DEVICE, dtype=torch.float32)
        spec = _relu_and_spatial_mean.bind((x, shift)).config_spec
        seeds = [
            spec.normalized_config(helion.Config(**seed.config))
            for seed in spec.compiler_seed_configs
            if seed.config.get("cute_vloop_sink")
        ]
        (rows64,) = [seed for seed in seeds if seed.reduction_loops == [64]]
        for kernel, args, seed in (
            (_spatial_mean, (x,), None),
            (_relu_and_spatial_mean, (x, shift), None),
            (_relu_and_spatial_mean, (x, shift), seeds[0]),
            (_relu_and_spatial_mean, (x, shift), rows64),
        ):
            bound = kernel.bind(args)
            config = (
                _strided_reduction_config(bound, sink=True) if seed is None else seed
            )
            bound.set_config(config)
            self.assertIn("_vsink_vec", bound.to_code(config))
            result = bound(*args)
            if kernel is _spatial_mean:
                expected = x.float().mean(1).to(x.dtype)
                torch.testing.assert_close(result, expected, atol=1e-2, rtol=1e-2)
                continue
            act, pooled = cast("tuple[torch.Tensor, torch.Tensor]", result)
            y = torch.relu(x.float() - shift)
            torch.testing.assert_close(act, y.to(x.dtype), atol=0, rtol=0)
            torch.testing.assert_close(
                pooled, y.mean(1).to(x.dtype), atol=1e-2, rtol=1e-2
            )

    def test_a_column_sum_over_4_byte_aligned_rows(self) -> None:
        """The heuristic's first seed for rows of 1026 bf16 elements: 2-element
        packets on 128 vector threads, the last 256-column tile jagged."""
        x = torch.randn(2048, 1026, device=DEVICE, dtype=torch.bfloat16)
        bound = _column_sum.bind((x,))
        spec = bound.config_spec
        seed = next(
            spec.normalized_config(helion.Config(**seed.config))
            for seed in spec.compiler_seed_configs
            if seed.config.get("cute_vloop_sink")
        )
        bound.set_config(seed)
        self.assertIn("_cute_load_v4b_if(", bound.to_code(seed))
        torch.testing.assert_close(bound(x), x.float().sum(0), atol=1e-2, rtol=1e-3)

    def _check_channel_affine(self, rows: int, config: dict[str, Any]) -> None:
        """``config`` matches eager, and its output with the channel operands
        hoisted (``cute_vloop_sink``) is bitwise the knob-off kernel's."""
        args = _channel_affine_args(DEVICE, rows=rows)
        outputs = []
        for sink in (True, False):
            code, out = code_and_output(
                _channel_affine_relu, args, **(config | {"cute_vloop_sink": sink})
            )
            self.assertEqual("_vsink_vec" in code, sink)
            outputs.append(out)
        sunk, unsunk = outputs
        torch.testing.assert_close(sunk, unsunk, atol=0, rtol=0)
        x, mean, var, weight, bias = (t.float() for t in args)
        expected = torch.relu((x - mean) * torch.rsqrt(var + 1e-05) * weight + bias)
        torch.testing.assert_close(
            sunk, expected.to(torch.bfloat16), atol=1e-2, rtol=1e-2
        )

    def test_channel_affine_with_parameters_hoisted_out_of_the_row_loop(self) -> None:
        self._check_channel_affine(512, dict(_channel_affine_config(sink=True).config))

    def test_channel_affine_hoist_seed(self) -> None:
        """The first broadcast-hoist seed (64x128 tiles, eight unrolled rows
        per thread, three CTAs per SM), on whole and on ragged row tiles."""
        for rows in (4096, 500):
            bound = _channel_affine_relu.bind(_channel_affine_args(DEVICE, rows=rows))
            self._check_channel_affine(rows, _hoist_seeds(bound)[0])
