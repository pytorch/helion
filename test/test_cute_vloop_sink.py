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
import re
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
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
from helion._compiler.cute import sink_vector_loops
from helion._compiler.cute.autotune.heuristics.cute import CuteColumnReductionHeuristic
from helion._compiler.cute.lane_loops import root_passes
from helion._compiler.cute.sink_vector_loops import VloopSinkNotApplied
from helion._compiler.cute.state import cute_env_state
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import cpu_render_target
from helion._testing import import_example
from helion._testing import onlyBackends
from helion._testing import skipIfNotCUDA
from helion._testing.cute_codegen import cpu_bind
import helion.language as hl

pytestmark = pytest.mark.cpu_only

rms_norm_fwd = import_example("rms_norm").rms_norm_fwd
softmax = import_example("softmax").softmax

if TYPE_CHECKING:
    from collections.abc import Iterator

    from helion._compiler.generate_ast import GenerateAST

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
