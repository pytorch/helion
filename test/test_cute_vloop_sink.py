"""Grid vector-loop sinking for serial reductions (``cute_vloop_sink``).

A column sum (``acc += torch.sum(x[tile_m, tile_n], dim=0)`` with the vector
along the grid axis ``tile_n``) used to run its constexpr V-loop OUTSIDE the
serial row loops: every thread walked its row strip V times with 2-byte
scalar loads and paid V cross-thread combines per row tile.  With
``cute_vloop_sink`` the V-loop is distributed over the body and interchanged
with the loops: one V-wide vector load per row, V register accumulators, one
fragment grouped combine per row tile, strength-reduced row addressing and an
optional load-first unroll of the row lane loop (``cute_lane_unroll``).

CPU code generation only; ``test_cute_vloop_sink_cuda.py`` runs the kernels.
"""

from __future__ import annotations

import ast
import builtins
from contextlib import ExitStack
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from unittest.mock import patch

from examples.rms_norm import rms_norm_fwd
from examples.softmax import softmax
import pytest
import torch

from test._cute_binding import _cpu_bind
from test._cute_binding import _mock_cuda_unavailable
from test._cute_vloop_sink_kernels import FRAGMENT_REDUCE
from test._cute_vloop_sink_kernels import TWO_STAGE_REDUCE
from test._cute_vloop_sink_kernels import _sink_config
from test._cute_vloop_sink_kernels import col_reduce_sum_atomic_then_load_static
from test._cute_vloop_sink_kernels import col_reduce_sum_dynamic
from test._cute_vloop_sink_kernels import col_reduce_sum_from8_dynamic
from test._cute_vloop_sink_kernels import col_reduce_sum_from8_static
from test._cute_vloop_sink_kernels import col_reduce_sum_gather_then_atomic_static
from test._cute_vloop_sink_kernels import col_reduce_sum_lower_triangle_static
from test._cute_vloop_sink_kernels import col_reduce_sum_pair_static
from test._cute_vloop_sink_kernels import col_reduce_sum_rescaled_const_static
from test._cute_vloop_sink_kernels import col_reduce_sum_rescaled_static
from test._cute_vloop_sink_kernels import col_reduce_sum_static
from test._cute_vloop_sink_kernels import col_weighted_mean_static

import helion
from helion._compiler import generate_ast
from helion._compiler.ast_read_writes import ReadWrites
from helion._compiler.autotuner_heuristics.cute import CuteColumnReductionHeuristic
from helion._compiler.cute import sink_vector_loops
from helion._testing import skipUnlessBackends
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Iterator

cutlass = pytest.importorskip("cutlass")


@pytest.fixture
def cpu_only() -> Iterator[None]:
    with (
        _mock_cuda_unavailable(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CPU-only test")),
        patch(
            "helion._compiler.reduction_strategy._cute_shared_memory_budget_bytes",
            return_value=232448,
        ),
    ):
        yield


def _cpu_code(
    kernel: helion.Kernel, args: tuple[torch.Tensor, ...], **config: object
) -> str:
    bound = _cpu_bind(kernel, args)
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
    return [
        stmt
        for stmt in body
        if isinstance(stmt, ast.Assign)
        and "cute.arch.load(" in ast.unparse(stmt)
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


@skipUnlessBackends(["cute"])
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


@skipUnlessBackends(["cute"])
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


@skipUnlessBackends(["cute"])
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


@skipUnlessBackends(["cute"])
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


@skipUnlessBackends(["cute"])
def test_dynamic_shapes_guard_whole_vectors_with_a_uniform_mask(
    cpu_only: None,
) -> None:
    """With ``n % V == 0`` the column mask is the same for all V lanes: it is
    evaluated once at the chunk base, guards the vector transaction (masked
    threads read the tensor's first chunk) and still gates every value."""
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
    assert (
        "if mask_0_u0 and mask_1 else x.iterator + cutlass.Int32(0), ir.VectorType"
        in ast.unparse(loads[0])
    )
    assert "mask_0_u1 = indices_0_u1 < m" in ast.unparse(lane_loop)
    assert (
        ".bitcast(cutlass.BFloat16) if mask_0_u1 and mask_1 else cutlass.BFloat16(0)"
        in ast.unparse(lane_loop)
    )
    # The finished columns are stored under the same uniform mask.
    assert "if mask_1:" in code
    assert code.count(FRAGMENT_REDUCE) == 1


@skipUnlessBackends(["cute"])
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


@skipUnlessBackends(["cute"])
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


@skipUnlessBackends(["cute"])
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


@skipUnlessBackends(["cute"])
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


@skipUnlessBackends(["cute"])
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


@skipUnlessBackends(["cute"])
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


@skipUnlessBackends(["cute"])
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
        (load,) = [text for text in statements if "cute.arch.load(" in text]
        assert f"else {tensor}.iterator + cutlass.Int32(0)" in load
        # The row index and the row mask of this loop come first, in the
        # same lane iteration, ahead of the vector transaction they address
        # and guard.
        (row_index,) = [text for text in statements if text.startswith("indices_")]
        (row_mask,) = [text for text in statements if text.startswith("mask_")]
        assert statements.index(row_index) < statements.index(row_mask)
        assert statements.index(row_mask) < statements.index(load)
    assert code.count(FRAGMENT_REDUCE) == 2
    assert ".load()" not in code


@skipUnlessBackends(["cute"])
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


@skipUnlessBackends(["cute"])
def test_a_load_after_a_bound_atomic_stays_per_lane(cpu_only: None) -> None:
    """``old = hl.atomic_add(count, [tile_n], 1); first = count[tile.id * B]``:
    with its result used and no guard, the atomic is an assignment rather
    than a store statement, yet it is a memory effect of the segment.  The
    lane-invariant load of ``count`` stays in the V-loop after it (lane 0 must
    observe its own increment) while the row loop still sinks."""
    x = torch.empty(1024, 1024, dtype=torch.bfloat16)
    count = torch.empty(1024, dtype=torch.float32)
    code = _cpu_code(
        col_reduce_sum_atomic_then_load_static,
        (x, count),
        **_sink_config(block_sizes=[1024, 16], num_threads=[1, 4], vec=[1, 4]),
    )
    _assert_defined_before_use(code)
    assert "_vsink_vec" in code
    kernel = _kernel_def(code)
    every_statement = [
        ast.unparse(stmt) for stmt in ast.walk(kernel) if isinstance(stmt, ast.stmt)
    ]
    (prologue,) = [
        loop
        for loop in ast.walk(kernel)
        if isinstance(loop, ast.For)
        and "range_constexpr" in ast.unparse(loop.iter)
        and "cute.arch.atomic_add(" in ast.unparse(loop)
    ]
    statements = [ast.unparse(stmt) for stmt in prologue.body]
    (atomic,) = [
        text for text in statements if text.startswith("old = cute.arch.atomic_add(")
    ]
    (load,) = [
        text
        for text in statements
        if text.startswith("first = (count.iterator") and text.endswith(".load()")
    ]
    # The load is issued once per lane, after that lane's atomic, and nowhere
    # else (not widened, not run once ahead of the V-loop).
    assert statements.index(atomic) < statements.index(load)
    assert every_statement.count(load) == 1
    assert sum(text.startswith("first = ") for text in every_statement) == 1
    assert "v_0_frag[vec_lane_1] = old * first" in statements
    # The pure tile-origin arithmetic of the load's address is still hoisted.
    assert "mul = _BLOCK_SIZE_1 * tile_id" not in statements
    assert "mul = _BLOCK_SIZE_1 * tile_id" in every_statement


@skipUnlessBackends(["cute"])
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


@skipUnlessBackends(["cute"])
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
        assert bound.env.cute_fission_plan is not None
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

        def not_applied(self: generate_ast.GenerateAST) -> None:
            restarts.append(True)
            raise generate_ast.VloopSinkNotApplied

        with patch.object(
            generate_ast.GenerateAST, "_sink_grid_vector_loops", not_applied
        ):
            retried = code(True)
        assert restarts == [True, True]
        assert retried == off


@skipUnlessBackends(["cute"])
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


@skipUnlessBackends(["cute"])
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
        bound = _cpu_bind(kernel, args)
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


@skipUnlessBackends(["cute"])
def test_lane_unroll_requires_sinking_or_a_grid_lane_loop(cpu_only: None) -> None:
    x = torch.empty(4096, 4096, dtype=torch.bfloat16)
    bound = _cpu_bind(col_reduce_sum_static, (x,))

    def normalized_unroll(num_threads: list[int]) -> int:
        config = bound.config_spec.normalized_config(
            helion.Config(
                block_sizes=[4096, 16],
                num_threads=num_threads,
                cute_vector_widths=[1, 4],
                cute_vloop_sink=False,
                cute_lane_unroll=8,
            )
        )
        return config.config["cute_lane_unroll"]

    # Without sinking the unroll belongs to the loads-first lane unroll
    # (``cute/unroll_lane_loads.py``), which needs a grid lane loop: four
    # threads over 16 columns form one, sixteen do not.
    assert normalized_unroll([128, 4]) == 8
    assert normalized_unroll([128, 16]) == 1
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


@skipUnlessBackends(["cute"])
def test_column_reduction_heuristic_seeds_the_coalesced_layout(
    cpu_only: None,
) -> None:
    x = torch.empty(4096, 4096, dtype=torch.bfloat16)
    bound = _cpu_bind(col_reduce_sum_dynamic, (x,))
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
    static_bound = _cpu_bind(col_reduce_sum_static, (x,))
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
        declined = _cpu_bind(kernel, (torch.empty(*shape, dtype=torch.bfloat16),))
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
        declined = _cpu_bind(kernel, (x,))
        assert (
            CuteColumnReductionHeuristic.register_facts(
                declined.env, declined.host_function.device_ir
            )
            == frozenset()
        )
    # Row reductions are not column reductions.
    rows_bound = _cpu_bind(
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
