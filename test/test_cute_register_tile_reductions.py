"""Register-tile lowering of lane reductions over one-vector tiles (CPU codegen).

The fused linear-attention recurrent step loads a ``[D, dv]`` state tile,
writes the updated tile to a new state tensor at the same index (``new_state =
state * decay + k[:, None] * v[None, :]``) and reduces ``q[:, None] *
new_state`` over ``D``.  With the reduction split across a persistent synthetic
lane loop the CuTe backend used to nest that lane loop inside the tile's
constexpr V-loop, so the state load stayed scalar, each lane iteration waited
for its own load, and the same-index read-modify-write (the in-place form of the
step, kept here as ``recurrent_step_in_place``) was rejected by the aliasing
gate because the tensor strides were not proven.

Now the scalar reduction lane nests OUTSIDE the one-vector tile wrappers as a
trace-time register tile: one packet load per lane, every lane's loads issued
before the first store, per-tile-element accumulators combined across the
thread group once with the shared-memory column reduce, and the vectorized
tile axis on ``thread_idx[0]`` so a warp's packet loads cover whole rows.
"""

from __future__ import annotations

import ast
import dataclasses
import itertools
import logging
import pathlib
import types
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from unittest.mock import patch

import pytest
import torch

from test._cute_register_tile_kernels import col_scale
from test._cute_register_tile_kernels import col_scale_into
from test._cute_register_tile_kernels import col_scale_three
from test._cute_register_tile_kernels import col_softmax
from test._cute_register_tile_kernels import col_stats
from test._cute_register_tile_kernels import col_sum
from test._cute_register_tile_kernels import col_sum_dynamic
from test._cute_register_tile_kernels import col_sum_f32_out
from test._cute_register_tile_kernels import col_sum_guarded
from test._cute_register_tile_kernels import col_sum_nested
from test._cute_register_tile_kernels import col_var
from test._cute_register_tile_kernels import column_config
from test._cute_register_tile_kernels import recurrent_step_in_place
from test._cute_register_tile_kernels import rms_norm_bwd
from test._cute_register_tile_kernels import rms_norm_bwd_arguments

import helion
from helion._compiler import generate_ast as generate_ast_module
from helion._compiler.backend import Backend
from helion._compiler.cute.cache_policy_loads import _CUTE_CACHE_LOAD_HELPER_NAMES
from helion._compiler.cute.cache_policy_loads import MASKED_PACKET_LOAD_HELPER_NAMES
from helion._compiler.cute.persistent_branch_vec import _generated_access_pointer
from helion._compiler.cute.register_tile_admission import (
    _CUTE_REGISTER_TILE_CONTROL_FLOW,
)
from helion._compiler.cute.register_tile_admission import RegisterTileUnsupported
from helion._compiler.emitter_protocols import RetryPolicy
from helion._testing import DEVICE
from helion._testing import skipIfRefEager
from helion._testing.cute_codegen import cpu_bind
from helion.experimental.linear_attention import recurrent_step_fused
import helion.language as hl
from helion.language import _tracing_ops

pytestmark = [
    pytest.mark.cpu_only,
    pytest.mark.cpu_render(reduction_smem_budget_bytes=232448),
]

if TYPE_CHECKING:
    from collections.abc import Callable

BH, D, DV = 8, 128, 128
COLUMN_REDUCE = "_cute_grouped_reduce_shared_columns("
CONSTEXPR_OWNER = "for synthetic_lane_1 in cutlass.range_constexpr("
ROLLED_OWNER = "for synthetic_lane_1 in range("
FALLBACK_LOG = "regenerating with the rolled reduction lane"
LOOPED_RETRY_LOG = "the retry loops the reduction"


def _kernel() -> helion.Kernel:
    return helion.kernel(
        recurrent_step_fused.fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
        ignore_warnings=[helion.exc.TensorOperationInWrapper],
    )


def _arguments(
    state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    if state is None:
        state = torch.empty((BH, D, DV), dtype=torch.float32)
    return (
        torch.empty((BH, D), dtype=torch.float32),
        torch.empty((BH, D), dtype=torch.float32),
        torch.empty((BH, DV), dtype=torch.float32),
        state,
        torch.empty((BH,), dtype=torch.float32),
    )


def _bind(
    arguments: tuple[torch.Tensor, ...] | None = None,
) -> tuple[Any, tuple[torch.Tensor, ...]]:
    """Bind on CPU; the caller keeps ``arguments`` alive while generating code
    (the vector alignment proof reads the live inputs through weak references).
    """
    if arguments is None:
        arguments = _arguments()
    return cpu_bind(_kernel(), arguments), arguments


def _config(
    bound: Any,
    *,
    dv_block: int,
    dv_threads: int,
    reduction_threads: int,
    vec: int = 4,
    reduction_loop: int | None = None,
    **extra: Any,
) -> helion.Config:
    spec = bound.config_spec
    (reduction,) = [
        block.block_id for block in bound.env.block_sizes if block.reduction
    ]
    (dv,) = spec.block_sizes.valid_block_ids()
    threads = {dv: dv_threads, reduction: reduction_threads}
    return spec.normalized_config(
        helion.Config(
            block_sizes=[dv_block],
            num_threads=[
                threads.get(block_id, 0)
                for block_id in spec.num_threads.valid_block_ids()
            ],
            reduction_loops=[reduction_loop],
            cute_vector_widths=[
                vec if block_id == dv else 1
                for block_id in spec.cute_vector_widths.valid_block_ids()
            ],
            **extra,
        )
    )


def _kernel_loops(code: str) -> list[ast.For]:
    """Every ``for`` loop of the generated device kernel, outermost first."""
    module = ast.parse(code)
    (kernel,) = [
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("_helion_")
    ]
    return [node for node in ast.walk(kernel) if isinstance(node, ast.For)]


def _iter_name(loop: ast.For) -> str:
    assert isinstance(loop.iter, ast.Call)
    return ast.unparse(loop.iter.func)


def _owner_loops(code: str) -> list[ast.For]:
    """The top-level reduction lane loops, in program order."""
    module = ast.parse(code)
    (kernel,) = [
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("_helion_")
    ]
    return [
        node
        for node in kernel.body
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id.startswith("synthetic_lane_")
    ]


def _assert_defined_before_use(code: str) -> None:
    """Every local name of the device kernel is assigned before its first
    lexical use.  All loops unroll at trace time, so lexical order is
    execution order."""
    module = ast.parse(code)
    (kernel,) = [
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("_helion_")
    ]
    local_names = {
        node.id
        for node in ast.walk(kernel)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    }
    assigned = {arg.arg for arg in kernel.args.args}

    def check(node: ast.AST) -> None:
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load):
                assert sub.id not in local_names or sub.id in assigned, (
                    f"line {sub.lineno}: {sub.id} is used before its assignment"
                )

    def visit(statements: list[ast.stmt]) -> None:
        for stmt in statements:
            if isinstance(stmt, ast.For):
                check(stmt.iter)
                assert isinstance(stmt.target, ast.Name)
                assigned.add(stmt.target.id)
                visit(stmt.body)
            elif isinstance(stmt, ast.If):
                check(stmt.test)
                visit(stmt.body)
                visit(stmt.orelse)
            elif isinstance(stmt, ast.Assign):
                check(stmt.value)
                for target in stmt.targets:
                    if isinstance(target, ast.Name):
                        assigned.add(target.id)
                    else:
                        check(target)
            else:
                check(stmt)

    visit(kernel.body)


def test_same_index_state_update_keeps_vector_loads_and_stores() -> None:
    bound, _arguments = _bind()
    code = bound.to_code(_config(bound, dv_block=32, dv_threads=8, reduction_threads=8))
    # Every lane loop unrolls at trace time; the per-lane values live in
    # Python lists instead of a rolled loop's re-loads.
    assert "_lane_stash" in code
    assert all(
        _iter_name(loop) == "cutlass.range_constexpr" for loop in _kernel_loops(code)
    )
    # One 16-byte packet per lane for the state tile and one for ``v``; the
    # state tensor is read exactly once (the packet) and the new state written
    # exactly once (the flush) per lane -- no scalar re-load of the stored
    # fragment.
    assert code.count("cute.arch.load(") == 2
    assert code.count("ir.VectorType.get([4], cutlass.Uint32.mlir_type)") == 2
    assert code.count("(state.iterator") == 1
    assert code.count("(new_state.iterator") == 1
    assert code.count("_cute_store_u32_vec(new_state.iterator") == 1
    assert code.count("_cute_store_u32_vec(out.iterator") == 1
    # Accumulate nest: all loads, no stores.  Consume nest: the state flush.
    accumulate, consume = _owner_loops(code)
    accumulate_source = ast.unparse(accumulate)
    consume_source = ast.unparse(consume)
    assert accumulate_source.count("cute.arch.load(") == 2
    assert ".load()" in accumulate_source
    assert "_cute_store" not in accumulate_source
    assert "_cute_store_u32_vec(new_state.iterator" in consume_source
    assert ".load(" not in consume_source
    # The vectorized tile axis owns thread_idx[0]; the reduction sits above it.
    assert (
        "lane_base_1 = tile_offset_1 + cutlass.Int32(cute.arch.thread_idx()[0]) * 4"
        in code
    )
    assert (
        "indices_2 = cutlass.Int32(cute.arch.thread_idx()[1]) + cutlass.Int32(synthetic_lane_2) * 8"
        in code
    )
    assert "block=(8, 8, 1)" in code


def test_column_reduce_runs_once_per_tile() -> None:
    bound, _arguments = _bind()
    code = bound.to_code(_config(bound, dv_block=32, dv_threads=8, reduction_threads=8))
    assert code.count(COLUMN_REDUCE) == 1
    assert "pre=8, group_span=64, group_count=1)" in code
    assert "_cute_grouped_reduce_shared_two_stage" not in code
    assert "warp_reduction" not in code
    assert "_helion_lane_reduce" not in code
    # No cross-thread reduce or barrier inside any element loop.
    for loop in _kernel_loops(code):
        body = "\n".join(ast.unparse(stmt) for stmt in loop.body)
        assert "_cute_grouped_reduce" not in body
        assert "sync_threads" not in body
    # The reduced output is stored by the reduction-coordinate-zero thread of
    # each column group only.
    assert "% 64 < 8:" in code
    assert code.count("_cute_store_u32_vec(out.iterator") == 1


def test_multiwarp_group_keeps_its_thread_count() -> None:
    # 16 reduction threads above 8 vector threads span four warps; the
    # register tile keeps them (no one-warp collapse) and combines with the
    # strided column reduce.
    bound, _arguments = _bind()
    code = bound.to_code(
        _config(bound, dv_block=32, dv_threads=8, reduction_threads=16)
    )
    assert "block=(8, 16, 1)" in code
    assert "for synthetic_lane_2 in cutlass.range_constexpr(8):" in code
    assert code.count(COLUMN_REDUCE) == 1
    assert "pre=8, group_span=128, group_count=1)" in code


def test_mismatched_vector_width_keeps_rolled_lanes() -> None:
    # Eight elements per thread with V=4 is two vectors per lane, not one: the
    # established rolled nesting is kept.
    bound, _arguments = _bind()
    code = bound.to_code(_config(bound, dv_block=64, dv_threads=8, reduction_threads=8))
    assert "_lane_stash" not in code
    assert COLUMN_REDUCE not in code
    assert "for synthetic_lane_2 in range(16):" in code


def test_aliasing_inputs_reject_the_split(caplog: pytest.LogCaptureFixture) -> None:
    # ``q`` viewing the state storage removes the disjointness proof.  The
    # deferred vector store of ``state`` may not flush past the later load of
    # ``q`` (``demote_reordered_tile_vec_stores``, which logs the pair), so it
    # keeps its scalar form inside the element loop, which the register tile
    # cannot schedule (a structural rejection ahead of its aliasing proof); the
    # rolled nesting the kernel is regenerated with is then rejected by the
    # aliasing gate, as the loads of one lane may not move above another
    # lane's state store.  Each of the three stages names its reason.  The
    # example's step writes a new state tensor, which no input can alias, so
    # this test runs the in-place form of the same step.
    state = torch.empty((BH, D, DV), dtype=torch.float32)
    q, k, v, _state, alpha = _arguments(state)
    arguments = (state[:, :, 0], k, v, state, alpha)
    bound = cpu_bind(recurrent_step_in_place, arguments)
    with (
        caplog.at_level(logging.DEBUG, logger="helion._compiler.cute.memory_ops"),
        caplog.at_level(
            logging.DEBUG, logger="helion._compiler.cute.kernel_generators"
        ),
        pytest.raises(helion.exc.BackendUnsupported, match="aliasing write"),
    ):
        bound.to_code(_config(bound, dv_block=32, dv_threads=8, reduction_threads=8))
    assert (
        "deferred store of state restored to its scalar form: q accessed later"
        in caplog.text
    )
    assert FALLBACK_LOG in caplog.text
    assert "a store repeats inside an element loop" in caplog.text


def test_explicit_reduction_threads_stay_persistent() -> None:
    bound, _arguments = _bind()
    explicit = _config(bound, dv_block=64, dv_threads=16, reduction_threads=8)
    assert explicit.reduction_loops == [None]
    # An automatic reduction thread count that cannot cover the extent next to
    # 16 tile threads is still rolled.
    automatic = _config(bound, dv_block=64, dv_threads=16, reduction_threads=0)
    assert automatic.reduction_loops == [64]
    code = bound.to_code(explicit)
    assert "block=(16, 8, 1)" in code
    assert "for synthetic_lane_2 in cutlass.range_constexpr(16):" in code
    assert code.count(COLUMN_REDUCE) == 1
    assert "pre=16, group_span=128, group_count=1)" in code


def test_flattened_column_tile_forms_a_register_tile() -> None:
    # A single flattened tile axis (``PerThreadFlattenedTileStrategy``) with a
    # persistent row reduction gets the same lowering: the column axis on
    # thread_idx[0], one packet load per lane, one column reduce, and the
    # reduced columns stored once by the reduction-coordinate-zero threads.
    arguments = (torch.empty((128, 1024), dtype=torch.float32),)
    bound = cpu_bind(col_sum, arguments)
    config = column_config(bound, block=32, tile_threads=8, reduction_threads=8)
    assert config.reduction_loops == [None]
    code = bound.to_code(config)
    assert "block=(8, 8, 1)" in code
    assert (
        "offsets_base_0 = pid_flat * _BLOCK_SIZE_0 + cutlass.Int32(cute.arch.thread_idx()[0]) * 4"
        in code
    )
    assert "for synthetic_lane_1 in cutlass.range_constexpr(16):" in code
    assert code.count("cute.arch.load(") == 1
    assert code.count(COLUMN_REDUCE) == 1
    assert "pre=8, group_span=64, group_count=1)" in code
    assert "warp_reduction" not in code
    assert code.count("_cute_store_u32_vec(out.iterator") == 1
    assert "% 64 < 8:" in code


def test_reduction_derived_scale_is_emitted_in_the_consume_nest() -> None:
    # ``r = 1 / (sum(t * t) + 1)`` is lane-invariant but feeds the lane-varying
    # store of every row, so the consume nest must define it before the
    # products.
    arguments = (torch.empty((128, 1024), dtype=torch.float32),)
    bound = cpu_bind(col_scale, arguments)
    code = bound.to_code(
        column_config(bound, block=32, tile_threads=8, reduction_threads=8)
    )
    _assert_defined_before_use(code)
    accumulate, consume = _owner_loops(code)
    consume_source = ast.unparse(consume)
    assert consume_source.count("_lane_results[") == 1
    assert " / " in consume_source
    assert "_cute_store_u32_vec(y.iterator" in consume_source
    assert "_lane_results" not in ast.unparse(accumulate)
    assert code.count(COLUMN_REDUCE) == 1


@pytest.mark.parametrize(
    ("policy", "helper"),
    (
        ("l2_last", "_cute_load_l2_evict_last("),
        ("l1_l2_first", "_cute_load_l1_l2_evict_first("),
    ),
)
def test_eviction_policy_loads_form_a_register_tile(policy: str, helper: str) -> None:
    # The L2-policy packet helpers are loads like ``cute.arch.load``: the
    # register tile schedules them in its accumulate nest.
    arguments = (torch.empty((128, 1024), dtype=torch.float32),)
    bound = cpu_bind(col_sum, arguments)
    code = bound.to_code(
        column_config(
            bound,
            block=32,
            tile_threads=8,
            reduction_threads=8,
            load_eviction_policies=[policy],
        )
    )
    assert "for synthetic_lane_1 in cutlass.range_constexpr(16):" in code
    assert code.count(helper) == 1
    assert "cute.arch.load(" not in code
    assert code.count(COLUMN_REDUCE) == 1
    _assert_defined_before_use(code)


def test_nested_device_loop_keeps_the_rolled_lane() -> None:
    # The device IR shows a rolled loop inside the tile body, so the lane
    # nesting the register tile needs is never chosen: the established rolled
    # synthetic lane compiles instead of a late rejection.
    arguments = (
        torch.empty((128, 1024), dtype=torch.float32),
        torch.empty((4, 1024), dtype=torch.float32),
    )
    bound = cpu_bind(col_sum_nested, arguments)
    code = bound.to_code(
        column_config(bound, block=32, tile_threads=8, reduction_threads=8)
    )
    assert "for synthetic_lane_1 in range(16):" in code
    assert COLUMN_REDUCE not in code
    assert "_lane_stash" not in code


def test_two_reductions_share_one_register_tile() -> None:
    arguments = (torch.empty((128, 1024), dtype=torch.float32),)
    bound = cpu_bind(col_stats, arguments)
    code = bound.to_code(
        column_config(bound, block=32, tile_threads=8, reduction_threads=8)
    )
    _assert_defined_before_use(code)
    # One accumulate nest feeds both column reduces; both reduced tiles are
    # stored once (lane-invariant tail) by the reduction-coordinate-zero
    # threads.
    assert len(_owner_loops(code)) == 1
    assert code.count(COLUMN_REDUCE) == 2
    assert code.count("_cute_store_u32_vec(out.iterator") == 2
    assert code.count("% 64 < 8:") == 2


def test_single_vector_thread_uses_the_warp_column_path() -> None:
    # One vector thread under 64 consecutive reduction threads: the column
    # reduce folds each warp with a shuffle before the shared-memory exchange.
    arguments = (torch.empty((128, 1024), dtype=torch.float32),)
    bound = cpu_bind(col_sum, arguments)
    code = bound.to_code(
        column_config(bound, block=4, tile_threads=1, reduction_threads=64)
    )
    assert "for synthetic_lane_1 in cutlass.range_constexpr(2):" in code
    assert code.count(COLUMN_REDUCE) == 1
    # The shared-memory exchange is emitted keyed on the full runtime thread
    # id with a group count covering the largest launch block (a redundant
    # thread axis may still be mapped later in codegen); once the launch
    # shape is final, ``finalize_shared_reduce_groups`` sizes it for the one
    # 64-thread group that launches and keys it on ``thread_idx()[0]``.
    assert "block=(64, 1, 1)" in code
    assert "sum_1_lane_acc_lane = cutlass.Int32(cute.arch.thread_idx()[0])\n" in code
    assert "block_dim()" not in code
    assert "pre=1, group_span=64, group_count=1)" in code


def _rolled_code(bound: Any, config: helion.Config) -> str:
    """``config``'s code with the register tile never predicted: the rolled
    lane nesting the persistent strategy used before register tiles existed."""
    with patch(
        "helion._compiler.cute.thread_budget.register_tile_shape",
        return_value=False,
    ):
        return bound.to_code(config)


def _f32(*shape: int) -> torch.Tensor:
    return torch.empty(shape, dtype=torch.float32)


@pytest.mark.parametrize(
    ("kernel", "make_arguments", "owner_loops", "stashes", "once"),
    (
        # Load, max combine; ``exp(t - max)`` once in the level-1 nest (stashed
        # for the store), sum combine; the store nest reads it back.
        pytest.param(
            col_softmax,
            lambda: (_f32(128, 1024),),
            3,
            ("t_lane_stash", "v_1_lane_stash"),
            "cute.math.exp2(",
            id="column-softmax",
        ),
        # Load, sum combine; the squared deviations in the level-1 nest, sum
        # combine; both results flushed by the once-only tail (no store nest).
        pytest.param(
            col_var,
            lambda: (_f32(128, 1024),),
            2,
            ("t_lane_stash",),
            "cute.arch.load(x.iterator",
            id="two-pass-variance",
        ),
    ),
)
def test_chained_reductions_run_one_accumulate_nest_per_level(
    kernel: helion.Kernel,
    make_arguments: Callable[[], tuple[Any, ...]],
    owner_loops: int,
    stashes: tuple[str, ...],
    once: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # One reduction feeding another: the device IR admits the chain and the
    # schedule runs a second accumulate nest between the two combines, reading
    # the stashed tile instead of loading it again.
    arguments = make_arguments()
    bound = cpu_bind(kernel, arguments)
    config = column_config(bound, block=32, tile_threads=8, reduction_threads=8)
    with caplog.at_level(
        logging.DEBUG, logger="helion._compiler.cute.kernel_generators"
    ):
        code = bound.to_code(config)
    assert FALLBACK_LOG not in caplog.text
    assert ROLLED_OWNER not in code
    assert code.count(CONSTEXPR_OWNER) == owner_loops
    for stash in stashes:
        assert f"{stash} = []" in code
    assert code.count(once) == 1
    assert code.count("cute.arch.load(x.iterator") == 1
    statements = _kernel_statements(code)
    combines = [
        i
        for i, source in enumerate(statements)
        if "_reduced" in source and "_lane_acc" in source and "= _cute_" in source
    ]
    owners = [
        i for i, s in enumerate(statements) if s.startswith("for synthetic_lane_")
    ]
    # First nest, first combine(s), second nest, second combine, the rest.
    assert owners[0] < combines[0] < owners[1] < combines[-1]
    assert all(owner < combines[-1] for owner in owners[:2])
    assert all(owner > combines[-1] for owner in owners[2:])


def _kernel_statements(code: str) -> list[str]:
    """The kernel body's top-level statements, unparsed, in program order."""
    module = ast.parse(code)
    (kernel,) = [
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("_helion_")
    ]
    return [ast.unparse(node) for node in kernel.body]


@pytest.mark.parametrize(
    ("kernel", "make_arguments", "block", "reduction_threads", "vec", "capability"),
    (
        pytest.param(
            col_sum_guarded,
            lambda: (_f32(128, 1024), 1),
            32,
            8,
            4,
            (10, 0),
            id="guard-inside-the-element-loops",
        ),
        pytest.param(
            col_sum_f32_out,
            lambda: (torch.empty((128, 1024), dtype=torch.bfloat16),),
            64,
            16,
            8,
            (9, 0),
            id="bf16-tile-fp32-store",
        ),
        pytest.param(
            col_scale_three,
            lambda: (_f32(128, 1024), _f32(128, 1024), _f32(128, 1024)),
            32,
            8,
            4,
            (10, 0),
            id="three-stashed-tiles",
        ),
    ),
)
def test_unschedulable_bodies_fall_back_to_the_rolled_lane(
    kernel: helion.Kernel,
    make_arguments: Callable[[], tuple[Any, ...]],
    block: int,
    reduction_threads: int,
    vec: int,
    capability: tuple[int, int],
    caplog: pytest.LogCaptureFixture,
) -> None:
    # The device IR admits these bodies, so the register-tile nesting is
    # chosen, but the generated tile body has a statement the two-pass
    # schedule cannot place (a guard inside the element loops, a 32-byte output
    # packet on SM90, a live-value budget overrun). The kernel is regenerated with the
    # rolled lane nesting and compiles exactly as it did before register tiles
    # existed.
    arguments = make_arguments()
    bound = cpu_bind(kernel, arguments)
    bound.config_spec.target_device_capability = capability
    config = column_config(
        bound,
        block=block,
        tile_threads=8,
        reduction_threads=reduction_threads,
        vec=vec,
    )
    assert config.reduction_loops == [None]
    with caplog.at_level(
        logging.DEBUG, logger="helion._compiler.cute.kernel_generators"
    ):
        code = bound.to_code(config)
    assert FALLBACK_LOG in caplog.text
    # 16 reduction threads cover 128 rows without the register tile too, so
    # the retried config is unchanged and the reduction stays persistent.
    assert LOOPED_RETRY_LOG not in caplog.text
    assert ROLLED_OWNER in code
    assert CONSTEXPR_OWNER not in code
    assert "_lane_stash" not in code
    assert COLUMN_REDUCE not in code
    assert code == _rolled_code(bound, config)


def test_bf16_register_tile_flushes_fp32_results_on_sm100(
    caplog: pytest.LogCaptureFixture,
) -> None:
    arguments = (torch.empty((128, 1024), dtype=torch.bfloat16),)
    bound = cpu_bind(col_sum_f32_out, arguments)
    bound.config_spec.target_device_capability = (10, 0)
    config = column_config(bound, block=64, tile_threads=8, reduction_threads=16, vec=8)
    with caplog.at_level(
        logging.DEBUG, logger="helion._compiler.cute.kernel_generators"
    ):
        code = bound.to_code(config)
    assert FALLBACK_LOG not in caplog.text
    assert CONSTEXPR_OWNER in code
    assert ROLLED_OWNER not in code
    assert code.count(COLUMN_REDUCE) == 1
    assert "ir.VectorType.get([8], cutlass.Uint16.mlir_type)" in code
    assert code.count("_cute_store_u32_vec(out.iterator") == 1


@pytest.mark.parametrize(
    ("kernel", "make_arguments"),
    (
        pytest.param(
            col_sum_guarded,
            lambda: (_f32(1024, 1024), 1),
            id="guard-inside-the-element-loops",
        ),
        pytest.param(
            col_scale_three,
            lambda: (_f32(1024, 1024), _f32(1024, 1024), _f32(1024, 1024)),
            id="three-stashed-tiles",
        ),
    ),
)
def test_rejected_register_tiles_retry_as_the_looped_reduction(
    kernel: helion.Kernel,
    make_arguments: Callable[[], tuple[Any, ...]],
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Over 1024 rows with threads [8, 64] the register tile is the only reason
    # normalization keeps the reduction persistent: 64 threads cannot cover
    # the extent otherwise.  When the split rejects the body, the retry
    # re-normalizes the config with register tiles withheld and compiles the
    # looped reduction the geometry meant before register tiles existed
    # (reduction_loops=[128], 64 reduction threads), not a 32-thread lane
    # over 32 scalar rows each.
    arguments = make_arguments()
    bound = cpu_bind(kernel, arguments)
    config = column_config(bound, block=32, tile_threads=8, reduction_threads=64)
    assert config.reduction_loops == [None]
    with caplog.at_level(
        logging.DEBUG, logger="helion._compiler.cute.kernel_generators"
    ):
        code = bound.to_code(config)
    assert FALLBACK_LOG in caplog.text
    assert LOOPED_RETRY_LOG in caplog.text
    looped = column_config(
        bound, block=32, tile_threads=8, reduction_threads=64, reduction_loop=128
    )
    assert looped.reduction_loops == [128]
    assert code == bound.to_code(looped)
    assert "synthetic_lane_1" not in code
    assert "_REDUCTION_BLOCK_1" in code
    assert "block=(64, 8, 1)" in code


def _sinking(bound: Any, config: helion.Config) -> helion.Config:
    """``config`` with vector-loop sinking (``cute_vloop_sink``) switched on."""
    sinking = bound.config_spec.normalized_config(
        helion.Config.from_dict(
            {**config.config, "cute_vloop_sink": True, "cute_lane_unroll": 8}
        )
    )
    assert sinking.config["cute_vloop_sink"] is True
    return sinking


def _passes(bound: Any, config: helion.Config) -> tuple[str, list[helion.Config]]:
    """``config``'s code and the config of every codegen pass it took."""
    passes: list[helion.Config] = []
    generate = generate_ast_module._generate_ast

    def counting_generate(*args: Any, **kwargs: Any) -> ast.Module:
        passes.append(args[1])
        return generate(*args, **kwargs)

    with patch.object(generate_ast_module, "_generate_ast", counting_generate):
        code = bound.to_code(config)
    return code, passes


def test_vector_loop_sinking_leaves_the_register_tile_alone() -> None:
    # ``cute_vloop_sink`` interchanges a grid V-loop into the serial reduction
    # nest it wraps (``cute/sink_vector_loops.py``).  A register tile has no
    # such nest: its constexpr reduction lane sits outside the one-vector tile
    # loops and every load is already a packet, so the pass finds nothing to
    # sink, and with two thread axes the knob shapes no layout either.  The
    # knob-on code is the knob-off register tile, generated in one pass.
    bound, _arguments = _bind()
    config = _config(bound, dv_block=32, dv_threads=8, reduction_threads=8)
    code, passes = _passes(bound, _sinking(bound, config))
    assert len(passes) == 1
    assert "_lane_stash" in code
    assert all(
        _iter_name(loop) == "cutlass.range_constexpr" for loop in _kernel_loops(code)
    )
    assert COLUMN_REDUCE in code
    assert "_vsink_" not in code
    assert code == bound.to_code(config)


def test_the_looped_retry_keeps_the_sink_knob() -> None:
    # The retry after a rejected register tile re-normalizes the config with
    # register tiles withheld and nothing else: ``cute_vloop_sink`` survives
    # into the looped pass, where the sink hook runs on the rolled body (this
    # grid records no V-loop wrapper, so it leaves the body untouched).  The
    # knob-on code is the knob-off looped code.
    arguments = (_f32(1024, 1024), 1)
    bound = cpu_bind(col_sum_guarded, arguments)
    config = column_config(bound, block=32, tile_threads=8, reduction_threads=64)
    code, passes = _passes(bound, _sinking(bound, config))
    assert [attempt.reduction_loops for attempt in passes] == [[None], [128]]
    assert all(attempt.config["cute_vloop_sink"] is True for attempt in passes)
    assert "_REDUCTION_BLOCK_1" in code
    assert "_vsink_" not in code
    assert code == bound.to_code(config)


@helion.kernel(backend="triton", static_shapes=True, autotune_effort="none")
def add_rows(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size()):
        out[tile] = x[tile] + y[tile]
    return out


@skipIfRefEager("the codegen passes this test retries do not run in ref mode")
def test_the_retry_starts_the_memory_transforms_afresh() -> None:
    # Inductor's prologue transform remembers the fused inputs whose
    # placeholder it already emitted so a second load of the same input reuses
    # the result variable.  That memory belongs to one codegen pass: the pass
    # a backend's retry policy (the register-tile split) rejects is thrown
    # away, so the regenerated pass must emit every placeholder again.  The
    # record lives on the pass's ``GenerateAST``, which the retry creates anew.
    arguments = (_f32(64, 64), _f32(64, 64))
    bound = cpu_bind(add_rows, arguments)
    host = bound.host_function
    assert host is not None
    passes: list[list[dict[str, str]]] = []
    repairs: list[BaseException] = []
    generate = generate_ast_module._generate_ast

    @dataclasses.dataclass(frozen=True)
    class RetryOnce(RetryPolicy):
        signals: tuple[type[BaseException], ...] = (RegisterTileUnsupported,)

        def repair(
            self, signal: BaseException, config: helion.Config, *, caller_graphs: bool
        ) -> helion.Config | None:
            # One repair only: a second signal propagates and fails the test
            # instead of regenerating forever.
            if repairs:
                return None
            repairs.append(signal)
            return config

    def generate_failing_once(*args: Any, **kwargs: Any) -> ast.Module:
        passes.append([])
        module = generate(*args, **kwargs)
        if len(passes) == 1:
            raise RegisterTileUnsupported("the test rejects the first pass")
        return module

    def remember_loads(
        state: Any,
        tensor: torch.Tensor,
        subscript: list[object],
        extra_mask: object,
        eviction_policy: object,
        cache_modifier: object,
        codegen_load: Callable[..., ast.expr],
    ) -> ast.expr:
        record = state.codegen.prologue_first_indexing
        passes[-1].append(dict(record))
        record.setdefault(state.device_function.tensor_arg(tensor).name, "emitted")
        return codegen_load(
            state, tensor, [*subscript], extra_mask, eviction_policy, cache_modifier
        )

    with (
        bound.env,
        patch.object(generate_ast_module, "_generate_ast", generate_failing_once),
        patch.object(Backend, "retry_policy", return_value=RetryOnce()),
    ):
        generate_ast_module.generate_ast(
            host,
            bound.config_spec.default_config(),
            False,
            load_transform=remember_loads,
        )
    assert len(passes) == 2
    assert passes[0] == passes[1] == [{}, {"x": "emitted"}]


def test_register_budget_keeps_wide_stashes_rolled(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Three loaded tiles kept live until the consume pass (3 x 16 lanes x 4
    # elements) plus the accumulators exceed the live-value budget; half the
    # lanes per thread fit.
    arguments = (_f32(128, 1024), _f32(128, 1024), _f32(128, 1024))
    bound = cpu_bind(col_scale_three, arguments)
    with caplog.at_level(
        logging.DEBUG, logger="helion._compiler.cute.kernel_generators"
    ):
        code = bound.to_code(
            column_config(bound, block=32, tile_threads=8, reduction_threads=8)
        )
    assert "live values exceed" in caplog.text
    assert ROLLED_OWNER in code
    assert "_lane_stash" not in code
    caplog.clear()
    with caplog.at_level(
        logging.DEBUG, logger="helion._compiler.cute.kernel_generators"
    ):
        code = bound.to_code(
            column_config(bound, block=32, tile_threads=8, reduction_threads=16)
        )
    assert FALLBACK_LOG not in caplog.text
    assert code.count("_lane_stash = []") == 3
    _assert_defined_before_use(code)


def test_eight_byte_policy_loads_are_stashed() -> None:
    # fp32 V=2 under ``l1_l2_first`` loads through the 8-byte policy helper.
    # It is a load like the 16-byte form: the loaded tile is stashed for the
    # consume nest instead of being re-loaded behind the barrier.
    arguments = (_f32(128, 1024), _f32(128, 1024))
    bound = cpu_bind(col_scale_into, arguments)
    code = bound.to_code(
        column_config(
            bound,
            block=16,
            tile_threads=8,
            reduction_threads=8,
            vec=2,
            load_eviction_policies=["l1_l2_first"],
        )
    )
    assert code.count("_cute_load_l1_l2_evict_first_8b(") == 1
    assert code.count("_lane_stash = []") == 1
    assert "cute.arch.load(" not in code
    assert code.count(COLUMN_REDUCE) == 1
    _assert_defined_before_use(code)


@pytest.mark.parametrize(
    ("vec", "block", "policy"),
    ((2, 16, None), (4, 32, "l1_l2_first"), (2, 16, "l1_l2_first")),
    ids=("plain-loads", "16-byte-helper", "8-byte-helper"),
)
def test_aliasing_views_reject_every_load_form(
    vec: int, block: int, policy: str | None, caplog: pytest.LogCaptureFixture
) -> None:
    # ``x`` and ``y`` overlap, so a lane's load may alias another lane's
    # store.  Neither the register tile (its aliasing proof, the store being
    # the tile's last access) nor the rolled split may reorder them,
    # whichever helper the load goes through.
    base = _f32(129, 1024)
    arguments = (base[:128], base[1:])
    bound = cpu_bind(col_scale_into, arguments)
    extra: dict[str, Any] = (
        {} if policy is None else {"load_eviction_policies": [policy]}
    )
    with (
        caplog.at_level(
            logging.DEBUG, logger="helion._compiler.cute.kernel_generators"
        ),
        pytest.raises(helion.exc.BackendUnsupported, match="aliasing write"),
    ):
        bound.to_code(
            column_config(
                bound,
                block=block,
                tile_threads=8,
                reduction_threads=8,
                vec=vec,
                **extra,
            )
        )
    assert "a load of one lane may alias a store of another lane" in caplog.text


def test_load_helper_names_match_the_backend_imports() -> None:
    # Every ``_cute_load_*`` helper the backend binds in generated code is in
    # one of the two lists the lane-loop passes read: the cache-hinted loads
    # (relocation, load collection and the aliasing proofs) or the predicated
    # packet loads of the sunk V-loop (``MASKED_PACKET_LOAD_HELPER_NAMES``,
    # which the relocation proof reads; they appear only in bodies the sink
    # already unrolled), so a new helper cannot slip past them.
    bound, _arguments = _bind()
    bound_names = {
        name
        for name in bound.env.backend.library_imports
        if name.startswith("_cute_load_")
    }
    assert bound_names == (
        _CUTE_CACHE_LOAD_HELPER_NAMES | MASKED_PACKET_LOAD_HELPER_NAMES
    )


def test_control_flow_names_are_tracing_ops() -> None:
    for name in _CUTE_REGISTER_TILE_CONTROL_FLOW:
        assert isinstance(getattr(_tracing_ops, name), types.FunctionType)


def test_column_reduce_returns_behind_a_barrier() -> None:
    # Both branches of the column reduce read other threads' slots after one
    # barrier and must not return before a second one: a persistent CTA calls
    # the helper once per tile with no other barrier per trip, and the GPU
    # test of that shape cannot catch a missing barrier reliably.
    path = pathlib.Path(helion.__file__).parent / "_compiler/cute/reduce_helpers.py"
    module = ast.parse(path.read_text())
    (function,) = [
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_cute_grouped_reduce_shared_columns"
    ]
    returns = 0
    for node in ast.walk(function):
        for field in ("body", "orelse"):
            statements = getattr(node, field, None)
            if not isinstance(statements, list):
                continue
            for previous, statement in itertools.pairwise(statements):
                if isinstance(statement, ast.Return):
                    returns += 1
                    assert ast.unparse(previous) == "cute.arch.sync_threads()"
    # The warp-fold branch (``fold='warp'``), the one-warp-per-slot branch
    # (``pre == 1``) and the serial ``pre > 1`` branch.
    assert returns == 3


def test_register_tile_blocks_follow_the_body_and_the_extent() -> None:
    # Config normalization, the seed and codegen share one admission: the
    # reduction block is listed only for a body the schedule may lower (no
    # rolled loop, no effects beyond loads, stores and per-element values;
    # chained reductions take one accumulate nest per level) and a static
    # (unmasked) extent.
    for kernel, arguments, admitted in (
        (col_sum, (_f32(128, 1024),), True),
        (col_stats, (_f32(128, 1024),), True),
        (col_scale, (_f32(128, 1024),), True),
        (col_softmax, (_f32(128, 1024),), True),
        (col_var, (_f32(128, 1024),), True),
        (col_sum_nested, (_f32(128, 1024), _f32(4, 1024)), False),
        (col_sum_dynamic, (_f32(1024, 1024),), False),
        (rms_norm_bwd, rms_norm_bwd_arguments(), False),
    ):
        bound = cpu_bind(kernel, arguments)
        (reduction,) = [
            block.block_id for block in bound.env.block_sizes if block.reduction
        ]
        blocks = bound.config_spec.cute_register_tile_reduction_blocks
        assert (reduction in blocks) is admitted, kernel.name


def test_tile_axis_reductions_keep_the_rolled_nesting(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # The autodiff rms_norm backward reduces over the row tile itself for
    # ``grad_weight`` beside its two chained feature reductions.  Under the
    # register-tile nesting that marker is owned by the row tile's vector lane,
    # which the schedule cannot place, so the admission refuses the body: no
    # register-tile seed is promoted and the autotune-off default renders the
    # warp-reduction layout it had before chained bodies were admitted.
    arguments = rms_norm_bwd_arguments()
    bound = cpu_bind(rms_norm_bwd, arguments)
    spec = bound.config_spec
    assert not spec.cute_register_tile_reduction_blocks
    assert "cute_register_tile" not in spec.autotuner_heuristics
    assert spec.compiler_default_config is None
    default = spec.default_config()
    assert default.config["cute_vector_widths"] == [1, 1]
    assert "num_threads" not in default.config
    with caplog.at_level(
        logging.DEBUG, logger="helion._compiler.cute.kernel_generators"
    ):
        code = bound.to_code(default)
    assert FALLBACK_LOG not in caplog.text
    assert "block=(32, 32, 1)" in code
    assert "cute.arch.warp_reduction_sum(" in code
    assert CONSTEXPR_OWNER not in code
    assert COLUMN_REDUCE not in code
    assert "_lane_stash" not in code


def test_tile_axis_reduction_under_a_register_tile_falls_back(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Were the device IR to admit the body all the same, the lowering fails
    # closed: the row tile's one-vector wrapper under the register-tile owner
    # carries the tile-axis reduction, so the lane interchange raises
    # ``RegisterTileUnsupported`` (not the vector lane's ownership proof) and
    # ``generate_ast`` regenerates with the rolled nesting, which rejects this
    # config as it always did (a rolled reduction lane below the vectorized
    # row tile), instead of a kernel-level failure of the seed's default.
    arguments = rms_norm_bwd_arguments()
    with patch(
        "helion._compiler.cute.register_tile_admission.register_tile_body_admitted",
        return_value=True,
    ):
        bound = cpu_bind(rms_norm_bwd, arguments)
    spec = bound.config_spec
    (reduction,) = [
        block.block_id for block in bound.env.block_sizes if block.reduction
    ]
    assert reduction in spec.cute_register_tile_reduction_blocks
    assert "cute_register_tile" in spec.autotuner_heuristics
    seed = spec.default_config()
    assert seed.config["num_threads"] == [8, 4]
    assert seed.config["cute_vector_widths"] == [1, 4]
    with (
        caplog.at_level(
            logging.DEBUG, logger="helion._compiler.cute.kernel_generators"
        ),
        pytest.raises(
            helion.exc.BackendUnsupported, match="nested in a different lane owner"
        ),
    ):
        bound.to_code(seed)
    assert FALLBACK_LOG in caplog.text
    assert "owned by that tile lane, not by the register-tile owner" in caplog.text
    assert LOOPED_RETRY_LOG not in caplog.text


def test_dynamic_extent_forces_the_looped_reduction() -> None:
    # With ``static_shapes=False`` a persistent lane loop would bake its lane
    # count from the size hint and a later, larger extent would drop rows; a
    # shrunk thread count is forced looped, with the trip count computed from
    # the runtime extent on the host.
    arguments = (_f32(1024, 1024),)
    bound = cpu_bind(col_sum_dynamic, arguments)
    config = column_config(bound, block=32, tile_threads=8, reduction_threads=64)
    assert config.reduction_loops == [128]
    code = bound.to_code(config)
    assert "synthetic_lane_1" not in code
    assert (
        "_REDUCTION_TRIPS_1 = (m + _REDUCTION_BLOCK_1 - 1) // _REDUCTION_BLOCK_1"
        in code
    )
    assert (
        "cutlass.Int32(_REDUCTION_TRIPS_1 * _REDUCTION_BLOCK_1), "
        "cutlass.Int32(_REDUCTION_BLOCK_1)"
    ) in code


def _access(source: str) -> tuple[str, int] | None:
    call = ast.parse(source, mode="eval").body
    assert isinstance(call, ast.Call)
    access = _generated_access_pointer(call)
    return None if access is None else (ast.unparse(access[0]), access[1])


def test_access_pointer_widths() -> None:
    # The aliasing proofs see the pointer and element count of every generated
    # access form; byte-packed word loads hide their element count and fail
    # closed.
    vector = "ir.VectorType.get([4], cutlass.Float32.mlir_type)"
    assert _access(f"cute.arch.load(p, {vector})") == ("p", 4)
    assert _access(f"_cute_load_l2_evict_last(p, {vector})") == ("p", 4)
    assert _access(
        "_cute_load_l1_l2_evict_first(p, ir.VectorType.get([8], cutlass.Uint16.mlir_type))"
    ) == ("p", 8)
    assert _access(
        "_cute_load_l1_l2_evict_last_8b(p, ir.VectorType.get([2], cutlass.Uint32.mlir_type))"
    ) == ("p", 2)
    assert _access(
        "cute.arch.load(p, cutlass.Float32, level1_eviction_priority='evict_first')"
    ) == ("p", 1)
    assert _access("cute.arch.load(p, cutlass.Uint64)") is None
    assert _access("cute.arch.load(p, cutlass.Uint32)") is None
    assert _access("p.load()") == ("p", 1)
    assert _access("p.store(v)") == ("p", 1)
    assert _access("_cute_store_u32_vec(p, vals)") == ("p", 4)
    assert _access("_cute_store_u16x8_l2_evict_last(p, vals)") == ("p", 8)
    assert _access("cute.arch.atomic_add(p, v)") is None


def test_register_tile_seed_is_the_default_config() -> None:
    # The autotune-off default is the stride-aware layout the seed describes:
    # the stride-1 dv axis on vectorized lanes (8 threads x 4 fp32), the
    # strided D reduction on the remaining threads, instead of the base
    # default's reduction on the fastest thread index.
    bound, _arguments = _bind()
    spec = bound.config_spec
    assert "cute_register_tile" in spec.autotuner_heuristics
    assert spec.compiler_default_config is not None
    default = spec.default_config()
    (reduction,) = [
        block.block_id for block in bound.env.block_sizes if block.reduction
    ]
    (dv,) = spec.block_sizes.valid_block_ids()
    thread_ids = spec.num_threads.valid_block_ids()
    vector_ids = spec.cute_vector_widths.valid_block_ids()
    num_threads = default.config["num_threads"]
    vector_widths = default.config["cute_vector_widths"]
    assert default.config["block_sizes"] == [32]
    assert default.config["reduction_loops"] == [None]
    assert num_threads[thread_ids.index(dv)] == 8
    assert num_threads[thread_ids.index(reduction)] == 8
    assert vector_widths[vector_ids.index(dv)] == 4
    code = bound.to_code(default)
    assert COLUMN_REDUCE in code
    assert "block=(8, 8, 1)" in code


def test_default_stays_the_base_config_without_the_register_tile_shape() -> None:
    # A dv extent the vector tile does not divide has no register-tile
    # geometry: nothing is promoted, the base default stands.
    state = torch.empty((BH, D, 48), dtype=torch.float32)
    arguments = (
        torch.empty((BH, D), dtype=torch.float32),
        torch.empty((BH, D), dtype=torch.float32),
        torch.empty((BH, 48), dtype=torch.float32),
        state,
        torch.empty((BH,), dtype=torch.float32),
    )
    bound, _arguments = _bind(arguments)
    spec = bound.config_spec
    assert "cute_register_tile" not in spec.autotuner_heuristics
    assert spec.compiler_default_config is None


# GPU numerics for register-tile lane reductions (see
# ``test_cute_register_tile_reductions.py`` for the codegen shape).


BH_DEVICE = 256


def _check_recurrent_step(
    *,
    dtype: torch.dtype,
    dv_block: int,
    dv_threads: int,
    reduction_threads: int,
    vec: int = 4,
    **extra: Any,
) -> str:
    """Run the fused recurrent step with a register-tile config on two input
    sets (a stale output or new state cannot pass twice) and return the
    generated code."""
    kernel = helion.kernel(
        recurrent_step_fused.fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
        ignore_warnings=[helion.exc.TensorOperationInWrapper],
    )
    torch.manual_seed(0)
    q = torch.randn(BH_DEVICE, D, device=DEVICE).to(dtype)
    k = torch.randn(BH_DEVICE, D, device=DEVICE).to(dtype)
    v = torch.randn(BH_DEVICE, DV, device=DEVICE).to(dtype)
    alpha = torch.rand(BH_DEVICE, device=DEVICE).to(dtype)
    initial = torch.randn(BH_DEVICE, D, DV, device=DEVICE).to(dtype)
    state = initial.clone()
    bound = kernel.bind((q, k, v, state, alpha))
    config = _config(
        bound,
        dv_block=dv_block,
        dv_threads=dv_threads,
        reduction_threads=reduction_threads,
        vec=vec,
        **extra,
    )
    code = bound.to_code(config)
    assert "_lane_stash" in code
    compiled = bound.compile_config(config)
    # The kernel computes in fp32 and rounds the state and the output once.
    rtol, atol = (1e-3, 1e-3) if dtype is torch.float32 else (2e-2, 2e-1)
    for _ in range(2):
        out, new_state = compiled(q, k, v, state, alpha)
        torch.cuda.synchronize()
        expected_state = (
            alpha.float()[:, None, None] * initial.float()
            + k.float()[:, :, None] * v.float()[:, None, :]
        )
        expected_out = (q.float()[:, :, None] * expected_state).sum(1)
        torch.testing.assert_close(
            new_state.float(), expected_state, rtol=rtol, atol=atol
        )
        torch.testing.assert_close(out.float(), expected_out, rtol=rtol, atol=atol)
        assert torch.equal(state, initial)
        q.copy_(torch.rand_like(q, dtype=torch.float32).mul_(2).sub_(1).to(dtype))
        v.copy_(torch.rand_like(v, dtype=torch.float32).mul_(2).sub_(1).to(dtype))
    return code


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.cuda
@pytest.mark.parametrize(
    ("dv_block", "dv_threads", "reduction_threads"),
    (
        pytest.param(32, 8, 8, id="one-warp-columns-64-elements"),
        pytest.param(32, 8, 16, id="four-warp-columns"),
        pytest.param(64, 16, 8, id="two-warp-wide-tile"),
    ),
)
def test_register_tile_matches_reference(
    dv_block: int, dv_threads: int, reduction_threads: int
) -> None:
    _check_recurrent_step(
        dtype=torch.float32,
        dv_block=dv_block,
        dv_threads=dv_threads,
        reduction_threads=reduction_threads,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.cuda
@pytest.mark.parametrize("pid_type", ("persistent_blocked", "persistent_interleaved"))
def test_persistent_pid_reuses_the_column_reduce_safely(pid_type: str) -> None:
    # A persistent CTA runs the column reduce once per tile with no other
    # barrier in between; the helper's trailing barrier keeps a fast warp's
    # next-tile partials from overwriting slots a slower warp still reads.
    # The race is timing dependent, so a pass here does not prove the barrier
    # is present: ``test_column_reduce_returns_behind_a_barrier`` pins it in
    # the helper's source; this test checks the persistent register tile runs
    # and its numerics.
    code = _check_recurrent_step(
        dtype=torch.float32,
        dv_block=32,
        dv_threads=8,
        reduction_threads=16,
        pid_type=pid_type,
    )
    assert "for virtual_pid in range(" in code
    assert code.count(COLUMN_REDUCE) == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.cuda
def test_bf16_register_tile_flushes_u16_packets() -> None:
    # bf16 state with V=8: one 16-byte packet per lane and a u16 flush.
    code = _check_recurrent_step(
        dtype=torch.bfloat16,
        dv_block=64,
        dv_threads=8,
        reduction_threads=16,
        vec=8,
    )
    assert "ir.VectorType.get([8], cutlass.Uint16.mlir_type)" in code
    assert code.count("_cute_store_u16_vec(new_state.iterator") == 1


def _col_sum_reference(x: torch.Tensor) -> torch.Tensor:
    return x.sum(0)


def _col_scale_reference(x: torch.Tensor) -> torch.Tensor:
    return x / ((x * x).sum(0) + 1.0)[None, :]


def _col_stats_reference(x: torch.Tensor) -> torch.Tensor:
    return torch.stack([x.sum(0), (x * x).sum(0)])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.cuda
@pytest.mark.parametrize(
    ("kernel", "reference", "block", "tile_threads", "reduction_threads", "extra"),
    (
        pytest.param(col_sum, _col_sum_reference, 32, 8, 8, {}, id="col-sum"),
        pytest.param(
            col_sum,
            _col_sum_reference,
            4,
            1,
            64,
            {},
            id="col-sum-warp-columns",
        ),
        pytest.param(
            col_sum,
            _col_sum_reference,
            4,
            1,
            64,
            {"pid_type": "persistent_blocked"},
            id="col-sum-warp-columns-persistent",
        ),
        pytest.param(
            col_sum,
            _col_sum_reference,
            32,
            8,
            8,
            {"load_eviction_policies": ["l2_last"]},
            id="col-sum-l2-last",
        ),
        pytest.param(
            col_sum,
            _col_sum_reference,
            32,
            8,
            8,
            {"load_eviction_policies": ["l1_l2_first"]},
            id="col-sum-l1-l2-first",
        ),
        pytest.param(col_scale, _col_scale_reference, 32, 8, 8, {}, id="col-scale"),
        pytest.param(
            col_scale,
            _col_scale_reference,
            32,
            8,
            8,
            {"pid_type": "persistent_interleaved"},
            id="col-scale-persistent",
        ),
        pytest.param(col_stats, _col_stats_reference, 32, 8, 8, {}, id="col-stats"),
    ),
)
def test_column_register_tiles_match_reference(
    kernel: helion.Kernel,
    reference: Callable[[torch.Tensor], torch.Tensor],
    block: int,
    tile_threads: int,
    reduction_threads: int,
    extra: dict[str, Any],
) -> None:
    torch.manual_seed(0)
    x = torch.randn(128, 16384, device=DEVICE)
    bound = kernel.bind((x,))
    config = column_config(
        bound,
        block=block,
        tile_threads=tile_threads,
        reduction_threads=reduction_threads,
        **extra,
    )
    code = bound.to_code(config)
    assert COLUMN_REDUCE in code
    compiled = bound.compile_config(config)
    for _ in range(2):
        out = compiled(x)
        torch.cuda.synchronize()
        torch.testing.assert_close(out, reference(x), rtol=1e-4, atol=1e-3)
        x.uniform_(-1, 1)


def _col_sum_f32_reference(x: torch.Tensor) -> torch.Tensor:
    return x.float().sum(0)


def _col_scale_three_reference(
    x: torch.Tensor, u: torch.Tensor, w: torch.Tensor
) -> torch.Tensor:
    r = 1.0 / ((x * x).sum(0) + 1.0)[None, :]
    return x * r + u * r + w


def _randn_like_inputs(inputs: tuple[torch.Tensor, ...]) -> None:
    for tensor in inputs:
        tensor.copy_(
            torch.rand_like(tensor, dtype=torch.float32)
            .mul_(2)
            .sub_(1)
            .to(tensor.dtype)
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.cuda
@pytest.mark.parametrize(
    ("kernel", "reference", "dtype", "inputs", "block", "reduction_threads", "vec"),
    (
        pytest.param(
            col_sum_f32_out,
            _col_sum_f32_reference,
            torch.bfloat16,
            1,
            64,
            16,
            8,
            id="col-sum-bf16-to-fp32",
        ),
        pytest.param(
            col_scale_three,
            _col_scale_three_reference,
            torch.float32,
            3,
            32,
            8,
            4,
            id="col-scale-three-tiles",
        ),
    ),
)
def test_supported_store_and_rolled_fallback_match_reference(
    kernel: helion.Kernel,
    reference: Callable[..., torch.Tensor],
    dtype: torch.dtype,
    inputs: int,
    block: int,
    reduction_threads: int,
    vec: int,
) -> None:
    # SM100 can pack eight widened fp32 results into one 32-byte output packet.
    # Older targets and live-value budget overruns retain rolled fallback.
    torch.manual_seed(0)
    tensors = tuple(
        torch.randn(128, 16384, device=DEVICE).to(dtype) for _ in range(inputs)
    )
    bound = kernel.bind(tensors)
    config = column_config(
        bound, block=block, tile_threads=8, reduction_threads=reduction_threads, vec=vec
    )
    assert config.reduction_loops == [None]
    code = bound.to_code(config)
    wide_store = dtype is torch.bfloat16 and torch.cuda.get_device_capability() >= (
        10,
        0,
    )
    if wide_store:
        assert "for synthetic_lane_1 in cutlass.range_constexpr(" in code
        assert COLUMN_REDUCE in code
        assert code.count("_cute_store_u32_vec(out.iterator") == 1
    else:
        assert "for synthetic_lane_1 in range(" in code
    assert "_lane_stash" not in code
    compiled = bound.compile_config(config)
    for _ in range(2):
        out = compiled(*tensors)
        torch.cuda.synchronize()
        torch.testing.assert_close(
            out.float(), reference(*tensors).float(), rtol=1e-4, atol=1e-4
        )
        _randn_like_inputs(tensors)


def _col_sum_guarded_reference(x: torch.Tensor, flag: int) -> torch.Tensor:
    return x.sum(0) * (2.0 if flag > 0 else 1.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.cuda
@pytest.mark.parametrize(
    ("kernel", "reference", "inputs", "extra"),
    (
        pytest.param(
            col_sum_guarded,
            _col_sum_guarded_reference,
            1,
            (1,),
            id="guard-inside-the-element-loops",
        ),
        pytest.param(
            col_scale_three,
            _col_scale_three_reference,
            3,
            (),
            id="three-stashed-tiles",
        ),
    ),
)
def test_looped_retry_matches_reference(
    kernel: helion.Kernel,
    reference: Callable[..., torch.Tensor],
    inputs: int,
    extra: tuple[int, ...],
) -> None:
    # Over 1024 rows with threads [8, 64] the register tile is the only reason
    # the config keeps the reduction persistent.  When the split rejects the
    # body, the retry compiles the looped reduction (reduction_loops=[128],
    # 64 reduction threads) the geometry meant before register tiles existed.
    torch.manual_seed(0)
    tensors = tuple(torch.randn(1024, 4096, device=DEVICE) for _ in range(inputs))
    bound = kernel.bind((*tensors, *extra))
    config = column_config(bound, block=32, tile_threads=8, reduction_threads=64)
    assert config.reduction_loops == [None]
    code = bound.to_code(config)
    assert "synthetic_lane_1" not in code
    assert "_REDUCTION_BLOCK_1" in code
    assert "block=(64, 8, 1)" in code
    compiled = bound.compile_config(config)
    for _ in range(2):
        out = compiled(*tensors, *extra)
        torch.cuda.synchronize()
        torch.testing.assert_close(
            out, reference(*tensors, *extra), rtol=1e-4, atol=1e-4
        )
        _randn_like_inputs(tensors)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.cuda
@pytest.mark.parametrize("reduction_threads", (8, 16))
def test_column_softmax_chains_two_register_tile_levels(
    reduction_threads: int,
) -> None:
    # One reduction feeding another: the tile is loaded once, the max is
    # combined, ``exp(t - max)`` is computed once more in a second accumulate
    # nest and stashed for the store, then the sum is combined.
    torch.manual_seed(0)
    x = torch.randn(128, 16384, device=DEVICE)
    bound = col_softmax.bind((x,))
    config = column_config(
        bound, block=32, tile_threads=8, reduction_threads=reduction_threads
    )
    assert config.reduction_loops == [None]
    code = bound.to_code(config)
    assert "for synthetic_lane_1 in range(" not in code
    assert code.count("for synthetic_lane_1 in cutlass.range_constexpr(") == 3
    assert code.count("cute.math.exp2(") == 1
    assert code.count("cute.arch.load(x.iterator") == 1
    compiled = bound.compile_config(config)
    for _ in range(2):
        out = compiled(x)
        torch.cuda.synchronize()
        torch.testing.assert_close(out, torch.softmax(x, dim=0), rtol=1e-4, atol=1e-5)
        x.uniform_(-1, 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.cuda
def test_two_pass_variance_chains_two_register_tile_levels() -> None:
    # The second reduction reads the first one's mean: level-1 accumulate
    # nest from the stashed tile, both results flushed by the once-only tail.
    torch.manual_seed(0)
    x = torch.randn(128, 16384, device=DEVICE)
    bound = col_var.bind((x,))
    config = column_config(bound, block=32, tile_threads=8, reduction_threads=16)
    code = bound.to_code(config)
    assert code.count("for synthetic_lane_1 in cutlass.range_constexpr(") == 2
    assert code.count(COLUMN_REDUCE) == 2
    compiled = bound.compile_config(config)
    for _ in range(2):
        out = compiled(x)
        torch.cuda.synchronize()
        torch.testing.assert_close(out, x.var(0, unbiased=False), rtol=1e-4, atol=1e-5)
        x.uniform_(-1, 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.cuda
def test_eight_byte_policy_register_tile_matches_reference() -> None:
    # fp32 V=2 under ``l1_l2_first``: the 8-byte policy helper is a load the
    # register tile stashes for its consume nest.
    torch.manual_seed(0)
    x = torch.randn(128, 16384, device=DEVICE)
    y = torch.empty_like(x)
    bound = col_scale_into.bind((x, y))
    config = column_config(
        bound,
        block=16,
        tile_threads=8,
        reduction_threads=8,
        vec=2,
        load_eviction_policies=["l1_l2_first"],
    )
    code = bound.to_code(config)
    assert code.count("_cute_load_l1_l2_evict_first_8b(") == 1
    assert "_lane_stash" in code
    compiled = bound.compile_config(config)
    for _ in range(2):
        compiled(x, y)
        torch.cuda.synchronize()
        torch.testing.assert_close(y, _col_scale_reference(x), rtol=1e-4, atol=1e-3)
        x.uniform_(-1, 1)
        y.zero_()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.cuda
def test_dynamic_extent_binding_covers_larger_extents() -> None:
    # ``static_shapes=False``: the shrunk reduction thread count is forced
    # looped, so a binding made at 1024 rows reads the extent at runtime and
    # covers 2048 rows too instead of baking a 32-lane loop from the hint.
    torch.manual_seed(0)
    small = torch.randn(1024, 4096, device=DEVICE)
    bound = col_sum_dynamic.bind((small,))
    config = column_config(bound, block=32, tile_threads=8, reduction_threads=64)
    assert config.reduction_loops == [128]
    compiled = bound.compile_config(config)
    for x in (small, torch.randn(2048, 4096, device=DEVICE)):
        out = compiled(x)
        torch.cuda.synchronize()
        torch.testing.assert_close(out, x.sum(0), rtol=1e-4, atol=1e-3)
