"""Register-tile lowering of lane reductions over one-vector tiles (CPU codegen).

The fused linear-attention recurrent step loads a ``[D, dv]`` state tile,
updates it in place (``state = state * decay + k[:, None] * v[None, :]``) and
reduces ``q[:, None] * state`` over ``D``.  With the reduction split across a
persistent synthetic lane loop the CuTe backend used to nest that lane loop
inside the tile's constexpr V-loop, so the state load stayed scalar, each lane
iteration waited for its own load, and the same-index read-modify-write was
rejected by the aliasing gate because the tensor strides were not proven.

Now the scalar reduction lane nests OUTSIDE the one-vector tile wrappers as a
trace-time register tile: one packet load per lane, every lane's loads issued
before the first store, per-tile-element accumulators combined across the
thread group once with the shared-memory column reduce, and the vectorized
tile axis on ``thread_idx[0]`` so a warp's packet loads cover whole rows.
"""

from __future__ import annotations

import ast
import itertools
import logging
import pathlib
import types
from typing import TYPE_CHECKING
from typing import Any
from unittest.mock import patch

from examples.linear.linear_attention_engine import recurrent_step_fused
import pytest
import torch

from test._cute_binding import _cpu_bind
from test._cute_binding import _mock_cuda_unavailable
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

import helion
from helion._compiler import generate_ast as generate_ast_module
from helion._compiler.cute.cache_policy_loads import _CUTE_CACHE_LOAD_HELPER_NAMES
from helion._compiler.cute.persistent_branch_vec import _generated_access_pointer
from helion._compiler.cute.register_tile_admission import (
    _CUTE_REGISTER_TILE_CONTROL_FLOW,
)
from helion._compiler.cute.register_tile_admission import RegisterTileUnsupported
from helion._compiler.cute.thread_budget import MAX_THREADS_PER_BLOCK
from helion._testing import skipIfRefEager
from helion._testing import skipUnlessBackends
import helion.language as hl
from helion.language import _tracing_ops

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterator

BH, D, DV = 8, 128, 128
COLUMN_REDUCE = "_cute_grouped_reduce_shared_columns("
CONSTEXPR_OWNER = "for synthetic_lane_1 in cutlass.range_constexpr("
ROLLED_OWNER = "for synthetic_lane_1 in range("
FALLBACK_LOG = "regenerating with the rolled reduction lane"
LOOPED_RETRY_LOG = "the retry loops the reduction"


@pytest.fixture(autouse=True)
def _cpu_only() -> Iterator[None]:
    with (
        _mock_cuda_unavailable(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CPU-only test")),
        patch(
            "helion._compiler.reduction_strategy._cute_shared_memory_budget_bytes",
            return_value=232448,
        ),
    ):
        yield


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
    return _cpu_bind(_kernel(), arguments), arguments


def _config(
    bound: Any,
    *,
    dv_block: int,
    dv_threads: int,
    reduction_threads: int,
    vec: int = 4,
    reduction_loop: int | None = None,
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


@skipUnlessBackends(["cute"])
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
    # state tensor is read exactly once (the packet) and written exactly once
    # (the flush) per lane -- no scalar re-load of the stored fragment.
    assert code.count("cute.arch.load(") == 2
    assert code.count("ir.VectorType.get([4], cutlass.Uint32.mlir_type)") == 2
    assert code.count("state.iterator") == 2
    assert code.count("_cute_store_u32_vec(state.iterator") == 1
    assert code.count("_cute_store_u32_vec(out.iterator") == 1
    # Accumulate nest: all loads, no stores.  Consume nest: the state flush.
    accumulate, consume = _owner_loops(code)
    accumulate_source = ast.unparse(accumulate)
    consume_source = ast.unparse(consume)
    assert accumulate_source.count("cute.arch.load(") == 2
    assert ".load()" in accumulate_source
    assert "_cute_store" not in accumulate_source
    assert "_cute_store_u32_vec(state.iterator" in consume_source
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


@skipUnlessBackends(["cute"])
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


@skipUnlessBackends(["cute"])
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


@skipUnlessBackends(["cute"])
def test_mismatched_vector_width_keeps_rolled_lanes() -> None:
    # Eight elements per thread with V=4 is two vectors per lane, not one: the
    # established rolled nesting is kept.
    bound, _arguments = _bind()
    code = bound.to_code(_config(bound, dv_block=64, dv_threads=8, reduction_threads=8))
    assert "_lane_stash" not in code
    assert COLUMN_REDUCE not in code
    assert "for synthetic_lane_2 in range(16):" in code


@skipUnlessBackends(["cute"])
def test_aliasing_inputs_reject_the_split(caplog: pytest.LogCaptureFixture) -> None:
    # ``q`` viewing the state storage removes the disjointness proof, so the
    # loads of one lane may not move above another lane's state store: the
    # register tile rejects the split, and so does the rolled nesting the
    # kernel is regenerated with.
    state = torch.empty((BH, D, DV), dtype=torch.float32)
    q, k, v, _state, alpha = _arguments(state)
    bound, _arguments_alive = _bind((state[:, :, 0], k, v, state, alpha))
    with (
        caplog.at_level(logging.DEBUG, logger="helion._compiler.generate_ast"),
        pytest.raises(helion.exc.BackendUnsupported, match="aliasing write"),
    ):
        bound.to_code(_config(bound, dv_block=32, dv_threads=8, reduction_threads=8))
    assert "may alias a store of another lane" in caplog.text


@skipUnlessBackends(["cute"])
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


@skipUnlessBackends(["cute"])
def test_flattened_column_tile_forms_a_register_tile() -> None:
    # A single flattened tile axis (``PerThreadFlattenedTileStrategy``) with a
    # persistent row reduction gets the same lowering: the column axis on
    # thread_idx[0], one packet load per lane, one column reduce, and the
    # reduced columns stored once by the reduction-coordinate-zero threads.
    arguments = (torch.empty((128, 1024), dtype=torch.float32),)
    bound = _cpu_bind(col_sum, arguments)
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


@skipUnlessBackends(["cute"])
def test_reduction_derived_scale_is_emitted_in_the_consume_nest() -> None:
    # ``r = 1 / (sum(t * t) + 1)`` is lane-invariant but feeds the lane-varying
    # store of every row, so the consume nest must define it before the
    # products.
    arguments = (torch.empty((128, 1024), dtype=torch.float32),)
    bound = _cpu_bind(col_scale, arguments)
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


@skipUnlessBackends(["cute"])
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
    bound = _cpu_bind(col_sum, arguments)
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


@skipUnlessBackends(["cute"])
def test_nested_device_loop_keeps_the_rolled_lane() -> None:
    # The device IR shows a rolled loop inside the tile body, so the lane
    # nesting the register tile needs is never chosen: the established rolled
    # synthetic lane compiles instead of a late rejection.
    arguments = (
        torch.empty((128, 1024), dtype=torch.float32),
        torch.empty((4, 1024), dtype=torch.float32),
    )
    bound = _cpu_bind(col_sum_nested, arguments)
    code = bound.to_code(
        column_config(bound, block=32, tile_threads=8, reduction_threads=8)
    )
    assert "for synthetic_lane_1 in range(16):" in code
    assert COLUMN_REDUCE not in code
    assert "_lane_stash" not in code


@skipUnlessBackends(["cute"])
def test_two_reductions_share_one_register_tile() -> None:
    arguments = (torch.empty((128, 1024), dtype=torch.float32),)
    bound = _cpu_bind(col_stats, arguments)
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


@skipUnlessBackends(["cute"])
def test_single_vector_thread_uses_the_warp_column_path() -> None:
    # One vector thread under 64 consecutive reduction threads: the column
    # reduce folds each warp with a shuffle before the shared-memory exchange.
    arguments = (torch.empty((128, 1024), dtype=torch.float32),)
    bound = _cpu_bind(col_sum, arguments)
    code = bound.to_code(
        column_config(bound, block=4, tile_threads=1, reduction_threads=64)
    )
    assert "for synthetic_lane_1 in cutlass.range_constexpr(2):" in code
    assert code.count(COLUMN_REDUCE) == 1
    # The shared-memory exchange is keyed on the full runtime thread id (a
    # redundant thread axis may still be mapped later in codegen), so its
    # group count covers the largest launch block.
    assert (
        "sum_1_lane_acc_lane = cutlass.Int32(cute.arch.thread_idx()[0]) "
        "+ cutlass.Int32(cute.arch.thread_idx()[1]) "
        "* cutlass.Int32(cute.arch.block_dim()[0])"
    ) in code
    assert f"pre=1, group_span=64, group_count={MAX_THREADS_PER_BLOCK // 64})" in code


def _rolled_code(bound: Any, config: helion.Config) -> str:
    """``config``'s code with the register tile never predicted: the rolled
    lane nesting the persistent strategy used before register tiles existed."""
    with patch(
        "helion._compiler.reduction_strategy._cute_register_tile_shape",
        return_value=False,
    ):
        return bound.to_code(config)


def _f32(*shape: int) -> torch.Tensor:
    return torch.empty(shape, dtype=torch.float32)


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize(
    ("kernel", "make_arguments"),
    (
        pytest.param(col_softmax, lambda: (_f32(128, 1024),), id="column-softmax"),
        pytest.param(col_var, lambda: (_f32(128, 1024),), id="two-pass-variance"),
    ),
)
def test_chained_reductions_keep_the_rolled_lane_up_front(
    kernel: helion.Kernel,
    make_arguments: Callable[[], tuple[Any, ...]],
    caplog: pytest.LogCaptureFixture,
) -> None:
    # One reduction feeding another needs a second accumulate pass; the device
    # IR shows the chain, so the rolled lane nesting is chosen directly (no
    # register tile, no second codegen pass) exactly as before register tiles.
    arguments = make_arguments()
    bound = _cpu_bind(kernel, arguments)
    config = column_config(bound, block=32, tile_threads=8, reduction_threads=8)
    with caplog.at_level(logging.DEBUG, logger="helion._compiler.generate_ast"):
        code = bound.to_code(config)
    assert FALLBACK_LOG not in caplog.text
    assert ROLLED_OWNER in code
    assert CONSTEXPR_OWNER not in code
    assert "_lane_stash" not in code
    assert code == _rolled_code(bound, config)


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize(
    ("kernel", "make_arguments", "block", "reduction_threads", "vec"),
    (
        pytest.param(
            col_sum_guarded,
            lambda: (_f32(128, 1024), 1),
            32,
            8,
            4,
            id="guard-inside-the-element-loops",
        ),
        pytest.param(
            col_sum_f32_out,
            lambda: (torch.empty((128, 1024), dtype=torch.bfloat16),),
            64,
            16,
            8,
            id="bf16-tile-fp32-store",
        ),
        pytest.param(
            col_scale_three,
            lambda: (_f32(128, 1024), _f32(128, 1024), _f32(128, 1024)),
            32,
            8,
            4,
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
    caplog: pytest.LogCaptureFixture,
) -> None:
    # The device IR admits these bodies, so the register-tile nesting is
    # chosen, but the generated tile body has a statement the two-pass
    # schedule cannot place (a guard inside the element loops, a per-element
    # store, a live-value budget overrun).  The kernel is regenerated with the
    # rolled lane nesting and compiles exactly as it did before register tiles
    # existed.
    arguments = make_arguments()
    bound = _cpu_bind(kernel, arguments)
    config = column_config(
        bound,
        block=block,
        tile_threads=8,
        reduction_threads=reduction_threads,
        vec=vec,
    )
    assert config.reduction_loops == [None]
    with caplog.at_level(logging.DEBUG, logger="helion._compiler.generate_ast"):
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


@skipUnlessBackends(["cute"])
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
    bound = _cpu_bind(kernel, arguments)
    config = column_config(bound, block=32, tile_threads=8, reduction_threads=64)
    assert config.reduction_loops == [None]
    with caplog.at_level(logging.DEBUG, logger="helion._compiler.generate_ast"):
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


@skipUnlessBackends(["cute"])
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


@skipUnlessBackends(["cute"])
def test_the_looped_retry_keeps_the_sink_knob() -> None:
    # The retry after a rejected register tile re-normalizes the config with
    # register tiles withheld and nothing else: ``cute_vloop_sink`` survives
    # into the looped pass, where the sink hook runs on the rolled body (this
    # grid records no V-loop wrapper, so it leaves the body untouched).  The
    # knob-on code is the knob-off looped code.
    arguments = (_f32(1024, 1024), 1)
    bound = _cpu_bind(col_sum_guarded, arguments)
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
    # the register-tile split rejects is thrown away, so the regenerated pass
    # must emit every placeholder again.  The record lives on the pass's
    # ``GenerateAST``, which the retry creates anew.
    arguments = (_f32(64, 64), _f32(64, 64))
    bound = _cpu_bind(add_rows, arguments)
    host = bound.host_function
    assert host is not None
    passes: list[list[dict[str, str]]] = []
    generate = generate_ast_module._generate_ast

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
    ):
        generate_ast_module.generate_ast(
            host,
            bound.config_spec.default_config(),
            False,
            load_transform=remember_loads,
        )
    assert len(passes) == 2
    assert passes[0] == passes[1] == [{}, {"x": "emitted"}]


@skipUnlessBackends(["cute"])
def test_register_budget_keeps_wide_stashes_rolled(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Three loaded tiles kept live until the consume pass (3 x 16 lanes x 4
    # elements) plus the accumulators exceed the live-value budget; half the
    # lanes per thread fit.
    arguments = (_f32(128, 1024), _f32(128, 1024), _f32(128, 1024))
    bound = _cpu_bind(col_scale_three, arguments)
    with caplog.at_level(logging.DEBUG, logger="helion._compiler.generate_ast"):
        code = bound.to_code(
            column_config(bound, block=32, tile_threads=8, reduction_threads=8)
        )
    assert "live values exceed" in caplog.text
    assert ROLLED_OWNER in code
    assert "_lane_stash" not in code
    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger="helion._compiler.generate_ast"):
        code = bound.to_code(
            column_config(bound, block=32, tile_threads=8, reduction_threads=16)
        )
    assert FALLBACK_LOG not in caplog.text
    assert code.count("_lane_stash = []") == 3
    _assert_defined_before_use(code)


@skipUnlessBackends(["cute"])
def test_eight_byte_policy_loads_are_stashed() -> None:
    # fp32 V=2 under ``l1_l2_first`` loads through the 8-byte policy helper.
    # It is a load like the 16-byte form: the loaded tile is stashed for the
    # consume nest instead of being re-loaded behind the barrier.
    arguments = (_f32(128, 1024), _f32(128, 1024))
    bound = _cpu_bind(col_scale_into, arguments)
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


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize(
    ("vec", "block", "policy"),
    ((2, 16, None), (4, 32, "l1_l2_first"), (2, 16, "l1_l2_first")),
    ids=("plain-loads", "16-byte-helper", "8-byte-helper"),
)
def test_aliasing_views_reject_every_load_form(
    vec: int, block: int, policy: str | None
) -> None:
    # ``x`` and ``y`` overlap, so a lane's load may alias another lane's
    # store.  Neither the register tile nor the rolled split may reorder them,
    # whichever helper the load goes through.
    base = _f32(129, 1024)
    arguments = (base[:128], base[1:])
    bound = _cpu_bind(col_scale_into, arguments)
    extra: dict[str, Any] = (
        {} if policy is None else {"load_eviction_policies": [policy]}
    )
    with pytest.raises(helion.exc.BackendUnsupported, match="aliasing write"):
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


@skipUnlessBackends(["cute"])
def test_load_helper_names_match_the_backend_imports() -> None:
    # Every ``_cute_load_*`` helper the backend binds in generated code is in
    # the one list the lane-loop passes read (relocation, load collection and
    # the aliasing proofs), so a new helper cannot slip past them.
    bound, _arguments = _bind()
    bound_names = {
        name
        for name in bound.env.backend.library_imports
        if name.startswith("_cute_load_")
    }
    assert bound_names == _CUTE_CACHE_LOAD_HELPER_NAMES


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
    assert returns == 2


@skipUnlessBackends(["cute"])
def test_register_tile_blocks_follow_the_body_and_the_extent() -> None:
    # Config normalization, the seed and codegen share one admission: the
    # reduction block is listed only for a body the two-pass schedule may
    # lower (independent reductions, no rolled loop, no reduction fed by
    # another) and a static (unmasked) extent.
    for kernel, arguments, admitted in (
        (col_sum, (_f32(128, 1024),), True),
        (col_stats, (_f32(128, 1024),), True),
        (col_scale, (_f32(128, 1024),), True),
        (col_softmax, (_f32(128, 1024),), False),
        (col_var, (_f32(128, 1024),), False),
        (col_sum_nested, (_f32(128, 1024), _f32(4, 1024)), False),
        (col_sum_dynamic, (_f32(1024, 1024),), False),
    ):
        bound = _cpu_bind(kernel, arguments)
        (reduction,) = [
            block.block_id for block in bound.env.block_sizes if block.reduction
        ]
        blocks = bound.config_spec.cute_register_tile_reduction_blocks
        assert (reduction in blocks) is admitted, kernel.name


@skipUnlessBackends(["cute"])
def test_dynamic_extent_forces_the_looped_reduction() -> None:
    # With ``static_shapes=False`` a persistent lane loop would bake its lane
    # count from the size hint and a later, larger extent would drop rows; a
    # shrunk thread count is forced looped, with the trip count computed from
    # the runtime extent on the host.
    arguments = (_f32(1024, 1024),)
    bound = _cpu_bind(col_sum_dynamic, arguments)
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
