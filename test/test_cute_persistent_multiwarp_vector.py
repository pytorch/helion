"""Multi-warp one-vector-per-thread persistent row reductions (CPU codegen).

``PersistentReductionStrategy`` used to cap every synthetic-lane row to one
warp: a 1024-wide bf16 row asked to run on 128 threads with V=8 came out as a
32-thread CTA looping over 32 scalar lanes, so the one-LDG.128-per-thread shape
of the Triton kernel was unreachable.  A row whose vector width exactly covers
each thread's slice now keeps its warp-aligned thread count and combines the
per-thread V-folds once with the cross-warp two-stage shared reduce, keyed on
the full runtime thread id like the non-synthetic cross-warp path.
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import cast
from unittest.mock import patch

from examples.rms_norm import rms_norm_fwd
from examples.softmax import softmax
import pytest
import torch

from test._cute_binding import _cpu_bind
from test._cute_binding import _forbid_native_compile
from test._cute_binding import _mock_cuda_unavailable

import helion
from helion._compiler.autotuner_heuristics import get_heuristics
from helion._compiler.autotuner_heuristics.cute import CuteReductionTileHeuristic
from helion._testing import skipUnlessBackends
from helion.autotuner.config_generation import ConfigGeneration
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Iterator

TWO_STAGE = "_cute_grouped_reduce_shared_two_stage"
MAX_THREADS_PER_BLOCK = 1024
# The two-stage shared reduce is emitted keyed on the linear thread id across
# ALL launch-block threads, taken from the runtime block dims, with a group
# for every warp-row the 1024-thread budget allows: a redundant thread axis
# mapped later in codegen must not alias the slots.  Once the launch shape is
# final, ``finalize_shared_reduce_groups`` sizes the groups for the threads
# that launch and reduces the lane of a one-dimensional block to
# ``thread_idx()[0]``.
X_LANE = "cutlass.Int32(cute.arch.thread_idx()[0])"
RUNTIME_LANE = (
    "cutlass.Int32(cute.arch.thread_idx()[0])"
    " + cutlass.Int32(cute.arch.thread_idx()[1])"
    " * cutlass.Int32(cute.arch.block_dim()[0])"
    " + cutlass.Int32(cute.arch.thread_idx()[2])"
    " * cutlass.Int32(cute.arch.block_dim()[0])"
    " * cutlass.Int32(cute.arch.block_dim()[1])"
)


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


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _row_rms_static(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    rows, width = x.shape
    out = torch.empty_like(x)
    for tile_rows in hl.tile(rows):
        cols = hl.arange(width)
        x_tile = x[tile_rows, cols].float()
        inv_rms = torch.rsqrt((x_tile * x_tile).sum(-1) / width + 1e-5)
        out[tile_rows, cols] = (
            x_tile * inv_rms[:, None] * weight[cols].float()[None, :]
        ).to(x.dtype)
    return out


def _example_kernel(
    fn: Callable[..., Any], *, static_shapes: bool = True
) -> helion.Kernel:
    return helion.kernel(
        fn,
        backend="cute",
        static_shapes=static_shapes,
        autotune_effort="none",
        ignore_warnings=[helion.exc.TensorOperationInWrapper],
    )


def _bind(
    kernel: helion.Kernel, arguments: tuple[torch.Tensor, ...]
) -> tuple[Any, tuple[torch.Tensor, ...]]:
    """Bind on CPU and hand the arguments back with the bound kernel.

    The caller keeps ``arguments`` alive while generating code: the vector
    alignment proof reads the live input tensors' metadata (the kernel only
    holds weak references), and a collected tensor silently leaves the loads
    scalar.
    """
    return _cpu_bind(kernel, arguments), arguments


def _bind_rms_norm(
    rows: int, width: int, dtype: torch.dtype
) -> tuple[Any, tuple[torch.Tensor, ...]]:
    return _bind(
        _example_kernel(rms_norm_fwd.fn),
        (torch.empty((rows, width), dtype=dtype), torch.empty(width, dtype=dtype)),
    )


def _reduction_block_id(bound: Any) -> int:
    (block_id,) = [block.block_id for block in bound.env.block_sizes if block.reduction]
    return block_id


def _row_config(
    bound: Any,
    *,
    reduction_threads: int,
    vec: int,
    row_block: int = 1,
    row_threads: int = 0,
    load_eviction_policies: list[str] | None = None,
) -> helion.Config:
    spec = bound.config_spec
    reduction = _reduction_block_id(bound)
    extra: dict[str, object] = {}
    if load_eviction_policies is not None:
        extra["load_eviction_policies"] = load_eviction_policies
    return spec.normalized_config(
        helion.Config(
            block_sizes=[row_block for _ in spec.block_sizes.valid_block_ids()],
            num_threads=[
                reduction_threads if block_id == reduction else row_threads
                for block_id in spec.num_threads.valid_block_ids()
            ],
            cute_vector_widths=[
                vec if block_id == reduction else 1
                for block_id in spec.cute_vector_widths.valid_block_ids()
            ],
            **extra,
        )
    )


def _kernel_body(code: str) -> str:
    """Source of the device kernel (``def _helion_*``) up to the host wrapper."""
    lines = code.splitlines()
    starts = [index for index, line in enumerate(lines) if line.startswith("def ")]
    kernel = next(index for index in starts if lines[index].startswith("def _helion_"))
    end = next((index for index in starts if index > kernel), len(lines))
    return "\n".join(lines[kernel:end])


def _top_level_store_lines(code: str, store: str) -> list[str]:
    """Lines calling ``store`` directly in the kernel body (not under an if)."""
    return [
        line
        for line in _kernel_body(code).splitlines()
        if line.startswith(f"    {store}(")
    ]


def _element_loop_bodies(code: str) -> list[str]:
    """Source of every constexpr per-element V-loop body."""
    return [
        "\n".join(ast.unparse(stmt) for stmt in node.body)
        for node in ast.walk(ast.parse(code))
        if isinstance(node, ast.For)
        and isinstance(node.iter, ast.Call)
        and ast.unparse(node.iter.func) == "cutlass.range_constexpr"
    ]


def _two_stage_reduces(code: str) -> list[tuple[str, dict[str, int]]]:
    """``(lane_expr, keyword args)`` of every two-stage shared reduce call.

    ``lane_expr`` is the value assigned to the call's ``lane`` argument: the
    expression the helper keys its shared-memory groups on.
    """
    tree = ast.parse(code)
    assigned = {
        node.targets[0].id: ast.unparse(node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
    }
    reduces = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == TWO_STAGE
        ):
            lane = node.args[3]
            assert isinstance(lane, ast.Name)
            kwargs = {
                keyword.arg: keyword.value.value
                for keyword in node.keywords
                if keyword.arg is not None
                and isinstance(keyword.value, ast.Constant)
                and isinstance(keyword.value.value, int)
            }
            reduces.append((assigned[lane.id], kwargs))
    return reduces


def _assert_reduces_once_per_reduction(
    code: str,
    *,
    reductions: int,
    group_span: int,
    broadcast_stores: int = 1,
    block_rows: int = 1,
) -> None:
    reduces = _two_stage_reduces(code)
    assert len(reduces) == reductions
    for lane_expr, kwargs in reduces:
        # Sized for the launched block: one group of ``group_span``
        # consecutive lanes per row of the launch block.  A one-dimensional
        # block owns a single group keyed on ``thread_idx()[0]``; rows sharing
        # a CTA keep the runtime thread id (``thread_idx()[1]`` selects the
        # row's group).
        assert lane_expr == (X_LANE if block_rows == 1 else RUNTIME_LANE)
        assert kwargs == {
            "pre": 1,
            "group_span": group_span,
            "group_count": block_rows,
        }
    # The runtime key must not leak into the ownership analysis: a broadcast
    # result is stored by lane 0 of the (static) reduce axis, and nothing is
    # guarded on the runtime lane.
    owner_guard = f"if cutlass.Int32(cute.arch.thread_idx()[0]) % {group_span} < 1:"
    assert code.count(owner_guard) == broadcast_stores
    assert f"if ({RUNTIME_LANE})" not in code
    # Every marker was lowered, and no shuffle tree is left over: a warp
    # shuffle cannot span the multi-warp group.
    assert "_helion_lane_reduce" not in code
    assert "warp_reduction" not in code
    # No loop-carried scalar lanes: each thread owns exactly one V-fold.
    assert "in range(" not in _kernel_body(code)
    bodies = _element_loop_bodies(code)
    assert len(bodies) >= reductions
    for body in bodies:
        assert "_cute_grouped_reduce" not in body
        assert "warp_reduction" not in body
        assert "sync_threads" not in body


def _assert_one_warp_vector_row(
    code: str, *, vec: int, element: str, store: str
) -> None:
    """A 32-thread row loading one V-wide vector per thread for x and weight."""
    assert "block=(32, 1, 1)" in code
    vector_loads = [line for line in code.splitlines() if "cute.arch.load(" in line]
    assert len(vector_loads) == 2
    assert sum("x.iterator" in line for line in vector_loads) == 1
    assert sum("weight.iterator" in line for line in vector_loads) == 1
    assert code.count(f"ir.VectorType.get([{vec}], cutlass.{element}.mlir_type)") == 2
    assert f"_cute_store_{store}_vec(out.iterator" in code
    # One warp folds the per-thread V-folds with a plain warp shuffle.
    assert "threads_in_group=32" in code
    assert TWO_STAGE not in code
    assert "in range(" not in _kernel_body(code)


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize(
    "width,threads,vec", [(256, 96, 2), (512, 224, 2), (1024, 224, 4)]
)
def test_whole_warp_thread_request_covers_the_full_row(
    width: int, threads: int, vec: int
) -> None:
    # These requests are legal whole-warp counts, but floor division would
    # falsely admit a single packet per thread and omit the end of the row.
    bound, _arguments = _bind_rms_norm(8, width, torch.bfloat16)
    config = _row_config(bound, reduction_threads=threads, vec=vec)
    reduction = _reduction_block_id(bound)
    assert config.reduction_loops == [None]
    assert (
        bound.config_spec.num_threads.config_get(config.num_threads, reduction, 0)
        == threads
    )
    with _forbid_native_compile():
        source = ast.parse(bound.to_code(config))
    lane_extents = {
        ast.literal_eval(node.iter.args[0])
        for node in ast.walk(source)
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id.startswith("synthetic_lane_")
        and isinstance(node.iter, ast.Call)
    }
    block = next(
        ast.literal_eval(keyword.value)
        for node in ast.walk(source)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_launcher"
        for keyword in node.keywords
        if keyword.arg == "block"
    )
    assert len(lane_extents) == 1
    assert block[1:] == (1, 1)
    assert block[0] * next(iter(lane_extents)) == width


@skipUnlessBackends(["cute"])
def test_rms_norm_1024_bf16_one_vector_per_thread_uses_four_warps() -> None:
    bound, _arguments = _bind_rms_norm(256, 1024, torch.bfloat16)
    config = _row_config(bound, reduction_threads=128, vec=8)
    assert config.reduction_loops == [None]
    code = bound.to_code(config)
    assert "block=(128, 1, 1)" in code
    _assert_reduces_once_per_reduction(code, reductions=1, group_span=128)
    # One LDG.128 per thread for x and one for weight; the consume sweep
    # reuses the x fragment from registers instead of reloading it.
    vector_loads = [line for line in code.splitlines() if "cute.arch.load(" in line]
    assert len(vector_loads) == 2
    assert sum("x.iterator" in line for line in vector_loads) == 1
    assert sum("weight.iterator" in line for line in vector_loads) == 1
    assert code.count("ir.VectorType.get([8], cutlass.Uint16.mlir_type)") == 2
    # Every thread stores its own vector of ``out`` unguarded; only the
    # broadcast inv_rms result is stored by lane 0 of the row group.
    assert len(_top_level_store_lines(code, "_cute_store_u16_vec")) == 1
    assert "_cute_store_u16_vec(out.iterator" in code
    assert code.count("if cutlass.Int32(cute.arch.thread_idx()[0]) % 128 < 1:") == 1
    assert code.count(" % 128 < 1:") == 1


@skipUnlessBackends(["cute"])
def test_multiwarp_row_reduce_is_sized_for_the_launched_block() -> None:
    # The thread axes known when the reduce is emitted only cover the row's
    # own axis; a sibling branch can still map a redundant axis onto
    # thread_idx()[1]/[2] later in codegen, so the marker is lowered keyed on
    # the full runtime thread id with a group for every possible warp-row
    # (see ``test_finalize_shared_reduce_groups_*``).  The launch shape is
    # final before the late passes run: a 128-thread row ends up as the one
    # group spanning the CTA, keyed on ``thread_idx()[0]`` -- the form the
    # helper's cheaper serial fold and the replicated-reduction rewrite need.
    bound, _arguments = _bind_rms_norm(256, 1024, torch.bfloat16)
    code = bound.to_code(_row_config(bound, reduction_threads=128, vec=8))
    [(lane_expr, kwargs)] = _two_stage_reduces(code)
    assert lane_expr == X_LANE
    assert kwargs == {"pre": 1, "group_span": 128, "group_count": 1}
    assert "block_dim()" not in _kernel_body(code)
    # The static lane still drives ownership: the per-thread ``out`` vector
    # store is emitted for every thread, and only the broadcast inv_rms store
    # gets the lane-0 guard.
    assert len(_top_level_store_lines(code, "_cute_store_u16_vec")) == 1
    assert code.count("if cutlass.Int32(cute.arch.thread_idx()[0]) % 128 < 1:") == 1
    # The lane setup is shared with the non-synthetic persistent cross-warp
    # path: a 1024-thread row's reduce is one group as well.
    wide = bound.to_code(_row_config(bound, reduction_threads=0, vec=1))
    assert "block=(1024, 1, 1)" in wide
    [(wide_lane_expr, wide_kwargs)] = _two_stage_reduces(wide)
    assert wide_lane_expr == X_LANE
    assert wide_kwargs == {"pre": 1, "group_span": 1024, "group_count": 1}


def _finalize(source: str, dims: tuple[int, int, int]) -> str:
    from helion._compiler.cute.finalize_reduce_groups import (
        finalize_shared_reduce_groups,
    )

    body = ast.parse(source).body
    return "\n".join(
        ast.unparse(stmt)
        for stmt in finalize_shared_reduce_groups(body, thread_block_dims=dims)
    )


_EMITTED_REDUCE = (
    f"lane = {RUNTIME_LANE}\n"
    "lane_in_group = lane % 128\n"
    "lane_mod_pre = lane_in_group % 1\n"
    "acc = _cute_grouped_reduce_shared_two_stage(part, 'sum', cutlass.Float32(0), "
    "lane, lane_in_group, lane_mod_pre, pre=1, group_span=128, group_count=8)"
)


def test_finalize_shared_reduce_groups_sizes_one_dimensional_block() -> None:
    # Emitted for the 1024-thread budget (8 groups of 128), launched as one
    # 128-thread row: one group, lane reduced to thread_idx()[0].
    code = _finalize(_EMITTED_REDUCE, (128, 1, 1))
    assert f"lane = {X_LANE}" in code
    assert "group_count=1)" in code
    assert "block_dim()" not in code


def test_finalize_shared_reduce_groups_keeps_rows_sharing_a_cta_apart() -> None:
    # Two rows per CTA (block (128, 2, 1)): two groups, keyed on the runtime
    # thread id so thread_idx()[1] selects the row's group.
    code = _finalize(_EMITTED_REDUCE, (128, 2, 1))
    assert f"lane = {RUNTIME_LANE}" in code
    assert "group_count=2)" in code


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _row_sumsq_per_arange_row(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    # x: [M, N], w: [4, N]; the free arange(4) selects the w row and feeds the
    # reduced values, so the threads along its axis reduce different data.
    rows, width = x.shape
    out = torch.empty((rows, 4), dtype=torch.float32, device=x.device)
    for tile_rows in hl.tile(rows):
        cols = hl.arange(width)
        j = hl.arange(4)
        x_tile = x[tile_rows, cols].float()
        w_tile = w[j[:, None], cols[None, :]].float()
        vals = x_tile[:, None, :] * w_tile[None, :, :]
        out[tile_rows, j] = (vals * vals).sum(-1)
    return out


@skipUnlessBackends(["cute"])
def test_free_arange_thread_axis_keeps_the_runtime_lane_form() -> None:
    # The free arange claims a synthetic thread axis the strategies' static
    # shape does not know about: the launch is block=(128, 4, 1), not the
    # (128, 1, 1) the row planned.  Sizing the shared stage for the planned
    # shape would let the four arange rows alias one group's slots, so the
    # finalization declines and the conservative runtime-lane form stays.
    bound, _arguments = _bind(
        _row_sumsq_per_arange_row,
        (
            torch.empty((64, 1024), dtype=torch.bfloat16),
            torch.empty((4, 1024), dtype=torch.bfloat16),
        ),
    )
    code = bound.to_code(_row_config(bound, reduction_threads=128, vec=8))
    assert "block=(128, 4, 1)" in code
    [(lane_expr, kwargs)] = _two_stage_reduces(code)
    assert lane_expr == RUNTIME_LANE
    assert kwargs["group_span"] == 128
    assert kwargs["group_count"] >= 4
    assert kwargs == {
        "pre": 1,
        "group_span": 128,
        "group_count": MAX_THREADS_PER_BLOCK // 128,
    }


def test_finalize_shared_reduce_groups_for_launch_declines_wider_claims() -> None:
    from helion._compiler.cute.finalize_reduce_groups import (
        finalize_shared_reduce_groups_for_launch,
    )

    body = ast.parse(_EMITTED_REDUCE).body
    # A claimed axis wider than the static shape (a free arange on axis 1):
    # untouched, nothing recorded.
    result, sized_for = finalize_shared_reduce_groups_for_launch(
        body, thread_block_dims=(128, 1, 1), claimed_axis_sizes={0: 128, 1: 4}
    )
    assert sized_for is None
    assert "group_count=8)" in "\n".join(ast.unparse(stmt) for stmt in result)
    # Claims within the static shape: rewritten, the shape recorded.
    result, sized_for = finalize_shared_reduce_groups_for_launch(
        body, thread_block_dims=(128, 1, 1), claimed_axis_sizes={0: 128}
    )
    assert sized_for == (128, 1, 1)
    assert "group_count=1)" in "\n".join(ast.unparse(stmt) for stmt in result)
    # Nothing to rewrite (two rows per CTA already sized as two groups, the
    # runtime lane kept for the multi-dimensional block): nothing recorded.
    exact = ast.parse(_EMITTED_REDUCE.replace("group_count=8", "group_count=2")).body
    _result, sized_for = finalize_shared_reduce_groups_for_launch(
        exact, thread_block_dims=(128, 2, 1), claimed_axis_sizes={0: 128, 1: 2}
    )
    assert sized_for is None
    # The lane simplification alone is a rewrite that assumes the shape.
    one_group = ast.parse(
        _EMITTED_REDUCE.replace("group_count=8", "group_count=1")
    ).body
    _result, sized_for = finalize_shared_reduce_groups_for_launch(
        one_group, thread_block_dims=(128, 1, 1), claimed_axis_sizes={}
    )
    assert sized_for == (128, 1, 1)


def test_finalize_shared_reduce_groups_matches_the_int64_lane() -> None:
    # ``index_dtype=torch.int64`` spells the runtime lane with cutlass.Int64;
    # the rewrite keeps the index type.
    source = _EMITTED_REDUCE.replace("cutlass.Int32", "cutlass.Int64")
    code = _finalize(source, (128, 1, 1))
    assert "lane = cutlass.Int64(cute.arch.thread_idx()[0])" in code
    assert "group_count=1)" in code
    assert "block_dim()" not in code


def test_finalize_shared_reduce_groups_leaves_other_shapes_alone() -> None:
    # A span that does not divide the block keeps its count (the lane of a
    # one-dimensional block still reduces: thread_idx()[1]/[2] are zero); a
    # count already at or below the launched groups is not the conservative
    # form and stays, as does the runtime lane of a multi-dimensional block.
    odd = _finalize(_EMITTED_REDUCE, (96, 1, 1))
    assert "group_count=8)" in odd
    assert f"lane = {X_LANE}" in odd
    exact = _EMITTED_REDUCE.replace("group_count=8", "group_count=1")
    assert "group_count=1)" in _finalize(exact, (1024, 1, 1))
    assert f"lane = {RUNTIME_LANE}" in _finalize(exact, (256, 2, 2))
    assert "group_count=1)" in _finalize(exact, (256, 2, 2))


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize(
    "dtype,threads,vec,suffix",
    [
        (torch.bfloat16, 128, 8, ""),
        (torch.float16, 256, 4, "_8b"),
        (torch.float16, 512, 2, "_4b"),
        (torch.float32, 256, 4, ""),
        (torch.float32, 512, 2, "_8b"),
    ],
)
def test_hinted_row_loads_use_the_cache_policy_helper_of_their_width(
    dtype: torch.dtype, threads: int, vec: int, suffix: str
) -> None:
    # ``load_eviction_policies`` L2 hints are inline-PTX helpers, one per
    # packet width (16, 8 and 4 bytes).  Dropping the hint below 16 bytes
    # cost the 1024-wide fp16 row a DRAM round trip per replay under an L2
    # flush (the Triton backend's evict_last keeps the row resident).
    bound, _arguments = _bind_rms_norm(256, 1024, dtype)
    code = bound.to_code(
        _row_config(
            bound,
            reduction_threads=threads,
            vec=vec,
            load_eviction_policies=["l1_l2_last", "l2_last"],
        )
    )
    assert f"block=({threads}, 1, 1)" in code
    body = _kernel_body(code)
    assert f"_cute_load_l1_l2_evict_last{suffix}(x.iterator" in body
    assert f"_cute_load_l2_evict_last{suffix}(weight.iterator" in body
    assert "cute.arch.load(" not in body
    element = "Uint16" if dtype.itemsize == 2 else "Uint32"
    assert body.count(f"ir.VectorType.get([{vec}], cutlass.{element}.mlir_type)") == 2
    _assert_reduces_once_per_reduction(code, reductions=1, group_span=threads)


@skipUnlessBackends(["cute"])
def test_fp32_row_uses_eight_warps_with_v4() -> None:
    bound, _arguments = _bind_rms_norm(256, 1024, torch.float32)
    code = bound.to_code(_row_config(bound, reduction_threads=256, vec=4))
    assert "block=(256, 1, 1)" in code
    _assert_reduces_once_per_reduction(code, reductions=1, group_span=256)
    assert code.count("ir.VectorType.get([4], cutlass.Uint32.mlir_type)") == 2
    assert len(_top_level_store_lines(code, "_cute_store_u32_vec")) == 1
    assert "_cute_store_u32_vec(out.iterator" in code


@skipUnlessBackends(["cute"])
def test_softmax_reduces_once_per_reduction() -> None:
    bound, _arguments = _bind(
        _example_kernel(softmax.fn),
        (torch.empty((256, 1024), dtype=torch.bfloat16),),
    )
    code = bound.to_code(_row_config(bound, reduction_threads=128, vec=8))
    assert "block=(128, 1, 1)" in code
    # softmax has no broadcast store: both reductions only feed the row.
    _assert_reduces_once_per_reduction(
        code, reductions=2, group_span=128, broadcast_stores=0
    )
    assert "'max', cutlass.Float32(float('-inf'))" in code
    assert "'sum', cutlass.Float32(0)" in code
    # Every thread stores its own output vector.
    assert len(_top_level_store_lines(code, "_cute_store_u16_vec")) == 1


@skipUnlessBackends(["cute"])
def test_rows_sharing_a_cta_reduce_in_separate_groups() -> None:
    bound, _arguments = _bind(
        _row_rms_static,
        (
            torch.empty((64, 1024), dtype=torch.bfloat16),
            torch.empty(1024, dtype=torch.bfloat16),
        ),
    )
    code = bound.to_code(
        _row_config(bound, reduction_threads=128, vec=8, row_block=2, row_threads=2)
    )
    assert "block=(128, 2, 1)" in code
    # The shared-memory groups are keyed on the runtime thread id across both
    # rows (thread_idx()[1] selects the row's group), so the rows never fold
    # into each other.  The kernel has no broadcast store; every thread stores
    # its own output vector (under the row's bounds mask, not an owner guard).
    _assert_reduces_once_per_reduction(
        code, reductions=1, group_span=128, broadcast_stores=0, block_rows=2
    )
    assert code.count("_cute_store_u16_vec(out.iterator") == 1
    assert "% 128 < 1" not in code


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize(
    "width,dtype,threads,vec,element,store",
    [
        (256, torch.bfloat16, 64, 8, "Uint16", "u16"),
        (256, torch.bfloat16, 128, 8, "Uint16", "u16"),
        (128, torch.float32, 64, 4, "Uint32", "u32"),
    ],
)
def test_capped_row_keeps_one_warp_vector_loads(
    width: int,
    dtype: torch.dtype,
    threads: int,
    vec: int,
    element: str,
    store: str,
) -> None:
    # Too many threads for one vector each (e.g. 4 elements per thread at 64
    # threads on a 256-wide bf16 row): the one-warp cap still applies, and V
    # is tested against the slice AFTER the cap, where it covers exactly one
    # vector per thread again.
    bound, _arguments = _bind_rms_norm(256, width, dtype)
    code = bound.to_code(_row_config(bound, reduction_threads=threads, vec=vec))
    _assert_one_warp_vector_row(code, vec=vec, element=element, store=store)
    # Identical to asking for the one warp directly.
    one_warp, _one_warp_arguments = _bind_rms_norm(256, width, dtype)
    assert code == one_warp.to_code(
        _row_config(one_warp, reduction_threads=32, vec=vec)
    )


@skipUnlessBackends(["cute"])
def test_mismatched_vector_width_keeps_single_warp_lane_loop() -> None:
    # 64 threads would give each thread 16 elements, not one V=8 vector: the
    # established one-warp scalar lane loop is kept.
    bound, _arguments = _bind_rms_norm(256, 1024, torch.bfloat16)
    code = bound.to_code(_row_config(bound, reduction_threads=64, vec=8))
    assert "block=(32, 1, 1)" in code
    assert "for synthetic_lane_1 in range(32):" in code
    assert "threads_in_group=32" in code
    assert TWO_STAGE not in code


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize(
    "dtype,threads,vec",
    [(torch.bfloat16, 128, 8), (torch.float16, 128, 8), (torch.float32, 256, 4)],
)
def test_reduction_tile_heuristic_seeds_multiwarp_vector_row(
    dtype: torch.dtype, threads: int, vec: int
) -> None:
    bound, _arguments = _bind_rms_norm(256, 1024, dtype)
    host = bound.host_function
    assert host is not None
    spec = bound.config_spec
    assert CuteReductionTileHeuristic in get_heuristics("cute")
    seeds = CuteReductionTileHeuristic.get_seed_configs(bound.env, host.device_ir)
    assert seeds is not None and len(seeds) == 2
    primary, alternate = seeds
    assert primary == CuteReductionTileHeuristic.get_seed_config(
        bound.env, host.device_ir
    )
    reduction = _reduction_block_id(bound)
    threads_index = spec.num_threads.valid_block_ids().index(reduction)
    vec_index = spec.cute_vector_widths.valid_block_ids().index(reduction)
    assert alternate.block_sizes == [1]
    assert alternate.reduction_loops == [None]
    assert alternate.num_threads[threads_index] == threads
    assert cast("list[int]", alternate.config["cute_vector_widths"])[vec_index] == vec
    assert alternate in spec.compiler_seed_configs
    # The seed survives normalization and the autotuner's flat round trip.
    normalized = spec.normalized_config(alternate)
    generation = ConfigGeneration(spec)
    surviving = [config for _flat, config in generation.seed_flat_config_pairs()]
    flat, roundtrip = generation.canonicalize_flat(generation.flatten(alternate))
    assert roundtrip in surviving
    assert generation.unflatten(flat) == roundtrip
    assert roundtrip.num_threads == normalized.num_threads
    assert (
        roundtrip.config["cute_vector_widths"]
        == normalized.config["cute_vector_widths"]
    )
    assert roundtrip.reduction_loops == [None]
    code = bound.to_code(roundtrip)
    assert f"block=({threads}, 1, 1)" in code
    _assert_reduces_once_per_reduction(code, reductions=1, group_span=threads)


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("width", [1000, 128, 2048])
def test_reduction_tile_heuristic_skips_rows_without_the_layout(width: int) -> None:
    # 1000: masked (non power-of-two) row; 128: 16 threads is below one warp;
    # 2048: wider than the persistent thread budget (rolled instead).
    bound, _arguments = _bind_rms_norm(256, width, torch.bfloat16)
    host = bound.host_function
    assert host is not None
    assert (
        CuteReductionTileHeuristic.multiwarp_vector_row_seed_config(
            bound.env, host.device_ir
        )
        is None
    )
    seeds = CuteReductionTileHeuristic.get_seed_configs(bound.env, host.device_ir)
    assert seeds == [
        CuteReductionTileHeuristic.get_seed_config(bound.env, host.device_ir)
    ]


@skipUnlessBackends(["cute"])
def test_reduction_tile_heuristic_requires_static_extent() -> None:
    # A dynamic row is masked and never vectorized, however power-of-two its
    # size hint happens to be: the seed keys on the static extent, not the hint.
    bound, _arguments = _bind(
        _example_kernel(softmax.fn, static_shapes=False),
        (torch.empty((256, 1024), dtype=torch.bfloat16),),
    )
    host = bound.host_function
    assert host is not None
    spec = bound.config_spec
    assert spec.reduction_loops[0].size_hint == 1024
    assert not bound.env.block_sizes[_reduction_block_id(bound)].numel.is_Integer
    assert (
        CuteReductionTileHeuristic.multiwarp_vector_row_seed_config(
            bound.env, host.device_ir
        )
        is None
    )
    seeds = CuteReductionTileHeuristic.get_seed_configs(bound.env, host.device_ir)
    assert seeds == [
        CuteReductionTileHeuristic.get_seed_config(bound.env, host.device_ir)
    ]
