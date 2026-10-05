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
# The two-stage shared reduce keys its per-group shared memory on the linear
# thread id across ALL launch-block threads, taken from the runtime block dims
# so a redundant thread axis mapped later in codegen cannot alias the slots.
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
) -> helion.Config:
    spec = bound.config_spec
    reduction = _reduction_block_id(bound)
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
    code: str, *, reductions: int, group_span: int, broadcast_stores: int = 1
) -> None:
    reduces = _two_stage_reduces(code)
    assert len(reduces) == reductions
    for lane_expr, kwargs in reduces:
        # Shared memory keyed on the full runtime thread id: one group of
        # ``group_span`` consecutive lanes per row of the launch block, with
        # enough groups for every thread a redundant axis could add.
        assert lane_expr == RUNTIME_LANE
        assert kwargs == {
            "pre": 1,
            "group_span": group_span,
            "group_count": MAX_THREADS_PER_BLOCK // group_span,
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
def test_multiwarp_row_keys_shared_memory_on_runtime_thread_id() -> None:
    # The thread axes known when the reduce is emitted only cover the row's
    # own axis; a sibling branch can still map a redundant axis onto
    # thread_idx()[1]/[2] later in codegen.  Keying on the reduce axis alone
    # would let those redundant rows race on the same shared slots, so the
    # lane is the full runtime thread id (as for a non-synthetic persistent
    # cross-warp reduce) with a group for every possible warp-row.
    bound, _arguments = _bind_rms_norm(256, 1024, torch.bfloat16)
    code = bound.to_code(_row_config(bound, reduction_threads=128, vec=8))
    [(lane_expr, kwargs)] = _two_stage_reduces(code)
    assert lane_expr == RUNTIME_LANE
    assert kwargs == {"pre": 1, "group_span": 128, "group_count": 8}
    # Not keyed on the axes discovered so far.
    assert lane_expr != "cutlass.Int32(cute.arch.thread_idx()[0])"
    assert "_lane = cutlass.Int32(cute.arch.thread_idx()[0])\n" not in code
    # The runtime key is only the shared-memory key.  The marker's static lane
    # still drives ownership: the per-thread ``out`` vector store is emitted
    # for every thread, and only the broadcast inv_rms store gets the static
    # lane-0 guard (a guard on the runtime lane would leave one thread in 128
    # storing ``out``).
    assert len(_top_level_store_lines(code, "_cute_store_u16_vec")) == 1
    assert code.count("if cutlass.Int32(cute.arch.thread_idx()[0]) % 128 < 1:") == 1
    assert f"if ({RUNTIME_LANE})" not in code
    # The lane setup is shared with the non-synthetic persistent cross-warp
    # path: the same expression keys a 1024-thread row's reduce.
    wide = bound.to_code(_row_config(bound, reduction_threads=0, vec=1))
    assert "block=(1024, 1, 1)" in wide
    [(wide_lane_expr, wide_kwargs)] = _two_stage_reduces(wide)
    assert wide_lane_expr == RUNTIME_LANE
    assert wide_kwargs == {"pre": 1, "group_span": 1024, "group_count": 1}


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
        code, reductions=1, group_span=128, broadcast_stores=0
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
