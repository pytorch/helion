from __future__ import annotations

import ast
from dataclasses import replace
import importlib
import inspect
from typing import Any
from typing import cast

import pytest

from helion import exc
from helion._compiler.cute.chunk_recurrence_config import chunk_recurrence_pipeline
from helion._compiler.cute.warp_specialized_plan import MBarrierRegion
from helion._compiler.cute.warp_specialized_plan import SharedBufferRegion
from helion._compiler.cute.warp_specialized_plan import SharedBufferRequest
from helion._compiler.cute.warp_specialized_plan import WarpRole
from helion._compiler.cute.warp_specialized_plan import WarpSpecializedPipelinePlan
from helion._compiler.cute.warp_specialized_plan import accumulator_tmem_columns
from helion._compiler.cute.warp_specialized_plan import allocate_shared_regions
from helion._compiler.cute.warp_specialized_plan import allocated_tmem_columns
from helion._compiler.cute.warp_specialized_plan import chained_recurrence_tmem_layout
from helion._compiler.cute.warp_specialized_plan import packed_input_tmem_columns


def _plan() -> WarpSpecializedPipelinePlan:
    return WarpSpecializedPipelinePlan(
        threads=256,
        shared_bytes=4096,
        max_shared_bytes_per_block=48_000,
        shared_bytes_per_mp=65_536,
        tmem_columns=128,
        min_blocks_per_mp=1,
        max_threads_per_mp=2048,
        max_blocks_per_mp=32,
        roles=(
            WarpRole("producer", 0, 4, 96),
            WarpRole("consumer", 4, 4, 128),
        ),
        barriers=(
            MBarrierRegion("ready", 0, 2, 1),
            MBarrierRegion("free", 16, 2, 4),
        ),
        shared_buffers=(
            SharedBufferRegion("input", 256, 1024, 0, 2),
            SharedBufferRegion("output", 256, 1024, 2, 4),
        ),
    )


def test_warp_specialized_plan_accepts_lifetime_aliasing() -> None:
    plan = _plan()
    plan.validate()
    assert plan.warps == 8
    assert plan.register_words == 28_672


def test_existing_recurrence_profiles_share_the_resource_planner() -> None:
    wide = chunk_recurrence_pipeline("wide").resource_plan
    compact = chunk_recurrence_pipeline("compact").resource_plan
    assert wide.warps == compact.warps == 16
    assert wide.tmem_columns == 512
    assert compact.tmem_columns == 256
    assert wide.register_words == 59_392
    assert compact.register_words == 32_768


def test_bt32_pipeline_uses_the_shared_resource_contract() -> None:
    pytest.importorskip("cutlass.cute")
    from helion._compiler.cute.chunk_prefill_bt32 import common
    from helion._compiler.cute.chunk_prefill_bt32 import device

    plan = common.PIPELINE_PLAN
    plan.validate()
    assert plan.threads == 1024
    assert plan.register_words == 64_512
    assert {
        region.name: (region.byte_offset, region.stages, region.arrivals)
        for region in plan.barriers
    } == {
        "single_producer": (common.QK_FULL, 9 * common.STAGES, 1),
        "factor_team": (common.V_FREE, 4 * common.STAGES, 4),
        "output_empty": (common.OUT_EMPTY, 1, 1),
        "state_done": (common.DONE, 1, 8),
    }
    assert {
        region.name: (region.byte_offset, region.byte_size)
        for region in plan.shared_buffers
    } == {
        "tmem_address": (common.TMEM_ADDR, 4),
        "factor_stages": (common.QD, common.STAGES * common.STAGE_BYTES),
        "output_stages": (common.OUT, 2 * common.BT * common.DV * 2),
        "gate_bias": (common.GATE_BIAS, common.DK * 4),
    }
    assert plan.shared_buffers[-1].byte_end == 227_920
    initialization = inspect.getsource(device.init_barriers)
    for name in (
        "SINGLE_PRODUCER_BARRIER_OFFSET",
        "SINGLE_PRODUCER_BARRIER_STAGES",
        "SINGLE_PRODUCER_BARRIER_ARRIVALS",
        "FACTOR_TEAM_BARRIER_OFFSET",
        "FACTOR_TEAM_BARRIER_STAGES",
        "FACTOR_TEAM_BARRIER_ARRIVALS",
    ):
        assert name in initialization


def test_bt32_host_config_is_cutlass_free() -> None:
    from helion._compiler.cute.chunk_prefill_bt32 import config

    source = inspect.getsource(config).lower()
    assert "import cutlass" not in source
    config.PIPELINE_PLAN.validate()


def test_resource_planner_has_no_workload_vocabulary() -> None:
    from helion._compiler.cute import warp_specialized_plan

    source = inspect.getsource(warp_specialized_plan).lower()
    assert "kda" not in source
    assert "prefill" not in source
    assert "attention" not in source


@pytest.mark.parametrize(
    "module_name",
    [
        "helion._compiler.cute.chunk_prefill_tmem",
        "helion._compiler.cute.chunk_recurrence_sm100",
        "helion._compiler.cute.chunk_prefill_bt32.common",
        "helion._compiler.cute.chunk_prefill_bt32.factor",
    ],
)
def test_specialized_engines_do_not_redefine_shared_pipeline_primitives(
    module_name: str,
) -> None:
    pytest.importorskip("cutlass.cute")
    module = importlib.import_module(module_name)
    definitions = {
        node.name
        for node in ast.parse(inspect.getsource(module)).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert definitions.isdisjoint(
        {
            "advance_ring_stage",
            "copy_b16x8_async",
            "pack_input_b16x2_to_i32",
            "pack_output_b16x2_to_i32",
            "packed_f32x2_binary",
            "wait_and_flip_mbarrier",
        }
    )


def test_direct_affine_reuses_pipeline_copy_and_swizzle_primitives() -> None:
    pytest.importorskip("cutlass.cute")
    from helion._compiler.cute import short_affine_scan_mma
    from helion._compiler.cute import warp_specialized_primitives

    assert (
        short_affine_scan_mma.copy_b16x8_async
        is warp_specialized_primitives.copy_b16x8_async
    )
    assert (
        short_affine_scan_mma.segmented_swizzle_b16_element_index
        is warp_specialized_primitives.segmented_swizzle_b16_element_index
    )


def test_shared_swizzle_and_lane_mapping_match_legacy_formulas() -> None:
    pytest.importorskip("cutlass.cute")
    from helion._compiler.cute import warp_specialized_primitives as primitives

    for row in range(32):
        for column in range(128):
            byte = (column // 64) * 32 * 128 + row * 128 + (column % 64) * 2
            expected_byte = byte ^ (((byte >> 7) & 7) << 4)
            assert primitives.swizzle_b16_index(row, column, 128, 64, 32, 7) == (
                expected_byte
            )
            assert primitives.segmented_swizzle_b16_element_index(
                row, column, 32, 64, 8, 7
            ) == (expected_byte // 2)
    for lane in range(32):
        matrix_id = lane // 8
        expected = ((matrix_id // 2) * 8 + lane % 8, (matrix_id & 1) * 8)
        assert primitives.matrix_16x16_transposed_lane_coordinates(lane) == expected


def test_pipeline_primitive_source_preserves_instruction_contracts() -> None:
    pytest.importorskip("cutlass.cute")
    from helion._compiler.cute import warp_specialized_primitives as primitives

    source = inspect.getsource(primitives)
    for marker in (
        "mbarrier_wait_parity",
        "mbarrier_arrive_expect_tx",
        "fence_view_async_shared",
        "tcgen05_mma",
        "tcgen05_commit",
        "cp_async_shared_global",
    ):
        assert marker in source


@pytest.mark.parametrize(
    "plan",
    [
        replace(_plan(), threads=255),
        replace(_plan(), min_blocks_per_mp=3),
        replace(_plan(), min_blocks_per_mp=cast("Any", "2")),
        replace(_plan(), threads=128, min_blocks_per_mp=17),
        replace(_plan(), min_blocks_per_mp=33),
        replace(_plan(), shared_bytes=30_000, min_blocks_per_mp=3),
        replace(_plan(), shared_bytes=48_001),
        replace(_plan(), tmem_columns=False),
        replace(_plan(), tmem_columns=512, min_blocks_per_mp=2),
        replace(
            _plan(),
            roles=(
                WarpRole("a", 0, 1, 24),
                WarpRole("b", 1, 7, 32),
            ),
        ),
        replace(
            _plan(),
            roles=(
                WarpRole("a", 0, 5, 128),
                WarpRole("b", 4, 4, 128),
            ),
        ),
        replace(
            _plan(),
            barriers=(
                MBarrierRegion("a", 0, 2, 1),
                MBarrierRegion("b", 8, 2, 1),
            ),
        ),
        replace(
            _plan(),
            barriers=(MBarrierRegion("a", 0, 1, 1 << 20),),
        ),
        replace(
            _plan(),
            barriers=(MBarrierRegion("a", cast("Any", "0"), 1, 1),),
        ),
        replace(
            _plan(),
            barriers=(MBarrierRegion("a", 0, cast("Any", "1"), 1),),
        ),
        replace(
            _plan(),
            shared_buffers=(
                SharedBufferRegion("a", 256, 1024, 0, 3),
                SharedBufferRegion("b", 256, 1024, 2, 4),
            ),
        ),
        replace(
            _plan(),
            shared_buffers=(SharedBufferRegion("a", 257, 1024, 0, 3),),
        ),
        replace(
            _plan(),
            shared_buffers=(SharedBufferRegion("a", 256, 1024, 0, 3, alignment=3),),
        ),
    ],
)
def test_warp_specialized_plan_rejects_invalid_resources(
    plan: WarpSpecializedPipelinePlan,
) -> None:
    with pytest.raises(exc.BackendUnsupported):
        plan.validate()


@pytest.mark.parametrize(
    ("required", "allocated"), [(1, 32), (32, 32), (33, 64), (129, 256), (512, 512)]
)
def test_tmem_column_planning(required: int, allocated: int) -> None:
    assert allocated_tmem_columns(required) == allocated


def test_tmem_shape_planning_is_workload_neutral() -> None:
    assert accumulator_tmem_columns(8) == 8
    assert accumulator_tmem_columns(128) == 128
    assert packed_input_tmem_columns(16) == 8
    assert packed_input_tmem_columns(128) == 64


def test_shared_layout_planning_is_workload_neutral() -> None:
    layout = allocate_shared_regions(
        (
            SharedBufferRequest("first", 96, 64),
            SharedBufferRequest("second", 17, 128),
        ),
        final_alignment=128,
    )
    assert layout.region("first").byte_offset == 0
    assert layout.region("second").byte_offset == 128
    assert layout.allocated_bytes == 256


@pytest.mark.parametrize(
    "buffer_request",
    [
        SharedBufferRequest("bad", 16, alignment=3),
        SharedBufferRequest("too_aligned", 16, alignment=256),
    ],
)
def test_shared_layout_rejects_unrepresentable_alignment(
    buffer_request: SharedBufferRequest,
) -> None:
    with pytest.raises(ValueError):
        allocate_shared_regions((buffer_request,), final_alignment=128)


@pytest.mark.parametrize(
    ("step_width", "expected_offsets", "required_columns"),
    [
        (16, (0, 128, 192, 208, 224, 256), 272),
        (32, (0, 128, 192, 224, 256, 320), 352),
    ],
)
def test_chained_recurrence_tmem_layout(
    step_width: int,
    expected_offsets: tuple[int, ...],
    required_columns: int,
) -> None:
    layout = chained_recurrence_tmem_layout(
        state_width=128,
        step_width=step_width,
    )
    assert tuple(region.column_offset for region in layout.regions) == expected_offsets
    assert layout.required_columns == required_columns
    assert layout.allocated_columns == 512


@pytest.mark.parametrize(
    ("fn", "value"),
    [
        (accumulator_tmem_columns, 7),
        (packed_input_tmem_columns, 3),
        (allocated_tmem_columns, 513),
    ],
)
def test_tmem_shape_planning_rejects_invalid_values(fn, value: int) -> None:
    with pytest.raises(ValueError):
        fn(value)
