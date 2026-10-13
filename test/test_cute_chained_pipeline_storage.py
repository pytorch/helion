"""Pure storage checks, including accepted transports from real CPU codegen."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from dataclasses import replace
from itertools import combinations
from typing import TYPE_CHECKING
from typing import TypedDict
from unittest.mock import patch

import pytest
import torch

from ._cute_aux import _cpu_codegen
from .test_cute_chained_loop_tmem_carry_transport import _resident_carry_sequence
from .test_cute_chained_loop_tmem_transport import _inputs as _carry_inputs
from .test_cute_chained_preparation_cut import _kda_fixture
from .test_cute_chained_preparation_cut import _runtime_values
from .test_cute_chained_preparation_frame import _synthetic
from .test_cute_chained_recurrence_workspace import _pair
import helion
from helion._compiler.cute import chained_loop_tmem_carry_transport as carry_module
from helion._compiler.cute import chained_loop_tmem_transport as bridge_module
from helion._compiler.cute import chained_matmul as chain
from helion._compiler.cute import chained_preparation_pipeline as pipeline_module
from helion._compiler.cute.chained_pipeline_storage import capture_storage_revision
from helion._compiler.cute.chained_pipeline_storage import finalize_pipeline_storage
from helion._compiler.cute.chained_pipeline_storage import select_stage_transports
from helion._compiler.cute.chained_preparation_cohorts import plan_preparation_cohorts
from helion._compiler.cute.chained_preparation_frame import plan_preparation_frame
from helion._compiler.cute.chained_preparation_pipeline import PreparationPipeline
from helion._compiler.cute.chained_prepared_operands import plan_prepared_operands
from helion._compiler.cute.chained_recurrence_workspace import plan_recurrence_workspace
from helion._compiler.cute.chained_workspace import _align

if TYPE_CHECKING:
    from helion._compiler.cute.chained_loop_tmem_carry_transport import (
        LoopTmemCarryTransport,
    )
    from helion._compiler.cute.chained_loop_tmem_transport import LoopTmemTransport
    from helion._compiler.cute.chained_matmul import ChainedMatmulPlan
    from helion._compiler.cute.chained_pipeline_storage import StorageRevision


class _Observed(TypedDict, total=False):
    plan: ChainedMatmulPlan
    pipeline: PreparationPipeline
    revision: StorageRevision | None
    bridges: tuple[LoopTmemTransport, ...]
    carry: LoopTmemCarryTransport | None
    source: str


def _ordinary(kind="cache", *, multiple=False, grouped=False):
    plan, cut, shapes = _pair(grouped=grouped) if multiple else _synthetic(kind)
    if multiple:
        plan = replace(plan, warp_mma_stages=frozenset((0,)))
    plan = replace(plan, threads=512)
    frame = plan_preparation_frame(plan, cut, shapes)
    recurrence = plan_recurrence_workspace(plan, cut, shapes)
    assert frame is not None and recurrence is not None
    available = plan_prepared_operands(plan, frame, recurrence)
    assert available is not None
    pipeline = PreparationPipeline(frame, recurrence, None, 2, 384, 128, 384)
    revision = capture_storage_revision(plan, pipeline, shapes)
    stages = select_stage_transports(pipeline, ())
    assert revision is not None and stages is not None
    return plan, pipeline, revision, stages


def _finalize(case, capacity=232448, **kwargs):
    plan, pipeline, revision, stages = case
    return finalize_pipeline_storage(
        plan, pipeline, stages, revision=revision, capacity_bytes=capacity, **kwargs
    )


def _safe(result):
    for a, b in combinations(result.recurrence.layout.regions, 2):
        assert not (a.overlaps_lifetime(b) and a.overlaps_storage(b))
    assert result.charged_bytes == sum(_align(size) for _, size in result.allocations)
    assert all(size > 0 for _, size in result.allocations)
    assert all(view.byte_offset % 128 == 0 for view in result.carry_views)


@pytest.mark.parametrize("kind", ("dot", "collective", "cache"))
def test_no_selected_transport_preserves_all_requests(kind):
    case = _ordinary(kind)
    original = case[1]
    result = _finalize(case)
    assert result is not None
    _safe(result)
    assert result.recurrence == original.recurrence
    assert result.omitted_results == ()
    assert result.charged_bytes == original.shared_bytes
    assert all(view.pool == "recurrence" for view in result.carry_views)
    assert result.stages is case[3]
    with pytest.raises(FrozenInstanceError):
        result.stages = ()  # pyrefly: ignore [read-only]


@pytest.mark.parametrize("grouped", (False, True))
def test_multiple_ordinary_carries_keep_simultaneous_typed_storage(grouped):
    case = _ordinary(multiple=True, grouped=grouped)
    result = _finalize(case)
    assert result is not None
    _safe(result)
    assert len(result.carry_views) == 2
    assert all(
        view.dtype == torch.float32 and view.shape == (16, 16)
        for view in result.carry_views
    )
    left, right = result.carry_views
    assert (
        left.byte_offset + 1024 <= right.byte_offset
        or right.byte_offset + 1024 <= left.byte_offset
    )
    assert result.recurrence == case[1].recurrence
    if grouped:
        # The B allocation covers the concatenated issue, not one member.
        assert result.recurrence.b_bytes == 2 * 32 * 16


@pytest.mark.parametrize("quota", (True, False, -1, 1.5, "232448", None))
def test_capacity_type_and_range_are_strict(quota):
    assert _finalize(_ordinary(), quota) is None


def test_exact_quota_and_one_byte_overflow():
    case = _ordinary()
    result = _finalize(case)
    assert result is not None
    assert _finalize(case, result.charged_bytes) == result
    assert _finalize(case, result.charged_bytes - 1) is None


@pytest.mark.parametrize(
    "mutation", ("target", "args", "kwargs", "dtype", "shape", "root_args")
)
def test_revision_rejects_semantic_mutation_after_late_proof(mutation):
    case = _ordinary()
    plan = case[0]
    assert plan.region is not None and plan.loop is not None
    node = next(
        node for node in plan.region.nodes if node.target is torch.ops.aten.add.Tensor
    )
    if mutation == "target":
        node.target = torch.ops.aten.sub.Tensor
    elif mutation == "args":
        node.args = node.args[::-1]
    elif mutation == "kwargs":
        node.kwargs = {"alpha": 2}
    elif mutation == "dtype":
        node.meta["val"] = torch.empty_like(node.meta["val"], dtype=torch.float16)
    elif mutation == "shape":
        node.meta["val"] = torch.empty((8, 32))
    else:
        root = plan.loop.root.graph
        with root.inserting_before(next(iter(root.nodes))):
            root.placeholder("new_capture")
    assert _finalize(case) is None


@pytest.mark.parametrize(
    "mutation",
    ("plan", "frame", "interval", "alignment", "order", "partial_group", "shape"),
)
def test_stale_or_inconsistent_resource_proofs_reject(mutation):
    plan, pipeline, revision, stages = _ordinary(multiple=True, grouped=True)
    if mutation == "plan":
        plan = replace(plan)
    elif mutation == "frame":
        pipeline = replace(pipeline, frame=replace(pipeline.frame))
    elif mutation in ("interval", "alignment"):
        region = pipeline.recurrence.layout.regions[0]
        bad = (
            replace(region, live_until=region.live_until + 1)
            if mutation == "interval"
            else replace(region, alignment=256)
        )
        recurrence = replace(
            pipeline.recurrence,
            layout=replace(
                pipeline.recurrence.layout,
                regions=(bad, *pipeline.recurrence.layout.regions[1:]),
            ),
        )
        pipeline = replace(pipeline, recurrence=recurrence)
        revision = capture_storage_revision(plan, pipeline, dict(revision.shapes))
    elif mutation == "order":
        stages = ()
    elif mutation == "partial_group":
        stage = stages[0]
        group = replace(
            stage.group,
            stages=stage.group.stages[:1],
            geometries=stage.group.geometries[:1],
        )
        stages = (replace(stage, group=group),)
    else:
        shapes = dict(revision.shapes)
        shapes[plan.dots[-1]] = (8, 32)
        revision = capture_storage_revision(plan, pipeline, shapes)
    assert revision is not None
    assert _finalize((plan, pipeline, revision, stages)) is None


def _capture_late(kernel, args, config):
    """Observe accepted objects without changing any original emitted source."""
    observed: _Observed = {}
    original_replace = pipeline_module.replace
    original_bridges = bridge_module.prepare_loop_tmem_transports
    original_carry = carry_module.prepare_loop_tmem_carry

    def observe_replace(value, **kwargs):
        result = original_replace(value, **kwargs)
        if isinstance(result, PreparationPipeline):
            observed["pipeline"] = result
        return result

    def observe_bridges(cg, plan, frontiers, groups):
        shapes = {
            node: chain._shape(node)
            for node in plan.region.nodes
            if isinstance(node.meta.get("val"), torch.Tensor)
        }
        observed["plan"] = plan
        observed["revision"] = capture_storage_revision(
            plan, observed["pipeline"], shapes
        )
        result = original_bridges(cg, plan, frontiers, groups)
        observed["bridges"] = result[0]
        return result

    def observe_carry(*args, **kwargs):
        result = original_carry(*args, **kwargs)
        observed["carry"] = result
        return result

    with (
        _cpu_codegen(),
        patch.object(pipeline_module, "replace", observe_replace),
        patch.object(bridge_module, "prepare_loop_tmem_transports", observe_bridges),
        patch.object(carry_module, "prepare_loop_tmem_carry", observe_carry),
    ):
        bound = kernel._bind_isolated(args)
        with bound.env.use_runtime_arg_values(_runtime_values(kernel, args)):
            observed["source"] = bound.to_code(config)
    assert observed["revision"] is not None
    stages = select_stage_transports(
        observed["pipeline"], observed["bridges"], observed["carry"]
    )
    assert stages is not None
    return observed, stages


@pytest.fixture(scope="module")
def actual_transports():
    kernel, args = _kda_fixture()
    config = helion.Config(
        block_sizes=[128],
        num_warps=16,
        cute_chained_mma_schedule="tcgen05_tmem",
        cute_chained_group_contractions=True,
        cute_chained_warp_mma_rows=32,
        cute_chained_pointwise_cache_bytes=4096,
        cute_chained_scan_schedule="warp",
        cute_chained_scratch_layout="xor",
        cute_chained_pointwise_vectorize=True,
        cute_chained_pointwise_unroll=1,
        cute_chained_preparation_pipeline=True,
        cute_chained_pipeline_consumer_warps=4,
        cute_chained_leaf_pipeline="rectangular_tma",
        cute_chained_seed_tile_columns=32,
        cute_chained_pointwise_cache_layout="xor",
    )
    result = _capture_late(kernel, args, config)
    assert result[0]["carry"] is not None and len(result[0]["bridges"]) == 2
    return result


def _actual_case(actual_transports):
    objects, stages = actual_transports
    return objects["plan"], objects["pipeline"], objects["revision"], stages


def test_actual_late_transports_remove_only_omitted_publications(actual_transports):
    case = _actual_case(actual_transports)
    result = _finalize(case)
    assert result is not None
    _safe(result)
    assert case[1].shared_bytes == 225152
    assert case[1].frame.layout.allocated_bytes == 49920
    assert result.recurrence.a_bytes == result.recurrence.b_bytes == 0
    assert result.recurrence.layout.allocated_bytes == 16384
    assert result.charged_bytes == 116608
    assert len(result.carry_views) == 1
    assert result.carry_views[0].pool == "frames"
    assert result.carry_views[0].shape == (128, 128)
    assert result.carry_views[0].byte_offset == 0
    assert "a" not in dict(result.allocations) and "b" not in dict(result.allocations)
    original = {region.name: region for region in case[1].recurrence.layout.regions}
    for region in result.recurrence.layout.regions:
        assert (
            replace(region, byte_offset=original[region.name].byte_offset)
            == original[region.name]
        )


@pytest.mark.parametrize("missing", ("bridges", "carry", "singleton", "group", "all"))
def test_failed_late_proofs_retain_corresponding_materialization(
    actual_transports, missing
):
    objects, _ = actual_transports
    plan, pipeline = objects["plan"], objects["pipeline"]
    bridges = () if missing in ("bridges", "all") else objects["bridges"]
    # Carry slot placement depends on occupied bridge slots. An actual failed
    # bridge would re-run carry proof; conservatively reject that proof here.
    carry = None if missing in ("bridges", "carry", "all") else objects["carry"]
    if missing in ("singleton", "all"):
        pipeline = replace(pipeline, prepared_operands=())
    if missing in ("group", "all"):
        pipeline = replace(pipeline, prepared_groups=())
    revision = capture_storage_revision(
        plan, pipeline, dict(objects["revision"].shapes)
    )
    stages = select_stage_transports(pipeline, bridges, carry)
    assert revision is not None and stages is not None
    result = _finalize((plan, pipeline, revision, stages))
    assert result is not None
    _safe(result)
    if missing in ("bridges", "carry", "all"):
        assert all(view.pool == "recurrence" for view in result.carry_views)
        assert result.recurrence.a_bytes == 32768
    if missing == "group":
        assert result.recurrence.b_bytes == 2 * 160 * 32
    if missing == "all":
        assert result.recurrence == pipeline.recurrence


@pytest.mark.parametrize(
    "corruption",
    ("offset", "dtype", "group", "duplicate", "carry_offset", "carry_dtype"),
)
def test_changed_late_transport_ownership_rejects(actual_transports, corruption):
    objects, stages = actual_transports
    if corruption.startswith("carry"):
        carry = objects["carry"]
        carry = replace(
            carry,
            **(
                {"arena_offset": carry.arena_offset + 32}
                if corruption == "carry_offset"
                else {"dtype": "cutlass.Float16"}
            ),
        )
        stages = select_stage_transports(objects["pipeline"], objects["bridges"], carry)
    elif corruption == "duplicate":
        assert (
            select_stage_transports(
                objects["pipeline"], (*objects["bridges"], objects["bridges"][0])
            )
            is None
        )
        return
    else:
        item = objects["bridges"][0]
        if corruption == "offset":
            item = replace(
                item,
                slot=replace(item.slot, column_offset=item.slot.column_offset + 32),
            )
        elif corruption == "dtype":
            item = replace(item, dtype="cutlass.Float16")
        else:
            slot = replace(
                item.slot,
                candidate=replace(
                    item.slot.candidate, destination_group=stages[0].group
                ),
            )
            item = replace(item, slot=slot)
        stages = select_stage_transports(
            objects["pipeline"], (item, *objects["bridges"][1:]), objects["carry"]
        )
    assert stages is not None
    assert (
        _finalize((objects["plan"], objects["pipeline"], objects["revision"], stages))
        is None
    )


def test_cohort_frame_slab_and_independently_aligned_protocol(actual_transports):
    plan, pipeline, revision, stages = _actual_case(actual_transports)
    cohorts = plan_preparation_cohorts(512, 128, 3, has_tma=True)
    assert cohorts is not None
    assert _finalize((plan, pipeline, revision, stages), cohorts=cohorts) is None
    pipeline = replace(pipeline, slots=3)
    revision = capture_storage_revision(plan, pipeline, dict(revision.shapes))
    assert revision is not None
    case = plan, pipeline, revision, stages
    result = _finalize(case, cohorts=cohorts)
    assert result is not None
    assert dict(result.allocations)["frames"] == 3 * 49920
    assert dict(result.allocations)["slot_barriers"] == 3 * 3 * 8
    assert result.charged_bytes == 166528
    assert _finalize(case, result.charged_bytes, cohorts=cohorts) == result
    assert _finalize(case, result.charged_bytes - 1, cohorts=cohorts) is None


def test_large_slot_barrier_array_has_its_own_alignment_charge(actual_transports):
    plan, pipeline, revision, stages = _actual_case(actual_transports)
    plan = replace(plan, threads=1024)
    pipeline = replace(pipeline, slots=7, preparation_threads=896)
    revision = capture_storage_revision(plan, pipeline, dict(revision.shapes))
    cohorts = plan_preparation_cohorts(1024, 128, 7, has_tma=True)
    assert revision is not None and cohorts is not None
    result = _finalize((plan, pipeline, revision, stages), 1 << 20, cohorts=cohorts)
    assert result is not None
    assert dict(result.allocations)["slot_barriers"] == 168
    assert result.charged_bytes == 7 * 49920 + 16384 + 128 + 128 + 256


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
def test_small_frame_keeps_resident_endpoint_storage_separate(dtype):
    captured = _capture_late(
        _resident_carry_sequence,
        _carry_inputs("cpu", dtype),
        helion.Config(
            num_warps=16,
            cute_chained_mma_schedule="tcgen05_tmem",
            cute_chained_warp_mma_rows=32,
            cute_chained_preparation_pipeline=True,
            cute_chained_pipeline_consumer_warps=4,
        ),
    )
    assert captured[0]["carry"] is not None
    case = _actual_case(captured)
    result = _finalize(case)
    assert result is not None
    _safe(result)
    allocations = dict(result.allocations)
    assert allocations["frames"] < allocations["endpoints"] == 128 * 32 * 4
    assert result.carry_views[0].pool == "endpoints"
    assert _finalize(case, result.charged_bytes - 1) is None


def test_existing_aliases_may_grow_but_cannot_be_rebound():
    plan, pipeline, revision, stages = _ordinary()
    plan.tensor_aliases["input"] = "input_alias"
    revision = capture_storage_revision(plan, pipeline, dict(revision.shapes))
    assert revision is not None
    case = plan, pipeline, revision, stages
    plan.tensor_aliases["new_input"] = "new_alias"
    assert _finalize(case) is not None
    plan.tensor_aliases["input"] = "different_tensor"
    assert _finalize(case) is None
