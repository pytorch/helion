from __future__ import annotations

from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from .test_cute_chained_loop_tmem_transport import _source
from .test_cute_chained_preparation_cut import _kda_fixture
from .test_cute_chained_prepared_groups import _candidate
from .test_cute_chained_seed_tile_integration import _config
from helion import exc
from helion._compiler.cute import chained_preparation_pipeline as pipeline_module
from helion._compiler.cute.chained_frontier_groups import plan_frontier_group
from helion._compiler.cute.chained_frontier_ownership import FrontierOwnership
from helion._compiler.cute.chained_prepared_groups import coallocate_prepared_groups
from helion._compiler.cute.chained_prepared_groups import prepared_group_candidates
from helion._compiler.cute.chained_prepared_operands import plan_prepared_operands


def _native_frame(dtype=torch.bfloat16, size=32, transpose=(True, False)):
    plan, frame, recurrence = _candidate((size, size), size, dtype, transpose=transpose)
    candidates = prepared_group_candidates(plan, frame, recurrence)
    assert candidates is not None and len(candidates) == 1
    placement = coallocate_prepared_groups(
        plan, frame, recurrence, candidates, frame_capacity_bytes=1 << 20
    )
    assert placement is not None
    frame = placement.frame
    operands = plan_prepared_operands(plan, frame, recurrence)
    assert operands is not None
    first = next(action.event for action in frame.actions if action.kind == "frontier")
    group = plan_frontier_group(frame, first)
    assert group is not None
    return frame, group, operands, placement.groups


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("size,columns", [(16, 8), (32, 16), (64, 32)])
@pytest.mark.parametrize("threads", [32, 128, 256])
def test_exact_native_mixed_group_selects_without_mutating_or_activating(
    dtype, size, columns, threads
):
    frame, group, operands, bindings = _native_frame(dtype, size)
    before = (repr(frame), repr(group), repr(operands), repr(bindings))
    graph_before = str(frame.cut.region.graph)
    policy = FrontierOwnership(columns)
    with patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA forbidden")):
        selected = policy.select(frame, group, operands, bindings, threads)
    assert selected is not None and selected.changed
    assert selected.shape == (size, size) and selected.tile_columns == columns
    assert selected.thread_columns == columns // 8
    assert selected.thread_rows == threads // (columns // 8)
    assert not policy.activated
    assert before == (repr(frame), repr(group), repr(operands), repr(bindings))
    assert graph_before == str(frame.cut.region.graph)
    with pytest.raises(exc.BackendUnsupported, match="successfully emitted"):
        policy.validate()
    # Only a successful caller emission may publish activation.
    policy.activated = True
    policy.validate()


@pytest.mark.parametrize("columns", [0, 32, 64, 3, 24])
def test_default_identical_or_unrepresentable_geometry_is_not_effective(columns):
    frame, group, operands, bindings = _native_frame()
    policy = FrontierOwnership(columns)
    assert policy.select(frame, group, operands, bindings, 128) is None
    assert not policy.activated
    if columns:
        with pytest.raises(exc.BackendUnsupported):
            policy.validate()
    else:
        policy.validate()


def test_zero_does_not_inspect_frame_or_discover_native_groups():
    class Untouchable:
        def __getattribute__(self, name):
            raise AssertionError(name)

    policy = FrontierOwnership(0)
    item = Untouchable()
    assert policy.select(item, item, item, item, 128) is None  # type: ignore[arg-type]
    policy.validate()


@pytest.mark.parametrize("columns", [True, False, None, "32", 32.0, -1])
def test_constructor_requires_strict_nonnegative_integer(columns):
    with pytest.raises(ValueError, match="nonnegative integer"):
        FrontierOwnership(columns)


@pytest.mark.parametrize("threads", [True, 0, 16, 96, 384, 128.0])
def test_native_copy_requires_complete_power_of_two_warp_team(threads):
    assert FrontierOwnership(16).select(*_native_frame(), threads) is None


@pytest.mark.parametrize("transpose", [(True, True), (False, False)])
def test_same_orientation_is_not_selected(transpose):
    assert (
        FrontierOwnership(16).select(*_native_frame(transpose=transpose), 128) is None
    )


def test_discovery_candidate_omitted_by_late_admission_is_not_authority():
    frame, group, operands, bindings = _native_frame()
    assert bindings
    assert FrontierOwnership(16).select(frame, group, operands, (), 128) is None


@pytest.mark.parametrize(
    "bad",
    [
        "group_offset",
        "group_size",
        "group_shape",
        "group_layout",
        "group_lifetime",
        "group_stages",
        "member_offset",
        "member_row",
        "member_shape",
        "member_modes",
        "member_layout",
        "member_dtype",
        "member_stage",
        "member_operand",
        "member_geometry",
        "member_node",
        "duplicate_member",
        "partial_group",
    ],
)
def test_stale_complete_native_binding_cannot_select(bad):
    frame, group, operands, bindings = _native_frame()
    binding = bindings[0]
    candidate = binding.candidate
    member = candidate.members[0]
    if bad == "group_offset":
        binding = replace(binding, byte_offset=binding.byte_offset + 128)
    elif bad == "group_size":
        candidate = replace(candidate, byte_size=candidate.byte_size + 128)
    elif bad == "group_shape":
        candidate = replace(candidate, physical_shape=(32, 64))
    elif bad == "group_layout":
        candidate = replace(candidate, native_layout=())
    elif bad == "group_lifetime":
        candidate = replace(candidate, live_from=candidate.live_from + 1)
    elif bad == "group_stages":
        candidate = replace(candidate, group=replace(candidate.group, stages=(99, 100)))
    elif bad == "member_offset":
        member = replace(member, byte_offset=member.byte_offset + 128)
    elif bad == "member_row":
        member = replace(member, row_offset=member.row_offset + 16)
    elif bad == "member_shape":
        member = replace(member, physical_shape=(16, 32))
    elif bad == "member_modes":
        member = replace(member, logical_modes=(1, 0))
    elif bad == "member_layout":
        member = replace(member, native_layout=())
    elif bad == "member_dtype":
        member = replace(member, buffer=replace(member.buffer, dtype=torch.float16))
    elif bad == "member_stage":
        member = replace(member, stage=99)
    elif bad == "member_operand":
        member = replace(member, operand_index=1 - member.operand_index)
    elif bad == "member_geometry":
        member = replace(member, geometry=replace(member.geometry, transpose=False))
    elif bad == "member_node":
        member = replace(
            member, buffer=replace(member.buffer, node=group.buffers[1].node)
        )
    elif bad == "duplicate_member":
        candidate = replace(candidate, members=(member, member))
    elif bad == "partial_group":
        candidate = replace(candidate, members=(member,))
    if bad.startswith("member_"):
        candidate = replace(candidate, members=(member, *candidate.members[1:]))
    if bad != "group_offset":
        binding = replace(binding, candidate=candidate)
    assert FrontierOwnership(16).select(frame, group, operands, (binding,), 128) is None


@pytest.mark.parametrize(
    "bad",
    [
        "repacked",
        "future_write",
        "group_members",
        "metadata",
        "graph_revision",
        "shape",
        "xor",
    ],
)
def test_current_frame_and_original_typed_metadata_are_required(bad):
    frame, group, operands, bindings = _native_frame()
    if bad in ("repacked", "future_write"):
        selected = frame.layout.region(group.buffers[0].name)
        altered = replace(
            selected,
            byte_offset=frame.layout.allocated_bytes
            if bad == "repacked"
            else selected.byte_offset,
            live_from=group.first_event + 1
            if bad == "future_write"
            else selected.live_from,
        )
        frame = replace(
            frame,
            layout=replace(
                frame.layout,
                regions=tuple(
                    altered if item is selected else item
                    for item in frame.layout.regions
                ),
                allocated_bytes=frame.layout.allocated_bytes + selected.byte_size,
            ),
        )
    elif bad == "group_members":
        group = replace(group, buffers=group.buffers[::-1])
    elif bad == "metadata":
        assert group.buffers[0].node is not None
        group.buffers[0].node.meta["val"] = torch.empty((32, 32), dtype=torch.float32)
    elif bad == "graph_revision":
        frame.cut.region.graph.placeholder("later")
    elif bad in ("shape", "xor"):
        buffer = replace(
            group.buffers[0],
            shape=(16, 64) if bad == "shape" else (32, 32),
            dtype=torch.bfloat16 if bad == "shape" else torch.float32,
        )
        frame = replace(
            frame,
            buffers=tuple(
                buffer if item is group.buffers[0] else item for item in frame.buffers
            ),
        )
        group = replace(group, buffers=(buffer, *group.buffers[1:]))
    assert FrontierOwnership(16).select(frame, group, operands, bindings, 128) is None


def test_actual_kda_only_final_mixed_native_group_is_capable_cpu():
    kernel, args = _kda_fixture()
    config = _config(32, pipeline=True)
    config.config.update(
        cute_chained_leaf_pipeline="rectangular_tma",
        cute_chained_leaf_count=4,
        cute_chained_preparation_cohorts=3,
        cute_chained_preparation_unroll=1,
        cute_chained_register_islands=True,
        cute_chained_vector_group=True,
        cute_chained_operand_retention=True,
        cute_chained_pointwise_cache_layout="xor",
    )
    original = pipeline_module._prepare
    seen = []

    def observe(cg, plan, pipeline, *rest, **kwargs):
        frame = pipeline.frame
        policy = FrontierOwnership(32)
        matches = []
        for action in frame.actions:
            group = plan_frontier_group(frame, action.event)
            if group is None:
                continue
            selected = policy.select(
                frame, group, pipeline.prepared_operands, pipeline.prepared_groups, 128
            )
            if selected is not None:
                matches.append((group, selected))
        assert len(matches) == 1
        group, selected = matches[0]
        assert len(group.buffers) == 3 and selected.shape == (32, 128)
        assert (selected.thread_rows, selected.thread_columns, selected.trips) == (
            32,
            4,
            4,
        )
        assert not policy.activated
        # A singleton's implicit (0,1) contract participates alongside the
        # explicit (1,0) grouped member, but cannot activate alone.
        assert any(item.buffer in group.buffers for item in pipeline.prepared_operands)
        assert policy.select(frame, group, (), pipeline.prepared_groups, 128) is None
        assert policy.select(frame, group, pipeline.prepared_operands, (), 128) is None
        singleton = next(
            item for item in pipeline.prepared_operands if item.buffer in group.buffers
        )
        for altered in (
            replace(singleton, region=replace(singleton.region, byte_offset=0)),
            replace(singleton, native_layout=()),
            replace(singleton, operand_index=1 - singleton.operand_index),
            replace(singleton, physical_shape=(128, 32)),
        ):
            stale = tuple(
                altered if item is singleton else item
                for item in pipeline.prepared_operands
            )
            assert (
                policy.select(frame, group, stale, pipeline.prepared_groups, 128)
                is None
            )
        seen.append(True)
        return original(cg, plan, pipeline, *rest, **kwargs)

    before = torch.cuda.is_initialized()
    with patch.object(pipeline_module, "_prepare", observe):
        source = _source(kernel, args, config)
    assert seen == [True]
    assert "chain_retained_operand_0" in source
    assert torch.cuda.is_initialized() == before
