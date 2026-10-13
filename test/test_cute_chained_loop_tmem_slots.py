from __future__ import annotations

from dataclasses import FrozenInstanceError
from dataclasses import replace
from typing import Any
from typing import cast
from unittest.mock import patch

import pytest
import torch
from torch.fx import Graph

from .test_cute_chained_group_guards import _convert
from .test_cute_chained_group_guards import _dot
from .test_cute_chained_group_guards import _input
from .test_cute_chained_loop_tmem_bridges import _candidate
from .test_cute_chained_loop_workspace import _loop_plan
from .test_cute_chained_preparation_cut import _kda_fixture
from .test_cute_chained_preparation_frame import _capture
import helion
from helion._compiler.cute import chained_loop_tmem_slots as slots
from helion._compiler.cute.chained_contraction_groups import ContractionGroup
from helion._compiler.cute.chained_loop_tmem_bridges import plan_loop_tmem_bridges
from helion._compiler.cute.chained_tcgen_stage import StageGeometry
from helion._compiler.cute.warp_specialized_plan import VALID_TMEM_COLUMNS
from helion._compiler.cute.warp_specialized_plan import allocated_tmem_columns


def _one(width=32, **kwargs):
    plan, groups = _candidate(width, **kwargs)
    candidates = plan_loop_tmem_bridges(plan, groups)
    assert candidates is not None and len(candidates) == 1
    return candidates[0]


def _many(widths=(32, 48, 16)):
    graph = Graph()
    carries, outputs, groups = [], [], []
    for index, width in enumerate(widths):
        dtype = torch.bfloat16 if index % 2 == 0 else torch.float16
        left = _input(graph, f"left_{index}", (128, 16), dtype)
        right = _input(graph, f"right_{index}", (16, width), dtype)
        source = _dot(graph, left, right)
        operand = _convert(graph, source, dtype)
        other = _input(graph, f"other_{index}", (width, 16), dtype)
        state = _input(graph, f"state_{index}", (128, 16), torch.float32)
        outputs.append(_dot(graph, operand, other, state))
        carries.append(state)
        groups.extend(
            (
                ContractionGroup(
                    (2 * index,), (StageGeometry((128, width, 16), False),)
                ),
                ContractionGroup(
                    (2 * index + 1,), (StageGeometry((128, 16, width), False),)
                ),
            )
        )
    plan = replace(
        _loop_plan(graph, tuple(carries), tuple(outputs)),
        contraction_groups=tuple(groups),
    )
    candidates = plan_loop_tmem_bridges(plan, tuple(groups))
    assert candidates is not None and len(candidates) == len(widths)
    return candidates


@pytest.mark.parametrize("working", [1, 16, 32, 33, 64, 65, 128, 160, 256, 257, 512])
def test_empty_candidates_preserve_existing_working_allocation(working):
    result = slots.plan_tmem_operand_slots((), working)
    assert result == slots.TmemOperandSlots(
        (), working, working, allocated_tmem_columns(working)
    )


@pytest.mark.parametrize("working", [0, -1, True, False, 32.0, 513, None])
def test_invalid_working_columns_fail_closed(working):
    assert slots.plan_tmem_operand_slots((), working) is None


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "width,transpose",
    [
        (width, transpose)
        for width in (16, 32, 48, 64, 128)
        for transpose in (False, True)
    ]
    + [(256, False)],
)
def test_typed_physical_width_and_orientation_preserve_original_candidate(
    dtype, transpose, width
):
    candidate = _one(width, dtype=dtype, transpose=transpose, mode="rounding")
    result = slots.plan_tmem_operand_slots((candidate,), 256)
    assert result is not None
    (slot,) = result.slots
    assert slot.candidate is candidate
    assert (slot.column_offset, slot.columns) == (256, width // 2)
    assert result.required_columns == 256 + width // 2
    assert result.allocated_columns == 512
    assert slot.candidate.expression_nodes is candidate.expression_nodes


def test_disjoint_slots_preserve_order_and_count_alignment_gaps_without_reuse():
    candidates = _many()
    allocator = slots.allocate_tmem_regions
    with patch.object(slots, "allocate_tmem_regions", wraps=allocator) as allocate:
        result = slots.plan_tmem_operand_slots(candidates, 160)
    assert result is not None
    assert [(slot.column_offset, slot.columns) for slot in result.slots] == [
        (160, 16),
        (192, 24),
        (224, 8),
    ]
    assert (result.required_columns, result.allocated_columns) == (232, 256)
    assert tuple(slot.candidate for slot in result.slots) == candidates
    assert all(
        slot.candidate is original
        for slot, original in zip(result.slots, candidates, strict=True)
    )
    (requests,) = allocate.call_args.args
    assert [(item.name, item.columns) for item in requests] == [
        ("working", 160),
        ("packed_0", 16),
        ("padding_1", 16),
        ("packed_1", 24),
        ("padding_2", 8),
        ("packed_2", 8),
    ]
    occupied = set(range(result.working_columns))
    for slot in result.slots:
        addresses = set(range(slot.column_offset, slot.column_offset + slot.columns))
        assert slot.column_offset % 32 == 0
        assert not addresses & occupied
        occupied |= addresses
    assert candidates[0].issue_event < candidates[1].publication_event
    # An earlier completed consumer does not license recycling its packed slot.
    reversed_result = slots.plan_tmem_operand_slots(candidates[::-1], 160)
    assert reversed_result is not None
    assert [slot.candidate.source_stage for slot in reversed_result.slots] == [4, 2, 0]
    assert [slot.column_offset for slot in reversed_result.slots] == [160, 192, 224]
    assert reversed_result.required_columns == 240


@pytest.mark.parametrize(
    "width,working,required",
    [
        (16, 16, 40),
        (16, 480, 488),
        (32, 32, 48),
        (64, 480, 512),
        (128, 448, 512),
        (256, 384, 512),
    ],
)
def test_capacity_boundary_counts_only_final_live_columns(width, working, required):
    result = slots.plan_tmem_operand_slots((_one(width),), working)
    assert result is not None
    assert result.required_columns == required
    assert result.allocated_columns in VALID_TMEM_COLUMNS
    assert result.allocated_columns == allocated_tmem_columns(required)


@pytest.mark.parametrize(
    "width,working", [(16, 15), (16, 497), (32, 512), (64, 481), (128, 449), (256, 385)]
)
def test_understated_working_arena_and_overflow_reject(width, working):
    assert slots.plan_tmem_operand_slots((_one(width),), working) is None


def test_multiple_slots_can_overflow_even_when_each_alone_fits():
    candidates = _many((128, 128))
    assert all(
        slots.plan_tmem_operand_slots((candidate,), 416) is not None
        for candidate in candidates
    )
    assert slots.plan_tmem_operand_slots(candidates, 416) is None


def test_records_are_immutable_and_duplicate_or_foreign_graph_candidates_reject():
    first, second = _one(), _one()
    result = slots.plan_tmem_operand_slots((first,), 160)
    assert result is not None
    with pytest.raises(FrozenInstanceError):
        result.working_columns = 0  # pyrefly: ignore [read-only]
    with pytest.raises(FrozenInstanceError):
        result.slots[0].column_offset = 0  # pyrefly: ignore [read-only]
    assert slots.plan_tmem_operand_slots((first, first), 160) is None
    assert slots.plan_tmem_operand_slots((first, second), 160) is None
    assert slots.plan_tmem_operand_slots(cast("Any", (object(),)), 160) is None
    assert slots.plan_tmem_operand_slots(cast("Any", [first]), 160) is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_stage", True),
        ("source_stage", -1),
        ("source_stage", 20),
        ("physical_shape", (64, 32)),
        ("physical_shape", (128, 31)),
        ("physical_shape", (128, 32.0)),
        ("physical_shape", (128,)),
        ("physical_shape", None),
        ("dtype", torch.float32),
        ("dtype", torch.int16),
        ("publication_event", True),
        ("publication_event", 0),
        ("issue_event", 1),
        ("source_geometry", StageGeometry((64, 32, 16), False)),
        ("source_geometry", StageGeometry((128, 32, 16), True)),
        ("source_geometry", StageGeometry(cast("Any", (128, 32)), False)),
        ("source_geometry", StageGeometry((256, 128, 16), True)),
        ("source_geometry", None),
        ("region", None),
        ("operands", None),
        ("expression_nodes", ()),
        ("expression_nodes", None),
    ],
)
def test_malformed_candidate_fields_fail_closed(field, value):
    candidate = replace(_one(), **{field: value})
    assert slots.plan_tmem_operand_slots((candidate,), 160) is None


@pytest.mark.parametrize(
    "mode",
    [
        "revision",
        "source_node",
        "operand_node",
        "expression_node",
        "result_dtype",
        "operand_dtype",
        "group_member",
        "group_geometry",
        "source_kwargs",
    ],
)
def test_stale_or_cross_graph_ownership_and_metadata_fail_closed(mode):
    candidate = _one()
    foreign = _one()
    if mode == "revision":
        candidate.region.graph.placeholder("new_revision")
    elif mode == "source_node":
        candidate = replace(candidate, source=foreign.source)
    elif mode == "operand_node":
        candidate = replace(candidate, operands=foreign.operands)
    elif mode == "expression_node":
        candidate = replace(
            candidate, expression_nodes=(*candidate.expression_nodes, foreign.source)
        )
    elif mode == "result_dtype":
        candidate.source.meta["val"] = torch.empty((128, 32), dtype=torch.float16)
    elif mode == "operand_dtype":
        candidate.operands[0].meta["val"] = torch.empty((128, 32), dtype=torch.float16)
    elif mode == "group_member":
        candidate = replace(
            candidate,
            destination_group=replace(candidate.destination_group, stages=(0,)),
        )
    elif mode == "group_geometry":
        candidate = replace(
            candidate,
            destination_group=replace(
                candidate.destination_group,
                geometries=(StageGeometry((128, 16, 64), False),),
            ),
        )
    else:
        candidate.source.kwargs = {"unexpected": 1}
    assert slots.plan_tmem_operand_slots((candidate,), 160) is None


def test_actual_kda_candidates_need_208_columns_inside_existing_256_envelope():
    kernel, args = _kda_fixture()
    plan, _, _ = _capture(
        kernel,
        args,
        helion.Config(
            block_sizes=[128],
            num_warps=16,
            num_stages=2,
            cute_chained_mma_schedule="tcgen05_tmem",
            cute_chained_group_contractions=True,
            cute_chained_scratch_layout="xor",
            cute_chained_pointwise_vectorize=True,
            cute_chained_scan_schedule="warp",
            cute_chained_pointwise_cache_bytes=4096,
            cute_chained_pointwise_unroll=8,
            cute_chained_warp_mma_rows=32,
        ),
    )
    assert plan.contraction_groups is not None
    groups = tuple(
        group
        for group in plan.contraction_groups
        if group.stages[0] not in plan.warp_mma_stages
    )
    candidates = plan_loop_tmem_bridges(plan, groups)
    assert candidates is not None and len(candidates) == 2
    with patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA forbidden")):
        result = slots.plan_tmem_operand_slots(
            candidates, max(group.physical[1] for group in groups)
        )
    assert result is not None
    assert [
        (
            slot.candidate.source_stage,
            slot.candidate.destination_group.stages,
            slot.column_offset,
            slot.columns,
        )
        for slot in result.slots
    ] == [(10, (11,), 160, 16), (11, (13, 14), 192, 16)]
    assert (
        result.working_columns,
        result.required_columns,
        result.allocated_columns,
    ) == (160, 208, 256)
