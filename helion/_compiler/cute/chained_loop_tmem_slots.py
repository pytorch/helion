"""Conservative disjoint placement for loop packed-TMEM operand candidates.

This is a physical allocation, not permission to eagerly evaluate an operand or
omit its FP32 source. Callers separately prove original fragment coordinates,
side-input readiness, casts/masks, issue completion and cross-role publication.
Every accepted packed image occupies its own slot through the ordered loop end;
no release event or alias reuse is inferred from a candidate's issue event.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.fx import Node

from .chained_contraction_groups import ContractionGroup
from .chained_loop_tmem_bridges import LoopTmemBridgeCandidate
from .chained_prepared_operands import _matches_contraction
from .chained_tcgen_stage import StageGeometry
from .chained_tcgen_stage import stage_geometry
from .chained_tmem_accumulator import _full_m128
from .contraction_region import ContractionRegion
from .warp_specialized_plan import VALID_TMEM_COLUMNS
from .warp_specialized_plan import TensorMemoryRegionRequest
from .warp_specialized_plan import allocate_tmem_regions
from .warp_specialized_plan import packed_input_tmem_columns


@dataclass(frozen=True)
class TmemOperandSlot:
    candidate: LoopTmemBridgeCandidate
    column_offset: int
    columns: int


@dataclass(frozen=True)
class TmemOperandSlots:
    slots: tuple[TmemOperandSlot, ...]
    working_columns: int
    required_columns: int
    allocated_columns: int


def _valid_geometry(geometry: StageGeometry) -> bool:
    return (
        isinstance(geometry, StageGeometry)
        and isinstance(geometry.logical, tuple)
        and len(geometry.logical) == 3
        and all(type(size) is int and size > 0 for size in geometry.logical)
        and type(geometry.transpose) is bool
        and stage_geometry(geometry.logical) is not None
    )


def _valid_candidate(candidate: LoopTmemBridgeCandidate, working: int) -> bool:
    """Check retained ownership/geometry facts, without redoing the late proof."""
    region = candidate.region
    group = candidate.destination_group
    if (
        not isinstance(region, ContractionRegion)
        or region.nodes != tuple(region.graph.nodes)
        or type(candidate.source_stage) is not int
        or not 0 <= candidate.source_stage < len(region.contractions)
        or not _valid_geometry(candidate.source_geometry)
        or not _full_m128(candidate.source_geometry)
        or not isinstance(candidate.physical_shape, tuple)
        or len(candidate.physical_shape) != 2
        or any(type(size) is not int for size in candidate.physical_shape)
        or candidate.physical_shape != candidate.source_geometry.physical[:2]
        or candidate.dtype not in (torch.bfloat16, torch.float16)
        or not isinstance(group, ContractionGroup)
        or not isinstance(group.stages, tuple)
        or not isinstance(group.geometries, tuple)
        or not isinstance(candidate.operands, tuple)
        or not group.stages
        or len(group.stages) != len(group.geometries)
        or len(group.stages) != len(candidate.operands)
        or any(
            type(index) is not int
            or not candidate.source_stage < index < len(region.contractions)
            for index in group.stages
        )
        or tuple(sorted(set(group.stages))) != group.stages
        or any(not _valid_geometry(geometry) for geometry in group.geometries)
        or type(candidate.publication_event) is not int
        or candidate.publication_event != 2 * candidate.source_stage + 1
        or type(candidate.issue_event) is not int
        or candidate.issue_event != 2 * group.stages[0]
    ):
        return False
    source = region.contractions[candidate.source_stage]
    if (
        candidate.source is not source.node
        or source.result_dtype != torch.float32
        or not _matches_contraction(source)
        or any(
            type(old) is int and old != new
            for old, new in zip(
                source.shape, candidate.source_geometry.logical, strict=True
            )
        )
        or max(candidate.source_geometry.physical[1], group.physical[1]) > working
    ):
        return False
    nodes = set(region.nodes)
    if (
        not isinstance(candidate.expression_nodes, tuple)
        or any(
            not isinstance(node, Node)
            or node.graph is not region.graph
            or node not in nodes
            for node in (
                candidate.source,
                *candidate.operands,
                *candidate.expression_nodes,
            )
        )
        or candidate.source not in candidate.expression_nodes
        or any(
            operand not in candidate.expression_nodes for operand in candidate.operands
        )
        or len(set(candidate.expression_nodes)) != len(candidate.expression_nodes)
    ):
        return False
    for index, geometry, operand in zip(
        group.stages, group.geometries, candidate.operands, strict=True
    ):
        spec = region.contractions[index]
        if (
            not _matches_contraction(spec)
            or spec.node not in nodes
            or spec.node.graph is not region.graph
            or spec.operand_dtypes != (candidate.dtype, candidate.dtype)
            or operand is not (spec.rhs if geometry.transpose else spec.lhs)
            or geometry.logical[int(geometry.transpose)] != 128
            or geometry.physical[::2] != candidate.physical_shape
            or any(
                type(old) is int and old != new
                for old, new in zip(spec.shape, geometry.logical, strict=True)
            )
        ):
            return False
    return True


def plan_tmem_operand_slots(
    candidates: tuple[LoopTmemBridgeCandidate, ...], working_columns: int
) -> TmemOperandSlots | None:
    """Append unique 32-column-aligned slots after the complete working C arena.

    Candidate order and graph identities are preserved. The caller supplies
    the maximum physical C width of *all* issued groups, not merely groups
    mentioned by these candidates. Alignment gaps count toward the allocation;
    the last slot is not rounded until selecting the hardware envelope.
    """
    if (
        type(working_columns) is not int
        or not 0 < working_columns <= VALID_TMEM_COLUMNS[-1]
        or not isinstance(candidates, tuple)
        or any(not isinstance(item, LoopTmemBridgeCandidate) for item in candidates)
    ):
        return None
    if candidates and any(
        item.region is not candidates[0].region for item in candidates
    ):
        return None
    if (
        any(not _valid_candidate(item, working_columns) for item in candidates)
        or len({item.source_stage for item in candidates}) != len(candidates)
        or len({item.destination_group.stages[0] for item in candidates})
        != len(candidates)
    ):
        return None
    requests = [TensorMemoryRegionRequest("working", working_columns)]
    cursor = working_columns
    for index, candidate in enumerate(candidates):
        aligned = (cursor + 31) // 32 * 32
        columns = packed_input_tmem_columns(candidate.physical_shape[1])
        if aligned + columns > VALID_TMEM_COLUMNS[-1]:
            return None
        if aligned != cursor:
            requests.append(
                TensorMemoryRegionRequest(f"padding_{index}", aligned - cursor)
            )
        requests.append(TensorMemoryRegionRequest(f"packed_{index}", columns))
        cursor = aligned + columns
    layout = allocate_tmem_regions(tuple(requests))
    return TmemOperandSlots(
        tuple(
            TmemOperandSlot(
                item,
                layout.region(f"packed_{index}").column_offset,
                layout.region(f"packed_{index}").columns,
            )
            for index, item in enumerate(candidates)
        ),
        working_columns,
        layout.required_columns,
        layout.allocated_columns,
    )
