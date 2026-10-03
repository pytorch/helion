"""Exact FP32 accumulator transport across adjacent TCgen05 issues.

This does not fuse or reassociate an addition. A singleton result stays in the
same TMEM columns until the next issue consumes it as its direct accumulator.
The emitter must retain completion waits, preserve those columns while seeding
other members, and keep the role's TMEM allocation exclusively owned.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
import math
from typing import TYPE_CHECKING

import torch
from torch.fx.node import map_arg

from .chained_contraction_groups import ContractionGroup
from .chained_tcgen_stage import StageGeometry
from .chained_tcgen_stage import stage_geometry

if TYPE_CHECKING:
    from torch.fx import Node

    from .chained_matmul import ChainedMatmulPlan
    from .contraction_region import ContractionRegion
    from .contraction_region import ContractionSpec


@dataclass(frozen=True)
class TmemAccumulatorResidency:
    region: ContractionRegion
    source_stage: int
    destination_group: int
    destination_member: int
    source: Node
    destination: Node
    source_geometry: StageGeometry
    destination_geometry: StageGeometry
    offset: int
    columns: int


def _full_m128(geometry: StageGeometry) -> bool:
    m, n, k = geometry.logical
    if (
        any(type(size) is not int or size <= 0 for size in (m, n, k))
        or stage_geometry(geometry.logical) is None
    ):
        return False
    rows, columns = (n, m) if geometry.transpose else (m, n)
    # The full M128 FP32 fragment has ((M,N),1,1):((65536,1),0,0).
    # Initially exclude padding and sparse M64 ownership. Actual CuTe layout
    # and segmented-copy tests establish this physical transport contract.
    return rows == 128 and columns == geometry.physical[1] and columns <= 256


def _fp32_dot(spec: ContractionSpec) -> bool:
    node = spec.node
    value = node.meta.get("val")
    return (
        len(node.args) == 4
        and node.args[:3] == (spec.lhs, spec.rhs, spec.accumulator)
        and node.args[3] == spec.requested_out_dtype
        and spec.requested_out_dtype in (None, torch.float32)
        and spec.result_dtype == torch.float32
        and spec.operand_dtypes[0] in (torch.bfloat16, torch.float16)
        and spec.operand_dtypes[0] == spec.operand_dtypes[1]
        and all(
            isinstance(operand.meta.get("val"), torch.Tensor)
            and operand.meta["val"].dtype == dtype
            for operand, dtype in zip(
                (spec.lhs, spec.rhs), spec.operand_dtypes, strict=True
            )
        )
        and (
            spec.accumulator is None
            or spec.accumulator_dtype == torch.float32
            and isinstance(spec.accumulator.meta.get("val"), torch.Tensor)
            and spec.accumulator.meta["val"].dtype == torch.float32
        )
        and isinstance(value, torch.Tensor)
        and value.dtype == torch.float32
        and value.ndim == 2
    )


def _groups(plan: ChainedMatmulPlan) -> tuple[ContractionGroup, ...] | None:
    if plan.contraction_groups is not None:
        return plan.contraction_groups
    singletons = []
    for index, shape in enumerate(plan.shapes):
        if len(shape) != 3 or any(type(size) is not int or size <= 0 for size in shape):
            return None
        geometry = stage_geometry(shape)
        if geometry is None:
            return None
        singletons.append(ContractionGroup((index,), (geometry,)))
    return tuple(singletons)


def validate_tmem_accumulator_residency(
    plan: ChainedMatmulPlan, residency: TmemAccumulatorResidency
) -> bool:
    """Revalidate a retained proof against the live graph and issue sequence."""
    groups = _groups(plan)
    return (
        groups is not None
        and residency.region is plan.region
        and plan_tmem_accumulator_residency(
            plan,
            tuple(
                group
                for group in groups
                if group.stages and group.stages[0] not in plan.warp_mma_stages
            ),
        )
        == residency
    )


def plan_tmem_accumulator_residency(
    plan: ChainedMatmulPlan,
    groups: tuple[ContractionGroup, ...],
) -> TmemAccumulatorResidency | None:
    """Choose at most one exact edge in the caller's ordered TCgen role.

    ``groups`` must contain all non-warp issues from the admitted plan, in their
    original order. Preparation warp issues may overlap in another role but
    cannot touch this role's TMEM. Select the largest removable logical FP32 C,
    breaking ties by original source stage. No compiler context is consulted.

    Initial transport is singleton-to-first-member, offset zero, full M128 and
    identical logical coordinates/orientation. No implicit cast, identity view,
    arithmetic seed, or additional use is admitted by this narrow policy.
    """
    region = plan.region
    if (
        region is None
        or plan.strategy != "tcgen05_tmem"
        or plan.direct_output
        or plan.initialized_accumulator is not None
        or plan.late_rhs_reuse is not None
        or plan.k_schedule is not None
        or tuple(region.graph.nodes) != region.nodes
        or plan.dots != tuple(spec.node for spec in region.contractions)
        or len(plan.dots) != len(plan.shapes)
    ):
        return None
    positions = {node: index for index, node in enumerate(region.nodes)}
    if any(
        node.graph is not region.graph
        or any(
            positions.get(source, len(positions)) >= index
            for source in node.all_input_nodes
        )
        for index, node in enumerate(region.nodes)
    ):
        return None
    all_groups = _groups(plan)
    if all_groups is None:
        return None
    if tuple(stage for group in all_groups for stage in group.stages) != tuple(
        range(len(plan.dots))
    ):
        return None
    for group in all_groups:
        if not group.stages or len(group.stages) != len(group.geometries):
            return None
        if any(
            geometry.logical != plan.shapes[index]
            for index, geometry in zip(group.stages, group.geometries, strict=True)
        ):
            return None
    expected = tuple(
        group for group in all_groups if group.stages[0] not in plan.warp_mma_stages
    )
    if groups != expected:
        return None
    candidates = []
    for prior, following in pairwise(groups):
        if len(prior.stages) != 1:
            continue
        source_stage, destination_member = prior.stages[0], following.stages[0]
        source, destination = plan.dots[source_stage], plan.dots[destination_member]
        source_spec, destination_spec = (
            region.contractions[source_stage],
            region.contractions[destination_member],
        )
        source_geometry, destination_geometry = (
            prior.geometries[0],
            following.geometries[0],
        )
        if (
            not _fp32_dot(source_spec)
            or not _fp32_dot(destination_spec)
            or destination_spec.accumulator is not source
            or destination_spec.accumulator_dtype != torch.float32
            or tuple(source.users) != (destination,)
            or sum(item is source for item in destination.args) != 1
            or source in region.live_outs
            or any(carry.output is source for carry in region.carries)
            or source_geometry.logical[:2] != destination_geometry.logical[:2]
            or source_geometry.transpose != destination_geometry.transpose
            or not _full_m128(source_geometry)
            or not _full_m128(destination_geometry)
            or any(
                type(extent) is int and extent != configured
                for node, geometry in (
                    (source, source_geometry),
                    (destination, destination_geometry),
                )
                for extent, configured in zip(
                    node.meta["val"].shape, geometry.logical[:2], strict=True
                )
            )
            or following.offsets[0] != 0
            or following.physical[1] > 256
            or any(
                geometry.physical[::2] != following.physical[::2]
                for geometry in following.geometries
            )
            or any(
                source is entry.node or source in entry.dependencies
                for entry in (
                    () if plan.pointwise_cache is None else plan.pointwise_cache.entries
                )
            )
        ):
            continue
        # Dependency-only kwargs can duplicate the same direct user. They may
        # not turn a sole explicit-accumulator use into another compiler read.
        extra_uses: list[Node] = []
        map_arg(
            destination.kwargs, lambda node, uses=extra_uses: uses.append(node) or node
        )
        if source in extra_uses:
            continue
        candidates.append(
            TmemAccumulatorResidency(
                region,
                source_stage,
                following.stages[0],
                destination_member,
                source,
                destination,
                source_geometry,
                destination_geometry,
                0,
                source_geometry.physical[1],
            )
        )
    return min(
        candidates,
        key=lambda item: (
            -math.prod(item.source_geometry.logical[:2]),
            item.source_stage,
        ),
        default=None,
    )
