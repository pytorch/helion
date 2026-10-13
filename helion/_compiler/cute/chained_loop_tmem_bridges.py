"""Candidates for exact loop-result pointwise regions packed into TMEM A.

These records are not authorization to omit a result's materialization. Before
allocating or emitting a bridge, codegen must validate every original operand
with ``_Expression.fragments`` at the source's logical coordinates and retain
``_operand_domain`` masking. Packing also requires all original side inputs to
be available. A caller must allocate disjoint raw-C/packed-A TMEM lifetimes and
retain completion, publication, and consumer barriers. No in-place conversion,
carry residency, reassociation, or new schedule is implied here.
The consumer issue event is not a release: packed A remains live through that
issue's completion, or longer under a conservative caller-owned allocation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from ...language import _tracing_ops
from .chained_contraction_groups import _physical_left
from .chained_preparation_cut import _known_effects
from .chained_prepared_operands import _matches_contraction
from .chained_tcgen_stage import stage_geometry
from .chained_tmem_accumulator import _full_m128
from .chained_tmem_accumulator import _groups
from .contraction_region import _domain

if TYPE_CHECKING:
    from torch.fx import Node

    from .chained_contraction_groups import ContractionGroup
    from .chained_matmul import ChainedMatmulPlan
    from .chained_tcgen_stage import StageGeometry
    from .contraction_region import ContractionRegion


@dataclass(frozen=True)
class LoopTmemBridgeCandidate:
    region: ContractionRegion
    source_stage: int
    destination_group: ContractionGroup
    source: Node
    operands: tuple[Node, ...]
    source_geometry: StageGeometry
    physical_shape: tuple[int, int]
    dtype: torch.dtype
    expression_nodes: tuple[Node, ...]
    publication_event: int
    issue_event: int


def _ancestors(node: Node) -> set[Node]:
    result: set[Node] = set()
    pending = [node]
    while pending:
        current = pending.pop()
        if current not in result:
            result.add(current)
            pending.extend(current.all_input_nodes)
    return result


def _exchange_axes(node: Node) -> bool:
    if node.target is torch.ops.aten.t.default:
        return len(node.args) == 1 and not node.kwargs
    if node.target is torch.ops.aten.permute.default:
        return (
            len(node.args) == 2 and node.args[1] in ((1, 0), [1, 0]) and not node.kwargs
        )
    if node.target is torch.ops.aten.transpose.int:
        if len(node.args) != 3:
            return False
        first, second = node.args[1:]
        return (
            type(first) is int
            and type(second) is int
            and -2 <= first < 2
            and -2 <= second < 2
            and first % 2 != second % 2
            and not node.kwargs
        )
    return False


def _coordinate_path(
    operand: Node,
    coordinates: tuple[str, str],
    source: Node,
    expected: tuple[str, str],
    ancestors: dict[Node, set[Node]],
) -> set[Node] | None:
    """Prove only exact rank-two pointwise coordinates on C-dependent edges.

    This structural proof deliberately does not replace codegen's fragment and
    domain proof. Independent inputs retain their original broadcasting/index
    expressions; no source-dependent broadcast, reshape, or gather is admitted.
    """
    pending = [(operand, coordinates)]
    visited: set[tuple[Node, tuple[str, str]]] = set()
    result: set[Node] = set()
    while pending:
        node, coords = pending.pop()
        if (node, coords) in visited:
            continue
        visited.add((node, coords))
        result.add(node)
        if node is source:
            if coords != expected:
                return None
            continue
        value = node.meta.get("val")
        if (
            node.op != "call_function"
            or not isinstance(value, torch.Tensor)
            or value.ndim != 2
            or not _known_effects(node)
        ):
            return None
        dependent = tuple(
            child for child in node.all_input_nodes if source in ancestors[child]
        )
        if not dependent:
            return None
        if _exchange_axes(node):
            if len(dependent) != 1 or node.args[0] is not dependent[0]:
                return None
            child = dependent[0]
            child_value = child.meta.get("val")
            if (
                not isinstance(child_value, torch.Tensor)
                or child_value.ndim != 2
                or _domain(child_value)[::-1] != _domain(value)
                or child_value.dtype != value.dtype
            ):
                return None
            pending.append((child, (coords[1], coords[0])))
            continue
        identity = node.target is _tracing_ops._new_var
        pointwise = (
            isinstance(node.target, torch._ops.OpOverload)
            and torch.Tag.pointwise in node.target.tags
        ) or node.target is _tracing_ops._mask_to
        if not identity and not pointwise:
            return None
        if identity and (len(node.args) != 1 or node.kwargs):
            return None
        for child in dependent:
            child_value = child.meta.get("val")
            if not isinstance(child_value, torch.Tensor) or _domain(
                child_value
            ) != _domain(value):
                return None
            pending.append((child, coords))
    return result


def plan_loop_tmem_bridges(
    plan: ChainedMatmulPlan,
    groups: tuple[ContractionGroup, ...],
) -> tuple[LoopTmemBridgeCandidate, ...] | None:
    """Find exclusive complete physical-A candidates in one ordered TCgen role.

    ``groups`` must contain every non-warp issue in its original order. Invalid
    retained plans return ``None``; unsupported edges simply yield no candidate.
    Events use the existing recurrence convention: read/issue ``2*stage``, C
    publication ``2*stage+1``. Nonadjacent edges are allowed, without granting
    permission to overlap or reuse their live TMEM. Every record still requires
    late fragment-coordinate and domain validation for *each* group member.
    """
    region, loop = plan.region, plan.loop
    if (
        region is None
        or loop is None
        or loop.region.graph is not region.graph
        or loop.region.nodes != region.nodes
        or loop.region.contractions != region.contractions
        or loop.region.carries != region.carries
        or plan.strategy != "tcgen05_tmem"
        or plan.direct_output
        or plan.initialized_accumulator is not None
        or plan.late_rhs_reuse is not None
        or plan.k_schedule is not None
        or tuple(region.graph.nodes) != region.nodes
        or plan.dots != tuple(spec.node for spec in region.contractions)
        or len(plan.dots) != len(plan.shapes)
        or not all(_matches_contraction(spec) for spec in region.contractions)
    ):
        return None
    positions = {node: index for index, node in enumerate(region.nodes)}
    if any(
        node.graph is not region.graph
        or any(
            positions.get(child, len(positions)) >= index
            for child in node.all_input_nodes
        )
        for index, node in enumerate(region.nodes)
    ):
        return None
    all_groups = _groups(plan)
    if all_groups is None or tuple(
        index for group in all_groups for index in group.stages
    ) != tuple(range(len(plan.dots))):
        return None
    if any(
        not group.stages
        or len(group.stages) != len(group.geometries)
        or any(
            geometry.logical != plan.shapes[index]
            or len(geometry.logical) != 3
            or any(type(size) is not int or size <= 0 for size in geometry.logical)
            or stage_geometry(geometry.logical) is None
            or any(
                type(old) is int and old != new
                for old, new in zip(
                    region.contractions[index].shape, geometry.logical, strict=True
                )
            )
            for index, geometry in zip(group.stages, group.geometries, strict=True)
        )
        for group in all_groups
    ):
        return None
    # Instruction selection names issue leaders, not every concatenated member.
    if not plan.warp_mma_stages <= {group.stages[0] for group in all_groups}:
        return None
    if groups != tuple(
        group for group in all_groups if group.stages[0] not in plan.warp_mma_stages
    ):
        return None
    ancestors = {node: _ancestors(node) for node in region.nodes}
    forbidden = {
        *region.live_outs,
        *region.stores,
        *region.scans,
        *region.reductions,
        *(carry.output for carry in region.carries),
    }
    if plan.pointwise_cache is not None:
        forbidden.update(entry.node for entry in plan.pointwise_cache.entries)
        forbidden.update(
            node
            for entry in plan.pointwise_cache.entries
            for node in entry.dependencies
        )
    result = []
    for source_index, source_group in enumerate(groups):
        if len(source_group.stages) != 1:
            continue
        source_stage = source_group.stages[0]
        source_geometry = source_group.geometries[0]
        source = plan.dots[source_stage]
        source_spec = region.contractions[source_stage]
        if (
            not _full_m128(source_geometry)
            or source_spec.result_dtype != torch.float32
            or source_spec.operand_dtypes[0] not in (torch.bfloat16, torch.float16)
            or source_spec.operand_dtypes[0] != source_spec.operand_dtypes[1]
            or source in forbidden
        ):
            continue
        physical_shape = source_geometry.physical[:2]
        expected = source_geometry.result_coordinates("row", "k")
        for destination in groups[source_index + 1 :]:
            m, _, k = destination.physical
            if physical_shape != (m, k):
                continue
            operands = tuple(
                (region.contractions[index].lhs, region.contractions[index].rhs)[
                    int(geometry.transpose)
                ]
                for index, geometry in zip(
                    destination.stages, destination.geometries, strict=True
                )
            )
            dtype = operands[0].meta["val"].dtype
            if (
                dtype not in (torch.bfloat16, torch.float16)
                or any(
                    geometry.logical[int(geometry.transpose)] != 128
                    or geometry.physical[::2] != physical_shape
                    or region.contractions[index].operand_dtypes != (dtype, dtype)
                    or _physical_left(plan.dots[index], geometry)
                    != _physical_left(
                        plan.dots[destination.stages[0]], destination.geometries[0]
                    )
                    for index, geometry in zip(
                        destination.stages, destination.geometries, strict=True
                    )
                )
                or any(source not in ancestors[operand] for operand in operands)
            ):
                continue
            paths: set[Node] = set()
            valid = True
            for index, geometry, operand in zip(
                destination.stages, destination.geometries, operands, strict=True
            ):
                spec = region.contractions[index]
                other = spec.lhs if geometry.transpose else spec.rhs
                if source in ancestors[other] or (
                    spec.accumulator is not None
                    and source in ancestors[spec.accumulator]
                ):
                    valid = False
                    break
                path = _coordinate_path(
                    operand,
                    geometry.operand("a", "row", "k")[1],
                    source,
                    expected,
                    ancestors,
                )
                if path is None:
                    valid = False
                    break
                paths.update(path)
            if not valid or paths & forbidden:
                continue
            terminals = {
                (operand, plan.dots[index])
                for index, operand in zip(destination.stages, operands, strict=True)
            }
            if any(
                user not in paths and (node, user) not in terminals
                for node in paths
                for user in node.users
            ):
                continue
            result.append(
                LoopTmemBridgeCandidate(
                    region,
                    source_stage,
                    destination,
                    source,
                    operands,
                    source_geometry,
                    physical_shape,
                    dtype,
                    tuple(node for node in region.nodes if node in paths),
                    2 * source_stage + 1,
                    2 * destination.stages[0],
                )
            )
    return tuple(result)
