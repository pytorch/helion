"""Graph ownership for an FP32 carry and a separate narrowed TMEM snapshot.

The complete final-group C arena must remain disjoint from ordinary working C
and every packed operand. This candidate allocates nothing and grants no eager
evaluation or synchronization permission: callers still prove fragment/domain
mapping, independent-input readiness, and completion before overwriting C.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch.fx import Node

from ...language import _tracing_ops
from .chained_contraction_groups import _physical_left
from .chained_loop_tmem_bridges import _coordinate_path
from .chained_loop_tmem_bridges import plan_loop_tmem_bridges
from .chained_preparation_cut import _shape_input
from .chained_tmem_accumulator import _fp32_dot
from .chained_tmem_accumulator import _full_m128
from .chained_tmem_accumulator import validate_tmem_accumulator_residency
from .contraction_region import _domain

if TYPE_CHECKING:
    from .chained_contraction_groups import ContractionGroup
    from .chained_matmul import ChainedMatmulPlan
    from .chained_tcgen_stage import StageGeometry
    from .chained_tmem_accumulator import TmemAccumulatorResidency
    from .contraction_region import ContractionCarry
    from .contraction_region import ContractionRegion


@dataclass(frozen=True)
class TmemCarrySnapshotUse:
    group: ContractionGroup
    operands: tuple[Node, ...]


@dataclass(frozen=True)
class LoopTmemCarryCandidate:
    region: ContractionRegion
    carry: ContractionCarry
    final_group: ContractionGroup
    member: int
    member_offset: int
    geometry: StageGeometry
    arena_columns: int
    snapshot_shape: tuple[int, int]
    snapshot_dtype: torch.dtype
    snapshot_casts: tuple[torch.dtype, ...]
    snapshot_users: tuple[TmemCarrySnapshotUse, ...]
    snapshot_nodes: tuple[Node, ...]
    accumulator: Node
    accumulator_nodes: tuple[Node, ...]
    residency: TmemAccumulatorResidency | None

    @property
    def carry_index(self) -> int:
        return self.carry.input_index

    @property
    def input(self) -> Node:
        return self.carry.input

    @property
    def output(self) -> Node:
        return self.carry.output


def _value_ancestors(node: Node) -> set[Node]:
    """Exclude only already-validated shape queries, never arbitrary indices."""
    pending, result = [node], set()
    while pending:
        current = pending.pop()
        if current in result:
            continue
        result.add(current)
        if current.target is not torch.ops.aten.sym_size.int:
            pending.extend(current.all_input_nodes)
    return result


def plan_loop_tmem_carry(
    plan: ChainedMatmulPlan,
    groups: tuple[ContractionGroup, ...],
    residency: TmemAccumulatorResidency | None = None,
) -> LoopTmemCarryCandidate | None:
    """Select the first supported carry in original loop-port order.

    Reuses the bridge planner's retained graph/group validation, but not its
    exclusivity policy: this snapshot may serve multiple earlier groups. Every
    physical-A image must canonicalize to the same carry, orientation and full
    cast chain. The original accumulator expression is evaluated later from
    authoritative FP32, not from that narrowed snapshot.

    ``residency`` is the caller's active explicit-accumulator proof. If present,
    its producer must be routed into this same complete final arena, with a
    write window disjoint from the old carry. All other issues use working C.
    """
    if not groups or plan_loop_tmem_bridges(plan, groups) is None:
        return None
    region = plan.region
    assert region is not None and plan.loop is not None
    if any(
        node.target is torch.ops.aten.sym_size.int and not _shape_input(node)
        for node in region.nodes
    ):
        return None
    if residency is not None and not validate_tmem_accumulator_residency(
        plan, residency
    ):
        return None
    final = groups[-1]
    if final.stages[-1] != len(plan.dots) - 1 or final.physical[1] > 256:
        return None
    if any(
        item.physical[::2] != group.physical[::2]
        or _physical_left(plan.dots[index], item)
        != _physical_left(plan.dots[group.stages[0]], group.geometries[0])
        for group in groups
        for index, item in zip(group.stages, group.geometries, strict=True)
    ):
        return None
    if residency is not None and residency.destination_group != final.stages[0]:
        return None
    ancestors = {node: _value_ancestors(node) for node in region.nodes}
    for carry in region.carries:
        if carry.output not in (plan.dots[index] for index in final.stages):
            continue
        member_position = tuple(plan.dots[index] for index in final.stages).index(
            carry.output
        )
        member = final.stages[member_position]
        geometry = final.geometries[member_position]
        spec = region.contractions[member]
        value = carry.input.meta.get("val")
        if (
            not isinstance(value, torch.Tensor)
            or value.dtype != torch.float32
            or value.ndim != 2
            or _domain(value) != spec.result_domain
            or not _full_m128(geometry)
            or not _fp32_dot(spec)
            or spec.accumulator is None
            or spec.accumulator_dtype != torch.float32
            or carry.input not in ancestors[spec.accumulator]
            or any(
                other is not carry
                and (
                    other.input is carry.input
                    or carry.input in ancestors[other.output]
                    or other.input in ancestors[carry.output]
                    or other.output is carry.output
                )
                for other in region.carries
            )
            or any(
                user.op != "output" and not _shape_input(user)
                for user in carry.output.users
            )
        ):
            continue
        offset, columns = final.offsets[member_position], geometry.physical[1]
        if residency is not None and not (
            residency.offset + residency.columns <= offset
            or offset + columns <= residency.offset
        ):
            continue
        expected = geometry.result_coordinates("row", "column")
        accumulator_nodes = _coordinate_path(
            spec.accumulator, expected, carry.input, expected, ancestors
        )
        if accumulator_nodes is None or any(
            node.target is _tracing_ops._mask_to
            or node.target
            in (
                torch.ops.aten.where.self,
                torch.ops.aten.where.ScalarOther,
                torch.ops.aten.where.ScalarSelf,
            )
            for node in accumulator_nodes
        ):
            continue
        snapshot_shape = geometry.physical[:2]
        snapshot_users: list[TmemCarrySnapshotUse] = []
        snapshot_nodes: set[Node] = set()
        terminal_edges = {(spec.accumulator, carry.output)}
        terminal_positions = {(carry.output, 2, spec.accumulator)}
        canonical = None
        supported = True
        for group in groups[:-1]:
            keys = tuple(
                _physical_left(plan.dots[index], item)
                for index, item in zip(group.stages, group.geometries, strict=True)
            )
            if not any(key[0] is carry.input for key in keys):
                continue
            operands = tuple(
                (
                    region.contractions[index].rhs
                    if item.transpose
                    else region.contractions[index].lhs
                )
                for index, item in zip(group.stages, group.geometries, strict=True)
            )
            dtype = operands[0].meta["val"].dtype
            if (
                dtype not in (torch.bfloat16, torch.float16)
                or any(key != keys[0] for key in keys)
                or keys[0][0] is not carry.input
                or keys[0][1] != geometry.transpose
                or canonical is not None
                and canonical != (keys[0], dtype)
                or any(
                    item.logical[int(item.transpose)] != 128
                    or item.physical[::2] != snapshot_shape
                    or region.contractions[index].operand_dtypes != (dtype, dtype)
                    for index, item in zip(group.stages, group.geometries, strict=True)
                )
            ):
                supported = False
                break
            for index, item, operand in zip(
                group.stages, group.geometries, operands, strict=True
            ):
                path = _coordinate_path(
                    operand,
                    item.operand("a", "row", "column")[1],
                    carry.input,
                    expected,
                    ancestors,
                )
                if path is None:
                    supported = False
                    break
                snapshot_nodes.update(path)
                terminal_edges.add((operand, plan.dots[index]))
                terminal_positions.add((plan.dots[index], int(item.transpose), operand))
            if not supported:
                break
            canonical = keys[0], dtype
            snapshot_users.append(TmemCarrySnapshotUse(group, operands))
        if not supported or not snapshot_users or canonical is None:
            continue
        paths = snapshot_nodes | accumulator_nodes
        if any(
            user not in paths
            and (node, user) not in terminal_edges
            and not _shape_input(user)
            for node in paths
            for user in node.users
        ) or any(
            entry.node in paths
            or carry.output is entry.node
            or any(node in paths or node is carry.output for node in entry.dependencies)
            for entry in (
                () if plan.pointwise_cache is None else plan.pointwise_cache.entries
            )
        ):
            continue
        # A direct user edge is not an operand-position proof: the same node
        # could occur again as physical B, an explicit seed, or dependency kwarg.
        for dot in plan.dots:
            for position, argument in enumerate(dot.args):
                if not isinstance(argument, Node) or argument not in paths:
                    continue
                if (dot, position, argument) not in terminal_positions:
                    supported = False
            if any(
                node in paths for node in dot.all_input_nodes if node not in dot.args
            ):
                supported = False
        if not supported:
            continue
        return LoopTmemCarryCandidate(
            region,
            carry,
            final,
            member,
            offset,
            geometry,
            final.physical[1],
            snapshot_shape,
            canonical[1],
            canonical[0][2],
            tuple(snapshot_users),
            tuple(node for node in region.nodes if node in snapshot_nodes),
            spec.accumulator,
            tuple(node for node in region.nodes if node in accumulator_nodes),
            residency,
        )
    return None
