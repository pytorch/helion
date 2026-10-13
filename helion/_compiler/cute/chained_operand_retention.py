"""Exact typed operand images retained across preparation actions.

These are graph/storage candidates, not permission to change emission. The late
caller must prove original expression domains and native producer/read mappings,
and provide every selected early-write reservation before adopting the layout.
No candidate bypasses a cast, changes action order, or authorizes slot release.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from itertools import pairwise
import math
from typing import TYPE_CHECKING
from typing import Literal
from typing import cast

import torch
from torch.fx import Node

from ...language import _tracing_ops
from .chained_mma_selection import warp_mma_shape
from .chained_pipeline_storage import _freeze
from .chained_prepared_groups import _valid_frame
from .chained_prepared_operands import _matches_contraction
from .chained_workspace import _reuse_requests
from .contraction_region import _domain
from .warp_specialized_plan import SharedBufferRequest
from .warp_specialized_plan import SharedMemoryLayoutPlan

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .chained_contraction_groups import ContractionGroup
    from .chained_matmul import ChainedMatmulPlan
    from .chained_preparation_frame import PreparationFrame
    from .chained_prepared_groups import PreparedGroupBinding
    from .chained_tcgen_stage import StageGeometry


class _InvalidRetention(Exception):
    pass


def _require(condition: bool) -> None:
    if not condition:
        raise _InvalidRetention


@dataclass(frozen=True)
class _Revision:
    plan: ChainedMatmulPlan
    frame: PreparationFrame
    facts: tuple[object, ...]
    shapes: tuple[tuple[Node, tuple[int, ...]], ...]


def _revision(
    plan: ChainedMatmulPlan,
    frame: PreparationFrame,
    shapes: Mapping[Node, tuple[int, ...]],
) -> _Revision:
    region = plan.region
    _require(region is not None)
    assert region is not None
    _require(
        frame.cut.region.graph is region.graph
        and frame.cut.region.nodes == region.nodes
        and frame.cut.region.contractions == region.contractions
        and frame.cut.region.carries == region.carries
    )
    _require(region.nodes == tuple(region.graph.nodes) and _valid_frame(frame))
    _require(tuple(spec.node for spec in region.contractions) == plan.dots)
    _require(all(_matches_contraction(spec) for spec in region.contractions))
    facts = []
    resolved = []
    for node in region.nodes:
        value = node.meta.get("val")
        if isinstance(value, torch.Tensor):
            shape = shapes.get(node)
            _require(
                type(shape) is tuple
                and len(shape) == value.ndim
                and all(type(size) is int and size >= 0 for size in shape)
            )
            assert shape is not None
            _require(
                all(
                    type(old) is not int or old == new
                    for old, new in zip(value.shape, shape, strict=True)
                )
            )
            resolved.append((node, shape))
            metadata = value.dtype, _domain(value), _freeze(value.stride())
        else:
            metadata = _freeze(value)
        facts.append(
            (
                node,
                node.op,
                node.target,
                _freeze(node.args),
                _freeze(node.kwargs),
                metadata,
                node.meta.get("lowering"),
            )
        )
    return _Revision(
        plan,
        frame,
        (tuple(facts), plan.shapes, plan.contraction_groups, plan.warp_mma_stages),
        tuple(resolved),
    )


@dataclass(frozen=True)
class OperandRetentionCandidate:
    node: Node
    operand: Node
    view_path: tuple[Node, ...]
    group: ContractionGroup
    geometry: StageGeometry
    operand_index: int
    role: Literal["a", "b"]
    owner: str
    full_shape: tuple[int, int]
    logical_shape: tuple[int, int]
    logical_modes: tuple[int, int]
    row_offset: int
    dtype: torch.dtype
    publication_event: int
    consumer_events: tuple[int, ...]
    revision: _Revision

    @property
    def major_mode(self) -> Literal["k"]:
        return "k"

    @property
    def source_stage(self) -> int:
        if self.role == "a":
            return self.group.stages[0]
        return self.group.stages[self.group.offsets.index(self.row_offset)]

    def alias_lines(self, full_tensor: str, name: str) -> tuple[str, ...]:
        """Retain the FULL native layout; never manufacture a dense member.

        The original Node shape still supplies _Expression's bounds mask.
        Pointer/layout legality and source completion remain late obligations.
        """
        value = (
            full_tensor
            if self.row_offset == 0
            else f"cute.domain_offset(({self.row_offset}, 0), {full_tensor})"
        )
        if self.logical_modes == (0, 1):
            return (f"{name} = {value}",)
        return (
            f"{name}_physical = {value}",
            f"{name} = cute.make_tensor({name}_physical.iterator, cute.select({name}_physical.layout, mode=[1, 0]))",
        )


def _canonical_operand(
    operand: Node, coordinates: tuple[str, str]
) -> tuple[Node, tuple[Node, ...], tuple[int, int]]:
    node, path = operand, []
    while True:
        exchange = False
        if node.target is _tracing_ops._new_var:
            _require(len(node.args) == 1)
        elif node.target is torch.ops.aten.t.default or (
            node.target is torch.ops.aten.permute.default
            and len(node.args) == 2
            and node.args[1] in ((1, 0), [1, 0])
        ):
            exchange = True
        elif node.target is torch.ops.aten.transpose.int and len(node.args) == 3:
            left, right = node.args[1:]
            if not (
                type(left) is int
                and type(right) is int
                and -2 <= left < 2
                and -2 <= right < 2
                and left % 2 != right % 2
            ):
                break
            exchange = True
        else:
            break
        _require(not node.kwargs)
        source = node.args[0]
        _require(isinstance(source, Node))
        assert isinstance(source, Node)
        value = source.meta.get("val")
        _require(isinstance(value, torch.Tensor) and value.ndim == 2)
        path.append(node)
        node = source
        if exchange:
            coordinates = coordinates[1], coordinates[0]
    _require(coordinates in (("row", "k"), ("k", "row")))
    return node, tuple(path), (0, 1) if coordinates == ("row", "k") else (1, 0)


def _reads(
    frame: PreparationFrame,
    candidates: tuple[OperandRetentionCandidate, ...] = (),
) -> tuple[dict[int, tuple[str, ...]], dict[int, frozenset[Node]]]:
    """Follow original graph operands/kwargs until the current published cut."""
    buffers = {buffer.name: buffer for buffer in frame.buffers}
    published: dict[Node, str] = {}
    reads, visits = {}, {}
    for action in frame.actions:
        if action.kind in ("mma", "ready"):
            used = set(action.reads)
            visited: set[Node] = set()
        else:
            roots = (
                tuple(n for dot in action.nodes for n in dot.all_input_nodes)
                if action.kind == "fill"
                else action.nodes
            )
            retained = {
                item.node: item.owner
                for item in candidates
                if item.publication_event <= action.event
            }
            pending, visited, used = list(roots), set(), set()
            while pending:
                node = pending.pop()
                if node in visited:
                    continue
                visited.add(node)
                if node in retained:
                    used.add(retained[node])
                elif node in published:
                    used.add(published[node])
                else:
                    pending.extend(node.all_input_nodes)
        reads[action.event] = tuple(sorted(used))
        visits[action.event] = frozenset(visited)
        if action.kind in ("mma", "collective", "cache", "leaf"):
            for name in action.writes:
                node = buffers[name].node
                if node is not None:
                    published[node] = name
    return reads, visits


def discover_operand_retention(
    plan: ChainedMatmulPlan,
    frame: PreparationFrame,
    shapes: Mapping[Node, tuple[int, ...]],
) -> tuple[OperandRetentionCandidate, ...] | None:
    """Discover exact SSA images with later preparation reads, without CuTe.

    None denotes inconsistent retained facts; () denotes no eligible image.
    Domains/producer ownership are deliberately not inferred from graph shape.
    """
    try:
        revision = _revision(plan, frame, shapes)
        reads, visits = _reads(frame)
        _require(all(set(reads[a.event]) <= set(a.reads) for a in frame.actions))
        buffers = {buffer.name: buffer for buffer in frame.buffers}
        result = []
        seen = set()
        for stage in frame.stages:
            group = stage.group
            fills = tuple(
                a
                for a in frame.actions
                if a.kind == "fill" and a.stages == group.stages
            )
            _require(len(fills) == 1)
            fill = fills[0]
            _require(fill.event + 1 < len(frame.actions))
            mma = frame.actions[fill.event + 1]
            _require(mma.kind == "mma" and mma.stages == group.stages)
            _require(group.stages[0] in plan.warp_mma_stages)
            _require(len(group.stages) == len(group.geometries))
            _require(
                all(
                    g.logical == plan.shapes[i]
                    for i, g in zip(group.stages, group.geometries, strict=True)
                )
            )
            m, n, k = stage.shape
            _require(
                stage.shape == warp_mma_shape(group.geometries[0], group)
                and all(type(size) is int and size > 0 for size in stage.shape)
                and stage.a.byte_size >= 2 * m * k
                and stage.b.byte_size >= 2 * n * k
            )
            _require(
                buffers[stage.a.name].shape == (m, k)
                and buffers[stage.b.name].shape == (n, k)
            )
            physical_a = []
            members = []
            offset = 0
            for index, geometry in zip(group.stages, group.geometries, strict=True):
                spec = frame.cut.region.contractions[index]
                for role in ("a", "b"):
                    operand_index, coords = geometry.operand(role, "row", "k")
                    operand = (spec.lhs, spec.rhs)[operand_index]
                    node, path, modes = _canonical_operand(operand, coords)
                    item = (node, path, modes, operand, operand_index, geometry)
                    if role == "a":
                        physical_a.append(item)
                    else:
                        members.append((item, offset, geometry.physical[1]))
                offset += geometry.physical[1]
            _require(offset == n)
            operands = [
                ("b", item, offset, rows, stage.b.name)
                for item, offset, rows in members
            ]
            if all(
                (item[0], item[2]) == (physical_a[0][0], physical_a[0][2])
                for item in physical_a
            ):
                operands.insert(0, ("a", physical_a[0], 0, m, stage.a.name))
            for role, item, offset, rows, owner in operands:
                node, path, modes, operand, operand_index, geometry = item
                value = node.meta.get("val")
                logical = (rows, k) if modes == (0, 1) else (k, rows)
                consumers = tuple(
                    a.event
                    for a in frame.actions
                    if a.event >= mma.publication_event and node in visits[a.event]
                )
                if (
                    node in seen
                    or not consumers
                    or not isinstance(value, torch.Tensor)
                    or value.dtype not in (torch.bfloat16, torch.float16)
                    or shapes[node] != logical
                    or buffers[owner].dtype != value.dtype
                ):
                    continue
                seen.add(node)
                result.append(
                    OperandRetentionCandidate(
                        node,
                        operand,
                        path,
                        group,
                        geometry,
                        operand_index,
                        cast("Literal['a', 'b']", role),
                        owner,
                        (m, k) if role == "a" else (n, k),
                        logical,
                        modes,
                        offset,
                        value.dtype,
                        mma.publication_event,
                        consumers,
                        revision,
                    )
                )
        return tuple(result)
    except _InvalidRetention:
        return None


@dataclass(frozen=True)
class OperandRetentionPlan:
    frame: PreparationFrame
    candidates: tuple[OperandRetentionCandidate, ...]
    prepared_groups: tuple[PreparedGroupBinding, ...]
    leaf_reads: tuple[tuple[Node, tuple[int, ...]], ...]
    owner_offsets: tuple[tuple[str, int], ...]
    revision: _Revision
    original_groups: tuple[PreparedGroupBinding, ...]
    reservations: tuple[SharedBufferRequest, ...]
    _output_witness: tuple[object, ...]

    def matches(
        self,
        plan: ChainedMatmulPlan,
        frame: PreparationFrame,
        shapes: Mapping[Node, tuple[int, ...]],
        *,
        prepared_groups: tuple[PreparedGroupBinding, ...] = (),
        reservations: tuple[SharedBufferRequest, ...] = (),
    ) -> bool:
        try:
            return (
                self._output_witness
                == (
                    self.frame,
                    self.candidates,
                    self.prepared_groups,
                    self.leaf_reads,
                    self.owner_offsets,
                    self.revision,
                    self.original_groups,
                    self.reservations,
                )
                and self.revision.plan is plan
                and self.revision.frame is frame
                and self.revision == _revision(plan, frame, shapes)
                and prepared_groups == self.original_groups
                and reservations == self.reservations
            )
        except _InvalidRetention:
            return False


def plan_operand_retention_frame(
    plan: ChainedMatmulPlan,
    frame: PreparationFrame,
    candidates: tuple[OperandRetentionCandidate, ...],
    shapes: Mapping[Node, tuple[int, ...]],
    *,
    prepared_groups: tuple[PreparedGroupBinding, ...] = (),
    reservations: tuple[SharedBufferRequest, ...] = (),
    capacity_bytes: int,
) -> OperandRetentionPlan | None:
    """Rebuild complete reads/leases and pack all original owners atomically.

    Reservations are additional typed early-write/lifetime requirements from
    already selected emission plans. They may extend, never replace or shrink,
    original starts. The caller must supply ALL such requirements and reprove
    late emission after rebinding; this pure result is not execution authority.
    """
    try:
        available = discover_operand_retention(plan, frame, shapes)
        _require(available is not None and bool(candidates))
        assert available is not None
        _require(type(capacity_bytes) is int and capacity_bytes > 0)
        _require(
            all(
                item.revision.plan is plan and item.revision.frame is frame
                for item in candidates
            )
        )
        _require(candidates == tuple(item for item in available if item in candidates))
        _require(len({item.node for item in candidates}) == len(candidates))
        reads, _ = _reads(frame, candidates)
        regions = {region.name: region for region in frame.layout.regions}
        ends = {name: region.live_from + 1 for name, region in regions.items()}
        for action in frame.actions:
            for name in (*reads[action.event], *action.writes):
                _require(name in regions)
                ends[name] = max(ends[name], action.publication_event)
        requests = {
            name: SharedBufferRequest(
                name, region.byte_size, region.alignment, region.live_from, ends[name]
            )
            for name, region in regions.items()
        }
        for reservation in reservations:
            _require(
                type(reservation) is SharedBufferRequest
                and reservation.name in requests
            )
            original = requests[reservation.name]
            _require(
                reservation.byte_size == original.byte_size
                and reservation.alignment == original.alignment
                and type(reservation.live_from) is int
                and type(reservation.live_until) is int
                and 0
                <= reservation.live_from
                < reservation.live_until
                <= len(frame.actions)
            )
            requests[reservation.name] = replace(
                original,
                live_from=min(original.live_from, reservation.live_from),
                live_until=max(original.live_until, reservation.live_until),
            )
        owners = {}
        unions = []
        rebound_groups = []
        for binding in prepared_groups:
            group = binding.candidate
            _require(group.name not in requests and bool(group.members))
            _require(
                group.group in (plan.contraction_groups or ())
                and len(group.members) == len(group.group.stages)
            )
            member_requests = []
            spans = []
            for member in group.members:
                name = member.buffer.name
                _require(name in requests and name not in owners)
                _require(
                    member.buffer in frame.buffers
                    and member.stage in group.group.stages
                    and member.geometry
                    == group.group.geometries[group.group.stages.index(member.stage)]
                    and member.buffer.dtype == group.dtype
                )
                _require(
                    regions[name].byte_offset
                    == binding.byte_offset + member.byte_offset
                )
                _require(
                    regions[name].byte_size
                    == math.prod(member.physical_shape) * group.dtype.itemsize
                )
                owners[name] = (group.name, member.byte_offset)
                member_requests.append(requests[name])
                spans.append(
                    (member.byte_offset, member.byte_offset + requests[name].byte_size)
                )
            spans.sort()
            _require(
                spans[0][0] == 0
                and spans[-1][1] == group.byte_size
                and all(left[1] == right[0] for left, right in pairwise(spans))
            )
            union = SharedBufferRequest(
                group.name,
                group.byte_size,
                128,
                min(group.live_from, *(item.live_from for item in member_requests)),
                max(group.live_until, *(item.live_until for item in member_requests)),
            )
            unions.append(union)
            rebound_groups.append(
                replace(
                    binding,
                    candidate=replace(
                        group, live_from=union.live_from, live_until=union.live_until
                    ),
                )
            )
        packed = _reuse_requests(
            tuple(request for name, request in requests.items() if name not in owners)
            + tuple(unions)
        )
        _require(packed.allocated_bytes <= capacity_bytes)
        output_regions = []
        for original in frame.layout.regions:
            request = requests[original.name]
            if original.name in owners:
                group, offset = owners[original.name]
                output_regions.append(
                    replace(
                        original,
                        byte_offset=packed.region(group).byte_offset + offset,
                        live_from=request.live_from,
                        live_until=request.live_until,
                    )
                )
            else:
                output_regions.append(packed.region(original.name))
        layout = SharedMemoryLayoutPlan(tuple(output_regions), packed.allocated_bytes)
        output = replace(
            frame,
            layout=layout,
            actions=tuple(
                replace(action, reads=reads[action.event]) for action in frame.actions
            ),
            stages=tuple(
                replace(
                    stage, a=layout.region(stage.a.name), b=layout.region(stage.b.name)
                )
                for stage in frame.stages
            ),
            peak_live_bytes=max(
                sum(
                    region.byte_size
                    for region in packed.regions
                    if region.live_from <= action.event < region.live_until
                )
                for action in frame.actions
            ),
        )
        _require(_valid_frame(output))
        groups = tuple(
            replace(
                binding, byte_offset=packed.region(binding.candidate.name).byte_offset
            )
            for binding in rebound_groups
        )
        leaves = tuple(
            (
                buffer.node,
                tuple(
                    action.event
                    for action in output.actions
                    if buffer.name in action.reads
                ),
            )
            for buffer in frame.buffers
            if buffer.kind == "leaf" and buffer.node is not None
        )
        offsets = tuple((region.name, region.byte_offset) for region in layout.regions)
        witness = (
            output,
            candidates,
            groups,
            leaves,
            offsets,
            candidates[0].revision,
            prepared_groups,
            reservations,
        )
        return OperandRetentionPlan(
            output,
            candidates,
            groups,
            leaves,
            offsets,
            candidates[0].revision,
            prepared_groups,
            reservations,
            witness,
        )
    except _InvalidRetention:
        return None
