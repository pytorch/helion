"""Resident recurrence storage behind an exact prepared-value frontier.

Preparation frames belong to another role and are not part of this arena. A
consumer owns its ready frame until every recurrence/store/carry read finishes.
Recurrence input reads precede publication, with a role-local barrier between
them. All next carries must then be captured in registers, followed by a
barrier, all carry writes, and another barrier. This is a sequential role-local
lifetime plan, not permission to overlap recurrence iterations.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

import torch

from .chained_contraction_groups import ContractionGroup
from .chained_preparation_cut import _shape_input
from .chained_tcgen_stage import stage_geometry
from .chained_tmem_accumulator import plan_tmem_accumulator_residency
from .chained_tmem_accumulator import validate_tmem_accumulator_residency
from .chained_workspace import _align
from .chained_workspace import _reuse_requests
from .contraction_region import _domain
from .physical_use_frontier import InvalidPhysicalUse
from .physical_use_frontier import PhysicalPublication
from .physical_use_frontier import PhysicalReadPoint
from .physical_use_frontier import PhysicalUseFrontier
from .warp_specialized_plan import SharedBufferRequest

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Mapping

    from torch.fx import Node

    from .chained_matmul import ChainedMatmulPlan
    from .chained_preparation_cut import PreparationCut
    from .chained_tmem_accumulator import TmemAccumulatorResidency
    from .warp_specialized_plan import SharedMemoryLayoutPlan


@dataclass(frozen=True)
class RecurrenceStage:
    group: ContractionGroup
    read_event: int
    publication_event: int


@dataclass(frozen=True)
class RecurrenceWorkspace:
    cut: PreparationCut
    layout: SharedMemoryLayoutPlan
    bindings: tuple[tuple[Node, str], ...]
    stages: tuple[RecurrenceStage, ...]
    transition_event: int
    a_bytes: int
    b_bytes: int
    peak_live_bytes: int

    @property
    def shared_bytes(self) -> int:
        """Recurrence A/B and carry/C only; frames/protocol are caller-owned."""
        return self.a_bytes + self.b_bytes + self.layout.allocated_bytes


class _UnsupportedRecurrence(Exception):
    pass


def _require(condition: bool) -> None:
    if not condition:
        raise _UnsupportedRecurrence


def plan_recurrence_workspace(
    plan: ChainedMatmulPlan,
    cut: PreparationCut,
    shapes: Mapping[Node, tuple[int, ...]],
    residency: TmemAccumulatorResidency | None = None,
) -> RecurrenceWorkspace | None:
    """Pack admitted recurrence carries/C without inspecting compiler context.

    Logical shapes must be explicitly resolved by the caller. Existing stage
    IDs, grouped publication, typed frontier images, and original dependencies
    are preserved. ``peak_live_bytes`` is an aligned live-byte lower bound, not
    a claim that greedy placement always attains the minimum. No hardware
    capacity or pipeline protocol overhead is assumed here.
    """
    try:
        return _plan(plan, cut, shapes, residency)
    except (_UnsupportedRecurrence, InvalidPhysicalUse):
        return None


def _plan(
    plan: ChainedMatmulPlan,
    cut: PreparationCut,
    shapes: Mapping[Node, tuple[int, ...]],
    residency: TmemAccumulatorResidency | None,
) -> RecurrenceWorkspace:
    from .chained_matmul import _UnsupportedChain
    from .prepared_graph_schedule import ContractionGraph
    from .prepared_graph_schedule import LoopTerminalStorePlacement
    from .prepared_graph_schedule import schedule_contraction_groups

    region, loop = plan.region, plan.loop
    _require(region is not None and loop is not None)
    assert region is not None and loop is not None
    _require(
        region.graph is cut.region.graph is loop.body.graph
        and region.nodes == cut.region.nodes == loop.region.nodes
        and region.nodes == tuple(region.graph.nodes)
        and region.carries == cut.region.carries == cut.carries == loop.region.carries
        and bool(cut.carries)
        and loop.storage_key == cut.storage_proof_key
        and plan.dots == tuple(spec.node for spec in cut.region.contractions)
        and plan.dots == tuple(spec.node for spec in region.contractions)
        and len(plan.dots) == len(plan.shapes)
        and plan.strategy == "tcgen05_tmem"
        and not plan.direct_output
        and plan.initialized_accumulator is None
        and plan.late_rhs_reuse is None
        and plan.k_schedule is None
        and not plan.scan_exports
    )
    positions = {node: index for index, node in enumerate(region.nodes)}
    _require(
        all(
            node.graph is region.graph
            and all(
                positions.get(source, len(positions)) < index
                for source in node.all_input_nodes
            )
            for index, node in enumerate(region.nodes)
        )
    )
    preparation, recurrence, shared = map(
        set, (cut.preparation, cut.recurrence, cut.shared_inputs)
    )
    _require(
        preparation | recurrence | shared == set(region.nodes)
        and not (
            preparation & recurrence or preparation & shared or recurrence & shared
        )
        and {carry.input for carry in cut.carries} <= recurrence
        and set(cut.captures) <= shared
        and set(region.stores) <= recurrence
        and set(plan.dots) <= preparation | recurrence
        and {*region.scans, *region.reductions} <= preparation
        and (
            plan.pointwise_cache is None
            or all(entry.node in preparation for entry in plan.pointwise_cache.entries)
        )
    )

    def tensor(node: Node) -> tuple[torch.dtype, tuple[int, ...]]:
        value, shape = node.meta.get("val"), shapes.get(node)
        _require(isinstance(value, torch.Tensor) and shape is not None)
        assert isinstance(value, torch.Tensor) and shape is not None
        _require(
            type(shape) is tuple
            and len(shape) == value.ndim
            and all(type(size) is int and size > 0 for size in shape)
            and all(
                type(old) is not int or old == new
                for old, new in zip(value.shape, shape, strict=True)
            )
        )
        return value.dtype, shape

    images = {image.node for image in cut.images}
    _require(len(images) == len(cut.images) and bool(images))
    expected_images = {
        node for node in preparation if any(user in recurrence for user in node.users)
    }
    _require(images == expected_images)
    bindings: list[tuple[Node, str]] = []
    for index, image in enumerate(cut.images):
        dtype, _ = tensor(image.node)
        _require(
            dtype == image.dtype
            and _domain(image.node.meta["val"]) == image.logical_domain
            and image.consumers
            == tuple(
                sorted(
                    (user for user in image.node.users if user in recurrence),
                    key=positions.__getitem__,
                )
            )
        )
        bindings.append((image.node, f"chain_prepared_{index}"))
    # A preparation node may not hide a data dependence on current state.
    for node in preparation | shared:
        if node.target is torch.ops.aten.sym_size.int:
            _require(_shape_input(node))
        else:
            _require(not any(source in recurrence for source in node.all_input_nodes))

    groups = plan.contraction_groups
    if groups is None:
        singletons = []
        for index, shape in enumerate(plan.shapes):
            _require(
                len(shape) == 3
                and all(type(size) is int and size > 0 for size in shape)
            )
            geometry = stage_geometry(shape)
            _require(geometry is not None)
            assert geometry is not None
            singletons.append(ContractionGroup((index,), (geometry,)))
        groups = tuple(singletons)
    try:
        schedule = schedule_contraction_groups(
            ContractionGraph(region),
            groups,
            plan.shapes,
            partitions=(frozenset(preparation), frozenset(recurrence)),
            materializations=frozenset((*region.scans, *region.reductions)),
            terminal_stores=LoopTerminalStorePlacement(
                loop, region.stores, cut.storage_proof_key, frozenset(recurrence)
            ),
        )
    except _UnsupportedChain as error:
        raise _UnsupportedRecurrence from error
    stages = []
    starts: dict[Node, int] = {}
    ends: dict[Node, int] = {}
    sizes: dict[Node, int] = {}
    names: dict[Node, str] = {}
    a_bytes = b_bytes = 0
    for scheduled in schedule.groups:
        group = scheduled.group
        _require(bool(group.stages) and len(group.stages) == len(group.geometries))
        roles = {plan.dots[stage] in recurrence for stage in group.stages}
        _require(len(roles) == 1)
        for stage, geometry in zip(group.stages, group.geometries, strict=True):
            shape = plan.shapes[stage]
            _require(
                len(shape) == 3
                and all(type(size) is int and size > 0 for size in shape)
            )
            _require(geometry.logical == shape and stage_geometry(shape) is not None)
            rows, columns = shape[1::-1] if geometry.transpose else shape[:2]
            _require(rows <= 128 and columns <= 256)
        if False in roles:
            continue
        _require(not set(group.stages) & plan.warp_mma_stages)
        m, n, k = group.physical
        _require(
            n <= 256
            and all(geometry.physical[::2] == (m, k) for geometry in group.geometries)
        )
        stages.append(
            RecurrenceStage(group, scheduled.read_event, scheduled.publication_event)
        )
        group_dtype = None
        for stage in group.stages:
            node = plan.dots[stage]
            spec = region.contractions[stage]
            _require(
                len(node.args) == 4
                and node.args[:3] == (spec.lhs, spec.rhs, spec.accumulator)
            )
            dtype, lhs_shape = tensor(spec.lhs)
            rhs_dtype, rhs_shape = tensor(spec.rhs)
            result_dtype, result_shape = tensor(node)
            rows, columns, reduction = plan.shapes[stage]
            _require(
                dtype in (torch.bfloat16, torch.float16)
                and dtype == rhs_dtype
                and result_dtype == torch.float32
                and node.args[3] in (None, torch.float32)
                and spec.requested_out_dtype == node.args[3]
                and lhs_shape == (rows, reduction)
                and rhs_shape == (reduction, columns)
                and result_shape == (rows, columns)
                and (group_dtype is None or dtype == group_dtype)
            )
            group_dtype = dtype
            if spec.accumulator is not None:
                _require(tensor(spec.accumulator) == (torch.float32, result_shape))
            starts[node] = scheduled.publication_event
            ends[node] = starts[node] + 1
            sizes[node] = 4 * math.prod(result_shape)
            names[node] = f"chain_{stage}_c"
        a_bytes = max(a_bytes, _align(2 * m * k))
        b_bytes = max(b_bytes, _align(2 * n * k))
    _require(bool(stages))
    if residency is not None:
        _require(
            validate_tmem_accumulator_residency(plan, residency)
            and plan_tmem_accumulator_residency(
                plan, tuple(stage.group for stage in stages)
            )
            == residency
        )
    carry_names = set()
    for carry in cut.carries:
        dtype, shape = tensor(carry.input)
        _require(
            dtype == torch.float32
            and len(shape) == 2
            and tensor(carry.output) == (dtype, shape)
        )
        name = loop.carry_name(carry.input_index)
        _require(carry.input not in starts and name not in carry_names)
        carry_names.add(name)
        starts[carry.input], ends[carry.input] = -1, 0
        sizes[carry.input], names[carry.input] = 4 * math.prod(shape), name

    use_frontier = PhysicalUseFrontier(region.nodes)
    publications = {}
    live_ins = {carry.input for carry in cut.carries}
    for node, born in starts.items():
        if node in live_ins:
            use_frontier.register_live_in(node, cut)
        else:
            use_frontier.register_owner(node, cut, born)
        dtype, shape = tensor(node)
        publications[node] = PhysicalPublication(node, node, dtype, shape)

    def consume(roots: Iterable[Node], event: int) -> None:
        point = PhysicalReadPoint(cut, event, event + 1)
        for publication in use_frontier.resolve(
            roots,
            publications,
            external=frozenset(images | shared),
            required=frozenset(starts),
            traversable=frozenset(recurrence),
        ):
            node = publication.node
            if residency is not None and node is residency.source:
                # Preserve original publication ordering without manufacturing
                # a shared-C read for the proven resident TMEM accumulator.
                use_frontier.require_published(publication.owner, point)
                continue
            use_frontier.read(publication.owner, point)
            ends[node] = use_frontier.through(publication.owner, cut)

    for stage in stages:
        consume(
            (
                source
                for index in stage.group.stages
                for source in plan.dots[index].all_input_nodes
            ),
            stage.read_event,
        )
    transition = schedule.transition_event
    for store, event in schedule.store_events:
        consume((store,), event)
    consume(
        (
            plan.store,
            *region.live_outs,
            *(carry.output for carry in cut.carries),
        ),
        transition,
    )
    ordered = (
        *(carry.input for carry in cut.carries),
        *(
            node
            for node in plan.dots
            if node in starts and (residency is None or node is not residency.source)
        ),
    )
    use_frontier.check()
    requests = tuple(
        SharedBufferRequest(names[node], sizes[node], 128, starts[node], ends[node])
        for node in ordered
    )
    layout = _reuse_requests(requests)
    peak = max(
        sum(
            _align(request.byte_size)
            for request in requests
            if request.live_from <= event < request.live_until
        )
        for event in {
            -1,
            transition,
            *(stage.read_event for stage in stages),
            *(stage.publication_event for stage in stages),
        }
    )
    bindings.extend((node, names[node]) for node in ordered)
    return RecurrenceWorkspace(
        cut, layout, tuple(bindings), tuple(stages), transition, a_bytes, b_bytes, peak
    )
