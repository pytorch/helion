"""Lifetime-packed preparation storage for one exclusively owned loop slot.

This is a byte-interval plan, not an asynchronous pipeline or a physical CuTe
layout proof. Every action ends with a role-local publication barrier. A ready
frame remains owned by its consumer until whole-slot release; another iteration
must not use its dead scratch while its frontier is still being consumed.
"""

from __future__ import annotations

from bisect import bisect_right
from collections import Counter
from dataclasses import dataclass
from dataclasses import replace
import math
from typing import TYPE_CHECKING
from typing import Literal

import torch

from .chained_contraction_groups import ContractionGroup
from .chained_mma_selection import warp_mma_shape
from .chained_tcgen_stage import stage_geometry
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
    from .chained_preparation_cut import PreparationImage
    from .warp_specialized_plan import SharedBufferRegion
    from .warp_specialized_plan import SharedMemoryLayoutPlan


BufferKind = Literal["a", "b", "c", "collective", "cache", "frontier", "leaf"]
ActionKind = Literal["collective", "cache", "fill", "mma", "frontier", "ready", "leaf"]


@dataclass(frozen=True)
class PreparationBuffer:
    name: str
    kind: BufferKind
    node: Node | None
    dtype: torch.dtype
    shape: tuple[int, ...]


@dataclass(frozen=True)
class PreparationAction:
    kind: ActionKind
    event: int
    nodes: tuple[Node, ...]
    stages: tuple[int, ...]
    source_stage: int | None
    reads: tuple[str, ...]
    writes: tuple[str, ...]

    @property
    def publication_event(self) -> int:
        return self.event + 1


@dataclass(frozen=True)
class PreparationStage:
    group: ContractionGroup
    shape: tuple[int, int, int]
    a: SharedBufferRegion
    b: SharedBufferRegion


@dataclass(frozen=True)
class PreparationFrame:
    cut: PreparationCut
    layout: SharedMemoryLayoutPlan
    buffers: tuple[PreparationBuffer, ...]
    actions: tuple[PreparationAction, ...]
    stages: tuple[PreparationStage, ...]
    frontier_order: tuple[Node, ...]
    peak_live_bytes: int


class _UnsupportedFrame(Exception):
    pass


def _require(condition: bool) -> None:
    if not condition:
        raise _UnsupportedFrame


def _aligned(size: int) -> int:
    return (size + 127) // 128 * 128


class _Builder:
    def __init__(
        self,
        plan: ChainedMatmulPlan,
        cut: PreparationCut,
        shapes: Mapping[Node, tuple[int, ...]],
    ) -> None:
        self.plan = plan
        self.cut = cut
        self.shapes = shapes
        self.positions = {node: index for index, node in enumerate(cut.region.nodes)}
        self.preparation = set(cut.preparation)
        self.shared = set(cut.shared_inputs)
        self.collectives = set(cut.region.scans) | set(cut.region.reductions)
        self.cache = (
            () if plan.pointwise_cache is None else plan.pointwise_cache.entries
        )
        self.known = (
            set(plan.dots) | self.collectives | {item.node for item in self.cache}
        )
        self.materialized: dict[Node, str] = {}
        self.publications: dict[Node, PhysicalPublication] = {}
        self.use_frontier = PhysicalUseFrontier(cut.region.nodes)
        self.owners: dict[str, PreparationBuffer] = {}
        self.requests: dict[str, SharedBufferRequest] = {}
        self.buffers: list[PreparationBuffer] = []
        self.actions: list[PreparationAction] = []
        self.stage_names: list[
            tuple[ContractionGroup, tuple[int, int, int], str, str]
        ] = []

    def tensor(self, node: Node) -> tuple[torch.dtype, tuple[int, ...]]:
        value = node.meta.get("val")
        shape = self.shapes.get(node)
        _require(isinstance(value, torch.Tensor) and shape is not None)
        assert isinstance(value, torch.Tensor) and shape is not None
        _require(
            type(shape) is tuple
            and len(shape) == value.ndim
            and all(type(size) is int and size > 0 for size in shape)
            and value.dtype
            in (
                torch.bfloat16,
                torch.float16,
                torch.float32,
                torch.int32,
                torch.int64,
                torch.bool,
            )
        )
        # Static metadata must agree; symbolic extents are the caller's explicit
        # configured resolution, not an invitation to consult DeviceFunction.
        _require(
            all(
                type(old) is not int or old == new
                for old, new in zip(value.shape, shape, strict=True)
            )
        )
        return value.dtype, shape

    def dependencies(
        self, roots: Iterable[Node], *, expand: Node | None = None
    ) -> set[str]:
        return {
            self.materialized[value.node]
            for value in self.use_frontier.resolve(
                roots,
                self.publications,
                external=frozenset(self.shared),
                required=frozenset(self.known),
                traversable=frozenset(self.preparation),
                expand=expand,
            )
        }

    def allocate(
        self,
        name: str,
        kind: BufferKind,
        node: Node | None,
        dtype: torch.dtype,
        shape: tuple[int, ...],
    ) -> None:
        _require(name not in self.requests)
        event = len(self.actions)
        self.requests[name] = SharedBufferRequest(
            name,
            _aligned(math.prod(shape) * dtype.itemsize),
            128,
            event,
            event + 1,
        )
        owner = PreparationBuffer(name, kind, node, dtype, shape)
        self.buffers.append(owner)
        self.owners[name] = owner
        self.use_frontier.register_owner(owner, self.cut, event)

    def publish(self, node: Node, name: str) -> None:
        owner = self.owners[name]
        _require(owner.node is node and node not in self.publications)
        self.materialized[node] = name
        self.publications[node] = PhysicalPublication(
            node, owner, owner.dtype, owner.shape
        )

    def action(
        self,
        kind: ActionKind,
        nodes: tuple[Node, ...],
        stages: tuple[int, ...],
        reads: set[str],
        writes: tuple[str, ...],
        *,
        source_stage: int | None = None,
    ) -> None:
        event = len(self.actions)
        for name in reads:
            request = self.requests[name]
            owner = self.owners[name]
            self.use_frontier.read(owner, PhysicalReadPoint(self.cut, event, event + 1))
            self.requests[name] = replace(
                request, live_until=self.use_frontier.through(owner, self.cut)
            )
        self.actions.append(
            PreparationAction(
                kind,
                event,
                nodes,
                stages,
                stages[0] if stages else source_stage,
                tuple(sorted(reads)),
                writes,
            )
        )

    def materialize(
        self,
        node: Node,
        name: str,
        kind: Literal["collective", "cache"],
        source_stage: int,
    ) -> None:
        _require(node in self.preparation and node not in self.materialized)
        dtype, shape = self.tensor(node)
        if kind == "collective":
            _require(dtype == torch.float32)
        reads = self.dependencies((node,), expand=node)
        self.allocate(name, kind, node, dtype, shape)
        self.action(kind, (node,), (), reads, (name,), source_stage=source_stage)
        self.publish(node, name)

    def build(self) -> PreparationFrame:
        from .chained_matmul import _UnsupportedChain
        from .prepared_graph_schedule import ContractionGraph
        from .prepared_graph_schedule import LoopTerminalStorePlacement
        from .prepared_graph_schedule import schedule_contraction_groups

        plan, cut = self.plan, self.cut
        _require(
            plan.region is not None
            and plan.loop is not None
            and plan.strategy == "tcgen05_tmem"
        )
        assert plan.region is not None and plan.loop is not None
        _require(
            plan.region.graph is cut.region.graph
            and plan.region.nodes == cut.region.nodes == tuple(cut.region.graph.nodes)
            and plan.region.carries == cut.carries
            and plan.loop.region.nodes == cut.region.nodes
            and plan.loop.storage_key == cut.storage_proof_key
            and tuple(spec.node for spec in cut.region.contractions) == plan.dots
            and len(plan.shapes) == len(plan.dots)
            and bool(cut.images)
        )
        _require(
            set(cut.preparation) | set(cut.recurrence) | set(cut.shared_inputs)
            == set(cut.region.nodes)
            and not self.preparation.intersection(cut.recurrence)
            and not self.shared.intersection(cut.recurrence)
            and not self.shared.intersection(self.preparation)
            and self.collectives <= self.preparation
            and all(item.node in self.preparation for item in self.cache)
        )
        groups = plan.contraction_groups
        if groups is None:
            singletons = []
            for index, shape in enumerate(plan.shapes):
                geometry = stage_geometry(shape)
                _require(geometry is not None)
                assert geometry is not None
                singletons.append(ContractionGroup((index,), (geometry,)))
            groups = tuple(singletons)
        try:
            schedule = schedule_contraction_groups(
                ContractionGraph(cut.region),
                groups,
                plan.shapes,
                partitions=(frozenset(cut.preparation), frozenset(cut.recurrence)),
                materializations=frozenset(self.collectives),
                terminal_stores=LoopTerminalStorePlacement(
                    plan.loop,
                    cut.region.stores,
                    cut.storage_proof_key,
                    frozenset(cut.recurrence),
                ),
            )
        except _UnsupportedChain as error:
            raise _UnsupportedFrame from error
        groups = tuple(item.group for item in schedule.groups)
        for group in groups:
            _require(bool(group.stages) and len(group.stages) == len(group.geometries))
            _require(
                all(
                    geometry.logical == plan.shapes[index]
                    for index, geometry in zip(
                        group.stages, group.geometries, strict=True
                    )
                )
            )
            roles = {plan.dots[index] in self.preparation for index in group.stages}
            _require(len(roles) == 1)
            _require(
                not any(
                    self.positions[plan.dots[group.stages[0]]]
                    < self.positions[node]
                    < self.positions[plan.dots[group.stages[-1]]]
                    for node in self.collectives
                )
            )
            if True in roles:
                _require(group.stages[0] in plan.warp_mma_stages)
                _require(
                    all(plan.dots[index].args[2] is None for index in group.stages)
                )
        first_stages = {group.stages[0] for group in groups}
        _require(all(item.first_stage in first_stages for item in self.cache))
        _require(len({item.node for item in self.cache}) == len(self.cache))
        for item in self.cache:
            dtype, shape = self.tensor(item.node)
            _require(
                dtype == item.dtype
                and shape == item.shape
                and math.prod(shape) * dtype.itemsize == item.byte_size
                and item.allocated_bytes == _aligned(item.byte_size)
            )
        collectives = [node for node in cut.region.nodes if node in self.collectives]
        dot_positions = [self.positions[node] for node in plan.dots]
        pending = list(enumerate(collectives))
        for group in groups:
            first = group.stages[0]
            stop = self.positions[plan.dots[first]]
            for index, node in tuple(pending):
                if self.positions[node] < stop:
                    self.materialize(
                        node,
                        f"chain_collective_{index}",
                        "collective",
                        bisect_right(dot_positions, self.positions[node]),
                    )
                    pending.remove((index, node))
            for item in self.cache:
                if item.first_stage == first:
                    self.materialize(item.node, item.name, "cache", item.first_stage)
            if plan.dots[first] not in self.preparation:
                continue
            shape = warp_mma_shape(group.geometries[0], group)
            m, n, k = shape
            dtype = plan.operand_dtype(first)
            _require(dtype in (torch.bfloat16, torch.float16))
            _require(all(plan.operand_dtype(stage) == dtype for stage in group.stages))
            nodes = tuple(plan.dots[index] for index in group.stages)
            reads = self.dependencies(
                source for node in nodes for source in node.all_input_nodes
            )
            a, b = f"prep_{first}_a", f"prep_{first}_b"
            self.allocate(a, "a", None, dtype, (m, k))
            self.allocate(b, "b", None, dtype, (n, k))
            self.action("fill", nodes, group.stages, reads, (a, b))
            outputs = []
            for stage, node in zip(group.stages, nodes, strict=True):
                result_dtype, result_shape = self.tensor(node)
                _require(
                    result_dtype == torch.float32
                    and result_shape == plan.shapes[stage][:2]
                )
                name = f"chain_{stage}_c"
                self.allocate(name, "c", node, result_dtype, result_shape)
                self.publish(node, name)
                outputs.append(name)
            self.action("mma", nodes, group.stages, {a, b}, tuple(outputs))
            self.stage_names.append((group, shape, a, b))
        for index, node in pending:
            self.materialize(
                node,
                f"chain_collective_{index}",
                "collective",
                bisect_right(dot_positions, self.positions[node]),
            )
        _require(bool(self.stage_names))
        frontier = list(enumerate(cut.images))
        _require(len({image.node for _, image in frontier}) == len(frontier))
        dependencies = {
            image.node: self.dependencies((image.node,)) for _, image in frontier
        }
        sizes = {}
        for _, image in frontier:
            _require(image.node in self.preparation)
            dtype, shape = self.tensor(image.node)
            _require(
                dtype == image.dtype
                and _domain(image.node.meta["val"]) == image.logical_domain
            )
            sizes[image.node] = _aligned(math.prod(shape) * dtype.itemsize)
        order = []
        while frontier:
            users = Counter(
                name for _, image in frontier for name in dependencies[image.node]
            )

            def score(
                item: tuple[int, PreparationImage], remaining: Counter[str] = users
            ) -> tuple[int, int]:
                index = item[0]
                node = cut.images[index].node
                released = sum(
                    self.requests[name].byte_size
                    for name in dependencies[node]
                    if remaining[name] == 1
                )
                return released - sizes[node], -self.positions[node]

            index, image = max(frontier, key=score)
            dtype, shape = self.tensor(image.node)
            name = f"chain_prepared_{index}"
            self.allocate(name, "frontier", image.node, dtype, shape)
            self.action(
                "frontier", (image.node,), (), dependencies[image.node], (name,)
            )
            frontier.remove((index, image))
            order.append(image.node)
        names = {buffer.name for buffer in self.buffers if buffer.kind == "frontier"}
        self.action("ready", tuple(order), (), names, ())
        self.use_frontier.check()
        layout = _reuse_requests(tuple(self.requests.values()))
        peak = max(
            sum(
                region.byte_size
                for region in layout.regions
                if region.live_from <= action.event < region.live_until
            )
            for action in self.actions
        )
        return PreparationFrame(
            cut,
            layout,
            tuple(self.buffers),
            tuple(self.actions),
            tuple(
                PreparationStage(group, shape, layout.region(a), layout.region(b))
                for group, shape, a, b in self.stage_names
            ),
            tuple(order),
            peak,
        )


def plan_preparation_frame(
    plan: ChainedMatmulPlan,
    cut: PreparationCut,
    shapes: Mapping[Node, tuple[int, ...]],
) -> PreparationFrame | None:
    """Pack one slot using admitted, same-revision graph facts and explicit sizes.

    ``shapes`` resolves logical shapes of every materialized C/collective/cache
    and frontier node. No compiler context is consulted. This does not reserve
    recurrence storage, choose role sizes, prove physical layouts or claim a
    timing gain. The consumer must preserve the action/publication and whole-slot
    ownership protocol, then separately validate total resource capacity.
    """
    try:
        return _Builder(plan, cut, shapes).build()
    except (_UnsupportedFrame, InvalidPhysicalUse):
        return None
