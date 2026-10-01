"""Use-def recognition and issue ordering for original contraction regions.

Physical adapters bind original owners, role scopes and issue intervals. They
do not assign semantic dot ordinals here. This analysis has no readiness token
or renderer: the existing body still owns native publication and completion.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from itertools import pairwise
from typing import TYPE_CHECKING
from typing import Protocol

from torch.fx import Node

from . import chained_matmul as chain
from .chained_completed_store import _records

if TYPE_CHECKING:
    from collections.abc import Callable

    from .chained_contraction_groups import ContractionGroup
    from .chained_initialized_accumulator import InitializedAccumulator
    from .chained_loop import ChainedLoopPlan
    from .contraction_region import ContractionRegion
    from .contraction_region import ContractionSpec
    from .prepared_continuation import ContractionIssue


class DependencyKind(Enum):
    VALUE = "completed_value"
    ACCUMULATOR = "ordered_accumulator"
    TRANSFORMED = "completed_transformed_seed"


@dataclass(frozen=True)
class ContractionDependency:
    producer: ContractionSpec
    consumer: ContractionSpec
    value: Node
    kind: DependencyKind
    # Preserve every intermediate operation, including casts, masks and indices.
    path: frozenset[Node]


def _input_reader(
    dots: frozenset[Node],
) -> Callable[[Node], dict[Node, frozenset[Node]]]:
    """Memoized nearest ancestors within one immutable graph discovery."""
    cache: dict[Node, dict[Node, frozenset[Node]]] = {}
    active: set[Node] = set()

    def visit(node: Node) -> dict[Node, frozenset[Node]]:
        if node in dots:
            return {node: frozenset()}
        if node in cache:
            return cache[node]
        if node in active:
            raise chain._UnsupportedChain("cyclic contraction value dependency")
        active.add(node)
        result: dict[Node, set[Node]] = {}
        for parent in node.all_input_nodes:
            for source, path in visit(parent).items():
                result.setdefault(source, set()).update((*path, node))
        active.remove(node)
        cache[node] = {source: frozenset(path) for source, path in result.items()}
        return cache[node]

    return visit


def seed_pairs(nodes: tuple[Node, ...]) -> tuple[tuple[Node, Node], ...]:
    """Discover original add(seed(first), second) candidates, not their proof."""
    import torch

    from ...language.matmul_ops import dot

    dots = frozenset(node for node in nodes if node.target is dot)
    inputs = _input_reader(dots)
    result = []
    for join in nodes:
        if (
            join.target is not torch.ops.aten.add.Tensor
            or len(join.args) != 2
            or not isinstance(join.args[0], Node)
            or not isinstance(join.args[1], Node)
            or join.args[1] not in dots
        ):
            continue
        second = join.args[1]
        for first in inputs(join.args[0]):
            if first is not second:
                result.append((first, second))
    return tuple(result)


def dependency_nodes(
    nodes: tuple[Node, ...],
) -> tuple[tuple[Node, Node, Node, DependencyKind, frozenset[Node]], ...]:
    """Structural use-def edges also available before physical shape binding."""
    from ...language.matmul_ops import dot

    dots = frozenset(node for node in nodes if node.target is dot)
    return _dependency_nodes(nodes, dots, _input_reader(dots))


def _dependency_nodes(
    nodes: tuple[Node, ...],
    dots: frozenset[Node],
    inputs: Callable[[Node], dict[Node, frozenset[Node]]],
) -> tuple[tuple[Node, Node, Node, DependencyKind, frozenset[Node]], ...]:
    result = []
    for consumer in nodes:
        if consumer not in dots:
            continue
        if len(consumer.args) != 4 or consumer.kwargs:
            raise chain._UnsupportedChain("malformed contraction dependency")
        for value, role in (
            (consumer.args[0], DependencyKind.VALUE),
            (consumer.args[1], DependencyKind.VALUE),
            (consumer.args[2], DependencyKind.ACCUMULATOR),
        ):
            if value is None and role is DependencyKind.ACCUMULATOR:
                continue
            if not isinstance(value, Node):
                raise chain._UnsupportedChain("missing contraction dependency value")
            for producer, path in inputs(value).items():
                if producer is consumer:
                    raise chain._UnsupportedChain("self-dependent contraction")
                result.append(
                    (
                        producer,
                        consumer,
                        value,
                        role if not path else DependencyKind.VALUE,
                        path,
                    )
                )
    return tuple(result)


@dataclass(frozen=True)
class ContractionGraph:
    region: ContractionRegion
    transformed: tuple[InitializedAccumulator, ...] = ()
    dependencies: tuple[ContractionDependency, ...] = field(init=False)
    effects: tuple[tuple[Node, frozenset[Node]], ...] = field(init=False)
    _revision: object = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        dependencies, effects = self._discover()
        object.__setattr__(self, "dependencies", dependencies)
        object.__setattr__(self, "effects", effects)
        object.__setattr__(self, "_revision", self._current())

    def _discover(
        self,
    ) -> tuple[
        tuple[ContractionDependency, ...],
        tuple[tuple[Node, frozenset[Node]], ...],
    ]:
        region = self.region
        nodes = tuple(region.graph.nodes)
        if nodes != region.nodes or any(
            node.graph is not region.graph for node in nodes
        ):
            raise chain._UnsupportedChain("contraction region changed before discovery")
        by_node = {spec.node: spec for spec in region.contractions}
        if len(by_node) != len(region.contractions) or any(
            node not in nodes for node in by_node
        ):
            raise chain._UnsupportedChain("ambiguous contraction region")
        dots = frozenset(by_node)
        for spec in region.contractions:
            if (
                spec.node.args
                != (spec.lhs, spec.rhs, spec.accumulator, spec.requested_out_dtype)
                or spec.node.kwargs
            ):
                raise chain._UnsupportedChain("stale original contraction operands")
        inputs = _input_reader(dots)
        dependencies = [
            ContractionDependency(by_node[source], by_node[target], value, kind, path)
            for source, target, value, kind, path in _dependency_nodes(
                nodes, dots, inputs
            )
        ]
        if self.transformed:
            from .chained_initialized_accumulator import (
                classify_initialized_accumulator,
            )

            selected = classify_initialized_accumulator(nodes)
            if len(self.transformed) != 1 or selected != self.transformed[0]:
                raise chain._UnsupportedChain("foreign transformed seed relation")
            proof = self.transformed[0]
            if proof.first not in by_node or proof.second not in by_node:
                raise chain._UnsupportedChain("transformed seed outside region")
            path = inputs(proof.seed)
            if set(path) != {proof.first}:
                raise chain._UnsupportedChain("ambiguous transformed seed dependency")
            dependencies.append(
                ContractionDependency(
                    by_node[proof.first],
                    by_node[proof.second],
                    proof.seed,
                    DependencyKind.TRANSFORMED,
                    path[proof.first],
                )
            )
        # Store indices/masks are effects too, not only the stored payload.
        # Loop carried outputs remain original output ports rather than a
        # same-iteration edge back to their input placeholders.
        effects = tuple(
            (node, frozenset(inputs(node)))
            for node in dict.fromkeys((*region.stores, *region.live_outs))
        )
        return tuple(dependencies), effects

    def _current(self) -> object:
        return (
            id(self.region),
            _records(
                (
                    self.region.contractions,
                    self.region.live_ins,
                    self.region.live_outs,
                    self.region.loads,
                    self.region.stores,
                    self.region.scans,
                    self.region.reductions,
                    self.region.carries,
                    self.transformed,
                    self.dependencies,
                    self.effects,
                )
            ),
            tuple(
                (
                    node,
                    node.op,
                    node.target,
                    _records(node.args),
                    _records(node.kwargs),
                    _records(node.meta),
                    tuple(node.users),
                )
                for node in self.region.graph.nodes
            ),
        )

    def check(self) -> None:
        if self._current() != self._revision:
            raise chain._UnsupportedChain("contraction dependency graph changed")

    def predecessors(self, spec: ContractionSpec) -> frozenset[Node]:
        self.check()
        if not any(item is spec for item in self.region.contractions):
            raise chain._UnsupportedChain("foreign scheduled contraction")
        return frozenset(
            edge.producer.node for edge in self.dependencies if edge.consumer is spec
        )

    def order(self) -> tuple[ContractionSpec, ...]:
        """Topological order; original graph order breaks independent ties."""
        self.check()
        positions = {node: index for index, node in enumerate(self.region.nodes)}
        pending = {spec.node: spec for spec in self.region.contractions}
        predecessors = {node: set() for node in pending}
        for edge in self.dependencies:
            predecessors[edge.consumer.node].add(edge.producer.node)
        emitted: set[Node] = set()
        result = []
        while pending:
            ready = [
                spec for spec in pending.values() if predecessors[spec.node] <= emitted
            ]
            if not ready:
                raise chain._UnsupportedChain("cyclic contraction seed schedule")
            spec = min(ready, key=lambda item: positions[item.node])
            result.append(spec)
            emitted.add(spec.node)
            pending.pop(spec.node)
        return tuple(result)


@dataclass(frozen=True)
class ContractionPartition:
    """Selected original specs in a physical owner's legal issue priority.

    The full graph remains authoritative. Incoming preparation dependencies,
    outgoing consumers and intermediate casts are retained, not replaced by
    synthetic contraction nodes or claimed as covered by this partition.
    """

    graph: ContractionGraph
    specs: tuple[ContractionSpec, ...]

    def check(self) -> None:
        self.graph.check()
        selected = {id(spec): index for index, spec in enumerate(self.specs)}
        original = {id(spec) for spec in self.graph.region.contractions}
        if (
            not selected
            or len(selected) != len(self.specs)
            or not selected.keys() <= original
        ):
            raise chain._UnsupportedChain("invalid contraction partition")
        for consumer, spec in enumerate(self.specs):
            ancestors = chain._ancestors(spec.node)
            if any(item.node in ancestors for item in self.specs[consumer + 1 :]):
                raise chain._UnsupportedChain(
                    "partition reverses contraction dependency"
                )

    @property
    def boundary(self) -> tuple[ContractionDependency, ...]:
        self.check()
        selected = {id(spec) for spec in self.specs}
        return tuple(
            dependency
            for dependency in self.graph.dependencies
            if (id(dependency.producer) in selected)
            != (id(dependency.consumer) in selected)
        )


@dataclass(frozen=True)
class IssuePlacement:
    """An original issue interval and its bound issuer/member, not a dot rank."""

    issue: ContractionIssue
    owner: object
    role: object
    member: tuple[int, int]


@dataclass(frozen=True)
class ScheduledContractionGroup:
    """Original logical members with events assigned by their selected order."""

    group: ContractionGroup
    partition: int
    read_event: int
    publication_event: int


@dataclass(frozen=True)
class LoopTerminalStorePlacement:
    """The existing loop epilogue, conditional on its original storage proof.

    This is not an alias proof or permission to move stores. The accepted loop
    already emits these stores after the contraction body; its original bound
    runtime storage guard and downstream completion checks remain required.
    """

    loop: ChainedLoopPlan
    stores: tuple[Node, ...]
    storage_key: str
    recurrence: frozenset[Node]

    def check(
        self, graph: ContractionGraph, partitions: tuple[frozenset[Node], ...]
    ) -> None:
        region = graph.region
        if (
            self.loop.body.graph is not region.graph
            or self.loop.region.graph is not region.graph
            or self.loop.region.nodes != region.nodes
            or self.loop.region.carries != region.carries
            or self.loop.region.stores != self.stores
            or self.stores != region.stores
            or self.storage_key != self.loop.storage_key
            or sum(partition == self.recurrence for partition in partitions) != 1
            or not set(self.stores) <= self.recurrence
        ):
            raise chain._UnsupportedChain("original terminal store placement changed")


@dataclass(frozen=True)
class GroupedContractionSchedule:
    groups: tuple[ScheduledContractionGroup, ...]
    transition_event: int
    store_events: tuple[tuple[Node, int], ...]


def schedule_contraction_groups(
    graph: ContractionGraph,
    groups: tuple[ContractionGroup, ...],
    shapes: tuple[tuple[int, int, int], ...],
    *,
    partitions: tuple[frozenset[Node], ...],
    materializations: frozenset[Node] = frozenset(),
    terminal_stores: LoopTerminalStorePlacement | None = None,
) -> GroupedContractionSchedule:
    """Order unordered, already-bound physical groups before creating effects.

    This pass chooses no new grouping, role or alias. Each original group is
    indivisible, each logical stage occurs once, and native member order stays
    intact. Existing materialization cuts cannot lie inside a physical group.
    Independent groups keep original FX order; stage IDs remain identity keys,
    while events count the logical members in the selected schedule.
    """
    from ...language import memory_ops

    graph.check()
    specs = graph.region.contractions
    if len(shapes) != len(specs) or not groups or not partitions:
        raise chain._UnsupportedChain("incomplete grouped contraction schedule")
    positions = {node: index for index, node in enumerate(graph.region.nodes)}
    if not materializations <= positions.keys():
        raise chain._UnsupportedChain("foreign grouped materialization cut")
    stores = tuple(
        node for node in graph.region.nodes if node.target is memory_ops.store
    )
    if stores != graph.region.stores:
        raise chain._UnsupportedChain("original store inventory changed")
    if terminal_stores is not None:
        terminal_stores.check(graph, partitions)
    # An actual store is a cut unless the original physical loop places it in
    # the epilogue. A pure live-out definition is not its publication event.
    cuts = materializations | (
        frozenset(stores) if terminal_stores is None else frozenset()
    )
    membership = {}
    partitions_by_group = {}
    for index, group in enumerate(groups):
        if (
            not group.stages
            or len(group.stages) != len(group.geometries)
            or any(
                type(stage) is not int or not 0 <= stage < len(specs)
                for stage in group.stages
            )
            or tuple(sorted(group.stages)) != group.stages
            or group.stages != tuple(range(group.stages[0], group.stages[-1] + 1))
        ):
            raise chain._UnsupportedChain("invalid grouped logical members")
        roles = set()
        for stage, geometry in zip(group.stages, group.geometries, strict=True):
            if stage in membership or geometry.logical != shapes[stage]:
                raise chain._UnsupportedChain("duplicate or changed grouped geometry")
            membership[stage] = index
            matches = [
                role
                for role, nodes in enumerate(partitions)
                if specs[stage].node in nodes
            ]
            if len(matches) != 1:
                raise chain._UnsupportedChain(
                    "grouped stage lost its original partition"
                )
            roles.add(matches[0])
        if len(roles) != 1:
            raise chain._UnsupportedChain("physical group crosses original roles")
        partitions_by_group[index] = roles.pop()
        first, last = (
            positions[specs[stage].node]
            for stage in (group.stages[0], group.stages[-1])
        )
        if any(first < positions[node] < last for node in cuts):
            raise chain._UnsupportedChain("physical group crosses materialization")
    if set(membership) != set(range(len(specs))):
        raise chain._UnsupportedChain("missing grouped logical stage")
    stages = {spec.node: index for index, spec in enumerate(specs)}
    predecessors: dict[int, set[int]] = {index: set() for index in range(len(groups))}
    for edge in graph.dependencies:
        producer = membership[stages[edge.producer.node]]
        consumer = membership[stages[edge.consumer.node]]
        if producer == consumer:
            raise chain._UnsupportedChain(
                "physical group contains dependent contractions"
            )
        predecessors[consumer].add(producer)
    pending = set(predecessors)
    emitted: set[int] = set()
    result = []
    event = 0
    while pending:
        ready = [index for index in pending if predecessors[index] <= emitted]
        if not ready:
            raise chain._UnsupportedChain("cyclic grouped contraction schedule")
        selected = min(
            ready, key=lambda index: positions[specs[groups[index].stages[0]].node]
        )
        group = groups[selected]
        result.append(
            ScheduledContractionGroup(
                group, partitions_by_group[selected], event, event + 1
            )
        )
        event += 2 * len(group.stages)
        pending.remove(selected)
        emitted.add(selected)
    return GroupedContractionSchedule(
        tuple(result),
        event,
        ()
        if terminal_stores is None
        else tuple((store, event) for store in terminal_stores.stores),
    )


class NativeAction(Protocol):
    def facts(self) -> object: ...


@dataclass(frozen=True)
class NativeReadiness:
    """One existing wait/acquire with its instruction-specific address setup."""

    wait: NativeAction
    acquire: NativeAction | None = None
    address: NativeAction | None = None
    address_before_wait: bool = False

    def facts(self) -> object:
        return (
            self.wait.facts(),
            self.acquire.facts() if self.acquire is not None else None,
            self.address.facts() if self.address is not None else None,
            self.address_before_wait,
        )

    def actions(self) -> tuple[NativeAction, ...]:
        result = []
        if self.address_before_wait and self.address is not None:
            result.append(self.address)
        result.append(self.wait)
        if self.acquire is not None:
            result.append(self.acquire)
        if not self.address_before_wait and self.address is not None:
            result.append(self.address)
        for action in result:
            action.facts()
        return tuple(result)


@dataclass(frozen=True)
class OperandPublication:
    operand: str
    value: Node
    ready: NativeReadiness


@dataclass(frozen=True)
class NativeIssueBinding:
    placement: IssuePlacement
    inputs: tuple[OperandPublication, ...]
    available: NativeReadiness | None = None
    complete: NativeAction | None = None

    def facts(self) -> object:
        # Original actions validate their own operands. Their owner is checked
        # by the original bound issue, not recursively copied once per leaf.
        placement = self.placement
        return (
            id(placement.issue),
            id(placement.owner),
            id(placement.role),
            placement.member,
            tuple(
                (item.operand, item.value, item.ready.facts()) for item in self.inputs
            ),
            self.available.facts() if self.available is not None else None,
            self.complete.facts() if self.complete is not None else None,
        )


class NativeBoundIssue(NativeAction, Protocol):
    @property
    def issue(self) -> ContractionIssue: ...

    @property
    def binding(self) -> NativeIssueBinding: ...


def schedule_native_issues(
    graph: ContractionGraph, operations: tuple[NativeBoundIssue, ...]
) -> tuple[tuple[object, tuple[NativeAction, ...]], ...]:
    """Generate actual issue/wait actions from unordered original operand facts.

    Only the original operation can own its physical binding. Its facts()
    method retains native layout/owner/event authority; this function does not
    manufacture readiness. Native waits are shared only within one issuer and
    only when the exact original wait object is the same.
    """
    by_placement = {}
    for operation in operations:
        operation.facts()
        binding = operation.binding
        placement = binding.placement
        if operation.issue is not placement.issue or id(placement) in by_placement:
            raise chain._UnsupportedChain("foreign native issue placement")
        inputs = {item.operand: item for item in binding.inputs}
        if (
            len(inputs) != 2
            or len(binding.inputs) != 2
            or set(inputs) != {"lhs", "rhs"}
            or inputs["lhs"].value is not placement.issue.spec.lhs
            or inputs["rhs"].value is not placement.issue.spec.rhs
        ):
            raise chain._UnsupportedChain("native publication lost original operand")
        by_placement[id(placement)] = operation
    ordered = order_issue_placements(
        graph, tuple(operation.binding.placement for operation in operations)
    )
    result = []
    for role in ordered:
        actions: list[NativeAction] = []
        waited: dict[int, NativeReadiness] = {}
        for placement in role:
            operation = by_placement[id(placement)]
            operation.facts()
            binding = operation.binding
            inputs = {item.operand: item for item in binding.inputs}
            requirements = [inputs["lhs"].ready, inputs["rhs"].ready]
            if binding.available is not None:
                requirements.append(binding.available)
            for ready in requirements:
                key = id(ready.wait)
                if key in waited:
                    if ready != waited[key]:
                        raise chain._UnsupportedChain("shared native wait changed")
                    continue
                actions.extend(ready.actions())
                waited[key] = ready
            actions.append(operation)
            if binding.complete is not None:
                binding.complete.facts()
                actions.append(binding.complete)
        result.append((role[0].role, tuple(actions)))
    return tuple(result)


def order_issue_placements(
    graph: ContractionGraph,
    placements: tuple[IssuePlacement, ...],
    *,
    partition: ContractionPartition | None = None,
) -> tuple[tuple[IssuePlacement, ...], ...]:
    """Schedule each original issuer separately, retaining cross-role edges.

    Owner/readiness and member-to-native-layout validity remain obligations of
    the original physical binding. No wait or completion is inferred from this
    ordering alone. Every selected logical contraction must have a complete K
    cover for each of its original result members.
    """
    graph.check()
    if not placements or len({id(item) for item in placements}) != len(placements):
        raise chain._UnsupportedChain("missing or duplicate issue placement")
    if partition is not None:
        partition.check()
        if partition.graph is not graph:
            raise chain._UnsupportedChain("foreign contraction partition")
    selected = graph.region.contractions if partition is None else partition.specs
    specs = {id(spec): spec for spec in selected}
    if {id(item.issue.spec) for item in placements} != set(specs):
        raise chain._UnsupportedChain("incomplete contraction issue schedule")
    owners = {id(item.owner) for item in placements}
    if len(owners) != 1:
        raise chain._UnsupportedChain("foreign issue owner")
    groups: dict[tuple[int, tuple[int, int]], list[IssuePlacement]] = {}
    for item in placements:
        item.issue.check()
        if (
            item.issue.region is not graph.region
            or len(item.member) != 2
            or any(type(index) is not int for index in item.member)
            or not 0 <= item.member[0] < item.member[1]
        ):
            raise chain._UnsupportedChain("invalid issue member")
        groups.setdefault((id(item.issue.spec), item.member), []).append(item)
    for members in groups.values():
        ordered = sorted(members, key=lambda item: item.issue.begin)
        cursor = 0
        first = ordered[0]
        for item in ordered:
            if (
                item.role is not first.role
                or item.issue.atom_k != first.issue.atom_k
                or item.issue.begin != cursor
                or cursor != 0
                and not item.issue.initialized
            ):
                raise chain._UnsupportedChain("changed issue K prefix or role")
            cursor = item.issue.end
        k = chain._host_shape(first.issue.spec.lhs.meta["val"])[1]
        if cursor * first.issue.atom_k != k:
            raise chain._UnsupportedChain("incomplete scheduled K prefix")
    # Disconnected result members are allowed only without overlap. Their
    # full owner/layout correspondence is still checked by the physical port.
    for spec in selected:
        members = sorted(member for key, member in groups if key == id(spec))
        if any(left[1] > right[0] for left, right in pairwise(members)):
            raise chain._UnsupportedChain("overlapping issue result members")
    priority = graph.order() if partition is None else partition.specs
    rank = {id(spec): index for index, spec in enumerate(priority)}
    ordered = sorted(
        placements,
        key=lambda item: (
            rank[id(item.issue.spec)],
            item.member,
            item.issue.begin,
        ),
    )
    roles: list[object] = []
    for item in ordered:
        if all(role is not item.role for role in roles):
            roles.append(item.role)
    return tuple(tuple(item for item in ordered if item.role is role) for role in roles)
