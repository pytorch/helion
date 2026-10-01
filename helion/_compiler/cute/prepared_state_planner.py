"""Bound state-transfer scheduling inside already-established event cuts.

The inputs are unordered typed publications, not an instruction sequence.
This planner shares identical reads, keeps their FP32 values live through all
uses, and delays destructive stores/completion to the original consumer cuts.
It grants no owner, readiness, panel geometry or cross-generation authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING

from . import chained_matmul as chain
from .chained_completed_store import _records
from .physical_use_frontier import InvalidPhysicalUse
from .physical_use_frontier import PhysicalReadPoint
from .physical_use_frontier import PhysicalUseFrontier
from .prepared_state_body import StateEffect

if TYPE_CHECKING:
    from torch.fx import Node

    from .prepared_epoch_product import DescriptorProductBinding
    from .prepared_epoch_state import DescriptorStateBinding
    from .prepared_state_body import FragmentStateBinding

    StateBinding = (
        FragmentStateBinding | DescriptorStateBinding | DescriptorProductBinding
    )


@dataclass(frozen=True, eq=False)
class StateView:
    """An original full owner and its exactly admitted native member map."""

    owner: object
    scope: tuple[object, ...]
    mapping: tuple[object, ...]
    offset: int
    width: int

    def facts(self) -> object:
        if (
            type(self.offset) is not int
            or type(self.width) is not int
            or self.offset < 0
            or self.width <= 0
            or not self.scope
            or not self.mapping
        ):
            raise chain._UnsupportedChain("unbound state source view")
        return (
            id(self.owner),
            self.scope,
            _records(self.mapping),
            self.offset,
            self.width,
        )


@dataclass(frozen=True, eq=False)
class StateCut:
    """An existing event/consumer boundary, not a newly minted readiness token."""

    owner: object
    anchor: object
    scope: tuple[object, ...]

    def facts(self) -> object:
        from torch.fx import Node

        from .prepared_epoch_protocol import EpochEvent
        from .prepared_epoch_protocol import EpochSynchronizationAction
        from .prepared_epoch_state import StateArrival

        if isinstance(self.anchor, Node):
            anchor = self.anchor, _records((self.anchor.args, self.anchor.kwargs))
        elif isinstance(
            self.anchor, (EpochEvent, EpochSynchronizationAction, StateArrival)
        ):
            owner = (
                self.anchor.binding.owner
                if isinstance(self.anchor, StateArrival)
                else self.anchor.owner
            )
            if owner is not self.owner:
                raise chain._UnsupportedChain("state cut lost original event owner")
            anchor = self.anchor.facts()
        else:
            raise chain._UnsupportedChain("unsupported state consumer cut")
        if not self.scope or id(self.owner) != self.scope[0]:
            raise chain._UnsupportedChain("state boundary lost original scope")
        return id(self), id(self.owner), id(self.anchor), self.scope, anchor


@dataclass(frozen=True, eq=False)
class StatePublication:
    """Original value, native view and legal intervals of one publication.

    A ``*_before`` cut denotes the interval ending at that existing boundary.
    The representation adapter proves those intervals; the planner decides
    read placement, reuse, SSA order and in-place last-use constraints inside
    them. Completion can cover several disjoint panels at one original fence.
    """

    read: StateBinding
    value: StateBinding
    source_view: StateView
    destination_view: StateView
    transform_before: StateCut
    store_before: StateCut
    complete_before: StateCut | None

    @property
    def source(self) -> Node:
        return self.read.source

    @property
    def result(self) -> Node:
        return self.value.destination

    def facts(self) -> object:
        self.value.check_publication(self)
        if self.read.source is not self.value.source:
            raise chain._UnsupportedChain("state publication changed source value")
        if self.source.graph is not self.result.graph:
            raise chain._UnsupportedChain("state publication crosses graphs")
        if (
            self.source_view.facts() != self.read.state_view().facts()
            or self.destination_view.facts()
            != self.value.state_view(store=True).facts()
        ):
            raise chain._UnsupportedChain("state request lost original native view")
        return (
            id(self),
            self.read.facts(),
            self.value.facts(),
            self.source,
            self.result,
            self.source_view.facts(),
            self.destination_view.facts(),
            id(self.transform_before),
            id(self.store_before),
            id(self.complete_before),
        )


@dataclass(frozen=True)
class StateResidency:
    source: Node
    view: StateView
    first_interval: int
    last_interval: int
    consumers: tuple[Node, ...]


@dataclass(eq=False)
class StateTransferPlan:
    requests: tuple[StatePublication, ...]
    cuts: tuple[StateCut, ...]
    phases: tuple[tuple[StateEffect, ...], ...]
    residency: tuple[StateResidency, ...]
    _facts: object = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._facts = self._current()
        for phase in self.phases:
            for action in phase:
                object.__setattr__(action, "schedule", self)

    @property
    def actions(self) -> tuple[StateEffect, ...]:
        return tuple(action for phase in self.phases for action in phase)

    def _current(self) -> object:
        return (
            tuple(request.facts() for request in self.requests),
            tuple(cut.facts() for cut in self.cuts),
            tuple(
                tuple((id(a), a.kind, id(a.binding)) for a in p) for p in self.phases
            ),
            tuple(
                (
                    r.source,
                    r.view.facts(),
                    r.first_interval,
                    r.last_interval,
                    r.consumers,
                )
                for r in self.residency
            ),
        )

    def check(self) -> None:
        if self._current() != self._facts or any(
            action.schedule is not self for action in self.actions
        ):
            raise chain._UnsupportedChain("state transfer decision changed")


def _reject_dependent_groups(
    groups: list[list[StatePublication]], use_frontier: PhysicalUseFrontier
) -> None:
    """Offset order is legal only for independent original value groups.

    Multiple disjoint member views of one result are not producer/consumer
    edges. A proper def-use edge across groups, however, needs a different
    scheduling proof; do not load a value before another group produces it.
    """
    for index, members in enumerate(groups):
        dependencies = use_frontier.ancestors(
            node
            for request in members
            for node in (request.source, *request.result.all_input_nodes)
        )
        if any(
            request.result in dependencies
            for other, requests in enumerate(groups)
            if other != index
            for request in requests
        ):
            raise chain._UnsupportedChain("dependent state groups require a schedule")


def plan_state_transfers(
    requests: tuple[StatePublication, ...], cuts: tuple[StateCut, ...]
) -> StateTransferPlan:
    """Derive read residency and publication order from actual values/views."""
    try:
        return _plan_state_transfers(requests, cuts)
    except InvalidPhysicalUse as error:
        raise chain._UnsupportedChain(str(error)) from error


def _plan_state_transfers(
    requests: tuple[StatePublication, ...], cuts: tuple[StateCut, ...]
) -> StateTransferPlan:
    if not requests or not cuts or len({id(cut) for cut in cuts}) != len(cuts):
        raise chain._UnsupportedChain("missing or duplicated state boundary")
    if len({id(request) for request in requests}) != len(requests):
        raise chain._UnsupportedChain("duplicated state publication")
    cut_index = {id(cut): index for index, cut in enumerate(cuts)}
    graph = requests[0].source.graph
    node_order = {node: index for index, node in enumerate(graph.nodes)}
    intervals: dict[int, tuple[int, int, int | None]] = {}
    groups: dict[tuple[object, ...], list[StatePublication]] = {}
    for request in requests:
        request.facts()
        if request.source.graph is not graph:
            raise chain._UnsupportedChain("foreign state request graph")
        selected = (request.transform_before, request.store_before)
        if request.complete_before is not None:
            selected = (*selected, request.complete_before)
        if any(id(cut) not in cut_index for cut in selected):
            raise chain._UnsupportedChain("foreign state consumer boundary")
        if any(cut.scope != request.source_view.scope for cut in selected):
            raise chain._UnsupportedChain(
                "state boundary changed participant/generation"
            )
        transform, store = (cut_index[id(cut)] for cut in selected[:2])
        complete = (
            None
            if request.complete_before is None
            else cut_index[id(request.complete_before)]
        )
        if transform > store or complete is not None and store > complete:
            raise chain._UnsupportedChain("state publication precedes its value")
        intervals[id(request)] = transform, store, complete
        key = (request.source, request.source_view.facts())
        groups.setdefault(key, []).append(request)
    use_frontier = PhysicalUseFrontier(tuple(graph.nodes))
    _reject_dependent_groups(list(groups.values()), use_frontier)
    read_views = {id(members): members[0].source_view for members in groups.values()}
    for members in groups.values():
        # This is the original admitted native StateView, not a dense image of
        # its whole source Node or a shared-memory allocation. Existing facts()
        # and overlap guards retain its exact mapping/owner/generation proof.
        view = read_views[id(members)]
        use_frontier.register_live_in(view, view.scope)
        for request in members:
            transform = intervals[id(request)][0]
            use_frontier.read(
                view, PhysicalReadPoint(view.scope, transform, transform + 1)
            )
    for index, request in enumerate(requests):
        a = request.destination_view
        for other in requests[index + 1 :]:
            b = other.destination_view
            if (
                a.owner is b.owner
                and a.scope == b.scope
                and max(a.offset, b.offset)
                < min(a.offset + a.width, b.offset + b.width)
            ):
                raise chain._UnsupportedChain("overlapping state publications")
    # Distinct values or mappings which overlap one live native owner are not
    # established aliases. Preserve them as a rejection, not an inferred reuse.
    keys = list(groups)
    for index, key in enumerate(keys):
        left = groups[key][0]
        for other in keys[index + 1 :]:
            right = groups[other][0]
            a, b = left.source_view, right.source_view
            if (
                a.owner is b.owner
                and a.scope == b.scope
                and max(a.offset, b.offset)
                < min(a.offset + a.width, b.offset + b.width)
            ):
                raise chain._UnsupportedChain("overlapping nonidentical state reads")
    for request in requests:
        target = request.destination_view
        for members in groups.values():
            source = members[0].source_view
            overlap = (
                target.owner is source.owner
                and target.scope == source.scope
                and max(target.offset, source.offset)
                < min(target.offset + target.width, source.offset + source.width)
            )
            if overlap:
                if (
                    target.facts() != source.facts()
                    or target.facts() != request.source_view.facts()
                ):
                    raise chain._UnsupportedChain("partial state overwrite is unproven")
                read_view = read_views[id(members)]
                last = use_frontier.through(read_view, read_view.scope) - 1
                if intervals[id(request)][1] < last:
                    raise chain._UnsupportedChain("state overwrite precedes last use")
    ordered_groups = sorted(
        groups.values(),
        key=lambda members: (
            members[0].source_view.offset,
            node_order[members[0].source],
        ),
    )
    phases: list[list[StateEffect]] = [[] for _ in cuts]
    residencies = []
    for members in ordered_groups:
        members.sort(key=lambda request: node_order[request.result])
        source = read_views[id(members)]
        first = use_frontier.first_read(source, source.scope)
        last = use_frontier.through(source, source.scope) - 1
        # Different read bindings must still designate the same original RMEM
        # SSA value; a same-shaped foreign temporary cannot borrow this read.
        read = members[0].read
        if any(request.read is not read for request in members):
            raise chain._UnsupportedChain("equal state view lost original read binding")
        phases[first].append(StateEffect("read", read))
        deferred: list[tuple[int, StateEffect]] = []
        for request in members:
            transform, store, _ = intervals[id(request)]
            phases[transform].append(StateEffect("transform", request.value))
            action = StateEffect("store", request.value)
            if request.destination_view.facts() == request.source_view.facts():
                deferred.append((store, action))
            else:
                phases[store].append(action)
        for store, action in deferred:
            phases[store].append(action)
        residencies.append(
            StateResidency(
                read.source,
                members[0].source_view,
                first,
                last,
                tuple(request.result for request in members),
            )
        )
    # A fence retires exactly the already-issued stores due at that cut. One
    # original fragment fence covers its disjoint panels; descriptor snapshots
    # with different deadlines remain separate operations.
    for index, phase in enumerate(phases):
        due = [request for request in requests if intervals[id(request)][2] == index]
        if due:
            owner = due[0].destination_view
            if any(
                request.destination_view.owner is not owner.owner
                or request.destination_view.scope != owner.scope
                for request in due
            ):
                raise chain._UnsupportedChain("foreign state completion domain")
            issued = [
                action.binding
                for prior in phases[: index + 1]
                for action in prior
                if action.kind == "store"
            ]
            due_bindings = {id(request.value) for request in due}
            representative = next(
                binding for binding in reversed(issued) if id(binding) in due_bindings
            )
            phase.append(StateEffect("complete", representative))
    use_frontier.check()
    return StateTransferPlan(
        requests, cuts, tuple(tuple(p) for p in phases), tuple(residencies)
    )
