"""Append-only sets of exact raw preparation images and private TMA phases.

Capture the discovery witness before inserting the first leaf. This component
extends an already admitted single-leaf schedule; it neither chooses that first
leaf nor grants descriptor, domain, asynchronous-completion or storage-finalizer
authority. All original frame placements and action order survive. The caller
must register each new wrapper and finalize the complete resource plan only
after its ordinary late proofs.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from typing import TYPE_CHECKING

from .chained_pipeline_storage import _facts
from .chained_pipeline_storage import _freeze
from .chained_preparation_leaves import insert_preparation_leaf
from .chained_prepared_groups import _valid_frame
from .chained_prepared_groups import prepared_group_candidates
from .chained_prepared_operands import plan_prepared_operands

if TYPE_CHECKING:
    from .chained_matmul import ChainedMatmulPlan
    from .chained_preparation_frame import PreparationAction
    from .chained_preparation_frame import PreparationFrame
    from .chained_preparation_leaves import PreparationLeaf
    from .chained_prepared_groups import PreparedGroupBinding
    from .chained_prepared_operands import PreparedOperand
    from .chained_recurrence_workspace import RecurrenceWorkspace


@dataclass(frozen=True)
class LeafSetWitness:
    """Same-attempt semantic witness; not serializable proof authority."""

    plan: ChainedMatmulPlan
    frame: PreparationFrame
    candidates: tuple[PreparationLeaf, ...]
    prepared_groups: tuple[PreparedGroupBinding, ...]
    facts: tuple[object, ...]
    leaf_facts: tuple[object, ...]


@dataclass(frozen=True)
class PreparationLeafSet:
    frame: PreparationFrame
    leaves: tuple[PreparationLeaf, ...]
    prepared_operands: tuple[PreparedOperand, ...]
    prepared_groups: tuple[PreparedGroupBinding, ...]


def _leaf_fact(leaf: PreparationLeaf) -> object:
    return (leaf.node, leaf.name, leaf.proof, _freeze(leaf.wrapper))


def _candidate_fact(leaf: PreparationLeaf) -> object:
    return (_leaf_fact(leaf), leaf.first_event, leaf.last_event, leaf.read_events)


def _action_key(action: PreparationAction, leaf_names: frozenset[str]) -> tuple:
    return (
        action.kind,
        action.nodes,
        action.stages,
        action.source_stage,
        tuple(name for name in action.reads if name not in leaf_names),
        action.writes,
    )


def capture_leaf_set_witness(
    plan: ChainedMatmulPlan,
    frame: PreparationFrame,
    candidates: tuple[PreparationLeaf, ...],
    *,
    prepared_groups: tuple[PreparedGroupBinding, ...] = (),
) -> LeafSetWitness | None:
    """Retain original action identities and graph facts at leaf discovery."""
    if (
        plan.region is None
        or plan.loop is None
        or frame.cut.region.graph is not plan.region.graph
        or not _valid_frame(frame)
        or any(action.kind == "leaf" for action in frame.actions)
        or len({leaf.node for leaf in candidates}) != len(candidates)
        or len({leaf.name for leaf in candidates}) != len(candidates)
        or len({_action_key(action, frozenset()) for action in frame.actions})
        != len(frame.actions)
    ):
        return None
    for leaf in candidates:
        events = leaf.read_events
        if (
            leaf.node not in plan.region.nodes
            or not events
            or any(type(event) is not int for event in events)
            or tuple(sorted(set(events))) != events
            or not 0 <= leaf.first_event == events[0]
            or not events[-1] == leaf.last_event < frame.actions[-1].event
            or any(frame.actions[event].kind in ("mma", "ready") for event in events)
            or leaf.name in {buffer.name for buffer in frame.buffers}
        ):
            return None
    return LeafSetWitness(
        plan,
        frame,
        candidates,
        prepared_groups,
        _facts(plan),
        tuple(_candidate_fact(leaf) for leaf in candidates),
    )


def _remap(
    witness: LeafSetWitness, frame: PreparationFrame, leaf: PreparationLeaf
) -> PreparationLeaf | None:
    names = frozenset(buffer.name for buffer in frame.buffers if buffer.kind == "leaf")
    by_key: dict[tuple, list[int]] = {}
    for action in frame.actions:
        if action.kind != "leaf":
            by_key.setdefault(_action_key(action, names), []).append(action.event)
    events = []
    for event in leaf.read_events:
        matches = by_key.get(_action_key(witness.frame.actions[event], frozenset()), [])
        if len(matches) != 1:
            return None
        events.append(matches[0])
    return replace(
        leaf, first_event=events[0], last_event=events[-1], read_events=tuple(events)
    )


def _rebind(
    plan: ChainedMatmulPlan,
    frame: PreparationFrame,
    recurrence: RecurrenceWorkspace,
    selected: tuple[PreparedGroupBinding, ...],
) -> tuple[tuple[PreparedOperand, ...], tuple[PreparedGroupBinding, ...]] | None:
    operands = plan_prepared_operands(plan, frame, recurrence)
    groups = prepared_group_candidates(plan, frame, recurrence)
    if operands is None or groups is None:
        return None
    by_group = {candidate.group: candidate for candidate in groups}
    if len({item.candidate.group for item in selected}) != len(selected):
        return None
    rebound = []
    for binding in selected:
        candidate = by_group.get(binding.candidate.group)
        if candidate is None or any(
            frame.layout.region(member.buffer.name).byte_offset
            != binding.byte_offset + member.byte_offset
            for member in candidate.members
        ):
            return None
        rebound.append(replace(binding, candidate=candidate))
    return operands, tuple(rebound)


def extend_preparation_leaves(
    plan: ChainedMatmulPlan,
    witness: LeafSetWitness,
    frame: PreparationFrame,
    recurrence: RecurrenceWorkspace,
    leaves: tuple[PreparationLeaf, ...],
    prepared_operands: tuple[PreparedOperand, ...],
    prepared_groups: tuple[PreparedGroupBinding, ...],
    *,
    max_count: int,
    capacity: int,
) -> PreparationLeafSet | None:
    """Append repeated original-action loads without moving existing bytes.

    Missing/stale/ambiguous witnesses reject the extension. Capacity failures
    retain the ordinary load and try subsequent candidates. Returned leaf event
    fields refer to reads in the FINAL frame, rather than stale discovery events.
    The original single-leaf path need not call this helper.
    """
    if (
        type(max_count) is not int
        or max_count <= 0
        or type(capacity) is not int
        or capacity < frame.layout.allocated_bytes
        or witness.plan is not plan
        or witness.facts != _facts(plan)
        or witness.leaf_facts
        != tuple(_candidate_fact(leaf) for leaf in witness.candidates)
        or not leaves
        or len(leaves) > max_count
        or len({leaf.node for leaf in leaves}) != len(leaves)
    ):
        return None
    originals = {leaf.node: leaf for leaf in witness.candidates}
    if any(
        leaf.node not in originals
        or _leaf_fact(leaf) != _leaf_fact(originals[leaf.node])
        for leaf in leaves
    ):
        return None
    # Reproduce only allocation/action bookkeeping, not numerical emission.
    # This checks the complete current frame, including existing leaf aliases,
    # first-fit placement, exclusive-end lifetimes and all native group offsets.
    replay = witness.frame
    bindings = _rebind(plan, replay, recurrence, witness.prepared_groups)
    if bindings is None or bindings[1] != witness.prepared_groups:
        return None
    for ordinal, admitted in enumerate(leaves):
        current = _remap(witness, replay, originals[admitted.node])
        if current is None:
            return None
        next_frame = insert_preparation_leaf(
            replay,
            current,
            capacity=capacity,
            prepared_groups=bindings[1],
            append=ordinal > 0,
        )
        if next_frame is None:
            return None
        replay = next_frame
        bindings = _rebind(plan, replay, recurrence, bindings[1])
        if bindings is None:
            return None
    if replay != frame or bindings != (prepared_operands, prepared_groups):
        return None
    admitted_nodes = {leaf.node for leaf in leaves}
    for leaf in witness.candidates:
        if len(admitted_nodes) == max_count:
            break
        if leaf.node in admitted_nodes or len(leaf.read_events) < 2:
            continue
        current = _remap(witness, frame, leaf)
        if current is None:
            return None
        extended = insert_preparation_leaf(
            frame,
            current,
            capacity=capacity,
            prepared_groups=prepared_groups,
            append=True,
        )
        if extended is None:
            continue
        rebound = _rebind(plan, extended, recurrence, prepared_groups)
        if rebound is None:
            return None
        frame = extended
        prepared_operands, prepared_groups = rebound
        leaves = (*leaves, leaf)
        admitted_nodes.add(leaf.node)
    remapped = tuple(_remap(witness, frame, originals[leaf.node]) for leaf in leaves)
    if any(leaf is None for leaf in remapped):
        return None
    return PreparationLeafSet(
        frame,
        tuple(leaf for leaf in remapped if leaf is not None),
        prepared_operands,
        prepared_groups,
    )


@dataclass(frozen=True)
class LeafSetProtocol:
    """Private completion objects; caller supplies all publication/join fences.

    Independent cohorts advance each (leaf,slot) barrier once per generation.
    A single preparation team advances each leaf barrier once per iteration,
    regardless of its rotating frame slot. READY/EMPTY retain their old indices.
    These are shared-memory mbarriers, not the limited named-barrier ID space.
    """

    slots: int
    leaf_count: int
    cohorts: bool

    def __post_init__(self) -> None:
        if (
            type(self.slots) is not int
            or self.slots <= 0
            or type(self.leaf_count) is not int
            or self.leaf_count <= 0
            or type(self.cohorts) is not bool
        ):
            raise ValueError("invalid preparation leaf protocol")

    @property
    def barrier_count(self) -> int:
        return 2 * self.slots + self.leaf_count * (self.slots if self.cohorts else 1)

    @property
    def barrier_bytes(self) -> int:
        return 8 * self.barrier_count

    @property
    def allocated_bytes(self) -> int:
        return (self.barrier_bytes + 127) // 128 * 128

    def leaf_index(self, ordinal: int, slot: int = 0) -> int:
        if type(ordinal) is not int or not 0 <= ordinal < self.leaf_count:
            raise ValueError("leaf ordinal outside protocol")
        if type(slot) is not int or not 0 <= slot < self.slots:
            raise ValueError("slot outside protocol")
        return (
            2 * self.slots
            + ordinal * (self.slots if self.cohorts else 1)
            + (slot if self.cohorts else 0)
        )

    def leaf_pointer(self, ordinal: int) -> str:
        return f"chain_slot_bars + {self.leaf_index(ordinal)}" + (
            " + chain_slot" if self.cohorts else ""
        )

    @property
    def phase(self) -> str:
        return "chain_generation & 1" if self.cohorts else "chain_iteration & 1"
