"""Separate rectangular input issues from completion, without moving arithmetic.

The renderer is the intended eventual delegate of PreparationLeaf.emit. Its two
parts concatenate to the old emitter exactly. Neither rendered strings nor this
pure schedule authorize allocation/body installation: a future caller must
record actual pending/completed transitions and finalize the extended leases.
In particular, AcceptedPreparation.lines still describes the OLD synchronous
body and must never serve as a receipt for a scheduled body.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from itertools import pairwise
import math
from typing import TYPE_CHECKING
from typing import Literal
from typing import cast

from .chained_leaf_sets import LeafSetProtocol
from .chained_preparation_reads import resolved_preparation_reads
from .chained_scan_producer import ScanProducer

if TYPE_CHECKING:
    from torch.fx import Node

    from ..generate_ast import GenerateAST
    from .chained_execution import ChainedExecution
    from .chained_matmul import ChainedMatmulPlan
    from .chained_preparation_leaves import PreparationLeaf
    from .chained_preparation_pipeline import PreparationPipeline
    from .chained_preparation_storage import AcceptedPreparationStorage
    from .warp_specialized_plan import SharedBufferRegion


@dataclass(frozen=True)
class LeafTransferEmission:
    """Original issue/fallback and wait/join; not completion authority."""

    issue: tuple[str, ...]
    completion: tuple[str, ...]

    def lines(self) -> list[str]:
        return [*self.issue, *self.completion]


def emit_leaf_transfer(
    leaf: PreparationLeaf,
    cg: GenerateAST,
    plan: ChainedMatmulPlan,
    boundaries: dict[Node, str],
    execution: ChainedExecution,
    barrier: str,
    phase: str,
) -> LeafTransferEmission:
    """Use the existing typed expression/descriptor once, with no new proof.

    Both branches advance the same original barrier. The full branch retains
    its proxy fence and all-writer rendezvous; the fallback retains its original
    expression, masks, stores and arrival. Only completion publishes a boundary.
    """
    from .chained_matmul import _Expression
    from .chained_matmul import _indent

    prefix, proof = leaf.name, leaf.proof
    row, column = proof.origin
    atom, tensor = cast("list[str]", leaf.wrapper["kernel_args"])
    offset = f"{prefix}_offset"
    coords = (
        f"{offset} // {proof.tile_shape[1]}",
        f"{offset} % {proof.tile_shape[1]}",
    )
    expression = _Expression(cg, plan, boundaries)
    expression.coordinate_names.add(offset)
    value = expression.value(leaf.node, coords)
    fallback = [
        f"for {offset} in cutlass.range({execution.thread}, {math.prod(proof.tile_shape)}, {execution.threads}, unroll=1):",
        _indent(
            [*expression.lines, f"{prefix}[{coords[0]}, {coords[1]}] = {value}"], 4
        ),
        execution.sync,
        f"if {execution.thread} == 0:",
        f"    cute.arch.mbarrier_arrive({barrier})",
    ]
    return LeafTransferEmission(
        (
            f"if {proof.guard}:",
            "    cute.arch.fence_view_async_shared()",
            f"    {execution.sync}",
            f"    if {execution.thread} == 0:",
            f"        cute.arch.mbarrier_arrive_and_expect_tx({barrier}, {leaf.byte_size})",
            f"    if {execution.warp} == 0:",
            f"        {prefix}_origin = cute.domain_offset((cutlass.Int32({row}), cutlass.Int32({column})), {tensor})",
            f"        {prefix}_source = cute.local_tile({prefix}_origin, {proof.tile_shape!r}, (0, 0))",
            f"        {prefix}_shared, {prefix}_global = cute.nvgpu.cpasync.tma_partition({atom}, 0, cute.make_layout(1), cute.group_modes({prefix}, 0, 2), cute.group_modes({prefix}_source, 0, 2))",
            f"        cute.copy({atom}, {prefix}_global, {prefix}_shared, tma_bar_ptr={barrier})",
            "else:",
            _indent(fallback, 4),
        ),
        (f"cute.arch.mbarrier_wait({barrier}, {phase})", execution.sync),
    )


@dataclass(frozen=True)
class LeafScheduleAction:
    """An original accepted action or a checked scan-prelude subaction.

    ``phase`` uses the physical table's existing time scale. A scan body is
    opaque after its prelude; completing a leaf before it is conservative.
    """

    accepted_index: int
    event: int
    phase: int
    kind: Literal["leaf", "action", "scan_body"]
    reads: tuple[str, ...]
    writes: tuple[str, ...]


@dataclass(frozen=True)
class ScheduledLeaf:
    leaf: PreparationLeaf
    ordinal: int
    original_action: int
    issue_before: int
    complete_before: int
    owner: str


@dataclass(frozen=True)
class LeafScheduleStep:
    kind: Literal["issue", "complete", "action"]
    index: int


@dataclass(frozen=True)
class LeafSchedule:
    """Same-revision schedule proposal, with no permission to reuse old lines."""

    physical: AcceptedPreparationStorage
    actions: tuple[LeafScheduleAction, ...]
    leaves: tuple[ScheduledLeaf, ...]
    steps: tuple[LeafScheduleStep, ...]
    regions: tuple[SharedBufferRegion, ...]
    protocol: LeafSetProtocol
    max_pending: int
    _selection: tuple[object, ...]

    def _fields(self) -> tuple[object, ...]:
        return (
            self.physical,
            self.actions,
            self.leaves,
            self.steps,
            self.regions,
            self.protocol,
            self.max_pending,
        )

    def matches(self, plan: ChainedMatmulPlan, pipeline: PreparationPipeline) -> bool:
        if self._selection != self._fields():
            return False
        fresh = plan_leaf_schedule(plan, pipeline, self.physical)
        return fresh is not None and fresh._fields() == self._fields()


def _actions(
    plan: ChainedMatmulPlan,
    pipeline: PreparationPipeline,
    physical: AcceptedPreparationStorage,
) -> tuple[LeafScheduleAction, ...] | None:
    frame = pipeline.frame
    scans = tuple(
        action.proof for action in physical.accepted.actions if action.kind == "scan"
    )
    if len(scans) > 1 or (scans and not isinstance(scans[0], ScanProducer)):
        return None
    scan = cast("ScanProducer | None", scans[0] if scans else None)
    phases = scan.phases if scan is not None else tuple(range(len(frame.actions)))
    result = []
    for index, accepted in enumerate(physical.accepted.actions):
        original = frame.actions[accepted.first : accepted.stop]
        if accepted.kind != "scan":
            leaves = [action for action in original if action.kind == "leaf"]
            if leaves and (len(original) != 1 or accepted.kind != "ordinary"):
                return None
            result.append(
                LeafScheduleAction(
                    index,
                    accepted.first,
                    phases[accepted.first],
                    "leaf" if leaves else "action",
                    accepted.reads,
                    accepted.writes,
                )
            )
            continue
        if (
            scan is None
            or not isinstance(accepted.proof, ScanProducer)
            or accepted.proof is not scan
            or scan is not pipeline.scan_producer
            or not scan.matches(plan, pipeline, dict(physical.accepted.revision.shapes))
        ):
            return None
        boundaries = dict(accepted.inputs)
        published = accepted.inputs
        prelude_writes = set()
        for action in scan.prelude:
            if action.kind not in ("leaf", "collective"):
                return None
            reads = resolved_preparation_reads(
                pipeline,
                action.nodes,
                boundaries,
                use_frontier=physical.accepted.read_frontier,
                published=published,
            )
            result.append(
                LeafScheduleAction(
                    index,
                    action.event,
                    phases[action.event],
                    "leaf" if action.kind == "leaf" else "action",
                    reads,
                    action.writes,
                )
            )
            if len(action.nodes) != len(action.writes):
                return None
            boundaries.update(zip(action.nodes, action.writes, strict=True))
            published = tuple(
                {
                    **dict(published),
                    **dict(zip(action.nodes, action.writes, strict=True)),
                }.items()
            )
            prelude_writes.update(action.writes)
        roots = (
            scan.scan.node,
            *(
                source
                for dot in scan.stage.group.stages
                for source in plan.dots[dot].all_input_nodes
            ),
            *(node for action in scan.deferred for node in action.nodes),
        )
        # Original dependency closure stops at the actually published prelude
        # values. A pending leaf is completed before entering this opaque body.
        reads = resolved_preparation_reads(
            pipeline,
            roots,
            boundaries,
            use_frontier=physical.accepted.read_frontier,
            published=published,
        )
        result.append(
            LeafScheduleAction(
                index,
                scan.first_event,
                phases[scan.first_event],
                "scan_body",
                reads,
                tuple(name for name in accepted.writes if name not in prelude_writes),
            )
        )
    if any(left.phase > right.phase for left, right in pairwise(result)):
        return None
    return tuple(result)


def _schedule(
    actions: tuple[LeafScheduleAction, ...],
    leaves: tuple[tuple[PreparationLeaf, int, str], ...],
    regions: tuple[SharedBufferRegion, ...],
) -> (
    tuple[
        tuple[ScheduledLeaf, ...],
        tuple[LeafScheduleStep, ...],
        tuple[SharedBufferRegion, ...],
        int,
    ]
    | None
):
    """Extend starts only; preserve offsets, sizes, all ends and arithmetic order."""
    if len({region.name for region in regions}) != len(regions):
        return None
    if (
        not actions
        or len({leaf.name for leaf, _, _ in leaves}) != len(leaves)
        or len({ordinal for _, ordinal, _ in leaves}) != len(leaves)
        or len({owner for _, _, owner in leaves}) != len(leaves)
        or sorted(action.writes for action in actions if action.kind == "leaf")
        != sorted((leaf.name,) for leaf, _, _ in leaves)
        or any(type(action.phase) is not int or action.phase < 0 for action in actions)
        or any(left.phase > right.phase for left, right in pairwise(actions))
        or any(
            left.overlaps_storage(right) and left.overlaps_lifetime(right)
            for index, left in enumerate(regions)
            for right in regions[index + 1 :]
        )
    ):
        return None
    changed = {region.name: region for region in regions}
    selected = []
    for leaf, ordinal, owner in leaves:
        original = [
            i
            for i, action in enumerate(actions)
            if action.kind == "leaf" and action.writes == (leaf.name,)
        ]
        readers = [i for i, action in enumerate(actions) if leaf.name in action.reads]
        if (
            len(original) != 1
            or actions[original[0]].reads
            or not readers
            or min(readers) <= original[0]
        ):
            return None
        first, last = min(readers), max(readers)
        if owner not in changed:
            return None
        region = changed[owner]
        if actions[last].phase >= region.live_until:
            return None
        for issue in range(original[0] + 1):
            expanded = replace(
                region, live_from=min(region.live_from, actions[issue].phase)
            )
            if any(
                other.name != owner
                and expanded.overlaps_storage(other)
                and expanded.overlaps_lifetime(other)
                for other in changed.values()
            ):
                continue
            changed[owner] = expanded
            selected.append(
                ScheduledLeaf(leaf, ordinal, original[0], issue, first, owner)
            )
            break
        else:
            return None
    steps = []
    pending = set()
    maximum = 0
    for index, action in enumerate(actions):
        for selected_index, leaf in enumerate(selected):
            if leaf.issue_before == index:
                steps.append(LeafScheduleStep("issue", selected_index))
                pending.add(selected_index)
        maximum = max(maximum, len(pending))
        for selected_index, leaf in enumerate(selected):
            if leaf.complete_before == index:
                if selected_index not in pending:
                    return None
                steps.append(LeafScheduleStep("complete", selected_index))
                pending.remove(selected_index)
        if action.kind != "leaf":
            steps.append(LeafScheduleStep("action", index))
    if pending or maximum < 2:
        return None
    return (
        tuple(selected),
        tuple(steps),
        tuple(changed[r.name] for r in regions),
        maximum,
    )


def plan_leaf_schedule(
    plan: ChainedMatmulPlan,
    pipeline: PreparationPipeline,
    physical: AcceptedPreparationStorage,
) -> LeafSchedule | None:
    """Propose batching only within a caller-owned, acquired whole frame.

    This reuses the accepted physical table's whole native owners and time
    scale. Overlapping bytes are legal only with disjoint EXTENDED lifetimes.
    No future phase may use the original, shorter async-write reservation.
    The caller must preserve EMPTY/acquire, READY/release and endpoint joins.
    """
    shapes = dict(physical.accepted.revision.shapes)
    threads = (
        pipeline.preparation_threads
        if pipeline.cohorts is None
        else pipeline.cohorts.cohort_threads
    )
    if (
        not pipeline.prepared_leaves
        or not physical.matches(plan, pipeline, shapes)
        or physical.accepted.execution.threads != threads
    ):
        return None
    try:
        actions = _actions(plan, pipeline, physical)
    except ValueError:
        return None
    if actions is None:
        return None
    views = {view.original.semantic.name: view for view in physical.views}
    regions = {region.name: region for region in physical.layout.regions}
    leaves = []
    for ordinal, leaf in enumerate(pipeline.prepared_leaves):
        view = views.get(leaf.name)
        if view is None:
            return None
        if view.owner not in regions:
            return None
        region = regions[view.owner]
        if (
            view.original.semantic.node is not leaf.node
            or view.original.stored_node is not leaf.node
            or view.original.shape != leaf.proof.tile_shape
            or view.original.dtype.itemsize != leaf.proof.element_bytes
            or view.crop is not None
            or view.original.native_group is not None
            or view.byte_offset != region.byte_offset
            or view.declared_bytes != leaf.byte_size
            or view.accesses != ((region.byte_offset, leaf.byte_size),)
            or region.byte_size < leaf.byte_size
            or sum(other.owner == view.owner for other in physical.views) != 1
        ):
            return None
        # _actions uses the original published boundary set, not all eventual
        # frontier nodes. _schedule rejects any real materialized leaf input;
        # the existing rectangle proof owns immutable host origins and masks.
        leaves.append((leaf, ordinal, view.owner))
    positions = {
        action.writes[0]: index
        for index, action in enumerate(actions)
        if action.kind == "leaf" and len(action.writes) == 1
    }
    if any(leaf.name not in positions for leaf, _, _ in leaves):
        return None
    leaves.sort(key=lambda item: positions[item[0].name])
    scheduled = _schedule(actions, tuple(leaves), physical.layout.regions)
    if scheduled is None:
        return None
    selected, steps, updated, maximum = scheduled
    protocol = LeafSetProtocol(
        pipeline.slots, len(pipeline.prepared_leaves), pipeline.cohorts is not None
    )
    fields = (physical, actions, selected, steps, updated, protocol, maximum)
    return LeafSchedule(*fields, fields)
