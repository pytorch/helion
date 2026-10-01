"""Coordinate-owned CSE across adjacent, independently stored frontier images.

The frame remains unchanged. A group carries an additional early-write proof:
extending each destination's lifetime to the first action must not overlap any
other live allocation. In particular, a later image cannot overwrite scratch
that an earlier image still reads, even if ordinary sequential publication is
safe. One full-role barrier completes all outputs after the grouped publication.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
import math
from typing import TYPE_CHECKING

import torch

from ... import exc
from .chained_matmul import _indent
from .chained_native_stores import NativeStMatrixStore
from .chained_native_stores import plan_native_stmatrix_store
from .chained_vector_group import VectorGroupOutput
from .chained_vector_group import emit_materialized_group
from .chained_vector_group import emit_vector_group
from .chained_vector_ownership import plan_vector_ownership

if TYPE_CHECKING:
    from torch.fx import Node

    from ..generate_ast import GenerateAST
    from .chained_broadcast_retention import BroadcastOptions
    from .chained_execution import ChainedExecution
    from .chained_frontier_materialization import FrontierMaterializationAttempt
    from .chained_frontier_ownership import FrontierOwnership
    from .chained_matmul import ChainedMatmulPlan
    from .chained_pointwise_unroll import BoundedProducerUnroll
    from .chained_preparation_frame import PreparationBuffer
    from .chained_preparation_frame import PreparationFrame
    from .chained_preparation_reads import PreparationFrontierCompletion
    from .chained_prepared_groups import PreparedGroupBinding
    from .chained_prepared_operands import PreparedOperand
    from .chained_vector_native import NativeReadInputs
    from .chained_vector_ownership import VectorOwnership
    from .chained_vector_stage import VectorStaging


@dataclass(frozen=True)
class FrontierGroup:
    first_event: int
    stop_event: int
    buffers: tuple[PreparationBuffer, ...]


@dataclass
class FrontierStores:
    """Native stores for final accepted frontier images, not new producers."""

    enabled: bool
    activated: bool = False

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise ValueError("frontier StMatrix must be bool")

    def select(
        self,
        frame: PreparationFrame,
        group: FrontierGroup,
        prepared_groups: tuple[PreparedGroupBinding, ...],
        ownership: VectorOwnership,
    ) -> dict[str, NativeStMatrixStore]:
        # These are final accepted native groups; reuse their physical/typed
        # proof rather than interpreting scalar target names as native layouts.
        from .chained_frontier_ownership import _group_matches
        from .chained_frontier_ownership import _typed_buffer
        from .chained_prepared_groups import _valid_frame

        if (
            not self.enabled
            or not _valid_frame(frame)
            or frame.cut.region.nodes != tuple(frame.cut.region.graph.nodes)
            or any(not _typed_buffer(frame, buffer) for buffer in group.buffers)
            or plan_frontier_group(frame, group.first_event) != group
        ):
            return {}
        selected = {}
        for binding in prepared_groups:
            if not any(
                member.buffer in group.buffers and member.logical_modes == (1, 0)
                for member in binding.candidate.members
            ):
                continue
            if not _group_matches(frame, binding):
                return {}
            for member in binding.candidate.members:
                if member.buffer not in group.buffers or member.logical_modes != (1, 0):
                    continue
                geometry = plan_native_stmatrix_store(
                    binding.candidate.physical_shape,
                    (member.buffer.shape[0], member.buffer.shape[1]),
                    member.row_offset,
                    member.buffer.dtype,
                    ownership,
                )
                if geometry is not None:
                    if member.buffer.name in selected:
                        return {}
                    selected[member.buffer.name] = geometry
        return selected

    def validate(self) -> None:
        if self.enabled and not self.activated:
            raise exc.BackendUnsupported(
                "cute", "frontier StMatrix requires a successfully emitted native store"
            )


def plan_frontier_group(
    frame: PreparationFrame, first_event: int
) -> FrontierGroup | None:
    """Return the longest safe adjacent prefix, without changing placement.

    Geometry and byte lifetimes are necessary, not sufficient: the emitter
    separately proves exact operand domains, coordinate ownership and useful
    shared original-node work. Singletons keep their ordinary producer.
    """
    buffers = {buffer.name: buffer for buffer in frame.buffers}
    regions = {region.name: region for region in frame.layout.regions}
    writes = {name: action.event for action in frame.actions for name in action.writes}
    selected = []
    shape = None
    for action in frame.actions:
        if action.event < first_event:
            continue
        if action.event != first_event + len(selected) or action.kind != "frontier":
            break
        assert len(action.writes) == 1
        buffer = buffers[action.writes[0]]
        assert buffer.kind == "frontier" and buffer.node is not None
        if (
            len(buffer.shape) != 2
            or any(extent <= 0 for extent in buffer.shape)
            or buffer.dtype not in (torch.bfloat16, torch.float16, torch.float32)
            or shape is not None
            and buffer.shape != shape
        ):
            break
        region = regions[buffer.name]
        # Retention planning may already reserve an earlier grouped write.
        # Never shorten that reservation, or accept storage allocated later
        # than the original publication action.
        if region.live_from > action.event:
            break
        # Frame requests include alignment padding; check the whole allocated
        # interval for conflicts, not just the logical tensor's payload.
        assert region.byte_size >= math.prod(buffer.shape) * buffer.dtype.itemsize
        # All grouped reads happen during the first action. Ordinary frontier
        # publication does not install new preparation-side boundaries.
        if any(
            not regions[name].live_from < first_event < regions[name].live_until
            or name in writes
            and writes[name] >= first_event
            for name in action.reads
        ):
            break
        early = replace(region, live_from=min(region.live_from, first_event))
        if any(
            other.name != early.name
            and early.overlaps_storage(other)
            and early.overlaps_lifetime(other)
            for other in frame.layout.regions
        ):
            break
        shape = buffer.shape
        selected.append(buffer)
    if len(selected) < 2:
        return None
    return FrontierGroup(first_event, first_event + len(selected), tuple(selected))


def emit_frontier_group(
    cg: GenerateAST,
    plan: ChainedMatmulPlan,
    frame: PreparationFrame,
    first_event: int,
    boundaries: dict[Node, str],
    execution: ChainedExecution,
    vector: VectorStaging,
    unroll: BoundedProducerUnroll,
    *,
    scalar_targets: set[str],
    materialized: bool = False,
    ownership: FrontierOwnership | None = None,
    prepared_operands: tuple[PreparedOperand, ...] = (),
    prepared_groups: tuple[PreparedGroupBinding, ...] = (),
    native_inputs: NativeReadInputs | None = None,
    stores: FrontierStores | None = None,
    materializations: FrontierMaterializationAttempt | None = None,
    completion: PreparationFrontierCompletion | None = None,
) -> tuple[list[str], int] | None:
    """Use original expression lowering and native/scalar target contracts."""
    if not vector.enabled or not vector.group_enabled:
        return None
    group = plan_frontier_group(frame, first_event)
    if group is None:
        return None
    inputs = tuple(boundaries.items()) if completion is not None else ()
    if materializations is not None:
        selected = materializations.selection
        if (
            materializations.receipt is not None
            or not selected.matches()
            or selected.revision.plan is not plan
            or selected.pipeline.frame is not frame
            or selected.group != group
            or dict(selected.boundaries) != boundaries
            or not materialized
        ):
            return None
        ordinals = selected.emitted_ordinals
        if (
            any(
                type(index) is not int or not 0 <= index < len(group.buffers)
                for index in ordinals
            )
            or tuple(sorted(set(ordinals))) != ordinals
        ):
            return None
    else:
        ordinals = tuple(range(len(group.buffers)))
    # A physical-copy target needs a power-of-two thread tile, just like its
    # ordinary single-output producer. Scalar targets can share those owners;
    # all role participants still reach the publication barrier.
    threads = execution.threads
    if any(buffer.name not in scalar_targets for buffer in group.buffers):
        threads = 1 << (threads.bit_length() - 1)
    outputs = []
    for ordinal in ordinals:
        buffer = group.buffers[ordinal]
        assert buffer.node is not None
        outputs.append(
            VectorGroupOutput(
                buffer.node,
                buffer.name,
                lambda row, column: (row, column),
                vector_store=buffer.name not in scalar_targets,
                ordinal=ordinal if materializations is not None else None,
            )
        )
    shape = group.buffers[0].shape
    selected_ownership = (
        ownership.select(frame, group, prepared_operands, prepared_groups, threads)
        if ownership is not None
        else None
    )
    selected_stores = {}
    if stores is not None and stores.enabled:
        vector_ownership = selected_ownership or plan_vector_ownership(
            (shape[0], shape[1]), threads
        )
        if vector_ownership is not None:
            selected_stores = stores.select(
                frame, group, prepared_groups, vector_ownership
            )
            if materializations is not None:
                targets = {output.target for output in outputs}
                selected_stores = {
                    name: store
                    for name, store in selected_stores.items()
                    if name in targets
                }
            outputs = [
                replace(output, native_store=selected_stores.get(output.target))
                for output in outputs
            ]
    emit = emit_materialized_group if materialized else emit_vector_group
    local_inputs = (
        replace(native_inputs, activated=False, emitted=())
        if materializations is not None and native_inputs is not None
        else native_inputs
    )
    local_unroll = replace(unroll) if materializations is not None else unroll
    options: BroadcastOptions = {}
    if vector.broadcast is not None:
        options["broadcast"] = vector.broadcast
    broadcast_first = 0 if vector.broadcast is None else vector.broadcast.checkpoint()
    lines = emit(
        cg,
        plan,
        boundaries,
        outputs,
        shape=(shape[0], shape[1]),
        tag=f"{group.buffers[0].name}_group",
        producer_unroll=local_unroll,
        execution=replace(execution, threads=threads),
        ownership=selected_ownership,
        native_inputs=local_inputs,
        **options,
    )
    if lines is None:
        return None
    broadcast_placement = (
        None
        if vector.broadcast is None
        else vector.broadcast.place(
            broadcast_first,
            lines,
            1 if threads != execution.threads else 0,
            indent=4 if threads != execution.threads else 0,
        )
    )
    if materializations is not None:
        effective_ownership = selected_ownership or plan_vector_ownership(
            (shape[0], shape[1]), threads
        )
        assert effective_ownership is not None
        materializations.complete(
            effective_ownership,
            local_inputs.emitted if local_inputs is not None else (),
            tuple(
                (index, selected_stores[group.buffers[index].name])
                for index in ordinals
                if group.buffers[index].name in selected_stores
            ),
        )
        unroll.activated |= local_unroll.activated
        unroll.eliminated |= local_unroll.eliminated
        if native_inputs is not None and local_inputs is not None:
            native_inputs.activated |= local_inputs.activated
            native_inputs.emitted = local_inputs.emitted
    if threads != execution.threads:
        lines = [f"if {execution.thread} < {threads}:", _indent(lines, 4)]
    if materializations is not None:
        for _, crop in materializations.selection.aliases:
            full = f"chain_{crop.source.group.stages[0]}_{crop.source.role}"
            lines.extend(crop.alias_lines(full, crop.buffer.name))
    if not materialized:
        vector.activated = vector.group_activated = True
    if ownership is not None and selected_ownership is not None:
        ownership.activated = True
    if stores is not None and selected_stores:
        stores.activated = True
    # Every output copy and alias is complete before this full-role join.
    # Their logical publications share one synchronization point.
    lines = [*lines, execution.sync]
    if completion is not None:
        completion.record(cg, plan, group.buffers, execution, inputs, boundaries, lines)
    if vector.broadcast is not None:
        vector.broadcast.enclose(broadcast_first, lines, (broadcast_placement,))
    return lines, group.stop_event
