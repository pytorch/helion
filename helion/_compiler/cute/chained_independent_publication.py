"""Defer independent load-only frame publications to their original READY join.

Only a completed original frontier emitter can surrender its trailing join.
The dependency proof uses its actual boundary inventory, and the physical proof
uses the final native frame. Optional bounded copy batching retains typed
enqueue/settlement evidence; internal joins and the final shared proxy
fence/role join remain mandatory.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import torch

from ... import exc
from .chained_async_groups import ASYNC_GROUP_LIMITS
from .chained_async_groups import AsyncPreparationGroups
from .chained_matmul import _shape
from .chained_pipeline_storage import capture_storage_revision
from .chained_preparation_cut import is_load_only_preparation
from .chained_preparation_reads import EnqueuedPreparationFrontier
from .chained_preparation_reads import PreparationReadFrontier
from .chained_preparation_reads import resolved_preparation_reads

if TYPE_CHECKING:
    from collections.abc import Mapping

    from torch.fx import Node

    from ..generate_ast import GenerateAST
    from .chained_async_groups import AsyncCopyAttempt
    from .chained_execution import ChainedExecution
    from .chained_matmul import ChainedMatmulPlan
    from .chained_pipeline_storage import PipelineStorage
    from .chained_preparation_frame import PreparationAction
    from .chained_preparation_pipeline import PreparationPipeline
    from .chained_preparation_reads import PreparationFrontierCompletion


class IndependentPublication:
    """One lexical preparation walk, after complete storage finalization."""

    def __init__(
        self,
        cg: GenerateAST,
        plan: ChainedMatmulPlan,
        pipeline: PreparationPipeline,
        execution: ChainedExecution,
        storage: PipelineStorage | None,
    ) -> None:
        frame = pipeline.frame
        if (
            storage is None
            or storage.preparation is not None
            or plan.loop_workspace is not storage.recurrence.layout
            or dict(storage.allocations).get("frames")
            != pipeline.slots * frame.layout.allocated_bytes
            or not is_load_only_preparation(frame.cut)
            or frame.stages
            or pipeline.cohorts is not None
            or pipeline.prepared_leaves
            or pipeline.operand_retention is not None
            or pipeline.scan_producer is not None
            or not frame.actions
            or frame.actions[-1].kind != "ready"
            or any(action.kind != "frontier" for action in frame.actions[:-1])
        ):
            raise exc.BackendUnsupported(
                "cute", "independent publication requires a finalized load-only frame"
            )
        writes = tuple(name for action in frame.actions[:-1] for name in action.writes)
        if (
            len(writes) < 2
            or len(set(writes)) != len(writes)
            or set(writes) != {buffer.name for buffer in frame.buffers}
            or any(action.reads for action in frame.actions[:-1])
            or any(buffer.kind != "frontier" for buffer in frame.buffers)
        ):
            raise exc.BackendUnsupported(
                "cute", "independent publication has dependent frame fields"
            )
        regions = tuple(frame.layout.region(name) for name in writes)
        if any(
            region.byte_offset < 0
            or region.byte_size <= 0
            or region.byte_end > frame.layout.allocated_bytes
            or any(region.overlaps_storage(other) for other in regions[:index])
            for index, region in enumerate(regions)
        ):
            raise exc.BackendUnsupported(
                "cute", "independent publication has overlapping frame fields"
            )
        self.cg, self.plan, self.pipeline, self.execution = (
            cg,
            plan,
            pipeline,
            execution,
        )
        self.shapes = {
            node: _shape(node)
            for node in frame.cut.region.nodes
            if isinstance(node.meta.get("val"), torch.Tensor)
        }
        revision = capture_storage_revision(plan, pipeline, self.shapes)
        if revision is None:
            raise exc.BackendUnsupported(
                "cute", "independent publication has no semantic revision"
            )
        self.revision = revision
        self.reads = PreparationReadFrontier(pipeline, self.shapes)
        self.cursor = 0
        self.removed = 0
        self.finished = False
        limit = cg.device_function.config.config.get(
            "cute_chained_async_group_limit", 1
        )
        if type(limit) is not int or limit not in ASYNC_GROUP_LIMITS:
            raise exc.BackendUnsupported(
                "cute", "invalid async preparation group limit"
            )
        self.async_groups = AsyncPreparationGroups(self, limit) if limit != 1 else None

    def begin_async(
        self, action: PreparationAction, before: Mapping[Node, str]
    ) -> AsyncCopyAttempt | None:
        return (
            None
            if self.async_groups is None
            else self.async_groups.begin(action, before)
        )

    def ready(self, action: PreparationAction) -> list[str]:
        return [] if self.async_groups is None else self.async_groups.ready(action)

    def accept(
        self,
        action: PreparationAction,
        stop: int,
        before: Mapping[Node, str],
        after: Mapping[Node, str],
        completion: PreparationFrontierCompletion | None,
        lines: list[str],
    ) -> list[str]:
        """Remove exactly the successful emitter's last synchronization line."""
        frame = self.pipeline.frame
        if (
            self.finished
            or action.event != self.cursor
            or not self.cursor < stop < len(frame.actions)
            or completion is None
        ):
            raise exc.BackendUnsupported(
                "cute", "independent publication lost an ordered completion"
            )
        consume = completion.consume
        if self.async_groups is not None and isinstance(
            completion.publication, EnqueuedPreparationFrontier
        ):
            consume = completion.consume_enqueued
        completed = consume(
            self.cg,
            self.plan,
            self.execution,
            tuple(before.items()),
            tuple(after.items()),
            tuple(lines),
        )
        expected = tuple(
            name for item in frame.actions[self.cursor : stop] for name in item.writes
        )
        if (
            tuple(buffer.name for buffer in completed.buffers) != expected
            or any(buffer not in frame.buffers for buffer in completed.buffers)
            or completed.inputs != completed.outputs
            or not lines
            or lines[-1] != self.execution.sync
        ):
            raise exc.BackendUnsupported(
                "cute", "independent publication changed its original writes"
            )
        roots = tuple(
            buffer.node for buffer in completed.buffers if buffer.node is not None
        )
        if len(roots) != len(completed.buffers) or resolved_preparation_reads(
            self.pipeline,
            roots,
            before,
            use_frontier=self.reads,
            published=tuple(before.items()),
        ):
            raise exc.BackendUnsupported(
                "cute", "independent publication reads a shared image"
            )
        self.cursor = stop
        self.removed += 1
        if self.async_groups is not None:
            self.async_groups.accept(completed)
        return lines[:-1]

    def finish(self, action: PreparationAction, lines: list[str]) -> None:
        expected = (
            (
                list(self.async_groups.ready_lines or ())
                if self.async_groups is not None
                else []
            )
            + (
                ["cute.arch.fence_view_async_shared()"]
                if self.pipeline.prepared_operands or self.pipeline.prepared_groups
                else []
            )
            + [self.execution.sync]
        )
        current = capture_storage_revision(self.plan, self.pipeline, self.shapes)
        # Original expression emission may discover new host tensor arguments.
        # It must preserve every existing alias and all typed graph facts.
        if (
            self.finished
            or action is not self.pipeline.frame.actions[-1]
            or action.event != self.cursor
            or not self.removed
            or lines != expected
            or current is None
            or current.aliases[: len(self.revision.aliases)] != self.revision.aliases
            or replace(current, aliases=self.revision.aliases) != self.revision
        ):
            raise exc.BackendUnsupported(
                "cute", "independent publication lost its final READY join"
            )
        if self.async_groups is not None:
            self.async_groups.finish()
        self.finished = True
