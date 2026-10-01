"""A completed grouped warp member used at its original pointwise-cache cut.

Shared publication and register availability are separate. The original frame
reservation is retained even when every read of a member uses its native image.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING

import torch

from .chained_fragment_epilogue import bind_fragment_expression
from .chained_operand_retention import _revision
from .physical_use_frontier import PhysicalPublication
from .physical_use_frontier import PhysicalReadPoint

if TYPE_CHECKING:
    from torch.fx import Node

    from ..generate_ast import GenerateAST
    from .chained_execution import ChainedExecution
    from .chained_matmul import ChainedMatmulPlan
    from .chained_operand_retention import _Revision
    from .chained_pointwise_residency import PointwiseCacheEntry
    from .chained_preparation_actions import AcceptedPreparation
    from .chained_preparation_frame import PreparationAction
    from .chained_preparation_frame import PreparationStage
    from .chained_preparation_pipeline import PreparationPipeline
    from .chained_preparation_reads import PreparationReadFrontier
    from .chained_tcgen_stage import StageGeometry
    from .chained_warp_stage import CompletedWarpStage


@dataclass
class _FragmentState:
    completion: CompletedWarpStage | None = None
    original_completion: CompletedWarpStage | None = None
    revision: object = None
    aliases: tuple[tuple[str, str], ...] = ()
    stage_first: int | None = None
    cache_first: int | None = None
    cache_lines: tuple[str, ...] | None = None
    cache_inputs: tuple[tuple[Node, str], ...] | None = None
    cache_outputs: tuple[tuple[Node, str], ...] | None = None
    shared_reads: tuple[str, ...] | None = None
    publication: PhysicalPublication | None = None
    original_publication: PhysicalPublication | None = None


@dataclass(frozen=True)
class PreparationFragment:
    cg: GenerateAST
    plan: ChainedMatmulPlan
    pipeline: PreparationPipeline
    execution: ChainedExecution
    stage: PreparationStage
    member: tuple[int, StageGeometry, int]
    entry: PointwiseCacheEntry
    source_event: int
    cache_event: int
    keep_shared: bool
    frontier: PreparationReadFrontier
    revision: _Revision
    _selection: object = field(repr=False)
    _state: _FragmentState = field(default_factory=_FragmentState, repr=False)

    @property
    def source(self) -> Node:
        return self.plan.dots[self.member[0]]

    @property
    def source_name(self) -> str:
        return f"chain_{self.member[0]}_c"

    def facts(self) -> object:
        from .chained_warp_stage import _snapshot

        return _snapshot(
            (
                self.cg,
                self.plan,
                self.pipeline,
                self.execution,
                self.stage,
                self.member,
                self.entry,
                self.source_event,
                self.cache_event,
                self.keep_shared,
                self.frontier,
                self.revision,
                id(self._state),
            )
        )

    def check(self) -> None:
        from . import chained_matmul as chain

        if (
            self._selection != self.facts()
            or self.pipeline.frame is not self.revision.frame
            or self.stage not in self.pipeline.frame.stages
            or self.frontier.pipeline is not self.pipeline
            or self.revision
            != _revision(self.plan, self.pipeline.frame, dict(self.revision.shapes))
        ):
            raise chain._UnsupportedChain("grouped fragment plan changed")

    def bind(self, publication: CompletedWarpStage) -> None:
        """Called inside the original warp emitter, before its successful return."""
        from . import chained_matmul as chain
        from .chained_warp_stage import WarpMemberResult
        from .chained_warp_stage import warp_revision

        self.check()
        prepared = publication.prepared
        result = prepared.result
        if (
            self._state.completion is not None
            or self._state.original_completion is not None
            or not isinstance(result, WarpMemberResult)
            or result.fragment is not self
            or self.member not in result.members
            or prepared.threads != prepared.completion.execution.threads
            or prepared.completion.execution.thread != self.execution.thread
            or prepared.completion.execution.sync != self.execution.sync
            or not publication.matches(
                self.cg, self.plan, dict(publication.boundaries), publication.prefix
            )
            or (self.source in dict(publication.boundaries)) != self.keep_shared
        ):
            raise chain._UnsupportedChain("grouped fragment lacks original completion")
        self._state.completion = self._state.original_completion = publication
        self._state.revision = warp_revision(self.plan, aliases=False)
        self._state.aliases = tuple(self.plan.tensor_aliases.items())
        self.frontier.frontier.register_owner(
            prepared, self.pipeline.frame, self.source_event
        )
        self._state.publication = self._state.original_publication = (
            PhysicalPublication(
                self.source,
                prepared,
                self.source.meta["val"].dtype,
                self.member[1].logical[:2],
            )
        )

    def completed(self) -> CompletedWarpStage:
        from . import chained_matmul as chain
        from .chained_warp_stage import _execution
        from .chained_warp_stage import _snapshot
        from .chained_warp_stage import warp_revision

        self.check()
        publication = self._state.completion
        if publication is None or publication is not self._state.original_completion:
            raise chain._UnsupportedChain("missing original grouped fragment")
        prepared = publication.prepared
        completion = prepared.completion
        image = self._state.publication
        if (
            publication._selection != publication.facts()
            or completion._state.publication is not publication
            or completion._state.prepared is not prepared
            or not completion._state.consumed
            or not completion._consumed
            or prepared._selection != prepared.facts()
            or completion._selection != completion.fields()
            or completion.codegen is not self.cg
            or completion.plan is not self.plan
            or completion._context != _execution(completion.execution)
            or completion._config != _snapshot(self.cg.device_function.config.config)
            or self._state.revision != warp_revision(self.plan, aliases=False)
            or tuple(self.plan.tensor_aliases.items())[: len(self._state.aliases)]
            != self._state.aliases
            or image is None
            or image is not self._state.original_publication
            or image.node is not self.source
            or image.owner is not prepared
            or image.dtype != self.source.meta["val"].dtype
            or image.shape != self.member[1].logical[:2]
        ):
            raise chain._UnsupportedChain("original grouped fragment changed")
        return publication

    def accept_stage(self, lines: list[str], body_first: int) -> None:
        from . import chained_matmul as chain

        publication = self.completed()
        if self._state.stage_first is not None or publication.prefix != tuple(lines):
            raise chain._UnsupportedChain("grouped stage return changed")
        self._state.stage_first = body_first

    def shared_outputs(self, outputs: dict[Node, str]) -> dict[Node, str]:
        self.completed()
        result = dict(outputs)
        if not self.keep_shared:
            result.pop(self.source, None)
        return result

    def emit_cache(
        self, action: PreparationAction, boundaries: dict[Node, str], body_first: int
    ) -> list[str]:
        from ..compile_environment import CompileEnvironment
        from . import chained_matmul as chain
        from .chained_tcgen_stage import _scalar_publication
        from .chained_warp_stage import WarpMemberResult

        publication = self.completed()
        state = self._state
        if (
            action is not self.pipeline.frame.actions[self.cache_event]
            or action.nodes != (self.entry.node,)
            or state.cache_lines is not None
            or state.stage_first is None
            or self.entry.node in boundaries
            or (self.source in boundaries) != self.keep_shared
        ):
            raise chain._UnsupportedChain("grouped cache publication cut changed")
        inputs = tuple(boundaries.items())
        publications = {}
        for node, name in inputs:
            value = self.frontier.publication(node, name)
            if value is None:
                raise chain._UnsupportedChain("grouped cache has unowned shared input")
            publications[node] = value
        assert state.publication is not None
        publications[self.source] = state.publication
        cache = self.plan.pointwise_cache
        assert cache is not None
        uses = self.frontier.frontier.resolve(
            (self.entry.node,),
            publications,
            external=frozenset(),
            required=frozenset(
                (*boundaries, *self.plan.dots, *(e.node for e in cache.entries))
            ),
            traversable=frozenset(self.pipeline.frame.cut.region.nodes),
            expand=self.entry.node,
        )
        point = PhysicalReadPoint(
            self.pipeline.frame, self.cache_event, self.cache_event + 1
        )
        self.frontier.frontier.read(state.publication.owner, point)
        names = {id(buffer): buffer.name for buffer in self.pipeline.frame.buffers}
        reads = tuple(
            sorted(
                {
                    names[id(value.owner)]
                    for value in uses
                    if value is not state.publication
                }
            )
        )
        if state.publication not in uses:
            raise chain._UnsupportedChain("grouped cache lost its native source read")
        result = publication.prepared.result
        assert isinstance(result, WarpMemberResult)
        prefix = f"{self.entry.name}_fragment"
        layout = _scalar_publication(prefix, (self.member,))
        row, column, predicate = layout.coordinates(layout.targets[0])
        coords = (row, column)
        expression = bind_fragment_expression(
            self.cg,
            self.plan,
            boundaries,
            [],
            self.source,
            self.entry.node,
            coords,
            coords,
            f"{result.prefix}_values[{layout.index}]",
            CompileEnvironment.current().backend.dtype_str(self.entry.dtype),
            operand_domain=False,
        )
        lines = [
            f"for {layout.index} in cutlass.range_constexpr(cute.size({result.prefix}_values)):",
            f"    {layout.row}, {layout.column} = {result.prefix}_coords[{layout.index}]",
            f"    if {predicate}:",
            chain._indent(expression.expression.lines, 8),
            f"        {self.entry.name}[{row}, {column}] = {expression.dtype}({expression.value})",
            self.execution.sync,
        ]
        expression.check()
        boundaries[self.entry.node] = self.entry.name
        state.cache_first = body_first
        state.cache_lines = tuple(lines)
        state.cache_inputs = inputs
        state.cache_outputs = tuple(boundaries.items())
        state.shared_reads = reads
        return lines

    def accepted(self, accepted: AcceptedPreparation) -> bool:
        publication = self.completed()
        state = self._state
        source = tuple(
            a for a in accepted.actions if a.first <= self.source_event < a.stop
        )
        target = tuple(a for a in accepted.actions if a.first == self.cache_event)
        return (
            accepted.pipeline is self.pipeline
            and len(source) == len(target) == 1
            and source[0].fragment is self
            and target[0].fragment is self
            and (self.source in dict(source[0].outputs)) == self.keep_shared
            and (self.source_name in source[0].writes) == self.keep_shared
            and self.source_name not in source[0].omitted
            and target[0].inputs == state.cache_inputs
            and target[0].outputs == state.cache_outputs
            and target[0].reads == state.shared_reads
            and target[0].writes == (self.entry.name,)
            and state.stage_first is not None
            and state.cache_first is not None
            and state.cache_lines is not None
            and accepted.lines[
                state.stage_first : state.stage_first + len(publication.prefix)
            ]
            == publication.prefix
            and accepted.lines[
                state.cache_first : state.cache_first + len(state.cache_lines)
            ]
            == state.cache_lines
        )


def plan_preparation_fragment(
    cg: GenerateAST,
    plan: ChainedMatmulPlan,
    pipeline: PreparationPipeline,
    execution: ChainedExecution,
    frontier: PreparationReadFrontier,
) -> PreparationFragment | None:
    """Select a real same-role grouped result/cache relation, never a formula."""
    cache, scan = plan.pointwise_cache, pipeline.scan_producer
    if cache is None or scan is None:
        return None
    stage = scan.stage
    if len(stage.group.stages) < 2:
        return None
    frame = pipeline.frame
    source_event = scan.stop_event
    for entry in cache.entries:
        sources = tuple(
            index
            for index in stage.group.stages
            if plan.dots[index] in entry.dependencies
        )
        if len(sources) != 1 or len(entry.shape) != 2:
            continue
        ordinal = stage.group.stages.index(sources[0])
        geometry = stage.group.geometries[ordinal]
        if (
            geometry.logical[:2] != entry.shape
            or plan.dots[sources[0]].meta["val"].dtype != torch.float32
        ):
            continue
        target = tuple(
            a for a in frame.actions if a.kind == "cache" and a.nodes == (entry.node,)
        )
        if len(target) != 1 or target[0].event <= source_event:
            continue
        if any(
            a.kind not in ("cache", "collective", "leaf")
            for a in frame.actions[source_event + 1 : target[0].event]
        ):
            continue
        name = f"chain_{sources[0]}_c"
        keep = any(name in a.reads and a is not target[0] for a in frame.actions)
        offset = sum(g.physical[1] for g in stage.group.geometries[:ordinal])
        result = PreparationFragment(
            cg,
            plan,
            pipeline,
            execution,
            stage,
            (sources[0], geometry, offset),
            entry,
            source_event,
            target[0].event,
            keep,
            frontier,
            _revision(plan, frame, frontier.shapes),
            None,
        )
        object.__setattr__(result, "_selection", result.facts())
        return result
    return None
