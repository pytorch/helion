from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

from .. import exc
from .loop_dependency_checker import TileAccess
from .loop_dependency_checker import TileAccessKind
from .loop_dependency_checker import TileDependencyAnalysis
from .loop_dependency_checker import TileDependencyKind
from .loop_dependency_checker import TileDependencySynchronization

if TYPE_CHECKING:
    from ..runtime.tile_dependency import TileDependencySchedule


@dataclasses.dataclass(frozen=True)
class TileDependencyStage:
    """A source-ordered set of roots that can execute without a grid wait."""

    index: int
    roots: tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class TileDependencyEdge:
    """One analyzed storage dependence between two opaque tile programs.

    Access maps are scheduling metadata.  They may be used to prove when a
    consumer tile can start, but never to reconstruct or simplify either tile's
    computation.
    """

    producer_stage: int
    consumer_stage: int
    producer_root: int
    consumer_root: int
    storage: str
    kinds: frozenset[TileDependencyKind]
    synchronization: TileDependencySynchronization
    producer_accesses: tuple[TileAccess, ...]
    consumer_accesses: tuple[TileAccess, ...]

    @property
    def is_unsynchronized(self) -> bool:
        return self.synchronization is TileDependencySynchronization.UNSYNCHRONIZED


@dataclasses.dataclass(frozen=True)
class ResolvedTileDependencySchedule:
    """Compiler-resolved stage graph for top-level tile dependencies."""

    policy: TileDependencySchedule | None
    stages: tuple[TileDependencyStage, ...]
    edges: tuple[TileDependencyEdge, ...]
    stage_by_root: tuple[int, ...]
    implicit_stage_starts: frozenset[int]
    uses_tile_dependency_counters: bool

    @property
    def stage_count(self) -> int:
        return len(self.stages)

    def stage_for_root(self, root: int) -> int:
        return self.stage_by_root[root]

    @property
    def unsynchronized_edges(self) -> tuple[TileDependencyEdge, ...]:
        return tuple(edge for edge in self.edges if edge.is_unsynchronized)

    def edges_between(
        self,
        producer_root: int,
        consumer_root: int,
        *,
        unsynchronized_only: bool = True,
    ) -> tuple[TileDependencyEdge, ...]:
        """Return dependence edges between two opaque tile programs."""
        return tuple(
            edge
            for edge in self.edges
            if edge.producer_root == producer_root
            and edge.consumer_root == consumer_root
            and (not unsynchronized_only or edge.is_unsynchronized)
        )

    @property
    def implicit_phase_starts(self) -> frozenset[int]:
        """Compatibility spelling for diagnostics emitted by older probes."""
        return self.implicit_stage_starts


def build_tile_dependency_stage_graph(
    analysis: TileDependencyAnalysis,
    policy: TileDependencySchedule | None = None,
) -> ResolvedTileDependencySchedule:
    """Build the source-level stage DAG used by TileDependency lowering.

    Stage placement is computed over roots in source order. This graph records
    ordering requirements and normalized access maps only. It does not authorize
    rewriting a root body or silently selecting a whole-grid fallback. Device
    lowering must prove that every unsynchronized edge is implemented by one of
    its tile-granular placement modes or reject the schedule.
    """
    if policy is not None:
        for dependency in analysis.unsynchronized_tile_dependencies:
            producer_accesses = analysis.accesses_by_root[dependency.producer_root]
            consumer_accesses = analysis.accesses_by_root[dependency.consumer_root]
            producer_kinds = {
                access.kind
                for access in producer_accesses
                if access.storage == dependency.name and access.index is not None
            }
            consumer_kinds = {
                access.kind
                for access in consumer_accesses
                if access.storage == dependency.name and access.index is not None
            }
            required_producer = {
                TileAccessKind.WRITE
                if kind
                in (
                    TileDependencyKind.READ_AFTER_WRITE,
                    TileDependencyKind.WRITE_AFTER_WRITE,
                )
                else TileAccessKind.READ
                for kind in dependency.kinds
            }
            required_consumer = {
                TileAccessKind.READ
                if kind is TileDependencyKind.READ_AFTER_WRITE
                else TileAccessKind.WRITE
                for kind in dependency.kinds
            }
            if not (
                required_producer <= producer_kinds
                and required_consumer <= consumer_kinds
            ):
                raise exc.TileDependencyScheduleError(
                    f"{dependency.producer_root}->{dependency.consumer_root} "
                    f"through storage {dependency.name!r} has an opaque access footprint"
                )

    dependencies_by_consumer = {
        root: [
            dependency
            for dependency in analysis.tile_dependencies
            if dependency.consumer_root == root
            and dependency.synchronization
            is TileDependencySynchronization.UNSYNCHRONIZED
        ]
        for root in range(analysis.root_count)
    }
    stage_by_root_list: list[int] = []
    implicit_stage_starts: set[int] = set()
    stage = 0

    for root in range(analysis.root_count):
        if root in analysis.source_phase_starts:
            stage += 1
        if any(
            stage_by_root_list[dependency.producer_root] == stage
            for dependency in dependencies_by_consumer[root]
        ):
            implicit_stage_starts.add(root)
            stage += 1
        stage_by_root_list.append(stage)

    stage_by_root = tuple(stage_by_root_list)
    stage_count = stage_by_root[-1] + 1 if stage_by_root else 0
    stages = tuple(
        TileDependencyStage(
            index=stage,
            roots=tuple(
                root
                for root, root_stage in enumerate(stage_by_root)
                if root_stage == stage
            ),
        )
        for stage in range(stage_count)
    )
    edges = tuple(
        TileDependencyEdge(
            producer_stage=stage_by_root[dependency.producer_root],
            consumer_stage=stage_by_root[dependency.consumer_root],
            producer_root=dependency.producer_root,
            consumer_root=dependency.consumer_root,
            storage=dependency.name,
            kinds=dependency.kinds,
            synchronization=dependency.synchronization,
            producer_accesses=tuple(
                access
                for access in analysis.accesses_by_root[dependency.producer_root]
                if access.storage == dependency.name
            ),
            consumer_accesses=tuple(
                access
                for access in analysis.accesses_by_root[dependency.consumer_root]
                if access.storage == dependency.name
            ),
        )
        for dependency in analysis.tile_dependencies
    )
    return ResolvedTileDependencySchedule(
        policy=policy,
        stages=stages,
        edges=edges,
        stage_by_root=stage_by_root,
        implicit_stage_starts=frozenset(implicit_stage_starts),
        uses_tile_dependency_counters=(
            bool(implicit_stage_starts) and not analysis.source_phase_starts
        ),
    )
