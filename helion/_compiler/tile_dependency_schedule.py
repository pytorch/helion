from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

from .. import exc

if TYPE_CHECKING:
    from ..runtime.tile_dependency import TileDependencySchedule
    from .cross_loop_dependencies import CrossLoopDependencyPlan


@dataclasses.dataclass(frozen=True)
class TileDependencyStage:
    """A source-ordered set of roots that can execute without a grid wait."""

    index: int
    roots: tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class ResolvedTileDependencySchedule:
    """Source phase layout plus the requested cross-loop scheduling policy.

    Dependency edges are intentionally absent. DeviceIR's allocation graph is
    the sole dependency representation consumed by scheduler lowering.
    """

    policy: TileDependencySchedule | None
    stages: tuple[TileDependencyStage, ...]
    stage_by_root: tuple[int, ...]
    implicit_stage_starts: frozenset[int]
    uses_tile_dependency_counters: bool

    @property
    def stage_count(self) -> int:
        return len(self.stages)

    def stage_for_root(self, root: int) -> int:
        return self.stage_by_root[root]

    @property
    def implicit_phase_starts(self) -> frozenset[int]:
        """Compatibility spelling for diagnostics emitted by older probes."""
        return self.implicit_stage_starts


def resolve_tile_dependency_schedule(
    dependency_plan: CrossLoopDependencyPlan,
    policy: TileDependencySchedule | None = None,
    *,
    source_phase_starts: frozenset[int] = frozenset(),
) -> ResolvedTileDependencySchedule:
    """Resolve source phases and DeviceIR hazards into execution stages."""
    edges = dependency_plan.edges
    if edges and policy is None:
        names = sorted(edges[0].tensor_names)
        raise exc.LoopDependencyError(names[0] if names else "tensor allocation")
    if edges and source_phase_starts:
        raise exc.TileDependencyScheduleError(
            "mixing explicit hl.barrier() phases with implicit "
            "tile-dependency stages is not supported yet"
        )

    dependencies_by_consumer = {
        family.root: [edge for edge in edges if edge.consumer_root == family.root]
        for family in dependency_plan.task_families
    }
    stage_by_root_list: list[int] = []
    implicit_stage_starts: set[int] = set()
    stage = 0

    for root in range(len(dependency_plan.task_families)):
        if root in source_phase_starts:
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
    return ResolvedTileDependencySchedule(
        policy=policy,
        stages=stages,
        stage_by_root=stage_by_root,
        implicit_stage_starts=frozenset(implicit_stage_starts),
        uses_tile_dependency_counters=(
            bool(implicit_stage_starts) and not source_phase_starts
        ),
    )
