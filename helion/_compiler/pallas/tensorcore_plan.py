"""TensorCore plans for Pallas memory operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .memory_access import MemoryAccessKind
from .plan_tiling import TensorIndexPattern

if TYPE_CHECKING:
    from ...runtime.config import Config
    from ..device_ir import GraphInfo
    from .gather import GatherPlan
    from .gather import ScatterPlan
    from .memory_access import MemoryAccess


TENSORCORE_PLAN_META = "pallas_tensorcore_plan"


@dataclass(frozen=True)
class TensorCorePlan:
    """TensorCore implementation of one memory operation."""

    access: MemoryAccess
    indirect_positions: tuple[int, ...]  # Positions in access.subscript.


@dataclass(frozen=True)
class OneHotGatherPlan(TensorCorePlan):
    """Resident-table one-hot/MXU fallback for an indirect load."""

    plan: GatherPlan


@dataclass(frozen=True)
class OneHotScatterPlan(TensorCorePlan):
    """One-hot fallback for an indirect store."""

    plan: ScatterPlan


def _one_hot_gather(
    access: MemoryAccess, positions: list[int], config: Config
) -> OneHotGatherPlan:
    from .gather import build_gather_plan

    plan = build_gather_plan(
        access.tensor,
        list(access.subscript),
        positions,
        list(access.patterns),
        config,
    )
    return OneHotGatherPlan(access, tuple(positions), plan)


def _one_hot_scatter(
    access: MemoryAccess, positions: list[int], _config: Config
) -> OneHotScatterPlan:
    from .gather import build_scatter_plan

    plan = build_scatter_plan(access.tensor, list(access.subscript), positions)
    return OneHotScatterPlan(access, tuple(positions), plan)


def select_tensorcore_plan(
    access: MemoryAccess, config: Config
) -> TensorCorePlan | None:
    """Choose a TensorCore implementation without changing shared patterns."""
    positions = [
        index
        for index, pattern in enumerate(access.patterns)
        if isinstance(pattern, TensorIndexPattern)
    ]
    if not positions:
        return None
    # One-hot is the fallback while Pallas has no native TensorCore indirect access.
    if access.kind is MemoryAccessKind.LOAD:
        return _one_hot_gather(access, positions, config)
    if access.kind is MemoryAccessKind.STORE:
        return _one_hot_scatter(access, positions, config)
    if access.kind is MemoryAccessKind.ATOMIC:
        from ...language.atomic_ops import atomic_add

        if access.node.target is atomic_add:
            return _one_hot_scatter(access, positions, config)
    op = access.node.target
    op_name = getattr(op, "__name__", str(op))
    raise NotImplementedError(
        f"Pallas: tensor-indexed memory op is not supported for op={op_name}."
    )


def build_tensorcore_plans(graphs: list[GraphInfo], config: Config) -> None:
    """Select TensorCore plans after shared memory analysis."""
    from .memory_access import MEMORY_ACCESS_META
    from .memory_access import MemoryAccess

    for graph_info in graphs:
        for node in graph_info.graph.nodes:
            access = node.meta.get(MEMORY_ACCESS_META)
            if isinstance(access, MemoryAccess):
                node.meta[TENSORCORE_PLAN_META] = select_tensorcore_plan(access, config)
