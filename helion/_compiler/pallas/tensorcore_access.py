"""TensorCore policy for shared Pallas access sites."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Callable

from .access import AccessKind
from .plan_tiling import TensorIndexPattern

if TYPE_CHECKING:
    from ...runtime.config import Config
    from .access import AccessSite
    from .gather import GatherPlan
    from .gather import ScatterPlan


TENSORCORE_ACCESS_META = "pallas_tensorcore_access"


@dataclass(frozen=True)
class TensorCoreAccess:
    """TensorCore plan for one memory access."""

    site: AccessSite
    indirect_positions: tuple[int, ...]


@dataclass(frozen=True)
class OneHotGatherAccess(TensorCoreAccess):
    """Resident-table one-hot/MXU fallback for an indirect load."""

    fallback: GatherPlan


@dataclass(frozen=True)
class ProjectionScatterAccess(TensorCoreAccess):
    """One-hot projection fallback for an indirect store."""

    fallback: ScatterPlan


AccessCandidate = Callable[["AccessSite", list[int], "Config"], TensorCoreAccess | None]


def _one_hot_gather_fallback(
    site: AccessSite, positions: list[int], config: Config
) -> OneHotGatherAccess:
    from .gather import build_gather_plan

    plan = build_gather_plan(
        site.tensor, list(site.subscripts), positions, list(site.patterns), config
    )
    return OneHotGatherAccess(site, tuple(positions), plan)


def _projection_scatter_fallback(
    site: AccessSite, positions: list[int], config: Config
) -> ProjectionScatterAccess:
    del config
    from .gather import build_scatter_plan

    plan = build_scatter_plan(site.tensor, list(site.subscripts), positions)
    return ProjectionScatterAccess(site, tuple(positions), plan)


# Native implementations belong before these fallbacks.
_INDIRECT_LOAD_CANDIDATES: tuple[AccessCandidate, ...] = (_one_hot_gather_fallback,)
_INDIRECT_STORE_CANDIDATES: tuple[AccessCandidate, ...] = (
    _projection_scatter_fallback,
)


def _select_candidate(
    candidates: tuple[AccessCandidate, ...],
    site: AccessSite,
    positions: list[int],
    config: Config,
) -> TensorCoreAccess:
    for candidate in candidates:
        if plan := candidate(site, positions, config):
            return plan
    raise NotImplementedError(
        f"Pallas TensorCore has no lowering for {site.kind.value} "
        f"with tensor indices at {positions}"
    )


def select_tensorcore_access(
    site: AccessSite, config: Config
) -> TensorCoreAccess | None:
    """Choose a TC implementation without changing shared patterns."""
    positions = [
        index
        for index, pattern in enumerate(site.patterns)
        if isinstance(pattern, TensorIndexPattern)
    ]
    if not positions:
        return None
    if site.kind is AccessKind.LOAD:
        return _select_candidate(_INDIRECT_LOAD_CANDIDATES, site, positions, config)
    if site.kind is AccessKind.STORE:
        return _select_candidate(_INDIRECT_STORE_CANDIDATES, site, positions, config)
    op_name = getattr(site.target, "__name__", str(site.target))
    raise NotImplementedError(
        f"Pallas: tensor-indexed memory op is not supported for op={op_name}."
    )
