"""Explicit execution ownership for common contraction-region emitters.

A context describes an already-established role, not a launch or a pipeline.
The caller binds zero-based role-local thread/warp indices for a contiguous,
warp-aligned team, supplies its resources, and ensures every participant reaches
each synchronization statement. Named-barrier construction, stage readiness,
allocation, iteration coordinates and cross-role publication remain external.
"""

from __future__ import annotations

from dataclasses import dataclass

from .thread_budget import MAX_THREADS_PER_BLOCK
from .warp_specialized_plan import WARP_SIZE


@dataclass(frozen=True)
class ChainedExecution:
    """An immutable role-local view; defaults preserve the existing CTA emitter.

    ``threads`` is the role's participant count, not the kernel's launch size.
    ``thread`` and ``warp`` name caller-bound local indices. ``sync`` is a
    reusable full-team join, e.g. ``prep_barrier.arrive_and_wait()``, without
    ancillary phase bookkeeping. Pointer fields
    are caller-owned expressions and may be replaced for a particular frame.
    The context neither creates resources nor rewrites generated source.
    """

    threads: int
    thread: str = "chain_thread"
    warp: str = "chain_warp"
    sync: str = "cute.arch.sync_threads()"
    a_workspace: str = "chain_a_workspace"
    b_workspace: str = "chain_b_workspace"
    tmem: str = "chain_tptr"
    barriers: str = "chain_bars"

    def __post_init__(self) -> None:
        if (
            type(self.threads) is not int
            or not WARP_SIZE <= self.threads <= MAX_THREADS_PER_BLOCK
            or self.threads % WARP_SIZE
        ):
            raise ValueError(
                "execution participants must be whole warps within one CTA"
            )
