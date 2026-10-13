"""Independent preparation teams feeding one ordered recurrence team.

Each cohort owns one frame slot. Cohort j prepares iterations count*g+j and
reuses its slot only after the ordered consumer completes generation g-1.
READY follows all producer writes; EMPTY follows all consumer reads, including
asynchronous operations. This module supplies ownership and protocol coordinates,
not those completion fences, frame storage, arithmetic, or instruction layouts.

The existing single preparation team with two rotating slots is a different
schedule. A count of one deliberately returns no cohort plan: callers retain
that existing path unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .chained_execution import ChainedExecution
from .thread_budget import MAX_THREADS_PER_BLOCK
from .warp_specialized_plan import WARP_SIZE

if TYPE_CHECKING:
    from collections.abc import Mapping

_TEAM_THREADS = 4 * WARP_SIZE
_NAMED_BARRIER_COUNT = 16
# CTA/TMEM, the original role barriers and transport completion use IDs 0..4.
_FIRST_COHORT_BARRIER = 5


def supports_twenty_warp_preparation(config: Mapping[str, object]) -> bool:
    """The bounded non-power-of-two CTA belongs only to compact preparation.

    This is ownership/launch-policy admission, never storage or graph proof.
    The caller must still admit the common loop and finalize its actual table.
    """
    warps = config.get("num_warps")
    consumer = config.get("cute_chained_pipeline_consumer_warps", 4)
    count = config.get("cute_chained_preparation_cohorts", 1)
    minimum = config.get("cute_min_blocks_per_mp")
    if (
        type(warps) is not int
        or warps != 20
        or type(consumer) is not int
        or consumer not in (4, 8, 16)
        or type(count) is not int
        or type(minimum) is not int
        or minimum != 1
        or config.get("cute_chained_mma_schedule") != "tcgen05_tmem"
        or config.get("cute_chained_preparation_pipeline") is not True
        or config.get("cute_chained_compact_preparation") is not True
    ):
        return False
    cohorts = plan_preparation_cohorts(WARP_SIZE * warps, WARP_SIZE * consumer, count)
    return cohorts is not None and cohorts.cohort_threads == _TEAM_THREADS


def _valid(
    cta_threads: int,
    recurrence_threads: int,
    count: int,
    has_tma: bool,
    named_barrier_base: int,
) -> bool:
    if (
        any(
            type(value) is not int
            for value in (cta_threads, recurrence_threads, count, named_barrier_base)
        )
        or type(has_tma) is not bool
        or count <= 1
        or not 0 < cta_threads <= MAX_THREADS_PER_BLOCK
        or not 0 < recurrence_threads < cta_threads
        or cta_threads % _TEAM_THREADS
        or recurrence_threads % _TEAM_THREADS
        or named_barrier_base < _FIRST_COHORT_BARRIER
        or named_barrier_base + count > _NAMED_BARRIER_COUNT
    ):
        return False
    preparation_threads = cta_threads - recurrence_threads
    return preparation_threads % (count * _TEAM_THREADS) == 0


def _positive_step(step: int) -> None:
    if type(step) is not int or step <= 0:
        raise ValueError("cohort loop step must be a positive integer")


@dataclass(frozen=True)
class PreparationCohorts:
    """Validated CTA partition and disjoint per-slot barrier resources.

    Named barriers synchronize all threads of one preparation cohort, never a
    predicated MMA subset. Slot mbarriers are a separate shared-memory array:
    READY[0:count], EMPTY[count:2*count], and optional TMA[2*count:3*count].
    All are initialized with one arrival; only the role's elected warp signals
    READY/EMPTY. Private TMA completion advances once per cohort generation.
    """

    cta_threads: int
    recurrence_threads: int
    count: int
    has_tma: bool = False
    named_barrier_base: int = _FIRST_COHORT_BARRIER

    def __post_init__(self) -> None:
        if not _valid(
            self.cta_threads,
            self.recurrence_threads,
            self.count,
            self.has_tma,
            self.named_barrier_base,
        ):
            raise ValueError("invalid independent preparation cohort resources")

    @property
    def slots(self) -> int:
        return self.count

    @property
    def preparation_threads(self) -> int:
        return self.cta_threads - self.recurrence_threads

    @property
    def cohort_threads(self) -> int:
        return self.preparation_threads // self.count

    @property
    def named_barrier_ids(self) -> tuple[int, ...]:
        return tuple(
            range(self.named_barrier_base, self.named_barrier_base + self.count)
        )

    @property
    def slot_mbarrier_count(self) -> int:
        return self.count * (2 + int(self.has_tma))

    @property
    def slot_mbarrier_bytes(self) -> int:
        return 8 * self.slot_mbarrier_count

    @property
    def slot_mbarrier_allocated_bytes(self) -> int:
        """This array only; stage barriers and TMEM address storage are separate."""
        return (self.slot_mbarrier_bytes + 127) // 128 * 128

    def barrier_indices(self, slot: int) -> tuple[int, int, int | None]:
        """READY, EMPTY, optional private TMA indices for one immutable slot."""
        if type(slot) is not int or not 0 <= slot < self.count:
            raise ValueError("cohort slot is outside the preparation schedule")
        return (
            slot,
            self.count + slot,
            2 * self.count + slot if self.has_tma else None,
        )

    def barrier_pointers(self) -> tuple[str, str, str | None]:
        return (
            "chain_slot_bars + chain_slot",
            f"chain_slot_bars + {self.count} + chain_slot",
            f"chain_slot_bars + {2 * self.count} + chain_slot"
            if self.has_tma
            else None,
        )

    def preparation_bindings(self) -> list[str]:
        """Inside the preparation branch; all cohort threads share its barrier."""
        return [
            f"chain_cohort = chain_thread // {self.cohort_threads}",
            f"chain_prep_thread = chain_thread % {self.cohort_threads}",
            f"chain_prep_warp = chain_prep_thread // {WARP_SIZE}",
            f"chain_prep_barrier = chain_pipeline.NamedBarrier(barrier_id={self.named_barrier_base} + chain_cohort, num_threads={self.cohort_threads})",
        ]

    def recurrence_bindings(self) -> list[str]:
        return [
            f"chain_recurrence_thread = chain_thread - {self.preparation_threads}",
            f"chain_recurrence_warp = chain_recurrence_thread // {WARP_SIZE}",
        ]

    def preparation_execution(self) -> ChainedExecution:
        return ChainedExecution(
            self.cohort_threads,
            thread="chain_prep_thread",
            warp="chain_prep_warp",
            sync="chain_prep_barrier.arrive_and_wait()",
        )

    def producer_header(self, step: int) -> str:
        _positive_step(step)
        return f"for chain_loop_index in cutlass.range(chain_loop_begin + chain_cohort * {step}, chain_loop_end, {self.count * step}, unroll=1):"

    def consumer_header(self, step: int) -> str:
        _positive_step(step)
        return f"for chain_loop_index in cutlass.range(chain_loop_begin, chain_loop_end, {step}, unroll=1):"

    def iteration_bindings(self, step: int) -> list[str]:
        _positive_step(step)
        return [
            f"chain_iteration = ((chain_loop_index - chain_loop_begin) // {step})",
            f"chain_slot = chain_iteration % {self.count}",
            f"chain_generation = chain_iteration // {self.count}",
        ]

    @property
    def completion_phase(self) -> str:
        """READY and private TMA use slot generations, not global iterations."""
        return "chain_generation & 1"

    @property
    def reuse_phase(self) -> str:
        """Wait only when chain_generation > 0, before touching the frame."""
        return "(chain_generation - 1) & 1"


def plan_preparation_cohorts(
    cta_threads: int,
    recurrence_threads: int,
    count: int,
    *,
    has_tma: bool = False,
    named_barrier_base: int = _FIRST_COHORT_BARRIER,
) -> PreparationCohorts | None:
    """Validate requested physical teams; do not select counts or storage.

    None for count=1 means the existing two-slot, single-team fallback. Other
    invalid resource requests also return None; explicit-option validation is
    the caller's responsibility. Register and shared-memory capacity still
    require the caller's complete physical plan.
    """
    if not _valid(cta_threads, recurrence_threads, count, has_tma, named_barrier_base):
        return None
    return PreparationCohorts(
        cta_threads, recurrence_threads, count, has_tma, named_barrier_base
    )
