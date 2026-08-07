"""Balanced randomized experiment scheduling (§5, §10, §18).

The plan runs five repetitions per workload and arm with "balanced randomized
order inside workload-by-repetition blocks" (§10) and a frozen RNG seed for
reproducibility (§18). This module turns a workload list, an arm list, a
repetition count, and that frozen seed into a deterministic, ordered sequence of
:class:`ScheduledRun` units.

Two balance properties hold by construction:

* Within every ``(workload, repetition)`` block each arm appears exactly once
  (each block is a full permutation of the arms), so each arm appears exactly
  ``len(workloads) * repetitions`` times overall and each ``(workload, arm)``
  pair exactly ``repetitions`` times.
* Across blocks each arm appears equally often in each ordinal position, as far
  as the block count evenly divides the arm count. Block permutations are drawn
  from a per-seed randomized Latin square whose rows are traversed in a fresh
  random order every ``len(arms)`` blocks, so each complete cycle of blocks uses
  every Latin row (hence every position-by-arm assignment) exactly once.

Everything is a pure function of ``seed``: :func:`numpy.random.default_rng` is
the only randomness source and no global/``random`` state is touched.
"""

from __future__ import annotations

import dataclasses
from collections import Counter
from collections.abc import Sequence

import numpy as np


@dataclasses.dataclass(frozen=True)
class ScheduledRun:
    """One ``(workload, arm, repetition)`` run unit in the frozen execution order.

    ``order_index`` is the unique, contiguous position of this run in the emitted
    sequence; it is the stable primary key for the run in the research ledger.
    ``repetition`` is 1-indexed (``1 .. repetitions``).
    """

    workload: str
    arm: str
    repetition: int
    order_index: int


def schedule(
    workloads: Sequence[str],
    arms: Sequence[str],
    *,
    seed: int,
    repetitions: int = 5,
) -> list[ScheduledRun]:
    """Deterministic balanced randomized schedule (§5, §10, §18).

    Blocks are visited in ``(workload, repetition)`` order; within each block the
    arm order is a randomized-but-balanced permutation as described in the module
    docstring. Identical ``seed`` values yield identical schedules.
    """
    if repetitions < 1:
        raise ValueError(f"repetitions must be >= 1, got {repetitions}")
    if not arms:
        raise ValueError("arms must be non-empty")
    if len(set(arms)) != len(arms):
        raise ValueError("arms must be unique")

    rng = np.random.default_rng(seed)
    n = len(arms)

    # Randomized Latin square: rows are cyclic shifts of one random base
    # permutation, so every column (ordinal position) holds every arm once.
    base = [int(i) for i in rng.permutation(n)]
    latin = [[base[(pos + shift) % n] for pos in range(n)] for shift in range(n)]

    # Assign a Latin row to each block, refreshing the row order every ``n``
    # blocks so each complete cycle uses all rows once (position balance).
    num_blocks = len(workloads) * repetitions
    block_rows: list[int] = []
    while len(block_rows) < num_blocks:
        block_rows.extend(int(r) for r in rng.permutation(n))
    block_rows = block_rows[:num_blocks]

    runs: list[ScheduledRun] = []
    order_index = 0
    block = 0
    for workload in workloads:
        for repetition in range(1, repetitions + 1):
            for arm_idx in latin[block_rows[block]]:
                runs.append(
                    ScheduledRun(
                        workload=workload,
                        arm=arms[arm_idx],
                        repetition=repetition,
                        order_index=order_index,
                    )
                )
                order_index += 1
            block += 1
    return runs


def verify_balance(
    runs: Sequence[ScheduledRun],
    workloads: Sequence[str],
    arms: Sequence[str],
    *,
    repetitions: int = 5,
) -> bool:
    """True iff arm and ``(workload, arm)`` occurrence counts are balanced (§5, §10).

    Each arm must appear exactly ``len(workloads) * repetitions`` times and each
    ``(workload, arm)`` pair exactly ``repetitions`` times.
    """
    arm_counts = Counter(run.arm for run in runs)
    expected_per_arm = len(workloads) * repetitions
    if any(arm_counts[arm] != expected_per_arm for arm in arms):
        return False
    pair_counts = Counter((run.workload, run.arm) for run in runs)
    return all(
        pair_counts[(workload, arm)] == repetitions
        for workload in workloads
        for arm in arms
    )
