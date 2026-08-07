from __future__ import annotations

from collections import Counter

import pytest

from helion_rag.experiment.scheduler import ScheduledRun
from helion_rag.experiment.scheduler import schedule
from helion_rag.experiment.scheduler import verify_balance

_WORKLOADS = ["w0", "w1", "w2"]
_ARMS = ["cold", "control", "best", "qwen", "combined"]


def test_determinism_same_seed_identical_order():
    a = schedule(_WORKLOADS, _ARMS, seed=1234)
    b = schedule(_WORKLOADS, _ARMS, seed=1234)
    assert a == b


def test_every_block_is_a_full_permutation():
    runs = schedule(_WORKLOADS, _ARMS, seed=7, repetitions=5)
    # 3 workloads * 5 reps = 15 blocks, each a permutation of all 5 arms.
    assert len(runs) == len(_WORKLOADS) * 5 * len(_ARMS)
    for workload in _WORKLOADS:
        for repetition in range(1, 6):
            block = [
                run.arm
                for run in runs
                if run.workload == workload and run.repetition == repetition
            ]
            assert sorted(block) == sorted(_ARMS)


def test_balance_counts_and_helper():
    runs = schedule(_WORKLOADS, _ARMS, seed=99, repetitions=5)
    arm_counts = Counter(run.arm for run in runs)
    assert all(arm_counts[arm] == len(_WORKLOADS) * 5 for arm in _ARMS)
    pair_counts = Counter((run.workload, run.arm) for run in runs)
    assert all(pair_counts[(w, a)] == 5 for w in _WORKLOADS for a in _ARMS)
    assert verify_balance(runs, _WORKLOADS, _ARMS, repetitions=5)


def test_position_balance_when_blocks_divide_arms():
    # 5 workloads * 5 reps = 25 blocks = 5 full cycles of 5 arms => each arm
    # appears in each ordinal position exactly 5 times.
    workloads = [f"w{i}" for i in range(5)]
    runs = schedule(workloads, _ARMS, seed=3, repetitions=5)
    n = len(_ARMS)
    by_position: dict[int, Counter[str]] = {pos: Counter() for pos in range(n)}
    for run in runs:
        by_position[run.order_index % n][run.arm] = (
            by_position[run.order_index % n][run.arm] + 1
        )
    for pos in range(n):
        assert all(by_position[pos][arm] == 5 for arm in _ARMS)


def test_unique_contiguous_order_index():
    runs = schedule(_WORKLOADS, _ARMS, seed=42)
    indices = [run.order_index for run in runs]
    assert indices == list(range(len(runs)))
    assert len(set(indices)) == len(runs)


def test_different_seeds_differ_but_stay_balanced():
    a = schedule(_WORKLOADS, _ARMS, seed=1)
    b = schedule(_WORKLOADS, _ARMS, seed=2)
    assert a != b
    assert verify_balance(a, _WORKLOADS, _ARMS)
    assert verify_balance(b, _WORKLOADS, _ARMS)


def test_verify_balance_detects_imbalance():
    runs = schedule(_WORKLOADS, _ARMS, seed=5)
    dropped = [run for run in runs if run.arm != _ARMS[0]]
    assert not verify_balance(dropped, _WORKLOADS, _ARMS)


def test_single_arm_and_single_workload():
    runs = schedule(["only"], ["solo"], seed=0, repetitions=3)
    assert runs == [
        ScheduledRun("only", "solo", 1, 0),
        ScheduledRun("only", "solo", 2, 1),
        ScheduledRun("only", "solo", 3, 2),
    ]


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        schedule(_WORKLOADS, [], seed=0)
    with pytest.raises(ValueError):
        schedule(_WORKLOADS, ["a", "a"], seed=0)
    with pytest.raises(ValueError):
        schedule(_WORKLOADS, _ARMS, seed=0, repetitions=0)
