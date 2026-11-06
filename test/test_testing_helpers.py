"""Tests of the shared test helpers in ``helion._testing``."""

from __future__ import annotations

import itertools
from pathlib import Path

import pytest

from helion._testing import pairwise
from helion._testing.sharding import MIN_WEIGHT
from helion._testing.sharding import assign_shards
from helion._testing.sharding import load_durations
from helion._testing.sharding import shard_loads

pytestmark = pytest.mark.cpu_only


def _assert_all_pairs_covered(rows: list[tuple[object, ...]], **axes: tuple) -> None:
    names = list(axes)
    for i, j in itertools.combinations(range(len(names)), 2):
        seen = {(row[i], row[j]) for row in rows}
        assert seen == set(itertools.product(axes[names[i]], axes[names[j]]))


def test_pairwise_covers_every_pair_of_eight_flags() -> None:
    flags = {f"flag{i}": (False, True) for i in range(8)}
    rows = pairwise(**flags)
    _assert_all_pairs_covered(rows, **flags)
    assert len(rows) == 9  # a full product would be 256 rows


def test_pairwise_mixes_axis_sizes_and_keeps_axis_order() -> None:
    axes = {
        "dtype": ("f16", "bf16", "f32"),
        "lanes": (1, 2, 4),
        "largest": (True, False),
    }
    rows = pairwise(**axes)
    _assert_all_pairs_covered(rows, **axes)
    assert len(rows) == 9
    assert all(
        row[0] in axes["dtype"]
        and row[1] in axes["lanes"]
        and row[2] in axes["largest"]
        for row in rows
    )


def test_pairwise_is_deterministic() -> None:
    axes = {"a": (1, 2, 3), "b": ("x", "y"), "c": (0.5, 1.5)}
    assert pairwise(**axes) == pairwise(**axes)


def test_pairwise_single_axis_lists_every_value() -> None:
    assert pairwise(width=(16, 32, 64)) == [(16,), (32,), (64,)]


def test_pairwise_rejects_an_empty_axis() -> None:
    with pytest.raises(ValueError, match="no values"):
        pairwise(a=(1, 2), b=())


def test_assign_shards_balances_recorded_durations() -> None:
    durations = {"a": 10.0, "b": 9.0, "c": 5.0, "d": 4.0, "e": 1.0, "f": 1.0}
    assignment = assign_shards(durations, 2, durations)
    assert sorted(assignment) == sorted(durations)
    assert assignment["a"] != assignment["b"]  # the two heaviest stay apart
    assert shard_loads(assignment, 2, durations) == [15.0, 15.0]


def test_assign_shards_is_deterministic_in_module_order() -> None:
    durations = {f"m{i}": float(i % 7 + 1) for i in range(40)}
    modules = list(durations)
    forward = assign_shards(modules, 3, durations)
    backward = assign_shards(reversed(modules), 3, durations)
    assert forward == backward
    assert set(forward.values()) == {0, 1, 2}


def test_assign_shards_weighs_unrecorded_modules_as_the_median() -> None:
    durations = {"heavy": 100.0, "light1": 1.0, "light2": 1.0}
    assignment = assign_shards(
        ["heavy", "light1", "light2", "new1", "new2"], 2, durations
    )
    # The new modules weigh 1.0 (the median), so everything else fits opposite
    # the heavy module.
    assert {m for m, s in assignment.items() if s == assignment["heavy"]} == {"heavy"}


def test_assign_shards_single_shard_and_invalid_count() -> None:
    assert assign_shards(["a", "b"], 1, {}) == {"a": 0, "b": 0}
    with pytest.raises(ValueError, match="positive"):
        assign_shards(["a"], 0, {})


def test_assign_shards_counts_zero_second_modules_at_the_floor() -> None:
    durations = {"heavy": 30.0, "mid": 20.0, **{f"skipped{i}": 0.0 for i in range(20)}}
    assignment = assign_shards(durations, 2, durations)
    # Each skipped module weighs MIN_WEIGHT, so they fill the lighter shard up
    # to the heavier one and then alternate, instead of all tying onto one.
    assert shard_loads(assignment, 2, durations) == [
        30.0 + 5 * MIN_WEIGHT,
        20.0 + 15 * MIN_WEIGHT,
    ]
    assert assignment["heavy"] != assignment["mid"]


def test_recorded_shard_durations_name_existing_test_modules() -> None:
    durations = load_durations()
    assert durations, "test/shard_durations.json is missing or empty"
    root = Path(__file__).resolve().parents[1]
    missing = [m for m in durations if not (root / m).is_file()]
    assert not missing, (
        f"run scripts/shard_durations.py (without reports it drops them): {missing}"
    )
    assert all(m.startswith("test/") and m.endswith(".py") for m in durations)
    assert all(seconds >= 0 for seconds in durations.values())
