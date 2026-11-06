"""Tests of the shared test helpers in ``helion._testing``."""

from __future__ import annotations

import itertools

import pytest

from helion._testing import pairwise

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
