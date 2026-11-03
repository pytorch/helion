"""Table tests for the grouped static-persistent layout tail walker.

``helion._compiler.cute.grouped_static_tail.ordered_group_m_tail`` is shared by
the codegen-time analysis (``m_tail_preserve`` from the tail-epilogue proof)
and the bind-time specialization key (``layout_has_m_tail``, which always
preserves the M tail). The key half used to carry its own copy of the walk
with ``m_tail_preserve=True`` hard-wired; ``_reference_layout_has_m_tail`` is
that copy, kept here as the oracle the shared walker must agree with.
"""

from __future__ import annotations

import pytest
import torch

from helion._compiler.cute import grouped_static_tail

# Backend-independent planner test over CPU tensors: it runs on every backend row.


def _reference_layout_has_m_tail(
    layout_values: tuple[int, ...],
    *,
    bm: int,
    group_count: int,
) -> bool | None:
    cursor = 0
    has_m_tail = False
    for expected_group in range(group_count):
        if expected_group > 0:
            next_m_boundary = ((cursor + bm - 1) // bm) * bm
            while (
                cursor < len(layout_values)
                and cursor < next_m_boundary
                and layout_values[cursor] < 0
            ):
                cursor += 1
            if cursor != next_m_boundary or (
                cursor < len(layout_values) and layout_values[cursor] < 0
            ):
                return None
        if cursor >= len(layout_values) or layout_values[cursor] != expected_group:
            return None
        start = cursor
        while cursor < len(layout_values) and layout_values[cursor] == expected_group:
            cursor += 1
        actual_m = cursor - start
        if start % bm != 0:
            return None
        has_m_tail = has_m_tail or actual_m % bm != 0
    if cursor != len(layout_values):
        if all(value < 0 for value in layout_values[cursor:]):
            cursor = len(layout_values)
    if cursor != len(layout_values):
        return None
    return has_m_tail


def _layout(*runs: tuple[int, int]) -> tuple[int, ...]:
    """``(group_id, rows)`` runs; a negative group id is padding."""
    assert all(rows >= 0 for _group, rows in runs), runs
    return tuple(group for group, rows in runs for _ in range(rows))


_BM = 4
# (layout, group_count, expected with m_tail_preserve=True, expected without)
_CASES = [
    pytest.param(_layout((0, 8), (1, 4)), 2, False, False, id="dense"),
    pytest.param(_layout((0, 6), (-1, 1), (1, 4)), 2, None, None, id="bad-pad"),
    pytest.param(_layout((0, 6), (-1, 2), (1, 4)), 2, True, None, id="group-m-tail"),
    pytest.param(_layout((0, 8), (1, 3)), 2, True, True, id="last-group-tail"),
    pytest.param(_layout((0, 8), (-1, 4), (1, 4)), 2, None, None, id="whole-pad-tile"),
    pytest.param(_layout((0, 8), (1, 4), (-1, 3)), 2, False, None, id="trailing-pad"),
    pytest.param(_layout((0, 8), (1, 4), (2, 4)), 2, None, None, id="extra-group"),
    pytest.param(_layout((1, 4), (0, 4)), 2, None, None, id="unordered"),
    pytest.param(_layout((0, 8), (2, 4)), 3, None, None, id="skipped-group"),
    pytest.param(_layout((0, 8)), 2, None, None, id="missing-group"),
    pytest.param((), 1, None, None, id="empty"),
    pytest.param(_layout((0, 4)), 1, False, False, id="single-full"),
    pytest.param(_layout((0, 5), (-1, 3), (1, 1)), 2, True, None, id="padded-tails"),
]


@pytest.mark.parametrize(
    ("layout", "group_count", "expected_preserve", "expected_adjacent"), _CASES
)
def test_ordered_group_m_tail_matches_the_oracle(
    layout: tuple[int, ...],
    group_count: int,
    expected_preserve: bool | None,
    expected_adjacent: bool | None,
) -> None:
    assert (
        grouped_static_tail.ordered_group_m_tail(
            layout, group_count=group_count, bm=_BM, m_tail_preserve=True
        )
        is expected_preserve
    )
    assert (
        grouped_static_tail.ordered_group_m_tail(
            layout, group_count=group_count, bm=_BM, m_tail_preserve=False
        )
        is expected_adjacent
    )
    assert (
        grouped_static_tail.layout_has_m_tail(layout, bm=_BM, group_count=group_count)
        is _reference_layout_has_m_tail(layout, bm=_BM, group_count=group_count)
        is expected_preserve
    )


class _Env:
    def __init__(self, **tensors: torch.Tensor) -> None:
        self.runtime_arg_values_by_name = tensors


def test_analyze_grouped_static_tail_reads_the_bound_tensors() -> None:
    layout = torch.tensor(_layout((0, 8), (1, 3)), dtype=torch.int32)
    n_sizes = torch.tensor((64, 48), dtype=torch.int64)
    facts = grouped_static_tail.analyze_grouped_static_tail(
        _Env(layout=layout, n_sizes=n_sizes),  # pyrefly: ignore [bad-argument-type]
        layout_arg_name="layout",
        n_sizes_arg_name="n_sizes",
        group_count=2,
        bm=_BM,
        bn=32,
        n_size=64,
        m_tail_preserve=False,
    )
    assert facts == grouped_static_tail.GroupedStaticTailFacts(
        has_m_tail=True, has_n_tail=True
    )
    # Without ``n_sizes`` every group spans the full N extent.
    facts = grouped_static_tail.analyze_grouped_static_tail(
        _Env(layout=layout),  # pyrefly: ignore [bad-argument-type]
        layout_arg_name="layout",
        n_sizes_arg_name=None,
        group_count=2,
        bm=_BM,
        bn=32,
        n_size=64,
        m_tail_preserve=False,
    )
    assert facts == grouped_static_tail.GroupedStaticTailFacts(
        has_m_tail=True, has_n_tail=False
    )


@pytest.mark.parametrize(
    "n_sizes",
    [
        torch.tensor((64, 0), dtype=torch.int32),  # a zero-row group
        torch.tensor((64, 96), dtype=torch.int32),  # beyond n_size
        torch.tensor((64,), dtype=torch.int32),  # wrong group count
        torch.tensor((64.0, 32.0)),  # not an int tensor
    ],
)
def test_analyze_grouped_static_tail_declines_bad_n_sizes(
    n_sizes: torch.Tensor,
) -> None:
    layout = torch.tensor(_layout((0, 4), (1, 4)), dtype=torch.int32)
    assert (
        grouped_static_tail.analyze_grouped_static_tail(
            _Env(layout=layout, n_sizes=n_sizes),  # pyrefly: ignore [bad-argument-type]
            layout_arg_name="layout",
            n_sizes_arg_name="n_sizes",
            group_count=2,
            bm=_BM,
            bn=32,
            n_size=64,
            m_tail_preserve=True,
        )
        is None
    )


def test_tail_extractor_recomputes_both_facts() -> None:
    descriptor = ("cute_grouped_static_tail", 0, 2, _BM, 1, 32)
    extractor = grouped_static_tail.make_tail_extractor(descriptor)
    layout = torch.tensor(_layout((0, 8), (1, 3)), dtype=torch.int64)
    n_sizes = torch.tensor((64, 48), dtype=torch.int32)
    assert extractor((layout, n_sizes)) == (descriptor, True, True)
    layout[8:] = -1
    assert extractor((layout, n_sizes)) == (descriptor, None, True)
    n_sizes.copy_(torch.tensor((64, 64), dtype=torch.int32))
    assert extractor((layout, n_sizes)) == (descriptor, None, False)
    # Missing arguments leave the facts unknown rather than raising.
    assert extractor((layout,)) == (descriptor, None, None)
