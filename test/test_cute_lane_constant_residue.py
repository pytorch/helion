from __future__ import annotations

import ast
import itertools

import pytest

from helion._compiler.cute.persistent_branch_vec import (
    _scalar_pointers_are_modularly_lane_disjoint,
)


def check(
    left: str,
    right: str,
    *,
    width: int = 4,
    extent: int = 4,
    strides: dict[tuple[str, int], int] | None = None,
    unstable: tuple[str, ...] | set[str] = (),
) -> bool:
    return _scalar_pointers_are_modularly_lane_disjoint(
        ast.parse(left, mode="eval").body,
        ast.parse(right, mode="eval").body,
        lane_var="lane",
        lane_extent=extent,
        proven_tensor_stride_values=strides or {},
        unstable_names=set(unstable),
        width=width,
    )


@pytest.mark.parametrize("offset", [0, 4, -4, 8, -8, 124, -124])
def test_known_packet_offsets(offset: int) -> None:
    assert check(
        "state.iterator + slot * 524288 + lane * 128",
        f"state.iterator + slot * 524288 + lane * 128 + ({offset})",
        unstable={"slot"},
    )


@pytest.mark.parametrize("offset", [125, 126, 127, 128, 129, 130, 131])
@pytest.mark.parametrize("sign", [-1, 1])
def test_shifted_packets_overlap_in_one_direction(offset: int, sign: int) -> None:
    # Checking only positive deltas incorrectly accepts one of these signs.
    assert not check(
        "state.iterator + lane * 128",
        f"state.iterator + lane * 128 + ({offset * sign})",
    )


def test_wraparound_is_still_part_of_the_proof() -> None:
    assert not check(
        "state.iterator + lane * 2147483648",
        "state.iterator + lane * 2147483648",
        extent=3,
    )
    assert not check(
        "state.iterator + lane * 128",
        "state.iterator + lane * 128 + 4294967168",
    )
    assert check(
        "state.iterator + lane * 4294967424",
        "state.iterator + lane * 4294967424 + 4",
    )


@pytest.mark.parametrize(
    "left,right",
    [
        ("state.iterator + lane * 128", "other.iterator + lane * 128"),
        ("state.iterator + lane * 128", "state.iterator + lane * 64 + 4"),
        ("state.iterator + lane * 128", "state.iterator + unknown(lane)"),
        ("state.iterator + lane * 128", "state.iterator + 4"),
        ("state.iterator + lane * 128", "state.iterator + lane * lane + 4"),
        ("state.iterator + lane * 128", "state.iterator + lane * stride + 4"),
        (
            "state.iterator + (lane * 128 if active else 0)",
            "state.iterator + lane * 128 + 4",
        ),
        (
            "state.iterator + lane * state.layout.stride[0]",
            "state.iterator + lane * state.layout.stride[0] + 4",
        ),
    ],
)
def test_unknown_or_unproved_mapping_declines(left: str, right: str) -> None:
    assert not check(left, right)


def test_changed_base_is_not_assumed_identical() -> None:
    assert not check(
        "state.iterator + slot + lane * 128",
        "state.iterator + slot + lane * 128 + 4",
        unstable={"slot"},
    )
    assert check(
        "state.iterator + slot + lane * 128",
        "state.iterator + slot + lane * 128 + 4",
    )


def test_exact_stride_authority_and_negative_steps() -> None:
    for step in (128, -128):
        assert check(
            "state.iterator + lane * state.layout.stride[0]",
            "state.iterator + lane * state.layout.stride[0] + 4",
            strides={("state", 0): step},
        )
    assert not check(
        "state.iterator + lane * state.layout.stride[0]",
        "state.iterator + lane * state.layout.stride[0] + 4",
        strides={("state", 0): 4},
    )


def test_all_admitted_intervals_are_disjoint_in_bounded_modular_model() -> None:
    # Independent enumeration includes unrelated load/store slot values,
    # negative strides/displacements and Int32 wrap. It checks soundness,
    # rather than merely reimplementing the gcd predicate.
    accepted = rejected = 0
    for step, offset, width, base_stride in itertools.product(
        [-128, -8, -4, 0, 4, 8, 128, 1 << 31, (1 << 32) - 8],
        [-129, -128, -127, -8, -4, -1, 0, 1, 4, 8, 127, 128, 129],
        [1, 2, 4],
        [0, 4, 16, 128, 1024],
    ):
        admitted = check(
            f"state.iterator + load_slot * {base_stride} + lane * ({step})",
            f"state.iterator + store_slot * {base_stride} + lane * ({step}) + ({offset})",
            width=width,
        )
        if not admitted:
            rejected += 1
            continue
        accepted += 1
        for load_slot, store_slot, i, j in itertools.product(
            [-2, -1, 0, 1, 2], [-2, -1, 0, 1, 2], range(4), range(4)
        ):
            if i == j:
                continue
            left = {
                (load_slot * base_stride + step * i + element) % (1 << 32)
                for element in range(width)
            }
            right = {
                (store_slot * base_stride + step * j + offset + element) % (1 << 32)
                for element in range(width)
            }
            assert left.isdisjoint(right), (
                step,
                offset,
                width,
                base_stride,
                load_slot,
                store_slot,
                i,
                j,
            )
    assert accepted > 100 and rejected > 100
