"""Logical-coordinate layouts for typed, scalar-access pointwise caches.

The permutation stays within each row, so byte accounting, slot alignment and
lifetimes are unchanged. This does not describe a native MMA operand or grant
permission to vector-copy physically consecutive values into logical order.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ... import exc


def cache_xor_swizzle(
    shape: tuple[int, ...], dtype: torch.dtype
) -> tuple[int, int, int] | None:
    if len(shape) != 2 or dtype not in (
        torch.float32,
        torch.float16,
        torch.bfloat16,
    ):
        return None
    rows, columns = shape
    if (
        type(rows) is not int
        or type(columns) is not int
        or rows < 2
        or columns < 2
        or columns & (columns - 1)
    ):
        return None
    column_bits = columns.bit_length() - 1
    # A 32-bit bank word contains two half elements. For >=64-column half
    # rows, preserve that low bit and permute the five actual bank bits. Short
    # rows instead permute all available low columns; source/target fields
    # remain disjoint even at the smallest admitted width.
    base = int(dtype != torch.float32 and column_bits >= 6)
    shift = column_bits - base
    return min(5, shift), base, shift


@dataclass
class PointwiseCacheLayouts:
    """Attempt-local activation, independent of FP32 contraction scratch."""

    mode: str = "auto"
    activated: bool = False

    def __post_init__(self) -> None:
        if self.mode not in ("auto", "xor"):
            raise ValueError("pointwise cache layout must be auto or xor")

    def layout(self, shape: tuple[int, ...], dtype: torch.dtype, default: str) -> str:
        parameters = cache_xor_swizzle(shape, dtype) if self.mode == "xor" else None
        if parameters is None:
            return default
        self.activated = True
        bits, base, shift = parameters
        return (
            f"cute.make_composed_layout(cute.make_swizzle({bits}, {base}, {shift}), "
            f"0, cute.make_layout({shape!r}, stride=({shape[1]}, 1)))"
        )

    def validate(self) -> None:
        if self.mode == "xor" and not self.activated:
            raise exc.BackendUnsupported(
                "cute", "cache XOR layout requires an eligible materialized typed cache"
            )
