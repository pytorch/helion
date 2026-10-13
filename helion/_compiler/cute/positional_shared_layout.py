"""Logical-coordinate shared layouts for positional root allocations.

The optional XOR permutes elements only within the last dimension. It preserves
each 16-byte grain and the exact allocation extent, including non-power-of-two
outer dimensions. Consumers continue using logical tensor coordinates.
"""

from __future__ import annotations

import math


def positional_xor_swizzle(
    shape: tuple[int, ...], element_bytes: int
) -> tuple[int, int, int] | None:
    if (
        len(shape) < 2
        or any(size <= 0 for size in shape)
        or math.prod(shape[:-1]) < 2
        or element_bytes <= 0
        or element_bytes > 16
        or element_bytes & (element_bytes - 1)
    ):
        return None
    columns = shape[-1]
    if columns & (columns - 1):
        return None
    # Offsets are in elements. Preserve a 16-byte grain for every element width,
    # then XOR at most three row bits into column bits above that grain.
    base = (16 // element_bytes).bit_length() - 1
    column_bits = columns.bit_length() - 1
    shift = column_bits - base
    bits = min(3, shift)
    if bits <= 0:
        return None
    # Target bits [base, base+bits) lie entirely inside the last dimension;
    # source bits start at column_bits, so source and target never overlap.
    return bits, base, shift


def shared_tile_layout(shape: tuple[int, ...], element_bytes: int, mode: str) -> str:
    if mode not in ("column", "row", "xor"):
        raise ValueError(f"unsupported positional shared layout: {mode}")
    if not shape:
        return "cute.make_layout(1)"
    if mode == "column":
        return f"cute.make_layout({shape!r})"
    strides = tuple(math.prod(shape[i + 1 :]) for i in range(len(shape)))
    layout = f"cute.make_layout({shape!r}, stride={strides!r})"
    parameters = positional_xor_swizzle(shape, element_bytes) if mode == "xor" else None
    if parameters is None:
        return layout
    bits, base, shift = parameters
    return (
        f"cute.make_composed_layout(cute.make_swizzle({bits}, {base}, {shift}), "
        f"0, {layout})"
    )
