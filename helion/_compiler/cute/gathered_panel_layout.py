"""Static geometry and shared-memory bounds of the swizzled BF16 page ring."""

from __future__ import annotations

from .gathered_launch import GATHERED_SEED_THREADS

RING_STAGES = tuple(range(2, 9))
MMA_ROWS = 16
PAGE_ROWS = 64
DEPTH_PANEL = 64
VALUE_PANEL = 128
STAGE_BYTES = (MMA_ROWS + PAGE_ROWS) * DEPTH_PANEL * 2


def slab_bytes(stages: int) -> int:
    # A 16-byte aligned arena can need 112 extra bytes to reach a 128-byte
    # swizzle origin. Even the smallest ring can also hold the later V panel.
    return stages * STAGE_BYTES + 112


def row_scratch_size(warps: int) -> int:
    return (warps + 2) * MMA_ROWS


def working_set_bytes(rows: int, columns: int, width: int, stages: int) -> int:
    # Both incoming and outgoing (row max, denominator and accumulator) may
    # need storage. Private carries can use less, but are not assumed here.
    carries = 2 * (2 * rows + rows * width) * 4
    return (
        slab_bytes(stages)
        + carries
        + MMA_ROWS * columns * 2
        + row_scratch_size(GATHERED_SEED_THREADS // 32) * 4
    )
