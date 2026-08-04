from __future__ import annotations

import torch

from helion._compiler.pallas.sparsecore_compute import chunk_schedule


def test_lane_chunks() -> None:
    assert [
        (chunk.start, chunk.size) for chunk in chunk_schedule(48, torch.float32)
    ] == [
        (0, 16),
        (16, 16),
        (32, 16),
    ]
    assert [
        (chunk.start, chunk.size, chunk.unique_start)
        for chunk in chunk_schedule(48, torch.bfloat16)
    ] == [(0, 32, 0), (16, 32, 32)]
