from __future__ import annotations

from helion._compiler.pallas.sparsecore_compute import chunk_schedule


def test_lane_chunks() -> None:
    assert [(chunk.start, chunk.size) for chunk in chunk_schedule(48)] == [
        (0, 16),
        (16, 16),
        (32, 16),
    ]
