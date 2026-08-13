"""SparseCore launcher metadata shared with compiler codegen."""

from __future__ import annotations

from typing import TypedDict


class SparseCoreLauncherSpec(TypedDict):
    index_inputs: list[tuple[int, int, int, int]]
    value_inputs: list[tuple[int, int, int, int, int]]
    output_shapes: list[tuple[int, tuple[int, ...]]]
    reshape_outputs: list[int]
    scalar_outputs: list[int]
    int32_outputs: list[int]
    num_cores: int
    num_subcores: int
    dma_granule: int
