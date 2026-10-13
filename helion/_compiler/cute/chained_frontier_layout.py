"""Packet-preserving permutations of original raw-half frontier images.

The frame owns the allocation, domain and lifetime. This helper only changes
the logical view inside that allocation. XOR targets start above the three
low element bits, so each aligned eight-half packet stays contiguous. Its
source bits belong to the row and its target bits stay within the column.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

import torch

from ...language import memory_ops

if TYPE_CHECKING:
    from .chained_preparation_frame import PreparationBuffer
    from .warp_specialized_plan import SharedBufferRegion


def packet_xor_swizzle(shape: tuple[int, ...]) -> tuple[int, int, int] | None:
    if len(shape) != 2 or any(type(size) is not int for size in shape):
        return None
    rows, columns = shape
    if rows < 2 or columns < 16 or columns & (columns - 1):
        return None
    shift = columns.bit_length() - 1 - 3
    return min(3, shift), 3, shift


@dataclass(frozen=True)
class PacketXorVectorSink:
    """A composed tensor layout, without an embedded pointer swizzle.

    Unlike a native MMA image, Pointer.align cannot discard any layout here:
    the physical element offset already applied the complete permutation.
    Source guards and copy completion remain owned by the original emitter.
    """

    target: str
    shape: tuple[int, int]
    dtype: str
    swizzle: tuple[int, int, int]

    def layout(self) -> str:
        bits, base, shift = self.swizzle
        return (
            f"cute.make_composed_layout(cute.make_swizzle({bits}, {base}, {shift}), "
            f"0, cute.make_layout({self.shape!r}, stride=({self.shape[1]}, 1)))"
        )

    def pointer(self, row: str, column: str) -> str:
        return f"{self.target}.iterator + {self.target}.layout(({row}, {column}))"

    def aligned_pointer(self, pointer: str) -> str:
        return f"{pointer}.align(16)"


def packet_xor_sink(
    buffer: PreparationBuffer, region: SharedBufferRegion
) -> PacketXorVectorSink | None:
    """Select only a complete original raw image in the actual plain binder.

    Native, grouped, leaf, cache and physical replacement views have separate
    authorities and never call this helper. No interval is shortened or moved.
    The original frame finalizer must still accept every action and allocation.
    """
    from .chained_matmul import _shape

    node = buffer.node
    swizzle = packet_xor_swizzle(buffer.shape)
    if (
        swizzle is None
        or buffer.kind != "frontier"
        or buffer.dtype not in (torch.float16, torch.bfloat16)
        or node is None
        or node.target is not memory_ops.load
        or node.meta["val"].dtype != buffer.dtype
        or _shape(node) != buffer.shape
        or region.name != buffer.name
        or region.byte_offset % 16
        or region.byte_size < math.prod(buffer.shape) * buffer.dtype.itemsize
    ):
        return None
    return PacketXorVectorSink(
        buffer.name,
        (buffer.shape[0], buffer.shape[1]),
        "cutlass.Float16" if buffer.dtype == torch.float16 else "cutlass.BFloat16",
        swizzle,
    )
