"""Physical image for a complete computed prefix's optional input phase."""

from __future__ import annotations

from dataclasses import dataclass

from .positional_root import _Unsupported

KEY = "cute_computed_scan_schedule"


@dataclass(frozen=True)
class StagedScanLayout:
    shape: tuple[int, int]
    axis: int
    extent: int
    threads: int
    strides: tuple[int, int]
    elements: int
    allocated_bytes: int

    def check(self) -> None:
        if self != staged_scan_layout(self.shape, self.axis, self.extent, self.threads):
            raise _Unsupported("computed scan staging layout changed")

    @property
    def expression(self) -> str:
        self.check()
        return f"cute.make_layout({self.shape!r}, stride={self.strides!r})"


def staged_scan_layout(
    shape: tuple[int, ...], axis: int, extent: int, threads: int
) -> StagedScanLayout:
    """One injective FP32 image shared by pointwise staging and warp prefixes.

    The row pitch is odd, making a column's 32 FP32 words occupy distinct
    shared-memory banks. Adjacent columns remain contiguous. Reserving the
    entire final padded row makes the physical footprint explicit; both the
    planner and emitter charge the same 128-byte-rounded allocation.

    A complete CTA stages every physical cell. After its join, one full warp
    owns each independent scan vector, reading each cell before overwriting
    that same cell. Parts and vectors have disjoint addresses. No additional
    barrier is needed between consecutive parts of one vector.
    """
    if (
        len(shape) != 2
        or any(type(size) is not int or size <= 0 for size in shape)
        or type(axis) is not int
        or axis not in (0, 1)
        or type(extent) is not int
        or not 1 <= extent <= shape[axis]
        or type(threads) is not int
        or threads < 32
        or threads % 32
    ):
        raise _Unsupported("computed scan staging requires a complete rank-two tile")
    rows, columns = shape
    pitch = columns | 1
    elements = rows * pitch
    if elements > (2**31 - 1) // 4:
        raise _Unsupported("computed scan staging address exceeds Int32")
    return StagedScanLayout(
        (rows, columns),
        axis,
        extent,
        threads,
        (pitch, 1),
        elements,
        (elements * 4 + 127) // 128 * 128,
    )
