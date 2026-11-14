"""The layout ``cute_unpadded_rows`` runs a persistent reduction row in.

CuTe pads a persistent reduction row whose static length is not a power of
two to the next one and masks the padding, and a masked row stays on one warp
walking scalar lanes.  The knob instead runs such a row unpadded when the
config gives it a whole number of warps that one vector per thread covers
exactly: 768 columns as 96 threads of 8 elements is the multi-warp vector row,
combined across warps through shared memory.  The config normalization and the
reduction emitter decide it with the same predicate.
"""

from __future__ import annotations

from .reduction_lanes import WARP_THREADS


def unpadded_row_layout(threads: object, vec: object, numel: int) -> bool:
    """Whether ``threads`` threads of ``vec`` elements each cover a row of
    ``numel`` elements exactly, in whole warps."""
    return (
        isinstance(threads, int)
        and isinstance(vec, int)
        and vec > 1
        and threads > 0
        and threads % WARP_THREADS == 0
        and threads * vec == numel
    )
