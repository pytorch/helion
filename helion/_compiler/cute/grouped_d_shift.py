"""Bounded store-only rank-3 TensorMaps for live grouped row intervals."""

from __future__ import annotations


def bounded_d_shift_index_domain(m: int, n: int, row_tile: int) -> bool:
    """Bound element offsets and every padded signed TMA coordinate.

    A descriptor based at D-M*stride_m has row extent M and an extra extent
    M+1. For a frontend-clipped interval [start, start+length), coordinates
    (M-length+row, column, start+length) address D+(start+row)*stride_m
    while the row extent rejects every padded row. The shifted pointer is
    never used for a load.
    """
    limit = (1 << 31) - 1
    return (
        min(m, n, row_tile) > 0
        and m + 1 <= limit
        and m * n <= limit
        and m + row_tile - 1 <= limit
    )
