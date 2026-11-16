"""Trace-time checks for the split-K workspace producer's atomic epilogue.

The reducer-free epilogue (``split_k_workspace_codegen.
_rewrite_producer_for_atomic_epilogue``) reads the staged epilogue subtile
back from shared memory and addresses the output with static arithmetic that
mirrors CUTLASS's epilogue tiler (``compute_epilogue_tile_shape``: a
``tile_m:1`` M mode and a ``(tile_n // warp_n, warp_n):(1, cta_n // warp_n)``
N mode).  The generated kernel calls :func:`check_epilogue_tiler` on the tiler
it actually built, while it is traced (the layouts are static, so ``cute.size``
and ``cute.crd2idx`` return Python ints): a CUTLASS version that tiles the
epilogue differently fails the compile with a clear message instead of
reducing into the wrong output addresses.
"""

from __future__ import annotations

import cutlass.cute as cute


def check_epilogue_tiler(
    tile: object,
    *,
    rows: int,
    tile_m: int,
    tile_n: int,
    run: int,
    run_stride: int,
) -> None:
    """Fail the trace unless ``tile`` (the ``compute_epilogue_tile_shape``
    result) maps subtile row ``r`` to CTA row ``r`` and subtile column ``c``
    to CTA column ``c % run + c // run * run_stride`` for ``tile_m`` rows and
    ``tile_n`` columns, the mapping the atomic epilogue's addresses assume,
    and the subtile rows divide the CTA's ``rows``."""
    if not isinstance(tile, tuple) or len(tile) != 2:
        raise ValueError(
            f"split-K atomic epilogue: expected a (M, N) epilogue tiler, got {tile!r}"
        )
    m_layout, n_layout = tile
    actual_m, actual_n = cute.size(m_layout), cute.size(n_layout)
    if (
        not isinstance(actual_m, int)
        or not isinstance(actual_n, int)
        or (actual_m, actual_n) != (tile_m, tile_n)
        or rows % tile_m
    ):
        raise ValueError(
            "split-K atomic epilogue: the epilogue tiler covers "
            f"{actual_m}x{actual_n}, the epilogue walks {tile_m}x{tile_n} "
            f"subtiles of a {rows}-row CTA tile"
        )
    for row in range(tile_m):
        index = cute.crd2idx(row, m_layout)
        if not isinstance(index, int) or index != row:
            raise ValueError(
                f"split-K atomic epilogue: tiler row {row} maps to {index!r}, "
                f"the epilogue assumes row {row}"
            )
    for column in range(tile_n):
        expected = column % run + column // run * run_stride
        index = cute.crd2idx(column, n_layout)
        if not isinstance(index, int) or index != expected:
            raise ValueError(
                f"split-K atomic epilogue: tiler column {column} maps to "
                f"{index!r}, the epilogue assumes column {expected} "
                f"(runs of {run} every {run_stride})"
            )
