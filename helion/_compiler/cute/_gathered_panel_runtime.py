"""Aligned BF16 page-panel transfers and warp fragment mappings for four warps."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import cast

import cutlass
import cutlass.cute as cute

from . import gathered_panel_layout as panels
from ._flash_row_mma_runtime import mma_bf16
from .device.ptx import cp_async_16
from .device.ptx import cp_async_16_zfill
from .device.ptx import ldmatrix_x4 as _ldmatrix_x4
from .device.ptx import ldmatrix_x4_trans as _ldmatrix_x4_trans

if TYPE_CHECKING:
    from collections.abc import Callable

# The table-driven PTX wrappers return the declared four packed BF16 words.
ldmatrix_x4 = cast(
    "Callable[[object], tuple[cutlass.Uint32, cutlass.Uint32, cutlass.Uint32, cutlass.Uint32]]",
    _ldmatrix_x4,
)
ldmatrix_x4_trans = cast(
    "Callable[[object], tuple[cutlass.Uint32, cutlass.Uint32, cutlass.Uint32, cutlass.Uint32]]",
    _ldmatrix_x4_trans,
)


@cute.jit
def _address(
    base: cute.Numeric,
    row: cute.Numeric,
    packet: cute.Numeric,
    width: cutlass.Constexpr,
) -> cutlass.Int32:
    # Rows contain a power-of-two number of naturally aligned 16-byte
    # packets. XOR rotates their banks without changing the lane's bytes.
    col = packet ^ (row % (width // 8))
    return base + (row * (width // 8) + col) * 16


@cute.jit
def _issue_qk(
    qbase: cute.Numeric,
    kbase: cute.Numeric,
    qpitch: cute.Numeric,
    kpitch: cute.Numeric,
    panel: cute.Numeric,
    group: cutlass.Constexpr,
    stage: cute.Numeric,
    tid: cute.Numeric,
    valid_k: cutlass.Boolean,
) -> None:
    # Every Q/K panel has 64 depth columns: one packet per lane for
    # 16 rows of Q, and four packets per lane for 64 rows of K.
    row, col = tid // 8, tid % 8
    cp_async_16(
        _address(stage, row, col, 64),
        qbase + ((group * 16 + row) * qpitch + panel * 64 + col * 8) * 2,
    )
    for chunk in cutlass.range_constexpr(4):
        i = tid + 128 * chunk
        row, col = i // 8, i % 8
        cp_async_16_zfill(
            _address(stage + 2048, row, col, 64),
            kbase + (row * kpitch + panel * 64 + col * 8) * 2,
            cutlass.Int32(valid_k) * 16,
        )


@cute.jit
def query_key(
    scores: cute.Tensor,
    q: cute.Tensor,
    k: cute.Tensor,
    physical: cute.Numeric,
    kpage: cute.Numeric,
    qpitch: cute.Numeric,
    kpitch: cute.Numeric,
    group: cutlass.Constexpr,
    slab: cute.Numeric,
    tid: cute.Numeric,
    valid_k: cutlass.Boolean,
    depth: cutlass.Constexpr,
    stages: cutlass.Constexpr,
) -> None:
    # Each 10-KiB stage holds 16x64 Q and 64x64 K. An invalid physical
    # page uses a valid base for the zero-byte copy; the V extent is independent.
    qbase = cutlass.Int64(cast("cute.Pointer", q.iterator).toint())
    safe_k = cutlass.Int64(physical if valid_k else cutlass.Int64(0))
    kbase = cutlass.Int64(cast("cute.Pointer", k.iterator).toint()) + safe_k * kpage * 2
    lane = tid % 32
    quadrant = lane // 8
    warp = tid // 32
    for warm in cutlass.range_constexpr(stages - 1):
        if cutlass.const_expr(warm < depth // 64):
            _issue_qk(
                qbase,
                kbase,
                qpitch,
                kpitch,
                warm,
                group,
                slab + warm * panels.STAGE_BYTES,
                tid,
                valid_k,
            )
        cute.arch.cp_async_commit_group()
    for panel in range(depth // 64):
        future = panel + stages - 1
        if future < depth // 64:
            _issue_qk(
                qbase,
                kbase,
                qpitch,
                kpitch,
                future,
                group,
                slab + (future % stages) * panels.STAGE_BYTES,
                tid,
                valid_k,
            )
        # Empty prologue/tail groups keep a fixed distance even if K has
        # fewer 64-column panels than the number of prefetched groups.
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(stages - 1)
        cute.arch.sync_threads()
        cur = slab + (panel % stages) * panels.STAGE_BYTES
        for step in cutlass.range_constexpr(4):
            a0, a1, a2, a3 = ldmatrix_x4(
                _address(
                    cur, lane % 8 + (quadrant % 2) * 8, step * 2 + quadrant // 2, 64
                )
            )
            krow = lane % 8 + 8 * ((quadrant // 2) * 4 + warp)
            b0, b1, b2, b3 = ldmatrix_x4(
                _address(cur + 2048, krow, step * 2 + quadrant % 2, 64)
            )
            c0, c1, c2, c3 = mma_bf16(
                a0, a1, a2, a3, b0, b1, scores[0], scores[1], scores[2], scores[3]
            )
            scores[0], scores[1], scores[2], scores[3] = c0, c1, c2, c3
            c0, c1, c2, c3 = mma_bf16(
                a0, a1, a2, a3, b2, b3, scores[4], scores[5], scores[6], scores[7]
            )
            scores[4], scores[5], scores[6], scores[7] = c0, c1, c2, c3
        cute.arch.sync_threads()
    cute.arch.cp_async_wait_group(0)


@cute.jit
def issue_values(
    slab: cute.Numeric,
    vals: cute.Tensor,
    physical: cute.Numeric,
    page: cute.Numeric,
    row_pitch: cute.Numeric,
    dim0: cute.Numeric,
    tid: cute.Numeric,
    valid: cutlass.Boolean,
    width: cute.Numeric,
) -> None:
    # Recycle the two consumed QK stages for a 64x128 V panel. Its transfer
    # can overlap the unmodified FP32 online softmax and denominator.
    base = (
        cutlass.Int64(cast("cute.Pointer", vals.iterator).toint())
        + cutlass.Int64(physical if valid else cutlass.Int64(0)) * page * 2
    )
    for chunk in cutlass.range_constexpr(8):
        index = tid + 128 * chunk
        row, col = index // 16, index % 16
        vcol = cutlass.Int64(dim0) + col * 8
        live = valid & (vcol >= 0) & (vcol + 7 < width)
        cp_async_16_zfill(
            _address(slab, row, col, 128),
            base + (row * row_pitch + vcol) * 2,
            cutlass.Int32(live) * 16,
        )
    cute.arch.cp_async_commit_group()


@cute.jit
def probability_value(
    slab: cute.Numeric,
    probs: cute.Tensor,
    tid: cute.Numeric,
    nc: cutlass.Constexpr,
    pitch: cutlass.Constexpr,
    c0: cutlass.Float32,
    c1: cutlass.Float32,
    c2: cutlass.Float32,
    c3: cutlass.Float32,
) -> tuple[cutlass.Float32, cutlass.Float32, cutlass.Float32, cutlass.Float32]:
    lane = tid % 32
    quadrant = lane // 8
    pbase = cutlass.Int32(cast("cute.Pointer", probs.iterator).toint())
    for pair in cutlass.range_constexpr(2):
        row = pair * 32 + quadrant * 8 + lane % 8
        b0, b1, b2, b3 = ldmatrix_x4_trans(_address(slab, row, nc * 4 + tid // 32, 128))
        for half in cutlass.range_constexpr(2):
            prow = lane % 8 + (quadrant % 2) * 8
            col = pair * 32 + 16 * half + (quadrant // 2) * 8
            a0, a1, a2, a3 = ldmatrix_x4(pbase + (prow * pitch + col) * 2)
            if cutlass.const_expr(half == 0):
                c0, c1, c2, c3 = mma_bf16(a0, a1, a2, a3, b0, b1, c0, c1, c2, c3)
            else:
                c0, c1, c2, c3 = mma_bf16(a0, a1, a2, a3, b2, b3, c0, c1, c2, c3)
    return c0, c1, c2, c3
