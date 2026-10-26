# ruff: noqa: ANN001, ANN202
"""SM100 tcgen05 schedule for the gated-delta-rule chunk recurrence.

One CTA owns one ``(dstate tile, head, batch)`` state ``state^T [MMA_M,
DHEAD]`` kept in fp32 TMEM for the whole sequence.  Every tcgen05 row is one
dstate index and every warp may only touch the TMEM quadrant ``warp % 4``
(its SM sub-partition), so a dstate tile narrower than the MMA tile is
replicated into the quadrants it does not reach: all four sub-partitions then
share the per-chunk epilogue, each replica's warps handling a slice of the
chunk's tokens.  Per chunk of ``CHUNK`` tokens the epilogue warps:

* load the fp32 state, pack its bf16 image into TMEM (the ``A`` operand of the
  projection) and, after a named barrier, warp 0 issues ``acc = image @
  w_chunk^T`` (``M = MMA_M``, ``N = CHUNK``, ``K = DHEAD``; ``w`` is the
  K-major B operand) straight from the epilogue, one commit per token group;
* rescale the state by ``exp(g_last)`` for the update while the projection
  runs, read their token slice of ``acc`` as soon as its group's projection
  has committed, pack ``bf16((u - acc) * gate)`` into the update image (into
  every replica's rows of a shared-memory K-major tile, or straight into TMEM
  when the tile is not replicated) and arrive on their token group's named
  barrier; warp 0 waits on the groups in order and issues each group's K
  steps of ``state += update_image @ k_chunk`` (``N = DHEAD``, ``K = CHUNK``;
  ``k`` is the MN-major B operand), so the update of the first groups runs
  while the last groups still compute, and commits once after the last group.

``TOKEN_GROUPS`` (the ``cute_gdn_recurrence_token_groups`` knob) is that
pipelining depth: one group is the plain two-barrier schedule, more groups
trade extra commits and barriers for overlap.

``MMA_M`` (the ``cute_gdn_recurrence_mma_m`` knob) is the tcgen05 tile
height.  With ``M = 128`` a TMEM quadrant holds 32 rows, one per lane, and
the epilogue moves them with the ``32x32b`` shapes.  With ``M = 64`` (tiles of
at most 64 rows) a quadrant holds 16 rows in its lanes 0-15 and the epilogue
uses the 16-lane data path: thread ``4a + b`` of a warp holds rows ``a`` and
``a + 8`` of its quadrant and, of every eight consecutive columns, columns
``2b`` and ``2b + 1`` (``16x256b`` for fp32 columns, ``16x128b`` for the
packed bf16 pairs they become; the store warps read the image through
``16x64b``, one row and one word of every two per thread).  Both MMAs then
run at half the tensor time, the update image halves, and a 64-row tile
needs no replica at all.

The first chunk starts from the zero state, so its projection is exactly zero
and is never issued: only its ``k`` and ``u`` tiles are loaded and the epilogue
computes ``u * gate`` without an accumulator round trip.  The last chunk only
stores ``h``: its projection and update would feed a state nobody reads.

For a replicated tile one store warp per TMEM quadrant reads its replica's
share of the chunk's bf16 state image back from TMEM once the projection has
consumed it and writes it to ``h``, so the ``h`` store never sits on the
epilogue's critical path; the store warps arrive on the same barrier as the
update's commit, so the epilogue's wait for the updated state also proves the
image slot is free.  An unreplicated tile has no spare replica: its epilogue
warps store their own columns while the update MMA runs.

The ``tcgen05.commit`` round trips (projection group done, update done) are the
only mbarrier hops left on the per-chunk critical path.

The TMA warp streams ``w``/``k``/``u`` chunk tiles through a ``STAGES``-deep
shared-memory ring, issuing the first ``STAGES`` chunks before the CTA sync so
the loads overlap the setup (the other warps prefetch the first chunk's rows
into L2 at kernel entry, a few hundred nanoseconds before that burst can
issue), stages the raw ``g`` rows of a chunk through the same ring with
``cp.async`` a full ring depth ahead, and publishes
the per-row gate vector ``exp(g_last - g_row)`` (the ordinary exact ``exp2(x *
log2 e)`` lowering) together with the tile through the stage's full barrier.
Every computed chunk is full: only the last chunk can be ragged, and it is
never computed, so the source kernel's ``row < seqlen`` gate mask holds for
every row the device code touches and it carries no mask of its own.  One
``w``/``k`` tile row is one swizzle-atom row
(32/64/128 bytes for ``DHEAD`` 16/32/64), which is what lets the same
TMA-written tile serve as the K-major B of the projection and the MN-major B
of the update.

Roles: warps ``[0, EPI_WARPS)`` epilogue (``EPI_WARPS // 4`` column slices per
quadrant), warps ``EPI_WARPS + q`` the ``h`` store warp of quadrant ``q`` (when
the tile is replicated), then the TMA + gate producer warp.
"""

from __future__ import annotations

from typing import cast

import cutlass
import cutlass.cute as cute
import cutlass.experimental.cuda as cuda
import cutlass.experimental.primitives as prims

from .affine_recurrence_primitives import pack_bf16x2
from .affine_recurrence_primitives import prefetch_global_l2
from .gdn_recurrence_geometry import GDN_HALF_LANE_COLS
from .gdn_recurrence_geometry import GDN_MAX_UNROLLED_K_STEPS
from .gdn_recurrence_geometry import GDN_MMA_K
from .gdn_recurrence_geometry import gdn_acc_load_cols
from .gdn_recurrence_geometry import gdn_active_quadrants
from .gdn_recurrence_geometry import gdn_cta_warps
from .gdn_recurrence_geometry import gdn_half_lanes
from .gdn_recurrence_geometry import gdn_prefetch_tokens
from .gdn_recurrence_geometry import gdn_quadrant_rows
from .gdn_recurrence_geometry import gdn_state_replicas
from .gdn_recurrence_geometry import gdn_store_warps
from .gdn_recurrence_geometry import gdn_tma_transaction_bytes
from .gdn_recurrence_geometry import gdn_token_splits
from .gdn_recurrence_geometry import gdn_tokens_per_split

THREADS_PER_WARP: int = 32
LOG2_E: float = 1.4426950408889634
BF16_BYTES: int = 2
F32_BYTES: int = 4
SWIZZLE_ATOM_ROWS: int = 8
SMEM_TILE_ALIGNMENT_BYTES: int = 1024
K_MAJOR_LEADING_BYTES: int = 16
K_STEP_BYTES_K_MAJOR: int = GDN_MMA_K * BF16_BYTES
IMAGE_COLS_PER_K_STEP: int = GDN_MMA_K // 2
TMEM_SPACE: int = 6
# The shared-memory update image is the canonical unswizzled K-major tcgen05
# operand: 8-row x 16-byte core matrices, adjacent along K at CORE_MATRIX_BYTES
# and along M at ``CHUNK * 16`` bytes.
CORE_MATRIX_BYTES: int = 128
CORE_MATRIX_ROWS: int = 8
CORE_MATRIX_WORDS: int = CORE_MATRIX_BYTES // 4
K_STEP_BYTES_SMEM_IMAGE: int = (GDN_MMA_K // CORE_MATRIX_ROWS) * CORE_MATRIX_BYTES
# The 16-lane TMEM data path pairs lane ``a`` with lane ``a + 8`` in one thread.
HALF_LANE_ROW_STRIDE: int = 8
EPILOGUE_BARRIER_ID: int = 1
# Named barrier ``GROUP_BARRIER_ID + g`` syncs token group ``g`` (plus warp 0).
GROUP_BARRIER_ID: int = 2
# TMA tensor-map mode order for the (batch, seq, head, feature) inputs: the
# feature axis is innermost so the tile row is the contiguous swizzle-atom row.
TMA_STRIDE_ORDER: tuple[int, int, int, int] = (3, 2, 1, 0)


def thread_regs(cols: int, half_lanes: bool) -> int:
    """Registers one thread holds of a per-lane vector of ``cols`` columns.

    ``32x32b``: the thread is its lane and holds every column.  16-lane data
    path: the four threads sharing a lane pair each hold two columns of every
    eight, for two rows.
    """

    return cols // 2 if half_lanes else cols


def reg_row(reg: int, half_lanes: bool) -> int:
    """0 for the thread's first row (lane ``a``), 1 for its second (``a + 8``)."""

    return (reg % 4) // 2 if half_lanes else 0


def reg_col(reg: int, half_lanes: bool) -> int:
    """Static part of the lane column register ``reg`` holds.

    The 16-lane data path adds the thread's ``2 * (lane % 4)`` column shift.
    """

    return GDN_HALF_LANE_COLS * (reg // 4) + reg % 2 if half_lanes else reg


def reg_token_slot(reg: int, half_lanes: bool) -> int:
    """Index of the thread's token (column) that register ``reg`` belongs to;
    the two rows of a column share the slot."""

    return 2 * (reg // 4) + reg % 2 if half_lanes else reg


def slot_col(slot: int, half_lanes: bool) -> int:
    """Static part of the lane column of token slot ``slot``."""

    return GDN_HALF_LANE_COLS * (slot // 2) + slot % 2 if half_lanes else slot


def tcgen05_swizzle_for_row_bytes(row_bytes: int) -> prims.Tcgen05SmemSwizzle:
    """Descriptor swizzle whose atom row is exactly one tile row."""

    if row_bytes == 32:
        return prims.Tcgen05SmemSwizzle.SWIZZLE_32B
    if row_bytes == 64:
        return prims.Tcgen05SmemSwizzle.SWIZZLE_64B
    if row_bytes == 128:
        return prims.Tcgen05SmemSwizzle.SWIZZLE_128B
    raise ValueError(f"unsupported gdn factor tile row of {row_bytes} bytes")


def tma_swizzle_for_row_bytes(row_bytes: int) -> cuda.TensorMapSwizzle:
    """TMA swizzle matching :func:`tcgen05_swizzle_for_row_bytes`."""

    if row_bytes == 32:
        return cuda.TensorMapSwizzle.s32b
    if row_bytes == 64:
        return cuda.TensorMapSwizzle.s64b
    if row_bytes == 128:
        return cuda.TensorMapSwizzle.s128b
    raise ValueError(f"unsupported gdn factor tile row of {row_bytes} bytes")


@cute.jit
def mbarrier_wait(mbar, parity) -> None:
    """Spin until the mbarrier phase with ``parity`` has completed."""

    while not prims.mbarrier_wait_parity(mbar, parity, prims.MBarrierWait.TRY):
        pass


@cute.jit
def tmem_row_ptr(tmem_base, quadrant, column, dtype: cutlass.Constexpr):
    """TMEM pointer at lane ``32 * quadrant`` and absolute column ``column``."""

    address = ((quadrant * THREADS_PER_WARP) << 16) | (tmem_base + column)
    return cutlass.inttoptr(address, TMEM_SPACE, dtype)


@cute.jit
def load_f32_columns(ptr, COLS: cutlass.Constexpr, HALF_LANES: cutlass.Constexpr):
    """This thread's registers of the ``COLS`` fp32 lane columns at ``ptr``."""

    if cutlass.const_expr(HALF_LANES):
        values = prims.tcgen05_ld("16x256b", ptr, num=COLS // GDN_HALF_LANE_COLS)
    else:
        values = prims.tcgen05_ld("32x32b", ptr, num=COLS)
    return values


@cute.jit
def store_f32_columns(ptr, values, HALF_LANES: cutlass.Constexpr) -> None:
    """Store fp32 lane columns from the registers :func:`load_f32_columns` fills."""

    if cutlass.const_expr(HALF_LANES):
        prims.tcgen05_st("16x256b", ptr, values)
    else:
        prims.tcgen05_st("32x32b", ptr, values)


@cute.jit
def store_word_columns(ptr, words, HALF_LANES: cutlass.Constexpr) -> None:
    """Store the bf16 pairs packed from :func:`load_f32_columns` registers:
    word ``p`` is columns ``2p`` and ``2p + 1`` of the same row, which is the
    ``16x128b`` (or ``32x32b``) register order."""

    if cutlass.const_expr(HALF_LANES):
        prims.tcgen05_st("16x128b", ptr, words)
    else:
        prims.tcgen05_st("32x32b", ptr, words)


@cute.jit
def load_image_words(ptr, PAIRS: cutlass.Constexpr, HALF_LANES: cutlass.Constexpr):
    """The store warps' registers of ``PAIRS`` packed lane columns at ``ptr``:
    all of one lane, or (``16x64b``) word ``lane % 4 // 2`` of every two for
    lane ``lane // 4 + 8 * (lane % 2)``."""

    if cutlass.const_expr(HALF_LANES):
        words = prims.tcgen05_ld("16x64b", ptr, num=PAIRS // 2)
    else:
        words = prims.tcgen05_ld("32x32b", ptr, num=PAIRS)
    return words


@cute.jit
def epilogue_sync(warps: cutlass.Constexpr) -> None:
    """Named barrier across the epilogue warps."""

    prims.barrier_cta_sync(EPILOGUE_BARRIER_ID, thread_count=THREADS_PER_WARP * warps)


@cute.jit
def advance_ring(stage, phase, STAGES: cutlass.Constexpr):
    """Next ring slot and its parity without a division."""

    wrapped = stage == STAGES - 1
    next_stage = cutlass.Int32(0) if wrapped else stage + 1
    next_phase = phase ^ 1 if wrapped else phase
    return next_stage, next_phase


@cute.jit
def issue_chunk_tiles(
    tma_desc_w,
    tma_desc_k,
    tma_desc_u,
    w_smem,
    k_smem,
    u_smem,
    full_mbar,
    stage,
    t0,
    head,
    batch,
    v_base,
    factor_tile_elems: cutlass.Constexpr,
    value_tile_elems: cutlass.Constexpr,
    with_w: cutlass.Constexpr,
) -> None:
    """TMA the ``w``/``k``/``u`` tiles of the chunk at ``t0`` into ``stage``.

    The transaction bytes are expected later, once the gate is published, so
    the tile can land before its stage's arrive (the tx-count goes negative
    until then, which the mbarrier allows).  The first chunk has no
    projection and skips ``w``.
    """

    if with_w:
        prims.cp_async_bulk_tensor_shared_cta_global(
            w_smem.subview(stage * factor_tile_elems),
            tma_desc_w.get_ptr(),
            [cutlass.Int32(0), head, t0, batch],
            full_mbar.subview(stage),
        )
    prims.cp_async_bulk_tensor_shared_cta_global(
        k_smem.subview(stage * factor_tile_elems),
        tma_desc_k.get_ptr(),
        [cutlass.Int32(0), head, t0, batch],
        full_mbar.subview(stage),
    )
    prims.cp_async_bulk_tensor_shared_cta_global(
        u_smem.subview(stage * value_tile_elems),
        tma_desc_u.get_ptr(),
        [v_base, head, t0, batch],
        full_mbar.subview(stage),
    )


@cute.jit
def stage_gate_rows(
    g,
    raw_g_smem,
    stage,
    lane,
    t0,
    batch,
    head,
    CHUNK: cutlass.Constexpr,
) -> None:
    """``cp.async`` the chunk's raw ``g`` rows into the stage's staging slot.

    Only computed chunks are staged and those are always full (only the
    never-computed last chunk can be ragged), so every row is in range.
    """

    for row_base in cutlass.range_constexpr(0, CHUNK, THREADS_PER_WARP):
        row = row_base + lane
        if row < CHUNK:
            prims.cp_async_shared_global(
                raw_g_smem.subview(stage * CHUNK + row),
                g.iterator + cute.crd2idx((batch, t0 + row, head), g.layout),
                F32_BYTES,
                prims.LoadCacheModifier.CA,
            )


@cute.jit
def prefetch_first_chunk_rows(
    k,
    u,
    g,
    batch,
    head,
    v_base,
    dstate,
    thread,
    THREADS: cutlass.Constexpr,
    CHUNK: cutlass.Constexpr,
    DHEAD: cutlass.Constexpr,
    BLOCK_V: cutlass.Constexpr,
) -> None:
    """``prefetch.global.L2`` the first chunk's ``k``, ``u`` and ``g`` rows.

    The TMA burst needs the descriptors and barriers first and issues a few
    hundred nanoseconds into the kernel; these requests leave at once, so the
    burst finds its lines in flight (the L2 is cold after a flush).  Rows are
    16-byte aligned, so both ends of every row are touched.  The ``u`` tile
    may be wider than the state it covers (a partial last tile, or a tile the
    block search picked above ``dstate``): the requests stay inside the row.
    """

    u_last = cutlass.min(v_base + BLOCK_V, dstate) - v_base - 1
    for row_base in cutlass.range_constexpr(0, CHUNK, THREADS):
        row = row_base + thread
        if row < CHUNK:
            k_row = k.iterator + cute.crd2idx((batch, row, head, 0), k.layout)
            for offset in cutlass.range_constexpr(0, DHEAD * BF16_BYTES, 128):
                prefetch_global_l2(k_row + offset // BF16_BYTES)
            prefetch_global_l2(k_row + (DHEAD - 1))
            u_row = u.iterator + cute.crd2idx((batch, row, head, v_base), u.layout)
            for offset in cutlass.range_constexpr(0, BLOCK_V * BF16_BYTES, 128):
                if offset // BF16_BYTES <= u_last:
                    prefetch_global_l2(u_row + offset // BF16_BYTES)
            prefetch_global_l2(u_row + u_last)
            prefetch_global_l2(g.iterator + cute.crd2idx((batch, row, head), g.layout))


@cute.jit
def issue_update_step(state_tmem, update_operand, desc_k_step, idesc_update) -> None:
    """One K step of ``state += update_image @ k_chunk`` from the elected lane.

    ``update_operand`` is the step's update-image operand: the advanced
    shared-memory descriptor of a replicated tile, or the TMEM image view of
    an unreplicated tile.
    """

    if prims.elect_sync():
        prims.tcgen05_mma(
            prims.Tcgen05MMAKind.F16,
            prims.CTAGroup.CTA_1,
            state_tmem,
            update_operand,
            desc_k_step,
            idesc_update,
            True,
        )


@cute.jit
def rescale_state(
    state, scale, state_ptr, REGS: cutlass.Constexpr, HALF_LANES: cutlass.Constexpr
) -> None:
    """Store ``state * scale`` back to TMEM (the update accumulates onto it)."""

    scaled = cutlass.Array(cutlass.Float32, REGS, alignment=16)
    for reg in cutlass.range_constexpr(REGS):
        scaled[reg] = state[reg] * scale
    store_f32_columns(state_ptr, scaled[0:REGS], HALF_LANES)


@cute.jit
def store_h_share(
    h,
    state,
    stored_row0,
    stored_row1,
    replica,
    batch,
    chunk,
    head,
    state_col0,
    col_shift,
    v_index0,
    v_index1,
    H_COLS: cutlass.Constexpr,
    REGS: cutlass.Constexpr,
    HALF_LANES: cutlass.Constexpr,
) -> None:
    """Store this thread's part of replica ``replica``'s columns of ``bf16(state)``.

    Every replica's warps hold the whole column slice of their rows; each
    stores ``H_COLS`` of it.  ``.to(BFloat16)`` rounds to nearest even like
    the ``cvt.rn.bf16x2`` of the state image, so this equals the image words
    the store warps write.  With one lane per row the replica index is
    dynamic and each share is a static register slice behind its own branch
    (a dynamic index into the register vector would spill it to local
    memory); on the 16-lane data path a register's column also depends on
    the thread's column shift, so every register carries its own share test.
    """

    if cutlass.const_expr(HALF_LANES):
        for reg in cutlass.range_constexpr(REGS):
            col = reg_col(reg, True) + col_shift
            if reg_row(reg, True) == 0:
                if stored_row0 & (col // H_COLS == replica):
                    h[batch, chunk, head, state_col0 + col, v_index0] = state[reg].to(
                        cutlass.BFloat16
                    )
            else:
                if stored_row1 & (col // H_COLS == replica):
                    h[batch, chunk, head, state_col0 + col, v_index1] = state[reg].to(
                        cutlass.BFloat16
                    )
    else:
        for target in cutlass.range_constexpr(REGS // H_COLS):
            if stored_row0 & (replica == target):
                for col in cutlass.range_constexpr(H_COLS):
                    h[
                        batch, chunk, head, state_col0 + target * H_COLS + col, v_index0
                    ] = state[target * H_COLS + col].to(cutlass.BFloat16)


@cute.jit
def store_h_image(
    h,
    image,
    stored_row,
    batch,
    chunk,
    head,
    col0,
    pair_base,
    v_index,
    REGS: cutlass.Constexpr,
    PAIR_STRIDE: cutlass.Constexpr,
) -> None:
    """Store this thread's ``REGS`` packed bf16 column pairs of the state image
    to ``h``: register ``r`` is pair ``pair_base + r * PAIR_STRIDE``."""

    if stored_row:
        for reg in cutlass.range_constexpr(REGS):
            word = image[reg]
            col = col0 + 2 * (pair_base + reg * PAIR_STRIDE)
            h[batch, chunk, head, col, v_index] = cutlass.Int16(word & 0xFFFF).bitcast(
                cutlass.BFloat16
            )
            h[batch, chunk, head, col + 1, v_index] = cutlass.Int16(word >> 16).bitcast(
                cutlass.BFloat16
            )


@cute.kernel
def gdn_recurrence_kernel(
    tma_desc_w: cutlass.GridConstant[cuda.TensorMap],
    tma_desc_k: cutlass.GridConstant[cuda.TensorMap],
    tma_desc_u: cutlass.GridConstant[cuda.TensorMap],
    k: cute.Tensor,
    u: cute.Tensor,
    g: cute.Tensor,
    h: cute.Tensor,
    num_chunks: cutlass.Int32,
    dstate: cutlass.Int32,
    BLOCK_V: cutlass.Constexpr,
    CHUNK: cutlass.Constexpr,
    DHEAD: cutlass.Constexpr,
    EPI_WARPS: cutlass.Constexpr,
    STAGES: cutlass.Constexpr,
    TOKEN_GROUPS: cutlass.Constexpr,
    MMA_M: cutlass.Constexpr,
    STATE_COL: cutlass.Constexpr,
    STATE_IMAGE_COL: cutlass.Constexpr,
    ACC_COL: cutlass.Constexpr,
    UPDATE_IMAGE_COL: cutlass.Constexpr,
    TMEM_COLS: cutlass.Constexpr,
) -> None:
    tidx, _, _ = cute.arch.thread_idx()
    v_tile, head, batch = cute.arch.block_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    lane = tidx % THREADS_PER_WARP
    # Rows per TMEM quadrant and how a thread reaches them: one lane per row
    # (``32x32b``), or the 16-lane data path with two rows per thread.
    half_lanes: cutlass.Constexpr = gdn_half_lanes(MMA_M)
    quadrant_rows: cutlass.Constexpr = gdn_quadrant_rows(MMA_M)
    rows_per_thread: cutlass.Constexpr = 2 if half_lanes else 1
    quadrants: cutlass.Constexpr = gdn_active_quadrants(BLOCK_V, MMA_M)
    replicas: cutlass.Constexpr = gdn_state_replicas(BLOCK_V, MMA_M)
    slices: cutlass.Constexpr = EPI_WARPS // 4
    token_splits: cutlass.Constexpr = gdn_token_splits(CHUNK, BLOCK_V, EPI_WARPS, MMA_M)
    tokens_per_split: cutlass.Constexpr = gdn_tokens_per_split(
        CHUNK, BLOCK_V, EPI_WARPS, MMA_M
    )
    smem_update_image: cutlass.Constexpr = replicas > 1
    # Store warp q (quadrant q) follows the epilogue warps; the TMA warp is last.
    store_warps: cutlass.Constexpr = gdn_store_warps(BLOCK_V, MMA_M)
    tma_warp: cutlass.Constexpr = EPI_WARPS + store_warps
    threads: cutlass.Constexpr = THREADS_PER_WARP * gdn_cta_warps(
        BLOCK_V, EPI_WARPS, MMA_M
    )
    # Each store warp writes its replica's share of the image's column pairs.
    store_pairs: cutlass.Constexpr = DHEAD // 2 // replicas
    prefetch_tokens: cutlass.Constexpr = gdn_prefetch_tokens(
        CHUNK, BLOCK_V, EPI_WARPS, MMA_M
    )
    # Token groups: the chunk's tokens (projection N, update K) in pipelined
    # slices of ``group_tokens``; ``splits_per_group`` warps compute each.
    group_tokens: cutlass.Constexpr = CHUNK // TOKEN_GROUPS
    group_k_steps: cutlass.Constexpr = group_tokens // GDN_MMA_K
    unrolled_k_steps: cutlass.Constexpr = group_k_steps <= GDN_MAX_UNROLLED_K_STEPS
    splits_per_group: cutlass.Constexpr = token_splits // TOKEN_GROUPS
    group_warps: cutlass.Constexpr = EPI_WARPS // TOKEN_GROUPS
    factor_tile_elems: cutlass.Constexpr = CHUNK * DHEAD
    value_tile_elems: cutlass.Constexpr = CHUNK * BLOCK_V
    tx_bytes: cutlass.Constexpr = gdn_tma_transaction_bytes(CHUNK, DHEAD, BLOCK_V)
    first_tx_bytes: cutlass.Constexpr = tx_bytes - factor_tile_elems * BF16_BYTES
    row_bytes: cutlass.Constexpr = DHEAD * BF16_BYTES
    swizzle_stride_bytes: cutlass.Constexpr = SWIZZLE_ATOM_ROWS * row_bytes
    k_step_bytes_mn_major: cutlass.Constexpr = GDN_MMA_K * row_bytes
    smem_swizzle: cutlass.Constexpr = tcgen05_swizzle_for_row_bytes(row_bytes)
    v_base = v_tile * BLOCK_V
    # The last chunk only stores h: its projection and update feed a state
    # nobody reads, so neither its tiles nor its gate are ever needed.
    computed_chunks = num_chunks - 1

    # Every warp but the TMA warp (which is busy with the descriptors and
    # barriers) gets the first chunk's rows moving towards L2 right away.
    if (warp_idx != tma_warp) & (computed_chunks > 0):
        prefetch_first_chunk_rows(
            k,
            u,
            g,
            batch,
            head,
            v_base,
            dstate,
            tidx,
            THREADS_PER_WARP * (EPI_WARPS + store_warps),
            CHUNK,
            DHEAD,
            BLOCK_V,
        )

    w_smem = cutlass.Array(
        cutlass.BFloat16,
        STAGES * factor_tile_elems,
        space=cutlass.AddressSpace.smem,
        alignment=SMEM_TILE_ALIGNMENT_BYTES,
    )
    k_smem = cutlass.Array(
        cutlass.BFloat16,
        STAGES * factor_tile_elems,
        space=cutlass.AddressSpace.smem,
        alignment=SMEM_TILE_ALIGNMENT_BYTES,
    )
    u_smem = cutlass.Array(
        cutlass.BFloat16,
        STAGES * value_tile_elems,
        space=cutlass.AddressSpace.smem,
        alignment=SMEM_TILE_ALIGNMENT_BYTES,
    )
    gate_smem = cutlass.Array(
        cutlass.Float32,
        STAGES * CHUNK,
        space=cutlass.AddressSpace.smem,
        alignment=16,
    )
    raw_g_smem = cutlass.Array(
        cutlass.Float32,
        STAGES * CHUNK,
        space=cutlass.AddressSpace.smem,
        alignment=16,
    )
    scale_smem = cutlass.Array(
        cutlass.Float32, STAGES, space=cutlass.AddressSpace.smem, alignment=16
    )
    full_mbar = cutlass.Array(
        cutlass.Int64, STAGES, space=cutlass.AddressSpace.smem, alignment=8
    )
    empty_mbar = cutlass.Array(
        cutlass.Int64, STAGES, space=cutlass.AddressSpace.smem, alignment=8
    )
    # One projection commit per token group.
    acc_ready_mbar = cutlass.Array(
        cutlass.Int64, TOKEN_GROUPS, space=cutlass.AddressSpace.smem, alignment=8
    )
    # The update's commit plus the store warps, whose image reads must precede
    # the next chunk's image store.
    state_ready_mbar = cutlass.Array(
        cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8
    )
    tmem_ptr_i32 = cutlass.Array(
        cutlass.Int32, 1, space=cutlass.AddressSpace.smem, alignment=4
    )
    # bf16 pairs of the [MMA_M, CHUNK] update image (replicated tiles only).
    update_smem = cutlass.Array(
        cutlass.Int32,
        (MMA_M * CHUNK // 2) if smem_update_image else 4,
        space=cutlass.AddressSpace.smem,
        alignment=SMEM_TILE_ALIGNMENT_BYTES,
    )

    # Only the ring's full barriers gate the first TMA burst: the TMA warp
    # initializes those and issues, while warp 1 initializes the barriers the
    # CTA sync below publishes to everyone.
    if warp_idx == tma_warp:
        if prims.elect_sync():
            # Start the descriptor fetches first so they overlap the barrier
            # setup instead of the first TMA.
            prims.prefetch_tensormap(tma_desc_w.get_ptr())
            prims.prefetch_tensormap(tma_desc_k.get_ptr())
            prims.prefetch_tensormap(tma_desc_u.get_ptr())
            for stage in cutlass.range_constexpr(STAGES):
                prims.mbarrier_init(full_mbar.subview(stage), 1)
            # The TMA burst below completes transactions on these barriers
            # from the async proxy before the CTA sync: publish the inits to
            # it (``fence.mbarrier_init`` alone only covers the generic proxy).
            cute.arch.fence_proxy(kind="async.shared", space="cta")
    elif warp_idx == 1:
        if prims.elect_sync():
            for stage in cutlass.range_constexpr(STAGES):
                prims.mbarrier_init(empty_mbar.subview(stage), 1)
            for group in cutlass.range_constexpr(TOKEN_GROUPS):
                prims.mbarrier_init(acc_ready_mbar.subview(group), 1)
            prims.mbarrier_init(state_ready_mbar, 1 + store_warps)
    prims.fence_mbarrier_init()
    if warp_idx == tma_warp:
        # Get the first ring's worth of tiles and raw gate rows in flight
        # before the setup barrier: the initializing thread may use its own
        # barriers as soon as the init fence has run.  Exactly STAGES groups
        # (empty past the last chunk) so the ring loop's
        # ``wait_group(STAGES - 1)`` always covers the chunk it consumes.
        for chunk in cutlass.range_constexpr(STAGES):
            if chunk < computed_chunks:
                if prims.elect_sync():
                    # The first chunk has no projection: no w tile.
                    issue_chunk_tiles(
                        tma_desc_w,
                        tma_desc_k,
                        tma_desc_u,
                        w_smem,
                        k_smem,
                        u_smem,
                        full_mbar,
                        chunk,
                        chunk * CHUNK,
                        head,
                        batch,
                        v_base,
                        factor_tile_elems,
                        value_tile_elems,
                        chunk > 0,
                    )
                stage_gate_rows(
                    g, raw_g_smem, chunk, lane, chunk * CHUNK, batch, head, CHUNK
                )
            prims.cp_async_commit_group()
    if warp_idx == 0:
        prims.tcgen05_alloc(tmem_ptr_i32, TMEM_COLS, group=prims.CTAGroup.CTA_1)
        prims.tcgen05_relinquish_alloc_permit(group=prims.CTAGroup.CTA_1)
    prims.barrier_cta_sync(0, thread_count=threads)
    tmem_base = tmem_ptr_i32.load()

    if warp_idx == tma_warp:
        # ================= TMA ring + per-row gate producer =================
        stage = cutlass.Int32(0)
        ring_phase = cutlass.Int32(0)
        for chunk in cutlass.range(computed_chunks, unroll=1):
            t0 = chunk * CHUNK
            if chunk >= STAGES:
                mbarrier_wait(empty_mbar.subview(stage), ring_phase ^ 1)
                if prims.elect_sync():
                    issue_chunk_tiles(
                        tma_desc_w,
                        tma_desc_k,
                        tma_desc_u,
                        w_smem,
                        k_smem,
                        u_smem,
                        full_mbar,
                        stage,
                        t0,
                        head,
                        batch,
                        v_base,
                        factor_tile_elems,
                        value_tile_elems,
                        True,
                    )
            # The raw g rows of this chunk were committed STAGES groups ago
            # (every iteration commits exactly one group, empty or not).
            prims.cp_async_wait_group(STAGES - 1)
            prims.bar_warp_sync(cute.arch.FULL_MASK)
            gate_base = stage * CHUNK
            # A computed chunk is always full, so its last token is its last
            # row and no row needs the kernel's ``row < seqlen`` mask.
            g_last = raw_g_smem[gate_base + CHUNK - 1]
            for row_base in cutlass.range_constexpr(0, CHUNK, THREADS_PER_WARP):
                row = row_base + lane
                # A 16-token chunk fills only half the warp: lanes past the
                # chunk must not write the next stage's gate slots.
                if row < CHUNK:
                    gate_smem[gate_base + row] = cute.math.exp2(
                        (g_last - raw_g_smem[gate_base + row]) * LOG2_E
                    )
            if lane == 0:
                scale_smem[stage] = cute.math.exp2(g_last * LOG2_E)
            prims.bar_warp_sync(cute.arch.FULL_MASK)
            if prims.elect_sync():
                if chunk == 0:
                    # No w tile for the first chunk.
                    prims.mbarrier_arrive_expect_tx(
                        full_mbar.subview(stage), first_tx_bytes
                    )
                else:
                    prims.mbarrier_arrive_expect_tx(full_mbar.subview(stage), tx_bytes)
            # Refill the staging slot just consumed with the raw rows of the
            # chunk that will reuse it, a full ring depth ahead.
            next_chunk = chunk + STAGES
            if next_chunk < computed_chunks:
                stage_gate_rows(
                    g,
                    raw_g_smem,
                    stage,
                    lane,
                    next_chunk * CHUNK,
                    batch,
                    head,
                    CHUNK,
                )
            prims.cp_async_commit_group()
            stage, ring_phase = advance_ring(stage, ring_phase, STAGES)
    elif warp_idx < EPI_WARPS:
        # ================== TMEM epilogue + tcgen05 issue ===================
        quadrant = warp_idx % 4
        slice_index = warp_idx // 4
        # Quadrant q holds replica q // quadrants of row group q % quadrants.
        row_group = quadrant % quadrants
        replica = quadrant // quadrants
        state_cols: cutlass.Constexpr = DHEAD // slices
        state_regs: cutlass.Constexpr = thread_regs(state_cols, half_lanes)
        acc_load_cols: cutlass.Constexpr = gdn_acc_load_cols(
            CHUNK, BLOCK_V, EPI_WARPS, MMA_M
        )
        acc_load_blocks: cutlass.Constexpr = tokens_per_split // acc_load_cols
        acc_regs: cutlass.Constexpr = thread_regs(acc_load_cols, half_lanes)
        prefetch_regs: cutlass.Constexpr = thread_regs(prefetch_tokens, half_lanes)
        prefetch_slots: cutlass.Constexpr = prefetch_regs // rows_per_thread
        # Replicas split the columns of the slice for the h store.
        h_splits: cutlass.Constexpr = min(replicas, state_cols)
        h_cols: cutlass.Constexpr = state_cols // h_splits
        # This thread's rows of the quadrant: its lane, or (16-lane data path)
        # lanes ``lane // 4`` and ``lane // 4 + 8`` with columns ``2 * (lane
        # % 4)`` and the next of every eight.
        lane_row = lane // 4 if cutlass.const_expr(half_lanes) else lane
        col_shift = (
            2 * (lane % 4) if cutlass.const_expr(half_lanes) else cutlass.Int32(0)
        )
        row_local0 = row_group * quadrant_rows + lane_row
        row_local1 = row_local0 + HALF_LANE_ROW_STRIDE
        v_index0 = v_base + row_local0
        v_index1 = v_base + row_local1
        # Rows past the dstate tile are TMEM padding: they compute on a clamped
        # copy of the last valid u row and are never stored.
        stored_row0 = (row_local0 < BLOCK_V) & (v_index0 < dstate)
        stored_row1 = (row_local1 < BLOCK_V) & (v_index1 < dstate)
        u_row0 = cutlass.min(row_local0, BLOCK_V - 1)
        u_row1 = cutlass.min(row_local1, BLOCK_V - 1)
        state_col0 = slice_index * state_cols
        # This warp's token slice of the update; warps past the split count
        # (tiny chunks) only keep the state and h.
        split = replica * slices + slice_index
        computes = split < token_splits
        token0 = split * tokens_per_split
        # The token group this warp's slice belongs to (a warp that does not
        # compute joins group 0, which is then the only group).
        group = cutlass.min(split, token_splits - 1) // splits_per_group
        issues = warp_idx == 0
        # Word index of each of this thread's rows in every replica of the
        # update tile: core matrices of 8 rows x 16 bytes, CHUNK / 8 of them
        # per row group.
        update_row_words = cutlass.Array(cutlass.Int32, replicas * rows_per_thread)
        for target in cutlass.range_constexpr(replicas):
            for row in cutlass.range_constexpr(rows_per_thread):
                m = (
                    (target * quadrants + row_group) * quadrant_rows
                    + lane_row
                    + row * HALF_LANE_ROW_STRIDE
                )
                update_row_words[target * rows_per_thread + row] = (
                    m // CORE_MATRIX_ROWS
                ) * (CHUNK // CORE_MATRIX_ROWS) * CORE_MATRIX_WORDS + (
                    m % CORE_MATRIX_ROWS
                ) * (CORE_MATRIX_BYTES // CORE_MATRIX_ROWS // 4)
        state_ptr = tmem_row_ptr(
            tmem_base, quadrant, STATE_COL + state_col0, cutlass.Float32
        )
        state_image_ptr = tmem_row_ptr(
            tmem_base, quadrant, STATE_IMAGE_COL + state_col0 // 2, cutlass.Int32
        )
        idesc_projection = prims.Tcgen05InstrDesc.build(
            c_dtype=cutlass.Float32,
            a_dtype=cutlass.BFloat16,
            b_dtype=cutlass.BFloat16,
            n_dim=group_tokens,
            m_dim=MMA_M,
            a_major=0,
            b_major=0,
        )
        idesc_update = prims.Tcgen05InstrDesc.build(
            c_dtype=cutlass.Float32,
            a_dtype=cutlass.BFloat16,
            b_dtype=cutlass.BFloat16,
            n_dim=DHEAD,
            m_dim=MMA_M,
            a_major=0,
            b_major=1,
        )
        acc_group_tmem = [
            cutlass.inttoptr(
                tmem_base + ACC_COL + target * group_tokens, TMEM_SPACE, cutlass.Float32
            )
            for target in range(TOKEN_GROUPS)
        ]
        state_tmem = cutlass.inttoptr(
            tmem_base + STATE_COL, TMEM_SPACE, cutlass.Float32
        )
        image_tmem = prims.make_tmem_ptr(tmem_base, cutlass.Int8)

        # Zero the state: the first chunk stores it as h and the first
        # update accumulates onto it.
        zeros = cutlass.Array(cutlass.Float32, state_regs, alignment=16)
        for reg in cutlass.range_constexpr(state_regs):
            zeros[reg] = cutlass.Float32(0.0)
        store_f32_columns(state_ptr, zeros[0:state_regs], half_lanes)
        prims.tcgen05_wait(kind=prims.Tcgen05Wait.STORE)
        prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
        stage = cutlass.Int32(0)
        ring_phase = cutlass.Int32(0)
        for chunk in cutlass.range(computed_chunks, unroll=1):
            parity = chunk & 1
            later = chunk > 0
            if later:
                mbarrier_wait(state_ready_mbar, parity ^ 1)
                prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            state = load_f32_columns(state_ptr, state_cols, half_lanes)
            # The tile and gate normally landed long ago; check while the
            # state load is in flight.
            mbarrier_wait(full_mbar.subview(stage), ring_phase)
            prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)
            scale = scale_smem[stage]
            if later:
                packed_state = cutlass.Array(
                    cutlass.Int32, state_regs // 2, alignment=16
                )
                for pair in cutlass.range_constexpr(state_regs // 2):
                    packed_state[pair] = pack_bf16x2(
                        state[2 * pair], state[2 * pair + 1]
                    )
                store_word_columns(
                    state_image_ptr, packed_state[0 : state_regs // 2], half_lanes
                )
                prims.tcgen05_wait(kind=prims.Tcgen05Wait.STORE)
            prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
            epilogue_sync(EPI_WARPS)
            if issues & later:
                prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                desc_w = prims.Tcgen05SmemDesc.build(
                    w_smem.subview(stage * factor_tile_elems),
                    leading_byte_offset=K_MAJOR_LEADING_BYTES,
                    stride_byte_offset=swizzle_stride_bytes,
                    layout=smem_swizzle,
                )
                # One projection per token group over its N slice of the
                # w tile, committed separately so its warps start early.
                for target in cutlass.range_constexpr(TOKEN_GROUPS):
                    for k_step in cutlass.range_constexpr(DHEAD // GDN_MMA_K):
                        if prims.elect_sync():
                            prims.tcgen05_mma(
                                prims.Tcgen05MMAKind.F16,
                                prims.CTAGroup.CTA_1,
                                acc_group_tmem[target],
                                image_tmem.subview(
                                    STATE_IMAGE_COL + k_step * IMAGE_COLS_PER_K_STEP
                                ),
                                desc_w.advance_start_address(
                                    k_step * K_STEP_BYTES_K_MAJOR
                                    + target * group_tokens * row_bytes
                                ),
                                idesc_projection,
                                k_step > 0,
                            )
                    if prims.elect_sync():
                        prims.tcgen05_commit(
                            acc_ready_mbar.subview(target), group=prims.CTAGroup.CTA_1
                        )

            if later:
                # Rescale the state for the update while the projection runs.
                # The store only has to land before the update MMA: the store
                # wait ahead of the group barrier covers it.  (Issuing it
                # before the projection or after the accumulator load puts
                # its issue on the critical path instead.)
                rescale_state(state, scale, state_ptr, state_regs, half_lanes)
            # Read the leading u values and gates of this warp's token slice
            # while the projection runs: only the accumulator is on the
            # critical path.  A wide slice reads the rest as it goes.
            u_base = stage * value_tile_elems
            gate_base = stage * CHUNK
            u_vals = cutlass.Array(cutlass.BFloat16, prefetch_regs)
            gates = cutlass.Array(cutlass.Float32, prefetch_slots)
            if computes:
                for slot in cutlass.range_constexpr(prefetch_slots):
                    gates[slot] = gate_smem[
                        gate_base + token0 + slot_col(slot, half_lanes) + col_shift
                    ]
                for reg in cutlass.range_constexpr(prefetch_regs):
                    u_vals[reg] = u_smem[
                        u_base
                        + (token0 + reg_col(reg, half_lanes) + col_shift) * BLOCK_V
                        + (u_row1 if reg_row(reg, half_lanes) else u_row0)
                    ]
            if later:
                # The first chunk has no projection to wait for.
                mbarrier_wait(acc_ready_mbar.subview(group), (chunk - 1) & 1)
                prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            if computes:
                # Word index of this slice's first token in each replica of
                # the shared-memory update tile (one lane per row), or the
                # thread's word within a core-matrix row (16-lane data path:
                # its two columns of every eight are one word).
                word0 = (token0 // CORE_MATRIX_ROWS) * CORE_MATRIX_WORDS + (
                    token0 % CORE_MATRIX_ROWS
                ) // 2
                word_shift = lane % 4
                # One accumulator block (``acc_load_cols`` tokens) at a time:
                # load, wait, pack its update words and store them before the
                # next block's load, so a wide token slice never holds its
                # whole accumulator and update in registers (a slice of one
                # block, the small-chunk case, is one load and one wait).
                for block in cutlass.range_constexpr(acc_load_blocks):
                    first_token: cutlass.Constexpr = block * acc_load_cols
                    first_reg: cutlass.Constexpr = thread_regs(first_token, half_lanes)
                    # Register pairs of this block whose u and gate are in
                    # registers.  A pair is two adjacent tokens of one row.
                    block_ready: cutlass.Constexpr = (
                        max(0, min(acc_regs, prefetch_regs - first_reg)) // 2
                    )
                    packed_update = cutlass.Array(
                        cutlass.Int32, acc_regs // 2, alignment=16
                    )
                    if later:
                        acc = load_f32_columns(
                            tmem_row_ptr(
                                tmem_base,
                                quadrant,
                                ACC_COL + token0 + first_token,
                                cutlass.Float32,
                            ),
                            acc_load_cols,
                            half_lanes,
                        )
                        prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)
                        for pair in cutlass.range_constexpr(block_ready):
                            reg = first_reg + 2 * pair
                            slot = reg_token_slot(reg, half_lanes)
                            packed_update[pair] = pack_bf16x2(
                                (u_vals[reg].to(cutlass.Float32) - acc[2 * pair])
                                * gates[slot],
                                (
                                    u_vals[reg + 1].to(cutlass.Float32)
                                    - acc[2 * pair + 1]
                                )
                                * gates[slot + 1],
                            )
                        for pair in cutlass.range_constexpr(block_ready, acc_regs // 2):
                            reg = first_reg + 2 * pair
                            token = (
                                token0
                                + first_token
                                + reg_col(2 * pair, half_lanes)
                                + col_shift
                            )
                            u_row = u_row1 if reg_row(reg, half_lanes) else u_row0
                            u0 = u_smem[u_base + token * BLOCK_V + u_row].to(
                                cutlass.Float32
                            )
                            u1 = u_smem[u_base + (token + 1) * BLOCK_V + u_row].to(
                                cutlass.Float32
                            )
                            packed_update[pair] = pack_bf16x2(
                                (u0 - acc[2 * pair]) * gate_smem[gate_base + token],
                                (u1 - acc[2 * pair + 1])
                                * gate_smem[gate_base + token + 1],
                            )
                    else:
                        # The first chunk projects the zero state: ``u - 0``
                        # is ``u`` exactly, so no accumulator round trip.
                        for pair in cutlass.range_constexpr(block_ready):
                            reg = first_reg + 2 * pair
                            slot = reg_token_slot(reg, half_lanes)
                            packed_update[pair] = pack_bf16x2(
                                u_vals[reg].to(cutlass.Float32) * gates[slot],
                                u_vals[reg + 1].to(cutlass.Float32) * gates[slot + 1],
                            )
                        for pair in cutlass.range_constexpr(block_ready, acc_regs // 2):
                            reg = first_reg + 2 * pair
                            token = (
                                token0
                                + first_token
                                + reg_col(2 * pair, half_lanes)
                                + col_shift
                            )
                            u_row = u_row1 if reg_row(reg, half_lanes) else u_row0
                            packed_update[pair] = pack_bf16x2(
                                u_smem[u_base + token * BLOCK_V + u_row].to(
                                    cutlass.Float32
                                )
                                * gate_smem[gate_base + token],
                                u_smem[u_base + (token + 1) * BLOCK_V + u_row].to(
                                    cutlass.Float32
                                )
                                * gate_smem[gate_base + token + 1],
                            )
                    if smem_update_image:
                        if cutlass.const_expr(half_lanes):
                            # Word ``pair`` is core-matrix row (token octet)
                            # ``pair // 2`` of the thread's row ``pair % 2``:
                            # one 32-bit store per word into every replica.
                            block_octet = (token0 + first_token) // CORE_MATRIX_ROWS
                            for target in cutlass.range_constexpr(replicas):
                                for pair in cutlass.range_constexpr(acc_regs // 2):
                                    update_smem[
                                        update_row_words[
                                            target * rows_per_thread + pair % 2
                                        ]
                                        + (block_octet + pair // 2) * CORE_MATRIX_WORDS
                                        + word_shift
                                    ] = packed_update[pair]
                        else:
                            # Every replica's rows of this block, 16 bytes
                            # (one core-matrix row of eight tokens) at a time.
                            words: cutlass.Constexpr = acc_regs // 2
                            words_per_store: cutlass.Constexpr = min(words, 4)
                            block_word0 = (
                                word0
                                + (first_token // CORE_MATRIX_ROWS) * CORE_MATRIX_WORDS
                            )
                            for target in cutlass.range_constexpr(replicas):
                                row_word = update_row_words[target] + block_word0
                                for group_index in cutlass.range_constexpr(
                                    words // words_per_store
                                ):
                                    first_word = group_index * words_per_store
                                    update_smem.store(
                                        tuple(
                                            packed_update[first_word + i]
                                            for i in range(words_per_store)
                                        ),
                                        row_word
                                        + (first_word // 4) * CORE_MATRIX_WORDS,
                                        alignment=4 * words_per_store,
                                    )
                    else:
                        store_word_columns(
                            tmem_row_ptr(
                                tmem_base,
                                quadrant,
                                UPDATE_IMAGE_COL + (token0 + first_token) // 2,
                                cutlass.Int32,
                            ),
                            packed_update[0 : acc_regs // 2],
                            half_lanes,
                        )
            if smem_update_image:
                # The MMA reads the update image through the async proxy.
                cute.arch.fence_proxy(kind="async.shared", space="cta")
            prims.tcgen05_wait(kind=prims.Tcgen05Wait.STORE)
            prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
            # Each token group signals its own named barrier without waiting
            # (``bar.arrive``); warp 0 waits on the groups in order and
            # issues each group's K steps as soon as it is complete.
            if not issues:
                for target in cutlass.range_constexpr(TOKEN_GROUPS):
                    if group == target:
                        prims.barrier_cta_arrive(
                            GROUP_BARRIER_ID + target,
                            thread_count=THREADS_PER_WARP
                            * (group_warps + (1 if target > 0 else 0)),
                        )
            if issues:
                desc_k = prims.Tcgen05SmemDesc.build(
                    k_smem.subview(stage * factor_tile_elems),
                    leading_byte_offset=CHUNK * row_bytes,
                    stride_byte_offset=swizzle_stride_bytes,
                    layout=smem_swizzle,
                )
                # Built unconditionally: the staged ``if`` below may only use
                # names defined outside any control flow.
                desc_update = prims.Tcgen05SmemDesc.build(
                    update_smem,
                    leading_byte_offset=CORE_MATRIX_BYTES,
                    stride_byte_offset=(CHUNK // CORE_MATRIX_ROWS) * CORE_MATRIX_BYTES,
                    layout=prims.Tcgen05SmemSwizzle.NONE,
                )
                for target in cutlass.range_constexpr(TOKEN_GROUPS):
                    prims.barrier_cta_sync(
                        GROUP_BARRIER_ID + target,
                        thread_count=THREADS_PER_WARP
                        * (group_warps + (1 if target > 0 else 0)),
                    )
                    prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                    # Short groups issue unrolled with their operand
                    # descriptors precomputed; a long group (sixteen steps)
                    # would hoist more descriptors than the uniform register
                    # file holds, so it issues from a rolled loop.
                    if cutlass.const_expr(unrolled_k_steps):
                        for k_step in cutlass.range_constexpr(
                            target * group_k_steps, (target + 1) * group_k_steps
                        ):
                            if smem_update_image:
                                issue_update_step(
                                    state_tmem,
                                    desc_update.advance_start_address(
                                        k_step * K_STEP_BYTES_SMEM_IMAGE
                                    ),
                                    desc_k.advance_start_address(
                                        k_step * k_step_bytes_mn_major
                                    ),
                                    idesc_update,
                                )
                            else:
                                issue_update_step(
                                    state_tmem,
                                    image_tmem.subview(
                                        UPDATE_IMAGE_COL
                                        + k_step * IMAGE_COLS_PER_K_STEP
                                    ),
                                    desc_k.advance_start_address(
                                        k_step * k_step_bytes_mn_major
                                    ),
                                    idesc_update,
                                )
                    else:
                        for k_step in cutlass.range(
                            target * group_k_steps,
                            (target + 1) * group_k_steps,
                            1,
                            unroll=1,
                        ):
                            if smem_update_image:
                                issue_update_step(
                                    state_tmem,
                                    desc_update.advance_start_address(
                                        k_step * K_STEP_BYTES_SMEM_IMAGE
                                    ),
                                    desc_k.advance_start_address(
                                        k_step * k_step_bytes_mn_major
                                    ),
                                    idesc_update,
                                )
                            else:
                                issue_update_step(
                                    state_tmem,
                                    image_tmem.subview(
                                        UPDATE_IMAGE_COL
                                        + k_step * IMAGE_COLS_PER_K_STEP
                                    ),
                                    desc_k.advance_start_address(
                                        k_step * k_step_bytes_mn_major
                                    ),
                                    idesc_update,
                                )
                if prims.elect_sync():
                    prims.tcgen05_commit(state_ready_mbar, group=prims.CTAGroup.CTA_1)
                    # Every read of the stage (the u and gate reads of all
                    # epilogue warps precede the group barriers above, the w
                    # and k reads are the MMAs themselves) is done when this
                    # fires.
                    prims.tcgen05_commit(
                        empty_mbar.subview(stage), group=prims.CTAGroup.CTA_1
                    )
            if store_warps == 0:
                # No spare replica to hand the store to: write this warp's
                # columns of the state image while the update runs.
                store_h_share(
                    h,
                    state,
                    stored_row0,
                    stored_row1,
                    replica,
                    batch,
                    chunk,
                    head,
                    state_col0,
                    col_shift,
                    v_index0,
                    v_index1,
                    h_cols,
                    state_regs,
                    half_lanes,
                )
            stage, ring_phase = advance_ring(stage, ring_phase, STAGES)
        # The last chunk: store the final state, nothing else.
        if computed_chunks > 0:
            mbarrier_wait(state_ready_mbar, (computed_chunks - 1) & 1)
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
        state = load_f32_columns(state_ptr, state_cols, half_lanes)
        prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)
        store_h_share(
            h,
            state,
            stored_row0,
            stored_row1,
            replica,
            batch,
            computed_chunks,
            head,
            state_col0,
            col_shift,
            v_index0,
            v_index1,
            h_cols,
            state_regs,
            half_lanes,
        )
        prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
        epilogue_sync(EPI_WARPS)
        if warp_idx == 0:
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            prims.tcgen05_dealloc(
                cutlass.inttoptr(tmem_base, TMEM_SPACE, cutlass.Float32),
                TMEM_COLS,
                group=prims.CTAGroup.CTA_1,
            )
    elif warp_idx < EPI_WARPS + store_warps:
        # ======================= h store warps ===============================
        quadrant = warp_idx - EPI_WARPS
        row_group = quadrant % quadrants
        replica = quadrant // quadrants
        # One lane per row and all of its pairs, or (``16x64b``) row ``lane
        # // 4 + 8 * (lane % 2)`` and pair ``lane % 4 // 2`` of every two.
        lane_row = (
            lane // 4 + HALF_LANE_ROW_STRIDE * (lane % 2)
            if cutlass.const_expr(half_lanes)
            else lane
        )
        pair_base = (
            (lane % 4) // 2 if cutlass.const_expr(half_lanes) else cutlass.Int32(0)
        )
        pair_stride: cutlass.Constexpr = 2 if half_lanes else 1
        store_regs: cutlass.Constexpr = store_pairs // pair_stride
        row_local = row_group * quadrant_rows + lane_row
        v_index = v_base + row_local
        stored_row = (row_local < BLOCK_V) & (v_index < dstate)
        col0 = replica * (2 * store_pairs)
        image_ptr = tmem_row_ptr(
            tmem_base, quadrant, STATE_IMAGE_COL + replica * store_pairs, cutlass.Int32
        )
        zero_image = cutlass.Array(cutlass.Int32, store_regs)
        for reg in cutlass.range_constexpr(store_regs):
            zero_image[reg] = cutlass.Int32(0)
        for chunk in cutlass.range(computed_chunks, unroll=1):
            if chunk == 0:
                # The first chunk's h is the zero state: nothing to read back,
                # the image slot is free from the start.
                if prims.elect_sync():
                    prims.mbarrier_arrive(state_ready_mbar)
                store_h_image(
                    h,
                    zero_image,
                    stored_row,
                    batch,
                    chunk,
                    head,
                    col0,
                    pair_base,
                    v_index,
                    store_regs,
                    pair_stride,
                )
            else:
                # The last group's commit orders every epilogue warp's image
                # store (barrier + fences) and all projections before this read.
                mbarrier_wait(acc_ready_mbar.subview(TOKEN_GROUPS - 1), (chunk - 1) & 1)
                prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                image = load_image_words(image_ptr, store_pairs, half_lanes)
                prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)
                prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
                if prims.elect_sync():
                    prims.mbarrier_arrive(state_ready_mbar)
                store_h_image(
                    h,
                    image,
                    stored_row,
                    batch,
                    chunk,
                    head,
                    col0,
                    pair_base,
                    v_index,
                    store_regs,
                    pair_stride,
                )


@cute.jit
def host_gdn_recurrence(
    k: cute.Tensor,
    w: cute.Tensor,
    u: cute.Tensor,
    g: cute.Tensor,
    h: cute.Tensor,
    stream,
    BLOCK_V: cutlass.Constexpr,
    CHUNK: cutlass.Constexpr,
    DHEAD: cutlass.Constexpr,
    EPI_WARPS: cutlass.Constexpr,
    STAGES: cutlass.Constexpr,
    TOKEN_GROUPS: cutlass.Constexpr,
    MMA_M: cutlass.Constexpr,
    STATE_COL: cutlass.Constexpr,
    STATE_IMAGE_COL: cutlass.Constexpr,
    ACC_COL: cutlass.Constexpr,
    UPDATE_IMAGE_COL: cutlass.Constexpr,
    TMEM_COLS: cutlass.Constexpr,
) -> None:
    """Build the chunk-tile TMA descriptors and launch one CTA per state tile."""

    k_shape = cast("tuple[int | cutlass.Integer, ...]", k.shape)
    u_shape = cast("tuple[int | cutlass.Integer, ...]", u.shape)
    h_shape = cast("tuple[int | cutlass.Integer, ...]", h.shape)
    batch = k_shape[0]
    heads = k_shape[2]
    dstate = u_shape[3]
    num_chunks = h_shape[1]
    factor_swizzle = tma_swizzle_for_row_bytes(DHEAD * BF16_BYTES)
    # ``box_dims`` follow the tensors' own mode order (batch, seq, head, feature);
    # the kernel's TMA coordinates follow ``TMA_STRIDE_ORDER`` (feature, head,
    # seq, batch).  The explicit order also covers singleton batch/head modes
    # whose strides tie with a neighbour.
    tma_desc_w = cuda.create_tensor_map_tiled_from_view(
        w,
        box_dims=(1, CHUNK, 1, DHEAD),
        stride_order=TMA_STRIDE_ORDER,
        swizzle=factor_swizzle,
    )
    tma_desc_k = cuda.create_tensor_map_tiled_from_view(
        k,
        box_dims=(1, CHUNK, 1, DHEAD),
        stride_order=TMA_STRIDE_ORDER,
        swizzle=factor_swizzle,
    )
    tma_desc_u = cuda.create_tensor_map_tiled_from_view(
        u,
        box_dims=(1, CHUNK, 1, BLOCK_V),
        stride_order=TMA_STRIDE_ORDER,
        swizzle=cuda.TensorMapSwizzle.none,
    )
    gdn_recurrence_kernel(
        tma_desc_w,
        tma_desc_k,
        tma_desc_u,
        k,
        u,
        g,
        h,
        cutlass.Int32(num_chunks),
        cutlass.Int32(dstate),
        BLOCK_V,
        CHUNK,
        DHEAD,
        EPI_WARPS,
        STAGES,
        TOKEN_GROUPS,
        MMA_M,
        STATE_COL,
        STATE_IMAGE_COL,
        ACC_COL,
        UPDATE_IMAGE_COL,
        TMEM_COLS,
    ).launch(
        grid=((dstate + BLOCK_V - 1) // BLOCK_V, heads, batch),
        block=(THREADS_PER_WARP * gdn_cta_warps(BLOCK_V, EPI_WARPS, MMA_M), 1, 1),
        stream=stream,
        min_blocks_per_mp=1,
        preferred_smem_carveout=100,
    )


__all__ = ["gdn_recurrence_kernel", "host_gdn_recurrence"]
