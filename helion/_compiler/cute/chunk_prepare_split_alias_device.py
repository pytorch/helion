# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

# The CuTe DSL intentionally leaves its compile-time value types implicit.
# ruff: noqa: ANN001, ANN202

"""Exact BT16/K128 warp-specialized prepare device schedule.

This is a backend schedule implementation. Admission and argument binding live
in `helion._compiler.cute.chunk_prepare`; no source or kernel name is a
match predicate.
"""

from __future__ import annotations

from typing import Any
from typing import cast

import cutlass
from cutlass._mlir.dialects import llvm
import cutlass.cute as cute
from cutlass.cutlass_dsl import dsl_user_op
import cutlass.utils

from .kda_device_primitives import _mma_m16n8k16
from .kda_device_primitives import mma_m16n8k16_bf16
from .kda_device_primitives import movmatrix_b16
from .kda_device_primitives import pack_bf16x2
from .kda_device_primitives import store_vec8_bf16
from .kda_device_primitives import tma_store_3d
from .kda_device_primitives import tma_store_commit_group
from .kda_device_primitives import tma_store_wait_read
from .kda_device_primitives import vec4_f32
from .kda_device_primitives import vec8_bf16
from .kda_device_primitives import vec_at
from .kda_device_primitives import warp_arrive

BT = 16
DK = 128
make_rmem_tensor = cute.make_rmem_tensor
_LLVM_STRUCT_TYPE = cast("Any", llvm).StructType

PREPARE_DEVICE_THREADS = 128
PREPARE_DEVICE_WARPS = 4
BF16_SEG_ELEMS = 64
BF16_SEG_STRIDE = 1024
F32_SEG_ELEMS = 32
F32_SEG_STRIDE = 512
TMA_TX_BYTES_QK = 8192
TMA_TX_BYTES_G_FP32 = 8192
TMA_TX_BYTES_G_BF16 = 4096
MBAR_SLOT_GATE0 = 0
MBAR_SLOT_GATE1 = 1
MBAR_SLOT_QK0 = 2
MBAR_SLOT_QK1 = 3
MBAR_SLOT_K_HALF_READY = 4
MBAR_SLOT_K_FULL_READY = 5
MBAR_SLOT_RAW_RELEASED = 6
MBAR_SLOT_PAIRWISE_READY = 7
MBAR_SLOT_Q_RELEASED = 8
MBAR_SLOT_K_RELEASED = 9
PREFIX_FLOOR = -126.0
NORM_SS_FLOOR = 1.0e-24
LOG2_E = 1.4426950408889634


def ldmatrix_x4(smem_ptr, *, loc=None, ip=None):
    """``ldmatrix.sync.aligned.m8n8.x4.shared.b16`` -> four b32 registers."""
    from cutlass._mlir.extras import types as _T

    struct = llvm.inline_asm(
        _LLVM_STRUCT_TYPE.get_literal([_T.IntegerType.get_signless(32)] * 4),
        [smem_ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)],
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(
        cutlass.Int32(
            llvm.extractvalue(
                _T.IntegerType.get_signless(32), struct, [i], loc=loc, ip=ip
            )
        )
        for i in range(4)
    )


@dsl_user_op
def stmatrix_x4(smem_ptr, r0, r1, r2, r3, *, loc=None, ip=None):
    """``stmatrix.sync.aligned.m8n8.x4.shared.b16`` from four b32 registers."""

    llvm.inline_asm(
        None,
        [
            smem_ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip),
            cutlass.Int32(r0).ir_value(loc=loc, ip=ip),
            cutlass.Int32(r1).ir_value(loc=loc, ip=ip),
            cutlass.Int32(r2).ir_value(loc=loc, ip=ip),
            cutlass.Int32(r3).ir_value(loc=loc, ip=ip),
        ],
        "stmatrix.sync.aligned.m8n8.x4.shared.b16 [$0], {$1, $2, $3, $4};",
        "r,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def mma_m16n8k16_f16(a0, a1, a2, a3, b0, b1, c0, c1, c2, c3, *, loc=None, ip=None):
    """``mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32``."""

    return _mma_m16n8k16("f16", a0, a1, a2, a3, b0, b1, c0, c1, c2, c3, loc=loc, ip=ip)


@dsl_user_op
def pack_f16x2(lo: cutlass.Float32, hi: cutlass.Float32, *, loc=None, ip=None):
    """Round two FP32 values to FP16 and pack them into one b32 register.

    The FP16 twin of :func:`pack_bf16x2`, with the same half ordering:
    ``cvt.rn.f16x2.f32 d, hi, lo`` places ``lo`` in the low half.  Used by the
    inverse chain, whose operands are FP16 regardless of the kernel's input
    dtype -- FP16's 10-bit significand against BF16's 7 is worth 4-8x there,
    and the chain is the one stage where the extra bits survive.
    """
    from cutlass._mlir.extras import types as _T

    return cutlass.Int32(
        llvm.inline_asm(
            _T.IntegerType.get_signless(32),
            [
                cutlass.Float32(hi).ir_value(loc=loc, ip=ip),
                cutlass.Float32(lo).ir_value(loc=loc, ip=ip),
            ],
            "cvt.rn.f16x2.f32 $0, $1, $2;",
            "=r,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def mul_bf16x2(a, b, *, loc=None, ip=None):
    """Packed BF16 multiply of two b32 registers, rounding each product."""
    from cutlass._mlir.extras import types as _T

    return cutlass.Int32(
        llvm.inline_asm(
            _T.IntegerType.get_signless(32),
            [
                cutlass.Int32(a).ir_value(loc=loc, ip=ip),
                cutlass.Int32(b).ir_value(loc=loc, ip=ip),
            ],
            "mul.rn.bf16x2 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


def fence_tensormap_acquire(desc_addr, *, loc=None, ip=None):
    """Publish a host-written tensor map to the tensormap proxy.

    The descriptor is written by the host and read by the TMA unit through a
    different proxy, so the kernel has to acquire it before first use.
    """
    llvm.inline_asm(
        None,
        [cutlass.Int64(desc_addr).ir_value(loc=loc, ip=ip)],
        "fence.proxy.tensormap::generic.acquire.gpu [$0], 128;",
        "l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def tma_load_4d(smem_ptr, desc_addr, mbar_ptr, c0, c1, c2, c3, *, loc=None, ip=None):
    """One ``cp.async.bulk.tensor.4d`` box, global to shared."""
    llvm.inline_asm(
        None,
        [
            smem_ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip),
            cutlass.Int64(desc_addr).ir_value(loc=loc, ip=ip),
            mbar_ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip),
            cutlass.Int32(c0).ir_value(loc=loc, ip=ip),
            cutlass.Int32(c1).ir_value(loc=loc, ip=ip),
            cutlass.Int32(c2).ir_value(loc=loc, ip=ip),
            cutlass.Int32(c3).ir_value(loc=loc, ip=ip),
        ],
        "cp.async.bulk.tensor.4d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
        " [$0], [$1, {$3, $4, $5, $6}], [$2];",
        "r,l,r,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


def mbar_spin_wait(
    mbar_ptr: object, phase: object, wait_hint: int = 10_000_000
) -> None:
    """Busy-spin until an mbarrier reaches ``phase`` without sleep backoff.

    ``cute.arch.mbarrier_wait`` inserts ``NANOSLEEP`` between phase checks.
    Prepare's producer/consumer handoffs are short, so that wake-up latency is
    larger than the useful wait.  This matches FlashInfer's tight TRY-wait
    loop and preserves acquire semantics on the successful probe.
    """
    llvm.inline_asm(
        None,
        [
            cutlass.Int32(mbar_ptr.toint()).ir_value(),  # pyrefly: ignore[missing-attribute]
            cutlass.Int32(phase).ir_value(),  # pyrefly: ignore[bad-argument-type]
        ],
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT:\n\t"
        f"mbarrier.try_wait.parity.shared::cta.b64 P1, [$0], $1, {wait_hint};\n\t"
        "@P1 bra DONE;\n\t"
        "bra LAB_WAIT;\n\t"
        "DONE:\n\t"
        "}\n",
        "r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


def raw_bf16_idx(token, dim):
    seg = dim // BF16_SEG_ELEMS
    local = dim - seg * BF16_SEG_ELEMS
    group = local // 8
    inner = local - group * 8
    return (
        seg * BF16_SEG_STRIDE
        + token * BF16_SEG_ELEMS
        + (group ^ (token & 7)) * 8
        + inner
    )


@cute.jit
def raw_f32_idx(token, dim):
    seg = dim // F32_SEG_ELEMS
    local = dim - seg * F32_SEG_ELEMS
    group = local // 4
    inner = local - group * 4
    return (
        seg * F32_SEG_STRIDE + token * F32_SEG_ELEMS + (group ^ (token & 7)) * 4 + inner
    )


@cute.jit
def kr_ak_idx(token, dim):
    return raw_bf16_idx(token ^ 8, dim)


@cute.jit
def prepare_pair_idx(row, col):
    storage_col = col ^ 8
    byte_offset = 2 * (row * 16 + storage_col)
    return (byte_offset ^ (((byte_offset >> 7) & 1) << 4)) // 2


@cute.jit
def warp_row_sum_8(value: cutlass.Float32) -> cutlass.Float32:
    """Reduce the eight lanes that cooperate on one token row."""
    value = value + cutlass.Float32(cute.arch.shuffle_sync_bfly(value, offset=4))
    value = value + cutlass.Float32(cute.arch.shuffle_sync_bfly(value, offset=2))
    return value + cutlass.Float32(cute.arch.shuffle_sync_bfly(value, offset=1))


@cute.jit
def bf16_round(x: cutlass.Float32):
    return x.to(cutlass.BFloat16).to(cutlass.Float32)


def f16_round(x: cutlass.Float32):
    """Round through FP16, the inverse chain's operand dtype (Section 8.2)."""
    return x.to(cutlass.Float16).to(cutlass.Float32)


# ---------------------------------------------------------------------------
# Fragment helpers
# ---------------------------------------------------------------------------


@cute.jit
def load_a_fragment(
    smem,
    token_base,
    dim_base,
    lane,
    KEY_XOR: cutlass.Constexpr,
):
    """A-operand ``ldmatrix.x4``, undoing any storage-key permutation."""
    matrix_id = lane // 8
    row = token_base + (lane % 8) + 8 * (matrix_id % 2)
    col = (dim_base + 8 * (matrix_id // 2)) ^ KEY_XOR
    return ldmatrix_x4(smem + raw_bf16_idx(row, col))


@cute.jit
def load_b_fragment(smem, dim_base, lane):
    """B-operand ``ldmatrix.x4``.

    The row half is keyed on bit 4 of the lane, not bit 3.
    """
    matrix_id = lane // 8
    row = (lane % 8) + 8 * (lane // 16)
    col = dim_base + 8 * (matrix_id % 2)
    return ldmatrix_x4(smem + raw_bf16_idx(row, col))


@cute.jit
def load_pairwise_a_fragment(smem, lane):
    """Load a 16x16 pairwise tile in A layout."""
    matrix_id = lane // 8
    row = (lane % 8) + 8 * (matrix_id % 2)
    col = 8 * (matrix_id // 2)
    return ldmatrix_x4(smem + prepare_pair_idx(row, col))


@cute.jit
def a_to_b(a0, a1, a2, a3):
    """A layout -> B layout of the same matrix; identity register order."""
    return (
        movmatrix_b16(a0),
        movmatrix_b16(a1),
        movmatrix_b16(a2),
        movmatrix_b16(a3),
    )


@cute.jit
def a_to_a_transposed(a0, a1, a2, a3):
    """A layout -> A layout of the transpose; registers 1 and 2 swap."""
    return (
        movmatrix_b16(a0),
        movmatrix_b16(a2),
        movmatrix_b16(a1),
        movmatrix_b16(a3),
    )


@cute.jit
def mma_16x16(a0, a1, a2, a3, b0, b1, b2, b3, c):
    """One logical m16n16k16 step: two native N=8 MMAs."""
    n0 = mma_m16n8k16_bf16(a0, a1, a2, a3, b0, b1, c[0], c[1], c[2], c[3])
    n1 = mma_m16n8k16_bf16(a0, a1, a2, a3, b2, b3, c[4], c[5], c[6], c[7])
    return (n0[0], n0[1], n0[2], n0[3], n1[0], n1[1], n1[2], n1[3])


@cute.jit
def mma_blockdiag_8x8_f16(a0, a3, b0, b3):
    """Multiply two packed 8x8 block diagonals with one native MMA.

    ``a0``/``b0`` carry the upper-left block and ``a3``/``b3`` the
    lower-right block. Packing both right-hand blocks into the same N=8
    operand lets one m16n8 instruction compute the two independent 8x8
    products: accumulator slots 0:2 are the upper block and 2:4 the lower.
    """
    zero_i32 = cutlass.Int32(0)
    zero_f32 = cutlass.Float32(0.0)
    return mma_m16n8k16_f16(
        a0,
        zero_i32,
        zero_i32,
        a3,
        movmatrix_b16(b0),
        movmatrix_b16(b3),
        zero_f32,
        zero_f32,
        zero_f32,
        zero_f32,
    )


@cute.jit
def acc_to_a_fragment(c):
    """Register-local accumulator -> A-layout pack."""
    return (
        pack_bf16x2(c[0], c[1]),
        pack_bf16x2(c[2], c[3]),
        pack_bf16x2(c[4], c[5]),
        pack_bf16x2(c[6], c[7]),
    )


@cute.jit
def ZERO8():
    """Eight zeroed FP32 accumulator slots."""
    z = cutlass.Float32(0.0)
    return (z, z, z, z, z, z, z, z)


@cute.jit
def kk_half(lhs_ptr, ki_ptr, lane, half, c, FACTOR_KEY_XOR: cutlass.Constexpr):
    """``lhs @ Ki.T`` over the four K=16 phases of head-dimension half ``half``.

    Split so warp 0 can start K blocks 0-3 on ``k_half_ready`` and only wait
    for ``k_full_ready`` before blocks 4-7 (step 5 of the design).
    """
    for j in cutlass.range_constexpr(4):
        d0 = (half * 4 + j) * 16
        a0, a1, a2, a3 = load_a_fragment(lhs_ptr, 0, d0, lane, FACTOR_KEY_XOR)
        b0, b1, b2, b3 = load_b_fragment(ki_ptr, d0, lane)
        c = mma_16x16(a0, a1, a2, a3, b0, b1, b2, b3, c)
    return c


@cute.jit
def kk_over_dk(lhs_ptr, ki_ptr, lane, FACTOR_KEY_XOR: cutlass.Constexpr):
    """``lhs @ Ki.T`` over all eight K=16 phases of the head dimension."""
    return kk_half(
        lhs_ptr,
        ki_ptr,
        lane,
        1,
        kk_half(lhs_ptr, ki_ptr, lane, 0, ZERO8(), FACTOR_KEY_XOR),
        FACTOR_KEY_XOR,
    )


@cute.jit
def prepare_stmatrix_coord(lane):
    """Row/col of the 16-byte row segment lane ``lane`` feeds to stmatrix.x4.

    Same quadrant order as the A fragment.
    """
    matrix_id = lane // 8
    row = (lane - matrix_id * 8) + 8 * (matrix_id - (matrix_id // 2) * 2)
    col = 8 * (matrix_id // 2)
    return row, col


@cute.jit
def acc_coord(lane, slot):
    """``(row, col)`` of accumulator slot ``slot`` in ``[0, 8)``."""
    n_block = slot // 4
    reg = slot - n_block * 4
    row = (lane // 4) + 8 * (reg // 2)
    col = 8 * n_block + 2 * (lane % 4) + (reg % 2)
    return row, col


@cute.jit
def issue_chunk_tma(
    desc_q,
    desc_k,
    desc_g,
    p_q,
    p_k,
    p_g,
    mbar_gate,
    mbar_qk,
    token_base,
    head,
    G_FP32: cutlass.Constexpr,
):
    """Issue the first chunk's G, Q, and K full-tile TMA operations.

    Rank-4 descriptors fold the 64-element BF16 (or 32-element FP32) swizzle
    segments into their fourth box dimension.  Gate completion is tracked
    separately so prefix work can overlap the still-in-flight Q/K copies.  Q
    and K share one expected-byte transaction barrier even though later chunks
    issue them at different lifetime-safe points.
    """
    tma_load_4d(p_g, desc_g, mbar_gate, 0, token_base, head, 0)
    tma_load_4d(p_q, desc_q, mbar_qk, 0, token_base, head, 0)
    tma_load_4d(p_k, desc_k, mbar_qk, 0, token_base, head, 0)


@cute.jit
def clear_tail_rows(p_q, p_k, p_g, valid_rows, tidx, G_FP32: cutlass.Constexpr):
    """Zero the invalid rows of the raw stages.

    TMA only zero-fills coordinates outside the *tensor*, and a short chunk sits
    mid-tensor: the rows past ``valid_rows`` belong to the next sequence in the
    packed layout, so TMA faithfully loads real data there.  The ``cp.async``
    version this replaces got the zeros for free by passing src-size 0.

    Same 16-byte task map as the loads, so the writes stay one vector wide.
    """
    for rep in cutlass.range_constexpr(2):
        slot = tidx + rep * PREPARE_DEVICE_THREADS
        row = slot // 16
        d0 = (slot - row * 16) * 8
        if row >= valid_rows:
            zero8 = make_rmem_tensor(8, cutlass.BFloat16)
            for i in cutlass.range_constexpr(8):
                zero8[i] = cutlass.BFloat16(0.0)
            bidx = raw_bf16_idx(row, d0)
            store_vec8_bf16(p_q, bidx, zero8)
            store_vec8_bf16(p_k, bidx, zero8)

    if cutlass.const_expr(G_FP32):
        for rep in cutlass.range_constexpr(4):
            slot = tidx + rep * PREPARE_DEVICE_THREADS
            row = slot // 32
            d0 = (slot - row * 32) * 4
            if row >= valid_rows:
                zero4 = make_rmem_tensor(4, cutlass.Float32)
                for i in cutlass.range_constexpr(4):
                    zero4[i] = cutlass.Float32(0.0)
                cute.autovec_copy(zero4, vec_at(p_g, raw_f32_idx(row, d0), 4))
    else:
        # BF16 G is the same 4096-byte image as Q and K, so it takes the same
        # two-rep, 16-byte task map rather than the FP32 four-rep one.
        for rep in cutlass.range_constexpr(2):
            slot = tidx + rep * PREPARE_DEVICE_THREADS
            row = slot // 16
            d0 = (slot - row * 16) * 8
            if row >= valid_rows:
                zero8 = make_rmem_tensor(8, cutlass.BFloat16)
                for i in cutlass.range_constexpr(8):
                    zero8[i] = cutlass.BFloat16(0.0)
                store_vec8_bf16(p_g, raw_bf16_idx(row, d0), zero8)


@cute.jit
def load_beta_stage(gbeta, smem_beta, stage, token_base, valid_rows, head, lane, heads):
    """Activate one chunk's 16 beta logits into beta stage ``stage``.

    the design double-buffers ``smem_beta_act`` so that this strided
    column read out of ``[T, H]`` -- which cannot coalesce, one sector per
    token -- is issued a full chunk before the values are needed, instead of
    stalling the whole CTA at a barrier behind its DRAM latency.
    """
    if lane < BT:
        bv = cutlass.Float32(0.0)
        if lane < valid_rows:
            logit = cutlass.Float32(gbeta[(token_base + lane) * heads + head])
            half = cutlass.Float32(0.5)
            # Stored as FP32, unrounded.  The activated value has exactly two
            # consumers: the strict-lower scale, which is an FP32 multiply, and
            # the AINV column scale, which packs to BF16 itself.  Rounding here
            # is invisible to the second and only lossy to the first, at the
            # cost of two F2F conversions per lane (measured).
            bv = (
                cutlass.Float32(cute.math.tanh(logit * half, fastmath=True)) * half
                + half
            )
        smem_beta[stage * BT + lane] = bv


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------


@cute.jit
def emit_bt16_prepare(
    gq: cute.Tensor,
    gk: cute.Tensor,
    gg: cute.Tensor,
    gbeta: cute.Tensor,
    ga_log: cute.Tensor,
    gdt: cute.Tensor,
    gcu_seqlens: cute.Tensor,
    gcu_chunks: cute.Tensor,
    gchunk_to_seq: cute.Tensor,
    ws_kd: cute.Tensor,
    ws_qd: cute.Tensor,
    ws_ak: cute.Tensor,
    ws_aq: cute.Tensor,
    ws_gt: cute.Tensor,
    desc_q: cutlass.Int64,
    desc_k: cutlass.Int64,
    desc_g: cutlass.Int64,
    desc_factor: cutlass.Int64,
    SCALE: cutlass.Float32,
    GATE_SCALE_LOG2: cutlass.Float32,
    TOTAL_CHUNKS: cutlass.Int32,
    heads: cutlass.Int32,
    SCALE_OUTPUTS: cutlass.Constexpr,
    FACTOR_KEY_XOR: cutlass.Constexpr,
    CPC: cutlass.Constexpr,
    G_FP32: cutlass.Constexpr,
) -> None:
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    warp_id = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    lane = tidx % 32
    head = bidy

    alloc = cutlass.utils.SmemAllocator()
    p_kd = alloc.allocate_array(cutlass.BFloat16, BT * DK)
    # Qd overwrites raw Q and Ki overwrites raw K.  The first use is safe
    # warp-locally because every converged vector load precedes its vector
    # store (the unscaled layout's key-^8 Qd map only swaps source lanes). For
    # CPC>1 the next
    # Q/K TMAs are split below: Q waits for QK plus the outbound Qd store, and K
    # waits for both Ak producer warps to finish reading Ki.  G remains separate
    # and keeps its early prefetch overlap.
    p_q = alloc.allocate_array(cutlass.BFloat16, BT * DK)
    p_k = alloc.allocate_array(cutlass.BFloat16, BT * DK)
    p_g = alloc.allocate_array(cutlass.Float32, BT * DK)
    p_qd = p_q
    p_ki = p_k
    p_ainv = alloc.allocate_array(cutlass.BFloat16, BT * BT)
    p_gamma = alloc.allocate_array(cutlass.BFloat16, DK)
    # the design gives smem_beta_act 128 bytes: two 16-float stages, so a
    # chunk's beta is loaded and activated one chunk ahead of its use.
    p_beta = alloc.allocate_array(cutlass.Float32, 2 * BT)
    p_bar = alloc.allocate_array(cutlass.Int64, 10)
    p_qk = alloc.allocate_array(cutlass.BFloat16, BT * BT)

    # Pointers feed ldmatrix; the flat tensor views give dynamic scalar access.
    smem_g = cute.make_tensor(p_g, cute.make_layout(BT * DK))
    # A BF16 view of the same 8192-byte stage. With BF16 ``G`` the raw input
    # is the 4096-byte Q/K image living in its first half; the stage is then
    # overwritten in full by FP32 ``exp_g``. Unused when ``G`` is FP32.
    p_g_bf16 = cute.recast_ptr(p_g, dtype=cutlass.BFloat16)
    smem_g_bf16 = cute.make_tensor(p_g_bf16, cute.make_layout(BT * DK))
    # Trace-time selection: which view the raw G load and tail-clear address.
    p_g_raw = p_g if cutlass.const_expr(G_FP32) else p_g_bf16
    G_TX = TMA_TX_BYTES_G_FP32 if cutlass.const_expr(G_FP32) else TMA_TX_BYTES_G_BF16
    smem_ainv = cute.make_tensor(p_ainv, cute.make_layout(BT * BT))
    smem_gamma = cute.make_tensor(p_gamma, cute.make_layout(DK))
    smem_beta = cute.make_tensor(p_beta, cute.make_layout(2 * BT))
    smem_qk = cute.make_tensor(p_qk, cute.make_layout(BT * BT))

    # Gate and Q/K completion have independent two-stage barriers.  This lets
    # all warps consume the gate and run its prefix while the Q/K transactions
    # are still in flight.  Each pair alternates by chunk parity, so reuse two
    # chunks later toggles its phase.
    mbar_gate0 = p_bar + MBAR_SLOT_GATE0
    mbar_gate1 = p_bar + MBAR_SLOT_GATE1
    mbar_qk0 = p_bar + MBAR_SLOT_QK0
    mbar_qk1 = p_bar + MBAR_SLOT_QK1
    mbar_k_half = p_bar + MBAR_SLOT_K_HALF_READY
    mbar_k_full = p_bar + MBAR_SLOT_K_FULL_READY
    mbar_raw_released = p_bar + MBAR_SLOT_RAW_RELEASED
    mbar_pairwise = p_bar + MBAR_SLOT_PAIRWISE_READY
    mbar_q_released = p_bar + MBAR_SLOT_Q_RELEASED
    mbar_k_released = p_bar + MBAR_SLOT_K_RELEASED
    if warp_id == 0:
        if lane == 0:
            cute.arch.mbarrier_init(mbar_gate0, 1)
            cute.arch.mbarrier_init(mbar_gate1, 1)
            cute.arch.mbarrier_init(mbar_qk0, 1)
            cute.arch.mbarrier_init(mbar_qk1, 1)
            cute.arch.mbarrier_init(mbar_k_half, PREPARE_DEVICE_WARPS)
            cute.arch.mbarrier_init(mbar_k_full, PREPARE_DEVICE_WARPS)
            cute.arch.mbarrier_init(mbar_raw_released, PREPARE_DEVICE_WARPS)
            cute.arch.mbarrier_init(mbar_pairwise, 1)
            cute.arch.mbarrier_init(mbar_q_released, 2)
            cute.arch.mbarrier_init(mbar_k_released, 2)
            cute.arch.mbarrier_init_fence()
            fence_tensormap_acquire(desc_q)
            fence_tensormap_acquire(desc_k)
            fence_tensormap_acquire(desc_g)
            # One acquire, not three: Kd/Qd/Ak are 3H planes of one map.
            fence_tensormap_acquire(desc_factor)
    # step 1 of the design: nothing may arrive or wait before the
    # initialized state is published.
    cute.arch.barrier()

    cta_chunk_base = bidx * CPC
    my_chunks = TOTAL_CHUNKS - cta_chunk_base
    if my_chunks > CPC:
        my_chunks = cutlass.Int32(CPC)

    dt_value = cutlass.Float32(gdt[head * DK + tidx])
    a_log_value = cute.math.exp2(
        cutlass.Float32(ga_log[head]) * cutlass.Float32(LOG2_E), fastmath=True
    )
    row_in_warp = lane // 8
    lane_in_row = lane % 8
    my_token = warp_id * 4 + row_in_warp
    q4 = lane % 4

    # Prologue: start the first chunk's copies before entering the loop.
    seq0 = cutlass.Int32(gchunk_to_seq[cta_chunk_base])
    lc0 = cta_chunk_base - cutlass.Int32(gcu_chunks[seq0])
    tb0 = cutlass.Int32(gcu_seqlens[seq0]) + lc0 * BT
    vr0 = cutlass.Int32(gcu_seqlens[seq0 + 1]) - tb0
    if vr0 > BT:
        vr0 = cutlass.Int32(BT)
    if warp_id == 0:
        if lane == 0:
            cute.arch.mbarrier_arrive_and_expect_tx(mbar_gate0, G_TX)
            cute.arch.mbarrier_arrive_and_expect_tx(mbar_qk0, TMA_TX_BYTES_QK)
            issue_chunk_tma(
                desc_q,
                desc_k,
                desc_g,
                p_q,
                p_k,
                p_g_raw,
                mbar_gate0,
                mbar_qk0,
                tb0,
                head,
                G_FP32,
            )
        load_beta_stage(gbeta, smem_beta, 0, tb0, vr0, head, lane, heads)

    for lc in cutlass.range_constexpr(CPC):
        if lc < my_chunks:
            gchunk = cta_chunk_base + lc
            seq = cutlass.Int32(gchunk_to_seq[gchunk])
            local_c = gchunk - cutlass.Int32(gcu_chunks[seq])
            seq_start = cutlass.Int32(gcu_seqlens[seq])
            seq_end = cutlass.Int32(gcu_seqlens[seq + 1])
            token_base = seq_start + local_c * BT
            valid_rows = seq_end - token_base
            if valid_rows > BT:
                valid_rows = cutlass.Int32(BT)

            # ---- wait for the gate, overlap its prefix with Q/K TMA ------
            beta_stage = lc & 1
            tma_wait_phase = (lc >> 1) & 1
            if cutlass.const_expr(lc & 1):
                mbar_spin_wait(mbar_gate1, tma_wait_phase)
            else:
                mbar_spin_wait(mbar_gate0, tma_wait_phase)

            # Section 7.3: a short chunk sits mid-tensor, so TMA loaded the
            # next sequence's rows rather than zeros.  Clear them before the
            # gate is consumed.  Tail clearing also writes Q/K, so it must
            # first wait for their independent transaction barrier.
            if valid_rows < BT:
                if cutlass.const_expr(lc & 1):
                    mbar_spin_wait(mbar_qk1, tma_wait_phase)
                else:
                    mbar_spin_wait(mbar_qk0, tma_wait_phase)
                clear_tail_rows(p_q, p_k, p_g_raw, valid_rows, tidx, G_FP32)
                cute.arch.barrier()

            # ---- gate prefix --------------------------
            gate_regs = [cutlass.Float32(0.0) for _ in range(BT)]
            for r in cutlass.range_constexpr(BT):
                if cutlass.const_expr(G_FP32):
                    raw = smem_g[raw_f32_idx(r, tidx)]
                else:
                    raw = cutlass.Float32(smem_g_bf16[raw_bf16_idx(r, tidx)])
                inc = cutlass.Float32(0.0)
                if r < valid_rows:
                    x = raw + dt_value
                    half = cutlass.Float32(0.5)
                    sig = (
                        cutlass.Float32(
                            cute.math.tanh(a_log_value * x * half, fastmath=True)
                        )
                        * half
                        + half
                    )
                    inc = GATE_SCALE_LOG2 * sig
                gate_regs[r] = inc

            acc = cutlass.Float32(0.0)
            for p in cutlass.range_constexpr(BT // 2):
                r0 = p * 2
                g0 = gate_regs[r0]
                g1 = gate_regs[r0 + 1]
                prefix0 = acc + g0
                prefix1 = acc + (g0 + g1)
                gate_regs[r0] = prefix0
                gate_regs[r0 + 1] = prefix1
                acc = prefix1

            for r in cutlass.range_constexpr(BT):
                pv = gate_regs[r]
                if pv < cutlass.Float32(PREFIX_FLOOR):
                    pv = cutlass.Float32(PREFIX_FLOOR)
                gate_regs[r] = cutlass.Float32(cute.math.exp2(pv, fastmath=True))

            gamma = gate_regs[BT - 1]
            ws_gt[(head * TOTAL_CHUNKS + gchunk) * DK + tidx] = gamma
            smem_gamma[tidx] = gamma.to(cutlass.BFloat16)
            if cutlass.const_expr(not G_FP32):
                # The FP32 exp_g image about to be written spans all 8192 bytes
                # of the stage; the BF16 raw G it overwrites occupied only the
                # first 4096, in a different index map. A thread's write can
                # therefore land on a raw element another thread has not read
                # yet, so the read loop above must be complete CTA-wide first.
                # With FP32 G the two maps coincide and each thread rewrites
                # exactly the addresses it read, so no barrier is needed.
                cute.arch.barrier()
            for r in cutlass.range_constexpr(BT):
                smem_g[raw_f32_idx(r, tidx)] = gate_regs[r]
            cute.arch.barrier()

            # Q/K were launched after G but did not gate prefix evaluation.
            # Full chunks wait here; short chunks already waited before their
            # invalid rows were cleared.
            if valid_rows == BT:
                if cutlass.const_expr(lc & 1):
                    mbar_spin_wait(mbar_qk1, tma_wait_phase)
                else:
                    mbar_spin_wait(mbar_qk0, tma_wait_phase)

            # ---- norm + Kd/Ki/Qd --------------------
            # Every SMEM and global access below is 16 bytes wide.  The 8
            # features a lane owns are contiguous and 16-byte aligned under
            # raw_bf16_s128, and exp_g is read as 2 x float4 because 8
            # consecutive FP32 are two separate 4-element runs whose order
            # depends on the token parity.
            q_ss = cutlass.Float32(0.0)
            k_ss = cutlass.Float32(0.0)
            for h in cutlass.range_constexpr(2):
                d0 = 64 * h + 8 * lane_in_row
                fq = vec8_bf16(p_q, raw_bf16_idx(my_token, d0))
                fk = vec8_bf16(p_k, raw_bf16_idx(my_token, d0))
                for i in cutlass.range_constexpr(8):
                    qv = cutlass.Float32(fq[i])
                    kv = cutlass.Float32(fk[i])
                    q_ss = q_ss + qv * qv
                    k_ss = k_ss + kv * kv
            q_ss = warp_row_sum_8(q_ss)
            k_ss = warp_row_sum_8(k_ss)

            q_inv = cutlass.Float32(0.0)
            k_inv = cutlass.Float32(0.0)
            if my_token < valid_rows:
                qf = q_ss
                if qf < cutlass.Float32(NORM_SS_FLOOR):
                    qf = cutlass.Float32(NORM_SS_FLOOR)
                kf = k_ss
                if kf < cutlass.Float32(NORM_SS_FLOOR):
                    kf = cutlass.Float32(NORM_SS_FLOOR)
                q_inv = cutlass.Float32(cute.math.rsqrt(qf, fastmath=True))
                k_inv = cutlass.Float32(cute.math.rsqrt(kf, fastmath=True))

            s16 = (
                bf16_round(SCALE)
                if cutlass.const_expr(SCALE_OUTPUTS)
                else bf16_round(cutlass.Float32(1.0))
            )
            for h in cutlass.range_constexpr(2):
                d0 = 64 * h + 8 * lane_in_row
                bidx = raw_bf16_idx(my_token, d0)
                fq = vec8_bf16(p_q, bidx)
                fk = vec8_bf16(p_k, bidx)
                fg0 = vec4_f32(p_g, raw_f32_idx(my_token, d0))
                fg1 = vec4_f32(p_g, raw_f32_idx(my_token, d0 + 4))

                o_kd = make_rmem_tensor(8, cutlass.BFloat16)
                o_ki = make_rmem_tensor(8, cutlass.BFloat16)
                o_qd = make_rmem_tensor(8, cutlass.BFloat16)
                for j in cutlass.range_constexpr(4):
                    i = 0 + j
                    eg = fg0[j]
                    kv = cutlass.Float32(fk[i]) * k_inv
                    kd_v = bf16_round(bf16_round(kv) * bf16_round(eg))
                    ki_v = bf16_round(kv * cutlass.Float32(cute.arch.rcp_approx(eg)))
                    qv = bf16_round(cutlass.Float32(fq[i]) * q_inv)
                    qd_v = bf16_round(bf16_round(qv * bf16_round(eg)) * s16)
                    o_kd[i] = kd_v.to(cutlass.BFloat16)
                    o_ki[i] = ki_v.to(cutlass.BFloat16)
                    o_qd[i] = qd_v.to(cutlass.BFloat16)

                for j in cutlass.range_constexpr(4):
                    i = 4 + j
                    eg = fg1[j]
                    kv = cutlass.Float32(fk[i]) * k_inv
                    kd_v = bf16_round(bf16_round(kv) * bf16_round(eg))
                    ki_v = bf16_round(kv * cutlass.Float32(cute.arch.rcp_approx(eg)))
                    qv = bf16_round(cutlass.Float32(fq[i]) * q_inv)
                    qd_v = bf16_round(bf16_round(qv * bf16_round(eg)) * s16)
                    o_kd[i] = kd_v.to(cutlass.BFloat16)
                    o_ki[i] = ki_v.to(cutlass.BFloat16)
                    o_qd[i] = qd_v.to(cutlass.BFloat16)

                factor_bidx = raw_bf16_idx(my_token, d0 ^ FACTOR_KEY_XOR)
                store_vec8_bf16(p_kd, factor_bidx, o_kd)
                store_vec8_bf16(p_ki, bidx, o_ki)
                store_vec8_bf16(p_qd, factor_bidx, o_qd)
                # step 5 of the design: signal each Kd/Ki half the moment it
                # is in SMEM.  Kd and Qd leave for global by TMA later, read
                # straight out of these same images (Section 7.4), so there is
                # no separate global store here any more.
                if cutlass.const_expr(h == 0):
                    warp_arrive(mbar_k_half, lane)
                else:
                    warp_arrive(mbar_k_full, lane)

            # This warp is done with raw Q/K/exp-g and with Qd (Section 12.3
            # step 5); the raw stages may be overwritten once all four arrive.
            warp_arrive(mbar_raw_released, lane)
            phase = lc & 1

            # ---- warp 0: KK -> L -> AINV ------------------------------
            if warp_id == 0:
                mbar_spin_wait(mbar_k_half, phase)
                c = kk_half(p_kd, p_ki, lane, 0, ZERO8(), FACTOR_KEY_XOR)
                mbar_spin_wait(mbar_k_full, phase)
                c = kk_half(p_kd, p_ki, lane, 1, c, FACTOR_KEY_XOR)
                masked = [cutlass.Float32(0.0) for _ in range(8)]
                for slot in cutlass.range_constexpr(8):
                    row, col = acc_coord(lane, slot)
                    v = cutlass.Float32(0.0)
                    if row > col:
                        v = c[slot] * smem_beta[beta_stage * BT + row]
                    masked[slot] = v

                # Blockwise 8x8 inverse.  With D the block
                # diagonal of L and A21 its one off-diagonal block:
                #
                #   Binv = (I - D)(I + D^2)(I + D^4)   exact, blocks nilpotent
                #                                      at 8, so 3 factors
                #   AINV = Binv - Binv @ A21 @ Binv    exact, (Binv @ A21)^2 = 0
                #
                # The two diagonal 8x8 blocks share one native m16n8 issue:
                # its top eight rows carry the first block and its bottom
                # eight rows carry the second. This is the native CAKE
                # block-sparse formulation: two squares, two inverse updates,
                # and two lower-left coupling products are exactly six FP16
                # MMA issues, rather than six logical m16n16 operations.
                diagonal = [cutlass.Float32(0.0) for _ in range(4)]
                diagonal_slots = (0, 1, 6, 7)
                for index in cutlass.range_constexpr(4):
                    slot = diagonal_slots[index]
                    row, col = acc_coord(lane, slot)
                    eye = cutlass.Float32(0.0)
                    if row == col:
                        eye = cutlass.Float32(1.0)
                    diagonal[index] = eye - f16_round(masked[slot])

                d0 = pack_f16x2(masked[0], masked[1])
                d3 = pack_f16x2(masked[6], masked[7])
                d2 = mma_blockdiag_8x8_f16(d0, d3, d0, d3)
                d2_0 = pack_f16x2(d2[0], d2[1])
                d2_3 = pack_f16x2(d2[2], d2[3])

                diagonal_0 = pack_f16x2(diagonal[0], diagonal[1])
                diagonal_3 = pack_f16x2(diagonal[2], diagonal[3])
                product = mma_blockdiag_8x8_f16(diagonal_0, diagonal_3, d2_0, d2_3)
                for index in cutlass.range_constexpr(4):
                    diagonal[index] = f16_round(diagonal[index]) + product[index]

                d4 = mma_blockdiag_8x8_f16(d2_0, d2_3, d2_0, d2_3)
                d4_0 = pack_f16x2(d4[0], d4[1])
                d4_3 = pack_f16x2(d4[2], d4[3])
                diagonal_0 = pack_f16x2(diagonal[0], diagonal[1])
                diagonal_3 = pack_f16x2(diagonal[2], diagonal[3])
                product = mma_blockdiag_8x8_f16(diagonal_0, diagonal_3, d4_0, d4_3)
                for index in cutlass.range_constexpr(4):
                    diagonal[index] = f16_round(diagonal[index]) + product[index]

                # T1 = Binv @ A21 and X21 = -T1 @ Binv each have only the
                # lower-left output quadrant live, so each needs one N=8 MMA.
                zero_i32 = cutlass.Int32(0)
                zero_f32 = cutlass.Float32(0.0)
                binv_0 = pack_f16x2(diagonal[0], diagonal[1])
                binv_3 = pack_f16x2(diagonal[2], diagonal[3])
                a21_1 = pack_f16x2(masked[2], masked[3])
                t1 = mma_m16n8k16_f16(
                    binv_0,
                    zero_i32,
                    zero_i32,
                    binv_3,
                    zero_i32,
                    movmatrix_b16(a21_1),
                    zero_f32,
                    zero_f32,
                    zero_f32,
                    zero_f32,
                )
                correction_1 = pack_f16x2(t1[2], t1[3])
                correction = mma_m16n8k16_f16(
                    zero_i32,
                    correction_1,
                    zero_i32,
                    zero_i32,
                    movmatrix_b16(binv_0),
                    zero_i32,
                    zero_f32,
                    zero_f32,
                    zero_f32,
                    zero_f32,
                )

                inverse = (
                    diagonal[0],
                    diagonal[1],
                    -correction[2],
                    -correction[3],
                    zero_f32,
                    zero_f32,
                    diagonal[2],
                    diagonal[3],
                )
                for slot in cutlass.range_constexpr(8):
                    row, col = acc_coord(lane, slot)
                    smem_ainv[prepare_pair_idx(row, col)] = bf16_round(
                        inverse[slot]
                    ).to(cutlass.BFloat16)
                # step 7 of the design.  This arrival publishes AINV and
                # also proves warp 0 is done reading smem_kd_then_ak, which is
                # what lets warps 1 and 3 overwrite their Kd half with Ak.
                warp_arrive(mbar_pairwise, lane)
                if lc + 1 < CPC:
                    if lc + 1 < my_chunks:
                        # Qd aliases raw Q.  Its image becomes reusable only
                        # after warp 2 has consumed it for QK and warp 3's Qd
                        # TMA store has finished reading both segments.
                        mbar_spin_wait(mbar_q_released, phase)
                        nchunk = gchunk + 1
                        nseq = cutlass.Int32(gchunk_to_seq[nchunk])
                        nlc = nchunk - cutlass.Int32(gcu_chunks[nseq])
                        ntb = cutlass.Int32(gcu_seqlens[nseq]) + nlc * BT
                        if lane == 0:
                            if cutlass.const_expr((lc + 1) & 1):
                                cute.arch.mbarrier_arrive_and_expect_tx(
                                    mbar_qk1, TMA_TX_BYTES_QK
                                )
                                tma_load_4d(p_q, desc_q, mbar_qk1, 0, ntb, head, 0)
                            else:
                                cute.arch.mbarrier_arrive_and_expect_tx(
                                    mbar_qk0, TMA_TX_BYTES_QK
                                )
                                tma_load_4d(p_q, desc_q, mbar_qk0, 0, ntb, head, 0)

            # ---- warp 2: causal QK, staged through SMEM ---------------
            if warp_id == 2:
                mbar_spin_wait(mbar_k_full, phase)
                c = kk_over_dk(p_qd, p_ki, lane, FACTOR_KEY_XOR)
                for slot in cutlass.range_constexpr(8):
                    row, col = acc_coord(lane, slot)
                    v = cutlass.Float32(0.0)
                    if row >= col:
                        v = c[slot]
                    smem_qk[prepare_pair_idx(row, col)] = bf16_round(v).to(
                        cutlass.BFloat16
                    )
                # smem_qk is produced and consumed by this warp alone.
                cute.arch.sync_warp()
                if lc + 1 < CPC:
                    if lc + 1 < my_chunks:
                        warp_arrive(mbar_q_released, lane)

            # ---- warp 1: release raw G, then prefetch the next gate ------
            if warp_id == 1:
                mbar_spin_wait(mbar_raw_released, phase)
                # publish the ordinary Kd stores to the async
                # shared proxy before the TMA engine reads them, and converge
                # the warp so lane 0 speaks for all 32 lanes' writes.
                cute.arch.fence_view_async_shared()
                cute.arch.sync_warp()
                if lane == 0:
                    tma_store_3d(desc_factor, p_kd, 0, gchunk * BT, head)
                    tma_store_commit_group()
                if lc + 1 < CPC:
                    if lc + 1 < my_chunks:
                        nchunk = gchunk + 1
                        nseq = cutlass.Int32(gchunk_to_seq[nchunk])
                        nlc = nchunk - cutlass.Int32(gcu_chunks[nseq])
                        ntb = cutlass.Int32(gcu_seqlens[nseq]) + nlc * BT
                        nvr = cutlass.Int32(gcu_seqlens[nseq + 1]) - ntb
                        if nvr > BT:
                            nvr = cutlass.Int32(BT)
                        if lane == 0:
                            if cutlass.const_expr((lc + 1) & 1):
                                cute.arch.mbarrier_arrive_and_expect_tx(
                                    mbar_gate1, G_TX
                                )
                                tma_load_4d(
                                    p_g_raw, desc_g, mbar_gate1, 0, ntb, head, 0
                                )
                            else:
                                cute.arch.mbarrier_arrive_and_expect_tx(
                                    mbar_gate0, G_TX
                                )
                                tma_load_4d(
                                    p_g_raw, desc_g, mbar_gate0, 0, ntb, head, 0
                                )
                        load_beta_stage(
                            gbeta,
                            smem_beta,
                            (lc + 1) & 1,
                            ntb,
                            nvr,
                            head,
                            lane,
                            heads,
                        )
            elif warp_id == 3:
                # Warp 3 no longer issues any of the prefetch, but it still
                # acquires raw_operands_released: that is what makes the other
                # warps' Ki stores visible to its Ak path below, without
                # relying on a chained release/acquire through warp 0.
                mbar_spin_wait(mbar_raw_released, phase)
                cute.arch.fence_view_async_shared()
                cute.arch.sync_warp()
                if lane == 0:
                    # Section 7.4: Kd segment 1 plus both Qd segments, one group.
                    tma_store_3d(
                        desc_factor,
                        p_kd + BF16_SEG_STRIDE,
                        BF16_SEG_ELEMS,
                        gchunk * BT,
                        head,
                    )
                    tma_store_3d(desc_factor, p_qd, 0, gchunk * BT, heads + head)
                    tma_store_3d(
                        desc_factor,
                        p_qd + BF16_SEG_STRIDE,
                        BF16_SEG_ELEMS,
                        gchunk * BT,
                        heads + head,
                    )
                    tma_store_commit_group()

            # ---- AINV_beta, then Aq (warp 2) and Ak (warps 1 and 3) ---
            # pairwise_ready is a plain release/acquire pair on smem_ainv, whose
            # only writer is warp 0.  The Ki that warps 1 and 3 read below is
            # covered without leaning on chained visibility: warp 2 acquired
            # k_full_ready directly, and warps 1 and 3 acquired
            # raw_operands_released, on which every warp arrives after its
            # k_full_ready arrival and therefore after its Ki stores.
            if warp_id != 0:
                mbar_spin_wait(mbar_pairwise, phase)
                ai = load_pairwise_a_fragment(p_ainv, lane)
                bst = beta_stage * BT
                blo = pack_bf16x2(smem_beta[bst + 2 * q4], smem_beta[bst + 2 * q4 + 1])
                bhi = pack_bf16x2(
                    smem_beta[bst + 2 * q4 + 8], smem_beta[bst + 2 * q4 + 9]
                )
                ab0 = mul_bf16x2(ai[0], blo)
                ab1 = mul_bf16x2(ai[1], blo)
                ab2 = mul_bf16x2(ai[2], bhi)
                ab3 = mul_bf16x2(ai[3], bhi)

                if warp_id == 2:
                    bb = a_to_b(ab0, ab1, ab2, ab3)
                    qk_a = load_pairwise_a_fragment(p_qk, lane)
                    aq = mma_16x16(
                        qk_a[0],
                        qk_a[1],
                        qk_a[2],
                        qk_a[3],
                        bb[0],
                        bb[1],
                        bb[2],
                        bb[3],
                        ZERO8(),
                    )
                    base = (head * TOTAL_CHUNKS + gchunk) * (BT * BT)
                    # Stage through SMEM so the global store is one contiguous
                    # 16-byte run per lane. Direct stores cannot be:
                    # prepare_pair_idx swaps the column halves
                    # (col ^ 8), so one store instruction only ever fills half
                    # of each row's 32 bytes, spraying 128 B over 8 sectors.
                    # The four instructions tile the 512 B exactly, but L1 does
                    # not merge across instructions, so 32 sectors leave the SM
                    # where 16 suffice. Staging reduces this to five store
                    # instructions and 32 sectors per chunk.
                    #
                    # smem_qk is free here: warp 2 owns it alone and has just
                    # consumed it into qk_a, so this costs no shared memory.
                    cute.arch.sync_warp()
                    aqf = acc_to_a_fragment(aq)
                    srow, scol = prepare_stmatrix_coord(lane)
                    stmatrix_x4(
                        p_qk + prepare_pair_idx(srow, scol),
                        aqf[0],
                        aqf[1],
                        aqf[2],
                        aqf[3],
                    )
                    cute.arch.sync_warp()
                    cute.autovec_copy(
                        vec8_bf16(p_qk, lane * 8),
                        vec_at(ws_aq.iterator, base + lane * 8, 8),
                    )

                else:
                    # Section 12.3 step 8: pairwise_ready above proved warp 0 is
                    # done reading smem_kd_then_ak; this waits for the warp's own
                    # Kd TMA store to have finished *reading* its half, which is
                    # the other half of the condition for overwriting it.  The
                    # warp synchronization joins the two before any lane writes.
                    if lane == 0:
                        tma_store_wait_read(0)
                    cute.arch.sync_warp()
                    if warp_id == 3:
                        if lc + 1 < CPC:
                            if lc + 1 < my_chunks:
                                warp_arrive(mbar_q_released, lane)

                    at0, at1, at2, at3 = a_to_a_transposed(ab0, ab1, ab2, ab3)
                    tile_base = 0
                    if warp_id == 3:
                        tile_base = 4
                    for t in cutlass.range_constexpr(4):
                        d0 = (tile_base + t) * 16
                        ki0, ki1, ki2, ki3 = load_a_fragment(p_ki, 0, d0, lane, 0)
                        gl = pack_bf16x2(
                            cutlass.Float32(smem_gamma[d0 + 2 * q4]),
                            cutlass.Float32(smem_gamma[d0 + 2 * q4 + 1]),
                        )
                        gh = pack_bf16x2(
                            cutlass.Float32(smem_gamma[d0 + 2 * q4 + 8]),
                            cutlass.Float32(smem_gamma[d0 + 2 * q4 + 9]),
                        )
                        kb = a_to_b(
                            mul_bf16x2(ki0, gl),
                            mul_bf16x2(ki1, gl),
                            mul_bf16x2(ki2, gh),
                            mul_bf16x2(ki3, gh),
                        )
                        akc = mma_16x16(
                            at0, at1, at2, at3, kb[0], kb[1], kb[2], kb[3], ZERO8()
                        )
                        # publish Ak.T with stmatrix.x4 through
                        # the row-^8 image, into the Kd stage that is now dead.
                        # Warp 1 owns d < 64 -> bytes [0,2048), warp 3 the rest.
                        akf = acc_to_a_fragment(akc)
                        srow, scol = prepare_stmatrix_coord(lane)
                        stmatrix_x4(
                            p_kd + kr_ak_idx(srow, d0 + scol),
                            akf[0],
                            akf[1],
                            akf[2],
                            akf[3],
                        )

                    if lc + 1 < CPC:
                        if lc + 1 < my_chunks:
                            # Ki aliases raw K.  Both Ak producer warps arrive
                            # only after their final Ki fragment load.
                            warp_arrive(mbar_k_released, lane)
                            if warp_id == 1:
                                mbar_spin_wait(mbar_k_released, phase)
                                nchunk = gchunk + 1
                                nseq = cutlass.Int32(gchunk_to_seq[nchunk])
                                nlc = nchunk - cutlass.Int32(gcu_chunks[nseq])
                                ntb = cutlass.Int32(gcu_seqlens[nseq]) + nlc * BT
                                if lane == 0:
                                    if cutlass.const_expr((lc + 1) & 1):
                                        tma_load_4d(
                                            p_k, desc_k, mbar_qk1, 0, ntb, head, 0
                                        )
                                    else:
                                        tma_load_4d(
                                            p_k, desc_k, mbar_qk0, 0, ntb, head, 0
                                        )

                    # Section 7.4: each store warp moves its own Ak segment.
                    # This is what removes the CTA barrier and the 128-thread
                    # re-read the vector-store version needed -- the warp that
                    # produced the half is the warp that ships it.
                    cute.arch.fence_view_async_shared()
                    cute.arch.sync_warp()
                    if lane == 0:
                        seg = tile_base // 4
                        tma_store_3d(
                            desc_factor,
                            p_kd + seg * BF16_SEG_STRIDE,
                            seg * BF16_SEG_ELEMS,
                            gchunk * BT,
                            2 * heads + head,  # Ak is plane region 2
                        )
                        tma_store_commit_group()
                        # Section 12.3 step 8: the source stage cannot be reused
                        # until the store has read it.
                        tma_store_wait_read(0)

            # Chunk recycle (Section 12.3 step 9).
            cute.arch.barrier()
