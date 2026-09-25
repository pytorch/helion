# pyrefly: ignore-errors
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ruff: noqa: ANN001, ANN202
"""Five independent native-CuTe BT32 factor producers, explicit v2 math.

Based on the inspected FlashInfer 0.7.0 CAKE388 factor schedule; no FlashInfer
runtime, NVRTC/C++ source, or baseline binary is invoked.
"""

from __future__ import annotations

import cutlass
from cutlass._mlir.dialects import llvm
from cutlass._mlir.extras import types as _T
import cutlass.cute as cute
from cutlass.cutlass_dsl import dsl_user_op
import cutlass.experimental.primitives as prims

from . import common as cm
from helion._compiler.cute.chunk_prefill_tmem import movmatrix_b16
from helion._compiler.cute.chunk_prefill_tmem import pack_input_b16x2_to_i32
from helion._compiler.cute.chunk_prepare_split_alias_device import mma_blockdiag_8x8_f16
from helion._compiler.cute.chunk_prepare_split_alias_device import mma_m16n8k16_f16
from helion._compiler.cute.chunk_prepare_split_alias_device import pack_f16x2
from helion._compiler.cute.kda_device_primitives import mma_m16n8k16_bf16


@cute.jit
def sw128(row, col):
    """BT32 segment-major SW128 byte offset, 64 BF16 features per segment."""
    offset = (col // 64) * 4096 + row * 128 + (col % 64) * 2
    return offset ^ (((offset >> 7) & 7) << 4)


@cute.jit
def sw32(row, col):
    """32x32 inverse, 16 BF16 columns per SW32 segment."""
    offset = (col // 16) * 1024 + row * 32 + (col % 16) * 2
    return offset ^ (((offset >> 7) & 1) << 4)


@cute.jit
def work_sw128(row, col):
    """Inverse workspace padded to64 BF16 columns, matching CAKE388."""
    offset = row * 128 + col * 2
    return offset ^ (((offset >> 7) & 7) << 4)


@cute.jit
def zero8():
    z = cutlass.Float32(0.0)
    return z, z, z, z, z, z, z, z


@cute.jit
def mma16(a, b, acc, dtype: cutlass.Constexpr):
    """One16x16 result, two native m16n8k16 instructions."""
    lo = mma_m16n8k16_bf16(
        a[0], a[1], a[2], a[3], b[0], b[1], acc[0], acc[1], acc[2], acc[3]
    )
    hi = mma_m16n8k16_bf16(
        a[0], a[1], a[2], a[3], b[2], b[3], acc[4], acc[5], acc[6], acc[7]
    )
    return lo[0], lo[1], lo[2], lo[3], hi[0], hi[1], hi[2], hi[3]


@cute.jit
def bf16_pack8(values):
    return [
        pack_input_b16x2_to_i32(values[0], values[1], cutlass.BFloat16),
        pack_input_b16x2_to_i32(values[2], values[3], cutlass.BFloat16),
        pack_input_b16x2_to_i32(values[4], values[5], cutlass.BFloat16),
        pack_input_b16x2_to_i32(values[6], values[7], cutlass.BFloat16),
    ]


@cute.jit
def pairwise16(smem_base, lhs_offset, rhs_offset, row_base, col_base, lane):
    """[16,128] @ [16,128].T with unchanged eight K16 slice order."""
    acc = zero8()
    for k_slice in cutlass.range_constexpr(8):
        a_row = row_base + lane % 16
        a_col = k_slice * 16 + (lane // 16) * 8
        b_row = col_base + (lane // 16) * 8 + lane % 8
        b_col = k_slice * 16 + ((lane // 8) % 2) * 8
        a = prims.ldmatrix(
            cm.sptr(smem_base, lhs_offset + sw128(a_row, a_col), cutlass.BFloat16),
            4,
            prims.MMALayout.ROW,
        )
        b = prims.ldmatrix(
            cm.sptr(smem_base, rhs_offset + sw128(b_row, b_col), cutlass.BFloat16),
            4,
            prims.MMALayout.ROW,
        )
        acc = mma16(a, b, acc, cutlass.BFloat16)
    return acc


@cute.jit
def lower_values(smem_base, beta_offset, acc, row_base, col_base, lane):
    """FP32 beta multiply followed by the explicit BF16 strict-lower boundary."""
    beta_row0 = row_base + lane // 4
    beta0 = cm.sptr(smem_base, beta_offset + beta_row0 * 4, cutlass.Float32).load()
    beta1 = cm.sptr(
        smem_base, beta_offset + (beta_row0 + 8) * 4, cutlass.Float32
    ).load()
    values = cutlass.Array(cutlass.Float32, 8, space=cutlass.AddressSpace.rmem)
    for index in cutlass.range_constexpr(8):
        row = row_base + lane // 4 + ((index % 4) // 2) * 8
        col = col_base + (index // 4) * 8 + (lane % 4) * 2 + index % 2
        value = cutlass.Float32(0.0)
        if row > col:
            beta = beta0 if cutlass.const_expr(index % 4 < 2) else beta1
            value = cutlass.Float32(cutlass.BFloat16(acc[index] * beta))
        values[index] = value
    return values


@cute.jit
def inverse16_v2(values, lane):
    """Actual CAKE388: FP32 additive bases, degree3 coupling snapshot.

    Accumulator slots0/1 and6/7 are the two8x8 diagonals. The lower8x8
    coupling is slots2/3; upper-right remains exact zero. No old BT16
    inverse helper is reused because its casts/snapshot differ.
    """
    z = cutlass.Int32(0)
    fz = cutlass.Float32(0.0)
    d0 = pack_f16x2(values[0], values[1])
    d3 = pack_f16x2(values[6], values[7])
    d2 = mma_blockdiag_8x8_f16(d0, d3, d0, d3)
    d20 = pack_f16x2(d2[0], d2[1])
    d23 = pack_f16x2(d2[2], d2[3])
    row = lane // 4
    col = (lane % 4) * 2
    n00 = -values[0] + cutlass.Float32(row == col)
    n01 = -values[1] + cutlass.Float32(row == col + 1)
    n10 = -values[6] + cutlass.Float32(row == col)
    n11 = -values[7] + cutlass.Float32(row == col + 1)
    n0 = pack_f16x2(n00, n01)
    n3 = pack_f16x2(n10, n11)
    add2 = mma_blockdiag_8x8_f16(n0, n3, d20, d23)
    n00 = n00 + add2[0]
    n01 = n01 + add2[1]
    n10 = n10 + add2[2]
    n11 = n11 + add2[3]
    # Keep H(N1) for coupling; do not refresh it after the D4 correction.
    coupling0 = pack_f16x2(n00, n01)
    coupling3 = pack_f16x2(n10, n11)
    d4 = mma_blockdiag_8x8_f16(d20, d23, d20, d23)
    d40 = pack_f16x2(d4[0], d4[1])
    d43 = pack_f16x2(d4[2], d4[3])
    add4 = mma_blockdiag_8x8_f16(coupling0, coupling3, d40, d43)
    n00 = n00 + add4[0]
    n01 = n01 + add4[1]
    n10 = n10 + add4[2]
    n11 = n11 + add4[3]
    cross = pack_f16x2(values[2], values[3])
    first = mma_m16n8k16_f16(
        coupling0, z, z, coupling3, z, movmatrix_b16(cross), fz, fz, fz, fz
    )
    negative = pack_f16x2(-first[2], -first[3])
    coupled = mma_m16n8k16_f16(
        z, negative, z, z, movmatrix_b16(coupling0), z, fz, fz, fz, fz
    )
    return n00, n01, coupled[2], coupled[3], fz, fz, n10, n11


@cute.jit
def store_inverse_work(smem_base, offset, values, row_base, col_base, lane):
    row = row_base + lane % 16
    col = col_base + (lane // 16) * 8
    ptr = cm.sptr(smem_base, offset + work_sw128(row, col), cutlass.BFloat16)
    prims.stmatrix(
        ptr, bf16_pack8(values), prims.MMALayout.ROW, shape=prims.StoreShape.M8N8
    )


@cute.jit
def store_qk_transpose(smem_base, combined_offset, acc, row_base, col_base, lane):
    """Publish QK.T in the N128..159 slab of the final [32,160] operand."""
    values = cutlass.Array(cutlass.Float32, 8, space=cutlass.AddressSpace.rmem)
    for index in cutlass.range_constexpr(8):
        row = row_base + lane // 4 + ((index % 4) // 2) * 8
        col = col_base + (index // 4) * 8 + (lane % 4) * 2 + index % 2
        values[index] = acc[index] if row >= col else cutlass.Float32(0.0)
    packed = bf16_pack8(values)
    for pair in cutlass.range_constexpr(2):
        row = col_base + pair * 8 + lane % 8
        col = 128 + row_base + (lane // 8) * 8
        ptr = cm.sptr(smem_base, combined_offset + sw128(row, col), cutlass.BFloat16)
        prims.stmatrix(
            ptr,
            [packed[pair * 2], packed[pair * 2 + 1]],
            prims.MMALayout.COL,
            shape=prims.StoreShape.M8N8,
        )


@dsl_user_op
def late_lane_id(*, loc=None, ip=None):
    """Rematerialize a lane value at use, outside invariant-address hoisting."""
    return cutlass.Int32(
        llvm.inline_asm(
            _T.IntegerType.get_signless(32),
            [],
            "mov.u32 $0, %laneid;",
            "=r,~{memory}",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def late_tid_x(*, loc=None, ip=None):
    """Rematerialize the native thread index at its phase of use."""
    return cutlass.Int32(
        llvm.inline_asm(
            _T.IntegerType.get_signless(32),
            [],
            "mov.u32 $0, %tid.x;",
            "=r,~{memory}",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def finish_inverse32(smem_base, work_offset, inverse_offset, lane):
    """One warp assembles BF16 outer16 correction and publishes all32x32."""
    row = lane % 16
    col = (lane // 16) * 8
    d = prims.ldmatrix(
        cm.sptr(
            smem_base, work_offset + work_sw128(16 + row, 16 + col), cutlass.BFloat16
        ),
        4,
        prims.MMALayout.ROW,
    )
    c = prims.ldmatrix(
        cm.sptr(smem_base, work_offset + work_sw128(16 + row, col), cutlass.BFloat16),
        4,
        prims.MMALayout.COL,
    )
    first = mma16(d, c, zero8(), cutlass.BFloat16)
    negative = bf16_pack8(
        (
            -first[0],
            -first[1],
            -first[2],
            -first[3],
            -first[4],
            -first[5],
            -first[6],
            -first[7],
        )
    )
    a = prims.ldmatrix(
        cm.sptr(smem_base, work_offset + work_sw128(row, col), cutlass.BFloat16),
        4,
        prims.MMALayout.COL,
    )
    coupled = mma16(negative, a, zero8(), cutlass.BFloat16)
    store_lane = late_lane_id()
    store_row = store_lane % 16
    store_col = (store_lane // 16) * 8
    prims.stmatrix(
        cm.sptr(
            smem_base,
            inverse_offset + sw32(16 + store_row, 16 + store_col),
            cutlass.BFloat16,
        ),
        d,
        prims.MMALayout.ROW,
        shape=prims.StoreShape.M8N8,
    )
    prims.stmatrix(
        cm.sptr(
            smem_base, inverse_offset + sw32(store_row, store_col), cutlass.BFloat16
        ),
        a,
        prims.MMALayout.COL,
        shape=prims.StoreShape.M8N8,
    )
    prims.stmatrix(
        cm.sptr(
            smem_base,
            inverse_offset + sw32(16 + store_row, store_col),
            cutlass.BFloat16,
        ),
        bf16_pack8(coupled),
        prims.MMALayout.ROW,
        shape=prims.StoreShape.M8N8,
    )
    zi = cutlass.Int32(0)
    prims.stmatrix(
        cm.sptr(
            smem_base,
            inverse_offset + sw32(store_row, 16 + store_col),
            cutlass.BFloat16,
        ),
        [zi, zi, zi, zi],
        prims.MMALayout.ROW,
        shape=prims.StoreShape.M8N8,
    )


@cute.jit
def cp_async_bf16x8(destination, source, valid):
    """Copy one aligned eight-element BF16 vector, zero-filling invalid rows."""
    aligned_destination = cute.make_ptr(
        destination.dtype,
        destination.toint(),
        destination.memspace,
        assumed_align=16,
    )
    aligned_source = cute.make_ptr(
        source.dtype,
        source.toint(),
        source.memspace,
        assumed_align=16,
    )
    copy_size = cutlass.Int32(0)
    if valid:
        copy_size = cutlass.Int32(16)
    cute.arch.cp_async_shared_global(
        aligned_destination,
        aligned_source,
        16,
        "cg",
        cp_size=copy_size,
    )


@cute.jit
def tail_copy_qkg(
    smem_base, q, k, gate, begin, seqlen, head, heads, chunk, stage_bytes, tid
):
    """Predicated 16-byte copies with the same images as full-tile TMA."""
    for work_pass in cutlass.range_constexpr(4):
        item = work_pass * 128 + tid
        row = item // 16
        col = (item % 16) * 8
        token = begin + cutlass.Int64(chunk * cm.BT + row)
        valid = chunk * cm.BT + row < seqlen
        source_offset = (token * heads + head) * 128 + col
        cp_async_bf16x8(
            cm.sptr(
                smem_base,
                cm.Q_RAW_PREFETCH + stage_bytes + sw128(row, col),
                cutlass.BFloat16,
            ),
            q.iterator + source_offset,
            valid,
        )
        cp_async_bf16x8(
            cm.sptr(
                smem_base,
                cm.KD + stage_bytes + sw128(row, col),
                cutlass.BFloat16,
            ),
            k.iterator + source_offset,
            valid,
        )
        cp_async_bf16x8(
            cm.sptr(
                smem_base,
                cm.GATE_RAW + stage_bytes + (row * 128 + col) * 2,
                cutlass.BFloat16,
            ),
            gate.iterator + source_offset,
            valid,
        )
    cute.arch.cp_async_commit_group()
    cute.arch.cp_async_wait_group(0)


@cute.jit
def tail_copy_v(smem_base, v, begin, seqlen, head, chunk, stage_bytes, tid):
    for work_pass in cutlass.range_constexpr(4):
        item = work_pass * 128 + tid
        row = item // 16
        col = (item % 16) * 8
        token = begin + cutlass.Int64(chunk * cm.BT + row)
        valid = chunk * cm.BT + row < seqlen
        source_offset = (token * cutlass.Int64(v.shape[2]) + head) * 128 + col
        cp_async_bf16x8(
            cm.sptr(
                smem_base,
                cm.V + stage_bytes + (row * 128 + col) * 2,
                cutlass.BFloat16,
            ),
            v.iterator + source_offset,
            valid,
        )
    cute.arch.cp_async_commit_group()
    cute.arch.cp_async_wait_group(0)


@cute.jit
def materialize_centered(
    smem_base, stage_bytes, tid, scale, gate_scale_log2, FAST_RCP: cutlass.Constexpr
):
    """Each16-lane group normalizes one row; each thread owns8 features."""
    center = gate_scale_log2 * cutlass.Float32(16.0)
    for work_pass in cutlass.range(4, unroll=1):
        item = work_pass * 128 + tid
        row = item // 16
        col = (item % 16) * 8
        offset = sw128(row, col)
        qv = (
            cm.sptr(
                smem_base, cm.Q_RAW_PREFETCH + stage_bytes + offset, cutlass.BFloat16
            )
            .load(count=8, alignment=16)
            .to(cutlass.Float32)
        )
        kv = (
            cm.sptr(smem_base, cm.KD + stage_bytes + offset, cutlass.BFloat16)
            .load(count=8, alignment=16)
            .to(cutlass.Float32)
        )
        q0 = cutlass.Float32(0.0)
        q1 = cutlass.Float32(0.0)
        k0 = cutlass.Float32(0.0)
        k1 = cutlass.Float32(0.0)
        for pair in cutlass.range_constexpr(4):
            q0, q1 = cm.ffma2(
                (qv[pair * 2], qv[pair * 2 + 1]),
                (qv[pair * 2], qv[pair * 2 + 1]),
                (q0, q1),
            )
            k0, k1 = cm.ffma2(
                (kv[pair * 2], kv[pair * 2 + 1]),
                (kv[pair * 2], kv[pair * 2 + 1]),
                (k0, k1),
            )
        qs = q0 + q1
        ks = k0 + k1
        for shift in cutlass.range_constexpr(4):
            delta = 8 >> shift
            qs = qs + cutlass.Float32(
                prims.shfl_sync(cute.arch.FULL_MASK, qs, delta, 0x1F, prims.Shfl.BFLY)
            )
            ks = ks + cutlass.Float32(
                prims.shfl_sync(cute.arch.FULL_MASK, ks, delta, 0x1F, prims.Shfl.BFLY)
            )
        qi = cute.math.rsqrt(qs + cutlass.Float32(1e-6), fastmath=True)
        ki = cute.math.rsqrt(ks + cutlass.Float32(1e-6), fastmath=True)
        qd = cutlass.Array(
            cutlass.BFloat16, 8, space=cutlass.AddressSpace.rmem, alignment=16
        )
        kd = cutlass.Array(
            cutlass.BFloat16, 8, space=cutlass.AddressSpace.rmem, alignment=16
        )
        kinv = cutlass.Array(
            cutlass.BFloat16, 8, space=cutlass.AddressSpace.rmem, alignment=16
        )
        for pair in cutlass.range_constexpr(4):
            element = pair * 2
            prefix0 = cm.sptr(
                smem_base,
                cm.GATE_PREFIX + stage_bytes + (row * 128 + col + element) * 4,
                cutlass.Float32,
            ).load()
            prefix1 = cm.sptr(
                smem_base,
                cm.GATE_PREFIX + stage_bytes + (row * 128 + col + element + 1) * 4,
                cutlass.Float32,
            ).load()
            decay0 = cute.math.exp2(prefix0 - center, fastmath=True)
            decay1 = cute.math.exp2(prefix1 - center, fastmath=True)
            qn0, qn1 = cm.fmul2((qv[element], qv[element + 1]), (qi, qi))
            kn0, kn1 = cm.fmul2((kv[element], kv[element + 1]), (ki, ki))
            qscaled0, qscaled1 = cm.fmul2((qn0, qn1), (scale, scale))
            qd0, qd1 = cm.fmul2((qscaled0, qscaled1), (decay0, decay1))
            kd0, kd1 = cm.fmul2((kn0, kn1), (decay0, decay1))
            reciprocal0 = cute.math.rcp(decay0, approx=True, ftz=FAST_RCP)
            reciprocal1 = cute.math.rcp(decay1, approx=True, ftz=FAST_RCP)
            ki0, ki1 = cm.fmul2((kn0, kn1), (reciprocal0, reciprocal1))
            qd[element] = cutlass.BFloat16(qd0)
            qd[element + 1] = cutlass.BFloat16(qd1)
            kd[element] = cutlass.BFloat16(kd0)
            kd[element + 1] = cutlass.BFloat16(kd1)
            kinv[element] = cutlass.BFloat16(ki0)
            kinv[element + 1] = cutlass.BFloat16(ki1)
        cm.sptr(smem_base, cm.QD + stage_bytes + offset, cutlass.BFloat16).store(
            qd.data_ptr().load(count=8, alignment=16), alignment=16
        )
        cm.sptr(smem_base, cm.KD + stage_bytes + offset, cutlass.BFloat16).store(
            kd.data_ptr().load(count=8, alignment=16), alignment=16
        )
        cm.sptr(smem_base, cm.KI + stage_bytes + offset, cutlass.BFloat16).store(
            kinv.data_ptr().load(count=8, alignment=16), alignment=16
        )


@cute.jit
def restore_qk(
    smem_base,
    stage_bytes,
    local_warp,
    lane,
    FIRST_ROW: cutlass.Constexpr,
    ROW_STRIDE: cutlass.Constexpr,
    PASSES: cutlass.Constexpr,
):
    """Restore disjoint rows with the original BF16/FP32 cast boundaries."""
    restore_lane = late_lane_id()
    center_scale = cm.sptr(
        smem_base,
        cm.RESTORE_FACTOR + stage_bytes + 128 * 4,
        cutlass.Float32,
    ).load()
    col = (restore_lane & 15) << 3
    row_base = FIRST_ROW + (restore_lane >> 4)
    if cutlass.const_expr(ROW_STRIDE == 4):
        row_base = row_base + ((local_warp & 1) << 1)
    column_offset = ((restore_lane & 8) << 9) | ((restore_lane & 7) << 4)
    restore_factors = cm.sptr(
        smem_base,
        cm.RESTORE_FACTOR + stage_bytes + col * 4,
        cutlass.Float32,
    ).load(count=8, alignment=16)
    for work_pass in cutlass.range(PASSES, unroll=1):
        row = work_pass * ROW_STRIDE + row_base
        offset = (column_offset | (row << 7)) ^ ((row & 7) << 4)
        qv = (
            cm.sptr(smem_base, cm.QD + stage_bytes + offset, cutlass.BFloat16)
            .load(count=8, alignment=16)
            .to(cutlass.Float32)
        )
        kv = (
            cm.sptr(smem_base, cm.KI + stage_bytes + offset, cutlass.BFloat16)
            .load(count=8, alignment=16)
            .to(cutlass.Float32)
        )
        qr = cutlass.Array(
            cutlass.BFloat16, 8, space=cutlass.AddressSpace.rmem, alignment=16
        )
        kr = cutlass.Array(
            cutlass.BFloat16, 8, space=cutlass.AddressSpace.rmem, alignment=16
        )
        for pair in cutlass.range_constexpr(4):
            element = pair * 2
            qr0, qr1 = cm.fmul2(
                (qv[element], qv[element + 1]), (center_scale, center_scale)
            )
            kr0, kr1 = cm.fmul2(
                (kv[element], kv[element + 1]),
                (restore_factors[element], restore_factors[element + 1]),
            )
            qr[element] = cutlass.BFloat16(qr0)
            qr[element + 1] = cutlass.BFloat16(qr1)
            kr[element] = cutlass.BFloat16(kr0)
            kr[element + 1] = cutlass.BFloat16(kr1)
        cm.sptr(smem_base, cm.QD + stage_bytes + offset, cutlass.BFloat16).store(
            qr.data_ptr().load(count=8, alignment=16), alignment=16
        )
        cm.sptr(smem_base, cm.KR + stage_bytes + offset, cutlass.BFloat16).store(
            kr.data_ptr().load(count=8, alignment=16), alignment=16
        )


@cute.jit
def gate_prefix(
    smem_base,
    stage_bytes,
    tid,
    chunk,
    seqlen,
    gate_rate,
    bias,
    gate_scale_log2,
    MASK_TAIL: cutlass.Constexpr,
):
    """Serial FP32 gate-prefix DAG with a constexpr full-tile mask choice."""
    prefix = cutlass.Float32(0.0)
    for group in cutlass.range_constexpr(cm.BT // 4):
        row0 = group * 4 + 0
        row1 = group * 4 + 1
        row2 = group * 4 + 2
        row3 = group * 4 + 3
        raw0 = cutlass.Float32(
            cm.sptr(
                smem_base,
                cm.GATE_RAW + stage_bytes + (row0 * 128 + tid) * 2,
                cutlass.BFloat16,
            ).load()
        )
        raw1 = cutlass.Float32(
            cm.sptr(
                smem_base,
                cm.GATE_RAW + stage_bytes + (row1 * 128 + tid) * 2,
                cutlass.BFloat16,
            ).load()
        )
        raw2 = cutlass.Float32(
            cm.sptr(
                smem_base,
                cm.GATE_RAW + stage_bytes + (row2 * 128 + tid) * 2,
                cutlass.BFloat16,
            ).load()
        )
        raw3 = cutlass.Float32(
            cm.sptr(
                smem_base,
                cm.GATE_RAW + stage_bytes + (row3 * 128 + tid) * 2,
                cutlass.BFloat16,
            ).load()
        )
        increment0 = cutlass.Float32(0.0)
        increment1 = cutlass.Float32(0.0)
        increment2 = cutlass.Float32(0.0)
        increment3 = cutlass.Float32(0.0)
        if cutlass.const_expr(MASK_TAIL):
            if chunk * cm.BT + row0 < seqlen:
                activated0 = cute.math.tanh(
                    gate_rate * (raw0 + bias) * cutlass.Float32(0.5), approx=True
                ) * cutlass.Float32(0.5) + cutlass.Float32(0.5)
                increment0 = gate_scale_log2 * activated0
            if chunk * cm.BT + row1 < seqlen:
                activated1 = cute.math.tanh(
                    gate_rate * (raw1 + bias) * cutlass.Float32(0.5), approx=True
                ) * cutlass.Float32(0.5) + cutlass.Float32(0.5)
                increment1 = gate_scale_log2 * activated1
            if chunk * cm.BT + row2 < seqlen:
                activated2 = cute.math.tanh(
                    gate_rate * (raw2 + bias) * cutlass.Float32(0.5), approx=True
                ) * cutlass.Float32(0.5) + cutlass.Float32(0.5)
                increment2 = gate_scale_log2 * activated2
            if chunk * cm.BT + row3 < seqlen:
                activated3 = cute.math.tanh(
                    gate_rate * (raw3 + bias) * cutlass.Float32(0.5), approx=True
                ) * cutlass.Float32(0.5) + cutlass.Float32(0.5)
                increment3 = gate_scale_log2 * activated3
        else:
            half = cutlass.Float32(0.5)
            tanh0 = cute.math.tanh(gate_rate * (raw0 + bias) * half, approx=True)
            tanh1 = cute.math.tanh(gate_rate * (raw1 + bias) * half, approx=True)
            tanh2 = cute.math.tanh(gate_rate * (raw2 + bias) * half, approx=True)
            tanh3 = cute.math.tanh(gate_rate * (raw3 + bias) * half, approx=True)
            activated0, activated1 = cm.ffma2(
                (tanh0, tanh1), (half, half), (half, half)
            )
            activated2, activated3 = cm.ffma2(
                (tanh2, tanh3), (half, half), (half, half)
            )
            increment0 = gate_scale_log2 * activated0
            increment1 = gate_scale_log2 * activated1
            increment2 = gate_scale_log2 * activated2
            increment3 = gate_scale_log2 * activated3
        prefix = prefix + increment0
        cm.sptr(
            smem_base,
            cm.GATE_PREFIX + stage_bytes + (row0 * 128 + tid) * 4,
            cutlass.Float32,
        ).store(prefix)
        prefix = prefix + increment1
        cm.sptr(
            smem_base,
            cm.GATE_PREFIX + stage_bytes + (row1 * 128 + tid) * 4,
            cutlass.Float32,
        ).store(prefix)
        prefix = prefix + increment2
        cm.sptr(
            smem_base,
            cm.GATE_PREFIX + stage_bytes + (row2 * 128 + tid) * 4,
            cutlass.Float32,
        ).store(prefix)
        prefix = prefix + increment3
        cm.sptr(
            smem_base,
            cm.GATE_PREFIX + stage_bytes + (row3 * 128 + tid) * 4,
            cutlass.Float32,
        ).store(prefix)
    return prefix


@cute.jit
def factor_loop(
    smem_base,
    q,
    k,
    v,
    gate,
    beta,
    a_log,
    dt_bias,
    desc_q,
    desc_k,
    desc_gate,
    desc_beta,
    desc_v,
    begin,
    seqlen,
    head,
    heads,
    num_chunks,
    team,
    local_warp,
    lane,
    scale: cutlass.Float32,
    gate_scale_log2: cutlass.Float32,
):
    """Owner-slot loop; all128 threads of exactly one factor team participate."""
    tid = local_warp * 32 + lane
    stage_bytes = team * cm.STAGE_BYTES
    gate_rate = cute.math.exp2(
        cutlass.Float32(a_log[head]) * cutlass.Float32(cm.LOG2_E), fastmath=True
    )
    bias = cutlass.Float32(dt_bias[head, tid])
    iterations = (num_chunks + cm.STAGES - 1 - team) // cm.STAGES
    for iteration in cutlass.range(iterations, unroll=1):
        chunk = iteration * cm.STAGES + team
        phase = iteration % 2
        early_beta = cutlass.Float32(0.0)
        # Qd/gate and Kd retire after StateQ, before final Kr/QK consumers.
        cm.wait(cm.bptr(smem_base, cm.RAW_INPUTS_FREE, team), phase ^ 1)
        full = (chunk + 1) * cm.BT <= seqlen
        if full:
            if local_warp == 0:
                cm.expect_tx(cm.bptr(smem_base, cm.GATE_RAW_FULL, team), 8704)
                cm.expect_tx(cm.bptr(smem_base, cm.QK_RAW_FULL, team), 16384)
                if prims.elect_sync():
                    token = cutlass.Int32(begin) + chunk * cm.BT
                    prims.cp_async_bulk_tensor_shared_cta_global(
                        cm.sptr(smem_base, cm.GATE_RAW + stage_bytes, cutlass.BFloat16),
                        desc_gate.get_ptr(),
                        (0, head, token),
                        cm.bptr(smem_base, cm.GATE_RAW_FULL, team),
                    )
                    prims.cp_async_bulk_tensor_shared_cta_global(
                        cm.sptr(smem_base, cm.BETA_RAW + stage_bytes, cutlass.BFloat16),
                        desc_beta.get_ptr(),
                        ((head // 8) * 8, token),
                        cm.bptr(smem_base, cm.GATE_RAW_FULL, team),
                    )
                    prims.cp_async_bulk_tensor_shared_cta_global(
                        cm.sptr(smem_base, cm.KD + stage_bytes, cutlass.BFloat16),
                        desc_k.get_ptr(),
                        (0, token, 0, head, 0),
                        cm.bptr(smem_base, cm.QK_RAW_FULL, team),
                    )
            cm.wait(cm.bptr(smem_base, cm.GATE_RAW_FULL, team), phase)
            if local_warp == 2:
                raw = cutlass.Float32(
                    cm.sptr(
                        smem_base,
                        cm.BETA_RAW + stage_bytes + lane * 16 + (head % 8) * 2,
                        cutlass.BFloat16,
                    ).load()
                )
                early_beta = cute.math.tanh(
                    raw * cutlass.Float32(0.5), approx=True
                ) * cutlass.Float32(0.5) + cutlass.Float32(0.5)
        # Q raw aliases Kr/final QK, so its copy retains the final-use waits.
        cm.wait(cm.bptr(smem_base, cm.SMEM_FREE, team), phase ^ 1)
        cm.wait(cm.bptr(smem_base, cm.V_FREE, team), phase ^ 1)
        if full:
            if local_warp == 0:
                if prims.elect_sync():
                    token = cutlass.Int32(begin) + chunk * cm.BT
                    prims.cp_async_bulk_tensor_shared_cta_global(
                        cm.sptr(
                            smem_base, cm.Q_RAW_PREFETCH + stage_bytes, cutlass.BFloat16
                        ),
                        desc_q.get_ptr(),
                        (0, token, 0, head, 0),
                        cm.bptr(smem_base, cm.QK_RAW_FULL, team),
                    )
        else:
            tail_copy_qkg(
                smem_base,
                q,
                k,
                gate,
                begin,
                seqlen,
                head,
                heads,
                chunk,
                stage_bytes,
                tid,
            )
            cm.team_sync(team)
            if local_warp == 0:
                cm.arrive(cm.bptr(smem_base, cm.GATE_RAW_FULL, team))
                cm.arrive(cm.bptr(smem_base, cm.QK_RAW_FULL, team))
        if not full:
            cm.wait(cm.bptr(smem_base, cm.GATE_RAW_FULL, team), phase)
        phase_tid = late_tid_x() & 127
        phase_lane = phase_tid & 31
        if local_warp == 2:
            if not full:
                token = begin + cutlass.Int64(chunk * cm.BT + phase_lane)
                raw = cutlass.Float32(0.0)
                if chunk * cm.BT + phase_lane < seqlen:
                    raw = cutlass.Float32(beta[0, token, head])
                early_beta = cute.math.tanh(
                    raw * cutlass.Float32(0.5), approx=True
                ) * cutlass.Float32(0.5) + cutlass.Float32(0.5)
            cm.sptr(
                smem_base, cm.PREP_BETA + stage_bytes + phase_lane * 4, cutlass.Float32
            ).store(early_beta)
        prefix = cutlass.Float32(0.0)
        if full:
            prefix = gate_prefix(
                smem_base,
                stage_bytes,
                phase_tid,
                chunk,
                seqlen,
                gate_rate,
                bias,
                gate_scale_log2,
                False,
            )
        else:
            prefix = gate_prefix(
                smem_base,
                stage_bytes,
                phase_tid,
                chunk,
                seqlen,
                gate_rate,
                bias,
                gate_scale_log2,
                True,
            )
        restore = cute.math.exp2(
            prefix - gate_scale_log2 * cutlass.Float32(16.0), fastmath=True
        )
        cm.sptr(
            smem_base, cm.RESTORE_FACTOR + stage_bytes + phase_tid * 4, cutlass.Float32
        ).store(restore)
        if phase_tid == 0:
            center_scale = cute.math.exp2(
                gate_scale_log2 * cutlass.Float32(16.0), fastmath=True
            )
            cm.sptr(
                smem_base, cm.RESTORE_FACTOR + stage_bytes + 128 * 4, cutlass.Float32
            ).store(center_scale)
        cm.team_sync(team)
        cm.wait(cm.bptr(smem_base, cm.QK_RAW_FULL, team), phase)
        if (gate_scale_log2 >= cutlass.Float32(-7.5)) & (
            gate_scale_log2 <= cutlass.Float32(7.5)
        ):
            materialize_centered(
                smem_base, stage_bytes, phase_tid, scale, gate_scale_log2, True
            )
        else:
            materialize_centered(
                smem_base, stage_bytes, phase_tid, scale, gate_scale_log2, False
            )
        # The CTA barrier orders ordinary shared writes before pairwise reads.
        cm.team_sync(team)
        total = cm.sptr(
            smem_base, cm.PREFIX_LAST + stage_bytes + phase_tid * 4, cutlass.Float32
        ).load()
        cm.sptr(
            smem_base, cm.GAMMA + stage_bytes + phase_tid * 4, cutlass.Float32
        ).store(cute.math.exp2(total, fastmath=True))
        row_base = (local_warp // 2) * 16
        col_base = (local_warp % 2) * 16
        if row_base >= col_base:
            kk = pairwise16(
                smem_base,
                cm.KD + stage_bytes,
                cm.KI + stage_bytes,
                row_base,
                col_base,
                lane,
            )
            lower = lower_values(
                smem_base, cm.PREP_BETA + stage_bytes, kk, row_base, col_base, lane
            )
            if row_base == col_base:
                inv = inverse16_v2(lower, lane)
                store_inverse_work(
                    smem_base, cm.INV_WORK + stage_bytes, inv, row_base, col_base, lane
                )
            else:
                store_inverse_work(
                    smem_base,
                    cm.INV_WORK + stage_bytes,
                    lower,
                    row_base,
                    col_base,
                    lane,
                )
        else:
            store_inverse_work(
                smem_base, cm.INV_WORK + stage_bytes, zero8(), row_base, col_base, lane
            )
        qk = zero8()
        if row_base >= col_base:
            qk = pairwise16(
                smem_base,
                cm.QD + stage_bytes,
                cm.KI + stage_bytes,
                row_base,
                col_base,
                lane,
            )
        store_qk_transpose(
            smem_base, cm.FINAL_TRANS + stage_bytes, qk, row_base, col_base, lane
        )
        # The CTA barrier orders ordinary shared writes before restore reads.
        cm.team_sync(team)
        if local_warp == 0:
            finish_inverse32(
                smem_base, cm.INV_WORK + stage_bytes, cm.INV + stage_bytes, lane
            )
        if local_warp == 1:
            restore_qk(smem_base, stage_bytes, local_warp, lane, 0, 2, 4)
        elif local_warp >= 2:
            restore_qk(smem_base, stage_bytes, local_warp, lane, 8, 4, 6)
        cm.team_sync(team)
        if local_warp == 0:
            cm.fence_shared()
            cm.arrive(cm.bptr(smem_base, cm.QK_FULL, team))
        # V aliases inverse workspace, whose final reads precede the team sync.
        if full:
            if local_warp == 0:
                cm.expect_tx(cm.bptr(smem_base, cm.V_FULL, team), 8192)
                if prims.elect_sync():
                    token = cutlass.Int32(begin) + chunk * cm.BT
                    prims.cp_async_bulk_tensor_shared_cta_global(
                        cm.sptr(smem_base, cm.V + stage_bytes, cutlass.BFloat16),
                        desc_v.get_ptr(),
                        (0, head, token),
                        cm.bptr(smem_base, cm.V_FULL, team),
                    )
        else:
            tail_copy_v(smem_base, v, begin, seqlen, head, chunk, stage_bytes, tid)
            cm.team_sync(team)
            if local_warp == 0:
                cm.fence_shared()
                cm.arrive(cm.bptr(smem_base, cm.V_FULL, team))
