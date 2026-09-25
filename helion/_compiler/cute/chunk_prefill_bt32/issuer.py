# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# pyrefly: ignore-errors
# ruff: noqa: ANN001, ANN202
# SPDX-License-Identifier: BSD-3-Clause
"""Single-warp M128 TCGEN05 issue order from FlashInfer v0.7.0 CAKE388."""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import cutlass.experimental.primitives as prims

from . import common as cm


@cute.jit
def projection(smem_base, offset, tmem_base, destination):
    """Eight K16 slices: old BF16 state times a centered/restored operand."""
    descriptor = prims.Tcgen05SmemDesc.build(
        cm.sptr(smem_base, offset, cutlass.BFloat16),
        leading_byte_offset=16,
        stride_byte_offset=1024,
        layout=prims.Tcgen05SmemSwizzle.SWIZZLE_128B,
    )
    instruction = prims.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=cutlass.BFloat16,
        b_dtype=cutlass.BFloat16,
        n_dim=32,
        m_dim=128,
        b_major=0,
    )
    if prims.elect_sync():
        for part in cutlass.range_constexpr(8):
            address = (part % 4) * 32 + (part // 4) * 4096
            prims.tcgen05_mma(
                prims.Tcgen05MMAKind.F16,
                prims.CTAGroup.CTA_1,
                cutlass.inttoptr(tmem_base + destination, 6, cutlass.Float32),
                prims.make_tmem_ptr(tmem_base, cutlass.Int8).subview(part * 8),
                descriptor.advance_start_address(address),
                instruction,
                part != 0,
            )


@cute.jit
def inverse_product(smem_base, offset, tmem_base):
    """BF16 RHS times the transposed SW32 inverse: two ordered K16 slices."""
    descriptor = prims.Tcgen05SmemDesc.build(
        cm.sptr(smem_base, offset, cutlass.BFloat16),
        leading_byte_offset=16,
        stride_byte_offset=256,
        layout=prims.Tcgen05SmemSwizzle.SWIZZLE_32B,
    )
    instruction = prims.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=cutlass.BFloat16,
        b_dtype=cutlass.BFloat16,
        n_dim=32,
        m_dim=128,
        b_major=0,
    )
    if prims.elect_sync():
        for part in cutlass.range_constexpr(2):
            prims.tcgen05_mma(
                prims.Tcgen05MMAKind.F16,
                prims.CTAGroup.CTA_1,
                cutlass.inttoptr(tmem_base + cm.TMEM_UPDATE, 6, cutlass.Float32),
                prims.make_tmem_ptr(tmem_base, cutlass.Int8).subview(
                    cm.TMEM_RHS_U + part * 8
                ),
                descriptor.advance_start_address(part * 1024),
                instruction,
                part != 0,
            )


@cute.jit
def final_product(smem_base, offset, tmem_base):
    """One N160 destination: FP32 state128 followed by FP32 output32."""
    descriptor = prims.Tcgen05SmemDesc.build(
        cm.sptr(smem_base, offset, cutlass.BFloat16),
        leading_byte_offset=4096,
        stride_byte_offset=1024,
        layout=prims.Tcgen05SmemSwizzle.SWIZZLE_128B,
    )
    instruction = prims.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=cutlass.BFloat16,
        b_dtype=cutlass.BFloat16,
        n_dim=160,
        m_dim=128,
        b_major=1,
    )
    if prims.elect_sync():
        for part in cutlass.range_constexpr(2):
            prims.tcgen05_mma(
                prims.Tcgen05MMAKind.F16,
                prims.CTAGroup.CTA_1,
                cutlass.inttoptr(tmem_base + cm.TMEM_STATE, 6, cutlass.Float32),
                prims.make_tmem_ptr(tmem_base, cutlass.Int8).subview(
                    cm.TMEM_RHS_U + part * 8
                ),
                descriptor.advance_start_address(part * 2048),
                instruction,
                True,
            )


@cute.jit
def commit(pointer):
    if prims.elect_sync():
        prims.tcgen05_commit(pointer, group=prims.CTAGroup.CTA_1)


@cute.jit
def issuer_loop(smem_base, tmem_base, num_chunks):
    stage = cutlass.Int32(0)
    phase = cutlass.Int32(0)
    output_phase = cutlass.Int32(1)
    for _chunk in cutlass.range(num_chunks, unroll=1):
        stage_bytes = stage * cm.STAGE_BYTES
        cm.wait(cm.bptr(smem_base, cm.QK_FULL, stage), phase)
        cm.wait(cm.sptr(smem_base, cm.OUT_EMPTY, cutlass.Int64), output_phase)
        output_phase = output_phase ^ 1
        cm.wait(cm.bptr(smem_base, cm.STATE_INP_READY, stage), phase)
        prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
        projection(smem_base, cm.KD + stage_bytes, tmem_base, cm.TMEM_PROJECTION)
        commit(cm.bptr(smem_base, cm.OLD_OUT_READY, stage))
        projection(smem_base, cm.QD + stage_bytes, tmem_base, cm.TMEM_OUT)
        commit(cm.bptr(smem_base, cm.RAW_INPUTS_FREE, stage))
        cm.wait(cm.bptr(smem_base, cm.U_INP_READY, stage), phase)
        prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
        inverse_product(smem_base, cm.INV + stage_bytes, tmem_base)
        commit(cm.bptr(smem_base, cm.U2_ACC_READY, stage))
        cm.wait(cm.bptr(smem_base, cm.U2_INP_READY, stage), phase)
        prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
        final_product(smem_base, cm.FINAL_TRANS + stage_bytes, tmem_base)
        commit(cm.bptr(smem_base, cm.FINAL_READY, stage))
        commit(cm.bptr(smem_base, cm.SMEM_FREE, stage))
        stage = stage + 1
        if stage == cm.STAGES:
            stage = cutlass.Int32(0)
            phase = phase ^ 1
    # All eight consuming warps have completed TMEM reads and external stores.
    cm.wait(cm.sptr(smem_base, cm.DONE, cutlass.Int64), cutlass.Int32(0))
    prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
    prims.tcgen05_dealloc(
        prims.make_tmem_ptr(tmem_base, cutlass.Int8),
        cm.TMEM_COLS,
        group=prims.CTAGroup.CTA_1,
    )
