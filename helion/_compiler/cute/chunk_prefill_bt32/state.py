# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# pyrefly: ignore-errors
# ruff: noqa: ANN001, ANN202
# SPDX-License-Identifier: BSD-3-Clause
"""FP32-authoritative BT32 state worker; CAKE388 arithmetic and packing."""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import cutlass.experimental.primitives as prims

from . import common as cm


@cute.jit
def tptr(tmem_base, warp, column, dtype: cutlass.Constexpr):
    return cutlass.inttoptr(tmem_base + ((warp % 4) * 32 << 16) + column, 6, dtype)


@cute.jit
def initialize(tmem_base, initial_state, sequence, head, warp, lane):
    """Unlike CAKE388, preserve every external FP32 initial-state bit."""
    row = (warp % 4) * 32 + lane
    for panel in cutlass.range_constexpr(4):
        values = cutlass.Array(cutlass.Float32, 32, alignment=16)
        for column in cutlass.range_constexpr(32):
            value = cutlass.Float32(0.0)
            if cutlass.const_expr(initial_state is not None):
                value = initial_state[sequence, head, row, panel * 32 + column]
            values[column] = value
        prims.tcgen05_st(
            "32x32b",
            tptr(tmem_base, warp, cm.TMEM_STATE + panel * 32, cutlass.Float32),
            values[0:32],
        )
    prims.tcgen05_wait(kind=prims.Tcgen05Wait.STORE)


@cute.jit
def pack_scale_panel(
    smem_base, tmem_base, stage, warp, panel, PUBLISH: cutlass.Constexpr
):
    pointer = tptr(tmem_base, warp, cm.TMEM_STATE + panel * 32, cutlass.Float32)
    values = prims.tcgen05_ld("32x32b", pointer, num=32)
    prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)
    packed = cutlass.Array(cutlass.Int32, 16, alignment=16)
    for pair in cutlass.range_constexpr(16):
        packed[pair] = cm.pack_bf16(values[pair * 2], values[pair * 2 + 1])
    prims.tcgen05_st(
        "32x32b", tptr(tmem_base, warp, panel * 16, cutlass.Int32), packed[0:16]
    )
    if cutlass.const_expr(PUBLISH):
        # The issuer reads only the packed copy. The FP32 tail decay completes
        # before this worker can publish RHS/U and permit the final N160 update.
        prims.tcgen05_wait(kind=prims.Tcgen05Wait.STORE)
        prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
        cm.arrive(cm.bptr(smem_base, cm.STATE_INP_READY, stage))
    scaled = cutlass.Array(cutlass.Float32, 32, alignment=16)
    for half in cutlass.range_constexpr(2):
        gamma = cutlass.Array(cutlass.Float32, 16, alignment=16)
        for column in cutlass.range_constexpr(16):
            gamma[column] = cm.sptr(
                smem_base,
                cm.GAMMA
                + stage * cm.STAGE_BYTES
                + (panel * 32 + half * 16 + column) * 4,
                cutlass.Float32,
            ).load()
        for pair in cutlass.range_constexpr(8):
            index = half * 16 + pair * 2
            lo, hi = cm.fmul2(
                (values[index], values[index + 1]),
                (gamma[pair * 2], gamma[pair * 2 + 1]),
            )
            scaled[index] = lo
            scaled[index + 1] = hi
    prims.tcgen05_st("32x32b", pointer, scaled[0:32])


@cute.jit
def stage_rhs(
    smem_base,
    tmem_base,
    stage,
    warp,
    lane,
    center_scale,
    remaining,
    MASK_TAIL: cutlass.Constexpr,
):
    row = (warp % 4) * 32 + lane
    prediction = prims.tcgen05_ld(
        "32x32b", tptr(tmem_base, warp, cm.TMEM_PROJECTION, cutlass.Float32), num=32
    )
    prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)
    for half in cutlass.range_constexpr(2):
        values = cutlass.Array(cutlass.Float32, 16, alignment=16)
        beta = cutlass.Array(cutlass.Float32, 16, alignment=16)
        for column in cutlass.range_constexpr(16):
            token = half * 16 + column
            values[column] = cutlass.Float32(
                cm.sptr(
                    smem_base,
                    cm.V + stage * cm.STAGE_BYTES + (token * 128 + row) * 2,
                    cutlass.BFloat16,
                ).load()
            )
            beta[column] = cm.sptr(
                smem_base,
                cm.PREP_BETA + stage * cm.STAGE_BYTES + token * 4,
                cutlass.Float32,
            ).load()
        packed = cutlass.Array(cutlass.Int32, 8, alignment=16)
        for pair in cutlass.range_constexpr(8):
            token = half * 16 + pair * 2
            low, high = cm.ffma2(
                (prediction[token], prediction[token + 1]),
                (-center_scale, -center_scale),
                (values[pair * 2], values[pair * 2 + 1]),
            )
            low, high = cm.fmul2((low, high), (beta[pair * 2], beta[pair * 2 + 1]))
            # Padded rows are explicit zeros in the carrier, including when
            # zero operands meet a nonfinite state outside the finite domain.
            if cutlass.const_expr(MASK_TAIL):
                if token >= remaining:
                    low = cutlass.Float32(0.0)
                if token + 1 >= remaining:
                    high = cutlass.Float32(0.0)
            packed[pair] = cm.pack_bf16(low, high)
        prims.tcgen05_st(
            "32x32b",
            tptr(tmem_base, warp, cm.TMEM_RHS_U + half * 8, cutlass.Int32),
            packed[0:8],
        )
    prims.tcgen05_wait(kind=prims.Tcgen05Wait.STORE)
    prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
    cm.arrive(cm.bptr(smem_base, cm.V_FREE, stage))
    cm.arrive(cm.bptr(smem_base, cm.U_INP_READY, stage))


@cute.jit
def stage_update(smem_base, tmem_base, stage, warp):
    values = prims.tcgen05_ld(
        "32x32b", tptr(tmem_base, warp, cm.TMEM_UPDATE, cutlass.Float32), num=32
    )
    prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)
    packed = cutlass.Array(cutlass.Int32, 16, alignment=16)
    for pair in cutlass.range_constexpr(16):
        packed[pair] = cm.pack_bf16(values[pair * 2], values[pair * 2 + 1])
    prims.tcgen05_st(
        "32x32b", tptr(tmem_base, warp, cm.TMEM_RHS_U, cutlass.Int32), packed[0:16]
    )
    prims.tcgen05_wait(kind=prims.Tcgen05Wait.STORE)
    prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
    cm.arrive(cm.bptr(smem_base, cm.U2_INP_READY, stage))


@cute.jit
def state_loop(
    smem_base,
    tmem_base,
    initial_state,
    final_state,
    sequence,
    head,
    seqlen,
    num_chunks,
    warp,
    lane,
    gate_scale_log2,
):
    initialize(tmem_base, initial_state, sequence, head, warp, lane)
    center_scale = cute.math.exp2(
        gate_scale_log2 * cutlass.Float32(16.0), fastmath=True
    )
    stage = cutlass.Int32(0)
    phase = cutlass.Int32(0)
    for chunk in cutlass.range(num_chunks, unroll=1):
        cm.wait(cm.bptr(smem_base, cm.QK_FULL, stage), phase)
        for panel in cutlass.range(3, unroll=1):
            pack_scale_panel(smem_base, tmem_base, stage, warp, panel, False)
        pack_scale_panel(smem_base, tmem_base, stage, warp, cutlass.Int32(3), True)
        cm.wait(cm.bptr(smem_base, cm.V_FULL, stage), phase)
        cm.wait(cm.bptr(smem_base, cm.OLD_OUT_READY, stage), phase)
        prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
        remaining = seqlen - chunk * cm.BT
        if remaining >= cm.BT:
            stage_rhs(
                smem_base,
                tmem_base,
                stage,
                warp,
                lane,
                center_scale,
                remaining,
                False,
            )
        else:
            stage_rhs(
                smem_base,
                tmem_base,
                stage,
                warp,
                lane,
                center_scale,
                remaining,
                True,
            )
        cm.wait(cm.bptr(smem_base, cm.U2_ACC_READY, stage), phase)
        prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
        stage_update(smem_base, tmem_base, stage, warp)
        cm.wait(cm.bptr(smem_base, cm.FINAL_READY, stage), phase)
        prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
        stage = stage + 1
        if stage == cm.STAGES:
            stage = cutlass.Int32(0)
            phase = phase ^ 1
    if cutlass.const_expr(final_state is not None):
        row = (warp % 4) * 32 + lane
        for panel in cutlass.range_constexpr(4):
            values = prims.tcgen05_ld(
                "32x32b",
                tptr(tmem_base, warp, cm.TMEM_STATE + panel * 32, cutlass.Float32),
                num=32,
            )
            prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)
            for column in cutlass.range_constexpr(32):
                final_state[sequence, head, row, panel * 32 + column] = values[column]
    prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
    cm.arrive(cm.sptr(smem_base, cm.DONE, cutlass.Int64))
