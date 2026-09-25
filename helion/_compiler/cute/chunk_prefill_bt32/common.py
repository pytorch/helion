# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# pyrefly: ignore-errors
# ruff: noqa: ANN001, ANN202
# SPDX-License-Identifier: BSD-3-Clause
"""BT32 layout and primitives adapted from FlashInfer v0.7.0 CAKE.

Layout source: csrc/kda/flashkda_generated_bf16_fused_m128_388c6ad8eb.cu,
FlashInfer commit4d75a33f19aa. No FlashInfer runtime or compiled binary is used.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import cutlass.experimental.primitives as prims

BT = 32
DK = 128
DV = 128
STAGES = 5
THREADS = 1024
STAGE_BYTES = 41984
SMEM_BYTES = 227968
TMEM_COLS = 256
TMEM_STATE = 64
TMEM_PACKED_STATE = 0
TMEM_PROJECTION = 224
TMEM_RHS_U = 224
TMEM_UPDATE = 0
TMEM_OUT = 192
QD = 1024
GATE_RAW = 1024
KD = 9216
Q_RAW_PREFETCH = 17408
KI = 17408
FINAL_TRANS = 17408
KR = 17408
QK_SLAB = 25600
GATE_PREFIX = 25600
INV = 29696
GAMMA = 31744
V = 32384
INV_WORK = 32384
PREFIX_LAST = 41472
BETA_RAW = 41984
RESTORE_FACTOR = 41984
PREP_BETA = 42500
GATE_RATE = 42628
GATE_BIAS = 227408
OUT = 210944
QK_FULL = 0
GATE_RAW_FULL = 40
QK_RAW_FULL = 80
V_FULL = 120
SMEM_FREE = 160
RAW_INPUTS_FREE = 200
OLD_OUT_READY = 240
U2_ACC_READY = 280
FINAL_READY = 320
V_FREE = 360
STATE_INP_READY = 400
U_INP_READY = 440
U2_INP_READY = 480
OUT_EMPTY = 520
DONE = 528
TMEM_ADDR = 616
LOG2_E = 1.4426950408889634


@cute.jit
def sptr(base, offset, dtype: cutlass.Constexpr):
    return cutlass.inttoptr(base + offset, 3, dtype)


@cute.jit
def bptr(base, offset, stage):
    return sptr(base, offset + stage * 8, cutlass.Int64)


@cute.jit
def wait(pointer, phase):
    while not prims.mbarrier_wait_parity(pointer, phase, prims.MBarrierWait.TRY):
        pass


@cute.jit
def arrive(pointer):
    prims.bar_warp_sync(cute.arch.FULL_MASK)
    if prims.elect_sync():
        prims.mbarrier_arrive(pointer)


@cute.jit
def expect_tx(pointer, count):
    if prims.elect_sync():
        prims.mbarrier_arrive_expect_tx(pointer, count)


@cute.jit
def fence_shared():
    cute.arch.fence_view_async_shared()


@cute.jit
def team_sync(team):
    prims.barrier_cta_sync(10 + team, thread_count=128)


@cute.jit
def output_sync():
    prims.barrier_cta_sync(8, thread_count=128)


@cute.jit
def sw128(row, column):
    byte = (column // 64) * BT * 128 + row * 128 + (column % 64) * 2
    return byte ^ (((byte >> 7) & 7) << 4)


@cute.jit
def sw32(row, column):
    byte = (column // 16) * BT * 32 + row * 32 + (column % 16) * 2
    return byte ^ (((byte >> 7) & 1) << 4)


@cute.jit
def pack_bf16(lo, hi):
    return prims.inline_ptx_hl(
        "cvt.rn.bf16x2.f32 {$w0}, {$r1}, {$r0};",
        write_only_types=[cutlass.Int32],
        read_only_args=[cutlass.Float32(lo), cutlass.Float32(hi)],
    )


def ffma2(lhs, rhs, acc):
    a = cutlass.Vector.from_elements(lhs, cutlass.Float32)
    b = cutlass.Vector.from_elements(rhs, cutlass.Float32)
    c = cutlass.Vector.from_elements(acc, cutlass.Float32)
    result = prims.fma_packed_f32x2(a, b, c, ftz=False, rnd="rn")
    return cutlass.Float32(result[0]), cutlass.Float32(result[1])


def fmul2(lhs, rhs):
    a = cutlass.Vector.from_elements(lhs, cutlass.Float32)
    b = cutlass.Vector.from_elements(rhs, cutlass.Float32)
    result = prims.mul_packed_f32x2(a, b, ftz=False, rnd="rn")
    return cutlass.Float32(result[0]), cutlass.Float32(result[1])


def fsub2(lhs, rhs):
    a = cutlass.Vector.from_elements(lhs, cutlass.Float32)
    b = cutlass.Vector.from_elements(rhs, cutlass.Float32)
    result = prims.sub_packed_f32x2(a, b, ftz=False, rnd="rn")
    return cutlass.Float32(result[0]), cutlass.Float32(result[1])
