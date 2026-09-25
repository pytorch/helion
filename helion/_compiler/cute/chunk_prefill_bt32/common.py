# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# pyrefly: ignore-errors
# ruff: noqa: ANN001, ANN202
# SPDX-License-Identifier: BSD-3-Clause
"""BT32 layout and primitives adapted from FlashInfer v0.7.0 CAKE.

Layout source: csrc/kda/flashkda_generated_bf16_fused_m128_388c6ad8eb.cu,
FlashInfer commit4d75a33f19aa. No FlashInfer runtime or compiled binary is used.
"""

from __future__ import annotations

import cutlass.cute as cute

from ..affine_recurrence_primitives import ffma2 as ffma2
from ..affine_recurrence_primitives import fmul2 as fmul2
from ..affine_recurrence_primitives import fsub2 as fsub2
from ..affine_recurrence_primitives import pack_bf16x2_inline
from ..warp_specialized_primitives import arrive_mbarrier
from ..warp_specialized_primitives import expect_mbarrier_tx
from ..warp_specialized_primitives import fence_async_shared
from ..warp_specialized_primitives import named_barrier_sync
from ..warp_specialized_primitives import shared_pointer
from ..warp_specialized_primitives import staged_mbarrier_pointer
from ..warp_specialized_primitives import swizzle_b16_index
from ..warp_specialized_primitives import wait_mbarrier
from . import config

sptr = shared_pointer
bptr = staged_mbarrier_pointer
wait = wait_mbarrier
arrive = arrive_mbarrier
expect_tx = expect_mbarrier_tx
fence_shared = fence_async_shared
pack_bf16 = pack_bf16x2_inline

BT = config.BT
DK = config.DK
DV = config.DV
STAGES = config.STAGES
THREADS = config.THREADS
STAGE_BYTES = config.STAGE_BYTES
SMEM_BYTES = config.SMEM_BYTES
TMEM_COLS = config.TMEM_COLS
TMEM_STATE = 64
TMEM_PACKED_STATE = 0
TMEM_PROJECTION = 224
TMEM_RHS_U = 224
TMEM_UPDATE = 0
TMEM_OUT = 192
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
LOG2_E = 1.4426950408889634

PIPELINE_PLAN = config.PIPELINE_PLAN
SINGLE_PRODUCER_BARRIERS = config.SINGLE_PRODUCER_BARRIERS
FACTOR_TEAM_BARRIERS = config.FACTOR_TEAM_BARRIERS
SINGLE_PRODUCER_BARRIER_OFFSET = config.SINGLE_PRODUCER_BARRIER_OFFSET
SINGLE_PRODUCER_BARRIER_STAGES = config.SINGLE_PRODUCER_BARRIER_STAGES
SINGLE_PRODUCER_BARRIER_ARRIVALS = config.SINGLE_PRODUCER_BARRIER_ARRIVALS
FACTOR_TEAM_BARRIER_OFFSET = config.FACTOR_TEAM_BARRIER_OFFSET
FACTOR_TEAM_BARRIER_STAGES = config.FACTOR_TEAM_BARRIER_STAGES
FACTOR_TEAM_BARRIER_ARRIVALS = config.FACTOR_TEAM_BARRIER_ARRIVALS
QK_FULL = config.QK_FULL
GATE_RAW_FULL = config.GATE_RAW_FULL
QK_RAW_FULL = config.QK_RAW_FULL
V_FULL = config.V_FULL
SMEM_FREE = config.SMEM_FREE
RAW_INPUTS_FREE = config.RAW_INPUTS_FREE
OLD_OUT_READY = config.OLD_OUT_READY
U2_ACC_READY = config.U2_ACC_READY
FINAL_READY = config.FINAL_READY
V_FREE = config.V_FREE
STATE_INP_READY = config.STATE_INP_READY
U_INP_READY = config.U_INP_READY
U2_INP_READY = config.U2_INP_READY
OUT_EMPTY = config.OUT_EMPTY
DONE = config.DONE
TMEM_ADDR = config.TMEM_ADDR
QD = config.QD
OUT = config.OUT
GATE_BIAS = config.GATE_BIAS
STATE_LAST_WARP = config.STATE_LAST_WARP
STATE_REGISTERS = config.STATE_REGISTERS
OUTPUT_LAST_WARP = config.OUTPUT_LAST_WARP
OUTPUT_REGISTERS = config.OUTPUT_REGISTERS
SERVICE_LAST_WARP = config.SERVICE_LAST_WARP
SERVICE_REGISTERS = config.SERVICE_REGISTERS
FACTOR_FIRST_WARP = config.FACTOR_FIRST_WARP
FACTOR_REGISTERS = config.FACTOR_REGISTERS
MIN_BLOCKS_PER_MP = config.MIN_BLOCKS_PER_MP


@cute.jit
def team_sync(team):
    named_barrier_sync(10 + team, 128)


@cute.jit
def output_sync():
    named_barrier_sync(8, 128)


@cute.jit
def sw128(row, column):
    return swizzle_b16_index(row, column, 128, 64, BT, 7)


@cute.jit
def sw32(row, column):
    return swizzle_b16_index(row, column, 32, 16, BT, 1)
