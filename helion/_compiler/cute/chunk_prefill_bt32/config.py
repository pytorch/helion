"""CUTLASS-free physical configuration for the centered BT32 schedule."""

from __future__ import annotations

from ..warp_specialized_plan import MBarrierRegion
from ..warp_specialized_plan import SharedBufferRegion
from ..warp_specialized_plan import WarpRole
from ..warp_specialized_plan import WarpSpecializedPipelinePlan

BT = 32
DK = 128
DV = 128
STAGES = 5
THREADS = 1024
STAGE_BYTES = 41_984
SMEM_BYTES = 227_968
TMEM_COLS = 256
TMEM_ADDR_OFFSET = 616
FACTOR_STAGES_OFFSET = 1024
OUTPUT_STAGES_OFFSET = 210_944
GATE_BIAS_OFFSET = 227_408

PIPELINE_PLAN = WarpSpecializedPipelinePlan(
    threads=THREADS,
    shared_bytes=SMEM_BYTES,
    max_shared_bytes_per_block=232_448,
    shared_bytes_per_mp=233_472,
    tmem_columns=TMEM_COLS,
    min_blocks_per_mp=1,
    max_threads_per_mp=2048,
    max_blocks_per_mp=32,
    roles=(
        WarpRole("state", 0, 4, 144),
        WarpRole("output", 4, 4, 48),
        WarpRole("service", 8, 4, 32),
        WarpRole("factor", 12, 20, 56),
    ),
    barriers=(
        MBarrierRegion("single_producer", 0, 9 * STAGES, 1),
        MBarrierRegion("factor_team", 360, 4 * STAGES, 4),
        MBarrierRegion("output_empty", 520, 1, 1),
        MBarrierRegion("state_done", 528, 1, 8),
    ),
    shared_buffers=(
        SharedBufferRegion("tmem_address", TMEM_ADDR_OFFSET, 4, 0, 1, alignment=4),
        SharedBufferRegion(
            "factor_stages", FACTOR_STAGES_OFFSET, STAGES * STAGE_BYTES, 0, 4
        ),
        SharedBufferRegion(
            "output_stages", OUTPUT_STAGES_OFFSET, 2 * BT * DV * 2, 2, 5
        ),
        SharedBufferRegion("gate_bias", GATE_BIAS_OFFSET, DK * 4, 0, 5),
    ),
)
PIPELINE_PLAN.validate()

SINGLE_PRODUCER_BARRIERS = PIPELINE_PLAN.barrier("single_producer")
FACTOR_TEAM_BARRIERS = PIPELINE_PLAN.barrier("factor_team")
SINGLE_PRODUCER_BARRIER_OFFSET = SINGLE_PRODUCER_BARRIERS.byte_offset
SINGLE_PRODUCER_BARRIER_STAGES = SINGLE_PRODUCER_BARRIERS.stages
SINGLE_PRODUCER_BARRIER_ARRIVALS = SINGLE_PRODUCER_BARRIERS.arrivals
FACTOR_TEAM_BARRIER_OFFSET = FACTOR_TEAM_BARRIERS.byte_offset
FACTOR_TEAM_BARRIER_STAGES = FACTOR_TEAM_BARRIERS.stages
FACTOR_TEAM_BARRIER_ARRIVALS = FACTOR_TEAM_BARRIERS.arrivals

QK_FULL = SINGLE_PRODUCER_BARRIER_OFFSET
GATE_RAW_FULL = QK_FULL + 1 * STAGES * 8
QK_RAW_FULL = QK_FULL + 2 * STAGES * 8
V_FULL = QK_FULL + 3 * STAGES * 8
SMEM_FREE = QK_FULL + 4 * STAGES * 8
RAW_INPUTS_FREE = QK_FULL + 5 * STAGES * 8
OLD_OUT_READY = QK_FULL + 6 * STAGES * 8
U2_ACC_READY = QK_FULL + 7 * STAGES * 8
FINAL_READY = QK_FULL + 8 * STAGES * 8
V_FREE = FACTOR_TEAM_BARRIER_OFFSET
STATE_INP_READY = V_FREE + 1 * STAGES * 8
U_INP_READY = V_FREE + 2 * STAGES * 8
U2_INP_READY = V_FREE + 3 * STAGES * 8
OUT_EMPTY = PIPELINE_PLAN.barrier("output_empty").byte_offset
DONE = PIPELINE_PLAN.barrier("state_done").byte_offset

TMEM_ADDR = PIPELINE_PLAN.shared_buffer("tmem_address").byte_offset
QD = PIPELINE_PLAN.shared_buffer("factor_stages").byte_offset
OUT = PIPELINE_PLAN.shared_buffer("output_stages").byte_offset
GATE_BIAS = PIPELINE_PLAN.shared_buffer("gate_bias").byte_offset

STATE_LAST_WARP = PIPELINE_PLAN.role("state").last_warp
STATE_REGISTERS = PIPELINE_PLAN.role("state").registers_per_thread
OUTPUT_LAST_WARP = PIPELINE_PLAN.role("output").last_warp
OUTPUT_REGISTERS = PIPELINE_PLAN.role("output").registers_per_thread
SERVICE_LAST_WARP = PIPELINE_PLAN.role("service").last_warp
SERVICE_REGISTERS = PIPELINE_PLAN.role("service").registers_per_thread
FACTOR_FIRST_WARP = PIPELINE_PLAN.role("factor").first_warp
FACTOR_REGISTERS = PIPELINE_PLAN.role("factor").registers_per_thread
MIN_BLOCKS_PER_MP = PIPELINE_PLAN.min_blocks_per_mp

__all__ = [
    "BT",
    "DK",
    "DONE",
    "DV",
    "FACTOR_FIRST_WARP",
    "FACTOR_REGISTERS",
    "FACTOR_TEAM_BARRIERS",
    "FACTOR_TEAM_BARRIER_ARRIVALS",
    "FACTOR_TEAM_BARRIER_OFFSET",
    "FACTOR_TEAM_BARRIER_STAGES",
    "FINAL_READY",
    "GATE_BIAS",
    "GATE_RAW_FULL",
    "MIN_BLOCKS_PER_MP",
    "OLD_OUT_READY",
    "OUT",
    "OUTPUT_LAST_WARP",
    "OUTPUT_REGISTERS",
    "OUT_EMPTY",
    "PIPELINE_PLAN",
    "QD",
    "QK_FULL",
    "QK_RAW_FULL",
    "RAW_INPUTS_FREE",
    "SERVICE_LAST_WARP",
    "SERVICE_REGISTERS",
    "SINGLE_PRODUCER_BARRIERS",
    "SINGLE_PRODUCER_BARRIER_ARRIVALS",
    "SINGLE_PRODUCER_BARRIER_OFFSET",
    "SINGLE_PRODUCER_BARRIER_STAGES",
    "SMEM_BYTES",
    "SMEM_FREE",
    "STAGES",
    "STAGE_BYTES",
    "STATE_INP_READY",
    "STATE_LAST_WARP",
    "STATE_REGISTERS",
    "THREADS",
    "TMEM_ADDR",
    "TMEM_COLS",
    "U2_ACC_READY",
    "U2_INP_READY",
    "U_INP_READY",
    "V_FREE",
    "V_FULL",
]
