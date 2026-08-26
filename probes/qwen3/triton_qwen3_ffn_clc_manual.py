# ruff: noqa: ANN001, ANN201, ANN202
"""Qwen3-8B FP8 decode FFN megakernel using Cluster Launch Control.

The authoritative variant uses a monotonic command ticket after every
successful cancellation; canceled CTA IDs are measured only by historical
ablations and are not treated as dependency-safe task identity.  The final
producer for each gate/up group runs the existing SiLU/quant continuation
locally; an explicit-activation variant is retained as an ablation.

The benchmark compares with the three tuned standalone Helion kernels in one
CUDA graph.  Every measured replay is preceded by Triton's 256 MiB L2 flush.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics

from cuda.bindings import driver as cuda_driver
import torch
import triton
import triton.language as tl

from probes.common import capture
from probes.common import require_idle_visible_gpu
from probes.common import visible_gpu_pids
from probes.qwen3.helion_qwen3_layer_baseline import FFN_CONFIGS
from probes.qwen3.helion_qwen3_layer_baseline import FP8_MAX
from probes.qwen3.helion_qwen3_layer_baseline import FP8_MIN
from probes.qwen3.helion_qwen3_layer_baseline import FP8_MIN_SCALE
from probes.qwen3.helion_qwen3_layer_baseline import block_fp8_mm
from probes.qwen3.helion_qwen3_layer_baseline import compile_config
from probes.qwen3.helion_qwen3_layer_baseline import silu_and_mul_per_block_quant

HIDDEN = 4096
INTERMEDIATE = 12288
GROUP = 128
GATE_BLOCK = 16
DOWN_BLOCK = 8
GATE_TASKS = 2 * INTERMEDIATE // GATE_BLOCK
ACTIVATION_TASKS = INTERMEDIATE // GROUP
DOWN_TASKS = HIDDEN // DOWN_BLOCK
TOTAL_TASKS = GATE_TASKS + DOWN_TASKS
EXPLICIT_TOTAL_TASKS = GATE_TASKS + ACTIVATION_TASKS + DOWN_TASKS
FIRST_WAVE_GATE_TASKS = 1184
STATE_WORDS = 102
STARTED_SLOT = 98
CANCELED_SLOT = 99
TASK_CURSOR_SLOT = 100
PROCESSED_SLOT = 101
CLC_SCRATCH_BYTES = 12288

TL_HIDDEN = tl.constexpr(HIDDEN)
TL_INTERMEDIATE = tl.constexpr(INTERMEDIATE)
TL_GROUP = tl.constexpr(GROUP)
TL_GATE_BLOCK = tl.constexpr(GATE_BLOCK)
TL_DOWN_BLOCK = tl.constexpr(DOWN_BLOCK)
TL_GATE_TASKS = tl.constexpr(GATE_TASKS)
TL_ACTIVATION_TASKS = tl.constexpr(ACTIVATION_TASKS)
TL_DOWN_TASKS = tl.constexpr(DOWN_TASKS)
TL_STARTED_SLOT = tl.constexpr(STARTED_SLOT)
TL_CANCELED_SLOT = tl.constexpr(CANCELED_SLOT)
TL_TASK_CURSOR_SLOT = tl.constexpr(TASK_CURSOR_SLOT)
TL_PROCESSED_SLOT = tl.constexpr(PROCESSED_SLOT)
TL_EXPLICIT_DOWN_BASE = tl.constexpr(GATE_TASKS + ACTIVATION_TASKS)
TL_FP8_MAX = tl.constexpr(FP8_MAX)
TL_FP8_MIN = tl.constexpr(FP8_MIN)
TL_FP8_MIN_SCALE = tl.constexpr(FP8_MIN_SCALE)


@triton.jit
def _issue_clc_cancel():
    return tl.inline_asm_elementwise(
        asm=r"""
        {
            .reg .pred leader;
            .reg .b32 response_addr, mbar_addr, thread_id;
            .shared .align 16 .b8 qwen3_clc_scratch[12288];

            mov.u32 response_addr, qwen3_clc_scratch;
            add.u32 mbar_addr, response_addr, 16;
            mov.u32 thread_id, %tid.x;
            setp.eq.u32 leader, thread_id, 0;

            @leader mbarrier.init.shared::cta.b64 [mbar_addr], 1;
            @leader clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.b128 [response_addr], [mbar_addr];
            @leader mbarrier.arrive.expect_tx.relaxed.cta.shared::cta.b64 _, [mbar_addr], 16;

            mov.u32 $0, response_addr;
        }
        """,
        constraints="=r",
        args=[],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _wait_clc_cancel(response_addr):
    return tl.inline_asm_elementwise(
        asm=r"""
        {
            .reg .pred complete, canceled;
            .reg .b32 response_addr, mbar_addr, success, canceled_x;
            .reg .b128 response;

            mov.u32 response_addr, $2;
            add.u32 mbar_addr, response_addr, 16;
            mov.u32 success, 0;
            mov.u32 canceled_x, 0xffffffff;

        QWEN3_CLC_WAIT:
            mbarrier.try_wait.parity.relaxed.cta.shared.b64 complete, [mbar_addr], 0;
            @!complete bra QWEN3_CLC_WAIT;

            ld.shared.b128 response, [response_addr];
            clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 canceled, response;
            selp.u32 success, 1, 0, canceled;
            @canceled clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 canceled_x, response;

            mov.u32 $0, success;
            mov.u32 $1, canceled_x;
        }
        """,
        constraints="=r,=r,r",
        args=[response_addr],
        dtype=(tl.uint32, tl.uint32),
        is_pure=False,
        pack=1,
    )


@triton.jit
def _reissue_clc_cancel(response_addr):
    return tl.inline_asm_elementwise(
        asm=r"""
        {
            .reg .pred leader;
            .reg .b32 response_addr, mbar_addr, thread_id;

            mov.u32 response_addr, $1;
            add.u32 mbar_addr, response_addr, 16;
            mov.u32 thread_id, %tid.x;
            setp.eq.u32 leader, thread_id, 0;

            @leader fence.proxy.async.shared::cta;
            @leader clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.b128 [response_addr], [mbar_addr];
            @leader mbarrier.arrive.expect_tx.relaxed.cta.shared::cta.b64 _, [mbar_addr], 16;
            mov.u32 $0, response_addr;
        }
        """,
        constraints="=r,r",
        args=[response_addr],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _wait_clc_cancel_phase(response_addr, phase):
    return tl.inline_asm_elementwise(
        asm=r"""
        {
            .reg .pred complete, canceled;
            .reg .b32 response_addr, mbar_addr, success, phase;
            .reg .b128 response;

            mov.u32 response_addr, $1;
            add.u32 mbar_addr, response_addr, 16;
            mov.u32 phase, $2;
            mov.u32 success, 0;

        QWEN3_CLC_PHASE_WAIT:
            mbarrier.try_wait.parity.relaxed.cta.shared.b64 complete, [mbar_addr], phase;
            @!complete bra QWEN3_CLC_PHASE_WAIT;

            ld.shared.b128 response, [response_addr];
            clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 canceled, response;
            selp.u32 success, 1, 0, canceled;
            mov.u32 $0, success;
        }
        """,
        constraints="=r,r,r",
        args=[response_addr, phase],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _load_acquire(address):
    return tl.inline_asm_elementwise(
        asm="ld.acquire.gpu.global.u32 $0, [$1];",
        constraints="=r,l",
        args=[address],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _sync_warp():
    lanes = tl.arange(0, 32)
    return tl.inline_asm_elementwise(
        asm="bar.warp.sync 0xffffffff; mov.u32 $0, $1;",
        constraints="=r,r",
        args=[lanes],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _nanosleep(DELAY: tl.constexpr):
    lanes = tl.arange(0, 32)
    delay = tl.full([], DELAY, tl.uint32)
    return tl.inline_asm_elementwise(
        asm="nanosleep.u32 $1; mov.u32 $0, $2;",
        constraints="=r,r,r",
        args=[delay, lanes],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _wait_count(address, target, POLL_DELAY: tl.constexpr):
    value = _load_acquire(address)
    while value < target:
        if POLL_DELAY:
            _nanosleep(POLL_DELAY)
        value = _load_acquire(address)
    _sync_warp()


@triton.jit
def _physical_gate_task(virtual_task):
    group = virtual_task // 16
    within_group = virtual_task % 16
    return tl.where(
        within_group < 8,
        group * 8 + within_group,
        TL_GATE_TASKS // 2 + group * 8 + within_group - 8,
    )


@triton.jit
def _gate_task(
    input_q,
    input_scale,
    gate_weight_q,
    gate_weight_scale,
    gate_up,
    physical_task,
):
    output_index = physical_task * TL_GATE_BLOCK + tl.arange(0, TL_GATE_BLOCK)
    accumulator = tl.zeros((1, TL_GATE_BLOCK), dtype=tl.float32)
    for k_start in tl.range(
        0,
        TL_HIDDEN,
        TL_GROUP,
        loop_unroll_factor=2,
        num_stages=4,
        disallow_acc_multi_buffer=False,
        flatten=False,
    ):
        k = k_start + tl.arange(0, TL_GROUP)
        activation = tl.load(input_q + k[None, :])
        weight = tl.load(gate_weight_q + output_index[:, None] * TL_HIDDEN + k[None, :])
        partial = tl.dot(
            activation,
            tl.trans(weight),
            input_precision="tf32",
            out_dtype=tl.float32,
        )
        group = k_start // TL_GROUP
        activation_group_scale = tl.load(input_scale + group)
        weight_group_scale = tl.load(
            gate_weight_scale
            + (output_index // TL_GROUP) * (TL_HIDDEN // TL_GROUP)
            + group
        )
        accumulator += partial * activation_group_scale * weight_group_scale[None, :]
    tl.store(gate_up + output_index[None, :], accumulator.to(tl.bfloat16))


@triton.jit
def _activation_task(gate_up, activation_q, activation_scale, group):
    index = group * TL_GROUP + tl.arange(0, TL_GROUP)
    gate = tl.load(gate_up + index).to(tl.float32)
    up = tl.load(gate_up + TL_INTERMEDIATE + index).to(tl.float32)
    activated = gate * tl.sigmoid(gate) * up
    scale = tl.maximum(tl.max(tl.abs(activated), axis=0) / TL_FP8_MAX, TL_FP8_MIN_SCALE)
    tl.store(activation_scale + group, scale)
    quantized = tl.minimum(tl.maximum(activated / scale, TL_FP8_MIN), TL_FP8_MAX).to(
        tl.float8e4nv
    )
    tl.store(activation_q + index, quantized)


@triton.jit
def _gate_publish_task(
    input_q,
    input_scale,
    gate_weight_q,
    gate_weight_scale,
    gate_up,
    activation_q,
    activation_scale,
    state,
    virtual_task,
    FRONTIER: tl.constexpr,
):
    group = virtual_task // 16
    _gate_task(
        input_q,
        input_scale,
        gate_weight_q,
        gate_weight_scale,
        gate_up,
        _physical_gate_task(virtual_task),
    )
    _sync_warp()
    previous = tl.atomic_add(state + group, 1, sem="acq_rel", scope="gpu")
    if previous == 15:
        _activation_task(gate_up, activation_q, activation_scale, group)
        _sync_warp()
        milestone = tl.where(group < FRONTIER, 96, 97)
        tl.atomic_add(state + milestone, 1, sem="release", scope="gpu")


@triton.jit
def _gate_only_task(
    input_q,
    input_scale,
    gate_weight_q,
    gate_weight_scale,
    gate_up,
    state,
    virtual_task,
):
    group = virtual_task // 16
    _gate_task(
        input_q,
        input_scale,
        gate_weight_q,
        gate_weight_scale,
        gate_up,
        _physical_gate_task(virtual_task),
    )
    _sync_warp()
    tl.atomic_add(state + group, 1, sem="release", scope="gpu")


@triton.jit
def _down_segment(
    activation_q,
    activation_scale,
    down_weight_q,
    down_weight_scale,
    task,
    accumulator,
    START_GROUP: tl.constexpr,
    END_GROUP: tl.constexpr,
):
    output_index = task * TL_DOWN_BLOCK + tl.arange(0, TL_DOWN_BLOCK)
    for group in tl.range(
        START_GROUP,
        END_GROUP,
        1,
        loop_unroll_factor=4,
        num_stages=4,
        disallow_acc_multi_buffer=True,
        flatten=True,
    ):
        k = group * TL_GROUP + tl.arange(0, TL_GROUP)
        activation = tl.load(activation_q + k[None, :])
        weight = tl.load(
            down_weight_q + output_index[:, None] * TL_INTERMEDIATE + k[None, :]
        )
        partial = tl.dot(
            activation,
            tl.trans(weight),
            input_precision="tf32",
            out_dtype=tl.float32,
        )
        activation_group_scale = tl.load(activation_scale + group)
        weight_group_scale = tl.load(
            down_weight_scale
            + (output_index // TL_GROUP) * (TL_INTERMEDIATE // TL_GROUP)
            + group
        )
        accumulator += partial * activation_group_scale * weight_group_scale[None, :]
    return accumulator


@triton.jit
def _prefetch_l2(address):
    return tl.inline_asm_elementwise(
        asm="prefetch.global.L2 [$1]; mov.u32 $0, 0;",
        constraints="=r,l",
        args=[address],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _prefetch_l1(address):
    return tl.inline_asm_elementwise(
        asm="prefetch.global.L1 [$1]; mov.u32 $0, 0;",
        constraints="=r,l",
        args=[address],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _prefetch_down_groups(
    down_weight_q,
    task,
    FRONTIER: tl.constexpr,
    PREFETCH_DEPTH: tl.constexpr,
):
    output_index = task * TL_DOWN_BLOCK + tl.arange(0, TL_DOWN_BLOCK)
    for offset in tl.static_range(0, PREFETCH_DEPTH):
        group = FRONTIER + offset
        address = down_weight_q + output_index * TL_INTERMEDIATE + group * TL_GROUP
        _prefetch_l2(address)


@triton.jit
def _prefetch_down_groups_l1(
    down_weight_q,
    task,
    FRONTIER: tl.constexpr,
    PREFETCH_L1_DEPTH: tl.constexpr,
):
    output_index = task * TL_DOWN_BLOCK + tl.arange(0, TL_DOWN_BLOCK)
    for offset in tl.static_range(0, PREFETCH_L1_DEPTH):
        group = FRONTIER + offset
        address = down_weight_q + output_index * TL_INTERMEDIATE + group * TL_GROUP
        _prefetch_l1(address)


@triton.jit
def _store_down(output, task, accumulator):
    output_index = task * TL_DOWN_BLOCK + tl.arange(0, TL_DOWN_BLOCK)
    tl.store(output + output_index[None, :], accumulator.to(tl.bfloat16))


@triton.jit
def _run_down_task(
    activation_q,
    activation_scale,
    down_weight_q,
    down_weight_scale,
    output,
    state,
    task,
    FRONTIER: tl.constexpr,
    PREFETCH_DEPTH: tl.constexpr,
    PREFETCH_L1_DEPTH: tl.constexpr,
    FIRST_POLL_DELAY: tl.constexpr,
    SECOND_POLL_DELAY: tl.constexpr,
):
    accumulator = tl.zeros((1, TL_DOWN_BLOCK), dtype=tl.float32)
    _wait_count(state + 96, FRONTIER, FIRST_POLL_DELAY)
    accumulator = _down_segment(
        activation_q,
        activation_scale,
        down_weight_q,
        down_weight_scale,
        task,
        accumulator,
        0,
        FRONTIER,
    )
    if FRONTIER < TL_ACTIVATION_TASKS:
        if PREFETCH_DEPTH:
            _prefetch_down_groups(
                down_weight_q,
                task,
                FRONTIER,
                PREFETCH_DEPTH,
            )
        if PREFETCH_L1_DEPTH:
            _prefetch_down_groups_l1(
                down_weight_q,
                task,
                FRONTIER,
                PREFETCH_L1_DEPTH,
            )
        _wait_count(
            state + 97,
            TL_ACTIVATION_TASKS - FRONTIER,
            SECOND_POLL_DELAY,
        )
        accumulator = _down_segment(
            activation_q,
            activation_scale,
            down_weight_q,
            down_weight_scale,
            task,
            accumulator,
            FRONTIER,
            TL_ACTIVATION_TASKS,
        )
    _store_down(output, task, accumulator)


@triton.jit(noinline=True)
def _run_logical_task(
    input_q,
    input_scale,
    gate_weight_q,
    gate_weight_scale,
    down_weight_q,
    down_weight_scale,
    gate_up,
    activation_q,
    activation_scale,
    output,
    state,
    logical_task,
    FRONTIER: tl.constexpr,
    PREFETCH_DEPTH: tl.constexpr,
    PREFETCH_L1_DEPTH: tl.constexpr,
    FIRST_POLL_DELAY: tl.constexpr,
    SECOND_POLL_DELAY: tl.constexpr,
):
    if logical_task < TL_GATE_TASKS:
        _gate_publish_task(
            input_q,
            input_scale,
            gate_weight_q,
            gate_weight_scale,
            gate_up,
            activation_q,
            activation_scale,
            state,
            logical_task,
            FRONTIER,
        )
    else:
        _run_down_task(
            activation_q,
            activation_scale,
            down_weight_q,
            down_weight_scale,
            output,
            state,
            logical_task - TL_GATE_TASKS,
            FRONTIER,
            PREFETCH_DEPTH,
            PREFETCH_L1_DEPTH,
            FIRST_POLL_DELAY,
            SECOND_POLL_DELAY,
        )


@triton.jit
def qwen3_ffn_direct_kernel(
    input_q,
    input_scale,
    gate_weight_q,
    gate_weight_scale,
    down_weight_q,
    down_weight_scale,
    gate_up,
    activation_q,
    activation_scale,
    output,
    state,
    NUM_WORKERS: tl.constexpr,
    FRONTIER: tl.constexpr,
    PREFETCH_DEPTH: tl.constexpr,
    PREFETCH_L1_DEPTH: tl.constexpr,
    FIRST_POLL_DELAY: tl.constexpr,
    SECOND_POLL_DELAY: tl.constexpr,
):
    worker = tl.program_id(0)
    _gate_publish_task(
        input_q,
        input_scale,
        gate_weight_q,
        gate_weight_scale,
        gate_up,
        activation_q,
        activation_scale,
        state,
        worker,
        FRONTIER,
    )
    second_gate = worker + NUM_WORKERS
    if second_gate < TL_GATE_TASKS:
        _gate_publish_task(
            input_q,
            input_scale,
            gate_weight_q,
            gate_weight_scale,
            gate_up,
            activation_q,
            activation_scale,
            state,
            second_gate,
            FRONTIER,
        )
    down_worker_base: tl.constexpr = TL_GATE_TASKS - NUM_WORKERS
    if worker >= down_worker_base and worker < down_worker_base + TL_DOWN_TASKS:
        _run_down_task(
            activation_q,
            activation_scale,
            down_weight_q,
            down_weight_scale,
            output,
            state,
            worker - down_worker_base,
            FRONTIER,
            PREFETCH_DEPTH,
            PREFETCH_L1_DEPTH,
            FIRST_POLL_DELAY,
            SECOND_POLL_DELAY,
        )


@triton.jit
def qwen3_ffn_clc_kernel(
    input_q,
    input_scale,
    gate_weight_q,
    gate_weight_scale,
    down_weight_q,
    down_weight_scale,
    gate_up,
    activation_q,
    activation_scale,
    output,
    state,
    FRONTIER: tl.constexpr,
    PREFETCH_DEPTH: tl.constexpr,
    PREFETCH_L1_DEPTH: tl.constexpr,
    FIRST_POLL_DELAY: tl.constexpr,
    SECOND_POLL_DELAY: tl.constexpr,
    RECORD_STATS: tl.constexpr,
):
    logical_task = tl.program_id(0).to(tl.int32)
    if RECORD_STATS:
        tl.atomic_add(state + TL_STARTED_SLOT, 1, sem="relaxed", scope="gpu")

    response_addr = _issue_clc_cancel()
    _gate_publish_task(
        input_q,
        input_scale,
        gate_weight_q,
        gate_weight_scale,
        gate_up,
        activation_q,
        activation_scale,
        state,
        logical_task,
        FRONTIER,
    )
    success, canceled_task = _wait_clc_cancel(response_addr)
    if success != 0:
        if RECORD_STATS:
            tl.atomic_add(state + TL_CANCELED_SLOT, 1, sem="relaxed", scope="gpu")
        _run_logical_task(
            input_q,
            input_scale,
            gate_weight_q,
            gate_weight_scale,
            down_weight_q,
            down_weight_scale,
            gate_up,
            activation_q,
            activation_scale,
            output,
            state,
            canceled_task.to(tl.int32),
            FRONTIER,
            PREFETCH_DEPTH,
            PREFETCH_L1_DEPTH,
            FIRST_POLL_DELAY,
            SECOND_POLL_DELAY,
        )


@triton.jit
def qwen3_ffn_clc_token_kernel(
    input_q,
    input_scale,
    gate_weight_q,
    gate_weight_scale,
    down_weight_q,
    down_weight_scale,
    gate_up,
    activation_q,
    activation_scale,
    output,
    state,
    FRONTIER: tl.constexpr,
    PREFETCH_DEPTH: tl.constexpr,
    PREFETCH_L1_DEPTH: tl.constexpr,
    FIRST_POLL_DELAY: tl.constexpr,
    SECOND_POLL_DELAY: tl.constexpr,
    RECORD_STATS: tl.constexpr,
):
    logical_task = tl.program_id(0).to(tl.int32)
    if RECORD_STATS:
        tl.atomic_add(state + TL_STARTED_SLOT, 1, sem="relaxed", scope="gpu")

    response_addr = _issue_clc_cancel()
    _run_logical_task(
        input_q,
        input_scale,
        gate_weight_q,
        gate_weight_scale,
        down_weight_q,
        down_weight_scale,
        gate_up,
        activation_q,
        activation_scale,
        output,
        state,
        logical_task,
        FRONTIER,
        PREFETCH_DEPTH,
        PREFETCH_L1_DEPTH,
        FIRST_POLL_DELAY,
        SECOND_POLL_DELAY,
    )
    success, canceled_task = _wait_clc_cancel(response_addr)
    if success != 0:
        if RECORD_STATS:
            tl.atomic_add(state + TL_CANCELED_SLOT, 1, sem="relaxed", scope="gpu")
        _run_logical_task(
            input_q,
            input_scale,
            gate_weight_q,
            gate_weight_scale,
            down_weight_q,
            down_weight_scale,
            gate_up,
            activation_q,
            activation_scale,
            output,
            state,
            canceled_task.to(tl.int32),
            FRONTIER,
            PREFETCH_DEPTH,
            PREFETCH_L1_DEPTH,
            FIRST_POLL_DELAY,
            SECOND_POLL_DELAY,
        )


@triton.jit
def qwen3_ffn_clc_bootstrap_kernel(
    input_q,
    input_scale,
    gate_weight_q,
    gate_weight_scale,
    down_weight_q,
    down_weight_scale,
    gate_up,
    activation_q,
    activation_scale,
    output,
    state,
    FRONTIER: tl.constexpr,
    PREFETCH_DEPTH: tl.constexpr,
    PREFETCH_L1_DEPTH: tl.constexpr,
    FIRST_POLL_DELAY: tl.constexpr,
    SECOND_POLL_DELAY: tl.constexpr,
    RECORD_STATS: tl.constexpr,
):
    logical_task = tl.program_id(0).to(tl.int32)
    if RECORD_STATS:
        tl.atomic_add(state + TL_STARTED_SLOT, 1, sem="relaxed", scope="gpu")
        tl.atomic_add(state + TL_PROCESSED_SLOT, 1, sem="relaxed", scope="gpu")

    response_addr = _issue_clc_cancel()
    _gate_publish_task(
        input_q,
        input_scale,
        gate_weight_q,
        gate_weight_scale,
        gate_up,
        activation_q,
        activation_scale,
        state,
        logical_task,
        FRONTIER,
    )
    success, canceled_task = _wait_clc_cancel(response_addr)
    if success != 0:
        if RECORD_STATS:
            tl.atomic_add(state + TL_CANCELED_SLOT, 1, sem="relaxed", scope="gpu")
            tl.atomic_add(state + TL_PROCESSED_SLOT, 1, sem="relaxed", scope="gpu")
        _gate_publish_task(
            input_q,
            input_scale,
            gate_weight_q,
            gate_weight_scale,
            gate_up,
            activation_q,
            activation_scale,
            state,
            canceled_task.to(tl.int32),
            FRONTIER,
        )

    down_task = tl.atomic_add(
        state + TL_TASK_CURSOR_SLOT,
        1,
        sem="relaxed",
        scope="gpu",
    )
    if down_task < TL_DOWN_TASKS:
        if RECORD_STATS:
            tl.atomic_add(state + TL_PROCESSED_SLOT, 1, sem="relaxed", scope="gpu")
        _run_down_task(
            activation_q,
            activation_scale,
            down_weight_q,
            down_weight_scale,
            output,
            state,
            down_task,
            FRONTIER,
            PREFETCH_DEPTH,
            PREFETCH_L1_DEPTH,
            FIRST_POLL_DELAY,
            SECOND_POLL_DELAY,
        )


@triton.jit
def qwen3_ffn_clc_ticket_kernel(
    input_q,
    input_scale,
    gate_weight_q,
    gate_weight_scale,
    down_weight_q,
    down_weight_scale,
    gate_up,
    activation_q,
    activation_scale,
    output,
    state,
    FRONTIER: tl.constexpr,
    PREFETCH_DEPTH: tl.constexpr,
    PREFETCH_L1_DEPTH: tl.constexpr,
    FIRST_POLL_DELAY: tl.constexpr,
    SECOND_POLL_DELAY: tl.constexpr,
    RECORD_STATS: tl.constexpr,
):
    if RECORD_STATS:
        tl.atomic_add(state + TL_STARTED_SLOT, 1, sem="relaxed", scope="gpu")

    response_addr = _issue_clc_cancel()
    logical_task = tl.atomic_add(
        state + TL_TASK_CURSOR_SLOT,
        1,
        sem="relaxed",
        scope="gpu",
    )
    if RECORD_STATS:
        tl.atomic_add(state + TL_PROCESSED_SLOT, 1, sem="relaxed", scope="gpu")
    _run_logical_task(
        input_q,
        input_scale,
        gate_weight_q,
        gate_weight_scale,
        down_weight_q,
        down_weight_scale,
        gate_up,
        activation_q,
        activation_scale,
        output,
        state,
        logical_task,
        FRONTIER,
        PREFETCH_DEPTH,
        PREFETCH_L1_DEPTH,
        FIRST_POLL_DELAY,
        SECOND_POLL_DELAY,
    )
    success, _ = _wait_clc_cancel(response_addr)
    if success != 0:
        if RECORD_STATS:
            tl.atomic_add(
                state + TL_CANCELED_SLOT,
                1,
                sem="relaxed",
                scope="gpu",
            )
            tl.atomic_add(
                state + TL_PROCESSED_SLOT,
                1,
                sem="relaxed",
                scope="gpu",
            )
        logical_task = tl.atomic_add(
            state + TL_TASK_CURSOR_SLOT,
            1,
            sem="relaxed",
            scope="gpu",
        )
        _run_logical_task(
            input_q,
            input_scale,
            gate_weight_q,
            gate_weight_scale,
            down_weight_q,
            down_weight_scale,
            gate_up,
            activation_q,
            activation_scale,
            output,
            state,
            logical_task,
            FRONTIER,
            PREFETCH_DEPTH,
            PREFETCH_L1_DEPTH,
            FIRST_POLL_DELAY,
            SECOND_POLL_DELAY,
        )


@triton.jit
def qwen3_ffn_clc_explicit_activation_kernel(
    input_q,
    input_scale,
    gate_weight_q,
    gate_weight_scale,
    down_weight_q,
    down_weight_scale,
    gate_up,
    activation_q,
    activation_scale,
    output,
    state,
    FRONTIER: tl.constexpr,
    PREFETCH_DEPTH: tl.constexpr,
    PREFETCH_L1_DEPTH: tl.constexpr,
    FIRST_POLL_DELAY: tl.constexpr,
    SECOND_POLL_DELAY: tl.constexpr,
    RECORD_STATS: tl.constexpr,
):
    logical_task = tl.program_id(0).to(tl.int32)
    if RECORD_STATS:
        tl.atomic_add(state + TL_STARTED_SLOT, 1, sem="relaxed", scope="gpu")

    response_addr = _issue_clc_cancel()
    _gate_only_task(
        input_q,
        input_scale,
        gate_weight_q,
        gate_weight_scale,
        gate_up,
        state,
        logical_task,
    )
    success, canceled_task = _wait_clc_cancel(response_addr)
    if success != 0:
        if RECORD_STATS:
            tl.atomic_add(state + TL_CANCELED_SLOT, 1, sem="relaxed", scope="gpu")
        next_task = canceled_task.to(tl.int32)
        if next_task < TL_GATE_TASKS:
            _gate_only_task(
                input_q,
                input_scale,
                gate_weight_q,
                gate_weight_scale,
                gate_up,
                state,
                next_task,
            )
        elif next_task < TL_EXPLICIT_DOWN_BASE:
            group = next_task - TL_GATE_TASKS
            _wait_count(state + group, 16, FIRST_POLL_DELAY)
            _activation_task(gate_up, activation_q, activation_scale, group)
            _sync_warp()
            milestone = tl.where(group < FRONTIER, 96, 97)
            tl.atomic_add(state + milestone, 1, sem="release", scope="gpu")
        else:
            _run_down_task(
                activation_q,
                activation_scale,
                down_weight_q,
                down_weight_scale,
                output,
                state,
                next_task - TL_EXPLICIT_DOWN_BASE,
                FRONTIER,
                PREFETCH_DEPTH,
                PREFETCH_L1_DEPTH,
                FIRST_POLL_DELAY,
                SECOND_POLL_DELAY,
            )


def _capture_with_reset(fn, reset):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            reset()
            output = fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    with torch.cuda.stream(stream):
        reset()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = fn()
    torch.cuda.synchronize()
    reset()
    torch.cuda.synchronize()
    return graph, output


def _benchmark_graphs_cold_l2(entries, repeats: int):
    cache = triton.runtime.driver.active.get_empty_cache_for_benchmark()
    samples = {name: [] for name in entries}
    names = list(entries)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for sample in range(repeats):
        order = names[sample % len(names) :] + names[: sample % len(names)]
        for name in order:
            replay, reset = entries[name]
            reset()
            triton.runtime.driver.active.clear_cache(cache)
            torch.cuda.synchronize()
            start.record()
            replay()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end) * 1000.0)
    return {
        name: {
            "median_us": statistics.median(values),
            "mean_us": statistics.fmean(values),
            "p90_us": sorted(values)[min(len(values) - 1, int(0.9 * len(values)))],
        }
        for name, values in samples.items()
    }


def _resources(
    compiled, static_shared_bytes: int = CLC_SCRATCH_BYTES
) -> dict[str, int]:
    _ = compiled.run
    error, blocks_per_sm = cuda_driver.cuOccupancyMaxActiveBlocksPerMultiprocessor(
        cuda_driver.CUfunction(int(compiled.function)),
        32,
        int(compiled.metadata.shared),
    )
    if error != cuda_driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"CUDA occupancy query failed: {error}")
    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    return {
        "registers": int(compiled.n_regs),
        "spills": int(compiled.n_spills),
        "triton_dynamic_shared_bytes": int(compiled.metadata.shared),
        "clc_static_shared_bytes": static_shared_bytes,
        "total_shared_bytes": int(compiled.metadata.shared) + static_shared_bytes,
        "blocks_per_sm": int(blocks_per_sm),
        "device_blocks": int(blocks_per_sm) * int(sm_count),
    }


def _error_stats(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    difference = (actual.float() - expected.float()).abs()
    return {
        "max_abs": float(difference.max().item()),
        "mean_abs": float(difference.mean().item()),
    }


def run(args: argparse.Namespace) -> None:
    if not args.allow_busy:
        require_idle_visible_gpu()
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")
    if not 0 < args.frontier <= ACTIVATION_TASKS:
        raise ValueError("--frontier must be in [1, 96]")
    if args.prefetch_depth > ACTIVATION_TASKS - args.frontier:
        raise ValueError("--prefetch-depth exceeds the activation tail")
    if args.prefetch_l1_depth > ACTIVATION_TASKS - args.frontier:
        raise ValueError("--prefetch-l1-depth exceeds the activation tail")
    if not DOWN_TASKS <= args.direct_workers <= GATE_TASKS:
        raise ValueError("--direct-workers must be between down and gate task counts")

    torch.manual_seed(args.seed)
    device = "cuda"
    hidden_groups = HIDDEN // GROUP
    intermediate_groups = INTERMEDIATE // GROUP
    input_q = torch.randn((1, HIDDEN), device=device, dtype=torch.bfloat16).to(
        torch.float8_e4m3fn
    )
    input_scale = torch.rand((1, hidden_groups), device=device)
    gate_weight_q = torch.randn(
        (2 * INTERMEDIATE, HIDDEN), device=device, dtype=torch.bfloat16
    ).to(torch.float8_e4m3fn)
    gate_weight_scale = torch.rand(
        (2 * intermediate_groups, hidden_groups), device=device
    )
    down_weight_q = torch.randn(
        (HIDDEN, INTERMEDIATE), device=device, dtype=torch.bfloat16
    ).to(torch.float8_e4m3fn)
    down_weight_scale = torch.rand((hidden_groups, intermediate_groups), device=device)

    def make_outputs() -> tuple[torch.Tensor, ...]:
        return (
            torch.empty((1, 2 * INTERMEDIATE), device=device, dtype=torch.bfloat16),
            torch.empty((1, INTERMEDIATE), device=device, dtype=torch.float8_e4m3fn),
            torch.empty((1, ACTIVATION_TASKS), device=device, dtype=torch.float32),
            torch.empty((1, HIDDEN), device=device, dtype=torch.bfloat16),
        )

    clc = make_outputs()
    token = make_outputs()
    bootstrap = make_outputs()
    ticket = make_outputs()
    explicit = make_outputs()
    direct = make_outputs()
    clc_state = torch.zeros(STATE_WORDS, device=device, dtype=torch.int32)
    token_state = torch.zeros(STATE_WORDS, device=device, dtype=torch.int32)
    bootstrap_state = torch.zeros(STATE_WORDS, device=device, dtype=torch.int32)
    ticket_state = torch.zeros(STATE_WORDS, device=device, dtype=torch.int32)
    explicit_state = torch.zeros(STATE_WORDS, device=device, dtype=torch.int32)
    direct_state = torch.zeros(STATE_WORDS, device=device, dtype=torch.int32)

    _, helion_w13 = compile_config(
        block_fp8_mm,
        (input_q, input_scale, gate_weight_q, gate_weight_scale, GROUP),
        FFN_CONFIGS["w13"],
    )
    helion_gate = helion_w13(
        input_q, input_scale, gate_weight_q, gate_weight_scale, GROUP
    )
    _, helion_activation = compile_config(
        silu_and_mul_per_block_quant,
        (helion_gate, GROUP),
        FFN_CONFIGS["silu_quant"],
    )
    helion_activation_q, helion_activation_scale = helion_activation(helion_gate, GROUP)
    _, helion_w2 = compile_config(
        block_fp8_mm,
        (
            helion_activation_q,
            helion_activation_scale,
            down_weight_q,
            down_weight_scale,
            GROUP,
        ),
        FFN_CONFIGS["w2"],
    )
    helion_output = helion_w2(
        helion_activation_q,
        helion_activation_scale,
        down_weight_q,
        down_weight_scale,
        GROUP,
    )

    def launch_helion():
        gate = helion_w13(
            input_q,
            input_scale,
            gate_weight_q,
            gate_weight_scale,
            GROUP,
        )
        quantized, scale = helion_activation(gate, GROUP)
        return helion_w2(
            quantized,
            scale,
            down_weight_q,
            down_weight_scale,
            GROUP,
        )

    def launch_clc(record_stats: bool = False):
        return qwen3_ffn_clc_kernel[(TOTAL_TASKS,)](
            input_q,
            input_scale,
            gate_weight_q,
            gate_weight_scale,
            down_weight_q,
            down_weight_scale,
            clc[0],
            clc[1],
            clc[2],
            clc[3],
            clc_state,
            FRONTIER=args.frontier,
            PREFETCH_DEPTH=args.prefetch_depth,
            PREFETCH_L1_DEPTH=args.prefetch_l1_depth,
            FIRST_POLL_DELAY=args.first_poll_delay,
            SECOND_POLL_DELAY=args.second_poll_delay,
            RECORD_STATS=record_stats,
            num_warps=1,
            num_stages=2,
            num_ctas=1,
            launch_pdl=True,
        )

    def launch_direct():
        return qwen3_ffn_direct_kernel[(args.direct_workers,)](
            input_q,
            input_scale,
            gate_weight_q,
            gate_weight_scale,
            down_weight_q,
            down_weight_scale,
            direct[0],
            direct[1],
            direct[2],
            direct[3],
            direct_state,
            NUM_WORKERS=args.direct_workers,
            FRONTIER=args.frontier,
            PREFETCH_DEPTH=args.prefetch_depth,
            PREFETCH_L1_DEPTH=args.prefetch_l1_depth,
            FIRST_POLL_DELAY=args.first_poll_delay,
            SECOND_POLL_DELAY=args.second_poll_delay,
            num_warps=1,
            num_stages=2,
            launch_cooperative_grid=True,
        )

    def launch_token(record_stats: bool = False):
        return qwen3_ffn_clc_token_kernel[(TOTAL_TASKS,)](
            input_q,
            input_scale,
            gate_weight_q,
            gate_weight_scale,
            down_weight_q,
            down_weight_scale,
            token[0],
            token[1],
            token[2],
            token[3],
            token_state,
            FRONTIER=args.frontier,
            PREFETCH_DEPTH=args.prefetch_depth,
            PREFETCH_L1_DEPTH=args.prefetch_l1_depth,
            FIRST_POLL_DELAY=args.first_poll_delay,
            SECOND_POLL_DELAY=args.second_poll_delay,
            RECORD_STATS=record_stats,
            num_warps=1,
            num_stages=2,
            num_ctas=1,
            launch_pdl=True,
        )

    def launch_ticket(record_stats: bool = False):
        return qwen3_ffn_clc_ticket_kernel[(TOTAL_TASKS,)](
            input_q,
            input_scale,
            gate_weight_q,
            gate_weight_scale,
            down_weight_q,
            down_weight_scale,
            ticket[0],
            ticket[1],
            ticket[2],
            ticket[3],
            ticket_state,
            FRONTIER=args.frontier,
            PREFETCH_DEPTH=args.prefetch_depth,
            PREFETCH_L1_DEPTH=args.prefetch_l1_depth,
            FIRST_POLL_DELAY=args.first_poll_delay,
            SECOND_POLL_DELAY=args.second_poll_delay,
            RECORD_STATS=record_stats,
            num_warps=1,
            num_stages=2,
            num_ctas=1,
            launch_pdl=True,
        )

    def launch_bootstrap(record_stats: bool = False):
        return qwen3_ffn_clc_bootstrap_kernel[(GATE_TASKS,)](
            input_q,
            input_scale,
            gate_weight_q,
            gate_weight_scale,
            down_weight_q,
            down_weight_scale,
            bootstrap[0],
            bootstrap[1],
            bootstrap[2],
            bootstrap[3],
            bootstrap_state,
            FRONTIER=args.frontier,
            PREFETCH_DEPTH=args.prefetch_depth,
            PREFETCH_L1_DEPTH=args.prefetch_l1_depth,
            FIRST_POLL_DELAY=args.first_poll_delay,
            SECOND_POLL_DELAY=args.second_poll_delay,
            RECORD_STATS=record_stats,
            num_warps=1,
            num_stages=2,
            num_ctas=1,
            launch_pdl=True,
        )

    def launch_explicit(record_stats: bool = False):
        return qwen3_ffn_clc_explicit_activation_kernel[(EXPLICIT_TOTAL_TASKS,)](
            input_q,
            input_scale,
            gate_weight_q,
            gate_weight_scale,
            down_weight_q,
            down_weight_scale,
            explicit[0],
            explicit[1],
            explicit[2],
            explicit[3],
            explicit_state,
            FRONTIER=args.frontier,
            PREFETCH_DEPTH=args.prefetch_depth,
            PREFETCH_L1_DEPTH=args.prefetch_l1_depth,
            FIRST_POLL_DELAY=args.first_poll_delay,
            SECOND_POLL_DELAY=args.second_poll_delay,
            RECORD_STATS=record_stats,
            num_warps=1,
            num_stages=2,
            num_ctas=1,
            launch_pdl=True,
        )

    clc_state.zero_()
    stats_compiled = launch_clc(record_stats=True)
    token_state.zero_()
    token_stats_compiled = launch_token(record_stats=True)
    bootstrap_state.zero_()
    bootstrap_stats_compiled = launch_bootstrap(record_stats=True)
    ticket_state.zero_()
    ticket_stats_compiled = launch_ticket(record_stats=True)
    explicit_state.zero_()
    explicit_stats_compiled = launch_explicit(record_stats=True)
    direct_state.zero_()
    direct_compiled = launch_direct()
    torch.cuda.synchronize()

    started = int(clc_state[STARTED_SLOT].item())
    canceled = int(clc_state[CANCELED_SLOT].item())
    if started + canceled != TOTAL_TASKS:
        raise AssertionError(
            f"CLC partition mismatch: {started=} + {canceled=} != {TOTAL_TASKS}"
        )
    if started != FIRST_WAVE_GATE_TASKS:
        raise AssertionError(
            f"expected {FIRST_WAVE_GATE_TASKS} resident gate CTAs, got {started}"
        )
    if not torch.all(clc_state[:ACTIVATION_TASKS] == 16):
        raise AssertionError("not every activation group received 16 gate tiles")
    if int(clc_state[96].item()) != args.frontier:
        raise AssertionError("frontier activation count mismatch")
    if int(clc_state[97].item()) != ACTIVATION_TASKS - args.frontier:
        raise AssertionError("tail activation count mismatch")

    token_started = int(token_state[STARTED_SLOT].item())
    token_canceled = int(token_state[CANCELED_SLOT].item())
    if token_started + token_canceled != TOTAL_TASKS:
        raise AssertionError("token CLC launch partition mismatch")
    if not torch.all(token_state[:ACTIVATION_TASKS] == 16):
        raise AssertionError("token CLC activation dependencies were incomplete")
    if int(token_state[96].item()) != args.frontier:
        raise AssertionError("token CLC frontier activation count mismatch")
    if int(token_state[97].item()) != ACTIVATION_TASKS - args.frontier:
        raise AssertionError("token CLC tail activation count mismatch")

    bootstrap_started = int(bootstrap_state[STARTED_SLOT].item())
    bootstrap_canceled = int(bootstrap_state[CANCELED_SLOT].item())
    bootstrap_cursor = int(bootstrap_state[TASK_CURSOR_SLOT].item())
    bootstrap_processed = int(bootstrap_state[PROCESSED_SLOT].item())
    if bootstrap_started + bootstrap_canceled != GATE_TASKS:
        raise AssertionError("bootstrap CLC launch partition mismatch")
    if bootstrap_cursor != bootstrap_started:
        raise AssertionError("bootstrap tail must receive one claim per resident CTA")
    if bootstrap_processed != TOTAL_TASKS:
        raise AssertionError("bootstrap CLC command partition mismatch")
    if not torch.all(bootstrap_state[:ACTIVATION_TASKS] == 16):
        raise AssertionError("bootstrap CLC activation dependencies were incomplete")
    if int(bootstrap_state[96].item()) != args.frontier:
        raise AssertionError("bootstrap CLC frontier activation count mismatch")
    if int(bootstrap_state[97].item()) != ACTIVATION_TASKS - args.frontier:
        raise AssertionError("bootstrap CLC tail activation count mismatch")

    ticket_started = int(ticket_state[STARTED_SLOT].item())
    ticket_canceled = int(ticket_state[CANCELED_SLOT].item())
    ticket_cursor = int(ticket_state[TASK_CURSOR_SLOT].item())
    ticket_processed = int(ticket_state[PROCESSED_SLOT].item())
    if ticket_started + ticket_canceled != TOTAL_TASKS:
        raise AssertionError("ticket CLC launch partition mismatch")
    if ticket_cursor != TOTAL_TASKS or ticket_processed != TOTAL_TASKS:
        raise AssertionError("ticket CLC command partition mismatch")
    if not torch.all(ticket_state[:ACTIVATION_TASKS] == 16):
        raise AssertionError("ticket CLC activation dependencies were incomplete")
    if int(ticket_state[96].item()) != args.frontier:
        raise AssertionError("ticket CLC frontier activation count mismatch")
    if int(ticket_state[97].item()) != ACTIVATION_TASKS - args.frontier:
        raise AssertionError("ticket CLC tail activation count mismatch")

    explicit_started = int(explicit_state[STARTED_SLOT].item())
    explicit_canceled = int(explicit_state[CANCELED_SLOT].item())
    if explicit_started + explicit_canceled != EXPLICIT_TOTAL_TASKS:
        raise AssertionError("explicit CLC partition mismatch")
    if explicit_started != FIRST_WAVE_GATE_TASKS:
        raise AssertionError("explicit CLC residency no longer matches first wave")
    if not torch.all(explicit_state[:ACTIVATION_TASKS] == 16):
        raise AssertionError("explicit activation dependencies were incomplete")
    if int(explicit_state[96].item()) != args.frontier:
        raise AssertionError("explicit frontier activation count mismatch")
    if int(explicit_state[97].item()) != ACTIVATION_TASKS - args.frontier:
        raise AssertionError("explicit tail activation count mismatch")

    for actual, expected, atol, rtol in (
        (clc[0], helion_gate, 0.125, 3e-2),
        (clc[1], helion_activation_q, 64.0, 3e-2),
        (clc[2], helion_activation_scale, 2e-3, 3e-2),
        (clc[3], helion_output, 0.25, 5e-2),
        (token[0], helion_gate, 0.125, 3e-2),
        (token[1], helion_activation_q, 64.0, 3e-2),
        (token[2], helion_activation_scale, 2e-3, 3e-2),
        (token[3], helion_output, 0.25, 5e-2),
        (bootstrap[0], helion_gate, 0.125, 3e-2),
        (bootstrap[1], helion_activation_q, 64.0, 3e-2),
        (bootstrap[2], helion_activation_scale, 2e-3, 3e-2),
        (bootstrap[3], helion_output, 0.25, 5e-2),
        (ticket[0], helion_gate, 0.125, 3e-2),
        (ticket[1], helion_activation_q, 64.0, 3e-2),
        (ticket[2], helion_activation_scale, 2e-3, 3e-2),
        (ticket[3], helion_output, 0.25, 5e-2),
        (explicit[0], helion_gate, 0.125, 3e-2),
        (explicit[1], helion_activation_q, 64.0, 3e-2),
        (explicit[2], helion_activation_scale, 2e-3, 3e-2),
        (explicit[3], helion_output, 0.25, 5e-2),
        (direct[0], helion_gate, 0.125, 3e-2),
        (direct[1], helion_activation_q, 64.0, 3e-2),
        (direct[2], helion_activation_scale, 2e-3, 3e-2),
        (direct[3], helion_output, 0.25, 5e-2),
    ):
        torch.testing.assert_close(
            actual.float(), expected.float(), atol=atol, rtol=rtol
        )

    clc_state.zero_()
    compiled = launch_clc(record_stats=False)
    token_state.zero_()
    token_compiled = launch_token(record_stats=False)
    bootstrap_state.zero_()
    bootstrap_compiled = launch_bootstrap(record_stats=False)
    ticket_state.zero_()
    ticket_compiled = launch_ticket(record_stats=False)
    explicit_state.zero_()
    explicit_compiled = launch_explicit(record_stats=False)
    torch.cuda.synchronize()
    resources = _resources(compiled)
    explicit_resources = _resources(explicit_compiled)
    if resources["device_blocks"] != FIRST_WAVE_GATE_TASKS:
        raise AssertionError("compiled occupancy no longer matches the first wave")

    helion_graph, _ = capture(launch_helion)
    direct_graph, _ = _capture_with_reset(launch_direct, direct_state.zero_)
    clc_graph, _ = _capture_with_reset(launch_clc, clc_state.zero_)
    token_graph, _ = _capture_with_reset(launch_token, token_state.zero_)
    bootstrap_graph, _ = _capture_with_reset(
        launch_bootstrap,
        bootstrap_state.zero_,
    )
    ticket_graph, _ = _capture_with_reset(launch_ticket, ticket_state.zero_)
    explicit_graph, _ = _capture_with_reset(launch_explicit, explicit_state.zero_)
    pids = visible_gpu_pids()
    if not args.allow_busy and (foreign_pids := pids - {os.getpid()}):
        raise RuntimeError(
            f"GPU gained foreign compute processes {sorted(foreign_pids)}"
        )
    timings = _benchmark_graphs_cold_l2(
        {
            "helion_separate_graph": (helion_graph.replay, lambda: None),
            "triton_direct_static": (direct_graph.replay, direct_state.zero_),
            "triton_clc": (clc_graph.replay, clc_state.zero_),
            "triton_clc_token": (token_graph.replay, token_state.zero_),
            "triton_clc_bootstrap": (
                bootstrap_graph.replay,
                bootstrap_state.zero_,
            ),
            "triton_clc_ticket": (ticket_graph.replay, ticket_state.zero_),
            "triton_clc_explicit_activation": (
                explicit_graph.replay,
                explicit_state.zero_,
            ),
        },
        args.repeats,
    )
    if not args.allow_busy and visible_gpu_pids() != pids:
        raise RuntimeError("GPU process set changed during benchmark")

    helion_us = float(timings["helion_separate_graph"]["median_us"])
    direct_us = float(timings["triton_direct_static"]["median_us"])
    clc_us = float(timings["triton_clc"]["median_us"])
    token_us = float(timings["triton_clc_token"]["median_us"])
    bootstrap_us = float(timings["triton_clc_bootstrap"]["median_us"])
    ticket_us = float(timings["triton_clc_ticket"]["median_us"])
    ptx = compiled.asm["ptx"]
    issue_work_wait_order = (
        ptx.index("clusterlaunchcontrol.try_cancel")
        < ptx.index("mma.sync")
        < ptx.index("mbarrier.try_wait")
    )
    if not issue_work_wait_order:
        raise AssertionError("CLC issue -> useful work -> wait ordering changed")
    result = {
        "workload": "Qwen3-8B FP8 decode FFN CLC megakernel",
        "device": torch.cuda.get_device_name(),
        "shape": {
            "batch": 1,
            "hidden": HIDDEN,
            "intermediate": INTERMEDIATE,
            "group": GROUP,
            "gate_tasks": GATE_TASKS,
            "activation_tasks": ACTIVATION_TASKS,
            "down_tasks": DOWN_TASKS,
            "logical_tasks": TOTAL_TASKS,
            "first_wave_gate_tasks": FIRST_WAVE_GATE_TASKS,
        },
        "schedule": {
            "frontier_groups": args.frontier,
            "prefetch_depth": args.prefetch_depth,
            "prefetch_l1_depth": args.prefetch_l1_depth,
            "first_poll_delay": args.first_poll_delay,
            "second_poll_delay": args.second_poll_delay,
            "clc_scratch_bytes": CLC_SCRATCH_BYTES,
            "launch_pdl": True,
            "direct_workers": args.direct_workers,
            "stolen_id_order": "remaining gate tiles, then down tiles",
            "ticket_order": "monotonic all-CTA command cursor",
        },
        "clc": {
            "physically_started_ctas": started,
            "canceled_and_reassigned_ctas": canceled,
            "partition_total": started + canceled,
            "token_dispatch": {
                "physically_started_ctas": token_started,
                "canceled_and_reassigned_ctas": token_canceled,
                "partition_total": token_started + token_canceled,
            },
            "dependency_free_bootstrap": {
                "physically_started_ctas": bootstrap_started,
                "canceled_and_reassigned_ctas": bootstrap_canceled,
                "partition_total": bootstrap_started + bootstrap_canceled,
                "tail_claims": bootstrap_cursor,
                "processed_commands": bootstrap_processed,
            },
            "ticket_cursor": {
                "physically_started_ctas": ticket_started,
                "successful_cancellations": ticket_canceled,
                "partition_total": ticket_started + ticket_canceled,
                "claimed_commands": ticket_cursor,
                "processed_commands": ticket_processed,
            },
            "explicit_activation": {
                "physically_started_ctas": explicit_started,
                "canceled_and_reassigned_ctas": explicit_canceled,
                "partition_total": explicit_started + explicit_canceled,
            },
        },
        "correctness": {
            "gate_up": _error_stats(clc[0], helion_gate),
            "activation_q": _error_stats(clc[1], helion_activation_q),
            "activation_scale": _error_stats(clc[2], helion_activation_scale),
            "output": _error_stats(clc[3], helion_output),
            "token_output": _error_stats(token[3], helion_output),
            "bootstrap_output": _error_stats(bootstrap[3], helion_output),
            "ticket_output": _error_stats(ticket[3], helion_output),
            "direct_static_output": _error_stats(direct[3], helion_output),
            "explicit_activation_output": _error_stats(explicit[3], helion_output),
        },
        "cold_l2": {
            "flush_bytes": 256 * 1024 * 1024,
            "timings_us": timings,
            "speedup_vs_helion_separate_graph": helion_us / clc_us,
            "speedup_vs_direct_static": direct_us / clc_us,
            "token_speedup_vs_helion_separate_graph": helion_us / token_us,
            "token_speedup_vs_direct_id_clc": clc_us / token_us,
            "bootstrap_speedup_vs_helion_separate_graph": helion_us / bootstrap_us,
            "bootstrap_speedup_vs_direct_id_clc": clc_us / bootstrap_us,
            "ticket_speedup_vs_helion_separate_graph": helion_us / ticket_us,
            "ticket_speedup_vs_direct_id_clc": clc_us / ticket_us,
        },
        "resources": resources,
        "token_resources": _resources(token_compiled),
        "bootstrap_resources": _resources(bootstrap_compiled),
        "ticket_resources": _resources(ticket_compiled),
        "direct_static_resources": _resources(direct_compiled, 0),
        "explicit_activation_resources": explicit_resources,
        "ptx_checks": {
            "contains_clc": "clusterlaunchcontrol.try_cancel" in ptx,
            "contains_static_scratch": (
                ".shared .align 16 .b8 qwen3_clc_scratch[12288]" in ptx
            ),
            "contains_triton_global_smem": "global_smem" in ptx,
            "contains_acquire_load": "ld.acquire.gpu.global.u32" in ptx,
            "contains_release_atomic": "atom.global.gpu.release" in ptx,
            "issue_work_wait_order": issue_work_wait_order,
        },
        "instrumented_resources": _resources(stats_compiled),
        "token_instrumented_resources": _resources(token_stats_compiled),
        "bootstrap_instrumented_resources": _resources(bootstrap_stats_compiled),
        "ticket_instrumented_resources": _resources(ticket_stats_compiled),
        "explicit_activation_instrumented_resources": _resources(
            explicit_stats_compiled
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    args.output.with_suffix(".ptx").write_text(ptx)
    args.output.with_name(f"{args.output.stem}_explicit.ptx").write_text(
        explicit_compiled.asm["ptx"]
    )
    args.output.with_name(f"{args.output.stem}_ticket.ptx").write_text(
        ticket_compiled.asm["ptx"]
    )
    print("RESULT", json.dumps(result, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frontier", type=int, default=64)
    parser.add_argument("--prefetch-depth", type=int, default=8)
    parser.add_argument("--prefetch-l1-depth", type=int, default=0)
    parser.add_argument("--first-poll-delay", type=int, default=0)
    parser.add_argument("--second-poll-delay", type=int, default=0)
    parser.add_argument("--direct-workers", type=int, default=1184)
    parser.add_argument("--repeats", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow-busy", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/triton_qwen3_ffn_clc_manual_result.json"),
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
