# ruff: noqa: ANN001, ANN202
# pyrefly: ignore-errors
"""Schedule Helion-generated Gemma 4 root bodies from a raw Triton kernel.

The generated root functions are kept byte-for-byte.  Only their top-level
dispatch, waits, publications, and proven continuations are replaced.  This
isolates scheduling choices from differences in the computation codegen.

The initial probe supports the sliding, non-KV-shared representative layer.
That is the approximately 80 microsecond separate-kernel control and therefore
the clearest target for schedule exploration.
"""

from __future__ import annotations

import argparse
import ast
import copy
import ctypes
import functools
import json
import linecache
from pathlib import Path
import re
import textwrap

from benchmarks.gemma4.common import Gemma4E4BShape
from benchmarks.gemma4.common import allocate_layer
from benchmarks.gemma4.common import benchmark_interleaved
from benchmarks.gemma4.common import capture
from benchmarks.gemma4.common import layer_reference
from benchmarks.gemma4.common import require_idle_visible_gpu
from benchmarks.gemma4.common import visible_gpu_pids
import benchmarks.gemma4.helion_gemma4_e4b_layer as layer
import benchmarks.gemma4.helion_gemma4_e4b_megakernel as mega
import torch

import helion


@functools.cache
def _libcuda():
    return ctypes.CDLL("libcuda.so.1")


def resident_capacity(compiled, num_warps) -> tuple[int, int]:
    """Return the driver-proven co-resident worker capacity."""
    compiled._init_handles()
    properties = torch.cuda.get_device_properties(0)
    threads = num_warps * 32
    blocks = ctypes.c_int()
    status = _libcuda().cuOccupancyMaxActiveBlocksPerMultiprocessor(
        ctypes.byref(blocks),
        ctypes.c_void_p(compiled.function),
        ctypes.c_int(threads),
        ctypes.c_size_t(compiled.metadata.shared),
    )
    if status != 0:
        raise RuntimeError(
            f"cuOccupancyMaxActiveBlocksPerMultiprocessor failed: {status}"
        )
    per_sm = blocks.value
    return per_sm * properties.multi_processor_count, per_sm


SCHEDULER_SOURCE = r"""
@triton.jit
def _probe_sync_warp():
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
def _probe_load_acquire(address):
    return tl.inline_asm_elementwise(
        asm="ld.acquire.gpu.global.u32 $0, [$1];",
        constraints="=r,l",
        args=[address],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _probe_load_relaxed(address):
    return tl.inline_asm_elementwise(
        asm="ld.volatile.global.u32 $0, [$1];",
        constraints="=r,l",
        args=[address],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _probe_nanosleep(DELAY: tl.constexpr):
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
def _probe_globaltimer():
    return tl.inline_asm_elementwise(
        asm="mov.u64 $0, %globaltimer;",
        constraints="=l",
        args=[],
        dtype=tl.int64,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _probe_wait(
    address,
    epoch,
    POLL_DELAY: tl.constexpr,
    ENABLED: tl.constexpr = True,
    RELAXED_POLL: tl.constexpr = False,
):
    if ENABLED:
        value = (
            _probe_load_relaxed(address)
            if RELAXED_POLL
            else _probe_load_acquire(address)
        )
        while value != epoch:
            if POLL_DELAY:
                _probe_nanosleep(POLL_DELAY)
            value = (
                _probe_load_relaxed(address)
                if RELAXED_POLL
                else _probe_load_acquire(address)
            )
        if RELAXED_POLL:
            value = _probe_load_acquire(address)
            while value != epoch:
                value = _probe_load_acquire(address)
        _probe_sync_warp()


@triton.jit
def _probe_wait_count(
    address,
    target,
    POLL_DELAY: tl.constexpr,
    RELAXED_POLL: tl.constexpr,
):
    value = (
        _probe_load_relaxed(address)
        if RELAXED_POLL
        else _probe_load_acquire(address)
    )
    while value - target < 0:
        if POLL_DELAY:
            _probe_nanosleep(POLL_DELAY)
        value = (
            _probe_load_relaxed(address)
            if RELAXED_POLL
            else _probe_load_acquire(address)
        )
    if RELAXED_POLL:
        value = _probe_load_acquire(address)
        while value - target < 0:
            value = _probe_load_acquire(address)
    _probe_sync_warp()


@triton.jit
def _probe_wait_phase(
    arrivals,
    ready,
    phase: tl.constexpr,
    epoch,
    count: tl.constexpr,
    FUSED_SIGNALS: tl.constexpr,
    POLL_DELAY: tl.constexpr,
    RELAXED_POLL: tl.constexpr,
):
    if FUSED_SIGNALS != 2:
        if FUSED_SIGNALS:
            _probe_wait_count(
                arrivals + phase * 32,
                epoch * count,
                POLL_DELAY,
                RELAXED_POLL,
            )
        else:
            _probe_wait(
                ready + phase * 32,
                epoch,
                POLL_DELAY,
                True,
                RELAXED_POLL,
            )


@triton.jit
def _probe_publish(
    arrivals,
    ready,
    phase: tl.constexpr,
    epoch,
    count: tl.constexpr,
    FUSED_SIGNALS: tl.constexpr,
):
    tl.debug_barrier()
    if FUSED_SIGNALS:
        tl.atomic_add(arrivals + phase * 32, 1, sem="release", scope="gpu")
    else:
        previous = tl.atomic_add(
            arrivals + phase * 32,
            1,
            sem="acq_rel",
            scope="gpu",
        )
        if previous % count == count - 1:
            tl.atomic_xchg(ready + phase * 32, epoch, sem="release", scope="gpu")


@triton.jit
def _probe_counted_event_arrive(address, count):
    previous = tl.atomic_add(
        address,
        1,
        sem="acq_rel",
        scope="gpu",
    )
    return previous % count == count - 1


@triton.jit(noinline=True)
def _probe_down_two_splits(
    activation,
    down_weight,
    down,
    ffn_ready,
    task,
    epoch,
    H: tl.constexpr,
    INTERMEDIATE: tl.constexpr,
    FIRST_ACTIVATION_TASKS: tl.constexpr,
    DOWN_STAGES: tl.constexpr,
    DOWN_UNROLL: tl.constexpr,
    POLL_DELAY: tl.constexpr,
    WAITS: tl.constexpr,
    RELAXED_POLL: tl.constexpr,
):
    columns = task * _BLOCK_SIZE_26 + tl.arange(0, _BLOCK_SIZE_26).to(tl.int32)
    accumulator = tl.full([1, _BLOCK_SIZE_26], 0.0, tl.float32)
    first_k: tl.constexpr = FIRST_ACTIVATION_TASKS * _BLOCK_SIZE_24
    _probe_wait(ffn_ready, epoch, POLL_DELAY, WAITS, RELAXED_POLL)
    for k_base in tl.range(
        0,
        first_k,
        _BLOCK_SIZE_27,
        num_stages=DOWN_STAGES,
        loop_unroll_factor=DOWN_UNROLL,
        flatten=True,
    ):
        k = k_base + tl.arange(0, _BLOCK_SIZE_27).to(tl.int32)
        accumulator_copy = accumulator
        activation_values = tl.broadcast_to(
            tl.load(activation + k[None, :]),
            [1, _BLOCK_SIZE_27],
        )
        weights = tl.load(
            down_weight + columns[:, None] * INTERMEDIATE + k[None, :]
        )
        accumulator = tl.dot(
            tl.cast(activation_values, tl.bfloat16),
            tl.cast(tl.permute(weights, [1, 0]), tl.bfloat16),
            acc=accumulator_copy,
            input_precision="tf32",
            out_dtype=tl.float32,
        )
    _probe_wait(ffn_ready + 1, epoch, POLL_DELAY, WAITS, RELAXED_POLL)
    for k_base in tl.range(
        first_k,
        INTERMEDIATE,
        _BLOCK_SIZE_27,
        num_stages=DOWN_STAGES,
        loop_unroll_factor=DOWN_UNROLL,
        flatten=True,
    ):
        k = k_base + tl.arange(0, _BLOCK_SIZE_27).to(tl.int32)
        accumulator_copy = accumulator
        activation_values = tl.broadcast_to(
            tl.load(activation + k[None, :]),
            [1, _BLOCK_SIZE_27],
        )
        weights = tl.load(
            down_weight + columns[:, None] * INTERMEDIATE + k[None, :]
        )
        accumulator = tl.dot(
            tl.cast(activation_values, tl.bfloat16),
            tl.cast(tl.permute(weights, [1, 0]), tl.bfloat16),
            acc=accumulator_copy,
            input_precision="tf32",
            out_dtype=tl.float32,
        )
    tl.store(
        down + tl.broadcast_to(columns[None, :], [1, _BLOCK_SIZE_26]),
        tl.cast(accumulator, tl.bfloat16),
    )


@triton.jit(noinline=True)
def _probe_stream_ffn_producer(
    ff_input,
    gate_up_weight,
    gate_up,
    activation,
    activation_arrivals,
    ffn_split_arrivals,
    ffn_ready,
    trace,
    logical_task,
    epoch,
    INTERMEDIATE: tl.constexpr,
    ROOT_7_OFFSET: tl.constexpr,
    ROOT_8_OFFSET: tl.constexpr,
    ROOT_8_TASKS: tl.constexpr,
    FIRST_ACTIVATION_TASKS: tl.constexpr,
    IMMEDIATE_ACTIVATION: tl.constexpr,
    COUNTED_EVENT_ON_READY: tl.constexpr,
    TRACE: tl.constexpr,
):
    subtiles_per_activation: tl.constexpr = _BLOCK_SIZE_24 // _BLOCK_SIZE_21
    fan_in: tl.constexpr = 2 * subtiles_per_activation
    activation_task = logical_task // fan_in
    within_activation = logical_task % fan_in
    half_tasks: tl.constexpr = INTERMEDIATE // _BLOCK_SIZE_21
    physical_task = tl.where(
        within_activation < subtiles_per_activation,
        activation_task * subtiles_per_activation + within_activation,
        half_tasks
        + activation_task * subtiles_per_activation
        + within_activation
        - subtiles_per_activation,
    )
    tile_dependency_root_7(
        ff_input,
        gate_up_weight,
        gate_up,
        ROOT_7_OFFSET + physical_task,
    )
    tl.debug_barrier()
    if COUNTED_EVENT_ON_READY:
        activation_ready = _probe_counted_event_arrive(
            activation_arrivals + activation_task,
            fan_in,
        )
    else:
        previous = tl.atomic_add(
            activation_arrivals + activation_task,
            1,
            sem="acq_rel",
            scope="gpu",
        )
        activation_ready = previous % fan_in == fan_in - 1
    if activation_ready and IMMEDIATE_ACTIVATION:
        tile_dependency_root_8(
            gate_up,
            activation,
            ROOT_8_OFFSET + activation_task,
        )
        tl.debug_barrier()
        split = tl.where(
            activation_task < FIRST_ACTIVATION_TASKS,
            0,
            1,
        )
        split_count = tl.where(
            split == 0,
            FIRST_ACTIVATION_TASKS,
            ROOT_8_TASKS - FIRST_ACTIVATION_TASKS,
        ).to(tl.int32)
        if COUNTED_EVENT_ON_READY:
            split_ready = _probe_counted_event_arrive(
                ffn_split_arrivals + split,
                split_count,
            )
        else:
            split_previous = tl.atomic_add(
                ffn_split_arrivals + split,
                1,
                sem="acq_rel",
                scope="gpu",
            )
            split_ready = split_previous % split_count == split_count - 1
        if split_ready:
            tl.atomic_xchg(
                ffn_ready + split,
                epoch,
                sem="release",
                scope="gpu",
            )
            if TRACE:
                tl.store(trace + split, _probe_globaltimer())


@triton.jit
def gemma4_codegen_gate_probe(
    ff_input,
    gate_up_weight,
    gate_up,
    TOTAL_WORKERS: tl.constexpr,
    H: tl.constexpr,
    INTERMEDIATE: tl.constexpr,
    Q_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    SPLITS: tl.constexpr,
    CONSUMER_MAJOR: tl.constexpr,
):
    worker = tl.program_id(0)
    q_per_kv: tl.constexpr = Q_HEADS // KV_HEADS
    projected_width: tl.constexpr = (Q_HEADS + 2 * KV_HEADS) * HEAD_DIM
    root_7_offset: tl.constexpr = (
        1
        + tl.cdiv(projected_width, _BLOCK_SIZE_3)
        + (Q_HEADS + 2 * KV_HEADS) * tl.cdiv(HEAD_DIM, _BLOCK_SIZE_7)
        + SPLITS * KV_HEADS * tl.cdiv(q_per_kv, _BLOCK_SIZE_10)
        + KV_HEADS * tl.cdiv(q_per_kv, _BLOCK_SIZE_14)
        + tl.cdiv(H, _BLOCK_SIZE_17)
        + 1
    )
    tasks: tl.constexpr = tl.cdiv(2 * INTERMEDIATE, _BLOCK_SIZE_21)
    full_waves: tl.constexpr = tasks // TOTAL_WORKERS
    for wave in range(full_waves):
        logical_task = worker + wave * TOTAL_WORKERS
        if CONSUMER_MAJOR:
            subtiles: tl.constexpr = _BLOCK_SIZE_24 // _BLOCK_SIZE_21
            fan_in: tl.constexpr = 2 * subtiles
            activation_task = logical_task // fan_in
            within_activation = logical_task % fan_in
            half_tasks: tl.constexpr = INTERMEDIATE // _BLOCK_SIZE_21
            task = tl.where(
                within_activation < subtiles,
                activation_task * subtiles + within_activation,
                half_tasks
                + activation_task * subtiles
                + within_activation
                - subtiles,
            )
        else:
            task = logical_task
        tile_dependency_root_7(
            ff_input,
            gate_up_weight,
            gate_up,
            root_7_offset + task,
        )
    tail: tl.constexpr = tasks % TOTAL_WORKERS
    if tail and worker < tail:
        logical_task = full_waves * TOTAL_WORKERS + worker
        if CONSUMER_MAJOR:
            subtiles: tl.constexpr = _BLOCK_SIZE_24 // _BLOCK_SIZE_21
            fan_in: tl.constexpr = 2 * subtiles
            activation_task = logical_task // fan_in
            within_activation = logical_task % fan_in
            half_tasks: tl.constexpr = INTERMEDIATE // _BLOCK_SIZE_21
            task = tl.where(
                within_activation < subtiles,
                activation_task * subtiles + within_activation,
                half_tasks
                + activation_task * subtiles
                + within_activation
                - subtiles,
            )
        else:
            task = logical_task
        tile_dependency_root_7(
            ff_input,
            gate_up_weight,
            gate_up,
            root_7_offset + task,
        )


@triton.jit
def gemma4_codegen_schedule_probe(
    hidden_states,
    input_norm_weight,
    input_norm,
    qkv_weight,
    projected_qkv,
    q_norm_weight,
    k_norm_weight,
    slot_mapping,
    kv_cache,
    position,
    cos_sin,
    block_table,
    partial_out,
    partial_lse,
    attention,
    o_weight,
    attention_out,
    post_attention_norm_weight,
    residual,
    pre_ff_norm_weight,
    ff_input,
    gate_up_weight,
    gate_up,
    activation,
    down_weight,
    down,
    post_ff_norm_weight,
    hidden,
    ple_gate_weight,
    per_layer_input,
    ple_input,
    ple_projection_weight,
    ple_projection,
    layer_scalar,
    post_ple_norm_weight,
    output,
    worker_epoch,
    phase_arrivals,
    phase_ready,
    head_arrivals,
    group_arrivals,
    group_ready,
    attention_arrivals,
    activation_arrivals,
    ffn_split_arrivals,
    ffn_ready,
    trace,
    eps,
    TOTAL_WORKERS: tl.constexpr,
    H: tl.constexpr,
    INTERMEDIATE: tl.constexpr,
    PLE: tl.constexpr,
    Q_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    SPLITS: tl.constexpr,
    RDIM: tl.constexpr,
    QKV_CONTINUATION: tl.constexpr,
    ATTENTION_CONTINUATION: tl.constexpr,
    FFN_CONTINUATION: tl.constexpr,
    FFN_CONSUMER_MAJOR: tl.constexpr,
    FFN_STREAM: tl.constexpr,
    FFN_SCHEDULED_ACTIVATION: tl.constexpr,
    FFN_FIRST_GROUPS: tl.constexpr,
    FFN_CONSUMER_WORKERS: tl.constexpr,
    COUNTED_EVENT_ON_READY: tl.constexpr,
    STREAM_DOWN_STAGES: tl.constexpr,
    STREAM_DOWN_UNROLL: tl.constexpr,
    O_CONTINUATION: tl.constexpr,
    POLL_DELAY: tl.constexpr,
    TRACE: tl.constexpr,
    FUSED_SIGNALS: tl.constexpr,
    SPREAD_NARROW_ROOTS: tl.constexpr,
    RELAXED_POLL: tl.constexpr,
):
    worker = tl.program_id(0)
    epoch = tl.load(worker_epoch + worker) + 1

    phase_input: tl.constexpr = 0
    phase_qkv: tl.constexpr = 1
    phase_heads: tl.constexpr = 2
    phase_attention: tl.constexpr = 3
    phase_merged: tl.constexpr = 4
    phase_o: tl.constexpr = 5
    phase_post_attention: tl.constexpr = 6
    phase_gate: tl.constexpr = 7
    phase_activation: tl.constexpr = 8
    phase_down: tl.constexpr = 9
    phase_post_ff: tl.constexpr = 10
    phase_ple_gate: tl.constexpr = 11
    phase_ple_projection: tl.constexpr = 12
    phase_final: tl.constexpr = 13

    q_per_kv: tl.constexpr = Q_HEADS // KV_HEADS
    projected_width: tl.constexpr = (Q_HEADS + 2 * KV_HEADS) * HEAD_DIM
    root_0_offset: tl.constexpr = 0
    root_0_tasks: tl.constexpr = 1
    root_1_offset: tl.constexpr = root_0_offset + root_0_tasks
    root_1_tasks: tl.constexpr = tl.cdiv(projected_width, _BLOCK_SIZE_3)
    root_2_offset: tl.constexpr = root_1_offset + root_1_tasks
    root_2_tasks: tl.constexpr = (Q_HEADS + 2 * KV_HEADS) * tl.cdiv(
        HEAD_DIM, _BLOCK_SIZE_7
    )
    root_3_offset: tl.constexpr = root_2_offset + root_2_tasks
    root_3_tasks: tl.constexpr = SPLITS * KV_HEADS * tl.cdiv(
        q_per_kv, _BLOCK_SIZE_10
    )
    root_4_offset: tl.constexpr = root_3_offset + root_3_tasks
    root_4_tasks: tl.constexpr = KV_HEADS * tl.cdiv(q_per_kv, _BLOCK_SIZE_14)
    root_5_offset: tl.constexpr = root_4_offset + root_4_tasks
    root_5_tasks: tl.constexpr = tl.cdiv(H, _BLOCK_SIZE_17)
    root_6_offset: tl.constexpr = root_5_offset + root_5_tasks
    root_6_tasks: tl.constexpr = 1
    root_7_offset: tl.constexpr = root_6_offset + root_6_tasks
    root_7_tasks: tl.constexpr = tl.cdiv(2 * INTERMEDIATE, _BLOCK_SIZE_21)
    root_8_offset: tl.constexpr = root_7_offset + root_7_tasks
    root_8_tasks: tl.constexpr = tl.cdiv(INTERMEDIATE, _BLOCK_SIZE_24)
    root_9_offset: tl.constexpr = root_8_offset + root_8_tasks
    root_9_tasks: tl.constexpr = tl.cdiv(H, _BLOCK_SIZE_26)
    root_10_offset: tl.constexpr = root_9_offset + root_9_tasks
    root_10_tasks: tl.constexpr = tl.cdiv(H, _BLOCK_SIZE_29)
    root_11_offset: tl.constexpr = root_10_offset + root_10_tasks
    root_11_tasks: tl.constexpr = tl.cdiv(PLE, _BLOCK_SIZE_31)
    root_12_offset: tl.constexpr = root_11_offset + root_11_tasks
    root_12_tasks: tl.constexpr = tl.cdiv(H, _BLOCK_SIZE_34)
    root_13_offset: tl.constexpr = root_12_offset + root_12_tasks
    root_13_tasks: tl.constexpr = tl.cdiv(H, _BLOCK_SIZE_37)

    if worker == 0:
        tile_dependency_root_0(
            RDIM,
            hidden_states,
            eps,
            input_norm_weight,
            input_norm,
        )
        _probe_publish(
            phase_arrivals,
            phase_ready,
            phase_input,
            epoch,
            1,
            FUSED_SIGNALS,
        )

    root_1_participants: tl.constexpr = min(TOTAL_WORKERS, root_1_tasks)
    if worker < root_1_participants:
        _probe_wait_phase(
            phase_arrivals,
            phase_ready,
            phase_input,
            epoch,
            1,
            FUSED_SIGNALS,
            POLL_DELAY,
            RELAXED_POLL,
        )
        for task in tl.range(worker, root_1_tasks, TOTAL_WORKERS):
            tile_dependency_root_1(
                input_norm,
                qkv_weight,
                projected_qkv,
                root_1_offset + task,
            )
            if QKV_CONTINUATION:
                tl.debug_barrier()
                tiles_per_head: tl.constexpr = HEAD_DIM // _BLOCK_SIZE_3
                head = task // tiles_per_head
                previous = tl.atomic_add(
                    head_arrivals + head,
                    1,
                    sem="acq_rel",
                    scope="gpu",
                )
                if previous % tiles_per_head == tiles_per_head - 1:
                    tile_dependency_root_2(
                        eps,
                        projected_qkv,
                        q_norm_weight,
                        k_norm_weight,
                        slot_mapping,
                        kv_cache,
                        position,
                        cos_sin,
                        root_2_offset + head,
                    )
                    tl.debug_barrier()
                    if head < Q_HEADS:
                        group = head // q_per_kv
                    elif head < Q_HEADS + KV_HEADS:
                        group = head - Q_HEADS
                    else:
                        group = head - Q_HEADS - KV_HEADS
                    group_count: tl.constexpr = q_per_kv + 2
                    group_previous = tl.atomic_add(
                        group_arrivals + group,
                        1,
                        sem="acq_rel",
                        scope="gpu",
                    )
                    if group_previous % group_count == group_count - 1:
                        tl.atomic_xchg(
                            group_ready + group,
                            epoch,
                            sem="release",
                            scope="gpu",
                        )
        if not QKV_CONTINUATION:
            _probe_publish(
                phase_arrivals,
                phase_ready,
                phase_qkv,
                epoch,
                root_1_participants,
                FUSED_SIGNALS,
            )

    if not QKV_CONTINUATION:
        if worker < root_2_tasks:
            _probe_wait_phase(
                phase_arrivals,
                phase_ready,
                phase_qkv,
                epoch,
                root_1_participants,
                FUSED_SIGNALS,
                POLL_DELAY,
                RELAXED_POLL,
            )
            tile_dependency_root_2(
                eps,
                projected_qkv,
                q_norm_weight,
                k_norm_weight,
                slot_mapping,
                kv_cache,
                position,
                cos_sin,
                root_2_offset + worker,
            )
            _probe_publish(
                phase_arrivals,
                phase_ready,
                phase_heads,
                epoch,
                root_2_tasks,
                FUSED_SIGNALS,
            )

    root_3_participants: tl.constexpr = min(TOTAL_WORKERS, root_3_tasks)
    if worker < root_3_participants:
        for task in tl.range(worker, root_3_tasks, TOTAL_WORKERS):
            group = task // SPLITS
            if QKV_CONTINUATION:
                _probe_wait(
                    group_ready + group,
                    epoch,
                    POLL_DELAY,
                    FUSED_SIGNALS != 2,
                    RELAXED_POLL,
                )
            else:
                _probe_wait_phase(
                    phase_arrivals,
                    phase_ready,
                    phase_heads,
                    epoch,
                    root_2_tasks,
                    FUSED_SIGNALS,
                    POLL_DELAY,
                    RELAXED_POLL,
                )
            tile_dependency_root_3(
                HEAD_DIM,
                kv_cache,
                projected_qkv,
                block_table,
                partial_out,
                partial_lse,
                root_3_offset + task,
            )
            if ATTENTION_CONTINUATION:
                tl.debug_barrier()
                previous = tl.atomic_add(
                    attention_arrivals + group,
                    1,
                    sem="acq_rel",
                    scope="gpu",
                )
                if previous % SPLITS == SPLITS - 1:
                    for query in range(q_per_kv):
                        merge_task = group + KV_HEADS * query
                        tile_dependency_root_4(
                            HEAD_DIM,
                            partial_out,
                            partial_lse,
                            attention,
                            root_4_offset + merge_task,
                        )
                    _probe_publish(
                        phase_arrivals,
                        phase_ready,
                        phase_merged,
                        epoch,
                        KV_HEADS,
                        FUSED_SIGNALS,
                    )
        if not ATTENTION_CONTINUATION:
            _probe_publish(
                phase_arrivals,
                phase_ready,
                phase_attention,
                epoch,
                root_3_participants,
                FUSED_SIGNALS,
            )

    if not ATTENTION_CONTINUATION:
        if worker < root_4_tasks:
            _probe_wait_phase(
                phase_arrivals,
                phase_ready,
                phase_attention,
                epoch,
                root_3_participants,
                FUSED_SIGNALS,
                POLL_DELAY,
                RELAXED_POLL,
            )
            tile_dependency_root_4(
                HEAD_DIM,
                partial_out,
                partial_lse,
                attention,
                root_4_offset + worker,
            )
            _probe_publish(
                phase_arrivals,
                phase_ready,
                phase_merged,
                epoch,
                root_4_tasks,
                FUSED_SIGNALS,
            )

    root_5_participants: tl.constexpr = min(TOTAL_WORKERS, root_5_tasks)
    if worker < root_5_participants:
        merged_publishers: tl.constexpr = (
            KV_HEADS if ATTENTION_CONTINUATION else root_4_tasks
        )
        _probe_wait_phase(
            phase_arrivals,
            phase_ready,
            phase_merged,
            epoch,
            merged_publishers,
            FUSED_SIGNALS,
            POLL_DELAY,
            RELAXED_POLL,
        )
        for task in tl.range(worker, root_5_tasks, TOTAL_WORKERS):
            tile_dependency_root_5(
                attention,
                o_weight,
                attention_out,
                root_5_offset + task,
            )
        if O_CONTINUATION:
            tl.debug_barrier()
            previous = tl.atomic_add(
                phase_arrivals + phase_o * 32,
                1,
                sem="acq_rel",
                scope="gpu",
            )
            if previous % root_5_participants == root_5_participants - 1:
                tile_dependency_root_6(
                    RDIM,
                    hidden_states,
                    eps,
                    attention_out,
                    post_attention_norm_weight,
                    residual,
                    pre_ff_norm_weight,
                    ff_input,
                )
                _probe_publish(
                    phase_arrivals,
                    phase_ready,
                    phase_post_attention,
                    epoch,
                    1,
                    FUSED_SIGNALS,
                )
        else:
            _probe_publish(
                phase_arrivals,
                phase_ready,
                phase_o,
                epoch,
                root_5_participants,
                FUSED_SIGNALS,
            )

    root_6_worker: tl.constexpr = (
        root_5_participants if SPREAD_NARROW_ROOTS else 0
    )
    if not O_CONTINUATION and worker == root_6_worker:
        _probe_wait_phase(
            phase_arrivals,
            phase_ready,
            phase_o,
            epoch,
            root_5_participants,
            FUSED_SIGNALS,
            POLL_DELAY,
            RELAXED_POLL,
        )
        tile_dependency_root_6(
            RDIM,
            hidden_states,
            eps,
            attention_out,
            post_attention_norm_weight,
            residual,
            pre_ff_norm_weight,
            ff_input,
        )
        _probe_publish(
            phase_arrivals,
            phase_ready,
            phase_post_attention,
            epoch,
            1,
            FUSED_SIGNALS,
        )

    root_7_participants: tl.constexpr = min(TOTAL_WORKERS, root_7_tasks)
    root_8_participants: tl.constexpr = min(TOTAL_WORKERS, root_8_tasks)
    if FFN_STREAM:
        stream_fan_in: tl.constexpr = 2 * _BLOCK_SIZE_24 // _BLOCK_SIZE_21
        initial_tasks: tl.constexpr = FFN_FIRST_GROUPS * stream_fan_in
        producer_workers: tl.constexpr = TOTAL_WORKERS - FFN_CONSUMER_WORKERS
        tail_tasks: tl.constexpr = root_7_tasks - initial_tasks
        tail_producer_workers: tl.constexpr = min(
            producer_workers,
            tail_tasks,
        )
        if worker < TOTAL_WORKERS:
            _probe_wait_phase(
                phase_arrivals,
                phase_ready,
                phase_post_attention,
                epoch,
                1,
                FUSED_SIGNALS,
                POLL_DELAY,
                RELAXED_POLL,
            )
            if TRACE:
                tl.store(trace + 2 + 4 * worker, _probe_globaltimer())
            for logical_task in tl.range(worker, initial_tasks, TOTAL_WORKERS):
                _probe_stream_ffn_producer(
                    ff_input,
                    gate_up_weight,
                    gate_up,
                    activation,
                    activation_arrivals,
                    ffn_split_arrivals,
                    ffn_ready,
                    trace,
                    logical_task,
                    epoch,
                    INTERMEDIATE,
                    root_7_offset,
                    root_8_offset,
                    root_8_tasks,
                    FFN_FIRST_GROUPS,
                    not FFN_SCHEDULED_ACTIVATION,
                    COUNTED_EVENT_ON_READY,
                    TRACE,
                )
            if TRACE:
                tl.store(trace + 3 + 4 * worker, _probe_globaltimer())
            if worker < producer_workers:
                for logical_task in tl.range(
                    initial_tasks + worker,
                    root_7_tasks,
                    producer_workers,
                ):
                    _probe_stream_ffn_producer(
                        ff_input,
                        gate_up_weight,
                        gate_up,
                        activation,
                        activation_arrivals,
                        ffn_split_arrivals,
                        ffn_ready,
                        trace,
                        logical_task,
                        epoch,
                        INTERMEDIATE,
                        root_7_offset,
                        root_8_offset,
                        root_8_tasks,
                        FFN_FIRST_GROUPS,
                        not FFN_SCHEDULED_ACTIVATION,
                        COUNTED_EVENT_ON_READY,
                        TRACE,
                    )
                if TRACE:
                    tl.store(trace + 4 + 4 * worker, _probe_globaltimer())

        if FFN_SCHEDULED_ACTIVATION:
            activation_worker_base: tl.constexpr = tail_producer_workers
            activation_workers: tl.constexpr = (
                TOTAL_WORKERS - activation_worker_base
            )
            if worker >= activation_worker_base:
                for activation_task in tl.range(
                    worker - activation_worker_base,
                    root_8_tasks,
                    activation_workers,
                ):
                    if FUSED_SIGNALS != 2:
                        _probe_wait_count(
                            activation_arrivals + activation_task,
                            epoch * stream_fan_in,
                            POLL_DELAY,
                            RELAXED_POLL,
                        )
                    tile_dependency_root_8(
                        gate_up,
                        activation,
                        root_8_offset + activation_task,
                    )
                    tl.debug_barrier()
                    split = tl.where(
                        activation_task < FFN_FIRST_GROUPS,
                        0,
                        1,
                    )
                    split_count = tl.where(
                        split == 0,
                        FFN_FIRST_GROUPS,
                        root_8_tasks - FFN_FIRST_GROUPS,
                    ).to(tl.int32)
                    if COUNTED_EVENT_ON_READY:
                        split_ready = _probe_counted_event_arrive(
                            ffn_split_arrivals + split,
                            split_count,
                        )
                    else:
                        split_previous = tl.atomic_add(
                            ffn_split_arrivals + split,
                            1,
                            sem="acq_rel",
                            scope="gpu",
                        )
                        split_ready = (
                            split_previous % split_count == split_count - 1
                        )
                    if split_ready:
                        tl.atomic_xchg(
                            ffn_ready + split,
                            epoch,
                            sem="release",
                            scope="gpu",
                        )
                        if TRACE:
                            tl.store(trace + split, _probe_globaltimer())

    elif worker < root_7_participants:
        _probe_wait_phase(
            phase_arrivals,
            phase_ready,
            phase_post_attention,
            epoch,
            1,
            FUSED_SIGNALS,
            POLL_DELAY,
            RELAXED_POLL,
        )
        ffn_fan_in: tl.constexpr = 2 * _BLOCK_SIZE_24 // _BLOCK_SIZE_21
        for logical_task in tl.range(worker, root_7_tasks, TOTAL_WORKERS):
            task = logical_task
            if FFN_CONSUMER_MAJOR:
                subtiles: tl.constexpr = _BLOCK_SIZE_24 // _BLOCK_SIZE_21
                activation_task = logical_task // ffn_fan_in
                within_activation = logical_task % ffn_fan_in
                half_tasks: tl.constexpr = INTERMEDIATE // _BLOCK_SIZE_21
                task = tl.where(
                    within_activation < subtiles,
                    activation_task * subtiles + within_activation,
                    half_tasks
                    + activation_task * subtiles
                    + within_activation
                    - subtiles,
                )
            tile_dependency_root_7(
                ff_input,
                gate_up_weight,
                gate_up,
                root_7_offset + task,
            )
            if FFN_CONTINUATION:
                tl.debug_barrier()
                if not FFN_CONSUMER_MAJOR:
                    half_tasks: tl.constexpr = INTERMEDIATE // _BLOCK_SIZE_21
                    half_task = task % half_tasks
                    activation_task = (
                        half_task * _BLOCK_SIZE_21
                    ) // _BLOCK_SIZE_24
                previous = tl.atomic_add(
                    activation_arrivals + activation_task,
                    1,
                    sem="acq_rel",
                    scope="gpu",
                )
                if previous % ffn_fan_in == ffn_fan_in - 1:
                    tile_dependency_root_8(
                        gate_up,
                        activation,
                        root_8_offset + activation_task,
                    )
                    _probe_publish(
                        phase_arrivals,
                        phase_ready,
                        phase_activation,
                        epoch,
                        root_8_tasks,
                        FUSED_SIGNALS,
                    )
        if not FFN_CONTINUATION:
            _probe_publish(
                phase_arrivals,
                phase_ready,
                phase_gate,
                epoch,
                root_7_participants,
                FUSED_SIGNALS,
            )

    if not FFN_CONTINUATION:
        if worker < root_8_participants:
            _probe_wait_phase(
                phase_arrivals,
                phase_ready,
                phase_gate,
                epoch,
                root_7_participants,
                FUSED_SIGNALS,
                POLL_DELAY,
                RELAXED_POLL,
            )
            for task in tl.range(worker, root_8_tasks, TOTAL_WORKERS):
                tile_dependency_root_8(
                    gate_up,
                    activation,
                    root_8_offset + task,
                )
            _probe_publish(
                phase_arrivals,
                phase_ready,
                phase_activation,
                epoch,
                root_8_participants,
                FUSED_SIGNALS,
            )

    root_9_participants: tl.constexpr = min(TOTAL_WORKERS, root_9_tasks)
    if FFN_STREAM:
        consumer_base: tl.constexpr = TOTAL_WORKERS - FFN_CONSUMER_WORKERS
        if worker >= consumer_base:
            if TRACE:
                tl.store(trace + 4 + 4 * worker, _probe_globaltimer())
            for task in tl.range(
                worker - consumer_base,
                root_9_tasks,
                FFN_CONSUMER_WORKERS,
            ):
                _probe_down_two_splits(
                    activation,
                    down_weight,
                    down,
                    ffn_ready,
                    task,
                    epoch,
                    H,
                    INTERMEDIATE,
                    FFN_FIRST_GROUPS,
                    STREAM_DOWN_STAGES,
                    STREAM_DOWN_UNROLL,
                    POLL_DELAY,
                    FUSED_SIGNALS != 2,
                    RELAXED_POLL,
                )
            if TRACE:
                tl.store(trace + 5 + 4 * worker, _probe_globaltimer())
            _probe_publish(
                phase_arrivals,
                phase_ready,
                phase_down,
                epoch,
                FFN_CONSUMER_WORKERS,
                FUSED_SIGNALS,
            )
    else:
        if worker < root_9_participants:
            activation_publishers: tl.constexpr = (
                root_8_tasks if FFN_CONTINUATION else root_8_participants
            )
            _probe_wait_phase(
                phase_arrivals,
                phase_ready,
                phase_activation,
                epoch,
                activation_publishers,
                FUSED_SIGNALS,
                POLL_DELAY,
                RELAXED_POLL,
            )
            for task in tl.range(worker, root_9_tasks, TOTAL_WORKERS):
                tile_dependency_root_9(
                    activation,
                    down_weight,
                    down,
                    root_9_offset + task,
                )
            _probe_publish(
                phase_arrivals,
                phase_ready,
                phase_down,
                epoch,
                root_9_participants,
                FUSED_SIGNALS,
            )

    root_10_participants: tl.constexpr = min(TOTAL_WORKERS, root_10_tasks)
    if worker < root_10_participants:
        down_publishers: tl.constexpr = (
            FFN_CONSUMER_WORKERS if FFN_STREAM else root_9_participants
        )
        _probe_wait_phase(
            phase_arrivals,
            phase_ready,
            phase_down,
            epoch,
            down_publishers,
            FUSED_SIGNALS,
            POLL_DELAY,
            RELAXED_POLL,
        )
        for task in tl.range(worker, root_10_tasks, TOTAL_WORKERS):
            tile_dependency_root_10(
                RDIM,
                eps,
                residual,
                down,
                post_ff_norm_weight,
                hidden,
                root_10_offset + task,
            )
        _probe_publish(
            phase_arrivals,
            phase_ready,
            phase_post_ff,
            epoch,
            root_10_participants,
            FUSED_SIGNALS,
        )

    root_11_participants: tl.constexpr = min(TOTAL_WORKERS, root_11_tasks)
    if worker < root_11_participants:
        _probe_wait_phase(
            phase_arrivals,
            phase_ready,
            phase_post_ff,
            epoch,
            root_10_participants,
            FUSED_SIGNALS,
            POLL_DELAY,
            RELAXED_POLL,
        )
        for task in tl.range(worker, root_11_tasks, TOTAL_WORKERS):
            tile_dependency_root_11(
                hidden,
                ple_gate_weight,
                per_layer_input,
                ple_input,
                root_11_offset + task,
            )
        _probe_publish(
            phase_arrivals,
            phase_ready,
            phase_ple_gate,
            epoch,
            root_11_participants,
            FUSED_SIGNALS,
        )

    root_12_participants: tl.constexpr = min(TOTAL_WORKERS, root_12_tasks)
    if worker < root_12_participants:
        _probe_wait_phase(
            phase_arrivals,
            phase_ready,
            phase_ple_gate,
            epoch,
            root_11_participants,
            FUSED_SIGNALS,
            POLL_DELAY,
            RELAXED_POLL,
        )
        for task in tl.range(worker, root_12_tasks, TOTAL_WORKERS):
            tile_dependency_root_12(
                ple_input,
                ple_projection_weight,
                ple_projection,
                root_12_offset + task,
            )
        _probe_publish(
            phase_arrivals,
            phase_ready,
            phase_ple_projection,
            epoch,
            root_12_participants,
            FUSED_SIGNALS,
        )

    root_13_participants: tl.constexpr = min(TOTAL_WORKERS, root_13_tasks)
    root_13_base: tl.constexpr = (
        root_12_participants if SPREAD_NARROW_ROOTS else 0
    )
    root_13_worker = worker - root_13_base
    if worker >= root_13_base and root_13_worker < root_13_participants:
        _probe_wait_phase(
            phase_arrivals,
            phase_ready,
            phase_ple_projection,
            epoch,
            root_12_participants,
            FUSED_SIGNALS,
            POLL_DELAY,
            RELAXED_POLL,
        )
        for task in tl.range(root_13_worker, root_13_tasks, TOTAL_WORKERS):
            tile_dependency_root_13(
                RDIM,
                eps,
                hidden,
                ple_projection,
                layer_scalar,
                post_ple_norm_weight,
                output,
                root_13_offset + task,
            )
        _probe_publish(
            phase_arrivals,
            phase_ready,
            phase_final,
            epoch,
            root_13_participants,
            FUSED_SIGNALS,
        )

    tl.store(worker_epoch + worker, epoch)
"""


def _config_for_probe(bound, args, geometry):
    base_args = argparse.Namespace(
        config_mode="matched",
        config_path=args.config_path,
        attention_block=args.attention_block,
        full_splits=args.full_splits,
        sliding_splits=args.sliding_splits,
        worker_multiplier=2,
        cross_loop_workers=None,
        num_warps=args.num_warps,
        kernel_stages=args.kernel_stages,
    )
    config = mega._megakernel_config(bound, base_args, geometry)
    values = dict(config)
    overrides = {
        3: args.qkv_block_n,
        4: args.qkv_block_k,
        10: args.attention_q_block,
        17: args.o_block_n,
        18: args.o_block_k,
        21: args.gate_block_n,
        22: args.gate_block_k,
        24: args.activation_block,
        26: args.down_block_n,
        27: args.down_block_k,
        31: args.ple_gate_block_n,
        32: args.ple_gate_block_k,
        34: args.ple_projection_block_n,
        35: args.ple_projection_block_k,
    }
    values["block_sizes"] = [
        overrides.get(spec.block_id) or value
        for spec, value in zip(
            bound.config_spec.block_sizes,
            values["block_sizes"],
            strict=True,
        )
    ]
    if args.match_gate_eviction:
        for fact in bound.config_spec.memory_op_facts:
            if (
                fact.eviction_index is not None
                and fact.tensor_name is not None
                and "gate_up_weight" in fact.tensor_name
            ):
                values["load_eviction_policies"][fact.eviction_index] = "first"
    if args.disable_gate_warp_specialize:
        values["range_warp_specializes"] = [
            False if tuple(spec.block_ids) == (22,) else value
            for spec, value in zip(
                bound.config_spec.range_warp_specialize,
                values["range_warp_specializes"],
                strict=True,
            )
        ]
    if args.gate_unroll_factor is not None:
        values["range_unroll_factors"] = [
            args.gate_unroll_factor if tuple(spec.block_ids) == (22,) else value
            for spec, value in zip(
                bound.config_spec.range_unroll_factors,
                values["range_unroll_factors"],
                strict=True,
            )
        ]
    if args.gate_range_stages is not None:
        values["range_num_stages"] = [
            args.gate_range_stages if tuple(spec.block_ids) == (22,) else value
            for spec, value in zip(
                bound.config_spec.range_num_stages,
                values["range_num_stages"],
                strict=True,
            )
        ]
    if args.qkv_range_stages is not None:
        values["range_num_stages"] = [
            args.qkv_range_stages if tuple(spec.block_ids) == (4,) else value
            for spec, value in zip(
                bound.config_spec.range_num_stages,
                values["range_num_stages"],
                strict=True,
            )
        ]
    if args.qkv_unroll_factor is not None:
        values["range_unroll_factors"] = [
            args.qkv_unroll_factor if tuple(spec.block_ids) == (4,) else value
            for spec, value in zip(
                bound.config_spec.range_unroll_factors,
                values["range_unroll_factors"],
                strict=True,
            )
        ]
    if args.o_range_stages is not None:
        values["range_num_stages"] = [
            args.o_range_stages if tuple(spec.block_ids) == (18,) else value
            for spec, value in zip(
                bound.config_spec.range_num_stages,
                values["range_num_stages"],
                strict=True,
            )
        ]
    if args.o_unroll_factor is not None:
        values["range_unroll_factors"] = [
            args.o_unroll_factor if tuple(spec.block_ids) == (18,) else value
            for spec, value in zip(
                bound.config_spec.range_unroll_factors,
                values["range_unroll_factors"],
                strict=True,
            )
        ]
    if args.down_range_stages is not None:
        values["range_num_stages"] = [
            args.down_range_stages if tuple(spec.block_ids) == (27,) else value
            for spec, value in zip(
                bound.config_spec.range_num_stages,
                values["range_num_stages"],
                strict=True,
            )
        ]
    if args.down_unroll_factor is not None:
        values["range_unroll_factors"] = [
            args.down_unroll_factor if tuple(spec.block_ids) == (27,) else value
            for spec, value in zip(
                bound.config_spec.range_unroll_factors,
                values["range_unroll_factors"],
                strict=True,
            )
        ]
    if args.attention_range_stages is not None:
        values["range_num_stages"] = [
            args.attention_range_stages if tuple(spec.block_ids) == (12,) else value
            for spec, value in zip(
                bound.config_spec.range_num_stages,
                values["range_num_stages"],
                strict=True,
            )
        ]
    if args.match_effective_standalone_ranges:
        standalone_reduction_blocks = {(4,), (18,), (22,), (27,), (32,)}
        values["range_unroll_factors"] = [
            0 if tuple(spec.block_ids) in standalone_reduction_blocks else value
            for spec, value in zip(
                bound.config_spec.range_unroll_factors,
                values["range_unroll_factors"],
                strict=True,
            )
        ]
        values["range_num_stages"] = [
            0 if tuple(spec.block_ids) in standalone_reduction_blocks else value
            for spec, value in zip(
                bound.config_spec.range_num_stages,
                values["range_num_stages"],
                strict=True,
            )
        ]
    if args.match_standalone_eviction:
        policy_by_tensor = {
            "input_norm": "first",
            "qkv_weight": "first",
            "attention_flat": "last",
            "o_weight": "first",
            "gate_up_weight": "first",
            "activation": "first",
        }
        first_hidden_index = min(
            fact.eviction_index
            for fact in bound.config_spec.memory_op_facts
            if fact.eviction_index is not None and fact.tensor_name == "hidden"
        )
        for fact in bound.config_spec.memory_op_facts:
            if fact.eviction_index is None:
                continue
            policy = policy_by_tensor.get(fact.tensor_name)
            if fact.eviction_index == first_hidden_index:
                policy = "first"
            if policy is not None:
                values["load_eviction_policies"][fact.eviction_index] = policy
    config = helion.Config.from_dict(values)
    bound.config_spec.normalize(config.config)
    return config


class _SubstituteInlinedNames(ast.NodeTransformer):
    def __init__(self, arguments, local_names, prefix) -> None:
        self.arguments = arguments
        self.local_names = local_names
        self.prefix = prefix

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load) and node.id in self.arguments:
            return ast.copy_location(copy.deepcopy(self.arguments[node.id]), node)
        if node.id in self.local_names:
            return ast.copy_location(
                ast.Name(id=f"{self.prefix}{node.id}", ctx=node.ctx),
                node,
            )
        return node


class _InlineCallsInFunction(ast.NodeTransformer):
    def __init__(self, caller_name, callee) -> None:
        self.caller_name = caller_name
        self.callee = callee
        self.call_index = 0
        self.inside_caller = False
        self.parameters = [argument.arg for argument in callee.args.args]
        self.local_names = {
            node.id
            for statement in callee.body
            for node in ast.walk(statement)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
        }

    def visit_FunctionDef(self, node):
        if node.name != self.caller_name:
            return node
        was_inside = self.inside_caller
        self.inside_caller = True
        node = self.generic_visit(node)
        self.inside_caller = was_inside
        return node

    def visit_Expr(self, node):
        if not self.inside_caller:
            return node
        call = node.value
        if not (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id == self.callee.name
        ):
            return self.generic_visit(node)
        if call.keywords or len(call.args) != len(self.parameters):
            raise AssertionError(f"unexpected call shape for {self.callee.name}")
        self.call_index += 1
        arguments = dict(zip(self.parameters, call.args, strict=True))
        substituter = _SubstituteInlinedNames(
            arguments,
            self.local_names,
            f"_inlined_{self.callee.name}_{self.call_index}_",
        )
        return [
            substituter.visit(copy.deepcopy(statement))
            for statement in self.callee.body
        ]


def _inline_generated_helper(source: str, caller_name: str, callee_name: str) -> str:
    module = ast.parse(source)
    callee = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == callee_name
    )
    caller = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == caller_name
    )
    _InlineCallsInFunction(caller_name, callee).visit(caller)
    ast.fix_missing_locations(module)
    return ast.unparse(module) + "\n"


def _override_generated_range_keyword(
    source: str,
    function_name: str,
    loop_variable: str,
    keyword: str,
    value: int,
) -> str:
    """Override one emitted tl.range keyword for a generated-root probe."""
    lines = source.splitlines(keepends=True)
    function_start = next(
        index
        for index, line in enumerate(lines)
        if line.startswith(f"def {function_name}(")
    )
    function_end = next(
        (
            index
            for index in range(function_start + 1, len(lines))
            if lines[index].startswith("@triton.jit")
        ),
        len(lines),
    )
    matching_lines = [
        index
        for index in range(function_start, function_end)
        if f"for {loop_variable} in tl.range(" in lines[index]
    ]
    if len(matching_lines) != 1:
        raise AssertionError(
            f"expected one {loop_variable} range in {function_name}, "
            f"found {len(matching_lines)}"
        )
    index = matching_lines[0]
    pattern = rf"{re.escape(keyword)}=[^,)]+"
    if re.search(pattern, lines[index]):
        lines[index] = re.sub(pattern, f"{keyword}={value}", lines[index], count=1)
    else:
        lines[index] = lines[index].replace(
            "):\n",
            f", {keyword}={value}):\n",
            1,
        )
    return "".join(lines)


def _generated_root_namespace(bound, config, args):
    lowered = bound.to_triton_code(config, output_origin_lines=True)
    module = ast.parse(lowered)
    master = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("_helion_gemma4")
    )
    lines = lowered.splitlines(keepends=True)
    decorator_line = master.lineno - 1
    while decorator_line > 0 and lines[decorator_line - 1].lstrip().startswith("@"):
        decorator_line -= 1
    prefix = "".join(lines[:decorator_line])
    source = prefix + "\n" + textwrap.dedent(SCHEDULER_SOURCE)
    if args.force_emitted_gate_stages is not None:
        source = _override_generated_range_keyword(
            source,
            "tile_dependency_root_7",
            "offset_22",
            "num_stages",
            args.force_emitted_gate_stages,
        )
    if args.gate_root_noinline:
        source = source.replace(
            "@triton.jit\ndef tile_dependency_root_7(",
            "@triton.jit(noinline=True)\ndef tile_dependency_root_7(",
            1,
        )
    if args.inline_gate_body:
        source = _inline_generated_helper(
            source,
            "gemma4_codegen_gate_probe",
            "tile_dependency_root_7",
        )
    filename = str(Path(__file__).with_name("_generated_gemma4_schedule_probe.py"))
    linecache.cache[filename] = (
        len(source),
        None,
        source.splitlines(keepends=True),
        filename,
    )
    namespace: dict[str, object] = {"__name__": "_generated_gemma4_schedule_probe"}
    exec(compile(source, filename, "exec"), namespace)
    return namespace, source


def _config_block_size(bound, config, block_id: int) -> int:
    values = dict(config)["block_sizes"]
    return next(
        value
        for spec, value in zip(
            bound.config_spec.block_sizes,
            values,
            strict=True,
        )
        if spec.block_id == block_id
    )


def _validate_ffn_stream(bound, config, shape, args) -> None:
    if args.ffn_scheduled_activation and not args.ffn_stream:
        raise ValueError("--ffn-scheduled-activation requires --ffn-stream")
    if not args.ffn_stream:
        return

    gate_block = _config_block_size(bound, config, 21)
    activation_block = _config_block_size(bound, config, 24)
    down_block_n = _config_block_size(bound, config, 26)
    down_block_k = _config_block_size(bound, config, 27)
    if shape.intermediate % gate_block:
        raise ValueError("FFN streaming requires exact gate/up tiles")
    if shape.intermediate % activation_block:
        raise ValueError("FFN streaming requires exact activation tiles")
    if activation_block % gate_block:
        raise ValueError("activation tiles must contain whole gate/up tiles")
    if shape.hidden % down_block_n:
        raise ValueError("FFN streaming requires exact down output tiles")

    activation_tasks = shape.intermediate // activation_block
    if not 0 < args.ffn_first_groups < activation_tasks:
        raise ValueError("--ffn-first-groups must split the activation domain")
    first_k = args.ffn_first_groups * activation_block
    if first_k % down_block_k or (shape.intermediate - first_k) % down_block_k:
        raise ValueError("both streamed down K regions must contain whole K tiles")

    fan_in = 2 * activation_block // gate_block
    initial_tasks = args.ffn_first_groups * fan_in
    down_tasks = shape.hidden // down_block_n
    if not 0 < args.ffn_consumer_workers <= min(down_tasks, args.workers - 1):
        raise ValueError(
            "--ffn-consumer-workers must be positive and no larger than "
            "the down task count or workers - 1"
        )
    producer_workers = args.workers - args.ffn_consumer_workers
    if producer_workers <= 0:
        raise ValueError("FFN streaming requires at least one tail producer worker")
    gate_tasks = 2 * shape.intermediate // gate_block
    if gate_tasks < initial_tasks:
        raise ValueError("the streamed prefix exceeds the producer task domain")


def _allocate_buffers(tensors, shape, geometry, splits, workers, activation_block):
    device = tensors["hidden_states"].device
    q_width = shape.q_heads * geometry.head_dim
    qkv_width = q_width + 2 * shape.kv_heads * geometry.head_dim
    return {
        "input_norm": torch.empty(
            (1, shape.hidden), device=device, dtype=torch.bfloat16
        ),
        "projected_qkv": torch.empty(
            (1, qkv_width), device=device, dtype=torch.bfloat16
        ),
        "partial_out": torch.empty(
            (
                splits,
                shape.kv_heads,
                shape.q_heads // shape.kv_heads,
                geometry.head_dim,
            ),
            device=device,
            dtype=torch.float32,
        ),
        "partial_lse": torch.empty(
            (splits, shape.kv_heads, shape.q_heads // shape.kv_heads),
            device=device,
            dtype=torch.float32,
        ),
        "attention": torch.empty(
            (shape.kv_heads, shape.q_heads // shape.kv_heads, geometry.head_dim),
            device=device,
            dtype=torch.bfloat16,
        ),
        "attention_out": torch.empty(
            (1, shape.hidden), device=device, dtype=torch.bfloat16
        ),
        "residual": torch.empty((1, shape.hidden), device=device, dtype=torch.bfloat16),
        "ff_input": torch.empty((1, shape.hidden), device=device, dtype=torch.bfloat16),
        "gate_up": torch.empty(
            (1, 2 * shape.intermediate), device=device, dtype=torch.bfloat16
        ),
        "activation": torch.empty(
            (1, shape.intermediate), device=device, dtype=torch.bfloat16
        ),
        "down": torch.empty((1, shape.hidden), device=device, dtype=torch.bfloat16),
        "hidden": torch.empty((1, shape.hidden), device=device, dtype=torch.bfloat16),
        "ple_input": torch.empty((1, shape.ple), device=device, dtype=torch.bfloat16),
        "ple_projection": torch.empty(
            (1, shape.hidden), device=device, dtype=torch.bfloat16
        ),
        "output": torch.empty((1, shape.hidden), device=device, dtype=torch.bfloat16),
        "worker_epoch": torch.zeros(workers, device=device, dtype=torch.int32),
        "phase_arrivals": torch.zeros(14 * 32, device=device, dtype=torch.int32),
        "phase_ready": torch.zeros(14 * 32, device=device, dtype=torch.int32),
        "head_arrivals": torch.zeros(
            shape.q_heads + 2 * shape.kv_heads, device=device, dtype=torch.int32
        ),
        "group_arrivals": torch.zeros(shape.kv_heads, device=device, dtype=torch.int32),
        "group_ready": torch.zeros(shape.kv_heads, device=device, dtype=torch.int32),
        "attention_arrivals": torch.zeros(
            shape.kv_heads, device=device, dtype=torch.int32
        ),
        "activation_arrivals": torch.zeros(
            shape.intermediate // activation_block,
            device=device,
            dtype=torch.int32,
        ),
        "ffn_split_arrivals": torch.zeros(2, device=device, dtype=torch.int32),
        "ffn_ready": torch.zeros(2, device=device, dtype=torch.int32),
        "trace": torch.zeros(2 + 4 * workers, device=device, dtype=torch.int64),
    }


def _launch(
    kernel,
    tensors,
    buffers,
    shape,
    geometry,
    splits,
    args,
    *,
    counted_event_on_ready=None,
    scheduled_activation=None,
):
    kernel_args = (
        tensors["hidden_states"],
        tensors["input_norm_weight"],
        buffers["input_norm"],
        tensors["qkv_weight"],
        buffers["projected_qkv"],
        tensors["q_norm_weight"],
        tensors["k_norm_weight"],
        tensors["slot_mapping"],
        tensors["kv_cache"],
        tensors["position"],
        tensors["cos_sin"],
        tensors["block_table"],
        buffers["partial_out"],
        buffers["partial_lse"],
        buffers["attention"],
        tensors["o_weight"],
        buffers["attention_out"],
        tensors["post_attention_norm_weight"],
        buffers["residual"],
        tensors["pre_ff_norm_weight"],
        buffers["ff_input"],
        tensors["gate_up_weight"],
        buffers["gate_up"],
        buffers["activation"],
        tensors["down_weight"],
        buffers["down"],
        tensors["post_ff_norm_weight"],
        buffers["hidden"],
        tensors["ple_gate_weight"],
        tensors["per_layer_input"],
        buffers["ple_input"],
        tensors["ple_proj_weight"],
        buffers["ple_projection"],
        tensors["layer_scalar"],
        tensors["post_ple_norm_weight"],
        buffers["output"],
        buffers["worker_epoch"],
        buffers["phase_arrivals"],
        buffers["phase_ready"],
        buffers["head_arrivals"],
        buffers["group_arrivals"],
        buffers["group_ready"],
        buffers["attention_arrivals"],
        buffers["activation_arrivals"],
        buffers["ffn_split_arrivals"],
        buffers["ffn_ready"],
        buffers["trace"],
        shape.eps,
    )
    launch_options = {"num_warps": args.num_warps, "num_stages": args.kernel_stages}
    if args.maxnreg:
        launch_options["maxnreg"] = args.maxnreg
    return kernel[(args.workers,)](
        *kernel_args,
        args.workers,
        shape.hidden,
        shape.intermediate,
        shape.ple,
        shape.q_heads,
        shape.kv_heads,
        geometry.head_dim,
        splits,
        4096,
        args.qkv_continuation,
        args.attention_continuation,
        args.ffn_continuation or args.ffn_stream,
        args.ffn_consumer_major,
        args.ffn_stream,
        (
            args.ffn_scheduled_activation
            if scheduled_activation is None
            else scheduled_activation
        ),
        args.ffn_first_groups,
        args.ffn_consumer_workers,
        (
            args.counted_event_on_ready
            if counted_event_on_ready is None
            else counted_event_on_ready
        ),
        args.stream_down_stages,
        args.stream_down_unroll,
        args.o_continuation,
        args.poll_delay,
        args.trace,
        2 if args.no_waits else args.fused_signals,
        args.spread_narrow_roots,
        args.relaxed_poll,
        **launch_options,
    )


def _launch_gate(kernel, tensors, buffers, reference, shape, geometry, splits, args):
    launch_options = {"num_warps": args.num_warps, "num_stages": args.kernel_stages}
    if args.maxnreg:
        launch_options["maxnreg"] = args.maxnreg
    return kernel[(args.workers,)](
        reference["ff_input"],
        tensors["gate_up_weight"],
        buffers["gate_up"],
        args.workers,
        shape.hidden,
        shape.intermediate,
        shape.q_heads,
        shape.kv_heads,
        geometry.head_dim,
        splits,
        args.gate_consumer_major,
        **launch_options,
    )


def _assert_close(name, actual, expected, *, atol=2e-1, rtol=8e-2):
    torch.testing.assert_close(actual.float(), expected.float(), atol=atol, rtol=rtol)
    maximum = float((actual.float() - expected.float()).abs().max().item())
    print(f"codegen_probe_correctness {name} max_abs={maximum:.6f}", flush=True)


def _stream_trace_result(buffers, shape, bound, config, args):
    trace = buffers["trace"].cpu()
    worker_trace = trace[2:].view(args.workers, 4)
    start_values = worker_trace[:, 0]
    start = int(start_values[start_values != 0].min().item())

    def interval(values):
        values = values[values != 0]
        return {
            "min_us": round((int(values.min().item()) - start) / 1000.0, 3),
            "max_us": round((int(values.max().item()) - start) / 1000.0, 3),
        }

    consumer_base = args.workers - args.ffn_consumer_workers
    return {
        "split_ready_us": [
            round((int(value.item()) - start) / 1000.0, 3) for value in trace[:2]
        ],
        "ffn_start": interval(worker_trace[:, 0]),
        "initial_producer_done": interval(worker_trace[:, 1]),
        "tail_producer_done": interval(worker_trace[:consumer_base, 2]),
        "down_begin": interval(worker_trace[consumer_base:, 2]),
        "down_done": interval(worker_trace[consumer_base:, 3]),
    }


def run(args) -> None:
    require_idle_visible_gpu()
    if args.trace and not args.ffn_stream:
        raise ValueError("--trace currently records only --ffn-stream schedules")
    shape = Gemma4E4BShape(context=args.context, block_size=args.block_size)
    geometry = shape.layer_geometry(args.layer)
    if geometry.layer_type != "sliding" or geometry.kv_shared:
        raise ValueError("the first exact-codegen probe supports layer 0 geometry")
    splits = args.sliding_splits
    tensors = allocate_layer(shape, geometry, args.seed)
    reference = layer_reference(tensors, shape, geometry)
    kernel_args = mega._megakernel_args(tensors, shape, geometry, splits)
    bound = mega.NONSHARED_MEGAKERNEL.bind(kernel_args)
    config = _config_for_probe(bound, args, geometry)
    _validate_ffn_stream(bound, config, shape, args)
    namespace, lowered = _generated_root_namespace(bound, config, args)
    Path(args.lowered_output).write_text(lowered)
    print(f"LOWERED_TRITON_PATH {args.lowered_output}", flush=True)
    buffers = _allocate_buffers(
        tensors,
        shape,
        geometry,
        splits,
        args.workers,
        args.activation_block or 256,
    )
    if args.gate_only:
        kernel = namespace["gemma4_codegen_gate_probe"]
        compiled = _launch_gate(
            kernel,
            tensors,
            buffers,
            reference,
            shape,
            geometry,
            splits,
            args,
        )
        torch.cuda.synchronize()
        _assert_close("gate_up", buffers["gate_up"], reference["gate_up"])
        result = {
            "device": torch.cuda.get_device_name(),
            "layer": args.layer,
            "config": dict(config),
            "schedule": {
                "gate_only": True,
                "workers": args.workers,
                "consumer_major": args.gate_consumer_major,
            },
            "resources": {
                "registers": compiled.n_regs,
                "spills": compiled.n_spills,
                "shared": compiled.metadata.shared,
            },
        }
        if args.print_lowered:
            print(lowered)
        if args.benchmark:
            graph, _ = capture(
                lambda: (
                    _launch_gate(
                        kernel,
                        tensors,
                        buffers,
                        reference,
                        shape,
                        geometry,
                        splits,
                        args,
                    ),
                    buffers["gate_up"],
                )[1]
            )
            pids = visible_gpu_pids()
            result["timings"] = benchmark_interleaved(
                {"generated_gate_root": graph.replay},
                args.repeats,
                args.batch_replays,
            )
            if visible_gpu_pids() != pids:
                raise RuntimeError("GPU process set changed during benchmark")
        print("RESULT_JSON", json.dumps(result, sort_keys=True), flush=True)
        return

    kernel = namespace["gemma4_codegen_schedule_probe"]
    compiled = _launch(kernel, tensors, buffers, shape, geometry, splits, args)
    torch.cuda.synchronize()
    alternate_counted_event = None
    alternate_compiled = None
    if args.compare_counted_event:
        alternate_counted_event = not args.counted_event_on_ready
        alternate_compiled = _launch(
            kernel,
            tensors,
            buffers,
            shape,
            geometry,
            splits,
            args,
            counted_event_on_ready=alternate_counted_event,
        )
        torch.cuda.synchronize()
    alternate_scheduled_activation = None
    alternate_scheduled_compiled = None
    if args.compare_scheduled_activation:
        alternate_scheduled_activation = not args.ffn_scheduled_activation
        alternate_scheduled_compiled = _launch(
            kernel,
            tensors,
            buffers,
            shape,
            geometry,
            splits,
            args,
            scheduled_activation=alternate_scheduled_activation,
        )
        torch.cuda.synchronize()

    if not args.no_waits:
        _assert_close(
            "query",
            buffers["projected_qkv"][:, : shape.q_heads * geometry.head_dim].view_as(
                reference["query"]
            ),
            reference["query"],
            atol=1.5e-1,
            rtol=6e-2,
        )
        _assert_close(
            "attention",
            buffers["attention"].view_as(reference["attention"]),
            reference["attention"],
        )
        _assert_close("output", buffers["output"], reference["output"])
        slot = int(tensors["slot_mapping"][0].item())
        cache_block = slot // shape.block_size
        cache_offset = slot % shape.block_size
        _assert_close(
            "kv_cache_slot",
            tensors["kv_cache"][cache_block, cache_offset],
            reference["kv_cache"][cache_block, cache_offset],
            atol=1.5e-1,
            rtol=6e-2,
        )

    result = {
        "device": torch.cuda.get_device_name(),
        "layer": args.layer,
        "config": dict(config),
        "schedule": {
            "workers": args.workers,
            "qkv_continuation": args.qkv_continuation,
            "attention_continuation": args.attention_continuation,
            "ffn_continuation": args.ffn_continuation,
            "ffn_consumer_major": args.ffn_consumer_major,
            "ffn_stream": args.ffn_stream,
            "ffn_scheduled_activation": args.ffn_scheduled_activation,
            "ffn_first_groups": args.ffn_first_groups,
            "ffn_consumer_workers": args.ffn_consumer_workers,
            "counted_event_on_ready": args.counted_event_on_ready,
            "stream_down_stages": args.stream_down_stages,
            "stream_down_unroll": args.stream_down_unroll,
            "o_continuation": args.o_continuation,
            "fused_signals": args.fused_signals,
            "no_waits": args.no_waits,
            "spread_narrow_roots": args.spread_narrow_roots,
            "relaxed_poll": args.relaxed_poll,
        },
        "resources": {
            "registers": compiled.n_regs,
            "spills": compiled.n_spills,
            "shared": compiled.metadata.shared,
        },
        "generated_root_count": sum(
            name.startswith("tile_dependency_root_") and "scheduled_task" not in name
            for name in namespace
        ),
    }
    resident_workers, resident_blocks_per_sm = resident_capacity(
        compiled, args.num_warps
    )
    result["resources"].update(
        {
            "resident_blocks_per_sm": resident_blocks_per_sm,
            "resident_workers": resident_workers,
        }
    )
    if alternate_compiled is not None:
        result["counted_event_comparison"] = {
            "alternate_counted_event_on_ready": alternate_counted_event,
            "alternate_resources": {
                "registers": alternate_compiled.n_regs,
                "spills": alternate_compiled.n_spills,
                "shared": alternate_compiled.metadata.shared,
            },
        }
    if alternate_scheduled_compiled is not None:
        result["scheduled_activation_comparison"] = {
            "alternate_scheduled_activation": alternate_scheduled_activation,
            "alternate_resources": {
                "registers": alternate_scheduled_compiled.n_regs,
                "spills": alternate_scheduled_compiled.n_spills,
                "shared": alternate_scheduled_compiled.metadata.shared,
            },
        }
    if args.trace:
        result["stream_trace"] = _stream_trace_result(
            buffers,
            shape,
            bound,
            config,
            args,
        )
    helion_graph = None
    if args.compare_helion:
        configs = json.loads(Path(args.config_path).read_text())
        baseline_args = argparse.Namespace(
            tune=[],
            full_splits=args.full_splits,
            sliding_splits=args.sliding_splits,
        )
        built = layer.build_layer(
            baseline_args,
            tensors,
            shape,
            geometry,
            configs,
            Path(args.config_path),
        )
        baseline_output = built["launch_optimized"]()
        torch.cuda.synchronize()
        _assert_close("separate_helion", baseline_output, reference["output"])
        helion_graph, _ = capture(built["launch_optimized"])
    if args.print_lowered:
        print(lowered)
    if args.benchmark:
        graph, _ = capture(
            lambda: (
                _launch(kernel, tensors, buffers, shape, geometry, splits, args),
                buffers["output"],
            )[1]
        )
        pids = visible_gpu_pids()
        entries = {"triton_codegen_schedule": graph.replay}
        if alternate_counted_event is not None:
            alternate_graph, _ = capture(
                lambda: (
                    _launch(
                        kernel,
                        tensors,
                        buffers,
                        shape,
                        geometry,
                        splits,
                        args,
                        counted_event_on_ready=alternate_counted_event,
                    ),
                    buffers["output"],
                )[1]
            )
            entries[
                "triton_counted_event"
                if alternate_counted_event
                else "triton_direct_event"
            ] = alternate_graph.replay
        if alternate_scheduled_activation is not None:
            alternate_scheduled_graph, _ = capture(
                lambda: (
                    _launch(
                        kernel,
                        tensors,
                        buffers,
                        shape,
                        geometry,
                        splits,
                        args,
                        scheduled_activation=alternate_scheduled_activation,
                    ),
                    buffers["output"],
                )[1]
            )
            entries[
                "triton_scheduled_activation"
                if alternate_scheduled_activation
                else "triton_immediate_activation"
            ] = alternate_scheduled_graph.replay
        if helion_graph is not None:
            entries["separate_helion"] = helion_graph.replay
        result["timings"] = benchmark_interleaved(
            entries, args.repeats, args.batch_replays
        )
        if visible_gpu_pids() != pids:
            raise RuntimeError("GPU process set changed during benchmark")
        if not args.no_waits:
            _assert_close(
                "output_after_replays",
                buffers["output"],
                reference["output"],
            )
    print("RESULT_JSON", json.dumps(result, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--context", type=int, default=8192)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--sliding-splits", type=int, default=16)
    parser.add_argument("--full-splits", type=int, default=64)
    parser.add_argument("--attention-block", type=int, default=32)
    parser.add_argument("--attention-q-block", type=int)
    parser.add_argument("--attention-range-stages", type=int)
    parser.add_argument("--workers", type=int, default=296)
    parser.add_argument("--num-warps", type=int, choices=(1, 2, 4), default=2)
    parser.add_argument("--kernel-stages", type=int, choices=(1, 2, 3, 4), default=4)
    parser.add_argument("--maxnreg", type=int, default=0)
    parser.add_argument("--gate-only", action="store_true")
    parser.add_argument("--match-gate-eviction", action="store_true")
    parser.add_argument("--disable-gate-warp-specialize", action="store_true")
    parser.add_argument("--gate-unroll-factor", type=int)
    parser.add_argument("--gate-range-stages", type=int)
    parser.add_argument("--force-emitted-gate-stages", type=int)
    parser.add_argument("--down-unroll-factor", type=int)
    parser.add_argument("--down-range-stages", type=int)
    parser.add_argument(
        "--match-effective-standalone-ranges",
        action="store_true",
    )
    parser.add_argument("--match-standalone-eviction", action="store_true")
    parser.add_argument("--gate-root-noinline", action="store_true")
    parser.add_argument("--inline-gate-body", action="store_true")
    parser.add_argument(
        "--gate-consumer-major",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--qkv-block-n", type=int)
    parser.add_argument("--qkv-block-k", type=int)
    parser.add_argument("--qkv-range-stages", type=int)
    parser.add_argument("--qkv-unroll-factor", type=int)
    parser.add_argument("--o-block-n", type=int)
    parser.add_argument("--o-block-k", type=int)
    parser.add_argument("--o-range-stages", type=int)
    parser.add_argument("--o-unroll-factor", type=int)
    parser.add_argument("--gate-block-n", type=int)
    parser.add_argument("--gate-block-k", type=int)
    parser.add_argument("--activation-block", type=int)
    parser.add_argument("--down-block-n", type=int)
    parser.add_argument("--down-block-k", type=int)
    parser.add_argument("--ple-gate-block-n", type=int)
    parser.add_argument("--ple-gate-block-k", type=int)
    parser.add_argument("--ple-projection-block-n", type=int)
    parser.add_argument("--ple-projection-block-k", type=int)
    parser.add_argument(
        "--qkv-continuation", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--attention-continuation",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--ffn-continuation", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--ffn-consumer-major",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--ffn-stream", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--ffn-scheduled-activation",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--compare-scheduled-activation", action="store_true")
    parser.add_argument("--ffn-first-groups", type=int, default=20)
    parser.add_argument("--ffn-consumer-workers", type=int, default=160)
    parser.add_argument(
        "--counted-event-on-ready",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--compare-counted-event", action="store_true")
    parser.add_argument("--stream-down-stages", type=int, default=4)
    parser.add_argument("--stream-down-unroll", type=int, default=0)
    parser.add_argument(
        "--o-continuation", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--poll-delay", type=int, default=32)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument(
        "--fused-signals",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--no-waits", action="store_true")
    parser.add_argument("--spread-narrow-roots", action="store_true")
    parser.add_argument("--relaxed-poll", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--compare-helion", action="store_true")
    parser.add_argument(
        "--print-lowered",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--lowered-output",
        default="/tmp/gemma4_codegen_schedule_probe_lowered.py",
    )
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--batch-replays", type=int, default=20)
    parser.add_argument(
        "--config-path",
        default="benchmarks/gemma4/gemma4_e4b_b200_configs.json",
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
