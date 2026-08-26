# ruff: noqa: ANN001, ANN202
# pyrefly: ignore-errors
"""One-orchestrator-CTA tile-scheduling probe for the Qwen3 FP8 FFN.

The Helion-generated W13, SiLU/quant, and W2 tile bodies are kept unchanged.
Only the persistent dispatch is replaced.  CTA 0 issues the initial W13 range,
consumes worker completion records, writes ready activation tile descriptors,
and releases the W2 range when its first dependency frontier is satisfied.
Every other CTA is a worker: workers wait for commands, execute tile bodies,
and report completion.  In particular, workers do not publish successor work
in the default ``mailbox`` mode.

The probe also reconstructs the earlier local-on-ready policy from the same
generated roots, and compares both policies with the compiler's static
schedule and the three standalone Helion kernels under CUDA graphs.
"""

from __future__ import annotations

import argparse
import ast
import ctypes
import json
import linecache
import os
from pathlib import Path
import textwrap

import torch

from probes.common import benchmark_interleaved
from probes.common import capture
from probes.common import require_idle_visible_gpu
from probes.common import visible_gpu_pids
from probes.qwen3.helion_qwen3_ffn_tile_dependency import _helion_resources
from probes.qwen3.helion_qwen3_ffn_tile_dependency import _persistent_config
from probes.qwen3.helion_qwen3_ffn_tile_dependency import qwen3_ffn_tile_dependency
from probes.qwen3.helion_qwen3_layer_baseline import FFN_CONFIGS
from probes.qwen3.helion_qwen3_layer_baseline import FP8_MAX
from probes.qwen3.helion_qwen3_layer_baseline import FP8_MIN
from probes.qwen3.helion_qwen3_layer_baseline import FP8_MIN_SCALE
from probes.qwen3.helion_qwen3_layer_baseline import block_fp8_mm
from probes.qwen3.helion_qwen3_layer_baseline import compile_config
from probes.qwen3.helion_qwen3_layer_baseline import silu_and_mul_per_block_quant

SCHEDULER_SOURCE = r"""
@triton.jit
def _qwen_sync_warp():
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
def _qwen_load_acquire(address):
    return tl.inline_asm_elementwise(
        asm="ld.acquire.gpu.global.u32 $0, [$1];",
        constraints="=r,l",
        args=[address],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _qwen_nanosleep(DELAY: tl.constexpr):
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
def _qwen_wait_epoch(address, epoch, POLL_DELAY: tl.constexpr):
    value = _qwen_load_acquire(address)
    while value != epoch:
        if POLL_DELAY:
            _qwen_nanosleep(POLL_DELAY)
        value = _qwen_load_acquire(address)
    _qwen_sync_warp()


@triton.jit
def _qwen_wait_count(address, target, POLL_DELAY: tl.constexpr):
    value = _qwen_load_acquire(address)
    while value < target:
        if POLL_DELAY:
            _qwen_nanosleep(POLL_DELAY)
        value = _qwen_load_acquire(address)
    _qwen_sync_warp()


@triton.jit
def _qwen_wait_ready_bit(ready_masks, task, epoch, POLL_DELAY: tl.constexpr):
    parity = epoch & 1
    word = task // 32
    bit = task % 32
    mask = tl.full([], 1, tl.uint32) << bit
    address = ready_masks + parity * 3 + word
    value = _qwen_load_acquire(address)
    while value & mask == 0:
        if POLL_DELAY:
            _qwen_nanosleep(POLL_DELAY)
        value = _qwen_load_acquire(address)
    _qwen_sync_warp()


@triton.jit
def _qwen_w13_physical_task(
    logical_task,
    SUBTILES_PER_ACTIVATION: tl.constexpr,
    HALF_TASKS: tl.constexpr,
):
    fan_in: tl.constexpr = 2 * SUBTILES_PER_ACTIVATION
    activation_task = logical_task // fan_in
    within_activation = logical_task % fan_in
    return tl.where(
        within_activation < SUBTILES_PER_ACTIVATION,
        activation_task * SUBTILES_PER_ACTIVATION + within_activation,
        HALF_TASKS
        + activation_task * SUBTILES_PER_ACTIVATION
        + within_activation
        - SUBTILES_PER_ACTIVATION,
    )


@triton.jit(noinline=True)
def _qwen_orchestrated_w13(
    ffn_q,
    w13_q,
    ffn_scale,
    w13_scale,
    gate_up,
    w13_arrivals,
    activation_ready,
    logical_task,
    epoch,
    SUBTILES_PER_ACTIVATION: tl.constexpr,
    HALF_TASKS: tl.constexpr,
    ARRIVAL_STRIDE: tl.constexpr,
    WORKER_PUBLICATION: tl.constexpr,
):
    fan_in: tl.constexpr = 2 * SUBTILES_PER_ACTIVATION
    physical_task = _qwen_w13_physical_task(
        logical_task,
        SUBTILES_PER_ACTIVATION,
        HALF_TASKS,
    )
    tile_dependency_root_0(
        ffn_q,
        w13_q,
        ffn_scale,
        w13_scale,
        gate_up,
        physical_task,
    )
    tl.debug_barrier()
    activation_task = logical_task // fan_in
    if WORKER_PUBLICATION:
        previous = tl.atomic_add(
            w13_arrivals + activation_task * ARRIVAL_STRIDE,
            1,
            sem="acq_rel",
            scope="gpu",
        )
        if previous % fan_in == fan_in - 1:
            tl.atomic_xchg(
                activation_ready + activation_task,
                epoch,
                sem="release",
                scope="gpu",
            )
    else:
        tl.atomic_add(
            w13_arrivals + activation_task * ARRIVAL_STRIDE,
            1,
            sem="release",
            scope="gpu",
        )


@triton.jit(noinline=True)
def _qwen_queued_w13(
    ffn_q,
    w13_q,
    ffn_scale,
    w13_scale,
    gate_up,
    w13_arrivals,
    completion_tail,
    completion_tasks,
    completion_epochs,
    logical_task,
    epoch,
    ACTIVATION_TASKS: tl.constexpr,
    SUBTILES_PER_ACTIVATION: tl.constexpr,
    HALF_TASKS: tl.constexpr,
    ARRIVAL_STRIDE: tl.constexpr,
):
    fan_in: tl.constexpr = 2 * SUBTILES_PER_ACTIVATION
    physical_task = _qwen_w13_physical_task(
        logical_task,
        SUBTILES_PER_ACTIVATION,
        HALF_TASKS,
    )
    tile_dependency_root_0(
        ffn_q,
        w13_q,
        ffn_scale,
        w13_scale,
        gate_up,
        physical_task,
    )
    tl.debug_barrier()
    activation_task = logical_task // fan_in
    previous = tl.atomic_add(
        w13_arrivals + activation_task * ARRIVAL_STRIDE,
        1,
        sem="acq_rel",
        scope="gpu",
    )
    if previous % fan_in == fan_in - 1:
        ticket = tl.atomic_add(
            completion_tail,
            1,
            sem="relaxed",
            scope="gpu",
        )
        slot = ticket % ACTIVATION_TASKS
        tl.store(completion_tasks + slot, activation_task)
        tl.atomic_xchg(
            completion_epochs + slot,
            epoch,
            sem="release",
            scope="gpu",
        )


@triton.jit(noinline=True)
def _qwen_bitmap_w13(
    ffn_q,
    w13_q,
    ffn_scale,
    w13_scale,
    gate_up,
    w13_arrivals,
    completion_masks,
    logical_task,
    epoch,
    ACTIVATION_TASKS: tl.constexpr,
    SUBTILES_PER_ACTIVATION: tl.constexpr,
    HALF_TASKS: tl.constexpr,
    ARRIVAL_STRIDE: tl.constexpr,
):
    fan_in: tl.constexpr = 2 * SUBTILES_PER_ACTIVATION
    physical_task = _qwen_w13_physical_task(
        logical_task,
        SUBTILES_PER_ACTIVATION,
        HALF_TASKS,
    )
    tile_dependency_root_0(
        ffn_q,
        w13_q,
        ffn_scale,
        w13_scale,
        gate_up,
        physical_task,
    )
    tl.debug_barrier()
    activation_task = logical_task // fan_in
    previous = tl.atomic_add(
        w13_arrivals + activation_task * ARRIVAL_STRIDE,
        1,
        sem="acq_rel",
        scope="gpu",
    )
    if previous % fan_in == fan_in - 1:
        ready_words: tl.constexpr = tl.cdiv(ACTIVATION_TASKS, 32)
        word = activation_task // 32
        bit = activation_task % 32
        mask = tl.full([], 1, tl.uint32) << bit
        tl.atomic_or(
            completion_masks + (epoch & 1) * ready_words + word,
            mask,
            sem="release",
            scope="gpu",
        )


@triton.jit(noinline=True)
def _qwen_complete_activation(
    gate_up,
    activation_scale,
    activation_q,
    dependency_state,
    activation_task,
    epoch,
    ROOT_1_OFFSET: tl.constexpr,
    ACTIVATION_TASKS: tl.constexpr,
    FIRST_ACTIVATION_TASKS: tl.constexpr,
    SPLIT_BASE: tl.constexpr,
    FP8_MAX_VALUE: tl.constexpr,
    FP8_MIN_SCALE_VALUE: tl.constexpr,
    FP8_MIN_VALUE: tl.constexpr,
):
    tile_dependency_root_1(
        gate_up,
        FP8_MAX_VALUE,
        FP8_MIN_SCALE_VALUE,
        activation_scale,
        FP8_MIN_VALUE,
        activation_q,
        ROOT_1_OFFSET + activation_task,
    )
    tl.debug_barrier()
    split = tl.where(activation_task < FIRST_ACTIVATION_TASKS, 0, 1)
    tl.atomic_add(
        dependency_state + SPLIT_BASE + split * 32,
        1,
        sem="release",
        scope="gpu",
    )


@triton.jit(noinline=True)
def _qwen_orchestrated_activation(
    gate_up,
    activation_scale,
    activation_q,
    w2_ready,
    dependency_state,
    activation_task,
    epoch,
    ROOT_1_OFFSET: tl.constexpr,
    ACTIVATION_TASKS: tl.constexpr,
    FIRST_ACTIVATION_TASKS: tl.constexpr,
    SPLIT_BASE: tl.constexpr,
    WORKER_PUBLICATION: tl.constexpr,
    FP8_MAX_VALUE: tl.constexpr,
    FP8_MIN_SCALE_VALUE: tl.constexpr,
    FP8_MIN_VALUE: tl.constexpr,
):
    tile_dependency_root_1(
        gate_up,
        FP8_MAX_VALUE,
        FP8_MIN_SCALE_VALUE,
        activation_scale,
        FP8_MIN_VALUE,
        activation_q,
        ROOT_1_OFFSET + activation_task,
    )
    tl.debug_barrier()
    split = tl.where(activation_task < FIRST_ACTIVATION_TASKS, 0, 1)
    if WORKER_PUBLICATION:
        previous = tl.atomic_add(
            dependency_state + SPLIT_BASE + split * 32,
            1,
            sem="acq_rel",
            scope="gpu",
        )
        if split == 0:
            if previous % FIRST_ACTIVATION_TASKS == FIRST_ACTIVATION_TASKS - 1:
                tl.atomic_xchg(
                    w2_ready,
                    epoch,
                    sem="release",
                    scope="gpu",
                )
    else:
        tl.atomic_add(
            dependency_state + SPLIT_BASE + split * 32,
            1,
            sem="release",
            scope="gpu",
        )


@triton.jit(noinline=True)
def _qwen_local_w13(
    ffn_q,
    w13_q,
    ffn_scale,
    w13_scale,
    gate_up,
    activation_scale,
    activation_q,
    dependency_state,
    logical_task,
    epoch,
    EVENT_BASE: tl.constexpr,
    SPLIT_BASE: tl.constexpr,
    ROOT_1_OFFSET: tl.constexpr,
    ACTIVATION_TASKS: tl.constexpr,
    FIRST_ACTIVATION_TASKS: tl.constexpr,
    SUBTILES_PER_ACTIVATION: tl.constexpr,
    HALF_TASKS: tl.constexpr,
    FP8_MAX_VALUE: tl.constexpr,
    FP8_MIN_SCALE_VALUE: tl.constexpr,
    FP8_MIN_VALUE: tl.constexpr,
):
    fan_in: tl.constexpr = 2 * SUBTILES_PER_ACTIVATION
    physical_task = _qwen_w13_physical_task(
        logical_task,
        SUBTILES_PER_ACTIVATION,
        HALF_TASKS,
    )
    tile_dependency_root_0(
        ffn_q,
        w13_q,
        ffn_scale,
        w13_scale,
        gate_up,
        physical_task,
    )
    tl.debug_barrier()
    activation_task = logical_task // fan_in
    previous = tl.atomic_add(
        dependency_state + EVENT_BASE + activation_task * 32,
        1,
        sem="acq_rel",
        scope="gpu",
    )
    if previous % fan_in == fan_in - 1:
        _qwen_complete_activation(
            gate_up,
            activation_scale,
            activation_q,
            dependency_state,
            activation_task,
            epoch,
            ROOT_1_OFFSET,
            ACTIVATION_TASKS,
            FIRST_ACTIVATION_TASKS,
            SPLIT_BASE,
            FP8_MAX_VALUE,
            FP8_MIN_SCALE_VALUE,
            FP8_MIN_VALUE,
        )


@triton.jit(noinline=True)
def _qwen_orchestrate(
    w13_arrivals,
    activation_ready,
    w2_ready,
    dependency_state,
    epoch,
    ACTIVATION_TASKS: tl.constexpr,
    FAN_IN: tl.constexpr,
    FIRST_ACTIVATION_TASKS: tl.constexpr,
    SPLIT_BASE: tl.constexpr,
    READY_BLOCK: tl.constexpr,
    ARRIVAL_STRIDE: tl.constexpr,
    POLL_DELAY: tl.constexpr,
):
    task = tl.arange(0, READY_BLOCK)
    valid = task < ACTIVATION_TASKS
    safe_task = tl.minimum(task, ACTIVATION_TASKS - 1)
    published = tl.zeros([READY_BLOCK], tl.int1)
    remaining = tl.full([], ACTIVATION_TASKS, tl.int32)
    w2_published = tl.full([], 0, tl.int32)
    arrival_target = epoch * FAN_IN
    while (remaining > 0) | (w2_published == 0):
        if remaining > 0:
            arrivals = _qwen_load_acquire(
                w13_arrivals + safe_task * ARRIVAL_STRIDE
            )
            newly_ready = valid & ~published & (arrivals == arrival_target)
            tl.atomic_xchg(
                activation_ready + task,
                epoch,
                mask=newly_ready,
                sem="release",
                scope="gpu",
            )
            published = published | newly_ready
            remaining -= tl.sum(newly_ready.to(tl.int32), axis=0)
        if w2_published == 0:
            first_done = _qwen_load_acquire(dependency_state + SPLIT_BASE)
            if first_done == epoch * FIRST_ACTIVATION_TASKS:
                tl.atomic_xchg(
                    w2_ready,
                    epoch,
                    sem="release",
                    scope="gpu",
                )
                w2_published = 1
        if (remaining > 0) | (w2_published == 0):
            if POLL_DELAY:
                _qwen_nanosleep(POLL_DELAY)


@triton.jit(noinline=True)
def _qwen_orchestrate_ordered(
    w13_arrivals,
    activation_ready,
    w2_ready,
    dependency_state,
    epoch,
    ACTIVATION_TASKS: tl.constexpr,
    FAN_IN: tl.constexpr,
    FIRST_ACTIVATION_TASKS: tl.constexpr,
    SPLIT_BASE: tl.constexpr,
    ARRIVAL_STRIDE: tl.constexpr,
    POLL_DELAY: tl.constexpr,
):
    activation_task = tl.full([], 0, tl.int32)
    w2_published = tl.full([], 0, tl.int32)
    arrival_target = epoch * FAN_IN
    while (activation_task < ACTIVATION_TASKS) | (w2_published == 0):
        progressed = tl.full([], 0, tl.int32)
        if activation_task < ACTIVATION_TASKS:
            arrivals = _qwen_load_acquire(
                w13_arrivals + activation_task * ARRIVAL_STRIDE
            )
            if arrivals == arrival_target:
                tl.atomic_xchg(
                    activation_ready + activation_task,
                    epoch,
                    sem="release",
                    scope="gpu",
                )
                activation_task += 1
                progressed = 1
        if w2_published == 0:
            first_done = _qwen_load_acquire(dependency_state + SPLIT_BASE)
            if first_done == epoch * FIRST_ACTIVATION_TASKS:
                tl.atomic_xchg(
                    w2_ready,
                    epoch,
                    sem="release",
                    scope="gpu",
                )
                w2_published = 1
                progressed = 1
        if progressed == 0:
            if POLL_DELAY:
                _qwen_nanosleep(POLL_DELAY)


@triton.jit(noinline=True)
def _qwen_orchestrate_sharded(
    w13_arrivals,
    activation_ready,
    w2_ready,
    dependency_state,
    epoch,
    ACTIVATION_TASKS: tl.constexpr,
    FAN_IN: tl.constexpr,
    FIRST_ACTIVATION_TASKS: tl.constexpr,
    SPLIT_BASE: tl.constexpr,
    SCAN_BLOCK: tl.constexpr,
    ARRIVAL_STRIDE: tl.constexpr,
    POLL_DELAY: tl.constexpr,
):
    lane = tl.arange(0, SCAN_BLOCK)
    scan = tl.full([], 0, tl.int32)
    scans: tl.constexpr = tl.cdiv(ACTIVATION_TASKS, SCAN_BLOCK)
    remaining = tl.full([], ACTIVATION_TASKS, tl.int32)
    w2_published = tl.full([], 0, tl.int32)
    arrival_target = epoch * FAN_IN
    while (remaining > 0) | (w2_published == 0):
        progressed = tl.full([], 0, tl.int32)
        if remaining > 0:
            task = scan * SCAN_BLOCK + lane
            valid = task < ACTIVATION_TASKS
            safe_task = tl.minimum(task, ACTIVATION_TASKS - 1)
            arrivals = _qwen_load_acquire(
                w13_arrivals + safe_task * ARRIVAL_STRIDE
            )
            ready_epoch = _qwen_load_acquire(activation_ready + safe_task)
            newly_ready = (
                valid
                & (ready_epoch != epoch)
                & (arrivals == arrival_target)
            )
            tl.atomic_xchg(
                activation_ready + task,
                epoch,
                mask=newly_ready,
                sem="release",
                scope="gpu",
            )
            published_now = tl.sum(newly_ready.to(tl.int32), axis=0)
            remaining -= published_now
            progressed += published_now
            scan += 1
            if scan == scans:
                scan = 0
        if w2_published == 0:
            first_done = _qwen_load_acquire(dependency_state + SPLIT_BASE)
            if first_done == epoch * FIRST_ACTIVATION_TASKS:
                tl.atomic_xchg(
                    w2_ready,
                    epoch,
                    sem="release",
                    scope="gpu",
                )
                w2_published = 1
                progressed = 1
        if progressed == 0:
            if POLL_DELAY:
                _qwen_nanosleep(POLL_DELAY)


@triton.jit(noinline=True)
def _qwen_orchestrate_grouped(
    w13_arrivals,
    activation_ready,
    w2_ready,
    dependency_state,
    epoch,
    ACTIVATION_TASKS: tl.constexpr,
    FAN_IN: tl.constexpr,
    FIRST_ACTIVATION_TASKS: tl.constexpr,
    SPLIT_BASE: tl.constexpr,
    READY_GROUP_SIZE: tl.constexpr,
    ARRIVAL_STRIDE: tl.constexpr,
    POLL_DELAY: tl.constexpr,
):
    lane = tl.arange(0, READY_GROUP_SIZE)
    ready_groups: tl.constexpr = tl.cdiv(
        ACTIVATION_TASKS,
        READY_GROUP_SIZE,
    )
    ready_group = tl.full([], 0, tl.int32)
    remaining = tl.full([], ready_groups, tl.int32)
    w2_published = tl.full([], 0, tl.int32)
    arrival_target = epoch * FAN_IN
    while (remaining > 0) | (w2_published == 0):
        progressed = tl.full([], 0, tl.int32)
        if remaining > 0:
            ready_epoch = _qwen_load_acquire(
                activation_ready + ready_group
            )
            if ready_epoch != epoch:
                task = ready_group * READY_GROUP_SIZE + lane
                valid = task < ACTIVATION_TASKS
                safe_task = tl.minimum(task, ACTIVATION_TASKS - 1)
                arrivals = _qwen_load_acquire(
                    w13_arrivals + safe_task * ARRIVAL_STRIDE
                )
                all_ready = tl.min(
                    tl.where(valid, arrivals == arrival_target, True).to(
                        tl.int32
                    ),
                    axis=0,
                )
                if all_ready:
                    tl.atomic_xchg(
                        activation_ready + ready_group,
                        epoch,
                        sem="release",
                        scope="gpu",
                    )
                    remaining -= 1
                    progressed = 1
            ready_group += 1
            if ready_group == ready_groups:
                ready_group = 0
        if w2_published == 0:
            first_done = _qwen_load_acquire(dependency_state + SPLIT_BASE)
            if first_done == epoch * FIRST_ACTIVATION_TASKS:
                tl.atomic_xchg(
                    w2_ready,
                    epoch,
                    sem="release",
                    scope="gpu",
                )
                w2_published = 1
                progressed = 1
        if progressed == 0:
            if POLL_DELAY:
                _qwen_nanosleep(POLL_DELAY)


@triton.jit(noinline=True)
def _qwen_orchestrate_bitmask(
    w13_arrivals,
    activation_ready,
    w2_ready,
    dependency_state,
    epoch,
    ACTIVATION_TASKS: tl.constexpr,
    FAN_IN: tl.constexpr,
    FIRST_ACTIVATION_TASKS: tl.constexpr,
    SPLIT_BASE: tl.constexpr,
    ARRIVAL_STRIDE: tl.constexpr,
    POLL_DELAY: tl.constexpr,
):
    lanes = tl.arange(0, 32)
    ready_words: tl.constexpr = tl.cdiv(ACTIVATION_TASKS, 32)
    parity = epoch & 1
    word = tl.full([], 0, tl.int32)
    remaining = tl.full([], ready_words, tl.int32)
    w2_published = tl.full([], 0, tl.int32)
    arrival_target = epoch * FAN_IN
    full_mask = tl.full([], 0xFFFFFFFF, tl.uint32)
    while (remaining > 0) | (w2_published == 0):
        progressed = tl.full([], 0, tl.int32)
        if remaining > 0:
            ready_address = activation_ready + parity * ready_words + word
            published = _qwen_load_acquire(ready_address)
            if published != full_mask:
                task = word * 32 + lanes
                valid = task < ACTIVATION_TASKS
                safe_task = tl.minimum(task, ACTIVATION_TASKS - 1)
                arrivals = _qwen_load_acquire(
                    w13_arrivals + safe_task * ARRIVAL_STRIDE
                )
                ready_bits = tl.sum(
                    tl.where(
                        valid & (arrivals == arrival_target),
                        tl.full([32], 1, tl.uint32) << lanes,
                        0,
                    ),
                    axis=0,
                )
                if ready_bits != published:
                    previous = tl.atomic_or(
                        ready_address,
                        ready_bits,
                        sem="release",
                        scope="gpu",
                    )
                    published = previous | ready_bits
                    progressed = 1
                if published == full_mask:
                    remaining -= 1
            word += 1
            if word == ready_words:
                word = 0
        if w2_published == 0:
            first_done = _qwen_load_acquire(dependency_state + SPLIT_BASE)
            if first_done == epoch * FIRST_ACTIVATION_TASKS:
                tl.atomic_xchg(
                    w2_ready,
                    epoch,
                    sem="release",
                    scope="gpu",
                )
                w2_published = 1
                progressed = 1
        if progressed == 0:
            if POLL_DELAY:
                _qwen_nanosleep(POLL_DELAY)
    next_parity = parity ^ 1
    tl.store(
        activation_ready + next_parity * ready_words + lanes,
        0,
        mask=lanes < ready_words,
    )


@triton.jit(noinline=True)
def _qwen_orchestrate_worklist(
    w13_arrivals,
    activation_worklist,
    activation_tail,
    w13_ready,
    w2_ready,
    dependency_state,
    epoch,
    ACTIVATION_TASKS: tl.constexpr,
    FAN_IN: tl.constexpr,
    FIRST_ACTIVATION_TASKS: tl.constexpr,
    SPLIT_BASE: tl.constexpr,
    READY_BLOCK: tl.constexpr,
    ARRIVAL_STRIDE: tl.constexpr,
    POLL_DELAY: tl.constexpr,
):
    tl.atomic_xchg(
        w13_ready,
        epoch,
        sem="release",
        scope="gpu",
    )
    task = tl.arange(0, READY_BLOCK)
    valid = task < ACTIVATION_TASKS
    safe_task = tl.minimum(task, ACTIVATION_TASKS - 1)
    published = tl.zeros([READY_BLOCK], tl.int1)
    published_count = tl.full([], 0, tl.int32)
    w2_published = tl.full([], 0, tl.int32)
    arrival_target = epoch * FAN_IN
    epoch_base = (epoch - 1) * ACTIVATION_TASKS
    while (published_count < ACTIVATION_TASKS) | (w2_published == 0):
        progressed = tl.full([], 0, tl.int32)
        if published_count < ACTIVATION_TASKS:
            arrivals = _qwen_load_acquire(
                w13_arrivals + safe_task * ARRIVAL_STRIDE
            )
            newly_ready = valid & ~published & (arrivals == arrival_target)
            ready_count = tl.sum(newly_ready.to(tl.int32), axis=0)
            if ready_count > 0:
                ready_rank = tl.cumsum(
                    newly_ready.to(tl.int32),
                    axis=0,
                ) - 1
                tl.store(
                    activation_worklist + published_count + ready_rank,
                    task,
                    mask=newly_ready,
                )
                tl.debug_barrier()
                published_count += ready_count
                tl.atomic_xchg(
                    activation_tail,
                    epoch_base + published_count,
                    sem="release",
                    scope="gpu",
                )
                published = published | newly_ready
                progressed = 1
        if w2_published == 0:
            first_done = _qwen_load_acquire(dependency_state + SPLIT_BASE)
            if first_done == epoch * FIRST_ACTIVATION_TASKS:
                tl.atomic_xchg(
                    w2_ready,
                    epoch,
                    sem="release",
                    scope="gpu",
                )
                w2_published = 1
                progressed = 1
        if progressed == 0:
            if POLL_DELAY:
                _qwen_nanosleep(POLL_DELAY)


@triton.jit(noinline=True)
def _qwen_orchestrate_completion_queue(
    completion_tasks,
    completion_epochs,
    activation_worklist,
    activation_tail,
    w13_ready,
    w2_ready,
    dependency_state,
    epoch,
    ACTIVATION_TASKS: tl.constexpr,
    FIRST_ACTIVATION_TASKS: tl.constexpr,
    SPLIT_BASE: tl.constexpr,
    QUEUE_BATCH: tl.constexpr,
    POLL_DELAY: tl.constexpr,
):
    tl.atomic_xchg(
        w13_ready,
        epoch,
        sem="release",
        scope="gpu",
    )
    lane = tl.arange(0, QUEUE_BATCH)
    published = tl.full([], 0, tl.int32)
    w2_published = tl.full([], 0, tl.int32)
    epoch_base = (epoch - 1) * ACTIVATION_TASKS
    while (published < ACTIVATION_TASKS) | (w2_published == 0):
        progressed = tl.full([], 0, tl.int32)
        if published < ACTIVATION_TASKS:
            remaining = ACTIVATION_TASKS - published
            batch_count = tl.minimum(remaining, QUEUE_BATCH)
            ticket = epoch_base + published + lane
            slot = ticket % ACTIVATION_TASKS
            valid = lane < batch_count
            slot_epoch = _qwen_load_acquire(
                completion_epochs + slot
            )
            batch_ready = tl.min(
                tl.where(valid, slot_epoch == epoch, True).to(tl.int32),
                axis=0,
            )
            if batch_ready:
                task = tl.load(completion_tasks + slot, mask=valid)
                tl.store(
                    activation_worklist + published + lane,
                    task,
                    mask=valid,
                )
                tl.debug_barrier()
                published += batch_count
                tl.atomic_xchg(
                    activation_tail,
                    epoch_base + published,
                    sem="release",
                    scope="gpu",
                )
                progressed = 1
        if w2_published == 0:
            first_done = _qwen_load_acquire(dependency_state + SPLIT_BASE)
            if first_done == epoch * FIRST_ACTIVATION_TASKS:
                tl.atomic_xchg(
                    w2_ready,
                    epoch,
                    sem="release",
                    scope="gpu",
                )
                w2_published = 1
                progressed = 1
        if progressed == 0:
            if POLL_DELAY:
                _qwen_nanosleep(POLL_DELAY)


@triton.jit(noinline=True)
def _qwen_orchestrate_mailboxes(
    completion_masks,
    activation_ready,
    activation_worklist,
    w13_ready,
    w2_ready,
    dependency_state,
    epoch,
    ACTIVATION_TASKS: tl.constexpr,
    FIRST_ACTIVATION_TASKS: tl.constexpr,
    SPLIT_BASE: tl.constexpr,
    READY_BLOCK: tl.constexpr,
    POLL_DELAY: tl.constexpr,
):
    tl.atomic_xchg(
        w13_ready,
        epoch,
        sem="release",
        scope="gpu",
    )
    ready_words: tl.constexpr = tl.cdiv(ACTIVATION_TASKS, 32)
    task = tl.arange(0, READY_BLOCK)
    valid = task < ACTIVATION_TASKS
    word = task // 32
    safe_word = tl.minimum(word, ready_words - 1)
    bit = task % 32
    published = tl.zeros([READY_BLOCK], tl.int1)
    published_count = tl.full([], 0, tl.int32)
    w2_published = tl.full([], 0, tl.int32)
    parity = epoch & 1
    while (published_count < ACTIVATION_TASKS) | (w2_published == 0):
        progressed = tl.full([], 0, tl.int32)
        if published_count < ACTIVATION_TASKS:
            reported_word = _qwen_load_acquire(
                completion_masks + parity * ready_words + safe_word
            )
            reported = (
                reported_word & (tl.full([], 1, tl.uint32) << bit)
            ) != 0
            newly_ready = valid & ~published & reported
            ready_count = tl.sum(newly_ready.to(tl.int32), axis=0)
            if ready_count > 0:
                ready_rank = tl.cumsum(
                    newly_ready.to(tl.int32),
                    axis=0,
                ) - 1
                mailbox = published_count + ready_rank
                tl.store(
                    activation_worklist + mailbox,
                    task,
                    mask=newly_ready,
                )
                tl.debug_barrier()
                tl.atomic_xchg(
                    activation_ready + mailbox,
                    epoch,
                    mask=newly_ready,
                    sem="release",
                    scope="gpu",
                )
                published_count += ready_count
                published = published | newly_ready
                progressed = 1
        if w2_published == 0:
            first_done = _qwen_load_acquire(dependency_state + SPLIT_BASE)
            if first_done == epoch * FIRST_ACTIVATION_TASKS:
                tl.atomic_xchg(
                    w2_ready,
                    epoch,
                    sem="release",
                    scope="gpu",
                )
                w2_published = 1
                progressed = 1
        if progressed == 0:
            if POLL_DELAY:
                _qwen_nanosleep(POLL_DELAY)
    next_parity = parity ^ 1
    ready_word = tl.arange(0, 4)
    tl.store(
        completion_masks + next_parity * ready_words + ready_word,
        0,
        mask=ready_word < ready_words,
    )


@triton.jit(noinline=True)
def _qwen_orchestrate_direct(
    completion_masks,
    activation_ready,
    w13_ready,
    w2_ready,
    dependency_state,
    epoch,
    ACTIVATION_TASKS: tl.constexpr,
    FIRST_ACTIVATION_TASKS: tl.constexpr,
    SPLIT_BASE: tl.constexpr,
    READY_BLOCK: tl.constexpr,
    POLL_DELAY: tl.constexpr,
):
    tl.atomic_xchg(
        w13_ready,
        epoch,
        sem="release",
        scope="gpu",
    )
    ready_words: tl.constexpr = tl.cdiv(ACTIVATION_TASKS, 32)
    task = tl.arange(0, READY_BLOCK)
    valid = task < ACTIVATION_TASKS
    word = task // 32
    safe_word = tl.minimum(word, ready_words - 1)
    bit = task % 32
    published = tl.zeros([READY_BLOCK], tl.int1)
    remaining = tl.full([], ACTIVATION_TASKS, tl.int32)
    w2_published = tl.full([], 0, tl.int32)
    parity = epoch & 1
    while (remaining > 0) | (w2_published == 0):
        progressed = tl.full([], 0, tl.int32)
        if remaining > 0:
            reported_word = _qwen_load_acquire(
                completion_masks + parity * ready_words + safe_word
            )
            reported = (
                reported_word & (tl.full([], 1, tl.uint32) << bit)
            ) != 0
            newly_ready = valid & ~published & reported
            ready_count = tl.sum(newly_ready.to(tl.int32), axis=0)
            if ready_count > 0:
                tl.atomic_xchg(
                    activation_ready + task,
                    epoch,
                    mask=newly_ready,
                    sem="release",
                    scope="gpu",
                )
                remaining -= ready_count
                published = published | newly_ready
                progressed = 1
        if w2_published == 0:
            first_done = _qwen_load_acquire(dependency_state + SPLIT_BASE)
            if first_done == epoch * FIRST_ACTIVATION_TASKS:
                tl.atomic_xchg(
                    w2_ready,
                    epoch,
                    sem="release",
                    scope="gpu",
                )
                w2_published = 1
                progressed = 1
        if progressed == 0:
            if POLL_DELAY:
                _qwen_nanosleep(POLL_DELAY)
    next_parity = parity ^ 1
    ready_word = tl.arange(0, 4)
    tl.store(
        completion_masks + next_parity * ready_words + ready_word,
        0,
        mask=ready_word < ready_words,
    )


@triton.jit(noinline=True)
def _qwen_wait_for_worker_publication(
    dependency_state,
    epoch,
    ACTIVATION_TASKS: tl.constexpr,
    FIRST_ACTIVATION_TASKS: tl.constexpr,
    SPLIT_BASE: tl.constexpr,
    POLL_DELAY: tl.constexpr,
):
    second_count: tl.constexpr = ACTIVATION_TASKS - FIRST_ACTIVATION_TASKS
    value = _qwen_load_acquire(dependency_state + SPLIT_BASE + 32)
    while value != epoch * second_count:
        if POLL_DELAY:
            _qwen_nanosleep(POLL_DELAY)
        value = _qwen_load_acquire(dependency_state + SPLIT_BASE + 32)


@triton.jit
def qwen3_ffn_local_on_ready(
    ffn_q,
    w13_q,
    ffn_scale,
    w13_scale,
    gate_up,
    activation_scale,
    activation_q,
    w2_q,
    w2_scale,
    output,
    dependency_state,
    NUM_SM: tl.constexpr,
    TOTAL_WORKERS: tl.constexpr,
    W13_TASKS: tl.constexpr,
    ACTIVATION_TASKS: tl.constexpr,
    W2_TASKS: tl.constexpr,
    ROOT_1_OFFSET: tl.constexpr,
    ROOT_2_OFFSET: tl.constexpr,
    EVENT_BASE: tl.constexpr,
    SPLIT_BASE: tl.constexpr,
    FIRST_ACTIVATION_TASKS: tl.constexpr,
    SUBTILES_PER_ACTIVATION: tl.constexpr,
    HALF_TASKS: tl.constexpr,
    FP8_MAX_VALUE: tl.constexpr,
    FP8_MIN_SCALE_VALUE: tl.constexpr,
    FP8_MIN_VALUE: tl.constexpr,
):
    worker = tl.program_id(0)
    epoch = tl.load(dependency_state + worker) + 1
    for logical_task in tl.range(worker, W13_TASKS, TOTAL_WORKERS):
        _qwen_local_w13(
            ffn_q,
            w13_q,
            ffn_scale,
            w13_scale,
            gate_up,
            activation_scale,
            activation_q,
            dependency_state,
            logical_task,
            epoch,
            EVENT_BASE,
            SPLIT_BASE,
            ROOT_1_OFFSET,
            ACTIVATION_TASKS,
            FIRST_ACTIVATION_TASKS,
            SUBTILES_PER_ACTIVATION,
            HALF_TASKS,
            FP8_MAX_VALUE,
            FP8_MIN_SCALE_VALUE,
            FP8_MIN_VALUE,
        )
    consumer_base: tl.constexpr = TOTAL_WORKERS - W2_TASKS
    if worker >= consumer_base:
        tile_dependency_root_2(
            __ROOT_2_NUM_SM_ARGUMENT__
            activation_scale,
            activation_q,
            w2_q,
            w2_scale,
            output,
            dependency_state,
            ROOT_2_OFFSET + worker - consumer_base,
            epoch,
        )
    tl.store(dependency_state + worker, epoch)


@triton.jit
def qwen3_ffn_orchestrator(
    ffn_q,
    w13_q,
    ffn_scale,
    w13_scale,
    gate_up,
    activation_scale,
    activation_q,
    w2_q,
    w2_scale,
    output,
    worker_epochs,
    w13_arrivals,
    activation_ready,
    activation_worklist,
    activation_tail,
    completion_tail,
    completion_tasks,
    completion_epochs,
    activation_claimed,
    w13_ready,
    w2_ready,
    w2_claimed,
    dependency_state,
    NUM_SM: tl.constexpr,
    TOTAL_PROGRAMS: tl.constexpr,
    W13_TASKS: tl.constexpr,
    ACTIVATION_TASKS: tl.constexpr,
    W2_TASKS: tl.constexpr,
    ROOT_1_OFFSET: tl.constexpr,
    ROOT_2_OFFSET: tl.constexpr,
    SPLIT_BASE: tl.constexpr,
    FIRST_ACTIVATION_TASKS: tl.constexpr,
    SUBTILES_PER_ACTIVATION: tl.constexpr,
    HALF_TASKS: tl.constexpr,
    ACTIVATION_REPLICAS: tl.constexpr,
    ACTIVATION_WORKER_BASE: tl.constexpr,
    W2_WORKER_BASE: tl.constexpr,
    DYNAMIC_W2: tl.constexpr,
    CENTRAL_PUBLICATION: tl.constexpr,
    CENTRAL_SCAN: tl.constexpr,
    ORCHESTRATOR_WAITS: tl.constexpr,
    READY_BLOCK: tl.constexpr,
    READY_GROUP_SIZE: tl.constexpr,
    COMPLETION_BATCH: tl.constexpr,
    ARRIVAL_STRIDE: tl.constexpr,
    POLL_DELAY: tl.constexpr,
    FP8_MAX_VALUE: tl.constexpr,
    FP8_MIN_SCALE_VALUE: tl.constexpr,
    FP8_MIN_VALUE: tl.constexpr,
):
    program = tl.program_id(0)
    epoch = tl.load(worker_epochs + program) + 1
    if program == 0:
        if CENTRAL_PUBLICATION:
            if CENTRAL_SCAN == 1:
                _qwen_orchestrate_ordered(
                    w13_arrivals,
                    activation_ready,
                    w2_ready,
                    dependency_state,
                    epoch,
                    ACTIVATION_TASKS,
                    2 * SUBTILES_PER_ACTIVATION,
                    FIRST_ACTIVATION_TASKS,
                    SPLIT_BASE,
                    ARRIVAL_STRIDE,
                    POLL_DELAY,
                )
            elif CENTRAL_SCAN == 2:
                _qwen_orchestrate_sharded(
                    w13_arrivals,
                    activation_ready,
                    w2_ready,
                    dependency_state,
                    epoch,
                    ACTIVATION_TASKS,
                    2 * SUBTILES_PER_ACTIVATION,
                    FIRST_ACTIVATION_TASKS,
                    SPLIT_BASE,
                    32,
                    ARRIVAL_STRIDE,
                    POLL_DELAY,
                )
            elif CENTRAL_SCAN == 3:
                _qwen_orchestrate_grouped(
                    w13_arrivals,
                    activation_ready,
                    w2_ready,
                    dependency_state,
                    epoch,
                    ACTIVATION_TASKS,
                    2 * SUBTILES_PER_ACTIVATION,
                    FIRST_ACTIVATION_TASKS,
                    SPLIT_BASE,
                    READY_GROUP_SIZE,
                    ARRIVAL_STRIDE,
                    POLL_DELAY,
                )
            elif CENTRAL_SCAN == 4:
                _qwen_orchestrate_bitmask(
                    w13_arrivals,
                    activation_ready,
                    w2_ready,
                    dependency_state,
                    epoch,
                    ACTIVATION_TASKS,
                    2 * SUBTILES_PER_ACTIVATION,
                    FIRST_ACTIVATION_TASKS,
                    SPLIT_BASE,
                    ARRIVAL_STRIDE,
                    POLL_DELAY,
                )
            elif CENTRAL_SCAN == 5:
                _qwen_orchestrate_worklist(
                    w13_arrivals,
                    activation_worklist,
                    activation_tail,
                    w13_ready,
                    w2_ready,
                    dependency_state,
                    epoch,
                    ACTIVATION_TASKS,
                    2 * SUBTILES_PER_ACTIVATION,
                    FIRST_ACTIVATION_TASKS,
                    SPLIT_BASE,
                    READY_BLOCK,
                    ARRIVAL_STRIDE,
                    POLL_DELAY,
                )
            elif CENTRAL_SCAN == 6:
                _qwen_orchestrate_completion_queue(
                    completion_tasks,
                    completion_epochs,
                    activation_worklist,
                    activation_tail,
                    w13_ready,
                    w2_ready,
                    dependency_state,
                    epoch,
                    ACTIVATION_TASKS,
                    FIRST_ACTIVATION_TASKS,
                    SPLIT_BASE,
                    COMPLETION_BATCH,
                    POLL_DELAY,
                )
            elif CENTRAL_SCAN == 7:
                _qwen_orchestrate_mailboxes(
                    completion_epochs,
                    activation_ready,
                    activation_worklist,
                    w13_ready,
                    w2_ready,
                    dependency_state,
                    epoch,
                    ACTIVATION_TASKS,
                    FIRST_ACTIVATION_TASKS,
                    SPLIT_BASE,
                    READY_BLOCK,
                    POLL_DELAY,
                )
            elif CENTRAL_SCAN == 8:
                _qwen_orchestrate_direct(
                    completion_epochs,
                    activation_ready,
                    w13_ready,
                    w2_ready,
                    dependency_state,
                    epoch,
                    ACTIVATION_TASKS,
                    FIRST_ACTIVATION_TASKS,
                    SPLIT_BASE,
                    READY_BLOCK,
                    POLL_DELAY,
                )
            else:
                _qwen_orchestrate(
                    w13_arrivals,
                    activation_ready,
                    w2_ready,
                    dependency_state,
                    epoch,
                    ACTIVATION_TASKS,
                    2 * SUBTILES_PER_ACTIVATION,
                    FIRST_ACTIVATION_TASKS,
                    SPLIT_BASE,
                    READY_BLOCK,
                    ARRIVAL_STRIDE,
                    POLL_DELAY,
                )
        elif ORCHESTRATOR_WAITS:
            _qwen_wait_for_worker_publication(
                dependency_state,
                epoch,
                ACTIVATION_TASKS,
                FIRST_ACTIVATION_TASKS,
                SPLIT_BASE,
                POLL_DELAY,
            )
    else:
        worker = program - 1
        worker_count: tl.constexpr = TOTAL_PROGRAMS - 1
        if CENTRAL_PUBLICATION and (
            CENTRAL_SCAN == 5
            or CENTRAL_SCAN == 6
            or CENTRAL_SCAN == 7
            or CENTRAL_SCAN == 8
        ):
            _qwen_wait_epoch(w13_ready, epoch, POLL_DELAY)
        for logical_task in tl.range(worker, W13_TASKS, worker_count):
            if CENTRAL_PUBLICATION and CENTRAL_SCAN == 6:
                _qwen_queued_w13(
                    ffn_q,
                    w13_q,
                    ffn_scale,
                    w13_scale,
                    gate_up,
                    w13_arrivals,
                    completion_tail,
                    completion_tasks,
                    completion_epochs,
                    logical_task,
                    epoch,
                    ACTIVATION_TASKS,
                    SUBTILES_PER_ACTIVATION,
                    HALF_TASKS,
                    ARRIVAL_STRIDE,
                )
            elif CENTRAL_PUBLICATION and (
                CENTRAL_SCAN == 7 or CENTRAL_SCAN == 8
            ):
                _qwen_bitmap_w13(
                    ffn_q,
                    w13_q,
                    ffn_scale,
                    w13_scale,
                    gate_up,
                    w13_arrivals,
                    completion_epochs,
                    logical_task,
                    epoch,
                    ACTIVATION_TASKS,
                    SUBTILES_PER_ACTIVATION,
                    HALF_TASKS,
                    ARRIVAL_STRIDE,
                )
            else:
                _qwen_orchestrated_w13(
                    ffn_q,
                    w13_q,
                    ffn_scale,
                    w13_scale,
                    gate_up,
                    w13_arrivals,
                    activation_ready,
                    logical_task,
                    epoch,
                    SUBTILES_PER_ACTIVATION,
                    HALF_TASKS,
                    ARRIVAL_STRIDE,
                    not CENTRAL_PUBLICATION,
                )

        activation_workers: tl.constexpr = (
            ACTIVATION_REPLICAS * ACTIVATION_TASKS
        )
        activation_worker_end: tl.constexpr = (
            ACTIVATION_WORKER_BASE + activation_workers
        )
        is_activation_worker = (
            worker >= ACTIVATION_WORKER_BASE
        ) & (worker < activation_worker_end)
        if is_activation_worker:
            activation_rank = worker - ACTIVATION_WORKER_BASE
            if CENTRAL_PUBLICATION and (CENTRAL_SCAN == 5 or CENTRAL_SCAN == 6):
                _qwen_wait_count(
                    activation_tail,
                    (epoch - 1) * ACTIVATION_TASKS + activation_rank + 1,
                    POLL_DELAY,
                )
                activation_task = tl.load(
                    activation_worklist + activation_rank
                ).to(tl.int32)
            elif CENTRAL_PUBLICATION and CENTRAL_SCAN == 7:
                _qwen_wait_epoch(
                    activation_ready + activation_rank,
                    epoch,
                    POLL_DELAY,
                )
                activation_task = tl.load(
                    activation_worklist + activation_rank
                ).to(tl.int32)
            else:
                activation_task = activation_rank % ACTIVATION_TASKS
                ready_key = activation_task
                if CENTRAL_PUBLICATION and CENTRAL_SCAN == 4:
                    _qwen_wait_ready_bit(
                        activation_ready,
                        activation_task,
                        epoch,
                        POLL_DELAY,
                    )
                else:
                    if CENTRAL_PUBLICATION and CENTRAL_SCAN == 3:
                        ready_key = activation_task // READY_GROUP_SIZE
                    _qwen_wait_epoch(
                        activation_ready + ready_key,
                        epoch,
                        POLL_DELAY,
                    )
            if ACTIVATION_REPLICAS == 1:
                _qwen_orchestrated_activation(
                    gate_up,
                    activation_scale,
                    activation_q,
                    w2_ready,
                    dependency_state,
                    activation_task,
                    epoch,
                    ROOT_1_OFFSET,
                    ACTIVATION_TASKS,
                    FIRST_ACTIVATION_TASKS,
                    SPLIT_BASE,
                    not CENTRAL_PUBLICATION,
                    FP8_MAX_VALUE,
                    FP8_MIN_SCALE_VALUE,
                    FP8_MIN_VALUE,
                )
            else:
                previous = tl.atomic_cas(
                    activation_claimed + activation_task,
                    epoch - 1,
                    epoch,
                    sem="acq_rel",
                    scope="gpu",
                )
                if previous == epoch - 1:
                    _qwen_orchestrated_activation(
                        gate_up,
                        activation_scale,
                        activation_q,
                        w2_ready,
                        dependency_state,
                        activation_task,
                        epoch,
                        ROOT_1_OFFSET,
                        ACTIVATION_TASKS,
                        FIRST_ACTIVATION_TASKS,
                        SPLIT_BASE,
                        not CENTRAL_PUBLICATION,
                        FP8_MAX_VALUE,
                        FP8_MIN_SCALE_VALUE,
                        FP8_MIN_VALUE,
                    )
        else:
            w2_rank = tl.where(
                worker < ACTIVATION_WORKER_BASE,
                worker,
                worker - activation_workers,
            )
            if DYNAMIC_W2:
                w2_task = w2_rank % W2_TASKS
                _qwen_wait_epoch(w2_ready, epoch, POLL_DELAY)
                previous = tl.atomic_cas(
                    w2_claimed + w2_task,
                    epoch - 1,
                    epoch,
                    sem="acq_rel",
                    scope="gpu",
                )
                if previous == epoch - 1:
                    tile_dependency_root_2(
                        __ROOT_2_NUM_SM_ARGUMENT__
                        activation_scale,
                        activation_q,
                        w2_q,
                        w2_scale,
                        output,
                        dependency_state,
                        ROOT_2_OFFSET + w2_task,
                        epoch,
                    )
            elif (
                worker >= W2_WORKER_BASE
                and worker < W2_WORKER_BASE + W2_TASKS
            ):
                _qwen_wait_epoch(w2_ready, epoch, POLL_DELAY)
                tile_dependency_root_2(
                    __ROOT_2_NUM_SM_ARGUMENT__
                    activation_scale,
                    activation_q,
                    w2_q,
                    w2_scale,
                    output,
                    dependency_state,
                    ROOT_2_OFFSET + worker - W2_WORKER_BASE,
                    epoch,
                )
    tl.store(worker_epochs + program, epoch)
"""


def _generated_namespace(bound, config, lowered_output: Path):
    lowered = bound.to_triton_code(config, output_origin_lines=True)
    module = ast.parse(lowered)
    master = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name.startswith("_helion_qwen3_ffn")
    )
    lines = lowered.splitlines(keepends=True)
    decorator_line = master.lineno - 1
    while decorator_line > 0 and lines[decorator_line - 1].lstrip().startswith("@"):
        decorator_line -= 1
    prefix = "".join(lines[:decorator_line])
    root_2 = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "tile_dependency_root_2"
    )
    root_2_has_num_sm = root_2.args.args[0].arg.lower().endswith("num_sm")
    scheduler_source = SCHEDULER_SOURCE.replace(
        "__ROOT_2_NUM_SM_ARGUMENT__",
        "NUM_SM," if root_2_has_num_sm else "",
    )
    source = prefix + "\n" + textwrap.dedent(scheduler_source)
    filename = str(Path(__file__).with_name("_generated_qwen3_ffn_orchestrator.py"))
    linecache.cache[filename] = (
        len(source),
        None,
        source.splitlines(keepends=True),
        filename,
    )
    namespace: dict[str, object] = {"__name__": "_generated_qwen3_ffn_orchestrator"}
    exec(compile(source, filename, "exec"), namespace)
    lowered_output.write_text(source)
    return namespace, source


def _constant(namespace, name: str) -> int:
    value = namespace[name]
    return int(getattr(value, "value", value))


def _geometry(namespace, args, static_workers: int) -> dict[str, int]:
    w13_block = _constant(namespace, "_BLOCK_SIZE_1")
    activation_block = _constant(namespace, "_BLOCK_SIZE_4")
    w2_block = _constant(namespace, "_BLOCK_SIZE_6")
    if args.batch != 1:
        raise ValueError("the orchestrator probe currently supports --batch 1")
    if args.intermediate % activation_block:
        raise ValueError("activation tiles must exactly divide the intermediate size")
    if activation_block % w13_block:
        raise ValueError("activation tiles must contain whole W13 tiles")
    if args.hidden % w2_block:
        raise ValueError("W2 tiles must exactly divide the hidden size")
    activation_tasks = args.intermediate // activation_block
    subtiles = activation_block // w13_block
    fan_in = 2 * subtiles
    w13_tasks = 2 * args.intermediate // w13_block
    w2_tasks = args.hidden // w2_block
    first_activation_tasks = min(activation_tasks, static_workers // fan_in)
    if first_activation_tasks == activation_tasks:
        raise ValueError("the generated W2 root must contain a two-part wait")
    event_base = (static_workers + 31) // 32 * 32
    split_base = event_base + activation_tasks * 32
    return {
        "w13_tasks": w13_tasks,
        "activation_tasks": activation_tasks,
        "w2_tasks": w2_tasks,
        "root_1_offset": w13_tasks,
        "root_2_offset": w13_tasks + activation_tasks,
        "event_base": event_base,
        "split_base": split_base,
        "state_size": split_base + 64,
        "first_activation_tasks": first_activation_tasks,
        "subtiles_per_activation": subtiles,
        "half_tasks": args.intermediate // w13_block,
        "ready_block": 1 << (activation_tasks - 1).bit_length(),
    }


def _resident_capacity(compiled, num_warps: int) -> tuple[int, int]:
    compiled._init_handles()
    cuda = ctypes.CDLL("libcuda.so.1")
    blocks = ctypes.c_int()
    status = cuda.cuOccupancyMaxActiveBlocksPerMultiprocessor(
        ctypes.byref(blocks),
        ctypes.c_void_p(compiled.function),
        ctypes.c_int(num_warps * 32),
        ctypes.c_size_t(compiled.metadata.shared),
    )
    if status != 0:
        raise RuntimeError(
            f"cuOccupancyMaxActiveBlocksPerMultiprocessor failed: {status}"
        )
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    return blocks.value * sms, blocks.value


def _triton_resources(compiled, num_warps: int) -> dict[str, int]:
    capacity, blocks_per_sm = _resident_capacity(compiled, num_warps)
    return {
        "registers": compiled.n_regs,
        "spills": compiled.n_spills,
        "shared": compiled.metadata.shared,
        "resident_programs": capacity,
        "blocks_per_sm": blocks_per_sm,
        "ptx_atomics": compiled.asm["ptx"].count("atom."),
        "ptx_acquire_loads": compiled.asm["ptx"].count("ld.acquire"),
    }


def _allocate_inputs(args) -> tuple[torch.Tensor, ...]:
    device = "cuda"
    torch.manual_seed(args.seed)
    ffn_q = torch.randn(
        (args.batch, args.hidden), device=device, dtype=torch.bfloat16
    ).to(torch.float8_e4m3fn)
    ffn_scale = torch.rand(
        (args.batch, args.hidden // args.group),
        device=device,
        dtype=torch.float32,
    )
    w13_q = torch.randn(
        (2 * args.intermediate, args.hidden),
        device=device,
        dtype=torch.bfloat16,
    ).to(torch.float8_e4m3fn)
    w13_scale = torch.rand(
        (2 * args.intermediate // args.group, args.hidden // args.group),
        device=device,
        dtype=torch.float32,
    )
    w2_q = torch.randn(
        (args.hidden, args.intermediate), device=device, dtype=torch.bfloat16
    ).to(torch.float8_e4m3fn)
    w2_scale = torch.rand(
        (args.hidden // args.group, args.intermediate // args.group),
        device=device,
        dtype=torch.float32,
    )
    return ffn_q, ffn_scale, w13_q, w13_scale, w2_q, w2_scale


def _allocate_outputs(args) -> dict[str, torch.Tensor]:
    return {
        "gate_up": torch.empty(
            (args.batch, 2 * args.intermediate),
            device="cuda",
            dtype=torch.bfloat16,
        ),
        "activation_q": torch.empty(
            (args.batch, args.intermediate),
            device="cuda",
            dtype=torch.float8_e4m3fn,
        ),
        "activation_scale": torch.empty(
            (args.batch, args.intermediate // args.group),
            device="cuda",
            dtype=torch.float32,
        ),
        "output": torch.empty(
            (args.batch, args.hidden), device="cuda", dtype=torch.bfloat16
        ),
    }


def _allocate_local_state(geometry) -> torch.Tensor:
    return torch.zeros(geometry["state_size"], device="cuda", dtype=torch.uint32)


def _allocate_orchestrator_state(
    geometry, programs: int, arrival_stride: int
) -> dict[str, torch.Tensor]:
    return {
        "worker_epochs": torch.zeros(programs, device="cuda", dtype=torch.uint32),
        "w13_arrivals": torch.zeros(
            (geometry["activation_tasks"] - 1) * arrival_stride + 1,
            device="cuda",
            dtype=torch.uint32,
        ),
        "activation_ready": torch.zeros(
            geometry["activation_tasks"]
            + 2 * ((geometry["activation_tasks"] + 31) // 32),
            device="cuda",
            dtype=torch.uint32,
        ),
        "activation_worklist": torch.empty(
            geometry["activation_tasks"], device="cuda", dtype=torch.uint32
        ),
        "activation_tail": torch.zeros(1, device="cuda", dtype=torch.uint32),
        "completion_tail": torch.zeros(1, device="cuda", dtype=torch.uint32),
        "completion_tasks": torch.empty(
            geometry["activation_tasks"], device="cuda", dtype=torch.uint32
        ),
        "completion_epochs": torch.zeros(
            geometry["activation_tasks"], device="cuda", dtype=torch.uint32
        ),
        "activation_claimed": torch.zeros(
            geometry["activation_tasks"], device="cuda", dtype=torch.uint32
        ),
        "w13_ready": torch.zeros(1, device="cuda", dtype=torch.uint32),
        "w2_ready": torch.zeros(1, device="cuda", dtype=torch.uint32),
        "w2_claimed": torch.zeros(
            geometry["w2_tasks"], device="cuda", dtype=torch.uint32
        ),
        "dependency_state": torch.zeros(
            geometry["state_size"], device="cuda", dtype=torch.uint32
        ),
    }


def _local_arguments(inputs, outputs, state, geometry, args, num_sm, workers):
    ffn_q, ffn_scale, w13_q, w13_scale, w2_q, w2_scale = inputs
    return (
        ffn_q,
        w13_q,
        ffn_scale,
        w13_scale,
        outputs["gate_up"],
        outputs["activation_scale"],
        outputs["activation_q"],
        w2_q,
        w2_scale,
        outputs["output"],
        state,
        num_sm,
        workers,
        geometry["w13_tasks"],
        geometry["activation_tasks"],
        geometry["w2_tasks"],
        geometry["root_1_offset"],
        geometry["root_2_offset"],
        geometry["event_base"],
        geometry["split_base"],
        geometry["first_activation_tasks"],
        geometry["subtiles_per_activation"],
        geometry["half_tasks"],
        FP8_MAX,
        FP8_MIN_SCALE,
        FP8_MIN,
    )


def _orchestrator_arguments(
    inputs,
    outputs,
    state,
    geometry,
    args,
    num_sm,
    programs,
):
    ffn_q, ffn_scale, w13_q, w13_scale, w2_q, w2_scale = inputs
    workers = programs - 1
    second_wave_workers = max(0, geometry["w13_tasks"] - workers)
    activation_workers = args.activation_replicas * geometry["activation_tasks"]
    w2_worker_base = workers - geometry["w2_tasks"]
    activation_worker_base = w2_worker_base - activation_workers
    if activation_worker_base < second_wave_workers:
        activation_worker_base = second_wave_workers
        w2_worker_base = 0
        if geometry["w2_tasks"] > activation_worker_base:
            raise ValueError(
                "not enough workers for disjoint activation and W2 cohorts"
            )
    return (
        ffn_q,
        w13_q,
        ffn_scale,
        w13_scale,
        outputs["gate_up"],
        outputs["activation_scale"],
        outputs["activation_q"],
        w2_q,
        w2_scale,
        outputs["output"],
        state["worker_epochs"],
        state["w13_arrivals"],
        state["activation_ready"],
        state["activation_worklist"],
        state["activation_tail"],
        state["completion_tail"],
        state["completion_tasks"],
        state["completion_epochs"],
        state["activation_claimed"],
        state["w13_ready"],
        state["w2_ready"],
        state["w2_claimed"],
        state["dependency_state"],
        num_sm,
        programs,
        geometry["w13_tasks"],
        geometry["activation_tasks"],
        geometry["w2_tasks"],
        geometry["root_1_offset"],
        geometry["root_2_offset"],
        geometry["split_base"],
        geometry["first_activation_tasks"],
        geometry["subtiles_per_activation"],
        geometry["half_tasks"],
        args.activation_replicas,
        activation_worker_base,
        w2_worker_base,
        args.dynamic_w2,
        args.central_publication,
        {
            "vector": 0,
            "ordered": 1,
            "sharded": 2,
            "grouped": 3,
            "bitmask": 4,
            "worklist": 5,
            "completion_queue": 6,
            "mailbox": 7,
            "direct": 8,
        }[args.central_scan],
        args.orchestrator_waits,
        geometry["ready_block"],
        args.ready_group_size,
        args.completion_batch,
        args.arrival_stride,
        args.poll_delay,
        FP8_MAX,
        FP8_MIN_SCALE,
        FP8_MIN,
    )


def _compile_and_check(kernel, kernel_args, programs: int, name: str):
    compiled = kernel.warmup(
        *kernel_args,
        grid=(programs,),
        num_warps=1,
        num_stages=2,
    )
    capacity, _ = _resident_capacity(compiled, 1)
    if programs > capacity:
        raise RuntimeError(
            f"{name} needs {programs} co-resident CTAs but only {capacity} fit"
        )
    return compiled


def _launch(kernel, kernel_args, programs: int):
    return kernel[(programs,)](
        *kernel_args,
        num_warps=1,
        num_stages=2,
    )


def _assert_exact(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    if torch.equal(actual, expected):
        return
    difference = (actual.float() - expected.float()).abs()
    raise AssertionError(
        f"{name} changed numerics: max_abs={difference.max().item()}, "
        f"mean_abs={difference.mean().item()}"
    )


def _assert_close(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(
        actual.float(),
        expected.float(),
        atol=0.25,
        rtol=5e-2,
        msg=name,
    )


def run(args) -> None:
    if not args.allow_busy:
        require_idle_visible_gpu()
    inputs = _allocate_inputs(args)
    kernel_args = (*inputs, args.group)
    bound = qwen3_ffn_tile_dependency.bind(kernel_args)
    config = _persistent_config(bound, args)
    lowered_output = Path(args.lowered_output)
    namespace, _ = _generated_namespace(bound, config, lowered_output)
    local_kernel = namespace["qwen3_ffn_local_on_ready"]
    orchestrator_kernel = namespace["qwen3_ffn_orchestrator"]

    num_sm = torch.cuda.get_device_properties(0).multi_processor_count
    static_workers = args.cross_loop_workers or num_sm * args.worker_multiplier
    programs = args.programs or static_workers + 1
    if args.central_scan in {"worklist", "completion_queue", "mailbox"}:
        if args.activation_replicas != 1:
            raise ValueError(f"{args.central_scan} requires --activation-replicas 1")
    geometry = _geometry(namespace, args, static_workers)

    static_compiled = bound.compile_config(config)
    static_output, static_gate, static_q, static_scale = static_compiled(*kernel_args)

    local_outputs = _allocate_outputs(args)
    local_state = _allocate_local_state(geometry)
    local_args = _local_arguments(
        inputs,
        local_outputs,
        local_state,
        geometry,
        args,
        num_sm,
        static_workers,
    )
    local_binary = _compile_and_check(
        local_kernel,
        local_args,
        static_workers,
        "local-on-ready",
    )
    _launch(local_kernel, local_args, static_workers)

    orchestrator_outputs = _allocate_outputs(args)
    orchestrator_state = _allocate_orchestrator_state(
        geometry,
        programs,
        args.arrival_stride,
    )
    orchestrator_args = _orchestrator_arguments(
        inputs,
        orchestrator_outputs,
        orchestrator_state,
        geometry,
        args,
        num_sm,
        programs,
    )
    orchestrator_binary = _compile_and_check(
        orchestrator_kernel,
        orchestrator_args,
        programs,
        "orchestrator",
    )
    _launch(orchestrator_kernel, orchestrator_args, programs)

    ffn_q, ffn_scale, w13_q, w13_scale, w2_q, w2_scale = inputs
    _, w13 = compile_config(
        block_fp8_mm,
        (ffn_q, ffn_scale, w13_q, w13_scale, args.group),
        FFN_CONFIGS["w13"],
    )
    separate_gate = w13(ffn_q, ffn_scale, w13_q, w13_scale, args.group)
    _, activation = compile_config(
        silu_and_mul_per_block_quant,
        (separate_gate, args.group),
        FFN_CONFIGS["silu_quant"],
    )
    separate_q, separate_scale = activation(separate_gate, args.group)
    _, w2 = compile_config(
        block_fp8_mm,
        (separate_q, separate_scale, w2_q, w2_scale, args.group),
        FFN_CONFIGS["w2"],
    )
    separate_output = w2(separate_q, separate_scale, w2_q, w2_scale, args.group)
    torch.cuda.synchronize()

    static_tensors = (static_output, static_gate, static_q, static_scale)
    for candidate_name, outputs in (
        ("local_on_ready", local_outputs),
        ("orchestrator", orchestrator_outputs),
    ):
        for tensor_name, actual, expected in zip(
            ("output", "gate_up", "activation_q", "activation_scale"),
            (
                outputs["output"],
                outputs["gate_up"],
                outputs["activation_q"],
                outputs["activation_scale"],
            ),
            static_tensors,
            strict=True,
        ):
            _assert_exact(f"{candidate_name}_{tensor_name}_vs_static", actual, expected)
    for tensor_name, actual, expected in (
        ("gate_up", static_gate, separate_gate),
        ("activation_q", static_q, separate_q),
        ("activation_scale", static_scale, separate_scale),
        ("output", static_output, separate_output),
    ):
        _assert_close(f"static_{tensor_name}_vs_separate", actual, expected)

    static_graph, static_graph_output = capture(lambda: static_compiled(*kernel_args))
    local_graph, _ = capture(
        lambda: (
            _launch(local_kernel, local_args, static_workers),
            local_outputs["output"],
        )[1]
    )
    orchestrator_graph, _ = capture(
        lambda: (
            _launch(orchestrator_kernel, orchestrator_args, programs),
            orchestrator_outputs["output"],
        )[1]
    )

    def launch_separate():
        gate = w13(ffn_q, ffn_scale, w13_q, w13_scale, args.group)
        quant, scale = activation(gate, args.group)
        return w2(quant, scale, w2_q, w2_scale, args.group)

    separate_graph, separate_graph_output = capture(launch_separate)
    for _ in range(args.correctness_replays):
        orchestrator_graph.replay()
    local_graph.replay()
    static_graph.replay()
    separate_graph.replay()
    torch.cuda.synchronize()
    _assert_exact(
        "orchestrator_replay_vs_static",
        orchestrator_outputs["output"],
        static_graph_output[0],
    )
    _assert_exact(
        "local_replay_vs_static",
        local_outputs["output"],
        static_graph_output[0],
    )
    _assert_close(
        "static_replay_vs_separate",
        static_graph_output[0],
        separate_graph_output,
    )

    pids = visible_gpu_pids()
    if not args.allow_busy and (foreign_pids := pids - {os.getpid()}):
        raise RuntimeError(
            f"GPU gained foreign compute processes {sorted(foreign_pids)}"
        )
    timings = benchmark_interleaved(
        {
            "static_schedule": static_graph.replay,
            "local_on_ready": local_graph.replay,
            "one_cta_orchestrator": orchestrator_graph.replay,
            "standalone_helion_graph": separate_graph.replay,
        },
        args.repeats,
        args.batch_replays,
    )
    if visible_gpu_pids() != pids:
        raise RuntimeError("GPU process set changed during benchmark")

    static_us = timings["static_schedule"]["median_us"]
    standalone_us = timings["standalone_helion_graph"]["median_us"]
    for name in ("local_on_ready", "one_cta_orchestrator"):
        value = timings[name]["median_us"]
        timings[name]["reduction_vs_static_pct"] = (
            100.0 * (static_us - value) / static_us
        )
        timings[name]["reduction_vs_standalone_pct"] = (
            100.0 * (standalone_us - value) / standalone_us
        )

    result = {
        "device": torch.cuda.get_device_name(),
        "shape": {
            "batch": args.batch,
            "hidden": args.hidden,
            "intermediate": args.intermediate,
            "group": args.group,
        },
        "schedule": {
            "static_workers": static_workers,
            "orchestrator_programs": programs,
            "orchestrator_workers": programs - 1,
            "activation_replicas": args.activation_replicas,
            "arrival_stride": args.arrival_stride,
            "dynamic_w2": args.dynamic_w2,
            "central_publication": args.central_publication,
            "central_scan": args.central_scan,
            "ready_group_size": args.ready_group_size,
            "completion_batch": args.completion_batch,
            "orchestrator_waits": args.orchestrator_waits,
            "first_activation_tasks": geometry["first_activation_tasks"],
            "task_counts": {
                "w13": geometry["w13_tasks"],
                "activation": geometry["activation_tasks"],
                "w2": geometry["w2_tasks"],
            },
        },
        "timings": timings,
        "resources": {
            "static_schedule": _helion_resources(static_compiled),
            "local_on_ready": _triton_resources(local_binary, 1),
            "one_cta_orchestrator": _triton_resources(orchestrator_binary, 1),
        },
        "lowered": str(lowered_output),
    }
    print("RESULT_JSON", json.dumps(result, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=12288)
    parser.add_argument("--group", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--batch-replays", type=int, default=20)
    parser.add_argument("--correctness-replays", type=int, default=20)
    parser.add_argument("--w13-stages", type=int, default=4)
    parser.add_argument("--w13-unroll", type=int, default=2)
    parser.add_argument("--w13-block-n", type=int, default=16)
    parser.add_argument("--w2-stages", type=int, default=4)
    parser.add_argument("--w2-unroll", type=int, default=4)
    parser.add_argument("--w2-block-n", type=int, default=8)
    parser.add_argument("--kernel-stages", type=int, default=2)
    parser.add_argument("--worker-multiplier", type=int, default=8)
    parser.add_argument("--cross-loop-workers", type=int)
    parser.add_argument("--evict-first", type=int, action="append", default=[])
    parser.add_argument("--evict-last", type=int, action="append", default=[])
    parser.add_argument(
        "--programs",
        type=int,
        default=0,
        help="total resident CTAs including the orchestrator; 0 uses the static grid",
    )
    parser.add_argument("--activation-replicas", type=int, default=1)
    parser.add_argument("--arrival-stride", type=int, choices=(1, 32), default=1)
    parser.add_argument(
        "--dynamic-w2",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--central-publication",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--central-scan",
        choices=(
            "vector",
            "ordered",
            "sharded",
            "grouped",
            "bitmask",
            "worklist",
            "completion_queue",
            "mailbox",
            "direct",
        ),
        default="mailbox",
    )
    parser.add_argument(
        "--ready-group-size",
        type=int,
        choices=(1, 2, 4, 8, 16, 32),
        default=8,
    )
    parser.add_argument(
        "--completion-batch",
        type=int,
        choices=(1, 2, 4, 8, 16, 32),
        default=4,
    )
    parser.add_argument(
        "--orchestrator-waits",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--poll-delay", type=int, default=32)
    parser.add_argument("--allow-busy", action="store_true")
    parser.add_argument(
        "--lowered-output",
        default="/tmp/qwen3_ffn_orchestrator_lowered.py",
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
