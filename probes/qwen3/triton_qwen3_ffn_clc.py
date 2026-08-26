# ruff: noqa: ANN001, ANN202
# pyrefly: ignore-errors
"""Blackwell Cluster Launch Control probe for the Qwen3 FP8 FFN.

The arithmetic tile bodies are the same Helion-generated W13, SiLU/quant, and
W2 roots used by the static tile-dependency schedule.  The logical grid has one
CTA ID per tile.  Every physically launched CTA issues a CLC cancellation while
it executes its current tile, then adopts the canceled CTA ID as its next tile.

W13 publishes each completed fragment with a release atomic.  SwiGLU tiles and
the two W2 K-ranges consume their dependency counters with acquire loads.  The
CLC responses and mbarriers live in a scoped bank of 32-byte static
shared-memory slots, separate from Triton's compiler-managed dynamic shared
memory.  The static reservation is tunable so the CLC worker pool can match the
resident worker count used by the static schedule without changing tile math.
"""

from __future__ import annotations

import argparse
import ast
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
from probes.qwen3.triton_qwen3_ffn_orchestrator import _allocate_inputs
from probes.qwen3.triton_qwen3_ffn_orchestrator import _allocate_outputs
from probes.qwen3.triton_qwen3_ffn_orchestrator import _assert_close
from probes.qwen3.triton_qwen3_ffn_orchestrator import _assert_exact
from probes.qwen3.triton_qwen3_ffn_orchestrator import _geometry
from probes.qwen3.triton_qwen3_ffn_orchestrator import _triton_resources

CLC_SOURCE = r'''
@triton.jit
def _qwen_clc_issue_cancel():
    return tl.inline_asm_elementwise(
        asm=r"""
        {
            .reg .pred leader;
            .reg .b32 response_addr, mbar_addr, thread_id;
            .shared .align 16 .b8 qwen_clc_scratch[__CLC_STATIC_SHARED_BYTES__];

            mov.u32 response_addr, qwen_clc_scratch;
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
def _qwen_clc_wait_cancel(response_addr):
    success, canceled_x = tl.inline_asm_elementwise(
        asm=r"""
        {
            .reg .pred complete, canceled;
            .reg .b32 response_addr, mbar_addr, success, canceled_x;
            .reg .b128 response;

            mov.u32 response_addr, $2;
            add.u32 mbar_addr, response_addr, 16;
            mov.u32 success, 0;
            mov.u32 canceled_x, 0xffffffff;

        QWEN_CLC_WAIT:
            mbarrier.try_wait.parity.relaxed.cta.shared.b64 complete, [mbar_addr], 0;
            @!complete bra QWEN_CLC_WAIT;

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
    return success, canceled_x


@triton.jit
def _qwen_clc_load_acquire(address):
    return tl.inline_asm_elementwise(
        asm="ld.acquire.gpu.global.u32 $0, [$1];",
        constraints="=r,l",
        args=[address],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _qwen_clc_sync_warp():
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
def _qwen_clc_nanosleep(DELAY: tl.constexpr):
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
def _qwen_clc_wait_count(address, target, POLL_DELAY: tl.constexpr):
    value = _qwen_clc_load_acquire(address)
    while value < target:
        if POLL_DELAY:
            _qwen_clc_nanosleep(POLL_DELAY)
        value = _qwen_clc_load_acquire(address)
    _qwen_clc_sync_warp()


@triton.jit
def _qwen_clc_w13_physical_task(
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
def _qwen_clc_w13_tile(
    ffn_q,
    w13_q,
    ffn_scale,
    w13_scale,
    gate_up,
    w13_arrivals,
    logical_task,
    SUBTILES_PER_ACTIVATION: tl.constexpr,
    HALF_TASKS: tl.constexpr,
    ARRIVAL_STRIDE: tl.constexpr,
):
    physical_task = _qwen_clc_w13_physical_task(
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
    fan_in: tl.constexpr = 2 * SUBTILES_PER_ACTIVATION
    activation_task = logical_task // fan_in
    tl.atomic_add(
        w13_arrivals + activation_task * ARRIVAL_STRIDE,
        1,
        sem="release",
        scope="gpu",
    )


@triton.jit(noinline=True)
def _qwen_clc_activation_tile(
    gate_up,
    activation_scale,
    activation_q,
    w13_arrivals,
    dependency_state,
    activation_task,
    epoch,
    ROOT_1_OFFSET: tl.constexpr,
    FIRST_ACTIVATION_TASKS: tl.constexpr,
    SUBTILES_PER_ACTIVATION: tl.constexpr,
    SPLIT_BASE: tl.constexpr,
    ARRIVAL_STRIDE: tl.constexpr,
    POLL_DELAY: tl.constexpr,
    FP8_MAX_VALUE: tl.constexpr,
    FP8_MIN_SCALE_VALUE: tl.constexpr,
    FP8_MIN_VALUE: tl.constexpr,
):
    fan_in: tl.constexpr = 2 * SUBTILES_PER_ACTIVATION
    _qwen_clc_wait_count(
        w13_arrivals + activation_task * ARRIVAL_STRIDE,
        epoch * fan_in,
        POLL_DELAY,
    )
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


@triton.jit
def _qwen_clc_w2_tile(
    activation_scale,
    activation_q,
    w2_q,
    w2_scale,
    output,
    dependency_state,
    w2_done,
    completed_epoch,
    w2_task,
    epoch,
    NUM_SM: tl.constexpr,
    ROOT_2_OFFSET: tl.constexpr,
    W2_TASKS: tl.constexpr,
):
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
    tl.debug_barrier()
    previous = tl.atomic_add(
        w2_done,
        1,
        sem="acq_rel",
        scope="gpu",
    )
    if previous + 1 == epoch * W2_TASKS:
        tl.atomic_xchg(
            completed_epoch,
            epoch,
            sem="release",
            scope="gpu",
        )


@triton.jit
def _qwen_clc_gate_publish_tile(
    ffn_q,
    w13_q,
    ffn_scale,
    w13_scale,
    gate_up,
    activation_scale,
    activation_q,
    w13_arrivals,
    dependency_state,
    logical_task,
    epoch,
    ROOT_1_OFFSET: tl.constexpr,
    FIRST_ACTIVATION_TASKS: tl.constexpr,
    SUBTILES_PER_ACTIVATION: tl.constexpr,
    HALF_TASKS: tl.constexpr,
    SPLIT_BASE: tl.constexpr,
    ARRIVAL_STRIDE: tl.constexpr,
    FP8_MAX_VALUE: tl.constexpr,
    FP8_MIN_SCALE_VALUE: tl.constexpr,
    FP8_MIN_VALUE: tl.constexpr,
):
    fan_in: tl.constexpr = 2 * SUBTILES_PER_ACTIVATION
    physical_task = _qwen_clc_w13_physical_task(
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


@triton.jit
def qwen3_ffn_clc_one_handoff(
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
    w13_arrivals,
    dependency_state,
    w2_done,
    completed_epoch,
    started,
    processed,
    successful_cancels,
    canceled_successor,
    NUM_SM: tl.constexpr,
    LOGICAL_GRID: tl.constexpr,
    W13_TASKS: tl.constexpr,
    ACTIVATION_TASKS: tl.constexpr,
    W2_TASKS: tl.constexpr,
    W13_HEAD_TASKS: tl.constexpr,
    TAIL_W13_ORDER: tl.constexpr,
    ROOT_1_OFFSET: tl.constexpr,
    ROOT_2_OFFSET: tl.constexpr,
    SPLIT_BASE: tl.constexpr,
    FIRST_ACTIVATION_TASKS: tl.constexpr,
    SUBTILES_PER_ACTIVATION: tl.constexpr,
    HALF_TASKS: tl.constexpr,
    ARRIVAL_STRIDE: tl.constexpr,
    POLL_DELAY: tl.constexpr,
    TRACKING: tl.constexpr,
    FP8_MAX_VALUE: tl.constexpr,
    FP8_MIN_SCALE_VALUE: tl.constexpr,
    FP8_MIN_VALUE: tl.constexpr,
):
    initial_task = tl.program_id(0)
    lanes = tl.arange(0, 32)
    lane_zero = lanes == 0
    if TRACKING:
        tl.store(started + initial_task + lanes, 1, mask=lane_zero)

    epoch = _qwen_clc_load_acquire(completed_epoch) + 1
    response_addr = _qwen_clc_issue_cancel()
    _qwen_clc_gate_publish_tile(
        ffn_q,
        w13_q,
        ffn_scale,
        w13_scale,
        gate_up,
        activation_scale,
        activation_q,
        w13_arrivals,
        dependency_state,
        initial_task,
        epoch,
        ROOT_1_OFFSET,
        FIRST_ACTIVATION_TASKS,
        SUBTILES_PER_ACTIVATION,
        HALF_TASKS,
        SPLIT_BASE,
        ARRIVAL_STRIDE,
        FP8_MAX_VALUE,
        FP8_MIN_SCALE_VALUE,
        FP8_MIN_VALUE,
    )
    success, canceled_task = _qwen_clc_wait_cancel(response_addr)
    canceled_task = canceled_task.to(tl.int32)

    if TRACKING:
        tl.atomic_add(
            processed + initial_task + lanes,
            1,
            mask=lane_zero,
            sem="relaxed",
            scope="gpu",
        )
    if success:
        if TRACKING:
            tl.store(
                canceled_successor + initial_task + lanes,
                canceled_task,
                mask=lane_zero,
            )
            tl.atomic_add(
                successful_cancels + lanes,
                1,
                mask=lane_zero,
                sem="relaxed",
                scope="gpu",
            )
        if canceled_task < W13_TASKS:
            _qwen_clc_gate_publish_tile(
                ffn_q,
                w13_q,
                ffn_scale,
                w13_scale,
                gate_up,
                activation_scale,
                activation_q,
                w13_arrivals,
                dependency_state,
                canceled_task,
                epoch,
                ROOT_1_OFFSET,
                FIRST_ACTIVATION_TASKS,
                SUBTILES_PER_ACTIVATION,
                HALF_TASKS,
                SPLIT_BASE,
                ARRIVAL_STRIDE,
                FP8_MAX_VALUE,
                FP8_MIN_SCALE_VALUE,
                FP8_MIN_VALUE,
            )
        else:
            _qwen_clc_w2_tile(
                activation_scale,
                activation_q,
                w2_q,
                w2_scale,
                output,
                dependency_state,
                w2_done,
                completed_epoch,
                canceled_task - W13_TASKS,
                epoch,
                NUM_SM,
                ROOT_2_OFFSET,
                W2_TASKS,
            )
        if TRACKING:
            tl.atomic_add(
                processed + canceled_task + lanes,
                1,
                mask=lane_zero,
                sem="relaxed",
                scope="gpu",
            )


@triton.jit
def qwen3_ffn_clc(
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
    w13_arrivals,
    dependency_state,
    w2_done,
    completed_epoch,
    started,
    processed,
    successful_cancels,
    canceled_successor,
    NUM_SM: tl.constexpr,
    LOGICAL_GRID: tl.constexpr,
    W13_TASKS: tl.constexpr,
    ACTIVATION_TASKS: tl.constexpr,
    W2_TASKS: tl.constexpr,
    W13_HEAD_TASKS: tl.constexpr,
    TAIL_W13_ORDER: tl.constexpr,
    ROOT_1_OFFSET: tl.constexpr,
    ROOT_2_OFFSET: tl.constexpr,
    SPLIT_BASE: tl.constexpr,
    FIRST_ACTIVATION_TASKS: tl.constexpr,
    SUBTILES_PER_ACTIVATION: tl.constexpr,
    HALF_TASKS: tl.constexpr,
    ARRIVAL_STRIDE: tl.constexpr,
    POLL_DELAY: tl.constexpr,
    TRACKING: tl.constexpr,
    FP8_MAX_VALUE: tl.constexpr,
    FP8_MIN_SCALE_VALUE: tl.constexpr,
    FP8_MIN_VALUE: tl.constexpr,
):
    initial_task = tl.program_id(0)
    lanes = tl.arange(0, 32)
    lane_zero = lanes == 0
    if TRACKING:
        tl.store(started + initial_task + lanes, 1, mask=lane_zero)

    epoch = _qwen_clc_load_acquire(completed_epoch) + 1
    current_task = initial_task
    active = tl.full([], 1, tl.int1)
    task_iteration = tl.full([], 0, tl.int32)
    while active:
        response_addr = _qwen_clc_issue_cancel()

        if TAIL_W13_ORDER:
            activation_base: tl.constexpr = W13_HEAD_TASKS
            w2_base: tl.constexpr = activation_base + ACTIVATION_TASKS
            w13_tail_base: tl.constexpr = w2_base + W2_TASKS
            if current_task < W13_HEAD_TASKS:
                _qwen_clc_w13_tile(
                    ffn_q,
                    w13_q,
                    ffn_scale,
                    w13_scale,
                    gate_up,
                    w13_arrivals,
                    current_task,
                    SUBTILES_PER_ACTIVATION,
                    HALF_TASKS,
                    ARRIVAL_STRIDE,
                )
            elif current_task < w2_base:
                _qwen_clc_activation_tile(
                    gate_up,
                    activation_scale,
                    activation_q,
                    w13_arrivals,
                    dependency_state,
                    current_task - activation_base,
                    epoch,
                    ROOT_1_OFFSET,
                    FIRST_ACTIVATION_TASKS,
                    SUBTILES_PER_ACTIVATION,
                    SPLIT_BASE,
                    ARRIVAL_STRIDE,
                    POLL_DELAY,
                    FP8_MAX_VALUE,
                    FP8_MIN_SCALE_VALUE,
                    FP8_MIN_VALUE,
                )
            elif current_task < w13_tail_base:
                _qwen_clc_w2_tile(
                    activation_scale,
                    activation_q,
                    w2_q,
                    w2_scale,
                    output,
                    dependency_state,
                    w2_done,
                    completed_epoch,
                    current_task - w2_base,
                    epoch,
                    NUM_SM,
                    ROOT_2_OFFSET,
                    W2_TASKS,
                )
            else:
                _qwen_clc_w13_tile(
                    ffn_q,
                    w13_q,
                    ffn_scale,
                    w13_scale,
                    gate_up,
                    w13_arrivals,
                    W13_HEAD_TASKS + current_task - w13_tail_base,
                    SUBTILES_PER_ACTIVATION,
                    HALF_TASKS,
                    ARRIVAL_STRIDE,
                )
        else:
            if current_task < W13_TASKS:
                _qwen_clc_w13_tile(
                    ffn_q,
                    w13_q,
                    ffn_scale,
                    w13_scale,
                    gate_up,
                    w13_arrivals,
                    current_task,
                    SUBTILES_PER_ACTIVATION,
                    HALF_TASKS,
                    ARRIVAL_STRIDE,
                )
            elif current_task < W13_TASKS + ACTIVATION_TASKS:
                _qwen_clc_activation_tile(
                    gate_up,
                    activation_scale,
                    activation_q,
                    w13_arrivals,
                    dependency_state,
                    current_task - W13_TASKS,
                    epoch,
                    ROOT_1_OFFSET,
                    FIRST_ACTIVATION_TASKS,
                    SUBTILES_PER_ACTIVATION,
                    SPLIT_BASE,
                    ARRIVAL_STRIDE,
                    POLL_DELAY,
                    FP8_MAX_VALUE,
                    FP8_MIN_SCALE_VALUE,
                    FP8_MIN_VALUE,
                )
            else:
                _qwen_clc_w2_tile(
                    activation_scale,
                    activation_q,
                    w2_q,
                    w2_scale,
                    output,
                    dependency_state,
                    w2_done,
                    completed_epoch,
                    current_task - W13_TASKS - ACTIVATION_TASKS,
                    epoch,
                    NUM_SM,
                    ROOT_2_OFFSET,
                    W2_TASKS,
                )

        success_lanes, canceled_lanes = _qwen_clc_wait_cancel(response_addr)
        success = tl.max(success_lanes, axis=0) != 0
        canceled_task = tl.min(canceled_lanes, axis=0).to(tl.int32)
        if TRACKING:
            tl.atomic_add(
                processed + current_task + lanes,
                1,
                mask=lane_zero,
                sem="relaxed",
                scope="gpu",
            )
            tl.store(
                canceled_successor + current_task + lanes,
                canceled_task,
                mask=lane_zero & success,
            )
            tl.atomic_add(
                successful_cancels + lanes,
                1,
                mask=lane_zero & success,
                sem="relaxed",
                scope="gpu",
            )
        current_task = canceled_task
        active = success & (canceled_task < LOGICAL_GRID)
        task_iteration += 1
'''


def _generated_namespace(bound, config, lowered_output: Path, clc_static_bytes: int):
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
    clc_source = CLC_SOURCE.replace(
        "__ROOT_2_NUM_SM_ARGUMENT__",
        "NUM_SM," if root_2_has_num_sm else "",
    )
    clc_source = clc_source.replace(
        "__CLC_STATIC_SHARED_BYTES__", str(clc_static_bytes)
    )
    source = prefix + "\n" + textwrap.dedent(clc_source)
    filename = str(Path(__file__).with_name("_generated_qwen3_ffn_clc.py"))
    linecache.cache[filename] = (
        len(source),
        None,
        source.splitlines(keepends=True),
        filename,
    )
    namespace: dict[str, object] = {"__name__": "_generated_qwen3_ffn_clc"}
    exec(compile(source, filename, "exec"), namespace)
    lowered_output.write_text(source)
    return namespace, source


def _allocate_clc_state(
    geometry, arrival_stride: int, logical_grid: int
) -> dict[str, torch.Tensor]:
    return {
        "w13_arrivals": torch.zeros(
            (geometry["activation_tasks"] - 1) * arrival_stride + 1,
            device="cuda",
            dtype=torch.uint32,
        ),
        "dependency_state": torch.zeros(
            geometry["state_size"], device="cuda", dtype=torch.uint32
        ),
        "w2_done": torch.zeros(1, device="cuda", dtype=torch.uint32),
        "completed_epoch": torch.zeros(1, device="cuda", dtype=torch.uint32),
        "started": torch.zeros(logical_grid, device="cuda", dtype=torch.uint32),
        "processed": torch.zeros(logical_grid, device="cuda", dtype=torch.uint32),
        "successful_cancels": torch.zeros(1, device="cuda", dtype=torch.uint32),
        "canceled_successor": torch.full(
            (logical_grid,), -1, device="cuda", dtype=torch.int32
        ),
    }


def _clc_arguments(
    inputs, outputs, state, geometry, args, logical_grid: int, tracking: bool
):
    ffn_q, ffn_scale, w13_q, w13_scale, w2_q, w2_scale = inputs
    num_sm = torch.cuda.get_device_properties(0).multi_processor_count
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
        state["w13_arrivals"],
        state["dependency_state"],
        state["w2_done"],
        state["completed_epoch"],
        state["started"],
        state["processed"],
        state["successful_cancels"],
        state["canceled_successor"],
        num_sm,
        logical_grid,
        geometry["w13_tasks"],
        geometry["activation_tasks"],
        geometry["w2_tasks"],
        args.w13_head_tasks,
        args.task_order == "tail-w13",
        geometry["root_1_offset"],
        geometry["root_2_offset"],
        geometry["split_base"],
        geometry["first_activation_tasks"],
        geometry["subtiles_per_activation"],
        geometry["half_tasks"],
        args.arrival_stride,
        args.poll_delay,
        tracking,
        FP8_MAX,
        FP8_MIN_SCALE,
        FP8_MIN,
    )


def _compile(kernel, kernel_args, logical_grid: int, args):
    return kernel.warmup(
        *kernel_args,
        grid=(logical_grid,),
        num_warps=args.num_warps,
        num_stages=args.kernel_stages,
        num_ctas=1,
        launch_pdl=True,
    )


def _launch(kernel, kernel_args, logical_grid: int, args):
    return kernel[(logical_grid,)](
        *kernel_args,
        num_warps=args.num_warps,
        num_stages=args.kernel_stages,
        num_ctas=1,
        launch_pdl=True,
    )


def _clc_resources(
    compiled, num_warps: int, clc_static_bytes: int
) -> dict[str, int | bool | str]:
    resources = _triton_resources(compiled, num_warps)
    ptx = compiled.asm["ptx"]
    resources.update(
        {
            "triton_dynamic_shared_bytes": int(compiled.metadata.shared),
            "inline_ptx_static_shared_bytes": clc_static_bytes,
            "expected_total_shared_bytes": (
                int(compiled.metadata.shared) + clc_static_bytes
            ),
            "uses_scoped_clc_scratch": (
                f".shared .align 16 .b8 qwen_clc_scratch[{clc_static_bytes}]" in ptx
            ),
            "uses_triton_global_smem": (
                ".extern .shared .align 16 .b8 global_smem[]" in ptx
            ),
            "contains_try_cancel": "clusterlaunchcontrol.try_cancel" in ptx,
            "contains_query_cancel": "clusterlaunchcontrol.query_cancel" in ptx,
            "ptx_version_line": next(
                line for line in ptx.splitlines() if line.startswith(".version")
            ),
            "ptx_target_line": next(
                line for line in ptx.splitlines() if line.startswith(".target")
            ),
        }
    )
    return resources


def _verify_clc_accounting(state, logical_grid: int, w13_tasks: int) -> dict[str, int]:
    started_count = int(state["started"].sum().item())
    cancel_count = int(state["successful_cancels"].item())
    processed = state["processed"]
    missing = int((processed == 0).sum().item())
    duplicated = int((processed != 1).sum().item()) - missing
    successors = state["canceled_successor"]
    invalid = int(((successors >= logical_grid) & (successors != -1)).sum().item())
    started_downstream = int(state["started"][w13_tasks:].sum().item())
    result = {
        "physically_started_ctas": started_count,
        "successful_cancels": cancel_count,
        "started_plus_canceled": started_count + cancel_count,
        "missing_logical_tiles": missing,
        "duplicated_logical_tiles": duplicated,
        "invalid_canceled_ids": invalid,
        "physically_started_downstream_tiles": started_downstream,
    }
    if started_count + cancel_count != logical_grid:
        raise AssertionError("physical starts plus successful cancels != logical grid")
    if missing or duplicated or invalid or started_downstream:
        raise AssertionError(f"CLC tile accounting failed: {result}")
    return result


def run(args) -> None:
    if not args.allow_busy:
        require_idle_visible_gpu()
    properties = torch.cuda.get_device_properties(0)
    if properties.major < 10:
        raise RuntimeError("Blackwell CLC requires compute capability 10.0+")

    inputs = _allocate_inputs(args)
    kernel_args = (*inputs, args.group)
    bound = qwen3_ffn_tile_dependency.bind(kernel_args)
    config = _persistent_config(bound, args)
    namespace, _ = _generated_namespace(
        bound, config, Path(args.lowered_output), args.clc_static_bytes
    )
    clc_kernel = namespace["qwen3_ffn_clc_one_handoff"]
    num_sm = properties.multi_processor_count
    static_workers = args.cross_loop_workers or num_sm * args.worker_multiplier
    geometry = _geometry(namespace, args, static_workers)
    fan_in = 2 * geometry["subtiles_per_activation"]
    if not 0 < args.w13_head_tasks < geometry["w13_tasks"]:
        raise ValueError("--w13-head-tasks must split the W13 task range")
    if args.w13_head_tasks % fan_in:
        raise ValueError("--w13-head-tasks must end on an activation fan-in group")
    logical_grid = geometry["w13_tasks"] + geometry["w2_tasks"]

    static_compiled = bound.compile_config(config)
    static_output, static_gate, static_q, static_scale = static_compiled(*kernel_args)

    tracking_outputs = _allocate_outputs(args)
    tracking_state = _allocate_clc_state(geometry, args.arrival_stride, logical_grid)
    tracking_args = _clc_arguments(
        inputs,
        tracking_outputs,
        tracking_state,
        geometry,
        args,
        logical_grid,
        True,
    )
    tracking_binary = _compile(clc_kernel, tracking_args, logical_grid, args)
    _launch(clc_kernel, tracking_args, logical_grid, args)

    clc_outputs = _allocate_outputs(args)
    clc_state = _allocate_clc_state(geometry, args.arrival_stride, logical_grid)
    clc_args = _clc_arguments(
        inputs,
        clc_outputs,
        clc_state,
        geometry,
        args,
        logical_grid,
        False,
    )
    clc_binary = _compile(clc_kernel, clc_args, logical_grid, args)
    _launch(clc_kernel, clc_args, logical_grid, args)

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

    clc_accounting = _verify_clc_accounting(
        tracking_state, logical_grid, geometry["w13_tasks"]
    )
    static_tensors = (static_output, static_gate, static_q, static_scale)
    for candidate_name, outputs in (
        ("clc_tracking", tracking_outputs),
        ("clc", clc_outputs),
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
    clc_graph, _ = capture(
        lambda: (
            _launch(clc_kernel, clc_args, logical_grid, args),
            clc_outputs["output"],
        )[1]
    )

    def launch_separate():
        gate = w13(ffn_q, ffn_scale, w13_q, w13_scale, args.group)
        quant, scale = activation(gate, args.group)
        return w2(quant, scale, w2_q, w2_scale, args.group)

    separate_graph, separate_graph_output = capture(launch_separate)
    for _ in range(args.correctness_replays):
        clc_graph.replay()
    static_graph.replay()
    separate_graph.replay()
    torch.cuda.synchronize()
    _assert_exact("clc_replay_vs_static", clc_outputs["output"], static_graph_output[0])
    _assert_close(
        "static_replay_vs_separate", static_graph_output[0], separate_graph_output
    )

    pids = visible_gpu_pids()
    if not args.allow_busy and (foreign_pids := pids - {os.getpid()}):
        raise RuntimeError(
            f"GPU gained foreign compute processes {sorted(foreign_pids)}"
        )
    timings = benchmark_interleaved(
        {
            "static_schedule": static_graph.replay,
            "clc_tile_scheduler": clc_graph.replay,
            "standalone_helion_graph": separate_graph.replay,
        },
        args.repeats,
        args.batch_replays,
    )
    if visible_gpu_pids() != pids:
        raise RuntimeError("GPU process set changed during benchmark")

    static_us = timings["static_schedule"]["median_us"]
    standalone_us = timings["standalone_helion_graph"]["median_us"]
    clc_us = timings["clc_tile_scheduler"]["median_us"]
    timings["clc_tile_scheduler"]["reduction_vs_static_pct"] = (
        100.0 * (static_us - clc_us) / static_us
    )
    timings["clc_tile_scheduler"]["reduction_vs_standalone_pct"] = (
        100.0 * (standalone_us - clc_us) / standalone_us
    )

    tracking_resources = _clc_resources(
        tracking_binary, args.num_warps, args.clc_static_bytes
    )
    clc_resources = _clc_resources(clc_binary, args.num_warps, args.clc_static_bytes)
    if not clc_resources["uses_scoped_clc_scratch"]:
        raise AssertionError("generated PTX does not contain scoped CLC scratch")
    if not clc_resources["contains_try_cancel"]:
        raise AssertionError("generated PTX does not contain CLC cancellation")
    if not clc_resources["contains_query_cancel"]:
        raise AssertionError("generated PTX does not contain CLC query")

    result = {
        "device": torch.cuda.get_device_name(),
        "shape": {
            "batch": args.batch,
            "hidden": args.hidden,
            "intermediate": args.intermediate,
            "group": args.group,
        },
        "schedule": {
            "logical_grid": logical_grid,
            "static_workers": static_workers,
            "num_warps": args.num_warps,
            "task_order": args.task_order,
            "w13_head_tasks": args.w13_head_tasks,
            "arrival_stride": args.arrival_stride,
            "clc_static_bytes": args.clc_static_bytes,
            "poll_delay": args.poll_delay,
            "first_activation_tasks": geometry["first_activation_tasks"],
            "handoffs_per_physical_cta": 1,
            "task_counts": {
                "w13": geometry["w13_tasks"],
                "activation_inline": geometry["activation_tasks"],
                "w2": geometry["w2_tasks"],
            },
        },
        "clc_accounting": clc_accounting,
        "timings": timings,
        "resources": {
            "static_schedule": _helion_resources(static_compiled),
            "clc_tracking": tracking_resources,
            "clc_tile_scheduler": clc_resources,
        },
        "lowered": str(Path(args.lowered_output).resolve()),
    }
    print("RESULT_JSON", json.dumps(result, sort_keys=True), flush=True)
    if args.output:
        output = Path(args.output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        output.with_suffix(".ptx").write_text(clc_binary.asm["ptx"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--num-warps", type=int, choices=(1, 2, 4, 8), default=1)
    parser.add_argument(
        "--clc-static-bytes",
        type=int,
        choices=(256, 4096, 8192, 12288, 16384, 20480),
        default=12288,
    )
    parser.add_argument("--task-order", choices=("stage", "tail-w13"), default="stage")
    parser.add_argument("--w13-head-tasks", type=int, default=1024)
    parser.add_argument("--worker-multiplier", type=int, default=8)
    parser.add_argument("--cross-loop-workers", type=int)
    parser.add_argument("--evict-first", type=int, action="append", default=[])
    parser.add_argument("--evict-last", type=int, action="append", default=[])
    parser.add_argument("--arrival-stride", type=int, choices=(1, 32), default=1)
    parser.add_argument("--poll-delay", type=int, default=0)
    parser.add_argument("--allow-busy", action="store_true")
    parser.add_argument("--lowered-output", default="/tmp/qwen3_ffn_clc_lowered.py")
    parser.add_argument("--output", default="")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
