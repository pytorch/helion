# ruff: noqa: ANN001, ANN202
# pyrefly: ignore-errors
"""CTA-0 tile-scheduler probes for one complete Qwen3 decode layer.

The Helion-generated operator bodies are retained unchanged.  CTA 0 consumes
tile-completion reports and publishes work to the other 1024 CTAs.  Selectable
ablation modes extend that scheduling from the FFN into QKV/attention using
group commands, compact worklists, individual tile descriptors, or a dynamic
claim queue.
"""

from __future__ import annotations

import argparse
import ast
import copy
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
from probes.qwen3 import helion_qwen3_granular_tile_dependency as granular
from probes.qwen3 import helion_qwen3_tile_dependency as layer_probe
from probes.qwen3.helion_qwen3_tile_dependency import _composite_args
from probes.qwen3.helion_qwen3_tile_dependency import _helion_resources
from probes.qwen3.helion_qwen3_tile_dependency import allocate_layer
from probes.qwen3.helion_qwen3_tile_dependency import build_helion_reference

import helion
import helion.runtime

OUTPUT_NAMES = (
    "output",
    "pre_q",
    "pre_scale",
    "qkv",
    "partial_out",
    "partial_lse",
    "attention",
    "attention_q",
    "attention_scale",
    "attention_out",
    "ffn_q",
    "ffn_scale",
    "gate_up",
    "activation_q",
    "activation_scale",
    "residual",
)

SCHEDULER_SOURCE = r"""
@triton.jit
def _qwen_layer_sync_warp():
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
def _qwen_layer_load_acquire(address):
    return tl.inline_asm_elementwise(
        asm="ld.acquire.gpu.global.u32 $0, [$1];",
        constraints="=r,l",
        args=[address],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _qwen_layer_nanosleep(DELAY: tl.constexpr):
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
def _qwen_layer_wait_epoch(address, epoch, POLL_DELAY: tl.constexpr):
    value = _qwen_layer_load_acquire(address)
    while value != epoch:
        if POLL_DELAY:
            _qwen_layer_nanosleep(POLL_DELAY)
        value = _qwen_layer_load_acquire(address)
    _qwen_layer_sync_warp()


@triton.jit
def _qwen_layer_wait_count(address, target, POLL_DELAY: tl.constexpr):
    value = _qwen_layer_load_acquire(address)
    while value < target:
        if POLL_DELAY:
            _qwen_layer_nanosleep(POLL_DELAY)
        value = _qwen_layer_load_acquire(address)
    _qwen_layer_sync_warp()


@triton.jit
def _qwen_layer_report_attention_group(
    tile_dependency_state,
    epoch,
    group,
    SCHED_BASE: tl.constexpr,
):
    attention_group_arrival_base: tl.constexpr = 197
    tl.atomic_add(
        tile_dependency_state
        + SCHED_BASE
        + attention_group_arrival_base
        + group,
        1,
        sem="release",
        scope="gpu",
    )


@triton.jit
def _qwen_layer_report_merge_chunk(
    tile_dependency_state,
    epoch,
    attention_task,
    SCHED_BASE: tl.constexpr,
):
    arrival_base: tl.constexpr = 221
    split = attention_task // 8
    group = attention_task % 8
    chunk = split // 8
    key = group * 16 + chunk
    tl.atomic_add(
        tile_dependency_state + SCHED_BASE + arrival_base + key,
        1,
        sem="release",
        scope="gpu",
    )


@triton.jit
def _qwen_layer_report_merge_head(
    tile_dependency_state,
    epoch,
    head,
    SCHED_BASE: tl.constexpr,
):
    arrival_base: tl.constexpr = 605
    tl.atomic_add(
        tile_dependency_state + SCHED_BASE + arrival_base + head,
        1,
        sem="release",
        scope="gpu",
    )


@triton.jit(noinline=True)
def _qwen_layer_schedule_ffn(
    tile_dependency_state,
    epoch,
    SCHED_BASE: tl.constexpr,
    ACTIVATION_TASKS: tl.constexpr,
    FIRST_ACTIVATION_TASKS: tl.constexpr,
    READY_BLOCK: tl.constexpr,
    POLL_DELAY: tl.constexpr,
):
    w13_ready_offset: tl.constexpr = 2
    w2_ready_offset: tl.constexpr = 3
    first_activation_count_offset: tl.constexpr = 4
    completion_base: tl.constexpr = 5
    command_base: tl.constexpr = completion_base + ACTIVATION_TASKS

    # The exact-1024 lowering ends with the 32-word post-quant counter.
    _qwen_layer_wait_count(
        tile_dependency_state + SCHED_BASE - 32,
        epoch * 32,
        POLL_DELAY,
    )
    tl.atomic_xchg(
        tile_dependency_state + SCHED_BASE + w13_ready_offset,
        epoch,
        sem="release",
        scope="gpu",
    )

    task = tl.arange(0, READY_BLOCK)
    valid = task < ACTIVATION_TASKS
    safe_task = tl.minimum(task, ACTIVATION_TASKS - 1)
    published = tl.zeros([READY_BLOCK], tl.int1)
    remaining = tl.full([], ACTIVATION_TASKS, tl.int32)
    w2_published = tl.full([], 0, tl.int32)
    while (remaining > 0) | (w2_published == 0):
        progressed = tl.full([], 0, tl.int32)
        if remaining > 0:
            completed = _qwen_layer_load_acquire(
                tile_dependency_state
                + SCHED_BASE
                + completion_base
                + safe_task
            )
            newly_ready = valid & ~published & (completed == epoch)
            ready_count = tl.sum(newly_ready.to(tl.int32), axis=0)
            if ready_count > 0:
                tl.atomic_xchg(
                    tile_dependency_state
                    + SCHED_BASE
                    + command_base
                    + task,
                    epoch,
                    mask=newly_ready,
                    sem="release",
                    scope="gpu",
                )
                published = published | newly_ready
                remaining -= ready_count
                progressed = 1
        if w2_published == 0:
            first_done = _qwen_layer_load_acquire(
                tile_dependency_state
                + SCHED_BASE
                + first_activation_count_offset
            )
            if first_done == epoch * FIRST_ACTIVATION_TASKS:
                tl.atomic_xchg(
                    tile_dependency_state + SCHED_BASE + w2_ready_offset,
                    epoch,
                    sem="release",
                    scope="gpu",
                )
                w2_published = 1
                progressed = 1
        if progressed == 0:
            if POLL_DELAY:
                _qwen_layer_nanosleep(POLL_DELAY)


@triton.jit(noinline=True)
def _qwen_layer_schedule_groups(
    tile_dependency_state,
    epoch,
    SCHED_BASE: tl.constexpr,
    ACTIVATION_TASKS: tl.constexpr,
    FIRST_ACTIVATION_TASKS: tl.constexpr,
    READY_BLOCK: tl.constexpr,
    POLL_DELAY: tl.constexpr,
):
    group = tl.arange(0, 8)
    arrival_base: tl.constexpr = 197
    command_base: tl.constexpr = 213
    published = tl.zeros([8], tl.int1)
    remaining = tl.full([], 8, tl.int32)
    while remaining > 0:
        completed = _qwen_layer_load_acquire(
            tile_dependency_state + SCHED_BASE + arrival_base + group
        )
        newly_ready = ~published & (completed >= epoch * 6)
        ready_count = tl.sum(newly_ready.to(tl.int32), axis=0)
        if ready_count > 0:
            tl.atomic_xchg(
                tile_dependency_state + SCHED_BASE + command_base + group,
                epoch,
                mask=newly_ready,
                sem="release",
                scope="gpu",
            )
            published = published | newly_ready
            remaining -= ready_count
        elif POLL_DELAY:
            _qwen_layer_nanosleep(POLL_DELAY)

    _qwen_layer_schedule_ffn(
        tile_dependency_state,
        epoch,
        SCHED_BASE,
        ACTIVATION_TASKS,
        FIRST_ACTIVATION_TASKS,
        READY_BLOCK,
        POLL_DELAY,
    )


@triton.jit(noinline=True)
def _qwen_layer_schedule_ready_order(
    tile_dependency_state,
    epoch,
    SCHED_BASE: tl.constexpr,
    ACTIVATION_TASKS: tl.constexpr,
    FIRST_ACTIVATION_TASKS: tl.constexpr,
    READY_BLOCK: tl.constexpr,
    POLL_DELAY: tl.constexpr,
):
    group = tl.arange(0, 8)
    arrival_base: tl.constexpr = 197
    worklist_base: tl.constexpr = 213
    tail_offset: tl.constexpr = 221
    published = tl.zeros([8], tl.int1)
    ready_groups = tl.full([], 0, tl.int32)
    epoch_base = (epoch - 1) * 8
    while ready_groups < 8:
        completed = _qwen_layer_load_acquire(
            tile_dependency_state + SCHED_BASE + arrival_base + group
        )
        newly_ready = ~published & (completed >= epoch * 6)
        ready_count = tl.sum(newly_ready.to(tl.int32), axis=0)
        if ready_count > 0:
            ready_rank = tl.cumsum(newly_ready.to(tl.int32), axis=0) - 1
            tl.store(
                tile_dependency_state
                + SCHED_BASE
                + worklist_base
                + ready_groups
                + ready_rank,
                group,
                mask=newly_ready,
            )
            tl.debug_barrier()
            published = published | newly_ready
            ready_groups += ready_count
            tl.atomic_xchg(
                tile_dependency_state + SCHED_BASE + tail_offset,
                epoch_base + ready_groups,
                sem="release",
                scope="gpu",
            )
        elif POLL_DELAY:
            _qwen_layer_nanosleep(POLL_DELAY)

    _qwen_layer_schedule_ffn(
        tile_dependency_state,
        epoch,
        SCHED_BASE,
        ACTIVATION_TASKS,
        FIRST_ACTIVATION_TASKS,
        READY_BLOCK,
        POLL_DELAY,
    )


@triton.jit(noinline=True)
def _qwen_layer_schedule_tiles(
    tile_dependency_state,
    epoch,
    SCHED_BASE: tl.constexpr,
    ACTIVATION_TASKS: tl.constexpr,
    FIRST_ACTIVATION_TASKS: tl.constexpr,
    READY_BLOCK: tl.constexpr,
    POLL_DELAY: tl.constexpr,
):
    group = tl.arange(0, 8)
    split = tl.arange(0, 128)
    arrival_base: tl.constexpr = 197
    worklist_base: tl.constexpr = 205
    tail_offset: tl.constexpr = 1229
    published = tl.zeros([8], tl.int1)
    ready_groups = tl.full([], 0, tl.int32)
    while ready_groups < 8:
        completed = _qwen_layer_load_acquire(
            tile_dependency_state + SCHED_BASE + arrival_base + group
        )
        newly_ready = ~published & (completed >= epoch * 6)
        if tl.sum(newly_ready.to(tl.int32), axis=0) > 0:
            ready_group = tl.argmax(newly_ready.to(tl.int32), axis=0)
            task_base = ready_groups * 128
            tl.store(
                tile_dependency_state
                + SCHED_BASE
                + worklist_base
                + task_base
                + split,
                split * 8 + ready_group,
            )
            tl.debug_barrier()
            published = published | (group == ready_group)
            tl.atomic_xchg(
                tile_dependency_state
                + SCHED_BASE
                + tail_offset
                + ready_groups,
                epoch,
                sem="release",
                scope="gpu",
            )
            ready_groups += 1
        elif POLL_DELAY:
            _qwen_layer_nanosleep(POLL_DELAY)

    _qwen_layer_schedule_ffn(
        tile_dependency_state,
        epoch,
        SCHED_BASE,
        ACTIVATION_TASKS,
        FIRST_ACTIVATION_TASKS,
        READY_BLOCK,
        POLL_DELAY,
    )


@triton.jit(noinline=True)
def _qwen_layer_schedule_tile_queue(
    tile_dependency_state,
    epoch,
    SCHED_BASE: tl.constexpr,
    ACTIVATION_TASKS: tl.constexpr,
    FIRST_ACTIVATION_TASKS: tl.constexpr,
    READY_BLOCK: tl.constexpr,
    POLL_DELAY: tl.constexpr,
):
    group = tl.arange(0, 8)
    split = tl.arange(0, 128)
    arrival_base: tl.constexpr = 197
    worklist_base: tl.constexpr = 205
    tail_offset: tl.constexpr = 1229
    published = tl.zeros([8], tl.int1)
    ready_groups = tl.full([], 0, tl.int32)
    epoch_base = (epoch - 1) * 1024
    while ready_groups < 8:
        completed = _qwen_layer_load_acquire(
            tile_dependency_state + SCHED_BASE + arrival_base + group
        )
        newly_ready = ~published & (completed >= epoch * 6)
        if tl.sum(newly_ready.to(tl.int32), axis=0) > 0:
            ready_group = tl.argmax(newly_ready.to(tl.int32), axis=0)
            task_base = ready_groups * 128
            tl.store(
                tile_dependency_state
                + SCHED_BASE
                + worklist_base
                + task_base
                + split,
                split * 8 + ready_group,
            )
            tl.debug_barrier()
            published = published | (group == ready_group)
            ready_groups += 1
            tl.atomic_xchg(
                tile_dependency_state + SCHED_BASE + tail_offset,
                epoch_base + ready_groups * 128,
                sem="release",
                scope="gpu",
            )
        elif POLL_DELAY:
            _qwen_layer_nanosleep(POLL_DELAY)

    _qwen_layer_schedule_ffn(
        tile_dependency_state,
        epoch,
        SCHED_BASE,
        ACTIVATION_TASKS,
        FIRST_ACTIVATION_TASKS,
        READY_BLOCK,
        POLL_DELAY,
    )


@triton.jit(noinline=True)
def _qwen_layer_schedule_pipeline(
    tile_dependency_state,
    epoch,
    SCHED_BASE: tl.constexpr,
    ACTIVATION_TASKS: tl.constexpr,
    FIRST_ACTIVATION_TASKS: tl.constexpr,
    READY_BLOCK: tl.constexpr,
    POLL_DELAY: tl.constexpr,
):
    w13_ready_offset: tl.constexpr = 2
    w2_ready_offset: tl.constexpr = 3
    first_activation_count_offset: tl.constexpr = 4
    completion_base: tl.constexpr = 5
    command_base: tl.constexpr = completion_base + ACTIVATION_TASKS

    attention_group = tl.arange(0, 8)
    attention_group_arrival_base: tl.constexpr = 197
    attention_group_worklist_base: tl.constexpr = 213
    attention_group_tail: tl.constexpr = 701
    attention_group_published = tl.zeros([8], tl.int1)
    attention_group_count = tl.full([], 0, tl.int32)
    attention_epoch_base = (epoch - 1) * 8
    while attention_group_count < 8:
        completed = _qwen_layer_load_acquire(
            tile_dependency_state
            + SCHED_BASE
            + attention_group_arrival_base
            + attention_group
        )
        newly_ready = ~attention_group_published & (completed >= epoch * 6)
        ready_count = tl.sum(newly_ready.to(tl.int32), axis=0)
        if ready_count > 0:
            ready_rank = tl.cumsum(newly_ready.to(tl.int32), axis=0) - 1
            tl.store(
                tile_dependency_state
                + SCHED_BASE
                + attention_group_worklist_base
                + attention_group_count
                + ready_rank,
                attention_group,
                mask=newly_ready,
            )
            tl.debug_barrier()
            attention_group_published = attention_group_published | newly_ready
            attention_group_count += ready_count
            tl.atomic_xchg(
                tile_dependency_state + SCHED_BASE + attention_group_tail,
                attention_epoch_base + attention_group_count,
                sem="release",
                scope="gpu",
            )
        elif POLL_DELAY:
            _qwen_layer_nanosleep(POLL_DELAY)

    merge_key = tl.arange(0, 128)
    merge_arrival_base: tl.constexpr = 221
    merge_worklist_base: tl.constexpr = 477
    merge_tail: tl.constexpr = 702
    merge_published = tl.zeros([128], tl.int1)
    merge_count = tl.full([], 0, tl.int32)
    merge_epoch_base = (epoch - 1) * 128
    while merge_count < 128:
        completed = _qwen_layer_load_acquire(
            tile_dependency_state
            + SCHED_BASE
            + merge_arrival_base
            + merge_key
        )
        newly_ready = ~merge_published & (completed >= epoch * 8)
        ready_count = tl.sum(newly_ready.to(tl.int32), axis=0)
        if ready_count > 0:
            ready_rank = tl.cumsum(newly_ready.to(tl.int32), axis=0) - 1
            tl.store(
                tile_dependency_state
                + SCHED_BASE
                + merge_worklist_base
                + merge_count
                + ready_rank,
                merge_key,
                mask=newly_ready,
            )
            tl.debug_barrier()
            merge_published = merge_published | newly_ready
            merge_count += ready_count
            tl.atomic_xchg(
                tile_dependency_state + SCHED_BASE + merge_tail,
                merge_epoch_base + merge_count,
                sem="release",
                scope="gpu",
            )
        elif POLL_DELAY:
            _qwen_layer_nanosleep(POLL_DELAY)

    head = tl.arange(0, 32)
    head_arrival_base: tl.constexpr = 605
    head_worklist_base: tl.constexpr = 669
    head_tail: tl.constexpr = 703
    head_published = tl.zeros([32], tl.int1)
    head_count = tl.full([], 0, tl.int32)
    head_epoch_base = (epoch - 1) * 32
    while head_count < 32:
        completed = _qwen_layer_load_acquire(
            tile_dependency_state
            + SCHED_BASE
            + head_arrival_base
            + head
        )
        newly_ready = ~head_published & (completed >= epoch * 16)
        ready_count = tl.sum(newly_ready.to(tl.int32), axis=0)
        if ready_count > 0:
            ready_rank = tl.cumsum(newly_ready.to(tl.int32), axis=0) - 1
            tl.store(
                tile_dependency_state
                + SCHED_BASE
                + head_worklist_base
                + head_count
                + ready_rank,
                head,
                mask=newly_ready,
            )
            tl.debug_barrier()
            head_published = head_published | newly_ready
            head_count += ready_count
            tl.atomic_xchg(
                tile_dependency_state + SCHED_BASE + head_tail,
                head_epoch_base + head_count,
                sem="release",
                scope="gpu",
            )
        elif POLL_DELAY:
            _qwen_layer_nanosleep(POLL_DELAY)

    # The exact-1024 lowering ends with the 32-word post-quant counter.
    _qwen_layer_wait_count(
        tile_dependency_state + SCHED_BASE - 32,
        epoch * 32,
        POLL_DELAY,
    )
    tl.atomic_xchg(
        tile_dependency_state + SCHED_BASE + w13_ready_offset,
        epoch,
        sem="release",
        scope="gpu",
    )

    task = tl.arange(0, READY_BLOCK)
    valid = task < ACTIVATION_TASKS
    safe_task = tl.minimum(task, ACTIVATION_TASKS - 1)
    published = tl.zeros([READY_BLOCK], tl.int1)
    remaining = tl.full([], ACTIVATION_TASKS, tl.int32)
    w2_published = tl.full([], 0, tl.int32)
    while (remaining > 0) | (w2_published == 0):
        progressed = tl.full([], 0, tl.int32)
        if remaining > 0:
            completed = _qwen_layer_load_acquire(
                tile_dependency_state
                + SCHED_BASE
                + completion_base
                + safe_task
            )
            newly_ready = valid & ~published & (completed == epoch)
            ready_count = tl.sum(newly_ready.to(tl.int32), axis=0)
            if ready_count > 0:
                tl.atomic_xchg(
                    tile_dependency_state
                    + SCHED_BASE
                    + command_base
                    + task,
                    epoch,
                    mask=newly_ready,
                    sem="release",
                    scope="gpu",
                )
                published = published | newly_ready
                remaining -= ready_count
                progressed = 1
        if w2_published == 0:
            first_done = _qwen_layer_load_acquire(
                tile_dependency_state
                + SCHED_BASE
                + first_activation_count_offset
            )
            if first_done == epoch * FIRST_ACTIVATION_TASKS:
                tl.atomic_xchg(
                    tile_dependency_state + SCHED_BASE + w2_ready_offset,
                    epoch,
                    sem="release",
                    scope="gpu",
                )
                w2_published = 1
                progressed = 1
        if progressed == 0:
            if POLL_DELAY:
                _qwen_layer_nanosleep(POLL_DELAY)
"""


class _ProgramIdToWorker(ast.NodeTransformer):
    def visit_Call(self, node: ast.Call) -> ast.AST:
        node = self.generic_visit(node)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "tl"
            and node.func.attr == "program_id"
            and len(node.args) == 1
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == 0
        ):
            return ast.copy_location(ast.Name(id="worker", ctx=ast.Load()), node)
        return node


class _RenameName(ast.NodeTransformer):
    def __init__(self, old: str, new: str) -> None:
        self.old = old
        self.new = new

    def visit_Name(self, node: ast.Name) -> ast.Name:
        if node.id == self.old:
            return ast.copy_location(ast.Name(id=self.new, ctx=node.ctx), node)
        return node


class _ReplaceCall(ast.NodeTransformer):
    def __init__(self, old: str, new: str, extra_argument: str) -> None:
        self.old = old
        self.new = new
        self.extra_argument = extra_argument

    def visit_Call(self, node: ast.Call) -> ast.Call:
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id == self.old:
            node.func.id = self.new
            node.args.append(ast.Name(id=self.extra_argument, ctx=ast.Load()))
        return node


def _called_roots(statement: ast.stmt) -> list[str]:
    return [
        node.func.id
        for node in ast.walk(statement)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id.startswith("tile_dependency_root_")
    ]


def _function_arguments(function: ast.FunctionDef) -> set[str]:
    return {argument.arg for argument in function.args.args}


def _contains_call(statement: ast.AST, function_name: str) -> bool:
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == function_name
        for node in ast.walk(statement)
    )


def _attribute_call(statement: ast.AST, attribute: str) -> ast.Call | None:
    for node in ast.walk(statement):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "tl"
            and node.func.attr == attribute
        ):
            return node
    return None


def _direct_call(statement: ast.stmt, function_name: str) -> ast.Call | None:
    if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
        return None
    call = statement.value
    if isinstance(call.func, ast.Name) and call.func.id == function_name:
        return call
    return None


def _insert_group_reports(
    statements: list[ast.stmt],
    function_name: str,
    state_name: str,
    epoch_name: str,
    group_expression,
) -> None:
    index = 0
    while index < len(statements):
        statement = statements[index]
        call = _direct_call(statement, function_name)
        if call is not None:
            insert_at = index + 1
            if (
                insert_at < len(statements)
                and _attribute_call(statements[insert_at], "inline_asm_elementwise")
                is not None
            ):
                insert_at += 1
            group = group_expression(call.args[-1])
            statements.insert(
                insert_at,
                ast.Expr(
                    value=ast.Call(
                        func=ast.Name(
                            id="_qwen_layer_report_attention_group",
                            ctx=ast.Load(),
                        ),
                        args=[
                            ast.Name(id=state_name, ctx=ast.Load()),
                            ast.Name(id=epoch_name, ctx=ast.Load()),
                            group,
                            ast.Name(id="SCHED_BASE", ctx=ast.Load()),
                        ],
                        keywords=[],
                    )
                ),
            )
            index = insert_at + 1
            continue
        for field in ("body", "orelse"):
            child = getattr(statement, field, None)
            if isinstance(child, list):
                _insert_group_reports(
                    child,
                    function_name,
                    state_name,
                    epoch_name,
                    group_expression,
                )
        index += 1


def _make_qkv_report_helper(
    scheduled: ast.FunctionDef,
    qk_root: str,
    cache_roots: tuple[str, ...],
) -> ast.FunctionDef:
    helper = copy.deepcopy(scheduled)
    helper.name = "_qwen_layer_qkv_report"
    helper.args.args.append(
        ast.arg(
            arg="SCHED_BASE",
            annotation=ast.Attribute(
                value=ast.Name(id="tl", ctx=ast.Load()),
                attr="constexpr",
                ctx=ast.Load(),
            ),
        )
    )
    state_name = next(
        argument.arg
        for argument in scheduled.args.args
        if "dependency_state" in argument.arg
    )
    epoch_name = scheduled.args.args[-1].arg

    def qk_group(task: ast.expr) -> ast.expr:
        return ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="tl", ctx=ast.Load()),
                attr="where",
                ctx=ast.Load(),
            ),
            args=[
                ast.Compare(
                    left=copy.deepcopy(task),
                    ops=[ast.Lt()],
                    comparators=[ast.Constant(value=32)],
                ),
                ast.BinOp(
                    left=copy.deepcopy(task),
                    op=ast.FloorDiv(),
                    right=ast.Constant(value=4),
                ),
                ast.BinOp(
                    left=copy.deepcopy(task),
                    op=ast.Sub(),
                    right=ast.Constant(value=32),
                ),
            ],
            keywords=[],
        )

    _insert_group_reports(
        helper.body,
        qk_root,
        state_name,
        epoch_name,
        qk_group,
    )
    for cache_root in cache_roots:
        _insert_group_reports(
            helper.body,
            cache_root,
            state_name,
            epoch_name,
            copy.deepcopy,
        )
    return helper


def _make_w13_report_helper(
    scheduled: ast.FunctionDef,
    activation_root: str,
) -> tuple[ast.FunctionDef, str, str, str]:
    helper = copy.deepcopy(scheduled)
    helper.name = "_qwen_layer_w13_report"
    helper.args.args.append(
        ast.arg(
            arg="SCHED_BASE",
            annotation=ast.Attribute(
                value=ast.Name(id="tl", ctx=ast.Load()),
                attr="constexpr",
                ctx=ast.Load(),
            ),
        )
    )
    activation_if = next(
        node
        for node in helper.body
        if isinstance(node, ast.If) and _contains_call(node, activation_root)
    )
    activation_call = next(
        node
        for node in ast.walk(activation_if)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == activation_root
    )
    activation_task = activation_call.args[-1]
    if not isinstance(activation_task, ast.Name):
        raise RuntimeError("expected named Qwen3 activation task")
    state_name = next(
        argument.arg
        for argument in scheduled.args.args
        if "dependency_state" in argument.arg
    )
    epoch_name = scheduled.args.args[-1].arg
    report_address = ast.BinOp(
        left=ast.BinOp(
            left=ast.BinOp(
                left=ast.Name(id=state_name, ctx=ast.Load()),
                op=ast.Add(),
                right=ast.Name(id="SCHED_BASE", ctx=ast.Load()),
            ),
            op=ast.Add(),
            right=ast.Constant(value=5),
        ),
        op=ast.Add(),
        right=copy.deepcopy(activation_task),
    )
    activation_if.body = [
        ast.Expr(
            value=ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="tl", ctx=ast.Load()),
                    attr="atomic_xchg",
                    ctx=ast.Load(),
                ),
                args=[
                    report_address,
                    ast.Name(id=epoch_name, ctx=ast.Load()),
                ],
                keywords=[
                    ast.keyword(arg="sem", value=ast.Constant(value="release")),
                    ast.keyword(arg="scope", value=ast.Constant(value="gpu")),
                ],
            )
        )
    ]
    return helper, activation_task.id, state_name, epoch_name


def _make_activation_helper(
    activation: ast.FunctionDef,
    scheduled_w13: ast.FunctionDef,
    activation_task_name: str,
    state_name: str,
    epoch_name: str,
) -> ast.FunctionDef:
    activation_if = next(
        node
        for node in scheduled_w13.body
        if isinstance(node, ast.If) and _contains_call(node, activation.name)
    )
    activation_call_index = next(
        index
        for index, statement in enumerate(activation_if.body)
        if _contains_call(statement, activation.name)
    )
    publication = copy.deepcopy(activation_if.body[activation_call_index + 1 :])
    publication = [
        _RenameName(activation_task_name, "activation_task").visit(statement)
        for statement in publication
    ]
    arguments = copy.deepcopy(activation.args)
    arguments.args[-1].arg = "activation_task"
    arguments.args.extend(
        [
            ast.arg(arg=state_name),
            ast.arg(arg=epoch_name),
            ast.arg(
                arg="SCHED_BASE",
                annotation=ast.Attribute(
                    value=ast.Name(id="tl", ctx=ast.Load()),
                    attr="constexpr",
                    ctx=ast.Load(),
                ),
            ),
            ast.arg(
                arg="FIRST_ACTIVATION_TASKS",
                annotation=ast.Attribute(
                    value=ast.Name(id="tl", ctx=ast.Load()),
                    attr="constexpr",
                    ctx=ast.Load(),
                ),
            ),
        ]
    )
    call_arguments = [
        ast.Name(id=argument.arg, ctx=ast.Load()) for argument in arguments.args[:-4]
    ]
    first_count_address = ast.BinOp(
        left=ast.BinOp(
            left=ast.Name(id=state_name, ctx=ast.Load()),
            op=ast.Add(),
            right=ast.Name(id="SCHED_BASE", ctx=ast.Load()),
        ),
        op=ast.Add(),
        right=ast.Constant(value=4),
    )
    first_count = ast.If(
        test=ast.Compare(
            left=ast.Name(id="activation_task", ctx=ast.Load()),
            ops=[ast.Lt()],
            comparators=[ast.Name(id="FIRST_ACTIVATION_TASKS", ctx=ast.Load())],
        ),
        body=[
            ast.Expr(
                value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id="tl", ctx=ast.Load()),
                        attr="atomic_add",
                        ctx=ast.Load(),
                    ),
                    args=[first_count_address, ast.Constant(value=1)],
                    keywords=[
                        ast.keyword(arg="sem", value=ast.Constant(value="release")),
                        ast.keyword(arg="scope", value=ast.Constant(value="gpu")),
                    ],
                )
            )
        ],
        orelse=[],
    )
    return ast.FunctionDef(
        name="_qwen_layer_activation",
        args=arguments,
        body=[
            ast.Expr(
                value=ast.Call(
                    func=ast.Name(id=activation.name, ctx=ast.Load()),
                    args=call_arguments,
                    keywords=[],
                )
            ),
            *publication,
            first_count,
        ],
        decorator_list=[
            ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="triton", ctx=ast.Load()),
                    attr="jit",
                    ctx=ast.Load(),
                ),
                args=[],
                keywords=[ast.keyword(arg="noinline", value=ast.Constant(value=True))],
            )
        ],
        returns=None,
        type_comment=None,
    )


def _wait_statement(address: str) -> ast.stmt:
    return ast.parse(
        f"_qwen_layer_wait_epoch({address}, tile_dependency_epoch, POLL_DELAY)"
    ).body[0]


def _scheduler_call(function: str, *arguments: str) -> ast.stmt:
    return ast.Expr(
        value=ast.Call(
            func=ast.Name(id=function, ctx=ast.Load()),
            args=[ast.parse(argument, mode="eval").body for argument in arguments],
            keywords=[],
        )
    )


def _parsed_statements(source: str) -> list[ast.stmt]:
    return ast.parse(textwrap.dedent(source)).body


def _scheduler_source(prefix_signal: str) -> str:
    source = textwrap.dedent(SCHEDULER_SOURCE)
    if prefix_signal == "native":
        return source
    native_wait = """\
    _qwen_layer_wait_count(
        tile_dependency_state + SCHED_BASE - 32,
        epoch * 32,
        POLL_DELAY,
    )
"""
    worker_wait = """\
    _qwen_layer_wait_epoch(
        tile_dependency_state + SCHED_BASE + 1,
        epoch,
        POLL_DELAY,
    )
"""
    if source.count(native_wait) != 2:
        raise RuntimeError("unexpected scheduler prefix-wait layout")
    return source.replace(native_wait, worker_wait)


def _rewrite_attention(
    prefix: list[ast.stmt],
    attention_index: int,
    attention_schedule: str,
) -> None:
    if attention_schedule == "static":
        return

    attention_statement = prefix[attention_index]
    if not isinstance(attention_statement, ast.If):
        raise RuntimeError("expected a guarded Qwen3 attention worker block")
    attention_loop_index = next(
        index
        for index, statement in enumerate(attention_statement.body)
        if isinstance(statement, ast.For)
    )
    attention_loop = attention_statement.body[attention_loop_index]
    if not isinstance(attention_loop, ast.For):
        raise RuntimeError("expected a Qwen3 attention tile loop")
    attention_suffix = copy.deepcopy(
        attention_statement.body[attention_loop_index + 1 :]
    )

    if attention_schedule == "group":
        attention_statement.body = [
            *_parsed_statements(
                """
                attention_group = worker % 8
                _qwen_layer_wait_epoch(
                    tile_dependency_state + SCHED_BASE + 213 + attention_group,
                    tile_dependency_epoch,
                    POLL_DELAY,
                )
                """
            ),
            copy.deepcopy(attention_loop),
            *attention_suffix,
        ]
        return

    if attention_schedule == "tile-worklist":
        attention_call = copy.deepcopy(attention_loop.body[0])
        attention_statement.body = [
            *_parsed_statements(
                """
                attention_slot = (worker + 256) % 1024
                _qwen_layer_wait_epoch(
                    tile_dependency_state
                    + SCHED_BASE
                    + 1229
                    + attention_slot // 128,
                    tile_dependency_epoch,
                    POLL_DELAY,
                )
                attention_task = tl.load(
                    tile_dependency_state + SCHED_BASE + 205 + attention_slot
                ).to(tl.int32)
                virtual_pid = 880 + attention_task
                """
            ),
            attention_call,
            *attention_suffix,
        ]
        return

    if attention_schedule == "tile-queue":
        attention_call = copy.deepcopy(attention_loop.body[0])
        attention_statement.body = [
            *_parsed_statements(
                """
                attention_position = tl.atomic_add(
                    tile_dependency_state + SCHED_BASE + 1230,
                    1,
                    sem="acq_rel",
                    scope="gpu",
                )
                attention_slot = attention_position % 1024
                _qwen_layer_wait_count(
                    tile_dependency_state + SCHED_BASE + 1229,
                    attention_position + 1,
                    POLL_DELAY,
                )
                attention_task = tl.load(
                    tile_dependency_state + SCHED_BASE + 205 + attention_slot
                ).to(tl.int32)
                virtual_pid = 880 + attention_task
                """
            ),
            attention_call,
            *attention_suffix,
        ]
        return

    tail_offset = 221 if attention_schedule == "ready-order" else 701
    attention_call = copy.deepcopy(attention_loop.body[0])
    attention_statement.body = [
        *_parsed_statements(
            f"""
            attention_slot = ((worker + 256) % 1024) // 128
            _qwen_layer_wait_count(
                tile_dependency_state + SCHED_BASE + {tail_offset},
                (tile_dependency_epoch - 1) * 8 + attention_slot + 1,
                POLL_DELAY,
            )
            attention_group = tl.load(
                tile_dependency_state + SCHED_BASE + 213 + attention_slot
            ).to(tl.int32)
            attention_task = (worker % 128) * 8 + attention_group
            virtual_pid = 880 + attention_task
            """
        ),
        attention_call,
        attention_suffix[0],
    ]
    if attention_schedule == "pipeline":
        attention_statement.body.append(
            _scheduler_call(
                "_qwen_layer_report_merge_chunk",
                "tile_dependency_state",
                "tile_dependency_epoch",
                "attention_task",
                "SCHED_BASE",
            )
        )
    else:
        attention_statement.body.extend(attention_suffix[1:])


def _rewrite_pipeline_tail(prefix: list[ast.stmt], attention_index: int) -> None:
    merge_statement = prefix[attention_index + 1]
    if not isinstance(merge_statement, ast.If):
        raise RuntimeError("expected a guarded Qwen3 merge worker block")
    merge_loop_index = next(
        index
        for index, statement in enumerate(merge_statement.body)
        if isinstance(statement, ast.For)
    )
    merge_loop = merge_statement.body[merge_loop_index]
    if not isinstance(merge_loop, ast.For):
        raise RuntimeError("expected a Qwen3 merge tile loop")
    merge_call = copy.deepcopy(merge_loop.body[0])
    merge_sync = copy.deepcopy(merge_statement.body[merge_loop_index + 1])
    merge_statement.body = [
        *_parsed_statements(
            """
            merge_slot = worker // 4
            _qwen_layer_wait_count(
                tile_dependency_state + SCHED_BASE + 702,
                (tile_dependency_epoch - 1) * 128 + merge_slot + 1,
                POLL_DELAY,
            )
            merge_key = tl.load(
                tile_dependency_state + SCHED_BASE + 477 + merge_slot
            ).to(tl.int32)
            merge_head = (merge_key // 16) * 4 + worker % 4
            merge_task = merge_head * 16 + merge_key % 16
            virtual_pid = 1904 + merge_task
            """
        ),
        merge_call,
        merge_sync,
        _scheduler_call(
            "_qwen_layer_report_merge_head",
            "tile_dependency_state",
            "tile_dependency_epoch",
            "merge_head",
            "SCHED_BASE",
        ),
    ]

    final_merge_statement = prefix[attention_index + 2]
    if not isinstance(final_merge_statement, ast.If):
        raise RuntimeError("expected a guarded Qwen3 final-merge worker block")
    final_merge_loop = next(
        statement
        for statement in final_merge_statement.body
        if isinstance(statement, ast.For)
    )
    final_merge_call = copy.deepcopy(final_merge_loop.body[0])
    final_merge_statement.body = [
        *_parsed_statements(
            """
            _qwen_layer_wait_count(
                tile_dependency_state + SCHED_BASE + 703,
                (tile_dependency_epoch - 1) * 32 + worker + 1,
                POLL_DELAY,
            )
            final_merge_head = tl.load(
                tile_dependency_state + SCHED_BASE + 669 + worker
            ).to(tl.int32)
            virtual_pid = 2416 + final_merge_head
            """
        ),
        final_merge_call,
    ]


def _generated_namespace(
    bound,
    config,
    lowered_output: Path,
    attention_schedule: str,
    prefix_signal: str,
    activation_worker_start: int,
):
    lowered = bound.to_triton_code(config, output_origin_lines=True)
    module = ast.parse(lowered)
    functions = {
        node.name: node for node in module.body if isinstance(node, ast.FunctionDef)
    }
    master = next(
        function
        for name, function in functions.items()
        if name.startswith("_helion_qwen3_layer_tile_dependency_source")
    )

    scheduled = [
        function
        for function in functions.values()
        if function.name.endswith("_scheduled_task")
    ]
    w13_scheduled = next(
        function for function in scheduled if "w13_q" in _function_arguments(function)
    )
    w2_scheduled = next(
        function for function in scheduled if "w2_q" in _function_arguments(function)
    )
    qkv_scheduled = next(
        function
        for function in scheduled
        if {"qkv_weight_q", "kv_cache"} <= _function_arguments(function)
    )
    qkv_called_roots = {
        call.func.id
        for call in ast.walk(qkv_scheduled)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id.startswith("tile_dependency_root_")
    }
    qk_root = next(
        name
        for name in qkv_called_roots
        if {"q_weight", "k_weight"} <= _function_arguments(functions[name])
    )
    cache_roots = tuple(
        name
        for name in qkv_called_roots
        if {"slot_mapping", "kv_cache"} <= _function_arguments(functions[name])
    )
    attention = next(
        function
        for function in functions.values()
        if function.name.startswith("tile_dependency_root_")
        and not function.name.endswith("_scheduled_task")
        and {"partial_out", "partial_lse", "kv_cache"} <= _function_arguments(function)
    )
    activation = next(
        function
        for function in functions.values()
        if function.name.startswith("tile_dependency_root_")
        and not function.name.endswith("_scheduled_task")
        and {"gate_up", "activation_scale", "activation_q"}
        <= _function_arguments(function)
        and "w13_q" not in _function_arguments(function)
    )
    w13_index = next(
        index
        for index, statement in enumerate(master.body)
        if w13_scheduled.name in _called_roots(statement)
    )
    w2_index = next(
        index
        for index, statement in enumerate(master.body)
        if w2_scheduled.name in _called_roots(statement)
    )
    if w2_index != w13_index + 1:
        raise RuntimeError("expected the exact-1024 Qwen3 FFN tail layout")
    qkv_index = next(
        index
        for index, statement in enumerate(master.body)
        if qkv_scheduled.name in _called_roots(statement)
    )
    attention_index = next(
        index
        for index, statement in enumerate(master.body)
        if attention.name in _called_roots(statement)
    )

    qkv_report_helper = _make_qkv_report_helper(
        qkv_scheduled,
        qk_root,
        cache_roots,
    )
    report_helper, activation_task_name, state_name, epoch_name = (
        _make_w13_report_helper(w13_scheduled, activation.name)
    )
    activation_helper = _make_activation_helper(
        activation,
        w13_scheduled,
        activation_task_name,
        state_name,
        epoch_name,
    )

    pid_transform = _ProgramIdToWorker()
    prefix = [
        pid_transform.visit(copy.deepcopy(statement))
        for statement in master.body[:w13_index]
    ]
    if attention_schedule != "static":
        prefix[qkv_index] = _ReplaceCall(
            qkv_scheduled.name,
            qkv_report_helper.name,
            "SCHED_BASE",
        ).visit(prefix[qkv_index])
    _rewrite_attention(prefix, attention_index, attention_schedule)
    if attention_schedule == "pipeline":
        _rewrite_pipeline_tail(prefix, attention_index)
    w13_statement = pid_transform.visit(copy.deepcopy(master.body[w13_index]))
    w13_statement = _ReplaceCall(
        w13_scheduled.name,
        report_helper.name,
        "SCHED_BASE",
    ).visit(w13_statement)
    w13_loop = next(
        node for node in ast.walk(w13_statement) if isinstance(node, ast.For)
    )
    w13_parent = next(
        node
        for node in ast.walk(w13_statement)
        if isinstance(node, ast.If) and w13_loop in node.body
    )
    w13_parent.body.insert(
        w13_parent.body.index(w13_loop),
        _wait_statement("tile_dependency_state + SCHED_BASE + 2"),
    )

    w2_statement = pid_transform.visit(copy.deepcopy(master.body[w2_index]))
    if not isinstance(w2_statement, ast.If):
        raise RuntimeError("expected a guarded Qwen3 W2 worker block")
    w2_statement.body.insert(
        0,
        _wait_statement("tile_dependency_state + SCHED_BASE + 3"),
    )

    master_arguments = copy.deepcopy(master.args)
    state_argument = master_arguments.args.pop()
    if state_argument.arg != state_name:
        raise RuntimeError("unexpected persistent-state argument")
    constexpr = ast.Attribute(
        value=ast.Name(id="tl", ctx=ast.Load()),
        attr="constexpr",
        ctx=ast.Load(),
    )
    master_arguments.args.extend(
        [
            ast.arg(arg="SCHED_BASE", annotation=copy.deepcopy(constexpr)),
            ast.arg(arg="POLL_DELAY", annotation=copy.deepcopy(constexpr)),
            state_argument,
        ]
    )

    activation_arguments = [argument.arg for argument in activation.args.args[:-1]]
    activation_call = ast.Expr(
        value=ast.Call(
            func=ast.Name(id=activation_helper.name, ctx=ast.Load()),
            args=[
                *[ast.Name(id=name, ctx=ast.Load()) for name in activation_arguments],
                ast.Name(id="activation_task", ctx=ast.Load()),
                ast.Name(id=state_name, ctx=ast.Load()),
                ast.Name(id=epoch_name, ctx=ast.Load()),
                ast.Name(id="SCHED_BASE", ctx=ast.Load()),
                ast.Constant(value=64),
            ],
            keywords=[],
        )
    )
    activation_body = [
        ast.Assign(
            targets=[ast.Name(id="activation_task", ctx=ast.Store())],
            value=ast.BinOp(
                left=ast.Name(id="worker", ctx=ast.Load()),
                op=ast.Sub(),
                right=ast.Constant(value=activation_worker_start),
            ),
        ),
        _wait_statement(
            "tile_dependency_state + SCHED_BASE + 5 + 96 + activation_task"
        ),
        activation_call,
    ]
    prefix_publication = (
        _parsed_statements(
            """
            if worker == 0:
                tl.atomic_xchg(
                    tile_dependency_state + SCHED_BASE + 1,
                    tile_dependency_epoch,
                    sem="release",
                    scope="gpu",
                )
            """
        )
        if prefix_signal == "worker"
        else []
    )
    worker_body = [
        ast.Assign(
            targets=[ast.Name(id="worker", ctx=ast.Store())],
            value=ast.BinOp(
                left=ast.Name(id="program", ctx=ast.Load()),
                op=ast.Sub(),
                right=ast.Constant(value=1),
            ),
        ),
        *prefix,
        *prefix_publication,
        w13_statement,
        ast.If(
            test=ast.BoolOp(
                op=ast.And(),
                values=[
                    ast.Compare(
                        left=ast.Name(id="worker", ctx=ast.Load()),
                        ops=[ast.GtE()],
                        comparators=[ast.Constant(value=activation_worker_start)],
                    ),
                    ast.Compare(
                        left=ast.Name(id="worker", ctx=ast.Load()),
                        ops=[ast.Lt()],
                        comparators=[ast.Constant(value=activation_worker_start + 96)],
                    ),
                ],
            ),
            body=activation_body,
            orelse=[],
        ),
        w2_statement,
        ast.Expr(
            value=ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="tl", ctx=ast.Load()),
                    attr="store",
                    ctx=ast.Load(),
                ),
                args=[
                    ast.BinOp(
                        left=ast.Name(id=state_name, ctx=ast.Load()),
                        op=ast.Add(),
                        right=ast.Name(id="worker", ctx=ast.Load()),
                    ),
                    ast.Name(id=epoch_name, ctx=ast.Load()),
                ],
                keywords=[],
            )
        ),
    ]
    scheduler_function = {
        "static": "_qwen_layer_schedule_ffn",
        "group": "_qwen_layer_schedule_groups",
        "ready-order": "_qwen_layer_schedule_ready_order",
        "tile-worklist": "_qwen_layer_schedule_tiles",
        "tile-queue": "_qwen_layer_schedule_tile_queue",
        "pipeline": "_qwen_layer_schedule_pipeline",
    }[attention_schedule]
    scheduler_body = [
        ast.Assign(
            targets=[ast.Name(id="scheduler_epoch", ctx=ast.Store())],
            value=ast.BinOp(
                left=ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id="tl", ctx=ast.Load()),
                        attr="atomic_add",
                        ctx=ast.Load(),
                    ),
                    args=[
                        ast.BinOp(
                            left=ast.Name(id=state_name, ctx=ast.Load()),
                            op=ast.Add(),
                            right=ast.Name(id="SCHED_BASE", ctx=ast.Load()),
                        ),
                        ast.Constant(value=1),
                    ],
                    keywords=[
                        ast.keyword(arg="sem", value=ast.Constant(value="acq_rel")),
                        ast.keyword(arg="scope", value=ast.Constant(value="gpu")),
                    ],
                ),
                op=ast.Add(),
                right=ast.Constant(value=1),
            ),
        ),
        ast.Expr(
            value=ast.Call(
                func=ast.Name(id=scheduler_function, ctx=ast.Load()),
                args=[
                    ast.Name(id=state_name, ctx=ast.Load()),
                    ast.Name(id="scheduler_epoch", ctx=ast.Load()),
                    ast.Name(id="SCHED_BASE", ctx=ast.Load()),
                    ast.Constant(value=96),
                    ast.Constant(value=64),
                    ast.Constant(value=128),
                    ast.Name(id="POLL_DELAY", ctx=ast.Load()),
                ],
                keywords=[],
            )
        ),
    ]
    orchestrator = ast.FunctionDef(
        name="qwen3_layer_orchestrator",
        args=master_arguments,
        body=[
            ast.Assign(
                targets=[ast.Name(id="program", ctx=ast.Store())],
                value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id="tl", ctx=ast.Load()),
                        attr="program_id",
                        ctx=ast.Load(),
                    ),
                    args=[ast.Constant(value=0)],
                    keywords=[],
                ),
            ),
            ast.If(
                test=ast.Compare(
                    left=ast.Name(id="program", ctx=ast.Load()),
                    ops=[ast.Eq()],
                    comparators=[ast.Constant(value=0)],
                ),
                body=scheduler_body,
                orelse=worker_body,
            ),
        ],
        decorator_list=[
            ast.Attribute(
                value=ast.Name(id="triton", ctx=ast.Load()),
                attr="jit",
                ctx=ast.Load(),
            )
        ],
        returns=None,
        type_comment=None,
    )

    generated_module = ast.fix_missing_locations(
        ast.Module(
            body=[
                qkv_report_helper,
                report_helper,
                activation_helper,
                orchestrator,
            ],
            type_ignores=[],
        )
    )
    generated_source = (
        _scheduler_source(prefix_signal) + "\n\n" + ast.unparse(generated_module) + "\n"
    )
    filename = str(Path(__file__).with_name("_generated_qwen3_layer_orchestrator.py"))
    combined_source = lowered + "\n\n" + generated_source
    linecache.cache[filename] = (
        len(combined_source),
        None,
        combined_source.splitlines(keepends=True),
        filename,
    )
    namespace: dict[str, object] = {"__name__": "_generated_qwen3_layer_orchestrator"}
    exec(compile(combined_source, filename, "exec"), namespace)
    lowered_output.write_text(combined_source)
    return namespace


def _orchestrator_launcher(kernel, poll_delay: int, scheduler_words: int):
    def launch(
        _original_kernel: object,
        grid: tuple[int, ...],
        *kernel_args: object,
        _persistent_state_specs: tuple[tuple[torch.Tensor, int, torch.dtype], ...] = (),
        _minimum_resident_programs: int = 0,
        **kwargs: object,
    ) -> object:
        if len(_persistent_state_specs) != 1:
            raise RuntimeError("expected one full-layer persistent state buffer")
        state_like, state_size, state_dtype = _persistent_state_specs[0]
        workers = int(grid[0])
        programs = workers + 1
        return helion.runtime.default_launcher(
            kernel,
            (programs,),
            *kernel_args,
            state_size,
            poll_delay,
            _persistent_state_specs=(
                (state_like, state_size + scheduler_words, state_dtype),
            ),
            _minimum_resident_programs=programs,
            **kwargs,
        )

    return launch


def _compiled_kernel(jit_kernel):
    device_cache = jit_kernel.device_caches[torch.cuda.current_device()][0]
    kernels = list(device_cache.values())
    if len(kernels) != 1:
        raise RuntimeError(
            f"expected one orchestrator specialization, found {len(kernels)}"
        )
    return kernels[0]


def _triton_resources(compiled, num_warps: int) -> dict[str, int]:
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
    return {
        "registers": compiled.n_regs,
        "spills": compiled.n_spills,
        "shared": compiled.metadata.shared,
        "blocks_per_sm": blocks.value,
        "resident_programs": blocks.value * sms,
    }


def _as_outputs(outputs) -> dict[str, torch.Tensor]:
    return dict(zip(OUTPUT_NAMES, outputs, strict=True))


def _build_granular_kernel():
    layer_probe.rms_norm_per_block_quant = granular.tiled_rms_norm_per_block_quant
    layer_probe.reshape_and_cache_flash = granular.tiled_reshape_and_cache_flash
    layer_probe.merge_attention_splits = granular.tiled_merge_attention_splits
    kernel, _ = layer_probe._build_composite_kernel()
    return helion.kernel(static_shapes=True, autotune_effort="none")(kernel.fn)


def _tuned_config(bound, args):
    base = granular._probe_config(bound, args)
    values = dict(base)
    for index, spec in enumerate(bound.config_spec.range_num_stages):
        if 37 in spec.block_ids:
            values["range_num_stages"][index] = args.w13_stages
        elif 42 in spec.block_ids:
            values["range_num_stages"][index] = args.w2_stages
    for index, spec in enumerate(bound.config_spec.range_unroll_factors):
        if 37 in spec.block_ids:
            values["range_unroll_factors"][index] = args.w13_unroll
        elif 42 in spec.block_ids:
            values["range_unroll_factors"][index] = args.w2_unroll
    config = helion.Config.from_dict(values)
    bound.config_spec.normalize(config.config)
    return config


def _assert_close(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    tolerances = {
        "output": (0.25, 5e-2),
        "qkv": (6e-2, 5e-2),
        "partial_out": (8e-2, 3e-2),
        "partial_lse": (8e-2, 3e-2),
        "attention": (8e-2, 3e-2),
        "attention_out": (0.125, 3e-2),
        "gate_up": (0.125, 3e-2),
        "activation_q": (64.0, 3e-2),
        "activation_scale": (2e-3, 3e-2),
    }
    tensor_name = name.removeprefix("static_").removeprefix("orchestrator_")
    atol, rtol = tolerances.get(tensor_name, (0.0, 0.0))
    actual_float = actual.view_as(expected).float()
    expected_float = expected.float()
    try:
        torch.testing.assert_close(
            actual_float,
            expected_float,
            atol=atol,
            rtol=rtol,
        )
    except AssertionError as error:
        difference = (actual_float - expected_float).abs()
        raise AssertionError(
            f"{name}: max_abs={difference.max().item()}, "
            f"mean_abs={difference.mean().item()}; {error}"
        ) from error


def run(args) -> None:
    if not args.allow_busy:
        require_idle_visible_gpu()
    if args.batch != 1:
        raise ValueError("the full-layer orchestrator currently supports --batch 1")
    if args.cross_loop_workers != 1024:
        raise ValueError("the current CTA placement requires --cross-loop-workers 1024")
    if not 512 <= args.activation_worker_start <= 896:
        raise ValueError("--activation-worker-start must be in [512, 896]")

    static_tensors = allocate_layer(args)
    static_args = _composite_args(static_tensors, args)
    granular_kernel = _build_granular_kernel()
    bound = granular_kernel.bind(static_args)
    config = _tuned_config(bound, args)
    namespace = _generated_namespace(
        bound,
        config,
        Path(args.lowered_output),
        args.attention_schedule,
        args.prefix_signal,
        args.activation_worker_start,
    )
    orchestrator_kernel = namespace["qwen3_layer_orchestrator"]
    orchestrator_wrapper = namespace["qwen3_layer_tile_dependency_source"]
    orchestrator_launcher = _orchestrator_launcher(
        orchestrator_kernel,
        args.poll_delay,
        {
            "static": 197,
            "group": 221,
            "ready-order": 222,
            "tile-worklist": 1237,
            "tile-queue": 1231,
            "pipeline": 704,
        }[args.attention_schedule],
    )

    static_compiled = bound.compile_config(config)
    static_outputs = _as_outputs(static_compiled(*static_args))

    orchestrator_tensors = allocate_layer(args)
    orchestrator_args = _composite_args(orchestrator_tensors, args)
    orchestrator_outputs = _as_outputs(
        orchestrator_wrapper(
            *orchestrator_args,
            _launcher=orchestrator_launcher,
        )
    )

    reference_tensors = allocate_layer(args)
    reference_launch, reference_outputs = build_helion_reference(
        args,
        reference_tensors,
    )
    reference_launch()
    torch.cuda.synchronize()

    for name, expected in reference_outputs.items():
        _assert_close(f"static_{name}", static_outputs[name], expected)
        _assert_close(f"orchestrator_{name}", orchestrator_outputs[name], expected)
    for name in OUTPUT_NAMES:
        if name in {"residual", "pre_q", "pre_scale", "ffn_q", "ffn_scale"}:
            continue
        torch.testing.assert_close(
            orchestrator_outputs[name],
            static_outputs[name],
            atol=0,
            rtol=0,
            msg=f"orchestrator_{name}_vs_static",
        )

    static_graph, _ = capture(lambda: static_compiled(*static_args))
    orchestrator_graph, _ = capture(
        lambda: orchestrator_wrapper(
            *orchestrator_args,
            _launcher=orchestrator_launcher,
        )
    )
    reference_graph, _ = capture(reference_launch)
    for _ in range(args.correctness_replays):
        static_graph.replay()
        orchestrator_graph.replay()
        reference_graph.replay()
    torch.cuda.synchronize()

    benchmark_pids = visible_gpu_pids()
    if not args.allow_busy and (foreign_pids := benchmark_pids - {os.getpid()}):
        raise RuntimeError(
            f"GPU gained foreign compute processes {sorted(foreign_pids)}"
        )
    timings = benchmark_interleaved(
        {
            "static_schedule": static_graph.replay,
            "cta0_tile_scheduler": orchestrator_graph.replay,
            "standalone_helion_graph": reference_graph.replay,
        },
        args.repeats,
        args.batch_replays,
    )
    if visible_gpu_pids() != benchmark_pids:
        raise RuntimeError("GPU process set changed during benchmark")

    static_us = timings["static_schedule"]["median_us"]
    standalone_us = timings["standalone_helion_graph"]["median_us"]
    scheduler_us = timings["cta0_tile_scheduler"]["median_us"]
    timings["cta0_tile_scheduler"]["reduction_vs_static_pct"] = (
        100.0 * (static_us - scheduler_us) / static_us
    )
    timings["cta0_tile_scheduler"]["reduction_vs_standalone_pct"] = (
        100.0 * (standalone_us - scheduler_us) / standalone_us
    )

    binary = _compiled_kernel(orchestrator_kernel)
    result = {
        "device": torch.cuda.get_device_name(),
        "cold_l2": os.environ.get("MEGAKERNEL_CLEAR_L2") == "1",
        "shape": {
            "batch": args.batch,
            "hidden": args.hidden,
            "intermediate": args.intermediate,
            "context": args.context,
            "attention_splits": args.attention_splits,
        },
        "schedule": {
            "workers": args.cross_loop_workers,
            "programs": args.cross_loop_workers + 1,
            "activation_workers": [
                args.activation_worker_start,
                args.activation_worker_start + 96,
            ],
            "w2_workers": [512, 1024],
            "attention": args.attention_schedule,
            "prefix_signal": args.prefix_signal,
            "poll_delay": args.poll_delay,
        },
        "timings": timings,
        "resources": {
            "static_schedule": _helion_resources(static_compiled),
            "cta0_tile_scheduler": _triton_resources(binary, 1),
        },
        "lowered": args.lowered_output,
    }
    print("RESULT_JSON", json.dumps(result, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=12288)
    parser.add_argument("--q-heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--context", type=int, default=8192)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--attention-splits", type=int, default=128)
    parser.add_argument("--helion-comparison-splits", type=int, default=32)
    parser.add_argument("--group", type=int, default=128)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--rope-theta", type=float, default=1_000_000.0)
    parser.add_argument("--projection-stages", type=int, default=4)
    parser.add_argument("--w13-stages", type=int, default=4)
    parser.add_argument("--w13-unroll", type=int, default=2)
    parser.add_argument("--w2-stages", type=int, default=4)
    parser.add_argument("--w2-unroll", type=int, default=4)
    parser.add_argument("--kernel-stages", type=int, default=2)
    parser.add_argument("--worker-multiplier", type=int, default=8)
    parser.add_argument("--cross-loop-workers", type=int, default=1024)
    parser.add_argument("--merge-split-block", type=int, default=32)
    parser.add_argument("--merge-q-block", type=int, default=4)
    parser.add_argument("--attention-context-block", type=int, default=32)
    parser.add_argument("--qk-head-block", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--poll-delay", type=int, default=0)
    parser.add_argument(
        "--prefix-signal",
        choices=("worker", "native"),
        default="worker",
    )
    parser.add_argument("--activation-worker-start", type=int, default=512)
    parser.add_argument(
        "--attention-schedule",
        choices=(
            "static",
            "group",
            "ready-order",
            "tile-worklist",
            "tile-queue",
            "pipeline",
        ),
        default="static",
    )
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--batch-replays", type=int, default=20)
    parser.add_argument("--correctness-replays", type=int, default=5)
    parser.add_argument("--allow-busy", action="store_true")
    parser.add_argument(
        "--config-path",
        default=str(Path(__file__).with_name("qwen3_layer_helion_b200_configs.json")),
    )
    parser.add_argument(
        "--lowered-output",
        default="/tmp/qwen3_layer_orchestrator_lowered.py",
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
