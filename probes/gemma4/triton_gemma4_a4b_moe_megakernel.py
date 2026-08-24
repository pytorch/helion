# ruff: noqa: ANN001, ANN201, ANN202
"""Single persistent Triton megakernel for the Gemma 4 26B-A4B MoE sub-layer.

The roots mirror the standalone Helion kernels in ``helion_gemma4_a4b_moe.py``
one for one; only their dispatch and the readiness edges between them change.
The point of this probe is the dispatch: MoE is the first workload in this
series whose task list is *ragged*, because the number of occupied expert tiles
and their occupancy are both decided by the router at run time.

Three dispatch policies are available for the two ragged roots, selected with
``--schedule``:

``static``       Traverse the static upper bound ``max_active_tiles`` with the
                 usual strided worker loop; tiles past the true count exit after
                 reading their guard.  This is what a tile-dependency scheduler
                 emits today when the trip count is not known at compile time.
``static-exact`` Same strided loop, but bounded by the device-computed count.
                 Isolates "knowing the real trip count" from "pulling work".
``dynamic``      A work queue: every worker claims the next tile with an atomic
                 increment on a shared cursor and stops when the cursor passes
                 the device-computed count.

``--expert-task expert`` additionally coarsens a task from one expert tile to
one whole expert, which is where a ragged workload's load imbalance actually
shows up.  Comparing the two granularities under the three policies separates
"dynamic dispatch pays" from "fine static tiling pays".
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from probes.gemma4.gemma4_26b_a4b_common import benchmark_interleaved
from probes.gemma4.gemma4_26b_a4b_common import capture
from probes.gemma4.gemma4_26b_a4b_common import require_idle_visible_gpu
from probes.gemma4.gemma4_26b_a4b_common import visible_gpu_pids
from probes.gemma4.gemma4_a4b_moe_common import Gemma4A4BMoEShape
from probes.gemma4.gemma4_a4b_moe_common import align_experts_reference
from probes.gemma4.gemma4_a4b_moe_common import allocate_moe
from probes.gemma4.gemma4_a4b_moe_common import max_aligned_tiles
from probes.gemma4.gemma4_a4b_moe_common import moe_reference
from probes.gemma4.gemma4_a4b_moe_common import routing_histogram
from probes.gemma4.gemma4_a4b_moe_common import tiles_per_expert as tiles_per_expert_of

# ---------------------------------------------------------------------------
# Signalling.  Identical primitives to the E4B megakernel: fused arrival
# counters on every edge, wrap-safe epoch-relative targets.
# ---------------------------------------------------------------------------

_LIBCUDA = None


def _libcuda():
    global _LIBCUDA
    if _LIBCUDA is None:
        _LIBCUDA = ctypes.CDLL("libcuda.so.1")
    return _LIBCUDA


LINE = 32  # int32 elements per 128-byte cache line

S_ROUTER = 0
S_PRE_NORM = 1
S_TILES = 2
S_ORDER = 3
S_GATE_UP = 4
S_DOWN = 5
S_REDUCE = 6
S_POST_NORM = 7
S_LOGITS = 8
S_GROUP = 9  # + group id, per-group gate/up completion (fine-grained readiness)
NUM_FIXED_SLOTS = 9

D_LINE = tl.constexpr(LINE)
D_NUM_FIXED_SLOTS = tl.constexpr(NUM_FIXED_SLOTS)
D_S_ROUTER = tl.constexpr(S_ROUTER)
D_S_PRE_NORM = tl.constexpr(S_PRE_NORM)
D_S_TILES = tl.constexpr(S_TILES)
D_S_ORDER = tl.constexpr(S_ORDER)
D_S_GATE_UP = tl.constexpr(S_GATE_UP)
D_S_DOWN = tl.constexpr(S_DOWN)
D_S_REDUCE = tl.constexpr(S_REDUCE)
D_S_POST_NORM = tl.constexpr(S_POST_NORM)
D_S_LOGITS = tl.constexpr(S_LOGITS)

SCHEDULE_STATIC = 0
SCHEDULE_STATIC_EXACT = 1
SCHEDULE_DYNAMIC = 2
SCHEDULE_NAMES = {
    "static": SCHEDULE_STATIC,
    "static-exact": SCHEDULE_STATIC_EXACT,
    "dynamic": SCHEDULE_DYNAMIC,
}


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
def _globaltimer():
    return tl.inline_asm_elementwise(
        asm="mov.u64 $0, %globaltimer;",
        constraints="=l",
        args=[],
        dtype=tl.int64,
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
def _load_acquire(address):
    return tl.inline_asm_elementwise(
        asm="ld.acquire.gpu.global.u32 $0, [$1];",
        constraints="=r,l",
        args=[address],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _wait_count(address, target, POLL_DELAY: tl.constexpr):
    """Wait for an arrival counter to reach ``target`` (wrap-safe signed test)."""
    value = _load_acquire(address)
    while value - target < 0:
        if POLL_DELAY:
            _nanosleep(POLL_DELAY)
        value = _load_acquire(address)
    _sync_warp()


@triton.jit
def _slot_base(signals, slot):
    return signals + slot * D_LINE


@triton.jit
def _publish(signals, slot):
    tl.debug_barrier()
    tl.atomic_add(_slot_base(signals, slot), 1, sem="release", scope="gpu")


@triton.jit
def _wait(signals, slot, target, POLL_DELAY: tl.constexpr, ENABLED: tl.constexpr):
    """Acquire readiness for ``slot``.

    ``ENABLED=False`` drops the waits while keeping every publication, which
    measures the no-wait floor: what the same roots cost in one launch with zero
    serialization.  The results are garbage; the time is not.
    """
    if ENABLED:
        _wait_count(_slot_base(signals, slot), target, POLL_DELAY)


@triton.jit
def _in(mask, slot):
    """Root selection for isolation benchmarks; -1 enables every root."""
    return mask < 0 or ((mask >> slot) & 1) == 1


@triton.jit
def _gelu_tanh(value):
    coefficient: tl.constexpr = 0.7978845608028654
    return (
        0.5
        * value
        * (
            1.0
            + libdevice.tanh(coefficient * (value + 0.044715 * value * value * value))
        )
    )


_OUTLINE_ROOTS = os.environ.get("GEMMA4_OUTLINE_ROOTS", "0") == "1"
_root_jit = triton.jit(noinline=True) if _OUTLINE_ROOTS else triton.jit


# ---------------------------------------------------------------------------
# Root bodies.
# ---------------------------------------------------------------------------


@_root_jit
def _root_router(
    residual,
    router_scale,
    router_weight,
    per_expert_scale,
    topk_weights,
    topk_ids,
    token,
    root_size,
    H: tl.constexpr,
    E: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EPS: tl.constexpr,
):
    """Router pre-norm, learned scale, FP32 projection, and top-k selection.

    Mirrors ``router_projection_topk``: the RMSNorm result is rounded to BF16
    before the learned scale, the projection accumulates in FP32, and the
    softmax runs over the selected logits only.
    """
    experts = tl.arange(0, E)
    square_sum = tl.zeros([1], tl.float32)
    for start in tl.range(0, H, BLOCK_K):
        offsets = start + tl.arange(0, BLOCK_K)
        values = tl.load(residual + token * H + offsets).to(tl.float32)
        square_sum += tl.sum(values * values)[None]
    inv_rms = tl.rsqrt(square_sum * (1.0 / H) + EPS)

    logits = tl.zeros([E], tl.float32)
    for start in tl.range(0, H, BLOCK_K):
        offsets = start + tl.arange(0, BLOCK_K)
        values = tl.load(residual + token * H + offsets).to(tl.float32)
        normalized = (values * inv_rms).to(tl.bfloat16)
        scaled = normalized * root_size * tl.load(router_scale + offsets)
        weight = tl.load(router_weight + experts[:, None] * H + offsets[None, :])
        logits += tl.sum(weight.to(tl.float32) * scaled.to(tl.float32)[None, :], axis=1)

    lanes = tl.arange(0, TOPK)
    chosen = tl.zeros([TOPK], tl.int32)
    weights = tl.zeros([TOPK], tl.float32)
    largest = tl.max(logits, axis=0)
    for slot in tl.static_range(TOPK):
        value = tl.max(logits, axis=0)
        index = tl.argmax(logits, axis=0)
        chosen = tl.where(lanes == slot, index, chosen)
        weights = tl.where(lanes == slot, tl.exp(value - largest), weights)
        logits = tl.where(experts == index, float("-inf"), logits)
    weights = weights / tl.sum(weights, axis=0)
    weights = weights * tl.load(per_expert_scale + chosen).to(tl.float32)
    tl.store(topk_weights + token * TOPK + lanes, weights)
    tl.store(topk_ids + token * TOPK + lanes, chosen)


@triton.jit
def _row_and_inv_rms(
    source, token, H: tl.constexpr, HPAD: tl.constexpr, EPS: tl.constexpr
):
    """Load a whole row and its inverse RMS in one pass.

    The blocked form is a chain of dependent loads and reductions -- eleven of
    them for H=2816 -- and on a root that owns one row that chain *is* the root's
    cost: the router measured 8.3 us and the post-norm 5.4 us for 0.72 MB and
    5.6 KB respectively.  The row is 5.6 KB, so it fits in registers padded to a
    power of two, and one load plus one reduction replaces the chain.
    """
    offsets = tl.arange(0, HPAD)
    values = tl.load(source + token * H + offsets, mask=offsets < H, other=0.0).to(
        tl.float32
    )
    return values, tl.rsqrt(tl.sum(values * values) * (1.0 / H) + EPS)


@_root_jit
def _root_router_project(
    residual,
    router_scale,
    router_weight,
    router_logits,
    token,
    expert_block,
    root_size,
    H: tl.constexpr,
    E: tl.constexpr,
    EXPERT_BLOCK: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HPAD: tl.constexpr,
    EPS: tl.constexpr,
):
    """One expert block of the router projection.

    Fission of ``_root_router``.  The whole-token RMS is recomputed per task
    rather than published by a predecessor: the input row is 5.6 KB and stays in
    L1, so recomputing it is cheaper than an extra readiness edge, and it keeps
    the task independent.  Splitting the expert axis is also what drops the
    register footprint that otherwise sets the whole megakernel's occupancy.
    """
    experts = expert_block * EXPERT_BLOCK + tl.arange(0, EXPERT_BLOCK)
    offsets = tl.arange(0, HPAD)
    row_mask = offsets < H
    row, inv_rms = _row_and_inv_rms(residual, token, H, HPAD, EPS)
    normalized = (row * inv_rms).to(tl.bfloat16)
    scaled = (
        normalized
        * root_size
        * tl.load(router_scale + offsets, mask=row_mask, other=0.0)
    ).to(tl.float32)
    weight = tl.load(
        router_weight + experts[:, None] * H + offsets[None, :],
        mask=row_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    logits = tl.sum(weight * scaled[None, :], axis=1)
    tl.store(router_logits + token * E + experts, logits)


@_root_jit
def _root_router_topk(
    router_logits,
    per_expert_scale,
    topk_weights,
    topk_ids,
    token,
    E: tl.constexpr,
    TOPK: tl.constexpr,
):
    """Top-k selection over one token's already-projected logits."""
    experts = tl.arange(0, E)
    logits = tl.load(router_logits + token * E + experts)
    lanes = tl.arange(0, TOPK)
    chosen = tl.zeros([TOPK], tl.int32)
    weights = tl.zeros([TOPK], tl.float32)
    largest = tl.max(logits, axis=0)
    for slot in tl.static_range(TOPK):
        value = tl.max(logits, axis=0)
        index = tl.argmax(logits, axis=0)
        chosen = tl.where(lanes == slot, index, chosen)
        weights = tl.where(lanes == slot, tl.exp(value - largest), weights)
        logits = tl.where(experts == index, float("-inf"), logits)
    weights = weights / tl.sum(weights, axis=0)
    weights = weights * tl.load(per_expert_scale + chosen).to(tl.float32)
    tl.store(topk_weights + token * TOPK + lanes, weights)
    tl.store(topk_ids + token * TOPK + lanes, chosen)


@_root_jit
def _root_rms_norm(
    source,
    weight,
    destination,
    token,
    H: tl.constexpr,
    BLOCK: tl.constexpr,
    HPAD: tl.constexpr,
    EPS: tl.constexpr,
):
    """vLLM RMSNorm semantics, including the BF16 pre-weight rounding point."""
    offsets = tl.arange(0, HPAD)
    row_mask = offsets < H
    values, inv_rms = _row_and_inv_rms(source, token, H, HPAD, EPS)
    normalized = (values * inv_rms).to(tl.bfloat16)
    tl.store(
        destination + token * H + offsets,
        normalized * tl.load(weight + offsets, mask=row_mask, other=0.0),
        mask=row_mask,
    )


@_root_jit
def _root_expert_tiles(
    topk_ids,
    expert_counts,
    tile_end,
    active_tiles,
    signals,
    gate_up_cursor,
    down_cursor,
    gate_up_counter,
    down_counter,
    A: tl.constexpr,
    E: tl.constexpr,
    TM: tl.constexpr,
    TPE: tl.constexpr,
    BLOCK_A: tl.constexpr,
    GROUP_SLOTS: tl.constexpr,
):
    """Per-expert token counts, tile prefix sums, and the packed active list.

    This root also zeroes the ragged roots' cursors, arrival counters, and
    per-group readiness slots.  Their fan-in is decided here, so it cannot be
    folded into the epoch-relative target the fixed-fan-in edges use; zeroing is
    safe because every producer and consumer of those words acquires this root's
    release first.
    """
    experts = tl.arange(0, E)
    counts = tl.zeros([E], tl.int32)
    for start in tl.range(0, A, BLOCK_A):
        offsets = start + tl.arange(0, BLOCK_A)
        ids = tl.load(topk_ids + offsets, mask=offsets < A, other=-1)
        counts += tl.sum((ids[None, :] == experts[:, None]).to(tl.int32), axis=1)
    tiles = (counts + (TM - 1)) // TM
    ends = tl.cumsum(tiles, axis=0).to(tl.int32)
    tl.store(expert_counts + experts, counts)
    tl.store(tile_end + experts, ends)
    starts = ends - tiles
    for local in tl.static_range(TPE):
        tl.store(
            active_tiles + starts + local,
            (experts * TPE + local).to(tl.int32),
            mask=local < tiles,
        )
    tl.store(gate_up_cursor, 0)
    tl.store(down_cursor, 0)
    tl.store(gate_up_counter, 0)
    tl.store(down_counter, 0)
    if GROUP_SLOTS > 0:
        for start in tl.range(0, GROUP_SLOTS, 64):
            slots = start + tl.arange(0, 64)
            tl.store(
                signals + (D_NUM_FIXED_SLOTS + slots) * D_LINE,
                tl.zeros([64], tl.int32),
                mask=slots < GROUP_SLOTS,
            )


@_root_jit
def _root_assignment_order(
    topk_ids,
    order,
    task,
    A: tl.constexpr,
    TM: tl.constexpr,
    TPE: tl.constexpr,
    BLOCK_A: tl.constexpr,
):
    """Stable expert-major placement of one block of assignments."""
    stride: tl.constexpr = TPE * TM
    mine_offsets = task * BLOCK_A + tl.arange(0, BLOCK_A)
    mine_valid = mine_offsets < A
    mine = tl.load(topk_ids + mine_offsets, mask=mine_valid, other=-1)
    rank = tl.zeros([BLOCK_A], tl.int32)
    for start in tl.range(0, A, BLOCK_A):
        offsets = start + tl.arange(0, BLOCK_A)
        other = tl.load(topk_ids + offsets, mask=offsets < A, other=-2)
        earlier = offsets[None, :] < mine_offsets[:, None]
        same = other[None, :] == mine[:, None]
        rank += tl.sum((earlier & same).to(tl.int32), axis=1)
    tl.store(order + mine * stride + rank, mine_offsets, mask=mine_valid)


@triton.jit
def _tile_rows(
    active_tiles,
    expert_counts,
    order,
    tile,
    TM: tl.constexpr,
    TPE: tl.constexpr,
    TOPK: tl.constexpr,
):
    """Decode one active tile into its expert, valid row mask, and assignments."""
    stride: tl.constexpr = TPE * TM
    group = tl.load(active_tiles + tile)
    expert = group // TPE
    local = group - expert * TPE
    rows = local * TM + tl.arange(0, TM)
    valid = rows < tl.load(expert_counts + expert)
    assignment = tl.load(order + expert * stride + rows, mask=valid, other=0)
    return expert, valid, assignment, assignment // TOPK


@_root_jit
def _root_grouped_gate_up(
    expert_input,
    expert_weight,
    activation,
    active_tiles,
    expert_counts,
    order,
    tile,
    column_block,
    H: tl.constexpr,
    INTERMEDIATE: tl.constexpr,
    TM: tl.constexpr,
    TPE: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    STAGES: tl.constexpr,
    LOAD_NK: tl.constexpr,
    EVICT: tl.constexpr,
):
    """Expert-grouped gate/up projection with its GeGLU epilogue.

    ``LOAD_NK`` picks how the weight tile is fetched.  The expert weight is
    ``[row, k]`` row-major, so ``k`` is the contiguous axis; loading the tile as
    ``[n, k]`` puts that axis last, which is the shape Triton coalesces, and
    then transposes in registers for the dot.  Loading it as ``[k, n]`` skips the
    transpose but asks for a strided fetch.
    """
    twice: tl.constexpr = 2 * INTERMEDIATE
    expert, valid, assignment, token = _tile_rows(
        active_tiles, expert_counts, order, tile, TM, TPE, TOPK
    )
    columns = column_block * BLOCK_N + tl.arange(0, BLOCK_N)
    column_valid = columns < INTERMEDIATE
    gate_rows = expert * twice + tl.where(column_valid, columns, 0)
    up_rows = gate_rows + INTERMEDIATE
    gate_acc = tl.zeros([TM, BLOCK_N], tl.float32)
    up_acc = tl.zeros([TM, BLOCK_N], tl.float32)
    for start in tl.range(0, H, BLOCK_K, num_stages=STAGES):
        offsets = start + tl.arange(0, BLOCK_K)
        values = tl.load(
            expert_input + token[:, None] * H + offsets[None, :],
            mask=valid[:, None],
            other=0.0,
        )
        if LOAD_NK:
            gate_weight = tl.load(
                expert_weight + gate_rows[:, None] * H + offsets[None, :],
                eviction_policy=EVICT,
            )
            up_weight = tl.load(
                expert_weight + up_rows[:, None] * H + offsets[None, :],
                eviction_policy=EVICT,
            )
            gate_acc = tl.dot(values, tl.trans(gate_weight), gate_acc)
            up_acc = tl.dot(values, tl.trans(up_weight), up_acc)
        else:
            gate_weight = tl.load(
                expert_weight + gate_rows[None, :] * H + offsets[:, None],
                eviction_policy=EVICT,
            )
            up_weight = tl.load(
                expert_weight + up_rows[None, :] * H + offsets[:, None],
                eviction_policy=EVICT,
            )
            gate_acc = tl.dot(values, gate_weight, gate_acc)
            up_acc = tl.dot(values, up_weight, up_acc)
    gate = gate_acc.to(tl.bfloat16).to(tl.float32)
    up = up_acc.to(tl.bfloat16)
    tl.store(
        activation + assignment[:, None] * INTERMEDIATE + columns[None, :],
        _gelu_tanh(gate).to(tl.bfloat16) * up,
        mask=valid[:, None] & column_valid[None, :],
    )


@_root_jit
def _root_gathered_gate_up(
    expert_input,
    residual,
    pre_ff_norm_weight,
    expert_weight,
    topk_ids,
    activation,
    task,
    H: tl.constexpr,
    INTERMEDIATE: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    STAGES: tl.constexpr,
    EVICT: tl.constexpr,
    FUSE_PRE_NORM: tl.constexpr,
    HPAD: tl.constexpr,
    EPS: tl.constexpr,
):
    """One (token, slot) assignment's gate/up GEMV plus its GeGLU epilogue.

    This is the batch-1 production formulation, mirroring
    ``expert_geglu_projection``: the output domain is the flat
    ``assignment x intermediate`` grid, so an assignment's weights are streamed
    once and there is no ragged tile count at all.  ``BLOCK_N`` must divide
    ``INTERMEDIATE`` so a task never straddles two assignments, which keeps the
    activation row a broadcast scalar rather than a per-row gather.
    """
    slot = (task * BLOCK_N) // INTERMEDIATE
    columns = (task * BLOCK_N) % INTERMEDIATE + tl.arange(0, BLOCK_N)
    expert = tl.load(topk_ids + slot)
    token = slot // TOPK
    gate_rows = expert * (2 * INTERMEDIATE) + columns
    up_rows = gate_rows + INTERMEDIATE
    gate_acc = tl.zeros([BLOCK_N], tl.float32)
    up_acc = tl.zeros([BLOCK_N], tl.float32)
    if FUSE_PRE_NORM:
        # The pre-MoE RMSNorm is one task on one worker, so as a separate root
        # every other worker stalls on it.  Recomputing it here costs one extra
        # pass over a 5.6 KB row that is already L1-resident, and removes both
        # the root and its barrier.
        _, inv_rms = _row_and_inv_rms(residual, token, H, HPAD, EPS)
    for start in tl.range(0, H, BLOCK_K, num_stages=STAGES):
        offsets = start + tl.arange(0, BLOCK_K)
        if FUSE_PRE_NORM:
            row = tl.load(residual + token * H + offsets).to(tl.float32)
            normalized = (row * inv_rms).to(tl.bfloat16)
            values = (normalized * tl.load(pre_ff_norm_weight + offsets)).to(tl.float32)
        else:
            values = tl.load(expert_input + token * H + offsets).to(tl.float32)
        gate_weight = tl.load(
            expert_weight + gate_rows[:, None] * H + offsets[None, :],
            eviction_policy=EVICT,
        ).to(tl.float32)
        up_weight = tl.load(
            expert_weight + up_rows[:, None] * H + offsets[None, :],
            eviction_policy=EVICT,
        ).to(tl.float32)
        gate_acc += tl.sum(gate_weight * values[None, :], axis=1)
        up_acc += tl.sum(up_weight * values[None, :], axis=1)
    gate = gate_acc.to(tl.bfloat16).to(tl.float32)
    up = up_acc.to(tl.bfloat16)
    tl.store(
        activation + slot * INTERMEDIATE + columns,
        _gelu_tanh(gate).to(tl.bfloat16) * up,
    )


@_root_jit
def _root_gathered_down(
    activation,
    expert_weight,
    topk_ids,
    expert_outputs,
    task,
    H: tl.constexpr,
    INTERMEDIATE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    STAGES: tl.constexpr,
    EVICT: tl.constexpr,
):
    """One assignment's down GEMV, mirroring ``expert_down``."""
    slot = (task * BLOCK_N) // H
    columns = (task * BLOCK_N) % H + tl.arange(0, BLOCK_N)
    expert = tl.load(topk_ids + slot)
    weight_rows = expert * H + columns
    accumulator = tl.zeros([BLOCK_N], tl.float32)
    for start in tl.range(0, INTERMEDIATE, BLOCK_K, num_stages=STAGES):
        offsets = start + tl.arange(0, BLOCK_K)
        values = tl.load(activation + slot * INTERMEDIATE + offsets).to(tl.float32)
        weight = tl.load(
            expert_weight + weight_rows[:, None] * INTERMEDIATE + offsets[None, :],
            eviction_policy=EVICT,
        ).to(tl.float32)
        accumulator += tl.sum(weight * values[None, :], axis=1)
    tl.store(expert_outputs + slot * H + columns, accumulator.to(tl.bfloat16))


@_root_jit
def _root_grouped_down(
    activation,
    expert_weight,
    expert_outputs,
    active_tiles,
    expert_counts,
    order,
    tile,
    column_block,
    H: tl.constexpr,
    INTERMEDIATE: tl.constexpr,
    TM: tl.constexpr,
    TPE: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    STAGES: tl.constexpr,
    LOAD_NK: tl.constexpr,
    EVICT: tl.constexpr,
):
    """Expert-grouped down projection, one row per (token, slot) assignment."""
    expert, valid, assignment, _ = _tile_rows(
        active_tiles, expert_counts, order, tile, TM, TPE, TOPK
    )
    columns = column_block * BLOCK_N + tl.arange(0, BLOCK_N)
    weight_rows = expert * H + columns
    accumulator = tl.zeros([TM, BLOCK_N], tl.float32)
    for start in tl.range(0, INTERMEDIATE, BLOCK_K, num_stages=STAGES):
        offsets = start + tl.arange(0, BLOCK_K)
        values = tl.load(
            activation + assignment[:, None] * INTERMEDIATE + offsets[None, :],
            mask=valid[:, None],
            other=0.0,
        )
        if LOAD_NK:
            weight = tl.load(
                expert_weight + weight_rows[:, None] * INTERMEDIATE + offsets[None, :],
                eviction_policy=EVICT,
            )
            accumulator = tl.dot(values, tl.trans(weight), accumulator)
        else:
            weight = tl.load(
                expert_weight + weight_rows[None, :] * INTERMEDIATE + offsets[:, None],
                eviction_policy=EVICT,
            )
            accumulator = tl.dot(values, weight, accumulator)
    tl.store(
        expert_outputs + assignment[:, None] * H + columns[None, :],
        accumulator.to(tl.bfloat16),
        mask=valid[:, None],
    )


@_root_jit
def _root_reduce(
    expert_outputs,
    topk_weights,
    moe_down,
    token,
    column_block,
    H: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Routing-weighted reduction over one token's top-k expert outputs."""
    columns = column_block * BLOCK_N + tl.arange(0, BLOCK_N)
    slots = tl.arange(0, TOPK)
    values = tl.load(
        expert_outputs + (token * TOPK + slots)[:, None] * H + columns[None, :]
    ).to(tl.float32)
    weights = tl.load(topk_weights + token * TOPK + slots)
    tl.store(
        moe_down + token * H + columns,
        tl.sum(values * weights[:, None], axis=0).to(tl.bfloat16),
    )


# ---------------------------------------------------------------------------
# The megakernel.
# ---------------------------------------------------------------------------


@triton.jit
def moe_megakernel(
    residual,
    pre_ff_norm_weight,
    router_scale,
    router_weight,
    per_expert_scale,
    expert_gate_up_weight,
    expert_down_weight,
    post_ff_norm_weight,
    root_size,
    expert_input,
    topk_weights,
    topk_ids,
    router_logits,
    expert_counts,
    tile_end,
    active_tiles,
    order,
    activation,
    expert_outputs,
    moe_down,
    moe_branch,
    worker_epoch,
    signals,
    gate_up_cursor,
    down_cursor,
    gate_up_counter,
    down_counter,
    TOTAL_WORKERS: tl.constexpr,
    B: tl.constexpr,
    H: tl.constexpr,
    INTERMEDIATE: tl.constexpr,
    E: tl.constexpr,
    TOPK: tl.constexpr,
    TM: tl.constexpr,
    TPE: tl.constexpr,
    MAX_ACTIVE: tl.constexpr,
    UNITS_STATIC: tl.constexpr,
    GROUP_SLOTS: tl.constexpr,
    NORM_BLOCK: tl.constexpr,
    ROUTER_BLOCK_K: tl.constexpr,
    ROUTER_EXPERT_BLOCK: tl.constexpr,
    HPAD: tl.constexpr,
    ROUTER_SPLIT: tl.constexpr,
    FUSE_PRE_NORM: tl.constexpr,
    ORDER_BLOCK: tl.constexpr,
    GATE_BLOCK_N: tl.constexpr,
    GATE_BLOCK_K: tl.constexpr,
    GATE_STAGES: tl.constexpr,
    DOWN_BLOCK_N: tl.constexpr,
    DOWN_BLOCK_K: tl.constexpr,
    DOWN_STAGES: tl.constexpr,
    REDUCE_BLOCK: tl.constexpr,
    LOAD_NK: tl.constexpr,
    EVICT: tl.constexpr,
    ROOT_MASK: tl.constexpr,
    GATHERED: tl.constexpr,
    DOWN_FANIN: tl.constexpr,
    DOWN_ARRIVALS: tl.constexpr,
    FUSE_TASKS: tl.constexpr,
    SCHEDULE: tl.constexpr,
    EXPERT_TASK: tl.constexpr,
    GROUP_CONTINUATION: tl.constexpr,
    POLL_DELAY: tl.constexpr,
    WAITS: tl.constexpr,
    EPS: tl.constexpr,
):
    worker = tl.program_id(0)
    epoch = tl.load(worker_epoch + worker) + 1

    assignments: tl.constexpr = B * TOPK
    order_tasks: tl.constexpr = (assignments + ORDER_BLOCK - 1) // ORDER_BLOCK
    gate_columns: tl.constexpr = (INTERMEDIATE + GATE_BLOCK_N - 1) // GATE_BLOCK_N
    down_columns: tl.constexpr = H // DOWN_BLOCK_N
    reduce_columns: tl.constexpr = H // REDUCE_BLOCK
    router_workers: tl.constexpr = min(TOTAL_WORKERS, B)
    norm_workers: tl.constexpr = min(TOTAL_WORKERS, B)
    expert_blocks: tl.constexpr = E // ROUTER_EXPERT_BLOCK
    project_tasks: tl.constexpr = B * expert_blocks
    project_workers: tl.constexpr = min(TOTAL_WORKERS, project_tasks)
    order_workers: tl.constexpr = min(TOTAL_WORKERS, order_tasks)
    reduce_tasks: tl.constexpr = B * reduce_columns
    reduce_workers: tl.constexpr = min(TOTAL_WORKERS, reduce_tasks)
    post_workers: tl.constexpr = min(TOTAL_WORKERS, B)

    # -- R0: router ---------------------------------------------------------
    #
    # ``ROUTER_SPLIT`` fissions the single fused router task into one task per
    # (token, expert block) plus a singleton top-k.  At batch 1 the fused form
    # is one task on one worker while 295 workers wait on it, and it is both the
    # largest root and the head of the dependency chain.
    if ROUTER_SPLIT:
        if worker < project_workers and _in(ROOT_MASK, D_S_ROUTER):
            for task in tl.range(worker, project_tasks, TOTAL_WORKERS):
                _root_router_project(
                    residual,
                    router_scale,
                    router_weight,
                    router_logits,
                    task // expert_blocks,
                    task % expert_blocks,
                    tl.load(root_size),
                    H,
                    E,
                    ROUTER_EXPERT_BLOCK,
                    ROUTER_BLOCK_K,
                    HPAD,
                    EPS,
                )
            _publish(signals, D_S_LOGITS)
        if worker < router_workers and _in(ROOT_MASK, D_S_ROUTER):
            _wait(signals, D_S_LOGITS, epoch * project_workers, POLL_DELAY, WAITS)
            for token in tl.range(worker, B, TOTAL_WORKERS):
                _root_router_topk(
                    router_logits,
                    per_expert_scale,
                    topk_weights,
                    topk_ids,
                    token,
                    E,
                    TOPK,
                )
            _publish(signals, D_S_ROUTER)
    else:
        if worker < router_workers and _in(ROOT_MASK, D_S_ROUTER):
            for token in tl.range(worker, B, TOTAL_WORKERS):
                _root_router(
                    residual,
                    router_scale,
                    router_weight,
                    per_expert_scale,
                    topk_weights,
                    topk_ids,
                    token,
                    tl.load(root_size),
                    H,
                    E,
                    TOPK,
                    ROUTER_BLOCK_K,
                    EPS,
                )
            _publish(signals, D_S_ROUTER)

    # -- R1: pre-MoE RMSNorm, independent of the router --------------------
    # The pre-norm root is only removable on the gathered path, whose gate root
    # recomputes it.  The grouped roots still consume ``expert_input``, so
    # eliding it there leaves them waiting on a release nobody publishes.
    elide_pre_norm: tl.constexpr = GATHERED and FUSE_PRE_NORM
    if worker < norm_workers and _in(ROOT_MASK, D_S_PRE_NORM) and not elide_pre_norm:
        for token in tl.range(worker, B, TOTAL_WORKERS):
            _root_rms_norm(
                residual,
                pre_ff_norm_weight,
                expert_input,
                token,
                H,
                NORM_BLOCK,
                HPAD,
                EPS,
            )
        _publish(signals, D_S_PRE_NORM)

    # -- R2-R5, gathered: the batch-1 production formulation ---------------
    #
    # One task per (assignment, column block).  Nothing here is ragged: an
    # assignment has exactly one expert, so the task count is a compile-time
    # constant and every fan-in is epoch-relative.  The expert-tile metadata and
    # assignment-order roots do not exist on this path at all.
    gathered_gate_tasks: tl.constexpr = (assignments * INTERMEDIATE) // GATE_BLOCK_N
    gathered_down_tasks: tl.constexpr = (assignments * H) // DOWN_BLOCK_N
    gathered_gate_workers: tl.constexpr = min(TOTAL_WORKERS, gathered_gate_tasks)
    gathered_down_workers: tl.constexpr = min(TOTAL_WORKERS, gathered_down_tasks)
    fused_tasks: tl.constexpr = gathered_gate_tasks + gathered_down_tasks
    gate_per_slot: tl.constexpr = INTERMEDIATE // GATE_BLOCK_N
    if GATHERED and FUSE_TASKS:
        # One task space over both roots.  Separate roots make each one's ragged
        # tail an exposed wave: at 296 workers gate needs ceil(352/296)=2 rounds
        # and down ceil(176/296)=1, three round-times for 1.8 rounds of work.
        # Fused, a worker that runs out of gate work takes a down task whose own
        # assignment is already complete, and the whole edge costs
        # ceil(528/296)=2.  Legality comes from the per-assignment readiness
        # words; termination comes from tasks being visited in increasing order,
        # so every gate task retires before any worker can block on one.
        _wait(signals, D_S_ROUTER, epoch * router_workers, POLL_DELAY, WAITS)
        if not FUSE_PRE_NORM:
            _wait(signals, D_S_PRE_NORM, epoch * norm_workers, POLL_DELAY, WAITS)
        for task in tl.range(worker, fused_tasks, TOTAL_WORKERS):
            if task < gathered_gate_tasks:
                _root_gathered_gate_up(
                    expert_input,
                    residual,
                    pre_ff_norm_weight,
                    expert_gate_up_weight,
                    topk_ids,
                    activation,
                    task,
                    H,
                    INTERMEDIATE,
                    TOPK,
                    GATE_BLOCK_N,
                    GATE_BLOCK_K,
                    GATE_STAGES,
                    EVICT,
                    FUSE_PRE_NORM,
                    HPAD,
                    EPS,
                )
                tl.debug_barrier()
                tl.atomic_add(
                    signals
                    + (D_NUM_FIXED_SLOTS + (task * GATE_BLOCK_N) // INTERMEDIATE)
                    * D_LINE,
                    1,
                    sem="release",
                    scope="gpu",
                )
            else:
                down_task = task - gathered_gate_tasks
                _wait(
                    signals,
                    D_NUM_FIXED_SLOTS + (down_task * DOWN_BLOCK_N) // H,
                    epoch * gate_per_slot,
                    POLL_DELAY,
                    WAITS,
                )
                _root_gathered_down(
                    activation,
                    expert_down_weight,
                    topk_ids,
                    expert_outputs,
                    down_task,
                    H,
                    INTERMEDIATE,
                    DOWN_BLOCK_N,
                    DOWN_BLOCK_K,
                    DOWN_STAGES,
                    EVICT,
                )
                _publish(signals, D_S_DOWN)
    elif GATHERED:
        if worker < gathered_gate_workers and _in(ROOT_MASK, D_S_GATE_UP):
            _wait(signals, D_S_ROUTER, epoch * router_workers, POLL_DELAY, WAITS)
            if not FUSE_PRE_NORM:
                _wait(signals, D_S_PRE_NORM, epoch * norm_workers, POLL_DELAY, WAITS)
            for task in tl.range(worker, gathered_gate_tasks, TOTAL_WORKERS):
                _root_gathered_gate_up(
                    expert_input,
                    residual,
                    pre_ff_norm_weight,
                    expert_gate_up_weight,
                    topk_ids,
                    activation,
                    task,
                    H,
                    INTERMEDIATE,
                    TOPK,
                    GATE_BLOCK_N,
                    GATE_BLOCK_K,
                    GATE_STAGES,
                    EVICT,
                    FUSE_PRE_NORM,
                    HPAD,
                    EPS,
                )
                if GROUP_CONTINUATION:
                    # A down task for assignment `a` needs only `activation[a]`,
                    # so it can start once that assignment's column blocks are
                    # done rather than after the whole root.  The assignment set
                    # is static here, so the target stays epoch-relative.
                    tl.debug_barrier()
                    tl.atomic_add(
                        signals
                        + (D_NUM_FIXED_SLOTS + (task * GATE_BLOCK_N) // INTERMEDIATE)
                        * D_LINE,
                        1,
                        sem="release",
                        scope="gpu",
                    )
            if not GROUP_CONTINUATION:
                _publish(signals, D_S_GATE_UP)
        if worker < gathered_down_workers and _in(ROOT_MASK, D_S_DOWN):
            if not GROUP_CONTINUATION:
                _wait(
                    signals,
                    D_S_GATE_UP,
                    epoch * gathered_gate_workers,
                    POLL_DELAY,
                    WAITS,
                )
            for task in tl.range(worker, gathered_down_tasks, TOTAL_WORKERS):
                if GROUP_CONTINUATION:
                    _wait(
                        signals,
                        D_NUM_FIXED_SLOTS + (task * DOWN_BLOCK_N) // H,
                        epoch * (INTERMEDIATE // GATE_BLOCK_N),
                        POLL_DELAY,
                        WAITS,
                    )
                _root_gathered_down(
                    activation,
                    expert_down_weight,
                    topk_ids,
                    expert_outputs,
                    task,
                    H,
                    INTERMEDIATE,
                    DOWN_BLOCK_N,
                    DOWN_BLOCK_K,
                    DOWN_STAGES,
                    EVICT,
                )
            _publish(signals, D_S_DOWN)
    else:
        # -- R2: expert tile metadata (singleton) ------------------------------
        if worker == 0:
            _wait(signals, D_S_ROUTER, epoch * router_workers, POLL_DELAY, WAITS)
            _root_expert_tiles(
                topk_ids,
                expert_counts,
                tile_end,
                active_tiles,
                signals,
                gate_up_cursor,
                down_cursor,
                gate_up_counter,
                down_counter,
                assignments,
                E,
                TM,
                TPE,
                ORDER_BLOCK,
                GROUP_SLOTS,
            )
            _publish(signals, D_S_TILES)

        # -- R3: stable expert-major assignment order --------------------------
        if worker < order_workers:
            _wait(signals, D_S_ROUTER, epoch * router_workers, POLL_DELAY, WAITS)
            for task in tl.range(worker, order_tasks, TOTAL_WORKERS):
                _root_assignment_order(
                    topk_ids, order, task, assignments, TM, TPE, ORDER_BLOCK
                )
            _publish(signals, D_S_ORDER)

        # -- R4: expert-grouped gate/up, the first ragged root -----------------
        #
        # ``units`` is the run-time number of dispatch units: active tiles, or all
        # experts when a task is a whole expert.  ``GATE_STATIC`` is the compile-time
        # upper bound the plain static schedule must walk instead.
        _wait(signals, D_S_TILES, epoch, POLL_DELAY, WAITS)
        _wait(signals, D_S_ORDER, epoch * order_workers, POLL_DELAY, WAITS)
        _wait(signals, D_S_PRE_NORM, epoch * norm_workers, POLL_DELAY, WAITS)
        active = tl.load(tile_end + (E - 1))
        units = active * 0 + UNITS_STATIC if EXPERT_TASK else active
        gate_tasks = units * gate_columns
        down_tasks = units * down_columns
        gate_static: tl.constexpr = UNITS_STATIC * gate_columns
        down_static: tl.constexpr = UNITS_STATIC * down_columns

        if SCHEDULE == 2:
            claimed = tl.atomic_add(gate_up_cursor, 1, sem="relaxed", scope="gpu")
            while claimed < gate_tasks:
                _dispatch_gate_up(
                    expert_input,
                    expert_gate_up_weight,
                    activation,
                    active_tiles,
                    expert_counts,
                    tile_end,
                    order,
                    signals,
                    gate_up_counter,
                    claimed,
                    units,
                    gate_columns,
                    GROUP_CONTINUATION,
                    H,
                    INTERMEDIATE,
                    TM,
                    TPE,
                    TOPK,
                    GATE_BLOCK_N,
                    GATE_BLOCK_K,
                    GATE_STAGES,
                    LOAD_NK,
                    EVICT,
                    EXPERT_TASK,
                )
                claimed = tl.atomic_add(gate_up_cursor, 1, sem="relaxed", scope="gpu")
        elif SCHEDULE == 1:
            for task in tl.range(worker, gate_tasks, TOTAL_WORKERS):
                _dispatch_gate_up(
                    expert_input,
                    expert_gate_up_weight,
                    activation,
                    active_tiles,
                    expert_counts,
                    tile_end,
                    order,
                    signals,
                    gate_up_counter,
                    task,
                    units,
                    gate_columns,
                    GROUP_CONTINUATION,
                    H,
                    INTERMEDIATE,
                    TM,
                    TPE,
                    TOPK,
                    GATE_BLOCK_N,
                    GATE_BLOCK_K,
                    GATE_STAGES,
                    LOAD_NK,
                    EVICT,
                    EXPERT_TASK,
                )
        else:
            for task in tl.range(worker, gate_static, TOTAL_WORKERS):
                _dispatch_gate_up(
                    expert_input,
                    expert_gate_up_weight,
                    activation,
                    active_tiles,
                    expert_counts,
                    tile_end,
                    order,
                    signals,
                    gate_up_counter,
                    task,
                    units,
                    gate_columns,
                    GROUP_CONTINUATION,
                    H,
                    INTERMEDIATE,
                    TM,
                    TPE,
                    TOPK,
                    GATE_BLOCK_N,
                    GATE_BLOCK_K,
                    GATE_STAGES,
                    LOAD_NK,
                    EVICT,
                    EXPERT_TASK,
                )

        # -- R5: expert-grouped down, the second ragged root -------------------
        if not GROUP_CONTINUATION:
            _wait(signals, D_S_GATE_UP, epoch, POLL_DELAY, WAITS)

        if SCHEDULE == 2:
            claimed = tl.atomic_add(down_cursor, 1, sem="relaxed", scope="gpu")
            while claimed < down_tasks:
                _dispatch_down(
                    activation,
                    expert_down_weight,
                    expert_outputs,
                    active_tiles,
                    expert_counts,
                    tile_end,
                    order,
                    signals,
                    down_counter,
                    claimed,
                    units,
                    down_columns,
                    GROUP_CONTINUATION,
                    POLL_DELAY,
                    WAITS,
                    H,
                    INTERMEDIATE,
                    TM,
                    TPE,
                    TOPK,
                    DOWN_BLOCK_N,
                    DOWN_BLOCK_K,
                    DOWN_STAGES,
                    LOAD_NK,
                    EVICT,
                    EXPERT_TASK,
                    gate_columns,
                )
                claimed = tl.atomic_add(down_cursor, 1, sem="relaxed", scope="gpu")
        elif SCHEDULE == 1:
            for task in tl.range(worker, down_tasks, TOTAL_WORKERS):
                _dispatch_down(
                    activation,
                    expert_down_weight,
                    expert_outputs,
                    active_tiles,
                    expert_counts,
                    tile_end,
                    order,
                    signals,
                    down_counter,
                    task,
                    units,
                    down_columns,
                    GROUP_CONTINUATION,
                    POLL_DELAY,
                    WAITS,
                    H,
                    INTERMEDIATE,
                    TM,
                    TPE,
                    TOPK,
                    DOWN_BLOCK_N,
                    DOWN_BLOCK_K,
                    DOWN_STAGES,
                    LOAD_NK,
                    EVICT,
                    EXPERT_TASK,
                    gate_columns,
                )
        else:
            for task in tl.range(worker, down_static, TOTAL_WORKERS):
                _dispatch_down(
                    activation,
                    expert_down_weight,
                    expert_outputs,
                    active_tiles,
                    expert_counts,
                    tile_end,
                    order,
                    signals,
                    down_counter,
                    task,
                    units,
                    down_columns,
                    GROUP_CONTINUATION,
                    POLL_DELAY,
                    WAITS,
                    H,
                    INTERMEDIATE,
                    TM,
                    TPE,
                    TOPK,
                    DOWN_BLOCK_N,
                    DOWN_BLOCK_K,
                    DOWN_STAGES,
                    LOAD_NK,
                    EVICT,
                    EXPERT_TASK,
                    gate_columns,
                )

    # -- R6: routing-weighted reduction ------------------------------------
    if worker < reduce_workers and _in(ROOT_MASK, D_S_REDUCE):
        _wait(signals, D_S_DOWN, epoch * DOWN_ARRIVALS, POLL_DELAY, WAITS)
        for task in tl.range(worker, reduce_tasks, TOTAL_WORKERS):
            _root_reduce(
                expert_outputs,
                topk_weights,
                moe_down,
                task // reduce_columns,
                task % reduce_columns,
                H,
                TOPK,
                REDUCE_BLOCK,
            )
        _publish(signals, D_S_REDUCE)

    # -- R7: post-MoE RMSNorm ----------------------------------------------
    if worker < post_workers and _in(ROOT_MASK, D_S_POST_NORM):
        _wait(signals, D_S_REDUCE, epoch * reduce_workers, POLL_DELAY, WAITS)
        for token in tl.range(worker, B, TOTAL_WORKERS):
            _root_rms_norm(
                moe_down,
                post_ff_norm_weight,
                moe_branch,
                token,
                H,
                NORM_BLOCK,
                HPAD,
                EPS,
            )

    tl.store(worker_epoch + worker, epoch)


@triton.jit
def _dispatch_gate_up(
    expert_input,
    expert_weight,
    activation,
    active_tiles,
    expert_counts,
    tile_end,
    order,
    signals,
    gate_up_counter,
    task,
    units,
    columns: tl.constexpr,
    GROUP_CONTINUATION: tl.constexpr,
    H: tl.constexpr,
    INTERMEDIATE: tl.constexpr,
    TM: tl.constexpr,
    TPE: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    STAGES: tl.constexpr,
    LOAD_NK: tl.constexpr,
    EVICT: tl.constexpr,
    EXPERT_TASK: tl.constexpr,
):
    """Run one gate/up task and account for it on the ragged edge.

    With ``EXPERT_TASK`` a task is a whole expert and the worker walks that
    expert's tiles itself, so its duration scales with the expert's token count.
    Otherwise a task is exactly one (tile, column block) pair and every task
    costs the same, because the weight slice a task streams does not depend on
    how many of its ``TM`` rows are occupied.
    """
    unit = task // columns
    column_block = task % columns
    if unit < units:
        if EXPERT_TASK:
            span = (tl.load(expert_counts + unit) + TM - 1) // TM
            first = tl.load(tile_end + unit) - span
            for local in tl.range(0, span):
                _root_grouped_gate_up(
                    expert_input,
                    expert_weight,
                    activation,
                    active_tiles,
                    expert_counts,
                    order,
                    first + local,
                    column_block,
                    H,
                    INTERMEDIATE,
                    TM,
                    TPE,
                    TOPK,
                    BLOCK_N,
                    BLOCK_K,
                    STAGES,
                    LOAD_NK,
                    EVICT,
                )
        else:
            _root_grouped_gate_up(
                expert_input,
                expert_weight,
                activation,
                active_tiles,
                expert_counts,
                order,
                unit,
                column_block,
                H,
                INTERMEDIATE,
                TM,
                TPE,
                TOPK,
                BLOCK_N,
                BLOCK_K,
                STAGES,
                LOAD_NK,
                EVICT,
            )
        tl.debug_barrier()
        if GROUP_CONTINUATION:
            tl.atomic_add(
                signals + (D_NUM_FIXED_SLOTS + unit) * D_LINE,
                1,
                sem="release",
                scope="gpu",
            )
        previous = tl.atomic_add(gate_up_counter, 1, sem="release", scope="gpu")
        if previous == units * columns - 1:
            _publish(signals, D_S_GATE_UP)


@triton.jit
def _dispatch_down(
    activation,
    expert_weight,
    expert_outputs,
    active_tiles,
    expert_counts,
    tile_end,
    order,
    signals,
    down_counter,
    task,
    units,
    columns: tl.constexpr,
    GROUP_CONTINUATION: tl.constexpr,
    POLL_DELAY: tl.constexpr,
    WAITS: tl.constexpr,
    H: tl.constexpr,
    INTERMEDIATE: tl.constexpr,
    TM: tl.constexpr,
    TPE: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    STAGES: tl.constexpr,
    LOAD_NK: tl.constexpr,
    EVICT: tl.constexpr,
    EXPERT_TASK: tl.constexpr,
    GATE_COLUMNS: tl.constexpr,
):
    """Run one down task, optionally gated on its own unit's gate/up only.

    The per-unit readiness words are zeroed by the tile-metadata root, so the
    target here is absolute rather than epoch-relative: which units exist is
    itself decided at run time, and an epoch-relative target would drift once
    the routing changes the active count between launches.
    """
    unit = task // columns
    column_block = task % columns
    if unit < units:
        if GROUP_CONTINUATION:
            _wait(signals, D_NUM_FIXED_SLOTS + unit, GATE_COLUMNS, POLL_DELAY, WAITS)
        if EXPERT_TASK:
            span = (tl.load(expert_counts + unit) + TM - 1) // TM
            first = tl.load(tile_end + unit) - span
            for local in tl.range(0, span):
                _root_grouped_down(
                    activation,
                    expert_weight,
                    expert_outputs,
                    active_tiles,
                    expert_counts,
                    order,
                    first + local,
                    column_block,
                    H,
                    INTERMEDIATE,
                    TM,
                    TPE,
                    TOPK,
                    BLOCK_N,
                    BLOCK_K,
                    STAGES,
                    LOAD_NK,
                    EVICT,
                )
        else:
            _root_grouped_down(
                activation,
                expert_weight,
                expert_outputs,
                active_tiles,
                expert_counts,
                order,
                unit,
                column_block,
                H,
                INTERMEDIATE,
                TM,
                TPE,
                TOPK,
                BLOCK_N,
                BLOCK_K,
                STAGES,
                LOAD_NK,
                EVICT,
            )
        tl.debug_barrier()
        previous = tl.atomic_add(down_counter, 1, sem="release", scope="gpu")
        if previous == units * columns - 1:
            _publish(signals, D_S_DOWN)


# ---------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------


def allocate_buffers(shape, workers, tile_tokens):
    device = "cuda"
    tpe = tiles_per_expert_of(shape, tile_tokens)
    max_active = max_aligned_tiles(shape, tile_tokens)
    assignments = shape.assignments
    zeros = lambda n: torch.zeros(n, device=device, dtype=torch.int32)  # noqa: E731
    return {
        "expert_input": torch.empty(
            (shape.batch, shape.hidden), device=device, dtype=torch.bfloat16
        ),
        "topk_weights": torch.zeros(
            (shape.batch, shape.top_k), device=device, dtype=torch.float32
        ),
        "topk_ids": zeros(shape.batch * shape.top_k).view(shape.batch, shape.top_k),
        "router_logits": torch.zeros(
            (shape.batch, shape.num_experts), device=device, dtype=torch.float32
        ),
        "expert_counts": zeros(shape.num_experts),
        "tile_end": zeros(shape.num_experts),
        "active_tiles": zeros(max(max_active, 1)),
        "order": zeros(shape.num_experts * tpe * tile_tokens),
        "activation": torch.empty(
            (assignments, shape.moe_intermediate), device=device, dtype=torch.bfloat16
        ),
        "expert_outputs": torch.empty(
            (assignments, shape.hidden), device=device, dtype=torch.bfloat16
        ),
        "moe_down": torch.empty(
            (shape.batch, shape.hidden), device=device, dtype=torch.bfloat16
        ),
        "moe_branch": torch.empty(
            (shape.batch, shape.hidden), device=device, dtype=torch.bfloat16
        ),
        "worker_epoch": zeros(workers),
        "signals": zeros((NUM_FIXED_SLOTS + max(max_active, shape.num_experts)) * LINE),
        "gate_up_cursor": zeros(LINE),
        "down_cursor": zeros(LINE),
        "gate_up_counter": zeros(LINE),
        "down_counter": zeros(LINE),
    }


BUFFER_ORDER = (
    "expert_input",
    "topk_weights",
    "topk_ids",
    "router_logits",
    "expert_counts",
    "tile_end",
    "active_tiles",
    "order",
    "activation",
    "expert_outputs",
    "moe_down",
    "moe_branch",
    "worker_epoch",
    "signals",
    "gate_up_cursor",
    "down_cursor",
    "gate_up_counter",
    "down_counter",
)


def kernel_arguments(tensors, buffers, shape, args):
    tpe = tiles_per_expert_of(shape, args.tile_tokens)
    gathered = 1 if args.formulation == "gathered" else 0
    return (
        tensors["residual"],
        tensors["pre_ff_norm_weight_2"],
        tensors["router_scale"],
        tensors["router_weight"],
        tensors["per_expert_scale"],
        tensors["expert_gate_up_weight"],
        tensors["expert_down_weight"],
        tensors["post_ff_norm_weight_2"],
        tensors["root_size"],
        *(buffers[name] for name in BUFFER_ORDER),
        args.workers,
        shape.batch,
        shape.hidden,
        shape.moe_intermediate,
        shape.num_experts,
        shape.top_k,
        args.tile_tokens,
        tpe,
        max_aligned_tiles(shape, args.tile_tokens),
        units_static(shape, args),
        units_static(shape, args) if args.group_continuation else 0,
        args.norm_block,
        args.router_block_k,
        args.router_expert_block,
        1 << (shape.hidden - 1).bit_length(),
        1 if args.router_split else 0,
        1 if args.fuse_pre_norm else 0,
        args.order_block,
        args.gate_block_n,
        args.gate_block_k,
        args.gate_stages,
        args.down_block_n,
        args.down_block_k,
        args.down_stages,
        args.reduce_block,
        1 if args.weight_major == "nk" else 0,
        args.eviction,
        args.root_mask,
        gathered,
        down_fanin(shape, args) if gathered else 1,
        down_arrivals(shape, args) if gathered else 1,
        1 if args.fuse_tasks else 0,
        SCHEDULE_NAMES[args.schedule],
        1 if args.expert_task == "expert" else 0,
        args.group_continuation,
        args.poll_delay,
        not args.no_waits,
        shape.eps,
    )


def launch(tensors, buffers, shape, args):
    return moe_megakernel[(args.workers,)](
        *kernel_arguments(tensors, buffers, shape, args),
        num_warps=args.num_warps,
        num_stages=args.kernel_stages,
    )


def compile_only(tensors, buffers, shape, args):
    return moe_megakernel.warmup(
        *kernel_arguments(tensors, buffers, shape, args),
        grid=(args.workers,),
        num_warps=args.num_warps,
        num_stages=args.kernel_stages,
    )


def resident_capacity(compiled, num_warps):
    """Workers that are provably co-resident.

    Full residency is the deadlock-safety invariant for a waiting persistent
    kernel.  Ask the driver rather than modelling it: an analytic estimate that
    ignores allocation granularity is optimistic, and optimistic hangs the GPU.
    """
    compiled._init_handles()
    properties = torch.cuda.get_device_properties(0)
    blocks = ctypes.c_int()
    status = _libcuda().cuOccupancyMaxActiveBlocksPerMultiprocessor(
        ctypes.byref(blocks),
        ctypes.c_void_p(compiled.function),
        ctypes.c_int(num_warps * 32),
        ctypes.c_size_t(compiled.metadata.shared),
    )
    if status != 0:
        raise RuntimeError(
            f"cuOccupancyMaxActiveBlocksPerMultiprocessor failed: {status}"
        )
    per_sm = blocks.value
    return per_sm * properties.multi_processor_count, per_sm


def _assert_close(name, actual, expected, *, atol=3e-1, rtol=1e-1):
    torch.testing.assert_close(actual.float(), expected.float(), atol=atol, rtol=rtol)
    print(
        f"correctness {name} "
        f"max_abs={float((actual.float() - expected.float()).abs().max()):.6f}",
        flush=True,
    )


def validate(buffers, reference, shape, args):
    torch.cuda.synchronize()
    torch.testing.assert_close(buffers["topk_ids"], reference["topk_ids"])
    _assert_close("topk_weights", buffers["topk_weights"], reference["topk_weights"])
    if args.formulation == "gathered":
        _assert_close(
            "activation",
            buffers["activation"].view(reference["expert_activation"].shape),
            reference["expert_activation"],
            atol=0.2,
            rtol=0.08,
        )
        _assert_close("moe_down", buffers["moe_down"], reference["moe_down"])
        _assert_close("moe_branch", buffers["moe_branch"], reference["moe_branch"])
        return {"num_active_tiles": 0}
    expected = align_experts_reference(
        reference["topk_ids"],
        shape.num_experts,
        args.tile_tokens,
        tiles_per_expert_of(shape, args.tile_tokens),
    )
    torch.testing.assert_close(buffers["expert_counts"], expected["expert_counts"])
    torch.testing.assert_close(buffers["tile_end"], expected["tile_end"])
    active = expected["num_active_tiles"]
    torch.testing.assert_close(
        buffers["active_tiles"][:active], expected["active_tiles"]
    )
    valid = expected["order"] >= 0
    torch.testing.assert_close(buffers["order"][valid], expected["order"][valid])
    _assert_close(
        "activation",
        buffers["activation"].view(reference["expert_activation"].shape),
        reference["expert_activation"],
        atol=0.2,
        rtol=0.08,
    )
    _assert_close("moe_down", buffers["moe_down"], reference["moe_down"])
    _assert_close("moe_branch", buffers["moe_branch"], reference["moe_branch"])
    return {"num_active_tiles": active}


def down_fanin(shape, args):
    """How many workers publish the gathered down root's completion."""
    tasks = shape.assignments * shape.hidden // args.down_block_n
    return min(args.workers, tasks)


def down_arrivals(shape, args):
    """Publications on the down edge: one per task when the space is fused."""
    if args.fuse_tasks:
        return shape.assignments * shape.hidden // args.down_block_n
    return down_fanin(shape, args)


def units_static(shape, args):
    """Compile-time upper bound on dispatch units for the two ragged roots."""
    if args.expert_task == "expert":
        return shape.num_experts
    return max_aligned_tiles(shape, args.tile_tokens)


def resolve_policy(args, shape):
    # The two formulations want different output-block widths and neither is a
    # good default for the other: the gathered path is a pure GEMV whose best
    # shape is narrow (16 columns, 12.45 us standalone versus 16.5 at 64), while
    # the grouped path feeds a 16-row ``tl.dot`` and wants the wide one
    # (batch 8: 112.4 us at 64/256 versus 195.3 at 16/128).
    if not args.gate_block_n:
        args.gate_block_n = 16 if args.formulation == "gathered" else 64
    # The gathered path's narrow tiles leave enough shared memory for a deep
    # pipeline (batch 1: 36.90 us at 2/3 stages, 30.98 at 3/5); the grouped
    # path's wide tiles do not -- anything deeper exceeds resident capacity.
    if not args.down_block_n:
        args.down_block_n = 64 if args.formulation == "gathered" else 256
    if not args.gate_stages:
        args.gate_stages = 3 if args.formulation == "gathered" else 2
    if not args.down_stages:
        args.down_stages = 5 if args.formulation == "gathered" else 3
    if args.formulation == "gathered":
        if shape.moe_intermediate % args.gate_block_n:
            raise ValueError(
                "gathered gate/up needs --gate-block-n to divide moe_intermediate"
            )
        if args.schedule != "static" or args.expert_task != "tile":
            raise ValueError(
                "the gathered formulation has no ragged dispatch to schedule"
            )
        return args
    if args.tile_tokens < 16:
        raise ValueError("tl.dot needs at least 16 rows; use --tile-tokens >= 16")
    if shape.hidden % args.down_block_n:
        raise ValueError("hidden must be a multiple of --down-block-n")
    if shape.hidden % args.reduce_block:
        raise ValueError("hidden must be a multiple of --reduce-block")
    if shape.moe_intermediate % args.down_block_k:
        raise ValueError("moe_intermediate must be a multiple of --down-block-k")
    if shape.hidden % args.gate_block_k:
        raise ValueError("hidden must be a multiple of --gate-block-k")
    return args


def build_helion_control(args, tensors, shape):
    """The tuned separate-kernel path this megakernel is measured against."""
    from probes.gemma4 import helion_gemma4_a4b_moe as baseline

    control_args = argparse.Namespace(
        tune=[],
        tile_tokens=args.tile_tokens,
        seed=args.seed,
        route_skew=args.route_skew,
    )
    config_path = args.control_config_path
    configs = (
        json.loads(open(config_path).read()) if os.path.exists(config_path) else {}
    )
    return baseline.build_moe(control_args, tensors, shape, configs, None)


def parse_variants(spec):
    """``schedule:task[:group]`` triples, e.g. ``dynamic:tile:group``."""
    variants = []
    for item in spec:
        fields = item.split(":")
        schedule = fields[0]
        task = fields[1] if len(fields) > 1 else "tile"
        group = len(fields) > 2 and fields[2] == "group"
        if schedule not in SCHEDULE_NAMES or task not in ("tile", "expert"):
            raise ValueError(f"bad variant {item!r}")
        variants.append((schedule, task, group))
    return variants


def variant_label(schedule, task, group, formulation="grouped"):
    if formulation == "gathered":
        return "mk_gathered"
    return f"mk_{schedule}_{task}" + ("_group" if group else "")


def prepare_variant(args, shape, tensors, reference, schedule, task, group):
    """Compile, launch once, and validate one dispatch policy."""
    local = argparse.Namespace(**vars(args))
    local.schedule = schedule
    local.expert_task = task
    local.group_continuation = group
    resolve_policy(local, shape)
    buffers = allocate_buffers(shape, local.workers, local.tile_tokens)
    compiled = compile_only(tensors, buffers, shape, local)
    capacity, per_sm = resident_capacity(compiled, local.num_warps)
    label = variant_label(schedule, task, group, local.formulation)
    if local.workers > capacity:
        print(
            "MEGAKERNEL_SKIP "
            + json.dumps(
                {
                    "variant": label,
                    "workers": local.workers,
                    "capacity": capacity,
                    "per_sm": per_sm,
                    "registers": compiled.n_regs,
                }
            ),
            flush=True,
        )
        return None
    launch(tensors, buffers, shape, local)
    if local.root_mask != -1:
        # A root subset does not produce the layer's output; only its time is
        # meaningful.
        launch(tensors, buffers, shape, local)
        torch.cuda.synchronize()
        stats = {"num_active_tiles": 0}
    elif local.no_waits:
        # Results are garbage without the waits, but the metadata roots still
        # run, so a second launch sees a real active-tile count and the timing
        # is the honest zero-serialization floor for these root bodies.
        launch(tensors, buffers, shape, local)
        torch.cuda.synchronize()
        stats = {"num_active_tiles": int(buffers["tile_end"][-1].item())}
    else:
        stats = validate(buffers, reference, shape, local)
    print(
        "MEGAKERNEL_VARIANT "
        + json.dumps(
            {
                "variant": label,
                "batch": shape.batch,
                "tile_tokens": local.tile_tokens,
                "tiles_per_expert": tiles_per_expert_of(shape, local.tile_tokens),
                "max_active_tiles": max_aligned_tiles(shape, local.tile_tokens),
                "units_static": units_static(shape, local),
                "workers": local.workers,
                "num_warps": local.num_warps,
                "registers": compiled.n_regs,
                "shared": compiled.metadata.shared,
                "resident_blocks_per_sm": per_sm,
                "resident_capacity": capacity,
                **stats,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return label, lambda: launch(tensors, buffers, shape, local)


def run(args):
    require_idle_visible_gpu()
    shape = Gemma4A4BMoEShape(batch=args.batch)
    resolve_policy(args, shape)
    tensors = allocate_moe(shape, args.seed, route_skew=args.route_skew)
    reference = moe_reference(tensors, shape)
    print(
        "MEGAKERNEL_WORKLOAD "
        + json.dumps(
            {
                "batch": args.batch,
                "route_skew": args.route_skew,
                "routing": routing_histogram(reference["topk_ids"], shape.num_experts),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    variants = parse_variants(
        args.sweep
        or [
            f"{args.schedule}:{args.expert_task}"
            + (":group" if args.group_continuation else "")
        ]
    )
    prepared = [
        prepare_variant(args, shape, tensors, reference, *variant)
        for variant in variants
    ]
    entries = dict(item for item in prepared if item is not None)
    if args.validate_only:
        return

    graphs = {}
    for label, call in entries.items():
        graph, _ = capture(call)
        graphs[label] = graph.replay
    if args.control:
        built = build_helion_control(args, tensors, shape)
        for name in ("launch_matched", "launch_optimized", "launch_grouped"):
            if name not in built:
                continue
            short = name.removeprefix("launch_")
            control_graph, control_output = capture(built[name])
            control_graph.replay()
            torch.cuda.synchronize()
            _assert_close(f"control_{short}", control_output, reference["moe_branch"])
            graphs[f"helion_{short}"] = control_graph.replay
    if not graphs:
        print("MEGAKERNEL_RESULT_EMPTY every variant exceeded residency", flush=True)
        return

    benchmark_pids = visible_gpu_pids()
    timings = benchmark_interleaved(graphs, args.repeats, args.batch_replays)
    if visible_gpu_pids() != benchmark_pids:
        raise RuntimeError("GPU process set changed during benchmark")
    print(
        "MEGAKERNEL_RESULT "
        + json.dumps(
            {"batch": args.batch, "timings": timings},
            sort_keys=True,
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--route-skew", type=float, default=0.0)
    parser.add_argument("--workers", type=int, default=352)
    parser.add_argument("--num-warps", type=int, default=4)
    parser.add_argument("--kernel-stages", type=int, default=1)
    parser.add_argument("--tile-tokens", type=int, default=16)
    parser.add_argument("--norm-block", type=int, default=256)
    parser.add_argument("--router-block-k", type=int, default=256)
    parser.add_argument("--router-expert-block", type=int, default=8)
    parser.add_argument("--no-router-split", dest="router_split", action="store_false")
    parser.add_argument(
        "--no-fuse-pre-norm", dest="fuse_pre_norm", action="store_false"
    )
    parser.add_argument("--fuse-tasks", action="store_true")
    parser.set_defaults(router_split=True, fuse_pre_norm=True)
    parser.add_argument("--order-block", type=int, default=64)
    parser.add_argument("--gate-block-n", type=int, default=0)
    parser.add_argument("--gate-block-k", type=int, default=256)
    parser.add_argument("--gate-stages", type=int, default=0)
    parser.add_argument("--down-block-n", type=int, default=0)
    parser.add_argument("--down-block-k", type=int, default=64)
    parser.add_argument("--down-stages", type=int, default=0)
    parser.add_argument("--reduce-block", type=int, default=256)
    parser.add_argument("--weight-major", choices=("nk", "kn"), default="kn")
    parser.add_argument(
        "--eviction", choices=("", "evict_first", "evict_last"), default=""
    )
    parser.add_argument("--schedule", choices=sorted(SCHEDULE_NAMES), default="static")
    parser.add_argument(
        "--formulation", choices=("grouped", "gathered"), default="grouped"
    )
    parser.add_argument("--expert-task", choices=("tile", "expert"), default="tile")
    parser.add_argument("--group-continuation", action="store_true")
    parser.add_argument(
        "--sweep",
        nargs="+",
        help="schedule:task[:group] variants benchmarked interleaved in one run",
    )
    parser.add_argument("--poll-delay", type=int, default=0)
    parser.add_argument("--no-waits", action="store_true")
    parser.add_argument("--control", action="store_true")
    parser.add_argument(
        "--control-config-path",
        default=str(Path(__file__).with_name("gemma4_a4b_moe_b200_configs.json")),
    )
    parser.add_argument("--root-mask", type=int, default=-1)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--batch-replays", type=int, default=20)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
