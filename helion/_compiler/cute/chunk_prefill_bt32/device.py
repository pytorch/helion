# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# pyrefly: ignore-errors
# ruff: noqa: ANN001, ANN202
# SPDX-License-Identifier: BSD-3-Clause
"""BT32 M128 CuTe engine for the explicit centered v2 policy.

The host ABI follows the existing BT16 prototype. Unsupported optional policies
are rejected at compilation; no argument is silently ignored. Input state is
separate, immutable FP32 and is copied exactly into authoritative FP32 TMEM.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import cutlass.experimental.cuda as cuda
import cutlass.experimental.primitives as prims

from . import common as cm
from .factor import factor_loop
from .issuer import issuer_loop
from .output import output_loop
from .state import state_loop

THREADS_PER_CTA = cm.THREADS
BT = cm.BT
DK = cm.DK
DV = cm.DV
NUMERICAL_POLICY = "centered_bt32_fp32_rhs_v2"


@cute.jit
def init_barriers(smem_base, lane):
    """Initialize the packed barrier regions cooperatively in one warp."""
    for index in cutlass.range(lane, 9 * cm.STAGES, 32, unroll=1):
        prims.mbarrier_init(cm.sptr(smem_base, index * 8, cutlass.Int64), 1)
    for index in cutlass.range(lane, 4 * cm.STAGES, 32, unroll=1):
        prims.mbarrier_init(cm.sptr(smem_base, cm.V_FREE + index * 8, cutlass.Int64), 4)
    if lane == 0:
        prims.mbarrier_init(cm.sptr(smem_base, cm.OUT_EMPTY, cutlass.Int64), 1)
        prims.mbarrier_init(cm.sptr(smem_base, cm.DONE, cutlass.Int64), 8)
    prims.fence_mbarrier_init()


@cute.kernel
def kernel(
    desc_q: cutlass.GridConstant[cuda.TensorMap],
    desc_k: cutlass.GridConstant[cuda.TensorMap],
    desc_gate: cutlass.GridConstant[cuda.TensorMap],
    desc_beta: cutlass.GridConstant[cuda.TensorMap],
    desc_v: cutlass.GridConstant[cuda.TensorMap],
    desc_out: cutlass.GridConstant[cuda.TensorMap],
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    gate: cute.Tensor,
    beta: cute.Tensor,
    a_log: cute.Tensor,
    dt_bias: cute.Tensor,
    cu_seqlens: cute.Tensor,
    seq_order: cute.Tensor | None,
    initial_state: cute.Tensor | None,
    out: cute.Tensor,
    final_state: cute.Tensor | None,
    scale: cutlass.Float32,
    gate_scale_log2: cutlass.Float32,
):
    thread, _, _ = cute.arch.thread_idx()
    warp = cute.arch.make_warp_uniform(thread // 32)
    lane = thread % 32
    head, sequence, _ = cute.arch.block_idx()
    head = cute.arch.make_warp_uniform(head)
    sequence = cute.arch.make_warp_uniform(sequence)
    if cutlass.const_expr(seq_order is not None):
        sequence = cute.arch.make_warp_uniform(cutlass.Int32(seq_order[sequence]))
    begin = cutlass.Int64(cu_seqlens[sequence])
    end = cutlass.Int64(cu_seqlens[sequence + 1])
    begin = cute.arch.make_warp_uniform(begin)
    seqlen = cute.arch.make_warp_uniform(cutlass.Int32(end - begin))
    num_chunks = cute.arch.make_warp_uniform((seqlen + cm.BT - 1) // cm.BT)
    heads = cutlass.Int32(q.shape[2])
    storage = cutlass.Array(
        cutlass.Uint8, cm.SMEM_BYTES, space=cutlass.AddressSpace.smem, alignment=1024
    )
    smem_base = cutlass.Int32(storage.data_ptr().toint())
    if warp == 10:
        init_barriers(smem_base, lane)
    if warp == 0:
        prims.tcgen05_alloc(
            cm.sptr(smem_base, cm.TMEM_ADDR, cutlass.Int32),
            cm.TMEM_COLS,
            group=prims.CTAGroup.CTA_1,
        )
        prims.tcgen05_relinquish_alloc_permit(group=prims.CTAGroup.CTA_1)
    prims.barrier_cta_sync(0, thread_count=cm.THREADS)
    prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
    tmem_base = cm.sptr(smem_base, cm.TMEM_ADDR, cutlass.Int32).load()
    tmem_base = cute.arch.make_warp_uniform(tmem_base)
    if warp < 4:
        prims.setmaxregister(144, prims.SetMaxRegisterAction.INCREASE)
        state_loop(
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
        )
    elif warp < 8:
        prims.setmaxregister(48, prims.SetMaxRegisterAction.DECREASE)
        output_loop(
            smem_base,
            tmem_base,
            out,
            desc_out,
            begin,
            seqlen,
            head,
            num_chunks,
            warp,
            lane,
        )
    elif warp < 12:
        prims.setmaxregister(32, prims.SetMaxRegisterAction.DECREASE)
        if warp == 9:
            issuer_loop(smem_base, tmem_base, num_chunks)
    else:
        prims.setmaxregister(56, prims.SetMaxRegisterAction.DECREASE)
        factor_loop(
            smem_base,
            q,
            k,
            v,
            gate,
            beta,
            a_log,
            dt_bias,
            desc_q,
            desc_k,
            desc_gate,
            desc_beta,
            desc_v,
            begin,
            seqlen,
            head,
            heads,
            num_chunks,
            (warp - 12) // 4,
            (warp - 12) % 4,
            lane,
            scale,
            gate_scale_log2,
        )


@cute.kernel
def empty_state_copy(initial_state: cute.Tensor | None, final_state: cute.Tensor):
    row, _, _ = cute.arch.thread_idx()
    head, sequence, _ = cute.arch.block_idx()
    for column in cutlass.range_constexpr(128):
        value = cutlass.Float32(0.0)
        if cutlass.const_expr(initial_state is not None):
            value = initial_state[sequence, head, row, column]
        final_state[sequence, head, row, column] = value


def _validate_metadata(
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    raw_gate: cute.Tensor,
    a_log: cute.Tensor,
    dt_bias: cute.Tensor,
    beta: cute.Tensor,
    cu_seqlens: cute.Tensor,
    seq_order: cute.Tensor | None,
    state_indices: cute.Tensor | None,
    initial_state: cute.Tensor | None,
    out: cute.Tensor,
    final_state: cute.Tensor | None,
    state_ckpt: cute.Tensor | None,
    cu_ckpts: cute.Tensor | None,
    checkpoint_state_indices: cute.Tensor | None,
    checkpoint_stride_chunks: cutlass.Constexpr,
    SAFE_GATE: cutlass.Constexpr,
    THREADS: cutlass.Constexpr,
    gate_dtype: cutlass.Constexpr,
    TASK_ORDER: cutlass.Constexpr,
    SEGMENT_BEGIN: cutlass.Constexpr,
    SEGMENT_SIZE: cutlass.Constexpr,
    SEQUENCE_BEGIN: cutlass.Constexpr,
    SEQUENCE_COUNT: cutlass.Constexpr,
):
    tokens = q.shape[1]
    if THREADS != cm.THREADS or not SAFE_GATE:
        raise ValueError("BT32 prototype requires THREADS=1024 and SAFE_GATE=True")
    if (
        TASK_ORDER != "identity"
        or SEGMENT_BEGIN != 0
        or SEGMENT_SIZE != -1
        or (SEQUENCE_COUNT != 0)
        or (SEQUENCE_BEGIN != 0)
        or (checkpoint_stride_chunks != 0)
        or (state_indices is not None)
        or (state_ckpt is not None)
        or (cu_ckpts is not None)
        or (checkpoint_state_indices is not None)
    ):
        raise ValueError(
            "BT32 prototype supports identity, unsegmented, unindexed calls only"
        )
    if seq_order is not None and (
        seq_order.element_type is not cutlass.Int32
        or seq_order.shape != (cu_seqlens.shape[0] - 1,)
        or seq_order.stride != (1,)
    ):
        raise ValueError(
            "seq_order must be contiguous Int32 with one entry per sequence"
        )
    if q.shape[0] != 1 or q.shape[3] != 128:
        raise ValueError("BT32 prototype requires packed [1,T,H,128] inputs")
    if (
        q.element_type is not cutlass.BFloat16
        or k.element_type is not cutlass.BFloat16
        or v.element_type is not cutlass.BFloat16
        or (raw_gate.element_type is not cutlass.BFloat16)
        or (beta.element_type is not cutlass.BFloat16)
        or (out.element_type is not cutlass.BFloat16)
        or (a_log.element_type is not cutlass.Float32)
        or (dt_bias.element_type is not cutlass.Float32)
        or (gate_dtype is not cutlass.BFloat16)
    ):
        raise ValueError(
            "BT32 prototype requires BF16 activations and FP32 gate parameters"
        )
    if initial_state is not None:
        if initial_state.element_type is not cutlass.Float32:
            raise ValueError("initial state must be FP32")
    if final_state is not None:
        if final_state.element_type is not cutlass.Float32:
            raise ValueError("final state must be FP32")
    if tokens > 2147483647 - 31:
        raise ValueError("BT32 TMA token coordinates require signed Int32 extent")


@cute.jit
def host(
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    raw_gate: cute.Tensor,
    a_log: cute.Tensor,
    dt_bias: cute.Tensor,
    beta: cute.Tensor,
    cu_seqlens: cute.Tensor,
    seq_order: cute.Tensor | None,
    state_indices: cute.Tensor | None,
    initial_state: cute.Tensor | None,
    out: cute.Tensor,
    final_state: cute.Tensor | None,
    stream,
    SCALE: cutlass.Float32,
    state_ckpt: cute.Tensor | None,
    cu_ckpts: cute.Tensor | None,
    checkpoint_state_indices: cute.Tensor | None,
    checkpoint_stride_chunks: cutlass.Constexpr,
    SAFE_GATE: cutlass.Constexpr,
    GATE_SCALE_LOG2: cutlass.Float32,
    THREADS: cutlass.Constexpr,
    gate_dtype: cutlass.Constexpr,
    TASK_ORDER: cutlass.Constexpr = "identity",
    SEGMENT_BEGIN: cutlass.Constexpr = 0,
    SEGMENT_SIZE: cutlass.Constexpr = -1,
    SEQUENCE_BEGIN: cutlass.Constexpr = 0,
    SEQUENCE_COUNT: cutlass.Constexpr = 0,
):
    _validate_metadata(
        q,
        k,
        v,
        raw_gate,
        a_log,
        dt_bias,
        beta,
        cu_seqlens,
        seq_order,
        state_indices,
        initial_state,
        out,
        final_state,
        state_ckpt,
        cu_ckpts,
        checkpoint_state_indices,
        checkpoint_stride_chunks,
        SAFE_GATE,
        THREADS,
        gate_dtype,
        TASK_ORDER,
        SEGMENT_BEGIN,
        SEGMENT_SIZE,
        SEQUENCE_BEGIN,
        SEQUENCE_COUNT,
    )
    tokens = q.shape[1]
    heads = q.shape[2]
    sequences = cu_seqlens.shape[0] - 1
    if cutlass.const_expr(tokens == 0):
        if cutlass.const_expr(final_state is not None):
            empty_state_copy(initial_state, final_state).launch(
                grid=(heads, sequences, 1), block=(128, 1, 1), stream=stream
            )
    else:
        qk_layout = cute.make_layout(
            (64, tokens, 2, heads, 1), stride=(1, 128 * heads, 64, 128, 128)
        )
        gv_layout = cute.make_layout((128, heads, tokens), stride=(1, 128, 128 * heads))
        out_layout = cute.make_layout(
            (64, tokens, heads, 2), stride=(1, 128 * heads, 128, 64)
        )
        desc_q = cuda.create_tensor_map_tiled_from_view(
            cute.make_tensor(q.iterator, qk_layout),
            box_dims=(64, 32, 2, 1, 1),
            stride_order=(0, 1, 2, 3, 4),
            swizzle=cuda.TensorMapSwizzle.s128b,
        )
        desc_k = cuda.create_tensor_map_tiled_from_view(
            cute.make_tensor(k.iterator, qk_layout),
            box_dims=(64, 32, 2, 1, 1),
            stride_order=(0, 1, 2, 3, 4),
            swizzle=cuda.TensorMapSwizzle.s128b,
        )
        desc_gate = cuda.create_tensor_map_tiled_from_view(
            cute.make_tensor(raw_gate.iterator, gv_layout),
            box_dims=(128, 1, 32),
            stride_order=(0, 1, 2),
            swizzle=cuda.TensorMapSwizzle.none,
        )
        beta_layout = cute.make_layout((heads, tokens), stride=(1, heads))
        desc_beta = cuda.create_tensor_map_tiled_from_view(
            cute.make_tensor(beta.iterator, beta_layout),
            box_dims=(8, 32),
            stride_order=(0, 1),
            swizzle=cuda.TensorMapSwizzle.none,
        )
        desc_v = cuda.create_tensor_map_tiled_from_view(
            cute.make_tensor(v.iterator, gv_layout),
            box_dims=(128, 1, 32),
            stride_order=(0, 1, 2),
            swizzle=cuda.TensorMapSwizzle.none,
        )
        desc_out = cuda.create_tensor_map_tiled_from_view(
            cute.make_tensor(out.iterator, out_layout),
            box_dims=(64, 32, 1, 2),
            stride_order=(0, 1, 2, 3),
            swizzle=cuda.TensorMapSwizzle.s128b,
        )
        kernel(
            desc_q,
            desc_k,
            desc_gate,
            desc_beta,
            desc_v,
            desc_out,
            q,
            k,
            v,
            raw_gate,
            beta,
            a_log,
            dt_bias,
            cu_seqlens,
            seq_order,
            initial_state,
            out,
            final_state,
            SCALE,
            GATE_SCALE_LOG2,
        ).launch(
            grid=(heads, sequences, 1),
            block=(THREADS, 1, 1),
            stream=stream,
            min_blocks_per_mp=1,
        )
