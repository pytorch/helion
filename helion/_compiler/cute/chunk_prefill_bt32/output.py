# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# pyrefly: ignore-errors
# ruff: noqa: ANN001, ANN202
# SPDX-License-Identifier: BSD-3-Clause
"""Four-warp output drain, two SMEM slots, and a separate tail store."""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import cutlass.experimental.cuda as cuda
import cutlass.experimental.primitives as prims

from ..warp_specialized_primitives import matrix_16x16_transposed_lane_coordinates
from . import common as cm
from .state import tptr


@cute.jit
def publish_empty(smem_base, local_warp):
    cm.output_sync()
    if local_warp == 0:
        cm.arrive(cm.sptr(smem_base, cm.OUT_EMPTY, cutlass.Int64))


@cute.jit
def stage_half(
    smem_base, output_stage, values, local_warp, lane, HALF: cutlass.Constexpr
):
    packed = cutlass.Array(cutlass.Int32, 8, alignment=16)
    for pair in cutlass.range_constexpr(8):
        packed[pair] = cm.pack_bf16(values[pair * 2], values[pair * 2 + 1])
    row, column = matrix_16x16_transposed_lane_coordinates(lane)
    for token_group in cutlass.range_constexpr(2):
        value = local_warp * 32 + HALF * 16 + column
        token = token_group * 16 + row
        pointer = cm.sptr(
            smem_base,
            cm.OUT + output_stage * 8192 + cm.sw128(token, value),
            cutlass.BFloat16,
        )
        prims.stmatrix(
            pointer,
            [
                packed[token_group * 4],
                packed[token_group * 4 + 1],
                packed[token_group * 4 + 2],
                packed[token_group * 4 + 3],
            ],
            prims.MMALayout.COL,
            shape=prims.StoreShape.M8N8,
        )


@cute.jit
def output_loop(
    smem_base,
    tmem_base,
    out,
    descriptor: cutlass.GridConstant[cuda.TensorMap],
    begin,
    seqlen,
    head,
    num_chunks,
    warp,
    lane,
):
    local_warp = warp - 4
    stage = cutlass.Int32(0)
    phase = cutlass.Int32(0)
    output_stage = cutlass.Int32(0)
    for chunk in cutlass.range(num_chunks, unroll=1):
        cm.wait(cm.bptr(smem_base, cm.FINAL_READY, stage), phase)
        prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
        if (chunk + 1) * cm.BT <= seqlen:
            values0 = prims.tcgen05_ld(
                "16x256b", tptr(tmem_base, warp, cm.TMEM_OUT, cutlass.Float32), num=4
            )
            values1 = prims.tcgen05_ld(
                "16x256b",
                cutlass.inttoptr(
                    tmem_base + (local_warp * 32 + 16 << 16) + cm.TMEM_OUT,
                    6,
                    cutlass.Float32,
                ),
                num=4,
            )
            prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)
            publish_empty(smem_base, local_warp)
            if local_warp == 0:
                if chunk >= 2:
                    prims.cp_async_bulk_wait_group(1, read=True)
            cm.output_sync()
            stage_half(smem_base, output_stage, values0, local_warp, lane, 0)
            stage_half(smem_base, output_stage, values1, local_warp, lane, 1)
            cm.output_sync()
            if local_warp == 0:
                cm.fence_shared()
                if prims.elect_sync():
                    prims.cp_async_bulk_tensor_global_shared_cta(
                        descriptor.get_ptr(),
                        cm.sptr(
                            smem_base, cm.OUT + output_stage * 8192, cutlass.BFloat16
                        ),
                        (0, cutlass.Int32(begin) + chunk * cm.BT, head, 0),
                    )
                    prims.cp_async_bulk_commit_group()
            output_stage = output_stage ^ 1
        else:
            values = prims.tcgen05_ld(
                "32x32b", tptr(tmem_base, warp, cm.TMEM_OUT, cutlass.Float32), num=32
            )
            prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)
            publish_empty(smem_base, local_warp)
            for column in cutlass.range_constexpr(32):
                token = chunk * cm.BT + column
                if token < seqlen:
                    out[
                        0, begin + cutlass.Int64(token), head, local_warp * 32 + lane
                    ] = cutlass.BFloat16(values[column])
        stage = stage + 1
        if stage == cm.STAGES:
            stage = cutlass.Int32(0)
            phase = phase ^ 1
    if local_warp == 0:
        prims.cp_async_bulk_wait_group(0, read=True)
    cm.output_sync()
    prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
    cm.arrive(cm.sptr(smem_base, cm.DONE, cutlass.Int64))
