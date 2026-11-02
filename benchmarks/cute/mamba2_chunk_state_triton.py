# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024, Tri Dao, Albert Gu.
# Adapted from https://github.com/state-spaces/mamba/blob/v2.2.4/mamba_ssm/ops/triton/ssd_chunk_state.py
# with the extra small-tile autotune configs vLLM added for hdim<=64 / dstate<=128
# (vllm/model_executor/layers/mamba/ops/ssd_chunk_state.py).
# ruff: noqa: ANN001, ANN202
"""Mamba2 ``_chunk_state_fwd`` Triton kernel: the state-of-the-art baseline for
``examples/mamba2_chunk_state.py``.

Same layouts as the Helion example: ``B (batch, seqlen, ngroups, dstate)``,
``x (batch, seqlen, nheads, headdim)``, ``dt``/``dA_cumsum (batch, nheads,
nchunks, chunk_size)``; returns ``states (batch, nchunks, nheads, headdim,
dstate)`` in ``states_dtype`` (the example's output dtype is ``B.dtype``).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32},
            num_stages=3,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32},
            num_stages=3,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32},
            num_stages=3,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64},
            num_stages=2,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64},
            num_stages=2,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64},
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 32},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32},
            num_stages=5,
            num_warps=2,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32},
            num_stages=5,
            num_warps=2,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32},
            num_stages=4,
            num_warps=2,
        ),
    ],
    key=["hdim", "dstate", "chunk_size"],
)
@triton.jit
def _chunk_state_fwd_kernel(
    x_ptr,
    b_ptr,
    states_ptr,
    dt_ptr,
    dA_cumsum_ptr,
    hdim,
    dstate,
    chunk_size,
    batch,
    seqlen,
    nheads_ngroups_ratio,
    stride_x_batch,
    stride_x_seqlen,
    stride_x_head,
    stride_x_hdim,
    stride_b_batch,
    stride_b_seqlen,
    stride_b_head,
    stride_b_dstate,
    stride_states_batch,
    stride_states_chunk,
    stride_states_head,
    stride_states_hdim,
    stride_states_dstate,
    stride_dt_batch,
    stride_dt_chunk,
    stride_dt_head,
    stride_dt_csize,
    stride_dA_cs_batch,
    stride_dA_cs_chunk,
    stride_dA_cs_head,
    stride_dA_cs_csize,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_bc = tl.program_id(axis=1)
    pid_c = pid_bc // batch
    pid_b = pid_bc - pid_c * batch
    pid_h = tl.program_id(axis=2)
    num_pid_n = tl.cdiv(dstate, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n
    b_ptr += (
        pid_b * stride_b_batch
        + pid_c * chunk_size * stride_b_seqlen
        + (pid_h // nheads_ngroups_ratio) * stride_b_head
    )
    x_ptr += (
        pid_b * stride_x_batch
        + pid_c * chunk_size * stride_x_seqlen
        + pid_h * stride_x_head
    )
    dt_ptr += pid_b * stride_dt_batch + pid_c * stride_dt_chunk + pid_h * stride_dt_head
    dA_cumsum_ptr += (
        pid_b * stride_dA_cs_batch
        + pid_c * stride_dA_cs_chunk
        + pid_h * stride_dA_cs_head
    )

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    x_ptrs = x_ptr + (
        offs_m[:, None] * stride_x_hdim + offs_k[None, :] * stride_x_seqlen
    )
    b_ptrs = b_ptr + (
        offs_n[None, :] * stride_b_dstate + offs_k[:, None] * stride_b_seqlen
    )
    dt_ptrs = dt_ptr + offs_k * stride_dt_csize
    dA_cs_last = tl.load(dA_cumsum_ptr + (chunk_size - 1) * stride_dA_cs_csize).to(
        tl.float32
    )
    dA_cumsum_ptrs = dA_cumsum_ptr + offs_k * stride_dA_cs_csize

    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, chunk_size_limit, BLOCK_SIZE_K):
        x = tl.load(
            x_ptrs,
            mask=(offs_m[:, None] < hdim) & (offs_k[None, :] < chunk_size_limit - k),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(offs_k[:, None] < chunk_size_limit - k) & (offs_n[None, :] < dstate),
            other=0.0,
        ).to(tl.float32)
        dA_cs_k = tl.load(
            dA_cumsum_ptrs, mask=offs_k < chunk_size_limit - k, other=0.0
        ).to(tl.float32)
        dt_k = tl.load(dt_ptrs, mask=offs_k < chunk_size_limit - k, other=0.0).to(
            tl.float32
        )
        scale = tl.exp(dA_cs_last - dA_cs_k) * dt_k
        b *= scale[:, None]
        b = b.to(x_ptr.dtype.element_ty)
        acc += tl.dot(x, b)
        x_ptrs += BLOCK_SIZE_K * stride_x_seqlen
        b_ptrs += BLOCK_SIZE_K * stride_b_seqlen
        dt_ptrs += BLOCK_SIZE_K * stride_dt_csize
        dA_cumsum_ptrs += BLOCK_SIZE_K * stride_dA_cs_csize
    states = acc.to(states_ptr.dtype.element_ty)

    states_ptr += (
        pid_b * stride_states_batch
        + pid_c * stride_states_chunk
        + pid_h * stride_states_head
    )
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    states_ptrs = states_ptr + (
        offs_m[:, None] * stride_states_hdim + offs_n[None, :] * stride_states_dstate
    )
    c_mask = (offs_m[:, None] < hdim) & (offs_n[None, :] < dstate)
    tl.store(states_ptrs, states, mask=c_mask)


def chunk_state_fwd(
    B: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    dA_cumsum: torch.Tensor,
    states: torch.Tensor | None = None,
    states_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, chunk_size = dt.shape
    _, _, ngroups, dstate = B.shape
    assert nheads % ngroups == 0
    assert B.shape == (batch, seqlen, ngroups, dstate)
    assert dt.shape == (batch, nheads, nchunks, chunk_size)
    assert dA_cumsum.shape == dt.shape
    if states is None:
        states = torch.empty(
            (batch, nchunks, nheads, headdim, dstate),
            device=x.device,
            dtype=B.dtype if states_dtype is None else states_dtype,
        )

    def grid(meta: dict[str, int]) -> tuple[int, int, int]:
        return (
            triton.cdiv(headdim, meta["BLOCK_SIZE_M"])
            * triton.cdiv(dstate, meta["BLOCK_SIZE_N"]),
            batch * nchunks,
            nheads,
        )

    _chunk_state_fwd_kernel[grid](
        x,
        B,
        states,
        dt,
        dA_cumsum,
        headdim,
        dstate,
        chunk_size,
        batch,
        seqlen,
        nheads // ngroups,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        B.stride(0),
        B.stride(1),
        B.stride(2),
        B.stride(-1),
        states.stride(0),
        states.stride(1),
        states.stride(2),
        states.stride(3),
        states.stride(4),
        dt.stride(0),
        dt.stride(2),
        dt.stride(1),
        dt.stride(3),
        dA_cumsum.stride(0),
        dA_cumsum.stride(2),
        dA_cumsum.stride(1),
        dA_cumsum.stride(3),
    )
    return states
