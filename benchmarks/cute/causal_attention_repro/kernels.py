"""Causal Helion CuTe kernels used by the GB300 forward/backward experiment.

Forward: attention-gym b530d94 attn_stage. Backward: Helion 61c840f39
flex_attention.py preprocess and fused gradient kernels. The kernel bodies
are preserved; explicit saved configs are used without autotuning.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

import helion
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Callable


def causal_stage(
    qk: torch.Tensor,
    b_idx: torch.Tensor,
    q_idx: torch.Tensor,
    kv_idx: torch.Tensor,
    b_index: torch.Tensor,
    tile_m: object,
    tile_n: object,
) -> torch.Tensor:
    return torch.where(q_idx >= kv_idx, qk, float("-inf"))


def score_mod(
    score: torch.Tensor,
    b: torch.Tensor,
    h: torch.Tensor,
    q_idx: torch.Tensor,
    kv_idx: torch.Tensor,
) -> torch.Tensor:
    return score


def score_mod_bwd(
    grad: torch.Tensor,
    score: torch.Tensor,
    b: torch.Tensor,
    h: torch.Tensor,
    q_idx: torch.Tensor,
    kv_idx: torch.Tensor,
) -> torch.Tensor:
    return grad


def causal_mask(
    b: torch.Tensor,
    h: torch.Tensor,
    q_idx: torch.Tensor,
    kv_idx: torch.Tensor,
) -> torch.Tensor:
    return q_idx >= kv_idx


@helion.kernel(static_shapes=True, fast_math=True, autotune_effort="none")
def attn_stage(
    q_in: torch.Tensor,
    k_in: torch.Tensor,
    v_in: torch.Tensor,
    stage: Callable[..., torch.Tensor],
) -> torch.Tensor:
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    assert n_dim == v_in.size(-2)
    head_dim = hl.specialize(q_in.size(-1))
    assert head_dim == k_in.size(-1) == v_in.size(-1)
    q_view = q_in.reshape([-1, m_dim, head_dim])
    v_view = v_in.reshape([-1, n_dim, head_dim])
    k_view = k_in.reshape([-1, n_dim, head_dim])
    out = torch.empty_like(q_view)
    qk_scale = (1.0 / math.sqrt(head_dim)) * 1.44269504
    for tile_b, tile_m in hl.tile([q_view.size(0), m_dim]):
        m_i = hl.full([tile_b, tile_m], -1e30, dtype=torch.float32)
        l_i = torch.full_like(m_i, 1.0)
        acc = hl.zeros([tile_b, tile_m, head_dim], dtype=torch.float32)
        qt = q_view[tile_b, tile_m, :]
        for tile_n in hl.tile(v_view.size(1)):
            kt = k_view[tile_b, tile_n, :]
            qk = torch.bmm(qt * qk_scale, kt.transpose(1, 2), torch.float32)
            qk = stage(
                qk,
                tile_b.index[:, None, None],
                tile_m.index[None, :, None],
                tile_n.index[None, None, :],
                tile_b.index,
                tile_m,
                tile_n,
            )
            m_ij_keepdim = torch.maximum(
                m_i[:, :, None], torch.amax(qk, -1, keepdim=True)
            )
            qk = qk - m_ij_keepdim
            m_ij = m_ij_keepdim.squeeze(-1)
            p = torch.exp2(qk)
            l_ij = torch.sum(p, -1)
            alpha = torch.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, :, None]
            vt = v_view[tile_b, tile_n, :]
            acc = torch.baddbmm(acc, p.to(vt.dtype), vt)
            m_i = m_ij
        acc = acc / l_i[:, :, None]
        out[tile_b, tile_m, :] = acc.to(out.dtype)
    return out.view(q_in.size())


@helion.kernel(static_shapes=True, fast_math=True, autotune_effort="none")
def _flex_attention_bwd_preprocess(
    o_in: torch.Tensor, do_in: torch.Tensor, lse_in: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """The per-row terms of the softmax backward: ``delta[b, h, m] = sum_d o * do``
    (fp32) and the forward's natural-log LSE in the exp2 base, ``lse2 = lse *
    log2(e)``, with a row that saw no key (``lse == -inf``) mapped to 0 so the
    main kernel's ``score * log2(e) - lse2`` stays finite (its P is 0 anyway:
    every score of such a row is masked)."""
    head_dim = hl.specialize(o_in.size(-1))
    o = o_in.reshape(-1, head_dim)
    do = do_in.reshape(-1, head_dim)
    lse = lse_in.reshape(-1)
    total = o.size(0)
    delta = torch.empty(total, device=o.device, dtype=torch.float32)
    lse2 = torch.empty(total, device=o.device, dtype=torch.float32)
    log_2_e = 1.44269504
    for tile in hl.tile(total):
        delta[tile] = torch.sum(
            o[tile, :].to(torch.float32) * do[tile, :].to(torch.float32), dim=-1
        )
        lse_row = lse[tile]
        lse2[tile] = torch.where(lse_row == float("-inf"), 0.0, lse_row) * log_2_e
    return delta.reshape(o_in.size()[:-1]), lse2.reshape(o_in.size()[:-1])


@helion.kernel(static_shapes=True, fast_math=True, autotune_effort="none")
def helion_flex_attention_backward_kernel(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    grad_out: torch.Tensor,
    lse2: torch.Tensor,
    delta: torch.Tensor,
    score_mod: Callable,
    score_mod_bwd: Callable,
    block_mask_q_num_blocks: torch.Tensor,
    block_mask_q_indices: torch.Tensor,
    block_mask_full_q_num_blocks: torch.Tensor,
    block_mask_full_q_indices: torch.Tensor,
    block_mask_mask_mod: Callable,
    block_mask_m: int,
    block_mask_n: int,
    scale: float,
    enable_gqa: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """FlexAttention-style fused attention backward (the twin of
    :func:`helion_flex_attention_kernel`).

    One work item per KV tile of one (batch, kv head): dK / dV accumulate in
    fp32 over the Q blocks the BlockMask lists for that KV block
    (``full_q_*``: no mask; ``q_*``: ``mask_mod`` applied), for every q head of
    the GQA group; dQ accumulates into an fp32 buffer with atomics (the
    wrapper converts).  ``lse2`` and ``delta`` are the ``_flex_attention_bwd_preprocess``
    row terms: the forward's natural-log logsumexp (its second output) in the
    exp2 base with no-visible-key rows mapped to 0, and the ``o * do`` row
    sums.  ``score_mod`` acts on the natural-unit scores like the forward;
    ``score_mod_bwd(grad, score, b, h, q_idx, kv_idx)`` maps the gradient
    w.r.t. the modified score back to the raw one (FlashAttention-4's
    interface; the identity for additive biases).  The CuTe backend lowers
    this form to its fused tcgen05 backward.
    """
    B, H, M, D = query.size()
    D = hl.specialize(D)
    Bk, Hk, N, Dk = key.size()
    Dv = hl.specialize(value.size(-1))
    assert Bk == B
    assert Dk == D
    assert grad_out.size(-1) == Dv
    if enable_gqa:
        assert H % Hk == 0
    else:
        assert Hk == H
    num_groups = H // Hk
    block_mask_m = hl.specialize(block_mask_m)
    block_mask_n = hl.specialize(block_mask_n)
    table_batch, table_heads, _ = block_mask_q_num_blocks.size()
    dq = torch.zeros((B, H, M, D), dtype=torch.float32, device=query.device)
    dk = torch.empty_like(key)
    dv = torch.empty_like(value)
    log_2_e = 1.44269504
    block_m = hl.register_block_size(min(256, block_mask_m))
    block_n = hl.register_block_size(min(256, block_mask_n))
    for tile_b, tile_hk, tile_n in hl.tile([B, Hk, N], block_size=[1, 1, block_n]):
        k_j = key[tile_b.begin, tile_hk.begin, tile_n, :]
        v_j = value[tile_b.begin, tile_hk.begin, tile_n, :]
        dk_acc = hl.zeros([tile_n, D], dtype=torch.float32)
        dv_acc = hl.zeros([tile_n, Dv], dtype=torch.float32)
        sparse_col = tile_n.begin // block_mask_n
        b_idx = tile_b.begin
        hk_idx = tile_hk.begin
        tb_idx = b_idx % table_batch
        bcast_b = (tile_b.begin + hl.arange(1))[:, None]
        bcast_n = (tile_n.begin + hl.arange(tile_n.block_size))[:, None]
        # The q heads sharing this KV head (one for MHA).
        for tile_h in hl.tile(
            hk_idx * num_groups, hk_idx * num_groups + num_groups, block_size=1
        ):
            h_idx = tile_h.begin
            th_idx = h_idx % table_heads
            bcast_h = (tile_h.begin + hl.arange(1))[:, None]

            # Q blocks that need no mask.
            full_num_blocks = block_mask_full_q_num_blocks[tb_idx, th_idx, sparse_col]
            for block_idx in hl.tile(full_num_blocks, block_size=1):
                start_m = (
                    block_mask_full_q_indices[tb_idx, th_idx, sparse_col, block_idx.id]
                    * block_mask_m
                )
                end_m = start_m + block_mask_m
                end_M = end_m.new_full([], M)
                end_m = torch.minimum(end_m, end_M)
                for tile_m in hl.tile(start_m, end_m, block_size=block_m):
                    q_i = query[b_idx, h_idx, tile_m, :]
                    do_i = grad_out[b_idx, h_idx, tile_m, :]
                    lse2_i = lse2[b_idx, h_idx, tile_m]
                    delta_i = delta[b_idx, h_idx, tile_m]
                    bcast_m = (tile_m.begin + hl.arange(tile_m.block_size))[None, :]
                    s_t = hl.dot(k_j, q_i.T, out_dtype=torch.float32)
                    s_t = s_t * scale
                    score = score_mod(s_t, bcast_b, bcast_h, bcast_m, bcast_n)
                    p_t = torch.exp2(score * log_2_e - lse2_i[None, :])
                    dp_t = hl.dot(v_j, do_i.T, out_dtype=torch.float32)
                    dv_acc = hl.dot(p_t.to(value.dtype), do_i, acc=dv_acc)
                    ds_t = p_t * (dp_t - delta_i[None, :])
                    ds_t = score_mod_bwd(ds_t, s_t, bcast_b, bcast_h, bcast_m, bcast_n)
                    ds_c = ds_t.to(query.dtype)
                    dq_i = hl.dot(ds_c.T, k_j, out_dtype=torch.float32)
                    hl.atomic_add(dq, [b_idx, h_idx, tile_m, slice(None)], dq_i * scale)
                    dk_acc = hl.dot(ds_c, q_i, acc=dk_acc)

            # Q blocks with the mask applied.
            num_blocks = block_mask_q_num_blocks[tb_idx, th_idx, sparse_col]
            for block_idx in hl.tile(num_blocks, block_size=1):
                start_m = (
                    block_mask_q_indices[tb_idx, th_idx, sparse_col, block_idx.id]
                    * block_mask_m
                )
                end_m = start_m + block_mask_m
                end_M = end_m.new_full([], M)
                end_m = torch.minimum(end_m, end_M)
                for tile_m in hl.tile(start_m, end_m, block_size=block_m):
                    q_i = query[b_idx, h_idx, tile_m, :]
                    do_i = grad_out[b_idx, h_idx, tile_m, :]
                    lse2_i = lse2[b_idx, h_idx, tile_m]
                    delta_i = delta[b_idx, h_idx, tile_m]
                    bcast_m = (tile_m.begin + hl.arange(tile_m.block_size))[None, :]
                    s_t = hl.dot(k_j, q_i.T, out_dtype=torch.float32)
                    s_t = s_t * scale
                    score = score_mod(s_t, bcast_b, bcast_h, bcast_m, bcast_n)
                    mask = block_mask_mask_mod(bcast_b, bcast_h, bcast_m, bcast_n)
                    score = torch.where(mask, score, -float("inf"))
                    p_t = torch.exp2(score * log_2_e - lse2_i[None, :])
                    dp_t = hl.dot(v_j, do_i.T, out_dtype=torch.float32)
                    dv_acc = hl.dot(p_t.to(value.dtype), do_i, acc=dv_acc)
                    ds_t = p_t * (dp_t - delta_i[None, :])
                    ds_t = score_mod_bwd(ds_t, s_t, bcast_b, bcast_h, bcast_m, bcast_n)
                    ds_c = ds_t.to(query.dtype)
                    dq_i = hl.dot(ds_c.T, k_j, out_dtype=torch.float32)
                    hl.atomic_add(dq, [b_idx, h_idx, tile_m, slice(None)], dq_i * scale)
                    dk_acc = hl.dot(ds_c, q_i, acc=dk_acc)

        dv[tile_b, tile_hk, tile_n, :] = dv_acc[None, None, :, :].to(dv.dtype)
        dk[tile_b, tile_hk, tile_n, :] = (dk_acc * scale)[None, None, :, :].to(dk.dtype)
    return dq, dk, dv
