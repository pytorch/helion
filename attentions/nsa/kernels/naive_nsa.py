# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import math
from typing import Optional, Union

import torch
import torch.nn.functional as F
from einops import rearrange, repeat


@torch.compile
def compression(
    k: torch.Tensor,
    v: torch.Tensor,
    block_size: int
) -> torch.Tensor:
    # Currently, we set mean pooling as our basic compression function.
    B, T, H = k.shape[:3]
    num_block = math.ceil(T / block_size)
    if k.shape[1] % block_size != 0:
        k = F.pad(k, (0, 0, 0, 0, 0, num_block * block_size - T))
        v = F.pad(v, (0, 0, 0, 0, 0, num_block * block_size - T))
    k_cmp = k.view(B, num_block, block_size, H, -1).mean(dim=2)
    v_cmp = v.view(B, num_block, block_size, H, -1).mean(dim=2)
    return k_cmp, v_cmp


def naive_nsa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_slc: torch.Tensor,
    g_swa: torch.Tensor,
    block_indices: torch.LongTensor,
    block_counts: Optional[Union[torch.LongTensor, int]] = None,
    block_size: int = 64,
    window_size: int = 0,
    scale: Optional[float] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    head_first: bool = False
) -> torch.Tensor:
    r"""
    Args:
        q (torch.Tensor):
            Queries of shape `[B, T, HQ, K]` if `head_first=False` else `[B, HQ, T, K]`.
        k (torch.Tensor):
            Keys of shape `[B, T, H, K]` if `head_first=False` else `[B, H, T, K]`.
            GQA is enforced here. The ratio of query heads (HQ) to key/value heads (H) must be a power of 2 and >=16.
        v (torch.Tensor):
            Values of shape `[B, T, H, V]` if `head_first=False` else `[B, H, T, V]`.
        g_slc (torch.Tensor):
            Gate score for selected attention of shape `[B, T, HQ]` if  `head_first=False` else `[B, HQ, T]`.
        g_swa (torch.Tensor):
            Gate score for sliding attentionof shape `[B, T, HQ]` if  `head_first=False` else `[B, HQ, T]`.
        block_indices (torch.LongTensor):
            Block indices of shape `[B, T, H, S]` if `head_first=False` else `[B, H, T, S]`.
            `S` is the maximum number of selected blocks for each query token, which is set to 16 in the paper.
        block_counts (Union[torch.LongTensor, int]):
            Number of selected blocks for each token.
            If a tensor is provided, with shape `[B, T, H]` if `head_first=True` else `[B, T, H]`,
            each token can select the same number of blocks.
            If not provided, it will default to `S`, Default: `None`.
        block_size (int):
            Selected block size. Default: 64.
        window_size (int):
            Sliding window size. Default: 0.
        scale (Optional[int]):
            Scale factor for attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
        head_first (Optional[bool]):
            Whether the inputs are in the head-first format. Default: `False`.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, HQ, V]` if `head_first=False` else `[B, HQ, T, V]`.
    """
    if scale is None:
        scale = k.shape[-1] ** -0.5
    if cu_seqlens is not None:
        assert q.shape[0] == 1, "batch size must be 1 when cu_seqlens are provided"
        if head_first:
            raise RuntimeError("Sequences with variable lengths are not supported for head-first mode")
    if head_first:
        q, k, v, block_indices = map(lambda x: rearrange(x, 'b h t d -> b t h d'), (q, k, v, block_indices))
        g_slc, g_swa = map(lambda x: rearrange(x, 'b h t -> b t h'), (g_slc, g_swa))
        if isinstance(block_counts, torch.Tensor):
            block_counts = rearrange(block_counts, 'b h t -> b t h')

    dtype = q.dtype
    G = q.shape[2] // k.shape[2]
    BS = block_size
    S = block_indices.shape[-1]
    k, v, block_indices = (repeat(x, 'b t h d -> b t (h g) d', g=G) for x in (k, v, block_indices))
    if isinstance(block_counts, torch.Tensor):
        block_counts = repeat(block_counts, 'b t h -> b t (h g)', g=G)
    c = torch.arange(S).repeat_interleave(BS).unsqueeze(1).expand(-1, q.shape[2]).to(q.device)
    q, k, v = map(lambda x: x.float(), (q, k, v))

    o_slc = torch.zeros_like(v)
    o_swa = torch.zeros_like(v) if window_size > 0 else None
    varlen = True
    if cu_seqlens is None:
        varlen = False
        B, T = q.shape[:2]
        cu_seqlens = torch.cat([block_indices.new_tensor(range(0, B*T, T)), block_indices.new_tensor([B*T])])

    for i in range(len(cu_seqlens) - 1):
        if not varlen:
            q_b, k_b, v_b, g_slc_b, g_swa_b, i_b = q[i], k[i], v[i], g_slc[i], g_swa[i], block_indices[i]
            if isinstance(block_counts, torch.Tensor):
                s_b = block_counts[i]
            else:
                s_b = block_counts
        else:
            T = cu_seqlens[i+1] - cu_seqlens[i]
            q_b, k_b, v_b, g_slc_b, g_swa_b, i_b = map(
                lambda x: x[0][cu_seqlens[i]:cu_seqlens[i+1]],
                (q, k, v, g_slc, g_swa, block_indices)
            )
            if isinstance(block_counts, torch.Tensor):
                s_b = block_counts[0][cu_seqlens[i]:cu_seqlens[i+1]]
            else:
                s_b = block_counts

        i_b = i_b.unsqueeze(-1) * BS + i_b.new_tensor(range(BS))
        # [T, S*BS, HQ]
        i_b = i_b.view(T, block_indices.shape[2], -1).transpose(1, 2)
        for i_q in range(T):
            # [HQ, D]
            q_i = q_b[i_q] * scale
            # [HQ]
            g_slc_i = g_slc_b[i_q]
            # [HQ]
            g_swa_i = g_swa_b[i_q]
            # [S*BS, HQ]
            i_i = i_b[i_q]
            # [HQ]
            if isinstance(block_counts, torch.Tensor):
                s_i = s_b[i_q]
            else:
                s_i = s_b
            # [S*BS, HQ, -1]
            k_i_slc, v_i_slc = map(lambda x: x.gather(0, i_i.clamp(
                0, T-1).unsqueeze(-1).expand(*i_i.shape, x.shape[-1])), (k_b, v_b))
            # [S*BS, HQ]
            attn_slc = torch.einsum('h d, n h d -> n h', q_i, k_i_slc).masked_fill(
                torch.logical_or(i_i < 0, i_i > i_q) | (c >= s_i if block_counts is not None else False),
                float('-inf')
            ).softmax(0)
            if not varlen:
                o_slc[i, i_q] = torch.einsum('n h, n h v -> h v', attn_slc, v_i_slc) * g_slc_i.unsqueeze(-1)
            else:
                o_slc[0][cu_seqlens[i]+i_q] = torch.einsum('n h, n h v -> h v', attn_slc, v_i_slc) * g_slc_i.unsqueeze(-1)
            if window_size > 0:
                k_i_swa, v_i_swa = map(lambda x: x[max(0, i_q - window_size + 1):i_q + 1], (k_b, v_b))
                attn_swa = torch.einsum('h d, n h d -> n h', q_i, k_i_swa).softmax(0)
                if not varlen:
                    o_swa[i, i_q] = torch.einsum('n h, n h v -> h v', attn_swa, v_i_swa) * g_swa_i.unsqueeze(-1)
                else:
                    o_swa[0][cu_seqlens[i]+i_q] = torch.einsum('n h, n h v -> h v', attn_swa, v_i_swa) * g_swa_i.unsqueeze(-1)

    if head_first:
        o_slc = rearrange(o_slc, 'b t h d -> b h t d')
        o_swa = rearrange(o_swa, 'b t h d -> b h t d')

    return o_slc.to(dtype) + o_swa.to(dtype) if o_swa is not None else o_slc.to(dtype)


def naive_nsa_compression(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_cmp: torch.Tensor,
    block_counts: Union[torch.LongTensor, int],
    block_size: int,
    scale: float,
    head_first: bool = False
) -> torch.LongTensor:
    dtype = q.dtype
    B, T = q.shape[0], q.shape[1]
    H, HQ = k.shape[2], q.shape[2]
    G = HQ//H
    BS = block_size
    if isinstance(block_counts, int):
        block_counts = torch.full((B, T, H), block_counts, dtype=torch.long, device=q.device)
    q, k, v = map(lambda x: x.float(), (q, k, v))
    k_cmp, v_cmp = compression(k, v, BS)
    C = k_cmp.shape[1]
    S = min(block_counts.max().item(), C)
    k_cmp, v_cmp = map(lambda x: repeat(x, 'b c h d -> b c (h g) d', g=G), (k_cmp, v_cmp))

    casual_mask = ((torch.arange(T) - BS + 1)[:, None] // BS < torch.arange(C)[None, :]).to(q.device)
    empty_mask = casual_mask.all(-1, True)
    local_mask = (torch.arange(T)[:, None] // BS == torch.arange(C)[None, :]).to(q.device)

    attn_cmp = torch.einsum('bqhd,bkhd->bhqk', q*scale, k_cmp)
    attn_cmp = attn_cmp.masked_fill(casual_mask & empty_mask.logical_not(), float('-inf'))
    attn_cmp = attn_cmp.softmax(-1).masked_fill(empty_mask, 0.0)
    o_cmp = torch.einsum('bhqk, bkhd -> bqhd', attn_cmp, v_cmp) * g_cmp.unsqueeze(-1)
    attn_select = attn_cmp.masked_fill(local_mask, float(1.0))
    attn_select = attn_select.view(B, H, G, T, C).sum(2)
    block_indices = attn_select.topk(S, -1)[1]

    block_indices = block_indices.masked_fill(block_indices > (block_indices.new_tensor(range(T))[:, None] // BS), -1)
    block_indices = block_indices.transpose(1, 2)

    if head_first:
        o_cmp = rearrange(o_cmp, 'b t h d -> b h t d')
    return block_indices, o_cmp.to(dtype)


def naive_nsa_compression_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_cmp: torch.Tensor,
    block_counts: Union[torch.LongTensor, int],
    block_size: int,
    scale: float,
    cu_seqlens: torch.LongTensor,
    head_first: bool = False
) -> torch.LongTensor:
    dtype = q.dtype
    B, T = q.shape[0], q.shape[1]
    H, HQ = k.shape[2], q.shape[2]
    D = v.shape[-1]
    G = HQ//H
    BS = block_size
    S = block_counts if isinstance(block_counts, int) else block_counts.max().item()
    C = math.ceil(T / block_size)
    S = min(S, C)
    block_indices = torch.zeros(B, T, H, S, dtype=torch.long, device=q.device)
    o_cmp = torch.zeros(B, T, HQ, D, dtype=dtype, device=q.device)
    for i in range(len(cu_seqlens) - 1):
        T_b = cu_seqlens[i+1] - cu_seqlens[i]
        C_b = math.ceil(T_b / block_size)
        q_b, k_b, v_b, g_cmp_b = map(
            lambda x: x[0][cu_seqlens[i]:cu_seqlens[i+1]],
            (q, k, v, g_cmp)
        )
        if isinstance(block_counts, torch.Tensor):
            s_b = block_counts[0][cu_seqlens[i]:cu_seqlens[i+1]]
        else:
            s_b = block_counts

        k_cmp, v_cmp = compression(k_b.unsqueeze(0), v_b.unsqueeze(0), BS)
        S_b = s_b if isinstance(s_b, int) else s_b.max().item()
        C_b = k_cmp.shape[1]
        S_b = min(S_b, C_b)
        k_cmp, v_cmp = map(lambda x: repeat(x.squeeze(0), 'c h d -> c (h g) d', g=G), (k_cmp, v_cmp))
        q_b, k_cmp, v_cmp = map(lambda x: x.float(), (q_b, k_cmp, v_cmp))

        casual_mask = ((torch.arange(T_b) - BS + 1)[:, None] // BS < torch.arange(C_b)[None, :]).to(q_b.device)
        local_mask = (torch.arange(T_b)[:, None] // BS == torch.arange(C_b)[None, :]).to(q.device)

        attn_cmp = torch.einsum('qhd,khd->hqk', q_b*scale, k_cmp)
        attn_cmp = attn_cmp.masked_fill(casual_mask, float('-inf'))
        attn_cmp = attn_cmp.softmax(-1)
        o_cmp[0][cu_seqlens[i]:cu_seqlens[i+1]] = torch.einsum('hqk,khd->qhd', attn_cmp, v_cmp).nan_to_num() *\
            g_cmp_b.unsqueeze(-1)
        attn_select = attn_cmp.masked_fill(local_mask, float(1.0))
        attn_select = attn_select.view(H, G, T_b, C_b).sum(1)
        block_indices_b = attn_select.topk(S_b, -1)[1]
        block_indices_b = block_indices_b.masked_fill(
            block_indices_b > (block_indices_b.new_tensor(range(T_b))[:, None]//BS),
            0
        )
        block_indices[0][cu_seqlens[i]:cu_seqlens[i+1], :, :S_b] = block_indices_b.transpose(0, 1)

    if head_first:
        o_cmp = rearrange(o_cmp, 'b t h d -> b h t d')
    return block_indices, o_cmp.to(dtype)


def naive_nsa_with_compression(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_cmp: torch.Tensor,
    g_slc: torch.Tensor,
    g_swa: torch.Tensor,
    block_counts: Union[torch.LongTensor, int],
    block_size: int = 64,
    window_size: int = 0,
    scale: Optional[float] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    head_first: bool = False
) -> torch.Tensor:
    r"""
    Args:
        q (torch.Tensor):
            Queries of shape `[B, T, HQ, K]` if `head_first=False` else `[B, HQ, T, K]`.
        k (torch.Tensor):
            Keys of shape `[B, T, H, K]` if `head_first=False` else `[B, H, T, K]`.
            GQA is enforced here. The ratio of query heads (HQ) to key/value heads (H) must be a power of 2 and >=16.
        v (torch.Tensor):
            Values of shape `[B, T, H, V]` if `head_first=False` else `[B, H, T, V]`.
        g_cmp (torch.Tensor):
            Gate score for compressed attention of shape `[B, T, HQ]` if  `head_first=False` else `[B, HQ, T]`.
        g_slc (torch.Tensor):
            Gate score for selected attention of shape `[B, T, HQ]` if  `head_first=False` else `[B, HQ, T]`.
        g_swa (torch.Tensor):
            Gate score for sliding attentionof shape `[B, T, HQ]` if  `head_first=False` else `[B, HQ, T]`.
        block_counts (Union[torch.LongTensor, int]):
            Number of selected blocks for each token.
            If a tensor is provided, with shape `[B, T, H]` if `head_first=True` else `[B, T, H]`,
            each token can select the same number of blocks.
        block_size (int):
            Selected block size. Default: 64.
        window_size (int):
            Sliding window size. Default: 0.
        scale (Optional[int]):
            Scale factor for attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        head_first (Optional[bool]):
            Whether the inputs are in the head-first format. Default: `False`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, HQ, V]` if `head_first=False` else `[B, HQ, T, V]`.
    """
    if scale is None:
        scale = k.shape[-1] ** -0.5
    if cu_seqlens is not None:
        assert q.shape[0] == 1, "batch size must be 1 when cu_seqlens are provided"
        if head_first:
            raise RuntimeError("Sequences with variable lengths are not supported for head-first mode")
    if head_first:
        q, k, v = map(lambda x: rearrange(x, 'b h t d -> b t h d'), (q, k, v))
        g_cmp, g_slc = map(lambda x: rearrange(x, 'b h t -> b t h'), (g_cmp, g_slc))
        if isinstance(block_counts, torch.Tensor):
            block_counts = rearrange(block_counts, 'b h t -> b t h')
    if cu_seqlens is not None:
        block_indices, o_cmp = naive_nsa_compression_varlen(
            q=q,
            k=k,
            v=v,
            g_cmp=g_cmp,
            block_counts=block_counts,
            block_size=block_size,
            scale=scale,
            cu_seqlens=cu_seqlens,
            head_first=False)
    else:
        block_indices, o_cmp = naive_nsa_compression(
            q=q,
            k=k,
            v=v,
            g_cmp=g_cmp,
            block_counts=block_counts,
            block_size=block_size,
            scale=scale,
            head_first=False)
    o = naive_nsa(
        q=q,
        k=k,
        v=v,
        g_slc=g_slc,
        g_swa=g_swa,
        block_indices=block_indices,
        block_counts=block_counts,
        block_size=block_size,
        window_size=window_size,
        scale=scale,
        cu_seqlens=cu_seqlens,
        head_first=False
    ) + o_cmp

    if head_first:
        o = rearrange(o, 'b t h d -> b h t d')

    return o, block_indices
