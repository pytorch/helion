"""Shared Helion attention kernels for CuTe matcher and emitter tests."""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(static_shapes=True, backend="cute")
def dot_attention_rows(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    check: hl.constexpr = False,  # pyrefly: ignore [bad-function-definition]
) -> torch.Tensor:
    """The 2-D form with the canonical ``v[bh, tile_n, :]`` tile and the
    scale on Q."""
    batch_heads = q.size(0)
    seq_len = q.size(1)
    head_dim = q.size(2)
    out = torch.empty([batch_heads, seq_len, head_dim], dtype=q.dtype, device=q.device)
    qk_scale = 0.18033688
    for bh, tile_m in hl.tile([batch_heads, seq_len], block_size=[1, None]):
        m_i = hl.full([tile_m], float("-inf"), dtype=torch.float32)
        l_i = hl.full([tile_m], 0.0, dtype=torch.float32)
        acc = hl.zeros([tile_m, head_dim], dtype=torch.float32)
        q_tile = q[bh.begin, tile_m, :]
        if check:
            torch._assert_async(q[bh.begin, tile_m.begin, 0].float() > 0)
        for tile_n in hl.tile(seq_len):
            q_scaled = q_tile * qk_scale
            k_tile = k[bh.begin, tile_n, :]
            qk = hl.dot(q_scaled, k_tile.transpose(0, 1))
            m_new = torch.maximum(m_i, torch.amax(qk, dim=-1))
            p = torch.exp2(qk - m_new[:, None])
            alpha = torch.exp2(m_i - m_new)
            l_i = l_i * alpha + torch.sum(p, dim=-1)
            acc = acc * alpha[:, None]
            v_tile = v[bh.begin, tile_n, :]
            acc = hl.dot(p.to(v.dtype), v_tile, acc=acc)
            m_i = m_new
        out[bh.begin, tile_m, :] = (acc / l_i[:, None]).to(q.dtype)
    return out


@helion.kernel(static_shapes=True, backend="cute")
def fp8_dot_attention_k_major(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    batch: int,
    heads: int,
) -> torch.Tensor:
    """``examples/fp8_attention.py`` with its query-tile loop on the launch
    grid: fp8 operands, the scale on the scores, a pre-transposed V and the
    4-D fp8 output."""
    batch_heads = q.size(0)
    seq_len = q.size(1)
    head_dim = q.size(2)
    out = torch.empty(
        [batch, heads, seq_len, head_dim], dtype=torch.float8_e4m3fn, device=q.device
    )
    sm_scale = 0.18033688
    for bh, tile_m in hl.tile([batch_heads, seq_len], block_size=[1, None]):
        b_index = bh.begin // heads
        h_index = bh.begin % heads
        m_i = hl.full([tile_m], float("-inf"), dtype=torch.float32)
        l_i = hl.full([tile_m], 0.0, dtype=torch.float32)
        acc = hl.zeros([tile_m, head_dim], dtype=torch.float32)
        q_tile = q[bh.begin, tile_m, :]
        for tile_n in hl.tile(seq_len):
            k_tile = k[bh.begin, tile_n, :]
            qk = hl.dot(q_tile, k_tile.transpose(0, 1)) * sm_scale
            m_new = torch.maximum(m_i, torch.amax(qk, dim=-1))
            p = torch.exp2(qk - m_new[:, None])
            alpha = torch.exp2(m_i - m_new)
            l_i = l_i * alpha + torch.sum(p, dim=-1)
            acc = acc * alpha[:, None]
            v_tile = v[bh.begin, :, tile_n]
            acc = hl.dot(p.to(v.dtype), v_tile.t(), acc=acc)
            m_i = m_new
        out[b_index, h_index, tile_m, :] = (acc / l_i[:, None]).to(torch.float8_e4m3fn)
    return out


@helion.kernel(static_shapes=True, backend="cute")
def dot_attention_k_major(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """The 2-D form with a pre-transposed ``v[bh, :, tile_n]`` and a 3-D
    output in the operand dtype."""
    batch_heads = q.size(0)
    seq_len = q.size(1)
    head_dim = q.size(2)
    out = torch.empty([batch_heads, seq_len, head_dim], dtype=q.dtype, device=q.device)
    sm_scale = 0.18033688
    for bh, tile_m in hl.tile([batch_heads, seq_len], block_size=[1, None]):
        m_i = hl.full([tile_m], float("-inf"), dtype=torch.float32)
        l_i = hl.full([tile_m], 0.0, dtype=torch.float32)
        acc = hl.zeros([tile_m, head_dim], dtype=torch.float32)
        q_tile = q[bh.begin, tile_m, :]
        for tile_n in hl.tile(seq_len):
            k_tile = k[bh.begin, tile_n, :]
            qk = hl.dot(q_tile, k_tile.transpose(0, 1)) * sm_scale
            m_new = torch.maximum(m_i, torch.amax(qk, dim=-1))
            p = torch.exp2(qk - m_new[:, None])
            alpha = torch.exp2(m_i - m_new)
            l_i = l_i * alpha + torch.sum(p, dim=-1)
            acc = acc * alpha[:, None]
            v_tile = v[bh.begin, :, tile_n]
            acc = hl.dot(p.to(v.dtype), v_tile.t(), acc=acc)
            m_i = m_new
        out[bh.begin, tile_m, :] = (acc / l_i[:, None]).to(q.dtype)
    return out
