"""Causal attention with two head-dim reductions, for the CuTe reduction tests.

Shared by the code generation checks of
``test/test_cute_single_thread_reduction_codegen.py`` and their GPU twins in
``test/test_cute_single_thread_reduction.py``: one kernel whose value head dim
differs from the q/k head dim, and one whose scores add a second matmul over
a shorter (RoPE) head dim.
"""

from __future__ import annotations

import math

import torch

import helion.language as hl


def attention_mla(
    q_in: torch.Tensor, k_in: torch.Tensor, v_in: torch.Tensor
) -> torch.Tensor:
    """Causal attention whose value head dim differs from the q/k head dim."""
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    head_dim = hl.specialize(q_in.size(-1))
    v_dim = hl.specialize(v_in.size(-1))
    q_view = q_in.reshape([-1, m_dim, head_dim])
    k_view = k_in.reshape([-1, n_dim, head_dim])
    v_view = v_in.reshape([-1, n_dim, v_dim])
    out = torch.empty(
        [q_view.size(0), m_dim, v_dim], dtype=q_in.dtype, device=q_in.device
    )
    qk_scale = (1.0 / math.sqrt(head_dim)) * 1.44269504
    for tile_b, tile_m in hl.tile([q_view.size(0), m_dim]):
        m_i = hl.full([tile_b, tile_m], -1e30, dtype=torch.float32)
        l_i = torch.full_like(m_i, 1.0)
        acc = hl.zeros([tile_b, tile_m, v_dim], dtype=torch.float32)
        qt = q_view[tile_b, tile_m, :]
        for tile_n in hl.tile(n_dim):
            kt = k_view[tile_b, tile_n, :]
            qk = torch.bmm(qt * qk_scale, kt.transpose(1, 2), torch.float32)
            # The ``tile.index`` slices keep their tiles' blocks.
            qk = torch.where(
                tile_m.index[None, :, None] >= tile_n.index[None, None, :],
                qk,
                float("-inf"),
            )
            m_ij_keepdim = torch.maximum(
                m_i[:, :, None], torch.amax(qk, -1, keepdim=True)
            )
            p = torch.exp2(qk - m_ij_keepdim)
            m_ij = m_ij_keepdim.squeeze(-1)
            alpha = torch.exp2(m_i - m_ij)
            l_i = l_i * alpha + torch.sum(p, -1)
            vt = v_view[tile_b, tile_n, :]
            acc = torch.baddbmm(acc * alpha[:, :, None], p.to(vt.dtype), vt)
            m_i = m_ij
        out[tile_b, tile_m, :] = (acc / l_i[:, :, None]).to(out.dtype)
    return out.view([q_in.size(0), q_in.size(1), m_dim, v_dim])


def attention_rope(
    q_in: torch.Tensor,
    k_in: torch.Tensor,
    v_in: torch.Tensor,
    qr_in: torch.Tensor,
    kr_in: torch.Tensor,
) -> torch.Tensor:
    """Causal attention whose scores add a second matmul over a shorter
    (RoPE) head dim."""
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    head_dim = hl.specialize(q_in.size(-1))
    rope_dim = hl.specialize(qr_in.size(-1))
    v_dim = hl.specialize(v_in.size(-1))
    q_view = q_in.reshape([-1, m_dim, head_dim])
    k_view = k_in.reshape([-1, n_dim, head_dim])
    qr_view = qr_in.reshape([-1, m_dim, rope_dim])
    kr_view = kr_in.reshape([-1, n_dim, rope_dim])
    v_view = v_in.reshape([-1, n_dim, v_dim])
    out = torch.empty(
        [q_view.size(0), m_dim, v_dim], dtype=q_in.dtype, device=q_in.device
    )
    qk_scale = (1.0 / math.sqrt(head_dim + rope_dim)) * 1.44269504
    for tile_b, tile_m in hl.tile([q_view.size(0), m_dim]):
        m_i = hl.full([tile_b, tile_m], -1e30, dtype=torch.float32)
        l_i = torch.full_like(m_i, 1.0)
        acc = hl.zeros([tile_b, tile_m, v_dim], dtype=torch.float32)
        qt = q_view[tile_b, tile_m, :]
        qr = qr_view[tile_b, tile_m, :]
        for tile_n in hl.tile(n_dim):
            kt = k_view[tile_b, tile_n, :]
            kr = kr_view[tile_b, tile_n, :]
            qk = torch.bmm(qt * qk_scale, kt.transpose(1, 2), torch.float32)
            qk = qk + torch.bmm(qr * qk_scale, kr.transpose(1, 2), torch.float32)
            # The ``tile.index`` slices keep their tiles' blocks.
            qk = torch.where(
                tile_m.index[None, :, None] >= tile_n.index[None, None, :],
                qk,
                float("-inf"),
            )
            m_ij_keepdim = torch.maximum(
                m_i[:, :, None], torch.amax(qk, -1, keepdim=True)
            )
            p = torch.exp2(qk - m_ij_keepdim)
            m_ij = m_ij_keepdim.squeeze(-1)
            alpha = torch.exp2(m_i - m_ij)
            l_i = l_i * alpha + torch.sum(p, -1)
            vt = v_view[tile_b, tile_n, :]
            acc = torch.baddbmm(acc * alpha[:, :, None], p.to(vt.dtype), vt)
            m_i = m_ij
        out[tile_b, tile_m, :] = (acc / l_i[:, :, None]).to(out.dtype)
    return out.view([q_in.size(0), q_in.size(1), m_dim, v_dim])


def attention_inputs(
    device: torch.device | str, *dims: int
) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(0)
    return tuple(
        torch.randn(1, 2, 256, dim, generator=generator, dtype=torch.bfloat16).to(
            device
        )
        for dim in dims
    )
