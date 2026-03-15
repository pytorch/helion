"""
Generic Chunkwise Linear Attention Recurrence
==============================================

A reusable Helion primitive for chunkwise linear attention models.

Usage:
    from chunk_recurrence import gated_delta_net_fwd_h

    kernel = gated_delta_net_fwd_h(chunk_size=64)
    h_out, v_new = kernel(k, w, u, g)
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel()
def _gated_delta_net_fwd_h_64(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K = k.shape
    K = hl.specialize(K)
    V = u.shape[-1]
    C = 64

    NT = (T + C - 1) // C
    h_out = torch.empty(B, NT, H, K, V, dtype=k.dtype, device=k.device)
    v_out = torch.empty_like(u)
    BH = B * H
    block_v = hl.register_block_size(V)

    for flat, tv in hl.tile([BH, V], block_size=[1, block_v]):
        i_b = flat.id // H
        i_h = flat.id % H

        state = hl.zeros([K, tv], dtype=torch.float32)

        for tc in hl.tile(T, block_size=C):
            h_out[i_b, tc.id, i_h, :, tv] = state.to(k.dtype)

            proj = hl.dot(
                w[i_b, tc, i_h, :], state, out_dtype=torch.float32
            )
            diff = u[i_b, tc, i_h, tv].to(torch.float32) - proj

            v_out[i_b, tc, i_h, tv] = diff.to(u.dtype)

            t_last = min(tc.begin + C, T) - 1
            g_last = g[i_b, t_last, i_h].to(torch.float32)
            g_t = g[i_b, tc, i_h].to(torch.float32)
            gate = torch.exp(g_last - g_t)
            diff_gated = diff * gate[:, None]
            state = state * torch.exp(g_last)

            p_k = k[i_b, tc, i_h, :]
            state = hl.dot(p_k.T, diff_gated, acc=state)

    return h_out, v_out


def gated_delta_net_fwd_h(chunk_size: int = 64):
    assert chunk_size == 64, "Only chunk_size=64 supported"
    return _gated_delta_net_fwd_h_64
