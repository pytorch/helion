"""
Generic Chunkwise Linear Attention Recurrence
==============================================

A reusable Helion primitive that captures the common computational pattern
shared by all chunkwise linear attention models: Gated DeltaNet, GLA,
DeltaNet, RetNet, Mamba2, and others.

Usage:
    from chunk_recurrence import make_chunk_fwd_h

    kernel = make_chunk_fwd_h(state_update="gated_delta", chunk_size=64)
    h_out, v_new = kernel(k, w, u, g)
"""

from __future__ import annotations

from typing import Literal

import torch

import helion
import helion.language as hl

SUPPORTED_UPDATES = ("gated_delta", "linear", "delta")


def make_chunk_fwd_h(
    state_update: Literal["gated_delta", "linear", "delta"] = "gated_delta",
    chunk_size: int = 64,
):
    """
    Factory function that creates a chunk_fwd_h kernel for the specified
    linear attention variant.

    Args:
        state_update: Which recurrence rule to use.
            - "gated_delta": Gated DeltaNet (Qwen3.5, Qwen3-Next)
            - "linear": Standard linear attention / RetNet / GLA
            - "delta": DeltaNet without gating
        chunk_size: Number of timesteps per chunk. Default 64.

    Returns:
        A Helion kernel function with signature:
            (k, w, u, g) -> (h_out, v_new)
    """
    assert state_update in SUPPORTED_UPDATES, (
        f"Unknown state_update={state_update!r}, must be one of {SUPPORTED_UPDATES}"
    )

    C = chunk_size
    use_delta_correction = state_update in ("gated_delta", "delta")
    use_gating = state_update in ("gated_delta", "linear")

    @helion.kernel()
    def chunk_fwd_h_kernel(
        k: torch.Tensor,
        w: torch.Tensor,
        u: torch.Tensor,
        g: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, H, K = k.shape
        K = hl.specialize(K)
        V = u.shape[-1]

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

                if use_delta_correction:
                    proj = hl.dot(
                        w[i_b, tc, i_h, :], state, out_dtype=torch.float32
                    )
                    diff = u[i_b, tc, i_h, tv].to(torch.float32) - proj
                else:
                    diff = u[i_b, tc, i_h, tv].to(torch.float32)

                v_out[i_b, tc, i_h, tv] = diff.to(u.dtype)

                if use_gating:
                    t_last = min(tc.begin + C, T) - 1
                    g_last = g[i_b, t_last, i_h].to(torch.float32)
                    g_t = g[i_b, tc, i_h].to(torch.float32)
                    gate = torch.exp(g_last - g_t)
                    diff_gated = diff * gate[:, None]
                    state = state * torch.exp(g_last)
                else:
                    diff_gated = diff

                p_k = k[i_b, tc, i_h, :]
                state = hl.dot(p_k.T, diff_gated, acc=state)

        return h_out, v_out

    return chunk_fwd_h_kernel


def gated_delta_net_fwd_h(chunk_size: int = 64):
    """Create a Gated DeltaNet chunk_fwd_h kernel (Qwen3.5, Qwen3-Next)."""
    return make_chunk_fwd_h(state_update="gated_delta", chunk_size=chunk_size)


def linear_attention_fwd_h(chunk_size: int = 64):
    """Create a linear attention / RetNet / GLA chunk_fwd_h kernel."""
    return make_chunk_fwd_h(state_update="linear", chunk_size=chunk_size)


def delta_net_fwd_h(chunk_size: int = 64):
    """Create a DeltaNet (ungated) chunk_fwd_h kernel."""
    return make_chunk_fwd_h(state_update="delta", chunk_size=chunk_size)
