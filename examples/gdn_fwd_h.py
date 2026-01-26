"""
Gated Delta Net Fwd H Kernel
============================

This code implements a fwd_h kernel as used in gated delta net
"""

# %%
# Imports
# -------
from __future__ import annotations

import math
from typing import Callable

import torch

import helion
from helion._testing import DEVICE
from helion._testing import run_example
import helion.language as hl


# %%
# Helion Kernel Implementation
# ----------------------------
@helion.kernel()
def helion_gdn_fwd_h(
    k: torch.Tensor, w: torch.Tensor, u: torch.Tensor, g: torch.Tensor, chunk_size: int
) -> torch.Tensor:
    """
    Argument:
        k: (batch, seqlen, nheads, dhead)
        w: (batch, seqlen, nheads, dhead)
        u: (batch, seqlen, nheads, expand_v*dhead)
        g: (batch, seqlen, nheads)
        chunk_size: int
    Return:
        h: (batch, nchunks, nheads, dhead, expand_v*dhead)
    """

    batch, seqlen, nheads, dhead = k.shape
    dhead = hl.specialize(dhead)
    chunk_size = hl.specialize(chunk_size)
    dstate = u.shape[-1]

    acc_dtype = torch.float32
    dtype = k.dtype

    nchunks = (seqlen + chunk_size - 1) // chunk_size
    h = torch.empty(batch, nchunks, nheads, dhead, dstate, dtype=dtype, device=k.device)
    block_v = hl.register_block_size(dstate)

    for i_b, i_h in hl.grid([batch, nheads]):
        for tile_v in hl.tile(dstate, block_size=block_v):
            b_h = hl.zeros([dhead, tile_v], dtype=acc_dtype)
            for t_i in hl.tile(seqlen, block_size=chunk_size):
                h[i_b, t_i.id, i_h, :, tile_v] = b_h.to(dtype)
                b_w = w[i_b, t_i, i_h, :]
                c_h = b_h.to(dtype)
                b_v = hl.dot(b_w, c_h, out_dtype=acc_dtype)
                p_v = u[i_b, t_i, i_h, tile_v].to(acc_dtype)
                b_v = p_v - b_v
                m_t = t_i.index < seqlen
                t_i_last = min(t_i.begin + chunk_size, seqlen) - 1
                b_g_last = g[i_b, t_i_last, i_h].to(acc_dtype)
                b_g = g[i_b, t_i, i_h].to(acc_dtype)
                b_v *= torch.where(m_t, torch.exp(b_g_last - b_g), 0)[:, None]
                b_g_last = torch.exp(b_g_last)
                b_h *= b_g_last
                b_v = b_v.to(dtype)
                p_k = k[i_b, t_i, i_h, :]
                b_h = hl.dot(p_k.T, b_v, acc=b_h)
    return h


def helion_gdn_fwd_h_tb(
    tb_obj: object,
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor,
    chunk_size: int,
) -> Callable[[], torch.Tensor]:
    """
    Argument:
        k: (batch, seqlen, nheads, dhead)
        w: (batch, seqlen, nheads, dhead)
        u: (batch, seqlen, nheads, expand_v*dhead)
        g: (batch, seqlen, nheads)
        chunk_size: int
    Return:
        h: (batch, nchunks, nheads, dhead, expand_v*dhead)
    """
    return lambda: helion_gdn_fwd_h(k, w, u, g, chunk_size)


# %%
# Reference Function
# -------------
def ref_gdn_fwd_h(
    k: torch.Tensor, w: torch.Tensor, u: torch.Tensor, g: torch.Tensor, chunk_size: int
) -> torch.Tensor:
    """
    Argument:
        k: (batch, seqlen, nheads, dhead)
        w: (batch, seqlen, nheads, dhead)
        u: (batch, seqlen, nheads, expand_v*dhead)
        g: (batch, seqlen, nheads)
        chunk_size: int
    Return:
        h: (batch, nchunks, nheads, dhead, expand_v*dhead)
    """

    batch, seqlen, nheads, dhead = k.shape
    expand_v = u.shape[-1] // dhead
    nchunks = (seqlen + chunk_size - 1) // chunk_size

    acc_dtype = torch.float32
    dtype = k.dtype

    h = torch.empty(
        batch, nchunks, nheads, dhead, expand_v * dhead, dtype=k.dtype, device=k.device
    )
    b_h = torch.zeros(
        batch, nheads, dhead, expand_v * dhead, dtype=acc_dtype, device=k.device
    )

    k_c = k.reshape(batch, nchunks, chunk_size, nheads, dhead)
    w_c = w.reshape(batch, nchunks, chunk_size, nheads, dhead)
    u_c = u.reshape(batch, nchunks, chunk_size, nheads, expand_v * dhead)
    g_c = g.reshape(batch, nchunks, chunk_size, nheads)
    for i_t in range(nchunks):
        h[:, i_t, :, :, :] = b_h.to(dtype)
        b_w = w_c[:, i_t, :, :, :].to(acc_dtype)
        c_h = b_h.to(dtype).to(acc_dtype)
        b_v = torch.einsum("bchk,bhkv->bchv", b_w, c_h)
        p_v = u_c[:, i_t, :, :, :].to(acc_dtype)
        b_v = p_v - b_v
        last_idx = min((i_t + 1) * chunk_size, seqlen) - 1
        m_t = (i_t * chunk_size + torch.arange(0, chunk_size, device=k.device)) < seqlen
        b_g_last = g[:, last_idx, :].to(acc_dtype)
        b_g = g_c[:, i_t, :, :].to(acc_dtype)  # batch, chunk, nheads
        b_v *= torch.where(
            m_t.unsqueeze(0).unsqueeze(-1), torch.exp(b_g_last.unsqueeze(1) - b_g), 0
        ).unsqueeze(-1)
        b_g_last = torch.exp(b_g_last)
        b_h *= b_g_last.unsqueeze(-1).unsqueeze(-1)
        b_v = b_v.to(dtype).to(acc_dtype)
        p_k = k_c[:, i_t, :, :, :].to(acc_dtype)
        b_h += torch.einsum("bchk,bchv->bhkv", p_k, b_v)
    return h


# %%
# Testing Function
# -------------
def test(
    batch: int,
    nheads: int,
    seqlen: int,
    chunk_size: int,
    dhead: int,
    dstate: int,
    dtype: torch.dtype = torch.float16,
) -> None:
    k = torch.randn(batch, seqlen, nheads, dhead, dtype=torch.bfloat16, device=DEVICE)
    k = torch.nn.functional.rms_norm(k, [dhead])
    w = torch.randn(
        batch,
        seqlen // chunk_size,
        chunk_size,
        nheads,
        dhead,
        dtype=torch.float32,
        device=DEVICE,
    )
    # w = torch.nn.functional.rms_norm(w.to(torch.bfloat16), (dhead,))
    wu, ws, wv = torch.linalg.svd(w.permute(0, 1, 3, 2, 4), full_matrices=False)
    w = torch.einsum("bnhik,bnhkj->bnhij", wu, wv)
    w = (
        w.permute(0, 1, 3, 2, 4)
        .reshape(batch, seqlen, nheads, dhead)
        .to(torch.bfloat16)
    )
    u = torch.randn(batch, seqlen, nheads, dstate, dtype=torch.bfloat16, device=DEVICE)
    u = torch.nn.functional.rms_norm(u, [dstate])
    g = torch.cumsum(
        0.5
        * math.log(1 / dhead)
        * torch.rand(batch, seqlen, nheads, dtype=torch.float32, device=DEVICE),
        dim=1,
    )
    args = (k, w, u, g, chunk_size)
    run_example(helion_gdn_fwd_h, ref_gdn_fwd_h, args)


# %%
# Main Function
# -----------
def main() -> None:
    """
    Main entry point that runs the attention kernel test with specific parameters.
    """
    test(8, 80, 4096, 256, 64, 128)


if __name__ == "__main__":
    main()
