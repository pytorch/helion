"""
Mamba2 Chunk Scan Kernel
========================

This code implements a chunked scan kernel as used for Mamba2
"""

# %%
# Imports
# -------
from __future__ import annotations

import functools

import torch

import helion
from helion._testing import DEVICE
from helion._testing import run_example
import helion.language as hl


# %%
# Helion Kernel Implementation
# ----------------------------
@helion.kernel()
def helion_mamba2_chunk_scan_kernel(
    cb: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    dA_cumsum: torch.Tensor,
    C: torch.Tensor,
    prev_states: torch.Tensor,
    D: torch.Tensor,
) -> torch.Tensor:
    """
    Argument:
        cb: (batch, nchunks, ngroups, chunk_size, chunk_size)
        x: (batch, seqlen, nheads, headdim)
        dt: (batch, nheads, nchunks, chunk_size)
        dA_cumsum: (batch, nheads, nchunks, chunk_size)
        C: (batch, seqlen, ngroups, dstate)
        prev_states: (batch, nchunks, nheads, headdim, dstate)
        D: (nheads,)
    Return:
        out: (batch, seqlen, nheads, headdim)
    """

    batch, nchunks, ngroups, chunk_size, _ = cb.shape
    _, seqlen, nheads, headdim = x.shape
    _, _, _, dstate = C.shape
    assert nchunks == (seqlen + chunk_size - 1) // chunk_size

    block_m = hl.register_block_size(chunk_size)
    block_n = hl.register_block_size(headdim)
    block_k = hl.register_block_size(64, 64)
    dstate = hl.specialize(dstate)

    assert cb.shape == (batch, nchunks, ngroups, chunk_size, chunk_size)
    assert x.shape == (batch, seqlen, nheads, headdim)
    assert dt.shape == (batch, nheads, nchunks, chunk_size)
    assert dA_cumsum.shape == (batch, nheads, nchunks, chunk_size)
    assert C.shape == (batch, seqlen, ngroups, dstate)
    assert prev_states.shape == (batch, nchunks, nheads, headdim, dstate)
    assert D.shape == (nheads,)

    dtype = cb.dtype
    accum_dtype = torch.float32
    assert (
        x.dtype
        == dt.dtype
        == dA_cumsum.dtype
        == C.dtype
        == prev_states.dtype
        == D.dtype
        == dtype
    )

    out = torch.empty_like(x)

    p = 1.44269504

    for tile_h, tile_m, tile_n, tile_b, tile_c in hl.tile(
        [nheads, chunk_size, headdim, batch, nchunks],
        block_size=[1, block_m, block_n, 1, 1],
    ):
        acc_o = hl.zeros([tile_m, tile_n], dtype=accum_dtype)
        dA_cumsum_local_m = dA_cumsum[
            tile_b.begin, tile_h.begin, tile_c.begin, tile_m
        ].to(torch.float32)
        scale_m_local = torch.exp2(dA_cumsum_local_m * p)

        C_local = C[
            tile_b.begin,
            tile_m.index + tile_c.begin * chunk_size,
            tile_h.begin // (nheads // ngroups),
            :,
        ]
        prev_states_local = prev_states[
            tile_b.begin, tile_c.begin, tile_h.begin, tile_n, :
        ]
        acc_o = hl.dot(C_local, prev_states_local.T, acc=acc_o)
        acc_o *= scale_m_local[:, None]

        for tile_k in hl.tile((tile_m.id + 1) * block_m, block_size=block_k):
            cb_local = cb[
                tile_b.begin,
                tile_c.begin,
                tile_h.begin // (nheads // ngroups),
                tile_m,
                tile_k,
            ]
            dA_cumsum_local_k = dA_cumsum[
                tile_b.begin, tile_h.begin, tile_c.begin, tile_k
            ].to(torch.float32)
            cb_local *= torch.exp2(
                dA_cumsum_local_m[:, None] * p - dA_cumsum_local_k[None, :] * p
            )
            dt_local = dt[tile_b.begin, tile_h.begin, tile_c.begin, tile_k].to(
                torch.float32
            )
            cb_local = (cb_local * dt_local[None, :]).to(dtype)
            pred = (tile_m.index + 0)[:, None] >= (tile_k.index + 0)[None, :]
            cb_local = torch.where(pred, cb_local, torch.zeros_like(cb_local))
            x_local = x[
                tile_b.begin,
                tile_c.begin * chunk_size + tile_k.index,
                tile_h.begin,
                tile_n,
            ]
            acc_o = hl.dot(cb_local, x_local, acc=acc_o)

        D_local = D[tile_h.begin].to(torch.float32)
        x_residual = x[
            tile_b.begin, tile_c.begin * chunk_size + tile_m.index, tile_h.begin, tile_n
        ].to(torch.float32)
        acc_o += x_residual * D_local
        out[
            tile_b.begin, tile_c.begin * chunk_size + tile_m.index, tile_h.begin, tile_n
        ] = acc_o.to(dtype=dtype)

    return out


# %%
# Reference Function
# -------------
def ref_chunk_scan(
    cb: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    dA_cumsum: torch.Tensor,
    C: torch.Tensor,
    prev_states: torch.Tensor,
    D: torch.Tensor,
) -> torch.Tensor:
    """
    Argument:
        cb: (batch, nchunks, ngroups, chunk_size, chunk_size)
        x: (batch, seqlen, nheads, dhead)
        dt: (batch, nheads, nchunks, chunk_size)
        dA_cumsum: (batch, nheads, nchunks, chunk_size)
        C: (batch, seqlen, ngroups, dstate)
        prev_states: (batch, nchunks, nheads, dhead, dstate)
        D: (nheads,)
    Return:
        out: (batch, seqlen, nheads, dhead)
    """
    _, _, ngroups, _, _ = cb.shape
    batch, seqlen, nheads, dhead = x.shape
    # _, _, ngroups, dstate = B.shape
    # assert B.shape == (batch, seqlen, ngroups, dstate)
    _, _, nchunks, chunk_size = dt.shape
    dstate = C.shape[-1]
    assert seqlen == nchunks * chunk_size
    # assert C.shape == B.shape
    # B = repeat(B, "b l g d -> b l (g h) d", h=nheads // ngroups)
    C = torch.repeat_interleave(C, nheads // ngroups, dim=2)
    cb = torch.repeat_interleave(cb, nheads // ngroups, dim=2)
    # CB = torch.einsum("bclhn,bcshn->bchls", rearrange(C, "b (c l) h n -> b c l h n", c=nchunks),
    #                   rearrange(B, "b (c s) h n -> b c s h n", c=nchunks))
    # (batch, nheads, nchunks, chunksize, chunksize)
    dt_segment_sum = dA_cumsum[:, :, :, :, None] - dA_cumsum[:, :, :, None, :]
    decay = torch.exp(dt_segment_sum)
    scores_decay = cb * decay.permute(0, 2, 1, 3, 4)
    causal_mask = torch.tril(
        torch.ones(chunk_size, chunk_size, device=x.device, dtype=torch.bool),
        diagonal=0,
    )
    scores_decay = scores_decay.masked_fill(~causal_mask, 0)
    out = torch.einsum(
        "bchls,bhcs,bcshp->bclhp",
        scores_decay.to(x.dtype),
        dt.to(x.dtype),
        x.reshape(batch, nchunks, chunk_size, nheads, dhead),
    )
    # state_decay_out = torch.exp(rearrange(dA_cumsum, "b h c l -> b c l h 1"))
    state_decay_out = torch.exp(dA_cumsum.permute(0, 2, 3, 1).unsqueeze(-1))
    out_prev = (
        torch.einsum(
            "bclhn,bchpn->bclhp",
            C.reshape(batch, nchunks, chunk_size, nheads, dstate),
            prev_states.to(C.dtype),
        )
        * state_decay_out
    )
    out = out + out_prev
    out = out.reshape(batch, seqlen, nheads, dhead)
    if D is not None:
        if D.dim() == 1:
            D = D.unsqueeze(-1)
        out = out + x * D
    return out


# %%
# Testing Function
# -------------
def test(
    init: str,
    batch: int,
    nheads: int,
    ngroups: int,
    seqlen: int,
    chunk_size: int,
    headdim: int,
    dstate: int,
    dtype: torch.dtype = torch.float16,
) -> None:
    INIT = {
        "r": functools.partial(torch.randn, dtype=dtype, device=DEVICE),
        "u": functools.partial(torch.rand, dtype=dtype, device=DEVICE),
        "z": functools.partial(torch.zeros, dtype=dtype, device=DEVICE),
        "o": functools.partial(torch.ones, dtype=dtype, device=DEVICE),
    }
    nchunks = (seqlen + chunk_size - 1) // chunk_size
    idx = 0

    def fn(*args: int) -> torch.Tensor:
        nonlocal idx
        ret = INIT[init[idx]](*args)
        idx += 1
        return ret

    cb = fn(batch, nchunks, ngroups, chunk_size, chunk_size)
    x = fn(batch, seqlen, nheads, headdim)
    dt = fn(batch, nheads, nchunks, chunk_size)
    dA_cumsum = fn(batch, nheads, nchunks, chunk_size)  # init range is too large
    C = fn(batch, seqlen, ngroups, dstate)
    prev_states = fn(batch, nchunks, nheads, headdim, dstate)
    D = fn(nheads)
    args = (cb, x, dt, dA_cumsum, C, prev_states, D)
    run_example(helion_mamba2_chunk_scan_kernel, ref_chunk_scan, args)


# %%
# Main Function
# -----------
def main() -> None:
    """
    Main entry point that runs the attention kernel test with specific parameters.
    Tests with batch size 2, 32 heads, 1024 sequence length, and 64-dimensional heads using float16.
    """
    test("zzzzzzz", 8, 80, 1, 4096, 256, 64, 128)
    test("zrzzzzr", 8, 80, 1, 4096, 256, 64, 128)  # D * x
    test("zzzzrrz", 8, 80, 1, 4096, 256, 64, 128)  # C * prev_state
    test("zzzrrrz", 8, 80, 1, 4096, 256, 64, 128)  # C * prev_state * dA
    test("rrrzzzz", 8, 80, 1, 4096, 256, 64, 128)  # cb * x * dt
    test("rrruzzz", 8, 80, 1, 4096, 256, 64, 128)  # cb * x * dt * dA


if __name__ == "__main__":
    main()
