"""
Mamba2 Chunk State Kernel
========================

This code implements a chunked state kernel as used for Mamba2
"""

# %%
# Imports
# -------
from __future__ import annotations

import functools

import torch
import torch.nn.functional as F

import helion
from helion._testing import DEVICE
from helion._testing import run_example
import helion.language as hl


# %%
# Helion Kernel Implementation
# ----------------------------
@helion.kernel()
def helion_mamba2_chunk_state_kernel(
    B: torch.Tensor, x: torch.Tensor, dt: torch.Tensor, dA_cumsum: torch.Tensor
) -> torch.Tensor:
    """
    Argument:
        B: (batch, seqlen, ngroups, dstate)
        x: (batch, seqlen, nheads, headdim)
        dt: (batch, nheads, nchunks, chunk_size)
        dA_cumsum: (batch, nheads, nchunks, chunk_size)
    Return:
        states: (batch, nchunks, nheads, headdim, dstate)
    """

    batch, seqlen, ngroups, dstate = B.shape
    batch, seqlen, nheads, headdim = x.shape
    batch, nheads, nchunks, chunk_size = dt.shape
    batch, nheads, nchunks, chunk_size = dA_cumsum.shape

    assert nchunks == (seqlen + chunk_size - 1) // chunk_size

    block_m = hl.register_block_size(headdim)
    block_n = hl.register_block_size(dstate)
    block_k = hl.register_block_size(chunk_size)

    assert B.shape == (batch, seqlen, ngroups, dstate)
    assert x.shape == (batch, seqlen, nheads, headdim)
    assert dt.shape == (batch, nheads, nchunks, chunk_size)
    assert dA_cumsum.shape == (batch, nheads, nchunks, chunk_size)

    dtype = B.dtype
    accum_dtype = torch.float32
    assert x.dtype == dt.dtype == dA_cumsum.dtype == dtype

    out = B.new_empty(batch, nchunks, nheads, headdim, dstate)

    p = 1.44269504

    for tile_h, tile_m, tile_n, tile_b, tile_c in hl.tile(
        [nheads, headdim, dstate, batch, nchunks],
        block_size=[1, block_m, block_n, 1, 1],
    ):
        dA_cumsum_last = dA_cumsum[
            tile_b.begin, tile_h.begin, tile_c.begin, chunk_size - 1
        ].to(accum_dtype)
        acc_o = hl.zeros([tile_m, tile_n], dtype=accum_dtype)
        for tile_k in hl.tile(chunk_size, block_size=block_k):
            x_local = x[
                tile_b.begin,
                tile_k.index + tile_c.begin * chunk_size,
                tile_h.begin,
                tile_m,
            ]
            dA_cumsum_local = dA_cumsum[
                tile_b.begin, tile_h.begin, tile_c.begin, tile_k
            ].to(accum_dtype)
            dt_local = dt[tile_b.begin, tile_h.begin, tile_c.begin, tile_k]
            scale = torch.exp2(dA_cumsum_last * p - dA_cumsum_local * p) * dt_local
            xt_local = (x_local.T * scale[None, :]).to(dtype)
            B_local = B[
                tile_b.begin,
                tile_c.begin * chunk_size + tile_k.index,
                tile_h.begin // (nheads // ngroups),
                tile_n,
            ]
            acc_o = hl.dot(xt_local, B_local, acc=acc_o)
        out[tile_b.begin, tile_c.begin, tile_h.begin, tile_m, tile_n] = acc_o.to(dtype)

    return out


# %%
# Reference Function
# -------------
def ref_chunk_state(
    B: torch.Tensor, x: torch.Tensor, dt: torch.Tensor, dA_cumsum: torch.Tensor
) -> torch.Tensor:
    """
    Argument:
        B: (batch, seqlen, ngroups, dstate)
        x: (batch, seqlen, nheads, dhead)
        dt: (batch, nheads, nchunks, chunk_size)
        dA_cumsum: (batch, nheads, nchunks, chunk_size)
    Return:
        states: (batch, nchunks, nheads, dhead, dstate)
    """
    # Check constraints.
    batch, seqlen, nheads, dhead = x.shape
    dstate = B.shape[-1]
    _, _, nchunks, chunk_size = dt.shape
    assert seqlen <= nchunks * chunk_size
    assert x.shape == (batch, seqlen, nheads, dhead)
    assert dt.shape == (batch, nheads, nchunks, chunk_size)
    ngroups = B.shape[2]
    assert nheads % ngroups == 0
    assert B.shape == (batch, seqlen, ngroups, dstate)
    B = torch.repeat_interleave(B, nheads // ngroups, dim=2)
    assert dA_cumsum.shape == (batch, nheads, nchunks, chunk_size)
    if seqlen < nchunks * chunk_size:
        x = F.pad(x, (0, 0, 0, 0, 0, nchunks * chunk_size - seqlen))
        B = F.pad(B, (0, 0, 0, 0, 0, nchunks * chunk_size - seqlen))
    x = x.reshape(batch, nchunks, chunk_size, nheads, dhead)
    B = B.reshape(batch, nchunks, chunk_size, nheads, dstate)
    decay_states = torch.exp(dA_cumsum[:, :, :, -1:] - dA_cumsum)
    return torch.einsum(
        "bclhn,bhcl,bhcl,bclhp->bchpn",
        B.to(x.dtype),
        decay_states.to(x.dtype),
        dt.to(x.dtype),
        x,
    )


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

    B = fn(batch, seqlen, ngroups, dstate)
    x = fn(batch, seqlen, nheads, headdim)
    dt = fn(batch, nheads, nchunks, chunk_size)
    dA_cumsum = fn(batch, nheads, nchunks, chunk_size)
    args = (B, x, dt, dA_cumsum)
    run_example(helion_mamba2_chunk_state_kernel, ref_chunk_state, args)


# %%
# Main Function
# -----------
def main() -> None:
    test("uuuu", 8, 80, 1, 4096, 256, 64, 128)


if __name__ == "__main__":
    main()
