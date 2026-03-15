"""
Gated DeltaNet Recompute W/U
============================

This example implements the chunk-local WY-transform recompute kernel used by
Gated DeltaNet.
"""

from __future__ import annotations

import torch

import helion
from helion._testing import DEVICE
import helion.language as hl


def _chunk_local_cumsum(g: torch.Tensor, chunk_size: int) -> torch.Tensor:
    batch, seqlen, nheads = g.shape
    return g.float().reshape(batch, seqlen // chunk_size, chunk_size, nheads).cumsum(
        dim=2
    ).reshape(batch, seqlen, nheads)


def _chunk_scaled_dot_kkt_fwd(
    k: torch.Tensor, g_cumsum: torch.Tensor, beta: torch.Tensor, chunk_size: int
) -> torch.Tensor:
    batch, seqlen, nheads, dhead = k.shape
    nchunks = seqlen // chunk_size
    k_c = k.float().reshape(batch, nchunks, chunk_size, nheads, dhead).permute(
        0, 1, 3, 2, 4
    )
    g_c = g_cumsum.float().reshape(batch, nchunks, chunk_size, nheads).permute(
        0, 1, 3, 2
    )
    beta_c = beta.float().reshape(batch, nchunks, chunk_size, nheads).permute(
        0, 1, 3, 2
    )
    kkt = k_c @ k_c.transpose(-1, -2)
    g_diff = g_c.unsqueeze(-1) - g_c.unsqueeze(-2)
    strict_lower = torch.tril(
        torch.ones(chunk_size, chunk_size, device=k.device), diagonal=-1
    )
    A = kkt * beta_c.unsqueeze(-1) * torch.exp(g_diff) * strict_lower
    return A.permute(0, 1, 3, 2, 4).reshape(batch, seqlen, nheads, chunk_size)


def _solve_tril(A: torch.Tensor, output_dtype: torch.dtype) -> torch.Tensor:
    batch, seqlen, nheads, chunk_size = A.shape
    nchunks = seqlen // chunk_size
    A_mat = A.float().reshape(batch, nchunks, chunk_size, nheads, chunk_size).permute(
        0, 1, 3, 2, 4
    )
    eye = torch.eye(chunk_size, device=A.device).expand_as(A_mat)
    result = torch.linalg.solve_triangular(eye + A_mat, eye, upper=False)
    return result.permute(0, 1, 3, 2, 4).reshape(batch, seqlen, nheads, chunk_size).to(
        output_dtype
    )


@helion.kernel()
def helion_gdn_recompute_w_u(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    g: torch.Tensor,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, seqlen, nheads, dhead = k.shape
    dvalue = v.shape[-1]
    chunk_size = hl.specialize(chunk_size)
    dhead = hl.specialize(dhead)
    dvalue = hl.specialize(dvalue)

    w_out = torch.empty_like(k)
    u_out = torch.empty_like(v)
    flat_heads = batch * nheads

    for tile_bh, tile_t in hl.tile([flat_heads, seqlen], block_size=[1, chunk_size]):
        batch_idx = tile_bh.begin // nheads
        head_idx = tile_bh.begin % nheads

        A_chunk = A[batch_idx, tile_t, head_idx, :].to(torch.float32)
        beta_chunk = beta[batch_idx, tile_t, head_idx].to(torch.float32)
        g_chunk = g[batch_idx, tile_t, head_idx].to(torch.float32)
        rhs_k = k[batch_idx, tile_t, head_idx, :].to(torch.float32)
        rhs_v = v[batch_idx, tile_t, head_idx, :].to(torch.float32)

        w_out[batch_idx, tile_t, head_idx, :] = hl.dot(
            A_chunk,
            rhs_k * (beta_chunk * torch.exp(g_chunk))[:, None],
            out_dtype=torch.float32,
        ).to(w_out.dtype)
        u_out[batch_idx, tile_t, head_idx, :] = hl.dot(
            A_chunk,
            rhs_v * beta_chunk[:, None],
            out_dtype=torch.float32,
        ).to(u_out.dtype)

    return w_out, u_out


def ref_gdn_recompute_w_u(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    g: torch.Tensor,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, seqlen, nheads, dhead = k.shape
    dvalue = v.shape[-1]
    nchunks = seqlen // chunk_size
    k_c = k.float().reshape(batch, nchunks, chunk_size, nheads, dhead).permute(
        0, 1, 3, 2, 4
    )
    v_c = v.float().reshape(batch, nchunks, chunk_size, nheads, dvalue).permute(
        0, 1, 3, 2, 4
    )
    beta_c = beta.float().reshape(batch, nchunks, chunk_size, nheads).permute(
        0, 1, 3, 2
    )
    g_c = g.float().reshape(batch, nchunks, chunk_size, nheads).permute(0, 1, 3, 2)
    A_c = A.float().reshape(batch, nchunks, chunk_size, nheads, chunk_size).permute(
        0, 1, 3, 2, 4
    )
    u_c = A_c @ (v_c * beta_c.unsqueeze(-1))
    w_c = A_c @ (k_c * (beta_c * torch.exp(g_c)).unsqueeze(-1))
    w = w_c.permute(0, 1, 3, 2, 4).reshape(batch, seqlen, nheads, dhead).to(k.dtype)
    u = u_c.permute(0, 1, 3, 2, 4).reshape(batch, seqlen, nheads, dvalue).to(v.dtype)
    return w, u


def make_inputs(
    batch: int,
    nheads: int,
    seqlen: int,
    chunk_size: int,
    dhead: int,
    dvalue: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    k = torch.randn(batch, seqlen, nheads, dhead, dtype=torch.float32, device=DEVICE)
    v = torch.randn(batch, seqlen, nheads, dvalue, dtype=torch.float32, device=DEVICE)
    beta = torch.sigmoid(
        torch.randn(batch, seqlen, nheads, dtype=torch.float32, device=DEVICE)
    )
    g = -torch.abs(
        torch.randn(batch, seqlen, nheads, dtype=torch.float32, device=DEVICE)
    ).cumsum(dim=1)
    g_cumsum = _chunk_local_cumsum(g, chunk_size)
    A = _chunk_scaled_dot_kkt_fwd(k, g_cumsum, beta, chunk_size)
    A = _solve_tril(A, k.dtype)
    return k, v, beta, A, g_cumsum, chunk_size


def test(
    batch: int, nheads: int, seqlen: int, chunk_size: int, dhead: int, dvalue: int
) -> None:
    args = make_inputs(batch, nheads, seqlen, chunk_size, dhead, dvalue)
    expected_w, expected_u = ref_gdn_recompute_w_u(*args)
    got_w, got_u = helion_gdn_recompute_w_u(*args)
    torch.testing.assert_close(got_w.to(torch.float32), expected_w.to(torch.float32))
    torch.testing.assert_close(got_u.to(torch.float32), expected_u.to(torch.float32))


def main() -> None:
    test(2, 4, 128, 64, 64, 64)


if __name__ == "__main__":
    main()
