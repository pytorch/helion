"""
Gated DeltaNet Forward O
========================

This example implements the output kernel used by Gated DeltaNet.
"""

from __future__ import annotations

import torch

import helion
from helion._testing import DEVICE
from helion._testing import HALF_DTYPE
from helion._testing import run_example
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


def _recompute_w_u(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    g: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, seqlen, nheads, dhead = k.shape
    dvalue = v.shape[-1]
    chunk_size = A.shape[-1]
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


def _chunk_fwd_h(
    k: torch.Tensor, w: torch.Tensor, u: torch.Tensor, g: torch.Tensor, chunk_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, seqlen, nheads, dhead = k.shape
    dvalue = u.shape[-1]
    nchunks = seqlen // chunk_size
    k_c = k.float().reshape(batch, nchunks, chunk_size, nheads, dhead).permute(
        0, 1, 3, 2, 4
    )
    w_c = w.float().reshape(batch, nchunks, chunk_size, nheads, dhead).permute(
        0, 1, 3, 2, 4
    )
    u_c = u.float().reshape(batch, nchunks, chunk_size, nheads, dvalue).permute(
        0, 1, 3, 2, 4
    )
    g_c = g.float().reshape(batch, nchunks, chunk_size, nheads).permute(0, 1, 3, 2)
    h_all = torch.zeros(batch, nchunks, nheads, dhead, dvalue, device=k.device)
    v_new_c = torch.zeros_like(u_c)
    h = torch.zeros(batch, nheads, dhead, dvalue, dtype=torch.float32, device=k.device)
    for chunk in range(nchunks):
        h_all[:, chunk] = h
        v_new_c[:, chunk] = u_c[:, chunk] - w_c[:, chunk] @ h
        g_last = g_c[:, chunk, :, -1]
        gate = torch.exp(g_last.unsqueeze(-1) - g_c[:, chunk])
        v_gated = v_new_c[:, chunk] * gate.unsqueeze(-1)
        h = h * torch.exp(g_last).unsqueeze(-1).unsqueeze(-1)
        h = h + k_c[:, chunk].transpose(-1, -2) @ v_gated
    v_new = v_new_c.permute(0, 1, 3, 2, 4).reshape(batch, seqlen, nheads, dvalue)
    return h_all.to(k.dtype), v_new.to(u.dtype)


@helion.kernel()
def gdn_fwd_o(
    q: torch.Tensor,
    k: torch.Tensor,
    v_new: torch.Tensor,
    state: torch.Tensor,
    g: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    batch, seqlen, nheads, dhead = q.shape
    dvalue = v_new.shape[-1]
    nchunks = seqlen // chunk_size
    chunk_size = hl.specialize(chunk_size)
    dhead = hl.specialize(dhead)
    dvalue = hl.specialize(dvalue)

    out = torch.empty_like(v_new)
    flat_heads = batch * nheads
    scale = dhead**-0.5

    for tile_bh, tile_chunk in hl.grid([flat_heads, nchunks]):
        batch_idx = tile_bh // nheads
        head_idx = tile_bh % nheads
        t_start = tile_chunk * chunk_size
        t_stop = t_start + chunk_size
        state_chunk = state[batch_idx, tile_chunk, head_idx, :, :].to(torch.float32)

        for tile_k in hl.tile(t_start, t_stop, block_size=chunk_size):
            k_chunk = k[batch_idx, tile_k, head_idx, :].to(torch.float32)
            v_chunk = v_new[batch_idx, tile_k, head_idx, :].to(torch.float32)
            g_chunk = g[batch_idx, tile_k, head_idx].to(torch.float32)
            for tile_q in hl.tile(t_start, t_stop, block_size=chunk_size):
                q_chunk = q[batch_idx, tile_q, head_idx, :].to(torch.float32)
                q_gate = g[batch_idx, tile_q, head_idx].to(torch.float32)
                sim = hl.dot(q_chunk, k_chunk.T, out_dtype=torch.float32)
                sim = sim * torch.exp(q_gate[:, None] - g_chunk[None, :])
                causal = (tile_q.index[:, None] >= tile_k.index[None, :]).to(
                    torch.float32
                )
                local_out = hl.dot(sim * causal, v_chunk, out_dtype=torch.float32)
                global_out = hl.dot(q_chunk, state_chunk, out_dtype=torch.float32)
                out[batch_idx, tile_q, head_idx, :] = (
                    (global_out * torch.exp(q_gate)[:, None] + local_out) * scale
                ).to(out.dtype)

    return out


def ref_gdn_fwd_o(
    q: torch.Tensor,
    k: torch.Tensor,
    v_new: torch.Tensor,
    state: torch.Tensor,
    g: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    batch, seqlen, nheads, dhead = q.shape
    dvalue = v_new.shape[-1]
    nchunks = seqlen // chunk_size
    scale = dhead**-0.5
    q_c = q.float().reshape(batch, nchunks, chunk_size, nheads, dhead).permute(
        0, 1, 3, 2, 4
    )
    k_c = k.float().reshape(batch, nchunks, chunk_size, nheads, dhead).permute(
        0, 1, 3, 2, 4
    )
    v_c = v_new.float().reshape(batch, nchunks, chunk_size, nheads, dvalue).permute(
        0, 1, 3, 2, 4
    )
    g_c = g.float().reshape(batch, nchunks, chunk_size, nheads).permute(0, 1, 3, 2)
    o_inter = (q_c @ state.float()) * torch.exp(g_c).unsqueeze(-1)
    qk = q_c @ k_c.transpose(-1, -2) * torch.exp(g_c.unsqueeze(-1) - g_c.unsqueeze(-2))
    causal = torch.tril(torch.ones(chunk_size, chunk_size, device=q.device))
    out = (o_inter + (qk * causal) @ v_c) * scale
    return out.permute(0, 1, 3, 2, 4).reshape(batch, seqlen, nheads, dvalue).to(q.dtype)


def make_inputs(
    batch: int,
    nheads: int,
    seqlen: int,
    chunk_size: int,
    dhead: int,
    dvalue: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    q = torch.randn(batch, seqlen, nheads, dhead, dtype=HALF_DTYPE, device=DEVICE)
    k = torch.randn(batch, seqlen, nheads, dhead, dtype=HALF_DTYPE, device=DEVICE)
    v = torch.randn(batch, seqlen, nheads, dvalue, dtype=HALF_DTYPE, device=DEVICE)
    beta = torch.sigmoid(
        torch.randn(batch, seqlen, nheads, dtype=torch.float32, device=DEVICE)
    )
    g = -torch.abs(
        torch.randn(batch, seqlen, nheads, dtype=torch.float32, device=DEVICE)
    ).cumsum(dim=1)
    g_cumsum = _chunk_local_cumsum(g, chunk_size)
    A = _chunk_scaled_dot_kkt_fwd(k, g_cumsum, beta, chunk_size)
    A = _solve_tril(A, k.dtype)
    w, u = _recompute_w_u(k, v, beta, A, g_cumsum)
    state, v_new = _chunk_fwd_h(k, w, u, g_cumsum, chunk_size)
    return q, k, v_new, state, g_cumsum, chunk_size


def test(
    batch: int, nheads: int, seqlen: int, chunk_size: int, dhead: int, dvalue: int
) -> None:
    args = make_inputs(batch, nheads, seqlen, chunk_size, dhead, dvalue)
    run_example(gdn_fwd_o, ref_gdn_fwd_o, args, atol=1e-2, rtol=1e-2)


def main() -> None:
    test(1, 4, 128, 64, 64, 64)


if __name__ == "__main__":
    main()
