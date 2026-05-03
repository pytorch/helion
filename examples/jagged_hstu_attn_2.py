"""
Jagged HSTU Attention (v2) Example
===================================

Jagged (unpadded) multi-head HSTU attention using data-dependent tile loops.

Unlike :mod:`jagged_hstu_attn`, which tiles over a fixed ``max_seq_len``
and guards with ``if tile_q.begin < seq_len``, this version iterates
directly over each sequence's ``[start, end)`` range via
``hl.tile(start, end)``.  The approach avoids wasted work on padding and
is compatible with both the Triton and Pallas backends.

Tensor shapes::

    q, k, v      : [L, H, D]   (L = total tokens across all sequences)
    seq_offsets  : [B + 1]      (int32 cumulative offsets)
    output       : [L, H, D]
"""

# %%
# Imports
# -------

# %%
from __future__ import annotations

import torch
import torch.nn.functional as F

import helion
from helion._testing import DEVICE
from helion._testing import run_example
import helion.language as hl

# %%
# Reference Implementation
# ------------------------


# %%
def _jagged_to_padded(
    values: torch.Tensor,
    offsets: torch.Tensor,
    max_len: int,
) -> torch.Tensor:
    """[L, D] jagged -> [B, max_len, D] padded with zeros."""
    B = offsets.shape[0] - 1
    D = values.shape[1]
    out = values.new_zeros(B, max_len, D)
    for i in range(B):
        s, e = int(offsets[i]), int(offsets[i + 1])
        n = min(e - s, max_len)
        out[i, :n] = values[s : s + n]
    return out


def _padded_to_jagged(
    padded: torch.Tensor,
    offsets: torch.Tensor,
    total_L: int,
) -> torch.Tensor:
    """[B, max_len, D] padded -> [L, D] jagged."""
    D = padded.shape[2]
    out = padded.new_zeros(total_L, D)
    B = offsets.shape[0] - 1
    for i in range(B):
        s, e = int(offsets[i]), int(offsets[i + 1])
        out[s:e] = padded[i, : e - s]
    return out


def reference_jagged_hstu_attention(
    max_seq_len: int,
    alpha: float,
    attn_scale: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
) -> torch.Tensor:
    """Pad-then-compute reference matching the production HSTU pattern."""
    L, H, D = q.shape

    padded_q = (
        _jagged_to_padded(q.reshape(L, H * D), seq_offsets, max_seq_len)
        .view(-1, max_seq_len, H, D)
        .transpose(1, 2)
    )  # [B, H, N, D]
    padded_k = (
        _jagged_to_padded(k.reshape(L, H * D), seq_offsets, max_seq_len)
        .view(-1, max_seq_len, H, D)
        .transpose(1, 2)
    )  # [B, H, N, D]
    padded_v = (
        _jagged_to_padded(v.reshape(L, H * D), seq_offsets, max_seq_len)
        .view(-1, max_seq_len, H, D)
        .transpose(1, 2)
    )  # [B, H, N, D]

    qk_attn = torch.einsum("bhxa,bhya->bhxy", padded_q, padded_k) * alpha
    qk_attn = F.silu(qk_attn) * attn_scale

    causal_mask = torch.ones(
        max_seq_len, max_seq_len, dtype=torch.bool, device=q.device
    ).tril()
    qk_attn = qk_attn * causal_mask.unsqueeze(0).unsqueeze(0)

    attn_out = torch.einsum("bhxd,bhdv->bhxv", qk_attn.to(v.dtype), padded_v)
    return _padded_to_jagged(
        attn_out.transpose(1, 2).flatten(2, 3),
        seq_offsets,
        L,
    ).view(L, H, D)


# %%
# Kernel
# ------


# %%
@helion.kernel(
    static_shapes=True,
    autotune_baseline_fn=reference_jagged_hstu_attention,
)
def jagged_hstu_attention(
    max_seq_len: int,
    alpha: float,
    attn_scale: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
) -> torch.Tensor:
    """Jagged HSTU attention forward pass.

    For each sequence defined by ``seq_offsets``, computes causal
    SiLU-gated attention::

        scores = silu(q @ k ^ T * alpha) * attn_scale
        scores = where(causal_mask, scores, 0)
        out = scores @ v
    """
    H = hl.specialize(q.size(1))
    D = hl.specialize(q.size(2))
    num_sequences = seq_offsets.size(0) - 1
    out = torch.empty_like(v)

    for seq_idx in hl.grid(num_sequences):
        start = seq_offsets[seq_idx]
        end = seq_offsets[seq_idx + 1]

        for tile_q in hl.tile(start, end):
            # [tile_q, H, D] -> [H, tile_q, D] for batched matmul
            q_blk = q[tile_q, :, :].transpose(0, 1)
            acc = hl.zeros([H, tile_q, D], dtype=torch.float32)

            for tile_kv in hl.tile(start, end):
                k_blk = k[tile_kv, :, :].transpose(0, 1)  # [H, tile_kv, D]
                v_blk = v[tile_kv, :, :].transpose(0, 1)  # [H, tile_kv, D]

                # [H, tile_q, D] @ [H, D, tile_kv] -> [H, tile_q, tile_kv]
                scores = torch.bmm(q_blk, k_blk.transpose(-2, -1)) * alpha
                scores = F.silu(scores) * attn_scale

                # causal mask: [tile_q, tile_kv] broadcast over H
                causal_mask = tile_q.index.unsqueeze(1) >= tile_kv.index.unsqueeze(0)
                scores = torch.where(causal_mask[None, :, :], scores, 0.0)

                # [H, tile_q, tile_kv] @ [H, tile_kv, D] -> [H, tile_q, D]
                acc = acc + torch.bmm(scores.to(v.dtype), v_blk)

            # [H, tile_q, D] -> [tile_q, H, D]
            out[tile_q, :, :] = acc.transpose(0, 1).to(out.dtype)

    return out


# %%
# Main
# ----


# %%
def main() -> None:
    torch.manual_seed(0)
    num_sequences = 16
    max_seq_len = 512
    heads = 32
    head_dim = 128
    alpha = 1.0 / head_dim**2
    attn_scale = 1.0 / max_seq_len
    dtype = torch.float32
    device = torch.device(DEVICE)

    lengths = torch.randint(
        max_seq_len // 2,
        max_seq_len + 1,
        (num_sequences,),
        dtype=torch.int32,
    )
    seq_offsets = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32),
            torch.cumsum(lengths, dim=0).to(torch.int32),
        ]
    ).to(device)
    L = int(seq_offsets[-1].item())

    q = torch.randn(L, heads, head_dim, dtype=dtype, device=device) * 0.1
    k = torch.randn(L, heads, head_dim, dtype=dtype, device=device) * 0.1
    v = torch.randn(L, heads, head_dim, dtype=dtype, device=device) * 0.1

    run_example(
        lambda *a: jagged_hstu_attention(*a),
        lambda *a: reference_jagged_hstu_attention(*a),
        (max_seq_len, alpha, attn_scale, q, k, v, seq_offsets),
        atol=1e-2,
        rtol=1e-2,
    )


if __name__ == "__main__":
    main()
