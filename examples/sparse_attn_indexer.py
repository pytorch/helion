"""
Sparse Attention Indexer Logits
===============================

The scoring step of the sparse-attention indexer used by DeepSeek V3.2 / V4,
GLM-5.2 and MiniMax M3: a small multi-query attention whose scores decide which
KV positions the real attention reads.

    logits[m, n] = sum_h relu(q[m, h, :] @ k[n, :]) * weights[m, h]

for ``H`` index heads of width ``D``, masked to the row's window ``[ks[m], ke[m])``.
``einsum("mhd,nd->hmn")`` would materialise an ``[H, M, N]`` tensor; both kernels
accumulate over heads inside the tile instead.

``mqa_logits`` runs one matmul per head, with the key tile hoisted out of the head
loop. ``mqa_logits_decode`` folds the heads into one batched matmul, which is
faster when there are few query rows -- on an A800 the two cross at 32 rows.
"""

# %%
from __future__ import annotations

import torch

import helion
from helion._testing import DEVICE
from helion._testing import run_example
import helion.language as hl


# %%
@helion.kernel(static_shapes=True)
def mqa_logits(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    ks: torch.Tensor,
    ke: torch.Tensor,
) -> torch.Tensor:
    """Indexer logits, one matmul per index head.

    Args:
        q: Index queries, ``[M, H, D]``
        k: Shared index keys, ``[N, D]``
        weights: Per-(row, head) score weights, ``[M, H]`` float32
        ks: Inclusive window start per query row, ``[M]``
        ke: Exclusive window end per query row, ``[M]``

    Returns:
        ``[M, N]`` float32 logits, ``-inf`` outside each row's window.
    """
    m_dim, h_dim, d_dim = q.size()
    n_dim = k.size(0)
    h_dim = hl.specialize(h_dim)
    d_dim = hl.specialize(d_dim)
    out = torch.empty([m_dim, n_dim], dtype=torch.float32, device=q.device)
    for tile_m, tile_n in hl.tile([m_dim, n_dim]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        kt = k[tile_n, :]
        for h in hl.grid(h_dim):
            score = torch.matmul(q[tile_m, h, :], kt.transpose(0, 1)).to(torch.float32)
            acc = acc + torch.relu(score) * weights[tile_m, h][:, None]
        in_window = (tile_n.index[None, :] >= ks[tile_m][:, None]) & (
            tile_n.index[None, :] < ke[tile_m][:, None]
        )
        out[tile_m, tile_n] = torch.where(in_window, acc, float("-inf"))
    return out


# %%
@helion.kernel(static_shapes=True)
def mqa_logits_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    ks: torch.Tensor,
    ke: torch.Tensor,
) -> torch.Tensor:
    """Indexer logits with the head axis folded into one batched matmul.

    Same arguments and result as :func:`mqa_logits`. Prefer this one when there
    are few query rows; see the module docstring.
    """
    m_dim, h_dim, d_dim = q.size()
    n_dim = k.size(0)
    h_dim = hl.specialize(h_dim)
    d_dim = hl.specialize(d_dim)
    out = torch.empty([m_dim, n_dim], dtype=torch.float32, device=q.device)
    for tile_m, tile_n in hl.tile([m_dim, n_dim]):
        kt = k[tile_n, :].transpose(0, 1)
        # [tile_m, H, D] @ [D, tile_n] -> [tile_m, H, tile_n]
        score = torch.matmul(q[tile_m, :, :], kt).to(torch.float32)
        acc = (torch.relu(score) * weights[tile_m, :][:, :, None]).sum(dim=1)
        in_window = (tile_n.index[None, :] >= ks[tile_m][:, None]) & (
            tile_n.index[None, :] < ke[tile_m][:, None]
        )
        out[tile_m, tile_n] = torch.where(in_window, acc, float("-inf"))
    return out


# %%
def ref_mqa_logits(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    ks: torch.Tensor,
    ke: torch.Tensor,
) -> torch.Tensor:
    """PyTorch reference. The einsum runs in the input dtype so that it and the
    kernels differ by the schedule alone, not by accumulation width."""
    pos = torch.arange(0, k.size(0), device=q.device)
    in_window = (pos[None, :] >= ks[:, None]) & (pos[None, :] < ke[:, None])
    score = torch.einsum("mhd,nd->hmn", q, k).float()
    logits = (score.relu() * weights.unsqueeze(-1).transpose(0, 1)).sum(dim=0)
    return logits.masked_fill(~in_window, float("-inf"))


# %%
def indexer_inputs(
    num_tokens: int,
    kv_len: int,
    num_heads: int = 32,
    head_dim: int = 128,
) -> tuple[torch.Tensor, ...]:
    """DeepSeek V3.2 indexer geometry by default (H=32, D=128)."""
    q = torch.randn(
        num_tokens, num_heads, head_dim, device=DEVICE, dtype=torch.bfloat16
    )
    k = torch.randn(kv_len, head_dim, device=DEVICE, dtype=torch.bfloat16)
    weights = torch.randn(num_tokens, num_heads, device=DEVICE, dtype=torch.float32)
    ks = torch.zeros(num_tokens, dtype=torch.int32, device=DEVICE)
    ke = torch.arange(num_tokens, dtype=torch.int32, device=DEVICE) + (
        kv_len - num_tokens
    )
    return q, k, weights, ks, ke


# %%
def check(num_tokens: int, kv_len: int) -> None:
    args = indexer_inputs(num_tokens, kv_len)
    run_example(
        {"helion": mqa_logits, "helion_decode": mqa_logits_decode},
        {
            "torch": ref_mqa_logits,
            "torch_compile": torch.compile(ref_mqa_logits, dynamic=False),
        },
        args,
        atol=1e-2,
        rtol=1e-2,
    )


# %%
def main() -> None:
    # decode: few query rows against a long KV window
    check(1, 4096)
    check(8, 16384)
    # prefill: the whole sequence scored at once
    check(512, 4096)
    check(4096, 8192)


if __name__ == "__main__":
    main()
