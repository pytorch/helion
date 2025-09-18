"""
Decoding Attention Example
==========================

Helion implementation of decoding attention with KV cache support, mirroring the
behavior of TritonBench's xFormers ``triton_splitk`` kernel. The kernel handles
grouped-query attention (GQA), variable cache lengths per sequence, and performs
an online softmax to keep numerical stability when streaming through long KV
caches.
"""

from __future__ import annotations

import math
from typing import Callable

import torch

import helion
import helion.language as hl


_LOG2E = 1.4426950408889634  # 1 / log(2)


@helion.kernel(static_shapes=False)
def decoding_attention_kernel(
    q: torch.Tensor,  # [batch, seq_len_q, num_q_heads, head_dim]
    k_cache: torch.Tensor,  # [batch, max_seq_len_kv, num_kv_heads, head_dim]
    v_cache: torch.Tensor,  # [batch, max_seq_len_kv, num_kv_heads, head_dim]
    cache_seqlens: torch.Tensor,  # [batch]
) -> torch.Tensor:
    """Compute decoding attention against a KV cache."""

    batch, seq_len_q, num_q_heads, head_dim = q.shape
    head_dim = hl.specialize(head_dim)
    max_seq_len_kv = k_cache.size(1)
    num_kv_heads = k_cache.size(2)

    # Collapse batch and head dimensions to simplify iteration order.
    q_flat = (
        q.permute(0, 2, 1, 3)
        .contiguous()
        .view(batch * num_q_heads, seq_len_q, head_dim)
    )
    out_flat = torch.empty_like(q_flat)

    total_heads = q_flat.size(0)

    max_seq_len_tensor = torch.tensor(max_seq_len_kv, device=q.device, dtype=torch.int32)

    for tile_bh, tile_m in hl.tile([total_heads, seq_len_q], block_size=[1, None]):
        head_idx = tile_bh.begin
        b_idx = head_idx // num_q_heads
        local_head = head_idx % num_q_heads
        kv_idx = local_head * num_kv_heads // num_q_heads

        q_rows = q_flat[tile_bh, tile_m, :]
        q_rows_fp32 = q_rows.to(torch.float32)

        k_rows = k_cache[b_idx, :, kv_idx, :]
        v_rows = v_cache[b_idx, :, kv_idx, :]

        kv_len = torch.clamp(cache_seqlens[b_idx], min=0).to(torch.int32)
        kv_len = torch.minimum(kv_len, max_seq_len_tensor)

        m_i = hl.full([tile_m], float("-inf"), dtype=torch.float32)
        l_i = hl.full([tile_m], 0.0, dtype=torch.float32)
        acc = hl.zeros([tile_m, head_dim], dtype=torch.float32)

        for tile_n in hl.tile(max_seq_len_kv, block_size=128):
            valid_cols = tile_n.index[None, :] < kv_len[:, None]
            if torch.all(~valid_cols):
                continue

            k_tile = k_rows[tile_n, :].to(torch.float32)
            v_tile = v_rows[tile_n, :].to(torch.float32)

            qk = torch.matmul(q_rows_fp32, k_tile.transpose(0, 1))
            sm_scale = (1.0 / math.sqrt(float(head_dim))) * _LOG2E
            qk = qk * sm_scale
            qk = torch.where(valid_cols, qk, torch.full_like(qk, float("-inf")))

            block_max = torch.amax(qk, dim=-1)
            m_new = torch.maximum(m_i, block_max)

            shifted = qk - m_new[:, None]
            p = torch.where(
                valid_cols,
                torch.exp2(shifted),
                torch.zeros_like(shifted),
            )
            l_ij = torch.sum(p, dim=-1)
            alpha = torch.where(
                torch.isfinite(m_i),
                torch.exp2(m_i - m_new),
                torch.zeros_like(m_i),
            )
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, None]
            acc = acc + torch.matmul(p, v_tile)
            m_i = torch.where(torch.isfinite(m_new), m_new, m_i)

        safe_l = torch.where(l_i > 0, l_i, torch.ones_like(l_i))
        out_rows = torch.where(
            l_i[:, None] > 0,
            acc / safe_l[:, None],
            torch.zeros_like(acc),
        )
        out_flat[tile_bh, tile_m, :] = out_rows.to(out_flat.dtype)

    return out_flat.view(batch, num_q_heads, seq_len_q, head_dim).permute(0, 2, 1, 3)


def decoding_attention_tritonbench(
    tb_op: object,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cache_seqlens: torch.Tensor,
) -> Callable[[], torch.Tensor]:
    """TritonBench wrapper that prepares inputs and invokes the Helion kernel."""
    q = q.contiguous()
    k_cache = k_cache.contiguous()
    v_cache = v_cache.contiguous()
    cache_seqlens = cache_seqlens.contiguous().to(torch.int32)
    return lambda: decoding_attention_kernel(q, k_cache, v_cache, cache_seqlens)


def _reference_decoding_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cache_seqlens: torch.Tensor,
) -> torch.Tensor:
    """Baseline decoder attention using plain PyTorch ops."""
    batch, seq_len_q, num_q_heads, head_dim = q.shape
    max_seq_len_kv = k_cache.size(1)
    num_kv_heads = k_cache.size(2)

    out = torch.zeros_like(q)
    scale = 1.0 / math.sqrt(float(head_dim))

    for b in range(batch):
        kv_len = int(cache_seqlens[b].item())
        kv_len = max(0, min(max_seq_len_kv, kv_len))
        for h in range(num_q_heads):
            kv_idx = min((h * num_kv_heads) // num_q_heads, num_kv_heads - 1)
            q_vec = q[b, :, h, :].to(torch.float32)
            if kv_len == 0:
                out[b, :, h, :] = 0
                continue
            k_slice = k_cache[b, :kv_len, kv_idx, :].to(torch.float32)
            v_slice = v_cache[b, :kv_len, kv_idx, :].to(torch.float32)
            scores = torch.matmul(q_vec, k_slice.transpose(0, 1)) * scale
            probs = torch.softmax(scores, dim=-1)
            out[b, :, h, :] = torch.matmul(probs, v_slice).to(out.dtype)
    return out


def main() -> None:
    """Run a quick correctness check against the reference implementation."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    batch, seq_len_q = 2, 1
    num_q_heads, num_kv_heads, head_dim = 8, 2, 128
    max_seq_len_kv = 256

    q = torch.randn(batch, seq_len_q, num_q_heads, head_dim, device=device, dtype=dtype)
    k_cache = torch.randn(
        batch, max_seq_len_kv, num_kv_heads, head_dim, device=device, dtype=dtype
    )
    v_cache = torch.randn_like(k_cache)
    cache_seqlens = torch.randint(
        1, max_seq_len_kv + 1, (batch,), device=device, dtype=torch.int32
    )

    ref = _reference_decoding_attention(q, k_cache, v_cache, cache_seqlens)
    hl_out = decoding_attention_kernel(q, k_cache, v_cache, cache_seqlens)

    torch.testing.assert_close(ref, hl_out, atol=1e-2, rtol=1e-2)
    print("Helion decoding attention matches reference output.")


if __name__ == "__main__":
    main()
