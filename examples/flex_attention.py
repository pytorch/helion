"""
Flex Attention Sketch
=====================

This module sketches a Helion implementation of the FlexAttention kernel. The goal
is to mirror Triton's structure: iterate over sparsity metadata, invoke score/mask
modifiers inside the device loop, and maintain the numerically-stable online softmax.

The code focuses on shaping the pattern. Some pieces may still require polish
before compiling with Helion, but the layout reflects how we plan to wire
everything together.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

import helion
from helion._testing import run_example
import helion.language as hl

from torch.nn.attention.flex_attention import (
    BlockMask,
    create_block_mask,
    flex_attention as torch_flex_attention,
)


ScoreMod = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
MaskMod = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


def _flatten_bh(x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    """Flatten batch and head dimensions to simplify grid traversal."""
    B, H, *rest = x.shape
    flat = x.contiguous().view(B * H, *rest)
    return flat, B, H


def _kv_head_index(batch: int, head: int, batch_count: int, kv_heads: int) -> int:
    """Map (batch, head) to flattened KV head index, handling broadcast batches."""
    return batch * kv_heads + head if batch_count > 1 else head


@helion.kernel(static_shapes=True)
def flex_attention_kernel(
    q: torch.Tensor,  # [B, Hq, M, D]
    k: torch.Tensor,  # [B or 1, Hkv, N, D]
    v: torch.Tensor,  # [B or 1, Hkv, N, Dv]
    kv_num_blocks: torch.Tensor,
    kv_indices: torch.Tensor,
    full_kv_num_blocks: torch.Tensor | None,
    full_kv_indices: torch.Tensor | None,
    block_size_q: int,
    block_size_kv: int,
    score_mod: ScoreMod | None,
    mask_mod: MaskMod | None,
    scale: float,
) -> torch.Tensor:
    """Helion FlexAttention sketch with score/mask modifiers inside the device loop."""
    B, Hq, M, Dh = q.shape
    Bkv, Hkv, N, Dv = v.shape
    q_flat, _, _ = _flatten_bh(q)
    k_flat = k.reshape(Bkv * Hkv, N, Dh)
    v_flat = v.reshape(Bkv * Hkv, N, Dv)

    out = torch.empty_like(q)
    rcp_ln2 = 1.44269504  # 1 / log(2), matches Triton kernel
    sm_scale = scale * rcp_ln2

    for bh, tile_m in hl.tile([B * Hq, M], block_size=[1, None]):
        q_tile = q_flat[bh, tile_m, :].to(torch.float32)
        tile_len = q_tile.shape[0]
        acc = hl.zeros([tile_len, Dv], dtype=torch.float32)
        m_i = hl.full([tile_len], float("-inf"), dtype=torch.float32)
        l_i = hl.ones([tile_len], dtype=torch.float32)

        batch_idx = int(bh // Hq)
        head_idx = int(bh % Hq)
        kv_batch_idx = 0 if Bkv == 1 else batch_idx
        kv_head_index = _kv_head_index(kv_batch_idx, head_idx, Bkv, Hkv)

        q_block_id = tile_m.begin // block_size_q

        def _process_blocks(
            block_counts: torch.Tensor | None,
            block_indices: torch.Tensor | None,
            apply_mask: bool,
        ) -> None:
            if block_counts is None or block_indices is None:
                return
            counts_row = block_counts[batch_idx, head_idx, q_block_id]
            indices_row = block_indices[batch_idx, head_idx, q_block_id]
            num_blocks = int(counts_row.item())  # pyright: ignore[reportAttributeAccessIssue]
            if num_blocks == 0:
                return
            for blk in range(num_blocks):
                kv_block_start = int(indices_row[blk].item()) * block_size_kv
                kv_indices_block = kv_block_start + torch.arange(
                    block_size_kv, device=q.device, dtype=torch.int64
                )
                k_block = k_flat[kv_head_index, kv_indices_block, :].to(torch.float32)
                v_block = v_flat[kv_head_index, kv_indices_block, :].to(torch.float32)
                logits = torch.matmul(q_tile, k_block.transpose(0, 1))
                logits = logits * sm_scale

                if score_mod is not None:
                    batch_ids = torch.full_like(logits, batch_idx, dtype=torch.int32)
                    head_ids = torch.full_like(logits, head_idx, dtype=torch.int32)
                    q_ids = (tile_m.begin + tile_m.index).to(torch.int32)
                    q_ids = q_ids.view(tile_len, 1).expand_as(logits)
                    kv_ids = kv_indices_block.to(torch.int32)
                    kv_ids = kv_ids.view(1, block_size_kv).expand_as(logits)
                    logits = score_mod(logits, batch_ids, head_ids, q_ids, kv_ids)

                if apply_mask and mask_mod is not None:
                    batch_vals = torch.full_like(logits, batch_idx, dtype=torch.int32)
                    head_vals = torch.full_like(logits, head_idx, dtype=torch.int32)
                    q_vals = (tile_m.begin + tile_m.index).to(torch.int32)
                    q_vals = q_vals.view(tile_len, 1).expand_as(logits)
                    kv_vals = kv_indices_block.to(torch.int32)
                    kv_vals = kv_vals.view(1, block_size_kv).expand_as(logits)
                    mask = mask_mod(batch_vals, head_vals, q_vals, kv_vals)
                    logits = torch.where(mask, logits, float("-inf"))

                m_new = torch.maximum(m_i, torch.amax(logits, dim=1))
                logits = logits - m_new[:, None]
                prob = torch.exp2(logits)
                alpha = torch.exp2(m_i - m_new)
                l_i = l_i * alpha + torch.sum(prob, dim=1)
                acc = acc * alpha[:, None]
                acc = acc + torch.matmul(prob, v_block)
                m_i = m_new

        _process_blocks(kv_num_blocks, kv_indices, apply_mask=True)
        _process_blocks(full_kv_num_blocks, full_kv_indices, apply_mask=False)

        l_i = torch.where(l_i == 0, torch.ones_like(l_i), l_i)
        acc = acc / l_i[:, None]
        out[batch_idx, head_idx, tile_m, :] = acc.to(out.dtype)

    return out


def _unpack_block_mask(block_mask: BlockMask) -> dict[str, Any]:
    """Extract the pieces needed by the kernel."""
    return {
        "kv_num_blocks": block_mask.kv_num_blocks,
        "kv_indices": block_mask.kv_indices,
        "full_kv_num_blocks": block_mask.full_kv_num_blocks,
        "full_kv_indices": block_mask.full_kv_indices,
        "block_q": block_mask.BLOCK_SIZE[0],
        "block_kv": block_mask.BLOCK_SIZE[1],
    }


def _causal_mask(
    batch: torch.Tensor,
    head: torch.Tensor,
    q_idx: torch.Tensor,
    kv_idx: torch.Tensor,
) -> torch.Tensor:
    """Simple causal mask helper."""
    return q_idx >= kv_idx


def check(
    batch_size: int,
    num_heads: int,
    query_len: int,
    kv_len: int,
    head_dim: int,
    value_dim: int | None = None,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device | str = "cuda",
) -> None:
    """Verify sketch against torch.nn.flex_attention using a causal mask."""
    device = torch.device(device)
    value_dim = value_dim or head_dim

    q = torch.randn(
        (batch_size, num_heads, query_len, head_dim),
        dtype=dtype,
        device=device,
    )
    k = torch.randn(
        (batch_size, num_heads, kv_len, head_dim),
        dtype=dtype,
        device=device,
    )
    v = torch.randn(
        (batch_size, num_heads, kv_len, value_dim),
        dtype=dtype,
        device=device,
    )

    block_mask = create_block_mask(
        _causal_mask, batch_size, num_heads, query_len, kv_len, device=device
    )
    mask_parts = _unpack_block_mask(block_mask)

    scale = 1.0 / float(head_dim) ** 0.5

    def helion_impl(q_in: torch.Tensor, k_in: torch.Tensor, v_in: torch.Tensor) -> torch.Tensor:
        return flex_attention_kernel(
            q_in,
            k_in,
            v_in,
            mask_parts["kv_num_blocks"],
            mask_parts["kv_indices"],
            mask_parts["full_kv_num_blocks"],
            mask_parts["full_kv_indices"],
            mask_parts["block_q"],
            mask_parts["block_kv"],
            None,
            _causal_mask,
            scale,
        )

    def baseline_impl(q_in: torch.Tensor, k_in: torch.Tensor, v_in: torch.Tensor) -> torch.Tensor:
        return torch_flex_attention(q_in, k_in, v_in, block_mask=block_mask)

    run_example(
        {"helion": helion_impl},
        {"baseline": baseline_impl},
        (q, k, v),
        atol=1e-2,
        rtol=1e-2,
    )


def main() -> None:
    """Ad-hoc driver for local experimentation."""
    check(batch_size=2, num_heads=4, query_len=128, kv_len=128, head_dim=64)


if __name__ == "__main__":
    main()
