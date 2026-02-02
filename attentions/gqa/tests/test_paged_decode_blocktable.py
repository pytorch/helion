# SPDX-License-Identifier: Apache-2.0
# Standalone test for block table style paged decode attention
# Origin: vLLM unified attention (triton_unified_attention.py)

import sys
from pathlib import Path

# Add kernels folder to path for standalone execution
sys.path.insert(0, str(Path(__file__).parent.parent / "kernels"))

import pytest
import torch

from paged_decode_blocktable import unified_attention_blocktable


# PyTorch reference implementation
def paged_decode_reference(
    q: torch.Tensor,  # [total_tokens, num_heads, head_dim]
    k_cache: torch.Tensor,  # [num_blocks, block_size, num_kv_heads, head_dim]
    v_cache: torch.Tensor,  # [num_blocks, block_size, num_kv_heads, head_dim]
    cu_seqlens_q: torch.Tensor,  # [num_seqs + 1]
    seqused_k: torch.Tensor,  # [num_seqs]
    block_table: torch.Tensor,  # [num_seqs, max_num_blocks]
    softmax_scale: float,
    causal: bool = True,
) -> torch.Tensor:
    """Reference implementation of paged decode attention with block table."""
    total_tokens, num_heads, head_dim = q.shape
    num_kv_heads = k_cache.shape[2]
    block_size = k_cache.shape[1]
    kv_group_num = num_heads // num_kv_heads

    o = torch.zeros_like(q)
    num_seqs = len(seqused_k)

    for seq_idx in range(num_seqs):
        q_start = cu_seqlens_q[seq_idx].item()
        q_end = cu_seqlens_q[seq_idx + 1].item()
        q_len = q_end - q_start
        kv_len = seqused_k[seq_idx].item()

        if q_len == 0 or kv_len == 0:
            continue

        # Gather K and V from paged cache
        num_blocks = (kv_len + block_size - 1) // block_size
        k_gathered = []
        v_gathered = []

        for blk_idx in range(num_blocks):
            physical_block = block_table[seq_idx, blk_idx].item()
            # Determine how many tokens in this block
            tokens_in_block = min(block_size, kv_len - blk_idx * block_size)

            k_block = k_cache[physical_block, :tokens_in_block]  # [tokens, num_kv_heads, head_dim]
            v_block = v_cache[physical_block, :tokens_in_block]
            k_gathered.append(k_block)
            v_gathered.append(v_block)

        k_seq = torch.cat(k_gathered, dim=0)  # [kv_len, num_kv_heads, head_dim]
        v_seq = torch.cat(v_gathered, dim=0)

        # For each query position in this sequence
        for q_pos in range(q_len):
            global_q_pos = q_start + q_pos
            q_h = q[global_q_pos]  # [num_heads, head_dim]

            for h in range(num_heads):
                kv_h = h // kv_group_num

                # Get query for this head
                q_vec = q_h[h, :]  # [head_dim]

                # Get KV for this head
                k_h = k_seq[:, kv_h, :]  # [kv_len, head_dim]
                v_h = v_seq[:, kv_h, :]

                # Compute attention scores
                scores = torch.matmul(q_vec.float().unsqueeze(0), k_h.float().T) * softmax_scale

                # Apply causal mask if needed
                if causal:
                    # Query position relative to start of this sequence
                    # In decode, query position is typically at the end
                    # Mask out future positions
                    kv_positions = torch.arange(kv_len, device=q.device)
                    # For decode, the query is at position (kv_len - q_len + q_pos)
                    current_pos = kv_len - q_len + q_pos
                    mask = kv_positions <= current_pos
                    scores = scores.masked_fill(~mask.unsqueeze(0), float('-inf'))

                # Softmax
                attn = torch.softmax(scores, dim=-1)

                # Output
                out_h = torch.matmul(attn, v_h.float())
                o[global_q_pos, h, :] = out_h.to(q.dtype).squeeze(0)

    return o


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("seq_len", [32, 64])
@pytest.mark.parametrize("num_heads", [8, 16])
@pytest.mark.parametrize("num_kv_heads", [8, 4])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_paged_decode_blocktable(
    batch_size: int,
    seq_len: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
):
    """Test block table style paged decode attention."""
    if num_heads < num_kv_heads:
        pytest.skip("num_heads must be >= num_kv_heads")
    if num_heads % num_kv_heads != 0:
        pytest.skip("num_heads must be divisible by num_kv_heads")

    torch.manual_seed(42)

    block_size = 16
    num_blocks_per_seq = (seq_len + block_size - 1) // block_size
    total_blocks = batch_size * num_blocks_per_seq

    # Query: [total_tokens, num_heads, head_dim] - for decode, 1 token per seq
    total_tokens = batch_size
    q = torch.randn(total_tokens, num_heads, head_dim, dtype=dtype, device="cuda")
    o = torch.zeros_like(q)

    # KV cache: [num_blocks, block_size, num_kv_heads, head_dim]
    k_cache = torch.randn(total_blocks, block_size, num_kv_heads, head_dim, dtype=dtype, device="cuda")
    v_cache = torch.randn(total_blocks, block_size, num_kv_heads, head_dim, dtype=dtype, device="cuda")

    # Block table: [num_seqs, max_num_blocks]
    block_table = torch.arange(total_blocks, dtype=torch.int32, device="cuda").view(batch_size, num_blocks_per_seq)

    # Sequence metadata
    cu_seqlens_q = torch.arange(0, batch_size + 1, dtype=torch.int32, device="cuda")
    seqused_k = torch.full((batch_size,), seq_len, dtype=torch.int32, device="cuda")

    # Run reference
    softmax_scale = 1.0 / (head_dim ** 0.5)
    o_ref = paged_decode_reference(
        q, k_cache, v_cache, cu_seqlens_q, seqused_k, block_table, softmax_scale, causal=True
    )

    # Run kernel
    unified_attention_blocktable(
        q=q,
        k=k_cache,
        v=v_cache,
        out=o,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=1,  # Decode: 1 token per seq
        seqused_k=seqused_k,
        max_seqlen_k=seq_len,
        softmax_scale=softmax_scale,
        causal=True,
        window_size=(-1, -1),  # No sliding window
        block_table=block_table,
        softcap=0.0,
    )

    # Compare
    rtol = 1e-2 if dtype == torch.bfloat16 else 1e-3
    torch.testing.assert_close(o.float(), o_ref.float(), atol=1e-2, rtol=rtol)


@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize("num_heads", [8])
@pytest.mark.parametrize("num_kv_heads", [4, 8])
@pytest.mark.parametrize("head_dim", [64])
@torch.inference_mode()
def test_paged_decode_blocktable_variable_length(
    batch_size: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
):
    """Test block table decode with variable-length KV sequences."""
    if num_heads % num_kv_heads != 0:
        pytest.skip("num_heads must be divisible by num_kv_heads")

    torch.manual_seed(42)
    dtype = torch.float16

    block_size = 16
    seq_lens = [32, 64]  # Variable KV lengths
    max_seq_len = max(seq_lens)
    max_blocks = (max_seq_len + block_size - 1) // block_size

    # Total blocks needed
    total_blocks = sum((s + block_size - 1) // block_size for s in seq_lens)

    # Query: 1 token per sequence (decode)
    total_tokens = batch_size
    q = torch.randn(total_tokens, num_heads, head_dim, dtype=dtype, device="cuda")
    o = torch.zeros_like(q)

    # KV cache
    k_cache = torch.randn(total_blocks, block_size, num_kv_heads, head_dim, dtype=dtype, device="cuda")
    v_cache = torch.randn(total_blocks, block_size, num_kv_heads, head_dim, dtype=dtype, device="cuda")

    # Block table with padding
    block_table = torch.zeros(batch_size, max_blocks, dtype=torch.int32, device="cuda")
    block_offset = 0
    for seq_idx, seq_len in enumerate(seq_lens):
        num_blocks = (seq_len + block_size - 1) // block_size
        for blk_idx in range(num_blocks):
            block_table[seq_idx, blk_idx] = block_offset + blk_idx
        block_offset += num_blocks

    # Sequence metadata
    cu_seqlens_q = torch.arange(0, batch_size + 1, dtype=torch.int32, device="cuda")
    seqused_k = torch.tensor(seq_lens, dtype=torch.int32, device="cuda")

    # Run reference
    softmax_scale = 1.0 / (head_dim ** 0.5)
    o_ref = paged_decode_reference(
        q, k_cache, v_cache, cu_seqlens_q, seqused_k, block_table, softmax_scale, causal=True
    )

    # Run kernel
    unified_attention_blocktable(
        q=q,
        k=k_cache,
        v=v_cache,
        out=o,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=1,
        seqused_k=seqused_k,
        max_seqlen_k=max_seq_len,
        softmax_scale=softmax_scale,
        causal=True,
        window_size=(-1, -1),
        block_table=block_table,
        softcap=0.0,
    )

    # Compare
    torch.testing.assert_close(o.float(), o_ref.float(), atol=1e-2, rtol=1e-3)


@pytest.mark.parametrize("head_dim", [64, 128])
@torch.inference_mode()
def test_paged_decode_blocktable_gqa(head_dim: int):
    """Test GQA configuration (multiple query heads per KV head)."""
    torch.manual_seed(42)
    dtype = torch.float16

    batch_size = 2
    seq_len = 64
    num_heads = 16
    num_kv_heads = 4  # 4:1 GQA ratio
    block_size = 16

    num_blocks_per_seq = (seq_len + block_size - 1) // block_size
    total_blocks = batch_size * num_blocks_per_seq

    # Query
    total_tokens = batch_size
    q = torch.randn(total_tokens, num_heads, head_dim, dtype=dtype, device="cuda")
    o = torch.zeros_like(q)

    # KV cache
    k_cache = torch.randn(total_blocks, block_size, num_kv_heads, head_dim, dtype=dtype, device="cuda")
    v_cache = torch.randn(total_blocks, block_size, num_kv_heads, head_dim, dtype=dtype, device="cuda")

    # Block table
    block_table = torch.arange(total_blocks, dtype=torch.int32, device="cuda").view(batch_size, num_blocks_per_seq)

    # Sequence metadata
    cu_seqlens_q = torch.arange(0, batch_size + 1, dtype=torch.int32, device="cuda")
    seqused_k = torch.full((batch_size,), seq_len, dtype=torch.int32, device="cuda")

    # Run reference
    softmax_scale = 1.0 / (head_dim ** 0.5)
    o_ref = paged_decode_reference(
        q, k_cache, v_cache, cu_seqlens_q, seqused_k, block_table, softmax_scale, causal=True
    )

    # Run kernel
    unified_attention_blocktable(
        q=q,
        k=k_cache,
        v=v_cache,
        out=o,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=1,
        seqused_k=seqused_k,
        max_seqlen_k=seq_len,
        softmax_scale=softmax_scale,
        causal=True,
        window_size=(-1, -1),
        block_table=block_table,
        softcap=0.0,
    )

    # Compare
    torch.testing.assert_close(o.float(), o_ref.float(), atol=1e-2, rtol=1e-3)


if __name__ == "__main__":
    # Run quick tests
    print("Testing block table paged decode attention (MHA)...")
    test_paged_decode_blocktable(
        batch_size=2, seq_len=64, num_heads=8, num_kv_heads=8, head_dim=64, dtype=torch.float16
    )
    print("  MHA test passed!")

    print("Testing block table paged decode attention (GQA)...")
    test_paged_decode_blocktable(
        batch_size=2, seq_len=64, num_heads=16, num_kv_heads=4, head_dim=64, dtype=torch.float16
    )
    print("  GQA test passed!")

    print("Testing block table paged decode with variable lengths...")
    test_paged_decode_blocktable_variable_length(
        batch_size=2, num_heads=8, num_kv_heads=4, head_dim=64
    )
    print("  Variable length test passed!")

    print("\nAll block table paged decode attention tests passed!")
