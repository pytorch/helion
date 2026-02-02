# SPDX-License-Identifier: Apache-2.0
# Standalone test for block table style chunked prefill attention
# Origin: vLLM chunked prefill paged decode (chunked_prefill_paged_decode.py)

import sys
from pathlib import Path

# Add attentions folder to path for package imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
import torch

from gqa.kernels.paged_prefill_blocktable import chunked_prefill_paged_decode_blocktable


# PyTorch reference implementation for prefill
def prefill_attention_reference(
    q: torch.Tensor,  # [total_tokens, num_heads, head_dim]
    k: torch.Tensor,  # [total_tokens, num_kv_heads, head_dim]
    v: torch.Tensor,  # [total_tokens, num_kv_heads, head_dim]
    query_start_loc: torch.Tensor,  # [num_seqs + 1]
    seq_lens: torch.Tensor,  # [num_seqs]
    is_causal: bool = True,
) -> torch.Tensor:
    """Reference implementation of prefill attention."""
    total_tokens, num_heads, head_dim = q.shape
    num_kv_heads = k.shape[1]
    kv_group_num = num_heads // num_kv_heads
    num_seqs = len(seq_lens)

    sm_scale = 1.0 / (head_dim ** 0.5)
    o = torch.zeros_like(q)

    for seq_idx in range(num_seqs):
        start = query_start_loc[seq_idx].item()
        end = query_start_loc[seq_idx + 1].item()
        seq_len = end - start

        if seq_len == 0:
            continue

        q_seq = q[start:end]  # [seq_len, num_heads, head_dim]
        k_seq = k[start:end]  # [seq_len, num_kv_heads, head_dim]
        v_seq = v[start:end]  # [seq_len, num_kv_heads, head_dim]

        for h in range(num_heads):
            kv_h = h // kv_group_num

            q_h = q_seq[:, h, :]  # [seq_len, head_dim]
            k_h = k_seq[:, kv_h, :]
            v_h = v_seq[:, kv_h, :]

            # [seq_len, seq_len]
            scores = torch.matmul(q_h.float(), k_h.float().T) * sm_scale

            if is_causal:
                mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device), diagonal=1).bool()
                scores = scores.masked_fill(mask, float('-inf'))

            attn = torch.softmax(scores, dim=-1)
            out_h = torch.matmul(attn, v_h.float())
            o[start:end, h, :] = out_h.to(q.dtype)

    return o


def create_5d_kv_cache(k_3d: torch.Tensor, block_size: int, x: int = 8):
    """
    Convert 3D K tensor to 5D cache layout.

    K layout: [total_tokens, num_kv_heads, head_dim]
    -> [num_blocks, num_kv_heads, head_dim//x, block_size, x]
    """
    total_tokens, num_kv_heads, head_dim = k_3d.shape
    num_blocks = (total_tokens + block_size - 1) // block_size

    # Pad to full blocks
    padded_tokens = num_blocks * block_size
    if padded_tokens > total_tokens:
        k_padded = torch.zeros(padded_tokens, num_kv_heads, head_dim, dtype=k_3d.dtype, device=k_3d.device)
        k_padded[:total_tokens] = k_3d
    else:
        k_padded = k_3d

    # Reshape to 5D: [num_blocks, block_size, num_kv_heads, head_dim]
    # -> [num_blocks, num_kv_heads, head_dim//x, block_size, x]
    k_4d = k_padded.view(num_blocks, block_size, num_kv_heads, head_dim)
    k_4d = k_4d.permute(0, 2, 3, 1)  # [num_blocks, num_kv_heads, head_dim, block_size]

    # Split head_dim into (head_dim//x, x)
    k_5d = k_4d.view(num_blocks, num_kv_heads, head_dim // x, x, block_size)
    k_5d = k_5d.permute(0, 1, 2, 4, 3)  # [num_blocks, num_kv_heads, head_dim//x, block_size, x]

    return k_5d.contiguous()


def create_4d_v_cache(v_3d: torch.Tensor, block_size: int):
    """
    Convert 3D V tensor to 4D cache layout.

    V layout: [total_tokens, num_kv_heads, head_dim]
    -> [num_blocks, num_kv_heads, head_dim, block_size]
    """
    total_tokens, num_kv_heads, head_dim = v_3d.shape
    num_blocks = (total_tokens + block_size - 1) // block_size

    # Pad to full blocks
    padded_tokens = num_blocks * block_size
    if padded_tokens > total_tokens:
        v_padded = torch.zeros(padded_tokens, num_kv_heads, head_dim, dtype=v_3d.dtype, device=v_3d.device)
        v_padded[:total_tokens] = v_3d
    else:
        v_padded = v_3d

    # Reshape: [num_blocks, block_size, num_kv_heads, head_dim]
    v_4d = v_padded.view(num_blocks, block_size, num_kv_heads, head_dim)
    # -> [num_blocks, num_kv_heads, head_dim, block_size]
    v_4d = v_4d.permute(0, 2, 3, 1)

    return v_4d.contiguous()


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("seq_len", [32, 64])
@pytest.mark.parametrize("num_heads", [8])
@pytest.mark.parametrize("num_kv_heads", [4, 8])
@pytest.mark.parametrize("head_dim", [64])
@pytest.mark.parametrize("dtype", [torch.float16])
@torch.inference_mode()
def test_chunked_prefill_blocktable(
    batch_size: int,
    seq_len: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
):
    """Test chunked prefill with block table style indexing."""
    if num_heads < num_kv_heads:
        pytest.skip("num_heads must be >= num_kv_heads")
    if num_heads % num_kv_heads != 0:
        pytest.skip("num_heads must be divisible by num_kv_heads")

    torch.manual_seed(42)

    block_size = 16
    x = 8  # Factor for vectorized K access (head_dim must be divisible by this)
    if head_dim % x != 0:
        pytest.skip(f"head_dim {head_dim} must be divisible by x={x}")

    total_tokens = batch_size * seq_len
    num_blocks_per_seq = (seq_len + block_size - 1) // block_size
    total_blocks = batch_size * num_blocks_per_seq

    # Dense tensors for prefill
    q = torch.randn(total_tokens, num_heads, head_dim, dtype=dtype, device="cuda")
    k = torch.randn(total_tokens, num_kv_heads, head_dim, dtype=dtype, device="cuda")
    v = torch.randn(total_tokens, num_kv_heads, head_dim, dtype=dtype, device="cuda")
    o = torch.zeros_like(q)

    # Create KV cache in block table format
    # K: [num_blocks, num_kv_heads, head_dim//x, block_size, x]
    # V: [num_blocks, num_kv_heads, head_dim, block_size]
    key_cache = create_5d_kv_cache(k, block_size, x)
    value_cache = create_4d_v_cache(v, block_size)

    # Block table: [num_seqs, max_num_blocks]
    block_table = torch.arange(total_blocks, dtype=torch.int32, device="cuda").view(batch_size, num_blocks_per_seq)

    # Sequence metadata
    query_start_loc = torch.arange(0, batch_size + 1, dtype=torch.int32, device="cuda") * seq_len
    seq_lens = torch.full((batch_size,), seq_len, dtype=torch.int32, device="cuda")

    # Run reference
    o_ref = prefill_attention_reference(q, k, v, query_start_loc, seq_lens, is_causal=True)

    # Run kernel
    chunked_prefill_paged_decode_blocktable(
        query=q,
        key=k,
        value=v,
        output=o,
        kv_cache_dtype="auto",
        key_cache=key_cache,
        value_cache=value_cache,
        block_table=block_table,
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        max_seq_len=seq_len,
        max_query_len=seq_len,
        k_scale=1.0,
        v_scale=1.0,
    )

    # Compare
    rtol = 1e-2 if dtype == torch.bfloat16 else 1e-3
    torch.testing.assert_close(o.float(), o_ref.float(), atol=1e-2, rtol=rtol)


@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize("num_heads", [8])
@pytest.mark.parametrize("head_dim", [64])
@torch.inference_mode()
def test_chunked_prefill_blocktable_variable_length(
    batch_size: int,
    num_heads: int,
    head_dim: int,
):
    """Test chunked prefill with variable-length sequences."""
    torch.manual_seed(42)
    dtype = torch.float16
    num_kv_heads = num_heads
    block_size = 16
    x = 8

    # Variable sequence lengths
    seq_lens_list = [32, 48]
    total_tokens = sum(seq_lens_list)
    max_seq_len = max(seq_lens_list)
    max_blocks = (max_seq_len + block_size - 1) // block_size

    # Calculate total blocks needed
    total_blocks = sum((s + block_size - 1) // block_size for s in seq_lens_list)

    # Dense tensors
    q = torch.randn(total_tokens, num_heads, head_dim, dtype=dtype, device="cuda")
    k = torch.randn(total_tokens, num_kv_heads, head_dim, dtype=dtype, device="cuda")
    v = torch.randn(total_tokens, num_kv_heads, head_dim, dtype=dtype, device="cuda")
    o = torch.zeros_like(q)

    # Create KV cache
    key_cache = create_5d_kv_cache(k, block_size, x)
    value_cache = create_4d_v_cache(v, block_size)

    # Block table with padding
    block_table = torch.zeros(batch_size, max_blocks, dtype=torch.int32, device="cuda")
    block_offset = 0
    for seq_idx, seq_len in enumerate(seq_lens_list):
        num_seq_blocks = (seq_len + block_size - 1) // block_size
        for blk_idx in range(num_seq_blocks):
            block_table[seq_idx, blk_idx] = block_offset + blk_idx
        block_offset += num_seq_blocks

    # Sequence metadata
    query_start_loc = torch.tensor([0, seq_lens_list[0], total_tokens], dtype=torch.int32, device="cuda")
    seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32, device="cuda")

    # Run reference
    o_ref = prefill_attention_reference(q, k, v, query_start_loc, seq_lens, is_causal=True)

    # Run kernel
    chunked_prefill_paged_decode_blocktable(
        query=q,
        key=k,
        value=v,
        output=o,
        kv_cache_dtype="auto",
        key_cache=key_cache,
        value_cache=value_cache,
        block_table=block_table,
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        max_seq_len=max_seq_len,
        max_query_len=max_seq_len,
        k_scale=1.0,
        v_scale=1.0,
    )

    # Compare
    torch.testing.assert_close(o.float(), o_ref.float(), atol=1e-2, rtol=1e-3)


@pytest.mark.parametrize("head_dim", [64, 128])
@torch.inference_mode()
def test_chunked_prefill_blocktable_gqa(head_dim: int):
    """Test GQA configuration with chunked prefill."""
    torch.manual_seed(42)
    dtype = torch.float16

    batch_size = 2
    seq_len = 64
    num_heads = 16
    num_kv_heads = 4  # 4:1 GQA ratio
    block_size = 16
    x = 8

    if head_dim % x != 0:
        pytest.skip(f"head_dim {head_dim} must be divisible by x={x}")

    total_tokens = batch_size * seq_len
    num_blocks_per_seq = (seq_len + block_size - 1) // block_size
    total_blocks = batch_size * num_blocks_per_seq

    # Dense tensors
    q = torch.randn(total_tokens, num_heads, head_dim, dtype=dtype, device="cuda")
    k = torch.randn(total_tokens, num_kv_heads, head_dim, dtype=dtype, device="cuda")
    v = torch.randn(total_tokens, num_kv_heads, head_dim, dtype=dtype, device="cuda")
    o = torch.zeros_like(q)

    # Create KV cache
    key_cache = create_5d_kv_cache(k, block_size, x)
    value_cache = create_4d_v_cache(v, block_size)

    # Block table
    block_table = torch.arange(total_blocks, dtype=torch.int32, device="cuda").view(batch_size, num_blocks_per_seq)

    # Sequence metadata
    query_start_loc = torch.arange(0, batch_size + 1, dtype=torch.int32, device="cuda") * seq_len
    seq_lens = torch.full((batch_size,), seq_len, dtype=torch.int32, device="cuda")

    # Run reference
    o_ref = prefill_attention_reference(q, k, v, query_start_loc, seq_lens, is_causal=True)

    # Run kernel
    chunked_prefill_paged_decode_blocktable(
        query=q,
        key=k,
        value=v,
        output=o,
        kv_cache_dtype="auto",
        key_cache=key_cache,
        value_cache=value_cache,
        block_table=block_table,
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        max_seq_len=seq_len,
        max_query_len=seq_len,
        k_scale=1.0,
        v_scale=1.0,
    )

    # Compare
    torch.testing.assert_close(o.float(), o_ref.float(), atol=1e-2, rtol=1e-3)


if __name__ == "__main__":
    # Run quick tests
    print("Testing chunked prefill with block table (MHA)...")
    test_chunked_prefill_blocktable(
        batch_size=2, seq_len=64, num_heads=8, num_kv_heads=8, head_dim=64, dtype=torch.float16
    )
    print("  MHA test passed!")

    print("Testing chunked prefill with block table (GQA)...")
    test_chunked_prefill_blocktable(
        batch_size=2, seq_len=64, num_heads=8, num_kv_heads=4, head_dim=64, dtype=torch.float16
    )
    print("  GQA test passed!")

    print("Testing chunked prefill with variable lengths...")
    test_chunked_prefill_blocktable_variable_length(
        batch_size=2, num_heads=8, head_dim=64
    )
    print("  Variable length test passed!")

    print("\nAll chunked prefill block table tests passed!")
