# Copyright 2023-2024 SGLang Team
# Standalone test for CSR-style paged decode attention
# Origin: sglang_attention/kernels/decode_attention.py

import sys
from pathlib import Path

# Add kernels folder to path for standalone execution
sys.path.insert(0, str(Path(__file__).parent.parent / "kernels"))

import pytest
import torch

from paged_decode_csr import decode_attention_fwd


# PyTorch reference implementation
def decode_attention_reference(
    q: torch.Tensor,  # [batch_size, num_heads, head_dim]
    k_buffer: torch.Tensor,  # [total_kv_tokens, num_kv_heads, head_dim]
    v_buffer: torch.Tensor,  # [total_kv_tokens, num_kv_heads, head_dim]
    kv_indptr: torch.Tensor,  # [batch_size + 1]
    kv_indices: torch.Tensor,  # [total_kv_tokens]
) -> torch.Tensor:
    """Reference implementation of paged decode attention with CSR indexing."""
    batch_size, num_heads, head_dim = q.shape
    num_kv_heads = k_buffer.shape[1]
    kv_group_num = num_heads // num_kv_heads

    sm_scale = 1.0 / (head_dim ** 0.5)
    o = torch.zeros_like(q)

    for b in range(batch_size):
        start = kv_indptr[b].item()
        end = kv_indptr[b + 1].item()
        seq_len = end - start

        if seq_len == 0:
            continue

        # Get KV indices for this batch
        kv_idx = kv_indices[start:end]

        # Extract K and V values using the indices
        k_b = k_buffer[kv_idx]  # [seq_len, num_kv_heads, head_dim]
        v_b = v_buffer[kv_idx]  # [seq_len, num_kv_heads, head_dim]

        for h in range(num_heads):
            kv_h = h // kv_group_num

            # Get query for this head
            q_h = q[b, h, :]  # [head_dim]

            # Get KV for this head
            k_h = k_b[:, kv_h, :]  # [seq_len, head_dim]
            v_h = v_b[:, kv_h, :]  # [seq_len, head_dim]

            # Compute attention scores [1, seq_len]
            scores = torch.matmul(q_h.float().unsqueeze(0), k_h.float().T) * sm_scale

            # Softmax
            attn = torch.softmax(scores, dim=-1)

            # Output [1, head_dim]
            out_h = torch.matmul(attn, v_h.float())
            o[b, h, :] = out_h.to(q.dtype).squeeze(0)

    return o


def _compute_num_kv_splits(seq_lens: torch.Tensor, max_seq_len: int) -> tuple:
    """Compute number of KV splits for each sequence."""
    # Simple heuristic: use more splits for longer sequences
    # In practice, this depends on block sizes and hardware
    min_block_kv = 32
    max_kv_splits = min(16, (max_seq_len + min_block_kv - 1) // min_block_kv)
    max_kv_splits = max(max_kv_splits, 1)

    num_kv_splits = torch.ones(len(seq_lens), dtype=torch.int32, device=seq_lens.device)
    for i, seq_len in enumerate(seq_lens.tolist()):
        # More splits for longer sequences
        num_kv_splits[i] = min(max_kv_splits, (seq_len + min_block_kv - 1) // min_block_kv)
        num_kv_splits[i] = max(num_kv_splits[i], 1)

    return num_kv_splits, max_kv_splits


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("seq_len", [32, 64, 128])
@pytest.mark.parametrize("num_heads", [8, 16])
@pytest.mark.parametrize("num_kv_heads", [8, 4])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_paged_decode_csr(
    batch_size: int,
    seq_len: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
):
    """Test CSR-style paged decode attention against PyTorch reference."""
    if num_heads < num_kv_heads:
        pytest.skip("num_heads must be >= num_kv_heads")
    if num_heads % num_kv_heads != 0:
        pytest.skip("num_heads must be divisible by num_kv_heads")

    torch.manual_seed(42)

    total_kv_tokens = batch_size * seq_len

    # Create query tensor [batch_size, num_heads, head_dim]
    q = torch.randn(batch_size, num_heads, head_dim, dtype=dtype, device="cuda")

    # Create KV buffer [total_kv_tokens, num_kv_heads, head_dim]
    k_buffer = torch.randn(total_kv_tokens, num_kv_heads, head_dim, dtype=dtype, device="cuda")
    v_buffer = torch.randn(total_kv_tokens, num_kv_heads, head_dim, dtype=dtype, device="cuda")

    # Create output tensor
    o = torch.zeros(batch_size, num_heads, head_dim, dtype=dtype, device="cuda")

    # CSR-style indexing - assume equal length sequences
    kv_indptr = torch.arange(0, batch_size + 1, dtype=torch.int32, device="cuda") * seq_len
    # Simple case: indices are just sequential
    kv_indices = torch.arange(0, total_kv_tokens, dtype=torch.int32, device="cuda")

    # Compute split parameters
    seq_lens = torch.full((batch_size,), seq_len, device="cuda")
    num_kv_splits, max_kv_splits = _compute_num_kv_splits(seq_lens, seq_len)

    # Allocate intermediate buffers for two-stage flash decoding
    attn_logits = torch.zeros(
        batch_size, num_heads, max_kv_splits, head_dim,
        dtype=torch.float32, device="cuda"
    )
    attn_lse = torch.zeros(
        batch_size, num_heads, max_kv_splits,
        dtype=torch.float32, device="cuda"
    )

    # Run reference
    o_ref = decode_attention_reference(q, k_buffer, v_buffer, kv_indptr, kv_indices)

    # Run CSR-style paged decode kernel
    sm_scale = 1.0 / (head_dim ** 0.5)
    decode_attention_fwd(
        q, k_buffer, v_buffer, o,
        kv_indptr, kv_indices,
        attn_logits, attn_lse,
        num_kv_splits, max_kv_splits,
        sm_scale,
    )

    # Compare
    rtol = 1e-2 if dtype == torch.bfloat16 else 1e-3
    torch.testing.assert_close(o.float(), o_ref.float(), atol=1e-2, rtol=rtol)


@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize("num_heads", [8])
@pytest.mark.parametrize("num_kv_heads", [4, 8])
@pytest.mark.parametrize("head_dim", [64])
@torch.inference_mode()
def test_paged_decode_csr_variable_length(
    batch_size: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
):
    """Test paged decode with variable-length sequences."""
    if num_heads % num_kv_heads != 0:
        pytest.skip("num_heads must be divisible by num_kv_heads")

    torch.manual_seed(42)
    dtype = torch.float16

    # Variable sequence lengths
    seq_lens = [32, 64]
    total_kv_tokens = sum(seq_lens)

    # Create tensors
    q = torch.randn(batch_size, num_heads, head_dim, dtype=dtype, device="cuda")
    k_buffer = torch.randn(total_kv_tokens, num_kv_heads, head_dim, dtype=dtype, device="cuda")
    v_buffer = torch.randn(total_kv_tokens, num_kv_heads, head_dim, dtype=dtype, device="cuda")
    o = torch.zeros(batch_size, num_heads, head_dim, dtype=dtype, device="cuda")

    # CSR-style indexing for variable lengths
    kv_indptr = torch.tensor([0, seq_lens[0], total_kv_tokens], dtype=torch.int32, device="cuda")
    kv_indices = torch.arange(0, total_kv_tokens, dtype=torch.int32, device="cuda")

    # Compute split parameters
    seq_lens_tensor = torch.tensor(seq_lens, device="cuda")
    num_kv_splits, max_kv_splits = _compute_num_kv_splits(seq_lens_tensor, max(seq_lens))

    attn_logits = torch.zeros(
        batch_size, num_heads, max_kv_splits, head_dim,
        dtype=torch.float32, device="cuda"
    )
    attn_lse = torch.zeros(
        batch_size, num_heads, max_kv_splits,
        dtype=torch.float32, device="cuda"
    )

    # Run reference
    o_ref = decode_attention_reference(q, k_buffer, v_buffer, kv_indptr, kv_indices)

    # Run kernel
    sm_scale = 1.0 / (head_dim ** 0.5)
    decode_attention_fwd(
        q, k_buffer, v_buffer, o,
        kv_indptr, kv_indices,
        attn_logits, attn_lse,
        num_kv_splits, max_kv_splits,
        sm_scale,
    )

    # Compare
    torch.testing.assert_close(o.float(), o_ref.float(), atol=1e-2, rtol=1e-3)


@pytest.mark.parametrize("head_dim", [64, 128])
@torch.inference_mode()
def test_paged_decode_csr_gqa(head_dim: int):
    """Test GQA configuration (multiple query heads per KV head)."""
    torch.manual_seed(42)
    dtype = torch.float16

    batch_size = 2
    seq_len = 64
    num_heads = 16
    num_kv_heads = 4  # 4:1 GQA ratio

    total_kv_tokens = batch_size * seq_len

    q = torch.randn(batch_size, num_heads, head_dim, dtype=dtype, device="cuda")
    k_buffer = torch.randn(total_kv_tokens, num_kv_heads, head_dim, dtype=dtype, device="cuda")
    v_buffer = torch.randn(total_kv_tokens, num_kv_heads, head_dim, dtype=dtype, device="cuda")
    o = torch.zeros(batch_size, num_heads, head_dim, dtype=dtype, device="cuda")

    kv_indptr = torch.arange(0, batch_size + 1, dtype=torch.int32, device="cuda") * seq_len
    kv_indices = torch.arange(0, total_kv_tokens, dtype=torch.int32, device="cuda")

    seq_lens = torch.full((batch_size,), seq_len, device="cuda")
    num_kv_splits, max_kv_splits = _compute_num_kv_splits(seq_lens, seq_len)

    attn_logits = torch.zeros(
        batch_size, num_heads, max_kv_splits, head_dim,
        dtype=torch.float32, device="cuda"
    )
    attn_lse = torch.zeros(
        batch_size, num_heads, max_kv_splits,
        dtype=torch.float32, device="cuda"
    )

    # Run reference
    o_ref = decode_attention_reference(q, k_buffer, v_buffer, kv_indptr, kv_indices)

    # Run kernel
    sm_scale = 1.0 / (head_dim ** 0.5)
    decode_attention_fwd(
        q, k_buffer, v_buffer, o,
        kv_indptr, kv_indices,
        attn_logits, attn_lse,
        num_kv_splits, max_kv_splits,
        sm_scale,
    )

    # Compare
    torch.testing.assert_close(o.float(), o_ref.float(), atol=1e-2, rtol=1e-3)


if __name__ == "__main__":
    # Run quick tests
    print("Testing CSR-style paged decode attention (MHA)...")
    test_paged_decode_csr(
        batch_size=2, seq_len=64, num_heads=8, num_kv_heads=8, head_dim=64, dtype=torch.float16
    )
    print("  MHA test passed!")

    print("Testing CSR-style paged decode attention (GQA)...")
    test_paged_decode_csr(
        batch_size=2, seq_len=64, num_heads=16, num_kv_heads=4, head_dim=64, dtype=torch.float16
    )
    print("  GQA test passed!")

    print("Testing CSR-style paged decode with variable lengths...")
    test_paged_decode_csr_variable_length(
        batch_size=2, num_heads=8, num_kv_heads=4, head_dim=64
    )
    print("  Variable length test passed!")

    print("\nAll CSR-style paged decode attention tests passed!")
