# Copyright 2023-2024 SGLang Team
# Standalone test for CSR-style paged prefill attention
# Origin: sglang_attention/tests/test_prefill_attention.py

import sys
from pathlib import Path

# Add kernels folder to path for standalone execution
sys.path.insert(0, str(Path(__file__).parent.parent / "kernels"))

import pytest
import torch

from paged_prefill_csr import context_attention_fwd


# PyTorch reference implementation
def attention_reference(
    q: torch.Tensor,  # [total_tokens, num_heads, head_dim]
    k: torch.Tensor,  # [total_tokens, num_kv_heads, head_dim]
    v: torch.Tensor,  # [total_tokens, num_kv_heads, head_dim]
    b_start_loc: torch.Tensor,  # [batch_size]
    b_seq_len: torch.Tensor,  # [batch_size]
    is_causal: bool = True,
) -> torch.Tensor:
    """Reference implementation of prefill attention."""
    total_tokens, num_heads, head_dim = q.shape
    num_kv_heads = k.shape[1]
    kv_group_num = num_heads // num_kv_heads
    batch_size = b_seq_len.shape[0]

    sm_scale = 1.0 / (head_dim ** 0.5)
    o = torch.zeros_like(q)

    for b in range(batch_size):
        start = b_start_loc[b].item()
        seq_len = b_seq_len[b].item()

        # Extract this batch's qkv
        q_b = q[start:start+seq_len]  # [seq_len, num_heads, head_dim]
        k_b = k[start:start+seq_len]  # [seq_len, num_kv_heads, head_dim]
        v_b = v[start:start+seq_len]  # [seq_len, num_kv_heads, head_dim]

        for h in range(num_heads):
            kv_h = h // kv_group_num

            # Compute attention scores
            q_h = q_b[:, h, :]  # [seq_len, head_dim]
            k_h = k_b[:, kv_h, :]  # [seq_len, head_dim]
            v_h = v_b[:, kv_h, :]  # [seq_len, head_dim]

            # [seq_len, seq_len]
            scores = torch.matmul(q_h.float(), k_h.float().T) * sm_scale

            if is_causal:
                # Create causal mask
                mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device), diagonal=1).bool()
                scores = scores.masked_fill(mask, float('-inf'))

            attn = torch.softmax(scores, dim=-1)
            out_h = torch.matmul(attn, v_h.float())  # [seq_len, head_dim]
            o[start:start+seq_len, h, :] = out_h.to(q.dtype)

    return o


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("seq_len", [32, 64, 128])
@pytest.mark.parametrize("num_heads", [8, 16])
@pytest.mark.parametrize("num_kv_heads", [8, 4])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_paged_prefill_csr(
    batch_size: int,
    seq_len: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
):
    """Test CSR-style paged prefill attention against PyTorch reference."""
    if num_heads < num_kv_heads:
        pytest.skip("num_heads must be >= num_kv_heads")
    if num_heads % num_kv_heads != 0:
        pytest.skip("num_heads must be divisible by num_kv_heads")

    torch.manual_seed(42)

    total_tokens = batch_size * seq_len

    # Create input tensors [total_tokens, num_heads, head_dim]
    q = torch.randn(total_tokens, num_heads, head_dim, dtype=dtype, device="cuda")
    k = torch.randn(total_tokens, num_kv_heads, head_dim, dtype=dtype, device="cuda")
    v = torch.randn(total_tokens, num_kv_heads, head_dim, dtype=dtype, device="cuda")
    o = torch.zeros_like(q)

    # Sequence metadata - assume equal length sequences for simplicity
    b_start_loc = torch.arange(0, total_tokens, seq_len, device="cuda")
    b_seq_len = torch.full((batch_size,), seq_len, device="cuda")

    # Run reference
    o_ref = attention_reference(q, k, v, b_start_loc, b_seq_len, is_causal=True)

    # Run CSR-style paged prefill kernel
    context_attention_fwd(q, k, v, o, b_start_loc, b_seq_len, seq_len, is_causal=True)

    # Compare
    rtol = 1e-2 if dtype == torch.bfloat16 else 1e-3
    torch.testing.assert_close(o.float(), o_ref.float(), atol=1e-2, rtol=rtol)


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("seq_len", [64])
@pytest.mark.parametrize("num_heads", [8])
@pytest.mark.parametrize("head_dim", [64])
@torch.inference_mode()
def test_paged_prefill_csr_non_causal(
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
):
    """Test non-causal prefill attention."""
    torch.manual_seed(42)
    dtype = torch.float16
    num_kv_heads = num_heads

    total_tokens = batch_size * seq_len

    q = torch.randn(total_tokens, num_heads, head_dim, dtype=dtype, device="cuda")
    k = torch.randn(total_tokens, num_kv_heads, head_dim, dtype=dtype, device="cuda")
    v = torch.randn(total_tokens, num_kv_heads, head_dim, dtype=dtype, device="cuda")
    o = torch.zeros_like(q)

    b_start_loc = torch.arange(0, total_tokens, seq_len, device="cuda")
    b_seq_len = torch.full((batch_size,), seq_len, device="cuda")

    # Run reference (non-causal)
    o_ref = attention_reference(q, k, v, b_start_loc, b_seq_len, is_causal=False)

    # Run kernel (non-causal)
    context_attention_fwd(q, k, v, o, b_start_loc, b_seq_len, seq_len, is_causal=False)

    # Compare
    torch.testing.assert_close(o.float(), o_ref.float(), atol=1e-2, rtol=1e-3)


@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize("num_heads", [8])
@pytest.mark.parametrize("num_kv_heads", [4, 8])
@pytest.mark.parametrize("head_dim", [64])
@torch.inference_mode()
def test_paged_prefill_csr_variable_length(
    batch_size: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
):
    """Test prefill with variable-length sequences."""
    if num_heads % num_kv_heads != 0:
        pytest.skip("num_heads must be divisible by num_kv_heads")

    torch.manual_seed(42)
    dtype = torch.float16

    # Variable sequence lengths
    seq_lens = [32, 64]
    total_tokens = sum(seq_lens)

    q = torch.randn(total_tokens, num_heads, head_dim, dtype=dtype, device="cuda")
    k = torch.randn(total_tokens, num_kv_heads, head_dim, dtype=dtype, device="cuda")
    v = torch.randn(total_tokens, num_kv_heads, head_dim, dtype=dtype, device="cuda")
    o = torch.zeros_like(q)

    b_start_loc = torch.tensor([0, seq_lens[0]], device="cuda")
    b_seq_len = torch.tensor(seq_lens, device="cuda")

    # Run reference
    o_ref = attention_reference(q, k, v, b_start_loc, b_seq_len, is_causal=True)

    # Run kernel
    context_attention_fwd(q, k, v, o, b_start_loc, b_seq_len, max(seq_lens), is_causal=True)

    # Compare
    torch.testing.assert_close(o.float(), o_ref.float(), atol=1e-2, rtol=1e-3)


if __name__ == "__main__":
    # Run quick tests
    print("Testing CSR-style paged prefill attention (causal)...")
    test_paged_prefill_csr(
        batch_size=2, seq_len=64, num_heads=8, num_kv_heads=8, head_dim=64, dtype=torch.float16
    )
    print("  Causal test passed!")

    print("Testing CSR-style paged prefill attention (non-causal)...")
    test_paged_prefill_csr_non_causal(
        batch_size=2, seq_len=64, num_heads=8, head_dim=64
    )
    print("  Non-causal test passed!")

    print("Testing CSR-style paged prefill with variable lengths...")
    test_paged_prefill_csr_variable_length(
        batch_size=2, num_heads=8, num_kv_heads=4, head_dim=64
    )
    print("  Variable length test passed!")

    print("\nAll CSR-style paged prefill attention tests passed!")
