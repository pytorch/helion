# Copyright 2023-2024 SGLang Team
# Standalone test for extend attention with CSR-style paged KV cache
# Origin: sglang_attention/kernels/extend_attention.py

import sys
from pathlib import Path

# Add attentions folder to path for package imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
import torch

from gqa.kernels.extend_attention_csr import extend_attention_fwd


# PyTorch reference implementation
def extend_attention_reference(
    q_extend: torch.Tensor,  # [total_extend_tokens, num_heads, head_dim]
    k_extend: torch.Tensor,  # [total_extend_tokens, num_kv_heads, head_dim]
    v_extend: torch.Tensor,  # [total_extend_tokens, num_kv_heads, head_dim]
    k_buffer: torch.Tensor,  # [total_kv_tokens, num_kv_heads, head_dim]
    v_buffer: torch.Tensor,  # [total_kv_tokens, num_kv_heads, head_dim]
    qo_indptr: torch.Tensor,  # [batch_size + 1] - query/output boundaries
    kv_indptr: torch.Tensor,  # [batch_size + 1] - KV boundaries in buffer
    kv_indices: torch.Tensor,  # [total_kv_tokens] - physical indices
    is_causal: bool = True,
) -> torch.Tensor:
    """
    Reference implementation of extend attention.

    Extend attention handles the case where we have:
    - A prefix KV cache (from k_buffer/v_buffer via kv_indices)
    - New extend tokens (k_extend, v_extend) that attend to prefix + themselves

    For each sequence:
    - prefix_len = kv_indptr[seq+1] - kv_indptr[seq]
    - extend_len = qo_indptr[seq+1] - qo_indptr[seq]
    - Query tokens attend to: prefix KV + extend KV (up to causal position)
    """
    total_extend, num_heads, head_dim = q_extend.shape
    num_kv_heads = k_extend.shape[1]
    kv_group_num = num_heads // num_kv_heads
    batch_size = len(qo_indptr) - 1

    sm_scale = 1.0 / (head_dim ** 0.5)
    o = torch.zeros_like(q_extend)

    for b in range(batch_size):
        # Get boundaries
        q_start = qo_indptr[b].item()
        q_end = qo_indptr[b + 1].item()
        extend_len = q_end - q_start

        kv_start = kv_indptr[b].item()
        kv_end = kv_indptr[b + 1].item()
        prefix_len = kv_end - kv_start

        if extend_len == 0:
            continue

        # Gather prefix KV from buffer
        if prefix_len > 0:
            prefix_indices = kv_indices[kv_start:kv_end]
            k_prefix = k_buffer[prefix_indices]  # [prefix_len, num_kv_heads, head_dim]
            v_prefix = v_buffer[prefix_indices]
        else:
            k_prefix = None
            v_prefix = None

        # Get extend KV
        k_ext = k_extend[q_start:q_end]  # [extend_len, num_kv_heads, head_dim]
        v_ext = v_extend[q_start:q_end]

        # For each query position in extend
        for q_pos in range(extend_len):
            q_h = q_extend[q_start + q_pos]  # [num_heads, head_dim]

            for h in range(num_heads):
                kv_h = h // kv_group_num

                q_vec = q_h[h, :].float()  # [head_dim]

                # Gather all KV this query can attend to
                if is_causal:
                    # Can attend to all prefix + extend up to current position
                    if k_prefix is not None:
                        k_all = torch.cat([k_prefix[:, kv_h, :], k_ext[:q_pos+1, kv_h, :]], dim=0)
                        v_all = torch.cat([v_prefix[:, kv_h, :], v_ext[:q_pos+1, kv_h, :]], dim=0)
                    else:
                        k_all = k_ext[:q_pos+1, kv_h, :]
                        v_all = v_ext[:q_pos+1, kv_h, :]
                else:
                    # Can attend to all prefix + all extend
                    if k_prefix is not None:
                        k_all = torch.cat([k_prefix[:, kv_h, :], k_ext[:, kv_h, :]], dim=0)
                        v_all = torch.cat([v_prefix[:, kv_h, :], v_ext[:, kv_h, :]], dim=0)
                    else:
                        k_all = k_ext[:, kv_h, :]
                        v_all = v_ext[:, kv_h, :]

                # Compute attention
                scores = torch.matmul(q_vec.unsqueeze(0), k_all.float().T) * sm_scale
                attn = torch.softmax(scores, dim=-1)
                out_h = torch.matmul(attn, v_all.float())

                o[q_start + q_pos, h, :] = out_h.to(q_extend.dtype).squeeze(0)

    return o


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("extend_len", [16, 32])
@pytest.mark.parametrize("prefix_len", [0, 32])
@pytest.mark.parametrize("num_heads", [8])
@pytest.mark.parametrize("num_kv_heads", [4, 8])
@pytest.mark.parametrize("head_dim", [64])
@pytest.mark.parametrize("dtype", [torch.float16])
@torch.inference_mode()
def test_extend_attention_csr(
    batch_size: int,
    extend_len: int,
    prefix_len: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
):
    """Test extend attention with CSR-style indexing."""
    if num_heads < num_kv_heads:
        pytest.skip("num_heads must be >= num_kv_heads")
    if num_heads % num_kv_heads != 0:
        pytest.skip("num_heads must be divisible by num_kv_heads")

    torch.manual_seed(42)

    total_extend = batch_size * extend_len
    total_prefix = batch_size * prefix_len

    # Extend tensors (new tokens)
    q_extend = torch.randn(total_extend, num_heads, head_dim, dtype=dtype, device="cuda")
    k_extend = torch.randn(total_extend, num_kv_heads, head_dim, dtype=dtype, device="cuda")
    v_extend = torch.randn(total_extend, num_kv_heads, head_dim, dtype=dtype, device="cuda")
    o_extend = torch.zeros_like(q_extend)

    # KV buffer (prefix cache)
    if total_prefix > 0:
        k_buffer = torch.randn(total_prefix, num_kv_heads, head_dim, dtype=dtype, device="cuda")
        v_buffer = torch.randn(total_prefix, num_kv_heads, head_dim, dtype=dtype, device="cuda")
    else:
        # Need at least a dummy buffer
        k_buffer = torch.randn(1, num_kv_heads, head_dim, dtype=dtype, device="cuda")
        v_buffer = torch.randn(1, num_kv_heads, head_dim, dtype=dtype, device="cuda")

    # Query/output boundaries (extend tokens)
    qo_indptr = torch.arange(0, batch_size + 1, dtype=torch.int32, device="cuda") * extend_len

    # KV boundaries (prefix in buffer)
    kv_indptr = torch.arange(0, batch_size + 1, dtype=torch.int32, device="cuda") * prefix_len

    # KV indices (sequential for simplicity)
    if total_prefix > 0:
        kv_indices = torch.arange(0, total_prefix, dtype=torch.int32, device="cuda")
    else:
        kv_indices = torch.zeros(1, dtype=torch.int32, device="cuda")

    # Run reference
    o_ref = extend_attention_reference(
        q_extend, k_extend, v_extend, k_buffer, v_buffer,
        qo_indptr, kv_indptr, kv_indices, is_causal=True
    )

    # Run kernel
    extend_attention_fwd(
        q_extend, k_extend, v_extend, o_extend,
        k_buffer, v_buffer,
        qo_indptr, kv_indptr, kv_indices,
        custom_mask=None,
        is_causal=True,
        mask_indptr=None,
        max_len_extend=extend_len,
    )

    # Compare
    rtol = 1e-2 if dtype == torch.bfloat16 else 1e-3
    torch.testing.assert_close(o_extend.float(), o_ref.float(), atol=1e-2, rtol=rtol)


@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize("num_heads", [8])
@pytest.mark.parametrize("head_dim", [64])
@torch.inference_mode()
def test_extend_attention_csr_variable_length(
    batch_size: int,
    num_heads: int,
    head_dim: int,
):
    """Test extend attention with variable-length sequences."""
    torch.manual_seed(42)
    dtype = torch.float16
    num_kv_heads = num_heads  # MHA for simplicity

    # Variable lengths
    extend_lens = [16, 32]
    prefix_lens = [24, 48]

    total_extend = sum(extend_lens)
    total_prefix = sum(prefix_lens)

    # Extend tensors
    q_extend = torch.randn(total_extend, num_heads, head_dim, dtype=dtype, device="cuda")
    k_extend = torch.randn(total_extend, num_kv_heads, head_dim, dtype=dtype, device="cuda")
    v_extend = torch.randn(total_extend, num_kv_heads, head_dim, dtype=dtype, device="cuda")
    o_extend = torch.zeros_like(q_extend)

    # KV buffer
    k_buffer = torch.randn(total_prefix, num_kv_heads, head_dim, dtype=dtype, device="cuda")
    v_buffer = torch.randn(total_prefix, num_kv_heads, head_dim, dtype=dtype, device="cuda")

    # Build indices
    qo_indptr = torch.tensor([0, extend_lens[0], total_extend], dtype=torch.int32, device="cuda")
    kv_indptr = torch.tensor([0, prefix_lens[0], total_prefix], dtype=torch.int32, device="cuda")
    kv_indices = torch.arange(0, total_prefix, dtype=torch.int32, device="cuda")

    # Run reference
    o_ref = extend_attention_reference(
        q_extend, k_extend, v_extend, k_buffer, v_buffer,
        qo_indptr, kv_indptr, kv_indices, is_causal=True
    )

    # Run kernel
    extend_attention_fwd(
        q_extend, k_extend, v_extend, o_extend,
        k_buffer, v_buffer,
        qo_indptr, kv_indptr, kv_indices,
        custom_mask=None,
        is_causal=True,
        mask_indptr=None,
        max_len_extend=max(extend_lens),
    )

    # Compare
    torch.testing.assert_close(o_extend.float(), o_ref.float(), atol=1e-2, rtol=1e-3)


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("extend_len", [32])
@pytest.mark.parametrize("prefix_len", [32])
@pytest.mark.parametrize("head_dim", [64])
@torch.inference_mode()
def test_extend_attention_csr_non_causal(
    batch_size: int,
    extend_len: int,
    prefix_len: int,
    head_dim: int,
):
    """Test non-causal extend attention."""
    torch.manual_seed(42)
    dtype = torch.float16
    num_heads = 8
    num_kv_heads = 8

    total_extend = batch_size * extend_len
    total_prefix = batch_size * prefix_len

    q_extend = torch.randn(total_extend, num_heads, head_dim, dtype=dtype, device="cuda")
    k_extend = torch.randn(total_extend, num_kv_heads, head_dim, dtype=dtype, device="cuda")
    v_extend = torch.randn(total_extend, num_kv_heads, head_dim, dtype=dtype, device="cuda")
    o_extend = torch.zeros_like(q_extend)

    k_buffer = torch.randn(total_prefix, num_kv_heads, head_dim, dtype=dtype, device="cuda")
    v_buffer = torch.randn(total_prefix, num_kv_heads, head_dim, dtype=dtype, device="cuda")

    qo_indptr = torch.arange(0, batch_size + 1, dtype=torch.int32, device="cuda") * extend_len
    kv_indptr = torch.arange(0, batch_size + 1, dtype=torch.int32, device="cuda") * prefix_len
    kv_indices = torch.arange(0, total_prefix, dtype=torch.int32, device="cuda")

    # Run reference (non-causal)
    o_ref = extend_attention_reference(
        q_extend, k_extend, v_extend, k_buffer, v_buffer,
        qo_indptr, kv_indptr, kv_indices, is_causal=False
    )

    # Run kernel (non-causal)
    extend_attention_fwd(
        q_extend, k_extend, v_extend, o_extend,
        k_buffer, v_buffer,
        qo_indptr, kv_indptr, kv_indices,
        custom_mask=None,
        is_causal=False,
        mask_indptr=None,
        max_len_extend=extend_len,
    )

    # Compare
    torch.testing.assert_close(o_extend.float(), o_ref.float(), atol=1e-2, rtol=1e-3)


if __name__ == "__main__":
    # Run quick tests
    print("Testing extend attention (with prefix, causal)...")
    test_extend_attention_csr(
        batch_size=2, extend_len=32, prefix_len=32,
        num_heads=8, num_kv_heads=8, head_dim=64, dtype=torch.float16
    )
    print("  Causal with prefix test passed!")

    print("Testing extend attention (no prefix, causal)...")
    test_extend_attention_csr(
        batch_size=2, extend_len=32, prefix_len=0,
        num_heads=8, num_kv_heads=8, head_dim=64, dtype=torch.float16
    )
    print("  Causal without prefix test passed!")

    print("Testing extend attention with variable lengths...")
    test_extend_attention_csr_variable_length(
        batch_size=2, num_heads=8, head_dim=64
    )
    print("  Variable length test passed!")

    print("Testing extend attention (non-causal)...")
    test_extend_attention_csr_non_causal(
        batch_size=2, extend_len=32, prefix_len=32, head_dim=64
    )
    print("  Non-causal test passed!")

    print("\nAll extend attention CSR tests passed!")
