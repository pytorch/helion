# Standalone tests for paged sliding window attention kernels

import sys
from pathlib import Path

# Add kernels folder to path for standalone execution
sys.path.insert(0, str(Path(__file__).parent.parent / "kernels"))

import pytest
import torch

from paged_swa_decode import paged_swa_decode_fwd, paged_swa_decode_blocktable_fwd
from paged_swa_prefill import paged_swa_prefill_fwd, paged_swa_extend_fwd


# PyTorch reference implementations
def swa_decode_reference(
    q: torch.Tensor,  # [batch_size, num_heads, head_dim]
    k_buffer: torch.Tensor,  # [total_kv_tokens, num_kv_heads, head_dim]
    v_buffer: torch.Tensor,  # [total_kv_tokens, num_kv_heads, head_dim]
    kv_indptr: torch.Tensor,  # [batch_size + 1]
    kv_indices: torch.Tensor,  # [total_kv_tokens]
    sliding_window: int,
) -> torch.Tensor:
    """Reference implementation of paged decode with sliding window."""
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

        # Apply sliding window
        if sliding_window > 0:
            window_start = max(0, seq_len - sliding_window)
        else:
            window_start = 0

        kv_idx = kv_indices[start + window_start:end]
        k_b = k_buffer[kv_idx]
        v_b = v_buffer[kv_idx]

        for h in range(num_heads):
            kv_h = h // kv_group_num
            q_h = q[b, h, :].float()
            k_h = k_b[:, kv_h, :].float()
            v_h = v_b[:, kv_h, :].float()

            scores = torch.matmul(q_h.unsqueeze(0), k_h.T) * sm_scale
            attn = torch.softmax(scores, dim=-1)
            out_h = torch.matmul(attn, v_h)
            o[b, h, :] = out_h.to(q.dtype).squeeze(0)

    return o


def swa_prefill_reference(
    q: torch.Tensor,  # [total_tokens, num_heads, head_dim]
    k: torch.Tensor,  # [total_tokens, num_kv_heads, head_dim]
    v: torch.Tensor,  # [total_tokens, num_kv_heads, head_dim]
    b_start_loc: torch.Tensor,
    b_seq_len: torch.Tensor,
    sliding_window: int,
    is_causal: bool = True,
) -> torch.Tensor:
    """Reference implementation of prefill with sliding window."""
    total_tokens, num_heads, head_dim = q.shape
    num_kv_heads = k.shape[1]
    kv_group_num = num_heads // num_kv_heads
    batch_size = b_seq_len.shape[0]

    sm_scale = 1.0 / (head_dim ** 0.5)
    o = torch.zeros_like(q)

    for b in range(batch_size):
        start = b_start_loc[b].item()
        seq_len = b_seq_len[b].item()

        q_b = q[start:start+seq_len]
        k_b = k[start:start+seq_len]
        v_b = v[start:start+seq_len]

        for h in range(num_heads):
            kv_h = h // kv_group_num

            q_h = q_b[:, h, :].float()
            k_h = k_b[:, kv_h, :].float()
            v_h = v_b[:, kv_h, :].float()

            scores = torch.matmul(q_h, k_h.T) * sm_scale

            # Build mask
            mask = torch.zeros(seq_len, seq_len, device=q.device, dtype=torch.bool)

            if is_causal:
                # Causal mask: positions where key > query are masked
                causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device), diagonal=1).bool()
                mask = mask | causal_mask

            if sliding_window > 0:
                # Window mask: positions where key < query - window + 1 are masked
                for i in range(seq_len):
                    for j in range(seq_len):
                        if j < i - sliding_window + 1:
                            mask[i, j] = True

            scores = scores.masked_fill(mask, float('-inf'))
            attn = torch.softmax(scores, dim=-1)
            out_h = torch.matmul(attn, v_h)
            o[start:start+seq_len, h, :] = out_h.to(q.dtype)

    return o


def _compute_num_kv_splits(seq_lens: torch.Tensor, max_seq_len: int) -> tuple:
    """Compute number of KV splits for each sequence."""
    min_block_kv = 32
    max_kv_splits = min(16, (max_seq_len + min_block_kv - 1) // min_block_kv)
    max_kv_splits = max(max_kv_splits, 1)

    num_kv_splits = torch.ones(len(seq_lens), dtype=torch.int32, device=seq_lens.device)
    for i, seq_len in enumerate(seq_lens.tolist()):
        num_kv_splits[i] = min(max_kv_splits, (seq_len + min_block_kv - 1) // min_block_kv)
        num_kv_splits[i] = max(num_kv_splits[i], 1)

    return num_kv_splits, max_kv_splits


# ============================================================================
# Tests for CSR-style paged SWA decode
# ============================================================================


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("seq_len", [64, 128])
@pytest.mark.parametrize("sliding_window", [32, 64])
@pytest.mark.parametrize("num_heads", [8])
@pytest.mark.parametrize("head_dim", [64])
@torch.inference_mode()
def test_paged_swa_decode_csr(
    batch_size: int,
    seq_len: int,
    sliding_window: int,
    num_heads: int,
    head_dim: int,
):
    """Test CSR-style paged decode with sliding window."""
    torch.manual_seed(42)
    dtype = torch.float16
    num_kv_heads = num_heads

    total_kv_tokens = batch_size * seq_len

    q = torch.randn(batch_size, num_heads, head_dim, dtype=dtype, device="cuda")
    k_buffer = torch.randn(total_kv_tokens, num_kv_heads, head_dim, dtype=dtype, device="cuda")
    v_buffer = torch.randn(total_kv_tokens, num_kv_heads, head_dim, dtype=dtype, device="cuda")
    o = torch.zeros(batch_size, num_heads, head_dim, dtype=dtype, device="cuda")

    kv_indptr = torch.arange(0, batch_size + 1, dtype=torch.int32, device="cuda") * seq_len
    kv_indices = torch.arange(0, total_kv_tokens, dtype=torch.int32, device="cuda")

    seq_lens = torch.full((batch_size,), seq_len, device="cuda")
    num_kv_splits, max_kv_splits = _compute_num_kv_splits(seq_lens, seq_len)

    attn_logits = torch.zeros(batch_size, num_heads, max_kv_splits, head_dim, dtype=torch.float32, device="cuda")
    attn_lse = torch.zeros(batch_size, num_heads, max_kv_splits, dtype=torch.float32, device="cuda")

    # Run reference
    o_ref = swa_decode_reference(q, k_buffer, v_buffer, kv_indptr, kv_indices, sliding_window)

    # Run kernel
    sm_scale = 1.0 / (head_dim ** 0.5)
    paged_swa_decode_fwd(
        q, k_buffer, v_buffer, o,
        kv_indptr, kv_indices,
        attn_logits, attn_lse,
        num_kv_splits, max_kv_splits,
        sm_scale, sliding_window,
    )

    # Compare
    torch.testing.assert_close(o.float(), o_ref.float(), atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("seq_len", [128])
@pytest.mark.parametrize("sliding_window", [32, 64, 128])  # Different window sizes
@torch.inference_mode()
def test_paged_swa_decode_window_sizes(seq_len: int, sliding_window: int):
    """Test different window sizes."""
    torch.manual_seed(42)
    dtype = torch.float16

    batch_size = 2
    num_heads = 8
    head_dim = 64
    num_kv_heads = num_heads

    total_kv_tokens = batch_size * seq_len

    q = torch.randn(batch_size, num_heads, head_dim, dtype=dtype, device="cuda")
    k_buffer = torch.randn(total_kv_tokens, num_kv_heads, head_dim, dtype=dtype, device="cuda")
    v_buffer = torch.randn(total_kv_tokens, num_kv_heads, head_dim, dtype=dtype, device="cuda")
    o = torch.zeros(batch_size, num_heads, head_dim, dtype=dtype, device="cuda")

    kv_indptr = torch.arange(0, batch_size + 1, dtype=torch.int32, device="cuda") * seq_len
    kv_indices = torch.arange(0, total_kv_tokens, dtype=torch.int32, device="cuda")

    seq_lens = torch.full((batch_size,), seq_len, device="cuda")
    num_kv_splits, max_kv_splits = _compute_num_kv_splits(seq_lens, seq_len)

    attn_logits = torch.zeros(batch_size, num_heads, max_kv_splits, head_dim, dtype=torch.float32, device="cuda")
    attn_lse = torch.zeros(batch_size, num_heads, max_kv_splits, dtype=torch.float32, device="cuda")

    # Run reference
    o_ref = swa_decode_reference(q, k_buffer, v_buffer, kv_indptr, kv_indices, sliding_window)

    # Run kernel
    sm_scale = 1.0 / (head_dim ** 0.5)
    paged_swa_decode_fwd(
        q, k_buffer, v_buffer, o,
        kv_indptr, kv_indices,
        attn_logits, attn_lse,
        num_kv_splits, max_kv_splits,
        sm_scale, sliding_window,
    )

    # Compare
    torch.testing.assert_close(o.float(), o_ref.float(), atol=1e-2, rtol=1e-2)


# ============================================================================
# Tests for block table style paged SWA decode
# ============================================================================


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("seq_len", [64])
@pytest.mark.parametrize("sliding_window", [32])
@pytest.mark.parametrize("head_dim", [64])
@torch.inference_mode()
def test_paged_swa_decode_blocktable(
    batch_size: int,
    seq_len: int,
    sliding_window: int,
    head_dim: int,
):
    """Test block table style paged decode with sliding window."""
    torch.manual_seed(42)
    dtype = torch.float16

    num_heads = 8
    num_kv_heads = num_heads
    block_size = 16

    num_blocks_per_seq = (seq_len + block_size - 1) // block_size
    total_blocks = batch_size * num_blocks_per_seq

    q = torch.randn(batch_size, num_heads, head_dim, dtype=dtype, device="cuda")
    o = torch.zeros(batch_size, num_heads, head_dim, dtype=dtype, device="cuda")

    # KV cache: [num_blocks, block_size, num_kv_heads, head_dim]
    k_cache = torch.randn(total_blocks, block_size, num_kv_heads, head_dim, dtype=dtype, device="cuda")
    v_cache = torch.randn(total_blocks, block_size, num_kv_heads, head_dim, dtype=dtype, device="cuda")

    block_table = torch.arange(total_blocks, dtype=torch.int32, device="cuda").view(batch_size, num_blocks_per_seq)
    seq_lens = torch.full((batch_size,), seq_len, dtype=torch.int32, device="cuda")

    # Flatten KV cache for reference
    k_flat = k_cache.view(-1, num_kv_heads, head_dim)[:batch_size * seq_len]
    v_flat = v_cache.view(-1, num_kv_heads, head_dim)[:batch_size * seq_len]

    kv_indptr = torch.arange(0, batch_size + 1, dtype=torch.int32, device="cuda") * seq_len
    kv_indices = torch.arange(0, batch_size * seq_len, dtype=torch.int32, device="cuda")

    # Run reference
    o_ref = swa_decode_reference(q, k_flat, v_flat, kv_indptr, kv_indices, sliding_window)

    # Run kernel
    sm_scale = 1.0 / (head_dim ** 0.5)
    paged_swa_decode_blocktable_fwd(
        q, k_cache, v_cache, o,
        block_table, seq_lens,
        sm_scale, sliding_window,
    )

    # Compare
    torch.testing.assert_close(o.float(), o_ref.float(), atol=1e-2, rtol=1e-2)


# ============================================================================
# Tests for paged SWA prefill
# ============================================================================


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("seq_len", [32, 64])
@pytest.mark.parametrize("sliding_window", [16, 32])
@pytest.mark.parametrize("num_heads", [8])
@pytest.mark.parametrize("head_dim", [64])
@torch.inference_mode()
def test_paged_swa_prefill(
    batch_size: int,
    seq_len: int,
    sliding_window: int,
    num_heads: int,
    head_dim: int,
):
    """Test prefill attention with sliding window."""
    torch.manual_seed(42)
    dtype = torch.float16
    num_kv_heads = num_heads

    total_tokens = batch_size * seq_len

    q = torch.randn(total_tokens, num_heads, head_dim, dtype=dtype, device="cuda")
    k = torch.randn(total_tokens, num_kv_heads, head_dim, dtype=dtype, device="cuda")
    v = torch.randn(total_tokens, num_kv_heads, head_dim, dtype=dtype, device="cuda")
    o = torch.zeros_like(q)

    b_start_loc = torch.arange(0, total_tokens, seq_len, dtype=torch.int32, device="cuda")
    b_seq_len = torch.full((batch_size,), seq_len, dtype=torch.int32, device="cuda")

    # Run reference
    o_ref = swa_prefill_reference(q, k, v, b_start_loc, b_seq_len, sliding_window, is_causal=True)

    # Run kernel
    paged_swa_prefill_fwd(
        q, k, v, o,
        b_start_loc, b_seq_len,
        seq_len, sliding_window,
        is_causal=True,
    )

    # Compare
    torch.testing.assert_close(o.float(), o_ref.float(), atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("sliding_window", [16, 32, 0])
@torch.inference_mode()
def test_paged_swa_prefill_window_sizes(sliding_window: int):
    """Test prefill with different window sizes."""
    torch.manual_seed(42)
    dtype = torch.float16

    batch_size = 2
    seq_len = 64
    num_heads = 8
    head_dim = 64
    num_kv_heads = num_heads

    total_tokens = batch_size * seq_len

    q = torch.randn(total_tokens, num_heads, head_dim, dtype=dtype, device="cuda")
    k = torch.randn(total_tokens, num_kv_heads, head_dim, dtype=dtype, device="cuda")
    v = torch.randn(total_tokens, num_kv_heads, head_dim, dtype=dtype, device="cuda")
    o = torch.zeros_like(q)

    b_start_loc = torch.arange(0, total_tokens, seq_len, dtype=torch.int32, device="cuda")
    b_seq_len = torch.full((batch_size,), seq_len, dtype=torch.int32, device="cuda")

    # Run reference
    o_ref = swa_prefill_reference(q, k, v, b_start_loc, b_seq_len, sliding_window, is_causal=True)

    # Run kernel
    paged_swa_prefill_fwd(
        q, k, v, o,
        b_start_loc, b_seq_len,
        seq_len, sliding_window,
        is_causal=True,
    )

    # Compare
    torch.testing.assert_close(o.float(), o_ref.float(), atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    # Run quick tests
    print("Testing paged SWA decode (CSR)...")
    test_paged_swa_decode_csr(
        batch_size=2, seq_len=64, sliding_window=32, num_heads=8, head_dim=64
    )
    print("  CSR decode test passed!")

    print("Testing paged SWA decode (block table)...")
    test_paged_swa_decode_blocktable(
        batch_size=2, seq_len=64, sliding_window=32, head_dim=64
    )
    print("  Block table decode test passed!")

    print("Testing paged SWA prefill...")
    test_paged_swa_prefill(
        batch_size=2, seq_len=64, sliding_window=32, num_heads=8, head_dim=64
    )
    print("  Prefill test passed!")

    print("Testing different window sizes...")
    for w in [16, 32, 128]:  # Test positive window sizes only
        test_paged_swa_decode_window_sizes(seq_len=128, sliding_window=w)
        test_paged_swa_prefill_window_sizes(sliding_window=w)
    print("  Window size tests passed!")

    print("\nAll paged SWA tests passed!")
