# MLA Standalone Tests
# Tests the MLA Triton decode attention implementation without external dependencies

import sys
from pathlib import Path

# Add paths for standalone execution
sys.path.insert(0, str(Path(__file__).parent.parent / "kernels"))

import pytest
import torch


def get_device():
    """Get the available device for testing."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def requires_cuda(func):
    """Skip test if CUDA is not available."""
    return pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available"
    )(func)


def get_abs_err(x, y):
    return (x - y).flatten().abs().max().item()


def get_err_ratio(x, y):
    err = (x - y).flatten().square().mean().sqrt().item()
    base = (x).flatten().square().mean().sqrt().item()
    return err / base if base > 0 else 0


def assert_close(prefix, ref, tri, ratio):
    msg = f"{prefix} diff: {get_abs_err(ref, tri):.6f} ratio: {get_err_ratio(ref, tri):.6f}"
    print(msg)
    assert get_err_ratio(ref, tri) < ratio, msg


def reference_attention(q, k, v, scale):
    """
    Reference implementation of attention for testing.

    Args:
        q: (batch, num_heads, head_dim)
        k: (batch, seq_len, num_kv_heads, head_dim)
        v: (batch, seq_len, num_kv_heads, head_dim_v)
        scale: softmax scale factor
    """
    batch, num_heads, head_dim = q.shape
    _, seq_len, num_kv_heads, _ = k.shape
    head_dim_v = v.shape[-1]

    kv_group_num = num_heads // num_kv_heads

    # Expand q for computation
    q = q.float()
    k = k.float()
    v = v.float()

    # Output shape: (batch, num_heads, head_dim_v)
    o = torch.zeros(batch, num_heads, head_dim_v, dtype=torch.float32, device=q.device)
    lse = torch.zeros(batch, num_heads, dtype=torch.float32, device=q.device)

    for b in range(batch):
        for h in range(num_heads):
            kv_h = h // kv_group_num

            # q_h: (head_dim,)
            q_h = q[b, h]

            # k_h: (seq_len, head_dim)
            k_h = k[b, :, kv_h]

            # v_h: (seq_len, head_dim_v)
            v_h = v[b, :, kv_h]

            # Compute attention scores
            scores = torch.matmul(q_h, k_h.T) * scale  # (seq_len,)

            # Softmax
            max_score = scores.max()
            exp_scores = torch.exp(scores - max_score)
            sum_exp = exp_scores.sum()
            attn_weights = exp_scores / sum_exp

            # Output
            o[b, h] = torch.matmul(attn_weights, v_h)
            lse[b, h] = max_score + torch.log(sum_exp)

    return o, lse


@requires_cuda
@pytest.mark.parametrize("batch,num_heads,num_kv_heads,seq_len,head_dim,head_dim_v", [
    (1, 4, 4, 32, 64, 64),      # MHA
    (2, 8, 2, 64, 64, 64),      # GQA
    (1, 16, 1, 128, 64, 64),    # MQA
    (2, 8, 1, 64, 128, 128),    # MLA-like
])
@torch.inference_mode()
def test_decode_attention_basic(batch: int, num_heads: int, num_kv_heads: int,
                                 seq_len: int, head_dim: int, head_dim_v: int):
    """Test basic decode attention against reference."""
    from triton_mla_decode import decode_attention_fwd

    device = torch.device("cuda")
    torch.manual_seed(42)

    dtype = torch.float16

    # Create inputs
    q = torch.randn(batch, num_heads, head_dim, dtype=dtype, device=device) * 0.1

    # For simplicity, use page_size=1 (no paging)
    page_size = 1
    k_buffer = torch.randn(batch * seq_len, page_size, num_kv_heads, head_dim,
                           dtype=dtype, device=device) * 0.1
    v_buffer = torch.randn(batch * seq_len, page_size, num_kv_heads, head_dim_v,
                           dtype=dtype, device=device) * 0.1

    # Request to token mapping - each batch maps to its own portion of the flattened KV buffer
    # For batch i, tokens are at indices [i*seq_len, i*seq_len+1, ..., i*seq_len+seq_len-1]
    req_to_token = torch.stack([
        torch.arange(seq_len, device=device) + i * seq_len
        for i in range(batch)
    ]).contiguous().int()

    # Sequence lengths
    b_seq_len = torch.full((batch,), seq_len, dtype=torch.int32, device=device)

    # Scale
    sm_scale = 1.0 / (head_dim ** 0.5)

    # Allocate outputs
    num_kv_splits = 4
    o = torch.zeros(batch, num_heads, head_dim_v, dtype=dtype, device=device)
    lse = torch.zeros(batch, num_heads, dtype=torch.float32, device=device)
    attn_logits = torch.zeros(batch, num_heads, num_kv_splits, head_dim_v + 1,
                              dtype=torch.float32, device=device)

    # Run Triton kernel
    decode_attention_fwd(
        q, k_buffer, v_buffer, o, lse,
        req_to_token, b_seq_len, attn_logits,
        num_kv_splits, sm_scale, page_size
    )

    # Compute reference (reshape k_buffer and v_buffer for reference)
    k_ref = k_buffer.view(batch, seq_len, num_kv_heads, head_dim)
    v_ref = v_buffer.view(batch, seq_len, num_kv_heads, head_dim_v)
    o_ref, lse_ref = reference_attention(q, k_ref, v_ref, sm_scale)

    # Check results
    assert o.shape == o_ref.shape, f"Output shape mismatch: {o.shape} vs {o_ref.shape}"
    assert torch.isfinite(o).all(), "Output contains NaN or inf"

    # Allow some tolerance due to different computation orders
    assert_close("decode_attention output", o_ref, o.float(), 0.05)


@requires_cuda
@pytest.mark.parametrize("batch,num_heads,num_kv_heads,seq_len,head_dim", [
    (1, 8, 1, 256, 576),   # DeepSeek-like MLA with 576 head_dim
    (2, 16, 1, 128, 288),  # DeepSeek-like MLA with 288 head_dim
])
@torch.inference_mode()
def test_decode_attention_mla_dims(batch: int, num_heads: int, num_kv_heads: int,
                                    seq_len: int, head_dim: int):
    """Test decode attention with MLA-specific dimensions (576, 288 head dims)."""
    from triton_mla_decode import decode_attention_fwd

    device = torch.device("cuda")
    torch.manual_seed(42)

    dtype = torch.float16
    head_dim_v = 512 if head_dim == 576 else 256

    # Create inputs
    q = torch.randn(batch, num_heads, head_dim, dtype=dtype, device=device) * 0.1

    page_size = 1
    k_buffer = torch.randn(batch * seq_len, page_size, num_kv_heads, head_dim,
                           dtype=dtype, device=device) * 0.1
    v_buffer = torch.randn(batch * seq_len, page_size, num_kv_heads, head_dim_v,
                           dtype=dtype, device=device) * 0.1

    # Request to token mapping - each batch maps to its own portion of the flattened KV buffer
    req_to_token = torch.stack([
        torch.arange(seq_len, device=device) + i * seq_len
        for i in range(batch)
    ]).contiguous().int()
    b_seq_len = torch.full((batch,), seq_len, dtype=torch.int32, device=device)
    sm_scale = 1.0 / (head_dim ** 0.5)

    num_kv_splits = 4
    o = torch.zeros(batch, num_heads, head_dim_v, dtype=dtype, device=device)
    lse = torch.zeros(batch, num_heads, dtype=torch.float32, device=device)
    attn_logits = torch.zeros(batch, num_heads, num_kv_splits, head_dim_v + 1,
                              dtype=torch.float32, device=device)

    # Run Triton kernel
    decode_attention_fwd(
        q, k_buffer, v_buffer, o, lse,
        req_to_token, b_seq_len, attn_logits,
        num_kv_splits, sm_scale, page_size
    )

    # Basic checks
    assert o.shape == (batch, num_heads, head_dim_v)
    assert torch.isfinite(o).all(), "Output contains NaN or inf"
    assert torch.isfinite(lse).all(), "LSE contains NaN or inf"


@requires_cuda
def test_decode_attention_shapes():
    """Test that output shapes are correct."""
    from triton_mla_decode import decode_attention_fwd

    device = torch.device("cuda")
    dtype = torch.float16

    batch, num_heads, num_kv_heads = 2, 8, 2
    seq_len, head_dim, head_dim_v = 64, 64, 64
    page_size = 1
    num_kv_splits = 4

    q = torch.randn(batch, num_heads, head_dim, dtype=dtype, device=device)
    k_buffer = torch.randn(batch * seq_len, page_size, num_kv_heads, head_dim,
                           dtype=dtype, device=device)
    v_buffer = torch.randn(batch * seq_len, page_size, num_kv_heads, head_dim_v,
                           dtype=dtype, device=device)
    # Request to token mapping - each batch maps to its own portion of the flattened KV buffer
    req_to_token = torch.stack([
        torch.arange(seq_len, device=device) + i * seq_len
        for i in range(batch)
    ]).contiguous().int()
    b_seq_len = torch.full((batch,), seq_len, dtype=torch.int32, device=device)
    sm_scale = 1.0 / (head_dim ** 0.5)

    o = torch.zeros(batch, num_heads, head_dim_v, dtype=dtype, device=device)
    lse = torch.zeros(batch, num_heads, dtype=torch.float32, device=device)
    attn_logits = torch.zeros(batch, num_heads, num_kv_splits, head_dim_v + 1,
                              dtype=torch.float32, device=device)

    decode_attention_fwd(
        q, k_buffer, v_buffer, o, lse,
        req_to_token, b_seq_len, attn_logits,
        num_kv_splits, sm_scale, page_size
    )

    assert o.shape == (batch, num_heads, head_dim_v), f"Output shape: {o.shape}"
    assert lse.shape == (batch, num_heads), f"LSE shape: {lse.shape}"


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available, skipping tests")
        exit(0)

    print("Testing decode attention basic...")
    test_decode_attention_basic(batch=1, num_heads=4, num_kv_heads=4,
                                 seq_len=32, head_dim=64, head_dim_v=64)
    print("  Basic test passed!")

    print("Testing decode attention GQA...")
    test_decode_attention_basic(batch=2, num_heads=8, num_kv_heads=2,
                                 seq_len=64, head_dim=64, head_dim_v=64)
    print("  GQA test passed!")

    print("Testing decode attention shapes...")
    test_decode_attention_shapes()
    print("  Shape test passed!")

    print("\nAll MLA standalone tests passed!")
