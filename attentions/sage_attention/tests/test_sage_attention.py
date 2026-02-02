# SPDX-License-Identifier: Apache-2.0
# SageAttention standalone test
# Tests INT8 quantized attention kernels against PyTorch reference

import sys
from pathlib import Path

# Add kernels folder to path for standalone execution
sys.path.insert(0, str(Path(__file__).parent.parent / "kernels"))

import pytest
import torch

from attn_qk_int8_per_block import forward as sage_attn_forward
from attn_qk_int8_per_block_causal import forward as sage_attn_forward_causal
from quant_per_block import per_block_int8


# PyTorch reference implementation of scaled dot-product attention
def attention_reference(
    q: torch.Tensor,  # [B, H, N, D]
    k: torch.Tensor,  # [B, H, N, D]
    v: torch.Tensor,  # [B, H, N, D]
    is_causal: bool = False,
    sm_scale: float = None,
) -> torch.Tensor:
    """Reference implementation of scaled dot-product attention."""
    if sm_scale is None:
        sm_scale = 1.0 / (q.size(-1) ** 0.5)

    # Compute attention scores
    attn = torch.matmul(q.float(), k.float().transpose(-2, -1)) * sm_scale

    if is_causal:
        seq_len = q.size(2)
        # Create causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float('-inf'))

    # Softmax
    attn = torch.softmax(attn, dim=-1)

    # Apply attention to values
    out = torch.matmul(attn, v.float())
    return out


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("num_heads", [8, 16])
@pytest.mark.parametrize("seq_len", [256, 512])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_sage_attention_non_causal(
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    dtype: torch.dtype,
):
    """Test non-causal SageAttention against PyTorch reference."""
    torch.manual_seed(42)

    # Create input tensors in HND layout [B, H, N, D]
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device="cuda")
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device="cuda")
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device="cuda")

    sm_scale = 1.0 / (head_dim ** 0.5)

    # Quantize Q and K to INT8
    q_int8, q_scale, k_int8, k_scale = per_block_int8(
        q, k, km=None, sm_scale=sm_scale, tensor_layout="HND"
    )

    # Run SageAttention kernel
    out_sage, _ = sage_attn_forward(
        q_int8, k_int8, v, q_scale, k_scale,
        tensor_layout="HND",
        output_dtype=dtype,
    )

    # Run reference
    out_ref = attention_reference(q, k, v, is_causal=False, sm_scale=sm_scale)

    # Compare - use relaxed tolerances due to INT8 quantization
    # SageAttention trades some precision for speed
    torch.testing.assert_close(
        out_sage.float(),
        out_ref.to(dtype).float(),
        atol=0.15,  # Relaxed tolerance for INT8 quantization error
        rtol=0.15,
    )


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("num_heads", [8, 16])
@pytest.mark.parametrize("seq_len", [256, 512])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_sage_attention_causal(
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    dtype: torch.dtype,
):
    """Test causal SageAttention against PyTorch reference."""
    torch.manual_seed(42)

    # Create input tensors in HND layout [B, H, N, D]
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device="cuda")
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device="cuda")
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device="cuda")

    sm_scale = 1.0 / (head_dim ** 0.5)

    # Quantize Q and K to INT8
    q_int8, q_scale, k_int8, k_scale = per_block_int8(
        q, k, km=None, sm_scale=sm_scale, tensor_layout="HND"
    )

    # Run SageAttention kernel (causal)
    out_sage, _ = sage_attn_forward_causal(
        q_int8, k_int8, v, q_scale, k_scale,
        tensor_layout="HND",
        output_dtype=dtype,
    )

    # Run reference (causal)
    out_ref = attention_reference(q, k, v, is_causal=True, sm_scale=sm_scale)

    # Compare - use relaxed tolerances due to INT8 quantization
    torch.testing.assert_close(
        out_sage.float(),
        out_ref.to(dtype).float(),
        atol=0.15,
        rtol=0.15,
    )


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("num_heads", [8])
@pytest.mark.parametrize("seq_len", [256])
@pytest.mark.parametrize("head_dim", [64, 128])
@torch.inference_mode()
def test_sage_attention_nhd_layout(
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
):
    """Test SageAttention with NHD tensor layout."""
    torch.manual_seed(42)
    dtype = torch.float16

    # Create input tensors in NHD layout [B, N, H, D]
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=dtype, device="cuda")
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=dtype, device="cuda")
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.float16, device="cuda")

    sm_scale = 1.0 / (head_dim ** 0.5)

    # Quantize Q and K to INT8 with NHD layout
    q_int8, q_scale, k_int8, k_scale = per_block_int8(
        q, k, km=None, sm_scale=sm_scale, tensor_layout="NHD"
    )

    # Run SageAttention kernel with NHD layout
    out_sage, _ = sage_attn_forward(
        q_int8, k_int8, v, q_scale, k_scale,
        tensor_layout="NHD",
        output_dtype=dtype,
    )

    # Convert to HND for reference
    q_hnd = q.permute(0, 2, 1, 3)  # [B, H, N, D]
    k_hnd = k.permute(0, 2, 1, 3)
    v_hnd = v.permute(0, 2, 1, 3)

    # Run reference
    out_ref = attention_reference(q_hnd, k_hnd, v_hnd, is_causal=False, sm_scale=sm_scale)
    out_ref = out_ref.permute(0, 2, 1, 3)  # Back to NHD

    # Compare
    torch.testing.assert_close(
        out_sage.float(),
        out_ref.to(dtype).float(),
        atol=0.15,
        rtol=0.15,
    )


if __name__ == "__main__":
    # Run quick tests
    print("Testing SageAttention non-causal...")
    test_sage_attention_non_causal(
        batch_size=2, num_heads=8, seq_len=256, head_dim=128, dtype=torch.float16
    )
    print("  Non-causal test passed!")

    print("Testing SageAttention causal...")
    test_sage_attention_causal(
        batch_size=2, num_heads=8, seq_len=256, head_dim=128, dtype=torch.float16
    )
    print("  Causal test passed!")

    print("Testing SageAttention NHD layout...")
    test_sage_attention_nhd_layout(
        batch_size=1, num_heads=8, seq_len=256, head_dim=128
    )
    print("  NHD layout test passed!")

    print("\nAll SageAttention tests passed!")
