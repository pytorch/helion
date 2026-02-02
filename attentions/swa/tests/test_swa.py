# SPDX-License-Identifier: Apache-2.0
# Standalone test - imports from local kernels folder

import sys
from pathlib import Path

# Add kernels folder to path for standalone execution
sys.path.insert(0, str(Path(__file__).parent.parent / "kernels"))

import torch
import pytest

from flash_attention import FlashAttention


# Inlined reference implementation
def multi_head_attention(Q, K, V, WINDOW_SIZE, attn_mode):
    """
    Implementation of MultiHead Self-Attention:
    - Global Attention
    - Causal Attention
    - Sliding Window Attention
    - No dropout
    - Float16
    """

    # Q, K, V are already: (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)
    _, _, SEQ_LEN, HEAD_DIM = Q.shape

    attn_bias = torch.zeros(SEQ_LEN, SEQ_LEN, dtype=Q.dtype, device=Q.device)
    softmax_factor = 1 / HEAD_DIM**0.5

    if attn_mode == "causal":
        MASK = torch.ones(SEQ_LEN, SEQ_LEN, dtype=torch.bool, device=Q.device).tril(
            diagonal=0
        )
        attn_bias.masked_fill_(MASK.logical_not(), float("-inf"))

    if attn_mode == "sliding_window":
        all_ones = torch.ones(SEQ_LEN, SEQ_LEN, dtype=torch.bool, device=Q.device)
        half_window = WINDOW_SIZE // 2
        MASK = torch.triu(all_ones, diagonal=-half_window) & torch.tril(
            all_ones, diagonal=half_window
        )
        attn_bias.masked_fill_(MASK.logical_not(), float("-inf"))

    P = torch.matmul(Q, K.transpose(-2, -1)) * softmax_factor
    P += attn_bias
    # Change P to float32 before softmax for numerical stability and precision and back to float16 afterwards
    attn_weights = torch.softmax(P.float(), dim=-1).half()
    attn_output = torch.matmul(attn_weights, V)

    return attn_output


def create_test_tensors(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16):
    """Create test tensors Q, K, V with normal distribution."""
    tensor_shape = (batch_size, num_heads, seq_len, head_dim)

    Q = (
        torch.empty(tensor_shape, dtype=dtype, device="cuda")
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )
    K = (
        torch.empty(tensor_shape, dtype=dtype, device="cuda")
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )
    V = (
        torch.empty(tensor_shape, dtype=dtype, device="cuda")
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )

    return Q, K, V


def run_pytorch_attention(Q, K, V, window_size, attn_mode, dO):
    """Run PyTorch attention implementation"""
    attn_out = multi_head_attention(Q, K, V, window_size, attn_mode)
    attn_out.backward(dO)

    # Extract and save gradients
    attn_dV, V.grad = V.grad.clone(), None
    attn_dK, K.grad = K.grad.clone(), None
    attn_dQ, Q.grad = Q.grad.clone(), None

    return attn_out, attn_dQ, attn_dK, attn_dV


def run_triton_attention(Q, K, V, window_size, attn_mode, dO):
    """Run Triton FlashAttention implementation"""
    flash_out = FlashAttention.apply(Q, K, V, window_size, attn_mode).half()
    flash_out.backward(dO)

    # Extract and save gradients
    flash_dV, V.grad = V.grad.clone(), None
    flash_dK, K.grad = K.grad.clone(), None
    flash_dQ, Q.grad = Q.grad.clone(), None

    return flash_out, flash_dQ, flash_dK, flash_dV


@pytest.mark.parametrize("batch_size", [2, 4])
@pytest.mark.parametrize("num_heads", [4])
@pytest.mark.parametrize("seq_len", [64, 256])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("attn_mode", ["global", "causal"])
def test_flash_attention_basic(batch_size, num_heads, seq_len, head_dim, attn_mode):
    """Test basic flash attention modes (global and causal)."""
    Q, K, V = create_test_tensors(batch_size, num_heads, seq_len, head_dim)
    dO = torch.randn_like(Q)

    # Run PyTorch implementation
    attn_out, attn_dQ, attn_dK, attn_dV = run_pytorch_attention(
        Q, K, V, None, attn_mode, dO
    )

    # Run Triton FlashAttention implementation
    flash_out, flash_dQ, flash_dK, flash_dV = run_triton_attention(
        Q, K, V, None, attn_mode, dO
    )

    # Compare forward pass
    torch.testing.assert_close(attn_out, flash_out, rtol=1e-2, atol=1e-2)

    # Compare backward pass
    torch.testing.assert_close(attn_dV, flash_dV, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(attn_dK, flash_dK, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(attn_dQ, flash_dQ, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize("num_heads", [4])
@pytest.mark.parametrize("seq_len", [128, 256])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("window_size", [64, 96])
def test_flash_attention_sliding_window(batch_size, num_heads, seq_len, head_dim, window_size):
    """Test sliding window attention."""
    Q, K, V = create_test_tensors(batch_size, num_heads, seq_len, head_dim)
    dO = torch.randn_like(Q)

    # Run PyTorch implementation
    attn_out, attn_dQ, attn_dK, attn_dV = run_pytorch_attention(
        Q, K, V, window_size, "sliding_window", dO
    )

    # Run Triton FlashAttention implementation
    flash_out, flash_dQ, flash_dK, flash_dV = run_triton_attention(
        Q, K, V, window_size, "sliding_window", dO
    )

    # Compare forward pass
    torch.testing.assert_close(attn_out, flash_out, rtol=1e-2, atol=1e-2)

    # Compare backward pass
    torch.testing.assert_close(attn_dV, flash_dV, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(attn_dK, flash_dK, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(attn_dQ, flash_dQ, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    # Run a quick test
    print("Testing global attention...")
    test_flash_attention_basic(
        batch_size=2, num_heads=4, seq_len=64, head_dim=64, attn_mode="global"
    )
    print("Global attention test passed!")

    print("Testing causal attention...")
    test_flash_attention_basic(
        batch_size=2, num_heads=4, seq_len=64, head_dim=64, attn_mode="causal"
    )
    print("Causal attention test passed!")

    print("Testing sliding window attention...")
    test_flash_attention_sliding_window(
        batch_size=2, num_heads=4, seq_len=128, head_dim=64, window_size=64
    )
    print("Sliding window attention test passed!")

    print("\nAll SWA tests passed!")
