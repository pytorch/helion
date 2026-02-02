import sys
from pathlib import Path

# Add local kernels path for standalone execution
sys.path.insert(0, str(Path(__file__).parent.parent / "kernels"))

import torch
import time
import argparse

from flash_attention import FlashAttention


# Inlined reference implementation (no external dependency)
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


###################################### Test Function ######################################


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark PyTorch vs Triton Attention Runtime"
    )
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument(
        "--num_heads", type=int, default=4, help="Number of attention heads"
    )
    parser.add_argument(
        "--head_dim", type=int, default=128, help="Dimension of each attention head"
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=None,
        help="Window size for sliding window attention",
    )
    parser.add_argument(
        "--attn_mode",
        type=str,
        default="global",
        choices=["global", "causal", "sliding_window"],
        help="Attention mode",
    )
    parser.add_argument(
        "--seq_len", type=int, nargs="+", default=4096, help="Sequence lengths"
    )
    return parser.parse_args()


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


def test(
    batch_size,
    num_heads,
    seq_len,
    head_dim,
    window_size,
    attn_mode,
    dtype=torch.float16,
):
    """
    This function verifies that the Triton and PyTorch implementations produce the same output.
    """
    # We initialize the Q, K, V vectors from a normal distribution
    # Q, K, V already went through the first linear layers
    Q, K, V = create_test_tensors(batch_size, num_heads, seq_len, head_dim, dtype)
    dO = torch.randn_like(Q)  # for the backward pass

    # Run PyTorch implementation
    attn_out, attn_dQ, attn_dK, attn_dV = run_pytorch_attention(
        Q, K, V, window_size, attn_mode, dO
    )

    # Run Triton FlashAttention implementation
    flash_out, flash_dQ, flash_dK, flash_dV = run_triton_attention(
        Q, K, V, window_size, attn_mode, dO
    )

    # Compare the two implementations and make sure the results match
    assert torch.allclose(attn_out, flash_out, rtol=0.0, atol=1e-2)
    print("\n> Forward pass matches.")

    assert torch.allclose(attn_dV, flash_dV, rtol=0.0, atol=1e-2)
    assert torch.allclose(attn_dK, flash_dK, rtol=0.0, atol=1e-2)
    assert torch.allclose(attn_dQ, flash_dQ, rtol=0.0, atol=1e-2)
    print("> Backward pass matches.")

    print("\nAll Tests PASSED!\n")


if __name__ == "__main__":
    # Restrictions for BLOCK_SIZE = 16.

    # For Global/Causal/Sliding Window:
    # --> SEQ_LEN >= 16

    # For Sliding Window:
    # --> SEQ_LEN >= 64
    # --> 2 * BLOCK_SIZE <= WINDOW_SIZE <= SEQ_LEN

    args = parse_args()
    test(
        args.batch_size,
        args.num_heads,
        args.seq_len,
        args.head_dim,
        args.window_size,
        args.attn_mode,
    )
