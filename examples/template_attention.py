"""
Template Attention Example
=========================

This code implements a templated attention kernel using Helion that mirrors the Triton template attention implementation.
It demonstrates masked causal attention with configurable parameters and optimization features.
"""

# %%
# Imports
# -------
from __future__ import annotations

from typing import Callable

import torch

import helion
from helion._testing import run_example
import helion.language as hl


# %%
# Template Attention Kernel with Causal Masking
# --------------------------------------------
@helion.kernel(config=helion.Config(block_sizes=[128, 64]))
def template_attention_causal(
    q_in: torch.Tensor,
    k_in: torch.Tensor,
    v_in: torch.Tensor,
) -> tuple[torch.Tensor]:
    """
    Computes scaled dot-product attention with causal masking.

    Based on the Triton template attention implementation, this kernel:
    - Uses causal masking (queries can only attend to keys at or before their position)
    - Implements flash attention algorithm for memory efficiency
    - Uses online softmax for numerical stability

    Args:
        q_in: Query tensor of shape [batch, heads, seq_len, D]
        k_in: Key tensor of shape [batch, heads, seq_len, D]
        v_in: Value tensor of shape [batch, heads, seq_len, D]

    Returns:
        Output tensor of shape [batch, heads, seq_len, D]
    """
    M = q_in.size(-2)  # seq_len
    N = k_in.size(-2)  # seq_len
    assert v_in.size(-2) == N
    D = hl.specialize(q_in.size(-1))
    assert D == k_in.size(-1) == v_in.size(-1)

    # Reshape to [batch*heads, seq_len, D]
    q_view = q_in.reshape([-1, M, D])
    v_view = v_in.reshape([-1, N, D])
    k_view = k_in.reshape([-1, N, D]).transpose(1, 2)  # [batch*heads, D, seq_len]

    out = torch.empty_like(q_view)

    # Scale factor (no exp2 conversion yet)
    # template_attention does not use 1.0 / math.sqrt(D)
    qk_scale = 1.0

    # Process in tiles: [batch*heads, seq_len_q]
    block_size_m = hl.register_block_size(M)
    block_size_n = hl.register_block_size(N)
    for tile_m, tile_b in hl.tile(
        [M, q_view.size(0)], block_size=[block_size_m, 1]
    ):  # BLOCK_M = 128
        # Initialize flash attention statistics
        m_i = hl.full([tile_b, tile_m], float("-inf"), dtype=torch.float32)
        l_i = hl.zeros([tile_b, tile_m], dtype=torch.float32)
        acc = hl.zeros([tile_b, tile_m, D], dtype=torch.float32)

        # Load query block
        q = q_view[tile_b, tile_m, :] * qk_scale

        # Iterate over key/value blocks
        for tile_n in hl.tile(N, block_size=block_size_n):  # BLOCK_N = 64
            # Load key and value
            k = k_view[tile_b, :, tile_n]  # [batch, D, block_n]
            v = v_view[tile_b, tile_n, :]  # [batch, block_n, D]

            # Compute attention scores: [batch, block_m, block_n]
            qk = torch.bmm(q, k) * qk_scale

            # Apply causal mask
            # Create indices for this tile
            q_indices = (tile_m.begin + hl.arange(tile_m.block_size))[:, None]
            k_indices = (tile_n.begin + hl.arange(tile_n.block_size))[None, :]

            # Causal condition: query_pos >= key_pos (can attend to current and previous)
            causal_mask = q_indices >= k_indices

            # Boundary mask
            tmp0 = hl.full([1], 1024, torch.int64)
            tmp1 = (q_indices) <= tmp0
            tmp2 = (k_indices) <= tmp0
            tmp3 = tmp1 & tmp2
            mask = tmp3 | causal_mask

            # Apply mask by setting invalid positions to -inf
            qk = torch.where(mask, qk, float("-inf"))

            # Online softmax (flash attention)
            row_max = torch.amax(qk, dim=-1)  # Row max
            m_i_new = torch.maximum(m_i, row_max)

            # Compute exponentials
            alpha = torch.exp(m_i - m_i_new)
            p = torch.exp(qk - m_i_new[:, :, None])

            # Update statistics
            l_i_new = l_i * alpha + torch.sum(p, dim=-1)

            # Update accumulator
            acc = acc * alpha[:, :, None]
            p = p.to(v.dtype)
            acc = torch.baddbmm(acc, p, v)

            # Update running statistics
            l_i = l_i_new
            m_i = m_i_new

        # Normalize and store output
        acc = acc / l_i[:, :, None]
        out[tile_b, tile_m, :] = acc.to(out.dtype)

    return (out.view(q_in.size()),)


# %%
# Template Attention with exp2 optimization
# ---------------------------------------
@helion.kernel(config=helion.Config(block_sizes=[128, 64]))
def template_attention_causal_exp2(
    q_in: torch.Tensor,
    k_in: torch.Tensor,
    v_in: torch.Tensor,
) -> tuple[torch.Tensor]:
    """
    Optimized version using exp2 for better performance, matching triton_tem_fused_with_exp2.

    This version includes optimizations from the Triton implementation:
    - Uses exp2 instead of exp for better numerical properties
    - Scales by log2(e) to convert between bases
    - Includes compiler optimization hints

    Args:
        q_in: Query tensor of shape [batch, heads, seq_len, D]
        k_in: Key tensor of shape [batch, heads, seq_len, D]
        v_in: Value tensor of shape [batch, heads, seq_len, D]

    Returns:
        Output tensor of shape [batch, heads, seq_len, D]
    """
    # SCORE_MOD_IS_LINEAR: Is the score modifier linear? If so, we can lift the
    # change of base out of the loop
    # ROWS_GUARANTEED_SAFE: Is it guaranteed that at least one value in each row
    # is not masked out? If so, we can skip an extra safety check
    SCORE_MOD_IS_LINEAR = False
    ROWS_GUARANTEED_SAFE = False

    M = q_in.size(-2)  # seq_len
    N = k_in.size(-2)  # seq_len
    assert v_in.size(-2) == N
    D = hl.specialize(q_in.size(-1))
    assert D == k_in.size(-1) == v_in.size(-1)

    # Reshape to [batch*heads, seq_len, D]
    q_view = q_in.reshape([-1, M, D])
    v_view = v_in.reshape([-1, N, D])
    k_view = k_in.reshape([-1, N, D]).transpose(1, 2)  # [batch*heads, D, seq_len]

    out = torch.empty_like(q_view)

    # Scale by log_2(e) for exp2 optimization
    # template_attention does not use 1.0 / math.sqrt(D)
    qk_scale = 1.0

    # Process in tiles: [batch*heads, seq_len_q]
    block_size_m = hl.register_block_size(M)
    block_size_n = hl.register_block_size(N)
    for tile_m, tile_b in hl.tile(
        [M, q_view.size(0)], block_size=[block_size_m, 1]
    ):  # BLOCK_M = 128
        # Initialize flash attention statistics
        m_i = hl.full([tile_b, tile_m], float("-inf"), dtype=torch.float32)
        l_i = hl.zeros([tile_b, tile_m], dtype=torch.float32)
        acc = hl.zeros([tile_b, tile_m, D], dtype=torch.float32)

        # Load and scale query by log_2(e)
        if SCORE_MOD_IS_LINEAR:
            qk_scale *= 1.44269504
        q = q_view[tile_b, tile_m, :] * qk_scale

        # Iterate over key/value blocks
        for tile_n in hl.tile(N, block_size=block_size_n):  # BLOCK_N = 64
            # Load key and value
            k = k_view[tile_b, :, tile_n]  # [batch, D, block_n]
            v = v_view[tile_b, tile_n, :]  # [batch, block_n, D]

            # Compute attention scores: [batch, block_m, block_n]
            qk = torch.bmm(q, k)

            # Apply causal mask
            # Create indices for this tile
            q_indices = (tile_m.begin + hl.arange(tile_m.block_size))[:, None]
            k_indices = (tile_n.begin + hl.arange(tile_n.block_size))[None, :]

            # Causal condition: query_pos >= key_pos (can attend to current and previous)
            causal_mask = q_indices >= k_indices

            # Boundary mask
            tmp0 = hl.full([1], 1024, torch.int64)
            tmp1 = (q_indices) <= tmp0
            tmp2 = (k_indices) <= tmp0
            tmp3 = tmp1 & tmp2
            mask = tmp3 | causal_mask

            # Apply mask by setting invalid positions to -inf
            qk = torch.where(mask, qk, float("-inf"))
            if not SCORE_MOD_IS_LINEAR:
                qk *= 1.44269504

            # Online softmax with exp2 (flash attention)
            row_max = torch.amax(qk, dim=-1)  # Row max
            m_i_new = torch.maximum(m_i, row_max)
            masked_out_rows = m_i_new == float("-inf")

            # Compute exponentials using exp2
            alpha = torch.exp2(m_i - m_i_new)
            p = torch.exp2(qk - m_i_new[:, :, None])
            if not ROWS_GUARANTEED_SAFE:
                alpha = torch.where(masked_out_rows, 0, alpha)
                p = torch.where(masked_out_rows[:, :, None], 0, p)

            # Update statistics
            l_i_new = l_i * alpha + torch.sum(p, dim=-1)

            # Update accumulator
            acc = acc * alpha[:, :, None]
            p = p.to(v.dtype)
            acc = torch.baddbmm(acc, p, v)

            # Update running statistics
            l_i = l_i_new
            m_i = m_i_new

        # Normalize and store output
        acc = acc / l_i[:, :, None]
        out[tile_b, tile_m, :] = acc.to(out.dtype)

    return (out.view(q_in.size()),)


# %%
# Testing Functions
# --------------
def ref_causal_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dtype: torch.dtype = torch.float16,
) -> tuple[torch.Tensor]:
    """Reference causal attention implementation with boundary mask"""
    # scale = 1.0 / math.sqrt(D)
    scale = 1.0
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale

    # Apply combined boundary and causal mask (matching lines 306-315 in triton_attention.py)
    seq_len = q.size(-2)

    # Create index tensors for query and key positions
    q_indices = torch.arange(seq_len, device=q.device)[:, None]
    k_indices = torch.arange(seq_len, device=q.device)[None, :]

    # Boundary condition: both query and key must be <= 1024
    boundary_threshold = 1024
    tmp1 = q_indices <= boundary_threshold
    tmp2 = k_indices <= boundary_threshold
    tmp3 = tmp1 & tmp2

    # Causal condition: query_pos >= key_pos
    tmp4 = q_indices >= k_indices

    # Combined mask: (both within boundary) OR (causal condition satisfied)
    tmp5 = tmp3 | tmp4

    # Apply mask by setting invalid positions to -inf
    scores = scores.masked_fill(~tmp5, float("-inf"))

    attn_weights = torch.softmax(scores, dim=-1).to(dtype)
    return (torch.matmul(attn_weights, v),)


def test_template_attention(
    batch_size: int,
    num_heads: int,
    seq_len: int,
    D: int,
    dtype: torch.dtype = torch.float16,
    device: torch.device | str = "cuda",
) -> None:
    """
    Test the template attention kernels against reference implementations.

    Args:
        batch_size: Batch size
        num_heads: Number of attention heads
        seq_len: Sequence length
        D: Head dimension
        dtype: Data type for tensors
        device: Device to run on
    """
    # Create test tensors
    q, k, v = [
        torch.randn((batch_size, num_heads, seq_len, D), dtype=dtype, device=device)
        for _ in range(3)
    ]

    # Create wrappers that extract only the first element of the tuple
    def baseline_template_attention_wrapper(
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """Wrapper that extracts first element from template_attention_causal output"""
        return ref_causal_attention(q, k, v)[0]

    def template_attention_causal_wrapper(
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """Wrapper that extracts first element from template_attention_causal output"""
        return template_attention_causal(q, k, v)[0]

    def template_attention_causal_exp2_wrapper(
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """Wrapper that extracts first element from template_attention_causal_exp2 output"""
        return template_attention_causal_exp2(q, k, v)[0]

    print("Testing template_attention_causal:")
    run_example(
        template_attention_causal_wrapper,
        baseline_template_attention_wrapper,
        (q, k, v),
    )

    print("\nTesting template_attention_causal_exp2:")
    run_example(
        template_attention_causal_exp2_wrapper,
        baseline_template_attention_wrapper,
        (q, k, v),
    )


# %%
# Tritonbench Integration
# -----------------------
def template_attention_tritonbench(
    tb_op: object, p1: torch.Tensor, p2: torch.Tensor, p3: torch.Tensor
) -> Callable:
    return lambda: template_attention_causal_exp2(p1, p2, p3)


# %%
# Main Function
# -----------
def main() -> None:
    """
    Main entry point that runs template attention tests.
    Tests with parameters similar to the Triton benchmark: 16 batch*heads, 4096 sequence length, 64 head dimension.
    """
    # Test with smaller sizes first for debugging
    test_template_attention(2, 8, 512, 64)

    # Test with tritonbench parameters: batch=16, heads=16, seq_len=4096, D=64
    test_template_attention(16, 16, 4096, 64)


if __name__ == "__main__":
    main()
