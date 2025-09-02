"""
Simplified Jagged HSTU Attention Forward Example
===============================================

This example demonstrates a simplified version of jagged HSTU attention using Helion.
"""

# %%
# Imports
# -------
from __future__ import annotations

from typing import Any

import torch

import helion
from helion._testing import run_example
import helion.language as hl

try:
    from generative_recommenders.ops.triton.triton_hstu_attention import (  # pyright: ignore[reportMissingImports]
        triton_hstu_mha,
    )

    HAS_HAMMER = True
except ImportError:
    HAS_HAMMER = False


def generate_inputs() -> dict[str, Any]:
    """Generate small inputs for HSTU attention for easier verification and testing"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    batch_size = 1024
    max_seq_len = 1024  # N parameter
    heads = 4
    head_dim = 128

    # Generate random sequence lengths
    min_seq_len = max_seq_len // 2
    seq_lengths = torch.randint(
        min_seq_len, max_seq_len + 1, (batch_size,), dtype=torch.int32, device=device
    )
    seq_offsets = torch.cat(
        [
            torch.tensor([0], dtype=torch.int32, device=device),
            torch.cumsum(seq_lengths, dim=0),
        ]
    )
    total_seq_len = int(seq_offsets[-1].item())

    # Generate tensors with ragged sequence length
    # q, k, v: [total_seq_len, heads, head_dim]
    q = torch.randn(
        (total_seq_len, heads, head_dim),
        dtype=dtype,
        device=device,
        requires_grad=True,
    )
    k = torch.randn(
        (total_seq_len, heads, head_dim),
        dtype=dtype,
        device=device,
        requires_grad=True,
    )
    v = torch.randn(
        (total_seq_len, heads, head_dim),
        dtype=dtype,
        device=device,
        requires_grad=True,
    )

    # Parameters
    alpha = 1.0 / (head_dim**0.5)  # Scaling factor
    invalid_attn_mask_type = "lower_triangular"

    # Optional parameters (set to None for simplicity)
    num_targets = None
    attn_scale = None
    attn_bias = None
    seq2_offsets = None

    # Integer parameters
    max_attn_len = 0
    contextual_seq_len = 0
    sort_by_length = False
    full_attn_size = 0

    return {
        "N": max_seq_len,
        "alpha": alpha,
        "q": q,
        "k": k,
        "v": v,
        "seq_offsets": seq_offsets,
        "invalid_attn_mask_type": invalid_attn_mask_type,
        "num_targets": num_targets,
        "attn_scale": attn_scale,
        "attn_bias": attn_bias,
        "seq2_offsets": seq2_offsets,
        "max_attn_len": max_attn_len,
        "contextual_seq_len": contextual_seq_len,
        "sort_by_length": sort_by_length,
        "full_attn_size": full_attn_size,
    }


def reference_jagged_hstu_kernel_pytorch(inputs: dict[str, Any]) -> torch.Tensor:
    """Simple PyTorch implementation of HSTU ragged attention using direct tensor slicing"""
    q = inputs["q"]
    k = inputs["k"]
    v = inputs["v"]
    seq_offsets = inputs["seq_offsets"]
    alpha = inputs["alpha"]
    N = inputs["N"]

    # Initialize output
    output = torch.zeros_like(v)

    # Scale factor
    scale = 1.0 / N

    # Compute per-batch sequence lengths
    seq_lens = seq_offsets[1:] - seq_offsets[:-1]

    q_split = torch.split(q, seq_lens.tolist(), dim=0)
    k_split = torch.split(k, seq_lens.tolist(), dim=0)
    v_split = torch.split(v, seq_lens.tolist(), dim=0)

    # Process each sequence in the batch using direct tensor slicing
    for i, (q_batch, k_batch, v_batch) in enumerate(
        zip(q_split, k_split, v_split, strict=False)
    ):
        q_batch = q_batch.transpose(0, 1)  # [heads, seq_len, head_dim]
        k_batch = k_batch.permute(1, 2, 0)  # [heads, head_dim, seq_len]
        v_batch = v_batch.transpose(0, 1)  # [heads, seq_len, head_dim]

        # Compute attention scores using batch matrix multiplication
        scores = torch.bmm(q_batch, k_batch) * alpha

        # Apply SiLU activation
        scores = (scores / (1.0 + torch.exp(-scores))) * scale

        # Apply invalid mask
        if inputs["invalid_attn_mask_type"] == "lower_triangular":
            invalid_mask = torch.tril(
                torch.ones_like(scores, dtype=torch.bool), diagonal=0
            )
            scores = torch.where(invalid_mask, scores, torch.zeros_like(scores))

        # Compute and store output
        output_batch = torch.bmm(scores, v_batch)
        output[seq_offsets[i] : seq_offsets[i + 1]] = output_batch.transpose(0, 1)

    return output


# @helion.kernel(config=helion.Config(block_sizes=[64, 16], indexing='pointer', l2_groupings=[1], loop_orders=[[2, 0, 1]], num_stages=7, num_warps=4, pid_type='flat', range_flattens=[None, None], range_multi_buffers=[None, None], range_num_stages=[0, 4], range_unroll_factors=[0, 1], range_warp_specializes=[]))
@helion.kernel()
def _helion_ragged_attention_kernel(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    alpha: float,
    invalid_mask_type: str,
    max_seq_len_tensor: torch.Tensor,
) -> torch.Tensor:
    max_seq_len = max_seq_len_tensor.numel()
    scale = 1.0 / max_seq_len

    num_heads = hl.specialize(q.size(1))
    num_batches = hl.specialize(seq_offsets.size(0) - 1)
    dimV = hl.specialize(v.size(2))

    out = torch.zeros_like(v)

    # --- Tile over batch batch, head, sequence ---
    for tile_b, tile_h, tile_q in hl.tile(
        [num_batches, num_heads, max_seq_len], block_size=[1, 1, None]
    ):
        starts = seq_offsets[tile_b.begin]
        ends = seq_offsets[tile_b.begin + 1]
        seq_len = ends - starts

        if tile_q.begin < seq_len:
            mask_q = tile_q.index < seq_len
            q_blk = q[tile_q.index + starts, tile_h.begin, :]
            acc = hl.zeros([tile_q, dimV], dtype=torch.float32)

            if invalid_mask_type == "lower_triangular":
                low = 0
                high = tile_q.end
            else:
                low = 0
                high = seq_len

            for tile_kv in hl.tile(low, high, block_size=None):
                mask_kv = tile_kv.index < seq_len
                k_blk = k[tile_kv.index + starts, tile_h.begin, :]
                v_blk = v[tile_kv.index + starts, tile_h.begin, :]

                scores = (
                    torch.nn.functional.silu(torch.matmul(q_blk, k_blk.T) * alpha)
                    * scale
                )

                if invalid_mask_type == "lower_triangular":
                    scores = torch.where(
                        (tile_q.index.unsqueeze(1) > tile_kv.index.unsqueeze(0))
                        & mask_q[:, None]
                        & mask_kv[None, :],
                        scores,
                        0.0,
                    )

                acc += torch.matmul(scores.to(v.dtype), v_blk)

            # Store result
            out[tile_q.index + starts, tile_h.begin, :] = acc

    return out


def helion_ragged_attention_function(inputs: dict[str, Any]) -> torch.Tensor:
    """
    Wrapper function for the Helion ragged attention kernel.
    """

    return _helion_ragged_attention_kernel(
        q=inputs["q"],
        k=inputs["k"],
        v=inputs["v"],
        seq_offsets=inputs["seq_offsets"],
        alpha=inputs["alpha"],
        invalid_mask_type=inputs["invalid_attn_mask_type"],
        max_seq_len_tensor=torch.empty(inputs["N"], device=inputs["q"].device),
    )


def tritonbench_hstu_attention_function(inputs: dict[str, Any]) -> torch.Tensor:
    """
    Wrapper function for the tritonbench HSTU attention implementation.

    Args:
        inputs: Dictionary containing all the input parameters

    Returns:
        Output tensor from tritonbench HSTU attention
    """
    if not HAS_HAMMER:
        # Return a dummy tensor with the same shape as expected output
        return torch.zeros_like(inputs["v"])

    return triton_hstu_mha(  # pyright: ignore[reportCallIssue,reportPossiblyUnboundVariable]
        N=inputs["N"],
        alpha=inputs["alpha"],
        q=inputs["q"],
        k=inputs["k"],
        v=inputs["v"],
        seq_offsets=inputs["seq_offsets"],
        num_targets=inputs["num_targets"],
        max_attn_len=inputs["max_attn_len"],
        contextual_seq_len=inputs["contextual_seq_len"],
        sort_by_length=inputs["sort_by_length"],
    )


def main() -> None:
    """
    Main entry point for testing the simplified jagged HSTU attention kernel.
    """
    inputs = generate_inputs()

    baselines = {
        "torch": lambda inputs: reference_jagged_hstu_kernel_pytorch(inputs),
    }
    if HAS_HAMMER:
        baselines["tritonbench"] = lambda inputs: tritonbench_hstu_attention_function(
            inputs
        )

    run_example(
        lambda inputs: helion_ragged_attention_function(inputs), baselines, (inputs,)
    )


if __name__ == "__main__":
    main()
