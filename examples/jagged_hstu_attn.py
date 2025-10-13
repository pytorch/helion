"""
Simplified Jagged HSTU Attention Forward Example
================================================

This example demonstrates a simplified version of jagged HSTU attention using Helion.
"""

# %%
# Imports
# -------

# %%
from __future__ import annotations

from typing import Callable

import torch

import helion
from helion._testing import DEVICE
from helion._testing import run_example
import helion.language as hl

try:
    from generative_recommenders.ops.triton.triton_hstu_attention import (  # pyright: ignore[reportMissingImports]
        triton_hstu_mha,
    )

    HAS_HAMMER = True
except ImportError:
    HAS_HAMMER = False


# %%
# Reference Implementation
# ------------------------


# %%
def reference_jagged_hstu_kernel_pytorch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    num_targets: torch.Tensor | None,
    max_seq_len: int,
) -> torch.Tensor:
    """Simple PyTorch implementation of HSTU jagged attention"""
    # Initialize output
    output = torch.zeros_like(v)

    # Scale factor
    scale = 1.0 / max_seq_len
    alpha = 1.0 / v.size(2) ** 2

    # Compute per-batch sequence lengths
    seq_lens = seq_offsets[1:] - seq_offsets[:-1]

    q_split = torch.split(q, seq_lens.tolist(), dim=0)
    k_split = torch.split(k, seq_lens.tolist(), dim=0)
    v_split = torch.split(v, seq_lens.tolist(), dim=0)

    # Get the batches
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

        # Apply lower triangular mask (causal attention)
        invalid_mask = torch.tril(torch.ones_like(scores, dtype=torch.bool), diagonal=0)
        scores = torch.where(invalid_mask, scores, torch.zeros_like(scores))

        # Compute and store output
        output_batch = torch.bmm(scores, v_batch)
        output[seq_offsets[i] : seq_offsets[i + 1]] = output_batch.transpose(0, 1)

    return output


# %%
# Jagged HSTU Attention Kernel
# ----------------------------


# %%
@helion.kernel()
def _helion_jagged_attention_kernel(
    max_seq_len: int,
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
) -> torch.Tensor:
    """Helion implementation of HSTU jagged attention"""
    scale = 1.0 / max_seq_len
    num_heads = hl.specialize(q.size(1))
    num_batches = hl.specialize(seq_offsets.size(0) - 1)
    dimV = hl.specialize(v.size(2))

    out = torch.zeros_like(v)

    # Tile over batch, head, sequence
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

            # Causal attention: only attend to previous tokens
            for tile_kv in hl.tile(0, tile_q.end, block_size=None):
                mask_kv = tile_kv.index < seq_len
                k_blk = k[tile_kv.index + starts, tile_h.begin, :]
                v_blk = v[tile_kv.index + starts, tile_h.begin, :]

                # Compute attention scores with SiLU activation
                scores = (
                    torch.nn.functional.silu(torch.matmul(q_blk, k_blk.T) * alpha)
                    * scale
                )

                # Apply causal mask: only attend to previous positions
                scores = torch.where(
                    (tile_q.index.unsqueeze(1) > tile_kv.index.unsqueeze(0))
                    & mask_q[:, None]
                    & mask_kv[None, :],
                    scores,
                    0.0,
                )

                acc += torch.matmul(scores.to(v.dtype), v_blk)

            # Store result
            out[tile_q.index + starts, tile_h.begin, :] = acc.to(out.dtype)

    return out


# %%
# Benchmark Wrapper
# -----------------


# %%
def ragged_attention_tritonbench(
    tb_op: object,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    num_targets: torch.Tensor | None,
    max_seq_len: int,
) -> Callable[[], torch.Tensor]:
    """Wrapper function for jagged attention kernel"""
    return lambda: _helion_jagged_attention_kernel(
        max_seq_len=max_seq_len,
        alpha=1.0 / v.size(2) ** 2,
        q=q,
        k=k,
        v=v,
        seq_offsets=seq_offsets,
    )


# %%
# Testing Function
# ----------------


# %%
def test(
    batch_size: int,
    max_seq_len: int,
    heads: int,
    head_dim: int,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device | str = "cuda",
) -> None:
    """
    Test the jagged HSTU attention kernel implementation.

    Args:
        batch_size: Number of sequences in the batch
        max_seq_len: Maximum sequence length
        heads: Number of attention heads
        head_dim: Dimension of each attention head
        dtype: Data type for the tensors
        device: Device to run the test on
    """
    device = torch.device(device)

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

    baselines = {
        "torch": reference_jagged_hstu_kernel_pytorch,
    }
    if HAS_HAMMER:

        def _triton_hstu_mha(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            seq_offsets: torch.Tensor,
            num_targets: torch.Tensor | None,
            max_seq_len: int,
        ) -> torch.Tensor:
            return triton_hstu_mha(  # pyright: ignore[reportPossiblyUnboundVariable,reportCallIssue]
                max_seq_len,
                alpha=1.0 / v.size(2) ** 2,
                q=q,
                k=k,
                v=v,
                seq_offsets=seq_offsets,
                num_targets=num_targets,
                max_attn_len=0,
                contextual_seq_len=0,
            )

        baselines["tritonbench"] = _triton_hstu_mha

    run_example(
        lambda *args: ragged_attention_tritonbench(None, *args)(),
        baselines,
        (q, k, v, seq_offsets, None, max_seq_len),
    )


# %%
# Main Function
# -------------


# %%
def main() -> None:
    """
    Main entry point for testing the simplified jagged HSTU attention kernel.
    """
    test(
        batch_size=1024,
        max_seq_len=1024,
        heads=4,
        head_dim=128,
        dtype=torch.bfloat16,
        device=DEVICE,
    )


if __name__ == "__main__":
    main()
