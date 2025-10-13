"""
Embedding Lookup Example
========================

This example demonstrates how to implement an embedding lookup operation using Helion.
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

# %%
# Embedding Kernel
# ----------------


# %%
@helion.kernel()
def embedding(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Performs embedding lookup for input indices.

    Maps indices in the input tensor to vectors from the embedding weight matrix.

    Args:
        x: Input tensor of indices of any shape
        weight: Embedding weight matrix of shape [num_embeddings, embedding_dim]

    Returns:
        Output tensor of shape [*x.shape, embedding_dim] containing the embedding vectors
    """
    x_flat = x.reshape(-1)  # collapse x into a single dimension
    _, embedding_dim = weight.size()
    out = torch.empty(
        [x_flat.size(0), embedding_dim], dtype=weight.dtype, device=weight.device
    )
    for tile_b, tile_e in hl.tile([x_flat.size(0), embedding_dim]):
        out[tile_b, tile_e] = weight[x_flat[tile_b], tile_e]
    # restore the original shape
    return out.view(*x.size(), embedding_dim)


# %%
# Benchmark Wrapper
# -----------------


# %%
def embedding_tritonbench(
    tb_op: object, V: int, D: int, inp: torch.Tensor, shared_weight: torch.Tensor
) -> Callable[[], torch.Tensor]:
    """
    Wrapper for tritonbench that matches its interface.

    Args:
        tb_op: TritonBench operator instance
        V: Vocabulary size (unused, provided for compatibility)
        D: Embedding dimension (unused, provided for compatibility)
        inp: Input tensor of indices
        shared_weight: Embedding weight matrix

    Returns:
        Callable that returns output tensor containing the embedding vectors
    """
    return lambda: embedding(inp, shared_weight)


# %%
# Main Function
# -------------


# %%
def main() -> None:
    """
    Main entry point that runs the embedding kernel verification.
    Tests with a batch of indices and an embedding table of size 16x64.
    """
    num_embeddings, embedding_dim = 16, 64
    x = torch.randint(0, num_embeddings, [256, 32], device=DEVICE, dtype=torch.int32)
    weight = torch.randn([num_embeddings, embedding_dim], device=DEVICE)
    run_example(
        embedding, torch.nn.functional.embedding, (x, weight), atol=0.0, rtol=0.0
    )


if __name__ == "__main__":
    main()
