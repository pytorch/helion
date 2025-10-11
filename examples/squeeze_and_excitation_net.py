"""
Helion Matmul Kernel Example
============================
This example demonstrates a Helion kernel implementation of queeze and Excitation
Net as those used in https://arxiv.org/abs/1709.01507.
"""

# %%
from __future__ import annotations

import torch
from torch import Tensor

import helion
from helion._testing import run_example
import helion.language as hl


# %%
@helion.kernel(
    # static_shapes=True gives a performance boost for matmuls
    static_shapes=True,
)
def squeeze_and_excitation_net_fwd(x: Tensor, a: Tensor, b: Tensor) -> Tensor:
    """
    Performs torch.sigmoid(torch.relu((x @ a)) @ b)
    Args:
        x: 2D tensor of shape [m, n].
        a: 2D tensor of shape [n, k].
        b: 2D tensor of shape [k, n].
    Returns:
        Tensor: Resulting matrix of shape [m, n].
    """
    m, n = x.size()
    k = a.size(1)
    
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            # Compute x @ a for this tile, apply relu, then multiply by b
            partial_xa = x[tile_m, :] @ a[:, tile_k]
            relu_xa = torch.relu(partial_xa)
            acc += relu_xa @ b[tile_k, tile_n]
        
        out[tile_m, tile_n] = torch.sigmoid(acc)
    
    return out


# %%
# Reference Implementation
# --------------------
def squeeze_and_excitation_net_pytorch(
    x: torch.Tensor, a: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:
    """
    PyTorch reference implementation of squeeze_and_excitation_net.

    Args:
        x, a, b: Input tensors

    Returns:
        tensor of torch.sigmoid(torch.relu((x @ a)) @ b)
    """
    return torch.sigmoid(torch.relu((x @ a)) @ b)


# %%
def check(m: int, k: int, n: int) -> None:
    """
    Checks the correctness of the def squeeze_and_excitation_net kernel against PyTorch baselines.
    Args:
        m (int): Number of rows in matrix x.
        k (int): Number of columns in matrix x and rows in matrix y.
        n (int): Number of columns in matrix y.
    """
    x = torch.randn([m, n], device="cuda", dtype=torch.float16)
    a = torch.randn([n, k], device="cuda", dtype=torch.float16)
    b = torch.randn([k, n], device="cuda", dtype=torch.float16)
    run_example(squeeze_and_excitation_net_fwd, squeeze_and_excitation_net_pytorch, (x, a, b))


# %%
def main() -> None:
    """
    Main function to run autotuning (commented out) and correctness checks.
    """
    # autotune(1024, 1024, 1024)
    check(1024, 1024, 1024)


# %%
if __name__ == "__main__":
    main()
