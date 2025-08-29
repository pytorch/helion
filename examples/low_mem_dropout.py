"""
Helion Low Memory Dropout Kernel
=================================
This example demonstrates Helion kernel implementations of dropout using pre-generated masks.
Since Helion doesn't yet support device-region RNG (tl.rand), we use pre-generated random masks.
"""

from __future__ import annotations

import torch

import helion
from helion._testing import run_example
import helion.language as hl


@helion.kernel()
def low_mem_dropout_helion(p: float, x: torch.Tensor) -> torch.Tensor:
    """
    Helion kernel implementing dropout with PyTorch RNG inside the kernel loop.
    This mimics the behavior of Triton's seeded_dropout with tl.rand().
    
    Args:
        p (float): Probability that an element is changed to zero
        x (torch.Tensor): Input tensor of shape [n]
        
    Returns:
        torch.Tensor: Dropout output tensor of the same shape
    """
    n = x.numel()
    output = torch.empty_like(x)
    
    # Apply dropout using tiling with RNG inside the loop
    for tile in hl.tile(n):
        x_val = x[tile]
        random_vals = torch.rand_like(x_val)
        keep_mask = random_vals > p
        # Standard inverted dropout: scale kept values by 1/(1-p)
        output[tile] = torch.where(keep_mask, x_val / (1 - p), torch.zeros_like(x_val))
    
    return output


def check() -> None:
    """
    Runs correctness checks comparing Helion dropout kernels against PyTorch's dropout.
    """
    # Test parameters
    p = 0.25
    size = 1024
    
    # Create input tensor
    x = torch.randn(size=(size,), device="cuda", dtype=torch.float32)
    
    # For validation, use a fixed seed for PyTorch dropout
    torch.manual_seed(123)
    
    kernels = {
        "helion": lambda x: low_mem_dropout_helion(p, x),
        "torch.compile": # TODO: add a torch.compile(dropout_op) impl
    }
    # TODO: set block size to full dim size so that we can test helion output accuracy vs. torch.compile(dropout_op)


def main() -> None:
    """
    Main function to run the dropout kernel correctness check.
    """
    check()


if __name__ == "__main__":
    main()