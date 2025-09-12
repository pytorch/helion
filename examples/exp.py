"""
Exponential Function Example
========================

This example demonstrates how to implement an element-wise exponential function using Helion.
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
# Exponential Kernel
# ---------------
@helion.kernel()
def exp(x: torch.Tensor) -> torch.Tensor:
    """
    Computes the exponential of all elements in the input tensor.

    Args:
        x: Input tensor

    Returns:
        Output tensor with the exponential of each element in the input
    """
    out = torch.empty_like(x)
    for tile in hl.tile(x.size()):
        out[tile] = torch.exp(x[tile])
    return out


# %%
# Benchmark Wrapper
# --------------
def exp_tritonbench(
    tb_op: object, x: torch.Tensor
) -> Callable[[], dict[str, torch.Tensor]]:
    """
    Wrapper for tritonbench that returns output in expected format.

    Args:
        tb_op: TritonBench operator instance
        x: Input tensor

    Returns:
        Callable that returns dictionary containing the output tensor
    """
    return lambda: {"output": exp(x)}


# %%
# Verification Function
# -------------------
def check(n: int) -> None:
    """
    Verify the exp kernel implementation against PyTorch's native exp function.

    Args:
        n: Size of the test tensor
    """
    x = torch.randn(n, device="cuda", dtype=torch.float32)
    run_example(exp, torch.exp, (x,))


# %%
# Main Function
# -----------
def main() -> None:
    """
    Main entry point that runs the exp kernel verification with a tensor of size 1M elements.
    """
    check(1024 * 1024)


if __name__ == "__main__":
    main()
