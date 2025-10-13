"""
Exponential Function Example
============================

This example demonstrates how to implement an element-wise exponential function using Helion.
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
@helion.kernel()
def exp_fwd(x: torch.Tensor) -> torch.Tensor:
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
@helion.kernel()
def exp_bwd(dy: torch.Tensor, exp_x: torch.Tensor) -> torch.Tensor:
    """
    Computes the gradient of the exponential function with respect to the input tensor.

    Args:
        dy: Gradient of the output tensor
        exp_x: Saved activation from the forward pass

    Returns:
        Gradient of the input tensor
    """
    dx = torch.empty_like(exp_x)
    for tile in hl.tile(exp_x.size()):
        dx[tile] = dy[tile] * exp_x[tile]
    return dx


# %%
# Exponential Kernel
# ------------------


# %%
class ExpFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: object,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for exp."""
        y = exp_fwd(x)
        ctx.save_for_backward(y)  # type: ignore[arg-type]
        return y

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: object,
        grad_output: torch.Tensor,
    ) -> torch.Tensor:
        """Backward pass for exp."""
        (x,) = ctx.saved_tensors  # type: ignore[attr-defined]
        return exp_bwd(grad_output, x)


# %%
def exp(x: torch.Tensor) -> torch.Tensor:
    """
    Exponential with forward and backward support.

    Args:
        x: Input tensor

    Returns:
        Output tensor with the exponential of each element in the input
    """
    return ExpFunction.apply(x)  # type: ignore[no-any-return]


# %%
# Benchmark Wrapper
# -----------------


# %%
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
# ---------------------


# %%
def check(n: int) -> None:
    """
    Verify the exp kernel implementation against PyTorch's native exp function.

    Args:
        n: Size of the test tensor
    """
    x = torch.randn(n, device=DEVICE, dtype=torch.float32, requires_grad=True)
    run_example(exp, torch.exp, (x,), bwd=True)


# %%
# Main Function
# -------------


# %%
def main() -> None:
    """
    Main entry point that runs the exp kernel verification.
    """
    check(10240 * 10240)


if __name__ == "__main__":
    main()
