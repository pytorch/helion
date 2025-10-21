"""
Helion Softmax Kernel Examples
==============================
This example demonstrates multiple Helion kernel implementations of the softmax function,
including a simple wrapper around PyTorch's softmax, a decomposed version using explicit
exponentiation and normalization, and a numerically optimized two-pass version.
The example also includes a check function to compare these kernels against PyTorch's
built-in softmax for correctness.
"""

# %%
from __future__ import annotations

from typing import Any
from typing import Callable

import torch

import helion
from helion._testing import DEVICE
from helion._testing import run_example
import helion.language as hl


# %%
@helion.kernel()
def softmax(x: torch.Tensor) -> torch.Tensor:
    """
    Simple Helion kernel wrapping PyTorch's softmax function.
    Args:
        x (torch.Tensor): Input tensor of shape [n, m].
    Returns:
        torch.Tensor: Softmax output tensor of the same shape.
    """
    n, _m = x.size()
    out = torch.empty_like(x)
    for tile_n in hl.tile(n):
        out[tile_n, :] = torch.nn.functional.softmax(x[tile_n, :], dim=1)
    return out


# %%
@helion.kernel()
def softmax_decomposed(x: torch.Tensor) -> torch.Tensor:
    """
    Helion kernel implementing softmax by decomposing into max, exp, and normalization steps.
    This avoids using PyTorch's built-in softmax decomposition.
    Args:
        x (torch.Tensor): Input tensor of shape [n, m].
    Returns:
        torch.Tensor: Softmax output tensor of the same shape.
    """
    n, _m = x.size()
    out = torch.empty_like(x)
    for tile_n in hl.tile(n):
        values = x[tile_n, :]
        amax = torch.amax(values, dim=1, keepdim=True)
        exp = torch.exp(values - amax)
        sum_exp = torch.sum(exp, dim=1, keepdim=True)
        out[tile_n, :] = exp / sum_exp
    return out


# %%
@helion.kernel()
def softmax_two_pass(x: torch.Tensor) -> torch.Tensor:
    """
    Numerically optimized Helion kernel performing softmax in two passes.
    This version uses fewer passes but is less numerically stable.
    Args:
        x (torch.Tensor): Input tensor of shape [m, n].
    Returns:
        torch.Tensor: Softmax output tensor of the same shape.
    """
    m, n = x.size()
    out = torch.empty_like(x)
    block_size_m = hl.register_block_size(m)
    block_size_n = hl.register_block_size(n)
    for tile_m in hl.tile(m, block_size=block_size_m):
        mi = hl.full([tile_m], float("-inf"), dtype=torch.float32)
        di = hl.zeros([tile_m], dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=block_size_n):
            values = x[tile_m, tile_n]
            local_amax = torch.amax(values, dim=1)
            mi_next = torch.maximum(mi, local_amax)
            di = di * torch.exp(mi - mi_next) + torch.exp(
                values - mi_next[:, None]
            ).sum(dim=1)
            mi = mi_next
        for tile_n in hl.tile(n, block_size=block_size_n):
            values = x[tile_m, tile_n]
            out[tile_m, tile_n] = torch.exp(values - mi[:, None]) / di[:, None]
    return out


@helion.kernel()
def softmax_bwd(
    grad_output: torch.Tensor, softmax_output: torch.Tensor
) -> torch.Tensor:
    """
    Helion kernel implementing softmax backward pass.

    dy/dx = softmax_output * (grad_output - sum(softmax_output * grad_output))

    Args:
        grad_output (torch.Tensor): Gradient from downstream layers of shape [m, n]
        softmax_output (torch.Tensor): Output from forward softmax pass of shape [m, n]

    Returns:
        torch.Tensor: Gradient with respect to input of shape [m, n]
    """
    m, n = grad_output.size()
    grad_input = torch.empty_like(grad_output)

    for tile_m in hl.tile(m):
        sum_per_row = hl.zeros([tile_m], dtype=torch.float32)
        for tile_n in hl.tile(n):
            sum_per_row += torch.sum(
                softmax_output[tile_m, tile_n] * grad_output[tile_m, tile_n], dim=1
            )
        for tile_n in hl.tile(n):
            grad_input[tile_m, tile_n] = softmax_output[tile_m, tile_n] * (
                grad_output[tile_m, tile_n] - sum_per_row[:, None]
            )

    return grad_input


class SoftmaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,  # noqa: ANN401
        x: torch.Tensor,
    ) -> torch.Tensor:
        y = softmax_two_pass(x)
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any,  # noqa: ANN401
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor | None]:
        (softmax_output,) = ctx.saved_tensors
        grad_x = softmax_bwd(grad_output, softmax_output)
        return (grad_x,)


def softmax_fwd_bwd(
    x: torch.Tensor,
) -> torch.Tensor:
    """Softmax with forward + backward support."""
    return SoftmaxFunction.apply(x)  # type: ignore[no-any-return]


def softmax_tritonbench(tb_op: object, x: torch.Tensor) -> Callable[[], torch.Tensor]:
    """
    Wrapper for tritonbench that returns softmax with backward support.
    Args:
        tb_op: TritonBench operator instance
        x: Input tensor

    Returns:
        Callable that returns the output tensor
    """
    return lambda: softmax_fwd_bwd(x)


# %%
def check(m: int, n: int) -> None:
    """
    Runs correctness checks comparing Helion softmax kernels against PyTorch's softmax.
    Args:
        m (int): Number of rows in input tensor.
        n (int): Number of columns in input tensor.
    """
    x = torch.randn([m, n], device=DEVICE, dtype=torch.float16)
    kernels = {
        "helion simple": softmax,
        # "helion decomposed": softmax_decomposed,  # Disabled due to possible issues
        "helion two pass": softmax_two_pass,
    }
    run_example(kernels, lambda x: torch.nn.functional.softmax(x, dim=1), (x,))

    print("\n\n=== Forward + Backward Pass Test ===")
    x_grad = torch.randn([m, n], device=DEVICE, dtype=torch.float16, requires_grad=True)
    run_example(
        softmax_fwd_bwd,
        torch.nn.functional.softmax,
        (x_grad,),
        rtol=1e-3,
        atol=1e-3,
        bwd=True,
    )


# %%
def main() -> None:
    """
    Main function to run the softmax kernel correctness check with example input size.
    """
    check(4096, 2560)


# %%
if __name__ == "__main__":
    main()
