"""
Matrix Multiplication with Split-K using Helion
===============================================
This example demonstrates a Helion kernel for matrix multiplication that uses a split-K
strategy to improve parallelism and performance. It supports an optional epilogue function
for post-processing the accumulator, such as adding bias.
The example includes:

- The Helion kernel implementation with static shapes for performance.

- A check function to validate correctness against PyTorch baselines.

- A wrapper for integration with tritonbench.
"""

# %%
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import helion
from helion._testing import DEVICE
from helion._testing import run_example
from helion.autotuner import PowerOfTwoFragment
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Callable


# %%
@helion.kernel(static_shapes=True)
def matmul_split_k(
    x: torch.Tensor,
    y: torch.Tensor,
    epilogue: Callable[
        [torch.Tensor, tuple[torch.Tensor, ...]], torch.Tensor
    ] = lambda acc, tile: acc,
) -> torch.Tensor:
    """
    Matrix multiplication kernel using split-K parallelism.
    This kernel splits the reduction (K) dimension into multiple fragments to improve
    parallelism and performance, especially for large K. The results from each split
    are accumulated atomically into the output tensor. An optional epilogue function
    can be applied to the accumulator, e.g., for adding bias.
    Args:
        x (torch.Tensor): Left input matrix of shape [m, k].
        y (torch.Tensor): Right input matrix of shape [k, n].
        epilogue (Callable, optional): Function applied to the accumulator and tile indices
            after the matmul. Defaults to identity (no change).
    Returns:
        torch.Tensor: Resulting matrix of shape [m, n].
    """
    m, k = x.size()
    k2, n = y.size()
    assert k == k2, f"size mismatch {k} != {k2}"
    out = torch.zeros(
        [m, n], dtype=torch.promote_types(x.dtype, y.dtype), device=x.device
    )
    split_k = hl.register_tunable("split_k", PowerOfTwoFragment(1, 256))
    k_block = helion.next_power_of_2(helion.cdiv(k, split_k))
    for tile_m, tile_n, outer_k in hl.tile([m, n, k], block_size=[None, None, k_block]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for inner_k in hl.tile(outer_k.begin, outer_k.end):
            acc = torch.addmm(acc, x[tile_m, inner_k], y[inner_k, tile_n])
        # Apply epilogue only on the first k-split iteration
        if outer_k.begin == 0:
            acc = epilogue(acc, (tile_m, tile_n))
        hl.atomic_add(out, [tile_m, tile_n], acc)
    return out


# %%
def check(m: int, k: int, n: int) -> None:
    """
    Validates the matmul_split_k kernel against PyTorch's matmul and linear functions.
    Runs two tests:
    - Without bias: compares to torch.matmul.
    - With bias: compares to torch.nn.functional.linear.
    Args:
        m (int): Number of rows in the left input matrix.
        k (int): Shared dimension.
        n (int): Number of columns in the right input matrix.
    """
    x = torch.randn([m, k], device=DEVICE, dtype=torch.float16)
    y = torch.randn([k, n], device=DEVICE, dtype=torch.float16)
    # Test without bias
    kernel_no_bias = lambda x, y: matmul_split_k(x, y)  # noqa: E731
    expected_no_bias = lambda x, y: torch.matmul(x, y)  # noqa: E731
    run_example(kernel_no_bias, expected_no_bias, (x, y), atol=1)
    # Test with bias using closure approach
    bias = torch.randn([n], device=DEVICE, dtype=torch.float16)
    kernel_with_bias = lambda x, y: matmul_split_k(  # noqa: E731
        x, y, epilogue=lambda acc, tile: acc + bias[tile[1]]
    )
    expected_with_bias = lambda x, y: torch.nn.functional.linear(x, y.T, bias)  # noqa: E731
    run_example(kernel_with_bias, expected_with_bias, (x, y), atol=1)


# %%
def matmul_split_k_tritonbench(
    tb_op: object, a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor | None
) -> Callable:
    """
    Wrapper for tritonbench that matches its interface.
    Args:
        tb_op: TritonBench operator instance
        a (torch.Tensor): Left input matrix.
        b (torch.Tensor): Right input matrix.
        bias (torch.Tensor or None): Optional bias to add in the epilogue.
    Returns:
        Callable: A callable that runs the matmul_split_k kernel with or without bias.
    """
    if bias is not None:
        return lambda: matmul_split_k(
            a, b, epilogue=lambda acc, tile: acc + bias[tile[1]]
        )
    return lambda: matmul_split_k(a, b)


# %%
def main() -> None:
    """
    Main function to run the matmul_split_k kernel correctness check with example input size.
    """
    check(64, 32768, 64)


# %%
if __name__ == "__main__":
    main()
