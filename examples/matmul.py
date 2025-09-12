"""
Helion Matmul Kernel Example
============================
This example demonstrates a Helion kernel implementation of matrix multiplication
with support for a customizable epilogue function. It includes autotuning,
correctness checks against PyTorch baselines, and integration with tritonbench.
"""

# %%
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

import helion
from helion._testing import run_example
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Callable


# %%
@helion.kernel(
    # static_shapes=True gives a performance boost for matmuls
    static_shapes=True,
)
def matmul(
    x: Tensor,
    y: Tensor,
    epilogue: Callable[[Tensor, tuple[Tensor, ...]], Tensor] = lambda acc, tile: acc,
) -> Tensor:
    """
    Performs matrix multiplication of x and y with an optional epilogue function.
    Args:
        x (Tensor): Left matrix of shape [m, k].
        y (Tensor): Right matrix of shape [k, n].
        epilogue (Callable, optional): Function applied to the accumulator and tile indices
            after the matmul. Defaults to identity (no change).
    Returns:
        Tensor: Resulting matrix of shape [m, n].
    """
    m, k = x.size()
    k2, n = y.size()
    assert k == k2, f"size mismatch {k} != {k2}"
    out = torch.empty(
        [m, n], dtype=torch.promote_types(x.dtype, y.dtype), device=x.device
    )
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = epilogue(acc, (tile_m, tile_n))
    return out


# %%
def autotune(m: int, k: int, n: int) -> None:
    """
    Runs autotuning on the matmul kernel with a ReLU epilogue and saves the best config.
    Args:
        m (int): Number of rows in matrix x.
        k (int): Number of columns in matrix x and rows in matrix y.
        n (int): Number of columns in matrix y.
    """
    x = torch.randn([m, k], device="cuda", dtype=torch.float16)
    y = torch.randn([k, n], device="cuda", dtype=torch.float16)
    bias = torch.randn([n], device="cuda", dtype=torch.float16)
    args = (x, y, lambda acc, tile: torch.relu(acc + bias[tile[1]]))
    best_config = matmul.autotune(args, force=True)
    print(f"Best config: {best_config}")
    best_config.save("best_config.json")


# %%
def check(m: int, k: int, n: int) -> None:
    """
    Checks the correctness of the matmul kernel against PyTorch baselines.
    Tests:
    - Plain matmul without bias.
    - Matmul with bias added in the epilogue.
    - Matmul with a more complex epilogue applying ReLU after bias addition.
    Args:
        m (int): Number of rows in matrix x.
        k (int): Number of columns in matrix x and rows in matrix y.
        n (int): Number of columns in matrix y.
    """
    x = torch.randn([m, k], device="cuda", dtype=torch.float16)
    y = torch.randn([k, n], device="cuda", dtype=torch.float16)
    bias = torch.randn([n], device="cuda", dtype=torch.float16)
    bias_scalar = torch.randn([1], device="cuda", dtype=torch.float16)
    # Test without bias
    run_example(matmul, torch.matmul, (x, y))

    # Test for addmm with scalar bias
    def addmm(bias: Tensor, mat1: Tensor, mat2: Tensor) -> Tensor:
        m, k = mat1.size()
        k2, n = mat2.size()
        bias = torch.broadcast_to(bias, [m, n])
        return matmul(mat1, mat2, lambda acc, tile: acc + bias[tile[0], tile[1]])

    run_example(addmm, torch.addmm, (bias_scalar, x, y))

    # Test with bias
    def helion_linear(x: Tensor, y: Tensor, bias: Tensor) -> Tensor:
        return matmul(x, y, lambda acc, tile: acc + bias[tile[1]])

    def baseline_linear(x: Tensor, y: Tensor, bias: Tensor) -> Tensor:
        return torch.nn.functional.linear(x, y.T, bias)

    run_example(helion_linear, baseline_linear, (x, y, bias))

    # Test more complex epilogue
    def epilogue(acc: Tensor, tile: tuple[Tensor, ...]) -> Tensor:
        # The epilogue can use the captured bias tensor that is implicitly lifted to a kernel arg
        return torch.relu(acc + bias[tile[1]])

    def kernel_wrapper(x: Tensor, y: Tensor) -> Tensor:
        return matmul(x, y, epilogue)

    def baseline_wrapper(x: Tensor, y: Tensor) -> Tensor:
        return torch.relu(x @ y + bias)

    run_example(
        kernel_wrapper,
        baseline_wrapper,
        (x, y),
    )


# %%
def matmul_tritonbench(
    tb_op: object, a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor | None
) -> Callable:
    """
    Wrapper for tritonbench that matches its interface.
    Args:
        tb_op: TritonBench operator instance
        a (torch.Tensor): Left matrix.
        b (torch.Tensor): Right matrix.
        bias (torch.Tensor or None): Optional bias to add in the epilogue.
    Returns:
        Callable: A callable that runs the matmul kernel with or without bias.
    """
    if bias is not None:
        return lambda: matmul(a, b, lambda acc, tile: acc + bias[tile[1]])
    return lambda: matmul(a, b)


def addmm_tritonbench(
    tb_op: object, bias: Tensor, mat1: Tensor, mat2: Tensor
) -> Callable:
    """
    Wrapper for tritonbench that performs a matrix multiplication of the matrices
    `mat1` and `mat2` followed by adding `bias` to the result.
    Args:
        bias (torch.Tensor): Bias to add in the epilogue.
        mat1 (torch.Tensor): Left matrix.
        mat2 (torch.Tensor): Right matrix.
    Returns:
        Callable: A callable that runs the matmul kernel with bias.
    """
    m, k = mat1.size()
    k2, n = mat2.size()
    bias = torch.broadcast_to(bias, [m, n])
    return lambda: matmul(mat1, mat2, lambda acc, tile: acc + bias[tile[0], tile[1]])


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
