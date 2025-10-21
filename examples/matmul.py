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
from typing import Any

import torch
from torch import Tensor

import helion
from helion._testing import DEVICE
from helion._testing import run_example
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Callable


# %%
@helion.kernel(
    # static_shapes=True gives a performance boost for matmuls
    static_shapes=True,
    # Disable autotung over unrolling/range_num_stages
    # tl.dot is pipelined with num_stages
    autotune_config_overrides={
        "range_unroll_factors": [0, 0],
        "range_num_stages": [0, 0],
    },
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


@helion.kernel
def matmul_bwd(
    grad_out: Tensor,  # [m, n] gradient w.r.t output
    mat1: Tensor,  # [m, k] first matrix
    mat2: Tensor,  # [k, n] second matrix
) -> tuple[Tensor, Tensor]:
    """
    Backward pass for matrix multiplication following Triton reference pattern.

    For C = A @ B, given grad_C, computes:
    - grad_A = grad_C @ B.T
    - grad_B = A.T @ grad_C

    Args:
        grad_out: Gradient w.r.t output [m, n]
        mat1: First matrix [m, k]
        mat2: Second matrix [k, n]

    Returns:
        tuple[Tensor, Tensor]: (grad_mat1, grad_mat2)
    """
    # Get all dimensions first
    m, n = grad_out.size()
    m2, k = mat1.size()
    k2, n2 = mat2.size()

    # All assertions at the top
    assert m == m2 and n == n2 and k == k2, "Size mismatch in matmul backward"

    # Declare ALL output tensors at the top before any loops
    grad_mat1 = torch.empty_like(mat1)
    grad_mat2 = torch.empty_like(mat2)

    # First loop block: compute grad_mat1 = grad_out @ mat2.T
    for tile_m1, tile_k1 in hl.tile([m, k]):
        acc1 = hl.zeros([tile_m1, tile_k1], dtype=torch.float32)
        for tile_n1 in hl.tile(n):
            # Need mat2.T: mat2 is [k, n], so mat2[tile_k, tile_n].T gives [tile_n, tile_k]
            acc1 = torch.addmm(
                acc1, grad_out[tile_m1, tile_n1], mat2[tile_k1, tile_n1].T
            )
        grad_mat1[tile_m1, tile_k1] = acc1.to(mat1.dtype)

    # Second loop block: compute grad_mat2 = mat1.T @ grad_out
    for tile_k2, tile_n2 in hl.tile([k, n]):
        acc2 = hl.zeros([tile_k2, tile_n2], dtype=torch.float32)
        for tile_m2 in hl.tile(m):
            # Need mat1.T: mat1 is [m, k], so mat1[tile_m, tile_k].T gives [tile_k, tile_m]
            acc2 = torch.addmm(
                acc2, mat1[tile_m2, tile_k2].T, grad_out[tile_m2, tile_n2]
            )
        grad_mat2[tile_k2, tile_n2] = acc2.to(mat2.dtype)

    return grad_mat1, grad_mat2


@helion.kernel
def addmm_bwd(
    grad_out: Tensor,  # [m, n] gradient w.r.t output
    bias: Tensor,  # [m, n] or broadcastable bias tensor
    mat1: Tensor,  # [m, k] first matrix
    mat2: Tensor,  # [k, n] second matrix
    alpha: float = 1.0,  # scalar multiplier for matmul
    beta: float = 1.0,  # scalar multiplier for bias
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Backward pass for addmm operation following Triton reference pattern.

    Forward: output = beta * bias + alpha * (mat1 @ mat2)

    Based on the Triton kernel analysis:
    - grad_input = beta * grad_out (with proper reduction for broadcasting)
    - grad_mat1 = alpha * (grad_out @ mat2.T)
    - grad_mat2 = alpha * (mat1.T @ grad_out)

    Args:
        grad_out: Gradient w.r.t output [m, n]
        bias: Bias tensor [m, n] (or broadcastable)
        mat1: First matrix [m, k]
        mat2: Second matrix [k, n]
        alpha: Scalar multiplier for matmul
        beta: Scalar multiplier for bias

    Returns:
        tuple[Tensor, Tensor, Tensor]: (grad_input, grad_mat1, grad_mat2)
    """
    # Get all dimensions first
    m, n = grad_out.size()
    m2, k = mat1.size()
    k2, n2 = mat2.size()

    # All assertions at the top
    assert m == m2 and n == n2 and k == k2, "Size mismatch in addmm backward"

    # Declare ALL output tensors at the top before any loops
    grad_input = torch.empty_like(bias)
    grad_mat1 = torch.empty_like(mat1)
    grad_mat2 = torch.empty_like(mat2)

    # Handle grad_input = beta * grad_out (assuming same shape for now)
    for tile_m3, tile_n3 in hl.tile([m, n]):
        grad_input[tile_m3, tile_n3] = beta * grad_out[tile_m3, tile_n3]

    # First loop block: compute grad_mat1 = alpha * (grad_out @ mat2.T)
    for tile_m1, tile_k1 in hl.tile([m, k]):
        acc1 = hl.zeros([tile_m1, tile_k1], dtype=torch.float32)
        for tile_n1 in hl.tile(n):
            acc1 = torch.addmm(
                acc1, grad_out[tile_m1, tile_n1], mat2[tile_k1, tile_n1].T
            )
        grad_mat1[tile_m1, tile_k1] = (alpha * acc1).to(mat1.dtype)

    # Second loop block: compute grad_mat2 = alpha * (mat1.T @ grad_out)
    for tile_k2, tile_n2 in hl.tile([k, n]):
        acc2 = hl.zeros([tile_k2, tile_n2], dtype=torch.float32)
        for tile_m2 in hl.tile(m):
            acc2 = torch.addmm(
                acc2, mat1[tile_m2, tile_k2].T, grad_out[tile_m2, tile_n2]
            )
        grad_mat2[tile_k2, tile_n2] = (alpha * acc2).to(mat2.dtype)

    return grad_input, grad_mat1, grad_mat2


# %%
class MatMulFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,  # noqa: ANN401
        mat1: Tensor,
        mat2: Tensor,
    ) -> Tensor:
        """Forward pass for matrix multiplication."""
        result = matmul(mat1, mat2)
        ctx.save_for_backward(mat1, mat2)
        return result

    @staticmethod
    def backward(
        ctx: Any,  # noqa: ANN401
        *grad_outputs: Tensor,
    ) -> tuple[Tensor | None, Tensor | None]:
        """Backward pass for matrix multiplication."""
        grad_out = grad_outputs[0]
        mat1, mat2 = ctx.saved_tensors
        grad_mat1, grad_mat2 = matmul_bwd(grad_out, mat1, mat2)
        return grad_mat1, grad_mat2


def matmul_autograd(mat1: Tensor, mat2: Tensor) -> Tensor:
    """Matrix multiplication with forward + backward support."""
    return MatMulFunction.apply(mat1, mat2)  # type: ignore[no-any-return]


class AddMMFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,  # noqa: ANN401
        bias: Tensor,
        mat1: Tensor,
        mat2: Tensor,
        alpha: float = 1.0,
        beta: float = 1.0,
    ) -> Tensor:
        """Forward pass for addmm operation using helion matmul with epilogue."""
        m, k = mat1.size()
        k2, n = mat2.size()
        input_broadcasted = torch.broadcast_to(bias, [m, n])

        # Define epilogue that adds bias: alpha * acc + beta * bias
        def addmm_epilogue(acc: Tensor, tile: tuple[Tensor, ...]) -> Tensor:
            return alpha * acc + beta * input_broadcasted[tile[0], tile[1]]

        result = matmul(mat1, mat2, addmm_epilogue)
        ctx.save_for_backward(bias, mat1, mat2)
        ctx.alpha = alpha
        ctx.beta = beta
        return result

    @staticmethod
    def backward(
        ctx: Any,  # noqa: ANN401
        *grad_outputs: Tensor,
    ) -> tuple[Tensor | None, Tensor | None, Tensor | None, None, None]:
        """Backward pass for addmm operation."""
        grad_out = grad_outputs[0]
        bias, mat1, mat2 = ctx.saved_tensors
        alpha = ctx.alpha
        beta = ctx.beta
        grad_input, grad_mat1, grad_mat2 = addmm_bwd(
            grad_out, bias, mat1, mat2, alpha, beta
        )
        return grad_input, grad_mat1, grad_mat2, None, None


def addmm_autograd(
    bias: Tensor, mat1: Tensor, mat2: Tensor, alpha: float = 1.0, beta: float = 1.0
) -> Tensor:
    """AddMM operation with forward + backward support."""
    return AddMMFunction.apply(bias, mat1, mat2, alpha, beta)  # type: ignore[no-any-return]


# %%
def autotune(m: int, k: int, n: int) -> None:
    """
    Runs autotuning on the matmul kernel with a ReLU epilogue and saves the best config.
    Args:
        m (int): Number of rows in matrix x.
        k (int): Number of columns in matrix x and rows in matrix y.
        n (int): Number of columns in matrix y.
    """
    x = torch.randn([m, k], device=DEVICE, dtype=torch.float16)
    y = torch.randn([k, n], device=DEVICE, dtype=torch.float16)
    bias = torch.randn([n], device=DEVICE, dtype=torch.float16)
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
    x = torch.randn([m, k], device=DEVICE, dtype=torch.float16)
    y = torch.randn([k, n], device=DEVICE, dtype=torch.float16)
    bias = torch.randn([n], device=DEVICE, dtype=torch.float16)
    bias_scalar = torch.randn([1], device=DEVICE, dtype=torch.float16)
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

    # Test matmul forward + backward pass
    print("\n\n=== MatMul Forward + Backward Pass Test ===")
    x_grad = torch.randn([m, k], device=DEVICE, dtype=torch.float16, requires_grad=True)
    y_grad = torch.randn([k, n], device=DEVICE, dtype=torch.float16, requires_grad=True)

    run_example(
        matmul_autograd,
        torch.matmul,
        (x_grad, y_grad),
        kernel_name="helion_matmul_autograd",
        baseline_name="torch",
        rtol=1e-2,
        atol=1e-2,
        bwd=True,
    )

    # Test addmm forward + backward pass
    print("\n\n=== AddMM Forward + Backward Pass Test ===")
    input_grad = torch.randn(
        [m, n], device=DEVICE, dtype=torch.float16, requires_grad=True
    )
    mat1_grad = torch.randn(
        [m, k], device=DEVICE, dtype=torch.float16, requires_grad=True
    )
    mat2_grad = torch.randn(
        [k, n], device=DEVICE, dtype=torch.float16, requires_grad=True
    )

    # Use lambda to handle the keyword argument format for torch.addmm
    run_example(
        addmm_autograd,
        lambda bias, mat1, mat2, alpha, beta: torch.addmm(
            bias, mat1, mat2, alpha=alpha, beta=beta
        ),
        (input_grad, mat1_grad, mat2_grad, 1.0, 1.0),
        kernel_name="helion_addmm_autograd",
        baseline_name="torch",
        rtol=1e-2,
        atol=1e-2,
        bwd=True,
    )

    # Test addmm forward + backward with different alpha/beta values
    print("\n\n=== AddMM Forward + Backward Test (Alpha=2.0, Beta=0.5) ===")
    run_example(
        addmm_autograd,
        lambda bias, mat1, mat2, alpha, beta: torch.addmm(
            bias, mat1, mat2, alpha=alpha, beta=beta
        ),
        (input_grad, mat1_grad, mat2_grad, 2.0, 0.5),
        kernel_name="helion_addmm_autograd_scaled",
        baseline_name="torch",
        rtol=1e-2,
        atol=1e-2,
        bwd=True,
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
        # For gemm with bias, use matmul_autograd and add bias
        return lambda: matmul_autograd(a, b) + bias
    return lambda: matmul_autograd(a, b)


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
        Callable: A callable that runs the addmm autograd function with bias.
    """
    return lambda: addmm_autograd(bias, mat1, mat2)


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
