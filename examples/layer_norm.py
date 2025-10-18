"""
Helion Layer Normalization Forward and Backward Example
=======================================================
This example demonstrates a Helion kernel implementation of 1D layer normalization
with both forward and backward passes using FP16 inputs and compares it against
PyTorch's built-in layer_norm function.
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
@helion.kernel
def layer_norm_fwd(
    x: torch.Tensor,
    normalized_shape: list[int],
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Performs 1D layer normalization on the input tensor using Helion.
    Args:
        x (torch.Tensor): Input tensor of shape [batch_size, dim], expected to be FP16.
        normalized_shape (list[int]): List containing the dimension to normalize over (should be length 1).
        weight (torch.Tensor): Learnable scale parameter of shape [dim].
        bias (torch.Tensor | None): Optional learnable bias parameter of shape [dim].
        eps (float, optional): Small value added to variance for numerical stability. Default is 1e-5.
    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - The layer-normalized output tensor of shape [batch_size, dim], in FP16.
            - Mean tensor of shape [batch_size], in FP32.
            - Reciprocal standard deviation tensor of shape [batch_size], in FP32.
    """
    m, n = x.size()
    assert weight.size(0) == n, f"weight size mismatch {weight.size(0)} != {n}"
    if bias is not None:
        assert bias.size(0) == n, f"bias size mismatch {bias.size(0)} != {n}"
    assert len(normalized_shape) == 1, (
        "Helion layer norm only supports 1D layer norm currently"
    )
    assert normalized_shape[0] == n, (
        f"normalized shape mismatch {normalized_shape[0]} != {n}"
    )
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    mean = torch.empty([m], dtype=torch.float32, device=x.device)
    rstd = torch.empty([m], dtype=torch.float32, device=x.device)

    for tile_m in hl.tile(m):
        acc = x[tile_m, :].to(torch.float32)
        # Compute mean
        mean_val = torch.sum(acc, dim=-1) / n
        # Compute variance
        centered = acc - mean_val[:, None]
        var_val = torch.sum(centered * centered, dim=-1) / n
        # Compute reciprocal standard deviation
        rstd_val = torch.rsqrt(var_val + eps)
        # Normalize
        normalized = centered * rstd_val[:, None]
        # Apply affine transformation
        if bias is not None:
            acc = normalized * (weight[:].to(torch.float32)) + (
                bias[:].to(torch.float32)
            )
        else:
            acc = normalized * (weight[:].to(torch.float32))
        out[tile_m, :] = acc.to(x.dtype)
        mean[tile_m] = mean_val
        rstd[tile_m] = rstd_val
    return out, mean, rstd


# %%
@helion.kernel
def layer_norm_bwd(
    grad_out: torch.Tensor,
    x: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    weight: torch.Tensor,
    compute_bias_grad: hl.constexpr = True,  # type: ignore[valid-type]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """
    Compute gradients for weight (dW) and optionally bias (dB) parameters.

    This kernel performs reduction across the batch dimension (M) to accumulate
    gradients for each feature dimension's weight and bias parameters.

    Args:
        grad_out: Gradient w.r.t layer norm output [M, N]
        x: Original input tensor [M, N]
        mean: Per-sample mean computed in forward pass [M]
        rstd: Per-sample reciprocal standard deviation from forward pass [M]
        weight: Weight parameter (used only for dtype/device info) [N]
        compute_bias_grad: Whether to compute bias gradient (default: True)

    Returns:
        (grad_x, grad_weight, grad_bias): Gradients for input, weight, and bias (if computed)
            grad_bias is None if compute_bias_grad is False
    """

    m_block = hl.register_block_size(x.size(0))
    n = hl.specialize(x.size(1))

    grad_x = torch.empty_like(x)
    num_blocks = (x.size(0) + m_block - 1) // m_block
    grad_weight_blocks = x.new_empty([num_blocks, n], dtype=torch.float32)
    grad_bias_blocks = x.new_empty([num_blocks, n], dtype=torch.float32)

    for mb_cta in hl.tile(x.size(0), block_size=m_block):
        grad_w_acc = weight.new_zeros(n, dtype=torch.float32)
        if compute_bias_grad:
            grad_b_acc = weight.new_zeros(n, dtype=torch.float32)
        weight_cta = weight[None, :].to(torch.float32)
        for mb in hl.tile(mb_cta.begin, mb_cta.end):
            x_mb = x[mb, :].to(torch.float32)
            dy_mb = grad_out[mb, :].to(torch.float32)
            mean_mb = mean[mb].to(torch.float32)
            rstd_mb = rstd[mb].to(torch.float32)

            x_hat = (x_mb - mean_mb[:, None]) * rstd_mb[:, None]

            grad_w_acc += torch.sum(dy_mb * x_hat, dim=0)
            if compute_bias_grad:
                grad_b_acc += torch.sum(dy_mb, dim=0)  # pyright: ignore[reportPossiblyUnboundVariable]

            wdy = weight_cta * dy_mb
            c1 = torch.sum(x_hat * wdy, dim=-1) / n
            c2 = torch.sum(wdy, dim=-1) / n
            dx = (wdy - (x_hat * c1[:, None] + c2[:, None])) * rstd_mb[:, None]
            grad_x[mb, :] = dx.to(x.dtype)

        grad_weight_blocks[mb_cta.id, :] = grad_w_acc
        if compute_bias_grad:
            grad_bias_blocks[mb_cta.id, :] = grad_b_acc  # type: ignore[index]

    grad_weight = grad_weight_blocks.sum(0).to(weight.dtype)
    if compute_bias_grad:
        grad_bias = grad_bias_blocks.sum(0).to(weight.dtype)
        return grad_x, grad_weight, grad_bias
    return grad_x, grad_weight, None


# %%
class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,  # noqa: ANN401
        x: torch.Tensor,
        normalized_shape: list[int],
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        eps: float,
    ) -> torch.Tensor:
        """Forward pass for layer normalization."""
        y, mean, rstd = layer_norm_fwd(x, normalized_shape, weight, bias, eps)
        ctx.save_for_backward(x, weight, bias, mean, rstd)  # type: ignore[arg-type]
        ctx.normalized_shape = normalized_shape  # type: ignore[attr-defined]
        return y

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any,  # noqa: ANN401
        grad_output: torch.Tensor,
    ) -> tuple[
        torch.Tensor | None, None, torch.Tensor | None, torch.Tensor | None, None
    ]:
        """Backward pass for layer normalization split into two separate kernels for efficiency."""
        grad_out = grad_output  # Use common name internally
        x, weight, bias, mean, rstd = ctx.saved_tensors  # type: ignore[attr-defined]

        # Check if bias gradient is needed
        compute_bias_grad = bias is not None

        grad_x, grad_weight, grad_bias = layer_norm_bwd(
            grad_out, x, mean, rstd, weight, compute_bias_grad
        )

        return grad_x, None, grad_weight, grad_bias, None


# %%
def layer_norm(
    x: torch.Tensor,
    normalized_shape: list[int],
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Layer normalization with forward + backward support."""
    return LayerNormFunction.apply(x, normalized_shape, weight, bias, eps)  # type: ignore[no-any-return]


# %%
# Benchmark Wrapper
# -----------------


# %%
def layer_norm_tritonbench(
    tb_op: object,
    x: torch.Tensor,
    normalized_shape: list[int],
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    eps: float = 1e-5,
) -> Callable[[], torch.Tensor]:
    """
    Wrapper for tritonbench that matches expected interface.

    Args:
        tb_op: TritonBench operator instance
        x: Input tensor
        normalized_shape: Shape to normalize over
        weight: Weight parameter
        bias: Bias parameter (optional)
        eps: Small constant for numerical stability

    Returns:
        Callable that returns normalized tensor
    """
    return lambda: layer_norm(x, normalized_shape, weight, bias, eps)


# %%
def main() -> None:
    """
    Main execution function for the layer normalization example.
    - Generates random input, weight, and bias tensors.
    - Runs the Helion layer normalization kernel and compares its output to PyTorch's
      built-in layer_norm function using the run_example utility.
    - Prints comparison results and checks for correctness within specified tolerances.
    """
    batch_size = 4096
    dim = 10240
    device = DEVICE

    # Test forward pass only
    print("\n=== Forward Pass Test ===")
    x = -2.3 + 0.5 * torch.randn([batch_size, dim], device=device, dtype=torch.float16)
    weight = torch.randn([dim], device=device, dtype=torch.float16)
    bias = torch.randn([dim], device=device, dtype=torch.float16)
    eps = 1e-4
    for b in [bias, None]:
        run_example(
            layer_norm,
            torch.nn.functional.layer_norm,
            (x, [dim], weight, b, eps),
            rtol=1e-3,
            atol=1e-3,
        )

    # Test forward + backward pass
    print("\n\n=== Forward + Backward Pass Test ===")
    x_grad = torch.randn(
        [batch_size, dim], device=device, dtype=torch.float16, requires_grad=True
    )
    weight_grad = torch.randn(
        [dim], device=device, dtype=torch.float16, requires_grad=True
    )
    bias_grad = torch.randn(
        [dim], device=device, dtype=torch.float16, requires_grad=True
    )
    for b in [bias_grad, None]:
        run_example(
            layer_norm,
            torch.nn.functional.layer_norm,
            (x_grad, [dim], weight_grad, b, eps),
            rtol=1e-3,
            atol=1e-3,
            bwd=True,
        )


# %%
if __name__ == "__main__":
    main()
