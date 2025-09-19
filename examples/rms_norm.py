"""
Root Mean Square Normalization Example
=================================

This example demonstrates how to implement a Root Mean Square (RMS) normalization
operation using Helion.
"""

# %%
# Imports
# -------
from __future__ import annotations

from typing import Any
from typing import Callable

import torch

import helion
from helion._testing import run_example
import helion.language as hl


# %%
# RMS Normalization Kernel
# ---------------------
@helion.kernel
def rms_norm_fwd(
    x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Performs Root Mean Square (RMS) normalization on the input tensor.

    RMS normalization normalizes by the root mean square of the elements:
    output = x / sqrt(mean(x^2) + eps) * weight

    Args:
        x: Input tensor of shape [M, N]
        weight: Scale parameter of shape [N]
        eps: Small constant for numerical stability

    Returns:
        Output tensor of shape [M, N] with RMS normalization applied
        RMS tensor of shape [M, 1] with RMS values for each element
    """
    m, n = x.size()
    assert weight.size(0) == n, f"weight size mismatch {weight.size(0)} != {n}"

    out = torch.empty_like(x)
    inv_rms = torch.empty([m, 1], dtype=x.dtype, device=x.device)

    for tile_m in hl.tile(m):
        x_tile = x[tile_m, :].to(torch.float32)

        # Compute inverse RMS: 1/sqrt(mean(x^2) + eps)
        x_squared = x_tile * x_tile
        mean_x_squared = torch.mean(x_squared, dim=-1, keepdim=True)
        inv_rms_tile = torch.rsqrt(mean_x_squared + eps)

        # Apply normalization and weight
        normalized = x_tile * inv_rms_tile
        out[tile_m, :] = (normalized * weight[:].to(torch.float32)).to(out.dtype)
        inv_rms[tile_m, :] = inv_rms_tile.to(out.dtype)

    return out, inv_rms


@helion.kernel
def rms_norm_bwd_dw(
    grad_out: torch.Tensor, x: torch.Tensor, weight: torch.Tensor, inv_rms: torch.Tensor
) -> torch.Tensor:
    """
    Compute gradients for weight (dW)

    This kernel performs reduction across the batch dimension (M) to accumulate
    gradients for each feature dimension's weight parameter.

    Args:
        grad_out: Gradient w.r.t rms norm output [M, N]
        x: Original input tensor [M, N]
        weight: Weight parameter (used only for dtype/device info) [N]
        inv_rms: Inverse RMS tensor [M, 1]

    Returns:
        grad_weight: Gradients for weight with shape [N]
    """
    m, n = x.shape

    dw = torch.empty([n], dtype=weight.dtype, device=weight.device)

    # Reduce across rows (M) inside the kernel without atomics
    rdim = hl.register_reduction_dim(m)

    for tile_n in hl.tile(n):
        rows = hl.arange(0, rdim)
        # Load slices for all rows in rdim and this tile of columns
        x_blk = x[rows, tile_n].to(torch.float32)
        dy_blk = grad_out[rows, tile_n].to(torch.float32)
        inv_rms_blk = inv_rms[rows, tile_n].to(torch.float32)

        # Compute normalized input: x_normalized = x * inv_rms
        x_normalized = x_blk * inv_rms_blk

        # Weight gradient: dw = sum_over_batch(dy * x_normalized)
        dw_tile = torch.sum(dy_blk * x_normalized, dim=0).to(weight.dtype)

        dw[tile_n] = dw_tile

    return dw


@helion.kernel
def rms_norm_bwd_dx(
    grad_out: torch.Tensor, x: torch.Tensor, weight: torch.Tensor, inv_rms: torch.Tensor
) -> torch.Tensor:
    """
    Compute gradient for input tensor (dX).

    This kernel computes per-sample gradients by performing reductions across
    the feature dimension (N) for each sample in the batch.

    Args:
        grad_out: Gradient w.r.t rms norm output [M, N]
        x: Original input tensor [M, N]
        weight: Weight parameter [N]
        inv_rms: Inverse RMS tensor [M, 1]

    Returns:
        grad_x: Gradient w.r.t input tensor, shape [M, N]
    """
    m, n = x.shape
    n = hl.specialize(n)

    grad_x = torch.empty_like(x)

    for tile_m in hl.tile(m):
        x_tile = x[tile_m, :].to(torch.float32)
        dy_tile = grad_out[tile_m, :].to(torch.float32)
        w = weight[:].to(torch.float32)
        inv_rms_tile = inv_rms[tile_m, :].to(torch.float32)

        dyw = dy_tile * w
        normed = x_tile * inv_rms_tile
        rowsum_dy_normed = (dyw * normed).sum(dim=-1, keepdim=True)
        dx = inv_rms_tile / n * (n * dyw - normed * rowsum_dy_normed)

        grad_x[tile_m, :] = dx.to(x.dtype)

    return grad_x


# %%
class RMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,  # noqa: ANN401
        x: torch.Tensor,
        weight: torch.Tensor,
        eps: float = 1e-5,
    ) -> torch.Tensor:
        """Forward pass for rms normalization."""
        y, rms = rms_norm_fwd(x, weight, eps)
        ctx.save_for_backward(x, weight)
        ctx.rms = rms  # type: ignore[attr-defined]
        return y

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any,  # noqa: ANN401
        grad_out: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, None]:
        """Backward pass for rms normalization split into two separate kernels for efficiency."""
        x, weight = ctx.saved_tensors  # type: ignore[attr-defined]
        rms = ctx.rms  # type: ignore[attr-defined]

        # First kernel: Compute gradients for weight by reducing across batch dimension (M)
        grad_weight = rms_norm_bwd_dw(grad_out, x, weight, rms)

        # Second kernel: Compute gradient for input (dx) using per-sample reductions across feature dimension (N)
        grad_x = rms_norm_bwd_dx(grad_out, x, weight, rms)

        return grad_x, grad_weight, None


# %%
def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """RMS normalization with forward + backward support."""
    return RMSNormFunction.apply(x, weight, eps)  # type: ignore[no-any-return]


# %%
# Benchmark Wrapper
# --------------
def rms_norm_tritonbench(
    tb_op: object, H: int, inp: torch.Tensor
) -> Callable[[], torch.Tensor]:
    """
    Wrapper for tritonbench that matches expected interface.

    Args:
        tb_op: TritonBench operator instance
        H: Hidden dimension size
        inp: Input tensor

    Returns:
        Callable that returns normalized tensor
    """
    weight = torch.ones(H, device=inp.device, dtype=inp.dtype, requires_grad=True)
    return lambda: rms_norm(inp, weight, eps=1e-6)


# %%
# Reference Implementation
# --------------------
def rms_norm_pytorch(
    x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5
) -> torch.Tensor:
    """
    PyTorch reference implementation of RMS normalization.

    Args:
        x: Input tensor
        weight: Scale parameter
        eps: Small constant for numerical stability

    Returns:
        Normalized tensor
    """
    input_dtype = x.dtype
    hidden_states = x.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    return weight * hidden_states.to(input_dtype)


# %%
# Verification Function
# -------------------
def check(m: int, n: int) -> None:
    """
    Verify the RMS norm kernel implementation against the PyTorch reference implementation.

    Args:
        m: First dimension of the test tensor
        n: Second dimension of the test tensor
    """
    x = torch.randn([m, n], device="cuda", dtype=torch.float16)
    weight = torch.randn([n], device="cuda", dtype=torch.float16)

    # Test forward pass only
    print("\n=== Forward Pass Test ===")
    run_example(
        rms_norm,
        rms_norm_pytorch,
        (x, weight, 1e-5),
        kernel_name="helion_fwd_kernel",
        baseline_name="torch",
        rtol=1e-3,
        atol=1e-3,
    )

    # Test forward + backward pass
    print("\n\n=== Forward + Backward Pass Test ===")
    x_grad = torch.randn([m, n], device="cuda", dtype=torch.float16, requires_grad=True)
    weight_grad = torch.randn(
        [n], device="cuda", dtype=torch.float16, requires_grad=True
    )

    run_example(
        rms_norm,
        rms_norm_pytorch,
        (x_grad, weight_grad, 1e-5),
        kernel_name="helion_autograd",
        baseline_name="torch",
        rtol=1e-3,
        atol=1e-3,
        bwd=True,
    )


# %%
# Main Function
# -----------
def main() -> None:
    """
    Main entry point that runs the RMS norm kernel verification with different tensor sizes.

    Tests with configurations:
    - 32x64
    - 128x256
    - 1024x1024
    - 2048x1024
    """
    check(32, 64)
    check(128, 256)
    check(1024, 1024)
    check(2048, 1024)


if __name__ == "__main__":
    main()
