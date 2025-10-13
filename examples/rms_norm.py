"""
Root Mean Square Normalization Example
======================================

This example demonstrates how to implement a Root Mean Square (RMS) normalization
operation using Helion.
"""

# %%
# Imports
# -------

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
# RMS Normalization Kernel
# ------------------------


# %%
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
    inv_rms = torch.empty([m], dtype=x.dtype, device=x.device)

    for tile_m in hl.tile(m):
        x_tile = x[tile_m, :].to(torch.float32)

        # Compute inverse RMS: 1/sqrt(mean(x^2) + eps)
        x_squared = x_tile * x_tile
        mean_x_squared = torch.mean(x_squared, dim=-1)
        inv_rms_tile = torch.rsqrt(mean_x_squared + eps)

        # Apply normalization and weight
        normalized = x_tile * inv_rms_tile[:, None]
        out[tile_m, :] = (normalized * weight[:].to(torch.float32)).to(out.dtype)
        inv_rms[tile_m] = inv_rms_tile.to(out.dtype)

    return out, inv_rms.reshape(-1, 1)


@helion.kernel
def rms_norm_bwd(
    grad_out: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    rsqrt: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute gradient for input tensor (dX) and weights (dW).

    This kernel computes per-sample gradients by performing reductions across
    the feature dimension (N) for each sample in the batch and across the batches
    in a split fashion.

    Args:
        grad_out: Gradient w.r.t rms norm output [M, N]
        x: Original input tensor [M, N]
        weight: Weight parameter [N]
        inv_rms: Inverse RMS tensor [M, 1]

    Returns:
        grad_x: Gradient w.r.t input tensor, shape [M, N]
        grad_weight: Gradient w.r.t eight tensor, shape [N]
    """
    m_block = hl.register_block_size(x.size(0))
    grad_x = torch.empty_like(x)
    grad_weight = x.new_empty(
        [(x.size(0) + m_block - 1) // m_block, *weight.shape], dtype=torch.float32
    )
    weight_shape = hl.specialize(weight.size(0))
    for mb_cta in hl.tile(x.size(0), block_size=m_block):
        grad_w_m = weight.new_zeros(weight_shape, dtype=torch.float32)
        for mb in hl.tile(mb_cta.begin, mb_cta.end):
            x_m = x[mb, :].to(torch.float32)
            do_m = grad_out[mb, :].to(torch.float32)
            rsqrt_m = rsqrt[mb, :].to(torch.float32)
            grad_w_m += (x_m * do_m * rsqrt_m).sum(0)
            w_m = weight[None, :].to(torch.float32)
            grad_x[mb, :] = (
                w_m * do_m * rsqrt_m
                - x_m * rsqrt_m**3 * (w_m * do_m * x_m).mean(-1)[:, None]
            ).to(x.dtype)
        grad_weight[mb_cta.id, :] = grad_w_m
    return grad_x, grad_weight.sum(0).to(weight.dtype)


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
        grad_x, grad_weight = rms_norm_bwd(grad_out, x, weight, rms)
        return grad_x, grad_weight, None


# %%
def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """RMS normalization with forward + backward support."""
    return RMSNormFunction.apply(x, weight, eps)  # type: ignore[no-any-return]


# %%
# Benchmark Wrapper
# -----------------


# %%
def rms_norm_tritonbench(
    tb_op: object, H: int, inp: torch.Tensor, weight: torch.Tensor
) -> Callable[[], torch.Tensor]:
    """
    Wrapper for tritonbench that matches expected interface.

    Args:
        tb_op: TritonBench operator instance
        H: Hidden dimension size
        inp: Input tensor
        weight: Weight tensor

    Returns:
        Callable that returns normalized tensor
    """
    return lambda: rms_norm(inp, weight, eps=1e-6)


# %%
# Reference Implementation
# ------------------------


# %%
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
# ---------------------


# %%
def check(m: int, n: int) -> None:
    """
    Verify the RMS norm kernel implementation against the PyTorch reference implementation.

    Args:
        m: First dimension of the test tensor
        n: Second dimension of the test tensor
    """
    x = torch.randn([m, n], device=DEVICE, dtype=torch.float16)
    weight = torch.randn([n], device=DEVICE, dtype=torch.float16)

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
    x_grad = torch.randn([m, n], device=DEVICE, dtype=torch.float16, requires_grad=True)
    weight_grad = torch.randn(
        [n], device=DEVICE, dtype=torch.float16, requires_grad=True
    )

    run_example(
        rms_norm,
        rms_norm_pytorch,
        (x_grad, weight_grad, 1e-5),
        kernel_name="helion_autograd",
        baseline_name="torch",
        rtol=1e-2,
        atol=1e-2,
        bwd=True,
    )


# %%
# Main Function
# -------------


# %%
def main() -> None:
    """
    Main entry point that runs the RMS norm kernel verification with different tensor sizes.
    """
    check(2048, 4096)
    check(2048, 8192)


if __name__ == "__main__":
    main()
