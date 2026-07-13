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
from helion._testing import HALF_DTYPE
from helion._testing import run_example
import helion.language as hl


def rms_norm_bwd_reference(
    grad_out: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    rsqrt: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """PyTorch reference for :func:`rms_norm_bwd` (autotune baseline, not Triton)."""
    x_f = x.to(torch.float32)
    do = grad_out.to(torch.float32)
    rsqrt_f = rsqrt.to(torch.float32)
    w = weight.to(torch.float32).unsqueeze(0)

    grad_weight = (x_f * do * rsqrt_f).sum(dim=0).to(weight.dtype)

    grad_x = w * do * rsqrt_f - x_f * rsqrt_f**3 * (w * do * x_f).mean(
        dim=-1, keepdim=True
    )
    grad_x = grad_x.to(dtype=x.dtype)

    return grad_x, grad_weight


# %%
# RMS Normalization Kernel
# ------------------------


# %%
@helion.kernel(
    autotune_ignore_errors=True,
    autotune_effort="full",
)
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
        Inverse RMS tensor of shape [M, 1] in ``x.dtype`` (e.g. FP16) for backward
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

@helion.kernel(
    autotune_ignore_errors=True,
    autotune_effort="full",
    autotune_baseline_fn=rms_norm_bwd_reference,
)
def rms_norm_bwd(
    grad_out: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    rsqrt: torch.Tensor,
    n_chunk_max: hl.constexpr = 1024,  # type: ignore[valid-type]
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute gradient for input tensor (dX) and weights (dW).

    Splits the feature dimension into tiles of at most ``n_chunk_max`` elements so
    on-chip buffers (no full-[N] fp32 ``grad_w_m``) fit Ascend UB; ``mean`` for the
    RMS backward term is accumulated in a separate pass over N-chunks per row.

    Args:
        grad_out: Gradient w.r.t rms norm output [M, N]
        x: Original input tensor [M, N]
        weight: Weight parameter [N]
        rsqrt: Inverse RMS tensor [M, 1]
        n_chunk_max: Max feature elements per tile (constexpr; lower if UB overflow).

    Returns:
        grad_x: Gradient w.r.t input tensor, shape [M, N]
        grad_weight: Gradient w.r.t weight tensor, shape [N]
    """
    m_block = hl.register_block_size(x.size(0))
    n = hl.specialize(weight.size(0))
    grad_x = torch.empty_like(x)
    grad_weight = x.new_zeros(
        [(x.size(0) + m_block - 1) // m_block, *weight.shape], dtype=torch.float32
    )
    for mb_cta in hl.tile(x.size(0), block_size=m_block):
        for mb in hl.tile(mb_cta.begin, mb_cta.end):
            s = torch.zeros_like(x[mb, 0], dtype=torch.float32)
            for tn in hl.tile(n, block_size=n_chunk_max):
                x_a = x[mb, tn].to(torch.float32)
                do_a = grad_out[mb, tn].to(torch.float32)
                w_a = weight[None, tn].to(torch.float32)
                s += (w_a * do_a * x_a).sum(-1)
            c = s / n

            for tile_n in hl.tile(n, block_size=n_chunk_max):
                x_m = x[mb, tile_n].to(torch.float32)
                do_m = grad_out[mb, tile_n].to(torch.float32)
                rsqrt_m = rsqrt[mb, :].to(torch.float32)
                w_m = weight[None, tile_n].to(torch.float32)
                grad_weight[mb_cta.id, tile_n] += (x_m * do_m * rsqrt_m).sum(0)
                grad_x[mb, tile_n] = (
                    w_m * do_m * rsqrt_m
                    - x_m * rsqrt_m**3 * c[:, None]
                ).to(x.dtype)
    return grad_x, grad_weight.sum(0).to(weight.dtype)


# %%
class RMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(  # pyrefly: ignore [bad-override]
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
class _RMSNormPytorchRef(torch.autograd.Function):
    """Reference that matches Helion: FP32 ``y``, FP16 ``inv_rms`` saved for backward."""

    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        x_f = x.to(torch.float32)
        variance = x_f.pow(2).mean(-1, keepdim=True)
        inv_rms_tile = torch.rsqrt(variance + eps)
        normalized = x_f * inv_rms_tile
        out = (normalized * weight.to(torch.float32)).to(x.dtype)
        ctx.save_for_backward(x, weight)
        ctx.inv_rms = inv_rms_tile.to(x.dtype).reshape(-1, 1)
        return out

    @staticmethod
    def backward(
        ctx: Any,
        grad_out: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        x, weight = ctx.saved_tensors
        inv_rms = ctx.inv_rms
        grad_x, grad_weight = rms_norm_bwd_reference(grad_out, x, weight, inv_rms)
        return grad_x, grad_weight, None


def rms_norm_pytorch(
    x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5
) -> torch.Tensor:
    """
    PyTorch reference for RMS normalization (aligned with Helion FP16 ``inv_rms``).

    Forward matches ``rms_norm_fwd`` (FP32 normalize then cast output). Backward uses
    the same FP16-stored inverse RMS as the Helion autograd path.
    """
    return _RMSNormPytorchRef.apply(x, weight, eps)


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
    x = torch.randn([m, n], device=DEVICE, dtype=HALF_DTYPE)
    weight = torch.randn([n], device=DEVICE, dtype=HALF_DTYPE)

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
    x_grad = torch.randn([m, n], device=DEVICE, dtype=HALF_DTYPE, requires_grad=True)
    weight_grad = torch.randn([n], device=DEVICE, dtype=HALF_DTYPE, requires_grad=True)

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
    # check(1024, 1024)
    check(2048, 4096)
    check(2048, 8192)


if __name__ == "__main__":
    import time
    time_st = time.time()
    main()
    print(f"time cost: {time.time() - time_st}")
