"""
Helion Layer Normalization Forward and Backward Example
=======================================================
This example demonstrates a Helion kernel implementation of 1D layer normalization
with both forward and backward passes using FP16 inputs. Forward correctness is
checked against :func:`torch.nn.functional.layer_norm`. Backward checks use an
analytic FP32-upcast baseline because PyTorch's native FP16 ``layer_norm``
backward can diverge from that closed form (same situation as ``rms_norm.py``).
"""

# %%
from __future__ import annotations

from typing import Any
from typing import Callable

import torch
import torch.nn.functional as F

import helion
from helion._testing import DEVICE
from helion._testing import HALF_DTYPE
from helion._testing import run_example
import helion.language as hl
from helion.language.constexpr import ConstExpr


def layer_norm_bwd_reference(
    grad_out: torch.Tensor,
    x: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    weight: torch.Tensor,
    compute_bias_grad: bool | ConstExpr,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """PyTorch reference for :func:`layer_norm_bwd` (used as autotune baseline, not Triton)."""
    if isinstance(compute_bias_grad, ConstExpr):
        compute_bias_grad = bool(compute_bias_grad.value)

    n = x.shape[1]
    x_f = x.to(torch.float32)
    dy_f = grad_out.to(torch.float32)
    mean_f = mean.to(torch.float32).unsqueeze(-1)
    rstd_f = rstd.to(torch.float32).unsqueeze(-1)
    w_f = weight.to(torch.float32)

    x_hat = (x_f - mean_f) * rstd_f
    grad_weight = (dy_f * x_hat).sum(dim=0).to(weight.dtype)

    wdy = w_f * dy_f
    c1 = (x_hat * wdy).sum(dim=-1) / n
    c2 = wdy.sum(dim=-1) / n
    grad_x = (wdy - (x_hat * c1.unsqueeze(-1) + c2.unsqueeze(-1))) * rstd_f
    grad_x = grad_x.to(dtype=x.dtype)

    if compute_bias_grad:
        grad_bias = dy_f.sum(dim=0).to(weight.dtype)
        return grad_x, grad_weight, grad_bias
    return grad_x, grad_weight, None


# %%
@helion.kernel(autotune_ignore_errors=True, autotune_effort="full")
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
@helion.kernel(
    autotune_ignore_errors=True,
    autotune_effort="full",
    autotune_baseline_fn=layer_norm_bwd_reference,
)
def layer_norm_bwd(
    grad_out: torch.Tensor,
    x: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    weight: torch.Tensor,
    compute_bias_grad: hl.constexpr = True,  # type: ignore[valid-type]
    n_chunk_max: hl.constexpr = 1024,  # type: ignore[valid-type]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """
    Compute gradients for weight (dW) and optionally bias (dB) parameters.

    This kernel performs reduction across the batch dimension (M) to accumulate
    gradients for each feature dimension's weight and bias parameters.

    The feature dimension ``N`` is tiled in chunks of at most ``n_chunk_max`` so
    on-chip buffers stay small (same idea as :func:`rms_norm_bwd` on Ascend UB).

    Args:
        grad_out: Gradient w.r.t layer norm output [M, N]
        x: Original input tensor [M, N]
        mean: Per-sample mean computed in forward pass [M]
        rstd: Per-sample reciprocal standard deviation from forward pass [M]
        weight: Weight parameter (used only for dtype/device info) [N]
        compute_bias_grad: Whether to compute bias gradient (default: True)
        n_chunk_max: Max feature elements per N-tile (constexpr; lower if UB overflow).

    Returns:
        (grad_x, grad_weight, grad_bias): Gradients for input, weight, and bias (if computed)
            grad_bias is None if compute_bias_grad is False
    """

    m_block = hl.register_block_size(x.size(0))
    n = hl.specialize(x.size(1))

    grad_x = torch.empty_like(x)
    num_blocks = (x.size(0) + m_block - 1) // m_block
    grad_weight_blocks = x.new_zeros([num_blocks, n], dtype=torch.float32)
    if compute_bias_grad:
        grad_bias_blocks = x.new_zeros([num_blocks, n], dtype=torch.float32)

    for mb_cta in hl.tile(x.size(0), block_size=m_block):
        for mb in hl.tile(mb_cta.begin, mb_cta.end):
            mean_mb = mean[mb].to(torch.float32)
            rstd_mb = rstd[mb].to(torch.float32)

            s1 = torch.zeros_like(x[mb, 0], dtype=torch.float32)
            s2 = torch.zeros_like(x[mb, 0], dtype=torch.float32)
            for tn in hl.tile(n, block_size=n_chunk_max):
                x_a = x[mb, tn].to(torch.float32)
                dy_a = grad_out[mb, tn].to(torch.float32)
                w_a = weight[None, tn].to(torch.float32)
                x_hat_a = (x_a - mean_mb[:, None]) * rstd_mb[:, None]
                wdy_a = w_a * dy_a
                s1 += (x_hat_a * wdy_a).sum(-1)
                s2 += wdy_a.sum(-1)

            c1 = s1 / n
            c2 = s2 / n

            for tile_n in hl.tile(n, block_size=n_chunk_max):
                x_m = x[mb, tile_n].to(torch.float32)
                dy_m = grad_out[mb, tile_n].to(torch.float32)
                w_m = weight[None, tile_n].to(torch.float32)
                x_hat = (x_m - mean_mb[:, None]) * rstd_mb[:, None]
                wdy = w_m * dy_m
                grad_weight_blocks[mb_cta.id, tile_n] += (dy_m * x_hat).sum(0)
                if compute_bias_grad:
                    grad_bias_blocks[mb_cta.id, tile_n] += dy_m.sum(0)
                dx = (wdy - (x_hat * c1[:, None] + c2[:, None])) * rstd_mb[:, None]
                grad_x[mb, tile_n] = dx.to(x.dtype)

    grad_weight = grad_weight_blocks.sum(0).to(weight.dtype)
    if compute_bias_grad:
        grad_bias = grad_bias_blocks.sum(0).to(weight.dtype)
        return grad_x, grad_weight, grad_bias
    return grad_x, grad_weight, None


# %%
class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(  # pyrefly: ignore [bad-override]
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
class _LayerNormAnalyticRef(torch.autograd.Function):
    """Forward matches ``F.layer_norm``; backward uses :func:`layer_norm_bwd_reference`."""

    @staticmethod
    def forward(  # pyrefly: ignore [bad-override]
        ctx: Any,  # noqa: ANN401
        x: torch.Tensor,
        normalized_shape: list[int],
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        eps: float,
    ) -> torch.Tensor:
        ctx.save_for_backward(x, weight, bias)
        ctx.eps = eps  # type: ignore[attr-defined]
        ctx.normalized_shape = normalized_shape  # type: ignore[attr-defined]
        return F.layer_norm(x, normalized_shape, weight, bias, eps=eps)

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any,  # noqa: ANN401
        grad_output: torch.Tensor,
    ) -> tuple[
        torch.Tensor | None, None, torch.Tensor | None, torch.Tensor | None, None
    ]:
        x, weight, bias = ctx.saved_tensors  # type: ignore[attr-defined]
        eps = ctx.eps  # type: ignore[attr-defined]
        xf = x.to(torch.float32)
        mean = xf.mean(-1)
        rstd = torch.rsqrt(xf.var(-1, unbiased=False) + eps)
        compute_bias_grad = bias is not None
        gx, gw, gb = layer_norm_bwd_reference(
            grad_output, x, mean, rstd, weight, compute_bias_grad
        )
        return gx, None, gw, gb, None


def layer_norm_pytorch_analytic(
    x: torch.Tensor,
    normalized_shape: list[int],
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Test baseline: ``F.layer_norm`` forward, analytic FP32-upcast backward."""
    return _LayerNormAnalyticRef.apply(x, normalized_shape, weight, bias, eps)  # type: ignore[no-any-return]


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
    x = -2.3 + 0.5 * torch.randn([batch_size, dim], device=device, dtype=HALF_DTYPE)
    weight = torch.randn([dim], device=device, dtype=HALF_DTYPE)
    bias = torch.randn([dim], device=device, dtype=HALF_DTYPE)
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
        [batch_size, dim], device=device, dtype=HALF_DTYPE, requires_grad=True
    )
    weight_grad = torch.randn(
        [dim], device=device, dtype=HALF_DTYPE, requires_grad=True
    )
    bias_grad = torch.randn([dim], device=device, dtype=HALF_DTYPE, requires_grad=True)
    for b in [bias_grad, None]:
        run_example(
            layer_norm,
            layer_norm_pytorch_analytic,
            (x_grad, [dim], weight_grad, b, eps),
            kernel_name="helion",
            baseline_name="analytic_fp32_bwd",
            rtol=1e-3,
            atol=1e-3,
            bwd=True,
        )


# %%
if __name__ == "__main__":
    main()
