"""Canonical K10 body and metadata helper captured without changing their AST."""

from __future__ import annotations

import math

import torch

import helion.language as hl


def _validate_metadata(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    rounded: bool,
) -> None:
    if (
        x.ndim != 2
        or min(x.shape) <= 0
        or residual.shape != x.shape
        or weight.shape != (x.shape[1],)
    ):
        raise ValueError("Require positive X/Residual[M,N] and Weight[N]")
    if (
        x.dtype not in (torch.bfloat16, torch.float16, torch.float32)
        or residual.dtype != x.dtype
        or weight.dtype != x.dtype
    ):
        raise ValueError("Inputs must share BF16, FP16 or FP32 dtype")
    if residual.device != x.device or weight.device != x.device:
        raise ValueError("Inputs must share a device")
    if (
        x.stride(1) != 1
        or residual.stride(1) != 1
        or x.stride(0) < x.shape[1]
        or residual.stride(0) < x.shape[1]
        or not weight.is_contiguous()
    ):
        raise ValueError(
            "Input rows and weights must be contiguous; padded row strides are allowed"
        )
    if type(rounded) is not bool or not math.isfinite(eps) or eps <= 0:
        raise ValueError("Use a bool normalization mode and finite positive epsilon")


def fused_add_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: hl.constexpr = 1e-6,
    normalize_rounded_residual: hl.constexpr = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (normalized, rounded residual) while preserving all inputs."""
    _validate_metadata(x, residual, weight, eps, normalize_rounded_residual)
    m, n = x.size()
    output = torch.empty((m, n), dtype=x.dtype, device=x.device)
    residual_out = torch.empty_like(output)
    for rows in hl.tile(m):
        total = x[rows, :].to(torch.float32) + residual[rows, :].to(torch.float32)
        stored = total.to(x.dtype)
        if normalize_rounded_residual:
            norm_input = stored.to(torch.float32)
        else:
            norm_input = total
        variance = torch.mean(norm_input * norm_input, dim=-1)
        inverse = torch.rsqrt(variance + eps)
        output[rows, :] = (
            norm_input * inverse[:, None] * weight[:].to(torch.float32)
        ).to(x.dtype)
        residual_out[rows, :] = stored
    return output, residual_out


def _fused_add_rms_norm_reference(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    normalize_rounded_residual: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Explicit PyTorch definition with FP32 addition and one residual store cast."""
    _validate_metadata(x, residual, weight, eps, normalize_rounded_residual)
    total = x.float() + residual.float()
    stored = total.to(x.dtype)
    values = stored.float() if normalize_rounded_residual else total
    inverse = torch.rsqrt(values.square().mean(dim=1, keepdim=True) + eps)
    return (values * inverse * weight.float()).to(x.dtype), stored
