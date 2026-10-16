"""Independent residual RMSNorm definition for both public precision tracks."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from numerical_policy import dtype_policies
from numerical_policy import tolerances
from numerics import check_accuracy
from numerics import full_precision_reference
import torch

if TYPE_CHECKING:
    from inputs import Case

# Compatibility views for callers displaying supported dtypes. Validation reads
# the policy on every call; these tables do not define acceptance thresholds.
TOLERANCES = {
    getattr(torch, dtype): (gate["atol"], gate["rtol"])
    for dtype, gate in dtype_policies("fused_add_rms_norm").items()
}
L2_LIMITS = {
    getattr(torch, dtype): gate["l2_limit"]
    for dtype, gate in dtype_policies("fused_add_rms_norm").items()
}


def reference(case: Case) -> tuple[torch.Tensor, torch.Tensor]:
    """FP64 normalization after the prescribed FP32 sum/residual boundary."""
    total = case.x.float() + case.residual.float()
    residual_out = total.to(case.x.dtype)
    norm_input = (
        residual_out.double() if case.normalize_rounded_residual else total.double()
    )
    inverse = (norm_input.square().sum(1) / norm_input.shape[1] + case.eps).rsqrt()
    output = norm_input * inverse[:, None] * case.weight.double()
    return output, residual_out


@full_precision_reference()
def conventional_reference(case: Case) -> tuple[torch.Tensor, torch.Tensor]:
    """PyTorch eager FP32 path, including both declared rounding modes."""
    total = case.x.float() + case.residual.float()
    stored = total.to(case.x.dtype)
    norm_input = stored.float() if case.normalize_rounded_residual else total
    inverse = (norm_input.square().mean(1) + case.eps).rsqrt()
    output = norm_input * inverse[:, None] * case.weight.float()
    return output.to(case.x.dtype), stored


def fp64_definition(case: Case) -> tuple[torch.Tensor, torch.Tensor]:
    x, residual, weight = (
        value.detach().cpu().double() for value in case.kernel_inputs()
    )
    # Addition still follows the specified FP32 step, before optional input-type
    # rounding; only the independent variance and normalization use CPU FP64.
    total = (x + residual).float().double()
    stored = total.to(case.x.dtype)
    norm_input = stored.double() if case.normalize_rounded_residual else total
    rows = []
    for row in norm_input:
        variance = math.fsum(value * value for value in row.tolist()) / row.numel()
        rows.append(row * (1 / math.sqrt(variance + case.eps)) * weight)
    return torch.stack(rows), stored


def validate_output(
    case: Case,
    output: tuple[torch.Tensor, torch.Tensor],
    expected: tuple[torch.Tensor, torch.Tensor],
    *,
    atol: float | None = None,
    rtol: float | None = None,
) -> dict:
    gate = tolerances("fused_add_rms_norm", str(case.x.dtype), atol=atol, rtol=rtol)
    if not isinstance(output, tuple) or len(output) != 2:
        raise AssertionError("Expected normalized output and updated residual")
    for value in output:
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != case.x.shape
            or value.dtype != case.x.dtype
            or value.device != case.x.device
            or not torch.isfinite(value).all()
        ):
            raise AssertionError("Both outputs must be finite and match input metadata")
    torch.testing.assert_close(output[1], expected[1], atol=0, rtol=0)
    total = case.x.float() + case.residual.float()
    norm_input = total.to(case.x.dtype) if case.normalize_rounded_residual else total
    metrics = check_accuracy(
        output[0],
        expected[0],
        atol=gate["atol"],
        rtol=gate["rtol"],
        l2_limit=gate["l2_limit"],
        group_dims=(-1,),
        exact_zero_mask=(norm_input == 0) | (case.weight[None, :] == 0),
        label="Add + RMSNorm output",
    )
    return metrics | {
        "status": "PASS",
        "reference": "Unrounded FP64 normalization after prescribed sum rounding",
        "relative_l2_scope": "global and per row",
        "residual_rounding": "exact input-dtype rounding of FP32 sum",
        "normalize_rounded_residual": case.normalize_rounded_residual,
    }
