"""Frozen forward-accuracy gates for the ROCm reproduction validators.

The recommendation's conventional-reference error is a calibration diagnostic,
never a candidate-specific allowance. All checks execute outside GPU timing.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import math
from pathlib import Path
from typing import TYPE_CHECKING

from numerical_policy import policy_identity
from numerical_policy import reference_policy
from numerical_policy import shared_policy
import torch

if TYPE_CHECKING:
    from collections.abc import Iterator

POLICY_VERSION = policy_identity()["version"]


def cap_tolerances(
    atol: float | None,
    rtol: float | None,
    default_atol: float,
    default_rtol: float,
) -> tuple[float, float]:
    """Allow explicit requests to tighten, but never relax, either component."""
    for value in (atol, rtol, default_atol, default_rtol):
        if value is not None and (not math.isfinite(value) or value < 0):
            raise ValueError("Use finite nonnegative tolerances")
    return (
        default_atol if atol is None else min(atol, default_atol),
        default_rtol if rtol is None else min(rtol, default_rtol),
    )


@contextmanager
def full_precision_reference() -> Iterator[None]:
    """Freeze conventional FP32 matmuls; restore the caller's backend settings."""
    backend = torch.backends.cuda.matmul
    names = (
        "allow_tf32",
        "allow_fp16_reduced_precision_reduction",
        "allow_bf16_reduced_precision_reduction",
    )
    previous = {name: getattr(backend, name) for name in names}
    try:
        for name in names:
            setattr(backend, name, False)
        yield
    finally:
        for name, value in previous.items():
            setattr(backend, name, value)


def policy_metadata(*, kernel: str | None = None) -> dict:
    path = Path(__file__)
    result = {
        **policy_identity(),
        "validator_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "reference_framework": "PyTorch eager",
        "torch_version": str(torch.__version__),
        "torch_commit": torch.version.git_version,
        **shared_policy(),
    }
    if kernel is not None:
        reference = reference_policy(kernel)
        result.update(
            oracle_dtype=reference["oracle_dtype"],
            oracle_final_rounding=reference["oracle_final_rounding"],
            comparison_dtype=reference["comparison_dtype"],
            reference_framework=reference.get("framework", "PyTorch eager"),
            reference=reference,
        )
    return result


def check_accuracy(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: float,
    rtol: float,
    l2_limit: float | None = None,
    group_l2_limit: float | None = None,
    group_dims: int | tuple[int, ...] | None = None,
    exact_zero_mask: torch.Tensor | None = None,
    label: str = "output",
    policy_kernel: str | None = None,
    compute_l2: bool = False,
) -> dict:
    """Measure elementwise and global/group errors using FP64 comparison tensors.

    The default oracle uses FP64 arithmetic. KDA low policies supply native
    FlashKDA or eager values widened losslessly to FP64 for error measurement.
    compute_l2 retains diagnostics when the selected policy has no L2 gate.

    Group norms use group_l2_limit when supplied, otherwise l2_limit.
    A zero reference norm has no relative L2 value. It still receives the
    elementwise check and any input-derived structural-zero assertion.
    """
    cap_tolerances(atol, rtol, atol, rtol)
    if l2_limit is not None and (not math.isfinite(l2_limit) or l2_limit < 0):
        raise ValueError("Use a finite nonnegative L2 limit")
    if group_l2_limit is not None:
        if (
            isinstance(group_l2_limit, bool)
            or not math.isfinite(group_l2_limit)
            or group_l2_limit < 0
        ):
            raise ValueError("Use a finite nonnegative group L2 limit")
        if group_dims is None:
            raise ValueError("A group L2 limit requires reduction dimensions")
    group_limit = l2_limit if group_l2_limit is None else group_l2_limit
    if actual.shape != expected.shape or actual.device != expected.device:
        raise AssertionError(f"{label}: output and oracle shape/device differ")
    if expected.dtype != torch.float64:
        if reference_policy(policy_kernel)["kind"] == "fp64":
            raise AssertionError(f"{label}: an unrounded FP64 oracle is required")
        raise AssertionError(
            f"{label}: widen the selected reference to FP64 for comparison"
        )
    if not torch.isfinite(expected).all():
        raise AssertionError(f"{label}: nonfinite oracle needs a separate contract")
    if not torch.isfinite(actual).all():
        raise AssertionError(f"{label}: output contains nonfinite values")
    value = actual.detach().double()
    expected = expected.detach()
    error = (value - expected).abs()
    if not torch.isclose(value, expected, atol=atol, rtol=rtol).all():
        oracle = {
            "fp64": "FP64",
            "torch_bf16_moe": "PyTorch BF16 MoE",
            "flashkda_prefill": "FlashKDA",
        }.get(reference_policy(policy_kernel)["kind"], "upstream eager")
        raise AssertionError(f"{label}: elementwise {oracle} accuracy gate failed")
    if exact_zero_mask is not None:
        mask = torch.broadcast_to(exact_zero_mask, actual.shape)
        if not torch.all(actual[mask] == 0):
            raise AssertionError(f"{label}: structural zeros must be exact")
    error_scale = float(error.max()) if error.numel() else 0.0
    rms_error = (
        error_scale * float((error / error_scale).square().mean().sqrt())
        if error_scale
        else 0.0
    )
    metrics = {
        "atol": atol,
        "rtol": rtol,
        "max_absolute_error": float(error.max()) if error.numel() else 0.0,
        "max_relative_error": float(
            (error / expected.abs().clamp_min(torch.finfo(torch.float64).tiny)).max()
        )
        if error.numel()
        else 0.0,
        "rms_error": rms_error,
        "numerical_policy": policy_metadata(kernel=policy_kernel),
    }
    if l2_limit is not None or group_l2_limit is not None or compute_l2:
        # Scale before squaring: a nonzero FP64 state near 1e-200 must not
        # masquerade as an exact-zero oracle when its squared norm underflows.
        def ratios(
            difference: torch.Tensor, reference: torch.Tensor, dims: tuple[int, ...]
        ) -> torch.Tensor:
            scale = reference.abs().amax(dim=dims, keepdim=True)
            scale = torch.where(scale > 0, scale, 1)
            norms = torch.linalg.vector_norm(reference / scale, dim=dims)
            errors = torch.linalg.vector_norm(difference / scale, dim=dims)
            nonzero = norms > 0
            return errors[nonzero] / norms[nonzero], nonzero

        if expected.numel():
            ratios_global, _ = ratios(error.reshape(-1), expected.reshape(-1), -1)
            relative = float(ratios_global[0]) if ratios_global.numel() else None
        else:
            relative = None
        if l2_limit is not None and relative is not None and relative > l2_limit:
            raise AssertionError(f"{label}: global relative L2 gate failed")
        metrics.update(relative_l2=relative, l2_limit=l2_limit)
        if group_dims is not None and expected.numel():
            relatives, nonzero = ratios(error, expected, group_dims)
            maximum = float(relatives.max()) if relatives.numel() else None
            if (
                group_limit is not None
                and maximum is not None
                and maximum > group_limit
            ):
                raise AssertionError(f"{label}: per-group relative L2 gate failed")
            metrics.update(
                max_group_relative_l2=maximum,
                zero_norm_groups=int((~nonzero).sum()),
            )
            if group_l2_limit is not None:
                metrics["group_l2_limit"] = group_l2_limit
    return metrics
