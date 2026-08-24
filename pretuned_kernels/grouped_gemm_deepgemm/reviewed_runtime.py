"""Runtime validation shared by the reviewed grouped-GEMM entry points."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING
from typing import Any

import torch

from . import reviewed_profiles

if TYPE_CHECKING:
    from collections.abc import Sequence

    from helion.runtime.kernel import BoundKernel


def normalized_difference(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """Return a finite, symmetric normalized difference for two tensors."""

    actual64 = actual.double()
    expected64 = expected.double()
    denominator = (actual64.square() + expected64.square()).sum()
    denominator_value = float(denominator.item())
    if not math.isfinite(denominator_value):
        return math.inf
    if denominator_value == 0.0:
        return 0.0
    value = float((1 - 2 * (actual64 * expected64).sum() / denominator).item())
    return max(0.0, value) if math.isfinite(value) else math.inf


def check_logical_outputs(
    actual: Sequence[torch.Tensor],
    oracle: Sequence[torch.Tensor],
    *,
    max_diff: float,
    rtol: float = 2e-2,
    atol: float = 2e-2,
) -> dict[str, Any]:
    """Validate every logical group independently against its FP32 oracle."""

    if len(actual) != len(oracle):
        raise ValueError(
            f"implementation produced {len(actual)} groups, expected {len(oracle)}"
        )
    groups: list[dict[str, Any]] = []
    passed = True
    max_abs = 0.0
    max_normalized_diff = 0.0
    mismatch_count = 0
    for group, (output, expected) in enumerate(zip(actual, oracle, strict=True)):
        shape_ok = output.shape == expected.shape
        dtype_ok = output.dtype is torch.bfloat16 and expected.dtype is torch.float32
        device_ok = output.device == expected.device
        if not (shape_ok and dtype_ok and device_ok):
            groups.append(
                {
                    "group": group,
                    "ok": False,
                    "shape_ok": shape_ok,
                    "dtype_ok": dtype_ok,
                    "device_ok": device_ok,
                    "normalized_diff": math.inf,
                    "max_abs": math.inf,
                    "mismatch_count": output.numel(),
                }
            )
            passed = False
            max_abs = math.inf
            max_normalized_diff = math.inf
            mismatch_count += output.numel()
            continue
        output_fp32 = output.float()
        difference = (output_fp32 - expected).abs()
        group_max_abs = float(difference.max().item()) if difference.numel() else 0.0
        finite = bool(
            torch.isfinite(output_fp32).all().item()
            and torch.isfinite(expected).all().item()
        )
        if not finite:
            group_max_abs = math.inf
        group_normalized_diff = (
            normalized_difference(output_fp32, expected) if finite else math.inf
        )
        close = torch.isclose(output_fp32, expected, rtol=rtol, atol=atol)
        group_mismatch_count = int((~close).sum().item()) if finite else output.numel()
        group_ok = (
            finite and group_normalized_diff <= max_diff and group_mismatch_count == 0
        )
        groups.append(
            {
                "group": group,
                "ok": group_ok,
                "shape_ok": shape_ok,
                "dtype_ok": dtype_ok,
                "device_ok": device_ok,
                "normalized_diff": group_normalized_diff,
                "max_abs": group_max_abs,
                "mismatch_count": group_mismatch_count,
            }
        )
        passed = passed and group_ok
        max_abs = max(max_abs, group_max_abs)
        max_normalized_diff = max(max_normalized_diff, group_normalized_diff)
        mismatch_count += group_mismatch_count
    return {
        "ok": passed,
        "max_normalized_diff": max_normalized_diff,
        "max_abs": max_abs,
        "mismatch_count": mismatch_count,
        "rtol": rtol,
        "atol": atol,
        "groups": groups,
    }


def effective_reviewed_config(
    bound: BoundKernel[Any],
    profile: reviewed_profiles.ReviewedWorklistProfile,
) -> dict[str, dict[str, object]]:
    """Validate the reviewed request and return its effective config."""

    requested = reviewed_profiles.reviewed_config_values(profile.config_name)
    actual = bound._config
    if actual is None or actual.config != requested:
        raise RuntimeError(
            "AOT evaluation did not select the exact reviewed config "
            f"{profile.config_name}"
        )
    effective = bound.config_spec.normalized_config(actual)
    return {
        "requested": dict(actual.config),
        "effective": dict(effective.config),
    }
