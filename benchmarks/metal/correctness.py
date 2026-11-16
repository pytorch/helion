"""Output comparison for the strict and dashboard verdicts.

Both verdicts compare in float64 on the host.  Structure, shape and the
NaN/inf pattern must match exactly, and finite values must satisfy
``|actual - expected| <= atol + rtol * |expected|`` elementwise.  Output
dtypes are checked against eager's by the harness.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils._pytree import tree_flatten

# The gate the GPU nightly applies against the eager baseline.
DASHBOARD_TOL = 1e-2
# Elements compared per float64 chunk.
CHUNK = 1 << 24


@dataclass
class Check:
    passed: bool
    reason: str | None = None
    max_abs: float = 0.0
    max_rel: float = 0.0
    mismatches: int = 0
    numel: int = 0


def compare(actual: object, expected: object, rtol: float, atol: float) -> Check:
    actual_leaves, actual_spec = tree_flatten(actual)
    expected_leaves, expected_spec = tree_flatten(expected)
    if actual_spec != expected_spec:
        return Check(False, f"output structure {actual_spec} != {expected_spec}")
    result = Check(True)
    for i, (a, e) in enumerate(zip(actual_leaves, expected_leaves, strict=True)):
        if not isinstance(a, torch.Tensor) or not isinstance(e, torch.Tensor):
            return Check(False, f"output {i} is not a tensor")
        if a.shape != e.shape:
            return Check(False, f"output {i} shape {list(a.shape)} != {list(e.shape)}")
        leaf = _compare_leaf(a, e, rtol, atol)
        result.max_abs = max(result.max_abs, leaf.max_abs)
        result.max_rel = max(result.max_rel, leaf.max_rel)
        result.mismatches += leaf.mismatches
        result.numel += leaf.numel
        if not leaf.passed and result.passed:
            result.passed = False
            result.reason = f"output {i}: {leaf.reason}"
    return result


def _compare_leaf(a: torch.Tensor, e: torch.Tensor, rtol: float, atol: float) -> Check:
    """Compare in float64, a bounded chunk at a time."""
    a, e = a.reshape(-1), e.reshape(-1)
    check = Check(True, numel=e.numel())
    nan_or_inf = 0
    for start in range(0, e.numel(), CHUNK):
        ac = a[start : start + CHUNK].double()
        ec = e[start : start + CHUNK].double()
        pattern = (torch.isnan(ac) != torch.isnan(ec)) | (
            torch.isinf(ac) != torch.isinf(ec)
        )
        pattern |= torch.isinf(ec) & (torch.sign(ac) != torch.sign(ec))
        finite = torch.isfinite(ac) & torch.isfinite(ec)
        diff = torch.where(finite, (ac - ec).abs(), 0.0)
        scale = ec.abs()
        bad = (diff > atol + rtol * scale) | pattern
        rel = torch.where(finite & (scale > 0), diff / scale, 0.0)
        nan_or_inf += int(pattern.sum())
        check.mismatches += int(bad.sum())
        check.max_abs = max(check.max_abs, float(diff.max()))
        check.max_rel = max(check.max_rel, float(rel.max()))
    check.passed = check.mismatches == 0
    if nan_or_inf:
        check.reason = "NaN/inf pattern differs"
    elif check.mismatches:
        check.reason = f"{check.mismatches}/{check.numel} values out of tolerance"
    return check
