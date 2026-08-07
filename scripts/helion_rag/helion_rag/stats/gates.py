"""Compound gate p-values and multiplicity control (§16, §17).

These functions operate purely on already-computed one-sided gate p-values (from
:func:`~helion_rag.stats.bootstrap.boundary_pvalue`). Keeping them separate from
the bootstrap makes the union / intersection-union / Holm logic exhaustively
testable with hand-chosen p-values.

Convention: every input is a one-sided p-value where *small means the gate is
satisfied*. Gates are combined with ``max`` (intersection-union: all must hold);
the two-dimensional branch uses a Bonferroni union across its two branches.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence


@dataclasses.dataclass(frozen=True)
class TwoDimensionalInputs:
    """The five p-values feeding the §16 two-dimensional union test."""

    p_runtime_superiority: float
    p_quality_superiority: float
    p_runtime_median_ni: float
    p_runtime_p95_ni: float
    p_quality_ni: float


def two_dimensional_pvalue(inp: TwoDimensionalInputs) -> float:
    """§16 two-dimensional union p-value.

    Branch A is runtime superiority with quality non-inferiority; branch B is
    quality superiority with runtime non-inferiority. The union takes the better
    branch and applies a Bonferroni factor of two.
    """
    p_runtime_ni = max(inp.p_runtime_median_ni, inp.p_runtime_p95_ni)
    p_branch_a = max(inp.p_runtime_superiority, inp.p_quality_ni)
    p_branch_b = max(inp.p_quality_superiority, p_runtime_ni)
    return min(1.0, 2.0 * min(p_branch_a, p_branch_b))


@dataclasses.dataclass(frozen=True)
class EqualBudgetInputs:
    """All scalar gate p-values for one equal-budget causal comparison (§16)."""

    p_success_ni: float
    p_ordinal_ni: float
    p_coverage_lcb: float
    p_reliability: float
    p_runtime_p95_ni: float
    two_dimensional: TwoDimensionalInputs


def comparison_pvalue(inp: EqualBudgetInputs) -> float:
    """§16 equal-budget comparison p-value: ``max`` over every required gate.

    Runtime p95 non-inferiority is mandatory regardless of which two-dimensional
    branch passes, so it enters both inside :func:`two_dimensional_pvalue` and
    again here.
    """
    return max(
        inp.p_success_ni,
        inp.p_ordinal_ni,
        inp.p_coverage_lcb,
        inp.p_reliability,
        inp.p_runtime_p95_ni,
        two_dimensional_pvalue(inp.two_dimensional),
    )


def intersection_union_pvalue(gate_pvalues: Sequence[float]) -> float:
    """Intersection-union p-value for a tuner claim: the maximum gate p (§17.1)."""
    if not gate_pvalues:
        raise ValueError("need at least one gate p-value")
    return max(gate_pvalues)


def holm_adjust(pvalues: Sequence[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values, in the input order (§17)."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[idx]))
        adjusted[idx] = running
    return adjusted


def holm_reject(pvalues: Sequence[float], alpha: float) -> list[bool]:
    """Which hypotheses Holm rejects at familywise level ``alpha`` (§17)."""
    return [p <= alpha for p in holm_adjust(pvalues)]
