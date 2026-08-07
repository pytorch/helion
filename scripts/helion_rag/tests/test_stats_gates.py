from __future__ import annotations

import pytest

from helion_rag.stats import EqualBudgetInputs
from helion_rag.stats import TwoDimensionalInputs
from helion_rag.stats import comparison_pvalue
from helion_rag.stats import holm_adjust
from helion_rag.stats import holm_reject
from helion_rag.stats import intersection_union_pvalue
from helion_rag.stats import two_dimensional_pvalue


def test_two_dimensional_branch_a_wins():
    # Branch A: runtime superiority (0.01) + quality NI (0.02) -> 0.02.
    # Branch B: quality superiority (0.5) + runtime NI max(0.01,0.02)=0.02 -> 0.5.
    inp = TwoDimensionalInputs(
        p_runtime_superiority=0.01,
        p_quality_superiority=0.5,
        p_runtime_median_ni=0.01,
        p_runtime_p95_ni=0.02,
        p_quality_ni=0.02,
    )
    assert two_dimensional_pvalue(inp) == pytest.approx(0.04)  # min(1, 2*0.02)


def test_two_dimensional_capped_at_one():
    inp = TwoDimensionalInputs(
        p_runtime_superiority=0.9,
        p_quality_superiority=0.8,
        p_runtime_median_ni=0.7,
        p_runtime_p95_ni=0.7,
        p_quality_ni=0.9,
    )
    # 2 * min(max(0.9,0.9), max(0.8,0.7)) = 2*0.8 = 1.6 -> capped to 1.0
    assert two_dimensional_pvalue(inp) == 1.0


def test_comparison_takes_worst_gate():
    two_dim = TwoDimensionalInputs(
        p_runtime_superiority=0.01,
        p_quality_superiority=0.5,
        p_runtime_median_ni=0.01,
        p_runtime_p95_ni=0.02,
        p_quality_ni=0.02,
    )
    inp = EqualBudgetInputs(
        p_success_ni=0.01,
        p_ordinal_ni=0.20,  # worst gate
        p_coverage_lcb=0.03,
        p_reliability=0.05,
        p_runtime_p95_ni=0.02,
        two_dimensional=two_dim,
    )
    assert comparison_pvalue(inp) == pytest.approx(0.20)


def test_intersection_union_is_max():
    assert intersection_union_pvalue([0.01, 0.2, 0.05]) == pytest.approx(0.2)
    with pytest.raises(ValueError):
        intersection_union_pvalue([])


def test_holm_adjust_known_answer():
    adj = holm_adjust([0.01, 0.03])
    assert adj == pytest.approx([0.02, 0.03])


def test_holm_adjust_monotone_and_capped():
    adj = holm_adjust([0.04, 0.02, 0.5])
    # sorted: 0.02(*3)=0.06, 0.04(*2)=0.08, 0.5(*1)=0.5 -> monotone, in input order
    assert adj == pytest.approx([0.08, 0.06, 0.5])


def test_holm_reject():
    assert holm_reject([0.01, 0.03], 0.05) == [True, True]
    assert holm_reject([0.04, 0.5], 0.05) == [False, False]
