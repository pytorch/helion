"""Paired statistical estimators and multiplicity gates for the head-to-head study.

Pure numpy/scipy; no retrieval, torch, or autotuner dependency.

- ``paired`` — Wilcoxon signed-rank, rank-biserial effect size, win/tie/loss
  counts, and the bootstrap geometric-mean ratio CI used for every arm pair.
- ``gates`` — Holm multiplicity control over the family of pairwise comparisons.
"""

from __future__ import annotations

from .gates import EqualBudgetInputs as EqualBudgetInputs
from .gates import TwoDimensionalInputs as TwoDimensionalInputs
from .gates import comparison_pvalue as comparison_pvalue
from .gates import holm_adjust as holm_adjust
from .gates import holm_reject as holm_reject
from .gates import intersection_union_pvalue as intersection_union_pvalue
from .gates import two_dimensional_pvalue as two_dimensional_pvalue
from .paired import bootstrap_geometric_mean_ci as bootstrap_geometric_mean_ci
from .paired import rank_biserial as rank_biserial
from .paired import wilcoxon_pvalue as wilcoxon_pvalue
from .paired import wins_ties_losses as wins_ties_losses
