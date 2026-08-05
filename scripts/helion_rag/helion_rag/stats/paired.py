"""Reusable paired-ratio inference for the head-to-head campaign.

Lifts the geometric mean, percentile kernel-bootstrap, Wilcoxon signed-rank, and
rank-biserial helpers (previously private to ``generate_conference_figures.py``)
into the stats package so the four-arm analysis can reuse them. Holm correction
lives in :mod:`helion_rag.stats.gates` (``holm_adjust``).
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
from scipy import stats

DEFAULT_BOOTSTRAP_RESAMPLES = 200_000
DEFAULT_BOOTSTRAP_SEED = 20_260_724


class BootstrapInterval(NamedTuple):
    estimate: float
    low: float
    high: float


def geometric_mean(values: list[float]) -> float:
    """Geometric mean of strictly positive finite values (NaN when empty)."""
    finite = [v for v in values if math.isfinite(v) and v > 0.0]
    if not finite:
        return math.nan
    return math.exp(sum(math.log(v) for v in finite) / len(finite))


def bootstrap_geometric_mean_ci(
    values: list[float],
    *,
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    alpha: float = 0.05,
) -> BootstrapInterval:
    """Percentile kernel-bootstrap CI for the geometric mean (deterministic)."""
    finite = [v for v in values if math.isfinite(v) and v > 0.0]
    estimate = geometric_mean(finite)
    if len(finite) < 2:
        return BootstrapInterval(estimate, math.nan, math.nan)
    logged = np.log(np.asarray(finite, dtype=float))
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(finite), size=(resamples, len(finite)))
    replicates = np.exp(logged[indices].mean(axis=1))
    low, high = np.quantile(replicates, [alpha / 2.0, 1.0 - alpha / 2.0])
    return BootstrapInterval(estimate, float(low), float(high))


def wilcoxon_pvalue(log_ratios: list[float]) -> float:
    """Two-sided Wilcoxon signed-rank p on log-ratios (NaN if undefined)."""
    finite = [x for x in log_ratios if math.isfinite(x)]
    nonzero = [x for x in finite if x != 0.0]
    if len(nonzero) < 1:
        return math.nan
    try:
        result = stats.wilcoxon(
            nonzero, zero_method="wilcox", alternative="two-sided", method="auto"
        )
    except ValueError:
        return math.nan
    return float(result.pvalue)


def rank_biserial(log_ratios: list[float]) -> float:
    """Paired rank-biserial effect from signed ranks of log-ratios."""
    array = np.asarray([x for x in log_ratios if math.isfinite(x)], dtype=float)
    nonzero = array[array != 0.0]
    if nonzero.size == 0:
        return math.nan
    ranks = stats.rankdata(np.abs(nonzero))
    total = float(ranks.sum())
    if total == 0.0:
        return math.nan
    return float((ranks[nonzero < 0].sum() - ranks[nonzero > 0].sum()) / total)


def wins_ties_losses(ratios: list[float]) -> tuple[int, int, int]:
    """Count kernels favoring the numerator (<1), tied (==1), or worse (>1)."""
    wins = sum(1 for r in ratios if math.isfinite(r) and r < 1.0)
    ties = sum(
        1 for r in ratios if math.isfinite(r) and math.isclose(r, 1.0, abs_tol=1e-12)
    )
    losses = sum(1 for r in ratios if math.isfinite(r) and r > 1.0)
    return wins, ties, losses
