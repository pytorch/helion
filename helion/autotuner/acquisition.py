from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import norm

if TYPE_CHECKING:
    from numpy.typing import NDArray


def expected_improvement(
    mu: NDArray[np.float64],
    sigma: NDArray[np.float64],
    best_so_far: float,
    xi: float = 0.01,
) -> NDArray[np.float64]:
    """
    Expected Improvement acquisition function.

    Balances exploration (high uncertainty) and exploitation (low predicted value).

    Args:
        mu: GP mean predictions (N,).
        sigma: GP uncertainty (standard deviation) (N,).
        best_so_far: Current best (minimum) performance observed.
        xi: Exploration parameter (higher = more exploration).

    Returns:
        Expected improvement scores (higher = more valuable to evaluate).
    """
    # Avoid division by zero
    sigma = np.maximum(sigma, 1e-9)

    # We're minimizing, so improvement is best_so_far - mu
    improvement = best_so_far - mu - xi
    Z = improvement / sigma

    # Expected improvement formula
    ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)

    # If sigma is very small, just use the improvement
    ei = np.where(sigma > 1e-9, ei, np.maximum(improvement, 0.0))

    return ei


def upper_confidence_bound(
    mu: NDArray[np.float64],
    sigma: NDArray[np.float64],
    beta: float = 2.0,
) -> NDArray[np.float64]:
    """
    Upper Confidence Bound acquisition function.

    For minimization, we use Lower Confidence Bound (LCB).

    Args:
        mu: GP mean predictions (N,).
        sigma: GP uncertainty (standard deviation) (N,).
        beta: Exploration parameter (higher = more exploration).

    Returns:
        UCB scores (lower = more valuable for minimization).
    """
    # For minimization, we want lower confidence bound
    lcb = mu - beta * sigma
    return lcb


def probability_of_improvement(
    mu: NDArray[np.float64],
    sigma: NDArray[np.float64],
    best_so_far: float,
    xi: float = 0.01,
) -> NDArray[np.float64]:
    """
    Probability of Improvement acquisition function.

    Args:
        mu: GP mean predictions (N,).
        sigma: GP uncertainty (standard deviation) (N,).
        best_so_far: Current best (minimum) performance observed.
        xi: Exploration parameter.

    Returns:
        Probability of improvement scores.
    """
    sigma = np.maximum(sigma, 1e-9)
    improvement = best_so_far - mu - xi
    Z = improvement / sigma
    pi = norm.cdf(Z)
    return pi


def cost_aware_ei(
    mu: NDArray[np.float64],
    sigma: NDArray[np.float64],
    best_so_far: float,
    cost: float = 1.0,
    xi: float = 0.01,
) -> NDArray[np.float64]:
    """
    Cost-aware Expected Improvement.

    Normalizes EI by evaluation cost, useful for multi-fidelity optimization.

    Args:
        mu: GP mean predictions (N,).
        sigma: GP uncertainty (standard deviation) (N,).
        best_so_far: Current best (minimum) performance observed.
        cost: Cost of evaluation at this fidelity.
        xi: Exploration parameter.

    Returns:
        Cost-normalized expected improvement scores.
    """
    ei = expected_improvement(mu, sigma, best_so_far, xi)
    return ei / np.sqrt(cost)
