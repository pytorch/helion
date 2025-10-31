#!/usr/bin/env python3
"""
Standalone test for Multi-Fidelity BO components using direct imports.
This tests the core ML components (GP, acquisition functions) in isolation.
"""

from __future__ import annotations

import os
import sys

# Add helion autotuner directory to path to allow direct imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "helion", "autotuner"))

import numpy as np


def test_gaussian_process():
    """Test that GP model can be trained and used for predictions."""
    print("Testing Gaussian Process...")

    # Direct import from the file
    from gaussian_process import MultiFidelityGP

    gp = MultiFidelityGP()

    # Create some synthetic training data
    rng = np.random.default_rng(42)
    X_train = rng.standard_normal((10, 5))
    y_train = rng.standard_normal(10)

    # Train low-fidelity model
    gp.fit_low(X_train, y_train)
    assert gp.fitted_low, "GP should be fitted after fit_low"

    # Make predictions
    X_test = rng.standard_normal((3, 5))
    mu, sigma = gp.predict_low(X_test, return_std=True)

    assert len(mu) == 3, f"Expected 3 predictions, got {len(mu)}"
    assert len(sigma) == 3, f"Expected 3 uncertainties, got {len(sigma)}"
    assert np.all(sigma >= 0), "Uncertainty should be non-negative"
    print(f"  Low-fidelity predictions: mu={mu}, sigma={sigma}")

    # Train high-fidelity model
    gp.fit_high(X_train[:5], y_train[:5])
    assert gp.fitted_high, "GP should be fitted after fit_high"

    mu_high, sigma_high = gp.predict_high(X_test, return_std=True)

    assert len(mu_high) == 3
    assert len(sigma_high) == 3
    print(f"  High-fidelity predictions: mu={mu_high}, sigma={sigma_high}")

    # Test multi-fidelity prediction
    mu_mf, sigma_mf = gp.predict_multifidelity(X_test)
    assert len(mu_mf) == 3
    assert len(sigma_mf) == 3
    print(f"  Multi-fidelity predictions: mu={mu_mf}, sigma={sigma_mf}")

    # Test best observed
    best = gp.get_best_observed()
    assert best <= np.min(y_train), "Best should be at most the minimum observed value"
    print(f"  Best observed: {best:.4f} (min y_train: {np.min(y_train):.4f})")

    print("✓ Gaussian Process tests passed")
    return True


def test_acquisition_functions():
    """Test acquisition functions work correctly."""
    print("\nTesting acquisition functions...")

    from acquisition import cost_aware_ei
    from acquisition import expected_improvement
    from acquisition import probability_of_improvement
    from acquisition import upper_confidence_bound

    mu = np.array([1.0, 2.0, 3.0])
    sigma = np.array([0.5, 1.0, 0.3])
    best_so_far = 2.5

    # Test Expected Improvement
    ei = expected_improvement(mu, sigma, best_so_far)
    assert len(ei) == 3, f"Expected 3 EI values, got {len(ei)}"
    assert np.all(ei >= 0), "EI should be non-negative"
    # Point with mu=1.0 should have highest EI since it's below best_so_far
    assert ei[0] > 0, "Best point should have positive EI"
    print(f"  Expected Improvement: {ei}")
    print(f"    Best candidate: index {np.argmax(ei)} with EI={np.max(ei):.4f}")

    # Test UCB/LCB
    lcb = upper_confidence_bound(mu, sigma, beta=2.0)
    assert len(lcb) == 3
    # LCB for minimization should prefer lower values
    assert lcb[0] < lcb[2], "Lower mean should have lower LCB"
    print(f"  Lower Confidence Bound: {lcb}")
    print(f"    Best candidate: index {np.argmin(lcb)} with LCB={np.min(lcb):.4f}")

    # Test Probability of Improvement
    pi = probability_of_improvement(mu, sigma, best_so_far)
    assert len(pi) == 3
    assert np.all(pi >= 0) and np.all(pi <= 1), "PI should be in [0, 1]"
    print(f"  Probability of Improvement: {pi}")

    # Test cost-aware EI
    cei = cost_aware_ei(mu, sigma, best_so_far, cost=2.0)
    assert len(cei) == 3
    assert np.all(cei >= 0), "Cost-aware EI should be non-negative"
    print(f"  Cost-aware EI (cost=2.0): {cei}")

    print("✓ Acquisition function tests passed")
    return True


def main():
    """Run all standalone tests."""
    print("=" * 60)
    print("Multi-Fidelity BO Component Tests")
    print("=" * 60)

    try:
        test_gaussian_process()
        test_acquisition_functions()

        print("\n" + "=" * 60)
        print("✓ All component tests passed!")
        print("=" * 60)
        return 0
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"✗ Test failed: {e}")
        print("=" * 60)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
