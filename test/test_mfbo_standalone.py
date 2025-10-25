#!/usr/bin/env python3
"""
Standalone test for Multi-Fidelity BO components that don't require Helion.
This tests the core ML components (GP, acquisition functions, encoding) in isolation.
"""

import sys
import numpy as np

def test_gaussian_process():
    """Test that GP model can be trained and used for predictions."""
    print("Testing Gaussian Process...")
    from helion.autotuner.gaussian_process import MultiFidelityGP

    gp = MultiFidelityGP()

    # Create some synthetic training data
    X_train = np.random.randn(10, 5)
    y_train = np.random.randn(10)

    # Train low-fidelity model
    gp.fit_low(X_train, y_train)
    assert gp.fitted_low, "GP should be fitted after fit_low"

    # Make predictions
    X_test = np.random.randn(3, 5)
    mu, sigma = gp.predict_low(X_test, return_std=True)

    assert len(mu) == 3, f"Expected 3 predictions, got {len(mu)}"
    assert len(sigma) == 3, f"Expected 3 uncertainties, got {len(sigma)}"
    assert np.all(sigma >= 0), "Uncertainty should be non-negative"

    # Train high-fidelity model
    gp.fit_high(X_train[:5], y_train[:5])
    assert gp.fitted_high, "GP should be fitted after fit_high"

    mu_high, sigma_high = gp.predict_high(X_test, return_std=True)

    assert len(mu_high) == 3
    assert len(sigma_high) == 3

    # Test multi-fidelity prediction
    mu_mf, sigma_mf = gp.predict_multifidelity(X_test)
    assert len(mu_mf) == 3
    assert len(sigma_mf) == 3

    # Test best observed
    best = gp.get_best_observed()
    assert best <= np.min(y_train), "Best should be at most the minimum observed value"

    print("✓ Gaussian Process tests passed")


def test_acquisition_functions():
    """Test acquisition functions work correctly."""
    print("Testing acquisition functions...")
    from helion.autotuner.acquisition import (
        expected_improvement,
        upper_confidence_bound,
        probability_of_improvement,
        cost_aware_ei,
    )

    mu = np.array([1.0, 2.0, 3.0])
    sigma = np.array([0.5, 1.0, 0.3])
    best_so_far = 2.5

    # Test Expected Improvement
    ei = expected_improvement(mu, sigma, best_so_far)
    assert len(ei) == 3, f"Expected 3 EI values, got {len(ei)}"
    assert np.all(ei >= 0), "EI should be non-negative"
    # Point with mu=1.0 should have highest EI since it's below best_so_far
    assert ei[0] > 0, "Best point should have positive EI"

    # Test UCB/LCB
    lcb = upper_confidence_bound(mu, sigma, beta=2.0)
    assert len(lcb) == 3
    # LCB for minimization should prefer lower values
    assert lcb[0] < lcb[2], "Lower mean should have lower LCB"

    # Test Probability of Improvement
    pi = probability_of_improvement(mu, sigma, best_so_far)
    assert len(pi) == 3
    assert np.all(pi >= 0) and np.all(pi <= 1), "PI should be in [0, 1]"

    # Test cost-aware EI
    cei = cost_aware_ei(mu, sigma, best_so_far, cost=2.0)
    assert len(cei) == 3
    assert np.all(cei >= 0), "Cost-aware EI should be non-negative"

    print("✓ Acquisition function tests passed")


def test_config_encoding_mock():
    """Test config encoding with mock data."""
    print("Testing config encoding (mock)...")
    from helion.autotuner.config_encoding import ConfigEncoder
    from helion.autotuner.config_fragment import (
        PowerOfTwoFragment,
        IntegerFragment,
        BooleanFragment,
        Category,
    )

    # Create mock config generation
    class MockConfigGen:
        def __init__(self):
            self.flat_spec = [
                PowerOfTwoFragment(16, 128, 32),  # Block size
                PowerOfTwoFragment(1, 8, 4),       # Num warps
                IntegerFragment(2, 5, 3),          # Num stages
                BooleanFragment(),                  # Some flag
            ]

    config_gen = MockConfigGen()
    encoder = ConfigEncoder(config_gen)

    # Test encoding
    flat_config = [32, 4, 3, True]
    encoded = encoder.encode(flat_config)

    assert encoded.ndim == 1, "Encoded should be 1D array"
    assert len(encoded) == encoder.encoded_dim, "Encoded dimension mismatch"
    assert len(encoded) > 0, "Encoded should not be empty"

    # Test bounds
    bounds = encoder.get_bounds()
    assert len(bounds) == len(encoded), "Bounds length should match encoding"

    # Test that different configs produce different encodings
    flat_config2 = [64, 8, 4, False]
    encoded2 = encoder.encode(flat_config2)
    assert not np.array_equal(encoded, encoded2), "Different configs should have different encodings"

    print("✓ Config encoding tests passed")


def main():
    """Run all standalone tests."""
    print("=" * 60)
    print("Multi-Fidelity BO Standalone Component Tests")
    print("=" * 60)
    print()

    try:
        test_gaussian_process()
        print()
        test_acquisition_functions()
        print()
        test_config_encoding_mock()
        print()
        print("=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        return 0
    except Exception as e:
        print()
        print("=" * 60)
        print(f"✗ Test failed: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
