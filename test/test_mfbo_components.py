#!/usr/bin/env python3
"""
Unit tests for Multi-Fidelity BO core components.
Tests the ML components (GP, acquisition functions) in isolation.
"""

from __future__ import annotations

import numpy as np

from helion._testing import TestCase


class TestMFBOComponents(TestCase):
    """Test Multi-Fidelity BO components (GP, acquisition functions)."""

    def test_gaussian_process(self):
        """Test that GP model can be trained and used for predictions."""
        from helion.autotuner.gaussian_process import MultiFidelityGP

        gp = MultiFidelityGP()

        # Create some synthetic training data
        rng = np.random.default_rng(42)
        X_train = rng.standard_normal((10, 5))
        y_train = rng.standard_normal(10)

        # Train low-fidelity model
        gp.fit_low(X_train, y_train)
        self.assertTrue(gp.fitted_low, "GP should be fitted after fit_low")

        # Make predictions
        X_test = rng.standard_normal((3, 5))
        mu, sigma = gp.predict_low(X_test, return_std=True)

        self.assertEqual(len(mu), 3, f"Expected 3 predictions, got {len(mu)}")
        self.assertEqual(len(sigma), 3, f"Expected 3 uncertainties, got {len(sigma)}")
        self.assertTrue(np.all(sigma >= 0), "Uncertainty should be non-negative")

        # Train high-fidelity model
        gp.fit_high(X_train[:5], y_train[:5])
        self.assertTrue(gp.fitted_high, "GP should be fitted after fit_high")

        mu_high, sigma_high = gp.predict_high(X_test, return_std=True)

        self.assertEqual(len(mu_high), 3)
        self.assertEqual(len(sigma_high), 3)

        # Test multi-fidelity prediction
        mu_mf, sigma_mf = gp.predict_multifidelity(X_test)
        self.assertEqual(len(mu_mf), 3)
        self.assertEqual(len(sigma_mf), 3)

        # Test best observed
        best = gp.get_best_observed()
        self.assertLessEqual(
            best, np.min(y_train), "Best should be at most the minimum observed value"
        )

    def test_expected_improvement(self):
        """Test Expected Improvement acquisition function."""
        from helion.autotuner.acquisition import expected_improvement

        mu = np.array([1.0, 2.0, 3.0])
        sigma = np.array([0.5, 1.0, 0.3])
        best_so_far = 2.5

        ei = expected_improvement(mu, sigma, best_so_far)
        self.assertEqual(len(ei), 3, f"Expected 3 EI values, got {len(ei)}")
        self.assertTrue(np.all(ei >= 0), "EI should be non-negative")
        # Point with mu=1.0 should have highest EI since it's below best_so_far
        self.assertGreater(ei[0], 0, "Best point should have positive EI")

    def test_upper_confidence_bound(self):
        """Test Upper Confidence Bound (UCB/LCB) acquisition function."""
        from helion.autotuner.acquisition import upper_confidence_bound

        mu = np.array([1.0, 2.0, 3.0])
        sigma = np.array([0.5, 1.0, 0.3])

        lcb = upper_confidence_bound(mu, sigma, beta=2.0)
        self.assertEqual(len(lcb), 3)
        # LCB for minimization should prefer lower values
        self.assertLess(lcb[0], lcb[2], "Lower mean should have lower LCB")

    def test_probability_of_improvement(self):
        """Test Probability of Improvement acquisition function."""
        from helion.autotuner.acquisition import probability_of_improvement

        mu = np.array([1.0, 2.0, 3.0])
        sigma = np.array([0.5, 1.0, 0.3])
        best_so_far = 2.5

        pi = probability_of_improvement(mu, sigma, best_so_far)
        self.assertEqual(len(pi), 3)
        self.assertTrue(np.all(pi >= 0) and np.all(pi <= 1), "PI should be in [0, 1]")

    def test_cost_aware_ei(self):
        """Test cost-aware Expected Improvement."""
        from helion.autotuner.acquisition import cost_aware_ei

        mu = np.array([1.0, 2.0, 3.0])
        sigma = np.array([0.5, 1.0, 0.3])
        best_so_far = 2.5

        cei = cost_aware_ei(mu, sigma, best_so_far, cost=2.0)
        self.assertEqual(len(cei), 3)
        self.assertTrue(np.all(cei >= 0), "Cost-aware EI should be non-negative")
