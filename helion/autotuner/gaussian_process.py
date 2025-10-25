from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel
from sklearn.gaussian_process.kernels import Matern

if TYPE_CHECKING:
    from numpy.typing import NDArray


class MultiFidelityGP:
    """
    Multi-fidelity Gaussian Process model for kernel autotuning.

    Uses separate GP models for low and high fidelity evaluations,
    with the low-fidelity model informing the high-fidelity predictions.
    """

    def __init__(self, noise_level: float = 1e-6) -> None:
        """
        Initialize the multi-fidelity GP model.

        Args:
            noise_level: Regularization parameter for numerical stability.
        """
        self.noise_level = noise_level
        # Separate GP for each fidelity level
        # Using Matérn 5/2 kernel (good for non-smooth functions)
        kernel = ConstantKernel(1.0) * Matern(nu=2.5, length_scale=1.0)

        self.gp_low = GaussianProcessRegressor(
            kernel=kernel,
            alpha=noise_level,
            normalize_y=True,
            n_restarts_optimizer=2,
            random_state=42,
        )
        self.gp_high = GaussianProcessRegressor(
            kernel=kernel,
            alpha=noise_level,
            normalize_y=True,
            n_restarts_optimizer=2,
            random_state=42,
        )

        self.X_low: NDArray[np.float64] | None = None
        self.y_low: NDArray[np.float64] | None = None
        self.X_high: NDArray[np.float64] | None = None
        self.y_high: NDArray[np.float64] | None = None
        self.fitted_low = False
        self.fitted_high = False

    def fit_low(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> None:
        """
        Train the low-fidelity GP model.

        Args:
            X: Input configurations (N x D).
            y: Performance measurements (N,).
        """
        if len(X) == 0 or len(y) == 0:
            return

        self.X_low = X.copy()
        self.y_low = y.copy()
        self.gp_low.fit(X, y)
        self.fitted_low = True

    def fit_high(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> None:
        """
        Train the high-fidelity GP model.

        Args:
            X: Input configurations (N x D).
            y: Performance measurements (N,).
        """
        if len(X) == 0 or len(y) == 0:
            return

        self.X_high = X.copy()
        self.y_high = y.copy()
        self.gp_high.fit(X, y)
        self.fitted_high = True

    def predict_low(
        self, X: NDArray[np.float64], return_std: bool = True
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]] | NDArray[np.float64]:
        """
        Predict performance at low fidelity.

        Args:
            X: Input configurations (N x D).
            return_std: Whether to return standard deviation.

        Returns:
            Mean predictions and optionally standard deviations.
        """
        if not self.fitted_low:
            if return_std:
                return np.zeros(len(X)), np.ones(len(X))
            return np.zeros(len(X))

        return self.gp_low.predict(X, return_std=return_std)  # type: ignore

    def predict_high(
        self, X: NDArray[np.float64], return_std: bool = True
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]] | NDArray[np.float64]:
        """
        Predict performance at high fidelity.

        If high-fidelity model is trained, use it.
        Otherwise, fall back to low-fidelity predictions.

        Args:
            X: Input configurations (N x D).
            return_std: Whether to return standard deviation.

        Returns:
            Mean predictions and optionally standard deviations.
        """
        if self.fitted_high:
            return self.gp_high.predict(X, return_std=return_std)  # type: ignore
        elif self.fitted_low:
            # Use low-fidelity as fallback with increased uncertainty
            mu_low, std_low = self.gp_low.predict(X, return_std=True)  # type: ignore
            if return_std:
                # Increase uncertainty since we're using low-fidelity
                return mu_low, std_low * 1.5  # type: ignore
            return mu_low  # type: ignore
        else:
            if return_std:
                return np.zeros(len(X)), np.ones(len(X))
            return np.zeros(len(X))

    def predict_multifidelity(
        self, X: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Predict using both fidelity levels when available.

        Combines low and high fidelity predictions with uncertainty-weighted averaging.

        Args:
            X: Input configurations (N x D).

        Returns:
            Combined mean predictions and standard deviations.
        """
        if self.fitted_high and self.fitted_low:
            mu_low, std_low = self.gp_low.predict(X, return_std=True)  # type: ignore
            mu_high, std_high = self.gp_high.predict(X, return_std=True)  # type: ignore

            # Variance-weighted combination
            var_low = std_low**2
            var_high = std_high**2

            # Avoid division by zero
            total_precision = 1.0 / (var_low + 1e-10) + 1.0 / (var_high + 1e-10)
            mu_combined = (mu_low / (var_low + 1e-10) + mu_high / (var_high + 1e-10)) / total_precision
            var_combined = 1.0 / total_precision
            std_combined = np.sqrt(var_combined)

            return mu_combined, std_combined  # type: ignore
        elif self.fitted_high:
            return self.predict_high(X, return_std=True)  # type: ignore
        else:
            return self.predict_low(X, return_std=True)  # type: ignore

    def get_best_observed(self) -> float:
        """
        Get the best (minimum) performance observed so far.

        Returns:
            The minimum performance value.
        """
        best = float("inf")
        if self.y_high is not None and len(self.y_high) > 0:
            best = min(best, float(np.min(self.y_high)))
        if self.y_low is not None and len(self.y_low) > 0:
            best = min(best, float(np.min(self.y_low)))
        return best
