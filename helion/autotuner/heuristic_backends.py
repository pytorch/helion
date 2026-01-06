"""
Pluggable Heuristic Backends for AOT Autotuning
================================================

This module provides a pluggable backend for generating configuration selection
heuristics. The backend implements a strategy for selecting optimal
configurations based on shape features.

Available backends:
- DecisionTreeBackend: Uses a simple hand-rolled decision tree (default)

The modular architecture allows registering custom backends via register_backend().
"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from dataclasses import field
import logging
from typing import TYPE_CHECKING
from typing import Any

import numpy as np

if TYPE_CHECKING:
    from ..runtime.config import Config

log: logging.Logger = logging.getLogger(__name__)


@dataclass
class FeatureSelectionResult:
    """Result of shape feature selection/pruning."""

    original_features: list[str]
    selected_features: list[str]
    removed_features: list[str]
    removal_reasons: dict[str, str]  # feature -> reason for removal


@dataclass
class ShapeConfigData:
    """Data for training heuristic models and storing measurement results."""

    kernel_name: str  # Name of the kernel
    shape_features: list[dict[str, Any]]  # Features for each shape
    timings: np.ndarray  # shape: (n_shapes, n_configs)
    configs: list[Config]  # All unique configs
    shape_hashes: list[str]  # Unique identifier for each shape
    config_hashes: list[str]  # Unique identifier for each config
    selected_config_indices: list[int] | None = (
        None  # Which configs were selected (set during heuristic generation)
    )


@dataclass
class HeuristicBackendResult:
    """Result from a heuristic backend."""

    generated_code: str
    model_accuracy: float
    feature_names: list[str]
    extra_files: dict[str, bytes] = field(default_factory=dict)  # filename -> content


class HeuristicBackend(ABC):
    """Base class for heuristic generation backends."""

    name: str = "base"

    @abstractmethod
    def generate_heuristic(
        self,
        kernel_name: str,
        data: ShapeConfigData,
        selected_configs: list[Config],
        feature_names: list[str],
    ) -> HeuristicBackendResult:
        """
        Generate heuristic code for config selection.

        Args:
            kernel_name: Name of the kernel
            data: Shape and config data
            selected_configs: Configs selected for the heuristic
            feature_names: Feature names to use

        Returns:
            HeuristicBackendResult with generated code and metadata
        """


def select_shape_features(
    shape_features: list[dict[str, Any]],
    verbose: bool = True,
) -> FeatureSelectionResult:
    """
    Select relevant shape features by removing redundant ones.

    Removes:
    - Features with only one unique value (constant)
    - Features that are fully dependent on other features (correlated)

    Args:
        shape_features: List of feature dicts for each shape
        verbose: Whether to print feature selection info

    Returns:
        FeatureSelectionResult with selected and removed features
    """
    import sys

    if not shape_features:
        return FeatureSelectionResult(
            original_features=[],
            selected_features=[],
            removed_features=[],
            removal_reasons={},
        )

    # Get all numeric feature names
    all_features: list[str] = []
    for key, value in shape_features[0].items():
        if isinstance(value, (int, float)):
            all_features.append(key)

    if not all_features:
        return FeatureSelectionResult(
            original_features=[],
            selected_features=[],
            removed_features=[],
            removal_reasons={},
        )

    # Build feature matrix
    n_shapes = len(shape_features)
    n_features = len(all_features)
    X = np.zeros((n_shapes, n_features))

    for i, features in enumerate(shape_features):
        for j, fname in enumerate(all_features):
            X[i, j] = features.get(fname, 0)

    # Track removed features and reasons
    removed_features: list[str] = []
    removal_reasons: dict[str, str] = {}

    # 1. Remove constant features (only one unique value)
    constant_mask = np.zeros(n_features, dtype=bool)
    for j in range(n_features):
        unique_values = np.unique(X[:, j])
        if len(unique_values) <= 1:
            constant_mask[j] = True
            removed_features.append(all_features[j])
            if len(unique_values) == 0:
                removal_reasons[all_features[j]] = "no values"
            else:
                removal_reasons[all_features[j]] = f"constant value: {unique_values[0]}"

    # 2. Remove features that are fully dependent on others
    # (perfectly correlated or anti-correlated)
    remaining_indices = np.where(~constant_mask)[0]
    dependent_mask = np.zeros(n_features, dtype=bool)

    for i, idx_i in enumerate(remaining_indices):
        if dependent_mask[idx_i]:
            continue
        for idx_j in remaining_indices[i + 1 :]:
            if dependent_mask[idx_j]:
                continue

            # Check if feature j is fully determined by feature i
            col_i = X[:, idx_i]
            col_j = X[:, idx_j]

            # Check for perfect correlation
            if np.std(col_i) > 0 and np.std(col_j) > 0:
                corr = np.corrcoef(col_i, col_j)[0, 1]
                if np.abs(corr) > 0.9999:
                    # Feature j is fully dependent on feature i
                    dependent_mask[idx_j] = True
                    removed_features.append(all_features[idx_j])
                    removal_reasons[all_features[idx_j]] = (
                        f"fully dependent on {all_features[idx_i]} (corr={corr:.4f})"
                    )

            # Check for exact functional dependency (j = f(i))
            # Group by values of i and check if j is constant within each group
            unique_i = np.unique(col_i)
            if len(unique_i) > 1:
                is_dependent = True
                for val_i in unique_i:
                    mask = col_i == val_i
                    if len(np.unique(col_j[mask])) > 1:
                        is_dependent = False
                        break
                if is_dependent and not dependent_mask[idx_j]:
                    dependent_mask[idx_j] = True
                    removed_features.append(all_features[idx_j])
                    removal_reasons[all_features[idx_j]] = (
                        f"functionally dependent on {all_features[idx_i]}"
                    )

    # Build selected features list
    selected_features = []
    for j, fname in enumerate(all_features):
        if not constant_mask[j] and not dependent_mask[j]:
            selected_features.append(fname)

    result = FeatureSelectionResult(
        original_features=all_features,
        selected_features=selected_features,
        removed_features=removed_features,
        removal_reasons=removal_reasons,
    )

    # Print summary if verbose
    if verbose:
        print("\n=== Shape Feature Selection ===", file=sys.stderr)
        print(f"Original features: {len(all_features)}", file=sys.stderr)
        print(f"Selected features: {len(selected_features)}", file=sys.stderr)
        print(f"Removed features: {len(removed_features)}", file=sys.stderr)

        if removed_features:
            print("\nRemoved features:", file=sys.stderr)
            for fname in removed_features:
                print(f"  - {fname}: {removal_reasons[fname]}", file=sys.stderr)

        if selected_features:
            print("\nSurviving features:", file=sys.stderr)
            for fname in selected_features:
                print(f"  + {fname}", file=sys.stderr)

        print("=" * 32, file=sys.stderr)

    return result


def print_score_matrix(
    data: ShapeConfigData,
    shape_labels: list[str] | None = None,
    config_labels: list[str] | None = None,
) -> None:
    """
    Print a matrix showing timings for each shape x config combination.

    Args:
        data: Shape and config data
        shape_labels: Optional labels for shapes (defaults to shape_hashes)
        config_labels: Optional labels for configs (defaults to config_hashes)
    """
    import sys

    n_shapes, n_configs = data.timings.shape

    if shape_labels is None:
        shape_labels = [h[:8] for h in data.shape_hashes]
    if config_labels is None:
        config_labels = [h[:8] for h in data.config_hashes]

    # Find best config for each shape
    best_per_shape = np.argmin(data.timings, axis=1)

    print("\n=== Score Matrix (shapes x configs) ===", file=sys.stderr)
    print("Times in ms, * = best for shape, - = invalid\n", file=sys.stderr)

    # Header
    header = f"{'Shape':<12}"
    for clabel in config_labels:
        header += f" {clabel:>10}"
    header += f" {'Best':>10}"
    print(header, file=sys.stderr)
    print("-" * len(header), file=sys.stderr)

    # Rows
    for i in range(n_shapes):
        row = f"{shape_labels[i]:<12}"
        for j in range(n_configs):
            timing = data.timings[i, j]
            if np.isinf(timing):
                cell = "-"
            elif j == best_per_shape[i]:
                cell = f"*{timing:.4f}"
            else:
                cell = f"{timing:.4f}"
            row += f" {cell:>10}"

        # Best timing
        best_timing = np.min(data.timings[i, :])
        if np.isinf(best_timing):
            row += f" {'-':>10}"
        else:
            row += f" {best_timing:.4f}"

        print(row, file=sys.stderr)

    print("\n" + "=" * 40, file=sys.stderr)

    # Summary statistics
    valid_timings = data.timings[~np.isinf(data.timings)]
    if len(valid_timings) > 0:
        print(f"Total measurements: {len(valid_timings)}", file=sys.stderr)
        print(
            f"Invalid measurements: {np.sum(np.isinf(data.timings))}", file=sys.stderr
        )
        print(f"Min timing: {np.min(valid_timings):.4f}ms", file=sys.stderr)
        print(f"Max timing: {np.max(valid_timings):.4f}ms", file=sys.stderr)


# Registry of available backends - populated lazily
HEURISTIC_BACKENDS: dict[str, type[HeuristicBackend]] = {}


def _ensure_backends_loaded() -> None:
    """Ensure built-in backends are registered."""
    if "decision_tree" not in HEURISTIC_BACKENDS:
        from .decision_tree_backend import DecisionTreeBackend

        HEURISTIC_BACKENDS["decision_tree"] = DecisionTreeBackend


def get_backend(name: str, **kwargs: int) -> HeuristicBackend:
    """
    Get a heuristic backend by name.

    Args:
        name: Backend name
        **kwargs: Backend-specific arguments

    Returns:
        HeuristicBackend instance
    """
    _ensure_backends_loaded()
    if name not in HEURISTIC_BACKENDS:
        raise ValueError(
            f"Unknown backend: {name}. Available: {list(HEURISTIC_BACKENDS.keys())}"
        )
    return HEURISTIC_BACKENDS[name](**kwargs)


def register_backend(name: str, backend_class: type[HeuristicBackend]) -> None:
    """Register a custom heuristic backend."""
    HEURISTIC_BACKENDS[name] = backend_class


def get_all_backend_names() -> list[str]:
    """Get list of all registered backend names."""
    _ensure_backends_loaded()
    return list(HEURISTIC_BACKENDS.keys())
