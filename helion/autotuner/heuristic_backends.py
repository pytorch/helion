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
    """Data for training heuristic models."""

    shape_features: list[dict[str, Any]]  # Features for each shape
    timings: np.ndarray  # shape: (n_shapes, n_configs)
    configs: list[Config]  # All unique configs
    shape_hashes: list[str]  # Unique identifier for each shape
    config_hashes: list[str]  # Unique identifier for each config
    selected_config_indices: list[int]  # Which configs were selected for the model


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


class DecisionTreeBackend(HeuristicBackend):
    """
    Simple hand-rolled decision tree backend.

    Creates a decision tree by recursively splitting on the feature
    that best separates the best configs.
    """

    name: str = "decision_tree"

    def __init__(self, max_depth: int = 6, min_samples_split: int = 2) -> None:
        """
        Initialize the backend.

        Args:
            max_depth: Maximum depth of the decision tree
            min_samples_split: Minimum samples required to split
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def generate_heuristic(
        self,
        kernel_name: str,
        data: ShapeConfigData,
        selected_configs: list[Config],
        feature_names: list[str],
    ) -> HeuristicBackendResult:
        """Generate decision tree heuristic code."""
        n_shapes = len(data.shape_features)

        # Build feature matrix
        X = np.zeros((n_shapes, len(feature_names)))
        for i, features in enumerate(data.shape_features):
            for j, fname in enumerate(feature_names):
                X[i, j] = features.get(fname, 0)

        # For each shape, determine which selected config is best
        y = np.zeros(n_shapes, dtype=int)
        for i in range(n_shapes):
            best_timing = np.inf
            best_config = 0
            for j, config_idx in enumerate(data.selected_config_indices):
                timing = data.timings[i, config_idx]
                if timing < best_timing:
                    best_timing = timing
                    best_config = j
            y[i] = best_config

        # Build decision tree
        tree = self._build_tree(X, y, feature_names, depth=0)

        # Compute accuracy
        predictions = np.array(
            [self._predict_tree(tree, X[i]) for i in range(n_shapes)]
        )
        accuracy = float(np.mean(predictions == y))

        # Generate code
        code = self._generate_code(
            kernel_name=kernel_name,
            configs=selected_configs,
            feature_names=feature_names,
            tree=tree,
        )

        return HeuristicBackendResult(
            generated_code=code,
            model_accuracy=accuracy,
            feature_names=feature_names,
        )

    def _build_tree(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str],
        depth: int,
    ) -> dict[str, Any]:
        """Recursively build a decision tree."""
        n_samples = len(y)
        n_classes = len(np.unique(y))

        # Base cases: leaf node
        if (
            depth >= self.max_depth
            or n_samples < self.min_samples_split
            or n_classes == 1
        ):
            # Return most common class
            unique, counts = np.unique(y, return_counts=True)
            return {"leaf": True, "class": int(unique[np.argmax(counts)])}

        # Find best split
        best_gain = -1
        best_feature = 0
        best_threshold = 0.0

        for j in range(X.shape[1]):
            values = np.unique(X[:, j])
            if len(values) <= 1:
                continue

            for threshold in values[:-1]:
                left_mask = X[:, j] <= threshold
                right_mask = ~left_mask

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                gain = self._information_gain(y, y[left_mask], y[right_mask])
                if gain > best_gain:
                    best_gain = gain
                    best_feature = j
                    best_threshold = float(threshold)

        # If no good split found, return leaf
        if best_gain <= 0:
            unique, counts = np.unique(y, return_counts=True)
            return {"leaf": True, "class": int(unique[np.argmax(counts)])}

        # Split and recurse
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        return {
            "leaf": False,
            "feature": best_feature,
            "feature_name": feature_names[best_feature],
            "threshold": best_threshold,
            "left": self._build_tree(
                X[left_mask], y[left_mask], feature_names, depth + 1
            ),
            "right": self._build_tree(
                X[right_mask], y[right_mask], feature_names, depth + 1
            ),
        }

    def _information_gain(
        self, parent: np.ndarray, left: np.ndarray, right: np.ndarray
    ) -> float:
        """Compute information gain for a split."""

        def entropy(y: np.ndarray) -> float:
            if len(y) == 0:
                return 0
            _, counts = np.unique(y, return_counts=True)
            probs = counts / len(y)
            return -np.sum(probs * np.log2(probs + 1e-10))

        n = len(parent)
        n_left = len(left)
        n_right = len(right)

        return (
            entropy(parent)
            - (n_left / n) * entropy(left)
            - (n_right / n) * entropy(right)
        )

    def _predict_tree(self, tree: dict[str, Any], x: np.ndarray) -> int:
        """Predict class for a single sample using the tree."""
        if tree["leaf"]:
            return tree["class"]
        if x[tree["feature"]] <= tree["threshold"]:
            return self._predict_tree(tree["left"], x)
        return self._predict_tree(tree["right"], x)

    def _generate_code(
        self,
        kernel_name: str,
        configs: list[Config],
        feature_names: list[str],
        tree: dict[str, Any],
    ) -> str:
        """Generate Python code for decision tree selection."""
        configs_code = "CONFIGS = [\n"
        for config in configs:
            configs_code += f"    {dict(config)!r},\n"
        configs_code += "]\n"

        # Generate tree code
        tree_code = self._tree_to_code(tree, indent=1)

        return f'''"""
Auto-generated decision tree heuristic for kernel: {kernel_name}
Backend: decision_tree

Uses a hand-rolled decision tree for config selection.
"""

{configs_code}

FEATURE_NAMES = {feature_names!r}


def _predict(features: dict) -> int:
    """Predict config index using decision tree."""
{tree_code}


def select_config_{kernel_name}(features: dict) -> dict:
    """Select the optimal config for the given shape features."""
    config_idx = _predict(features)
    return CONFIGS[config_idx]


def select_config(kernel_name: str, features: dict) -> dict:
    """Generic config selector."""
    if kernel_name == "{kernel_name}":
        return select_config_{kernel_name}(features)
    raise ValueError(f"Unknown kernel: {{kernel_name}}")
'''

    def _tree_to_code(self, tree: dict[str, Any], indent: int) -> str:
        """Convert tree dict to Python code."""
        prefix = "    " * indent
        if tree["leaf"]:
            return f"{prefix}return {tree['class']}"

        feature_name = tree["feature_name"]
        threshold = tree["threshold"]

        code = f'{prefix}if features.get("{feature_name}", 0) <= {threshold}:\n'
        code += self._tree_to_code(tree["left"], indent + 1) + "\n"
        code += f"{prefix}else:\n"
        code += self._tree_to_code(tree["right"], indent + 1)
        return code


# Registry of available backends
HEURISTIC_BACKENDS: dict[str, type[HeuristicBackend]] = {
    "decision_tree": DecisionTreeBackend,
}


def get_backend(name: str, **kwargs: Any) -> HeuristicBackend:
    """
    Get a heuristic backend by name.

    Args:
        name: Backend name
        **kwargs: Backend-specific arguments

    Returns:
        HeuristicBackend instance
    """
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
    return list(HEURISTIC_BACKENDS.keys())
