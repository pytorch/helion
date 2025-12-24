"""
Pluggable Heuristic Backends for AOT Autotuning
================================================

This module provides pluggable backends for generating configuration selection
heuristics. Each backend implements a different strategy for selecting optimal
configurations based on shape features.

Available backends:
- NearestNeighborsBackend: Selects config based on closest known shape
- DecisionTreeBackend: Uses a simple hand-rolled decision tree
- LightGBMBackend: Uses LightGBM for ML-based selection (default)
"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from dataclasses import field
import logging
from pathlib import Path
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


class NearestNeighborsBackend(HeuristicBackend):
    """
    Nearest neighbors heuristic backend.

    Selects the config that was best for the closest known shape
    (by Euclidean distance in feature space).
    """

    name: str = "nearest_neighbors"

    def __init__(self, normalize_features: bool = True) -> None:
        """
        Initialize the backend.

        Args:
            normalize_features: Whether to normalize features before distance computation
        """
        self.normalize_features = normalize_features

    def generate_heuristic(
        self,
        kernel_name: str,
        data: ShapeConfigData,
        selected_configs: list[Config],
        feature_names: list[str],
    ) -> HeuristicBackendResult:
        """Generate nearest neighbors heuristic code."""
        n_shapes = len(data.shape_features)

        # Build feature matrix
        X = np.zeros((n_shapes, len(feature_names)))
        for i, features in enumerate(data.shape_features):
            for j, fname in enumerate(feature_names):
                X[i, j] = features.get(fname, 0)

        # Compute normalization parameters
        if self.normalize_features:
            means = np.mean(X, axis=0)
            stds = np.std(X, axis=0)
            stds[stds == 0] = 1  # Avoid division by zero
        else:
            means = np.zeros(len(feature_names))
            stds = np.ones(len(feature_names))

        # For each shape, determine which selected config is best
        best_config_for_shape = np.zeros(n_shapes, dtype=int)
        for i in range(n_shapes):
            best_timing = np.inf
            best_config = 0
            for j, config_idx in enumerate(data.selected_config_indices):
                timing = data.timings[i, config_idx]
                if timing < best_timing:
                    best_timing = timing
                    best_config = j
            best_config_for_shape[i] = best_config

        # Compute accuracy (self-prediction)
        accuracy = 1.0  # Always correct for self-prediction

        # Generate code
        code = self._generate_code(
            kernel_name=kernel_name,
            configs=selected_configs,
            feature_names=feature_names,
            shape_features_list=[data.shape_features[i] for i in range(n_shapes)],
            best_config_for_shape=best_config_for_shape.tolist(),
            means=means.tolist(),
            stds=stds.tolist(),
        )

        return HeuristicBackendResult(
            generated_code=code,
            model_accuracy=accuracy,
            feature_names=feature_names,
        )

    def _generate_code(
        self,
        kernel_name: str,
        configs: list[Config],
        feature_names: list[str],
        shape_features_list: list[dict[str, Any]],
        best_config_for_shape: list[int],
        means: list[float],
        stds: list[float],
    ) -> str:
        """Generate Python code for nearest neighbors selection."""
        # Extract numeric features for each known shape
        known_shapes = []
        for features in shape_features_list:
            shape_vec = [features.get(fname, 0) for fname in feature_names]
            known_shapes.append(shape_vec)

        configs_code = "CONFIGS = [\n"
        for config in configs:
            configs_code += f"    {dict(config)!r},\n"
        configs_code += "]\n"

        return f'''"""
Auto-generated nearest neighbors heuristic for kernel: {kernel_name}
Backend: nearest_neighbors

Selects config based on closest known shape in feature space.
"""

{configs_code}

FEATURE_NAMES = {feature_names!r}

# Known shapes and their best configs
KNOWN_SHAPES = {known_shapes!r}
BEST_CONFIGS = {best_config_for_shape!r}

# Normalization parameters
FEATURE_MEANS = {means!r}
FEATURE_STDS = {stds!r}


def _extract_features(features: dict) -> list:
    """Extract feature vector from dict."""
    return [features.get(f, 0) for f in FEATURE_NAMES]


def _normalize(x: list) -> list:
    """Normalize feature vector."""
    return [(x[i] - FEATURE_MEANS[i]) / FEATURE_STDS[i] for i in range(len(x))]


def _distance(a: list, b: list) -> float:
    """Compute Euclidean distance between two vectors."""
    return sum((ai - bi) ** 2 for ai, bi in zip(a, b)) ** 0.5


def _find_nearest(x: list) -> int:
    """Find index of nearest known shape."""
    x_norm = _normalize(x)
    best_dist = float("inf")
    best_idx = 0
    for i, known in enumerate(KNOWN_SHAPES):
        known_norm = _normalize(known)
        dist = _distance(x_norm, known_norm)
        if dist < best_dist:
            best_dist = dist
            best_idx = i
    return best_idx


def _predict(features: dict) -> int:
    """Predict config index using nearest neighbors."""
    x = _extract_features(features)
    nearest_idx = _find_nearest(x)
    return BEST_CONFIGS[nearest_idx]


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


class LightGBMBackend(HeuristicBackend):
    """
    LightGBM-based heuristic backend.

    Uses LightGBM to train a classifier and generates code that loads
    the model at runtime.
    """

    name: str = "lightgbm"

    def generate_heuristic(
        self,
        kernel_name: str,
        data: ShapeConfigData,
        selected_configs: list[Config],
        feature_names: list[str],
    ) -> HeuristicBackendResult:
        """Generate LightGBM-based heuristic code."""
        try:
            import lightgbm as lgb
        except ImportError as e:
            raise ImportError(
                "LightGBM is required for this backend. Install with: pip install lightgbm"
            ) from e

        n_shapes = len(data.shape_features)

        # Build feature matrix
        X = np.zeros((n_shapes, len(feature_names)))
        for i, features in enumerate(data.shape_features):
            for j, fname in enumerate(feature_names):
                X[i, j] = features.get(fname, 0)

        # Build labels
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

        num_configs = len(selected_configs)

        # Handle edge cases
        if num_configs == 1:
            code = self._generate_single_config_code(
                kernel_name, selected_configs, feature_names
            )
            return HeuristicBackendResult(
                generated_code=code, model_accuracy=1.0, feature_names=feature_names
            )

        unique_labels = np.unique(y)
        if len(unique_labels) == 1:
            code = self._generate_single_config_code(
                kernel_name, selected_configs, feature_names
            )
            return HeuristicBackendResult(
                generated_code=code, model_accuracy=1.0, feature_names=feature_names
            )

        # Train model
        min_samples = max(1, min(5, n_shapes // 3))

        if num_configs == 2:
            params = {
                "objective": "binary",
                "metric": "binary_logloss",
                "verbosity": -1,
                "num_leaves": min(31, max(2, n_shapes)),
                "max_depth": min(6, max(1, n_shapes // 2)),
                "learning_rate": 0.1,
                "min_child_samples": min_samples,
            }
        else:
            params = {
                "objective": "multiclass",
                "num_class": num_configs,
                "metric": "multi_logloss",
                "verbosity": -1,
                "num_leaves": min(31, max(2, n_shapes)),
                "max_depth": min(6, max(1, n_shapes // 2)),
                "learning_rate": 0.1,
                "min_child_samples": min_samples,
            }

        num_boost_round = min(100, max(10, n_shapes * 5))
        train_data = lgb.Dataset(X, label=y, feature_name=feature_names)
        model = lgb.train(params, train_data, num_boost_round=num_boost_round)

        # Compute accuracy
        preds = model.predict(X)
        if num_configs == 2:
            predictions = (preds > 0.5).astype(int)
        else:
            predictions = np.argmax(preds, axis=1)
        accuracy = float(np.mean(predictions == y))

        # Generate code (runtime model loading)
        code = self._generate_code(
            kernel_name, selected_configs, feature_names, num_configs
        )

        # Save model to bytes for extra_files
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            model.save_model(f.name)
            model_bytes = Path(f.name).read_bytes()

        return HeuristicBackendResult(
            generated_code=code,
            model_accuracy=accuracy,
            feature_names=feature_names,
            extra_files={f"model_{kernel_name}.txt": model_bytes},
        )

    def _generate_single_config_code(
        self, kernel_name: str, configs: list[Config], feature_names: list[str]
    ) -> str:
        """Generate code for single config case."""
        configs_code = "CONFIGS = [\n"
        for config in configs:
            configs_code += f"    {dict(config)!r},\n"
        configs_code += "]\n"

        return f'''"""
Auto-generated LightGBM heuristic for kernel: {kernel_name}
Backend: lightgbm

Only one config selected - no model needed.
"""

{configs_code}

FEATURE_NAMES = {feature_names!r}


def _predict(features: dict) -> int:
    """Return the only config."""
    return 0


def select_config_{kernel_name}(features: dict) -> dict:
    """Select the optimal config for the given shape features."""
    return CONFIGS[0]


def select_config(kernel_name: str, features: dict) -> dict:
    """Generic config selector."""
    if kernel_name == "{kernel_name}":
        return select_config_{kernel_name}(features)
    raise ValueError(f"Unknown kernel: {{kernel_name}}")
'''

    def _generate_code(
        self,
        kernel_name: str,
        configs: list[Config],
        feature_names: list[str],
        num_configs: int,
    ) -> str:
        """Generate code that loads model at runtime."""
        configs_code = "CONFIGS = [\n"
        for config in configs:
            configs_code += f"    {dict(config)!r},\n"
        configs_code += "]\n"

        if num_configs == 2:
            predict_code = """
    probs = model.predict(x)
    return 1 if probs[0] > 0.5 else 0"""
        else:
            predict_code = """
    probs = model.predict(x)
    return int(np.argmax(probs))"""

        return f'''"""
Auto-generated LightGBM heuristic for kernel: {kernel_name}
Backend: lightgbm

Loads LightGBM model at runtime for prediction.
"""

import os
import numpy as np

{configs_code}

FEATURE_NAMES = {feature_names!r}

_model = None

def _load_model():
    global _model
    if _model is None:
        import lightgbm as lgb
        model_path = os.path.join(os.path.dirname(__file__), "model_{kernel_name}.txt")
        if os.path.exists(model_path):
            _model = lgb.Booster(model_file=model_path)
    return _model


def _predict(features: dict) -> int:
    """Predict config index using LightGBM model."""
    model = _load_model()
    if model is None:
        return 0  # Fallback
    x = np.array([[features.get(f, 0) for f in FEATURE_NAMES]]){predict_code}


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


class LGBMToCodeBackend(HeuristicBackend):
    """
    LightGBM backend that generates standalone Python code using lgbm-to-code.

    Trains a LightGBM model and converts it to human-readable Python code,
    so no model file is needed at runtime.
    """

    name: str = "lgbm_to_code"

    def generate_heuristic(
        self,
        kernel_name: str,
        data: ShapeConfigData,
        selected_configs: list[Config],
        feature_names: list[str],
    ) -> HeuristicBackendResult:
        """Generate standalone heuristic code using lgbm-to-code."""
        try:
            import lightgbm as lgb
        except ImportError as e:
            raise ImportError(
                "LightGBM is required for this backend. Install with: pip install lightgbm"
            ) from e

        try:
            from lgbm_to_code.lgbm_to_code import parse_lgbm_model
        except ImportError as e:
            raise ImportError(
                "lgbm-to-code is required for this backend. Install with: pip install lgbm-to-code"
            ) from e

        n_shapes = len(data.shape_features)

        # Build feature matrix
        X = np.zeros((n_shapes, len(feature_names)))
        for i, features in enumerate(data.shape_features):
            for j, fname in enumerate(feature_names):
                X[i, j] = features.get(fname, 0)

        # Build labels
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

        num_configs = len(selected_configs)

        # Handle edge cases
        if num_configs == 1:
            code = self._generate_single_config_code(
                kernel_name, selected_configs, feature_names
            )
            return HeuristicBackendResult(
                generated_code=code, model_accuracy=1.0, feature_names=feature_names
            )

        unique_labels = np.unique(y)
        if len(unique_labels) == 1:
            code = self._generate_single_config_code(
                kernel_name, selected_configs, feature_names
            )
            return HeuristicBackendResult(
                generated_code=code, model_accuracy=1.0, feature_names=feature_names
            )

        # Train model
        min_samples = max(1, min(5, n_shapes // 3))

        if num_configs == 2:
            params = {
                "objective": "binary",
                "metric": "binary_logloss",
                "verbosity": -1,
                "num_leaves": min(31, max(2, n_shapes)),
                "max_depth": min(6, max(1, n_shapes // 2)),
                "learning_rate": 0.1,
                "min_child_samples": min_samples,
            }
        else:
            params = {
                "objective": "multiclass",
                "num_class": num_configs,
                "metric": "multi_logloss",
                "verbosity": -1,
                "num_leaves": min(31, max(2, n_shapes)),
                "max_depth": min(6, max(1, n_shapes // 2)),
                "learning_rate": 0.1,
                "min_child_samples": min_samples,
            }

        num_boost_round = min(100, max(10, n_shapes * 5))
        train_data = lgb.Dataset(X, label=y, feature_name=feature_names)
        model = lgb.train(params, train_data, num_boost_round=num_boost_round)

        # Compute accuracy
        preds = model.predict(X)
        if num_configs == 2:
            predictions = (preds > 0.5).astype(int)
        else:
            predictions = np.argmax(preds, axis=1)
        accuracy = float(np.mean(predictions == y))

        # Generate code using lgbm-to-code
        code = self._generate_code(
            kernel_name,
            selected_configs,
            feature_names,
            model,
            num_configs,
            parse_lgbm_model,
        )

        return HeuristicBackendResult(
            generated_code=code,
            model_accuracy=accuracy,
            feature_names=feature_names,
        )

    def _generate_single_config_code(
        self, kernel_name: str, configs: list[Config], feature_names: list[str]
    ) -> str:
        """Generate code for single config case."""
        configs_code = "CONFIGS = [\n"
        for config in configs:
            configs_code += f"    {dict(config)!r},\n"
        configs_code += "]\n"

        return f'''"""
Auto-generated heuristic for kernel: {kernel_name}
Backend: lgbm_to_code

Only one config selected - no model needed.
"""

{configs_code}

FEATURE_NAMES = {feature_names!r}


def _predict(features: dict) -> int:
    """Return the only config."""
    return 0


def select_config_{kernel_name}(features: dict) -> dict:
    """Select the optimal config for the given shape features."""
    return CONFIGS[0]


def select_config(kernel_name: str, features: dict) -> dict:
    """Generic config selector."""
    if kernel_name == "{kernel_name}":
        return select_config_{kernel_name}(features)
    raise ValueError(f"Unknown kernel: {{kernel_name}}")
'''

    def _generate_code(
        self,
        kernel_name: str,
        configs: list[Config],
        feature_names: list[str],
        model: Any,
        num_configs: int,
        parse_lgbm_model: Any,
    ) -> str:
        """Generate standalone code using lgbm-to-code."""
        configs_code = "CONFIGS = [\n"
        for config in configs:
            configs_code += f"    {dict(config)!r},\n"
        configs_code += "]\n"

        # Get raw decision tree code from lgbm-to-code
        raw_code = parse_lgbm_model(model, "python")
        raw_code = self._fix_indentation(raw_code)

        # Get model info for multiclass handling
        model_json = model.dump_model()
        num_class = model_json.get("num_class", 1)
        num_trees = len(model_json.get("tree_info", []))

        if num_class > 1:
            # Multiclass: need to group trees by class and take argmax
            trees_per_class = num_trees // num_class  # noqa: F841
            class_tree_groups: list[list[int]] = []
            for c in range(num_class):
                trees = [i for i in range(num_trees) if i % num_class == c]
                class_tree_groups.append(trees)

            # Generate multiclass wrapper
            class_score_strs: list[str] = []
            for c, trees in enumerate(class_tree_groups):
                score_expr = " + ".join(f"func_{t}(x)" for t in trees)
                class_score_strs.append(f"        {score_expr},  # class {c}")

            decision_logic = f"""{raw_code}

def _predict(features: dict) -> int:
    \"\"\"Predict config index using decision trees.\"\"\"
    x = [features.get(f, 0) for f in FEATURE_NAMES]
    # Compute score for each class
    scores = [
{chr(10).join(class_score_strs)}
    ]
    return scores.index(max(scores))
"""
        else:
            # Binary: threshold at 0
            func_calls = " + ".join(f"func_{i}(x)" for i in range(num_trees))
            decision_logic = f"""{raw_code}

def _predict(features: dict) -> int:
    \"\"\"Predict config index using decision trees.\"\"\"
    x = [features.get(f, 0) for f in FEATURE_NAMES]
    total = {func_calls}
    return 1 if total > 0 else 0
"""

        return f'''"""
Auto-generated heuristic for kernel: {kernel_name}
Backend: lgbm_to_code

Standalone decision tree code generated from LightGBM model.
No model file needed at runtime.
"""

{configs_code}

FEATURE_NAMES = {feature_names!r}

{decision_logic}

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

    def _fix_indentation(self, code: str) -> str:
        """Fix indentation issues in lgbm-to-code output.

        lgbm-to-code generates code with no indentation like:
            def func_0(x):
            if x[0] <= 4608.0:
            if x[1] <= 3.0:
            return -0.693
            else:
            return -0.559
            else:
            return -0.826

        This function fixes it to proper Python indentation:
            def func_0(x):
                if x[0] <= 4608.0:
                    if x[1] <= 3.0:
                        return -0.693
                    else:
                        return -0.559
                else:
                    return -0.826
        """
        lines = code.split("\n")
        fixed_lines: list[str] = []
        # Stack to track (indent_level, in_else_body) for each nested if
        # in_else_body is True if we're in the else branch, False if in the if branch
        indent_stack: list[tuple[int, bool]] = []
        in_function = False

        for line in lines:
            stripped = line.strip()

            if not stripped:
                fixed_lines.append("")
                # Reset state if we hit an empty line between functions
                if in_function:
                    in_function = False
                    indent_stack = []
                continue

            if stripped.startswith("def "):
                # Start of a new function
                in_function = True
                indent_stack = []
                fixed_lines.append(stripped)
                continue

            if not in_function:
                fixed_lines.append(line)
                continue

            # Current indent level is 1 (base function level) + stack depth
            current_level = 1 + len(indent_stack)

            # Inside a function - handle if/else/return
            if stripped.startswith("if "):
                # if statement at current level
                fixed_lines.append("    " * current_level + stripped)
                # Push this if onto the stack (we're in its if-body, not else-body)
                indent_stack.append((current_level, False))
            elif stripped.startswith("else"):
                # else belongs to the most recent if
                if indent_stack:
                    if_level, _ = indent_stack[-1]
                    fixed_lines.append("    " * if_level + stripped)
                    # Mark that we're now in the else-body
                    indent_stack[-1] = (if_level, True)
                else:
                    # Fallback - shouldn't happen with valid lgbm-to-code output
                    fixed_lines.append("    " + stripped)
            elif stripped.startswith("return "):
                # return is at current body level
                fixed_lines.append("    " * current_level + stripped)
                # After return, check if we were in the else-body
                # If so, this if/else is complete, pop
                # If not (in if-body), else is coming, don't pop yet
                if indent_stack:
                    _, in_else = indent_stack[-1]
                    if in_else:
                        indent_stack.pop()
            else:
                # Other statements at current level
                fixed_lines.append("    " * current_level + stripped)

        return "\n".join(fixed_lines)


# Registry of available backends
HEURISTIC_BACKENDS: dict[str, type[HeuristicBackend]] = {
    "decision_tree": DecisionTreeBackend,
    "nearest_neighbors": NearestNeighborsBackend,
    "lightgbm": LightGBMBackend,
    "lgbm_to_code": LGBMToCodeBackend,
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


def _cross_validate_backend(
    backend: HeuristicBackend,
    kernel_name: str,
    data: ShapeConfigData,
    selected_configs: list[Config],
    feature_names: list[str],
    n_folds: int = 5,
) -> tuple[float, float]:
    """
    Cross-validate a backend to estimate generalization accuracy.

    Returns:
        Tuple of (train_accuracy, cv_accuracy)
    """
    n_shapes = len(data.shape_features)

    # If too few samples for CV, just return training accuracy
    if n_shapes < n_folds or n_shapes < 3:
        result = backend.generate_heuristic(
            kernel_name=kernel_name,
            data=data,
            selected_configs=selected_configs,
            feature_names=feature_names,
        )
        return result.model_accuracy, result.model_accuracy

    # Build labels for each shape
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

    # Adjust n_folds if needed
    n_folds = min(n_folds, n_shapes)

    # Create fold indices
    indices = np.arange(n_shapes)
    rng = np.random.default_rng(42)  # Reproducible
    rng.shuffle(indices)
    fold_size = n_shapes // n_folds

    cv_correct = 0
    cv_total = 0

    for fold in range(n_folds):
        # Split into train and test
        test_start = fold * fold_size
        test_end = test_start + fold_size if fold < n_folds - 1 else n_shapes
        test_indices = indices[test_start:test_end]
        train_indices = np.concatenate([indices[:test_start], indices[test_end:]])

        if len(train_indices) == 0 or len(test_indices) == 0:
            continue

        # Create training data
        train_features = [data.shape_features[i] for i in train_indices]
        train_timings = data.timings[train_indices, :]
        train_hashes = [data.shape_hashes[i] for i in train_indices]

        train_data = ShapeConfigData(
            shape_features=train_features,
            timings=train_timings,
            configs=data.configs,
            shape_hashes=train_hashes,
            config_hashes=data.config_hashes,
            selected_config_indices=data.selected_config_indices,
        )

        try:
            # Train on fold
            result = backend.generate_heuristic(
                kernel_name=kernel_name,
                data=train_data,
                selected_configs=selected_configs,
                feature_names=feature_names,
            )

            # Evaluate on test set by executing the generated code
            # We need to extract the _predict function and test it
            code = result.generated_code
            local_ns: dict[str, Any] = {}
            exec(compile(code, "<heuristic>", "exec"), local_ns)

            predict_fn = local_ns.get(f"select_config_{kernel_name}")
            if predict_fn is None:
                continue

            # Test predictions
            for test_idx in test_indices:
                test_features = data.shape_features[test_idx]
                true_label = y[test_idx]

                try:
                    pred_config = predict_fn(test_features)
                    # Find which index this config corresponds to
                    pred_label = None
                    for j, cfg in enumerate(selected_configs):
                        if dict(cfg) == pred_config:
                            pred_label = j
                            break

                    if pred_label is not None and pred_label == true_label:
                        cv_correct += 1
                except Exception:
                    pass  # Prediction failed

                cv_total += 1

        except Exception:
            continue

    # Compute CV accuracy
    cv_accuracy = cv_correct / cv_total if cv_total > 0 else 0.0

    # Also get training accuracy
    result = backend.generate_heuristic(
        kernel_name=kernel_name,
        data=data,
        selected_configs=selected_configs,
        feature_names=feature_names,
    )

    return result.model_accuracy, cv_accuracy


def evaluate_all_backends(
    kernel_name: str,
    data: ShapeConfigData,
    selected_configs: list[Config],
    feature_names: list[str],
    verbose: bool = True,
    cross_validate: bool = True,
    n_folds: int = 5,
) -> dict[str, dict[str, float]]:
    """
    Evaluate all available backends and return their accuracies.

    Args:
        kernel_name: Name of the kernel
        data: Shape and config data
        selected_configs: Selected configs for the heuristic
        feature_names: Feature names to use
        verbose: Whether to print results
        cross_validate: Whether to use cross-validation (default: True)
        n_folds: Number of CV folds (default: 5)

    Returns:
        Dict mapping backend name to dict with 'train' and 'cv' accuracies
    """
    import sys

    results: dict[str, dict[str, float]] = {}
    n_shapes = len(data.shape_features)

    if verbose:
        print(
            f"\n=== Backend Performance Comparison for {kernel_name} ===",
            file=sys.stderr,
        )
        if cross_validate and n_shapes >= 3:
            print(
                f"    ({n_folds}-fold cross-validation, {n_shapes} samples)",
                file=sys.stderr,
            )
        else:
            print(f"    (training accuracy only, {n_shapes} samples)", file=sys.stderr)

    for backend_name in HEURISTIC_BACKENDS:
        try:
            backend = get_backend(backend_name)

            if cross_validate and n_shapes >= 3:
                train_acc, cv_acc = _cross_validate_backend(
                    backend=backend,
                    kernel_name=kernel_name,
                    data=data,
                    selected_configs=selected_configs,
                    feature_names=feature_names,
                    n_folds=n_folds,
                )
                results[backend_name] = {"train": train_acc, "cv": cv_acc}
                if verbose:
                    print(
                        f"  {backend_name:<20} train: {train_acc:>6.1%}  cv: {cv_acc:>6.1%}",
                        file=sys.stderr,
                    )
            else:
                result = backend.generate_heuristic(
                    kernel_name=kernel_name,
                    data=data,
                    selected_configs=selected_configs,
                    feature_names=feature_names,
                )
                results[backend_name] = {
                    "train": result.model_accuracy,
                    "cv": result.model_accuracy,
                }
                if verbose:
                    print(
                        f"  {backend_name:<20} accuracy: {result.model_accuracy:.1%}",
                        file=sys.stderr,
                    )
        except Exception as e:
            results[backend_name] = {"train": -1.0, "cv": -1.0}
            if verbose:
                print(f"  {backend_name:<20} error: {e}", file=sys.stderr)

    if verbose:
        # Find best by CV score (or train if CV not available)
        score_key = "cv" if cross_validate else "train"
        best_backend = max(results, key=lambda k: results[k][score_key])
        print(f"  {'─' * 45}", file=sys.stderr)
        best_score = results[best_backend][score_key]
        print(f"  Best: {best_backend} ({best_score:.1%} {score_key})", file=sys.stderr)

    return results
