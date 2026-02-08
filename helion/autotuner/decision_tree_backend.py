"""
Decision Tree Backend for AOT Autotuning
=========================================

A simple hand-rolled decision tree backend for configuration selection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

import numpy as np

from .heuristic_generator import HeuristicBackend
from .heuristic_generator import HeuristicBackendResult
from .heuristic_generator import feature_to_var_name
from .heuristic_generator import generate_configs_code
from .heuristic_generator import generate_feature_extraction_code

if TYPE_CHECKING:
    from ..runtime.config import Config
    from .heuristic_generator import ShapeConfigData


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
        assert data.selected_config_indices is not None, (
            "selected_config_indices must be set"
        )
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
                return 0.0
            _, counts = np.unique(y, return_counts=True)
            probs = counts / len(y)
            return float(-np.sum(probs * np.log2(probs + 1e-10)))

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

    def _collect_used_features(self, tree: dict[str, Any]) -> set[str]:
        """Collect all feature names actually used in the tree."""
        if tree["leaf"]:
            return set()
        used = {tree["feature_name"]}
        used |= self._collect_used_features(tree["left"])
        used |= self._collect_used_features(tree["right"])
        return used

    def _generate_code(
        self,
        kernel_name: str,
        configs: list[Config],
        feature_names: list[str],
        tree: dict[str, Any],
    ) -> str:
        """Generate Python code for decision tree selection.

        Generates:
        - key_<kernel>(*args): Returns config index (cache key), with inlined extraction
        - autotune_<kernel>(*args): Returns config dict, with inlined configs
        """
        # Collect features actually used in decision tree
        used_features = self._collect_used_features(tree)
        used_features_list = sorted(used_features)

        # Generate inlined feature extraction code
        extract_lines = generate_feature_extraction_code(used_features_list)

        # Generate tree code using local variables
        tree_code = self._tree_to_code_inline(tree, indent=1)

        # Generate inlined configs for autotune
        configs_str = generate_configs_code(configs)

        return f'''"""
Auto-generated heuristic for kernel: {kernel_name}
Backend: decision_tree

Provides:
- key_{kernel_name}(*args): Returns config index (cache key)
- autotune_{kernel_name}(*args): Returns config dict for the given arguments
"""

import torch


def key_{kernel_name}(*args) -> int:
    """Select config index for the given arguments (also serves as cache key)."""
{extract_lines}
{tree_code}


def autotune_{kernel_name}(*args) -> dict:
    """Select the optimal config for the given arguments."""
    _C = [
{configs_str}
    ]
    return _C[key_{kernel_name}(*args)]
'''

    def _tree_to_code_inline(self, tree: dict[str, Any], indent: int) -> str:
        """Convert tree dict to Python code using local variables."""
        prefix = "    " * indent
        if tree["leaf"]:
            return f"{prefix}return {tree['class']}"

        feature_name = tree["feature_name"]
        var_name = feature_to_var_name(feature_name)
        threshold = tree["threshold"]

        code = f"{prefix}if {var_name} <= {threshold}:\n"
        code += self._tree_to_code_inline(tree["left"], indent + 1) + "\n"
        code += f"{prefix}else:\n"
        code += self._tree_to_code_inline(tree["right"], indent + 1)
        return code
