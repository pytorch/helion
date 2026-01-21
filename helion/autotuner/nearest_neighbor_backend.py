"""
Nearest Neighbor Backend for AOT Autotuning
============================================

A lookup-based backend that matches on categorical features exactly,
then finds the closest numeric match with a "prefer lower" heuristic.

This approach:
1. Separates features into categorical (dtype, etc.) vs numeric (sizes)
2. Requires categorical features to match exactly
3. For numeric features, finds the highest values ≤ current (safer configs)
4. Falls back to lowest values > current if no lower match exists
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .heuristic_generator import HeuristicBackend
from .heuristic_generator import HeuristicBackendResult
from .heuristic_generator import feature_to_var_name
from .heuristic_generator import generate_configs_code
from .heuristic_generator import generate_feature_extraction_code

if TYPE_CHECKING:
    from ..runtime.config import Config
    from .heuristic_generator import ShapeConfigData


# Features that are categorical (must match exactly)
CATEGORICAL_FEATURES = {"dtype", "dtype_cat", "dtype_size", "ndim"}


def is_categorical_feature(name: str) -> bool:
    """Check if a feature is categorical based on its name."""
    return any(cat in name for cat in CATEGORICAL_FEATURES)


class NearestNeighborBackend(HeuristicBackend):
    """
    Lookup-based backend with categorical matching and numeric interpolation.

    For a new input:
    1. Filter training points to those with matching categorical features
    2. Among matches, find the one with highest numeric values ≤ current
    3. If no such point, use the one with lowest numeric values > current
    """

    name: str = "nearest_neighbor"

    def generate_heuristic(
        self,
        kernel_name: str,
        data: ShapeConfigData,
        selected_configs: list[Config],
        feature_names: list[str],
    ) -> HeuristicBackendResult:
        """Generate nearest neighbor heuristic code."""
        n_shapes = len(data.shape_features)

        # Separate categorical vs numeric features
        categorical_features = [f for f in feature_names if is_categorical_feature(f)]
        numeric_features = [f for f in feature_names if not is_categorical_feature(f)]

        # Build feature data for each training point
        training_data: list[dict[str, float | int | str]] = []
        for features in data.shape_features:
            point: dict[str, float | int | str] = {}
            for fname in feature_names:
                point[fname] = features.get(fname, 0)
            training_data.append(point)

        # For each shape, determine which selected config is best
        assert data.selected_config_indices is not None, (
            "selected_config_indices must be set"
        )
        best_config_per_shape: list[int] = []
        for i in range(n_shapes):
            best_timing = np.inf
            best_config = 0
            for j, config_idx in enumerate(data.selected_config_indices):
                timing = data.timings[i, config_idx]
                if timing < best_timing:
                    best_timing = timing
                    best_config = j
            best_config_per_shape.append(best_config)

        # Compute accuracy using leave-one-out
        correct = 0
        for i in range(n_shapes):
            # Predict for point i using all other points
            predicted = self._predict(
                training_data,
                best_config_per_shape,
                training_data[i],
                categorical_features,
                numeric_features,
                exclude_idx=i,
            )
            if predicted == best_config_per_shape[i]:
                correct += 1
        accuracy = correct / n_shapes if n_shapes > 0 else 0.0

        # Generate code
        code = self._generate_code(
            kernel_name=kernel_name,
            configs=selected_configs,
            feature_names=feature_names,
            categorical_features=categorical_features,
            numeric_features=numeric_features,
            training_data=training_data,
            best_config_per_shape=best_config_per_shape,
        )

        return HeuristicBackendResult(
            generated_code=code,
            model_accuracy=accuracy,
            feature_names=feature_names,
        )

    def _predict(
        self,
        training_data: list[dict[str, float | int | str]],
        best_configs: list[int],
        query: dict[str, float | int | str],
        categorical_features: list[str],
        numeric_features: list[str],
        exclude_idx: int = -1,
    ) -> int:
        """Predict config for a query point."""
        # Filter to points with matching categorical features
        candidates: list[tuple[int, dict[str, float | int | str]]] = []
        for i, point in enumerate(training_data):
            if i == exclude_idx:
                continue
            # Check categorical match
            match = True
            for fname in categorical_features:
                if point.get(fname) != query.get(fname):
                    match = False
                    break
            if match:
                candidates.append((i, point))

        if not candidates:
            # No categorical match - fall back to first point
            return best_configs[0] if best_configs else 0

        if not numeric_features:
            # No numeric features - just return first match
            return best_configs[candidates[0][0]]

        # Among candidates, find best numeric match
        # Prefer highest values that are all <= query values
        best_lower_idx = -1
        best_lower_score = float("-inf")

        # Also track closest higher as fallback
        best_higher_idx = -1
        best_higher_score = float("inf")

        for idx, point in candidates:
            # Compute "score" - sum of numeric values
            score = sum(float(point.get(f, 0)) for f in numeric_features)

            # Check if all numeric values are <= query
            all_lower_or_equal = all(
                float(point.get(f, 0)) <= float(query.get(f, 0))
                for f in numeric_features
            )

            if all_lower_or_equal:
                if score > best_lower_score:
                    best_lower_score = score
                    best_lower_idx = idx
            else:
                if score < best_higher_score:
                    best_higher_score = score
                    best_higher_idx = idx

        # Prefer lower, fall back to higher
        if best_lower_idx >= 0:
            return best_configs[best_lower_idx]
        if best_higher_idx >= 0:
            return best_configs[best_higher_idx]

        # Shouldn't happen, but fall back to first candidate
        return best_configs[candidates[0][0]]

    def _generate_code(
        self,
        kernel_name: str,
        configs: list[Config],
        feature_names: list[str],
        categorical_features: list[str],
        numeric_features: list[str],
        training_data: list[dict[str, float | int | str]],
        best_config_per_shape: list[int],
    ) -> str:
        """Generate Python code for nearest neighbor selection."""
        # Generate feature extraction code
        extract_lines = generate_feature_extraction_code(feature_names)

        # Generate training data as list of tuples: (categorical_values, numeric_values, config_idx)
        # Format: [((cat1, cat2, ...), (num1, num2, ...), config_idx), ...]
        training_tuples: list[str] = []
        for i, point in enumerate(training_data):
            cat_vals = tuple(point.get(f, 0) for f in categorical_features)
            num_vals = tuple(float(point.get(f, 0)) for f in numeric_features)
            config_idx = best_config_per_shape[i]
            training_tuples.append(f"    ({cat_vals!r}, {num_vals!r}, {config_idx}),")

        training_str = "\n".join(training_tuples)

        # Generate config list
        configs_str = generate_configs_code(configs)

        # Generate categorical and numeric feature name lists
        cat_features_str = repr(categorical_features)
        num_features_str = repr(numeric_features)

        # Generate variable names for feature tuples (with trailing comma for single-element tuples)
        cat_var_names = ", ".join(feature_to_var_name(f) for f in categorical_features)
        num_var_names = ", ".join(feature_to_var_name(f) for f in numeric_features)

        # Generate tuple expressions (handle empty case properly)
        if categorical_features:
            cat_tuple_expr = f"({cat_var_names},)"
        else:
            cat_tuple_expr = "()"
        if numeric_features:
            num_tuple_expr = f"({num_var_names},)"
        else:
            num_tuple_expr = "()"

        return f'''"""
Auto-generated heuristic for kernel: {kernel_name}
Backend: nearest_neighbor

Provides:
- key_{kernel_name}(*args): Returns config index (cache key)
- autotune_{kernel_name}(*args): Returns config dict for the given arguments

Matching strategy:
1. Match categorical features exactly (dtype, etc.)
2. Find highest numeric values <= query (prefer lower/safer configs)
3. Fall back to lowest numeric values > query if no lower match
"""

import torch

# Training data: (categorical_values, numeric_values, config_idx)
_TRAIN_{kernel_name.upper()} = [
{training_str}
]

_CAT_FEATURES_{kernel_name.upper()} = {cat_features_str}
_NUM_FEATURES_{kernel_name.upper()} = {num_features_str}


def key_{kernel_name}(*args) -> int:
    """Select config index for the given arguments (also serves as cache key)."""
{extract_lines}

    # Build categorical and numeric feature tuples
    cat_vals = {cat_tuple_expr}
    num_vals = {num_tuple_expr}

    # Find matching training points
    candidates = []
    for train_cat, train_num, config_idx in _TRAIN_{kernel_name.upper()}:
        if train_cat == cat_vals:
            candidates.append((train_num, config_idx))

    if not candidates:
        # No categorical match - return first config
        return _TRAIN_{kernel_name.upper()}[0][2] if _TRAIN_{kernel_name.upper()} else 0

    if not num_vals or len(num_vals) == 0:
        # No numeric features - return first match
        return candidates[0][1]

    # Find best numeric match: highest values <= query, else lowest > query
    best_lower_idx = -1
    best_lower_score = float("-inf")
    best_higher_idx = -1
    best_higher_score = float("inf")

    for train_num, config_idx in candidates:
        score = sum(train_num)
        all_lower = all(t <= q for t, q in zip(train_num, num_vals))
        if all_lower:
            if score > best_lower_score:
                best_lower_score = score
                best_lower_idx = config_idx
        else:
            if score < best_higher_score:
                best_higher_score = score
                best_higher_idx = config_idx

    if best_lower_idx >= 0:
        return best_lower_idx
    if best_higher_idx >= 0:
        return best_higher_idx
    return candidates[0][1]


def autotune_{kernel_name}(*args) -> dict:
    """Select the optimal config for the given arguments."""
    _C = [
{configs_str}
    ]
    return _C[key_{kernel_name}(*args)]
'''
