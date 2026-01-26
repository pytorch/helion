"""
Heuristic Generator for AOT Autotuning
======================================

This module provides a pluggable backend for generating configuration selection
heuristics based on shape features.

Available backends:
- DecisionTreeBackend: Uses a simple hand-rolled decision tree (default)
- NearestNeighborBackend: Stores training shapes, finds closest match at runtime

The modular architecture allows registering custom backends via register_backend().

The workflow:
1. Load measurement data (kernel, shape, config, timing)
2. Determine the minimum set of configs needed to satisfy performance goals
3. Train a model (decision tree or nearest neighbor) to predict which config to use
4. Generate human-readable Python code
"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from dataclasses import field
import json
import logging
from pathlib import Path
import sys
from typing import Any
from typing import Literal

import numpy as np

from ..runtime.config import Config

log: logging.Logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes
# ============================================================================


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


PerformanceGoal = Literal["max_slowdown", "geomean_slowdown", "avg_slowdown"]


@dataclass
class PerformanceTarget:
    """Configuration for performance goals."""

    goal_type: PerformanceGoal = "max_slowdown"
    threshold: float = 1.1  # 10% slowdown allowed
    min_configs: int = 1
    max_configs: int = 10
    backend: str = (
        "decision_tree"  # Heuristic backend: decision_tree or nearest_neighbor
    )
    feature_selection: bool = True  # Whether to prune redundant features
    print_score_matrix: bool = True  # Whether to print the score matrix
    verbose: bool = True  # Verbose output
    skip_write: bool = False  # Skip writing files (for dump-code mode)


@dataclass
class HeuristicResult:
    """Result of heuristic generation."""

    selected_configs: list[Config]
    config_to_index: dict[str, int]
    performance_stats: dict[str, float]
    model_accuracy: float
    generated_code: str
    feature_selection_result: FeatureSelectionResult | None = None
    backend_used: str = "decision_tree"


# ============================================================================
# Code Generation Utilities
# ============================================================================


def generate_feature_extraction_code(feature_names: list[str]) -> str:
    """
    Generate Python code to extract features from kernel arguments.

    This is shared between heuristic backends to avoid code duplication.

    Supports two types of features:
    - Key-derived features (key_0, key_1, etc.): extract from positional args
    - Shape-derived features (arg0_dim1, etc.): extract tensor attributes

    Args:
        feature_names: List of feature names to extract

    Returns:
        Python code string that extracts features into local variables
    """
    import re

    if not feature_names:
        return "    # No features needed"

    # Check if these are key-derived features
    is_key_features = all(f.startswith("key_") for f in feature_names)

    extractions: dict[str, tuple[int, str]] = {}

    for feature in feature_names:
        if is_key_features:
            # Key-derived feature: key_0, key_1, etc.
            key_match = re.match(r"key_(\d+)", feature)
            if key_match:
                idx = int(key_match.group(1))
                expr = f"args[{idx}]"
                extractions[feature] = (idx, expr)
        else:
            # Shape-derived feature: arg0_dim1, etc.
            match = re.match(r"arg(\d+)_(.+)", feature)
            if match:
                arg_idx = int(match.group(1))
                attr = match.group(2)

                if attr == "ndim":
                    expr = f"args[{arg_idx}].ndim if len(args) > {arg_idx} and isinstance(args[{arg_idx}], torch.Tensor) else 0"
                elif attr.startswith("dim"):
                    dim_idx = int(attr[3:])
                    expr = f"int(args[{arg_idx}].shape[{dim_idx}]) if len(args) > {arg_idx} and isinstance(args[{arg_idx}], torch.Tensor) and args[{arg_idx}].ndim > {dim_idx} else 0"
                elif attr == "numel":
                    expr = f"int(args[{arg_idx}].numel()) if len(args) > {arg_idx} and isinstance(args[{arg_idx}], torch.Tensor) else 0"
                elif attr == "dtype":
                    expr = f"str(args[{arg_idx}].dtype) if len(args) > {arg_idx} and isinstance(args[{arg_idx}], torch.Tensor) else ''"
                elif attr == "dtype_size":
                    expr = f"args[{arg_idx}].element_size() if len(args) > {arg_idx} and isinstance(args[{arg_idx}], torch.Tensor) else 0"
                elif attr == "dtype_cat":
                    expr = f"_get_dtype_cat(args[{arg_idx}].dtype) if len(args) > {arg_idx} and isinstance(args[{arg_idx}], torch.Tensor) else 0"
                elif attr == "scalar":
                    expr = f"args[{arg_idx}] if len(args) > {arg_idx} and isinstance(args[{arg_idx}], (int, float)) else 0"
                else:
                    continue

                extractions[feature] = (arg_idx, expr)

    # Check if we need the dtype_cat helper
    needs_dtype_cat = (
        any("dtype_cat" in f for f in feature_names) and not is_key_features
    )

    lines: list[str] = []
    if needs_dtype_cat:
        lines.extend(
            [
                "    def _get_dtype_cat(dt):",
                "        if dt == torch.bool: return 0",
                "        if dt in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.uint16, torch.uint32, torch.uint64): return 1",
                "        if dt in (torch.float16, torch.bfloat16, torch.float32, torch.float64): return 2",
                "        if dt in (torch.complex64, torch.complex128): return 3",
                "        return 4",
                "",
            ]
        )

    for feature in sorted(extractions.keys()):
        _, expr = extractions[feature]
        var_name = feature_to_var_name(feature)
        lines.append(f"    {var_name} = {expr}")

    return "\n".join(lines)


def feature_to_var_name(feature: str) -> str:
    """Convert a feature name to a local variable name."""
    if feature.startswith("key_"):
        return f"_key_{feature[4:]}"  # _key_0, _key_1
    return feature.replace("arg", "_arg")  # _arg0_dim1


def generate_configs_code(configs: list[Config]) -> str:
    """Generate Python code for config list."""
    lines: list[str] = []
    for config in configs:
        lines.append(f"        {dict(config)!r},")
    return "\n".join(lines)


# ============================================================================
# Heuristic Backend Interface
# ============================================================================


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


# Registry of available backends - populated lazily
HEURISTIC_BACKENDS: dict[str, type[HeuristicBackend]] = {}


def _ensure_backends_loaded() -> None:
    """Ensure built-in backends are registered."""
    if "decision_tree" not in HEURISTIC_BACKENDS:
        from .decision_tree_backend import DecisionTreeBackend

        HEURISTIC_BACKENDS["decision_tree"] = DecisionTreeBackend

    if "nearest_neighbor" not in HEURISTIC_BACKENDS:
        from .nearest_neighbor_backend import NearestNeighborBackend

        HEURISTIC_BACKENDS["nearest_neighbor"] = NearestNeighborBackend


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


# ============================================================================
# Feature Selection
# ============================================================================


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


# ============================================================================
# Measurement Loading
# ============================================================================


def load_measurements(
    measurements_file: Path, kernel_name: str | None = None
) -> dict[str, ShapeConfigData]:
    """Load measurement data from CSV file."""
    import csv

    if not measurements_file.exists():
        return {}

    # Group measurements by kernel
    kernel_data: dict[str, dict[str, dict[str, Any]]] = {}

    with open(measurements_file, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            kname = row["kernel_name"]
            if kernel_name is not None and kname != kernel_name:
                continue

            if kname not in kernel_data:
                kernel_data[kname] = {}

            shape_hash = row["shape_hash"]
            config_hash = row["config_hash"]

            if shape_hash not in kernel_data[kname]:
                kernel_data[kname][shape_hash] = {
                    "features": json.loads(row["shape_features"]),
                    "configs": {},
                }

            kernel_data[kname][shape_hash]["configs"][config_hash] = {
                "config": json.loads(row["config"]),
                "timing_ms": float(row["timing_ms"]),
            }

    # Convert to ShapeConfigData format
    result: dict[str, ShapeConfigData] = {}
    for kname, shapes in kernel_data.items():
        # Get all unique configs
        all_config_hashes: set[str] = set()
        for shape_data in shapes.values():
            all_config_hashes.update(shape_data["configs"].keys())

        config_hashes = sorted(all_config_hashes)
        shape_hashes = sorted(shapes.keys())

        # Build configs list
        config_list: list[Config] = []
        config_map: dict[str, Config] = {}
        for shape_data in shapes.values():
            for chash, cdata in shape_data["configs"].items():
                if chash not in config_map:
                    config_map[chash] = Config(**cdata["config"])
        config_list = [config_map[h] for h in config_hashes]

        # Build timing matrix
        timings = np.full((len(shape_hashes), len(config_hashes)), np.inf)
        shape_features: list[dict[str, Any]] = []

        for i, shash in enumerate(shape_hashes):
            shape_data = shapes[shash]
            shape_features.append(shape_data["features"])
            for j, chash in enumerate(config_hashes):
                if chash in shape_data["configs"]:
                    timings[i, j] = shape_data["configs"][chash]["timing_ms"]

        result[kname] = ShapeConfigData(
            kernel_name=kname,
            shape_features=shape_features,
            timings=timings,
            configs=config_list,
            shape_hashes=shape_hashes,
            config_hashes=config_hashes,
        )

    return result


# ============================================================================
# Config Selection
# ============================================================================


def select_config_subset(
    data: ShapeConfigData,
    target: PerformanceTarget,
) -> tuple[list[int], dict[str, float]]:
    """
    Select a minimal subset of configs that satisfies the performance goal.

    Uses a greedy algorithm:
    1. Start with the config that is optimal for the most shapes
    2. Add configs until performance goal is met for all shapes

    Returns:
        Tuple of (selected config indices, performance stats)
    """
    n_shapes, n_configs = data.timings.shape

    # Find the best timing for each shape (oracle performance)
    best_per_shape = np.min(data.timings, axis=1)

    # Track which shapes are satisfied
    selected_indices: list[int] = []
    satisfied = np.zeros(n_shapes, dtype=bool)

    # Current best achievable timing with selected configs
    current_best = np.full(n_shapes, np.inf)

    while len(selected_indices) < target.max_configs:
        if satisfied.all():
            break

        # Score each remaining config by how many unsatisfied shapes it helps
        best_score = -1
        best_config_idx = -1

        for config_idx in range(n_configs):
            if config_idx in selected_indices:
                continue

            # Compute new best if we add this config
            new_best = np.minimum(current_best, data.timings[:, config_idx])
            slowdowns = new_best / best_per_shape

            if target.goal_type == "max_slowdown":
                score = np.sum(slowdowns <= target.threshold)
            elif target.goal_type == "geomean_slowdown":
                score = np.sum(
                    np.exp(np.mean(np.log(slowdowns + 1e-10))) <= target.threshold
                )
            else:  # avg_slowdown
                score = np.sum(np.mean(slowdowns) <= target.threshold)

            if score > best_score:
                best_score = score
                best_config_idx = config_idx

        if best_config_idx == -1:
            break

        selected_indices.append(best_config_idx)
        current_best = np.minimum(current_best, data.timings[:, best_config_idx])

        # Update satisfied
        slowdowns = current_best / best_per_shape
        if target.goal_type == "max_slowdown":
            satisfied = slowdowns <= target.threshold
        elif target.goal_type == "geomean_slowdown":
            geomean = np.exp(np.mean(np.log(slowdowns + 1e-10)))
            satisfied[:] = geomean <= target.threshold
        else:  # avg_slowdown
            avg = np.mean(slowdowns)
            satisfied[:] = avg <= target.threshold

    # Compute final stats
    slowdowns = current_best / best_per_shape
    stats = {
        "max_slowdown": float(np.max(slowdowns)),
        "geomean_slowdown": float(np.exp(np.mean(np.log(slowdowns + 1e-10)))),
        "avg_slowdown": float(np.mean(slowdowns)),
        "satisfied_ratio": float(np.mean(satisfied)),
        "num_configs": len(selected_indices),
    }

    return selected_indices, stats


# ============================================================================
# Heuristic Generation
# ============================================================================


def generate_heuristic(
    measurements_file: Path,
    output_dir: Path,
    kernel_name: str | None = None,
    target: PerformanceTarget | None = None,
    kernel_source_files: dict[str, str] | None = None,
) -> dict[str, HeuristicResult]:
    """
    Generate heuristics for all kernels in the measurements file.

    Args:
        measurements_file: Path to the measurements CSV
        output_dir: Directory to write heuristic files
        kernel_name: Optional specific kernel to process
        target: Performance target configuration
        kernel_source_files: Optional dict mapping kernel names to source file paths.
            If provided, heuristics are also saved next to source files as
            _<filename>_<device>_<compute>.py

    Returns:
        Dictionary mapping kernel names to HeuristicResult
    """
    from .aot_cache import get_hardware_info

    if target is None:
        target = PerformanceTarget()

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load measurements
    all_data = load_measurements(measurements_file, kernel_name)
    results: dict[str, HeuristicResult] = {}

    # Get device info for naming heuristic files
    hw = get_hardware_info()
    device_kind, compute_kind = hw.device_kind, hw.compute_capability

    for kname, data in all_data.items():
        log.info(f"Generating heuristic for kernel: {kname}")
        if target.verbose:
            print(
                f"\n=== Generating heuristic for kernel: {kname} ===", file=sys.stderr
            )

        # Print score matrix if requested
        if target.print_score_matrix:
            print_score_matrix(data)

        # Select config subset
        selected_indices, stats = select_config_subset(data, target)
        selected_configs = [data.configs[i] for i in selected_indices]

        log.info(
            f"  Selected {len(selected_configs)} configs with "
            f"max_slowdown={stats['max_slowdown']:.2f}x, "
            f"geomean_slowdown={stats['geomean_slowdown']:.2f}x"
        )
        if target.verbose:
            print(
                f"\nSelected {len(selected_configs)} configs: "
                f"max_slowdown={stats['max_slowdown']:.2f}x, "
                f"geomean_slowdown={stats['geomean_slowdown']:.2f}x",
                file=sys.stderr,
            )

        # Build config index mapping
        config_to_index = {
            data.config_hashes[i]: j for j, i in enumerate(selected_indices)
        }

        # Perform feature selection if requested
        feature_selection_result: FeatureSelectionResult | None = None
        if target.feature_selection:
            feature_selection_result = select_shape_features(
                data.shape_features, verbose=target.verbose
            )
            feature_names = feature_selection_result.selected_features
        else:
            # Get all numeric feature names
            feature_names = []
            if data.shape_features:
                for key, value in data.shape_features[0].items():
                    if isinstance(value, (int, float)):
                        feature_names.append(key)

        # Set selected config indices for the backend
        data.selected_config_indices = selected_indices

        # Generate heuristic using the selected backend
        backend = get_backend(target.backend)
        backend_result = backend.generate_heuristic(
            kernel_name=kname,
            data=data,
            selected_configs=selected_configs,
            feature_names=feature_names,
        )

        code = backend_result.generated_code
        accuracy = backend_result.model_accuracy

        log.info(f"  Model accuracy: {accuracy:.2%}")
        if target.verbose:
            print(f"\nModel accuracy: {accuracy:.2%}", file=sys.stderr)

        # Save heuristic code to output_dir (run directory)
        if not target.skip_write:
            heuristic_file = output_dir / f"heuristic_{kname}.py"
            heuristic_file.write_text(code)
            log.info(f"  Saved heuristic to {heuristic_file}")

        results[kname] = HeuristicResult(
            selected_configs=selected_configs,
            config_to_index=config_to_index,
            performance_stats=stats,
            model_accuracy=accuracy,
            generated_code=code,
            feature_selection_result=feature_selection_result,
            backend_used=target.backend,
        )

    # Group kernels by source file and save combined heuristics
    if kernel_source_files and not target.skip_write:
        # Group kernel names by their source file
        source_to_kernels: dict[str, list[str]] = {}
        for kname in results:
            if kname in kernel_source_files:
                source_file = kernel_source_files[kname]
                if source_file not in source_to_kernels:
                    source_to_kernels[source_file] = []
                source_to_kernels[source_file].append(kname)

        # Generate combined heuristic for each source file
        for source_file, knames in source_to_kernels.items():
            source_path = Path(source_file)
            if not source_path.exists():
                continue

            # Create heuristic filename: _helion_aot_<basename>_<device>_<compute>.py
            base_name = source_path.stem
            heuristic_name = f"_helion_aot_{base_name}_{device_kind}_{compute_kind}.py"
            source_heuristic_file = source_path.parent / heuristic_name

            # Combine heuristics for all kernels in this source file
            combined_code = _combine_heuristics(knames, results)
            source_heuristic_file.write_text(combined_code)
            log.info(
                f"  Saved combined heuristic for {len(knames)} kernel(s) to: {source_heuristic_file}"
            )

    return results


def _combine_heuristics(
    kernel_names: list[str],
    results: dict[str, HeuristicResult],
) -> str:
    """
    Combine heuristic code from multiple kernels into a single file.
    """
    # For a single kernel, just return its generated code directly
    if len(kernel_names) == 1:
        return results[kernel_names[0]].generated_code

    # For multiple kernels, we need to combine them
    lines = [
        '"""',
        f"Auto-generated heuristics for kernels: {', '.join(kernel_names)}",
        "Backend: decision_tree",
        "",
        "Provides for each kernel:",
        "- key_<kernel>(*args): Cache key function",
        "- autotune_<kernel>(*args): Config selection function",
        '"""',
        "",
        "import torch",
        "",
    ]

    # For each kernel, extract and include the relevant parts from generated code
    for kname in kernel_names:
        result = results[kname]
        code = result.generated_code

        # Find end of docstring - look for the second occurrence of """
        first_triple = code.find('"""')
        if first_triple >= 0:
            second_triple = code.find('"""', first_triple + 3)
            if second_triple >= 0:
                after_docstring = code[second_triple + 3 :].lstrip("\n")
            else:
                after_docstring = code
        else:
            after_docstring = code

        # Remove "import torch" line since we have it at the top
        after_docstring = after_docstring.replace("import torch\n", "")

        # Extract everything except the generic select_config function at the end
        kernel_code_lines = []
        code_lines = after_docstring.split("\n")
        for line in code_lines:
            # Stop at the generic select_config function
            if line.startswith("def select_config(kernel_name:"):
                break
            kernel_code_lines.append(line)

        # Make variable/function names kernel-specific to avoid clobbering
        kernel_code = "\n".join(kernel_code_lines)
        kernel_suffix = f"_{kname}"
        kernel_upper = f"_{kname.upper()}"

        # Rename globals and helper functions to be kernel-specific
        kernel_code = kernel_code.replace("CONFIGS", f"CONFIGS{kernel_upper}")
        kernel_code = kernel_code.replace(
            "FEATURE_NAMES", f"FEATURE_NAMES{kernel_upper}"
        )
        kernel_code = kernel_code.replace(
            "USED_FEATURES", f"USED_FEATURES{kernel_upper}"
        )
        kernel_code = kernel_code.replace(
            "def _extract_features(", f"def _extract_features{kernel_suffix}("
        )
        kernel_code = kernel_code.replace(
            "_extract_features(*args)", f"_extract_features{kernel_suffix}(*args)"
        )
        kernel_code = kernel_code.replace(
            "def _predict(", f"def _predict{kernel_suffix}("
        )
        kernel_code = kernel_code.replace(
            "_predict(features)", f"_predict{kernel_suffix}(features)"
        )

        lines.extend([f"# === Kernel: {kname} ===", kernel_code, ""])

    return "\n".join(lines)


# ============================================================================
# Heuristic Evaluation
# ============================================================================


def evaluate_heuristic(
    measurements_file: Path,
    heuristic_dir: Path,
    kernel_name: str | None = None,
) -> dict[str, dict[str, float]]:
    """
    Evaluate heuristic performance against measurements.

    Returns dict mapping kernel names to performance metrics.
    """
    import importlib.util

    all_data = load_measurements(measurements_file, kernel_name)
    results: dict[str, dict[str, float]] = {}

    for kname, data in all_data.items():
        heuristic_file = heuristic_dir / f"heuristic_{kname}.py"
        if not heuristic_file.exists():
            log.warning(f"Heuristic file not found for {kname}")
            continue

        # Load heuristic module
        spec = importlib.util.spec_from_file_location("heuristic", heuristic_file)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        select_fn = getattr(module, f"select_config_{kname}", None)
        if select_fn is None:
            continue

        # Evaluate on all shapes
        best_per_shape = np.min(data.timings, axis=1)
        heuristic_timings = np.zeros(len(data.shape_features))

        for i, features in enumerate(data.shape_features):
            try:
                selected_config = select_fn(features)
                # Find this config in our data
                for j, config in enumerate(data.configs):
                    if dict(config) == selected_config:
                        heuristic_timings[i] = data.timings[i, j]
                        break
                else:
                    heuristic_timings[i] = np.inf
            except Exception as e:
                log.warning(f"Heuristic failed for shape {i}: {e}")
                heuristic_timings[i] = np.inf

        # Compute stats
        slowdowns = heuristic_timings / best_per_shape
        valid = np.isfinite(slowdowns)

        results[kname] = {
            "max_slowdown": float(np.max(slowdowns[valid]))
            if valid.any()
            else float("inf"),
            "geomean_slowdown": float(np.exp(np.mean(np.log(slowdowns[valid] + 1e-10))))
            if valid.any()
            else float("inf"),
            "avg_slowdown": float(np.mean(slowdowns[valid]))
            if valid.any()
            else float("inf"),
            "coverage": float(np.mean(valid)),
        }

        log.info(
            f"Heuristic evaluation for {kname}: "
            f"max_slowdown={results[kname]['max_slowdown']:.2f}x, "
            f"coverage={results[kname]['coverage']:.1%}"
        )

    return results
