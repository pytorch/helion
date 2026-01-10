"""
Heuristic Generator for AOT Autotuning
======================================

This module provides a pluggable backend for generating configuration selection
heuristics and generates human-readable heuristics using decision trees
to select optimal configurations based on shape features.

Available backends:
- DecisionTreeBackend: Uses a simple hand-rolled decision tree (default)

The modular architecture allows registering custom backends via register_backend().

The workflow:
1. Load measurement data (kernel, shape, config, timing)
2. Determine the minimum set of configs needed to satisfy performance goals
3. Train a decision tree to predict which config to use
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
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal

import numpy as np

from ..runtime.config import Config

if TYPE_CHECKING:
    pass

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
    # Output hashes for each (shape, config) combination
    # Dict mapping (shape_hash, config_hash) -> output_hash
    # Used for computing config equivalence classes
    output_hashes: dict[tuple[str, str], str] | None = None
    # Batch handling mode: "blind" or "hash_equivalent"
    # When "hash_equivalent", equivalence classes are used during config selection
    batched_mode: str | None = None
    # Batched dimension specification from @aot_kernel(batched=...)
    # Format: [[0, None], None] = arg0 has 2 dims, dim0 is batched; arg1 is non-tensor
    # Used to group shapes into families by non-batch features
    batched_spec: list[list[int | None] | None] | None = None


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
    backend: str = "decision_tree"  # Heuristic backend (only decision_tree supported)
    feature_selection: bool = True  # Whether to prune redundant features
    print_score_matrix: bool = True  # Whether to print the score matrix
    verbose: bool = True  # Verbose output
    skip_write: bool = False  # Skip writing files (for dump-code mode)
    # When True, use output hashes to compute equivalence classes and restrict
    # config selection to configs that produce identical outputs
    use_equivalence_classes: bool = False


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
    # Track output hashes per kernel: {kernel_name: {(shape_hash, config_hash): output_hash}}
    kernel_output_hashes: dict[str, dict[tuple[str, str], str]] = {}
    # Track batched_mode per kernel
    kernel_batched_modes: dict[str, str] = {}

    with open(measurements_file, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            kname = row["kernel_name"]
            if kernel_name is not None and kname != kernel_name:
                continue

            if kname not in kernel_data:
                kernel_data[kname] = {}
                kernel_output_hashes[kname] = {}

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

            # Load output hash if present (backward compatible)
            output_hash = row.get("output_hash", "")
            if output_hash:
                kernel_output_hashes[kname][(shape_hash, config_hash)] = output_hash

            # Load batched_mode if present (backward compatible)
            batched_mode = row.get("batched_mode", "")
            if batched_mode and kname not in kernel_batched_modes:
                kernel_batched_modes[kname] = batched_mode

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

        # Get output hashes for this kernel (may be empty for old data)
        output_hashes = kernel_output_hashes.get(kname)
        if output_hashes and not output_hashes:
            output_hashes = None  # Convert empty dict to None

        result[kname] = ShapeConfigData(
            kernel_name=kname,
            shape_features=shape_features,
            timings=timings,
            configs=config_list,
            shape_hashes=shape_hashes,
            config_hashes=config_hashes,
            output_hashes=output_hashes if output_hashes else None,
            batched_mode=kernel_batched_modes.get(kname),
        )

    return result


# ============================================================================
# Config Selection
# ============================================================================


def compute_config_equivalence_classes(
    data: ShapeConfigData,
) -> dict[str, list[int]]:
    """
    Group configs into equivalence classes based on output hashes.

    Configs are equivalent if they produce the same output (same output hash)
    for ALL shapes that have been measured. This allows selecting configs
    that are guaranteed to produce bitwise-identical results.

    Args:
        data: Shape and config data with output_hashes populated

    Returns:
        Dict mapping output signature -> list of config indices.
        The signature is a concatenation of output hashes across all shapes.
        Returns empty dict if output_hashes is not available.
    """
    if data.output_hashes is None:
        return {}

    # For each config, compute a global signature by joining its output
    # hashes across all shapes (sorted for determinism)
    signatures: dict[int, str] = {}

    for config_idx, config_hash in enumerate(data.config_hashes):
        parts: list[str] = []
        for shape_hash in sorted(data.shape_hashes):
            key = (shape_hash, config_hash)
            output_hash = data.output_hashes.get(key, "missing")
            parts.append(output_hash)
        signatures[config_idx] = "|".join(parts)

    # Group by signature
    classes: dict[str, list[int]] = {}
    for config_idx, sig in signatures.items():
        classes.setdefault(sig, []).append(config_idx)

    return classes


def select_best_equivalence_class(
    data: ShapeConfigData,
    equivalence_classes: dict[str, list[int]],
) -> list[int]:
    """
    Select the equivalence class with the best overall performance.

    For each equivalence class, we compute the oracle performance
    (minimum timing achievable using any config in that class), then
    select the class with the best geomean of oracle timings.

    Args:
        data: Shape and config data
        equivalence_classes: Dict mapping signature -> config indices

    Returns:
        List of config indices in the best equivalence class
    """
    if not equivalence_classes:
        # No equivalence classes - return all configs
        return list(range(len(data.configs)))

    best_class_configs: list[int] = []
    best_geomean = float("inf")

    for sig, config_indices in equivalence_classes.items():
        # Compute oracle timing for each shape using only configs in this class
        oracle_timings = []
        for i in range(len(data.shape_hashes)):
            min_timing = float("inf")
            for config_idx in config_indices:
                timing = data.timings[i, config_idx]
                if timing < min_timing:
                    min_timing = timing
            oracle_timings.append(min_timing)

        # Compute geomean (skip inf timings)
        valid_timings = [t for t in oracle_timings if t < float("inf")]
        if not valid_timings:
            continue

        geomean = float(np.exp(np.mean(np.log(np.array(valid_timings) + 1e-10))))

        if geomean < best_geomean:
            best_geomean = geomean
            best_class_configs = config_indices

    return best_class_configs if best_class_configs else list(range(len(data.configs)))


def _extract_non_batch_key(
    features: dict[str, Any],
    batched_spec: list[list[int | None] | None] | None,
) -> str:
    """
    Extract a key string from non-batch features only.

    Args:
        features: Shape features dict (e.g., {"arg0_dim0": 32, "arg0_dim1": 1024, ...})
        batched_spec: Batched dimension spec (e.g., [[0, None], None])
            Format: list per arg, each is list of dims where int = batched, None = not batched

    Returns:
        String key built from non-batch features only.
    """
    if batched_spec is None:
        # No batch info - treat all features as non-batch
        return "|".join(f"{k}={v}" for k, v in sorted(features.items()))

    # Build set of batch dimension feature names
    batch_feature_names: set[str] = set()
    for arg_idx, arg_dims in enumerate(batched_spec):
        if arg_dims is None:
            continue  # Non-tensor arg
        for dim_idx, batch_flag in enumerate(arg_dims):
            if batch_flag is not None:  # This dim is batched
                batch_feature_names.add(f"arg{arg_idx}_dim{dim_idx}")

    # Build key from non-batch features
    non_batch_parts: list[str] = []
    for key, value in sorted(features.items()):
        if key not in batch_feature_names:
            non_batch_parts.append(f"{key}={value}")

    return "|".join(non_batch_parts)


def group_shapes_into_families(
    shape_features: list[dict[str, Any]],
    batched_spec: list[list[int | None] | None] | None,
) -> dict[str, list[int]]:
    """
    Group shape indices by non-batch features (shape families).

    Shapes in the same family have identical non-batch dimensions and differ
    only in their batch dimensions.

    Args:
        shape_features: List of feature dicts for each shape
        batched_spec: Batched dimension specification

    Returns:
        Dict mapping family_key -> list of shape indices in that family
    """
    families: dict[str, list[int]] = {}
    for i, features in enumerate(shape_features):
        family_key = _extract_non_batch_key(features, batched_spec)
        families.setdefault(family_key, []).append(i)
    return families


def compute_family_equivalence_classes(
    family_shape_indices: list[int],
    data: ShapeConfigData,
) -> dict[str, list[int]]:
    """
    Compute config equivalence classes within a single shape family.

    Configs are equivalent if they produce the same output for ALL shapes
    within this family. Different families can have different equivalence
    class structures.

    Args:
        family_shape_indices: Indices of shapes belonging to this family
        data: Shape and config data with output_hashes populated

    Returns:
        Dict mapping output signature -> list of config indices.
        Returns empty dict if output_hashes is not available.
    """
    if data.output_hashes is None:
        return {}

    # For each config, compute signature by joining output hashes
    # across shapes in this family only
    signatures: dict[int, str] = {}

    for config_idx, config_hash in enumerate(data.config_hashes):
        parts: list[str] = []
        for shape_idx in family_shape_indices:
            shape_hash = data.shape_hashes[shape_idx]
            key = (shape_hash, config_hash)
            output_hash = data.output_hashes.get(key, "missing")
            parts.append(output_hash)
        signatures[config_idx] = "|".join(parts)

    # Group by signature
    classes: dict[str, list[int]] = {}
    for config_idx, sig in signatures.items():
        classes.setdefault(sig, []).append(config_idx)

    return classes


def _is_batch_consistent_signature(signature: str) -> bool:
    """Check if a signature represents batch-consistent outputs.

    A batch-consistent signature has all the same hash parts, meaning
    the config produces identical outputs regardless of batch size.

    Example:
        "hash_a|hash_a|hash_a" -> True (all same)
        "hash_a|hash_b|hash_c" -> False (all different)
    """
    parts = signature.split("|")
    if not parts:
        return True
    return all(p == parts[0] for p in parts)


def select_best_config_for_family(
    family_shape_indices: list[int],
    equiv_classes: dict[str, list[int]],
    timings: np.ndarray,
    config_hashes: list[str] | None = None,
    verbose: bool = False,
    family_name: str = "",
) -> tuple[int, dict[str, Any]]:
    """
    Select the best config for a family from its equivalence classes.

    For hash_equivalent mode, we PREFER "batch-consistent" classes where
    the config produces the same output regardless of batch size. Only if
    no batch-consistent class exists do we fall back to inconsistent ones.

    Args:
        family_shape_indices: Indices of shapes in this family
        equiv_classes: Dict mapping signature -> config indices
        timings: Full timing matrix (n_shapes, n_configs)
        config_hashes: Optional config hash strings for logging
        verbose: Whether to print detailed debug info
        family_name: Name of family for logging

    Returns:
        Tuple of (best config index, debug info dict)
    """
    debug_info: dict[str, Any] = {
        "family": family_name,
        "n_shapes": len(family_shape_indices),
        "n_equiv_classes": len(equiv_classes),
        "equiv_class_details": [],
        "unrestricted_best": None,
        "unrestricted_geomean": None,
        "selected_config": None,
        "selected_geomean": None,
        "slowdown_from_restriction": None,
        "selected_batch_consistent": None,
    }

    # Compute unrestricted best (no equivalence class restriction)
    family_timings = timings[family_shape_indices, :]
    all_geomeans = np.exp(np.mean(np.log(family_timings + 1e-10), axis=0))
    unrestricted_best_idx = int(np.argmin(all_geomeans))
    unrestricted_best_geomean = float(all_geomeans[unrestricted_best_idx])
    debug_info["unrestricted_best"] = unrestricted_best_idx
    debug_info["unrestricted_geomean"] = unrestricted_best_geomean

    if not equiv_classes:
        # No equivalence info - pick config with best geomean for this family
        debug_info["selected_config"] = unrestricted_best_idx
        debug_info["selected_geomean"] = unrestricted_best_geomean
        debug_info["slowdown_from_restriction"] = 1.0
        debug_info["selected_batch_consistent"] = True
        return unrestricted_best_idx, debug_info

    # Separate batch-consistent from inconsistent classes
    consistent_classes: dict[str, list[int]] = {}
    inconsistent_classes: dict[str, list[int]] = {}

    for sig, config_indices in equiv_classes.items():
        if _is_batch_consistent_signature(sig):
            consistent_classes[sig] = config_indices
        else:
            inconsistent_classes[sig] = config_indices

    # Prefer batch-consistent classes if any exist
    prefer_consistent = len(consistent_classes) > 0
    classes_to_consider = consistent_classes if prefer_consistent else equiv_classes

    best_config_idx = -1
    best_geomean = float("inf")
    best_class_sig = ""

    # Analyze each equivalence class
    for sig, config_indices in equiv_classes.items():
        is_consistent = _is_batch_consistent_signature(sig)

        # Compute oracle timing for each shape in family using only this class's configs
        oracle_timings = []
        for shape_idx in family_shape_indices:
            min_timing = float("inf")
            for config_idx in config_indices:
                timing = timings[shape_idx, config_idx]
                if timing < min_timing:
                    min_timing = timing
            oracle_timings.append(min_timing)

        # Compute geomean
        valid_timings = [t for t in oracle_timings if t < float("inf")]
        if not valid_timings:
            continue

        class_geomean = float(np.exp(np.mean(np.log(np.array(valid_timings) + 1e-10))))

        # Find best single config in this class
        class_timings = timings[np.ix_(family_shape_indices, config_indices)]
        single_geomeans = np.exp(np.mean(np.log(class_timings + 1e-10), axis=0))
        best_in_class_local_idx = int(np.argmin(single_geomeans))
        best_in_class_idx = config_indices[best_in_class_local_idx]
        best_in_class_geomean = float(single_geomeans[best_in_class_local_idx])

        # Store class details for logging
        class_info = {
            "signature": sig[:16] + "..." if len(sig) > 16 else sig,
            "full_signature": sig,
            "n_configs": len(config_indices),
            "config_indices": config_indices,
            "oracle_geomean": class_geomean,
            "best_single_config": best_in_class_idx,
            "best_single_geomean": best_in_class_geomean,
            "batch_consistent": is_consistent,
        }
        debug_info["equiv_class_details"].append(class_info)

        # Only consider this class if it's in our preferred set
        if sig in classes_to_consider and class_geomean < best_geomean:
            best_geomean = class_geomean
            best_config_idx = best_in_class_idx
            best_class_sig = sig

    if best_config_idx == -1:
        # Fallback: best overall config for family
        best_config_idx = unrestricted_best_idx
        best_geomean = unrestricted_best_geomean
        debug_info["selected_batch_consistent"] = None
    else:
        debug_info["selected_batch_consistent"] = _is_batch_consistent_signature(
            best_class_sig
        )

    debug_info["selected_config"] = best_config_idx
    debug_info["selected_geomean"] = best_geomean
    debug_info["slowdown_from_restriction"] = best_geomean / unrestricted_best_geomean

    # Print detailed logging if verbose
    if verbose:
        print(f"\n  --- Family: {family_name} ---", file=sys.stderr)
        print(
            f"  Shapes: {len(family_shape_indices)}, "
            f"Equiv classes: {len(equiv_classes)} "
            f"({len(consistent_classes)} batch-consistent, "
            f"{len(inconsistent_classes)} inconsistent)",
            file=sys.stderr,
        )

        # Show unrestricted best
        cfg_name = (
            config_hashes[unrestricted_best_idx][:8]
            if config_hashes
            else str(unrestricted_best_idx)
        )
        print(
            f"  Unrestricted best: config {cfg_name} "
            f"(geomean={unrestricted_best_geomean:.4f}ms)",
            file=sys.stderr,
        )

        # Check if unrestricted best is in the selected class
        unrestricted_in_selected = False
        unrestricted_is_consistent = False
        for class_info in debug_info["equiv_class_details"]:
            if unrestricted_best_idx in class_info["config_indices"]:
                unrestricted_is_consistent = class_info["batch_consistent"]
                if best_config_idx in class_info["config_indices"]:
                    unrestricted_in_selected = True
                break

        if not unrestricted_is_consistent:
            print(
                f"    NOTE: Unrestricted best is NOT batch-consistent!",
                file=sys.stderr,
            )

        # Show each equivalence class
        print(f"\n  Equivalence classes:", file=sys.stderr)
        for i, class_info in enumerate(debug_info["equiv_class_details"]):
            is_selected = best_config_idx in class_info["config_indices"]
            has_unrestricted = unrestricted_best_idx in class_info["config_indices"]
            is_consistent = class_info["batch_consistent"]

            marker = " [SELECTED]" if is_selected else ""
            marker += " [HAS UNRESTRICTED BEST]" if has_unrestricted else ""
            consistency = "batch-consistent" if is_consistent else "INCONSISTENT"

            cfg_names = [
                config_hashes[idx][:8] if config_hashes else str(idx)
                for idx in class_info["config_indices"][:5]  # Show first 5
            ]
            if len(class_info["config_indices"]) > 5:
                cfg_names.append(f"...+{len(class_info['config_indices']) - 5} more")

            print(
                f"    Class {i + 1}: {class_info['n_configs']} configs, "
                f"oracle={class_info['oracle_geomean']:.4f}ms, "
                f"{consistency}{marker}",
                file=sys.stderr,
            )
            print(f"      Configs: {cfg_names}", file=sys.stderr)

        # Show selected config and slowdown
        selected_cfg_name = (
            config_hashes[best_config_idx][:8]
            if config_hashes
            else str(best_config_idx)
        )
        consistency_note = (
            " (batch-consistent)"
            if debug_info["selected_batch_consistent"]
            else " (WARNING: not batch-consistent!)"
            if debug_info["selected_batch_consistent"] is False
            else ""
        )
        print(
            f"\n  Selected: config {selected_cfg_name} "
            f"(geomean={best_geomean:.4f}ms){consistency_note}",
            file=sys.stderr,
        )

        slowdown = debug_info["slowdown_from_restriction"]
        if slowdown > 1.001:
            print(
                f"  Slowdown from batch-consistency requirement: "
                f"{(slowdown - 1) * 100:.1f}%",
                file=sys.stderr,
            )
            if not unrestricted_in_selected:
                print(
                    f"    -> Unrestricted best config is in a DIFFERENT equiv class",
                    file=sys.stderr,
                )

    return best_config_idx, debug_info


def select_configs_per_family(
    families: dict[str, list[int]],
    data: ShapeConfigData,
    verbose: bool = False,
) -> tuple[dict[str, int], list[int], list[dict[str, Any]]]:
    """
    For each shape family, select the best config from its best equivalence class.

    This allows different families (different non-batch shapes) to use configs
    from different equivalence classes while maintaining correctness within
    each family.

    Args:
        families: Dict mapping family_key -> list of shape indices
        data: Shape and config data
        verbose: Whether to print debug info

    Returns:
        Tuple of:
        - Dict mapping family_key -> best config index for that family
        - List of unique config indices selected across all families
        - List of debug info dicts for each family
    """
    family_configs: dict[str, int] = {}
    all_debug_info: list[dict[str, Any]] = []

    for family_key, shape_indices in families.items():
        # Compute equivalence classes for this family
        equiv_classes = compute_family_equivalence_classes(shape_indices, data)

        # Select best config for this family with detailed logging
        best_config, debug_info = select_best_config_for_family(
            shape_indices,
            equiv_classes,
            data.timings,
            config_hashes=data.config_hashes,
            verbose=verbose,
            family_name=family_key,
        )
        family_configs[family_key] = best_config
        all_debug_info.append(debug_info)

    # Collect unique configs
    unique_configs = sorted(set(family_configs.values()))

    # Print summary if verbose
    if verbose:
        total_slowdown = 1.0
        for debug_info in all_debug_info:
            if debug_info["slowdown_from_restriction"]:
                total_slowdown *= debug_info["slowdown_from_restriction"]

        avg_slowdown = total_slowdown ** (1.0 / len(all_debug_info))
        if avg_slowdown > 1.001:
            print(
                f"\n  SUMMARY: Avg slowdown from equiv restriction: "
                f"{(avg_slowdown - 1) * 100:.1f}%",
                file=sys.stderr,
            )

    return family_configs, unique_configs, all_debug_info


def select_config_subset(
    data: ShapeConfigData,
    target: PerformanceTarget,
    allowed_configs: list[int] | None = None,
) -> tuple[list[int], dict[str, float]]:
    """
    Select a minimal subset of configs that satisfies the performance goal.

    Uses a greedy algorithm:
    1. Start with the config that is optimal for the most shapes
    2. Add configs until performance goal is met for all shapes

    Args:
        data: Shape and config data
        target: Performance target configuration
        allowed_configs: Optional list of config indices to consider.
            If provided, only these configs will be selected (for equivalence class restriction).
            If None, all configs are considered.

    Returns:
        Tuple of (selected config indices, performance stats)
    """
    n_shapes, n_configs = data.timings.shape

    # Determine which configs to consider
    if allowed_configs is not None:
        candidate_configs = set(allowed_configs)
    else:
        candidate_configs = set(range(n_configs))

    # Find the best timing for each shape (oracle performance)
    # Only consider allowed configs for oracle computation
    if allowed_configs is not None:
        allowed_timings = data.timings[:, allowed_configs]
        best_per_shape = np.min(allowed_timings, axis=1)
    else:
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

        for config_idx in candidate_configs:
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
    batched_specs: dict[str, list[list[int | None] | None] | None] | None = None,
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
        batched_specs: Optional dict mapping kernel names to batched dimension specs.
            Used for per-family equivalence class computation in hash_equivalent mode.

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

        # Attach batched_spec to data if provided
        if batched_specs is not None and kname in batched_specs:
            data.batched_spec = batched_specs[kname]

        # Print score matrix if requested
        if target.print_score_matrix:
            print_score_matrix(data)

        # Determine if we should use equivalence classes
        use_equiv = (
            data.batched_mode == "hash_equivalent"
            or target.use_equivalence_classes  # CLI flag as fallback/override
        )

        # Use per-family equivalence for hash_equivalent mode with batched_spec
        if use_equiv and data.output_hashes and data.batched_spec is not None:
            # Group shapes into families by non-batch features
            families = group_shapes_into_families(
                data.shape_features, data.batched_spec
            )

            if target.verbose:
                print(
                    f"\n=== Per-Family Equivalence (batched_mode={data.batched_mode}) ===\n"
                    f"Found {len(families)} shape family(ies)",
                    file=sys.stderr,
                )

            # Select best config for each family from its equivalence classes
            family_configs, unique_configs, _debug_info = select_configs_per_family(
                families, data, verbose=target.verbose
            )

            selected_indices = unique_configs
            selected_configs = [data.configs[i] for i in selected_indices]

            # Compute stats for the per-family selection
            # Each shape uses its family's best config
            n_shapes = len(data.shape_hashes)
            best_per_shape = np.min(data.timings, axis=1)
            selected_timings = np.full(n_shapes, np.inf)
            for family_key, shape_indices in families.items():
                config_idx = family_configs[family_key]
                for shape_idx in shape_indices:
                    selected_timings[shape_idx] = data.timings[shape_idx, config_idx]

            slowdowns = selected_timings / best_per_shape
            stats = {
                "max_slowdown": float(np.max(slowdowns)),
                "geomean_slowdown": float(
                    np.exp(np.mean(np.log(slowdowns + 1e-10)))
                ),
                "avg_slowdown": float(np.mean(slowdowns)),
                "satisfied_ratio": 1.0,  # Per-family always satisfies
                "num_configs": len(selected_indices),
            }

            log.info(
                f"  Per-family selection: {len(families)} families, "
                f"{len(selected_configs)} unique configs"
            )
            if target.verbose:
                print(
                    f"\nPer-family selection: {len(families)} families, "
                    f"{len(selected_configs)} unique configs\n"
                    f"max_slowdown={stats['max_slowdown']:.2f}x, "
                    f"geomean_slowdown={stats['geomean_slowdown']:.2f}x",
                    file=sys.stderr,
                )
        elif use_equiv and data.output_hashes:
            # Fall back to global equivalence if no batched_spec
            equivalence_classes = compute_config_equivalence_classes(data)
            allowed_configs: list[int] | None = None
            if equivalence_classes:
                allowed_configs = select_best_equivalence_class(
                    data, equivalence_classes
                )
                if target.verbose:
                    print(
                        f"\n=== Global Equivalence (no batched_spec) ===\n"
                        f"Found {len(equivalence_classes)} equivalence class(es)\n"
                        f"Selected class with {len(allowed_configs)} config(s)",
                        file=sys.stderr,
                    )
                log.info(
                    f"  Found {len(equivalence_classes)} equivalence classes, "
                    f"selected class with {len(allowed_configs)} configs"
                )

            selected_indices, stats = select_config_subset(
                data, target, allowed_configs=allowed_configs
            )
            selected_configs = [data.configs[i] for i in selected_indices]

            log.info(
                f"  Selected {len(selected_configs)} configs with "
                f"max_slowdown={stats['max_slowdown']:.2f}x, "
                f"geomean_slowdown={stats['geomean_slowdown']:.2f}x"
            )
            if target.verbose:
                print(
                    f"\nSelected {len(selected_configs)} configs (from equivalence class): "
                    f"max_slowdown={stats['max_slowdown']:.2f}x, "
                    f"geomean_slowdown={stats['geomean_slowdown']:.2f}x",
                    file=sys.stderr,
                )
        else:
            # No equivalence - standard config subset selection
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
