"""
Heuristic Generator for AOT Autotuning
======================================

This module generates human-readable heuristics using decision trees
to select optimal configurations based on shape features.

The workflow:
1. Load measurement data (kernel, shape, config, timing)
2. Determine the minimum set of configs needed to satisfy performance goals
3. Train a decision tree to predict which config to use
4. Generate human-readable Python code
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Any
from typing import Literal

import numpy as np

from ..runtime.config import Config
from .heuristic_backends import FeatureSelectionResult
from .heuristic_backends import ShapeConfigData
from .heuristic_backends import get_backend
from .heuristic_backends import print_score_matrix
from .heuristic_backends import select_shape_features

log: logging.Logger = logging.getLogger(__name__)

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


@dataclass
class HeuristicResult:
    """Result of heuristic generation."""

    selected_configs: list[Config]
    config_to_index: dict[str, int]
    performance_stats: dict[str, float]
    model_accuracy: float
    generated_code: str
    feature_selection_result: FeatureSelectionResult | None = None


@dataclass
class MeasurementData:
    """Loaded measurement data for a kernel."""

    kernel_name: str
    shape_features: list[dict[str, Any]]
    configs: list[Config]
    timings: np.ndarray  # shape: (n_shapes, n_configs)
    shape_hashes: list[str]
    config_hashes: list[str]


def load_measurements(
    measurements_file: Path, kernel_name: str | None = None
) -> dict[str, MeasurementData]:
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

    # Convert to MeasurementData format
    result: dict[str, MeasurementData] = {}
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

        result[kname] = MeasurementData(
            kernel_name=kname,
            shape_features=shape_features,
            configs=config_list,
            timings=timings,
            shape_hashes=shape_hashes,
            config_hashes=config_hashes,
        )

    return result


def select_config_subset(
    data: MeasurementData,
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
    import sys

    from .aot_cache import get_device_compute_id

    if target is None:
        target = PerformanceTarget()

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load measurements
    all_data = load_measurements(measurements_file, kernel_name)
    results: dict[str, HeuristicResult] = {}

    # Get device info for naming heuristic files
    device_kind, compute_kind = get_device_compute_id()

    for kname, data in all_data.items():
        log.info(f"Generating heuristic for kernel: {kname}")
        if target.verbose:
            print(
                f"\n=== Generating heuristic for kernel: {kname} ===", file=sys.stderr
            )

        # Print score matrix if requested
        if target.print_score_matrix:
            # Create ShapeConfigData for the print function
            shape_data = ShapeConfigData(
                shape_features=data.shape_features,
                timings=data.timings,
                configs=data.configs,
                shape_hashes=data.shape_hashes,
                config_hashes=data.config_hashes,
                selected_config_indices=list(range(len(data.configs))),
            )
            print_score_matrix(shape_data)

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

        # Create ShapeConfigData for the backend
        shape_data = ShapeConfigData(
            shape_features=data.shape_features,
            timings=data.timings,
            configs=data.configs,
            shape_hashes=data.shape_hashes,
            config_hashes=data.config_hashes,
            selected_config_indices=selected_indices,
        )

        # Generate heuristic using the selected backend
        backend = get_backend(target.backend)
        backend_result = backend.generate_heuristic(
            kernel_name=kname,
            data=shape_data,
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

            # Create heuristic filename: _<basename>_<device>_<compute>.py
            base_name = source_path.stem
            heuristic_name = f"_{base_name}_{device_kind}_{compute_kind}.py"
            source_heuristic_file = source_path.parent / heuristic_name

            # Combine heuristics for all kernels in this source file
            combined_code = _combine_heuristics(knames, results)
            source_heuristic_file.write_text(combined_code)
            log.info(
                f"  Saved combined heuristic for {len(knames)} kernel(s) to: {source_heuristic_file}"
            )

            # Copy any extra files to source directory
            for filename, content in backend_result.extra_files.items():
                extra_dst = source_path.parent / filename
                extra_dst.write_bytes(content)
                log.info(f"  Copied extra file to: {extra_dst}")

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
        '"""',
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

        # Rename globals and functions to be kernel-specific
        kernel_code = kernel_code.replace("CONFIGS", f"CONFIGS{kernel_upper}")
        kernel_code = kernel_code.replace(
            "FEATURE_NAMES", f"FEATURE_NAMES{kernel_upper}"
        )
        kernel_code = kernel_code.replace(
            "def _predict(", f"def _predict{kernel_suffix}("
        )
        kernel_code = kernel_code.replace(
            "_predict(features)", f"_predict{kernel_suffix}(features)"
        )

        lines.extend([f"# === Kernel: {kname} ===", kernel_code, ""])

    # Generate unified select_config function
    lines.extend(
        [
            "def select_config(kernel_name: str, features: dict) -> dict:",
            '    """Generic config selector."""',
        ]
    )
    for kname in kernel_names:
        lines.extend(
            [
                f'    if kernel_name == "{kname}":',
                f"        return select_config_{kname}(features)",
            ]
        )
    lines.extend(['    raise ValueError(f"Unknown kernel: {kernel_name}")', ""])

    return "\n".join(lines)


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
