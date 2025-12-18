"""
Heuristic Generator for AOT Autotuning
======================================

This module generates human-readable heuristics using LightGBM and lgbm-to-code
to select optimal configurations based on shape features.

The workflow:
1. Load measurement data (kernel, shape, config, timing)
2. Determine the minimum set of configs needed to satisfy performance goals
3. Train a LightGBM classifier to predict which config to use
4. Generate human-readable Python code using lgbm-to-code
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

log: logging.Logger = logging.getLogger(__name__)

PerformanceGoal = Literal["max_slowdown", "geomean_slowdown", "avg_slowdown"]


def _fix_lgbm_to_code_indentation(code: str) -> str:
    """Fix indentation issues in lgbm-to-code output.

    lgbm-to-code generates code with improper indentation like:
        def func_0(x):
        return -0.810930216

    This function fixes it to:
        def func_0(x):
            return -0.810930216
    """
    lines = code.split("\n")
    fixed_lines = []
    in_function = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("def "):
            in_function = True
            fixed_lines.append(line)
        elif in_function and stripped and not stripped.startswith("def "):
            # This line should be indented as function body
            if not line.startswith("\t") and not line.startswith("    "):
                fixed_lines.append("\t" + line)
            else:
                fixed_lines.append(line)
        else:
            fixed_lines.append(line)

    return "\n".join(fixed_lines)


@dataclass
class PerformanceTarget:
    """Configuration for performance goals."""

    goal_type: PerformanceGoal = "max_slowdown"
    threshold: float = 1.1  # 10% slowdown allowed
    min_configs: int = 1
    max_configs: int = 10


@dataclass
class HeuristicResult:
    """Result of heuristic generation."""

    selected_configs: list[Config]
    config_to_index: dict[str, int]
    performance_stats: dict[str, float]
    model_accuracy: float
    generated_code: str


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


def train_config_selector(
    data: MeasurementData,
    selected_indices: list[int],
) -> tuple[Any, float, list[str]]:
    """
    Train a LightGBM classifier to predict which config to use.

    Returns:
        Tuple of (model, accuracy, feature_names)
    """
    try:
        import lightgbm as lgb
    except ImportError as e:
        raise ImportError(
            "LightGBM is required for heuristic generation. "
            "Install with: pip install lightgbm"
        ) from e

    n_shapes = len(data.shape_features)

    # Build feature matrix
    feature_names: list[str] = []
    if data.shape_features:
        # Get all numeric feature names
        for key, value in data.shape_features[0].items():
            if isinstance(value, (int, float)):
                feature_names.append(key)

    X = np.zeros((n_shapes, len(feature_names)))
    for i, features in enumerate(data.shape_features):
        for j, fname in enumerate(feature_names):
            X[i, j] = features.get(fname, 0)

    # Build labels: for each shape, which selected config is best?
    y = np.zeros(n_shapes, dtype=int)
    for i in range(n_shapes):
        best_timing = np.inf
        best_config = 0
        for j, config_idx in enumerate(selected_indices):
            timing = data.timings[i, config_idx]
            if timing < best_timing:
                best_timing = timing
                best_config = j
        y[i] = best_config

    # Train model
    if len(selected_indices) == 1:
        # Only one config, no need for a model
        return None, 1.0, feature_names

    params = {
        "objective": "multiclass",
        "num_class": len(selected_indices),
        "metric": "multi_logloss",
        "verbosity": -1,
        "num_leaves": 31,
        "max_depth": 6,
        "learning_rate": 0.1,
        "n_estimators": 100,
        "min_child_samples": 5,
    }

    train_data = lgb.Dataset(X, label=y, feature_name=feature_names)
    model = lgb.train(params, train_data, num_boost_round=100)

    # Compute accuracy
    predictions = np.argmax(model.predict(X), axis=1)
    accuracy = float(np.mean(predictions == y))

    return model, accuracy, feature_names


def generate_heuristic_code(
    kernel_name: str,
    model: Any,
    selected_configs: list[Config],
    feature_names: list[str],
    use_lgbm_to_code: bool = True,
) -> str:
    """
    Generate human-readable Python code for config selection.

    If lgbm-to-code is available and use_lgbm_to_code is True, generates
    interpretable decision tree code. Otherwise, generates a simpler
    fallback implementation.
    """
    configs_code = "CONFIGS = [\n"
    for config in selected_configs:
        configs_code += f"    {dict(config)!r},\n"
    configs_code += "]\n\n"

    if model is None:
        # Single config case
        return f'''"""
Auto-generated heuristic for kernel: {kernel_name}
Only one config selected - no decision logic needed.
"""

{configs_code}

def select_config_{kernel_name}(features: dict) -> dict:
    """Select the optimal config for the given shape features."""
    return CONFIGS[0]

def select_config(kernel_name: str, features: dict) -> dict:
    """Generic config selector."""
    if kernel_name == "{kernel_name}":
        return select_config_{kernel_name}(features)
    raise ValueError(f"Unknown kernel: {{kernel_name}}")
'''

    decision_logic = ""
    if use_lgbm_to_code:
        try:
            from lgbm_to_code.lgbm_to_code import parse_lgbm_model

            # Get raw decision tree code and fix indentation
            raw_code = parse_lgbm_model(model, "python")
            raw_code = _fix_lgbm_to_code_indentation(raw_code)

            # Get model info for multiclass handling
            model_json = model.dump_model()
            num_class = model_json.get("num_class", 1)
            num_trees = len(model_json.get("tree_info", []))

            if num_class > 1:
                # Multiclass: need to group trees by class and take argmax
                trees_per_class = num_trees // num_class
                class_tree_groups = []
                for c in range(num_class):
                    trees = [i for i in range(num_trees) if i % num_class == c]
                    class_tree_groups.append(trees)

                # Generate multiclass wrapper
                class_score_strs = []
                for c, trees in enumerate(class_tree_groups):
                    score_expr = "+".join(f"func_{t}(x)" for t in trees)
                    class_score_strs.append(f"        {score_expr},  # class {c}")

                decision_logic = f"""{raw_code}

def _predict(features: dict) -> int:
    \"\"\"Predict config index using decision trees.\"\"\"
    x = [features.get(f, 0) for f in {feature_names!r}]
    # Compute score for each class
    scores = [
{chr(10).join(class_score_strs)}
    ]
    return scores.index(max(scores))
"""
            else:
                # Binary: threshold at 0
                decision_logic = f"""{raw_code}

def _predict(features: dict) -> int:
    \"\"\"Predict config index using decision trees.\"\"\"
    x = [features.get(f, 0) for f in {feature_names!r}]
    return 1 if lgbminfer(x) > 0 else 0
"""
        except ImportError:
            log.warning(
                "lgbm-to-code not available. Install with: pip install lgbm-to-code"
            )
            use_lgbm_to_code = False

    if not use_lgbm_to_code:
        # Generate fallback code using model prediction
        decision_logic = f'''def _predict(features: dict) -> int:
    """Predict config index using embedded model weights."""
    import numpy as np
    # Feature extraction
    x = np.array([features.get(f, 0) for f in {feature_names!r}]).reshape(1, -1)

    # Load model and predict
    import lightgbm as lgb
    import tempfile
    import os

    model_path = os.path.join(os.path.dirname(__file__), "model_{kernel_name}.txt")
    if os.path.exists(model_path):
        model = lgb.Booster(model_file=model_path)
        probs = model.predict(x)
        return int(np.argmax(probs))
    return 0  # Fallback to first config
'''

    return f'''"""
Auto-generated heuristic for kernel: {kernel_name}
Generated by Helion AOT autotuning framework.

This file contains human-readable decision logic for selecting
the optimal kernel configuration based on input shape features.
"""

import numpy as np

{configs_code}

FEATURE_NAMES = {feature_names!r}

{decision_logic}

def select_config_{kernel_name}(features: dict) -> dict:
    """
    Select the optimal config for the given shape features.

    Args:
        features: Dictionary of shape features (e.g., dimensions, dtypes)

    Returns:
        Configuration dictionary for the kernel
    """
    config_idx = _predict(features)
    return CONFIGS[config_idx]


def select_config(kernel_name: str, features: dict) -> dict:
    """
    Generic config selector for any kernel.

    Args:
        kernel_name: Name of the kernel
        features: Dictionary of shape features

    Returns:
        Configuration dictionary for the kernel
    """
    if kernel_name == "{kernel_name}":
        return select_config_{kernel_name}(features)
    raise ValueError(f"Unknown kernel: {{kernel_name}}")
'''


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
    import shutil

    from .aot_cache import get_device_compute_id

    if target is None:
        target = PerformanceTarget()

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load measurements
    all_data = load_measurements(measurements_file, kernel_name)
    results: dict[str, HeuristicResult] = {}

    # Get device info for naming heuristic files
    device_kind, compute_kind = get_device_compute_id()

    # Track models and feature names for combined heuristic generation
    models: dict[str, Any] = {}
    feature_names_map: dict[str, list[str]] = {}

    for kname, data in all_data.items():
        log.info(f"Generating heuristic for kernel: {kname}")

        # Select config subset
        selected_indices, stats = select_config_subset(data, target)
        selected_configs = [data.configs[i] for i in selected_indices]

        log.info(
            f"  Selected {len(selected_configs)} configs with "
            f"max_slowdown={stats['max_slowdown']:.2f}x, "
            f"geomean_slowdown={stats['geomean_slowdown']:.2f}x"
        )

        # Build config index mapping
        config_to_index = {
            data.config_hashes[i]: j for j, i in enumerate(selected_indices)
        }

        # Train selector model
        model, accuracy, feature_names = train_config_selector(data, selected_indices)
        log.info(f"  Model accuracy: {accuracy:.2%}")

        # Store model and feature names for combined heuristic
        if model is not None:
            models[kname] = model
        feature_names_map[kname] = feature_names

        # Generate code
        code = generate_heuristic_code(kname, model, selected_configs, feature_names)

        # Save model if available
        if model is not None:
            model_file = output_dir / f"model_{kname}.txt"
            model.save_model(str(model_file))
            log.info(f"  Saved model to {model_file}")

        # Save heuristic code to output_dir (run directory)
        heuristic_file = output_dir / f"heuristic_{kname}.py"
        heuristic_file.write_text(code)
        log.info(f"  Saved heuristic to {heuristic_file}")

        results[kname] = HeuristicResult(
            selected_configs=selected_configs,
            config_to_index=config_to_index,
            performance_stats=stats,
            model_accuracy=accuracy,
            generated_code=code,
        )

    # Group kernels by source file and save combined heuristics
    if kernel_source_files:
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

            # Generate combined code for all kernels in this source file
            combined_code = _generate_combined_heuristic_code(
                knames,
                results,
                models=models,
                feature_names_map=feature_names_map,
                output_dir=output_dir,
            )
            source_heuristic_file.write_text(combined_code)
            log.info(
                f"  Saved combined heuristic for {len(knames)} kernel(s) to: {source_heuristic_file}"
            )

            # Copy model files to source directory for multi-config kernels
            for kname in knames:
                if kname in models:
                    model_src = output_dir / f"model_{kname}.txt"
                    model_dst = source_path.parent / f"model_{kname}.txt"
                    if model_src.exists():
                        shutil.copy2(model_src, model_dst)
                        log.info(f"  Copied model file to: {model_dst}")

    return results


def _generate_combined_heuristic_code(
    kernel_names: list[str],
    results: dict[str, HeuristicResult],
    models: dict[str, Any] | None = None,
    feature_names_map: dict[str, list[str]] | None = None,
    output_dir: Path | None = None,
) -> str:
    """Generate combined heuristic code for multiple kernels.

    Args:
        kernel_names: List of kernel names to include
        results: Dict mapping kernel names to HeuristicResult
        models: Optional dict mapping kernel names to trained LightGBM models
        feature_names_map: Optional dict mapping kernel names to feature name lists
        output_dir: Directory where model files are saved (for model loading path)
    """
    lines = [
        '"""',
        f"Auto-generated heuristics for kernels: {', '.join(kernel_names)}",
        '"""',
        "",
    ]

    # Check if any kernel needs the model loading infrastructure
    needs_model_loading = False
    for kname in kernel_names:
        result = results[kname]
        if len(result.selected_configs) > 1:
            needs_model_loading = True
            break

    if needs_model_loading:
        lines.extend(
            [
                "import os",
                "import numpy as np",
                "",
            ]
        )

    # Collect all configs for each kernel
    for kname in kernel_names:
        result = results[kname]
        configs_str = ",\n    ".join(str(dict(c)) for c in result.selected_configs)
        lines.extend(
            [
                f"CONFIGS_{kname.upper()} = [",
                f"    {configs_str}",
                "]",
                "",
            ]
        )

    # Generate select functions for each kernel
    for kname in kernel_names:
        result = results[kname]
        if len(result.selected_configs) == 1:
            # Single config - no decision logic needed
            lines.extend(
                [
                    f"def select_config_{kname}(features: dict) -> dict:",
                    f'    """Select the optimal config for {kname}."""',
                    f"    return CONFIGS_{kname.upper()}[0]",
                    "",
                ]
            )
        else:
            # Multi-config: generate decision logic using embedded model prediction
            fnames = feature_names_map.get(kname, []) if feature_names_map else []
            model = models.get(kname) if models else None

            # Try to generate lgbm-to-code decision tree first
            decision_code = None
            if model is not None:
                try:
                    from lgbm_to_code.lgbm_to_code import parse_lgbm_model

                    # Get raw decision tree code and fix indentation
                    raw_code = parse_lgbm_model(model, "python")
                    raw_code = _fix_lgbm_to_code_indentation(raw_code)

                    # Get model info for multiclass handling
                    model_json = model.dump_model()
                    num_class = model_json.get("num_class", 1)
                    num_trees = len(model_json.get("tree_info", []))

                    if num_class > 1:
                        # Multiclass: need to group trees by class and take argmax
                        class_tree_groups = []
                        for c in range(num_class):
                            trees = [i for i in range(num_trees) if i % num_class == c]
                            class_tree_groups.append(trees)

                        # Generate multiclass wrapper with unique function names
                        # First rename the raw functions to be kernel-specific
                        raw_code = raw_code.replace(
                            "def lgbminfer(x):", f"def _lgbminfer_{kname}(x):"
                        )
                        for i in range(num_trees):
                            raw_code = raw_code.replace(
                                f"def func_{i}(x):", f"def _func_{kname}_{i}(x):"
                            )
                            raw_code = raw_code.replace(
                                f"func_{i}(x)", f"_func_{kname}_{i}(x)"
                            )

                        class_score_strs = []
                        for c, trees in enumerate(class_tree_groups):
                            score_expr = "+".join(
                                f"_func_{kname}_{t}(x)" for t in trees
                            )
                            class_score_strs.append(
                                f"        {score_expr},  # class {c}"
                            )

                        decision_code = f"""{raw_code}

def _predict_{kname}(features: dict) -> int:
    \"\"\"Predict config index for {kname} using decision trees.\"\"\"
    x = [features.get(f, 0) for f in {fnames!r}]
    # Compute score for each class
    scores = [
{chr(10).join(class_score_strs)}
    ]
    return scores.index(max(scores))
"""
                    else:
                        # Binary: rename functions and threshold at 0
                        raw_code = raw_code.replace(
                            "def lgbminfer(x):", f"def _lgbminfer_{kname}(x):"
                        )
                        for i in range(num_trees):
                            raw_code = raw_code.replace(
                                f"def func_{i}(x):", f"def _func_{kname}_{i}(x):"
                            )
                            raw_code = raw_code.replace(
                                f"func_{i}(x)", f"_func_{kname}_{i}(x)"
                            )

                        decision_code = f"""{raw_code}

def _predict_{kname}(features: dict) -> int:
    \"\"\"Predict config index for {kname} using decision trees.\"\"\"
    x = [features.get(f, 0) for f in {fnames!r}]
    return 1 if _lgbminfer_{kname}(x) > 0 else 0
"""
                except ImportError:
                    pass

            if decision_code is not None:
                # Use the human-readable decision tree
                lines.extend(
                    [
                        f"# Decision logic for {kname}",
                        "",
                    ]
                )
                # Add the generated code
                for line in decision_code.strip().split("\n"):
                    lines.append(line)
                lines.extend(
                    [
                        "",
                        f"def select_config_{kname}(features: dict) -> dict:",
                        f'    """Select the optimal config for {kname}."""',
                        f"    config_idx = _predict_{kname}(features)",
                        f"    return CONFIGS_{kname.upper()}[config_idx]",
                        "",
                    ]
                )
            else:
                # Fallback: load model from file at runtime
                lines.extend(
                    [
                        f"# Decision logic for {kname} (model-based)",
                        f"FEATURE_NAMES_{kname.upper()} = {fnames!r}",
                        "",
                        f"def _predict_{kname}(features: dict) -> int:",
                        f'    """Predict config index for {kname} using LightGBM model."""',
                        f"    x = np.array([features.get(f, 0) for f in FEATURE_NAMES_{kname.upper()}]).reshape(1, -1)",
                        "    try:",
                        "        import lightgbm as lgb",
                        f'        model_path = os.path.join(os.path.dirname(__file__), "model_{kname}.txt")',
                        "        if os.path.exists(model_path):",
                        "            model = lgb.Booster(model_file=model_path)",
                        "            probs = model.predict(x)",
                        "            return int(np.argmax(probs))",
                        "    except Exception:",
                        "        pass",
                        "    return 0  # Fallback to first config",
                        "",
                        f"def select_config_{kname}(features: dict) -> dict:",
                        f'    """Select the optimal config for {kname}."""',
                        f"    config_idx = _predict_{kname}(features)",
                        f"    return CONFIGS_{kname.upper()}[config_idx]",
                        "",
                    ]
                )

    # Generate generic selector
    lines.extend(
        [
            "def select_config(kernel_name: str, features: dict) -> dict:",
            '    """Generic config selector."""',
        ]
    )
    for kname in kernel_names:
        lines.append(f'    if kernel_name == "{kname}":')
        lines.append(f"        return select_config_{kname}(features)")
    lines.append('    raise ValueError(f"Unknown kernel: {kernel_name}")')
    lines.append("")

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
