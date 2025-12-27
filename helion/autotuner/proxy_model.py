"""
Proxy models for autotuner testing.

Provides proxy models trained from autotuner CSV logs to simulate
benchmarking without running kernels. Uses KNN (k=5) combined with
positive-weight linear regression.
"""

from __future__ import annotations

import ast
import csv
from dataclasses import dataclass
from dataclasses import field
import math
import operator
from pathlib import Path
import re
from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterable
    from collections.abc import Sequence

    import numpy as np
    from numpy.typing import NDArray

try:
    import numpy as np
    from sklearn.linear_model import LinearRegression

    HAS_ML_DEPS = True
except ImportError:
    HAS_ML_DEPS = False
    np = None  # type: ignore[assignment]
    LinearRegression = None  # type: ignore[assignment, misc]


@dataclass
class AutotuneLogRecord:
    """A single record from an autotuner CSV log."""

    timestamp_s: float | None
    config_index: int
    generation: int
    status: str
    perf_ms: float | None
    compile_time_s: float | None
    config_str: str
    config_dict: dict[str, Any] = field(default_factory=dict)


def parse_config_str(config_str: str) -> dict[str, Any]:
    """Parse 'Config(block_sizes=[64, 64], num_warps=4)' into a dict."""
    # Remove 'Config(' prefix and ')' suffix
    match = re.match(r"Config\((.*)\)$", config_str.strip(), re.DOTALL)
    if not match:
        return {}

    inner = match.group(1)

    # Parse as Python dict-like syntax
    result: dict[str, Any] = {}

    # Use a simple state machine to parse key=value pairs
    # Handle nested structures like lists
    current_key = ""
    current_value = ""
    depth = 0
    in_key = True

    for char in inner:
        if char == "=" and depth == 0 and in_key:
            in_key = False
            continue
        if char in "[({":
            depth += 1
            if not in_key:
                current_value += char
        elif char in "])}":
            depth -= 1
            if not in_key:
                current_value += char
        elif char == "," and depth == 0:
            # End of a key-value pair
            if current_key and current_value:
                try:
                    result[current_key.strip()] = ast.literal_eval(
                        current_value.strip()
                    )
                except (ValueError, SyntaxError):
                    result[current_key.strip()] = current_value.strip()
            current_key = ""
            current_value = ""
            in_key = True
        elif in_key:
            current_key += char
        else:
            current_value += char

    # Handle last pair
    if current_key and current_value:
        try:
            result[current_key.strip()] = ast.literal_eval(current_value.strip())
        except (ValueError, SyntaxError):
            result[current_key.strip()] = current_value.strip()

    return result


def load_autotune_log(csv_path: str | Path) -> list[AutotuneLogRecord]:
    """Load records from an autotuner CSV log file."""
    records: list[AutotuneLogRecord] = []
    csv_path = Path(csv_path)

    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Skip 'started' entries - we want completed ones
            if row["status"] == "started":
                continue

            timestamp = float(row["timestamp_s"]) if row["timestamp_s"] else None
            config_index = int(row["config_index"]) if row["config_index"] else 0
            generation = int(row["generation"]) if row["generation"] else 0
            perf_ms = float(row["perf_ms"]) if row["perf_ms"] else None
            compile_time = (
                float(row["compile_time_s"]) if row["compile_time_s"] else None
            )
            config_str = row["config"]
            config_dict = parse_config_str(config_str)

            records.append(
                AutotuneLogRecord(
                    timestamp_s=timestamp,
                    config_index=config_index,
                    generation=generation,
                    status=row["status"],
                    perf_ms=perf_ms,
                    compile_time_s=compile_time,
                    config_str=config_str,
                    config_dict=config_dict,
                )
            )

    return records


def _extract_features_from_config_dict(config_dict: dict[str, Any]) -> list[float]:
    """Extract numerical features from a config dict for ML model input."""
    features: list[float] = []

    # Block sizes - log2 transform (up to 4 dimensions for M, N, K, etc.)
    block_sizes = config_dict.get("block_sizes", [])
    if isinstance(block_sizes, list):
        for i in range(4):  # Up to 4 block sizes
            if i < len(block_sizes):
                bs = block_sizes[i]
                if isinstance(bs, int) and bs > 0:
                    features.append(math.log2(bs))
                else:
                    features.append(0.0)
            else:
                features.append(0.0)
    else:
        features.extend([0.0] * 4)

    # num_warps - log2 transform
    num_warps = config_dict.get("num_warps", 4)
    if isinstance(num_warps, int) and num_warps > 0:
        features.append(math.log2(num_warps))
    else:
        features.append(2.0)  # log2(4)

    # num_stages
    num_stages = config_dict.get("num_stages", 1)
    if isinstance(num_stages, int):
        features.append(float(num_stages))
    else:
        features.append(1.0)

    # loop_orders - flatten first one (up to 3 elements)
    loop_orders = config_dict.get("loop_orders", [[0]])
    if isinstance(loop_orders, list) and loop_orders:
        first_order = (
            loop_orders[0] if isinstance(loop_orders[0], list) else loop_orders
        )
        for i in range(3):
            if i < len(first_order):
                features.append(float(first_order[i]))
            else:
                features.append(0.0)
    else:
        features.extend([0.0] * 3)

    # l2_groupings - log2 of first element if present
    l2_groupings = config_dict.get("l2_groupings", config_dict.get("l2_grouping", [1]))
    if isinstance(l2_groupings, list) and l2_groupings:
        val = l2_groupings[0]
        if isinstance(val, int) and val > 0:
            features.append(math.log2(val))
        else:
            features.append(0.0)
    elif isinstance(l2_groupings, int) and l2_groupings > 0:
        features.append(math.log2(l2_groupings))
    else:
        features.append(0.0)

    # indexing - can be string or list of strings
    # Encode as counts of each type
    indexing = config_dict.get("indexing", "pointer")
    indexing_types = ["pointer", "block_ptr", "tensor_descriptor"]

    if isinstance(indexing, list):
        # Count occurrences of each type
        for idx_type in indexing_types:
            count = sum(1 for x in indexing if x == idx_type)
            features.append(float(count))
    elif isinstance(indexing, str):
        for idx_type in indexing_types:
            features.append(1.0 if indexing == idx_type else 0.0)
    else:
        features.extend([0.0] * 3)

    # pid_type - one-hot encoding
    pid_type = config_dict.get("pid_type", "flat")
    pid_types = ["flat", "persistent_blocked", "persistent_interleaved"]
    for pt in pid_types:
        features.append(1.0 if pid_type == pt else 0.0)

    # range_flattens - count of True values
    range_flattens = config_dict.get("range_flattens", [])
    if isinstance(range_flattens, list):
        count = sum(1 for x in range_flattens if x is True)
        features.append(float(count))
    else:
        features.append(0.0)

    # range_multi_buffers - count of True values
    range_multi_buffers = config_dict.get("range_multi_buffers", [])
    if isinstance(range_multi_buffers, list):
        count = sum(1 for x in range_multi_buffers if x is True)
        features.append(float(count))
    else:
        features.append(0.0)

    # range_warp_specializes - count of True values or length
    range_warp_specializes = config_dict.get("range_warp_specializes", [])
    if isinstance(range_warp_specializes, list):
        if range_warp_specializes and isinstance(range_warp_specializes[0], bool):
            count = sum(1 for x in range_warp_specializes if x is True)
        else:
            count = len(range_warp_specializes)
        features.append(float(count))
    else:
        features.append(0.0)

    # Total block size (product) - useful feature for matmul
    if isinstance(block_sizes, list) and block_sizes:
        total = 1
        for bs in block_sizes:
            if isinstance(bs, int) and bs > 0:
                total *= bs
        features.append(math.log2(max(1, total)))
    else:
        features.append(0.0)

    # flatten_loops - count of True values
    flatten_loops = config_dict.get("flatten_loops", [])
    if isinstance(flatten_loops, list):
        count = sum(1 for x in flatten_loops if x is True)
        features.append(float(count))
    else:
        features.append(0.0)

    # range_unroll_factors - average unroll factor
    range_unroll_factors = config_dict.get("range_unroll_factors", [])
    if isinstance(range_unroll_factors, list) and range_unroll_factors:
        valid_factors = [
            f for f in range_unroll_factors if isinstance(f, int) and f > 0
        ]
        features.append(sum(valid_factors) / max(1, len(valid_factors)))
    else:
        features.append(1.0)

    # range_num_stages - average num stages for ranges
    range_num_stages = config_dict.get("range_num_stages", [])
    if isinstance(range_num_stages, list) and range_num_stages:
        valid_stages = [s for s in range_num_stages if isinstance(s, int) and s > 0]
        features.append(sum(valid_stages) / max(1, len(valid_stages)))
    else:
        features.append(1.0)

    # static_ranges - count of True values
    static_ranges = config_dict.get("static_ranges", [])
    if isinstance(static_ranges, list):
        count = sum(1 for x in static_ranges if x is True)
        features.append(float(count))
    else:
        features.append(0.0)

    return features


class _KNNLinearModel:
    """Ensemble model combining KNN (k=5) with positive-weight linear regression."""

    def __init__(self, k: int = 5) -> None:
        self.k = k
        self._X: Any = None
        self._y: Any = None
        self._linear_weights: Any = None
        self._feature_mean: Any = None
        self._feature_std: Any = None

    def fit(self, X: NDArray[Any], y: NDArray[Any]) -> None:
        assert np is not None
        self._X = np.array(X)
        self._y = np.array(y)

        # Normalize features for distance computation
        self._feature_mean = self._X.mean(axis=0)
        self._feature_std = self._X.std(axis=0)
        self._feature_std[self._feature_std == 0] = 1.0  # Avoid division by zero

        # Fit positive-weight linear regression using sklearn
        X_normalized = (self._X - self._feature_mean) / self._feature_std
        # Add bias column
        X_with_bias = np.column_stack([np.ones(len(X_normalized)), X_normalized])
        # Use LinearRegression with positive=True for non-negative weights
        assert LinearRegression is not None
        reg = LinearRegression(positive=True, fit_intercept=False)
        reg.fit(X_with_bias, self._y)
        self._linear_weights = reg.coef_

    def predict(self, X: NDArray[Any] | list[list[float]]) -> NDArray[Any]:
        assert np is not None
        X = np.atleast_2d(X)
        predictions = []

        for x in X:
            # KNN prediction
            x_normalized = (x - self._feature_mean) / self._feature_std
            distances = np.linalg.norm(
                (self._X - self._feature_mean) / self._feature_std - x_normalized,
                axis=1,
            )
            k = min(self.k, len(self._y))
            nearest_indices = np.argsort(distances)[:k]
            knn_pred = self._y[nearest_indices].mean()

            # Linear prediction
            x_with_bias = np.concatenate([[1.0], x_normalized])
            linear_pred = np.dot(x_with_bias, self._linear_weights)

            # Combine predictions (equal weight)
            pred = 0.5 * knn_pred + 0.5 * linear_pred
            predictions.append(max(pred, 0.001))  # Ensure positive prediction

        return np.array(predictions)


class PerformanceProxyModel:
    """Predicts kernel performance from config parameters."""

    def __init__(self) -> None:
        if not HAS_ML_DEPS:
            raise ImportError(
                "PerformanceProxyModel requires numpy and scikit-learn. "
                "Install them with: pip install numpy scikit-learn"
            )

        self._model = _KNNLinearModel(k=5)
        self._is_fitted = False
        self._config_strs: list[str] = []
        self._perfs: list[float] = []

    @classmethod
    def from_csv(cls, csv_path: str | Path) -> PerformanceProxyModel:
        """Create and train a proxy model from an autotuner CSV log."""
        proxy = cls()
        records = load_autotune_log(csv_path)
        proxy.fit_from_records(records)
        return proxy

    @classmethod
    def from_records(cls, records: list[AutotuneLogRecord]) -> PerformanceProxyModel:
        """Create and train a proxy model from autotuner log records."""
        proxy = cls()
        proxy.fit_from_records(records)
        return proxy

    def fit_from_records(self, records: list[AutotuneLogRecord]) -> None:
        """Train the model from autotuner log records."""
        X: list[list[float]] = []
        y: list[float] = []

        for record in records:
            if record.perf_ms is None or not math.isfinite(record.perf_ms):
                continue
            if record.status != "ok":
                continue

            features = _extract_features_from_config_dict(record.config_dict)
            X.append(features)
            y.append(record.perf_ms)
            self._config_strs.append(record.config_str)
            self._perfs.append(record.perf_ms)

        if len(X) < 2:
            raise ValueError(f"Need at least 2 valid records to train, got {len(X)}")

        assert np is not None
        self._model.fit(np.array(X), np.array(y))
        self._is_fitted = True

    def predict(self, config_dict: dict[str, Any]) -> float:
        """Predict performance for a config dictionary."""
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        features = _extract_features_from_config_dict(config_dict)
        return float(self._model.predict([features])[0])

    def predict_from_config_str(self, config_str: str) -> float:
        """Predict performance from a config string."""
        config_dict = parse_config_str(config_str)
        return self.predict(config_dict)

    def get_best_observed(self) -> tuple[str, float]:
        """Get the best (lowest perf) config observed during training."""
        if not self._perfs:
            raise RuntimeError("No training data available")
        best_idx = min(range(len(self._perfs)), key=lambda i: self._perfs[i])
        return self._config_strs[best_idx], self._perfs[best_idx]

    def get_all_observed(self) -> list[tuple[str, float]]:
        """Get all (config_str, perf_ms) pairs observed during training, sorted by perf."""
        pairs = list(zip(self._config_strs, self._perfs, strict=True))
        return sorted(pairs, key=operator.itemgetter(1))


class CompileTimeProxyModel:
    """Predicts compilation time from config parameters."""

    def __init__(self) -> None:
        if not HAS_ML_DEPS:
            raise ImportError(
                "CompileTimeProxyModel requires numpy and scikit-learn. "
                "Install them with: pip install numpy scikit-learn"
            )

        self._model = _KNNLinearModel(k=5)
        self._is_fitted = False

    @classmethod
    def from_csv(cls, csv_path: str | Path) -> CompileTimeProxyModel:
        """Create and train a compile time proxy model from an autotuner CSV log."""
        proxy = cls()
        records = load_autotune_log(csv_path)
        proxy.fit_from_records(records)
        return proxy

    def fit_from_records(self, records: list[AutotuneLogRecord]) -> None:
        """Train the model from autotuner log records."""
        X: list[list[float]] = []
        y: list[float] = []

        for record in records:
            if record.compile_time_s is None or not math.isfinite(
                record.compile_time_s
            ):
                continue

            features = _extract_features_from_config_dict(record.config_dict)
            X.append(features)
            y.append(record.compile_time_s)

        if len(X) < 2:
            raise ValueError(
                f"Need at least 2 valid records with compile time to train, got {len(X)}"
            )

        assert np is not None
        self._model.fit(np.array(X), np.array(y))
        self._is_fitted = True

    def predict(self, config_dict: dict[str, Any]) -> float:
        """Predict compile time for a config dictionary."""
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        features = _extract_features_from_config_dict(config_dict)
        return float(self._model.predict([features])[0])


@dataclass
class SearchEvaluationResult:
    """Result of evaluating a search algorithm with a proxy model."""

    search_name: str
    n_configs_evaluated: int
    best_perf_found: float
    true_best_perf: float
    percent_of_best: float  # (true_best / best_found) * 100
    total_estimated_compile_time: float | None
    config_evaluations: list[tuple[int, float]]  # (n_configs, best_perf_at_that_point)


@dataclass
class SimulatedBenchmarkResult:
    """Result of a single simulated benchmark call."""

    config_dict: dict[str, Any]
    predicted_perf: float
    predicted_compile_time: float | None
    config_index: int


class SimulatedBenchmark:
    """Simulated benchmark environment using proxy models."""

    def __init__(
        self,
        perf_model: PerformanceProxyModel,
        compile_time_model: CompileTimeProxyModel | None = None,
    ) -> None:
        self.perf_model = perf_model
        self.compile_time_model = compile_time_model
        self._config_counter = 0
        self._evaluation_history: list[SimulatedBenchmarkResult] = []

    @classmethod
    def from_csv(
        cls,
        csv_path: str | Path,
        include_compile_time: bool = True,
    ) -> SimulatedBenchmark:
        """Create a simulated benchmark from an autotuner CSV log."""
        perf_model = PerformanceProxyModel.from_csv(csv_path)

        compile_time_model = None
        if include_compile_time:
            import contextlib

            with contextlib.suppress(ValueError):
                compile_time_model = CompileTimeProxyModel.from_csv(csv_path)

        return cls(perf_model, compile_time_model)

    def reset(self) -> None:
        self._config_counter = 0
        self._evaluation_history = []

    def benchmark(self, config_dict: dict[str, Any]) -> SimulatedBenchmarkResult:
        """Simulate benchmarking a config."""
        self._config_counter += 1

        perf = self.perf_model.predict(config_dict)
        compile_time = (
            self.compile_time_model.predict(config_dict)
            if self.compile_time_model is not None
            else None
        )

        result = SimulatedBenchmarkResult(
            config_dict=config_dict,
            predicted_perf=perf,
            predicted_compile_time=compile_time,
            config_index=self._config_counter,
        )
        self._evaluation_history.append(result)
        return result

    def get_evaluation_history(self) -> list[SimulatedBenchmarkResult]:
        return list(self._evaluation_history)

    def get_best_at_n_configs(self, n: int | None = None) -> tuple[float, int]:
        """Get the best performance found after evaluating n configs."""
        if n is None:
            n = len(self._evaluation_history)

        if n == 0:
            return float("inf"), 0

        subset = self._evaluation_history[:n]
        best = min(subset, key=lambda r: r.predicted_perf)
        return best.predicted_perf, best.config_index

    def compute_metrics_at_n(
        self, n_values: Sequence[int] | None = None
    ) -> list[tuple[int, float, float]]:
        """Compute search quality metrics at various numbers of configs evaluated."""
        if n_values is None:
            # Default progression
            n_values = [1, 5, 10, 20, 50, 100, 200, 500, 1000]
            n_values = [n for n in n_values if n <= len(self._evaluation_history)]
            if (
                self._evaluation_history
                and len(self._evaluation_history) not in n_values
            ):
                n_values.append(len(self._evaluation_history))

        true_best = self.perf_model.get_best_observed()[1]

        results: list[tuple[int, float, float]] = []
        for n in n_values:
            best_at_n, _ = self.get_best_at_n_configs(n)
            percent = (true_best / best_at_n) * 100 if best_at_n > 0 else 0
            results.append((n, best_at_n, percent))

        return results

    def print_search_quality_report(self) -> None:
        metrics = self.compute_metrics_at_n()
        true_best_config, true_best_perf = self.perf_model.get_best_observed()

        print("\n" + "=" * 70)
        print("Search Quality Report")
        print("=" * 70)
        print(f"True best performance: {true_best_perf:.4f} ms")
        print(f"Total configs evaluated: {len(self._evaluation_history)}")
        print()
        print(f"{'N Configs':>12} {'Best Perf (ms)':>15} {'% of Best':>12}")
        print("-" * 42)

        for n, perf, percent in metrics:
            print(f"{n:>12} {perf:>15.4f} {percent:>11.1f}%")

        if self.compile_time_model is not None:
            total_compile = sum(
                r.predicted_compile_time or 0 for r in self._evaluation_history
            )
            print()
            print(f"Estimated total compile time: {total_compile:.1f}s")

        print("=" * 70 + "\n")


def evaluate_search_on_proxy(
    search_fn: Callable[[int], Iterable[dict[str, Any] | object]],
    csv_path: str | Path,
    n_configs: int = 100,
) -> SearchEvaluationResult:
    """Evaluate a search function using a proxy model."""
    sim = SimulatedBenchmark.from_csv(csv_path)

    # Generate and evaluate configs
    configs = search_fn(n_configs)
    for config in configs:
        if isinstance(config, dict):
            sim.benchmark(config)
        else:
            # Assume it has a config attribute or is convertible
            sim.benchmark(getattr(config, "config", {}))

    # Compute results
    true_best = sim.perf_model.get_best_observed()[1]
    best_found, _ = sim.get_best_at_n_configs()

    total_compile = None
    if sim.compile_time_model is not None:
        total_compile = sum(
            r.predicted_compile_time or 0 for r in sim.get_evaluation_history()
        )

    # Build trajectory
    trajectory: list[tuple[int, float]] = []
    best_so_far = float("inf")
    for i, result in enumerate(sim.get_evaluation_history(), 1):
        if result.predicted_perf < best_so_far:
            best_so_far = result.predicted_perf
            trajectory.append((i, best_so_far))

    return SearchEvaluationResult(
        search_name=getattr(search_fn, "__name__", "unknown"),
        n_configs_evaluated=len(sim.get_evaluation_history()),
        best_perf_found=best_found,
        true_best_perf=true_best,
        percent_of_best=(true_best / best_found) * 100 if best_found > 0 else 0,
        total_estimated_compile_time=total_compile,
        config_evaluations=trajectory,
    )
