"""
Proxy models for autotuner testing.

Provides proxy models trained from autotuner CSV logs to simulate
benchmarking without running kernels. Uses sklearn KNeighborsRegressor.
"""

from __future__ import annotations

import ast
import csv
from dataclasses import dataclass
from dataclasses import field
import math
import operator
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any

import numpy as np
from sklearn.neighbors import KNeighborsRegressor

from .. import Config
from .config_generation import ConfigGeneration

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .config_spec import ConfigSpec


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
    """Parse 'Config(block_sizes=[64, 64], num_warps=4)' into a dict.

    Uses ast.parse to handle the Config(...) format as a function call,
    extracting keyword arguments with their values parsed via literal_eval.
    """
    config_str = config_str.strip()
    if not config_str:
        return {}

    # Handle both "Config(...)" and "helion.Config(...)" formats
    config_str = config_str.removeprefix("helion.")

    if not config_str.startswith("Config(") or not config_str.endswith(")"):
        return {}

    try:
        # Parse as a function call expression
        tree = ast.parse(config_str, mode="eval")
        call = tree.body
        if not isinstance(call, ast.Call):
            return {}

        result: dict[str, Any] = {}
        for kw in call.keywords:
            if kw.arg is not None:
                # Use literal_eval on the unparsed value
                result[kw.arg] = ast.literal_eval(ast.unparse(kw.value))
        return result
    except (ValueError, SyntaxError):
        return {}


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


def infer_config_spec(configs: list[dict[str, Any]]) -> ConfigSpec:
    """Infer a minimal ConfigSpec from a list of config dictionaries.

    Takes a list of config dicts and infers the structure (number of dimensions
    for block_sizes, indexing, etc.) from the first non-empty config.
    """
    from .config_fragment import EnumFragment
    from .config_fragment import ListOf
    from .config_spec import BlockSizeSpec
    from .config_spec import ConfigSpec

    # Find structure from first valid config
    sample: dict[str, Any] = {}
    for config in configs:
        if config:
            sample = config
            break

    config_spec = ConfigSpec()

    # Block sizes determine the number of dimensions
    block_sizes = sample.get("block_sizes", [])
    for i in range(len(block_sizes)):
        config_spec.block_sizes.append(BlockSizeSpec(block_id=i, size_hint=64))

    # Set indexing length to match block_sizes
    n_dims = len(config_spec.block_sizes) or 1
    config_spec.indexing = ListOf(
        EnumFragment(choices=("pointer", "block_ptr", "tensor_descriptor")),
        length=n_dims,
    )

    # Set load_eviction_policies length if present
    eviction = sample.get("load_eviction_policies", [])
    if isinstance(eviction, list) and eviction:
        from .config_spec import VALID_EVICTION_POLICIES

        config_spec.load_eviction_policies = ListOf(
            EnumFragment(choices=VALID_EVICTION_POLICIES),
            length=len(eviction),
        )

    return config_spec


class PerformanceProxyModel:
    """Predicts kernel performance from config parameters using KNN."""

    def __init__(self, n_neighbors: int = 5) -> None:
        self._model = KNeighborsRegressor(n_neighbors=n_neighbors, weights="distance")
        self._is_fitted = False
        self._config_strs: list[str] = []
        self._perfs: list[float] = []
        self._config_gen: ConfigGeneration | None = None

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
        # Always create ConfigGeneration for prediction support
        configs = [r.config_dict for r in records if r.status == "ok" and r.config_dict]
        config_spec = infer_config_spec(configs)
        self._config_gen = ConfigGeneration(config_spec)

        X: list[list[float]] = []
        y: list[float] = []

        for record in records:
            if record.perf_ms is None or not math.isfinite(record.perf_ms):
                continue
            if record.status != "ok":
                continue

            config = Config(**record.config_dict)
            flat_config = self._config_gen.flatten(config)
            features = self._config_gen.encode_config(flat_config)
            X.append(features)
            y.append(record.perf_ms)
            self._config_strs.append(record.config_str)
            self._perfs.append(record.perf_ms)

        if len(X) < 2:
            raise ValueError(f"Need at least 2 valid records to train, got {len(X)}")

        self._model.fit(np.array(X), np.array(y))
        self._is_fitted = True

    def predict(self, config_dict: dict[str, Any]) -> float:
        """Predict performance for a config dictionary."""
        if not self._is_fitted or self._config_gen is None:
            raise RuntimeError("Model must be fitted before prediction")

        config = Config(**config_dict)
        flat_config = self._config_gen.flatten(config)
        features = self._config_gen.encode_config(flat_config)
        return max(0.001, float(self._model.predict([features])[0]))

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
    """Predicts compilation time from config parameters using KNN."""

    def __init__(self, n_neighbors: int = 5) -> None:
        self._model = KNeighborsRegressor(n_neighbors=n_neighbors, weights="distance")
        self._is_fitted = False
        self._config_gen: ConfigGeneration | None = None

    @classmethod
    def from_csv(cls, csv_path: str | Path) -> CompileTimeProxyModel:
        """Create and train a compile time proxy model from an autotuner CSV log."""
        proxy = cls()
        records = load_autotune_log(csv_path)
        proxy.fit_from_records(records)
        return proxy

    def fit_from_records(self, records: list[AutotuneLogRecord]) -> None:
        """Train the model from autotuner log records."""
        # Infer ConfigSpec from records and create ConfigGeneration for encoding
        configs = [r.config_dict for r in records if r.config_dict]
        config_spec = infer_config_spec(configs)
        self._config_gen = ConfigGeneration(config_spec)

        X: list[list[float]] = []
        y: list[float] = []

        for record in records:
            if record.compile_time_s is None or not math.isfinite(
                record.compile_time_s
            ):
                continue

            config = Config(**record.config_dict)
            flat_config = self._config_gen.flatten(config)
            features = self._config_gen.encode_config(flat_config)
            X.append(features)
            y.append(record.compile_time_s)

        if len(X) < 2:
            raise ValueError(
                f"Need at least 2 valid records with compile time to train, got {len(X)}"
            )

        self._model.fit(np.array(X), np.array(y))
        self._is_fitted = True

    def predict(self, config_dict: dict[str, Any]) -> float:
        """Predict compile time for a config dictionary."""
        if not self._is_fitted or self._config_gen is None:
            raise RuntimeError("Model must be fitted before prediction")

        config = Config(**config_dict)
        flat_config = self._config_gen.flatten(config)
        features = self._config_gen.encode_config(flat_config)
        return max(0.001, float(self._model.predict([features])[0]))


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
