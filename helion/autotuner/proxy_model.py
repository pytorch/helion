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
    from collections.abc import Callable
    from collections.abc import Iterable
    from collections.abc import Sequence


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


def _next_power_of_2(n: int) -> int:
    """Round up to the next power of 2."""
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def _prev_power_of_2(n: int) -> int:
    """Round down to the previous power of 2."""
    if n <= 1:
        return 1
    return 1 << (n.bit_length() - 1)


def _infer_config_spec_from_records(
    records: list[AutotuneLogRecord],
) -> ConfigSpec:
    """Infer a ConfigSpec from recorded autotuning data."""
    from .config_spec import VALID_EVICTION_POLICIES
    from .config_spec import BlockSizeSpec
    from .config_spec import ConfigSpec

    # Collect ranges from recorded configs
    block_size_dims: dict[int, tuple[int, int]] = {}  # dim -> (min, max)
    num_warps_range = (4, 4)  # (min, max)
    num_stages_range = (1, 1)
    indexing_types: set[str] = set()
    pid_types: set[str] = set()
    eviction_policy_length = 0

    for record in records:
        if record.status != "ok" or not record.config_dict:
            continue

        config = record.config_dict

        # Block sizes
        block_sizes = config.get("block_sizes", [])
        if isinstance(block_sizes, list):
            for i, bs in enumerate(block_sizes):
                if isinstance(bs, int) and bs > 0:
                    if i not in block_size_dims:
                        block_size_dims[i] = (bs, bs)
                    else:
                        cur_min, cur_max = block_size_dims[i]
                        block_size_dims[i] = (min(cur_min, bs), max(cur_max, bs))

        # num_warps
        nw = config.get("num_warps", 4)
        if isinstance(nw, int) and nw > 0:
            num_warps_range = (
                min(num_warps_range[0], nw),
                max(num_warps_range[1], nw),
            )

        # num_stages
        ns = config.get("num_stages", 1)
        if isinstance(ns, int) and ns > 0:
            num_stages_range = (
                min(num_stages_range[0], ns),
                max(num_stages_range[1], ns),
            )

        # indexing
        indexing = config.get("indexing", "pointer")
        if isinstance(indexing, str):
            indexing_types.add(indexing)
        elif isinstance(indexing, list):
            for idx in indexing:
                if isinstance(idx, str):
                    indexing_types.add(idx)

        # pid_type
        pid = config.get("pid_type", "flat")
        if isinstance(pid, str):
            pid_types.add(pid)

        # load_eviction_policies
        eviction = config.get("load_eviction_policies", [])
        if isinstance(eviction, list):
            eviction_policy_length = max(eviction_policy_length, len(eviction))

    # Build ConfigSpec
    config_spec = ConfigSpec()

    # Add block sizes (ensuring power-of-two min/max)
    for i in sorted(block_size_dims.keys()):
        min_bs, max_bs = block_size_dims[i]
        # Round min down and max up to nearest power of 2
        min_bs_p2 = max(1, _prev_power_of_2(min_bs))
        max_bs_p2 = max(min_bs_p2, _next_power_of_2(max_bs))
        config_spec.block_sizes.append(
            BlockSizeSpec(
                block_id=i,
                size_hint=max_bs_p2,
                min_size=min_bs_p2,
                max_size=max_bs_p2,
            )
        )

    # Set allowed pid types
    if pid_types:
        valid_pid_types = tuple(
            pt
            for pt in ("flat", "xyz", "persistent_blocked", "persistent_interleaved")
            if pt in pid_types
        )
        if valid_pid_types:
            config_spec.allowed_pid_types = valid_pid_types  # type: ignore[assignment]

    # Set indexing types
    if indexing_types:
        from .config_fragment import EnumFragment
        from .config_fragment import ListOf

        valid_indexing = tuple(
            it
            for it in ("pointer", "block_ptr", "tensor_descriptor")
            if it in indexing_types
        )
        if valid_indexing:
            config_spec.indexing = ListOf(
                EnumFragment(choices=valid_indexing),
                length=len(block_size_dims) if block_size_dims else 1,
            )

    # Set load_eviction_policies length
    if eviction_policy_length > 0:
        from .config_fragment import EnumFragment
        from .config_fragment import ListOf

        config_spec.load_eviction_policies = ListOf(
            EnumFragment(choices=VALID_EVICTION_POLICIES),
            length=eviction_policy_length,
        )

    return config_spec


# Type alias for ConfigSpec to avoid circular imports at runtime
if TYPE_CHECKING:
    from .config_spec import ConfigSpec


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
        config_spec = _infer_config_spec_from_records(records)
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
        config_spec = _infer_config_spec_from_records(records)
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
