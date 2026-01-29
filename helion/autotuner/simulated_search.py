"""
Simulated search environment for testing autotuner algorithms.

Provides infrastructure for running autotuner search algorithms against a
proxy model instead of actual kernel benchmarking, enabling fast CI testing
and reproducible comparisons between search strategies.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from math import inf
import random
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Literal

# Import the search algorithm registry
from . import search_algorithms
from .proxy_model import AutotuneLogRecord
from .proxy_model import CompileTimeProxyModel
from .proxy_model import PerformanceProxyModel
from .proxy_model import infer_config_spec
from .proxy_model import load_autotune_log

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from ..runtime.config import Config
    from .base_search import BenchmarkResult
    from .base_search import PrecompileFuture
    from .config_spec import ConfigSpec


@dataclass
class ConfigEvaluation:
    """Record of a single config evaluation during simulated search."""

    config_dict: dict[str, Any]
    config_str: str
    perf_ms: float
    evaluation_index: int
    is_best_so_far: bool
    compile_time_s: float = 0.0  # Estimated compile time


@dataclass
class SearchRunResult:
    """Result of running a single search algorithm."""

    search_name: str
    n_configs_evaluated: int
    best_config: dict[str, Any]
    best_perf: float
    true_best_perf: float
    percent_of_best: float  # Computed after comparison (relative to best found)
    evaluations: list[ConfigEvaluation]
    metrics_at_n: list[tuple[int, float, float]]  # (n, best_perf, percent)
    total_compile_time_s: float = 0.0  # Estimated total compile time


@dataclass
class SearchComparisonResult:
    """Result of comparing multiple search algorithms."""

    results: dict[str, SearchRunResult]
    true_best_perf: float
    true_best_config: str
    n_configs_per_search: int


class _MockCompiledConfig:
    def __call__(self, *args: object, **kwargs: object) -> None:
        return None


class _SimulatedSettings:
    """Mock settings object with attributes needed by search algorithms."""

    def __init__(self, seed: int = 42) -> None:
        self.autotune_accuracy_check = False
        self.autotune_ignore_errors = True
        self.autotune_random_seed = seed
        self.autotune_config_overrides: dict[str, object] | None = None
        self.autotune_benchmark_fn = None
        self.autotune_progress_bar = False
        self.autotune_log_level = 0
        self.autotune_precompile = None
        self.autotune_precompile_jobs = 1
        self.autotune_rebenchmark_threshold = 1.05
        self.autotune_compile_timeout = 300
        self.autotune_baseline_fn = None
        self.autotune_baseline_atol = None
        self.autotune_baseline_rtol = None
        self.autotune_log = None
        self.autotune_force_persistent = False
        self.autotune_max_generations = None
        self.autotune_effort = "quick"
        self.autotune_cache = "local"
        self.print_output_code = False

    def get_rebenchmark_threshold(self) -> float:
        return self.autotune_rebenchmark_threshold


class _SimulatedBoundKernel:
    """Mock BoundKernel that provides interfaces needed by search algorithms."""

    def __init__(
        self,
        config_spec: ConfigSpec,
        settings: _SimulatedSettings,
    ) -> None:
        self.config_spec = config_spec
        self.settings = settings
        self._name = "simulated_kernel"
        self.configs: list[Config] = []

    def compile_config(
        self, config: object, allow_print: bool = True
    ) -> _MockCompiledConfig:
        return _MockCompiledConfig()

    def format_kernel_decorator(self, config: object, settings: object) -> str:
        return "@helion.kernel(simulated=True)"

    def maybe_log_repro(
        self, log_fn: Callable[..., None], args: object, config: object
    ) -> None:
        pass

    def get_cached_path(self, config: object = None) -> str | None:
        return None


class SimulatedSearchMixin:
    """Mixin that replaces benchmarking with proxy model predictions."""

    # Class attributes set before instantiation
    _perf_lookup: Callable[[dict[str, Any]], float]
    _compile_time_lookup: Callable[[dict[str, Any]], float]
    _evaluations: list[ConfigEvaluation]
    _best_perf_tracker: list[float]

    # These will be provided by the actual search class
    counters: Any
    best_perf_so_far: float

    def _compute_baseline(self) -> tuple[object, bool, Sequence[object] | None]:
        return None, False, None

    def _compute_effective_tolerances(self) -> tuple[float, float]:
        return 1e-2, 1e-2

    def _clone_args(self, args: Sequence[object]) -> Sequence[object]:
        return args

    def _decide_num_jobs(self) -> int:
        return 1

    def _record_evaluation(self, config: Config, perf: float) -> None:
        config_dict = config.config if hasattr(config, "config") else dict(config)
        config_str = str(config)

        is_best = perf < self._best_perf_tracker[0]
        if is_best:
            self._best_perf_tracker[0] = perf

        compile_time = self._compile_time_lookup(config_dict)

        self._evaluations.append(
            ConfigEvaluation(
                config_dict=config_dict,
                config_str=config_str,
                perf_ms=perf,
                evaluation_index=len(self._evaluations) + 1,
                is_best_so_far=is_best,
                compile_time_s=compile_time,
            )
        )

        self.counters["benchmark"] += 1
        if perf < self.best_perf_so_far:
            self.best_perf_so_far = perf

    def benchmark(self, config: Config) -> tuple[Callable[..., object], float]:
        config_dict = config.config if hasattr(config, "config") else dict(config)
        perf = self._perf_lookup(config_dict)
        self._record_evaluation(config, perf)
        return _MockCompiledConfig(), perf

    def benchmark_function(self, config: Config, fn: object) -> float:
        config_dict = config.config if hasattr(config, "config") else dict(config)
        perf = self._perf_lookup(config_dict)
        self._record_evaluation(config, perf)
        return perf

    def start_precompile_and_check_for_hangs(
        self, config: Config, fn: object
    ) -> PrecompileFuture:
        from .base_search import PrecompileFuture

        return PrecompileFuture.skip(self, config, True)  # type: ignore[arg-type]

    def parallel_benchmark(
        self, configs: list[Config], *, desc: str = "Benchmarking"
    ) -> list[BenchmarkResult]:
        from .base_search import BenchmarkResult

        results: list[BenchmarkResult] = []
        for config in configs:
            fn, perf = self.benchmark(config)
            status: Literal["ok", "error", "timeout"] = (
                "ok" if math.isfinite(perf) else "error"
            )
            results.append(
                BenchmarkResult(
                    config=config,
                    fn=fn,
                    perf=perf,
                    status=status,
                    compile_time=0.0,
                )
            )
        return results

    def rebenchmark(self, members: list[Any], *, desc: str = "Rebenchmarking") -> None:
        pass


def _create_simulated_search(
    search_class: type,
    perf_lookup: Callable[[dict[str, Any]], float],
    compile_time_lookup: Callable[[dict[str, Any]], float],
    evaluations: list[ConfigEvaluation],
    best_perf_tracker: list[float],
) -> type:
    """Create a simulated version of a search class."""

    class SimulatedSearch(SimulatedSearchMixin, search_class):  # type: ignore[misc]
        _perf_lookup = staticmethod(perf_lookup)  # type: ignore[misc]
        _compile_time_lookup = staticmethod(compile_time_lookup)  # type: ignore[misc]
        _evaluations = evaluations
        _best_perf_tracker = best_perf_tracker

    return SimulatedSearch


class SimulatedSearchRunner:
    """Runner for evaluating search algorithms using proxy models."""

    def __init__(
        self,
        perf_model: PerformanceProxyModel,
        records: list[AutotuneLogRecord],
        compile_time_model: CompileTimeProxyModel | None = None,
    ) -> None:
        self.perf_model = perf_model
        self.compile_time_model = compile_time_model
        self.records = records
        self._true_best_config, self._true_best_perf = perf_model.get_best_observed()

        # Build config spec from records
        configs = [r.config_dict for r in records if r.status == "ok" and r.config_dict]
        self._config_spec = infer_config_spec(configs)

        # Build lookup for recorded performance values
        self._config_dict_to_perf: dict[str, float] = {}
        for record in records:
            if record.perf_ms is not None and record.status == "ok":
                self._config_dict_to_perf[str(record.config_dict)] = record.perf_ms

    @classmethod
    def from_csv(cls, csv_path: str | Path) -> SimulatedSearchRunner:
        """Create a search runner from an autotuner CSV log."""
        import contextlib

        records = load_autotune_log(csv_path)
        perf_model = PerformanceProxyModel.from_records(records)

        # Try to build compile time model (may fail if no compile time data)
        compile_time_model = None
        with contextlib.suppress(ValueError):
            compile_time_model = CompileTimeProxyModel.from_csv(csv_path)

        return cls(perf_model, records, compile_time_model)

    def _get_perf(self, config_dict: dict[str, Any]) -> float:
        """Get performance from recorded data or proxy model."""
        config_str = str(config_dict)
        if config_str in self._config_dict_to_perf:
            return self._config_dict_to_perf[config_str]
        return self.perf_model.predict(config_dict)

    def _get_compile_time(self, config_dict: dict[str, Any]) -> float:
        if self.compile_time_model is not None:
            return self.compile_time_model.predict(config_dict)
        return 0.0

    @staticmethod
    def get_available_searches() -> list[str]:
        """Get list of available search algorithm names."""
        return [name for name in search_algorithms if name != "FiniteSearch"]

    def _get_default_kwargs(
        self, search_class_name: str, n_configs: int
    ) -> dict[str, Any]:
        """Get default kwargs for a search algorithm based on n_configs budget."""
        defaults: dict[str, dict[str, Any]] = {
            "RandomSearch": {
                "count": n_configs,
            },
            "PatternSearch": {
                "initial_population": min(n_configs // 2, 20),
                "copies": 2,
                "max_generations": max(1, n_configs // 10),
            },
            "LFBOPatternSearch": {
                "initial_population": min(n_configs // 2, 20),
                "copies": 2,
                "max_generations": max(1, n_configs // 10),
            },
            "DifferentialEvolutionSearch": {
                "population_size": min(n_configs // 4, 20),
                "max_generations": max(2, n_configs // 10),
            },
            "DESurrogateHybrid": {
                "population_size": min(n_configs // 4, 20),
                "max_generations": max(2, n_configs // 10),
            },
        }
        return defaults.get(search_class_name, {})

    def run_search(
        self,
        search_class_name: str,
        n_configs: int,
        seed: int | None = None,
        **search_kwargs: object,
    ) -> SearchRunResult:
        """Run a search algorithm using the simulated environment."""
        if seed is not None:
            random.seed(seed)

        if search_class_name not in search_algorithms:
            available = ", ".join(self.get_available_searches())
            raise ValueError(
                f"Unknown search class: {search_class_name}. Available: {available}"
            )

        if search_class_name == "FiniteSearch":
            raise ValueError(
                "FiniteSearch requires explicit config list, "
                "not suitable for simulated search"
            )

        evaluations: list[ConfigEvaluation] = []
        best_perf_tracker = [inf]

        settings = _SimulatedSettings(seed=seed or 42)
        mock_kernel = _SimulatedBoundKernel(self._config_spec, settings)

        search_class = search_algorithms[search_class_name]

        final_kwargs = self._get_default_kwargs(search_class_name, n_configs)
        final_kwargs.update(search_kwargs)

        SimulatedSearch = _create_simulated_search(
            search_class,
            self._get_perf,
            self._get_compile_time,
            evaluations,
            best_perf_tracker,
        )

        search = SimulatedSearch(mock_kernel, [], **final_kwargs)  # type: ignore[arg-type]
        search.autotune()
        best_config: dict[str, Any] = {}
        best_perf = inf
        for ev in evaluations:
            if ev.perf_ms < best_perf:
                best_perf = ev.perf_ms
                best_config = ev.config_dict

        total_compile_time = sum(ev.compile_time_s for ev in evaluations)

        # Percent relative to training data best (capped at 100%)
        if best_perf > 0 and best_perf < inf:
            percent = min(100.0, (self._true_best_perf / best_perf) * 100)
        else:
            percent = 0.0

        metrics = self._compute_metrics_at_n(evaluations)

        return SearchRunResult(
            search_name=search_class_name,
            n_configs_evaluated=len(evaluations),
            best_config=best_config,
            best_perf=best_perf if best_perf < inf else 0.0,
            true_best_perf=self._true_best_perf,
            percent_of_best=percent,
            evaluations=evaluations,
            metrics_at_n=metrics,
            total_compile_time_s=total_compile_time,
        )

    def run_random_search(
        self,
        n_configs: int,
        seed: int | None = None,
    ) -> SearchRunResult:
        return self.run_search("RandomSearch", n_configs, seed=seed)

    def run_pattern_search(
        self,
        n_configs: int,
        initial_population: int = 10,
        seed: int | None = None,
    ) -> SearchRunResult:
        return self.run_search(
            "PatternSearch",
            n_configs,
            seed=seed,
            initial_population=initial_population,
        )

    def run_differential_evolution(
        self,
        n_configs: int,
        population_size: int = 20,
        seed: int | None = None,
    ) -> SearchRunResult:
        return self.run_search(
            "DifferentialEvolutionSearch",
            n_configs,
            seed=seed,
            population_size=population_size,
        )

    def _compute_metrics_at_n(
        self,
        evaluations: list[ConfigEvaluation],
        n_values: Sequence[int] | None = None,
        reference_best: float | None = None,
    ) -> list[tuple[int, float, float]]:
        """Compute % of best at various n values."""
        if not evaluations:
            return []

        if n_values is None:
            n_values = [1, 5, 10, 20, 50, 100, 200, 500, 1000]
            n_values = [n for n in n_values if n <= len(evaluations)]
            if evaluations and len(evaluations) not in n_values:
                n_values.append(len(evaluations))

        # Use provided reference or the best found in this run
        if reference_best is None:
            reference_best = min(e.perf_ms for e in evaluations)

        results: list[tuple[int, float, float]] = []
        for n in n_values:
            subset = evaluations[:n]
            best_at_n = min(e.perf_ms for e in subset)
            # Percent relative to reference best
            percent = (reference_best / best_at_n) * 100 if best_at_n > 0 else 0
            results.append((n, best_at_n, percent))

        return results

    def compare_searches(
        self,
        search_names: Sequence[str] | None = None,
        n_configs: int = 100,
        seed: int | None = 42,
    ) -> SearchComparisonResult:
        """Compare multiple search algorithms."""
        from dataclasses import replace

        if search_names is None:
            search_names = self.get_available_searches()

        results: dict[str, SearchRunResult] = {}

        for name in search_names:
            results[name] = self.run_search(name, n_configs, seed=seed)

        # Find the best performance across all algorithms
        overall_best = min(
            (r.best_perf for r in results.values() if r.best_perf > 0),
            default=inf,
        )

        # Update percent_of_best and metrics_at_n relative to the overall best
        for name, result in results.items():
            if result.best_perf > 0 and overall_best < inf:
                percent = (overall_best / result.best_perf) * 100
            else:
                percent = 0.0

            # Recompute metrics_at_n relative to overall best
            new_metrics = self._compute_metrics_at_n(
                result.evaluations, reference_best=overall_best
            )

            results[name] = replace(
                result, percent_of_best=percent, metrics_at_n=new_metrics
            )

        return SearchComparisonResult(
            results=results,
            true_best_perf=self._true_best_perf,
            true_best_config=self._true_best_config,
            n_configs_per_search=n_configs,
        )

    def print_comparison_report(self, comparison: SearchComparisonResult) -> None:
        best_found = min(
            (r.best_perf for r in comparison.results.values() if r.best_perf > 0),
            default=0.0,
        )

        print("\n" + "=" * 90)
        print("Search Algorithm Comparison Report")
        print("=" * 90)
        print(f"Best performance found: {best_found:.4f} ms")
        print(f"(Training data best: {comparison.true_best_perf:.4f} ms)")
        print(f"Configs per search: {comparison.n_configs_per_search}")
        print()

        # Summary table with compile time
        print(
            f"{'Algorithm':<28} {'Best (ms)':>12} {'% of Best':>11} "
            f"{'Configs':>9} {'Compile (s)':>12}"
        )
        print("-" * 75)

        for name, result in sorted(
            comparison.results.items(), key=lambda x: -x[1].percent_of_best
        ):
            compile_str = (
                f"{result.total_compile_time_s:>12.1f}"
                if result.total_compile_time_s > 0
                else f"{'N/A':>12}"
            )
            print(
                f"{name:<28} {result.best_perf:>12.4f} "
                f"{result.percent_of_best:>10.1f}% "
                f"{result.n_configs_evaluated:>9} {compile_str}"
            )

        # Detailed metrics
        print()
        print("% of Best at N configs evaluated:")
        print("-" * 75)

        # Collect all n values
        all_n: set[int] = set()
        for result in comparison.results.values():
            all_n.update(n for n, _, _ in result.metrics_at_n)

        # Print header
        header = f"{'N':>8}"
        for name in comparison.results:
            header += f" {name[:12]:>14}"
        print(header)
        print("-" * 75)

        # Print rows
        for n in sorted(all_n):
            row = f"{n:>8}"
            for result in comparison.results.values():
                percent = next((p for nn, _, p in result.metrics_at_n if nn == n), None)
                if percent is not None:
                    row += f" {percent:>13.1f}%"
                else:
                    row += f" {'N/A':>14}"
            print(row)

        print("=" * 90 + "\n")
