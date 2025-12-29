"""Tests for the proxy model and simulated search infrastructure."""

from __future__ import annotations

from pathlib import Path
import random
import tempfile

import pytest

from helion.autotuner.proxy_model import CompileTimeProxyModel
from helion.autotuner.proxy_model import PerformanceProxyModel
from helion.autotuner.proxy_model import SimulatedBenchmark
from helion.autotuner.proxy_model import load_autotune_log
from helion.autotuner.proxy_model import parse_config_str
from helion.autotuner.simulated_search import SimulatedSearchRunner

SAMPLE_AUTOTUNE_LOG = """\
timestamp_s,config_index,generation,status,perf_ms,compile_time_s,config
0.10,1,0,started,,,"Config(block_sizes=[32, 32], num_warps=4, num_stages=1, indexing='pointer')"
0.50,1,0,ok,1.234000,0.40,"Config(block_sizes=[32, 32], num_warps=4, num_stages=1, indexing='pointer')"
0.60,2,0,started,,,"Config(block_sizes=[64, 64], num_warps=4, num_stages=1, indexing='pointer')"
1.20,2,0,ok,0.876000,0.60,"Config(block_sizes=[64, 64], num_warps=4, num_stages=1, indexing='pointer')"
1.30,3,0,started,,,"Config(block_sizes=[128, 64], num_warps=8, num_stages=2, indexing='block_ptr')"
2.10,3,0,ok,0.654000,0.80,"Config(block_sizes=[128, 64], num_warps=8, num_stages=2, indexing='block_ptr')"
2.20,4,0,started,,,"Config(block_sizes=[64, 128], num_warps=8, num_stages=2, indexing='block_ptr')"
3.00,4,0,ok,0.712000,0.80,"Config(block_sizes=[64, 128], num_warps=8, num_stages=2, indexing='block_ptr')"
3.10,5,1,started,,,"Config(block_sizes=[128, 128], num_warps=8, num_stages=2, indexing='block_ptr')"
4.20,5,1,ok,0.543000,1.10,"Config(block_sizes=[128, 128], num_warps=8, num_stages=2, indexing='block_ptr')"
4.30,6,1,started,,,"Config(block_sizes=[256, 64], num_warps=8, num_stages=3, indexing='block_ptr')"
5.50,6,1,ok,0.621000,1.20,"Config(block_sizes=[256, 64], num_warps=8, num_stages=3, indexing='block_ptr')"
5.60,7,1,started,,,"Config(block_sizes=[64, 256], num_warps=16, num_stages=3, indexing='tensor_descriptor')"
6.90,7,1,ok,0.598000,1.30,"Config(block_sizes=[64, 256], num_warps=16, num_stages=3, indexing='tensor_descriptor')"
7.00,8,1,started,,,"Config(block_sizes=[128, 256], num_warps=16, num_stages=3, indexing='tensor_descriptor')"
8.40,8,1,error,,,"Config(block_sizes=[128, 256], num_warps=16, num_stages=3, indexing='tensor_descriptor')"
8.50,9,2,started,,,"Config(block_sizes=[256, 128], num_warps=8, num_stages=2, indexing='block_ptr')"
9.80,9,2,ok,0.489000,1.30,"Config(block_sizes=[256, 128], num_warps=8, num_stages=2, indexing='block_ptr')"
9.90,10,2,started,,,"Config(block_sizes=[256, 256], num_warps=8, num_stages=2, indexing='block_ptr')"
11.50,10,2,ok,0.512000,1.60,"Config(block_sizes=[256, 256], num_warps=8, num_stages=2, indexing='block_ptr')"
"""


def create_sample_csv(content: str = SAMPLE_AUTOTUNE_LOG) -> Path:
    """Create a temporary CSV file with sample autotuner data."""
    tmpdir = tempfile.mkdtemp()
    csv_path = Path(tmpdir) / "autotune_test.csv"
    csv_path.write_text(content)
    return csv_path


@pytest.fixture
def sample_csv():
    """Fixture that provides a sample CSV file path."""
    path = create_sample_csv()
    yield path
    # Cleanup
    import shutil

    shutil.rmtree(path.parent, ignore_errors=True)


class TestConfigParsing:
    def test_parse_simple_config(self):
        config_str = "Config(block_sizes=[64, 64], num_warps=4)"
        result = parse_config_str(config_str)
        assert result["block_sizes"] == [64, 64]
        assert result["num_warps"] == 4

    def test_parse_complex_config(self):
        config_str = "Config(block_sizes=[128, 256], num_warps=8, num_stages=2, indexing='block_ptr')"
        result = parse_config_str(config_str)
        assert result["block_sizes"] == [128, 256]
        assert result["num_warps"] == 8
        assert result["num_stages"] == 2
        assert result["indexing"] == "block_ptr"

    def test_parse_with_nested_lists(self):
        config_str = (
            "Config(block_sizes=[64], loop_orders=[[0, 1, 2]], flatten_loops=[True])"
        )
        result = parse_config_str(config_str)
        assert result["block_sizes"] == [64]
        assert result["loop_orders"] == [[0, 1, 2]]
        assert result["flatten_loops"] == [True]

    def test_parse_empty_string(self):
        result = parse_config_str("")
        assert result == {}

    def test_parse_invalid_format(self):
        result = parse_config_str("not a config")
        assert result == {}


class TestAutotuneLogLoading:
    def test_load_sample_log(self, sample_csv):
        records = load_autotune_log(sample_csv)

        # Should skip 'started' entries and only get completed ones
        assert len(records) == 10

        # Check first record
        first = records[0]
        assert first.config_index == 1
        assert first.generation == 0
        assert first.status == "ok"
        assert first.perf_ms == pytest.approx(1.234)
        assert first.compile_time_s == pytest.approx(0.40)
        assert first.config_dict["block_sizes"] == [32, 32]

    def test_load_finds_best_config(self, sample_csv):
        records = load_autotune_log(sample_csv)

        # Find best (lowest perf)
        best = min(
            (r for r in records if r.perf_ms is not None),
            key=lambda r: r.perf_ms,  # type: ignore[arg-type, return-value]
        )
        assert best.perf_ms == pytest.approx(0.489)  # Config 9
        assert best.config_dict["block_sizes"] == [256, 128]


class TestPerformanceProxyModel:
    def test_train_from_csv(self, sample_csv):
        model = PerformanceProxyModel.from_csv(sample_csv)
        assert model._is_fitted

    def test_predict_returns_float(self, sample_csv):
        model = PerformanceProxyModel.from_csv(sample_csv)

        config = {"block_sizes": [64, 64], "num_warps": 4, "num_stages": 1}
        perf = model.predict(config)

        assert isinstance(perf, float)
        assert perf > 0

    def test_predict_from_config_str(self, sample_csv):
        model = PerformanceProxyModel.from_csv(sample_csv)

        config_str = "Config(block_sizes=[64, 64], num_warps=4, num_stages=1)"
        perf = model.predict_from_config_str(config_str)

        assert isinstance(perf, float)
        assert perf > 0

    def test_get_best_observed(self, sample_csv):
        model = PerformanceProxyModel.from_csv(sample_csv)

        best_config, best_perf = model.get_best_observed()

        assert best_perf == pytest.approx(0.489)
        assert (
            "256" in best_config or "128" in best_config
        )  # Block sizes from best config


class TestCompileTimeProxyModel:
    def test_train_from_csv(self, sample_csv):
        model = CompileTimeProxyModel.from_csv(sample_csv)
        assert model._is_fitted

    def test_predict_returns_float(self, sample_csv):
        model = CompileTimeProxyModel.from_csv(sample_csv)

        config = {"block_sizes": [128, 128], "num_warps": 8, "num_stages": 2}
        compile_time = model.predict(config)

        assert isinstance(compile_time, float)
        assert compile_time > 0


class TestSimulatedBenchmark:
    def test_create_from_csv(self, sample_csv):
        sim = SimulatedBenchmark.from_csv(sample_csv)
        assert sim.perf_model._is_fitted

    def test_benchmark_returns_result(self, sample_csv):
        sim = SimulatedBenchmark.from_csv(sample_csv)

        config = {"block_sizes": [64, 64], "num_warps": 4}
        result = sim.benchmark(config)

        assert result.config_dict == config
        assert result.predicted_perf > 0
        assert result.config_index == 1

    def test_tracks_history(self, sample_csv):
        sim = SimulatedBenchmark.from_csv(sample_csv)

        configs = [
            {"block_sizes": [32, 32], "num_warps": 4},
            {"block_sizes": [64, 64], "num_warps": 8},
            {"block_sizes": [128, 128], "num_warps": 8},
        ]
        for cfg in configs:
            sim.benchmark(cfg)

        history = sim.get_evaluation_history()
        assert len(history) == 3

    def test_compute_metrics_at_n(self, sample_csv):
        sim = SimulatedBenchmark.from_csv(sample_csv)

        # Benchmark several configs
        for _ in range(20):
            config = {
                "block_sizes": [random.choice([32, 64, 128, 256]) for _ in range(2)],
                "num_warps": random.choice([4, 8, 16]),
                "num_stages": random.randint(1, 3),
            }
            sim.benchmark(config)

        metrics = sim.compute_metrics_at_n([1, 5, 10, 20])

        assert len(metrics) == 4
        for n, perf, percent in metrics:
            assert n in [1, 5, 10, 20]
            assert perf > 0
            assert 0 <= percent <= 100


class TestSimulatedSearchRunner:
    def test_create_from_csv(self, sample_csv):
        runner = SimulatedSearchRunner.from_csv(sample_csv)
        assert runner.perf_model._is_fitted
        assert len(runner.records) > 0

    def test_random_search(self, sample_csv):
        runner = SimulatedSearchRunner.from_csv(sample_csv)
        result = runner.run_random_search(n_configs=50, seed=42)

        assert result.search_name == "RandomSearch"
        assert result.n_configs_evaluated <= 50
        assert result.n_configs_evaluated >= 1
        assert result.best_perf > 0
        assert 0 < result.percent_of_best <= 100

    def test_pattern_search(self, sample_csv):
        runner = SimulatedSearchRunner.from_csv(sample_csv)
        result = runner.run_pattern_search(n_configs=50, seed=42)

        assert result.search_name == "PatternSearch"
        assert result.n_configs_evaluated >= 1
        assert result.best_perf > 0

    def test_differential_evolution(self, sample_csv):
        runner = SimulatedSearchRunner.from_csv(sample_csv)
        result = runner.run_differential_evolution(n_configs=50, seed=42)

        assert result.search_name == "DifferentialEvolutionSearch"
        assert result.best_perf > 0

    def test_compare_searches(self, sample_csv):
        runner = SimulatedSearchRunner.from_csv(sample_csv)
        comparison = runner.compare_searches(
            ["RandomSearch", "PatternSearch"],
            n_configs=30,
            seed=42,
        )

        assert "RandomSearch" in comparison.results
        assert "PatternSearch" in comparison.results
        assert comparison.true_best_perf > 0

    def test_metrics_at_n_progression(self, sample_csv):
        runner = SimulatedSearchRunner.from_csv(sample_csv)
        result = runner.run_random_search(n_configs=100, seed=42)

        metrics = result.metrics_at_n
        prev_best_perf = float("inf")
        for _n, best_perf, _percent in metrics:
            assert best_perf <= prev_best_perf + 0.001
            prev_best_perf = best_perf

    def test_get_available_searches(self, sample_csv):
        runner = SimulatedSearchRunner.from_csv(sample_csv)
        available = runner.get_available_searches()

        assert "RandomSearch" in available
        assert "PatternSearch" in available
        assert "DifferentialEvolutionSearch" in available
        assert "LFBOPatternSearch" in available
        assert "DESurrogateHybrid" in available
        assert "FiniteSearch" not in available

    def test_run_search_generic(self, sample_csv):
        runner = SimulatedSearchRunner.from_csv(sample_csv)
        for search_name in runner.get_available_searches():
            result = runner.run_search(search_name, n_configs=20, seed=42)
            assert result.search_name == search_name
            assert result.best_perf > 0

    def test_compare_searches_all(self, sample_csv):
        runner = SimulatedSearchRunner.from_csv(sample_csv)
        comparison = runner.compare_searches(None, n_configs=20, seed=42)
        available = runner.get_available_searches()
        assert set(comparison.results.keys()) == set(available)


class TestSearchQualityMetrics:
    def test_print_search_quality_report(self, sample_csv, capsys):
        runner = SimulatedSearchRunner.from_csv(sample_csv)
        random.seed(42)

        comparison = runner.compare_searches(
            ["RandomSearch", "PatternSearch", "DifferentialEvolutionSearch"],
            n_configs=50,
            seed=42,
        )

        # Print the report for CI visibility
        print("\n" + "=" * 80)
        print("AUTOTUNER SEARCH QUALITY TEST RESULTS")
        print("=" * 80)
        print(
            f"True best performance (from training data): {comparison.true_best_perf:.4f} ms"
        )
        print(f"Configs evaluated per algorithm: {comparison.n_configs_per_search}")
        print()

        print("Final Results:")
        print("-" * 60)
        for name, result in sorted(
            comparison.results.items(), key=lambda x: -x[1].percent_of_best
        ):
            print(
                f"  {name:<30} "
                f"Best: {result.best_perf:.4f} ms  "
                f"({result.percent_of_best:.1f}% of optimal)"
            )

        print()
        print("% of Best @ N Configs Evaluated:")
        print("-" * 60)

        # Get all N values
        all_n = sorted(
            {n for r in comparison.results.values() for n, _, _ in r.metrics_at_n}
        )

        header = f"{'N':>6}"
        for name in comparison.results:
            header += f" {name[:15]:>16}"
        print(header)

        for n in all_n:
            row = f"{n:>6}"
            for result in comparison.results.values():
                percent = next((p for nn, _, p in result.metrics_at_n if nn == n), None)
                if percent is not None:
                    row += f" {percent:>15.1f}%"
                else:
                    row += f" {'N/A':>16}"
            print(row)

        print("=" * 80)

        captured = capsys.readouterr()
        assert "AUTOTUNER SEARCH QUALITY TEST RESULTS" in captured.out
        assert "% of Best" in captured.out

    def test_search_quality_benchmark(self, sample_csv):
        runner = SimulatedSearchRunner.from_csv(sample_csv)
        results = {}
        for search_name in [
            "RandomSearch",
            "PatternSearch",
            "DifferentialEvolutionSearch",
        ]:
            if search_name == "RandomSearch":
                results[search_name] = runner.run_random_search(n_configs=50, seed=42)
            elif search_name == "PatternSearch":
                results[search_name] = runner.run_pattern_search(n_configs=50, seed=42)
            else:
                results[search_name] = runner.run_differential_evolution(
                    n_configs=50, seed=42
                )

        for name, result in results.items():
            assert result.percent_of_best > 10, f"{name} performed very poorly"

        print("\n--- Search Quality Summary ---")
        for name, result in sorted(
            results.items(), key=lambda x: -x[1].percent_of_best
        ):
            print(
                f"{name}: {result.percent_of_best:.1f}% of best after {result.n_configs_evaluated} configs"
            )


class TestFlattenUnflattenRoundTrip:
    """Tests for flatten/unflatten round-trip in ConfigGeneration."""

    def test_flat_values_round_trip(self, sample_csv):
        """Test that flat values round-trip: flatten(unflatten(flat)) == flat."""
        from helion.autotuner.config_generation import ConfigGeneration
        from helion.autotuner.proxy_model import infer_config_spec

        records = load_autotune_log(sample_csv)
        configs = [r.config_dict for r in records if r.status == "ok" and r.config_dict]
        config_spec = infer_config_spec(configs)
        config_gen = ConfigGeneration(config_spec)

        # Test with default flat values
        default_flat = config_gen.default_flat()
        config = config_gen.unflatten(default_flat)
        flat_again = config_gen.flatten(config)
        assert default_flat == flat_again, (
            f"Default flat round-trip failed\n"
            f"Original: {default_flat}\n"
            f"After round-trip: {flat_again}"
        )

        # Test with random flat values
        random.seed(42)
        for _ in range(10):
            flat = config_gen.random_flat()
            config = config_gen.unflatten(flat)
            flat_again = config_gen.flatten(config)
            assert flat == flat_again, (
                f"Random flat round-trip failed\n"
                f"Original: {flat}\n"
                f"After round-trip: {flat_again}"
            )

    def test_normalized_config_round_trip(self, sample_csv):
        """Test that normalized configs round-trip: unflatten(flatten(config)) == config."""
        from helion.autotuner.config_generation import ConfigGeneration
        from helion.autotuner.proxy_model import infer_config_spec

        records = load_autotune_log(sample_csv)
        configs = [r.config_dict for r in records if r.status == "ok" and r.config_dict]
        config_spec = infer_config_spec(configs)
        config_gen = ConfigGeneration(config_spec)

        # Create normalized configs via unflatten (which always produces normalized configs)
        random.seed(42)
        for _ in range(10):
            flat = config_gen.random_flat()
            config = config_gen.unflatten(flat)
            flat_again = config_gen.flatten(config)
            recovered = config_gen.unflatten(flat_again)

            assert config.config == recovered.config, (
                f"Normalized config round-trip failed\n"
                f"Original: {config.config}\n"
                f"Recovered: {recovered.config}"
            )

    def test_csv_configs_flatten_consistently(self, sample_csv):
        """Test that CSV configs produce consistent flat values and encodings."""
        from helion import Config
        from helion.autotuner.config_generation import ConfigGeneration
        from helion.autotuner.proxy_model import infer_config_spec

        records = load_autotune_log(sample_csv)
        configs = [r.config_dict for r in records if r.status == "ok" and r.config_dict]
        config_spec = infer_config_spec(configs)
        config_gen = ConfigGeneration(config_spec)

        for record in records:
            if record.status != "ok" or not record.config_dict:
                continue

            config = Config(**record.config_dict)
            flat1 = config_gen.flatten(config)
            flat2 = config_gen.flatten(config)

            # Flattening should be deterministic
            assert flat1 == flat2, f"Flatten not deterministic for {record.config_str}"

            # After flatten/unflatten, we should get a normalized config
            # that then round-trips perfectly
            normalized = config_gen.unflatten(flat1)
            flat3 = config_gen.flatten(normalized)
            assert flat1 == flat3, (
                f"Normalized config should produce same flat values\n"
                f"Original flat: {flat1}\n"
                f"After normalize: {flat3}"
            )


class TestWithDataFile:
    @pytest.fixture
    def data_csv(self):
        return Path(__file__).parent / "data" / "sample_autotune_log.csv"

    def test_load_from_data_file(self, data_csv):
        if not data_csv.exists():
            pytest.skip("Sample data file not found")

        records = load_autotune_log(data_csv)
        assert len(records) > 10

        model = PerformanceProxyModel.from_csv(data_csv)
        assert model._is_fitted

    def test_full_search_comparison_with_data_file(self, data_csv, capsys):
        if not data_csv.exists():
            pytest.skip("Sample data file not found")

        runner = SimulatedSearchRunner.from_csv(data_csv)
        comparison = runner.compare_searches(
            ["RandomSearch", "PatternSearch", "DifferentialEvolutionSearch"],
            n_configs=100,
            seed=42,
        )
        runner.print_comparison_report(comparison)

        for name, result in comparison.results.items():
            assert result.percent_of_best > 50, (
                f"{name} only achieved {result.percent_of_best:.1f}% of optimal"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
