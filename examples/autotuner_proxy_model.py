#!/usr/bin/env python3
"""
Example: Evaluating Autotuner Search Algorithms with Proxy Models

Demonstrates how to use proxy models to evaluate autotuner search algorithm
quality without running kernels.

Usage:
    python autotuner_proxy_model.py                           # Use sample data
    python autotuner_proxy_model.py --generate-csv data.csv --size 512  # Generate CSV (requires GPU)
    python autotuner_proxy_model.py --csv data.csv            # Use custom CSV
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tempfile

import numpy as np

from helion.autotuner.proxy_model import CompileTimeProxyModel
from helion.autotuner.proxy_model import PerformanceProxyModel
from helion.autotuner.proxy_model import SimulatedBenchmark
from helion.autotuner.proxy_model import load_autotune_log
from helion.autotuner.simulated_search import SimulatedSearchRunner


def generate_autotuner_csv(output_path: str, matrix_size: int = 512) -> Path:
    """Generate an autotuner CSV log by running actual autotuning on a matmul kernel."""
    try:
        import torch
    except ImportError as e:
        print(f"ERROR: Cannot generate CSV - missing dependencies: {e}")
        print("Make sure torch and helion are installed.")
        sys.exit(1)

    if not torch.cuda.is_available():
        print("ERROR: GPU required. Use --csv with an existing log file.")
        sys.exit(1)

    import os

    device = "cuda"
    print(f"\n{'=' * 70}")
    print("GENERATING AUTOTUNER CSV LOG")
    print(f"{'=' * 70}")
    print(f"Matrix size: {matrix_size}x{matrix_size}")
    print(f"Autotuner: {os.environ.get('HELION_AUTOTUNER', 'default')}")
    print(f"Effort: {os.environ.get('HELION_AUTOTUNE_EFFORT', 'default')}")
    print(f"Output: {output_path}")
    print()

    from examples.matmul import matmul as original_matmul

    import helion

    matmul = helion.kernel(
        original_matmul.fn,
        autotune_log=output_path,
    )

    print(f"Creating {matrix_size}x{matrix_size} test matrices...")
    a = torch.randn(matrix_size, matrix_size, device=device, dtype=torch.float16)
    b = torch.randn(matrix_size, matrix_size, device=device, dtype=torch.float16)

    print("Running autotuning (this may take a few minutes)...")
    bound_kernel = matmul.bind((a, b))
    best_config = bound_kernel.autotune((a, b))

    print("\nAutotuning complete!")
    print(f"Best config found: {best_config}")

    result = bound_kernel.compile_config(best_config)(a, b)
    expected = a @ b
    torch.testing.assert_close(result, expected, rtol=1e-2, atol=1e-1)
    print("Result verified against PyTorch matmul")

    csv_path = Path(f"{output_path}")
    if csv_path.exists():
        records = load_autotune_log(csv_path)
        print(f"\nGenerated CSV with {len(records)} config evaluations")
        print(f"CSV file: {csv_path}")
        print(f"Log file: {output_path}.log")
    else:
        print(f"WARNING: CSV file not found at {csv_path}")

    return csv_path


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
11.60,11,2,started,,,"Config(block_sizes=[512, 64], num_warps=4, num_stages=2, indexing='block_ptr')"
13.10,11,2,ok,0.445000,1.50,"Config(block_sizes=[512, 64], num_warps=4, num_stages=2, indexing='block_ptr')"
13.20,12,2,started,,,"Config(block_sizes=[64, 512], num_warps=4, num_stages=2, indexing='block_ptr')"
14.70,12,2,ok,0.467000,1.50,"Config(block_sizes=[64, 512], num_warps=4, num_stages=2, indexing='block_ptr')"
14.80,13,3,started,,,"Config(block_sizes=[512, 128], num_warps=8, num_stages=2, indexing='block_ptr')"
16.60,13,3,ok,0.398000,1.80,"Config(block_sizes=[512, 128], num_warps=8, num_stages=2, indexing='block_ptr')"
16.70,14,3,started,,,"Config(block_sizes=[128, 512], num_warps=8, num_stages=2, indexing='block_ptr')"
18.50,14,3,ok,0.412000,1.80,"Config(block_sizes=[128, 512], num_warps=8, num_stages=2, indexing='block_ptr')"
18.60,15,3,started,,,"Config(block_sizes=[256, 256], num_warps=16, num_stages=3, indexing='tensor_descriptor')"
20.80,15,3,ok,0.356000,2.20,"Config(block_sizes=[256, 256], num_warps=16, num_stages=3, indexing='tensor_descriptor')"
20.90,16,3,started,,,"Config(block_sizes=[512, 256], num_warps=8, num_stages=2, indexing='block_ptr')"
23.30,16,3,ok,0.334000,2.40,"Config(block_sizes=[512, 256], num_warps=8, num_stages=2, indexing='block_ptr')"
23.40,17,4,started,,,"Config(block_sizes=[256, 512], num_warps=8, num_stages=2, indexing='block_ptr')"
25.80,17,4,ok,0.345000,2.40,"Config(block_sizes=[256, 512], num_warps=8, num_stages=2, indexing='block_ptr')"
25.90,18,4,started,,,"Config(block_sizes=[512, 512], num_warps=8, num_stages=3, indexing='tensor_descriptor')"
29.00,18,4,ok,0.312000,3.10,"Config(block_sizes=[512, 512], num_warps=8, num_stages=3, indexing='tensor_descriptor')"
29.10,19,4,started,,,"Config(block_sizes=[1024, 256], num_warps=8, num_stages=2, indexing='block_ptr')"
32.50,19,4,ok,0.298000,3.40,"Config(block_sizes=[1024, 256], num_warps=8, num_stages=2, indexing='block_ptr')"
32.60,20,4,started,,,"Config(block_sizes=[256, 1024], num_warps=8, num_stages=2, indexing='block_ptr')"
36.00,20,4,ok,0.305000,3.40,"Config(block_sizes=[256, 1024], num_warps=8, num_stages=2, indexing='block_ptr')"
"""


def create_sample_csv() -> Path:
    """Create a temporary CSV file with sample autotuner data."""
    tmpdir = tempfile.mkdtemp()
    csv_path = Path(tmpdir) / "sample_autotune_log.csv"
    csv_path.write_text(SAMPLE_AUTOTUNE_LOG)
    return csv_path


def demo_basic_proxy_model(csv_path: Path) -> None:
    """Demonstrate basic proxy model usage."""
    print("\n" + "=" * 70)
    print("1. BASIC PROXY MODEL DEMO")
    print("=" * 70)

    # Load and analyze the CSV log
    records = load_autotune_log(csv_path)
    print(f"\nLoaded {len(records)} config evaluations from CSV log")

    # Train performance proxy model
    print("\nTraining performance proxy model...")
    perf_model = PerformanceProxyModel.from_csv(csv_path)

    # Get best observed config
    best_config_str, best_perf = perf_model.get_best_observed()
    print("\nBest config in training data:")
    print(f"  Config: {best_config_str}")
    print(f"  Performance: {best_perf:.4f} ms")

    # Predict performance for some configs
    print("\nPredicting performance for new configs:")
    test_configs = [
        {"block_sizes": [64, 64], "num_warps": 4, "num_stages": 1},
        {"block_sizes": [128, 128], "num_warps": 8, "num_stages": 2},
        {"block_sizes": [256, 256], "num_warps": 8, "num_stages": 2},
        {"block_sizes": [512, 256], "num_warps": 8, "num_stages": 3},
    ]

    for cfg in test_configs:
        pred = perf_model.predict(cfg)
        print(f"  {cfg} -> {pred:.4f} ms")

    # Train compile time proxy model
    print("\nTraining compile time proxy model...")
    try:
        compile_model = CompileTimeProxyModel.from_csv(csv_path)
        print("Predicting compile times:")
        for cfg in test_configs[:2]:
            pred = compile_model.predict(cfg)
            print(f"  {cfg} -> {pred:.2f}s")
    except ValueError as e:
        print(f"  Skipped: {e}")


def demo_simulated_benchmark(csv_path: Path) -> None:
    """Demonstrate simulated benchmark environment."""
    print("\n" + "=" * 70)
    print("2. SIMULATED BENCHMARK DEMO")
    print("=" * 70)

    # Create simulated benchmark from CSV
    sim = SimulatedBenchmark.from_csv(csv_path)

    # Benchmark some configs
    print("\nSimulating benchmarks (no actual kernel execution):")
    configs = [
        {"block_sizes": [64, 64], "num_warps": 4},
        {"block_sizes": [128, 128], "num_warps": 8},
        {"block_sizes": [256, 256], "num_warps": 8},
    ]

    for cfg in configs:
        result = sim.benchmark(cfg)
        compile_str = (
            f", compile: {result.predicted_compile_time:.2f}s"
            if result.predicted_compile_time
            else ""
        )
        print(
            f"  Config #{result.config_index}: perf={result.predicted_perf:.4f} ms{compile_str}"
        )

    # Show metrics
    sim.print_search_quality_report()


def demo_search_comparison(
    csv_path: Path,
    algorithms: list[str] | None,
    n_configs: int,
    seed: int,
) -> None:
    """Demonstrate search algorithm comparison."""
    print("\n" + "=" * 70)
    print("3. SEARCH ALGORITHM COMPARISON")
    print("=" * 70)

    # Create search runner
    runner = SimulatedSearchRunner.from_csv(csv_path)

    # Use all available algorithms if none specified
    if algorithms is None:
        algorithms = runner.get_available_searches()

    print(f"\nComparing algorithms: {', '.join(algorithms)}")
    print(f"Configs per algorithm: {n_configs}")
    print(f"Random seed: {seed}")

    # Run comparison
    comparison = runner.compare_searches(algorithms, n_configs=n_configs, seed=seed)

    # Print detailed report
    runner.print_comparison_report(comparison)

    # Additional analysis
    print("Analysis:")
    print("-" * 40)

    # Find best algorithm and overall best performance
    best_algo = max(comparison.results.items(), key=lambda x: x[1].percent_of_best)
    overall_best = min(
        r.best_perf for r in comparison.results.values() if r.best_perf > 0
    )
    print(
        f"Best algorithm: {best_algo[0]} ({best_algo[1].percent_of_best:.1f}% of optimal)"
    )

    # Show how many configs needed to reach 90% of the overall optimal
    # (90% of overall best means: perf <= overall_best / 0.90)
    threshold_perf = overall_best / 0.90
    for name, result in comparison.results.items():
        for n, perf, _percent in result.metrics_at_n:
            if perf <= threshold_perf:
                actual_percent = (overall_best / perf) * 100 if perf > 0 else 0
                print(
                    f"{name}: reached 90% of optimal at {n} configs ({actual_percent:.1f}%)"
                )
                break
        else:
            print(
                f"{name}: did not reach 90% of optimal ({result.percent_of_best:.1f}%)"
            )


def demo_multiple_runs(csv_path: Path, n_runs: int = 5) -> None:
    """Show variance across multiple random runs."""
    print("\n" + "=" * 70)
    print("4. VARIANCE ANALYSIS (Multiple Random Seeds)")
    print("=" * 70)

    runner = SimulatedSearchRunner.from_csv(csv_path)

    # Use all available algorithms
    algorithms = runner.get_available_searches()
    results_by_algo: dict[str, list[float]] = {algo: [] for algo in algorithms}

    print(f"\nRunning {n_runs} trials per algorithm...")

    for seed in range(n_runs):
        comparison = runner.compare_searches(algorithms, n_configs=50, seed=seed)
        for algo, result in comparison.results.items():
            results_by_algo[algo].append(result.percent_of_best)

    print("\nResults (% of optimal):")
    print(f"{'Algorithm':<30} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("-" * 70)

    for algo in algorithms:
        values = results_by_algo[algo]
        mean = np.mean(values)
        std = np.std(values)
        print(
            f"{algo:<30} {mean:>9.1f}% {std:>9.1f}% {min(values):>9.1f}% {max(values):>9.1f}%"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate autotuner search algorithms using proxy models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # CSV generation options
    parser.add_argument(
        "--generate-csv",
        type=str,
        metavar="OUTPUT_PATH",
        help="Generate a CSV log by running actual autotuning (requires GPU). "
        "Provide base path, will create OUTPUT_PATH.csv and OUTPUT_PATH.log",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=512,
        help="Matrix size for CSV generation (default: 512)",
    )

    # Analysis options
    parser.add_argument(
        "--csv",
        type=Path,
        help="Path to autotuner CSV log file (uses sample data if not provided)",
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=None,
        help="Search algorithms to compare (default: all available)",
    )
    parser.add_argument(
        "--n-configs",
        type=int,
        default=100,
        help="Number of configs to evaluate per algorithm (default: 100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--variance-runs",
        type=int,
        default=0,
        help="Number of runs for variance analysis (0 to skip)",
    )
    parser.add_argument(
        "--skip-basic-demo",
        action="store_true",
        help="Skip basic proxy model demo",
    )
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="Only generate CSV, don't run analysis",
    )

    args = parser.parse_args()

    # Handle CSV generation
    if args.generate_csv:
        csv_path = generate_autotuner_csv(
            args.generate_csv,
            matrix_size=args.size,
        )
        if args.generate_only:
            print("\nCSV generation complete. Run with --csv to analyze.")
            return
        print("\nNow analyzing the generated CSV...")
    elif args.csv:
        if not args.csv.exists():
            print(f"ERROR: CSV file not found: {args.csv}")
            sys.exit(1)
        csv_path = args.csv
        print(f"Using CSV log: {csv_path}")
    else:
        csv_path = create_sample_csv()
        print(f"Using sample data (created at: {csv_path})")
        print("Tip: Use --generate-csv to create a CSV from real autotuning")

    # Run demos
    if not args.skip_basic_demo:
        demo_basic_proxy_model(csv_path)
        demo_simulated_benchmark(csv_path)

    demo_search_comparison(csv_path, args.algorithms, args.n_configs, args.seed)

    if args.variance_runs > 0:
        demo_multiple_runs(csv_path, args.variance_runs)

    print("\nDone!")


if __name__ == "__main__":
    main()
