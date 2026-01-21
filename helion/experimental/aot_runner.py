"""
AOT Autotuning Runner
=====================

Command-line tool for running the AOT autotuning workflow.

Usage:
    python -m helion.experimental.aot_runner python my_benchmark.py [options]

The workflow has three phases:
1. collect: Tune each shape individually, record (kernel, shape, config) triples
2. measure: Measure each shape with all observed configs
3. evaluate: Generate heuristics and validate performance goals

Each phase runs the benchmark with different HELION_AOT_MODE settings.

Each run creates a unique timestamped subdirectory to avoid overwriting previous data.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
import logging
import operator
import os
from pathlib import Path
import subprocess
import sys
from typing import Any
import uuid

from ..autotuner.aot_cache import get_hardware_info
from ..autotuner.heuristic_generator import PerformanceTarget
from ..autotuner.heuristic_generator import evaluate_heuristic
from ..autotuner.heuristic_generator import generate_heuristic

log: logging.Logger = logging.getLogger(__name__)


def generate_run_id() -> str:
    """Generate a unique run ID using timestamp and short UUID."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_uuid = uuid.uuid4().hex[:6]
    return f"{timestamp}_{short_uuid}"


@dataclass
class RunConfig:
    """Configuration for a benchmark run."""

    benchmark_cmd: list[str]
    output_dir: Path
    hardware_id: str
    run_id: str

    # Performance target
    goal_type: str = "max_slowdown"
    threshold: float = 1.1
    min_configs: int = 1
    max_configs: int = 10

    # Heuristic generation options
    backend: str = "decision_tree"
    feature_selection: bool = True  # Whether to prune redundant features
    print_score_matrix: bool = True  # Whether to print the score matrix
    dump_code: bool = False  # Whether to print generated code to stdout

    # Benchmark overrides per phase
    collect_benchmark: list[str] | None = None
    measure_benchmark: list[str] | None = None
    evaluate_benchmark: list[str] | None = None

    # Kernel filtering
    kernels: list[str] | None = None  # Filter which kernels to tune

    @property
    def run_dir(self) -> Path:
        """Get the unique directory for this run."""
        return self.output_dir / self.run_id

    @property
    def run_log_dir(self) -> Path:
        """Get the log directory for this run."""
        return self.run_dir / "logs"


def run_benchmark(
    cmd: list[str],
    env: dict[str, str],
    log_file: Path,
    phase: str,
    kernels: list[str] | None = None,
) -> tuple[int, str, str]:
    """
    Run a benchmark command with the given environment.

    Returns:
        Tuple of (return_code, stdout, stderr)
    """
    log.info(f"Running {phase} phase: {' '.join(cmd)}")
    log.info(f"Environment overrides: {env}")

    # Merge with current environment
    full_env = os.environ.copy()
    full_env.update(env)

    # Add kernel filter if specified
    if kernels:
        full_env["HELION_AOT_KERNELS"] = ",".join(kernels)

    # Open log file for output
    with open(log_file, "w") as f:
        f.write(f"# {phase} phase\n")
        f.write(f"# Command: {' '.join(cmd)}\n")
        f.write(f"# Environment: {json.dumps(env)}\n")
        f.write(f"# Started: {datetime.now().isoformat()}\n\n")

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=full_env,
            text=True,
        )

        stdout_lines: list[str] = []
        if process.stdout is not None:
            for line in process.stdout:
                f.write(line)
                f.flush()
                stdout_lines.append(line)
                # Also print to console
                print(line, end="")

        return_code = process.wait()
        stdout = "".join(stdout_lines)

        f.write(f"\n# Finished: {datetime.now().isoformat()}\n")
        f.write(f"# Return code: {return_code}\n")

    return return_code, stdout, ""


def run_collect_phase(config: RunConfig) -> bool:
    """
    Run the collect phase: tune each shape individually.

    Returns True if successful.
    """
    log.info("=" * 60)
    log.info("PHASE 1: Collecting tuned configs")
    log.info("=" * 60)

    cmd = config.collect_benchmark or config.benchmark_cmd
    log_file = config.run_log_dir / f"collect_{config.hardware_id}.log"

    env = {
        "HELION_AOT_MODE": "collect",
        "HELION_AOT_DATA_DIR": str(config.run_dir),
        "HELION_AUTOTUNE_CACHE": "AOTAutotuneCache",
    }

    return_code, _, _ = run_benchmark(cmd, env, log_file, "collect", config.kernels)

    if return_code != 0:
        log.error(f"Collect phase failed with return code {return_code}")
        return False

    # Check that we collected some configs
    configs_file = config.run_dir / f"tuned_configs_{config.hardware_id}.json"
    if not configs_file.exists():
        log.error("No configs were collected")
        return False

    data = json.loads(configs_file.read_text())
    total_configs = sum(len(v) for v in data.values())
    log.info(f"Collected {total_configs} configs for {len(data)} kernels")

    return True


def run_measure_phase(config: RunConfig) -> bool:
    """
    Run the measure phase: measure all configs across all shapes.

    Returns True if successful.
    """
    log.info("=" * 60)
    log.info("PHASE 2: Measuring configs across shapes")
    log.info("=" * 60)

    cmd = config.measure_benchmark or config.benchmark_cmd
    log_file = config.run_log_dir / f"measure_{config.hardware_id}.log"

    env = {
        "HELION_AOT_MODE": "measure",
        "HELION_AOT_DATA_DIR": str(config.run_dir),
        "HELION_AUTOTUNE_CACHE": "AOTAutotuneCache",
    }

    return_code, _, _ = run_benchmark(cmd, env, log_file, "measure", config.kernels)

    if return_code != 0:
        log.error(f"Measure phase failed with return code {return_code}")
        return False

    # Check that we have measurements
    measurements_file = config.run_dir / f"measurements_{config.hardware_id}.csv"
    if not measurements_file.exists():
        log.error("No measurements were recorded")
        return False

    # Count measurements
    with open(measurements_file) as f:
        num_measurements = sum(1 for _ in f) - 1  # Subtract header
    log.info(f"Recorded {num_measurements} measurements")

    return True


def run_build_heuristic_phase(config: RunConfig) -> bool:
    """
    Build heuristics from measurement data.

    Returns True if successful.
    """
    from ..autotuner.aot_cache import load_kernel_source_files

    log.info("=" * 60)
    log.info("PHASE 3: Building heuristics")
    log.info("=" * 60)

    measurements_file = config.run_dir / f"measurements_{config.hardware_id}.csv"

    target = PerformanceTarget(
        goal_type=config.goal_type,  # type: ignore[arg-type]
        threshold=config.threshold,
        min_configs=config.min_configs,
        max_configs=config.max_configs,
        backend=config.backend,  # type: ignore[arg-type]
        feature_selection=config.feature_selection,
        print_score_matrix=config.print_score_matrix,
        verbose=not config.dump_code,  # Quiet when dumping code
        skip_write=config.dump_code,  # Don't write files when dumping
    )

    # Load kernel source files from tuned configs
    kernel_source_files = load_kernel_source_files(config.run_dir, config.hardware_id)

    try:
        results = generate_heuristic(
            measurements_file=measurements_file,
            output_dir=config.run_dir,
            target=target,
            kernel_source_files=kernel_source_files,
        )

        # Dump generated code to stdout if requested
        if config.dump_code:
            for kernel_name, result in results.items():
                print(f"\n{'=' * 60}")
                print(f"# Generated heuristic for: {kernel_name}")
                print(f"# Backend: {result.backend_used}")
                print(f"# Accuracy: {result.model_accuracy:.2%}")
                print(f"{'=' * 60}\n")
                print(result.generated_code)

        # Save summary (skip when just dumping code)
        if not config.dump_code:
            summary: dict[str, Any] = {}
            for kernel_name, result in results.items():
                summary[kernel_name] = {
                    "num_configs": len(result.selected_configs),
                    "model_accuracy": result.model_accuracy,
                    "performance_stats": result.performance_stats,
                    "backend": result.backend_used,
                }

            summary_file = (
                config.run_dir / f"heuristic_summary_{config.hardware_id}.json"
            )
            summary_file.write_text(json.dumps(summary, indent=2))
            log.info(f"Saved heuristic summary to {summary_file}")

        return True

    except Exception:
        log.exception("Failed to build heuristics")
        return False


def run_evaluate_phase(config: RunConfig) -> bool:
    """
    Run the evaluate phase: validate performance using heuristics.

    Returns True if performance goals are met.
    """
    log.info("=" * 60)
    log.info("PHASE 4: Evaluating heuristics")
    log.info("=" * 60)

    # First evaluate against measurement data
    measurements_file = config.run_dir / f"measurements_{config.hardware_id}.csv"
    eval_results = evaluate_heuristic(
        measurements_file=measurements_file,
        heuristic_dir=config.run_dir,
    )

    # Check if performance goals are met
    all_passed = True
    for kernel_name, stats in eval_results.items():
        if config.goal_type == "max_slowdown":
            passed = stats["max_slowdown"] <= config.threshold
        elif config.goal_type == "geomean_slowdown":
            passed = stats["geomean_slowdown"] <= config.threshold
        else:
            passed = stats["avg_slowdown"] <= config.threshold

        status = "PASS" if passed else "FAIL"
        log.info(
            f"  {kernel_name}: {status} (max_slowdown={stats['max_slowdown']:.2f}x)"
        )

        if not passed:
            all_passed = False

    # Optionally run the benchmark in evaluate mode
    if config.evaluate_benchmark or config.benchmark_cmd:
        cmd = config.evaluate_benchmark or config.benchmark_cmd
        log_file = config.run_log_dir / f"evaluate_{config.hardware_id}.log"

        env = {
            "HELION_AOT_MODE": "evaluate",
            "HELION_AOT_DATA_DIR": str(config.run_dir),
            "HELION_AUTOTUNE_CACHE": "AOTAutotuneCache",
        }

        return_code, _, _ = run_benchmark(
            cmd, env, log_file, "evaluate", config.kernels
        )

        if return_code != 0:
            log.warning(f"Evaluate benchmark failed with return code {return_code}")

    # Save evaluation results
    eval_file = config.run_dir / f"evaluation_{config.hardware_id}.json"
    eval_file.write_text(json.dumps(eval_results, indent=2))
    log.info(f"Saved evaluation results to {eval_file}")

    return all_passed


def list_previous_runs(output_dir: Path) -> None:
    """List all previous runs in the output directory."""
    if not output_dir.exists():
        print(f"No runs found (directory {output_dir} does not exist)")
        return

    runs: list[tuple[str, dict[str, Any]]] = []
    for run_dir in output_dir.iterdir():
        if not run_dir.is_dir():
            continue
        meta_file = run_dir / "run_metadata.json"
        if meta_file.exists():
            try:
                meta = json.loads(meta_file.read_text())
                runs.append((run_dir.name, meta))
            except json.JSONDecodeError:
                runs.append((run_dir.name, {}))

    if not runs:
        print(f"No runs found in {output_dir}")
        return

    # Sort by run ID (which starts with timestamp)
    runs.sort(key=operator.itemgetter(0), reverse=True)

    print(f"Previous runs in {output_dir}:\n")
    print(f"{'Run ID':<30} {'Hardware':<30} {'Status':<10} {'Started':<20}")
    print("-" * 90)

    for run_id, meta in runs:
        hardware = meta.get("hardware_id", "unknown")[:28]
        status = "OK" if meta.get("success") else ("FAIL" if "success" in meta else "?")
        started = meta.get("started_at", "")[:19]
        print(f"{run_id:<30} {hardware:<30} {status:<10} {started:<20}")

    print("\nTo continue a run, use: --run-id <run_id> --phase <phase>")


def run_full_workflow(config: RunConfig) -> bool:
    """
    Run the full AOT autotuning workflow.

    Returns True if all phases succeed and performance goals are met.
    """
    log.info("Starting AOT autotuning workflow")
    log.info(f"Run ID: {config.run_id}")
    log.info(f"Run directory: {config.run_dir}")
    log.info(f"Hardware ID: {config.hardware_id}")
    log.info(f"Performance goal: {config.goal_type} <= {config.threshold}")

    # Phase 1: Collect
    if not run_collect_phase(config):
        log.error("Collect phase failed, aborting workflow")
        return False

    # Phase 2: Measure
    if not run_measure_phase(config):
        log.error("Measure phase failed, aborting workflow")
        return False

    # Phase 3: Build heuristics
    if not run_build_heuristic_phase(config):
        log.error("Heuristic building failed, aborting workflow")
        return False

    # Phase 4: Evaluate
    if not run_evaluate_phase(config):
        log.warning("Performance goals not fully met")
        return False

    log.info("=" * 60)
    log.info("AOT autotuning workflow completed successfully!")
    log.info("=" * 60)
    log.info("")
    log.info(
        "TIP: To use the generated heuristics automatically, use @helion.experimental.aot_kernel():"
    )
    log.info("")
    log.info("     @helion.experimental.aot_kernel()")
    log.info("     def my_kernel(...):")
    log.info("         ...")
    log.info("")
    return True


def main() -> None:
    """Main entry point for the AOT runner CLI."""
    parser = argparse.ArgumentParser(
        description="Helion AOT Autotuning Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full workflow with a benchmark script (use -- to separate runner args from benchmark)
  python -m helion.experimental.aot_runner -- python my_benchmark.py

  # Run only the collect phase
  python -m helion.experimental.aot_runner --phase collect -- python my_benchmark.py

  # Benchmark with its own arguments
  python -m helion.experimental.aot_runner --phase collect -- python benchmark.py --kernel softmax

  # Use different benchmarks for different phases
  python -m helion.experimental.aot_runner \\
    --collect-benchmark "python benchmark.py --full" \\
    --measure-benchmark "python benchmark.py --quick" \\
    -- python benchmark.py

  # Set performance target
  python -m helion.experimental.aot_runner --goal max_slowdown --threshold 1.05 -- python benchmark.py

  # Select only a single config (useful with batched kernels)
  python -m helion.experimental.aot_runner --single-config -- python benchmark.py

  # List previous runs
  python -m helion.experimental.aot_runner --list-runs

  # Continue a previous run (run individual phases)
  python -m helion.experimental.aot_runner --run-id 20241217_143022_abc123 --phase measure \\
    -- python benchmark.py

  # Alternative: use --benchmark with quoted command (no -- needed)
  python -m helion.experimental.aot_runner --benchmark "python my_benchmark.py --arg"
        """,
    )

    parser.add_argument(
        "benchmark_args",
        nargs=argparse.REMAINDER,
        help="Benchmark command to run after -- (e.g., -- python my_benchmark.py)",
    )

    parser.add_argument(
        "--benchmark",
        type=str,
        help="Benchmark command (alternative to positional args, e.g., 'python my_benchmark.py')",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=".helion_aot",
        help="Directory for AOT data files (default: .helion_aot)",
    )

    parser.add_argument(
        "--phase",
        type=str,
        choices=["collect", "measure", "build", "evaluate", "all"],
        default="all",
        help="Which phase to run (default: all)",
    )

    parser.add_argument(
        "--collect-benchmark",
        type=str,
        help="Override benchmark command for collect phase",
    )

    parser.add_argument(
        "--measure-benchmark",
        type=str,
        help="Override benchmark command for measure phase",
    )

    parser.add_argument(
        "--evaluate-benchmark",
        type=str,
        help="Override benchmark command for evaluate phase",
    )

    parser.add_argument(
        "--goal",
        type=str,
        choices=["max_slowdown", "geomean_slowdown", "avg_slowdown"],
        default="max_slowdown",
        help="Performance goal type (default: max_slowdown)",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=1.1,
        help="Performance threshold as slowdown factor (default: 1.1 = 10%% slowdown)",
    )

    parser.add_argument(
        "--max-configs",
        type=int,
        default=10,
        help="Maximum number of configs to select per kernel (default: 10)",
    )

    parser.add_argument(
        "--single-config",
        action="store_true",
        help="Select only a single config per kernel (equivalent to --max-configs 1). "
        "Useful when batch dimensions vary but non-batch dimensions are fixed.",
    )

    parser.add_argument(
        "--backend",
        type=str,
        choices=["decision_tree", "nearest_neighbor"],
        default="decision_tree",
        help="Heuristic generation backend (default: decision_tree). "
        "nearest_neighbor stores all training shapes and finds closest match at runtime.",
    )

    parser.add_argument(
        "--no-feature-selection",
        action="store_true",
        help="Disable automatic feature selection (pruning of redundant features)",
    )

    parser.add_argument(
        "--no-score-matrix",
        action="store_true",
        help="Disable printing of the score matrix during heuristic generation",
    )

    parser.add_argument(
        "--dump-code",
        action="store_true",
        help="Print generated heuristic code to stdout (build phase only)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    parser.add_argument(
        "--run-id",
        type=str,
        help="Specify a run ID to continue a previous run (for running individual phases). "
        "If not specified, a new unique run ID is generated.",
    )

    parser.add_argument(
        "--list-runs",
        action="store_true",
        help="List all previous runs in the output directory and exit.",
    )

    parser.add_argument(
        "--kernel",
        "-k",
        type=str,
        action="append",
        dest="kernels",
        help="Filter which kernel(s) to tune. Can be repeated for multiple kernels. "
        "Sets HELION_AOT_KERNELS env var for the benchmark subprocess.",
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    output_dir = Path(args.output_dir)

    # Handle --list-runs
    if args.list_runs:
        list_previous_runs(output_dir)
        sys.exit(0)

    # Parse benchmark command: prefer positional args (after --), fall back to --benchmark
    benchmark_args = args.benchmark_args
    # Strip leading '--' if present (argparse.REMAINDER captures it)
    if benchmark_args and benchmark_args[0] == "--":
        benchmark_args = benchmark_args[1:]

    if benchmark_args:
        benchmark_cmd = benchmark_args
    elif args.benchmark:
        benchmark_cmd = args.benchmark.split()
    else:
        parser.error(
            "benchmark command is required. Use -- to separate runner args from benchmark "
            "(e.g., -- python my_benchmark.py) or use --benchmark"
        )
        sys.exit(1)  # parser.error already exits, but this satisfies type checker

    # Generate or use provided run ID
    if args.run_id:
        run_id = args.run_id
        log.info(f"Continuing run: {run_id}")
    else:
        run_id = generate_run_id()
        log.info(f"Starting new run: {run_id}")

    # Handle --single-config flag
    max_configs = 1 if args.single_config else args.max_configs

    config = RunConfig(
        benchmark_cmd=benchmark_cmd,
        output_dir=output_dir,
        hardware_id=get_hardware_info().hardware_id,
        run_id=run_id,
        goal_type=args.goal,
        threshold=args.threshold,
        max_configs=max_configs,
        backend=args.backend,
        feature_selection=not args.no_feature_selection,
        print_score_matrix=not args.no_score_matrix,
        dump_code=args.dump_code,
        collect_benchmark=args.collect_benchmark.split()
        if args.collect_benchmark
        else None,
        measure_benchmark=args.measure_benchmark.split()
        if args.measure_benchmark
        else None,
        evaluate_benchmark=args.evaluate_benchmark.split()
        if args.evaluate_benchmark
        else None,
        kernels=args.kernels,
    )

    # Create directories
    output_dir.mkdir(parents=True, exist_ok=True)
    config.run_dir.mkdir(parents=True, exist_ok=True)
    config.run_log_dir.mkdir(parents=True, exist_ok=True)

    # Add file handler to capture all logging to run directory
    runner_log_file = config.run_log_dir / f"runner_{config.hardware_id}.log"
    file_handler = logging.FileHandler(runner_log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    )
    logging.getLogger().addHandler(file_handler)

    # Also capture helion autotuner logging
    helion_logger = logging.getLogger("helion")
    helion_logger.addHandler(file_handler)
    helion_logger.setLevel(log_level)

    log.info(f"Logging to: {runner_log_file}")

    # Save run metadata
    run_meta = {
        "run_id": run_id,
        "hardware_id": config.hardware_id,
        "benchmark_cmd": benchmark_cmd,
        "goal_type": config.goal_type,
        "threshold": config.threshold,
        "max_configs": config.max_configs,
        "started_at": datetime.now().isoformat(),
    }
    meta_file = config.run_dir / "run_metadata.json"
    meta_file.write_text(json.dumps(run_meta, indent=2))

    success = False

    if args.phase == "all":
        success = run_full_workflow(config)
    elif args.phase == "collect":
        success = run_collect_phase(config)
    elif args.phase == "measure":
        success = run_measure_phase(config)
    elif args.phase == "build":
        success = run_build_heuristic_phase(config)
    elif args.phase == "evaluate":
        success = run_evaluate_phase(config)

    # Update metadata with completion status
    run_meta["completed_at"] = datetime.now().isoformat()
    run_meta["success"] = success
    meta_file.write_text(json.dumps(run_meta, indent=2))

    log.info(f"Run directory: {config.run_dir}")
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
