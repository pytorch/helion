"""TPU/Pallas benchmark runner for Helion examples.

Runs selected Helion examples with autotuning on TPU and reports results
in the same JSON format as the GPU benchmark runner (benchmarks/run.py).

Usage:
    # Run all default kernels
    HELION_BACKEND=pallas python benchmarks/run_tpu.py

    # Run specific kernels
    HELION_BACKEND=pallas python benchmarks/run_tpu.py --kernel exp,add

    # Output results to JSON (compatible with pytorch benchmark hub)
    HELION_BACKEND=pallas python benchmarks/run_tpu.py --output results.json

    # List available kernels
    HELION_BACKEND=pallas python benchmarks/run_tpu.py --list-kernels
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from dataclasses import field
import functools
import importlib.util
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable

import torch

from helion._testing import DEVICE
from helion._testing import run_example

if TYPE_CHECKING:
    import types

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"


# Shape generators for multi-shape benchmarking.
# Each returns a list of (label, args_tuple) pairs.
def _exp_shapes() -> list[tuple[str, tuple[Any, ...]]]:
    sizes = [1024, 4096, 16384, 65536, 262144, 1048576]
    return [
        (
            f"[{n}]",
            (torch.randn(n, device=DEVICE, dtype=torch.float32, requires_grad=True),),
        )
        for n in sizes
    ]


def _add_shapes() -> list[tuple[str, tuple[Any, ...]]]:
    sizes = [(128, 128), (256, 256), (512, 512), (1024, 1024), (2048, 2048)]
    return [
        (
            f"[{m},{n}]",
            (
                torch.randn(m, n, device=DEVICE, dtype=torch.bfloat16),
                torch.randn(m, n, device=DEVICE, dtype=torch.bfloat16),
            ),
        )
        for m, n in sizes
    ]


def _softmax_shapes() -> list[tuple[str, tuple[Any, ...]]]:
    shapes = [(1024, 256), (1024, 512), (1024, 1024), (1024, 2048), (1024, 4096)]
    return [
        (
            f"[{m},{n}]",
            (torch.randn(m, n, device=DEVICE, dtype=torch.bfloat16),),
        )
        for m, n in shapes
    ]


# Kernel mappings for TPU/Pallas benchmarks.
# Format: kernel_name -> (module_file, kernel_fn_name, baseline_fn, shapes_fn)
#   module_file: filename in examples/ (without .py)
#   kernel_fn_name: attribute name of the helion kernel in the module
#   baseline_fn: callable that produces reference output (None = call main())
#   shapes_fn: callable returning list of (label, args) pairs (None = call main())
#
# This list contains only kernels that reliably pass on Pallas/TPU.
KernelMapping = tuple[
    str,
    str,
    Callable[..., Any] | None,
    Callable[[], list[tuple[str, tuple[Any, ...]]]] | None,
]
KERNEL_MAPPINGS: dict[str, KernelMapping] = {
    "exp": ("exp", "exp", torch.exp, _exp_shapes),
    "add": ("add", "add", torch.add, _add_shapes),
    "softmax_two_pass": (
        "softmax",
        "softmax_two_pass",
        functools.partial(torch.softmax, dim=-1),
        _softmax_shapes,
    ),
    "welford": ("welford", "welford", None, None),
    "layer_norm": ("layer_norm", "layer_norm", None, None),
}


@dataclass
class ShapeResult:
    shape: str
    passed: bool
    kernel_time_ms: float = 0.0
    baseline_time_ms: float = 0.0
    speedup: float = 0.0
    error: str | None = None


@dataclass
class KernelResult:
    name: str
    passed: bool
    kernel_time_ms: float = 0.0
    error: str | None = None
    shape_results: list[ShapeResult] = field(default_factory=list)


def import_example(module_file: str) -> types.ModuleType:
    """Import an example module by filename."""
    module_path = EXAMPLES_DIR / f"{module_file}.py"
    spec = importlib.util.spec_from_file_location(
        f"examples.{module_file}", module_path
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


KERNEL_TIMEOUT = int(os.environ.get("HELION_BENCHMARK_KERNEL_TIMEOUT", "1200"))
NUM_SHAPES: int | None = None  # Set from CLI; None means all shapes


def _run_kernel_impl(name: str, result_queue: multiprocessing.Queue) -> None:  # type: ignore[type-arg]
    """Run a single kernel in a subprocess (target for multiprocessing)."""
    try:
        result_queue.put(run_kernel_inner(name))
    except Exception as e:
        result_queue.put(KernelResult(name=name, passed=False, error=str(e)))


def run_kernel(name: str) -> KernelResult:
    """Run a single kernel benchmark with a timeout."""
    queue: multiprocessing.Queue = multiprocessing.Queue()  # type: ignore[type-arg]
    proc = multiprocessing.Process(target=_run_kernel_impl, args=(name, queue))
    proc.start()
    proc.join(timeout=KERNEL_TIMEOUT)
    if proc.is_alive():
        proc.kill()
        proc.join()
        return KernelResult(
            name=name,
            passed=False,
            error=f"Timed out after {KERNEL_TIMEOUT}s",
        )
    if not queue.empty():
        return queue.get()
    return KernelResult(
        name=name, passed=False, error="Kernel process exited unexpectedly"
    )


def run_kernel_inner(name: str) -> KernelResult:
    """Run a single kernel benchmark: accuracy check + timing vs baseline."""
    if name not in KERNEL_MAPPINGS:
        return KernelResult(name=name, passed=False, error=f"Unknown kernel: {name}")

    module_file, kernel_fn_name, baseline_fn, shapes_fn = KERNEL_MAPPINGS[name]

    try:
        mod = import_example(module_file)
        kernel_fn = getattr(mod, kernel_fn_name)

        # For kernels with None baseline/shapes, call main() directly
        # (they have complex setup that's hard to replicate here)
        if baseline_fn is None or shapes_fn is None:
            start = time.perf_counter()
            mod.main()
            elapsed = time.perf_counter() - start
            return KernelResult(
                name=name,
                passed=True,
                kernel_time_ms=elapsed * 1000,
            )

        shapes = shapes_fn()
        if NUM_SHAPES is not None:
            shapes = shapes[:NUM_SHAPES]
        all_passed = True
        shape_results: list[ShapeResult] = []

        for label, args in shapes:
            print(f"  Shape {label}:", file=sys.stderr)
            try:
                timings = run_example(kernel_fn, baseline_fn, args)
                kernel_ms = timings.get("helion", 0.0)
                baseline_ms = timings.get("torch", 0.0)
                speedup = baseline_ms / kernel_ms if kernel_ms > 0 else 0.0
                shape_results.append(
                    ShapeResult(
                        shape=label,
                        passed=True,
                        kernel_time_ms=kernel_ms,
                        baseline_time_ms=baseline_ms,
                        speedup=speedup,
                    )
                )
            except Exception as e:
                print(f"    FAIL: {e}", file=sys.stderr)
                shape_results.append(
                    ShapeResult(shape=label, passed=False, error=str(e))
                )
                all_passed = False

        return KernelResult(name=name, passed=all_passed, shape_results=shape_results)

    except Exception as e:
        return KernelResult(name=name, passed=False, error=str(e))


def write_results_json(output: str, results: list[KernelResult]) -> None:
    """Write results in the same JSON format as benchmarks/run.py for pytorch benchmark hub."""
    device = os.environ.get("HELION_BACKEND", "pallas")
    records: list[dict[str, Any]] = []
    for result in results:
        if result.shape_results:
            for sr in result.shape_results:
                records.append(
                    {
                        "benchmark": {
                            "name": "Helion TPU Benchmark",
                            "extra_info": {"device": device},
                        },
                        "model": {"name": result.name},
                        "metric": {
                            "name": "accuracy",
                            "benchmark_values": [1.0 if sr.passed else 0.0],
                        },
                        "shape": [sr.shape],
                    }
                )
        else:
            records.append(
                {
                    "benchmark": {
                        "name": "Helion TPU Benchmark",
                        "extra_info": {"device": device},
                    },
                    "model": {"name": result.name},
                    "metric": {
                        "name": "accuracy",
                        "benchmark_values": [1.0 if result.passed else 0.0],
                    },
                    "shape": [],
                }
            )
        if result.kernel_time_ms > 0:
            records.append(
                {
                    "benchmark": {
                        "name": "Helion TPU Benchmark",
                        "extra_info": {"device": device},
                    },
                    "model": {"name": result.name},
                    "metric": {
                        "name": "kernel_time_ms",
                        "benchmark_values": [result.kernel_time_ms],
                    },
                    "shape": [],
                }
            )

    if os.path.exists(output):
        try:
            with open(output) as f:
                existing = json.load(f)
                if isinstance(existing, list):
                    records = existing + records
        except (OSError, json.JSONDecodeError):
            pass

    with open(output, "w") as f:
        json.dump(records, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TPU/Pallas benchmark runner for Helion examples",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--kernel",
        "--op",
        type=str,
        dest="kernel",
        help="Comma-separated list of kernels to run. If not specified, runs all kernels.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path (compatible with pytorch benchmark hub)",
    )
    parser.add_argument(
        "--num-shapes",
        type=int,
        default=None,
        help="Max number of shapes to benchmark per kernel (default: all)",
    )
    parser.add_argument(
        "--list-kernels",
        action="store_true",
        help="List available kernel names and exit",
    )
    args = parser.parse_args()

    global NUM_SHAPES
    NUM_SHAPES = args.num_shapes

    if args.list_kernels:
        for name in KERNEL_MAPPINGS:
            print(name)
        return

    if args.kernel:
        kernel_names = [k.strip() for k in args.kernel.split(",") if k.strip()]
        # Validate
        for name in kernel_names:
            if name not in KERNEL_MAPPINGS:
                print(f"Error: Unknown kernel '{name}'", file=sys.stderr)
                print(
                    f"Available kernels: {', '.join(KERNEL_MAPPINGS.keys())}",
                    file=sys.stderr,
                )
                sys.exit(1)
    else:
        kernel_names = list(KERNEL_MAPPINGS.keys())

    print(
        f"Running {len(kernel_names)} TPU kernels: {', '.join(kernel_names)}",
        file=sys.stderr,
    )
    print(
        f"HELION_BACKEND={os.environ.get('HELION_BACKEND', '(not set)')}",
        file=sys.stderr,
    )
    print("=" * 65, file=sys.stderr)

    results: list[KernelResult] = []
    for name in kernel_names:
        print(f"\n{'=' * 65}", file=sys.stderr)
        print(f"Kernel: {name}", file=sys.stderr)
        print(f"{'=' * 65}", file=sys.stderr)
        result = run_kernel(name)
        results.append(result)

        status = "PASS" if result.passed else "FAIL"
        print(f"  Status: {status}", file=sys.stderr)
        if result.error:
            print(f"  Error: {result.error}", file=sys.stderr)
        if result.shape_results:
            for sr in result.shape_results:
                sr_status = "PASS" if sr.passed else "FAIL"
                print(f"    {sr.shape}: {sr_status}", file=sys.stderr)

    # Summary table
    print(f"\n{'=' * 75}", file=sys.stderr)
    print("Summary", file=sys.stderr)
    print(f"{'=' * 75}", file=sys.stderr)
    print(
        f"{'Kernel':<22} {'Shape':<16} {'Status':<8} {'Helion (ms)':<14} {'Torch (ms)':<14} {'Speedup':<10}",
        file=sys.stderr,
    )
    print(f"{'-' * 75}", file=sys.stderr)
    for result in results:
        if result.shape_results:
            for sr in result.shape_results:
                status = "PASS" if sr.passed else "FAIL"
                kernel_str = (
                    f"{sr.kernel_time_ms:.4f}" if sr.kernel_time_ms > 0 else "-"
                )
                baseline_str = (
                    f"{sr.baseline_time_ms:.4f}" if sr.baseline_time_ms > 0 else "-"
                )
                speedup_str = f"{sr.speedup:.2f}x" if sr.speedup > 0 else "-"
                print(
                    f"{result.name:<22} {sr.shape:<16} {status:<8} {kernel_str:<14} {baseline_str:<14} {speedup_str:<10}",
                    file=sys.stderr,
                )
        else:
            status = "PASS" if result.passed else "FAIL"
            time_str = (
                f"{result.kernel_time_ms:.1f}" if result.kernel_time_ms > 0 else "-"
            )
            print(
                f"{result.name:<22} {'main()':<16} {status:<8} {time_str:<14} {'-':<14} {'-':<10}",
                file=sys.stderr,
            )

    passed = sum(1 for r in results if r.passed)
    total = len(results)
    print(f"{'-' * 75}", file=sys.stderr)
    print(f"Total: {passed}/{total} passed", file=sys.stderr)
    print(f"{'=' * 75}\n", file=sys.stderr)

    if args.output:
        write_results_json(args.output, results)
        print(f"Results written to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
