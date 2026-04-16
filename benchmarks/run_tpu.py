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
import os
from pathlib import Path
import signal
import sys
import time
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable

import torch
from torch import nn

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


def _welford_shapes() -> list[tuple[str, tuple[Any, ...]]]:
    configs = [(262144, 1024), (262144, 1536), (262144, 2048)]
    return [
        (
            f"[{s},{d}]",
            (
                torch.rand(d, device=DEVICE, dtype=torch.float32),
                torch.rand(d, device=DEVICE, dtype=torch.float32),
                torch.rand(s, d, device=DEVICE, dtype=torch.float32),
            ),
        )
        for s, d in configs
    ]


def _welford_baseline(
    weight: torch.Tensor, bias: torch.Tensor, x: torch.Tensor, eps: float = 1e-05
) -> torch.Tensor:
    return torch.nn.functional.layer_norm(
        x, normalized_shape=[x.shape[-1]], weight=weight, bias=bias, eps=eps
    )


def _attention_shapes() -> list[tuple[str, tuple[Any, ...]]]:
    configs = [
        (1, 4, 256, 64),
        (1, 4, 512, 64),
        (1, 4, 1024, 64),
        (2, 8, 512, 64),
    ]
    return [
        (
            f"[{z},{h},{n},{d}]",
            tuple(
                torch.randn(z, h, n, d, device=DEVICE, dtype=torch.float32)
                for _ in range(3)
            ),
        )
        for z, h, n, d in configs
    ]


def _bmm_shapes() -> list[tuple[str, tuple[Any, ...]]]:
    configs = [(16, 512, 768, 1024), (8, 256, 512, 256), (4, 1024, 512, 512)]
    return [
        (
            f"[{b},{m},{k},{n}]",
            (
                torch.randn(b, m, k, device=DEVICE, dtype=torch.bfloat16),
                torch.randn(b, k, n, device=DEVICE, dtype=torch.bfloat16),
            ),
        )
        for b, m, k, n in configs
    ]


def _broadcast_matmul_shapes() -> list[tuple[str, tuple[Any, ...]]]:
    configs = [(16, 512, 768, 1024), (8, 256, 512, 256), (4, 1024, 512, 512)]
    return [
        (
            f"[{b},{m},{k},{n}]",
            (
                torch.randn(b, m, k, device=DEVICE, dtype=torch.bfloat16),
                torch.randn(k, n, device=DEVICE, dtype=torch.bfloat16),
            ),
        )
        for b, m, k, n in configs
    ]


def _geglu_shapes() -> list[tuple[str, tuple[Any, ...]]]:
    shapes = [(1024, 512), (2048, 1024), (4096, 2048)]
    return [
        (
            f"[{m},{n}]",
            (
                torch.randn(m, n, device=DEVICE, dtype=torch.bfloat16),
                torch.randn(m, n, device=DEVICE, dtype=torch.bfloat16),
            ),
        )
        for m, n in shapes
    ]


def _geglu_baseline(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return nn.functional.gelu(a, approximate="tanh").to(b.dtype) * b


def _swiglu_shapes() -> list[tuple[str, tuple[Any, ...]]]:
    shapes = [(1024, 512), (2048, 1024), (4096, 2048)]
    return [
        (
            f"[{m},{n}]",
            (
                torch.randn(m, n, device=DEVICE, dtype=torch.bfloat16),
                torch.randn(m, n, device=DEVICE, dtype=torch.bfloat16),
            ),
        )
        for m, n in shapes
    ]


def _low_mem_dropout_shapes() -> list[tuple[str, tuple[Any, ...]]]:
    sizes = [4096, 16384, 65536, 262144]
    return [
        (
            f"[{n}]",
            (0.25, torch.randn(n, device=DEVICE, dtype=torch.float32), 123),
        )
        for n in sizes
    ]


def _low_mem_dropout_baseline(p: float, x: torch.Tensor, seed: int) -> torch.Tensor:
    return nn.functional.dropout(x, p=p, training=True)


def _swiglu_baseline(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return nn.functional.silu(a).to(b.dtype) * b


def _batch_softmax_shapes() -> list[tuple[str, tuple[Any, ...]]]:
    configs = [(16, 512, 1024), (8, 256, 2048), (4, 1024, 512)]
    return [
        (
            f"[{b},{m},{n}]",
            (torch.randn(b, m, n, device=DEVICE, dtype=torch.bfloat16),),
        )
        for b, m, n in configs
    ]


def _rms_norm_shapes() -> list[tuple[str, tuple[Any, ...]]]:
    configs = [(2048, 4096), (2048, 8192), (4096, 4096)]
    return [
        (
            f"[{m},{n}]",
            (
                torch.randn(m, n, device=DEVICE, dtype=torch.bfloat16),
                torch.randn(n, device=DEVICE, dtype=torch.bfloat16),
            ),
        )
        for m, n in configs
    ]


def _rms_norm_baseline(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    hidden = x.to(torch.float32)
    variance = hidden.pow(2).mean(-1, keepdim=True)
    return (weight * (hidden * torch.rsqrt(variance + 1e-5))).to(x.dtype)


def _sum_shapes() -> list[tuple[str, tuple[Any, ...]]]:
    shapes = [(5120, 2560), (10240, 10240), (2048, 8192)]
    return [
        (
            f"[{m},{n}]",
            (torch.randn(m, n, device=DEVICE, dtype=torch.bfloat16),),
        )
        for m, n in shapes
    ]


# Kernel mappings for TPU/Pallas benchmarks.
# Format: kernel_name -> (module_file, kernel_fn_name, baseline_fn, shapes_fn,
#                         max_mismatch_pct)
#   module_file: filename in examples/ (without .py)
#   kernel_fn_name: attribute name of the helion kernel in the module
#   baseline_fn: callable that produces reference output (None = call main())
#   shapes_fn: callable returning list of (label, args) pairs (None = call main())
#   max_mismatch_pct: fraction of elements allowed to mismatch (default None = strict)
#
# This list contains only kernels that reliably pass on Pallas/TPU.
KernelMapping = tuple[
    str,
    str,
    Callable[..., Any] | None,
    Callable[[], list[tuple[str, tuple[Any, ...]]]] | None,
    float | None,
]
KERNEL_MAPPINGS: dict[str, KernelMapping] = {
    "exp": ("exp", "exp", torch.exp, _exp_shapes, None),
    "add": ("add", "add", torch.add, _add_shapes, None),
    "softmax_two_pass": (
        "softmax",
        "softmax_two_pass",
        functools.partial(torch.softmax, dim=-1),
        _softmax_shapes,
        None,
    ),
    "welford": ("welford", "welford", _welford_baseline, _welford_shapes, None),
    "attention": (
        "attention",
        "attention",
        torch.nn.functional.scaled_dot_product_attention,
        _attention_shapes,
        None,
    ),
    "bmm": ("bmm", "bmm", torch.bmm, _bmm_shapes, None),
    "broadcast_matmul": (
        "broadcast_matmul",
        "broadcast_matmul",
        torch.matmul,
        _broadcast_matmul_shapes,
        None,
    ),
    "geglu": ("geglu", "geglu", _geglu_baseline, _geglu_shapes, None),
    # low_mem_dropout: helion uses a deterministic seed-based mask while
    # torch.nn.functional.dropout uses a random mask, so outputs always differ.
    # Allow 100% mismatch to skip correctness but still benchmark both.
    "low_mem_dropout": (
        "low_mem_dropout",
        "low_mem_dropout",
        _low_mem_dropout_baseline,
        _low_mem_dropout_shapes,
        1.0,
    ),
    "swiglu": ("swiglu", "swiglu_fwd", _swiglu_baseline, _swiglu_shapes, None),
    "batch_softmax": (
        "batch_softmax",
        "batch_softmax",
        functools.partial(torch.softmax, dim=-1),
        _batch_softmax_shapes,
        None,
    ),
    "sum": (
        "sum",
        "sum_kernel",
        functools.partial(torch.sum, dim=-1),
        _sum_shapes,
        None,
    ),
    # rms_norm_fwd returns (output, inv_rms); the _unwrap_first wrapper in
    # run_kernel_inner extracts just the first element for run_example.
    "rms_norm": (
        "rms_norm",
        "rms_norm_fwd",
        _rms_norm_baseline,
        _rms_norm_shapes,
        None,
    ),
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


class _KernelTimeout(Exception):
    """Raised by SIGALRM when a kernel exceeds its timeout."""


def _alarm_handler(signum: int, frame: object) -> None:
    raise _KernelTimeout


def run_kernel(name: str) -> KernelResult:
    """Run a single kernel benchmark with a signal-based timeout.

    Uses SIGALRM instead of multiprocessing to avoid fork-after-TPU-init
    deadlocks on Linux.
    """
    old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(KERNEL_TIMEOUT)
    try:
        return run_kernel_inner(name)
    except _KernelTimeout:
        return KernelResult(
            name=name,
            passed=False,
            error=f"Timed out after {KERNEL_TIMEOUT}s",
        )
    except Exception as e:
        return KernelResult(name=name, passed=False, error=str(e))
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def run_kernel_inner(name: str) -> KernelResult:
    """Run a single kernel benchmark: accuracy check + timing vs baseline."""
    if name not in KERNEL_MAPPINGS:
        return KernelResult(name=name, passed=False, error=f"Unknown kernel: {name}")

    module_file, kernel_fn_name, baseline_fn, shapes_fn, max_mismatch_pct = (
        KERNEL_MAPPINGS[name]
    )

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

        # Wrap kernel functions that return tuples so run_example sees a single tensor
        def _unwrap_first(fn: Callable[..., Any]) -> Callable[..., torch.Tensor]:
            @functools.wraps(fn)
            def wrapper(*a: object) -> torch.Tensor:
                result = fn(*a)
                return result[0] if isinstance(result, tuple) else result

            return wrapper

        kernel_fn = _unwrap_first(kernel_fn)

        for label, args in shapes:
            print(f"  Shape {label}:", file=sys.stderr)
            try:
                timings = run_example(
                    kernel_fn, baseline_fn, args, max_mismatch_pct=max_mismatch_pct
                )
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
