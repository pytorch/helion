#!/usr/bin/env python3
"""
LLM-Guided Autotuner MVP for Helion.

Compares three strategies across multiple kernels from examples/:
  1. Default autotuner  — Helion's built-in search
  2. Random baseline    — pick K random configs and benchmark them
  3. LLM-guided         — LLM ranks N candidates, benchmark only the top K

Usage:
    # Test all supported kernels:
    python autotuner_mvp.py --kernel all

    # Test a specific kernel:
    python autotuner_mvp.py --kernel softmax
    python autotuner_mvp.py --kernel matmul
    python autotuner_mvp.py --kernel layer_norm
    python autotuner_mvp.py --kernel add

    # Dry-run (skip LLM call):
    python autotuner_mvp.py --kernel softmax --dry-run

Environment variables:
    OPENAI_API_KEY         — API key
    OPENAI_BASE_URL        — custom endpoint (optional)
    AUTOTUNER_LLM_MODEL    — override model name (default: gpt-5-mini-2025-08-07)
"""

from __future__ import annotations

import argparse
import datetime
import functools
import json
import logging
import os
import random
import sys
import time

import torch

# Add the repo root so we can import from examples/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# Stub helion._testing before importing examples (it requires pytest which may not be installed)
import types
if "helion._testing" not in sys.modules:
    _stub = types.ModuleType("helion._testing")
    _stub.DEVICE = "cuda"  # type: ignore[attr-defined]
    _stub.HALF_DTYPE = torch.float16  # type: ignore[attr-defined]
    _stub.run_example = lambda *a, **kw: None  # type: ignore[attr-defined]
    sys.modules["helion._testing"] = _stub

from helion.autotuner.benchmarking import do_bench
from helion.runtime.config import Config

from llm_ranker import llm_rank_configs

log = logging.getLogger("autotuner_mvp")


# ---------------------------------------------------------------------------
# Kernel registry — each entry defines how to set up and run a kernel
# ---------------------------------------------------------------------------

def _make_add_entry(m: int, n: int, device: str):
    from examples.add import add as add_kernel

    x = torch.randn(m, n, device=device, dtype=torch.float32)
    y = torch.randn(m, n, device=device, dtype=torch.float32)
    return {
        "kernel": add_kernel,
        "args": (x, y),
        "features": {
            "kernel_type": "elementwise",
            "operation": "add",
            "dtype": "float32",
            "shape": [m, n],
            "numel": m * n,
            "memory_pattern": "contiguous",
            "reduction": False,
            "memory_bound": True,
            "bytes_accessed": m * n * 4 * 3,
        },
    }


def _make_softmax_entry(m: int, n: int, device: str):
    from examples.softmax import softmax

    x = torch.randn(m, n, device=device, dtype=torch.float16)
    return {
        "kernel": softmax,
        "args": (x,),
        "features": {
            "kernel_type": "softmax",
            "operation": "row-wise softmax",
            "dtype": "float16",
            "shape": [m, n],
            "rows": m,
            "cols": n,
            "memory_pattern": "row_contiguous",
            "reduction": True,
            "reduction_dim": 1,
            "memory_bound": n < 4096,
            "compute_bound": n >= 4096,
            "bytes_accessed": m * n * 2 * 2,
        },
    }


def _make_matmul_entry(m: int, k: int, n: int, device: str):
    from examples.matmul import matmul

    x = torch.randn(m, k, device=device, dtype=torch.float16)
    y = torch.randn(k, n, device=device, dtype=torch.float16)
    return {
        "kernel": matmul,
        "args": (x, y),
        "features": {
            "kernel_type": "matmul",
            "operation": "matrix multiplication",
            "dtype": "float16",
            "shape_x": [m, k],
            "shape_y": [k, n],
            "shape_out": [m, n],
            "M": m,
            "K": k,
            "N": n,
            "memory_pattern": "tiled_2d",
            "reduction": True,
            "reduction_dim": "k",
            "compute_bound": True,
            "flops": 2 * m * k * n,
        },
    }


def _make_layer_norm_entry(m: int, n: int, device: str):
    from examples.layer_norm import layer_norm_fwd

    x = torch.randn(m, n, device=device, dtype=torch.float16)
    weight = torch.randn(n, device=device, dtype=torch.float16)
    bias = torch.randn(n, device=device, dtype=torch.float16)
    return {
        "kernel": layer_norm_fwd,
        "args": (x, [n], weight, bias, 1e-5),
        "features": {
            "kernel_type": "layer_norm",
            "operation": "layer normalization forward",
            "dtype": "float16",
            "shape": [m, n],
            "batch_size": m,
            "normalized_dim": n,
            "memory_pattern": "row_contiguous",
            "reduction": True,
            "reduction_dim": 1,
            "has_affine": True,
            "memory_bound": True,
            "bytes_accessed": m * n * 2 * 4,  # x, weight, bias reads + output write
        },
    }


KERNEL_FACTORIES = {
    "add": lambda args: _make_add_entry(args.m, args.n, "cuda"),
    "softmax": lambda args: _make_softmax_entry(args.m, args.n, "cuda"),
    "matmul": lambda args: _make_matmul_entry(args.m, args.k, args.n, "cuda"),
    "layer_norm": lambda args: _make_layer_norm_entry(args.m, args.n, "cuda"),
}


# ---------------------------------------------------------------------------
# Benchmarking helpers (kernel-agnostic)
# ---------------------------------------------------------------------------

def benchmark_config(
    kernel, args: tuple, config: Config,
) -> float | None:
    """Compile kernel with config and return median latency in ms."""
    try:
        bound = kernel.bind(args)
        fn = bound.compile_config(config, allow_print=False)
        torch.cuda.synchronize()
        fn(*args)
        torch.cuda.synchronize()
        latency = do_bench(functools.partial(fn, *args), warmup=1, rep=50, return_mode="median")
        return float(latency)
    except Exception as e:
        log.debug("Config failed: %s — %s", config, e)
        return None


def benchmark_configs(
    kernel, args: tuple, configs: list[Config], label: str,
) -> tuple[Config | None, float, int]:
    """Benchmark a list of configs. Returns (best_config, best_latency_ms, num_evaluated)."""
    best_config: Config | None = None
    best_latency = float("inf")
    evaluated = 0

    for i, cfg in enumerate(configs):
        latency = benchmark_config(kernel, args, cfg)
        if latency is None:
            log.info("  [%s] config %d: FAILED", label, i)
            continue
        evaluated += 1
        log.info("  [%s] config %d: %.4f ms  %s", label, i, latency, dict(cfg))
        if latency < best_latency:
            best_latency = latency
            best_config = cfg

    return best_config, best_latency, evaluated


def generate_configs(kernel, args: tuple, count: int) -> list[Config]:
    """Generate candidate configs using Helion's config generation API."""
    bound = kernel.bind(args)
    config_gen = bound.config_spec.create_config_generation()
    return config_gen.random_population(count)


# ---------------------------------------------------------------------------
# Strategy runners (kernel-agnostic)
# ---------------------------------------------------------------------------

def run_default_autotuner(
    kernel, args: tuple,
) -> tuple[Config, float, float]:
    """Run Helion's default autotuner."""
    log.info("=== Strategy 1: Default Helion autotuner ===")
    kernel.reset()
    t0 = time.perf_counter()
    best_config = kernel.autotune(args, force=True)
    search_time = time.perf_counter() - t0

    latency = benchmark_config(kernel, args, best_config)
    assert latency is not None
    log.info("Default autotuner: %.4f ms (search took %.1f s)", latency, search_time)
    return best_config, latency, search_time


def run_random_baseline(
    kernel, args: tuple, features: dict,
    n_candidates: int, top_k: int,
) -> tuple[Config | None, float, float, int]:
    """Pick top_k random configs and benchmark them."""
    log.info("=== Strategy 2: Random baseline (pick %d of %d) ===", top_k, n_candidates)
    kernel.reset()
    t0 = time.perf_counter()
    candidates = generate_configs(kernel, args, count=n_candidates)
    chosen = random.sample(candidates, min(top_k, len(candidates)))
    best_config, best_latency, evaluated = benchmark_configs(kernel, args, chosen, "random")
    search_time = time.perf_counter() - t0
    log.info(
        "Random baseline: %.4f ms (evaluated %d, search %.1f s)",
        best_latency, evaluated, search_time,
    )
    return best_config, best_latency, search_time, evaluated


def run_llm_guided(
    kernel, args: tuple, features: dict,
    n_candidates: int, top_k: int,
    *, dry_run: bool = False,
) -> tuple[Config | None, float, float, int]:
    """LLM ranks n_candidates configs, benchmark top_k."""
    log.info(
        "=== Strategy 3: LLM-guided (rank %d, benchmark top %d) ===",
        n_candidates, top_k,
    )
    kernel.reset()
    t0 = time.perf_counter()
    candidates = generate_configs(kernel, args, count=n_candidates)

    if dry_run:
        log.info("  [dry-run] Skipping LLM call — using random shuffle as stand-in")
        ranked = list(candidates)
        random.shuffle(ranked)
    else:
        ranked = llm_rank_configs(candidates, features)

    top = ranked[:top_k]
    best_config, best_latency, evaluated = benchmark_configs(kernel, args, top, "llm")
    search_time = time.perf_counter() - t0
    log.info(
        "LLM-guided: %.4f ms (evaluated %d of %d, search %.1f s)",
        best_latency, evaluated, n_candidates, search_time,
    )
    return best_config, best_latency, search_time, evaluated


# ---------------------------------------------------------------------------
# Run one kernel
# ---------------------------------------------------------------------------

def run_kernel(
    kernel_name: str, entry: dict, args_ns: argparse.Namespace, timestamp: str,
) -> dict[str, dict[str, object]]:
    """Run all strategies for a single kernel and return results dict."""
    kernel = entry["kernel"]
    kargs = entry["args"]
    features = entry["features"]

    log.info("=" * 70)
    log.info("KERNEL: %s", kernel_name)
    log.info("Features: %s", json.dumps(features, default=str))
    log.info("=" * 70)

    results: dict[str, dict[str, object]] = {}

    # --- Strategy 1: Default autotuner ---
    if not args_ns.skip_default:
        cfg1, lat1, time1 = run_default_autotuner(kernel, kargs)
        results["default_autotuner"] = {
            "best_latency_ms": lat1,
            "search_time_s": time1,
            "config": dict(cfg1),
        }

    # --- Strategy 2: Random baseline ---
    cfg2, lat2, time2, eval2 = run_random_baseline(
        kernel, kargs, features, args_ns.candidates, args_ns.top_k,
    )
    results["random_baseline"] = {
        "best_latency_ms": lat2,
        "search_time_s": time2,
        "configs_evaluated": eval2,
        "config": dict(cfg2) if cfg2 else None,
    }

    # --- Strategy 3: LLM-guided ---
    cfg3, lat3, time3, eval3 = run_llm_guided(
        kernel, kargs, features,
        args_ns.candidates, args_ns.top_k, dry_run=args_ns.dry_run,
    )
    results["llm_guided"] = {
        "best_latency_ms": lat3,
        "search_time_s": time3,
        "configs_evaluated": eval3,
        "config": dict(cfg3) if cfg3 else None,
    }

    # --- Print summary ---
    print(f"\n{'=' * 70}")
    print(f"RESULTS: {kernel_name}")
    print("=" * 70)
    for name, res in results.items():
        print(f"\n{name}:")
        for k, v in res.items():
            if k == "config":
                continue
            print(f"  {k}: {v}")

    if "default_autotuner" in results and "llm_guided" in results:
        default_lat = results["default_autotuner"]["best_latency_ms"]
        llm_lat = results["llm_guided"]["best_latency_ms"]
        assert isinstance(default_lat, float) and isinstance(llm_lat, float)
        ratio = llm_lat / default_lat if default_lat > 0 else float("inf")
        print(f"\nLLM vs default: {ratio:.2f}x (1.0 = same, <1.0 = LLM faster)")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-Guided Autotuner MVP")
    parser.add_argument("--kernel", type=str, default="add",
                        choices=[*KERNEL_FACTORIES.keys(), "all"],
                        help="Which kernel to test (or 'all')")
    parser.add_argument("--m", type=int, default=1024, help="Tensor dim 0 / batch size")
    parser.add_argument("--n", type=int, default=1024, help="Tensor dim 1")
    parser.add_argument("--k", type=int, default=1024, help="Inner dim for matmul")
    parser.add_argument("--candidates", type=int, default=20, help="Number of candidate configs")
    parser.add_argument("--top-k", type=int, default=5, help="Configs to benchmark after LLM ranking")
    parser.add_argument("--dry-run", action="store_true", help="Skip LLM call")
    parser.add_argument("--skip-default", action="store_true", help="Skip default autotuner")
    parser.add_argument("--log-dir", type=str, default="logs", help="Directory for logs")
    args = parser.parse_args()

    # --- Set up logging ---
    os.makedirs(args.log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(args.log_dir, f"run_{timestamp}.log")

    fmt = logging.Formatter("%(asctime)s %(name)s %(levelname)s  %(message)s")
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(fmt)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    log.info("Logging to %s", log_file)

    # --- Determine which kernels to run ---
    if args.kernel == "all":
        kernel_names = list(KERNEL_FACTORIES.keys())
    else:
        kernel_names = [args.kernel]

    all_results: dict[str, dict] = {}

    for kernel_name in kernel_names:
        entry = KERNEL_FACTORIES[kernel_name](args)
        results = run_kernel(kernel_name, entry, args, timestamp)
        all_results[kernel_name] = {
            "features": entry["features"],
            "results": results,
        }

    # --- Save all results ---
    results_file = os.path.join(args.log_dir, f"results_{timestamp}.json")
    with open(results_file, "w") as f:
        json.dump(
            {
                "timestamp": timestamp,
                "kernels": kernel_names,
                "args": {"m": args.m, "n": args.n, "k": args.k,
                         "candidates": args.candidates, "top_k": args.top_k},
                "kernel_results": all_results,
            },
            f,
            indent=2,
            default=str,
        )
    log.info("Results saved to %s", results_file)

    # --- Cross-kernel summary ---
    if len(kernel_names) > 1:
        print(f"\n{'=' * 70}")
        print("CROSS-KERNEL SUMMARY")
        print("=" * 70)
        print(f"\n{'Kernel':<15} {'Random (ms)':>12} {'LLM (ms)':>12} {'Default (ms)':>13} {'LLM/Random':>12}")
        print("-" * 68)
        for kn, kr in all_results.items():
            res = kr["results"]
            r = res.get("random_baseline", {}).get("best_latency_ms", float("inf"))
            l = res.get("llm_guided", {}).get("best_latency_ms", float("inf"))
            d = res.get("default_autotuner", {}).get("best_latency_ms", float("nan"))
            ratio = l / r if r > 0 else float("inf")
            print(f"{kn:<15} {r:>12.4f} {l:>12.4f} {d:>13.4f} {ratio:>12.2f}x")


if __name__ == "__main__":
    main()
