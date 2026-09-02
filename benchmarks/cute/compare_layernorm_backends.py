"""Compare LayerNorm forward implementations across backends on B200-class HW.

All impls compute the training-mode forward: y (input dtype), plus per-row
mean and rstd in fp32 (what ``torch.native_layer_norm`` returns and what the
backward pass consumes).

Implementations:

- ``aten``: eager ``torch.native_layer_norm``
- ``compile``: ``torch.compile``'d ``torch.native_layer_norm``
- ``quack``: Quack CuTe layernorm (analytical-heuristic config), direct
  compiled-kernel call (bypasses the ``torch.library.custom_op`` wrapper and
  its per-call host overhead)
- ``quack-tuned``: same kernel, but sweeps Quack's full fwd config space
  (``get_all_fwd_configs``) with a short do_bench each and reports the best —
  the strongest baseline
- ``helion-triton``: ``examples/layer_norm.py`` fwd kernel under
  ``HELION_BACKEND=triton``
- ``helion-cute``: same kernel under ``HELION_BACKEND=cute``

Methodology (same as ``benchmarks/cute/compare_softmax_backends.py``):

- every impl is measured in a fresh subprocess (env isolation; avoids
  ``kernel.bind`` memoization pitfalls),
- CUDA-event ``triton.testing.do_bench`` for every impl (same timer for all),
- median over ``--num-runs`` do_bench medians is the gate metric,
- pre-measurement cooldown to ``--cooldown-temp-c`` equalizes thermal state,
- layernorm is memory bound: results are reported in GB/s
  (``2*M*N*elem + 2*N*elem + 2*M*4`` bytes per call: x read, y write,
  weight+bias read, mean+rstd write).

Examples::

    # Full comparison, one shape
    python benchmarks/cute/compare_layernorm_backends.py --m 32768 --n 8192

    # Single impl with cold full autotune, JSON to stdout
    python benchmarks/cute/compare_layernorm_backends.py --impl helion-cute \\
        --m 32768 --n 8192 --autotune force

    # Fixed Helion config (skip autotune) for A/B experiments
    python benchmarks/cute/compare_layernorm_backends.py --impl helion-cute \\
        --m 32768 --n 8192 --helion-config '{"block_sizes": [1], ...}'
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
QUACK_PATH = REPO_ROOT / "quack"

DEFAULT_IMPLS = (
    "aten",
    "compile",
    "quack",
    "quack-tuned",
    "helion-triton",
    "helion-cute",
)
DEFAULT_SHAPES = (
    "32768x1024",
    "32768x4096",
    "32768x8192",
    "32768x16384",
    "32768x32768",
    "32768x65536",
    "16384x131072",
    "8192x262144",
)


def _dtype_from_name(name: str):  # noqa: ANN202
    import torch

    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def _physical_gpu_index() -> str:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    first = visible.split(",")[0].strip()
    return first or "0"


def _gpu_temp_c() -> float | None:
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                _physical_gpu_index(),
                "--query-gpu=temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
        return float(out.splitlines()[0])
    except Exception:
        return None


def _gpu_info() -> dict[str, Any]:
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                _physical_gpu_index(),
                "--query-gpu=name,power.limit,clocks.max.sm",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
        name, power, clock = (part.strip() for part in out.split(",")[:3])
        return {"gpu": name, "power_limit": power, "max_sm_clock": clock}
    except Exception:
        return {}


def _wait_for_cooldown(target_c: float, timeout_s: float) -> dict[str, Any]:
    """Idle until the GPU cools to ``target_c`` (or timeout). Equalizes
    thermal state across impls measured after very different workloads."""
    import torch

    torch.cuda.synchronize()
    start_temp = _gpu_temp_c()
    t0 = time.time()
    temp = start_temp
    while temp is not None and temp > target_c:
        if time.time() - t0 > timeout_s:
            break
        time.sleep(5)
        temp = _gpu_temp_c()
    return {
        "start_temp_c": start_temp,
        "end_temp_c": temp,
        "waited_s": round(time.time() - t0, 1),
        "target_c": target_c,
    }


def _bench(
    fn: Callable[[], object],
    *,
    num_runs: int,
    warmup_ms: int,
    rep_ms: int,
    cooldown_temp_c: float,
    cooldown_timeout_s: float,
) -> dict[str, Any]:
    from triton.testing import do_bench

    # Populate per-launch caches before the thermal cooldown.
    for _ in range(3):
        fn()
    cooldown = _wait_for_cooldown(cooldown_temp_c, cooldown_timeout_s)
    runs = [
        float(
            do_bench(fn, warmup=warmup_ms, rep=rep_ms, return_mode="median")  # pyrefly: ignore [bad-argument-type]
        )
        for _ in range(num_runs)
    ]
    return {
        "best_ms": min(runs),
        "median_ms": statistics.median(runs),
        "runs_ms": runs,
        "cooldown": cooldown,
    }


def _gbps(m: int, n: int, elem_bytes: int, ms: float) -> float:
    nbytes = 2 * m * n * elem_bytes + 2 * n * elem_bytes + 2 * m * 4
    return nbytes / (ms * 1e-3) / 1e9


def _make_inputs(args: argparse.Namespace):  # noqa: ANN202
    import torch

    torch.manual_seed(args.seed)
    dtype = _dtype_from_name(args.dtype)
    x = torch.randn(args.m, args.n, device="cuda", dtype=dtype)
    weight = torch.randn(args.n, device="cuda", dtype=dtype)
    bias = torch.randn(args.n, device="cuda", dtype=dtype)
    return x, weight, bias


EPS = 1e-5


def _check(x, weight, bias, y, mean, rstd) -> None:  # noqa: ANN001
    import torch

    xf = x.float()
    ref_y = torch.nn.functional.layer_norm(
        xf, [x.size(1)], weight.float(), bias.float(), EPS
    )
    torch.testing.assert_close(y.float(), ref_y, rtol=1e-2, atol=1e-2)
    ref_mean = xf.mean(dim=-1)
    ref_rstd = torch.rsqrt(xf.var(dim=-1, unbiased=False) + EPS)
    # aten returns mean/rstd shaped [M, 1]; others return [M]
    torch.testing.assert_close(mean.float().reshape(-1), ref_mean, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(rstd.float().reshape(-1), ref_rstd, rtol=1e-3, atol=1e-3)


def _quack_compiled_kernel(x, weight, bias, config=None):  # noqa: ANN001, ANN202
    """Compile Quack's layernorm fwd (with mean+rstd stored) and return a
    zero-overhead callable ``fn(x, w, b, out, rstd, mean)``-closed over."""
    import torch

    sys.path.insert(0, str(QUACK_PATH))
    from quack.cute_dsl_utils import (  # pyrefly: ignore [missing-import]
        torch2cute_dtype_map,
    )
    from quack.rmsnorm import _compile_rmsnorm_fwd  # pyrefly: ignore [missing-import]

    dt = torch2cute_dtype_map[x.dtype]
    wdt = torch2cute_dtype_map[weight.dtype]
    bdt = torch2cute_dtype_map[bias.dtype]
    kernel = _compile_rmsnorm_fwd(
        dt,
        dt,
        None,
        wdt,
        bdt,
        None,
        x.size(1),
        True,
        True,
        True,
        False,
        0.0,
        config=config,
    )
    out = torch.empty_like(x)
    mean = torch.empty(x.size(0), device=x.device, dtype=torch.float32)
    rstd = torch.empty(x.size(0), device=x.device, dtype=torch.float32)
    fn = lambda: kernel(x, weight, bias, None, out, None, rstd, mean, EPS)  # noqa: E731
    return fn, out, mean, rstd


def _run_impl(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    x, weight, bias = _make_inputs(args)
    elem_bytes = x.element_size()
    config_used: object = None
    extra: dict[str, Any] = {}
    n = args.n

    if args.impl == "aten":
        fn = lambda: torch.native_layer_norm(x, [n], weight, bias, EPS)  # noqa: E731
        y, mean, rstd = fn()
    elif args.impl == "compile":
        compiled = torch.compile(
            lambda t, w, b: torch.native_layer_norm(t, [n], w, b, EPS)
        )
        fn = lambda: compiled(x, weight, bias)  # noqa: E731
        y, mean, rstd = fn()
    elif args.impl == "quack":
        fn, y, mean, rstd = _quack_compiled_kernel(x, weight, bias)
        fn()
    elif args.impl == "quack-tuned":
        sys.path.insert(0, str(QUACK_PATH))
        from quack.rmsnorm_config import (  # pyrefly: ignore [missing-import]
            get_all_fwd_configs,
        )
        from triton.testing import do_bench

        t0 = time.time()
        best_cfg = None
        best_ms = float("inf")
        for cfg in get_all_fwd_configs():
            if cfg.threads_per_row * cfg.cluster_n > n:
                continue
            try:
                cand_fn, *_ = _quack_compiled_kernel(x, weight, bias, config=cfg)
                cand_fn()
                torch.cuda.synchronize()
            except Exception:
                continue
            ms = float(do_bench(cand_fn, warmup=10, rep=50, return_mode="median"))  # pyrefly: ignore [bad-argument-type]
            if ms < best_ms:
                best_ms, best_cfg = ms, cfg
        extra["tune_s"] = round(time.time() - t0, 1)
        fn, y, mean, rstd = _quack_compiled_kernel(x, weight, bias, config=best_cfg)
        fn()
        config_used = repr(best_cfg)
    elif args.impl in ("helion-triton", "helion-cute"):
        backend = args.impl.split("-")[1]
        assert os.environ.get("HELION_BACKEND", backend) == backend, (
            "driver must set HELION_BACKEND to match --impl"
        )
        os.environ["HELION_BACKEND"] = backend
        from examples.layer_norm import layer_norm_fwd

        import helion

        kernel = layer_norm_fwd
        kernel.settings.print_output_code = bool(args.print_code)
        kernel_args = (x, [n], weight, bias, EPS)
        # --helion-config pins the cute config; --helion-config-triton pins
        # the triton config (for interleaved ABAB verify runs where both
        # backends alternate with their own autotune winners).
        config_json = (
            args.helion_config_triton if backend == "triton" else args.helion_config
        )
        if config_json:
            config = helion.Config(**json.loads(config_json))
            kernel.configs = [config]
            bound = kernel.bind(kernel_args)
            bound.set_config(config)
            config_used = config.to_json()
        elif args.autotune == "force":
            t0 = time.time()
            config = kernel.autotune(kernel_args, force=True)
            extra["autotune_s"] = round(time.time() - t0, 1)
            extra["autotune_seed"] = kernel.settings.autotune_random_seed
            bound = kernel.bind(kernel_args)
            config_used = config.to_json()
        else:
            bound = kernel.bind(kernel_args)
            cfg = getattr(bound, "_config", None)
            config_used = cfg.to_json() if cfg is not None else None
        fn = lambda: bound(*kernel_args)  # noqa: E731
        y, mean, rstd = fn()
    else:
        raise SystemExit(f"unknown impl {args.impl!r}")

    if not args.skip_correctness:
        _check(x, weight, bias, y, mean, rstd)

    stats = _bench(
        fn,
        num_runs=args.num_runs,
        warmup_ms=args.warmup_ms,
        rep_ms=args.rep_ms,
        cooldown_temp_c=args.cooldown_temp_c,
        cooldown_timeout_s=args.cooldown_timeout_s,
    )
    return {
        "impl": args.impl,
        "m": args.m,
        "n": args.n,
        "dtype": args.dtype,
        "median_ms": stats["median_ms"],
        "best_ms": stats["best_ms"],
        "median_gbps": _gbps(args.m, args.n, elem_bytes, stats["median_ms"]),
        "best_gbps": _gbps(args.m, args.n, elem_bytes, stats["best_ms"]),
        "runs_ms": stats["runs_ms"],
        "cooldown": stats["cooldown"],
        "config": config_used,
        "helion_fast_math": os.environ.get("HELION_FAST_MATH", ""),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        **_gpu_info(),
        **extra,
    }


def _spawn_impl(args: argparse.Namespace, impl: str, shape: str) -> dict[str, Any]:
    m, n = (int(v) for v in shape.split("x"))
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--impl",
        impl,
        "--m",
        str(m),
        "--n",
        str(n),
        "--dtype",
        args.dtype,
        "--autotune",
        args.autotune,
        "--num-runs",
        str(args.num_runs),
        "--warmup-ms",
        str(args.warmup_ms),
        "--rep-ms",
        str(args.rep_ms),
        "--cooldown-temp-c",
        str(args.cooldown_temp_c),
        "--cooldown-timeout-s",
        str(args.cooldown_timeout_s),
        "--seed",
        str(args.seed),
    ]
    if args.helion_config:
        cmd.extend(["--helion-config", args.helion_config])
    if args.helion_config_triton:
        cmd.extend(["--helion-config-triton", args.helion_config_triton])
    if args.skip_correctness:
        cmd.append("--skip-correctness")
    env = dict(os.environ)
    if impl.startswith("helion-"):
        env["HELION_BACKEND"] = impl.split("-")[1]
    else:
        env.pop("HELION_BACKEND", None)
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    marker = "RESULT_JSON: "
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith(marker):
            return json.loads(line[len(marker) :])
    return {
        "impl": impl,
        "m": m,
        "n": n,
        "error": (proc.stderr or proc.stdout)[-2000:],
        "returncode": proc.returncode,
    }


def _driver(args: argparse.Namespace) -> None:
    impls = args.impls.split(",")
    shapes = args.shapes.split(",")
    results: list[dict[str, Any]] = []
    for shape in shapes:
        for impl in impls:
            record = _spawn_impl(args, impl, shape)
            record["tag"] = args.tag
            results.append(record)
            if "error" in record:
                print(f"[{shape}] {impl}: ERROR\n{record['error']}")
            else:
                print(
                    f"[{shape}] {impl}: {record['median_ms']:.4f} ms "
                    f"{record['median_gbps']:.0f} GB/s"
                )
            if args.output:
                with open(args.output, "a") as f:
                    f.write(json.dumps(record) + "\n")
    _print_table(results, shapes, impls)


def _print_table(
    results: list[dict[str, Any]], shapes: list[str], impls: list[str]
) -> None:
    by_key = {(f"{r['m']}x{r['n']}", r["impl"]): r for r in results if "error" not in r}
    header = ["shape".ljust(14)] + [i.rjust(14) for i in impls]
    print("\nGB/s (median):")
    print("".join(header))
    for shape in shapes:
        row = [shape.ljust(14)]
        for impl in impls:
            r = by_key.get((shape, impl))
            row.append(f"{r['median_gbps']:.0f}".rjust(14) if r else "ERR".rjust(14))
        print("".join(row))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--impl", default=None, help="run a single impl in-process")
    parser.add_argument("--impls", default=",".join(DEFAULT_IMPLS))
    parser.add_argument("--shapes", default=",".join(DEFAULT_SHAPES))
    parser.add_argument("--m", type=int, default=32768)
    parser.add_argument("--n", type=int, default=8192)
    parser.add_argument(
        "--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"]
    )
    parser.add_argument(
        "--autotune",
        default="force",
        choices=["force", "cache", "none"],
        help="force = cold full autotune; cache = default bind path; "
        "none = requires HELION_AUTOTUNE_EFFORT=none in env",
    )
    parser.add_argument(
        "--helion-config",
        default=None,
        help="JSON helion.Config kwargs for helion-cute; skips autotune",
    )
    parser.add_argument(
        "--helion-config-triton",
        default=None,
        help="JSON helion.Config kwargs for helion-triton; skips autotune",
    )
    parser.add_argument("--num-runs", type=int, default=5)
    parser.add_argument("--warmup-ms", type=int, default=100)
    parser.add_argument("--rep-ms", type=int, default=300)
    parser.add_argument("--cooldown-temp-c", type=float, default=55.0)
    parser.add_argument("--cooldown-timeout-s", type=float, default=600.0)
    parser.add_argument("--skip-correctness", action="store_true")
    parser.add_argument("--print-code", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", default="")
    parser.add_argument("--output", default=None, help="append JSONL records here")
    args = parser.parse_args()

    if args.impl is None:
        _driver(args)
        return
    result = _run_impl(args)
    print("RESULT_JSON: " + json.dumps(result))


if __name__ == "__main__":
    main()
