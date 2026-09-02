"""Compare cross-entropy forward implementations across backends on B200-class HW.

All impls compute the per-row loss (``reduction="none"``):
``loss[i] = logsumexp(x[i, :]) - x[i, target[i]]`` for logits ``x [M, N]`` and
int64 ``target [M]``. quack/helion write fp32 losses; aten/compile return the
input dtype (checked with looser tolerance).

Implementations:

- ``aten``: eager ``F.cross_entropy(..., reduction="none")``
- ``compile``: ``torch.compile``'d ``F.cross_entropy(..., reduction="none")``
- ``quack``: Quack CuTe cross-entropy (analytical-heuristic config), direct
  compiled-kernel call (bypasses the ``torch.library.custom_op`` wrapper and
  its per-call host overhead)
- ``quack-tuned``: same kernel, but sweeps threads_per_row / num_threads /
  cluster_n / online_softmax / reload_from with a short do_bench each and
  reports the best — the strongest baseline
- ``helion-triton``: the Helion kernel below under ``HELION_BACKEND=triton``
- ``helion-cute``: same kernel under ``HELION_BACKEND=cute``

Methodology (same as ``benchmarks/cute/compare_rmsnorm_backends.py``):

- every impl is measured in a fresh subprocess (env isolation; avoids
  ``kernel.bind`` memoization pitfalls),
- CUDA-event ``triton.testing.do_bench`` for every impl (same timer for all),
- median over ``--num-runs`` do_bench medians is the gate metric,
- pre-measurement cooldown to ``--cooldown-temp-c`` equalizes thermal state,
- cross-entropy fwd is memory bound: results are reported in GB/s
  (``M*N*elem + M*8 + M*4`` bytes per call: x read, target read, loss write —
  the same accounting as quack's own benchmark).

Variants are ``MxNxdtype`` strings, e.g. ``32768x8192xbf16``.

Examples::

    # Full comparison, one variant
    python benchmarks/cute/compare_cross_entropy_backends.py \\
        --shapes 32768x8192xbf16

    # Single impl with cold full autotune, JSON to stdout
    python benchmarks/cute/compare_cross_entropy_backends.py \\
        --impl helion-cute --m 32768 --n 8192 --dtype bf16 --autotune force

    # Fixed Helion config (skip autotune) for A/B experiments
    python benchmarks/cute/compare_cross_entropy_backends.py \\
        --impl helion-cute --m 32768 --n 8192 --dtype bf16 \\
        --helion-config '{"block_sizes": [1], ...}'
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any
from typing import Callable

import torch

import helion
import helion.language as hl

REPO_ROOT = Path(__file__).resolve().parents[2]
QUACK_PATH = REPO_ROOT / "quack"


@helion.kernel()
def helion_cross_entropy_fwd(
    logits: torch.Tensor,  # [M, N]
    target: torch.Tensor,  # [M]
) -> torch.Tensor:
    m, n = logits.shape
    losses = torch.empty([m], dtype=torch.float32, device=logits.device)
    logits_flat = logits.view(-1)
    for tile_m in hl.tile(m):
        target_tile = target[tile_m]
        flat_indices = tile_m.index * n + target_tile
        logits_at_target = hl.load(logits_flat, [flat_indices]).to(torch.float32)
        rows = logits[tile_m, :].to(torch.float32)
        max_logits = torch.amax(rows, dim=-1)
        sum_exp = torch.sum(torch.exp(rows - max_logits[:, None]), dim=-1)
        losses[tile_m] = max_logits + torch.log(sum_exp) - logits_at_target
    return losses


DEFAULT_IMPLS = (
    "aten",
    "compile",
    "quack",
    "quack-tuned",
    "helion-triton",
    "helion-cute",
)
DEFAULT_SHAPES = (
    "65536x512xbf16",
    "32768x1024xfp16",
    "32768x2048xfp32",
    "32768x4096xbf16",
    "32768x8192xfp16",
    "32768x8192xfp32",
    "32768x16384xbf16",
    "32768x32768xfp16",
    "32768x32768xfp32",
    "32768x50257xfp16",
    "32768x65536xbf16",
    "16384x131072xbf16",
    "16384x131072xfp16",
    "8192x131072xfp32",
    "8192x262144xbf16",
    "8192x262144xfp16",
)


def _dtype_from_name(name: str):  # noqa: ANN202
    import torch

    return {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
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
    nbytes = m * n * elem_bytes + m * 8 + m * 4
    return nbytes / (ms * 1e-3) / 1e9


def _make_inputs(args: argparse.Namespace):  # noqa: ANN202
    import torch

    torch.manual_seed(args.seed)
    dtype = _dtype_from_name(args.dtype)
    # 0.1*randn matches quack's own benchmark data distribution.
    x = 0.1 * torch.randn(args.m, args.n, device="cuda", dtype=dtype)
    target = torch.randint(0, args.n, (args.m,), device="cuda", dtype=torch.int64)
    return x, target


def _check(x, target, losses) -> None:  # noqa: ANN001
    import torch
    import torch.nn.functional as F

    ref = F.cross_entropy(x.float(), target, reduction="none")
    if losses.dtype == torch.float32:
        rtol, atol = 1e-3, 2e-3
    else:  # aten/compile return the input dtype; storage rounding dominates
        rtol, atol = 1e-2, 1e-1
    torch.testing.assert_close(losses.float(), ref, rtol=rtol, atol=atol)


def _quack_config_space(n: int, dtype_width: int) -> list[dict[str, Any]]:
    """Enumerate valid (threads_per_row, num_threads, cluster_n,
    online_softmax, reload_from) combos for quack's CrossEntropy."""
    vecsize = math.gcd(n, 128 // dtype_width)
    smem_limit = 224 * 1024
    out: list[dict[str, Any]] = []
    for online in (True, False):
        for num_threads in (128, 256):
            for tpr in (32, 64, 128, 256):
                if tpr > num_threads:
                    continue
                # cap so every peer CTA owns a distinct, non-empty N-tile
                max_cluster = max(1, (n // vecsize) // tpr)
                for cluster in (1, 2, 4, 8, 16):
                    if cluster > 1 and cluster > max_cluster:
                        continue
                    blocks_n = math.ceil(n / vecsize / (tpr * cluster))
                    tiler_n = vecsize * blocks_n * tpr
                    smem = (num_threads // tpr) * tiler_n * (dtype_width // 8)
                    if smem > smem_limit:
                        continue
                    reload_opts = (None,) if online or n <= 16384 else (None, "smem")
                    for reload_from in reload_opts:
                        out.append(
                            {
                                "online_softmax": online,
                                "num_threads": num_threads,
                                "threads_per_row": tpr,
                                "cluster_n": cluster,
                                "reload_from": reload_from,
                            }
                        )
    return out


def _quack_compiled_kernel(x, target, config=None):  # noqa: ANN001, ANN202
    """Compile Quack's cross-entropy fwd and return a zero-overhead callable
    plus the loss buffer it writes."""
    import torch

    sys.path.insert(0, str(QUACK_PATH))
    import cutlass
    from cutlass import Float32  # pyrefly: ignore [missing-import]
    from cutlass import Int32  # pyrefly: ignore [missing-import]
    import cutlass.cute as cute  # pyrefly: ignore [missing-import]
    from quack.compile_utils import make_fake_tensor  # pyrefly: ignore [missing-import]
    from quack.cross_entropy import CrossEntropy  # pyrefly: ignore [missing-import]
    from quack.cute_dsl_utils import (  # pyrefly: ignore [missing-import]
        torch2cute_dtype_map,
    )

    cutlass.cuda.initialize_cuda_context()
    dtype = torch2cute_dtype_map[x.dtype]
    target_dtype = torch2cute_dtype_map[target.dtype]
    n = x.size(1)

    if config is None:
        op = CrossEntropy(dtype, n, online_softmax=True)
    else:

        class TunedCrossEntropy(CrossEntropy):
            def _threads_per_row(self):  # noqa: ANN202
                return config["threads_per_row"]

            def _num_threads(self):  # noqa: ANN202
                return config["num_threads"]

            def _set_cluster_n(self) -> None:
                self.cluster_n = config["cluster_n"]

        op = TunedCrossEntropy(dtype, n, online_softmax=config["online_softmax"])
        op.reload_from = config["reload_from"]

    batch_sym = cute.sym_int()
    div = math.gcd(128 // dtype.width, n)
    x_cute = make_fake_tensor(dtype, (batch_sym, n), div)
    target_cute = make_fake_tensor(target_dtype, (batch_sym,))
    loss_cute = make_fake_tensor(Float32, (batch_sym,))
    compiled = cute.compile(
        op,
        x_cute,
        target_cute,
        None,  # target_logit
        loss_cute,
        None,  # lse
        None,  # dx
        None,  # weight
        Int32(0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )
    loss = torch.empty(x.size(0), device=x.device, dtype=torch.float32)
    fn = lambda: compiled(x, target, None, loss, None, None, None, Int32(-100))  # noqa: E731
    return fn, loss


def _run_impl(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    x, target = _make_inputs(args)
    elem_bytes = x.element_size()
    config_used: object = None
    extra: dict[str, Any] = {}

    if args.impl == "aten":
        fn = lambda: F.cross_entropy(x, target, reduction="none")  # noqa: E731
        losses = fn()
    elif args.impl == "compile":
        compiled = torch.compile(lambda x, t: F.cross_entropy(x, t, reduction="none"))
        fn = lambda: compiled(x, target)  # noqa: E731
        losses = fn()
    elif args.impl == "quack":
        fn, losses = _quack_compiled_kernel(x, target)
        fn()
    elif args.impl == "quack-tuned":
        from triton.testing import do_bench

        t0 = time.time()
        best_cfg = None
        best_ms = float("inf")
        for cfg in _quack_config_space(args.n, elem_bytes * 8):
            try:
                cand_fn, _ = _quack_compiled_kernel(x, target, config=cfg)
                cand_fn()
                torch.cuda.synchronize()
            except Exception:
                continue
            ms = float(do_bench(cand_fn, warmup=10, rep=50, return_mode="median"))  # pyrefly: ignore [bad-argument-type]
            if ms < best_ms:
                best_ms, best_cfg = ms, cfg
        extra["tune_s"] = round(time.time() - t0, 1)
        fn, losses = _quack_compiled_kernel(x, target, config=best_cfg)
        fn()
        config_used = repr(best_cfg)
    elif args.impl in ("helion-triton", "helion-cute"):
        backend = args.impl.split("-")[1]
        # The kernel is decorated at module import, so the backend must come
        # from the environment before this process even started.
        assert os.environ.get("HELION_BACKEND") == backend, (
            "HELION_BACKEND must be set to match --impl before launching"
        )

        kernel = helion_cross_entropy_fwd
        kernel.settings.print_output_code = bool(args.print_code)
        kernel_args = (x, target)
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
        losses = fn()
    else:
        raise SystemExit(f"unknown impl {args.impl!r}")

    if not args.skip_correctness:
        _check(x, target, losses)

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
    m, n, dtype = shape.split("x")
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--impl",
        impl,
        "--m",
        m,
        "--n",
        n,
        "--dtype",
        dtype,
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
        "m": int(m),
        "n": int(n),
        "dtype": dtype,
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
    by_key = {
        (f"{r['m']}x{r['n']}x{r['dtype']}", r["impl"]): r
        for r in results
        if "error" not in r
    }
    header = ["shape".ljust(20)] + [i.rjust(14) for i in impls]
    print("\nGB/s (median):")
    print("".join(header))
    for shape in shapes:
        row = [shape.ljust(20)]
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
    parser.add_argument("--dtype", default="bf16", choices=["fp16", "bf16", "fp32"])
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
