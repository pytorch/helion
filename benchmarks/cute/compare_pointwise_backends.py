"""Compare pointwise kernel implementations across backends on B200-class HW.

16 distinct pointwise kernels (op x shape x dtype), all memory-bound
(>=134MB moved per call). Each variant is a separate Helion kernel written in
the natural style (``for tile in hl.tile(out.size())``).

Implementations:

- ``aten``: eager PyTorch op
- ``compile``: ``torch.compile(mode="max-autotune-no-cudagraphs")``
- ``cute-manual``: handwritten CuTe-DSL elementwise kernel
  (``cute_pointwise_manual.py``) with a mini config sweep — the structural
  comparator (quack has no standalone pointwise kernels)
- ``helion-triton``: the Helion kernel under ``HELION_BACKEND=triton``
- ``helion-cute``: the Helion kernel under ``HELION_BACKEND=cute``

Methodology (same as ``benchmarks/cute/compare_cross_entropy_backends.py``):

- every impl is measured in a fresh subprocess (env isolation; avoids
  ``kernel.bind`` memoization pitfalls),
- CUDA-event ``triton.testing.do_bench`` for every impl (same timer for all),
- median over ``--num-runs`` do_bench medians is the gate metric,
- pre-measurement cooldown to ``--cooldown-temp-c`` equalizes thermal state,
- pointwise kernels are memory bound: results are reported in GB/s using
  exact per-variant byte accounting (all input bytes read + output bytes
  written).

Variants are selected by name, e.g. ``--variants add,gelu_tanh``.

Examples::

    # Full comparison, one variant
    python benchmarks/cute/compare_pointwise_backends.py --variants add

    # Single impl with cold full autotune, JSON to stdout
    python benchmarks/cute/compare_pointwise_backends.py \\
        --impl helion-cute --variant add --autotune force

    # Fixed Helion config (skip autotune) for A/B experiments
    python benchmarks/cute/compare_pointwise_backends.py \\
        --impl helion-cute --variant add --helion-config '{...}'
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any
from typing import Callable

import torch
import torch.nn.functional as F

import helion
import helion.language as hl

REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Helion kernels (one per op, written in the natural pointwise style)
# ---------------------------------------------------------------------------


@helion.kernel()
def k_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(out.size()):
        out[tile] = x[tile] + y[tile]
    return out


@helion.kernel()
def k_mul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(out.size()):
        out[tile] = x[tile] * y[tile]
    return out


@helion.kernel()
def k_copy(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(out.size()):
        out[tile] = x[tile]
    return out


@helion.kernel()
def k_cast(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty(x.shape, dtype=torch.bfloat16, device=x.device)
    for tile in hl.tile(out.size()):
        out[tile] = x[tile].to(torch.bfloat16)
    return out


@helion.kernel()
def k_relu(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(out.size()):
        out[tile] = torch.relu(x[tile])
    return out


# gelu/sigmoid opt into approximate math via the ``fast_math`` SETTING (the
# sanctioned user-level opt-in; it is deliberately NOT an autotuner knob):
# their structural baseline (cute-manual, quack-style) uses MUFU approx
# transcendentals, and accurate math measurably trails it there (gelu's
# accurate tanh; sigmoid by ~2%).  silu/tanh/exp/rsqrt stay accurate —
# they reach baseline parity without it.
@helion.kernel(fast_math=True)
def k_gelu_tanh(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(out.size()):
        v = x[tile].to(torch.float32)
        # FMA-friendly gelu-tanh algebra (same form quack/flash-attention
        # use): 0.5*v*(1+tanh(b*(v+c*v^3))) == hv + hv*tanh(v*(b + b*c*v^2))
        v_sq = v * v
        inner = v * (0.7978845608028654 + 0.035677408136300125 * v_sq)
        half_v = 0.5 * v
        out[tile] = (half_v + half_v * torch.tanh(inner)).to(x.dtype)
    return out


@helion.kernel()
def k_silu(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(out.size()):
        v = x[tile].to(torch.float32)
        out[tile] = (v * torch.sigmoid(v)).to(x.dtype)
    return out


@helion.kernel(fast_math=True)
def k_sigmoid(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(out.size()):
        v = x[tile].to(torch.float32)
        out[tile] = torch.sigmoid(v).to(x.dtype)
    return out


@helion.kernel()
def k_tanh(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(out.size()):
        out[tile] = torch.tanh(x[tile])
    return out


@helion.kernel()
def k_exp(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(out.size()):
        out[tile] = torch.exp(x[tile])
    return out


@helion.kernel()
def k_rsqrt(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(out.size()):
        out[tile] = torch.rsqrt(x[tile])
    return out


@helion.kernel()
def k_addcmul(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(out.size()):
        out[tile] = x[tile] + y[tile] * z[tile]
    return out


@helion.kernel()
def k_saxpy(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(out.size()):
        out[tile] = 2.5 * x[tile] + y[tile]
    return out


@helion.kernel()
def k_bias_add(x: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile_m, tile_n in hl.tile(out.size()):
        out[tile_m, tile_n] = x[tile_m, tile_n] + b[tile_n]
    return out


@helion.kernel()
def k_leaky_relu(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(out.size()):
        v = x[tile]
        out[tile] = torch.where(v > 0, v, 0.01 * v)
    return out


@helion.kernel()
def k_clamp(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(out.size()):
        out[tile] = torch.clamp(x[tile], -2.0, 2.0)
    return out


# ---------------------------------------------------------------------------
# Variant registry
# ---------------------------------------------------------------------------

DT = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


def _randn(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    return torch.randn(shape, device="cuda", dtype=dtype)


@dataclasses.dataclass
class Variant:
    name: str
    shape: tuple[int, ...]
    dtype: str
    helion_kernel: Any
    aten_fn: Callable[..., torch.Tensor]
    make_inputs: Callable[[], tuple[torch.Tensor, ...]]
    nbytes: int
    manual_op: str


def _make_variants() -> dict[str, Variant]:
    v: dict[str, Variant] = {}

    def reg(
        name: str,
        shape: tuple[int, ...],
        dtype: str,
        kernel: object,
        aten_fn: Callable[..., torch.Tensor],
        make_inputs: Callable[[], tuple[torch.Tensor, ...]],
        nbytes: int,
        manual_op: str,
    ) -> None:
        v[name] = Variant(
            name, shape, dtype, kernel, aten_fn, make_inputs, nbytes, manual_op
        )

    def numel(shape: tuple[int, ...]) -> int:
        n = 1
        for s in shape:
            n *= s
        return n

    def esz(dtype: str) -> int:
        return {"fp16": 2, "bf16": 2, "fp32": 4}[dtype]

    # 1. add: x + y (bf16 2D)
    s, d = (16384, 8192), "bf16"
    reg(
        "add",
        s,
        d,
        k_add,
        torch.add,
        lambda s=s, d=d: (_randn(s, DT[d]), _randn(s, DT[d])),
        3 * numel(s) * esz(d),
        "add",
    )
    # 2. mul: x * y (fp32 2D)
    s, d = (8192, 8192), "fp32"
    reg(
        "mul",
        s,
        d,
        k_mul,
        torch.mul,
        lambda s=s, d=d: (_randn(s, DT[d]), _randn(s, DT[d])),
        3 * numel(s) * esz(d),
        "mul",
    )
    # 3. copy (bf16 1D)
    s, d = (2**27,), "bf16"
    reg(
        "copy",
        s,
        d,
        k_copy,
        lambda x: x.clone(),
        lambda s=s, d=d: (_randn(s, DT[d]),),
        2 * numel(s) * esz(d),
        "copy",
    )
    # 4. cast fp32 -> bf16 (1D)
    s, d = (2**26,), "fp32"
    reg(
        "cast",
        s,
        d,
        k_cast,
        lambda x: x.to(torch.bfloat16),
        lambda s=s, d=d: (_randn(s, DT[d]),),
        numel(s) * (4 + 2),
        "cast",
    )
    # 5. relu (fp16 2D)
    s, d = (16384, 4096), "fp16"
    reg(
        "relu",
        s,
        d,
        k_relu,
        torch.relu,
        lambda s=s, d=d: (_randn(s, DT[d]),),
        2 * numel(s) * esz(d),
        "relu",
    )
    # 6. gelu tanh approx (bf16 2D)
    s, d = (32768, 4096), "bf16"
    reg(
        "gelu_tanh",
        s,
        d,
        k_gelu_tanh,
        lambda x: F.gelu(x, approximate="tanh"),
        lambda s=s, d=d: (_randn(s, DT[d]),),
        2 * numel(s) * esz(d),
        "gelu_tanh",
    )
    # 7. silu (bf16 2D, non-pow2 N)
    s, d = (16384, 11008), "bf16"
    reg(
        "silu",
        s,
        d,
        k_silu,
        F.silu,
        lambda s=s, d=d: (_randn(s, DT[d]),),
        2 * numel(s) * esz(d),
        "silu",
    )
    # 8. sigmoid (fp16 1D)
    s, d = (2**26,), "fp16"
    reg(
        "sigmoid",
        s,
        d,
        k_sigmoid,
        torch.sigmoid,
        lambda s=s, d=d: (_randn(s, DT[d]),),
        2 * numel(s) * esz(d),
        "sigmoid",
    )
    # 9. tanh (fp32 2D)
    s, d = (4096, 16384), "fp32"
    reg(
        "tanh",
        s,
        d,
        k_tanh,
        torch.tanh,
        lambda s=s, d=d: (_randn(s, DT[d]),),
        2 * numel(s) * esz(d),
        "tanh",
    )
    # 10. exp (fp32 1D)
    s, d = (2**26,), "fp32"
    reg(
        "exp",
        s,
        d,
        k_exp,
        torch.exp,
        lambda s=s, d=d: (_randn(s, DT[d]),),
        2 * numel(s) * esz(d),
        "exp",
    )
    # 11. rsqrt (fp32 1D, small-end coverage)
    s, d = (2**24,), "fp32"
    reg(
        "rsqrt",
        s,
        d,
        k_rsqrt,
        torch.rsqrt,
        lambda s=s, d=d: (_randn(s, DT[d]).abs() + 0.5,),
        2 * numel(s) * esz(d),
        "rsqrt",
    )
    # 12. addcmul: x + y*z (bf16 2D, 3 reads)
    s, d = (8192, 4096), "bf16"
    reg(
        "addcmul",
        s,
        d,
        k_addcmul,
        lambda x, y, z: torch.addcmul(x, y, z),
        lambda s=s, d=d: (_randn(s, DT[d]), _randn(s, DT[d]), _randn(s, DT[d])),
        4 * numel(s) * esz(d),
        "addcmul",
    )
    # 13. saxpy: 2.5*x + y (fp32 1D)
    s, d = (2**26,), "fp32"
    reg(
        "saxpy",
        s,
        d,
        k_saxpy,
        lambda x, y: torch.add(y, x, alpha=2.5),
        lambda s=s, d=d: (_randn(s, DT[d]), _randn(s, DT[d])),
        3 * numel(s) * esz(d),
        "saxpy",
    )
    # 14. bias_add: x[M,N] + b[N] (fp16, ODD N)
    s, d = (4096, 50257), "fp16"
    reg(
        "bias_add",
        s,
        d,
        k_bias_add,
        torch.add,
        lambda s=s, d=d: (_randn(s, DT[d]), _randn((s[1],), DT[d])),
        2 * numel(s) * esz(d) + s[1] * esz(d),
        "bias_add",
    )
    # 15. leaky_relu (fp16 1D)
    s, d = (2**26,), "fp16"
    reg(
        "leaky_relu",
        s,
        d,
        k_leaky_relu,
        lambda x: F.leaky_relu(x, 0.01),
        lambda s=s, d=d: (_randn(s, DT[d]),),
        2 * numel(s) * esz(d),
        "leaky_relu",
    )
    # 16. clamp (fp32 1D)
    s, d = (2**25,), "fp32"
    reg(
        "clamp",
        s,
        d,
        k_clamp,
        lambda x: torch.clamp(x, -2.0, 2.0),
        lambda s=s, d=d: (_randn(s, DT[d]),),
        2 * numel(s) * esz(d),
        "clamp",
    )
    return v


VARIANTS = _make_variants()
DEFAULT_IMPLS = ("aten", "compile", "cute-manual", "helion-triton", "helion-cute")


def _variant_label(v: Variant) -> str:
    return f"{v.name}-{'x'.join(str(s) for s in v.shape)}-{v.dtype}"


# ---------------------------------------------------------------------------
# Measurement (same methodology as the cross_entropy harness)
# ---------------------------------------------------------------------------


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


def _check(variant: Variant, inputs: tuple[torch.Tensor, ...], out) -> None:  # noqa: ANN001
    ref = variant.aten_fn(*inputs)
    if out.dtype in (torch.float16, torch.bfloat16):
        rtol, atol = 1e-2, 1e-2
    else:
        rtol, atol = 1.5e-5, 1e-5
    torch.testing.assert_close(out.to(ref.dtype), ref, rtol=rtol, atol=atol)


def _run_impl(args: argparse.Namespace) -> dict[str, Any]:
    variant = VARIANTS[args.variant]
    torch.manual_seed(args.seed)
    inputs = variant.make_inputs()
    config_used: object = None
    extra: dict[str, Any] = {}

    if args.impl == "aten":
        fn = lambda: variant.aten_fn(*inputs)  # noqa: E731
        out = fn()
    elif args.impl == "compile":
        compiled = torch.compile(
            variant.aten_fn, mode="max-autotune-no-cudagraphs", dynamic=False
        )
        fn = lambda: compiled(*inputs)  # noqa: E731
        out = fn()
    elif args.impl == "cute-manual":
        from cute_pointwise_manual import (  # pyrefly: ignore [missing-import]
            make_manual_fn,
        )

        fn, out, config_used, tune_s = make_manual_fn(
            variant.manual_op, inputs, sweep=not args.manual_no_sweep
        )
        extra["tune_s"] = tune_s
    elif args.impl in ("helion-triton", "helion-cute"):
        backend = args.impl.split("-")[1]
        # The kernel is decorated at module import, so the backend must come
        # from the environment before this process even started.
        assert os.environ.get("HELION_BACKEND") == backend, (
            "HELION_BACKEND must be set to match --impl before launching"
        )
        kernel = variant.helion_kernel
        kernel.settings.print_output_code = bool(args.print_code)
        kernel_args = inputs
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
        out = fn()
    else:
        raise SystemExit(f"unknown impl {args.impl!r}")

    if not args.skip_correctness:
        _check(variant, inputs, out)

    stats = _bench(
        fn,
        num_runs=args.num_runs,
        warmup_ms=args.warmup_ms,
        rep_ms=args.rep_ms,
        cooldown_temp_c=args.cooldown_temp_c,
        cooldown_timeout_s=args.cooldown_timeout_s,
    )
    gbps_median = variant.nbytes / (stats["median_ms"] * 1e-3) / 1e9
    gbps_best = variant.nbytes / (stats["best_ms"] * 1e-3) / 1e9
    return {
        "impl": args.impl,
        "variant": args.variant,
        "label": _variant_label(variant),
        "nbytes": variant.nbytes,
        "median_ms": stats["median_ms"],
        "best_ms": stats["best_ms"],
        "median_gbps": gbps_median,
        "best_gbps": gbps_best,
        "runs_ms": stats["runs_ms"],
        "cooldown": stats["cooldown"],
        "config": config_used,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        **_gpu_info(),
        **extra,
    }


def _spawn_impl(args: argparse.Namespace, impl: str, variant: str) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--impl",
        impl,
        "--variant",
        variant,
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
    if args.manual_no_sweep:
        cmd.append("--manual-no-sweep")
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
        "variant": variant,
        "error": (proc.stderr or proc.stdout)[-2000:],
        "returncode": proc.returncode,
    }


def _driver(args: argparse.Namespace) -> None:
    impls = args.impls.split(",")
    variants = args.variants.split(",")
    results: list[dict[str, Any]] = []
    for variant in variants:
        for impl in impls:
            record = _spawn_impl(args, impl, variant)
            record["tag"] = args.tag
            results.append(record)
            if "error" in record:
                print(f"[{variant}] {impl}: ERROR\n{record['error']}")
            else:
                print(
                    f"[{variant}] {impl}: {record['median_ms']:.4f} ms "
                    f"{record['median_gbps']:.0f} GB/s"
                )
            if args.output:
                with open(args.output, "a") as f:
                    f.write(json.dumps(record) + "\n")
    _print_table(results, variants, impls)


def _print_table(
    results: list[dict[str, Any]], variants: list[str], impls: list[str]
) -> None:
    by_key = {(r.get("variant"), r["impl"]): r for r in results if "error" not in r}
    header = ["variant".ljust(28)] + [i.rjust(14) for i in impls]
    print("\nGB/s (median):")
    print("".join(header))
    for variant in variants:
        v = VARIANTS[variant]
        row = [_variant_label(v).ljust(28)]
        for impl in impls:
            r = by_key.get((variant, impl))
            row.append(f"{r['median_gbps']:.0f}".rjust(14) if r else "ERR".rjust(14))
        print("".join(row))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--impl", default=None, help="run a single impl in-process")
    parser.add_argument("--impls", default=",".join(DEFAULT_IMPLS))
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--variant", default=None, help="single variant (with --impl)")
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
    parser.add_argument("--manual-no-sweep", action="store_true")
    parser.add_argument("--num-runs", type=int, default=5)
    parser.add_argument("--warmup-ms", type=int, default=100)
    parser.add_argument("--rep-ms", type=int, default=300)
    parser.add_argument("--cooldown-temp-c", type=float, default=42.0)
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
    assert args.variant is not None, "--impl requires --variant"
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    result = _run_impl(args)
    print("RESULT_JSON: " + json.dumps(result))


if __name__ == "__main__":
    main()
