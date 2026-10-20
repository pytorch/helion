"""Compare Mamba2 chunk-state forward implementations across backends.

The op (``examples/mamba2_chunk_state.py``): for every (batch, chunk, head),
``states[p, n] = sum_l x[l, p] * exp(dA_cs[last] - dA_cs[l]) * dt[l] * B[l, n]``
over the ``chunk_size`` positions ``l`` of the chunk (B is shared by the
``nheads // ngroups`` heads of a group). Outputs are in the input dtype.

Implementations:

- ``aten``: eager PyTorch as one grouped batched matmul (``chunk_state_grouped``;
  the example's einsum ``ref_chunk_state`` materializes multi-GB intermediates
  at these shapes)
- ``compile``: ``torch.compile``'d ``chunk_state_grouped``
- ``mamba-triton``: the Mamba2 ``_chunk_state_fwd`` Triton kernel
  (``benchmarks/cute/mamba2_chunk_state_triton.py``, autotuned over its own
  config list)
- ``helion-triton``: ``examples/mamba2_chunk_state.py`` under
  ``HELION_BACKEND=triton``
- ``helion-cute``: same kernel under ``HELION_BACKEND=cute``

Methodology (same as ``benchmarks/cute/compare_layernorm_backends.py``):
every impl runs in a fresh subprocess, every impl is timed with CUDA-event
``triton.testing.do_bench``, the gate metric is the median over
``--num-runs`` do_bench medians, and a pre-measurement cooldown equalizes
thermal state. The op is memory bound, so results are reported in GB/s over
the compulsory traffic: x, B (once per group), dt and dA_cumsum reads plus
the states write.

Shapes are ``b{batch}_h{nheads}_g{ngroups}_L{seqlen}_c{chunk}_p{headdim}_n{dstate}``.

Examples::

    # Full comparison over the default shapes
    python benchmarks/cute/compare_mamba2_chunk_state_backends.py

    # Single impl with a cold full autotune, JSON to stdout
    python benchmarks/cute/compare_mamba2_chunk_state_backends.py \\
        --impl helion-cute --shape b8_h80_g1_L4096_c256_p64_n128

    # Pinned configs for interleaved ABAB verify runs
    python benchmarks/cute/compare_mamba2_chunk_state_backends.py \\
        --impls helion-cute,mamba-triton --abab 3 --shapes ... \\
        --helion-config '{...}' --mamba-config '{...}'
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable

if TYPE_CHECKING:
    import torch

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_IMPLS = (
    "aten",
    "compile",
    "mamba-triton",
    "helion-triton",
    "helion-cute",
)
# Mamba2 model configs (headdim 64, dstate 128, chunk 256 unless noted):
# 2.7B (80 heads), 1.3B (64), 370M long context (32), Codestral-7B (128
# heads, 8 groups), a Nemotron-H-style chunk-128 8-group layer, a dstate-64
# (Zamba2-style) layer, a headdim-128 layer, and a single long sequence.
DEFAULT_SHAPES = (
    "b8_h80_g1_L4096_c256_p64_n128",
    "b8_h64_g1_L2048_c256_p64_n128",
    "b4_h32_g1_L16384_c256_p64_n128",
    "b4_h128_g8_L4096_c256_p64_n128",
    "b2_h128_g8_L8192_c128_p64_n128",
    "b8_h80_g1_L4096_c256_p64_n64",
    "b4_h64_g1_L4096_c256_p128_n128",
    "b1_h128_g1_L32768_c256_p64_n128",
)
_SHAPE_RE = re.compile(r"b(\d+)_h(\d+)_g(\d+)_L(\d+)_c(\d+)_p(\d+)_n(\d+)")


def parse_shape(shape: str) -> tuple[int, int, int, int, int, int, int]:
    match = _SHAPE_RE.fullmatch(shape)
    if match is None:
        raise SystemExit(f"bad shape {shape!r}")
    batch, nheads, ngroups, seqlen, chunk, headdim, dstate = map(int, match.groups())
    return batch, nheads, ngroups, seqlen, chunk, headdim, dstate


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


def traffic_bytes(shape: str, elem_bytes: int = 2) -> int:
    batch, nheads, ngroups, seqlen, chunk, headdim, dstate = parse_shape(shape)
    nchunks = seqlen // chunk
    x = batch * seqlen * nheads * headdim
    b = batch * seqlen * ngroups * dstate
    dt_da = 2 * batch * nheads * nchunks * chunk
    states = batch * nchunks * nheads * headdim * dstate
    return (x + b + dt_da + states) * elem_bytes


def _gbps(shape: str, ms: float) -> float:
    return traffic_bytes(shape) / (ms * 1e-3) / 1e9


def make_inputs(
    shape: str, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Same generator as benchmarks/cute/example_cases.py's catalogue cases."""
    import torch

    batch, nheads, ngroups, seqlen, chunk, headdim, dstate = parse_shape(shape)
    assert seqlen % chunk == 0
    torch.manual_seed(seed)
    nchunks = seqlen // chunk
    dtype = torch.bfloat16
    x = torch.randn(batch, seqlen, nheads, headdim, device="cuda", dtype=dtype)
    dt = torch.rand(batch, nheads, nchunks, chunk, device="cuda", dtype=dtype)
    dA_cumsum = -torch.rand_like(dt).cumsum(-1)
    B = torch.randn(batch, seqlen, ngroups, dstate, device="cuda", dtype=dtype)
    return B, x, dt, dA_cumsum


def chunk_state_grouped(
    B: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    dA_cumsum: torch.Tensor,
    compute_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """The chunk states as one batched matmul per (batch, chunk, group):
    ``[r * headdim, chunk] @ [chunk, dstate]`` over the ``r`` heads of the
    group.  Like the Helion kernel, the decay scale is computed in fp32 and
    the scaled x is rounded to ``compute_dtype`` (default: the input dtype)
    before the matmul."""
    import torch

    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, chunk = dt.shape
    ngroups, dstate = B.shape[2], B.shape[3]
    dtype = x.dtype if compute_dtype is None else compute_dtype
    dA = dA_cumsum.float()
    scale = torch.exp(dA[..., -1:] - dA) * dt.float()  # (b, h, c, l)
    xs = x.view(batch, nchunks, chunk, nheads, headdim).float()
    xs = (xs * scale.permute(0, 2, 3, 1)[..., None]).to(dtype)
    # (b, c, g, r * p, l) @ (b, c, g, l, n)
    xs = xs.view(batch, nchunks, chunk, ngroups, nheads // ngroups * headdim)
    xs = xs.permute(0, 1, 3, 4, 2)
    Bg = B.view(batch, nchunks, chunk, ngroups, dstate).permute(0, 1, 3, 2, 4)
    states = torch.matmul(xs, Bg.to(dtype))
    return states.view(batch, nchunks, nheads, headdim, dstate).to(x.dtype)


def reference_fp32(
    B: torch.Tensor, x: torch.Tensor, dt: torch.Tensor, dA_cumsum: torch.Tensor
) -> torch.Tensor:
    return chunk_state_grouped(
        B.float(), x.float(), dt.float(), dA_cumsum.float()
    ).float()


def _check(out: torch.Tensor, ref: torch.Tensor) -> dict[str, float]:
    import torch

    torch.testing.assert_close(out.float(), ref, rtol=1e-2, atol=1e-1)
    diff = (out.float() - ref).abs()
    return {"max_abs_err": float(diff.max()), "mean_abs_err": float(diff.mean())}


def _run_impl(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    sys.path.insert(0, str(REPO_ROOT))
    B, x, dt, dA_cumsum = make_inputs(args.shape, args.seed)
    kernel_args = (B, x, dt, dA_cumsum)
    config_used: object = None
    extra: dict[str, Any] = {}

    if args.impl == "aten":
        fn = lambda: chunk_state_grouped(*kernel_args)  # noqa: E731
    elif args.impl == "compile":
        compiled = torch.compile(chunk_state_grouped)
        fn = lambda: compiled(*kernel_args)  # noqa: E731
    elif args.impl == "mamba-triton":
        import triton

        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from mamba2_chunk_state_triton import (  # pyrefly: ignore [missing-import]
            _chunk_state_fwd_kernel,
        )
        from mamba2_chunk_state_triton import (  # pyrefly: ignore [missing-import]
            chunk_state_fwd,
        )

        if args.mamba_config:
            spec = json.loads(args.mamba_config)
            num_warps = spec.pop("num_warps")
            num_stages = spec.pop("num_stages")
            _chunk_state_fwd_kernel.configs = [
                triton.Config(spec, num_warps=num_warps, num_stages=num_stages)
            ]
        states = torch.empty(
            x.shape[0],
            dt.shape[2],
            x.shape[2],
            x.shape[3],
            B.shape[3],
            device=x.device,
            dtype=x.dtype,
        )
        t0 = time.time()
        fn = lambda: chunk_state_fwd(B, x, dt, dA_cumsum, states=states)  # noqa: E731
        fn()
        extra["autotune_s"] = round(time.time() - t0, 1)
        best = _chunk_state_fwd_kernel.best_config
        config_used = {
            **best.kwargs,
            "num_warps": best.num_warps,
            "num_stages": best.num_stages,
        }
    elif args.impl in ("helion-triton", "helion-cute"):
        backend = args.impl.split("-")[1]
        assert os.environ.get("HELION_BACKEND", backend) == backend, (
            "driver must set HELION_BACKEND to match --impl"
        )
        os.environ["HELION_BACKEND"] = backend
        from examples.mamba2_chunk_state import helion_mamba2_chunk_state_kernel

        import helion

        kernel = helion_mamba2_chunk_state_kernel
        kernel.settings.print_output_code = bool(args.print_code)
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
    else:
        raise SystemExit(f"unknown impl {args.impl!r}")

    out = fn()
    torch.cuda.synchronize()
    if not args.skip_correctness:
        extra.update(_check(out, reference_fp32(*kernel_args)))

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
        "shape": args.shape,
        "median_ms": stats["median_ms"],
        "best_ms": stats["best_ms"],
        "median_gbps": _gbps(args.shape, stats["median_ms"]),
        "best_gbps": _gbps(args.shape, stats["best_ms"]),
        "runs_ms": stats["runs_ms"],
        "cooldown": stats["cooldown"],
        "config": config_used,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        **_gpu_info(),
        **extra,
    }


def _spawn_impl(args: argparse.Namespace, impl: str, shape: str) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--impl",
        impl,
        "--shape",
        shape,
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
    for flag, value in (
        ("--helion-config", args.helion_config),
        ("--helion-config-triton", args.helion_config_triton),
        ("--mamba-config", args.mamba_config),
    ):
        if value:
            cmd.extend([flag, value])
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
        "shape": shape,
        "error": (proc.stderr or proc.stdout)[-2000:],
        "returncode": proc.returncode,
    }


def _driver(args: argparse.Namespace) -> None:
    impls = args.impls.split(",")
    shapes = args.shapes.split(",")
    results: list[dict[str, Any]] = []
    for shape in shapes:
        # --abab N alternates the impls N times on the same GPU so thermal
        # drift hits every impl equally.
        for rep in range(max(1, args.abab)):
            for impl in impls:
                record = _spawn_impl(args, impl, shape)
                record["tag"] = args.tag
                record["abab_round"] = rep
                results.append(record)
                if "error" in record:
                    print(f"[{shape}] {impl}: ERROR\n{record['error']}")
                else:
                    print(
                        f"[{shape}] {impl}: {record['median_ms']:.4f} ms "
                        f"{record['median_gbps']:.0f} GB/s",
                        flush=True,
                    )
                if args.output:
                    with open(args.output, "a") as f:
                        f.write(json.dumps(record) + "\n")
    _print_table(results, shapes, impls)


def _print_table(
    results: list[dict[str, Any]], shapes: list[str], impls: list[str]
) -> None:
    by_key: dict[tuple[str, str], list[float]] = {}
    for r in results:
        if "error" not in r:
            by_key.setdefault((r["shape"], r["impl"]), []).append(r["median_gbps"])
    header = ["shape".ljust(34)] + [i.rjust(14) for i in impls]
    print("\nGB/s (median):")
    print("".join(header))
    for shape in shapes:
        row = [shape.ljust(34)]
        for impl in impls:
            vals = by_key.get((shape, impl))
            row.append(
                f"{statistics.median(vals):.0f}".rjust(14) if vals else "ERR".rjust(14)
            )
        print("".join(row))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--impl", default=None, help="run a single impl in-process")
    parser.add_argument("--impls", default=",".join(DEFAULT_IMPLS))
    parser.add_argument("--shape", default=DEFAULT_SHAPES[0])
    parser.add_argument("--shapes", default=",".join(DEFAULT_SHAPES))
    parser.add_argument(
        "--autotune",
        default="force",
        choices=["force", "cache"],
        help="force = cold full autotune; cache = default bind path",
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
    parser.add_argument(
        "--mamba-config",
        default=None,
        help="JSON BLOCK_SIZE_M/N/K + num_warps/num_stages for mamba-triton",
    )
    parser.add_argument("--abab", type=int, default=1)
    parser.add_argument("--num-runs", type=int, default=5)
    parser.add_argument("--warmup-ms", type=int, default=100)
    parser.add_argument("--rep-ms", type=int, default=300)
    parser.add_argument("--cooldown-temp-c", type=float, default=50.0)
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
