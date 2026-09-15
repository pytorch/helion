"""Compare attention-BACKWARD backends on B200+.

Variants (shapes / causal / GQA / dtype) are defined in
``attnbwd_variants.py``; the Helion kernels live in ``attnbwd_kernels.py``.
Methodology (steady-state event-timed do_bench, thermal warmup,
fresh-subprocess isolation per impl) reuses compare_attention_backends
helpers. Every impl times the backward-only closure: forward runs once
outside the timed region, then ``torch.autograd.grad(out, (q, k, v), do,
retain_graph=True)`` (baselines) or the Helion backward wrapper (which
includes the delta-preprocess kernel).

Impls:
    helion-cute     Helion kernels, HELION_BACKEND=cute (optimization target)
    helion-triton   Helion kernels, default triton backend (reference)
    fa4             flash-attention main CuTe DSL backward (flash_bwd_sm100)
    cudnn           torch SDPA, cuDNN backend
    flex            torch.compile FlexAttention backward (triton)
    flex-ma         same, mode=max-autotune-no-cudagraphs

Accuracy: dq/dk/dv vs a chunked fp32 reference, max relative-to-peak error.

Examples:
    CUDA_VISIBLE_DEVICES=6 python benchmarks/cute/compare_attnbwd_backends.py \
        --variant d64-causal --impl fa4 --json out.jsonl

    # Cold full autotune for helion-cute
    CUDA_VISIBLE_DEVICES=6 HELION_AUTOTUNE_EFFORT=full HELION_SKIP_CACHE=1 \
        python benchmarks/cute/compare_attnbwd_backends.py \
        --variant d64-causal --impl helion-cute --force-autotune 1 --json out.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "benchmarks" / "cute"))

import attnbwd_variants as abv  # noqa: E402  # pyrefly: ignore [missing-import]
import torch  # noqa: E402

HELION_IMPLS = ("helion-cute", "helion-triton")
ALL_IMPLS = ("helion-triton", "helion-cute", "fa4", "cudnn", "flex", "flex-ma")
_IMPL_ENV = {
    "helion-cute": {"HELION_BACKEND": "cute"},
    "helion-triton": {"HELION_BACKEND": "triton"},
}
_FA_MAIN_ROOT = "/tmp/fa-main"

Grads = tuple[torch.Tensor, ...]


def _bench_helpers() -> tuple[Callable[..., Any], Callable[..., Any]]:
    import compare_attention_backends as cab  # pyrefly: ignore [missing-import]

    return cab._bench_steady, cab._gpu_warmup


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def _result(
    args: argparse.Namespace,
    variant: abv.Variant,
    stats: dict[str, Any] | None,
    *,
    accuracy: str,
    max_rel_diff: float | None,
    config: str | None = None,
    notes: list[str] | None = None,
) -> dict[str, Any]:
    flops = abv.variant_flops(variant)
    payload: dict[str, Any] = {
        "variant": variant.name,
        "impl": args.impl,
        "shape": {
            "b": variant.b,
            "hq": variant.hq,
            "hkv": variant.hkv,
            "s": variant.s,
            "d": variant.d,
            "causal": variant.causal,
            "dtype": str(variant.dtype).replace("torch.", ""),
        },
        "flops": flops,
        "gpu": torch.cuda.get_device_name(),
        "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
        "helion_commit": _git_commit(),
        "accuracy": accuracy,
        "max_rel_diff": max_rel_diff,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "autotune_effort": os.environ.get("HELION_AUTOTUNE_EFFORT", "default"),
        "skip_cache": os.environ.get("HELION_SKIP_CACHE", ""),
        "force_autotune": bool(args.force_autotune),
        "input_seed": args.seed,
        "helion_mode": args.helion_mode,
    }
    if stats is not None:
        payload.update(
            {
                "best_ms": stats["best_ms"],
                "median_ms": stats["median_ms"],
                "mean_ms": stats["mean_ms"],
                "std_ms": stats["std_ms"],
                "runs_ms": stats["runs_ms"],
                "best_tflops": flops / stats["best_ms"] / 1e9,
                "median_tflops": flops / stats["median_ms"] / 1e9,
            }
        )
    if config is not None:
        payload["config"] = config
    if notes:
        payload["notes"] = notes
    return payload


def _skipped(
    args: argparse.Namespace, variant: abv.Variant, reason: str
) -> dict[str, Any]:
    return {
        "variant": variant.name,
        "impl": args.impl,
        "accuracy": "SKIP",
        "skip_reason": reason,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def _run_impl(
    args: argparse.Namespace,
    variant: abv.Variant,
    fn: Callable[[], Grads],
    ref_grads: Grads,
    *,
    config: str | None = None,
    notes: list[str] | None = None,
) -> dict[str, Any]:
    bench_steady, _gpu_warmup = _bench_helpers()
    accuracy = "SKIP"
    max_rel = None
    if not args.skip_correctness:
        got = fn()
        accuracy, max_rel = abv.check_grads(variant, got, ref_grads)
        del got
        torch.cuda.empty_cache()
    stats = bench_steady(
        fn,
        num_runs=args.num_runs,
        warmup_ms=args.warmup_ms,
        rep_ms=args.rep_ms,
        cooldown_max_temp_c=args.cooldown_temp if args.cooldown_temp > 0 else None,
    )
    return _result(
        args,
        variant,
        stats,
        accuracy=accuracy,
        max_rel_diff=max_rel,
        config=config,
        notes=notes,
    )


def _prepared(
    variant: abv.Variant,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple, Grads]:
    device = torch.device("cuda")
    q, k, v, do = abv.make_inputs(variant, device, seed=0)
    o, lse2, dq_ref, dk_ref, dv_ref = abv.prepare_reference(variant, q, k, v, do)
    return q, k, v, do, (o, lse2), (dq_ref, dk_ref, dv_ref)


def _benchmark_helion(args: argparse.Namespace, variant: abv.Variant) -> dict[str, Any]:
    backend = "cute" if args.impl == "helion-cute" else "triton"
    assert os.environ.get("HELION_BACKEND", "triton") == backend, (
        f"run {args.impl} with HELION_BACKEND={backend} (subprocess mode does this)"
    )
    import attnbwd_kernels as abk  # pyrefly: ignore [missing-import]

    import helion

    q, k, v, do, (o, lse2), ref_grads = _prepared(variant)
    sm_scale = variant.d**-0.5
    # Default mode per backend: the cute backend's flash-bwd emitter matches
    # the fused atomic-dQ kernel; triton runs the no-atomics split pair.
    mode = args.helion_mode or ("fused" if backend == "cute" else "split")
    calls = abk.bwd_kernel_calls(q, k, v, o, lse2, do, sm_scale, variant.causal, mode)
    notes = [f"HELION_BACKEND={backend}", f"mode={mode}"]
    pinned: dict[str, Any] = {}
    if args.config:
        pinned = json.loads(Path(args.config).read_text())
        notes.append(f"pinned config from {args.config}")
    configs: dict[str, Any] = {}
    for name, kernel, call_args in calls:
        bound = kernel.bind(call_args)  # type: ignore[attr-defined]
        if pinned:
            config = helion.Config(**pinned[name])
            bound.set_config(config)
        else:
            t0 = time.time()
            config = bound.autotune(call_args, force=bool(args.force_autotune))
            notes.append(f"autotune[{name}] wall {time.time() - t0:.0f}s")
        configs[name] = config.to_json() if hasattr(config, "to_json") else config
    config_blob = json.dumps({name: json.loads(c) for name, c in configs.items()})
    if args.save_config:
        save_path = Path(args.save_config)
        if save_path.is_dir():
            save_path = save_path / f"{variant.name}.{args.impl}.json"
        save_path.write_text(config_blob)

    def fn() -> Grads:
        return abk.helion_attention_backward(
            q, k, v, o, lse2, do, sm_scale, variant.causal, mode
        )

    return _run_impl(args, variant, fn, ref_grads, config=config_blob, notes=notes)


def _benchmark_fa4(args: argparse.Namespace, variant: abv.Variant) -> dict[str, Any]:
    import compare_attention_backends as cab  # pyrefly: ignore [missing-import]

    try:
        fc = cab._import_fa4()
    except Exception as e:
        return _skipped(args, variant, f"FA4 import failed: {e}")
    q, k, v, do, _, ref_grads = _prepared(variant)
    qt = q.transpose(1, 2).contiguous().requires_grad_(True)
    kt = k.transpose(1, 2).contiguous().requires_grad_(True)
    vt = v.transpose(1, 2).contiguous().requires_grad_(True)
    dot = do.transpose(1, 2).contiguous()
    out = fc.flash_attn_func(qt, kt, vt, causal=variant.causal)
    if isinstance(out, (tuple, list)):
        out = out[0]
    ref_t = tuple(g.transpose(1, 2) for g in ref_grads)

    def fn() -> Grads:
        return torch.autograd.grad(out, (qt, kt, vt), dot, retain_graph=True)

    return _run_impl(
        args,
        variant,
        fn,
        ref_t,  # type: ignore[arg-type]
        notes=[
            f"flash-attention cute bwd, root={os.environ.get('HELION_FA4_ROOT', '')}"
        ],
    )


def _benchmark_cudnn(args: argparse.Namespace, variant: abv.Variant) -> dict[str, Any]:
    from torch.nn.attention import SDPBackend
    from torch.nn.attention import sdpa_kernel

    q, k, v, do, _, ref_grads = _prepared(variant)
    qr = q.detach().requires_grad_(True)
    kr = k.detach().requires_grad_(True)
    vr = v.detach().requires_grad_(True)
    enable_gqa = variant.hq != variant.hkv
    try:
        with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
            out = torch.nn.functional.scaled_dot_product_attention(
                qr, kr, vr, is_causal=variant.causal, enable_gqa=enable_gqa
            )
    except Exception as e:
        return _skipped(args, variant, f"cudnn sdpa failed: {e}")

    def fn() -> Grads:
        return torch.autograd.grad(out, (qr, kr, vr), do, retain_graph=True)

    return _run_impl(args, variant, fn, ref_grads, notes=["torch SDPA cudnn bwd"])


def _benchmark_flex(args: argparse.Namespace, variant: abv.Variant) -> dict[str, Any]:
    from torch.nn.attention.flex_attention import create_block_mask
    from torch.nn.attention.flex_attention import flex_attention

    q, k, v, do, _, ref_grads = _prepared(variant)
    qr = q.detach().requires_grad_(True)
    kr = k.detach().requires_grad_(True)
    vr = v.detach().requires_grad_(True)
    mode = None
    notes = ["FlexAttention triton bwd"]
    if args.impl == "flex-ma":
        mode = "max-autotune-no-cudagraphs"
        notes.append("torch.compile mode=max-autotune-no-cudagraphs")
    compiled = torch.compile(flex_attention, fullgraph=True, mode=mode)
    block_mask = None
    if variant.causal:

        def causal_mask(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ) -> torch.Tensor:
            return q_idx >= kv_idx

        block_mask = create_block_mask(
            causal_mask, None, None, variant.s, variant.s, device=str(q.device)
        )
    out = compiled(
        qr,
        kr,
        vr,
        block_mask=block_mask,
        enable_gqa=variant.hq != variant.hkv,
    )

    def fn() -> Grads:
        return torch.autograd.grad(out, (qr, kr, vr), do, retain_graph=True)

    return _run_impl(args, variant, fn, ref_grads, notes=notes)


def _run_single(args: argparse.Namespace) -> dict[str, Any]:
    variant = abv.VARIANTS[args.variant]
    if args.impl in HELION_IMPLS:
        return _benchmark_helion(args, variant)
    if args.impl == "fa4":
        return _benchmark_fa4(args, variant)
    if args.impl == "cudnn":
        return _benchmark_cudnn(args, variant)
    if args.impl in ("flex", "flex-ma"):
        return _benchmark_flex(args, variant)
    raise SystemExit(f"unknown impl {args.impl!r}")


def _spawn(args: argparse.Namespace, variant: str, impl: str) -> dict[str, Any]:
    """Run one (variant, impl) in a fresh subprocess; return its result."""
    out_path = Path(args.json or "attnbwd_results.jsonl").with_suffix(
        f".{variant}.{impl}.tmp.json"
    )
    out_path.unlink(missing_ok=True)
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--variant",
        variant,
        "--impl",
        impl,
        "--json",
        str(out_path),
        "--num-runs",
        str(args.num_runs),
        "--warmup-ms",
        str(args.warmup_ms),
        "--rep-ms",
        str(args.rep_ms),
        "--cooldown-temp",
        str(args.cooldown_temp),
        "--seed",
        str(args.seed),
        "--force-autotune",
        str(int(args.force_autotune)),
        "--skip-correctness",
        str(int(args.skip_correctness)),
    ]
    if args.helion_mode:
        cmd += ["--helion-mode", args.helion_mode]
    if args.config:
        cmd += ["--config", args.config]
    if args.save_config:
        cmd += ["--save-config", args.save_config]
    env = dict(os.environ)
    env.pop("HELION_BACKEND", None)
    env.update(_IMPL_ENV.get(impl, {}))
    if impl == "fa4" and "HELION_FA4_ROOT" not in env and Path(_FA_MAIN_ROOT).is_dir():
        env["HELION_FA4_ROOT"] = _FA_MAIN_ROOT
    print(f"=== spawn {variant} / {impl} ===", flush=True)
    proc = subprocess.run(cmd, env=env, check=False)
    if proc.returncode != 0:
        return {
            "variant": variant,
            "impl": impl,
            "accuracy": "ERROR",
            "skip_reason": f"subprocess exit {proc.returncode}",
        }
    result = json.loads(out_path.read_text())
    out_path.unlink(missing_ok=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default=None, help="variant name or 'all'")
    parser.add_argument("--impl", default=None, help="impl name or 'all'")
    parser.add_argument("--impls", default=None, help="comma list for --variant all")
    parser.add_argument("--json", default=None, help="append result JSON lines here")
    parser.add_argument("--num-runs", type=int, default=5)
    parser.add_argument("--warmup-ms", type=int, default=1000)
    parser.add_argument("--rep-ms", type=int, default=500)
    parser.add_argument("--cooldown-temp", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force-autotune", type=int, default=0)
    parser.add_argument("--skip-correctness", type=int, default=0)
    parser.add_argument("--helion-mode", default=None, choices=["split", "fused"])
    parser.add_argument("--config", default=None, help="path to config-dict JSON")
    parser.add_argument("--save-config", default=None, help="write winning configs")
    args = parser.parse_args()

    variants = list(abv.VARIANTS) if args.variant in (None, "all") else [args.variant]
    if args.impl == "all" or args.impls:
        impls = args.impls.split(",") if args.impls else list(ALL_IMPLS)
        for variant in variants:
            for impl in impls:
                result = _spawn(args, variant, impl)
                if args.json:
                    with open(args.json, "a") as f:
                        f.write(json.dumps(result) + "\n")
                med = result.get("median_ms")
                tflops = result.get("median_tflops")
                print(
                    f"{variant:14s} {impl:14s} acc={result.get('accuracy'):5s} "
                    f"med={med if med is None else f'{med:.4f}'}ms "
                    f"tflops={tflops if tflops is None else f'{tflops:.1f}'}",
                    flush=True,
                )
        return

    assert args.variant and args.impl, "--variant and --impl required"
    try:
        result = _run_single(args)
    except Exception as e:
        import traceback

        traceback.print_exc()
        result = {
            "variant": args.variant,
            "impl": args.impl,
            "accuracy": "ERROR",
            "skip_reason": f"{type(e).__name__}: {str(e)[:300]}",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
    line = json.dumps(result)
    if args.json:
        if ".tmp." in args.json:
            Path(args.json).write_text(line + "\n")
        else:
            with open(args.json, "a") as f:
                f.write(line + "\n")
    print(line, flush=True)


if __name__ == "__main__":
    main()
