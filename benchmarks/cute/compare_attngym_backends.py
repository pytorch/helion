"""Compare attention backends on the attention-gym variant suite (B200+).

Variants (16 attention flavors from meta-pytorch/attention-gym) are defined in
``attngym_variants.py``; the Helion kernels (canonical flash envelope + inline
score/mask stages) live in ``attngym_kernels.py``. Methodology (steady-state
event-timed do_bench, thermal warmup, fresh-subprocess isolation per impl)
reuses compare_attention_backends helpers.

Impls:
    helion-cute     Helion kernel, HELION_BACKEND=cute (the optimization target)
    helion-triton   Helion kernel, default triton backend (secondary reference)
    flex-triton     torch.compile FlexAttention, Inductor Triton template
    flex-triton-ma  same, mode=max-autotune-no-cudagraphs (stronger baseline)
    flex-cute       FlexAttention experimental CuTeDSL FLASH backend
    sdpa            torch SDPA (only variants expressible: gqa-causal)
    fa4             FlashAttention-4 CuTe fwd (only variants expressible)

Every impl of a variant shares the same FLOP model (4*D*unmasked_pairs) and is
checked against a chunked fp32 reference implementing FlexAttention semantics.

Examples:
    # One impl, one variant, JSON line appended to artifacts file
    CUDA_VISIBLE_DEVICES=6 python benchmarks/cute/compare_attngym_backends.py \
        --variant alibi --impl flex-triton --json out.jsonl

    # Cold full autotune for helion-cute
    CUDA_VISIBLE_DEVICES=6 HELION_AUTOTUNE_EFFORT=full HELION_SKIP_CACHE=1 \
        python benchmarks/cute/compare_attngym_backends.py \
        --variant alibi --impl helion-cute --force-autotune 1 --json out.jsonl

    # All impls for one variant (fresh subprocess per impl)
    CUDA_VISIBLE_DEVICES=6 python benchmarks/cute/compare_attngym_backends.py \
        --variant alibi --impl all --json out.jsonl
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

import attngym_variants as agv  # noqa: E402  # pyrefly: ignore [missing-import]
import torch  # noqa: E402

HELION_IMPLS = ("helion-cute", "helion-triton")
ALL_IMPLS = (
    "helion-triton",
    "helion-cute",
    "flex-triton",
    "flex-triton-ma",
    "flex-cute",
    "sdpa",
    "fa4",
)
# Variants each restricted impl supports (flex/helion support everything).
SDPA_VARIANTS = {"gqa-causal"}
FA4_VARIANTS = {"gqa-causal"}
_IMPL_ENV = {
    "helion-cute": {"HELION_BACKEND": "cute"},
    "helion-triton": {"HELION_BACKEND": "triton"},
}


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


def _check_accuracy(
    variant: agv.Variant, out: torch.Tensor, ref: torch.Tensor
) -> tuple[str, float]:
    diff = (out.float() - ref.float()).abs().max().item()
    tol = 3e-2 if variant.dtype is torch.bfloat16 else 2e-2
    return ("PASS" if diff < tol else "FAIL"), diff


def _result(
    args: argparse.Namespace,
    variant: agv.Variant,
    stats: dict[str, Any] | None,
    *,
    accuracy: str,
    max_abs_diff: float | None,
    config: str | None = None,
    notes: list[str] | None = None,
) -> dict[str, Any]:
    device = torch.device("cuda")
    flops = agv.variant_flops(variant, device)
    payload: dict[str, Any] = {
        "variant": variant.name,
        "impl": args.impl,
        "shape": {
            "b": variant.b,
            "hq": variant.hq,
            "hkv": variant.hkv,
            "m": variant.m,
            "n": variant.n,
            "d": variant.d,
            "dtype": str(variant.dtype).replace("torch.", ""),
        },
        "flops": flops,
        "gpu": torch.cuda.get_device_name(),
        "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
        "helion_commit": _git_commit(),
        "accuracy": accuracy,
        "max_abs_diff": max_abs_diff,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "autotune_effort": os.environ.get("HELION_AUTOTUNE_EFFORT", "default"),
        "skip_cache": os.environ.get("HELION_SKIP_CACHE", ""),
        "force_autotune": bool(args.force_autotune),
        "input_seed": args.seed,
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
    args: argparse.Namespace, variant: agv.Variant, reason: str
) -> dict[str, Any]:
    return {
        "variant": variant.name,
        "impl": args.impl,
        "accuracy": "SKIP",
        "skip_reason": reason,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def _make_flex_inputs(
    variant: agv.Variant, device: torch.device
) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], object, object]:

    q, k, v = agv.make_qkv(variant, device, seed=0)
    score_mod, mask_mod = variant.make_mods(device)
    block_mask = None
    if mask_mod is not None:
        block_mask = agv.build_block_mask(variant, mask_mod, device)
    return (q, k, v), score_mod, block_mask


def _benchmark_flex(args: argparse.Namespace, variant: agv.Variant) -> dict[str, Any]:
    from torch.nn.attention.flex_attention import flex_attention

    device = torch.device("cuda")
    (q, k, v), score_mod, block_mask = _make_flex_inputs(variant, device)
    backend = "FLASH" if args.impl == "flex-cute" else "TRITON"
    notes = [f"FlexAttention BACKEND={backend}"]
    if args.impl == "flex-cute":
        import compare_attention_backends as cab  # pyrefly: ignore [missing-import]

        cab._import_fa4()
    mode = None
    if args.impl == "flex-triton-ma":
        mode = "max-autotune-no-cudagraphs"
        notes.append("torch.compile mode=max-autotune-no-cudagraphs")
    compiled = torch.compile(flex_attention, fullgraph=True, mode=mode)
    kernel_options: Any = {"BACKEND": backend}

    def fn() -> torch.Tensor:
        return compiled(  # pyrefly: ignore [no-matching-overload]
            q,
            k,
            v,
            score_mod=score_mod,
            block_mask=block_mask,
            enable_gqa=variant.enable_gqa,
            kernel_options=kernel_options,
        )

    return _run_impl(args, variant, fn, (q, k, v), notes=notes)


def _benchmark_sdpa(args: argparse.Namespace, variant: agv.Variant) -> dict[str, Any]:
    if variant.name not in SDPA_VARIANTS:
        return _skipped(args, variant, "variant not expressible with SDPA")
    device = torch.device("cuda")
    q, k, v = agv.make_qkv(variant, device, seed=0)

    def fn() -> torch.Tensor:
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=True, enable_gqa=variant.enable_gqa
        )

    return _run_impl(args, variant, fn, (q, k, v), notes=["torch SDPA"])


def _benchmark_fa4(args: argparse.Namespace, variant: agv.Variant) -> dict[str, Any]:
    if variant.name not in FA4_VARIANTS:
        return _skipped(args, variant, "variant not expressible with FA4")
    import compare_attention_backends as cab  # pyrefly: ignore [missing-import]

    try:
        fc = cab._import_fa4()
    except Exception as e:
        return _skipped(args, variant, f"FA4 import failed: {e}")
    device = torch.device("cuda")
    q, k, v = agv.make_qkv(variant, device, seed=0)
    # FA4 takes [B, S, H, D]
    qt = q.transpose(1, 2).contiguous()
    kt = k.transpose(1, 2).contiguous()
    vt = v.transpose(1, 2).contiguous()

    def fn() -> torch.Tensor:
        out, _lse = fc.flash_attn_func(qt, kt, vt, softmax_scale=None, causal=True)
        return out.transpose(1, 2)

    return _run_impl(args, variant, fn, (q, k, v), notes=["FA4 cute fwd"])


def _benchmark_helion(args: argparse.Namespace, variant: agv.Variant) -> dict[str, Any]:
    backend = "cute" if args.impl == "helion-cute" else "triton"
    assert os.environ.get("HELION_BACKEND", "triton") == backend, (
        f"run {args.impl} with HELION_BACKEND={backend} (subprocess mode does this)"
    )
    from attngym_kernels import kernel_call_args  # pyrefly: ignore [missing-import]

    import helion

    device = torch.device("cuda")
    q, k, v = agv.make_qkv(variant, device, seed=0)
    kernel, call_args = kernel_call_args(variant.name, q, k, v, device)
    bound = kernel.bind(call_args)  # type: ignore[attr-defined]
    notes = [f"HELION_BACKEND={backend}"]
    if args.config:
        config = helion.Config.from_json(Path(args.config).read_text())
        bound.set_config(config)
        notes.append(f"pinned config from {args.config}")
    else:
        t0 = time.time()
        config = bound.autotune(call_args, force=bool(args.force_autotune))
        notes.append(f"autotune wall {time.time() - t0:.0f}s")
    config_json = config.to_json()
    if args.save_config:
        save_path = Path(args.save_config)
        if save_path.is_dir():
            save_path = save_path / f"{variant.name}.{args.impl}.json"
        save_path.write_text(config_json)

    def fn() -> torch.Tensor:
        return bound(*call_args)

    return _run_impl(args, variant, fn, (q, k, v), config=config_json, notes=notes)


def _run_impl(
    args: argparse.Namespace,
    variant: agv.Variant,
    fn: Callable[[], torch.Tensor],
    qkv: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    config: str | None = None,
    notes: list[str] | None = None,
) -> dict[str, Any]:
    bench_steady, _gpu_warmup = _bench_helpers()
    accuracy = "SKIP"
    max_abs_diff = None
    if not args.skip_correctness:
        q, k, v = qkv
        score_mod, mask_mod = variant.make_mods(q.device)
        ref = agv.reference_output(variant, q, k, v, score_mod, mask_mod)
        out = fn()
        accuracy, max_abs_diff = _check_accuracy(variant, out, ref)
        del ref, out
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
        max_abs_diff=max_abs_diff,
        config=config,
        notes=notes,
    )


def _run_single(args: argparse.Namespace) -> dict[str, Any]:
    variant = agv.VARIANTS[args.variant]
    if args.impl in HELION_IMPLS:
        result = _benchmark_helion(args, variant)
    elif args.impl in ("flex-triton", "flex-triton-ma", "flex-cute"):
        result = _benchmark_flex(args, variant)
    elif args.impl == "sdpa":
        result = _benchmark_sdpa(args, variant)
    elif args.impl == "fa4":
        result = _benchmark_fa4(args, variant)
    else:
        raise SystemExit(f"unknown impl {args.impl!r}")
    return result


def _spawn(args: argparse.Namespace, variant: str, impl: str) -> dict[str, Any]:
    """Run one (variant, impl) in a fresh subprocess; return its result."""
    out_path = Path(args.json or "attngym_results.jsonl").with_suffix(
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
    if args.config:
        cmd += ["--config", args.config]
    if args.save_config:
        cmd += ["--save-config", args.save_config]
    env = dict(os.environ)
    env.pop("HELION_BACKEND", None)
    env.update(_IMPL_ENV.get(impl, {}))
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
    parser.add_argument("--config", default=None, help="path to Config JSON to pin")
    parser.add_argument("--save-config", default=None, help="write winning Config JSON")
    args = parser.parse_args()

    variants = list(agv.VARIANTS) if args.variant in (None, "all") else [args.variant]
    if args.impl == "all" or args.impls:
        impls = (
            args.impls.split(",")
            if args.impls
            else [
                "helion-cute",
                "helion-triton",
                "flex-triton",
                "flex-triton-ma",
                "flex-cute",
                "sdpa",
                "fa4",
            ]
        )
        results = []
        for variant in variants:
            for impl in impls:
                result = _spawn(args, variant, impl)
                results.append(result)
                if args.json:
                    with open(args.json, "a") as f:
                        f.write(json.dumps(result) + "\n")
                med = result.get("median_ms")
                tflops = result.get("median_tflops")
                print(
                    f"{variant:16s} {impl:15s} acc={result.get('accuracy'):5s} "
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
