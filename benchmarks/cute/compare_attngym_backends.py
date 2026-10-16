"""Compare attention backends on the attention-gym variant suite (B200+).

Variants (16 attention flavors from meta-pytorch/attention-gym) are defined in
``attngym_variants.py``; the Helion kernels (canonical flash envelope + inline
score/mask stages) live in ``attngym_kernels.py``. Methodology (steady-state
event-timed do_bench, thermal warmup, fresh-subprocess isolation per impl)
reuses compare_attention_backends helpers.

Use --suite figure for the 29-variant, 87-case suite in attngym_figure.py.
--variant accepts a row name or full case ID; --shape-id selects one case.

Impls:
    helion-cute     Helion kernel, HELION_BACKEND=cute (the optimization target)
    helion-triton   Helion kernel, default triton backend (secondary reference)
    flex-triton     torch.compile FlexAttention, Inductor Triton template
    flex-triton-ma  same, mode=max-autotune-no-cudagraphs (stronger baseline)
    flex-cute       FlexAttention experimental CuTeDSL FLASH backend
    sdpa            torch SDPA (only variants expressible: gqa-causal)
    fa4             FlashAttention-4 CuTe fwd (only variants expressible)

Every impl shares the same FLOP model (2*(Dqk+Dv)*unmasked_pairs) and is
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
from contextlib import nullcontext
from itertools import accumulate
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

from benchmarks.cute import attngym_variants as agv  # noqa: E402
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
FIGURE_IMPLS = tuple(impl for impl in ALL_IMPLS if impl != "helion-triton")
# Variants each restricted impl supports (flex/helion support everything).
SDPA_VARIANTS = {"gqa-causal"}
FA4_VARIANTS = {"gqa-causal"}
FIGURE_FA4_VARIANTS = {
    "noop",
    "causal",
    "gqa-causal",
    "mla-causal",
    "sliding-window",
    "softcap",
    "gemma2",
    "document",
}
_IMPL_ENV = {
    "helion-cute": {"HELION_BACKEND": "cute"},
    "helion-triton": {"HELION_BACKEND": "triton"},
}


def _suite_cases(suite: str) -> dict[str, agv.Variant]:
    if suite == "figure":
        from benchmarks.cute.attngym_figure import CASES

        return CASES
    return agv.VARIANTS


def _case_id(variant: agv.Variant) -> str:
    return variant.shape_id or variant.name


def _select_cases(
    suite: str, names: list[str] | None = None, shape_id: str | None = None
) -> list[agv.Variant]:
    cases = _suite_cases(suite)
    known = set(cases) | {variant.name for variant in cases.values()}
    if names and set(names) - known:
        raise ValueError(f"unknown variants: {set(names) - known}")
    selected = [
        variant
        for key, variant in cases.items()
        if (not names or variant.name in names or key in names)
        and (shape_id is None or key == shape_id)
    ]
    if not selected:
        raise ValueError("no matching attention-gym cases")
    return selected


def _case_metadata(variant: agv.Variant, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "variant": variant.name,
        "suite": variant.suite,
        "shape_id": _case_id(variant),
        "shape": {
            "b": variant.b,
            "hq": variant.hq,
            "hkv": variant.hkv,
            "m": variant.m,
            "n": variant.n,
            "d": variant.d,
            "dv": variant.value_dim,
            "dtype": str(variant.dtype).replace("torch.", ""),
        },
        "parameters": variant.extras,
        "input_seed": args.seed,
        "num_runs": args.num_runs,
        "warmup_ms": args.warmup_ms,
        "rep_ms": args.rep_ms,
        "cooldown_temp": args.cooldown_temp,
        "skip_correctness": bool(args.skip_correctness),
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
) -> tuple[str, float | None]:
    if variant.suite == "figure":
        if (
            out.shape != (variant.b, variant.hq, variant.m, variant.value_dim)
            or out.dtype != variant.dtype
        ):
            return "FAIL", None
        if not bool(torch.isfinite(out).all()) or not bool(torch.isfinite(ref).all()):
            return "FAIL", None
        diff = (out.float() - ref.float()).abs().max().item()
        return (
            "PASS"
            if torch.allclose(out.float(), ref.float(), atol=5e-2, rtol=2e-2)
            else "FAIL"
        ), diff
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
        **_case_metadata(variant, args),
        "impl": args.impl,
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
        **_case_metadata(variant, args),
        "impl": args.impl,
        "accuracy": "SKIP",
        "skip_reason": reason,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def _make_flex_inputs(
    variant: agv.Variant,
    device: torch.device,
    seed: int = 0,
    block_size: int | tuple[int, int] = 128,
) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], object, object]:
    q, k, v = agv.make_qkv(variant, device, seed=seed)
    score_mod, mask_mod = variant.make_mods(device)
    block_mask = None
    if mask_mod is not None:
        block_mask = agv.build_block_mask(variant, mask_mod, device, block_size)
    return (q, k, v), score_mod, block_mask


def _benchmark_flex(args: argparse.Namespace, variant: agv.Variant) -> dict[str, Any]:
    from torch.nn.attention.flex_attention import flex_attention

    device = torch.device("cuda")
    backend = "FLASH" if args.impl == "flex-cute" else "TRITON"
    (q, k, v), score_mod, block_mask = _make_flex_inputs(
        variant, device, args.seed, (256, 128) if backend == "FLASH" else 128
    )
    notes = [f"FlexAttention BACKEND={backend}"]
    if args.impl == "flex-cute":
        import compare_attention_backends as cab  # pyrefly: ignore [missing-import]

        cab._import_fa4()
    mode = None
    if args.impl == "flex-triton-ma":
        mode = "max-autotune-no-cudagraphs"
        notes.append("torch.compile mode=max-autotune-no-cudagraphs")
    compiled = torch.compile(flex_attention, fullgraph=True, dynamic=False, mode=mode)
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


def _prepare_cudnn_bias(bias: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    # Row constants cancel in softmax; centering preserves the influential
    # differences in positional biases before cuDNN's input-dtype conversion.
    row_max = bias.amax(dim=-1, keepdim=True)
    row_max = torch.where(torch.isfinite(row_max), row_max, 0.0)
    return (bias - row_max).to(dtype)


def _benchmark_sdpa(args: argparse.Namespace, variant: agv.Variant) -> dict[str, Any]:
    device = torch.device("cuda")
    bias, causal = None, True
    context = nullcontext()
    if variant.suite == "figure":
        from benchmarks.cute import attngym_figure as figure
        from torch.nn.attention import SDPBackend
        from torch.nn.attention import sdpa_kernel

        if variant.name not in figure.SDPA_VARIANTS:
            return _skipped(args, variant, "variant not expressible with SDPA")
        causal = variant.name in {"causal", "gqa-causal", "mla-causal"}
        bias = None if causal else figure.dense_bias(variant, device)
        if bias is not None:
            bias = _prepare_cudnn_bias(bias, variant.dtype)

        context = sdpa_kernel([SDPBackend.CUDNN_ATTENTION])
    elif variant.name not in SDPA_VARIANTS:
        return _skipped(args, variant, "variant not expressible with SDPA")
    q, k, v = agv.make_qkv(variant, device, seed=args.seed)

    def fn() -> torch.Tensor:
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=bias, is_causal=causal, enable_gqa=variant.enable_gqa
        )

    with context:
        return _run_impl(args, variant, fn, (q, k, v), notes=["torch SDPA"])


def _benchmark_fa4(args: argparse.Namespace, variant: agv.Variant) -> dict[str, Any]:
    supported = FIGURE_FA4_VARIANTS if variant.suite == "figure" else FA4_VARIANTS
    if variant.name not in supported:
        return _skipped(args, variant, "variant not expressible with FA4")
    import compare_attention_backends as cab  # pyrefly: ignore [missing-import]

    try:
        fc = cab._import_fa4()
    except Exception as e:
        return _skipped(args, variant, f"FA4 import failed: {e}")
    device = torch.device("cuda")
    q, k, v = agv.make_qkv(variant, device, seed=args.seed)
    # FA4 takes [B, S, H, D]
    qt = q.transpose(1, 2).contiguous()
    kt = k.transpose(1, 2).contiguous()
    vt = v.transpose(1, 2).contiguous()
    options: dict[str, Any] = {
        "softmax_scale": None,
        "causal": True,
        "return_lse": False,
    }
    if variant.suite == "figure":
        options.update(
            causal=bool(variant.extras.get("causal", False)),
            window_size=tuple(variant.extras.get("window_size", (None, None))),
            softcap=variant.extras.get("softcap", 0.0),
        )
    document = variant.suite == "figure" and variant.name == "document"
    if document:
        lengths = variant.extras["document_lengths"] * variant.b
        cu_seqlens = torch.tensor(
            [0, *accumulate(lengths)], dtype=torch.int32, device=device
        )
        qt = qt.reshape(-1, variant.hq, variant.d)
        kt = kt.reshape(-1, variant.hkv, variant.d)
        vt = vt.reshape(-1, variant.hkv, variant.value_dim)

    def fn() -> torch.Tensor:
        if document:
            out, _lse = fc.flash_attn_varlen_func(
                qt,
                kt,
                vt,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max(lengths),
                max_seqlen_k=max(lengths),
                **options,
            )
            out = out.reshape(variant.b, variant.m, variant.hq, variant.value_dim)
        else:
            out, _lse = fc.flash_attn_func(qt, kt, vt, **options)
        return out.transpose(1, 2)

    return _run_impl(args, variant, fn, (q, k, v), notes=["FA4 cute fwd"])


def _benchmark_helion(args: argparse.Namespace, variant: agv.Variant) -> dict[str, Any]:
    backend = "cute" if args.impl == "helion-cute" else "triton"
    assert os.environ.get("HELION_BACKEND", "triton") == backend, (
        f"run {args.impl} with HELION_BACKEND={backend} (subprocess mode does this)"
    )
    import helion

    device = torch.device("cuda")
    q, k, v = agv.make_qkv(variant, device, seed=args.seed)
    if variant.suite == "figure":
        from benchmarks.cute.attngym_flex import kernel_call_args

        kernel, call_args = kernel_call_args(variant, q, k, v, device)
    else:
        from attngym_kernels import kernel_call_args  # pyrefly: ignore [missing-import]

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
            save_path = save_path / f"{_case_id(variant)}.{args.impl}.json"
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
    variant = _suite_cases(args.suite)[args.variant]
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
        "--suite",
        args.suite,
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
            **_case_metadata(_suite_cases(args.suite)[variant], args),
            "impl": impl,
            "accuracy": "ERROR",
            "skip_reason": f"subprocess exit {proc.returncode}",
        }
    result = json.loads(out_path.read_text())
    out_path.unlink(missing_ok=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("legacy", "figure"), default="legacy")
    parser.add_argument("--shape-id", default=None, help="one full case ID")
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
    assert args.impl or args.impls, "--impl or --impls required"

    try:
        selected = _select_cases(
            args.suite,
            args.variant.split(",") if args.variant not in (None, "all") else None,
            args.shape_id,
        )
    except ValueError as e:
        parser.error(str(e))
    variants = [_case_id(variant) for variant in selected]
    if (
        args.impl == "all"
        or args.impls
        or (len(variants) > 1 and (args.variant is not None or args.suite == "figure"))
    ):
        impls = (
            args.impls.split(",")
            if args.impls
            else [args.impl]
            if args.impl not in (None, "all")
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
        if args.suite == "figure" and args.impl == "all" and not args.impls:
            impls.remove("helion-triton")
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

    assert (args.variant or args.shape_id) and args.impl, (
        "--variant and --impl required"
    )
    args.variant = variants[0]
    try:
        result = _run_single(args)
    except Exception as e:
        import traceback

        traceback.print_exc()
        result = {
            **_case_metadata(selected[0], args),
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
