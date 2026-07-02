"""Compare attention backends per-shape on B200+ hardware.

Mirrors the methodology of ``compare_matmul_backends.py`` (steady-state
do_bench, thermal warmup, fresh-subprocess isolation per impl) for scaled
dot-product attention.

Impls:
    sdpa          torch.nn.functional.scaled_dot_product_attention -- the gold
                  reference baseline (B200 fused flash). Used as the
                  correctness reference for the other impls.
    fa4           FlashAttention-4 (the CuTe-DSL fwd, flash_attn.cute) -- the
                  upstream design our cute flash kernel is modelled on, so the
                  most meaningful throughput target. Loaded from HELION_FA4_ROOT,
                  an existing tritonbench flash-attention submodule, or, when
                  HELION_FA4_AUTO_DOWNLOAD=1, an auto-cloned
                  benchmarks/flash-attention checkout with a small CUDA-12.9 ->
                  CUDA-13 cutlass-ABI shim (see _import_fa4);
                  supports causal too.
    helion-triton examples.attention attention kernels with the DEFAULT
                  (triton) backend.
    helion-cute   examples.attention attention kernels with
                  HELION_BACKEND=cute. By default this uses output-only
                  variants for dense, causal, and biased attention so Helion
                  does not compute an aux output that SDPA/FA4 omit.
                  NOTE: the cute attention path is being fixed in parallel and
                  may currently be numerically WRONG or slow. The harness flags
                  a cute correctness mismatch (accuracy=FAIL) but still reports
                  timing -- it never crashes on cute's current state.
    flexattention torch.nn.attention.flex_attention.flex_attention under
                  torch.compile(fullgraph=True). Causal uses a BlockMask;
                  biased uses score_mod.

Each impl runs in a fresh subprocess (so the HELION_BACKEND env mutation and
example imports do not leak between impls), with steady-state methodology
(10 s thermal warmup, do_bench warmup=1 s + rep=500 ms, 5 runs) and reports
best plus mom-median ms/TFLOP/s plus speedup vs sdpa.

Default is non-causal. Use ``--causal 1`` to benchmark causal kernels.

Examples:

    # Single-shape A/B across all impls
    CUDA_VISIBLE_DEVICES=6 python benchmarks/cute/compare_attention_backends.py \\
        --impl all --z 2 --h 8 --seq-len 512 --head-dim 64 --dtype float16

    # One impl, JSON line (used by --impl all subprocess collection)
    CUDA_VISIBLE_DEVICES=6 python benchmarks/cute/compare_attention_backends.py \\
        --impl sdpa --z 2 --h 32 --seq-len 1024 --head-dim 64 --json

    # Representative shape sweep -> Markdown table
    CUDA_VISIBLE_DEVICES=6 python benchmarks/cute/compare_attention_backends.py \\
        --all-shapes --output attention_sweep.md \\
        --csv-output attention_sweep.csv --plot-output attention_sweep.png

    # Variant-focused sweep -> compact table across dense/causal/biased
    CUDA_VISIBLE_DEVICES=6 HELION_AUTOTUNE_EFFORT=full \\
        python benchmarks/cute/compare_attention_backends.py \\
        --all-shapes --shape-suite variants --helion-force-autotune 0 \\
        --stream-subprocesses \\
        --csv-output attention_variants.csv --plot-output attention_variants.png
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time
import types
from typing import Any
from typing import Callable
from typing import Iterator
from typing import cast

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_IMPLS = ("helion-triton", "helion-cute", "flexattention", "sdpa", "fa4")
ALL_IMPLS = ("helion-triton", "helion-cute", "flexattention", "sdpa", "fa4")
HELION_IMPLS = ("helion-cute", "helion-triton")

_FA4_REPO = "https://github.com/Dao-AILab/flash-attention.git"
_FA4_DEFAULT_REF = "v2.8.3"
_FA4_ROOT_ENV = "HELION_FA4_ROOT"
_FA4_REF_ENV = "HELION_FA4_REF"
_FA4_AUTO_DOWNLOAD_ENV = "HELION_FA4_AUTO_DOWNLOAD"
_TRITONBENCH_ROOT = REPO_ROOT / "benchmarks" / "tritonbench"
_FA4_TRITONBENCH_ROOT = _TRITONBENCH_ROOT / "submodules" / "flash-attention"
_FA4_STANDALONE_ROOT = REPO_ROOT / "benchmarks" / "flash-attention"

# (z, h, seq_len, head_dim, dtype, causal, biased)
_REPRESENTATIVE_SHAPES: tuple[tuple[int, int, int, int, str, int, int], ...] = (
    (1, 4, 512, 64, "float16", 0, 0),
    (2, 8, 512, 64, "float16", 0, 0),
    (2, 32, 1024, 64, "float16", 0, 0),
    (2, 32, 2048, 64, "float16", 0, 0),
    (4, 32, 4096, 128, "bfloat16", 0, 0),
    (8, 32, 8192, 128, "bfloat16", 0, 0),
    # causal variants
    (2, 32, 1024, 64, "float16", 1, 0),
    (2, 32, 4096, 64, "float16", 1, 0),
    (4, 32, 4096, 128, "bfloat16", 1, 0),
    # biased variant
    (2, 32, 1024, 64, "float16", 0, 1),
)

_VARIANT_SHAPES: tuple[tuple[int, int, int, int, str, int, int], ...] = (
    (2, 32, 2048, 64, "float16", 0, 0),
    (2, 32, 4096, 64, "float16", 1, 0),
    (1, 2, 128, 64, "float16", 0, 1),
)

_DENSE_CAUSAL8_SHAPES: tuple[tuple[int, int, int, int, str, int, int], ...] = (
    (2, 32, 32768, 64, "float16", 0, 0),
    (2, 32, 65536, 64, "float16", 0, 0),
    (2, 32, 131072, 64, "float16", 0, 0),
    (2, 32, 262144, 64, "float16", 0, 0),
    (2, 32, 65536, 64, "float16", 1, 0),
    (2, 32, 131072, 64, "float16", 1, 0),
    (2, 32, 262144, 64, "float16", 1, 0),
    (2, 32, 524288, 64, "float16", 1, 0),
)

_SHAPE_SUITES = {
    "representative": _REPRESENTATIVE_SHAPES,
    "variants": _VARIANT_SHAPES,
    "dense_causal8": _DENSE_CAUSAL8_SHAPES,
}

_DISPLAY_IMPLS = ("helion-triton", "helion-cute", "flexattention", "sdpa", "fa4")
_IMPL_LABELS = {
    "helion-triton": "Helion+Triton",
    "helion-cute": "Helion+CuTe",
    "flexattention": "FlexAttention",
    "sdpa": "torch SDPA",
    "fa4": "FA4",
}
_IMPL_KEYS = {
    "helion-triton": "helion_triton",
    "helion-cute": "helion_cute",
    "flexattention": "flexattention",
    "sdpa": "torch_sdpa",
    "fa4": "fa4",
}

_ALLOWED_PHYSICAL_GPUS_ENV = "HELION_BENCHMARK_ALLOWED_PHYSICAL_GPUS"


def _parse_key_value(value: str) -> tuple[str, str]:
    key, sep, raw_value = value.partition("=")
    if not sep or not key:
        raise argparse.ArgumentTypeError(f"expected KEY=VALUE, got {value!r}")
    return key, raw_value


def _parse_config_override(value: str) -> tuple[str, object]:
    key, raw_value = _parse_key_value(value)
    try:
        parsed_value = json.loads(raw_value)
    except json.JSONDecodeError:
        parsed_value = raw_value
    return key, parsed_value


def _dtype_from_name(name: str) -> torch.dtype:
    return {"float16": torch.float16, "bfloat16": torch.bfloat16}[name]


def _attention_flops(args: argparse.Namespace) -> float:
    """Attention FLOPs = 4 * z * h * seq^2 * head_dim (x0.5 if causal)."""
    flops = 4.0 * args.z * args.h * args.seq_len * args.seq_len * args.head_dim
    if args.causal:
        flops *= 0.5
    return flops


def _tflops(args: argparse.Namespace, ms: float) -> float:
    return _attention_flops(args) / (ms * 1e9)


def _env_flag(name: str, *, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in ("1", "true", "on", "yes")


def _valid_fa4_root(root: Path) -> bool:
    return (root / "flash_attn" / "cute").is_dir()


def _fa4_ref() -> str:
    return os.environ.get(_FA4_REF_ENV, _FA4_DEFAULT_REF)


def _git_commit(root: Path, rev: str) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--verify", f"{rev}^{{commit}}"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError:
        return None
    return proc.stdout.strip()


def _fa4_checkout_matches_ref(root: Path, ref: str) -> bool:
    head = _git_commit(root, "HEAD")
    target = _git_commit(root, ref)
    return head is not None and head == target


def _run_fa4_setup(cmd: list[str], *, cwd: Path | None = None) -> None:
    try:
        subprocess.run(cmd, cwd=cwd, check=True)
    except subprocess.CalledProcessError as exc:
        command = " ".join(cmd)
        raise SystemExit(f"failed to set up FA4 checkout with `{command}`") from exc


def _checkout_fa4_ref(root: Path, ref: str) -> None:
    _run_fa4_setup(["git", "fetch", "--tags", "origin"], cwd=root)
    checkout_ref = ref
    if _git_commit(root, ref) is None:
        _run_fa4_setup(["git", "fetch", "origin", ref], cwd=root)
        if _git_commit(root, ref) is None:
            checkout_ref = "FETCH_HEAD"
    _run_fa4_setup(["git", "checkout", checkout_ref], cwd=root)
    if not _valid_fa4_root(root):
        raise SystemExit(
            f"{root} is checked out at {ref}, but flash_attn/cute is missing; "
            f"try another {_FA4_REF_ENV} value"
        )


def _fa4_root_at_ref(
    root: Path,
    ref: str,
    *,
    auto_checkout: bool,
    label: str,
) -> Path | None:
    if not _valid_fa4_root(root):
        return None
    if auto_checkout:
        try:
            _checkout_fa4_ref(root, ref)
        except SystemExit as exc:
            print(f"Skipping {label}: {exc}", file=sys.stderr)
            return None
        return root
    if _fa4_checkout_matches_ref(root, ref):
        return root
    print(
        f"Skipping {label}: {root} is not checked out at {ref}",
        file=sys.stderr,
    )
    return None


def _ensure_fa4_checkout(root: Path) -> Path:
    ref = _fa4_ref()
    if _valid_fa4_root(root):
        _checkout_fa4_ref(root, ref)
        return root
    if root.exists():
        raise SystemExit(
            f"{root} exists but does not look like a flash-attention checkout; "
            f"set {_FA4_ROOT_ENV}=<path> or remove the incomplete directory"
        )
    root.parent.mkdir(parents=True, exist_ok=True)
    print(f"Cloning FlashAttention for FA4 benchmark into {root}", file=sys.stderr)
    _run_fa4_setup(["git", "clone", "--filter=blob:none", _FA4_REPO, str(root)])
    _checkout_fa4_ref(root, ref)
    return root


def _ensure_fa4_tritonbench_submodule() -> Path | None:
    ref = _fa4_ref()
    if not (_TRITONBENCH_ROOT / ".git").exists():
        return None
    print(
        "Initializing TritonBench flash-attention submodule for FA4 benchmark",
        file=sys.stderr,
    )
    try:
        _run_fa4_setup(
            [
                "git",
                "submodule",
                "update",
                "--init",
                "--recursive",
                "submodules/flash-attention",
            ],
            cwd=_TRITONBENCH_ROOT,
        )
    except SystemExit as exc:
        print(f"Skipping TritonBench FA4 submodule: {exc}", file=sys.stderr)
        return None
    tritonbench_root = _fa4_root_at_ref(
        _FA4_TRITONBENCH_ROOT,
        ref,
        auto_checkout=True,
        label="TritonBench FA4 submodule",
    )
    if tritonbench_root is not None:
        return tritonbench_root
    print(
        f"Skipping TritonBench FA4 submodule: {_FA4_TRITONBENCH_ROOT} "
        "is missing flash_attn/cute after submodule update",
        file=sys.stderr,
    )
    return None


def _resolve_fa4_root() -> Path:
    for env_name in (_FA4_ROOT_ENV, "FLASH_ATTENTION_ROOT"):
        root_str = os.environ.get(env_name)
        if root_str:
            root = Path(root_str).expanduser().resolve()
            if _valid_fa4_root(root):
                return root
            raise SystemExit(f"{env_name}={root} does not contain flash_attn/cute")
    auto_download = _env_flag(_FA4_AUTO_DOWNLOAD_ENV, default=False)
    ref = _fa4_ref()
    tritonbench_root = _fa4_root_at_ref(
        _FA4_TRITONBENCH_ROOT,
        ref,
        auto_checkout=auto_download,
        label="TritonBench FA4 submodule",
    )
    if tritonbench_root is not None:
        return tritonbench_root
    if _valid_fa4_root(_FA4_STANDALONE_ROOT):
        if auto_download:
            return _ensure_fa4_checkout(_FA4_STANDALONE_ROOT)
        if _fa4_checkout_matches_ref(_FA4_STANDALONE_ROOT, ref):
            return _FA4_STANDALONE_ROOT
        raise SystemExit(
            f"{_FA4_STANDALONE_ROOT} is not checked out at {ref}; "
            f"set {_FA4_AUTO_DOWNLOAD_ENV}=1 to update it or set "
            f"{_FA4_ROOT_ENV}=<path>"
        )
    if not auto_download:
        raise SystemExit(
            "FlashAttention checkout not found; set "
            f"{_FA4_ROOT_ENV}=<path>, initialize the TritonBench submodule, "
            f"or enable {_FA4_AUTO_DOWNLOAD_ENV}=1"
        )
    tritonbench_fa4_root = _ensure_fa4_tritonbench_submodule()
    if tritonbench_fa4_root is not None:
        return tritonbench_fa4_root
    return _ensure_fa4_checkout(_FA4_STANDALONE_ROOT)


def _uses_bias(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "biased", 0))


def _check_gpu_policy() -> None:
    """Optionally restrict sweeps to a caller-provided CUDA_VISIBLE_DEVICES set."""
    allowed_raw = os.environ.get(_ALLOWED_PHYSICAL_GPUS_ENV, "").strip()
    if not allowed_raw:
        return
    allowed = tuple(item.strip() for item in allowed_raw.split(",") if item.strip())
    if not allowed:
        return
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        raise SystemExit(
            "Refusing to run without CUDA_VISIBLE_DEVICES set because "
            f"{_ALLOWED_PHYSICAL_GPUS_ENV}={allowed_raw!r} is configured."
        )
    requested = [item.strip() for item in visible.split(",") if item.strip()]
    bad = [gpu for gpu in requested if gpu not in allowed]
    if bad:
        raise SystemExit(
            f"CUDA_VISIBLE_DEVICES={visible!r} selects disallowed GPU(s) {bad}; "
            f"{_ALLOWED_PHYSICAL_GPUS_ENV} allows only {allowed}."
        )


def _make_inputs(
    args: argparse.Namespace, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(args.seed)
    shape = (args.z, args.h, args.seq_len, args.head_dim)
    q = torch.randn(shape, device="cuda", dtype=dtype)
    k = torch.randn(shape, device="cuda", dtype=dtype)
    v = torch.randn(shape, device="cuda", dtype=dtype)
    return q, k, v


def _make_bias(args: argparse.Namespace, dtype: torch.dtype) -> torch.Tensor | None:
    if not _uses_bias(args):
        return None
    torch.manual_seed(args.seed + 1)
    shape = (args.z, args.h, args.seq_len, args.seq_len)
    return torch.randn(shape, device="cuda", dtype=dtype) * 0.25


def _sdpa_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    return torch.nn.functional.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=bias,
        is_causal=causal,
    )


def _check_close(
    actual: torch.Tensor, expected: torch.Tensor, dtype: torch.dtype
) -> bool:
    """Dtype-aware correctness check; returns True on pass.

    bf16/fp16 attention accumulates softmax rounding noise, so benchmark smoke
    checks use a looser threshold than unit tests.
    """
    try:
        torch.testing.assert_close(
            actual.float(), expected.float(), atol=5e-2, rtol=2e-2
        )
    except AssertionError:
        return False
    return True


def _gpu_warmup(duration_ms: int = 10000) -> None:
    """Drive the GPU to a stable clock state with sustained matmul work.

    Without warmup the first benchmark in a new process is at the mercy of the
    GPU's current clock state (idle vs sustained boost on B200, with a ~5-7 s
    cold-to-boost ramp). The steady-state number under sustained load is the
    canonical one; the warmup ensures we start there.
    """
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    torch.cuda.synchronize()
    target = duration_ms / 1000.0
    t0 = time.time()
    while time.time() - t0 < target:
        for _ in range(50):
            a = a @ a
        torch.cuda.synchronize()


def _bench_steady(
    fn: Callable[[], object],
    *,
    num_runs: int,
    warmup_ms: int,
    rep_ms: int,
    do_bench_fn: Callable[..., Any] | None = None,
    cache_warmup_calls: int = 5,
    thermal_warmup_ms: int = 10000,
) -> dict[str, Any]:
    """Steady-state benchmark.

    1. Cache warmup: call fn() a few times to populate per-launch caches.
    2. Thermal warmup: drive the GPU to a stable clock state.
    3. Measurement: ``num_runs`` of do_bench(warmup, rep). Returns
       best/mom-median/mean across runs; mom-median is the gate metric,
       best-of-N is diagnostic only. CuTe can inject the backend wall-clock
       timer here because CUDA-event timing mis-times CuTe kernels on Blackwell.
    """
    bench_fn = do_bench_fn
    use_backend_timer = bench_fn is not None
    if bench_fn is None:
        from triton.testing import do_bench

        bench_fn = cast("Callable[..., Any]", do_bench)

    for _ in range(cache_warmup_calls):
        fn()
    torch.cuda.synchronize()

    _gpu_warmup(thermal_warmup_ms)

    runs: list[float] = []
    for _ in range(num_runs):
        if use_backend_timer:
            # Helion autotune also requests return_mode="median" from backend
            # timers, so use the same statistic when validating the winner.
            ms = bench_fn(fn, warmup=warmup_ms, rep=rep_ms, return_mode="median")
        else:
            ms = bench_fn(fn, warmup=warmup_ms, rep=rep_ms)
        if isinstance(ms, tuple):
            ms = ms[0]
        assert isinstance(ms, float)
        runs.append(ms)

    return {
        "best_ms": min(runs),
        "median_ms": statistics.median(runs),
        "mean_ms": sum(runs) / len(runs),
        "std_ms": statistics.stdev(runs) if len(runs) > 1 else 0.0,
        "runs_ms": runs,
    }


def _result(
    impl: str,
    args: argparse.Namespace,
    stats: dict[str, Any],
    *,
    accuracy: str,
    benchmark_timer: str,
    config: object = None,
    codegen: dict[str, bool] | None = None,
    helion_overrides: dict[str, Any] | None = None,
    notes: list[str] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "impl": impl,
        "shape": _shape_dict(args),
        "accuracy": accuracy,
        "benchmark_timer": benchmark_timer,
        "best_ms": stats["best_ms"],
        "median_ms": stats["median_ms"],
        "mom_median_ms": stats["median_ms"],
        "mean_ms": stats["mean_ms"],
        "std_ms": stats["std_ms"],
        "runs_ms": stats["runs_ms"],
        "best_tflops": _tflops(args, stats["best_ms"]),
        "median_tflops": _tflops(args, stats["median_ms"]),
        "mom_median_tflops": _tflops(args, stats["median_ms"]),
    }
    if config is not None:
        payload["config"] = config
    if codegen is not None:
        payload["codegen"] = codegen
    if helion_overrides is not None:
        payload["helion_overrides"] = helion_overrides
    if notes:
        payload["notes"] = notes
    return payload


def _shape_dict(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "z": args.z,
        "h": args.h,
        "seq_len": args.seq_len,
        "head_dim": args.head_dim,
        "dtype": args.dtype,
        "causal": int(args.causal),
        "biased": int(_uses_bias(args)),
    }


def _helion_codegen_markers(code: str) -> dict[str, bool]:
    return {
        "uses_tcgen05": "cute.nvgpu.tcgen05.CtaGroup" in code,
        "uses_tcgen05_two_cta": "cute.nvgpu.tcgen05.CtaGroup.TWO" in code,
        "uses_tma_umma_pipeline": "PipelineTmaUmma.create(" in code,
    }


def _helion_override_args(args: argparse.Namespace) -> list[str]:
    result: list[str] = []
    for key, value in _helion_env_overrides(args).items():
        result.extend(["--helion-env", f"{key}={value}"])
    for key, value in getattr(args, "helion_config", ()):
        result.extend(["--helion-config", f"{key}={json.dumps(value)}"])
    return result


def _helion_env_overrides(args: argparse.Namespace) -> dict[str, str]:
    env_overrides = dict(getattr(args, "helion_env", ()))
    helion_autotune_effort = getattr(args, "helion_autotune_effort", None)
    helion_autotune_budget_seconds = getattr(
        args, "helion_autotune_budget_seconds", None
    )
    helion_autotune_max_generations = getattr(
        args, "helion_autotune_max_generations", None
    )
    helion_autotune_best_of_k = getattr(args, "helion_autotune_best_of_k", None)
    helion_autotune_benchmark_timeout = getattr(
        args, "helion_autotune_benchmark_timeout", None
    )
    helion_autotune_accuracy_check = getattr(
        args, "helion_autotune_accuracy_check", None
    )
    helion_autotuner_initial_population = getattr(
        args, "helion_autotuner_initial_population", None
    )
    if helion_autotune_effort is not None:
        env_overrides["HELION_AUTOTUNE_EFFORT"] = helion_autotune_effort
    if helion_autotune_budget_seconds is not None:
        env_overrides["HELION_AUTOTUNE_BUDGET_SECONDS"] = str(
            helion_autotune_budget_seconds
        )
    if helion_autotune_max_generations is not None:
        env_overrides["HELION_AUTOTUNE_MAX_GENERATIONS"] = str(
            helion_autotune_max_generations
        )
    if helion_autotune_best_of_k is not None:
        env_overrides["HELION_AUTOTUNE_BEST_OF_K"] = str(helion_autotune_best_of_k)
    if helion_autotune_benchmark_timeout is not None:
        env_overrides["HELION_AUTOTUNE_BENCHMARK_TIMEOUT"] = str(
            helion_autotune_benchmark_timeout
        )
    if helion_autotune_accuracy_check is not None:
        env_overrides["HELION_AUTOTUNE_ACCURACY_CHECK"] = str(
            int(helion_autotune_accuracy_check)
        )
    if helion_autotuner_initial_population is not None:
        env_overrides["HELION_AUTOTUNER_INITIAL_POPULATION"] = (
            helion_autotuner_initial_population
        )
    return env_overrides


def _apply_helion_env(args: argparse.Namespace) -> dict[str, str]:
    env_overrides = _helion_env_overrides(args)
    os.environ.update(env_overrides)
    return env_overrides


def _write_json_output(args: argparse.Namespace, payload: dict[str, Any]) -> None:
    if args.json_output:
        Path(args.json_output).write_text(json.dumps(payload) + "\n")


def _make_helion_config(
    args: argparse.Namespace,
    compiler_seed_config: dict[str, object] | None = None,
) -> tuple[dict[str, object] | None, dict[str, object]]:
    config_overrides = dict(getattr(args, "helion_config", ()))
    if not args.helion_force_flash_config and not config_overrides:
        return None, config_overrides

    config: dict[str, object] = {}
    if args.helion_force_flash_config:
        if compiler_seed_config is not None:
            config.update(compiler_seed_config)
        else:
            config["block_sizes"] = [1, 128, 128]
    config.update(config_overrides)
    return config, config_overrides


def _benchmark_sdpa(args: argparse.Namespace) -> dict[str, Any]:
    dtype = _dtype_from_name(args.dtype)
    q, k, v = _make_inputs(args, dtype)
    bias = _make_bias(args, dtype)
    causal = bool(args.causal)
    fn = lambda: _sdpa_reference(q, k, v, causal=causal, bias=bias)  # noqa: E731
    # sdpa is the reference, so it is "PASS" by definition.
    stats = _bench_steady(
        fn,
        num_runs=args.num_runs,
        warmup_ms=args.warmup_ms,
        rep_ms=args.rep_ms,
    )
    return _result(
        "sdpa", args, stats, accuracy="PASS", benchmark_timer="event", config=None
    )


def _benchmark_flexattention(args: argparse.Namespace) -> dict[str, Any]:
    from torch.nn.attention.flex_attention import create_block_mask
    from torch.nn.attention.flex_attention import flex_attention

    dtype = _dtype_from_name(args.dtype)
    q, k, v = _make_inputs(args, dtype)
    bias = _make_bias(args, dtype)
    causal = bool(args.causal)
    compiled = cast(
        "Callable[..., torch.Tensor]", torch.compile(flex_attention, fullgraph=True)
    )

    if causal:

        def causal_mask(
            b: torch.Tensor,
            h: torch.Tensor,
            q_idx: torch.Tensor,
            kv_idx: torch.Tensor,
        ) -> torch.Tensor:
            return q_idx >= kv_idx

        block_mask = create_block_mask(
            causal_mask,
            None,
            None,
            args.seq_len,
            args.seq_len,
            device=q.device,
            BLOCK_SIZE=128,
        )
        fn = lambda: compiled(q, k, v, block_mask=block_mask)  # noqa: E731
    elif bias is not None:
        bias_tensor = bias

        def bias_score_mod(
            score: torch.Tensor,
            b: torch.Tensor,
            h: torch.Tensor,
            q_idx: torch.Tensor,
            kv_idx: torch.Tensor,
        ) -> torch.Tensor:
            return score + bias_tensor[b, h, q_idx, kv_idx]

        fn = lambda: compiled(q, k, v, score_mod=bias_score_mod)  # noqa: E731
    else:
        fn = lambda: compiled(q, k, v)  # noqa: E731

    accuracy = "PASS"
    if not args.skip_correctness:
        expected = _sdpa_reference(q, k, v, causal=causal, bias=bias)
        out = fn()
        accuracy = "PASS" if _check_close(out, expected, dtype) else "FAIL"
    stats = _bench_steady(
        fn,
        num_runs=args.num_runs,
        warmup_ms=args.warmup_ms,
        rep_ms=args.rep_ms,
    )
    return _result(
        "flexattention",
        args,
        stats,
        accuracy=accuracy,
        benchmark_timer="event",
        config=None,
    )


def _import_fa4() -> types.ModuleType:
    """Import FlashAttention-4 (``flash_attn.cute.flash_attn_func``) in this env.

    FA4 v2.8.3 targets the CUDA-12.9 cutlass DSL; Helion's env ships the CUDA-13
    cutlass 4.5.1 build. The checkout is resolved lazily from HELION_FA4_ROOT, an
    existing tritonbench submodule, or, when HELION_FA4_AUTO_DOWNLOAD=1, an
    auto-cloned benchmarks/flash-attention tree. Three things must be adapted to
    bridge the cutlass skew, and the
    top-level package must be bypassed:

    * The top-level ``flash_attn/__init__`` eagerly imports the unbuilt FA2 CUDA
      extension (``flash_attn_2_cuda``); register a bare ``flash_attn`` namespace
      package first so only the independent ``flash_attn.cute`` subpackage loads.
    * 4.5.1 only re-exports the nvvm enums (``ProxyKind`` etc.) from
      ``cute.arch`` on the 12.9 wheel -- inject them from ``_mlir.dialects.nvvm``.
    * FA4's primitive wrappers use the old binding ABI: ``nvvm.fmax`` takes an
      explicit result-type first arg (4.5.1 infers it), the packed-f32x2 ops take
      a ``RoundingModeKind`` enum (4.5.1 wants the ``'rn'`` string), and
      ``fence_proxy`` takes ``ProxyKind``/``SharedSpace`` enums (4.5.1 wants string
      literals). Shim each. These are numerically exact (max/min associativity,
      same rounding mode, same proxy/space).

    Returns the ``flash_attn.cute`` module (``.flash_attn_func`` is the fwd entry).
    """
    import cutlass._mlir.dialects.nvvm as nvvm
    import cutlass.cute as cute

    fa4_root = _resolve_fa4_root()
    for sym in (
        "ProxyKind",
        "SharedSpace",
        "RoundingModeKind",
        "ReduxKind",
        "AtomicOpKind",
    ):
        if not hasattr(cute.arch, sym) and hasattr(nvvm, sym):
            setattr(cute.arch, sym, getattr(nvvm, sym))

    def _strip_type_arg(orig: Callable[..., object]) -> Callable[..., object]:
        def wrapped(*args: object, **kw: object) -> object:
            if len(args) == 3:  # (result_type, a, b) -> (a, b)
                args = args[1:]
            return orig(*args, **kw)

        return wrapped

    nvvm.fmax = _strip_type_arg(nvvm.fmax)  # pyrefly: ignore[bad-assignment]
    nvvm.fmin = _strip_type_arg(nvvm.fmin)  # pyrefly: ignore[bad-assignment]

    proxy_str = {
        "alias": "alias",
        "async_": "async",
        "async_global": "async.global",
        "async_shared": "async.shared",
        "generic": "generic",
        "tensormap": "tensormap",
    }
    space_str = {"shared_cta": "cta", "shared_cluster": "cluster"}
    orig_fence_proxy = cute.arch.fence_proxy

    def _fence_proxy(kind: object, *, space: object = None, **kw: object) -> object:
        if hasattr(kind, "name"):
            kind = proxy_str.get(kind.name, kind.name)
        if space is not None and hasattr(space, "name"):
            space = space_str.get(space.name, space.name)
        return orig_fence_proxy(kind, space=space, **kw)

    cute.arch.fence_proxy = _fence_proxy

    fa4_root_str = str(fa4_root)
    if fa4_root_str not in sys.path:
        sys.path.insert(0, fa4_root_str)
    if "flash_attn" not in sys.modules:
        pkg = types.ModuleType("flash_attn")
        pkg.__path__ = [str(fa4_root / "flash_attn")]
        pkg.__package__ = "flash_attn"
        sys.modules["flash_attn"] = pkg

    import functools

    import flash_attn.cute as fc  # pyrefly: ignore[missing-import]
    import flash_attn.cute.utils as fu  # pyrefly: ignore[missing-import]

    fu.fma_packed_f32x2 = functools.partial(cute.arch.fma_packed_f32x2, rnd="rn")
    fu.mul_packed_f32x2 = functools.partial(cute.arch.mul_packed_f32x2, rnd="rn")
    fu.add_packed_f32x2 = functools.partial(cute.arch.add_packed_f32x2, rnd="rn")
    fu.sub_packed_f32x2 = functools.partial(
        cute.arch.calc_packed_f32x2_op,
        src_c=None,
        calc_func=nvvm.sub_packed_f32x2,
        rnd="rn",
    )
    return fc


def _benchmark_fa4(args: argparse.Namespace) -> dict[str, Any]:
    """FlashAttention-4 (CuTe-DSL fwd) baseline -- the upstream design target.

    FA4's tensor layout is ``(B, S, H, D)``; our harness builds ``(B, H, S, D)``
    (the SDPA convention), so we transpose in and out. FA4 returns ``(out, lse)``.
    """
    if _uses_bias(args):
        return {
            "impl": "fa4",
            "shape": _shape_dict(args),
            "accuracy": "SKIP",
            "skipped_reason": "FA4 harness does not support additive score bias",
        }
    fc = _import_fa4()
    dtype = _dtype_from_name(args.dtype)
    q, k, v = _make_inputs(args, dtype)
    causal = bool(args.causal)

    qt = q.transpose(1, 2).contiguous()  # (B, H, S, D) -> (B, S, H, D)
    kt = k.transpose(1, 2).contiguous()
    vt = v.transpose(1, 2).contiguous()

    def run() -> torch.Tensor:
        out, _lse = fc.flash_attn_func(qt, kt, vt, softmax_scale=None, causal=causal)
        return out

    with _scrubbed_argv():
        accuracy = "PASS"
        if not args.skip_correctness:
            expected = _sdpa_reference(q, k, v, causal=causal)
            out = run()  # (B, S, H, D)
            got = out.transpose(1, 2)  # back to (B, H, S, D)
            accuracy = "PASS" if _check_close(got, expected, dtype) else "FAIL"
        stats = _bench_steady(
            run,
            num_runs=args.num_runs,
            warmup_ms=args.warmup_ms,
            rep_ms=args.rep_ms,
        )
    return _result(
        "fa4", args, stats, accuracy=accuracy, benchmark_timer="event", config=None
    )


def _helion_benchmark_timer(args: argparse.Namespace, backend: str) -> str:
    if backend == "cute":
        return str(getattr(args, "helion_cute_benchmark_timer", "wall"))
    return "event"


def _helion_do_bench_fn(
    bound: object, args: argparse.Namespace, backend: str
) -> Callable[..., Any] | None:
    if _helion_benchmark_timer(args, backend) == "wall":
        return cast("Any", bound).env.backend.get_do_bench()
    return None


@contextlib.contextmanager
def _scrubbed_argv() -> Iterator[None]:
    """Hide our CLI argv from libraries that parse ``sys.argv`` on import/compile.

    The CuTe DSL's ``base_dsl.dsl.diagnostic()`` calls ``parse_known_args()`` on
    the process argv during kernel compilation/launch and aborts the process
    (printing its own ``-diagnostic`` usage banner) when it trips over our
    ``--impl``/``--dtype`` flags. Reducing argv to ``[argv0]`` for the duration
    of the cute work makes that parser a no-op without affecting our own parse,
    which has already completed.
    """
    saved = sys.argv
    sys.argv = sys.argv[:1]
    try:
        yield
    finally:
        sys.argv = saved


def _benchmark_helion(args: argparse.Namespace) -> dict[str, Any]:
    """Helion attention via examples/attention.py.

    Backend is determined by ``args.helion_backend`` (cute or triton). The cute
    path may currently be numerically wrong; we never let that crash the
    harness -- we record accuracy=FAIL and still report timing.

    When ``--helion-force-flash-config`` is set, skip autotune and directly use
    the compiler-promoted flash seed, including any shape-specific heuristic
    keys. This is useful for benchmarking individual knob variants without
    waiting for autotuner search.
    """
    env_overrides = _apply_helion_env(args)
    backend = args.helion_backend
    os.environ["HELION_BACKEND"] = backend
    impl_label = f"helion-{backend}"

    from examples.attention import attention
    from examples.attention import attention_output
    from examples.attention import biased_attention
    from examples.attention import biased_attention_output
    from examples.attention import causal_attention
    from examples.attention import causal_attention_output

    dtype = _dtype_from_name(args.dtype)
    q, k, v = _make_inputs(args, dtype)
    bias = _make_bias(args, dtype)
    causal = bool(args.causal)
    output_only = not bool(args.helion_return_lse)
    if bias is not None:
        kernel = biased_attention_output if output_only else biased_attention
        kernel_args = (q, k, v, bias)
    elif output_only:
        kernel = causal_attention_output if causal else attention_output
        kernel_args = (q, k, v)
    elif causal:
        kernel = causal_attention
        kernel_args = (q, k, v)
    else:
        kernel = attention
        kernel_args = (q, k, v)

    with _scrubbed_argv():
        bound = kernel.bind(kernel_args)
        compiler_seed_config = (
            dict(bound.config_spec.default_config().config)
            if backend == "cute"
            and bound.config_spec.compiler_default_config is not None
            else None
        )
        fixed_config, config_overrides = _make_helion_config(args, compiler_seed_config)
        notes: list[str] = []
        if bias is not None and backend == "cute":
            biased_seed_config = cast(
                "dict[str, object]",
                compiler_seed_config or {"block_sizes": [1, 128, 128]},
            )
            fixed_config = {**biased_seed_config, **(fixed_config or {})}
            notes.append(
                "CuTe biased attention starts from the fixed 128x128 flash seed "
                "and applies user overrides; biased autotune search is not "
                "characterized yet."
            )
        if fixed_config is not None:
            bound.set_config(fixed_config)
            active_config = fixed_config
            autotuned = False
        else:
            active_config = bound.autotune(
                kernel_args, force=bool(args.helion_force_autotune)
            )
            autotuned = True

        code = bound.to_triton_code(active_config)
        codegen = _helion_codegen_markers(code)

        accuracy = "PASS"
        if not args.skip_correctness:
            expected = _sdpa_reference(q, k, v, causal=causal, bias=bias)
            actual = bound(*kernel_args)
            out = cast("torch.Tensor", actual if output_only else actual[0])
            accuracy = "PASS" if _check_close(out, expected, dtype) else "FAIL"

        fn = lambda: bound(*kernel_args)  # noqa: E731
        benchmark_timer = _helion_benchmark_timer(args, backend)
        stats = _bench_steady(
            fn,
            num_runs=args.num_runs,
            warmup_ms=args.warmup_ms,
            rep_ms=args.rep_ms,
            do_bench_fn=_helion_do_bench_fn(bound, args, backend),
        )
    return _result(
        impl_label,
        args,
        stats,
        accuracy=accuracy,
        benchmark_timer=benchmark_timer,
        config=repr(active_config),
        codegen=codegen,
        helion_overrides={
            "env_overrides": env_overrides,
            "config_overrides": config_overrides,
            "autotuned": autotuned,
            "benchmark_timer": benchmark_timer,
            "force_autotune": bool(args.helion_force_autotune),
            "return_lse": not output_only,
        },
        notes=notes,
    )


def _run_impl(args: argparse.Namespace) -> dict[str, Any]:
    if args.impl == "helion-cute":
        args.helion_backend = "cute"
        return _benchmark_helion(args)
    if args.impl == "helion-triton":
        args.helion_backend = "triton"
        return _benchmark_helion(args)
    if args.impl == "sdpa":
        return _benchmark_sdpa(args)
    if args.impl == "flexattention":
        return _benchmark_flexattention(args)
    if args.impl == "fa4":
        return _benchmark_fa4(args)
    raise SystemExit(f"unknown impl {args.impl!r}")


def _build_subprocess_cmd(args: argparse.Namespace, impl: str) -> list[str]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--impl",
        impl,
        "--z",
        str(args.z),
        "--h",
        str(args.h),
        "--seq-len",
        str(args.seq_len),
        "--head-dim",
        str(args.head_dim),
        "--dtype",
        args.dtype,
        "--causal",
        str(int(args.causal)),
        "--biased",
        str(int(_uses_bias(args))),
        "--num-runs",
        str(args.num_runs),
        "--warmup-ms",
        str(args.warmup_ms),
        "--rep-ms",
        str(args.rep_ms),
        "--seed",
        str(args.seed),
        "--skip-correctness",
        str(int(args.skip_correctness)),
        "--helion-force-flash-config",
        str(int(getattr(args, "helion_force_flash_config", 0))),
        "--helion-force-autotune",
        str(int(getattr(args, "helion_force_autotune", 1))),
        "--helion-return-lse",
        str(int(getattr(args, "helion_return_lse", 0))),
        "--helion-cute-benchmark-timer",
        str(getattr(args, "helion_cute_benchmark_timer", "wall")),
        "--json",
    ]
    cmd.extend(_helion_override_args(args))
    return cmd


def _run_json_subprocess(
    cmd: list[str], args: argparse.Namespace
) -> tuple[int, dict[str, Any] | None, str, str]:
    if args.stream_subprocesses:
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "result.json"
            proc = subprocess.run(
                [*cmd, "--json-output", str(json_path)],
                cwd=REPO_ROOT,
                check=False,
            )
            if proc.returncode != 0:
                return proc.returncode, None, "", ""
            try:
                return proc.returncode, json.loads(json_path.read_text()), "", ""
            except (FileNotFoundError, json.JSONDecodeError) as err:
                return proc.returncode, None, "", f"failed to read {json_path}: {err}"

    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return proc.returncode, None, proc.stdout, proc.stderr
    stdout_lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    try:
        return proc.returncode, json.loads(stdout_lines[-1]), proc.stdout, proc.stderr
    except (IndexError, json.JSONDecodeError):
        return proc.returncode, None, proc.stdout, proc.stderr


def _run_all(args: argparse.Namespace) -> dict[str, Any]:
    impls = args.impls or list(DEFAULT_IMPLS)
    results: list[dict[str, Any]] = []
    for impl in impls:
        if impl not in ALL_IMPLS:
            print(f"unknown impl {impl!r}, skipping", file=sys.stderr)
            continue
        print(f"\n=== Running {impl} ===", flush=True)
        cmd = _build_subprocess_cmd(args, impl)
        returncode, payload, stdout, stderr = _run_json_subprocess(cmd, args)
        if returncode != 0:
            print(
                f"impl {impl} failed (rc={returncode})\n"
                f"--- stdout ---\n{stdout}\n"
                f"--- stderr ---\n{stderr}",
                file=sys.stderr,
            )
            results.append(
                {
                    "impl": impl,
                    "shape": _shape_dict(args),
                    "accuracy": "ERROR",
                    "error": f"subprocess rc={returncode}",
                }
            )
            continue
        if payload is None:
            print(
                f"impl {impl} produced no JSON output\n"
                f"--- stdout ---\n{stdout}\n"
                f"--- stderr ---\n{stderr}",
                file=sys.stderr,
            )
            results.append(
                {
                    "impl": impl,
                    "shape": _shape_dict(args),
                    "accuracy": "ERROR",
                    "error": "no JSON output",
                }
            )
        else:
            results.append(payload)
    return {"shape": _shape_dict(args), "results": results}


def _shape_label(shape: dict[str, Any]) -> str:
    return (
        f"z{shape['z']}_h{shape['h']}_s{shape['seq_len']}_d{shape['head_dim']}"
        f"_{shape['dtype']}_causal{shape['causal']}_biased{shape.get('biased', 0)}"
    )


def _print_summary(payload: dict[str, Any]) -> None:
    shape = payload["shape"]
    print(
        f"\nshape z={shape['z']} h={shape['h']} seq={shape['seq_len']} "
        f"head_dim={shape['head_dim']} dtype={shape['dtype']} "
        f"causal={shape['causal']} biased={shape.get('biased', 0)}"
    )
    impl_width = max(
        16,
        max(
            (len(str(result.get("impl", ""))) for result in payload["results"]),
            default=0,
        ),
    )
    separator_width = impl_width + 70
    print("=" * separator_width)
    header = (
        f"{'impl':>{impl_width}}  {'acc':>6}  {'best ms':>10}  "
        f"{'mom-med ms':>10}  {'best TF/s':>10}  {'mom-med TF/s':>12}"
    )
    print(header)
    print("-" * separator_width)
    sdpa_mom: float | None = None
    for r in payload["results"]:
        if r.get("impl") == "sdpa":
            sdpa_mom = r.get("mom_median_tflops")
    for r in payload["results"]:
        impl = r.get("impl", "")
        acc = r.get("accuracy", "?")
        if "best_ms" not in r:
            print(
                f"{impl:>{impl_width}}  {acc:>6}  "
                f"{'--':>10}  {'--':>10}  {'--':>10}  {'--':>12}"
            )
            for note in r.get("notes", ()):
                print(f"{'':>{impl_width}}  note: {note}")
            continue
        mom_ms = r.get("mom_median_ms", r["median_ms"])
        mom_tflops = r.get("mom_median_tflops", r["median_tflops"])
        line = (
            f"{impl:>{impl_width}}  {acc:>6}  "
            f"{r['best_ms']:>10.4f}  {mom_ms:>10.4f}  "
            f"{r['best_tflops']:>10.1f}  {mom_tflops:>12.1f}"
        )
        if sdpa_mom is not None and impl != "sdpa" and mom_tflops:
            line += f"   {mom_tflops / sdpa_mom * 100:>6.1f}% sdpa"
        print(line)
        details: list[str] = []
        if "benchmark_timer" in r:
            details.append(f"timer={r['benchmark_timer']}")
        helion_overrides = r.get("helion_overrides")
        if isinstance(helion_overrides, dict) and "autotuned" in helion_overrides:
            details.append(f"autotuned={helion_overrides['autotuned']}")
        if "config" in r:
            details.append(f"config={r['config']}")
        if details:
            print(f"{'':>{impl_width}}  {'; '.join(details)}")
        for note in r.get("notes", ()):
            print(f"{'':>{impl_width}}  note: {note}")
    print()


_MARKDOWN_COLUMNS = (
    "shape",
    "dtype",
    "causal",
    "biased",
    "impl",
    "acc",
    "timer",
    "best_ms",
    "mom_med_ms",
    "best_tflops",
    "mom_med_tflops",
    "pct_sdpa",
)


def _markdown_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    shape = payload["shape"]
    shape_str = f"{shape['z']}x{shape['h']}x{shape['seq_len']}x{shape['head_dim']}"
    sdpa_mom: float | None = None
    for r in payload["results"]:
        if r.get("impl") == "sdpa":
            sdpa_mom = r.get("mom_median_tflops")
    rows: list[dict[str, Any]] = []
    for r in payload["results"]:
        impl = r.get("impl", "")
        acc = r.get("accuracy", "?")
        if "best_ms" not in r:
            rows.append(
                {
                    "shape": shape_str,
                    "dtype": shape["dtype"],
                    "causal": shape["causal"],
                    "biased": shape.get("biased", 0),
                    "impl": impl,
                    "acc": acc,
                    "timer": r.get("benchmark_timer", ""),
                    "best_ms": "",
                    "mom_med_ms": "",
                    "best_tflops": "",
                    "mom_med_tflops": "",
                    "pct_sdpa": "",
                }
            )
            continue
        mom_ms = r.get("mom_median_ms", r["median_ms"])
        mom_tflops = r.get("mom_median_tflops", r["median_tflops"])
        pct = ""
        if sdpa_mom is not None and impl != "sdpa" and mom_tflops:
            pct = f"{mom_tflops / sdpa_mom * 100:.1f}%"
        rows.append(
            {
                "shape": shape_str,
                "dtype": shape["dtype"],
                "causal": shape["causal"],
                "biased": shape.get("biased", 0),
                "impl": impl,
                "acc": acc,
                "timer": r.get("benchmark_timer", ""),
                "best_ms": f"{r['best_ms']:.4f}",
                "mom_med_ms": f"{mom_ms:.4f}",
                "best_tflops": f"{r['best_tflops']:.1f}",
                "mom_med_tflops": f"{mom_tflops:.1f}",
                "pct_sdpa": pct,
            }
        )
    return rows


def _render_markdown_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| " + " | ".join(_MARKDOWN_COLUMNS) + " |",
        "| " + " | ".join("---" for _ in _MARKDOWN_COLUMNS) + " |",
    ]
    for row in rows:
        lines.append(
            "| " + " | ".join(str(row[col]) for col in _MARKDOWN_COLUMNS) + " |"
        )
    return "\n".join(lines)


def _render_report_notes(payloads: list[dict[str, Any]]) -> str:
    error_rows: list[str] = []
    fixed_config_rows: list[str] = []
    for payload in payloads:
        shape = payload["shape"]
        shape_str = f"{shape['z']}x{shape['h']}x{shape['seq_len']}x{shape['head_dim']}"
        variant = _variant_label(shape)
        for result in payload["results"]:
            impl = result.get("impl", "<unknown>")
            if result.get("accuracy") == "ERROR":
                error_rows.append(f"- {variant} {shape_str} {impl}")
                continue
            helion_overrides = result.get("helion_overrides")
            if (
                isinstance(helion_overrides, dict)
                and helion_overrides.get("autotuned") is False
            ):
                fixed_config_rows.append(f"- {variant} {shape_str} {impl}")
    sections: list[str] = []
    if error_rows:
        sections.append(
            "Rows marked `ERROR` did not produce timing data and are omitted "
            "from the bar graph:\n" + "\n".join(error_rows)
        )
    if fixed_config_rows:
        sections.append(
            "Rows below used a fixed Helion config rather than full autotuning:\n"
            + "\n".join(fixed_config_rows)
        )
    if not sections:
        return ""
    return "\n\n" + "\n\n".join(sections)


def _append_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    file_exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(_MARKDOWN_COLUMNS))
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


def _variant_label(shape: dict[str, Any]) -> str:
    if shape.get("biased"):
        return "biased"
    if shape["causal"]:
        return "causal"
    return "dense"


def _wide_rows(payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for payload in payloads:
        shape = payload["shape"]
        row: dict[str, Any] = {
            "variant": _variant_label(shape),
            "shape": f"{shape['z']}x{shape['h']}x{shape['seq_len']}x{shape['head_dim']}",
            "z": shape["z"],
            "h": shape["h"],
            "seq_len": shape["seq_len"],
            "head_dim": shape["head_dim"],
            "dtype": shape["dtype"],
            "causal": shape["causal"],
            "biased": shape.get("biased", 0),
        }
        results_by_impl = {r.get("impl"): r for r in payload["results"]}
        sdpa_mom = results_by_impl.get("sdpa", {}).get("mom_median_tflops")
        for impl in _DISPLAY_IMPLS:
            key = _IMPL_KEYS[impl]
            result = results_by_impl.get(impl, {})
            row[f"{key}_acc"] = result.get("accuracy", "")
            row[f"{key}_timer"] = result.get("benchmark_timer", "")
            if "best_ms" not in result:
                row[f"{key}_best_ms"] = ""
                row[f"{key}_mom_med_ms"] = ""
                row[f"{key}_best_tflops"] = ""
                row[f"{key}_mom_med_tflops"] = ""
                row[f"{key}_pct_sdpa"] = ""
                continue
            row[f"{key}_best_ms"] = f"{result['best_ms']:.4f}"
            mom_ms = result.get("mom_median_ms", result["median_ms"])
            row[f"{key}_mom_med_ms"] = f"{mom_ms:.4f}"
            row[f"{key}_best_tflops"] = f"{result['best_tflops']:.1f}"
            mom_tflops = result.get("mom_median_tflops", result["median_tflops"])
            row[f"{key}_mom_med_tflops"] = f"{mom_tflops:.1f}"
            if sdpa_mom and impl != "sdpa":
                row[f"{key}_pct_sdpa"] = f"{mom_tflops / sdpa_mom * 100:.1f}%"
            else:
                row[f"{key}_pct_sdpa"] = ""
        rows.append(row)
    return rows


def _write_wide_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    base_columns = (
        "variant",
        "shape",
        "z",
        "h",
        "seq_len",
        "head_dim",
        "dtype",
        "causal",
        "biased",
    )
    impl_columns: list[str] = []
    for impl in _DISPLAY_IMPLS:
        key = _IMPL_KEYS[impl]
        impl_columns.extend(
            [
                f"{key}_acc",
                f"{key}_timer",
                f"{key}_best_ms",
                f"{key}_mom_med_ms",
                f"{key}_best_tflops",
                f"{key}_mom_med_tflops",
                f"{key}_pct_sdpa",
            ]
        )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*base_columns, *impl_columns])
        writer.writeheader()
        writer.writerows(rows)


def _format_plot_cell(result: dict[str, Any]) -> str:
    accuracy = result.get("accuracy", "")
    if "best_ms" not in result:
        return str(accuracy)
    mom_ms = result.get("mom_median_ms", result["median_ms"])
    mom_tflops = result.get("mom_median_tflops", result["median_tflops"])
    label = f"{mom_tflops:.1f}\n{mom_ms:.4f} ms"
    if accuracy and accuracy != "PASS":
        label += f"\n{accuracy}"
    return label


def _shape_plot_label(shape: dict[str, Any]) -> str:
    variant = _variant_label(shape)
    return (
        f"{variant}\n{shape['z']}x{shape['h']}x{shape['seq_len']}x{shape['head_dim']}"
    )


def _write_matplotlib_bar_graph(path: Path, payloads: list[dict[str, Any]]) -> None:
    import matplotlib  # pyrefly: ignore[missing-import]

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # pyrefly: ignore[missing-import]
    import numpy as np

    labels: list[str] = []
    values_by_impl: dict[str, list[float]] = {impl: [] for impl in _DISPLAY_IMPLS}
    for payload in payloads:
        shape = payload["shape"]
        labels.append(_shape_plot_label(shape))
        results_by_impl = {r.get("impl"): r for r in payload["results"]}
        for impl in _DISPLAY_IMPLS:
            result = results_by_impl.get(impl, {})
            if result.get("accuracy") == "PASS" and "mom_median_tflops" in result:
                values_by_impl[impl].append(float(result["mom_median_tflops"]))
            else:
                values_by_impl[impl].append(np.nan)

    x = np.arange(len(labels))
    width = 0.15
    fig_width = max(13.5, 1.45 * len(labels) + 4.0)
    fig, ax = plt.subplots(figsize=(fig_width, 6.5))
    for index, impl in enumerate(_DISPLAY_IMPLS):
        offsets = x + (index - (len(_DISPLAY_IMPLS) - 1) / 2) * width
        ax.bar(offsets, values_by_impl[impl], width, label=_IMPL_LABELS[impl])

    ax.set_ylabel("TFLOP/s")
    ax.set_title("TFLOP/s")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha="center")
    ax.legend(ncols=1, loc="upper left")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_matplotlib_table(path: Path, payloads: list[dict[str, Any]]) -> None:
    import matplotlib  # pyrefly: ignore[missing-import]

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # pyrefly: ignore[missing-import]

    columns = [
        "Variant",
        "Shape\nzxhxSxD",
        "dtype",
        *(_IMPL_LABELS[impl] for impl in _DISPLAY_IMPLS),
    ]
    cell_text: list[list[str]] = []
    for payload in payloads:
        shape = payload["shape"]
        results_by_impl = {r.get("impl"): r for r in payload["results"]}
        cell_text.append(
            [
                _variant_label(shape),
                f"{shape['z']}x{shape['h']}x{shape['seq_len']}x{shape['head_dim']}",
                shape["dtype"],
                *(
                    _format_plot_cell(results_by_impl.get(impl, {}))
                    for impl in _DISPLAY_IMPLS
                ),
            ]
        )

    fig_height = max(3.0, 0.72 * len(cell_text) + 1.4)
    fig_width = 13.5
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")
    table = ax.table(
        cellText=cell_text,
        colLabels=columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.75)
    for (row, _col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#e9eef6")
        elif row % 2 == 0:
            cell.set_facecolor("#f7f7f7")
    ax.set_title(
        "Attention backend performance: TFLOP/s (ms)",
        fontsize=12,
        pad=18,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _run_shape_subprocess(
    args: argparse.Namespace, shape: tuple[int, int, int, int, str, int, int]
) -> dict[str, Any]:
    z, h, seq_len, head_dim, dtype, causal, biased = shape
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--impl",
        "all",
        "--z",
        str(z),
        "--h",
        str(h),
        "--seq-len",
        str(seq_len),
        "--head-dim",
        str(head_dim),
        "--dtype",
        dtype,
        "--causal",
        str(causal),
        "--biased",
        str(biased),
        "--num-runs",
        str(args.num_runs),
        "--warmup-ms",
        str(args.warmup_ms),
        "--rep-ms",
        str(args.rep_ms),
        "--seed",
        str(args.seed),
        "--skip-correctness",
        str(int(args.skip_correctness)),
        "--helion-force-flash-config",
        str(int(getattr(args, "helion_force_flash_config", 0))),
        "--helion-force-autotune",
        str(int(getattr(args, "helion_force_autotune", 1))),
        "--helion-return-lse",
        str(int(getattr(args, "helion_return_lse", 0))),
        "--helion-cute-benchmark-timer",
        str(getattr(args, "helion_cute_benchmark_timer", "wall")),
        "--json",
    ]
    if args.impls:
        cmd.extend(["--impls", *args.impls])
    if args.stream_subprocesses:
        cmd.append("--stream-subprocesses")
    cmd.extend(_helion_override_args(args))
    shape_dict = {
        "z": z,
        "h": h,
        "seq_len": seq_len,
        "head_dim": head_dim,
        "dtype": dtype,
        "causal": causal,
        "biased": biased,
    }
    returncode, payload, stdout, stderr = _run_json_subprocess(cmd, args)
    if returncode != 0:
        print(
            f"shape {_shape_label(shape_dict)} failed (rc={returncode})\n"
            f"--- stdout ---\n{stdout}\n"
            f"--- stderr ---\n{stderr}",
            file=sys.stderr,
        )
        return {"shape": shape_dict, "results": []}
    if payload is None:
        print(
            f"shape {_shape_label(shape_dict)} produced no JSON output\n"
            f"--- stdout ---\n{stdout}\n"
            f"--- stderr ---\n{stderr}",
            file=sys.stderr,
        )
        return {"shape": shape_dict, "results": []}
    return payload


def _write_sweep_outputs(
    args: argparse.Namespace, payloads: list[dict[str, Any]]
) -> None:
    all_rows: list[dict[str, Any]] = []
    for payload in payloads:
        all_rows.extend(_markdown_rows(payload))

    table = _render_markdown_table(all_rows)
    notes = _render_report_notes(payloads)
    print("\n## Attention backend sweep\n")
    print(table + notes)

    if args.output:
        out_path = Path(args.output)
        out_path.write_text("## Attention backend sweep\n\n" + table + notes + "\n")
        print(f"\nWrote Markdown table to {out_path}", file=sys.stderr)
    if args.append_csv:
        _append_csv(Path(args.append_csv), all_rows)
        print(f"Appended {len(all_rows)} rows to {args.append_csv}", file=sys.stderr)
    wide_rows = _wide_rows(payloads)
    if args.csv_output:
        _write_wide_csv(Path(args.csv_output), wide_rows)
        print(f"Wrote wide CSV table to {args.csv_output}", file=sys.stderr)
    if args.plot_output:
        _write_matplotlib_bar_graph(Path(args.plot_output), payloads)
        print(
            f"Wrote matplotlib TFLOP/s bar graph to {args.plot_output}", file=sys.stderr
        )


def _run_all_shapes(args: argparse.Namespace) -> None:
    _check_gpu_policy()
    payloads: list[dict[str, Any]] = []
    for shape in _SHAPE_SUITES[args.shape_suite]:
        print(f"\n##### shape {shape} #####", flush=True)
        payload = _run_shape_subprocess(args, shape)
        payloads.append(payload)
        _print_summary(payload)

    _write_sweep_outputs(args, payloads)


def _run_merge_json(args: argparse.Namespace) -> None:
    payloads = [json.loads(Path(path).read_text()) for path in args.merge_json]
    for payload in payloads:
        _print_summary(payload)
    _write_sweep_outputs(args, payloads)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--impl", choices=("all", *ALL_IMPLS), default="all")
    parser.add_argument(
        "--impls",
        nargs="*",
        default=None,
        help=(
            "Override DEFAULT_IMPLS for --impl all "
            f"(default: {' '.join(DEFAULT_IMPLS)})"
        ),
    )
    parser.add_argument("--z", type=int, default=2, help="batch size")
    parser.add_argument("--h", type=int, default=8, help="number of heads")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--causal", type=int, choices=(0, 1), default=0)
    parser.add_argument(
        "--biased",
        type=int,
        choices=(0, 1),
        default=0,
        help="Use an additive attention score bias. Not compatible with --causal.",
    )
    parser.add_argument("--num-runs", type=int, default=5)
    parser.add_argument("--warmup-ms", type=int, default=1000)
    parser.add_argument("--rep-ms", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-correctness", type=int, choices=(0, 1), default=0)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON to stdout (used by --impl all subprocess collection).",
    )
    parser.add_argument(
        "--json-output",
        default=None,
        help="Write the JSON payload to this file, in addition to normal output.",
    )
    parser.add_argument(
        "--stream-subprocesses",
        action="store_true",
        help=(
            "Stream nested subprocess stdout/stderr directly. Useful for long "
            "Helion autotune runs; JSON is collected through a sidecar file."
        ),
    )
    parser.add_argument(
        "--all-shapes",
        action="store_true",
        help=(
            "Sweep the selected shape list (each via an --impl all "
            "subprocess) and emit a Markdown table to stdout."
        ),
    )
    parser.add_argument(
        "--merge-json",
        nargs="+",
        default=None,
        help=(
            "Merge saved per-shape JSON payloads from --impl all into the "
            "--output/--append-csv/--csv-output/--plot-output reports without "
            "rerunning benchmarks."
        ),
    )
    parser.add_argument(
        "--shape-suite",
        choices=tuple(_SHAPE_SUITES),
        default="representative",
        help="(--all-shapes) Shape list to sweep.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="(--all-shapes) Write the Markdown table to this file.",
    )
    parser.add_argument(
        "--append-csv",
        default=None,
        help="(--all-shapes) Append sweep rows to this CSV file.",
    )
    parser.add_argument(
        "--csv-output",
        default=None,
        help="(--all-shapes) Write one wide CSV row per shape to this file.",
    )
    parser.add_argument(
        "--plot-output",
        default=None,
        help="(--all-shapes) Write a matplotlib-rendered TFLOP/s bar graph to this file.",
    )
    # Internal: backend selector threaded through the helion dispatch.
    parser.add_argument("--helion-backend", default="triton", help=argparse.SUPPRESS)
    parser.add_argument(
        "--helion-force-flash-config",
        type=int,
        choices=(0, 1),
        default=0,
        help=(
            "Skip autotune and use the compiler-promoted flash seed when one "
            "exists, otherwise Config(block_sizes=[1,128,128]). Fast compile "
            "path; verifies the flash kernel fires without full search."
        ),
    )
    parser.add_argument(
        "--helion-force-autotune",
        type=int,
        choices=(0, 1),
        default=1,
        help=(
            "Pass force=True to bound.autotune for Helion impls. Set to 0 to "
            "allow full-effort cache reads while still autotuning cache misses."
        ),
    )
    parser.add_argument(
        "--helion-return-lse",
        type=int,
        choices=(0, 1),
        default=0,
        help=(
            "Use the LSE-returning Helion attention examples. The default "
            "uses output-only kernels for dense, causal, and biased attention so "
            "Helion does not compute an aux output that SDPA/FA4 omit."
        ),
    )
    parser.add_argument(
        "--helion-cute-benchmark-timer",
        choices=("wall", "event"),
        default="wall",
        help=(
            "Timer for Helion-CuTe benchmark samples. The default wall-clock "
            "path matches CuTe autotune timing; event mode uses the same CUDA "
            "event timing path as FlexAttention/SDPA/FA4 for opt-in comparisons."
        ),
    )
    parser.add_argument(
        "--helion-autotune-effort",
        choices=("none", "quick", "full"),
        default=None,
        help="Set HELION_AUTOTUNE_EFFORT for Helion impl subprocesses.",
    )
    parser.add_argument(
        "--helion-autotune-budget-seconds",
        type=int,
        default=None,
        help="Set HELION_AUTOTUNE_BUDGET_SECONDS for Helion impl subprocesses.",
    )
    parser.add_argument(
        "--helion-autotune-max-generations",
        type=int,
        default=None,
        help="Set HELION_AUTOTUNE_MAX_GENERATIONS for Helion impl subprocesses.",
    )
    parser.add_argument(
        "--helion-autotune-best-of-k",
        type=int,
        default=None,
        help="Set HELION_AUTOTUNE_BEST_OF_K for Helion impl subprocesses.",
    )
    parser.add_argument(
        "--helion-autotune-benchmark-timeout",
        type=int,
        default=None,
        help=(
            "Set HELION_AUTOTUNE_BENCHMARK_TIMEOUT for Helion impl subprocesses. "
            "Use a larger value for very long attention shapes whose slow "
            "candidate configs can exceed the default per-config timeout."
        ),
    )
    parser.add_argument(
        "--helion-autotune-accuracy-check",
        type=int,
        choices=(0, 1),
        default=None,
        help=(
            "Set HELION_AUTOTUNE_ACCURACY_CHECK for Helion impl subprocesses. "
            "Set to 0 for performance-only sweeps after correctness is checked "
            "separately."
        ),
    )
    parser.add_argument(
        "--helion-autotuner-initial-population",
        choices=("from_random", "from_best_available"),
        default=None,
        help=(
            "Set HELION_AUTOTUNER_INITIAL_POPULATION for Helion impl "
            "subprocesses. Use from_best_available for long full-effort "
            "attention sweeps so cached and compiler-seeded configs are tried "
            "before random exploration."
        ),
    )
    parser.add_argument(
        "--helion-env",
        action="append",
        type=_parse_key_value,
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Set an environment variable for Helion impls only. Repeat for "
            "multiple vars; forwarded through --impl all and --all-shapes."
        ),
    )
    parser.add_argument(
        "--helion-config",
        action="append",
        type=_parse_config_override,
        default=[],
        metavar="KEY=JSON",
        help=(
            "Set a helion.Config kwarg for Helion impls only, parsing VALUE as "
            "JSON when possible and as a string otherwise. Repeat to sweep "
            "prospective knobs, e.g. --helion-config block_sizes='[1,128,128]'."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.merge_json:
        _run_merge_json(args)
        return

    if args.causal and _uses_bias(args):
        raise SystemExit("--biased 1 is not compatible with --causal 1")

    if args.all_shapes:
        _run_all_shapes(args)
        return

    if args.impl == "all":
        payload = _run_all(args)
        _write_json_output(args, payload)
        if args.json:
            print(json.dumps(payload))
        else:
            _print_summary(payload)
        return

    result = _run_impl(args)
    _write_json_output(args, result)
    if args.json:
        print(json.dumps(result))
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
