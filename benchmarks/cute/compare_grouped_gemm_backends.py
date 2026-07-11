# ruff: noqa: ANN401
"""Compare Helion grouped GEMM against the CUTLASS CuTeDSL Blackwell path.

This harness reproduces the saved Helion-vs-CUTLASS grouped GEMM comparison
used for the grouped GEMM PR. Saved timing values are informational regression
references; current benchmark artifacts remain the source of truth. The module
intentionally keeps imports lightweight: Torch, Helion, Triton, and CUTLASS are
imported only by the per-case benchmark worker.

Examples:

    CUDA_VISIBLE_DEVICES=3 python benchmarks/cute/compare_grouped_gemm_backends.py \\
        --out-dir grouped_gemm_cutlass_results

    python benchmarks/cute/compare_grouped_gemm_backends.py --list-cases

    CUDA_VISIBLE_DEVICES=3 python benchmarks/cute/compare_grouped_gemm_backends.py \\
        --case doc_original --quick
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses
import importlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]

CTA_M = 128
CTA_N = 64
CTA_K = 64
DEFAULT_OUT_DIR = Path("grouped_gemm_cutlass_results")
HELION_DIRECT_CALL = (
    "benchmarks.cute.blackwell_grouped_gemm_direct.blackwell_grouped_gemm_nt_direct"
)
CUTLASS_CALL = "benchmarks.cute.grouped_deepgemm.blackwell_grouped_gemm_nt"
SAVED_FINAL_GEOMEAN_HELION_OVER_CUTLASS = 1.0880407542392165
SAVED_BASELINE_GEOMEAN_HELION_OVER_CUTLASS = 1.086090533826298
POST_CLEANUP_SAVED_GEOMEAN_HELION_OVER_CUTLASS = 1.087482028485604
SAVED_RESULT_NOTE = (
    "Saved timing values are informational regression references from prior "
    "validation runs; current benchmark artifacts are authoritative."
)

CACHE_ENV_DIRS = {
    "HELION_CACHE_DIR": "helion",
    "CUTE_DSL_CACHE_DIR": "cute_dsl",
    "TORCHINDUCTOR_CACHE_DIR": "torchinductor",
    "TRITON_CACHE_DIR": "triton",
    "CUDA_CACHE_PATH": "cuda_driver",
    "TORCH_EXTENSIONS_DIR": "torch_extensions",
}
ENV_SNAPSHOT_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "CUDA_HOME",
    "CUDA_CACHE_PATH",
    "CUTE_DSL_CACHE_DIR",
    "CUTE_DSL_DUMP_DIR",
    "CUTE_DSL_KEEP",
    "HELION_AUTOTUNE_EFFORT",
    "HELION_BACKEND",
    "HELION_CUTE_BLACKWELL_GROUPED_GEMM_DIRECT",
    "HELION_CUTE_MMA_IMPL",
    "HELION_PRINT_OUTPUT_CODE",
    "PYTHONPATH",
    "TORCHINDUCTOR_CACHE_DIR",
    "TORCH_EXTENSIONS_DIR",
    "TRITON_CACHE_DIR",
)


class BenchmarkSetupError(RuntimeError):
    pass


@dataclasses.dataclass(frozen=True)
class GroupedGemmCase:
    name: str
    label: str
    problem_sizes: tuple[tuple[int, int, int, int], ...]
    note: str
    saved_helion_over_cutlass: float | None = None
    saved_helion_median_ms: float | None = None
    saved_cutlass_median_ms: float | None = None

    @property
    def shape_label(self) -> str:
        groups = ", ".join(
            f"g{group}: {m}x{n}x{k}"
            for group, (m, n, k, _l_mode) in enumerate(self.problem_sizes)
        )
        return f"G{len(self.problem_sizes)} [{groups}]"


DEFAULT_PROBLEM_SIZES = (
    (128, 128, 128, 1),
    (512, 128, 128, 1),
    (128, 256, 128, 1),
)
DOC_PROBLEM_SIZES = (
    (8192, 1280, 32, 1),
    (16, 384, 1536, 1),
    (640, 1280, 16, 1),
    (640, 160, 16, 1),
)
DOC_NO_MN_TAIL_PROBLEM_SIZES = (
    (8192, 1280, 32, 1),
    (128, 384, 1536, 1),
    (640, 1280, 16, 1),
    (640, 128, 16, 1),
)

SAVED_CASES = (
    GroupedGemmCase(
        name="default_small_parity",
        label="default small parity",
        problem_sizes=DEFAULT_PROBLEM_SIZES,
        note="Existing no-tail small default shape; known parity control.",
        saved_helion_over_cutlass=1.000953546,
        saved_helion_median_ms=0.010232018,
        saved_cutlass_median_ms=0.010222271,
    ),
    GroupedGemmCase(
        name="doc_no_mn_tail",
        label="doc no M/N tail",
        problem_sizes=DOC_NO_MN_TAIL_PROBLEM_SIZES,
        note="Doc envelope with M/N made full-tile.",
        saved_helion_over_cutlass=1.102637306,
        saved_helion_median_ms=0.022596965,
        saved_cutlass_median_ms=0.020493561,
    ),
    GroupedGemmCase(
        name="doc_no_mn_g3_192_extra_full",
        label="doc no M/N tail, g3 192",
        problem_sizes=(
            (8192, 1280, 32, 1),
            (128, 384, 1536, 1),
            (640, 1280, 16, 1),
            (640, 192, 16, 1),
        ),
        note="Doc envelope, no M/N tails, but group 3 has three full N tiles.",
        saved_helion_over_cutlass=1.100041509,
        saved_helion_median_ms=0.022540174,
        saved_cutlass_median_ms=0.020490294,
    ),
    GroupedGemmCase(
        name="doc_original",
        label="doc original",
        problem_sizes=DOC_PROBLEM_SIZES,
        note="Existing documented mixed-tail shape.",
        saved_helion_over_cutlass=1.106228864,
        saved_helion_median_ms=0.022663946,
        saved_cutlass_median_ms=0.020487574,
    ),
    GroupedGemmCase(
        name="doc_mtail_g1_g3_192_extra_full",
        label="doc M-tail g1, g3 192",
        problem_sizes=(
            (8192, 1280, 32, 1),
            (16, 384, 1536, 1),
            (640, 1280, 16, 1),
            (640, 192, 16, 1),
        ),
        note="Doc envelope, original M tail plus group 3 three full N tiles.",
        saved_helion_over_cutlass=1.102121328,
        saved_helion_median_ms=0.022576911,
        saved_cutlass_median_ms=0.020484960,
    ),
    GroupedGemmCase(
        name="doc_mtail_g1_only",
        label="doc M-tail g1 only",
        problem_sizes=(
            (8192, 1280, 32, 1),
            (16, 384, 1536, 1),
            (640, 1280, 16, 1),
            (640, 128, 16, 1),
        ),
        note="Doc envelope, only the original small-M group has an M tail.",
        saved_helion_over_cutlass=1.103667708,
        saved_helion_median_ms=0.022496863,
        saved_cutlass_median_ms=0.020383728,
    ),
    GroupedGemmCase(
        name="doc_ntail_g3_160",
        label="doc N-tail g3 160",
        problem_sizes=(
            (8192, 1280, 32, 1),
            (128, 384, 1536, 1),
            (640, 1280, 16, 1),
            (640, 160, 16, 1),
        ),
        note="Doc envelope, only original N=160 tail; adds one N tile for group 3.",
        saved_helion_over_cutlass=1.104962994,
        saved_helion_median_ms=0.022645762,
        saved_cutlass_median_ms=0.020494589,
    ),
)
SAVED_CASE_NAMES = tuple(case.name for case in SAVED_CASES)
CASES_BY_NAME = {case.name: case for case in SAVED_CASES}
ALL_CASE_NAMES = SAVED_CASE_NAMES
IMPL_NAMES = ("helion_retained", "helion_raw", "cutlass")


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def shape_stats(
    problem_sizes: Sequence[tuple[int, int, int, int]],
) -> dict[str, object]:
    per_group: list[dict[str, object]] = []
    total_ctas = 0
    total_tail_any = 0
    total_tail_m = 0
    total_tail_n = 0
    total_k_tail_ctas = 0
    total_flops = 0
    padded_m = 0
    max_n = 0
    max_k = 0
    for group, (m, n, k, l_mode) in enumerate(problem_sizes):
        if l_mode != 1:
            raise BenchmarkSetupError(f"unsupported L mode in group {group}: {l_mode}")
        m_tiles = _ceil_div(m, CTA_M)
        n_tiles = _ceil_div(n, CTA_N)
        k_tiles = _ceil_div(k, CTA_K)
        ctas = m_tiles * n_tiles
        m_tail = m % CTA_M != 0
        n_tail = n % CTA_N != 0
        k_tail = k % CTA_K != 0
        tail_m_ctas = n_tiles if m_tail else 0
        tail_n_ctas = m_tiles if n_tail else 0
        tail_any = tail_m_ctas + tail_n_ctas - (1 if m_tail and n_tail else 0)
        k_tail_ctas = ctas if k_tail else 0
        padded_group_m = m_tiles * CTA_M
        flops = 2 * m * n * k
        per_group.append(
            {
                "group": group,
                "m": m,
                "n": n,
                "k": k,
                "m_tiles": m_tiles,
                "n_tiles": n_tiles,
                "k_tiles": k_tiles,
                "ctas": ctas,
                "m_tail": m_tail,
                "n_tail": n_tail,
                "k_tail": k_tail,
                "d_tail_ctas_any": tail_any,
                "m_tail_ctas": tail_m_ctas,
                "n_tail_ctas": tail_n_ctas,
                "k_tail_ctas": k_tail_ctas,
                "padded_m": padded_group_m,
                "flops": flops,
            }
        )
        total_ctas += ctas
        total_tail_any += tail_any
        total_tail_m += tail_m_ctas
        total_tail_n += tail_n_ctas
        total_k_tail_ctas += k_tail_ctas
        total_flops += flops
        padded_m += padded_group_m
        max_n = max(max_n, n)
        max_k = max(max_k, k)
    return {
        "groups": len(problem_sizes),
        "padded_m": padded_m,
        "max_n": max_n,
        "max_k": max_k,
        "total_ctas": total_ctas,
        "d_tail_ctas_any": total_tail_any,
        "m_tail_ctas": total_tail_m,
        "n_tail_ctas": total_tail_n,
        "k_tail_ctas": total_k_tail_ctas,
        "d_tail_cta_fraction": total_tail_any / total_ctas if total_ctas else 0.0,
        "m_tail_cta_fraction": total_tail_m / total_ctas if total_ctas else 0.0,
        "n_tail_cta_fraction": total_tail_n / total_ctas if total_ctas else 0.0,
        "k_tail_cta_fraction": total_k_tail_ctas / total_ctas if total_ctas else 0.0,
        "total_flops": total_flops,
        "per_group": per_group,
    }


def case_to_dict(case: GroupedGemmCase) -> dict[str, object]:
    return {
        "name": case.name,
        "label": case.label,
        "shape_label": case.shape_label,
        "problem_sizes": [list(item) for item in case.problem_sizes],
        "shape_stats": shape_stats(case.problem_sizes),
        "note": case.note,
        "saved_helion_over_cutlass": case.saved_helion_over_cutlass,
        "saved_helion_median_ms": case.saved_helion_median_ms,
        "saved_cutlass_median_ms": case.saved_cutlass_median_ms,
    }


def geomean(values: Sequence[float]) -> float | None:
    positives = [float(value) for value in values if float(value) > 0.0]
    if not positives:
        return None
    return math.exp(sum(math.log(value) for value in positives) / len(positives))


def _float_value(value: object) -> float:
    if not isinstance(value, int | float | str):
        raise TypeError(f"expected numeric value, got {type(value).__name__}")
    return float(value)


def ratio_summary(
    rows: Sequence[Mapping[str, object]],
    *,
    ratio_key: str = "retained_over_cutlass",
) -> dict[str, object]:
    ratios = [
        _float_value(row[ratio_key])
        for row in rows
        if row.get("status", "ok") == "ok" and row.get(ratio_key) is not None
    ]
    return {
        "count": len(ratios),
        "wins": sum(ratio < 1.0 for ratio in ratios),
        "geomean": geomean(ratios),
        "best": min(ratios) if ratios else None,
        "worst": max(ratios) if ratios else None,
    }


def compare_regression(fresh: float | None, baseline: float) -> dict[str, object]:
    if fresh is None:
        return {"status": "missing", "baseline": baseline}
    return {
        "fresh": fresh,
        "baseline": baseline,
        "delta": fresh - baseline,
        "ratio": fresh / baseline,
        "regresses": fresh > baseline,
    }


def json_ready(value: object) -> object:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return json_ready(dataclasses.asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n")


def _default_cuda_home() -> Path | None:
    raw = os.environ.get("CUDA_HOME")
    if raw:
        return Path(raw)
    default = Path("/usr/local/cuda-13.0")
    return default if default.exists() else None


def _positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {text}")
    return value


def _non_negative_int(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError(f"expected a non-negative integer, got {text}")
    return value


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark benchmarks.cute.blackwell_grouped_gemm_direct."
            "blackwell_grouped_gemm_nt_direct "
            "against the CUTLASS CuTeDSL Blackwell grouped GEMM path."
        )
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--list-cases",
        action="store_true",
        help="Print saved case definitions without importing CUDA dependencies.",
    )
    mode.add_argument(
        "--summarize-only",
        action="store_true",
        help="Summarize existing per-case result.json files under --out-dir.",
    )
    mode.add_argument(
        "--run-case",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--cases",
        choices=("saved", "all"),
        default="saved",
        help="Case suite to run when --case is not specified.",
    )
    parser.add_argument(
        "--case",
        choices=ALL_CASE_NAMES,
        action="append",
        help="Run one saved case; may be repeated.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--summary-md", type=Path)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--cuda-home", type=Path, default=_default_cuda_home())
    parser.add_argument("--cuda-visible-devices")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compile-warmups", type=_positive_int, default=2)
    parser.add_argument("--num-runs", type=_positive_int, default=5)
    parser.add_argument("--warmup-ms", type=_non_negative_int, default=1000)
    parser.add_argument("--rep-ms", type=_positive_int, default=500)
    parser.add_argument("--cache-warmup-calls", type=_non_negative_int, default=5)
    parser.add_argument("--thermal-warmup-ms", type=_non_negative_int, default=10000)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use short timing windows for smoke checks, not saved-result reproduction.",
    )
    parser.add_argument(
        "--impls",
        default="helion_retained,helion_raw,cutlass",
        help="Comma-separated subset of helion_retained,helion_raw,cutlass.",
    )
    parser.add_argument(
        "--stream-subprocesses",
        action="store_true",
        help="Stream worker stdout/stderr while also saving them under --out-dir.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON for list mode.")
    return parser


def _apply_quick_preset(args: argparse.Namespace) -> None:
    if not args.quick:
        return
    args.compile_warmups = min(args.compile_warmups, 1)
    args.num_runs = min(args.num_runs, 1)
    args.warmup_ms = min(args.warmup_ms, 25)
    args.rep_ms = min(args.rep_ms, 25)
    args.cache_warmup_calls = min(args.cache_warmup_calls, 1)
    args.thermal_warmup_ms = min(args.thermal_warmup_ms, 0)


def selected_cases(args: argparse.Namespace) -> tuple[GroupedGemmCase, ...]:
    if args.case:
        return tuple(CASES_BY_NAME[name] for name in args.case)
    if args.cases in ("saved", "all"):
        return SAVED_CASES
    raise AssertionError(f"unexpected case suite {args.cases!r}")


def _parse_impls(value: str) -> tuple[str, ...]:
    impls = tuple(item.strip() for item in value.split(",") if item.strip())
    invalid = sorted(set(impls) - set(IMPL_NAMES))
    if invalid:
        joined = ", ".join(invalid)
        raise argparse.ArgumentTypeError(f"unknown impl(s): {joined}")
    if not impls:
        raise argparse.ArgumentTypeError("expected at least one impl")
    return impls


def list_cases(args: argparse.Namespace) -> int:
    payload = {
        "saved_case_names": SAVED_CASE_NAMES,
        "saved_result_note": SAVED_RESULT_NOTE,
        "saved_final_geomean_helion_over_cutlass": (
            SAVED_FINAL_GEOMEAN_HELION_OVER_CUTLASS
        ),
        "cases": [case_to_dict(case) for case in selected_cases(args)],
    }
    if args.json:
        print(json.dumps(json_ready(payload), indent=2, sort_keys=True))
        return 0
    for case in selected_cases(args):
        print(f"{case.name}: {case.label}")
        print(f"  {case.shape_label}")
        print(f"  {case.note}")
        if case.saved_helion_over_cutlass is not None:
            print(f"  saved Helion/CUTLASS: {case.saved_helion_over_cutlass:.9f}")
    return 0


def _repo_pythonpath(env: Mapping[str, str]) -> str:
    parts = [str(REPO_ROOT)]
    for raw in env.get("PYTHONPATH", "").split(os.pathsep):
        if not raw:
            continue
        try:
            path = Path(raw).resolve()
        except OSError:
            parts.append(raw)
            continue
        if path != REPO_ROOT:
            parts.append(raw)
    return os.pathsep.join(parts)


def base_child_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = _repo_pythonpath(env)
    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    if args.cuda_home is not None:
        env["CUDA_HOME"] = str(args.cuda_home)
        env["PATH"] = str(args.cuda_home / "bin") + os.pathsep + env.get("PATH", "")
    return env


def configure_case_env(case_dir: Path) -> dict[str, str]:
    os.environ["HELION_BACKEND"] = "cute"
    os.environ["HELION_CUTE_MMA_IMPL"] = "tcgen05"
    os.environ["HELION_PRINT_OUTPUT_CODE"] = "0"
    os.environ["HELION_AUTOTUNE_EFFORT"] = ""
    os.environ.pop("HELION_CUTE_BLACKWELL_GROUPED_GEMM_DIRECT", None)
    os.environ["CUTE_DSL_KEEP"] = "ir,ptx,cubin"
    os.environ["CUTE_DSL_DUMP_DIR"] = str(case_dir / "dump")
    for env_name, subdir in CACHE_ENV_DIRS.items():
        os.environ[env_name] = str(case_dir / "cache" / subdir)
    for path in [
        case_dir / "dump",
        *(case_dir / "cache" / subdir for subdir in CACHE_ENV_DIRS.values()),
    ]:
        path.mkdir(parents=True, exist_ok=True)
    return {key: os.environ.get(key, "") for key in ENV_SNAPSHOT_KEYS}


def _ensure_repo_on_path() -> None:
    repo_text = str(REPO_ROOT)
    sys.path[:] = [repo_text, *[item for item in sys.path if item != repo_text]]


def _torch_dtype(torch_module: Any, name: str) -> Any:
    return {"float16": torch_module.float16}[name]


def _make_outputs(
    torch_module: Any,
    group_a: Sequence[Any],
    group_b: Sequence[Any],
    dtype: Any,
) -> tuple[Any, ...]:
    return tuple(
        torch_module.empty(int(a.size(0)), int(b.size(0)), device=a.device, dtype=dtype)
        for a, b in zip(group_a, group_b, strict=True)
    )


def _fill_outputs(torch_module: Any, outputs: Sequence[Any], value: float) -> None:
    for out in outputs:
        out.fill_(value)
    torch_module.cuda.synchronize()


def _max_abs_and_close(
    torch_module: Any,
    actual: Sequence[Any],
    expected: Sequence[Any],
) -> dict[str, object]:
    max_abs = 0.0
    max_rel = 0.0
    for got, ref in zip(actual, expected, strict=True):
        diff = (got.float() - ref.float()).abs()
        if int(diff.numel()):
            max_abs = max(max_abs, float(diff.max().item()))
            denom = ref.float().abs().clamp_min(1.0)
            max_rel = max(max_rel, float((diff / denom).max().item()))
        torch_module.testing.assert_close(got, ref, rtol=2e-2, atol=2e-2)
    return {"ok": True, "max_abs": max_abs, "max_rel": max_rel}


def _capture_graph(
    torch_module: Any,
    fn: Callable[[], object],
) -> tuple[Any, object]:
    graph = torch_module.cuda.CUDAGraph()
    with torch_module.cuda.graph(graph):
        value = fn()
    torch_module.cuda.synchronize()
    return graph, value


def _get_generated_launch(grouped_mod: Any) -> Any:
    launch = getattr(grouped_mod, "_BLACKWELL_GENERATED_LAST_STABLE_LAUNCH", None)
    if launch is None:
        raise BenchmarkSetupError("generated stable launch cache was not populated")
    cuda_graph = getattr(launch, "cuda_graph", None)
    if cuda_graph is None:
        raise BenchmarkSetupError("generated stable launch did not retain a CUDA graph")
    return launch


def _gpu_warmup(torch_module: Any, duration_ms: int) -> None:
    if duration_ms <= 0:
        return
    a = torch_module.randn(4096, 4096, device="cuda", dtype=torch_module.bfloat16)
    torch_module.cuda.synchronize()
    target_s = duration_ms / 1000.0
    start = time.time()
    while time.time() - start < target_s:
        for _ in range(50):
            a = a @ a
        torch_module.cuda.synchronize()


def bench_steady_cuda_events(
    torch_module: Any,
    fn: Callable[[], object],
    *,
    num_runs: int,
    warmup_ms: int,
    rep_ms: int,
    cache_warmup_calls: int,
    thermal_warmup_ms: int,
) -> dict[str, object]:
    from triton.testing import do_bench

    for _ in range(cache_warmup_calls):
        fn()
    torch_module.cuda.synchronize()
    _gpu_warmup(torch_module, thermal_warmup_ms)
    runs: list[float] = []
    for _ in range(num_runs):
        ms = do_bench(fn, warmup=warmup_ms, rep=rep_ms, return_mode="mean")
        if isinstance(ms, tuple):
            ms = ms[0]
        runs.append(_float_value(ms))
    return {
        "best_ms": min(runs),
        "median_ms": statistics.median(runs),
        "mean_ms": statistics.fmean(runs),
        "std_ms": statistics.stdev(runs) if len(runs) > 1 else 0.0,
        "runs_ms": runs,
        "method": "triton.testing.do_bench_cuda_events",
        "inner_return_mode": "mean",
        "gate_metric": "median_ms across do_bench runs",
        "num_runs": num_runs,
        "warmup_ms": warmup_ms,
        "rep_ms": rep_ms,
        "cache_warmup_calls": cache_warmup_calls,
        "thermal_warmup_ms": thermal_warmup_ms,
    }


def benchmark_settings(args: argparse.Namespace) -> dict[str, object]:
    return {
        "fresh_subprocess_per_case": not args.run_case,
        "timer": "CUDA events via triton.testing.do_bench",
        "cache_warmup_calls": args.cache_warmup_calls,
        "thermal_warmup_ms": args.thermal_warmup_ms,
        "num_runs": args.num_runs,
        "warmup_ms": args.warmup_ms,
        "rep_ms": args.rep_ms,
        "compile_warmups": args.compile_warmups,
        "gate_metric": "median_ms across do_bench runs",
        "diagnostic_metric": "best_ms",
        "quick": bool(args.quick),
    }


def _import_confirmation(
    blackwell_direct: Any,
    helion_mod: Any,
    cutlass_grouped: Any,
) -> dict[str, object]:
    return {
        "repo_root": str(REPO_ROOT),
        "blackwell_direct_module": blackwell_direct.__name__,
        "blackwell_direct_file": str(Path(blackwell_direct.__file__).resolve()),
        "helion_file": str(Path(helion_mod.__file__).resolve()),
        "cutlass_grouped_file": str(Path(cutlass_grouped.__file__).resolve()),
        "helion_call": HELION_DIRECT_CALL,
        "cutlass_call": CUTLASS_CALL,
        "python": sys.executable,
        "cwd": os.getcwd(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "sys_path_prefix": sys.path[:6],
        "hidden_env_route": os.environ.get(
            "HELION_CUTE_BLACKWELL_GROUPED_GEMM_DIRECT",
            "",
        ),
    }


def run_case(args: argparse.Namespace) -> int:
    if not args.case or len(args.case) != 1:
        raise BenchmarkSetupError("--run-case requires exactly one --case")
    case = CASES_BY_NAME[args.case[0]]
    impls = _parse_impls(args.impls)
    case_dir = (args.out_dir / "cutlass" / case.name).resolve()
    case_dir.mkdir(parents=True, exist_ok=True)
    env_snapshot = configure_case_env(case_dir)

    _ensure_repo_on_path()
    import torch

    import helion

    if not torch.cuda.is_available():
        raise BenchmarkSetupError("CUDA is required to run grouped GEMM timings")
    from benchmarks.cute import grouped_deepgemm as cutlass_grouped

    blackwell_direct = importlib.import_module(
        "benchmarks.cute.blackwell_grouped_gemm_direct"
    )
    import_confirmation = _import_confirmation(
        blackwell_direct,
        helion,
        cutlass_grouped,
    )
    print(json.dumps({"import_confirmation": import_confirmation}, sort_keys=True))

    torch.cuda.set_device(0)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda", 0)
    dtype = _torch_dtype(torch, "float16")

    (group_a, group_b), expected = blackwell_direct.make_blackwell_grouped_gemm_nt_args(
        case.problem_sizes,
        dtype=dtype,
        device=device,
    )
    out_helion = _make_outputs(torch, group_a, group_b, dtype)
    out_cutlass = _make_outputs(torch, group_a, group_b, dtype)

    def helion_call() -> object:
        return blackwell_direct.blackwell_grouped_gemm_nt_direct(
            group_a,
            group_b,
            out_groups=out_helion,
        )

    def cutlass_call() -> object:
        return cutlass_grouped.blackwell_grouped_gemm_nt(
            group_a,
            group_b,
            out_groups=out_cutlass,
        )

    setup_start = time.perf_counter()
    for _ in range(args.compile_warmups):
        helion_call()
    torch.cuda.synchronize()
    launch = _get_generated_launch(blackwell_direct)
    helion_raw_graph, _helion_raw_captured = _capture_graph(torch, helion_call)
    torch.cuda.synchronize()
    helion_setup_s = time.perf_counter() - setup_start

    setup_start = time.perf_counter()
    for _ in range(args.compile_warmups):
        cutlass_call()
    torch.cuda.synchronize()
    cutlass_graph, _cutlass_captured = _capture_graph(torch, cutlass_call)
    torch.cuda.synchronize()
    cutlass_setup_s = time.perf_counter() - setup_start

    _fill_outputs(torch, out_helion, -3.0)
    launch.cuda_graph.replay()
    torch.cuda.synchronize()
    helion_retained_correctness = _max_abs_and_close(torch, out_helion, expected)
    _fill_outputs(torch, out_helion, -5.0)
    helion_raw_graph.replay()
    torch.cuda.synchronize()
    helion_raw_correctness = _max_abs_and_close(torch, out_helion, expected)
    _fill_outputs(torch, out_cutlass, -7.0)
    cutlass_graph.replay()
    torch.cuda.synchronize()
    cutlass_correctness = _max_abs_and_close(torch, out_cutlass, expected)

    replays: dict[str, Callable[[], object]] = {
        "helion_retained": launch.cuda_graph.replay,
        "helion_raw": helion_raw_graph.replay,
        "cutlass": cutlass_graph.replay,
    }
    timings: dict[str, object] = {}
    for impl in impls:
        timings[impl] = bench_steady_cuda_events(
            torch,
            replays[impl],
            num_runs=args.num_runs,
            warmup_ms=args.warmup_ms,
            rep_ms=args.rep_ms,
            cache_warmup_calls=args.cache_warmup_calls,
            thermal_warmup_ms=args.thermal_warmup_ms,
        )

    ratios: dict[str, float] = {}
    deltas_ms: dict[str, float] = {}
    if "cutlass" in timings:
        cutlass_timing = timings["cutlass"]
        assert isinstance(cutlass_timing, Mapping)
        cutlass_ms = float(cutlass_timing["median_ms"])
        for name, timing in timings.items():
            if name == "cutlass":
                continue
            assert isinstance(timing, Mapping)
            median_ms = float(timing["median_ms"])
            ratios[name] = median_ms / cutlass_ms
            deltas_ms[name] = median_ms - cutlass_ms

    result: dict[str, object] = {
        "comparison": "cutlass_cutedsl_blackwell_grouped_gemm",
        "case": case.name,
        "label": case.label,
        "shape_label": case.shape_label,
        "note": case.note,
        "problem_sizes": [list(item) for item in case.problem_sizes],
        "shape_stats": shape_stats(case.problem_sizes),
        "settings": benchmark_settings(args)
        | {
            "seed": args.seed,
            "impls": impls,
            "env": env_snapshot,
        },
        "device_name": torch.cuda.get_device_name(device),
        "capability": list(torch.cuda.get_device_capability(device)),
        "import_confirmation": import_confirmation,
        "correctness": {
            "helion_retained": helion_retained_correctness,
            "helion_raw": helion_raw_correctness,
            "cutlass": cutlass_correctness,
        },
        "compile_excluded_setup": {
            "helion_setup_s": helion_setup_s,
            "cutlass_setup_s": cutlass_setup_s,
        },
        "timings": timings,
        "ratios_to_cutlass": ratios,
        "deltas_to_cutlass_ms": deltas_ms,
        "saved": {
            "note": SAVED_RESULT_NOTE,
            "helion_over_cutlass": case.saved_helion_over_cutlass,
            "helion_median_ms": case.saved_helion_median_ms,
            "cutlass_median_ms": case.saved_cutlass_median_ms,
        },
    }
    out_path = case_dir / "result.json"
    write_json(out_path, result)
    print(
        json.dumps(
            {
                "case": case.name,
                "result": str(out_path),
                "ratios_to_cutlass": ratios,
            },
            sort_keys=True,
        )
    )
    return 0


def summarize_cutlass_results(
    out_dir: Path,
    cases: Sequence[GroupedGemmCase],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for case in cases:
        path = out_dir / "cutlass" / case.name / "result.json"
        if not path.exists():
            rows.append(
                {
                    "case": case.name,
                    "label": case.label,
                    "shape_label": case.shape_label,
                    "status": "missing",
                    "path": str(path),
                    "saved_helion_over_cutlass": case.saved_helion_over_cutlass,
                }
            )
            continue
        data = json.loads(path.read_text())
        timings = data["timings"]
        stats = data["shape_stats"]
        ratios = data.get("ratios_to_cutlass", {})
        helion_retained = timings.get("helion_retained", {})
        helion_raw = timings.get("helion_raw", {})
        cutlass = timings.get("cutlass", {})
        rows.append(
            {
                "case": case.name,
                "label": data.get("label", case.label),
                "shape_label": data.get("shape_label", case.shape_label),
                "status": "ok",
                "problem_sizes": data["problem_sizes"],
                "total_ctas": stats["total_ctas"],
                "d_tail_ctas_any": stats["d_tail_ctas_any"],
                "m_tail_ctas": stats["m_tail_ctas"],
                "n_tail_ctas": stats["n_tail_ctas"],
                "k_tail_ctas": stats["k_tail_ctas"],
                "helion_retained_median_ms": helion_retained.get("median_ms"),
                "helion_retained_best_ms": helion_retained.get("best_ms"),
                "helion_raw_median_ms": helion_raw.get("median_ms"),
                "cutlass_median_ms": cutlass.get("median_ms"),
                "cutlass_best_ms": cutlass.get("best_ms"),
                "retained_over_cutlass": ratios.get("helion_retained"),
                "raw_over_cutlass": ratios.get("helion_raw"),
                "retained_delta_ms": data.get("deltas_to_cutlass_ms", {}).get(
                    "helion_retained"
                ),
                "result_path": str(path),
                "import_confirmation": data.get("import_confirmation", {}),
                "saved_helion_over_cutlass": case.saved_helion_over_cutlass,
            }
        )
    return rows


def _fmt_ratio(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{_float_value(value):.9f}"


def write_markdown_summary(path: Path, summary: Mapping[str, object]) -> None:
    cutlass = summary["cutlass"]
    assert isinstance(cutlass, Mapping)
    rows = cutlass["rows"]
    assert isinstance(rows, Sequence)
    command = summary.get("top_level_command")
    lines = [
        "# Grouped GEMM Helion vs CUTLASS CuTeDSL",
        "",
        "## Command",
        "",
        f"`{command}`" if command else "`n/a`",
        "",
        "## CUTLASS Ratios",
        "",
        (
            "- Geomean Helion/CUTLASS: "
            f"{_fmt_ratio(cutlass.get('geomean_helion_over_cutlass'))}"
        ),
        (
            "- Saved final geomean: "
            f"{_fmt_ratio(cutlass.get('saved_final_geomean_helion_over_cutlass'))}"
        ),
        (
            "- Saved baseline geomean: "
            f"{_fmt_ratio(cutlass.get('saved_baseline_geomean_helion_over_cutlass'))}"
        ),
        f"- Saved values note: {cutlass.get('saved_result_note', SAVED_RESULT_NOTE)}",
        "",
        "| Case | Label | Helion/CUTLASS | Saved | Helion ms | CUTLASS ms |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        assert isinstance(row, Mapping)
        lines.append(
            "| "
            f"{row['case']} | "
            f"{row.get('label', '')} | "
            f"{_fmt_ratio(row.get('retained_over_cutlass'))} | "
            f"{_fmt_ratio(row.get('saved_helion_over_cutlass'))} | "
            f"{_fmt_ratio(row.get('helion_retained_median_ms'))} | "
            f"{_fmt_ratio(row.get('cutlass_median_ms'))} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _summary_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    summary_json = args.summary_json or args.out_dir / "summary.json"
    summary_md = args.summary_md or args.out_dir / "summary.md"
    return summary_json.resolve(), summary_md.resolve()


def build_summary(
    args: argparse.Namespace,
    cases: Sequence[GroupedGemmCase],
    commands: Sequence[Mapping[str, object]],
    failures: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    rows = summarize_cutlass_results(args.out_dir, cases)
    ratio_stats = ratio_summary(rows)
    geomean_ratio = ratio_stats["geomean"]
    assert geomean_ratio is None or isinstance(geomean_ratio, float)
    return {
        "artifact_dir": str(args.out_dir),
        "repo": {
            "root": str(REPO_ROOT),
            "head": _git_output("rev-parse", "HEAD"),
            "branch": _git_output("branch", "--show-current"),
            "status_short": _git_output("status", "--short"),
        },
        "commands": list(commands),
        "methodology": benchmark_settings(args),
        "helion_call": HELION_DIRECT_CALL,
        "cutlass_call": CUTLASS_CALL,
        "cutlass": {
            "rows": rows,
            "saved_result_note": SAVED_RESULT_NOTE,
            "wins": ratio_stats["wins"],
            "total": ratio_stats["count"],
            "geomean_helion_over_cutlass": geomean_ratio,
            "best_helion_over_cutlass": ratio_stats["best"],
            "worst_helion_over_cutlass": ratio_stats["worst"],
            "saved_final_geomean_helion_over_cutlass": (
                SAVED_FINAL_GEOMEAN_HELION_OVER_CUTLASS
            ),
            "saved_baseline_geomean_helion_over_cutlass": (
                SAVED_BASELINE_GEOMEAN_HELION_OVER_CUTLASS
            ),
            "post_cleanup_saved_geomean_helion_over_cutlass": (
                POST_CLEANUP_SAVED_GEOMEAN_HELION_OVER_CUTLASS
            ),
            "regression_vs_saved_final": compare_regression(
                geomean_ratio,
                SAVED_FINAL_GEOMEAN_HELION_OVER_CUTLASS,
            ),
            "regression_vs_saved_baseline": compare_regression(
                geomean_ratio,
                SAVED_BASELINE_GEOMEAN_HELION_OVER_CUTLASS,
            ),
            "regression_vs_post_cleanup_saved": compare_regression(
                geomean_ratio,
                POST_CLEANUP_SAVED_GEOMEAN_HELION_OVER_CUTLASS,
            ),
        },
        "failures": list(failures),
        "top_level_command": " ".join(sys.argv),
    }


def _git_output(*args: str) -> str:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError:
        return ""
    return proc.stdout.strip()


def summarize_only(args: argparse.Namespace) -> int:
    cases = selected_cases(args)
    summary = build_summary(args, cases, commands=(), failures=())
    summary_json, summary_md = _summary_paths(args)
    write_json(summary_json, summary)
    write_markdown_summary(summary_md, summary)
    print(json.dumps({"summary": str(summary_json), "markdown": str(summary_md)}))
    return 0


def _run_worker_command(
    args: argparse.Namespace,
    case: GroupedGemmCase,
    env: Mapping[str, str],
) -> tuple[int, str, str, list[str]]:
    cmd = [
        str(args.python),
        str(Path(__file__).resolve()),
        "--run-case",
        "--case",
        case.name,
        "--out-dir",
        str(args.out_dir),
        "--seed",
        str(args.seed),
        "--compile-warmups",
        str(args.compile_warmups),
        "--num-runs",
        str(args.num_runs),
        "--warmup-ms",
        str(args.warmup_ms),
        "--rep-ms",
        str(args.rep_ms),
        "--cache-warmup-calls",
        str(args.cache_warmup_calls),
        "--thermal-warmup-ms",
        str(args.thermal_warmup_ms),
        "--impls",
        args.impls,
    ]
    if args.cuda_home is not None:
        cmd.extend(["--cuda-home", str(args.cuda_home)])
    if args.cuda_visible_devices is not None:
        cmd.extend(["--cuda-visible-devices", args.cuda_visible_devices])
    if args.quick:
        cmd.append("--quick")
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        env=dict(env),
        text=True,
        capture_output=True,
        check=False,
    )
    if args.stream_subprocesses:
        print(proc.stdout, end="")
        print(proc.stderr, end="", file=sys.stderr)
    return proc.returncode, proc.stdout, proc.stderr, cmd


def run_all(args: argparse.Namespace) -> int:
    cases = selected_cases(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    env = base_child_env(args)
    commands: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for case in cases:
        label = f"cutlass:{case.name}"
        print(json.dumps({"starting": label}, sort_keys=True), flush=True)
        returncode, stdout, stderr, cmd = _run_worker_command(args, case, env)
        safe_label = label.replace(":", "_").replace("/", "_")
        stdout_path = args.out_dir / f"{safe_label}.stdout.txt"
        stderr_path = args.out_dir / f"{safe_label}.stderr.txt"
        stdout_path.write_text(stdout)
        stderr_path.write_text(stderr)
        command_record: dict[str, object] = {
            "label": label,
            "cmd": cmd,
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
            "returncode": returncode,
        }
        commands.append(command_record)
        if returncode != 0:
            failures.append(command_record)
            if not args.stream_subprocesses:
                print(stderr, file=sys.stderr)

    summary = build_summary(args, cases, commands=commands, failures=failures)
    summary_json, summary_md = _summary_paths(args)
    write_json(summary_json, summary)
    write_markdown_summary(summary_md, summary)
    print(json.dumps({"summary": str(summary_json), "markdown": str(summary_md)}))
    return 1 if failures else 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    _apply_quick_preset(args)
    args.out_dir = args.out_dir.resolve()
    try:
        _parse_impls(args.impls)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    if args.list_cases:
        return list_cases(args)
    if args.summarize_only:
        return summarize_only(args)
    if args.run_case:
        return run_case(args)
    return run_all(args)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print(
            "Running the saved grouped GEMM CUTLASS comparison. "
            "Use --list-cases to inspect cases or --quick for smoke timing.",
            file=sys.stderr,
        )
    raise SystemExit(main())
