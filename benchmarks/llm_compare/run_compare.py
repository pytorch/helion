"""Compare LFBOTreeSearch vs LLMSeededLFBOTreeSearch across a fixed kernel set.

Runs each (autotuner, kernel) cell sequentially on a single GPU, captures
end-to-end wall-clock and the helion-vs-baseline timings printed by each
example's `run_example` call, then writes a results table.

Usage:
    python benchmarks/llm_compare/run_compare.py [--gpu 0] [--timeout 2400]
                                                 [--out-tag mytag]

Set HELION_LLM_API_KEY (e.g. from a key helper) before invocation when the
LLM-seeded cell needs it. The harness sets HELION_SKIP_CACHE=1 per cell so
neither autotuner reuses the other's cached best_config.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as dt
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CONDA_PY = sys.executable

KERNELS = [
    "attention",
    "matmul",
    "softmax",
    "gdn_fwd_h",
    "layer_norm",
    "rope",
]

AUTOTUNERS = [
    {
        "label": "LFBO",
        "autotuner": "LFBOTreeSearch",
        "llm": False,
    },
    {
        "label": "LLMSeededLFBO_opus47_max_fast",
        "autotuner": "LLMSeededLFBOTreeSearch",
        "llm": True,
        "llm_env": {
            "HELION_LLM_PROVIDER": "anthropic",
            "HELION_LLM_MODEL": "claude-opus-4-7",
            "HELION_LLM_EFFORT_LEVEL": "max",
            "HELION_LLM_FAST_MODE": "1",
        },
    },
]


BENCH_RE = re.compile(
    r"^(?P<impl>\S+)\s+(?P<ms>\d+\.\d+)\s+(?P<speedup>\S+)",
    re.MULTILINE,
)


@dataclasses.dataclass
class CellResult:
    autotuner_label: str
    kernel: str
    wall_clock_s: float
    exit_code: int
    helion_timings: dict[str, float]
    baseline_timings: dict[str, float]
    error: str | None


def parse_bench_blocks(stderr_text: str) -> tuple[dict[str, float], dict[str, float]]:
    """Extract helion-vs-baseline ms timings from `run_example` output."""
    baseline_names = {"torch", "ref", "flex", "compiled", "torch_compile", "eager"}
    helion: dict[str, float] = {}
    baselines: dict[str, float] = {}
    for match in BENCH_RE.finditer(stderr_text):
        impl = match.group("impl")
        ms = float(match.group("ms"))
        speedup = match.group("speedup")
        if speedup.endswith("(ref)") or impl in baseline_names:
            baselines.setdefault(impl, ms)
        else:
            helion.setdefault(impl, ms)
    return helion, baselines


def run_cell(
    *,
    kernel: str,
    autotuner_cfg: dict[str, object],
    gpu: int,
    timeout_s: int,
    out_dir: Path,
) -> CellResult:
    label = str(autotuner_cfg["label"])
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONPATH"] = f"{REPO_ROOT}:" + env.get("PYTHONPATH", "")
    env["HELION_AUTOTUNE_EFFORT"] = "full"
    env["HELION_AUTOTUNER"] = str(autotuner_cfg["autotuner"])
    env["HELION_AUTOTUNE_LOG_LEVEL"] = "INFO"
    # HELION_SKIP_CACHE=1 disables read+write of the best_config cache and the
    # FROM_BEST_AVAILABLE disk-cache scan in stage 2 of LFBO. In-memory
    # LLM->LFBO seed handoff is unaffected. This is the only knob we need for
    # fair cross-cell comparison; we leave inductor/triton caches alone so we
    # don't touch shared state owned by other processes on this machine.
    env["HELION_SKIP_CACHE"] = "1"

    if autotuner_cfg.get("llm"):
        for key, value in autotuner_cfg.get("llm_env", {}).items():
            env[key] = str(value)
        if not env.get("HELION_LLM_API_KEY"):
            return CellResult(
                autotuner_label=label,
                kernel=kernel,
                wall_clock_s=0.0,
                exit_code=-1,
                helion_timings={},
                baseline_timings={},
                error="HELION_LLM_API_KEY not set; required for LLM-seeded cells",
            )

    log_path = out_dir / "logs" / f"{label}_{kernel}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [CONDA_PY, str(REPO_ROOT / "examples" / f"{kernel}.py")]
    t0 = time.perf_counter()
    try:
        result = subprocess.run(
            cmd,
            env=env,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        wall = time.perf_counter() - t0
        log_path.write_text(
            f"=== cmd: {' '.join(cmd)}\n"
            f"=== exit: {result.returncode}\n"
            f"=== wall_s: {wall:.2f}\n"
            f"=== stdout ===\n{result.stdout}\n"
            f"=== stderr ===\n{result.stderr}\n"
        )
        helion, baselines = parse_bench_blocks(result.stderr)
        return CellResult(
            autotuner_label=label,
            kernel=kernel,
            wall_clock_s=wall,
            exit_code=result.returncode,
            helion_timings=helion,
            baseline_timings=baselines,
            error=None
            if result.returncode == 0
            else f"nonzero exit {result.returncode}",
        )
    except subprocess.TimeoutExpired:
        wall = time.perf_counter() - t0
        log_path.write_text(f"TIMEOUT after {wall:.2f}s (limit={timeout_s}s)\n")
        return CellResult(
            autotuner_label=label,
            kernel=kernel,
            wall_clock_s=wall,
            exit_code=-1,
            helion_timings={},
            baseline_timings={},
            error=f"timeout after {timeout_s}s",
        )


def write_outputs(out_dir: Path, results: list[CellResult]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "results.jsonl").open("w") as f:
        for r in results:
            f.write(json.dumps(dataclasses.asdict(r)) + "\n")
    with (out_dir / "results.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "autotuner",
                "kernel",
                "wall_clock_s",
                "exit_code",
                "helion_impl",
                "helion_ms",
                "error",
            ]
        )
        for r in results:
            if r.helion_timings:
                for impl, ms in r.helion_timings.items():
                    writer.writerow(
                        [
                            r.autotuner_label,
                            r.kernel,
                            f"{r.wall_clock_s:.2f}",
                            r.exit_code,
                            impl,
                            f"{ms:.4f}",
                            r.error or "",
                        ]
                    )
            else:
                writer.writerow(
                    [
                        r.autotuner_label,
                        r.kernel,
                        f"{r.wall_clock_s:.2f}",
                        r.exit_code,
                        "",
                        "",
                        r.error or "no helion timings parsed",
                    ]
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--timeout", type=int, default=2400, help="Per-cell timeout in seconds"
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "benchmarks" / "llm_compare" / "runs",
        help="Parent directory for run output",
    )
    parser.add_argument(
        "--out-tag",
        default=dt.datetime.now().strftime("%Y%m%d_%H%M%S"),
        help="Subdirectory name under --out-dir",
    )
    parser.add_argument(
        "--kernels",
        default=",".join(KERNELS),
        help="Comma-separated kernel list (override for partial runs)",
    )
    parser.add_argument(
        "--autotuners",
        default=",".join(str(a["label"]) for a in AUTOTUNERS),
        help="Comma-separated autotuner labels (override for partial runs)",
    )
    args = parser.parse_args()

    kernels = [k for k in args.kernels.split(",") if k]
    chosen_autotuners = [
        a for a in AUTOTUNERS if str(a["label"]) in args.autotuners.split(",")
    ]

    out_dir = args.out_dir / args.out_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_dir}", file=sys.stderr)

    (out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "gpu": args.gpu,
                "timeout_s": args.timeout,
                "kernels": kernels,
                "autotuners": [a["label"] for a in chosen_autotuners],
                "started": dt.datetime.now().isoformat(timespec="seconds"),
            },
            indent=2,
        )
    )

    results: list[CellResult] = []
    total = len(kernels) * len(chosen_autotuners)
    cell_idx = 0
    overall_t0 = time.perf_counter()
    for autotuner_cfg in chosen_autotuners:
        label = str(autotuner_cfg["label"])
        for kernel in kernels:
            cell_idx += 1
            print(
                f"[{cell_idx}/{total}] {label} / {kernel} starting "
                f"(elapsed total {time.perf_counter() - overall_t0:.0f}s)",
                file=sys.stderr,
                flush=True,
            )
            r = run_cell(
                kernel=kernel,
                autotuner_cfg=autotuner_cfg,
                gpu=args.gpu,
                timeout_s=args.timeout,
                out_dir=out_dir,
            )
            results.append(r)
            write_outputs(out_dir, results)  # checkpoint after each cell
            print(
                f"[{cell_idx}/{total}] {label} / {kernel} done in "
                f"{r.wall_clock_s:.1f}s exit={r.exit_code} "
                f"helion_impls={list(r.helion_timings)} err={r.error}",
                file=sys.stderr,
                flush=True,
            )

    print(f"\nWrote: {out_dir / 'results.csv'}", file=sys.stderr)
    print(f"Wrote: {out_dir / 'results.jsonl'}", file=sys.stderr)
    print(
        f"Run `python benchmarks/llm_compare/summarize.py "
        f"{out_dir / 'results.jsonl'}` for a head-to-head table.",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
