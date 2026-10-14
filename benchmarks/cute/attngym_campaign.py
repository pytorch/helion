"""Launch an attention-gym measurement campaign across selected GPUs.

Shards cases across GPUs, running every implementation serially per case in
fresh subprocesses. Helion runs cold full-effort autotuning by default,
bounded by --budget. The figure suite has 29 variants and 87 cases.

Writes per-GPU JSONL (<out>_gpu<i>.jsonl), per-case configs under <out>_configs/,
logs under <out>_logs/, and a manifest with shapes and GPU assignments.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.cute import attngym_variants as agv  # noqa: E402
from benchmarks.cute import compare_attngym_backends as harness  # noqa: E402

DEFAULT_IMPLS = (
    "flex-triton",
    "flex-triton-ma",
    "flex-cute",
    "sdpa",
    "fa4",
    "helion-triton",
    "helion-cute",
)
FIGURE_IMPLS = (
    "sdpa",
    "fa4",
    "flex-triton",
    "flex-triton-ma",
    "flex-cute",
    "helion-cute",
)
HELION_IMPLS = set(harness.HELION_IMPLS)


def _shard_cases(
    cases: list[agv.Variant], gpus: list[int]
) -> dict[int, list[agv.Variant]]:
    if not gpus or len(gpus) != len(set(gpus)):
        raise ValueError("--gpus must name at least one GPU without duplicates")
    shards: dict[int, list[agv.Variant]] = {gpu: [] for gpu in gpus}
    for index, variant in enumerate(cases):
        shards[gpus[index % len(gpus)]].append(variant)
    return shards


def run_gpu(
    gpu: int, cases: list[agv.Variant], impls: list[str], args: argparse.Namespace
) -> int:
    out_base = Path(args.out)
    jsonl = Path(f"{out_base}_gpu{gpu}.jsonl")
    config_dir = Path(f"{out_base}_configs")
    log_dir = Path(f"{out_base}_logs")
    config_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    failures = 0
    for variant in cases:
        for impl in impls:
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env.pop("HELION_BACKEND", None)
            env.update(harness._IMPL_ENV.get(impl, {}))
            force = "0"
            if impl in HELION_IMPLS:
                env["HELION_AUTOTUNE_EFFORT"] = args.effort
                env["HELION_SKIP_CACHE"] = "1"
                if args.budget:
                    env["HELION_AUTOTUNE_BUDGET_SECONDS"] = str(args.budget)
                force = "1"
            cmd = [
                sys.executable,
                str(REPO_ROOT / "benchmarks" / "cute" / "compare_attngym_backends.py"),
                "--suite",
                args.suite,
                "--variant",
                variant.name,
                "--impl",
                impl,
                "--json",
                str(jsonl),
                "--num-runs",
                str(args.num_runs),
                "--warmup-ms",
                str(args.warmup_ms),
                "--rep-ms",
                str(args.rep_ms),
                "--cooldown-temp",
                str(args.cooldown_temp),
                "--force-autotune",
                force,
                "--save-config",
                str(config_dir),
            ]
            if variant.shape_id is not None:
                cmd += ["--shape-id", variant.shape_id]
            log_path = log_dir / (f"{harness._case_id(variant)}.{impl}.gpu{gpu}.log")
            offset = jsonl.stat().st_size if jsonl.exists() else 0
            print(
                f"[gpu{gpu}] {harness._case_id(variant)} / {impl} -> {log_path}",
                flush=True,
            )
            with log_path.open("w") as log:
                proc = subprocess.run(
                    cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=False
                )
            result = None
            if jsonl.exists():
                with jsonl.open() as results:
                    results.seek(offset)
                    for line in results:
                        if line.strip():
                            result = json.loads(line)
            if result is None:
                result = {
                    **harness._case_metadata(variant, args),
                    "impl": impl,
                    "accuracy": "ERROR",
                    "skip_reason": f"subprocess exit {proc.returncode} without a result",
                    "physical_gpu": str(gpu),
                }
                with jsonl.open("a") as results:
                    results.write(json.dumps(result) + "\n")
            status = result["accuracy"]
            failures += int(proc.returncode != 0 or status in {"ERROR", "FAIL"})
            if proc.returncode != 0:
                status = f"EXIT {proc.returncode}"
            print(
                f"[gpu{gpu}] {harness._case_id(variant)} / {impl}: {status}",
                flush=True,
            )
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--suite", choices=("legacy", "figure"), default="legacy")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--variants", help="comma-separated variant row labels")
    parser.add_argument("--shape-id", help="one complete case ID")
    parser.add_argument("--impls", help="comma-separated implementations")
    parser.add_argument("--effort", default="full")
    parser.add_argument("--budget", type=int, default=7200)
    parser.add_argument("--num-runs", type=int, default=5)
    parser.add_argument("--warmup-ms", type=int, default=1000)
    parser.add_argument("--rep-ms", type=int, default=500)
    parser.add_argument("--cooldown-temp", type=float, default=55.0)
    parser.add_argument("--manifest-output", help="default: <out>_manifest.json")
    parser.add_argument(
        "--dry-run", action="store_true", help="write manifest without GPU work"
    )
    parser.set_defaults(seed=0, skip_correctness=0)
    args = parser.parse_args()
    names = (
        args.variants.split(",") if args.variants and args.variants != "all" else None
    )
    try:
        gpus = [int(gpu) for gpu in args.gpus.split(",")]
        cases = harness._select_cases(args.suite, names, args.shape_id)
        shards = _shard_cases(cases, gpus)
    except ValueError as exc:
        parser.error(str(exc))
    impls = (
        args.impls.split(",")
        if args.impls
        else list(FIGURE_IMPLS if args.suite == "figure" else DEFAULT_IMPLS)
    )
    unknown = set(impls) - set(harness.ALL_IMPLS)
    if unknown:
        parser.error(f"unknown implementations: {', '.join(sorted(unknown))}")
    manifest: dict[str, Any]
    if args.suite == "figure":
        from benchmarks.cute import attngym_figure as figure

        manifest = figure.manifest()
    else:
        manifest = {
            "suite": "legacy",
            "cases": [
                harness._case_metadata(variant, args)
                for variant in harness._suite_cases("legacy").values()
            ],
        }
    manifest["campaign"] = {
        "selected_cases": [harness._case_id(variant) for variant in cases],
        "case_gpus": {
            harness._case_id(variant): gpu
            for gpu, gpu_cases in shards.items()
            for variant in gpu_cases
        },
        "gpus": gpus,
        "impls": impls,
        "input_seed": 0,
        "autotune_effort": args.effort,
        "autotune_budget_seconds": args.budget,
        "num_runs": args.num_runs,
        "warmup_ms": args.warmup_ms,
        "rep_ms": args.rep_ms,
        "cooldown_temp": args.cooldown_temp,
        "skip_correctness": False,
    }
    manifest_path = Path(args.manifest_output or f"{args.out}_manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    if args.dry_run:
        print(f"{len(cases)} cases, {len(impls)} implementations: {manifest_path}")
        return
    failures_by_gpu: dict[int, int] = {}

    def worker(gpu: int, gpu_cases: list[agv.Variant]) -> None:
        failures_by_gpu[gpu] = run_gpu(gpu, gpu_cases, impls, args)

    threads = []
    for gpu, gpu_cases in shards.items():
        if not gpu_cases:
            continue
        thread = threading.Thread(target=worker, args=(gpu, gpu_cases))
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()
    if len(failures_by_gpu) != len(threads):
        raise RuntimeError("A GPU worker failed before completing its cases")
    failures = sum(failures_by_gpu.values())
    print(f"campaign complete: {failures} failed or errored jobs", flush=True)


if __name__ == "__main__":
    main()
