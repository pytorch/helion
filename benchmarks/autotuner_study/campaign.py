"""Orchestrate many autotuning runs across GPUs for the autotuner study.

Each run executes run_one.py in a fresh subprocess with a pinned GPU, a fixed
random seed, cache disabled, and per-candidate CSV logging enabled. Runs for
the same kernel case are always assigned the same GPU so timings are
apples-to-apples; different cases run in parallel on different GPUs.

Usage:
    python benchmarks/autotuner_study/campaign.py --plan audit --out /tmp/autotuner_study/audit
    python benchmarks/autotuner_study/campaign.py --plan-file plan.json --out DIR
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from kernels import KERNEL_CASES  # noqa: E402  # pyrefly: ignore [missing-import]

PRECOMPILE_JOBS = 48  # keep host CPU load fair across up to 8 concurrent runs


def default_gpus() -> list[int]:
    """All visible GPUs. On shared machines pass --gpus to avoid busy devices."""
    import torch

    return list(range(max(1, torch.cuda.device_count())))


@dataclasses.dataclass
class RunSpec:
    case: str
    algorithm: str | None  # None = default autotuner (LFBOTreeSearch)
    seed: int
    autotune_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)
    extra_env: dict[str, str] = dataclasses.field(default_factory=dict)
    tag: str = ""

    @property
    def run_id(self) -> str:
        alg = self.algorithm or "default"
        tag = f"-{self.tag}" if self.tag else ""
        return f"{self.case}--{alg}{tag}--s{self.seed}"


def audit_plan() -> list[RunSpec]:
    """~126 runs auditing current algorithms across 14 kernel cases."""
    runs: list[RunSpec] = []
    for case in KERNEL_CASES:
        for seed in (101, 102, 103, 104):
            runs.append(RunSpec(case, None, seed))
        for seed in (101, 102, 103):
            runs.append(RunSpec(case, "PatternSearch", seed))
        for seed in (101, 102):
            runs.append(RunSpec(case, "DifferentialEvolutionSearch", seed))
    return runs


def run_env(spec: RunSpec, gpu: int, run_dir: Path) -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "HELION_AUTOTUNE_RANDOM_SEED": str(spec.seed),
            "HELION_SKIP_CACHE": "1",
            "HELION_AUTOTUNE_LOG": str(run_dir / "autotune"),
            "HELION_AUTOTUNE_LOG_DETAILS": "1",
            "HELION_AUTOTUNE_PRECOMPILE_JOBS": str(PRECOMPILE_JOBS),
            "HELION_AUTOTUNE_PROGRESS_BAR": "0",
        }
    )
    if spec.algorithm is not None:
        env["HELION_AUTOTUNER"] = spec.algorithm
    else:
        env.pop("HELION_AUTOTUNER", None)
    env.update(spec.extra_env)
    return env


def execute_run(spec: RunSpec, gpu: int, out_root: Path) -> dict[str, Any]:
    run_dir = out_root / spec.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    # The autotune log sink appends; clear any telemetry left by a failed
    # earlier attempt so analyzers never merge two attempts into one run.
    for stale in run_dir.glob("autotune.*"):
        stale.unlink()
    (run_dir / "spec.json").write_text(json.dumps(dataclasses.asdict(spec), indent=2))
    log_path = run_dir / "run.log"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "benchmarks" / "autotuner_study" / "run_one.py"),
        "--case",
        spec.case,
        "--out",
        str(run_dir),
        "--autotune-kwargs",
        json.dumps(spec.autotune_kwargs),
    ]
    start = time.time()
    with log_path.open("w") as log_file:
        proc = subprocess.run(
            cmd,
            env=run_env(spec, gpu, run_dir),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=str(REPO_ROOT),
            check=False,
        )
    return {
        "run_id": spec.run_id,
        "gpu": gpu,
        "returncode": proc.returncode,
        "elapsed_s": time.time() - start,
    }


def worker(
    gpu: int,
    queue: list[RunSpec],
    lock: threading.Lock,
    out_root: Path,
    results: list[dict[str, Any]],
) -> None:
    while True:
        with lock:
            if not queue:
                return
            spec = queue.pop(0)
        done_marker = out_root / spec.run_id / "summary.json"
        if done_marker.exists():
            print(f"[gpu{gpu}] SKIP (done) {spec.run_id}", flush=True)
            continue
        print(f"[gpu{gpu}] START {spec.run_id}", flush=True)
        result = execute_run(spec, gpu, out_root)
        status = "OK" if result["returncode"] == 0 else f"FAIL({result['returncode']})"
        print(
            f"[gpu{gpu}] {status} {spec.run_id} in {result['elapsed_s']:.0f}s",
            flush=True,
        )
        with lock:
            results.append(result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", choices=["audit"], default=None)
    parser.add_argument("--plan-file", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--gpus", default=None, help="comma-separated GPU indices")
    parser.add_argument("--cases", default=None, help="comma-separated case filter")
    args = parser.parse_args()

    if args.plan == "audit":
        runs = audit_plan()
    elif args.plan_file:
        raw = json.loads(Path(args.plan_file).read_text())
        runs = [RunSpec(**spec) for spec in raw]
    else:
        raise SystemExit("need --plan or --plan-file")

    if args.cases:
        wanted = set(args.cases.split(","))
        runs = [r for r in runs if r.case in wanted]

    gpus = [int(g) for g in args.gpus.split(",")] if args.gpus else default_gpus()
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    # Pin every run of a case to one GPU (stable across invocations and
    # case filters: the mapping is derived from the full registry).
    all_cases = sorted(KERNEL_CASES)
    case_gpu = {case: gpus[i % len(gpus)] for i, case in enumerate(all_cases)}
    queues: dict[int, list[RunSpec]] = {gpu: [] for gpu in gpus}
    for run in runs:
        queues[case_gpu[run.case]].append(run)

    print(f"{len(runs)} runs over GPUs {gpus}")
    for gpu in gpus:
        print(f"  gpu{gpu}: {len(queues[gpu])} runs")

    lock = threading.Lock()
    results: list[dict[str, Any]] = []
    threads = [
        threading.Thread(
            target=worker, args=(gpu, queues[gpu], lock, out_root, results)
        )
        for gpu in gpus
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    (out_root / "campaign_results.json").write_text(json.dumps(results, indent=2))
    failed = [r for r in results if r["returncode"] != 0]
    print(f"complete: {len(results)} executed, {len(failed)} failed")
    for r in failed:
        print(f"  FAILED {r['run_id']}")


if __name__ == "__main__":
    main()
