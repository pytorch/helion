"""Launch a full attention-backward measurement round across GPUs.

Shards the variants across the given GPUs, running every impl serially per
variant in fresh subprocesses via compare_attnbwd_backends. Helion impls run
a cold full-effort autotune (HELION_SKIP_CACHE=1, force) bounded per-kernel
by HELION_AUTOTUNE_BUDGET_SECONDS.

Usage:
    python benchmarks/cute/attnbwd_campaign.py \
        --out artifacts/attnbwd-2026-09-02/before \
        --gpus 0,2,3,5,6,7 [--variants a,b] [--impls x,y] [--budget 3600]

Writes per-GPU JSONL files (<out>_gpu<i>.jsonl), autotuner configs under
<out>_configs/, and per-job logs under <out>_logs/.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import threading

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "benchmarks" / "cute"))

from attnbwd_variants import VARIANTS  # noqa: E402  # pyrefly: ignore [missing-import]

DEFAULT_IMPLS = (
    "fa4",
    "cudnn",
    "flex",
    "flex-ma",
    "helion-triton",
    "helion-cute",
)
HELION_IMPLS = {"helion-triton", "helion-cute"}
_FA_MAIN_ROOT = "/tmp/fa-main"


def run_gpu(
    gpu: int, variants: list[str], impls: list[str], args: argparse.Namespace
) -> None:
    out_base = Path(args.out)
    jsonl = f"{out_base}_gpu{gpu}.jsonl"
    config_dir = Path(f"{out_base}_configs")
    log_dir = Path(f"{out_base}_logs")
    config_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    for variant in variants:
        for impl in impls:
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env.pop("HELION_BACKEND", None)
            if impl == "helion-cute":
                env["HELION_BACKEND"] = "cute"
            elif impl == "helion-triton":
                env["HELION_BACKEND"] = "triton"
            elif impl == "fa4" and Path(_FA_MAIN_ROOT).is_dir():
                env.setdefault("HELION_FA4_ROOT", _FA_MAIN_ROOT)
            force = "0"
            if impl in HELION_IMPLS:
                env["HELION_AUTOTUNE_EFFORT"] = args.effort
                env["HELION_SKIP_CACHE"] = "1"
                if args.budget:
                    env["HELION_AUTOTUNE_BUDGET_SECONDS"] = str(args.budget)
                force = "1"
            cmd = [
                sys.executable,
                str(REPO_ROOT / "benchmarks" / "cute" / "compare_attnbwd_backends.py"),
                "--variant",
                variant,
                "--impl",
                impl,
                "--json",
                jsonl,
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
            if args.helion_mode:
                cmd += ["--helion-mode", args.helion_mode]
            log_path = log_dir / f"{variant}.{impl}.gpu{gpu}.log"
            print(f"[gpu{gpu}] {variant} / {impl} -> {log_path}", flush=True)
            with open(log_path, "w") as log:
                proc = subprocess.run(
                    cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=False
                )
            status = "ok" if proc.returncode == 0 else f"EXIT {proc.returncode}"
            print(f"[gpu{gpu}] {variant} / {impl}: {status}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--gpus", default="0,2,3,5,6,7")
    parser.add_argument("--variants", default=None)
    parser.add_argument("--impls", default=",".join(DEFAULT_IMPLS))
    parser.add_argument("--effort", default="full")
    parser.add_argument("--budget", type=int, default=3600)
    parser.add_argument("--num-runs", type=int, default=5)
    parser.add_argument("--warmup-ms", type=int, default=1000)
    parser.add_argument("--rep-ms", type=int, default=500)
    parser.add_argument("--cooldown-temp", type=float, default=55.0)
    parser.add_argument("--helion-mode", default=None, choices=["split", "fused"])
    args = parser.parse_args()

    gpus = [int(g) for g in args.gpus.split(",")]
    variants = args.variants.split(",") if args.variants else list(VARIANTS)
    impls = args.impls.split(",")
    shards: dict[int, list[str]] = {g: [] for g in gpus}
    for i, variant in enumerate(variants):
        shards[gpus[i % len(gpus)]].append(variant)

    threads = []
    for gpu, vs in shards.items():
        if not vs:
            continue
        t = threading.Thread(target=run_gpu, args=(gpu, vs, impls, args))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    print("campaign complete", flush=True)


if __name__ == "__main__":
    main()
