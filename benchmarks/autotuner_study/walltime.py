"""Sequential wall-time comparison: baseline vs v2 bundle, one run at a time.

Historical study artifact: the --v2-tree it drives must be a checkout of the
autotuner-v2-fixes/-simple study branches, which carried the
HELION_AUTOTUNER_V2 / HELION_LISTOF_UNIFORM_NEIGHBORS flags (removed from
the final tree). Runs alternate baseline and v2 so thermal/clock drift
affects both arms equally; exactly one autotuning run executes at a time on
the given GPU.

Usage:
    python benchmarks/autotuner_study/walltime.py --gpu 1 \
        --baseline-tree /data/users/jansel/ws7/helion \
        --v2-tree /data/users/jansel/ws7/helion-proto2 \
        --out /tmp/autotuner_study/walltime
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

CASES = [
    "attention-2k64",
    "matmul-4096",
    "gathergemv-8192",
    "layernorm-10240",
    "crossentropy-131k",
]
SEEDS = [401, 402]


def run_one(
    tree: Path, case: str, seed: int, gpu: int, run_dir: Path, v2: bool
) -> dict[str, object]:
    run_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "HELION_AUTOTUNE_RANDOM_SEED": str(seed),
            "HELION_SKIP_CACHE": "1",
            "HELION_AUTOTUNE_LOG": str(run_dir / "autotune"),
            "HELION_AUTOTUNE_LOG_DETAILS": "1",
            "HELION_AUTOTUNE_PROGRESS_BAR": "0",
        }
    )
    env.pop("HELION_AUTOTUNER", None)
    env.pop("HELION_AUTOTUNER_V2", None)
    env.pop("HELION_LISTOF_UNIFORM_NEIGHBORS", None)
    env.pop("HELION_AUTOTUNE_PRECOMPILE_JOBS", None)  # full host for wall time
    if v2:
        env["HELION_AUTOTUNER_V2"] = "1"
    start = time.time()
    with (run_dir / "run.log").open("w") as log:
        proc = subprocess.run(
            [
                sys.executable,
                str(tree / "benchmarks" / "autotuner_study" / "run_one.py"),
                "--case",
                case,
                "--out",
                str(run_dir),
            ],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            cwd=str(tree),
            check=False,
        )
    return {
        "case": case,
        "seed": seed,
        "variant": "v2" if v2 else "baseline",
        "returncode": proc.returncode,
        "elapsed_s": time.time() - start,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--baseline-tree", required=True)
    parser.add_argument("--v2-tree", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    results = []
    for case in CASES:
        for seed in SEEDS:
            for v2 in (False, True):  # alternate to balance drift
                name = f"{case}--{'v2' if v2 else 'baseline'}--s{seed}"
                run_dir = out_root / name
                if (run_dir / "summary.json").exists():
                    print(f"SKIP {name}", flush=True)
                    continue
                tree = Path(args.v2_tree if v2 else args.baseline_tree)
                print(f"START {name}", flush=True)
                result = run_one(tree, case, seed, args.gpu, run_dir, v2)
                print(
                    f"DONE {name} rc={result['returncode']} {result['elapsed_s']:.0f}s",
                    flush=True,
                )
                results.append(result)
    (out_root / "walltime_results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
