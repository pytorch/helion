"""Re-benchmark one recorded hillclimb step, interleaved against FA4.

    python benchmarks/cute/hillclimb/attn_dense32768/rebench.py \
        benchmarks/cute/hillclimb/attn_dense32768/step00_baseline.json --gpu 3

Reads the pinned ``config`` out of the step file and drives
``compare_attention_backends.py`` with it, alternating Helion and FA4 on the
same GPU so board-level power contention hits both sides equally.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
BENCH = REPO / "benchmarks" / "cute" / "compare_attention_backends.py"
SHAPE = ["--z", "2", "--h", "32", "--seq-len", "32768", "--head-dim", "64",
         "--dtype", "float16", "--causal", "0", "--biased", "0"]


def run(impl: str, gpu: str, out: Path, extra: list[str]) -> dict:
    env = dict(os.environ)
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env.setdefault("HELION_FA4_ROOT", "/home/shangdiy/flash-attention")
    cmd = [
        sys.executable, str(BENCH), "--impl", impl, *SHAPE,
        "--num-runs", "9", "--warmup-ms", "1000", "--rep-ms", "500",
        "--seed", "2026090301", "--helion-cute-benchmark-timer", "event",
        "--json", "--json-output", str(out), *extra,
    ]
    subprocess.run(cmd, env=env, check=True, cwd=REPO,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return json.loads(out.read_text())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step_file")
    ap.add_argument("--gpu", default="3")
    ap.add_argument("--rounds", type=int, default=3)
    args = ap.parse_args()

    step = json.loads(Path(args.step_file).read_text())
    extra = ["--helion-force-autotune", "0"]
    for key, value in step["config"].items():
        extra += ["--helion-config", f"{key}={json.dumps(value)}"]

    cute, fa4 = [], []
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(args.rounds):
            fa4.append(run("fa4", args.gpu, Path(tmp) / f"fa4_{i}.json", []))
            cute.append(
                run("helion-cute", args.gpu, Path(tmp) / f"cute_{i}.json", extra)
            )

    mc = statistics.median(r["median_tflops"] for r in cute)
    mf = statistics.median(r["median_tflops"] for r in fa4)
    print(json.dumps({
        "step": step["step"],
        "name": step["name"],
        "recorded": step["measurement"],
        "now": {
            "helion_cute_tflops": round(mc, 1),
            "fa4_tflops": round(mf, 1),
            "ratio": round(mc / mf, 4),
            "accuracy": "PASS" if all(r["accuracy"] == "PASS" for r in cute) else "FAIL",
        },
    }, indent=2))


if __name__ == "__main__":
    main()
