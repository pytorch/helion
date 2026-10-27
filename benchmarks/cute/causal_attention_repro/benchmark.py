"""Benchmark the bundled causal Helion CuTe kernels in fresh worker processes."""

from __future__ import annotations

import argparse
from datetime import datetime
from datetime import timezone
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("forward", "backward"), required=True)
    parser.add_argument("--gpu", type=int, required=True, help="physical GPU index")
    parser.add_argument(
        "--sequence", choices=("all", "2048", "4096", "8192"), default="all"
    )
    parser.add_argument("--out", type=Path, required=True, help="new output directory")
    parser.add_argument(
        "--python", type=Path, help="use an existing pinned environment"
    )
    args = parser.parse_args()
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=False)
    python = args.python or ROOT / ".envs" / args.mode / "bin/python"
    engine = ROOT / ".engines" / args.mode
    library = subprocess.check_output(
        [
            str(python),
            "-c",
            "from importlib.metadata import distribution; "
            "print(distribution('nvidia-cutlass-dsl-libs-cu13').locate_file("
            "'nvidia_cutlass_dsl/cu13/lib/libcute_dsl_runtime.so'))",
        ],
        text=True,
    ).strip()
    env = dict(os.environ)
    env.update(
        CUDA_VISIBLE_DEVICES=str(args.gpu),
        HELION_BACKEND="cute",
        HELION_AUTOTUNE_EFFORT="none",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONUNBUFFERED="1",
        PYTHONPATH=str(engine),
        CUTE_DSL_LIBS=library,
        LD_LIBRARY_PATH=str(Path(library).parent),
    )
    env.pop("HELION_FA4_ROOT", None)
    env.pop("FA4_PYTHONPATH", None)
    processes = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            str(args.gpu),
            "--query-compute-apps=pid",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()
    if processes:
        raise RuntimeError(f"GPU {args.gpu} has active compute processes: {processes}")
    pins = json.loads((ROOT / "pins.json").read_text())
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "mode": args.mode,
                "gpu": args.gpu,
                "python": str(python),
                "helion_source": pins["sources"][args.mode],
                "software_versions": pins["software_versions"][args.mode],
                "shape": pins["shape"],
                "timing": {
                    "runs": 5,
                    "warmup_ms": 500,
                    "measurement_ms": 500,
                    "thermal_warmup_ms": 5000,
                    "idle_between_cases_s": 20,
                },
            },
            indent=2,
        )
        + "\n"
    )
    sequences = [2048, 4096, 8192] if args.sequence == "all" else [int(args.sequence)]
    for index, sequence in enumerate(sequences):
        if index:
            time.sleep(20)
        result = output / f"s{sequence}.json"
        with (output / f"s{sequence}.log").open("w") as log:
            subprocess.run(
                [
                    str(python),
                    str(ROOT / "worker.py"),
                    "--mode",
                    args.mode,
                    "--sequence",
                    str(sequence),
                    "--result",
                    str(result),
                ],
                env=env,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        record = json.loads(result.read_text())
        with (output / "results.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        print(
            f"{args.mode} S={sequence}: {record['median_ms']:.6f} ms, "
            f"{record['median_tflops']:.1f} TFLOP/s, {record['accuracy']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
