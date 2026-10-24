"""Interleaved A/B/A/B timing of two implementations of one catalogue case.

Each reading is a fresh ``example_case_bench.py`` process on the same GPU, so
thermal drift hits both sides equally and configs never share a process.

  python benchmarks/cute/example_abab_verify.py --gpu 4 --module rope --shape 2 \
      --case 00_forward --rounds 4 --output abab_rope_s2.jsonl \
      --a cute:helion --a-pin-from after/raw/00002.json \
      --b triton:helion --b-pin-from after/raw/00004.json

``--a`` / ``--b`` are ``provider:impl``; ``--a-pin-from`` / ``--b-pin-from``
pin Helion kernels from saved records (baselines need no pin). The JSONL
output has one line per reading and a final ``summary`` line with the median
of each side and ``ratio = median_b / median_a`` (>1 means A is faster), on
the catalogue's primary latency (``graph_long_ms``: the sustained per-call
time of a CUDA graph of at least ``--graph-calls`` calls, grown until one
replay takes 400 us); the other timings are recorded too.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCH = REPO_ROOT / "benchmarks" / "cute" / "example_case_bench.py"


def run_side(
    args: argparse.Namespace, side: str, spec: str, pin_from: Path | None
) -> dict:
    provider, _, impl = spec.partition(":")
    with tempfile.TemporaryDirectory(prefix="abab-") as directory:
        out = Path(directory) / "result.json"
        command = [
            sys.executable,
            str(BENCH),
            "--module",
            args.module,
            "--shape",
            str(args.shape),
            "--case",
            args.case,
            "--provider",
            provider,
            "--impl",
            impl or "helion",
            "--reps",
            str(args.reps),
            "--graph-calls",
            str(args.graph_calls),
            "--json",
            str(out),
        ]
        if pin_from is not None:
            command += ["--pin-from", str(pin_from)]
        env = {k: v for k, v in os.environ.items() if not k.startswith("HELION_")}
        env.update(
            CUDA_VISIBLE_DEVICES=args.gpu,
            HELION_CACHE_DIR=str(Path(directory) / "helion"),
            PYTHONPATH=os.pathsep.join([str(REPO_ROOT), env.get("PYTHONPATH", "")]),
        )
        started = time.monotonic()
        proc = subprocess.run(
            command, env=env, cwd=REPO_ROOT, capture_output=True, text=True
        )
        record = {
            "side": side,
            "spec": spec,
            "pin_from": str(pin_from) if pin_from else None,
            "returncode": proc.returncode,
            "wall_seconds": time.monotonic() - started,
        }
        if out.exists():
            data = json.loads(out.read_text())
            record.update(
                bench_ms=data.get("bench_ms"),
                bench_error=data.get("bench_error"),
                graph_ms=data.get("graph_ms"),
                graph_calls_ms=data.get("graph_calls_ms"),
                graph_long_ms=data.get("graph_long_ms"),
                graph_long_calls=data.get("graph_long_calls"),
                event_ms=data.get("event_ms"),
                graph_error=data.get("graph_error"),
                bound_kernels=data.get("bound_kernels"),
            )
        else:
            record["stderr_tail"] = proc.stderr[-2000:]
        return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--module", required=True)
    parser.add_argument("--shape", required=True, type=int)
    parser.add_argument("--case", default="00_forward")
    parser.add_argument("--a", required=True, help="provider:impl, e.g. cute:helion")
    parser.add_argument("--b", required=True, help="provider:impl")
    parser.add_argument("--a-pin-from", type=Path, default=None)
    parser.add_argument("--b-pin-from", type=Path, default=None)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument(
        "--graph-calls",
        type=int,
        default=16,
        help="Minimum calls per CUDA graph for the sustained per-call timing "
        "(graph_calls_ms; graph_long_ms grows it until a replay takes 400 us)",
    )
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    readings: dict[str, list[float]] = {"a": [], "b": []}
    with args.output.open("a") as stream:
        stream.write(
            json.dumps(
                {
                    "header": True,
                    "module": args.module,
                    "shape": args.shape,
                    "case": args.case,
                    "gpu": args.gpu,
                    "a": args.a,
                    "b": args.b,
                    "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }
            )
            + "\n"
        )
        for _ in range(args.rounds):
            for side, spec, pin in (
                ("a", args.a, args.a_pin_from),
                ("b", args.b, args.b_pin_from),
            ):
                record = run_side(args, side, spec, pin)
                stream.write(json.dumps(record) + "\n")
                stream.flush()
                reading = record.get("graph_long_ms")
                if reading is None:
                    reading = record.get("graph_calls_ms")
                if reading is not None:
                    readings[side].append(reading)
                print(
                    f"{side} {spec}: bench_ms={record.get('bench_ms')} "
                    f"graph_calls_ms={record.get('graph_calls_ms')} "
                    f"graph_long_ms={record.get('graph_long_ms')} "
                    f"({record.get('graph_long_calls')} calls) "
                    f"rc={record['returncode']}",
                    flush=True,
                )
        summary: dict[str, object] = {"summary": True}
        if readings["a"] and readings["b"]:
            med_a = statistics.median(readings["a"])
            med_b = statistics.median(readings["b"])
            summary.update(
                field="graph_long_ms (graph_calls_ms where a record lacks it)",
                median_a_ms=med_a,
                median_b_ms=med_b,
                ratio=med_b / med_a,
                readings_a=readings["a"],
                readings_b=readings["b"],
                balanced=len(readings["a"]) == len(readings["b"]),
            )
            if len(readings["a"]) != len(readings["b"]):
                print(
                    f"WARNING: unbalanced readings ({len(readings['a'])} vs "
                    f"{len(readings['b'])}): a side failed in some rounds",
                    file=sys.stderr,
                )
        stream.write(json.dumps(summary) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
