# ruff: noqa: ANN401
"""Profile one implementation of a better-benchmark repro under Nsight Compute.

Builds the implementation the way ``bench.py`` does (Inductor with or
without coordinate descent, the oracle, or a Helion port pinned to its
autotuned decorators), warms it up, captures it in a CUDA graph as
``bench.py``'s timers do, and profiles one replay of the graph with ``ncu``
(an implementation may take another path under capture: the dropout oracles
draw their random numbers in the kernel), printing per-kernel duration, DRAM
and L2 traffic, achieved DRAM bandwidth, SM throughput, occupancy and launch
geometry.

Example::

    python benchmarks/cute/better_benchmark/ncu_profile.py var_mean_42fad1ece813 \\
        --impl helion_cute --configs-dir artifacts/bb-run/configs
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import operator
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from benchmarks.cute.better_benchmark import common

METRICS = (
    "gpu__time_duration.sum",
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "lts__t_bytes.sum",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "launch__registers_per_thread",
    "launch__block_size",
    "launch__grid_size",
    "launch__shared_mem_per_block_static",
    "launch__shared_mem_per_block_dynamic",
    "smsp__inst_executed.sum",
)


def build(args: argparse.Namespace) -> Any:
    """The implementation as a callable taking the input list."""
    shape = args.shape or common.default_shape(args.repro_id)
    if args.impl in ("inductor", "inductor_cd"):
        instance = common.load_repro_module(args.repro_id).Repro()
        compiled = common.compile_inductor(
            args.repro_id, shape, instance, cd=args.impl == "inductor_cd"
        )
        return lambda inp: compiled(*inp)
    if args.impl == "oracle":
        return common.load_oracle(
            args.repro_id, common.make_inputs(args.repro_id, shape)
        )
    backend = args.impl.removeprefix("helion_")
    if args.configs_dir is not None:
        port, _ = common.load_autotuned_port(
            args.configs_dir, args.repro_id, shape, backend
        )
    else:
        pinned = (
            json.loads(args.decorators.read_text())
            if args.decorators is not None
            else None
        )
        port = common.load_port(args.repro_id, backend, pinned=pinned)
    return port.forward


def inner(args: argparse.Namespace) -> None:
    # The outer process, which only runs ncu, does without torch.
    import torch

    torch.cuda.set_device(0)
    shape = args.shape or common.default_shape(args.repro_id)
    call = build(args)
    inputs = common.make_inputs(args.repro_id, shape)
    with torch.no_grad():
        for _ in range(3):
            call(inputs)
        torch.cuda.synchronize()
        graph, _ = common.capture_graph(lambda: call(inputs))
        graph.replay()
        torch.cuda.synchronize()
        cudart = torch.cuda.cudart()
        assert cudart is not None
        cudart.cudaProfilerStart()
        graph.replay()
        torch.cuda.synchronize()
        cudart.cudaProfilerStop()


def outer(args: argparse.Namespace) -> list[dict[str, Any]]:
    ncu = shutil.which("ncu") or "/usr/local/cuda/bin/ncu"
    cmd = [
        ncu,
        "--profile-from-start",
        "off",
        "--csv",
        "--metrics",
        ",".join(METRICS),
        sys.executable,
        str(Path(__file__).resolve()),
        args.repro_id,
        "--impl",
        args.impl,
        "--inner",
    ]
    if args.shape:
        cmd += ["--shape", args.shape]
    if args.configs_dir is not None:
        cmd += ["--configs-dir", str(args.configs_dir)]
    if args.decorators is not None:
        cmd += ["--decorators", str(args.decorators)]
    env = dict(
        os.environ,
        HELION_AUTOTUNE_EFFORT=os.environ.get("HELION_AUTOTUNE_EFFORT", "none"),
    )
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"ncu failed ({proc.returncode}):\n{proc.stderr[-4000:]}")
    lines = [line for line in proc.stdout.splitlines() if line.startswith('"')]
    rows = list(csv.DictReader(io.StringIO("\n".join(lines))))
    kernels: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (row["ID"], row["Kernel Name"])
        entry = kernels.setdefault(
            key, {"id": int(row["ID"]), "kernel": row["Kernel Name"]}
        )
        value = row["Metric Value"].replace(",", "")
        try:
            entry[row["Metric Name"]] = float(value)
        except ValueError:
            entry[row["Metric Name"]] = value
        entry[f"{row['Metric Name']}__unit"] = row["Metric Unit"]
    return sorted(kernels.values(), key=operator.itemgetter("id"))


# The units ncu reports sizes and durations in: each one's kind and factor.
_UNITS = {
    "byte": ("bytes", 1.0),
    "Kbyte": ("bytes", 1e3),
    "Mbyte": ("bytes", 1e6),
    "Gbyte": ("bytes", 1e9),
    "Tbyte": ("bytes", 1e12),
    "nsecond": ("seconds", 1e-9),
    "ns": ("seconds", 1e-9),
    "usecond": ("seconds", 1e-6),
    "us": ("seconds", 1e-6),
    "msecond": ("seconds", 1e-3),
    "ms": ("seconds", 1e-3),
    "second": ("seconds", 1.0),
}


def _scaled(entry: dict[str, Any], metric: str, unit: str) -> float:
    """``metric`` converted to ``unit`` (ncu picks the unit per value); a
    unit missing from ``_UNITS``, or of another kind, raises ValueError
    rather than printing a wrong number."""
    value = float(entry.get(metric, 0.0))
    have = entry.get(f"{metric}__unit", unit)
    if have not in _UNITS or unit not in _UNITS:
        raise ValueError(f"{metric}: cannot convert ncu unit {have!r} to {unit!r}")
    (have_kind, have_factor), (kind, factor) = _UNITS[have], _UNITS[unit]
    if have_kind != kind:
        raise ValueError(f"{metric}: {have!r} is not a unit of {kind}")
    return value * have_factor / factor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("repro_id")
    parser.add_argument(
        "--shape",
        default=None,
        help="shape hash (default: the repro's selection.json point, else its first)",
    )
    parser.add_argument(
        "--impl",
        default="helion_cute",
        choices=("inductor", "inductor_cd", "oracle", "helion_cute", "helion_triton"),
    )
    parser.add_argument("--configs-dir", type=Path, default=None)
    parser.add_argument(
        "--decorators",
        type=Path,
        default=None,
        help="JSON {kernel: decorator} to pin instead of an autotune record",
    )
    parser.add_argument("--inner", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()
    if args.inner:
        inner(args)
        return
    kernels = outer(args)
    total_us = 0.0
    for k in kernels:
        us = _scaled(k, "gpu__time_duration.sum", "us")
        read = _scaled(k, "dram__bytes_read.sum", "Mbyte")
        write = _scaled(k, "dram__bytes_write.sum", "Mbyte")
        l2 = _scaled(k, "lts__t_bytes.sum", "Mbyte")
        total_us += us
        gbps = (read + write) * 1e6 / (us * 1e-6) / 1e9 if us else 0.0
        print(
            f"{us:9.2f}us dram r/w {read:8.1f}/{write:8.1f}MB ({gbps:6.0f} GB/s, "
            f"{k.get('dram__throughput.avg.pct_of_peak_sustained_elapsed', 0):5.1f}% peak) "
            f"L2 {l2:8.1f}MB sm {k.get('sm__throughput.avg.pct_of_peak_sustained_elapsed', 0):5.1f}% "
            f"occ {k.get('sm__warps_active.avg.pct_of_peak_sustained_active', 0):5.1f}% "
            f"regs {k.get('launch__registers_per_thread', 0):.0f} "
            f"grid {k.get('launch__grid_size', 0):.0f}x{k.get('launch__block_size', 0):.0f} "
            f"smem {k.get('launch__shared_mem_per_block_static', 0)}+"
            f"{k.get('launch__shared_mem_per_block_dynamic', 0)} "
            f"{k['kernel'][:70]}"
        )
    print(f"total {total_us:.2f}us over {len(kernels)} kernel(s)")
    if args.json:
        args.json.write_text(json.dumps(kernels, indent=1) + "\n")


if __name__ == "__main__":
    main()
