#!/usr/bin/env python3
"""Run every pretuned kernel's benchmark sweep and emit aggregate metrics as JSON.

Drives the nightly pretuned-kernel dashboard pipeline. Each kernel module under
``pretuned_kernels/<name>/<name>.py`` defines a ``main()`` that benchmarks the
Helion kernel against PyTorch eager across its checked-in shape sweep and prints
a machine-parseable line::

    SUMMARY: helion_wins=.. total=.. geomean=.. best_speedup=..

This script runs each ``main()``, parses that line, and writes a JSON list of
per-kernel records that ``.github/dashboard/build_pretuned_dashboard_data.py``
aggregates for the dashboard's Pretuned tab. A kernel that crashes (e.g. no
heuristic for the current GPU) is recorded with an ``error`` field so the rest
of the sweep still reports.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import re
import sys
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from types import ModuleType

PRETUNED_KERNELS_DIR = Path(__file__).resolve().parent

# Kernel directory == module file stem == kernel name. Ordered cheap-to-expensive
# so a partial run (e.g. timeout) still captures the quick kernels.
KERNELS = [
    "vector_add",
    "rope",
    "rms_norm",
    "layer_norm",
    "softmax",
    "scaled_mm",
    "cross_entropy",
]

_SUMMARY_RE = re.compile(
    r"^SUMMARY:\s+helion_wins=(?P<wins>\d+)\s+total=(?P<total>\d+)\s+"
    r"geomean=(?P<geomean>[\d.]+)\s+best_speedup=(?P<best>[\d.]+)\s*$",
    re.MULTILINE,
)

# Map the running GPU's compute capability to the hardware alias each kernel
# declares via pretuned_hardware(). The nightly runs one GPU per alias.
_HARDWARE_BY_COMPUTE = {"sm90": "h100", "sm100": "b200"}


def _import_kernel_module(name: str) -> ModuleType:
    # Private module name avoids clashing with examples/<name>.py.
    module_name = f"_helion_pretuned_run.{name}"
    file_path = PRETUNED_KERNELS_DIR / name / f"{name}.py"
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _compute_capability() -> str:
    major, minor = torch.cuda.get_device_capability()
    return f"sm{major}{minor}"


def run_kernel(name: str, current_hardware: str) -> dict:
    record: dict[str, object] = {"kernel": name}
    try:
        module = _import_kernel_module(name)
        # Only benchmark a kernel on the hardware its checked-in config targets.
        hardware = module.pretuned_hardware()
        record["pretuned_hardware"] = hardware
        if hardware != current_hardware:
            record["skipped"] = (
                f"pretuned for {hardware}; current hardware is {current_hardware}"
            )
            print(f"Skipping {name}: {record['skipped']}")
            return record
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            module.main()
        output = buf.getvalue()
        # Relay so CI logs show the per-shape table.
        print(output)
        match = _SUMMARY_RE.search(output)
        if match is None:
            record["error"] = "no SUMMARY line in output"
            return record
        record.update(
            helion_wins=int(match["wins"]),
            total=int(match["total"]),
            geomean=float(match["geomean"]),
            best_speedup=float(match["best"]),
            # Each kernel module declares its benchmark method; far more robust
            # than scraping a token out of the printed SUMMARY line.
            cudagraph=bool(module.use_cudagraph()),
        )
    except (
        Exception
    ) as exc:  # Keep going: one kernel's failure shouldn't sink the report.
        record["error"] = f"{type(exc).__name__}: {exc}"
        print(f"ERROR running {name}: {record['error']}", file=sys.stderr)
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="pretuned-bench.json")
    parser.add_argument(
        "--kernels",
        default=",".join(KERNELS),
        help="Comma-separated kernel names to run (default: all).",
    )
    parser.add_argument(
        "--hardware",
        default="",
        help="Hardware alias to match against each kernel's pretuned_hardware() "
        "(default: derived from the GPU's compute capability).",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA device required for pretuned kernel benchmarks.")

    device = torch.cuda.get_device_name()
    compute = _compute_capability()
    current_hardware = args.hardware or _HARDWARE_BY_COMPUTE.get(compute, compute)
    print(f"GPU: {device} ({compute}); hardware={current_hardware}")

    kernels = [k.strip() for k in args.kernels.split(",") if k.strip()]
    records = []
    for name in kernels:
        print(f"\n{'=' * 60}\nBenchmarking pretuned kernel: {name}\n{'=' * 60}")
        record = run_kernel(name, current_hardware)
        record["device"] = device
        record["compute_capability"] = compute
        records.append(record)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(records, f, indent=2)
    ok = sum(1 for r in records if "geomean" in r)
    print(f"\nWrote {len(records)} kernel records ({ok} with data) to {args.output}")


if __name__ == "__main__":
    main()
