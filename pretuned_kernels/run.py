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
import importlib.util
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
    "scale_mm_cute",
    "nvfp4_gemv",
    "nvfp4_gemv_cute",
    "cross_entropy",
    # Ported from vLLM (vllm/kernels/helion/ops); torch-native baselines.
    "silu_mul_fp8",
    "dynamic_per_token_scaled_fp8_quant",
    "per_token_group_fp8_quant",
    "rms_norm_dynamic_per_token_quant",
    "rms_norm_per_block_quant",
    "silu_and_mul_per_block_quant",
    "fused_qk_norm_rope",
]

# Map a compute capability (heuristic file suffix) to a hardware alias. The
# nightly runs one GPU per alias.
_HARDWARE_BY_COMPUTE = {"sm90": "h100", "sm100": "b200"}


def _supported_hardware(name: str) -> set[str]:
    """Hardware a kernel is pretuned for, inferred from its checked-in heuristics.

    Each ``_helion_aot_<name>_cuda_sm<NN>.py`` file declares a compute capability
    the kernel ships a config for; map those to hardware aliases. This is the
    single source of truth -- no per-kernel declaration to keep in sync.
    """
    hardware = set()
    for path in (PRETUNED_KERNELS_DIR / name).glob(f"_helion_aot_{name}_cuda_sm*.py"):
        match = re.search(r"_cuda_(sm\d+)\.py$", path.name)
        if match:
            hardware.add(_HARDWARE_BY_COMPUTE.get(match.group(1), match.group(1)))
    return hardware


def _import_kernel_module(name: str) -> ModuleType:
    # Flat private module name (no dotted parent package, which Helion's
    # global-scope resolution would try to import) that avoids clashing with
    # examples/<name>.py.
    module_name = f"_helion_pretuned_run_{name}"
    file_path = PRETUNED_KERNELS_DIR / name / f"{name}.py"
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register before exec so Helion can resolve kernels that reference
    # module-level globals (global_scope_origin does sys.modules[__name__]).
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _compute_capability() -> str:
    major, minor = torch.cuda.get_device_capability()
    return f"sm{major}{minor}"


def run_kernel(name: str, current_hardware: str) -> dict:
    record: dict[str, object] = {"kernel": name}
    # Only benchmark a kernel on the hardware its checked-in heuristics target.
    supported = sorted(_supported_hardware(name))
    record["pretuned_hardware"] = supported
    if current_hardware not in supported:
        record["skipped"] = (
            f"pretuned for {supported}; current hardware is {current_hardware}"
        )
        print(f"Skipping {name}: {record['skipped']}")
        return record
    try:
        module = _import_kernel_module(name)
        # main(verbose=True) prints the per-shape table (relayed to CI logs) and
        # returns the metrics dict directly -- no stdout scraping.
        metrics = module.main(verbose=True)
        record.update(
            helion_wins=int(metrics["helion_wins"]),
            total=int(metrics["total"]),
            geomean=float(metrics["geomean"]),
            best_speedup=float(metrics["best_speedup"]),
            cudagraph=bool(module.use_cudagraph()),
        )
        # Optional per-baseline breakdown (helion vs each specific baseline),
        # for the dashboard's per-kernel dropdown.
        if metrics.get("baselines"):
            record["baselines"] = metrics["baselines"]
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
        help="Hardware alias to match against each kernel's checked-in heuristic "
        "files (default: derived from the GPU's compute capability).",
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
