# ruff: noqa: ANN401
"""Check a Helion port against its better-benchmark repro.

For each backend the port runs with the default config (no autotuning) and,
with ``--random-configs N``, with N random configs per kernel. Every run gets
fresh inputs and must match eager PyTorch in output count, shapes, dtypes and
strides, with integer and boolean outputs exact. Float outputs must pass the
anchored numerics gate (``common.AnchoredGate``, a loosened variant of
better-benchmark's: the error against its reference, eager PyTorch on
float64 copies of the inputs with the graph's own casts kept, may be at most
3x Inductor's or 1e-5 of the output's magnitude, and a bf16 or fp16 element
may also be one unit in the last place off), and outputs that depend on
random numbers must match Inductor's compiled outputs (masks exactly).
Without the gate (``--no-anchored``) float outputs must be allclose to eager
instead and random outputs go unchecked.

A random config that fails to compile or run is classified the way the
autotuner treats the failure: ``rejected`` when the backend declines the
config (``InvalidConfig``, ``BackendUnsupported``) or the autotuner skips the
failure quietly (resource limits, expected Triton compiler errors, errors
raised inside the CuTe DSL); ``warn`` when it skips it with a warning (a
PTXAS failure, an unexpected error inside the CuTe backend), which can be a
codegen bug worth reporting; ``error`` otherwise and for every wrong output.
Errors fail the check; warnings are printed in full but do not.

Examples::

    python benchmarks/cute/better_benchmark/check_port.py var_mean_004f9a9ea198
    python benchmarks/cute/better_benchmark/check_port.py var_mean_004f9a9ea198 \\
        --backends cute --random-configs 8 --print-code
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    # The default config is enough to check a port; no search.
    os.environ.setdefault("HELION_AUTOTUNE_EFFORT", "none")

from benchmarks.cute.better_benchmark import common
import torch

import helion
from helion.autotuner.logger import classify_triton_exception
from helion.autotuner.logger import match_unrecoverable_runtime_error


def classify_failure(error: BaseException, backend: Any) -> str:
    """``rejected``, ``warn`` or ``error`` for a random config that failed to
    compile or run, following the autotuner: the backend's
    ``classify_autotune_exception``, then ``classify_triton_exception``."""
    if isinstance(error, (helion.exc.BackendUnsupported, helion.exc.InvalidConfig)):
        return "rejected"
    if match_unrecoverable_runtime_error(error):
        return "error"
    action = (
        backend.classify_autotune_exception(error) if backend is not None else None
    ) or classify_triton_exception(error)
    return {"debug": "rejected", "warn": "warn"}.get(action, "error")


def judge(
    out: Any,
    eager_out: Any,
    gate: common.AnchoredGate | None,
    run: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Raise unless ``out`` is acceptable; returns what was checked."""
    record: dict[str, Any] = {"errors_vs_eager": common.output_errors(out, eager_out)}
    try:
        common.check_outputs(
            out,
            eager_out,
            rtol=args.rtol,
            atol=args.atol,
            stochastic=args.stochastic,
        )
    except common.FloatMismatch as error:
        if gate is None:
            raise
        record["eager_allclose_warning"] = str(error)[:1000]
    if gate is not None:
        verdict = gate.check(run)
        record["anchored"] = verdict
        if not verdict["pass"]:
            raise AssertionError(f"anchored numerics gate failed: {verdict}")
    return record


def check_backend(
    repro_id: str,
    shape: str,
    backend: str,
    args: argparse.Namespace,
    eager_out: Any,
    gate: common.AnchoredGate | None,
) -> dict[str, Any]:
    record: dict[str, Any] = {"backend": backend, "status": "error"}
    try:
        port = common.load_port(repro_id, backend)
        with torch.no_grad():
            out = port.forward(common.make_inputs(repro_id, shape))
        torch.cuda.synchronize()
        record.update(judge(out, eager_out, gate, port.forward, args))
        if args.print_code:
            for name, kernel in common.port_kernels(port).items():
                for bound in kernel._bound_kernels.values():
                    if bound._config is not None:
                        print(f"# ===== {backend}:{name} =====")
                        print(common.generated_code(bound))
        samples = []
        for sample in range(args.random_configs):
            picked: dict[str, str] = {}
            entry: dict[str, Any] = {"configs": picked}
            backend_impl = None
            lost_context = False
            try:
                for name, kernel in common.port_kernels(port).items():
                    for bound in kernel._bound_kernels.values():
                        backend_impl = bound.config_spec.backend
                        generation = bound.config_spec.create_config_generation()
                        config = generation.random_config()
                        picked[name] = config.to_json()
                        bound.set_config(config)
                with torch.no_grad():
                    out = port.forward(common.make_inputs(repro_id, shape))
                torch.cuda.synchronize()
            except Exception as error:
                entry["status"] = classify_failure(error, backend_impl)
                entry["error"] = f"{type(error).__name__}: {str(error)[:2000]}"
                lost_context = match_unrecoverable_runtime_error(error)
            else:
                try:
                    entry.update(judge(out, eager_out, gate, port.forward, args))
                    entry["status"] = "ok"
                except Exception as error:
                    # Wrong outputs, or a failure while checking them.
                    entry["status"] = "error"
                    entry["error"] = f"{type(error).__name__}: {str(error)[:2000]}"
                    lost_context = match_unrecoverable_runtime_error(error)
            samples.append(entry)
            print(f"[{backend}] random config {sample}: {entry['status']}", flush=True)
            if entry["status"] in ("warn", "error"):
                print(json.dumps(entry, indent=1, default=str)[:4000], flush=True)
            if lost_context:
                print(f"[{backend}] the CUDA context is lost; stopping", flush=True)
                break
        record["random_configs"] = samples
        failed = [s for s in samples if s["status"] == "error"]
        record["status"] = "ok" if not failed else "random_config_failures"
    except Exception:
        record["error"] = traceback.format_exc()
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("repro_id")
    parser.add_argument(
        "--shape",
        default=None,
        help="shape hash (default: the repro's selection.json point, else its first)",
    )
    parser.add_argument("--backends", default="triton,cute")
    parser.add_argument("--random-configs", type=int, default=0)
    parser.add_argument("--rtol", type=float, default=2e-2)
    parser.add_argument("--atol", type=float, default=2e-2)
    parser.add_argument(
        "--no-anchored",
        action="store_true",
        help="skip Inductor and the anchored gate (eager allclose only; "
        "random outputs go unchecked)",
    )
    parser.add_argument("--print-code", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    torch.cuda.set_device(0)
    shape = args.shape or common.default_shape(args.repro_id)
    instance = common.load_repro_module(args.repro_id).Repro()
    with torch.no_grad():
        eager_out = instance(*common.make_inputs(args.repro_id, shape))
    gate = None
    if not args.no_anchored:
        torch._dynamo.reset()
        compiled = common.compile_inductor(args.repro_id, shape, instance, cd=False)
        gate = common.AnchoredGate(args.repro_id, shape, instance, compiled)
        if gate.ref_fallback is not None:
            print(
                "note: the gate compares with eager PyTorch on the original "
                f"inputs; its float64 reference failed: {gate.ref_fallback}",
                flush=True,
            )
    args.stochastic = (
        gate.stochastic
        if gate is not None
        else common.stochastic_outputs(args.repro_id, shape, instance)
    )
    started = time.time()
    results: dict[str, Any] = {
        "repro": args.repro_id,
        "shape": shape,
        "stochastic_outputs": sorted(args.stochastic),
        "random_outputs": None if gate is None else sorted(gate.random),
        "backends": {},
    }
    ok = True
    warned = 0
    for backend in args.backends.split(","):
        record = check_backend(args.repro_id, shape, backend, args, eager_out, gate)
        results["backends"][backend] = record
        status = record["status"]
        ok = ok and status == "ok"
        print(f"[{backend}] {status}", flush=True)
        if "error" in record:
            print(record["error"], flush=True)
            continue
        for row in record["errors_vs_eager"]:
            print(f"[{backend}]   output {row}", flush=True)
        if "anchored" in record:
            print(f"[{backend}]   anchored gate pass={record['anchored']['pass']}")
            for leaf in record["anchored"].get("per_output", []):
                if leaf.get("reference") == "inductor":
                    print(f"[{backend}]   random output vs Inductor: {leaf}")
        if "eager_allclose_warning" in record:
            print(
                f"[{backend}]   note: not allclose to eager but the anchored gate "
                f"passed: {record['eager_allclose_warning'][:300]}",
                flush=True,
            )
        samples = record.get("random_configs") or []
        if samples:
            counts = {
                status: sum(s["status"] == status for s in samples)
                for status in ("ok", "rejected", "warn", "error")
            }
            print(f"[{backend}]   random configs {counts}", flush=True)
            warned += counts["warn"]
    results["seconds"] = round(time.time() - started, 1)
    if args.output:
        args.output.write_text(json.dumps(results, indent=1, default=str) + "\n")
    note = (
        f" ({warned} random configs failed with warnings, see above)" if warned else ""
    )
    print(f"{'PASS' if ok else 'FAIL'}{note}", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
