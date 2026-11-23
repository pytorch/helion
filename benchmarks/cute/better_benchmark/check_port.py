# ruff: noqa: ANN401
"""Check a Helion port against its better-benchmark repro.

For each backend the port runs with the default config (no autotuning),
with ``--random-configs N`` with N random configs per kernel, and with
``--configs-dir DIR`` pinned to the autotune record that DIR holds for it (a
campaign's pick, as ``bench.py`` pins it). Every run gets fresh inputs and
free device memory filled with poison (``common.call_on_poisoned_memory``),
and must match eager PyTorch in output count, shapes, dtypes and strides,
with integer and boolean outputs exact. A port that leaves part of an output
unwritten fails whatever that memory held (``common.check_written``: before
the fill, ``torch.empty`` handed a config the block that the config checked
before it had just freed, right values included). Float outputs must pass the
anchored numerics gate (``common.AnchoredGate``, a loosened variant of
better-benchmark's: the error against its reference, eager PyTorch on
float64 copies of the inputs with the graph's own casts kept, may be at most
3x Inductor's or 1e-5 of the output's magnitude, and a bf16 or fp16 element
may also be one unit in the last place off), and outputs that depend on
random numbers must match Inductor's compiled outputs (masks exactly).
An integer or boolean output that Inductor's own compiled output does not
reproduce either (eager leaves the order of ``topk``'s tied values to its
implementation, and Inductor's decomposition returns them in index order)
must equal Inductor's exactly instead; the check prints each such output
with the number of elements in which Inductor's differs from eager's.
Without the gate (``--no-anchored``) nothing is compiled by Inductor: float
outputs must be allclose to eager instead, random outputs go unchecked and
every integer and boolean output must equal eager's.

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
    # The default config is enough to check a port; no search. An empty
    # value names no effort, for load_port as for Helion's settings: left
    # in place beside HELION_FORCE_AUTOTUNE=1 it would be a full search.
    if not os.environ.get("HELION_AUTOTUNE_EFFORT", "").strip():
        os.environ["HELION_AUTOTUNE_EFFORT"] = "none"

from benchmarks.cute.better_benchmark import common
import torch

import helion
from helion.autotuner.logger import classify_triton_exception
from helion.autotuner.logger import match_unrecoverable_runtime_error


def classify_failure(error: BaseException, backend: Any) -> str:
    """``rejected``, ``warn`` or ``error`` for a random config that failed to
    compile or run, following the autotuner: the backend's
    ``classify_autotune_exception``, then ``classify_triton_exception``. A
    call that could not be judged (``common.UnfilledMemory``) is the
    check's own failure, not one of the config: an ``error``."""
    if isinstance(error, common.UnfilledMemory):
        return "error"
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
    port: Any,
    repro_id: str,
    shape: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Raise unless ``out``, the port's outputs on the shape point from
    ``common.forward_on_poisoned_memory``, is acceptable; returns what was
    checked. An output that the port left partly unwritten is never
    acceptable (``common.check_written``, which calls the port again when an
    output still holds the poison)."""
    record: dict[str, Any] = {"errors_vs_eager": common.output_errors(out, eager_out)}
    common.check_written(
        out,
        lambda byte: common.forward_on_poisoned_memory(port, repro_id, shape, byte),
    )
    vs_inductor: list[dict[str, Any]] = []
    try:
        common.check_outputs(
            out,
            eager_out,
            rtol=args.rtol,
            atol=args.atol,
            stochastic=args.stochastic,
            inductor=None if gate is None else lambda: gate.baseline,
            vs_inductor=vs_inductor,
        )
    except common.FloatMismatch as error:
        if gate is None:
            raise
        record["eager_allclose_warning"] = str(error)[:1000]
    if vs_inductor:
        record["exact_vs_inductor"] = vs_inductor
    if gate is not None:
        verdict = gate.check(port.forward)
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
        out = common.forward_on_poisoned_memory(port, repro_id, shape)
        record.update(judge(out, eager_out, gate, port, repro_id, shape, args))
        del out
        if args.print_code:
            for name, kernel in common.port_kernels(port).items():
                for bound in kernel._bound_kernels.values():
                    if bound._config is not None:
                        print(f"# ===== {backend}:{name} =====")
                        print(common.generated_code(bound))
        samples = []
        lost_context = False
        for sample in range(args.random_configs):
            picked: dict[str, str] = {}
            entry: dict[str, Any] = {"configs": picked}
            backend_impl = None
            try:
                for name, kernel in common.port_kernels(port).items():
                    for bound in kernel._bound_kernels.values():
                        backend_impl = bound.config_spec.backend
                        generation = bound.config_spec.create_config_generation()
                        config = generation.random_config()
                        picked[name] = config.to_json()
                        bound.set_config(config)
                out = common.forward_on_poisoned_memory(port, repro_id, shape)
            except Exception as error:
                entry["status"] = classify_failure(error, backend_impl)
                entry["error"] = f"{type(error).__name__}: {str(error)[:2000]}"
                lost_context = match_unrecoverable_runtime_error(error)
            else:
                try:
                    entry.update(
                        judge(out, eager_out, gate, port, repro_id, shape, args)
                    )
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
        if args.configs_dir is not None and not lost_context:
            pinned = check_pinned(repro_id, shape, backend, args, eager_out, gate)
            record["pinned"] = pinned
            if pinned["status"] == "error" and record["status"] == "ok":
                record["status"] = "pinned_config_failure"
    except Exception:
        record["error"] = traceback.format_exc()
    return record


def check_pinned(
    repro_id: str,
    shape: str,
    backend: str,
    args: argparse.Namespace,
    eager_out: Any,
    gate: common.AnchoredGate | None,
) -> dict[str, Any]:
    """Check the port with its kernels pinned to the autotune record that
    ``--configs-dir`` holds for it (a campaign's pick, as ``bench.py`` pins
    it): ``ok`` with what was checked, ``error`` with why, or ``missing``
    when the directory has no record of the port on this backend, or the
    record of an autotune that failed. A record that is there and cannot be
    read is an error: nothing was checked. ``code_matches_autotune`` says
    whether the pinned kernels still generate the code the record holds
    (``common.pin_roundtrip``: the source-line comments do not count, except
    for a record that holds only the digest of the whole text)."""
    path = common.config_record_path(args.configs_dir, repro_id, shape, backend)
    record: dict[str, Any] = {"autotune_record": str(path), "status": "error"}
    if not path.exists():
        record["status"] = "missing"
        return record
    try:
        tuned = json.loads(path.read_text())
        if tuned.get("status") != "ok":
            record["status"] = "missing"
            return record
        port = common.load_port(
            repro_id, backend, pinned=common.per_kernel(tuned["kernels"], "decorator")
        )
        out = common.forward_on_poisoned_memory(port, repro_id, shape)
        record.update(judge(out, eager_out, gate, port, repro_id, shape, args))
        roundtrip = common.pin_roundtrip(port, tuned["kernels"])
        record["code_matches_autotune"] = all(roundtrip.values())
        record["status"] = "ok"
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
    parser.add_argument(
        "--configs-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="also check each backend's port pinned to its autotune record "
        "in DIR (a campaign's configs), where DIR has an ok one",
    )
    parser.add_argument("--print-code", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.configs_dir is not None and not args.configs_dir.is_dir():
        # A mistyped directory would read as no record for any backend.
        parser.error(f"--configs-dir {args.configs_dir}: not a directory")

    torch.cuda.set_device(0)
    shape = args.shape or common.default_shape(args.repro_id)
    instance = common.load_repro_module(args.repro_id).Repro()
    with torch.no_grad():
        eager_out = instance(*common.make_inputs(args.repro_id, shape))
    gate = None
    if not args.no_anchored:
        torch._dynamo.reset()
        compiled = common.compile_inductor(args.repro_id, shape, instance)
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
        for entry in record.get("exact_vs_inductor", []):
            print(
                f"[{backend}]   exact output vs Inductor, whose own differs from "
                f"eager's: {entry}",
                flush=True,
            )
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
        pinned = record.get("pinned")
        if pinned is not None:
            print(
                f"[{backend}]   pinned config ({pinned['autotune_record']}): "
                f"{pinned['status']}",
                flush=True,
            )
            if "error" in pinned:
                print(pinned["error"], flush=True)
            elif pinned["status"] == "ok":
                if not pinned["code_matches_autotune"]:
                    print(
                        f"[{backend}]   note: the pinned kernels generate other "
                        "code than the autotune recorded",
                        flush=True,
                    )
                print(
                    f"[{backend}]   pinned anchored gate "
                    f"pass={pinned.get('anchored', {}).get('pass')}",
                    flush=True,
                )
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
