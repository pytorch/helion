"""Benchmark complete raw decode calls with the original output/state oracle.

Run with python -m benchmarks.cute.compare_recurrent_decode_backends. This public
runner records new measurements; it does not reuse historical qualifications.
Optional baseline packages are imported only for the requested implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
from typing import Any

from benchmarks.cute.recurrent_decode_inputs import RETENTION_ROUTES
from benchmarks.cute.recurrent_decode_inputs import build
from benchmarks.cute.recurrent_decode_inputs import cases
from benchmarks.cute.recurrent_decode_objective import baseline_workload
from benchmarks.cute.recurrent_decode_objective import make_objective
from benchmarks.cute.recurrent_decode_objective import selected_guards
from examples.linear.recurrent_decode import kernel_args
import torch

import helion
from helion.autotuner.benchmark_objective import BenchmarkRequest
from helion.autotuner.benchmark_objective import BenchmarkWorkload
from helion.autotuner.kernel_args import _clone_args
from helion.autotuner.objective_timing import ObjectiveTiming


def run(
    case: dict[str, Any],
    implementation: str,
    *,
    config: dict[str, Any] | None,
    effort: str,
    samples: int,
) -> dict[str, Any]:
    bundle = build(case, implementation)
    device = torch.cuda.get_device_properties(torch.device(case["device"]))
    flush = torch.zeros(
        2 * device.L2_cache_size, device=case["device"], dtype=torch.uint8
    )

    def clear_cache() -> None:
        flush.add_(1)

    timing = ObjectiveTiming(
        clear_cache,
        "required",
        json.dumps({"device": str(device.uuid), "flush_bytes": flush.numel()}),
        final_samples=samples,
        search_samples=30,
    )
    record = {
        "case": case["name"],
        "implementation": implementation,
        "contract": bundle["metadata"],
        "timing": timing.cache_record(),
    }
    if implementation == "helion":
        kernel = next(iter(bundle["kernels"].values()))
        kernel.settings.autotune_effort = effort
        objective = make_objective(bundle, case, timing.cache_record())
        # Strongly retain temporary squeeze/index views through compile and timing.
        prepared = kernel_args(case["family"], objective.raw_args[1])
        bound = kernel.bind(prepared)
        if config is not None:
            requested = helion.Config(**config)
            _, selected = (
                bound.config_spec.create_config_generation().strict_config_pair(
                    requested
                )
            )
        elif effort == "none":
            selected = bound.config_spec.default_config()
        else:
            selected = bound.autotune(
                prepared, benchmark_objective=objective, objective_timing=timing
            )
        compiled = bound.compile_config(selected)
        source = Path(compiled.__code__.co_filename)
        source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        request = BenchmarkRequest.create(selected, compiled, source_hash)
        record["guards"] = selected_guards(
            objective, request, bundle, case, timing.cache_record()
        )
        record.update(
            config=selected.config, source_sha256=source_hash, source=str(source)
        )
        private = tuple(_clone_args(objective.raw_args, None))
        workload = objective.bind(request, private)
    else:
        workload = baseline_workload(bundle)
    original_run = workload.run
    last_output = []

    def retained_run() -> object:
        output = original_run()
        last_output[:] = [output]
        return output

    workload = BenchmarkWorkload(
        retained_run, workload.reset, workload.validate, workload.close
    )
    try:
        for _ in range(3):
            workload.reset()
            workload.validate(workload.run())
        measured = timing.measure([workload], warmup_ms=1, rep_ms=50, samples=samples)[
            0
        ]
        # This is the retained graph output from the actual last measured replay.
        workload.validate(last_output[0])
        record.update(
            status="passed",
            samples_ms=list(measured),
            median_us=statistics.median(measured) * 1000,
        )
    finally:
        workload.reset()
        workload.close()
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=[case["name"] for case in cases()])
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--implementation", default="helion")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--config", type=Path, help="Explicit verification config; no shape dispatch"
    )
    parser.add_argument("--effort", choices=("none", "quick", "full"), default="none")
    parser.add_argument("--samples", type=int, default=120)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.list:
        print(json.dumps(cases(), indent=2))
        return
    if args.case is None:
        parser.error("--case is required unless --list is supplied")
    case = dict(
        next(row for row in cases() if row["name"] == args.case), device=args.device
    )
    if case["family"] == "retention-decode":
        case["implementations"] = ["helion", *RETENTION_ROUTES]
    config = None if args.config is None else json.loads(args.config.read_text())
    result = run(
        case,
        args.implementation,
        config=config,
        effort=args.effort,
        samples=args.samples,
    )
    rendered = json.dumps(result, indent=2)
    if args.output is not None:
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
