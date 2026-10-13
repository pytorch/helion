"""Benchmark the complete T128 fixed-decay Retention output/state call.

Run with python -m benchmarks.cute.compare_retention_prefill_backends. Explicit
--config is verification only; ordinary discovery uses all natural compiler
seeds and the original 30/120 whole-call objective. Measurements are a new epoch,
not a transfer of historical qualification. FLA is optional and loaded on demand.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
from typing import Any
from typing import Literal

from benchmarks.cute.retention_prefill_inputs import BASELINES
from benchmarks.cute.retention_prefill_inputs import STAGES
from benchmarks.cute.retention_prefill_inputs import arguments
from benchmarks.cute.retention_prefill_inputs import build
from benchmarks.cute.retention_prefill_inputs import cases
from benchmarks.cute.retention_prefill_inputs import make_kernel
from benchmarks.cute.retention_prefill_inputs import sha
from benchmarks.cute.retention_prefill_objective import RAW_NAMES
from benchmarks.cute.retention_prefill_objective import baseline_workload
from benchmarks.cute.retention_prefill_objective import make_objective
from benchmarks.cute.retention_prefill_objective import selected_guards
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
    output: Path,
    config: dict[str, Any] | None = None,
    effort: Literal["none", "quick", "full"] = "none",
    source_workers: int = 0,
    native_workers: int = 0,
    compile_timeout: int = 120,
) -> dict[str, Any]:
    if config is not None and effort != "none":
        raise ValueError("Explicit verification configs cannot seed a discovery run")
    output.mkdir(parents=True, exist_ok=False)
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
        json.dumps(
            {
                "device": str(device.uuid),
                "flush": "uint8 add_ 1",
                "bytes": flush.numel(),
            },
            sort_keys=True,
        ),
        final_samples=120,
        search_samples=30,
    )
    record = {
        "case": case["name"],
        "implementation": implementation,
        "status": "running",
        "historical_qualification": False,
        "natural_full_search_complete": False,
        "contract": bundle["metadata"],
        "timing": timing.cache_record(),
    }
    (output / "result.json").write_text(json.dumps(record, indent=2) + "\n")
    if implementation == "helion":
        kernel = make_kernel(
            effort=effort,
            source_workers=source_workers,
            native_workers=native_workers,
            compile_timeout=compile_timeout,
            log=str(output / "candidates.csv"),
        )
        bundle["kernels"][STAGES[0]] = kernel
        objective = make_objective(bundle, case, timing.cache_record())
        original_record = objective.cache_record()
        # Keep the actual raw owners and every call argument alive through search,
        # compilation, graph capture and the final complete-workload measurement.
        prepared = arguments(
            dict(zip(RAW_NAMES, objective.raw_args, strict=True)), case, objective.scale
        )[0]
        bound = kernel.bind(prepared)
        generation = bound.config_spec.create_config_generation()
        record.update(
            default=bound.config_spec.default_config().config,
            natural_seeds=[c.config for _, c in generation.seed_flat_config_pairs()],
            source_workers=source_workers,
            native_workers=native_workers,
            compile_timeout_seconds=compile_timeout,
            effort=effort,
        )
        if config is not None:
            _, selected = generation.strict_config_pair(helion.Config(**config))
        elif effort == "none":
            selected = bound.config_spec.default_config()
        else:
            selected = bound.autotune(
                prepared,
                force=True,
                benchmark_objective=objective,
                objective_timing=timing,
            )
            record["natural_full_search_complete"] = effort == "full"
        compiled = bound.compile_config(selected)
        source = Path(compiled.__code__.co_filename)
        digest = sha(source)
        assert bound.config_spec.backend.generated_source_hash(compiled) == digest
        request = BenchmarkRequest.create(selected, compiled, digest)
        record["guards"] = selected_guards(objective, request)
        assert objective.cache_record() == original_record
        private = tuple(_clone_args(objective.raw_args, None))
        workload = objective.bind(request, private)
        record.update(
            config=selected.config, source_sha256=digest, final_state_assessed=True
        )
        (output / "selected.py").write_bytes(source.read_bytes())
        (output / "config.json").write_text(
            json.dumps(selected.config, indent=2) + "\n"
        )
    else:
        workload = baseline_workload(bundle)
    original_run = workload.run
    last_output = []

    def retained_run() -> object:
        result = original_run()
        last_output[:] = [result]
        return result

    workload = BenchmarkWorkload(
        retained_run, workload.reset, workload.validate, workload.close
    )
    try:
        for _ in range(3):
            workload.reset()
            workload.validate(workload.run())
        measured = timing.measure([workload], warmup_ms=1, rep_ms=50, samples=120)[0]
        workload.validate(last_output[0])
        if implementation == "helion":
            assert objective.cache_record() == original_record
        record.update(
            status="passed",
            samples_ms=list(measured),
            median_us=statistics.median(measured) * 1000,
        )
    finally:
        workload.reset()
        workload.close()
    (output / "result.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--case", choices=[case["name"] for case in cases()])
    parser.add_argument(
        "--implementation", choices=("helion", *BASELINES), default="helion"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--config",
        type=Path,
        help="Explicit verification config, never a production shape dispatch",
    )
    parser.add_argument("--effort", choices=("none", "quick", "full"), default="none")
    parser.add_argument("--source-workers", type=int, default=0)
    parser.add_argument("--native-workers", type=int, default=0)
    parser.add_argument("--compile-timeout", type=int, default=120)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.list:
        print(json.dumps(cases(), indent=2))
        return
    if args.case is None or args.output is None:
        parser.error("--case and --output are required")
    case = dict(
        next(case for case in cases() if case["name"] == args.case), device=args.device
    )
    result = run(
        case,
        args.implementation,
        output=args.output,
        config=None if args.config is None else json.loads(args.config.read_text()),
        effort=args.effort,
        source_workers=args.source_workers,
        native_workers=args.native_workers,
        compile_timeout=args.compile_timeout,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
