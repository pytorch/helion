"""Benchmark complete SSD output calls, or run cold full discovery per role.

Use --configs only for explicit verification. --effort full starts from ordinary
defaults/seeds and searches every stage under the complete raw-input objective.
Every public record is a new measurement epoch, not a historical qualification.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
from typing import Any

from benchmarks.cute.mamba2_ssd_inputs import BASELINES
from benchmarks.cute.mamba2_ssd_inputs import RAW_NAMES
from benchmarks.cute.mamba2_ssd_inputs import build
from benchmarks.cute.mamba2_ssd_inputs import cases
from benchmarks.cute.mamba2_ssd_inputs import configure
from benchmarks.cute.mamba2_ssd_inputs import sha
from benchmarks.cute.mamba2_ssd_inputs import stages
from benchmarks.cute.mamba2_ssd_objective import MambaObjective
from benchmarks.cute.mamba2_ssd_objective import baseline_workload
from benchmarks.cute.mamba2_ssd_objective import selected_guards
from benchmarks.cute.mamba2_ssd_objective import stage_arguments
from benchmarks.cute.mamba2_ssd_search import compile_defaults
from benchmarks.cute.mamba2_ssd_search import run_sweep
import torch

import helion
from helion.autotuner.benchmark_objective import BenchmarkRequest
from helion.autotuner.benchmark_objective import BenchmarkWorkload
from helion.autotuner.kernel_args import _clone_args
from helion.autotuner.objective_timing import ObjectiveTiming


def compile_verification(
    case: dict[str, Any],
    bundle: dict[str, Any],
    configs: dict[str, Any],
    timing: ObjectiveTiming,
) -> dict[str, Any]:
    """Compile an explicitly supplied tuple, retaining every temporary argument."""
    assert set(configs) == set(stages(case))
    raw = {name: bundle["inputs"][name] for name in RAW_NAMES}
    values, retained_args, requests, bounds = [], [], [], []
    for index, (name, kernel) in enumerate(
        bundle["objective_binding"]["kernels"].items()
    ):
        configure(kernel)
        args = stage_arguments(raw, case, index, values)
        retained_args.append(args)
        bound = kernel.bind(args)
        _, config = bound.config_spec.create_config_generation().strict_config_pair(
            helion.Config(**configs[name])
        )
        compiled = bound.compile_config(config)
        digest = bound.config_spec.backend.generated_source_hash(compiled)
        assert digest == sha(compiled.__code__.co_filename)
        requests.append(BenchmarkRequest.create(config, compiled, digest))
        bounds.append(bound)
        values.append(bundle["objective_binding"]["references"][index](*args))
    objective = MambaObjective(
        bundle,
        case=case,
        selected=len(requests) - 1,
        partners=dict(enumerate(requests[:-1])),
        provenance={"explicit_verification_tuple": True},
        timing_policy=timing.cache_record(),
    )
    return {
        "requests": tuple(requests),
        "bounds": tuple(bounds),
        "objective": objective,
        "representatives": (values, retained_args),
    }


def run(
    case: dict[str, Any],
    implementation: str,
    *,
    output: Path,
    configs: dict[str, Any] | None = None,
    effort: str = "none",
    samples: int = 120,
) -> dict[str, Any]:
    if configs is not None and effort != "none":
        raise ValueError("Explicit verification tuples cannot seed a cold full search")
    output.mkdir(parents=True, exist_ok=False)
    bundle = build(case, implementation)
    device = torch.cuda.get_device_properties(torch.device(case["device"]))
    stream = torch.cuda.current_stream()
    flush = torch.zeros(
        2 * device.L2_cache_size, device=case["device"], dtype=torch.uint8
    )

    def clear_cache() -> None:
        flush.add_(1)

    timing = ObjectiveTiming(
        clear_cache,
        "required",
        json.dumps({"device": str(device.uuid), "flush_bytes": flush.numel()}),
        final_samples=120,
        search_samples=30,
    )
    record = {
        "case": case["name"],
        "implementation": implementation,
        "status": "running",
        "output_only": True,
        "final_state_assessed": False,
        "cold_full_roles_completed": False,
        "historical_qualification": False,
        "timing": timing.cache_record(),
        "contract": bundle["metadata"],
    }
    (output / "result.json").write_text(json.dumps(record, indent=2) + "\n")
    selected: dict[str, Any]
    if implementation == "helion":
        if configs is not None:
            selected = compile_verification(case, bundle, configs, timing)
        else:
            binding = compile_defaults(case, bundle, output, stream.synchronize)
            if effort == "full":
                selected = run_sweep(
                    binding,
                    provenance={"public_cold_full": True},
                    timing=timing,
                    output=output,
                    synchronize=stream.synchronize,
                )
                record["cold_full_roles_completed"] = True
            else:
                requests = binding["requests"]
                objective = MambaObjective(
                    bundle,
                    case=case,
                    selected=len(requests) - 1,
                    partners=dict(enumerate(requests[:-1])),
                    provenance={"ordinary_partner_bootstrap": True},
                    timing_policy=timing.cache_record(),
                )
                selected = {
                    "requests": requests,
                    "bounds": binding["bounds"],
                    "objective": objective,
                    "representatives": binding,
                }
        objective = selected["objective"]
        request = selected["requests"][-1]
        record["guards"] = selected_guards(objective, request)
        private = tuple(_clone_args(objective.raw_args, None))
        workload = objective.bind(request, private)
        records = []
        for name, member in zip(stages(case), selected["requests"], strict=True):
            path = Path(member.compiled.__code__.co_filename)
            assert sha(path) == member.source_hash
            (output / (name + ".py")).write_bytes(path.read_bytes())
            records.append(
                {
                    "stage": name,
                    "config": member.config.config,
                    "source_sha256": member.source_hash,
                }
            )
        record["stages"] = records
        record["no_fixed_partner_injection"] = configs is None
    else:
        workload = baseline_workload(bundle)
    last_output = []
    original_run = workload.run

    def retained_run() -> object:
        value = original_run()
        last_output[:] = [value]
        return value

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
        workload.validate(last_output[0])
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
        "--configs", type=Path, help="Explicit stage-name to config JSON mapping"
    )
    parser.add_argument("--effort", choices=("none", "full"), default="none")
    parser.add_argument("--samples", type=int, default=120)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.list:
        print(json.dumps(cases(), indent=2))
        return
    if args.case is None or args.output is None:
        parser.error("--case and --output are required")
    case = dict(
        next(row for row in cases() if row["name"] == args.case), device=args.device
    )
    result = run(
        case,
        args.implementation,
        output=args.output,
        effort=args.effort,
        configs=None if args.configs is None else json.loads(args.configs.read_text()),
        samples=args.samples,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
