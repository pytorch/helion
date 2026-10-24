"""Time one example-catalogue case with a pinned, default, or autotuned config.

Companion to ``compare_example_backends.py``: the same inputs, references,
tolerances and CUDA-graph timer, but one (module, shape, case, provider, impl)
per process so configs can be pinned independently per implementation.

  python benchmarks/cute/example_case_bench.py --module rope --shape 2 \
      --case 00_forward --provider cute --mode default
  python benchmarks/cute/example_case_bench.py --module rope --shape 2 \
      --case 00_forward --provider cute --pin-from <sweep raw record>.json
  python benchmarks/cute/example_case_bench.py --module rope --shape 2 \
      --case 00_forward --provider triton --mode autotune --effort full
  python benchmarks/cute/example_case_bench.py --module rope --shape 2 \
      --case 00_forward --provider baselines --impl torch.compile

``--pin KERNEL=JSON`` pins one Helion kernel (``examples.<module>.<fn>`` or the
bare function name) to a config; ``--pin-from`` reads the ``bound_kernels``
of a saved comparison record and pins every kernel it lists. HELION_BACKEND is
set from ``--provider`` before the example is imported.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if __name__ == "__main__":
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", required=True)
    parser.add_argument("--shape", required=True, type=int, choices=[0, 1, 2])
    parser.add_argument("--case", default="00_forward")
    parser.add_argument(
        "--provider", default="cute", choices=["cute", "triton", "baselines"]
    )
    parser.add_argument(
        "--impl", default="helion", help="Kernel name or baseline name in the case"
    )
    parser.add_argument(
        "--mode",
        default="default",
        choices=["default", "autotune", "pinned"],
        help="default: config_spec default; autotune: fresh search; "
        "pinned: use --pin/--pin-from (implied when either is given)",
    )
    parser.add_argument("--effort", default="full", help="Autotune effort")
    parser.add_argument("--seed", type=int, default=None, help="Autotune seed")
    parser.add_argument("--pin", action="append", default=[], metavar="KERNEL=JSON")
    parser.add_argument("--pin-from", type=Path, default=None)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument(
        "--kernel-profile",
        action="store_true",
        help="Also run torch.profiler over --reps eager calls and report the "
        "median per-call sum of GPU kernel durations as kernel_us (with a "
        "per-kernel breakdown on stderr)",
    )
    parser.add_argument(
        "--graph-calls",
        type=int,
        default=0,
        help="Also time one CUDA graph holding this many back-to-back calls "
        "(graph_calls_ms) and, when one replay takes < 400 us, a graph grown "
        "until it does (graph_long_ms; event times are quantized to 1.953 us)",
    )
    parser.add_argument("--no-graph", action="store_true")
    parser.add_argument(
        "--no-bench",
        action="store_true",
        help="Skip the cold-L2 interleaved_bench measurement (bench_ms)",
    )
    parser.add_argument("--print-code", action="store_true")
    parser.add_argument("--json", type=Path, default=None, help="Write result here")
    return parser.parse_args()


def configure_environment(args: argparse.Namespace) -> None:
    backend = args.provider if args.provider in {"cute", "triton"} else "triton"
    os.environ["HELION_BACKEND"] = backend
    os.environ.setdefault("HELION_BENCHMARK_CUDAGRAPH", "1")
    os.environ.pop("HELION_FORCE_AUTOTUNE", None)
    if args.pin or args.pin_from is not None:
        args.mode = "pinned"
    if args.mode == "autotune":
        os.environ["HELION_AUTOTUNE_EFFORT"] = args.effort
        os.environ["HELION_FORCE_AUTOTUNE"] = "1"
        # Same regime as the catalogue's autotunes: a flush, then a
        # back-to-back batch of calls, like the harness's 16-call graph.
        os.environ.setdefault("HELION_AUTOTUNE_L2_FLUSH", "replay")
        if args.seed is not None:
            os.environ["HELION_AUTOTUNE_RANDOM_SEED"] = str(args.seed)
    else:
        # Construct kernels with a real effort so the CuTe structural policy is
        # the one the autotuner would use; the config is pinned afterwards.
        os.environ.setdefault("HELION_AUTOTUNE_EFFORT", "full")
    if args.print_code:
        os.environ["HELION_PRINT_OUTPUT_CODE"] = "1"


def example_kernels(module: str) -> dict[str, Any]:
    from benchmarks.cute.example_baselines import live_example_kernels

    found = live_example_kernels()
    if not found:
        raise RuntimeError(f"No Helion kernels imported for examples.{module}")
    return found


def pin_configs(args: argparse.Namespace, kernels: dict[str, Any]) -> None:
    import helion

    # A record lists one config per bound kernel (specialization); a backward
    # case binds the same kernel several times (forward, each gradient) with
    # configs that need not be valid for each other's specialization, so every
    # distinct config is pinned and Helion's finite search picks per binding.
    pins: dict[str, list[dict]] = {}
    if args.pin_from is not None:
        record = json.loads(args.pin_from.read_text())
        for bound in record.get("bound_kernels") or []:
            distinct = list(dict.fromkeys(c for c in bound["configs"] if c))
            if distinct:
                pins[bound["kernel"]] = [json.loads(c) for c in distinct]
    for item in args.pin:
        name, _, text = item.partition("=")
        pins.setdefault(name, []).append(json.loads(text))
    pinned = 0
    for name, configs in pins.items():
        # ``name@2`` marks a second kernel built from the same function; every
        # kernel of that function takes the pin.
        matches = [
            k
            for k in kernels
            if k.split("@")[0] == name or k.split("@")[0].rsplit(".", 1)[-1] == name
        ]
        if not matches:
            raise ValueError(f"Pin {name!r} matches nothing among {list(kernels)}")
        for match in matches:
            kernels[match].configs = [helion.Config(**c) for c in configs]
            pinned += 1
            for config in configs:
                print(f"pinned {match}: {json.dumps(config)}", file=sys.stderr)
    if not pinned:
        raise SystemExit(
            "--mode pinned pinned no kernel: the record has no bound_kernels "
            "configs and no --pin was given (a full autotune would run instead)"
        )


def main() -> None:
    args = parse_args()
    configure_environment(args)
    from benchmarks.cute import example_baselines as baseline
    from benchmarks.cute.example_cases import collect
    import torch
    from torch.utils._pytree import tree_map_only

    torch.cuda.set_device(0)
    cases = collect(args.module, args.shape)
    case = next(c for c in cases if c.name == args.case)
    torch.set_grad_enabled(case.mode == "backward")
    kernels = example_kernels(args.module)
    if args.provider in {"cute", "triton"}:
        if args.mode == "pinned":
            pin_configs(args, kernels)
        elif args.mode == "default":
            for kernel in kernels.values():
                kernel.settings.autotune_effort = "none"
    expected = baseline.make_callable(next(iter(case.baselines.values())), case)()
    expected = tree_map_only(torch.Tensor, lambda t: t.detach().clone(), expected)
    output_dtype = case.output_dtypes.get(f"{args.provider}:{args.impl}")
    if args.provider == "baselines":
        candidate = baseline.baseline_factories(case)[args.impl]()
        if isinstance(candidate, baseline.PreparedCandidate):
            print(f"baseline details: {candidate.details}", file=sys.stderr)
            candidate = candidate.fn
    else:
        candidate = case.kernels[args.impl]
    call = baseline.make_callable(candidate, case)
    started = time.monotonic()
    actual = call()
    torch.cuda.synchronize()
    first_call_s = time.monotonic() - started
    baseline.assert_correct(actual, expected, case, output_dtype=output_dtype)
    result: dict[str, Any] = {
        "module": args.module,
        "shape": args.shape,
        "case": args.case,
        "provider": args.provider,
        "impl": args.impl,
        "mode": args.mode,
        "description": case.description,
        "first_call_seconds": first_call_s,
        "eager_correctness": "pass",
    }
    if args.no_graph:
        torch.cuda.synchronize()
        runs = [baseline._event_ms(call) for _ in range(args.reps)]
        result.update(event_ms=statistics.median(runs), event_runs_ms=runs)
    else:
        result.update(
            baseline.measure(
                call,
                args.reps,
                expected,
                case,
                output_dtype=output_dtype,
                graph_nodes=args.kernel_profile,
            )
        )
    if args.kernel_profile:
        result.update(baseline.kernel_profile(call, max(args.reps, 20)))
    if args.graph_calls > 1 and not args.no_graph:
        result.update(
            baseline.measure_batched(
                call,
                args.reps,
                args.graph_calls,
                expected,
                case,
                output_dtype=output_dtype,
            )
        )
    if not args.no_graph and not args.no_bench:
        # Secondary: the autotuner's cold-L2 regime (bench_ms).
        result.update(baseline.measure_cold(call, args.reps))
    if args.provider in {"cute", "triton"}:
        result["bound_kernels"] = baseline.active_kernel_configs()
        for bound in result["bound_kernels"]:
            for config in bound["configs"]:
                print(f"config {bound['kernel']}: {config}", file=sys.stderr)
    summary = {
        k: v
        for k, v in result.items()
        if k
        in {
            "bench_ms",
            "bench_error",
            "event_ms",
            "graph_ms",
            "graph_error",
            "graph_calls_ms",
            "graph_long_ms",
            "graph_long_calls",
            "graph_calls_error",
            "kernel_us",
            "first_call_seconds",
        }
    }
    print(
        f"RESULT {args.module}/{args.case}/s{args.shape} {args.provider}:{args.impl} "
        f"[{args.mode}] {json.dumps(summary)}"
    )
    if args.json is not None:
        args.json.write_text(json.dumps(result, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
