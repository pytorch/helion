"""Compare the non-distributed examples with ATen, Triton, CuTe and baselines.

Run ``--help`` for the CLI. Optional baseline packages are used when installed;
their failures remain in the result. Each implementation gets a fresh process
and private compiler caches. FULL searches have no time or generation budget.
This is a comparison tool, not the historical hillclimb qualification protocol.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from datetime import timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import secrets
import subprocess
import sys
import tempfile
import time
import traceback
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if __name__ == "__main__":
    sys.path.insert(0, str(REPO_ROOT))

SCHEMA = "example-backend-comparison-v1"


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n")


def worker(args: argparse.Namespace) -> None:
    from benchmarks.cute import example_baselines as baseline
    from benchmarks.cute.example_cases import collect
    import torch
    from torch.utils._pytree import tree_map_only

    from helion.autotuner.metrics import register_post_autotune_hook

    result: dict[str, Any] = {
        "module": args.module,
        "shape": args.shape,
        "case": args.case,
        "provider": args.provider,
        "impl": args.impl,
        "status": "error",
        "autotune_seed": args.seed,
    }
    started = time.monotonic()
    try:
        torch.cuda.set_device(0)
        cases = collect(args.module, args.shape)
        if args.provider == "inventory":
            result.update(
                status="ok",
                cases=[
                    {
                        "name": case.name,
                        "kernels": list(case.kernels),
                        "baselines": list(baseline.baseline_factories(case)),
                    }
                    for case in cases
                ],
            )
            return
        case = next(case for case in cases if case.name == args.case)
        torch.set_grad_enabled(case.mode == "backward")
        result.update(
            input_seed=20260911 + args.shape,
            args=baseline.metadata(case.args),
            captured_inputs=baseline.captured_metadata(case.kernels),
            description=case.description,
            mode=case.mode,
            rtol=case.rtol,
            atol=case.atol,
            check=case.check,
            device=torch.cuda.get_device_name(0),
        )
        expected = baseline.make_callable(next(iter(case.baselines.values())), case)()
        expected = tree_map_only(torch.Tensor, lambda t: t.detach().clone(), expected)
        output_dtype = case.output_dtypes.get(f"{args.provider}:{args.impl}")
        result.update(
            reference_output=baseline.metadata(expected),
            output_contract=baseline.output_contract(
                expected, output_dtype=output_dtype
            ),
            reference_note=case.reference_note,
        )
        autotunes: list[dict[str, Any]] = []
        register_post_autotune_hook(lambda metrics: autotunes.append(metrics.to_dict()))
        candidate = (
            baseline.baseline_factories(case)[args.impl]()
            if args.provider == "baselines"
            else case.kernels[args.impl]
        )
        if isinstance(candidate, baseline.PreparedCandidate):
            result["baseline_config"] = candidate.details
            candidate = candidate.fn
        call = baseline.make_callable(candidate, case)
        actual = call()
        result["output"] = baseline.metadata(actual)
        baseline.assert_correct(actual, expected, case, output_dtype=output_dtype)
        result["eager_correctness"] = "pass"
        result.update(
            baseline.measure(
                call, args.repetitions, expected, case, output_dtype=output_dtype
            )
        )
        actual = call()
        result["post_measurement_output"] = baseline.metadata(actual)
        baseline.assert_correct(actual, expected, case, output_dtype=output_dtype)
        result["post_measurement_correctness"] = "pass"
        result["autotunes"] = autotunes
        result["bound_kernels"] = baseline.active_kernel_configs()
        if args.provider in {"cute", "triton"} and any(
            kernel["backend"] != args.provider for kernel in result["bound_kernels"]
        ):
            raise RuntimeError("An example used a different Helion backend")
        if result.get("graph_correctness") != "pass":
            raise RuntimeError("CUDA graph measurement or correctness failed")
        result["status"] = "ok"
    except Exception:
        result["error"] = traceback.format_exc()
    finally:
        result["wall_seconds"] = time.monotonic() - started
        write_json(args.output, result)


def worker_environment(
    gpu: str, provider: str, cache: Path, seed: int
) -> dict[str, str]:
    # Do not inherit an unrelated experiment's search budget, forced config,
    # kernel overrides, or backend setting. Optional package paths still work.
    env = {
        key: value for key, value in os.environ.items() if not key.startswith("HELION_")
    }
    env.update(
        CUDA_VISIBLE_DEVICES=gpu,
        HELION_BACKEND=provider if provider in {"cute", "triton"} else "triton",
        HELION_AUTOTUNE_EFFORT="full",
        HELION_FORCE_AUTOTUNE="1",
        HELION_AUTOTUNE_RANDOM_SEED=str(seed),
        HELION_BENCHMARK_CUDAGRAPH="1",
        HELION_CACHE_DIR=str(cache / "helion"),
        TRITON_CACHE_DIR=str(cache / "triton"),
        TORCHINDUCTOR_CACHE_DIR=str(cache / "inductor"),
        CUTE_DSL_CACHE_DIR=str(cache / "cute"),
        CUDA_CACHE_PATH=str(cache / "cuda"),
        QUACK_CACHE_DIR=str(cache / "quack"),
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        PYTHONPATH=os.pathsep.join([str(REPO_ROOT), env.get("PYTHONPATH", "")]),
    )
    return env


def check_gpu_idle(gpu: str) -> str:
    """Refuse to start a new worker while another GPU process is present."""
    query = subprocess.run(
        ["nvidia-smi", "-i", gpu, "--query-gpu=uuid", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    )
    uuids = query.stdout.strip().splitlines()
    if len(uuids) != 1:
        raise ValueError("--gpu must identify exactly one physical GPU")
    uuid = uuids[0]
    processes = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    if any(
        line.split(",", 1)[0].strip() == uuid for line in processes.stdout.splitlines()
    ):
        raise RuntimeError(f"GPU {gpu} is busy; existing results have been preserved")
    return uuid


def provenance(gpu: str) -> dict[str, Any]:
    import torch

    query = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            gpu,
            "--query-gpu=uuid,name,power.limit,driver_version",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    uuid, name, power, driver = next(
        csv.reader(query.stdout.splitlines(), skipinitialspace=True)
    )
    versions = {"python": sys.version.split()[0], "cuda_runtime": torch.version.cuda}
    for package in ("torch", "triton", "nvidia-cutlass-dsl", "quack-kernels"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "gpu_properties": {
            "uuid": uuid,
            "name": name,
            "power_limit_w": float(power),
            "driver_version": driver,
        },
        "software": versions,
    }


def run(args: argparse.Namespace) -> None:
    from benchmarks.cute.compare_attention_backends import _git_source_snapshot
    from benchmarks.cute.compare_attention_backends import _git_worktree_root_matches
    from benchmarks.cute.example_cases import MODULES

    modules = args.modules.split(",") if args.modules else MODULES
    if set(modules) - set(MODULES):
        raise ValueError(f"Unknown example modules: {set(modules) - set(MODULES)}")
    if len(set(modules)) != len(modules):
        raise ValueError("Duplicate example modules")
    shapes = [int(value) for value in args.shapes.split(",")]
    if not shapes or len(set(shapes)) != len(shapes) or set(shapes) - {0, 1, 2}:
        raise ValueError("Shapes must be distinct members of 0,1,2")
    seed = args.seed if args.seed is not None else secrets.randbits(63)
    args.output.mkdir(parents=True, exist_ok=False)
    raw = args.output / "raw"
    raw.mkdir()
    source = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    record: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": source.stdout.strip()
        if source.returncode == 0 and _git_worktree_root_matches(REPO_ROOT)
        else None,
        "source_snapshot": _git_source_snapshot(
            REPO_ROOT, ("helion", "examples", "benchmarks/cute")
        ),
        "modules": modules,
        "shapes": shapes,
        "gpu": args.gpu,
        "gpu_uuid": check_gpu_idle(args.gpu),
        "effort": "full",
        "budget_seconds": None,
        "repetitions": args.repetitions,
        "autotune_seed": seed,
        **provenance(args.gpu),
        "timer": "CUDA graph replay; do_bench(warmup=10, rep=50, return_mode=median)",
        "jobs": [],
    }

    def execute(
        module: str, shape: int, provider: str, case: str = "", impl: str = ""
    ) -> dict[str, Any]:
        check_gpu_idle(args.gpu)
        index = len(record["jobs"])
        worker_seed = (seed + index) % (1 << 63)
        output = raw / f"{index:05d}.json"
        log = raw / f"{index:05d}.log"
        with tempfile.TemporaryDirectory(prefix="hbench-") as directory:
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "worker",
                "--module",
                module,
                "--shape",
                str(shape),
                "--provider",
                provider,
                "--case",
                case,
                "--impl",
                impl,
                "--output",
                str(output),
                "--repetitions",
                str(args.repetitions),
                "--seed",
                str(worker_seed),
            ]
            with log.open("w") as stream:
                process = subprocess.run(
                    command,
                    env=worker_environment(
                        args.gpu, provider, Path(directory), worker_seed
                    ),
                    cwd=REPO_ROOT,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            result = (
                json.loads(output.read_text())
                if output.exists()
                else {
                    "module": module,
                    "shape": shape,
                    "provider": provider,
                    "case": case,
                    "impl": impl,
                    "status": "process_error",
                }
            )
            if process.returncode:
                result.update(status="process_error", returncode=process.returncode)
                write_json(output, result)
        record["jobs"].append(
            {
                "path": str(output.relative_to(args.output)),
                "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                "log": str(log.relative_to(args.output)),
                "status": result["status"],
                "autotune_seed": worker_seed,
            }
        )
        write_json(args.output / "results.json", record)
        print(
            f"{module}/{case}/shape{shape} {provider}:{impl}: {result['status']}",
            flush=True,
        )
        return result

    try:
        for module in modules:
            for shape in shapes:
                inventory = execute(module, shape, "inventory")
                if inventory["status"] != "ok":
                    continue
                for case in inventory["cases"]:
                    jobs = [
                        (provider, name)
                        for provider in ("cute", "triton")
                        for name in case["kernels"]
                    ] + [("baselines", name) for name in case["baselines"]]
                    random.Random(seed + shape).shuffle(jobs)
                    for provider, name in jobs:
                        execute(module, shape, provider, case["name"], name)
        record["status"] = "complete"
    finally:
        write_json(args.output / "results.json", record)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    commands.add_parser(
        "list", help="List supported example modules without running kernels"
    )
    runner = commands.add_parser(
        "run", help="Run an uncapped FULL comparison on one GPU"
    )
    runner.add_argument("--gpu", required=True)
    runner.add_argument(
        "--seed", type=int, help="Run seed; default: a fresh random seed"
    )
    runner.add_argument(
        "--modules", default="", help="Comma-separated module names; default: all"
    )
    runner.add_argument("--shapes", default="0,1,2")
    runner.add_argument("--output", required=True, type=Path)
    runner.add_argument("--repetitions", type=int, default=5)
    child = commands.add_parser("worker", help=argparse.SUPPRESS)
    child.add_argument("--module", required=True)
    child.add_argument("--seed", type=int, required=True)
    child.add_argument("--shape", required=True, type=int)
    child.add_argument(
        "--provider",
        required=True,
        choices=["inventory", "baselines", "cute", "triton"],
    )
    child.add_argument("--case", default="")
    child.add_argument("--impl", default="")
    child.add_argument("--output", required=True, type=Path)
    child.add_argument("--repetitions", type=int, default=5)
    report = commands.add_parser(
        "report", help="Create CSV, Markdown and a matplotlib bar chart"
    )
    report.add_argument("--input", required=True, type=Path)
    report.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.action == "list":
        from benchmarks.cute.example_cases import MODULES

        print("\n".join(MODULES))
    elif args.action == "report":
        from benchmarks.cute.example_report import report_results

        report_results(args.input, args.output)
    else:
        if args.seed is not None and not 0 <= args.seed < (1 << 63):
            parser.error("--seed must be in [0, 2**63)")
        if args.repetitions < 1:
            parser.error("--repetitions must be positive")
        args.output = args.output.resolve()
        (worker if args.action == "worker" else run)(args)


if __name__ == "__main__":
    main()
