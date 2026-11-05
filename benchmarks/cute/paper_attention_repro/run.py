"""Run the paper's attention workloads with isolated archived harnesses."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime
from datetime import timezone
import hashlib
import io
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time
from typing import Any
from typing import cast

if __package__:
    from . import protocol
else:
    import protocol  # pyrefly: ignore [missing-import]

HERE = Path(__file__).resolve().parent
SOURCE_FIELDS = ("helion", "harness")
EMPTY_DIFF_SHA256 = hashlib.sha256(b"").hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_runtime(path: Path) -> tuple[dict[str, Any], Path]:
    runtime = protocol.read_json(path)
    if "targets" in runtime:
        return runtime, path
    artifact_root = (path.parent / runtime["artifact_root"]).resolve()
    manifest = artifact_root / "runtime_manifest.json"
    if manifest.exists():
        return protocol.read_json(manifest), manifest
    return {
        "schema_version": 1,
        "artifact_root": str(artifact_root),
        "status": "preparing",
        "targets": {},
    }, manifest


def command_output(
    command: list[str], *, env: dict[str, str] | None = None, cwd: str | None = None
) -> str:
    return subprocess.check_output(
        command, text=True, timeout=30, env=env, cwd=cwd
    ).strip()


def environment(target: dict[str, Any], gpu: int | None = None) -> dict[str, str]:
    env = dict(os.environ)
    env.update(target.get("environment", {}))
    paths = [target["helion_root"], target["harness_root"]]
    fa4 = target.get("fa4_pythonpath")
    if fa4:
        paths.append(fa4)
        env["FA4_PYTHONPATH"] = fa4
    env["PYTHONPATH"] = os.pathsep.join(paths)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = "" if gpu is None else str(gpu)
    env["PAPER_REPRO_HELION_ROOT"] = target["helion_root"]
    env["PAPER_REPRO_HARNESS_ROOT"] = target["harness_root"]
    env.pop("HELION_BACKEND", None)
    return env


SOFTWARE_PROBE = """
import importlib.metadata as metadata
import importlib
import json
import os
import platform
from pathlib import Path
import sys

sys.path[:0] = os.environ["PYTHONPATH"].split(os.pathsep)

import cutlass
import helion
import torch
import triton

if Path(helion.__file__).resolve().parents[1] != Path(os.environ["PAPER_REPRO_HELION_ROOT"]).resolve():
    raise RuntimeError("Software probe imported Helion from an unexpected checkout")
kernels = importlib.import_module("benchmarks.helion_attn.kernels")
if Path(os.environ["PAPER_REPRO_HARNESS_ROOT"]).resolve() not in Path(kernels.__file__).resolve().parents:
    raise RuntimeError("Software probe imported kernels from an unexpected harness")

def version(package):
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None

print(json.dumps({
    "python": platform.python_version(),
    "machine": platform.machine(),
    "torch": torch.__version__,
    "triton": version("triton"),
    "triton_module_version": triton.__version__,
    "cuda": torch.version.cuda,
    "cudnn": version("nvidia-cudnn-cu13") or version("nvidia-cudnn-cu12"),
    "cudnn_runtime": torch.backends.cudnn.version(),
    "cutlass": cutlass.__version__,
    "cutlass_distribution_version": version("nvidia-cutlass-dsl"),
    "cute_library_versions": {
        package: version(package) for package in (
            "nvidia-cutlass-dsl-libs-base",
            "nvidia-cutlass-dsl-libs-core",
            "nvidia-cutlass-dsl-libs-cu13",
        )
    },
    "cute_runtime_path": os.environ.get("CUTE_DSL_LIBS"),
    "cute_runtime_exists": Path(os.environ["CUTE_DSL_LIBS"]).is_file() if "CUTE_DSL_LIBS" in os.environ else False,
    "flash-attn-4": version("flash-attn-4"),
    "quack-kernels": version("quack-kernels"),
    "cuda-python": version("cuda-python"),
    "cuda-bindings": version("cuda-bindings"),
    "cuda-core": version("cuda-core"),
    "kernel_import_validated": True,
    "module_paths": {
        "helion": helion.__file__,
        "harness": kernels.__file__,
        "torch": torch.__file__,
        "cutlass": cutlass.__file__,
        "triton": triton.__file__,
    },
}))
"""


def preflight(
    runtime: dict[str, Any], target_names: set[str], *, probe_versions: bool
) -> dict[str, Any]:
    errors = []
    checks: dict[str, Any] = {}
    for name in sorted(target_names):
        target = runtime["targets"].get(name)
        if target is None:
            errors.append(f"Runtime target missing: {name}")
            continue
        checked: dict[str, Any] = {
            "source_status": target.get("source_status"),
            "exact_code": target.get("exact_code"),
            "missing_measured_refs": target.get("missing_measured_refs", []),
        }
        checks[name] = checked
        if not target.get("software_validated"):
            errors.append(f"{name}: pinned software has not been validated")
        if (
            not target.get("source_status")
            or "exact_code" not in target
            or (not target["exact_code"] and not target.get("code_deviation"))
        ):
            errors.append(f"{name}: source status/deviation is not declared")
        if target.get("exact_code") and target.get("missing_measured_refs"):
            errors.append(
                f"{name}: exact source claimed with missing measured revisions"
            )
        python = Path(target["python"])
        if not python.is_file():
            errors.append(f"{name}: Python interpreter missing: {python}")
        for source in SOURCE_FIELDS:
            root = Path(target[f"{source}_root"])
            expected = target[f"{source}_commit"]
            if not root.is_dir():
                errors.append(f"{name}: {source} checkout missing: {root}")
                continue
            try:
                actual = command_output(["git", "-C", str(root), "rev-parse", "HEAD"])
                diff = subprocess.check_output(
                    ["git", "-C", str(root), "diff", "--binary", "HEAD"], timeout=30
                )
            except subprocess.SubprocessError as error:
                errors.append(f"{name}: cannot verify {source}: {error}")
                continue
            diff_sha = hashlib.sha256(diff).hexdigest()
            checked[source] = {"commit": actual, "diff_sha256": diff_sha}
            if not actual.startswith(expected):
                errors.append(f"{name}: {source} HEAD {actual} differs from {expected}")
            expected_diff = target.get(f"{source}_diff_sha256", EMPTY_DIFF_SHA256)
            if diff_sha != expected_diff:
                errors.append(f"{name}: {source} has unrecorded tracked changes")
        if target.get("harness_patch_sha256"):
            patch = target.get("harness_patch_path")
            if not patch or not Path(patch).is_file():
                errors.append(f"{name}: recorded harness patch file is missing")
            elif protocol.sha256(Path(patch)) != target["harness_patch_sha256"]:
                errors.append(f"{name}: recorded harness patch hash differs")
        if probe_versions and python.is_file() and target.get("software_validated"):
            try:
                versions = json.loads(
                    command_output(
                        [str(python), "-c", SOFTWARE_PROBE],
                        env=environment(target),
                        cwd=target["harness_root"],
                    )
                )
            except (subprocess.SubprocessError, json.JSONDecodeError) as error:
                errors.append(f"{name}: software probe failed: {error}")
                continue
            checked["software_versions"] = versions
            for package, expected in target["software_targets"].items():
                if versions.get(package) != expected:
                    errors.append(
                        f"{name}: {package} {versions.get(package)!r} differs from {expected!r}"
                    )
            cudnn = target["software_targets"].get("cudnn")
            if cudnn:
                major, minor, patch = (int(value) for value in cudnn.split(".")[:3])
                expected_cudnn = major * 10000 + minor * 100 + patch
                if versions["cudnn_runtime"] != expected_cudnn:
                    errors.append(f"{name}: loaded cuDNN differs from {expected_cudnn}")
            cutlass = target["software_targets"].get("cutlass")
            if cutlass:
                if not versions["cute_runtime_exists"]:
                    errors.append(f"{name}: selected CuTe runtime library is missing")
                for package, version in versions["cute_library_versions"].items():
                    if version != cutlass:
                        errors.append(
                            f"{name}: {package} {version!r} differs from {cutlass!r}"
                        )
    return {
        "checked_utc": utc_now(),
        "runtime_status": runtime["status"],
        "ready": not errors,
        "errors": errors,
        "live_software_probe": probe_versions,
        "targets": checks,
    }


def make_jobs(
    cases: list[dict[str, Any]],
    args: argparse.Namespace,
    runtime: dict[str, Any],
) -> list[dict[str, Any]]:
    jobs = []
    for gpu, gpu_cases in protocol.shard_cases(cases, args.gpus).items():
        for case in gpu_cases:
            for impl in args.impls:
                helion = impl in protocol.HELION_IMPLEMENTATIONS
                target_name = protocol.runtime_target(case, args.mode)
                target = runtime["targets"].get(target_name, {})
                config = (
                    protocol.config_file(case, impl, args.config_dir)
                    if helion and args.config_dir
                    else None
                )
                skip = None
                if (
                    args.mode == "backward"
                    and helion
                    and case["variant"] == "mla-causal"
                ):
                    skip = (
                        "Archived backward source has no MLA D192/Dv128 kernel; "
                        "latest pass7 source/configs/full metadata are unavailable"
                    )
                    config = None
                if impl == "flex-cute" and case["variant"] == "vsa":
                    skip = (
                        "Original VSA metadata has 128-token query tiles; "
                        "FlexAttention FLASH requires 256-row query blocks"
                    )
                key = f"{protocol.case_id(case)}.{impl}"
                contract = protocol.contract(case, args.mode, impl)
                measurement = dict(protocol.MEASUREMENT)
                if helion and args.config_dir:
                    measurement.update(
                        autotune_effort="pinned", autotune_budget_seconds=0
                    )
                jobs.append(
                    {
                        "key": key,
                        "case": case,
                        "mode": args.mode,
                        "impl": impl,
                        "physical_gpu": gpu,
                        "runtime_target": target_name,
                        "runtime": target,
                        "retune": bool(helion and args.retune),
                        "config": str(config.resolve()) if config else None,
                        "save_config": str(args.out / "configs"),
                        "sdpa_preparation": "paper",
                        "skip_reason": skip,
                        "result_file": str(args.out / "cells" / f"{key}.json"),
                        "record_metadata": {
                            "experiment": "paper_attention_repro",
                            "mode": "bwd" if args.mode == "backward" else "fwd",
                            "case_id": protocol.case_id(case),
                            "shape_metadata": case["shape"],
                            "parameters": case["parameters"],
                            "physical_gpu": str(gpu),
                            "runtime_target": target_name,
                            "source_status": target.get("source_status"),
                            "exact_code": target.get("exact_code"),
                            "code_deviation": target.get("code_deviation"),
                            "software_versions": target.get("software_versions"),
                            "sdpa_preparation": "paper",
                            "measurement": measurement,
                            "contract": contract,
                        },
                    }
                )
    return jobs


def hardware() -> list[dict[str, str]]:
    fields = [
        "index",
        "name",
        "uuid",
        "driver_version",
        "power.limit",
        "power.default_limit",
        "power.max_limit",
        "clocks.max.sm",
        "memory.total",
        "temperature.gpu",
    ]
    output = command_output(
        [
            "nvidia-smi",
            f"--query-gpu={','.join(fields)}",
            "--format=csv,noheader,nounits",
        ]
    )
    return [
        dict(zip(fields, (value.strip() for value in row), strict=False))
        for row in csv.reader(io.StringIO(output))
    ]


def check_gpu_idle(gpu: int, devices: list[dict[str, str]]) -> None:
    device = next((item for item in devices if item["index"] == str(gpu)), None)
    if device is None:
        raise RuntimeError(f"GPU {gpu} does not exist")
    output = command_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ]
    )
    active = [
        row[1].strip()
        for row in csv.reader(io.StringIO(output))
        if row and row[0].strip() == device["uuid"]
    ]
    if active:
        raise RuntimeError(
            f"GPU {gpu} has active compute processes: {', '.join(active)}"
        )


def summarize(
    records: list[dict[str, Any]], jobs: list[dict[str, Any]]
) -> dict[str, Any]:
    outcomes: dict[str, dict[str, int]] = {}
    cells: dict[str, dict[str, dict[str, Any]]] = {}
    for record in records:
        impl = record["impl"]
        status = record["accuracy"]
        counts = outcomes.setdefault(impl, {})
        counts[status] = counts.get(status, 0) + 1
        cells.setdefault(record["case_id"], {})[impl] = record
    ratios = []
    for implementations in cells.values():
        helion = implementations.get("helion-cute")
        baselines = [
            record
            for impl, record in implementations.items()
            if impl in protocol.IMPLEMENTATIONS[1:]
            and record["accuracy"] == "PASS"
            and record.get("median_ms", 0) > 0
        ]
        if helion and helion["accuracy"] == "PASS" and baselines:
            ratios.append(
                min(record["median_ms"] for record in baselines) / helion["median_ms"]
            )
    return {
        "finished_utc": utc_now(),
        "expected_records": len(jobs),
        "records": len(records),
        "complete": len(records) == len(jobs),
        "outcomes": outcomes,
        "compared_shapes": len(ratios),
        "geomean_vs_best_nonhelion": statistics.geometric_mean(ratios)
        if ratios
        else None,
        "aggregation": "PASS-only per-case fastest displayed non-Helion median / Helion median; geometric mean",
    }


def save_backward_config(record: dict[str, Any], out: Path) -> None:
    if (
        record["mode"] == "bwd"
        and record["impl"] in protocol.HELION_IMPLEMENTATIONS
        and record.get("config")
    ):
        configs = {
            name: json.loads(value) if isinstance(value, str) else value
            for name, value in record["config"].items()
        }
        path = out / "configs" / f"{record['case_id']}.{record['impl']}.json"
        path.write_text(json.dumps(configs, indent=2) + "\n")


def execute(jobs: list[dict[str, Any]], out: Path) -> int:
    devices = hardware()
    (out / "hardware_before.json").write_text(json.dumps(devices, indent=2) + "\n")
    for gpu in sorted({job["physical_gpu"] for job in jobs}):
        check_gpu_idle(gpu, devices)
    for name in ("jobs", "cells", "logs", "configs", "source"):
        (out / name).mkdir()
    for name in ("run.py", "protocol.py", "worker.py", "reference_cases.json"):
        (out / "source" / name).write_bytes((HERE / name).read_bytes())
    records: list[dict[str, Any]] = []
    lock = threading.Lock()

    def run_gpu(gpu: int) -> None:
        gpu_jobs = [job for job in jobs if job["physical_gpu"] == gpu]
        for index, job in enumerate(gpu_jobs):
            if index:
                time.sleep(
                    cast("int", protocol.MEASUREMENT["idle_between_processes_seconds"])
                )
            check_gpu_idle(gpu, devices)
            job_path = out / "jobs" / f"{job['key']}.json"
            job_path.write_text(json.dumps(job, indent=2) + "\n")
            if job["skip_reason"]:
                record = {
                    "variant": job["case"]["variant"],
                    "shape": job["case"]["label"],
                    "impl": job["impl"],
                    "accuracy": "SKIP",
                    "skip_reason": job["skip_reason"],
                    **job["record_metadata"],
                }
                Path(job["result_file"]).write_text(json.dumps(record) + "\n")
                (out / "logs" / f"{job['key']}.log").write_text(
                    job["skip_reason"] + "\n"
                )
            else:
                env = environment(job["runtime"], gpu)
                if job["impl"] in protocol.HELION_IMPLEMENTATIONS:
                    env["HELION_BACKEND"] = job["impl"].removeprefix("helion-")
                    env["HELION_AUTOTUNE_EFFORT"] = "full"
                    env["HELION_SKIP_CACHE"] = "1"
                    env["HELION_AUTOTUNE_BUDGET_SECONDS"] = "1800"
                with (out / "logs" / f"{job['key']}.log").open("w") as log:
                    try:
                        subprocess.run(
                            [
                                job["runtime"]["python"],
                                str(out / "source" / "worker.py"),
                                "--job",
                                str(job_path),
                            ],
                            env=env,
                            cwd=job["runtime"]["harness_root"],
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            check=False,
                            timeout=5400,
                        )
                    except subprocess.TimeoutExpired:
                        log.write("\nWorker exceeded the 5400-second timeout.\n")
                result = Path(job["result_file"])
                if result.exists():
                    record = protocol.read_json(result)
                else:
                    record = {
                        "variant": job["case"]["variant"],
                        "shape": job["case"]["label"],
                        "impl": job["impl"],
                        "accuracy": "ERROR",
                        "skip_reason": "Worker exited without a result; see log",
                        **job["record_metadata"],
                    }
                    result.write_text(json.dumps(record) + "\n")
            save_backward_config(record, out)
            with lock:
                records.append(record)
                with (out / "results.jsonl").open("a") as result_stream:
                    result_stream.write(json.dumps(record) + "\n")
                (out / "summary.json").write_text(
                    json.dumps(summarize(records, jobs), indent=2) + "\n"
                )
            print(f"GPU {gpu}: {job['key']} {record['accuracy']}", flush=True)

    gpus = sorted({job["physical_gpu"] for job in jobs})
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = [pool.submit(run_gpu, gpu) for gpu in gpus]
        for future in futures:
            future.result()
    (out / "hardware_after.json").write_text(json.dumps(hardware(), indent=2) + "\n")
    return int(any(record["accuracy"] in {"FAIL", "ERROR"} for record in records))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("forward", "backward"), default="forward")
    parser.add_argument(
        "--gpus", required=True, help="physical GPU indices, comma separated"
    )
    parser.add_argument("--variants", default="all")
    parser.add_argument(
        "--shapes", default="0,1,2", help="indices into the paper's three shapes"
    )
    parser.add_argument("--impls", default="all")
    parser.add_argument(
        "--out", type=Path, required=True, help="new or empty output directory"
    )
    configs = parser.add_mutually_exclusive_group()
    configs.add_argument(
        "--config-dir", type=Path, help="pinned configs; never modified"
    )
    configs.add_argument(
        "--retune", action="store_true", help="cold full 1800-second Helion searches"
    )
    parser.add_argument("--runtime", type=Path, default=HERE / "runtime.json")
    parser.add_argument(
        "--dry-run", action="store_true", help="write manifests without running kernels"
    )
    args = parser.parse_args()
    try:
        args.gpus = protocol.parse_gpus(args.gpus)
        args.impls = protocol.parse_impls(args.impls)
        args.out = args.out.resolve()
        profile_path = HERE / "reference_cases.json"
        profile = protocol.read_json(profile_path)
        cases = protocol.select_cases(profile, args.variants, args.shapes)
        runtime, runtime_path = load_runtime(args.runtime)
        if args.config_dir:
            args.config_dir = args.config_dir.resolve()
        if not args.dry_run and set(args.impls) & protocol.HELION_IMPLEMENTATIONS:
            if not args.retune and not args.config_dir:
                raise ValueError("Select --retune or --config-dir for Helion runs")
        protected = [HERE]
        if args.config_dir:
            protected.append(args.config_dir)
        for target in runtime["targets"].values():
            protected.extend(Path(target[f"{source}_root"]) for source in SOURCE_FIELDS)
        protocol.protected_output(args.out, protected)
        jobs = make_jobs(cases, args, runtime)
        checks = preflight(
            runtime,
            {job["runtime_target"] for job in jobs},
            probe_versions=not args.dry_run,
        )
        for job in jobs:
            observed = checks["targets"].get(job["runtime_target"], {})
            if "software_versions" in observed:
                job["record_metadata"]["software_versions"] = observed[
                    "software_versions"
                ]
            if job["config"] and not Path(job["config"]).is_file():
                checks["errors"].append(f"Pinned config missing: {job['config']}")
                checks["ready"] = False
    except (ValueError, KeyError, FileNotFoundError) as error:
        parser.error(str(error))
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "mode": args.mode,
        "dry_run": args.dry_run,
        "command": sys.argv,
        "reference_profile": str(profile_path),
        "reference_profile_sha256": protocol.sha256(profile_path),
        "runner_source_sha256": {
            name: protocol.sha256(HERE / name)
            for name in ("run.py", "protocol.py", "worker.py")
        },
        "paper_forward_commit": profile["paper_forward_commit"],
        "runtime_manifest": str(runtime_path),
        "runtime_manifest_sha256": protocol.sha256(runtime_path)
        if runtime_path.exists()
        else None,
        "measurement": dict(protocol.MEASUREMENT),
        "sdpa_preparation": "paper",
        "preparation_scope": "Layouts, BlockMask lists and closure tables prepared outside timing; SDPA mask/bias placement follows the paper",
        "case_count": len(cases),
        "expected_records": len(jobs),
        "cases": cases,
        "jobs": jobs,
        "preflight": checks,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.out / "preflight.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(
        f"{len(cases)} cases, {len(jobs)} cells: {args.out / 'manifest.json'}",
        flush=True,
    )
    if args.dry_run:
        print(f"Runtime ready: {checks['ready']}; no kernels run", flush=True)
        return
    if not checks["ready"]:
        raise SystemExit("Runtime preflight failed:\n" + "\n".join(checks["errors"]))
    raise SystemExit(execute(jobs, args.out))


if __name__ == "__main__":
    main()
