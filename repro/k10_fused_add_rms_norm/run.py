"""Run captured K10 configs through public APIs and the original 33 checks."""

from __future__ import annotations

import argparse
from datetime import datetime
from datetime import timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=int, choices=range(12), required=True)
    parser.add_argument("--fixed", action="store_true", required=True)
    parser.add_argument("--core", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify-frozen-source", action="store_true")
    parser.add_argument("--gpu-uuid")
    parser.add_argument("--physical-gpu", type=int, choices=(0, 1))
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    sources = json.loads((here / "provenance/source_pins.json").read_text())
    if args.core is not None:
        core = args.core.resolve(strict=True)
    elif args.verify_frozen_source:
        core = Path(sources["preferred_core"]["runtime"]).resolve(strict=True)
    else:
        core = None
    if (
        args.gpu_uuid is not None
        and os.environ.get("CUDA_VISIBLE_DEVICES") != args.gpu_uuid
    ):
        raise ValueError("The runner must use the controller's exact GPU UUID")
    manifest = json.loads((here / "manifest_v1.json").read_text())

    def source_check() -> dict:
        selected = next(
            item
            for item in (sources["preferred_core"], sources["baseline_main"])
            if Path(item["runtime"]).resolve() == core
        )
        source_path = Path(selected["manifest"]["path"])
        if (
            hashlib.sha256(source_path.read_bytes()).hexdigest()
            != selected["manifest"]["sha256"]
        ):
            raise AssertionError("The frozen source manifest changed")
        source_files = json.loads(source_path.read_text())
        for name, expected in source_files["files"].items():
            if hashlib.sha256((core / name).read_bytes()).hexdigest() != expected:
                raise AssertionError("The frozen core changed: " + name)
        for name, target in source_files["symlinks"].items():
            if os.readlink(core / name) != target:
                raise AssertionError("A frozen source alias changed: " + name)
        for name, expected in manifest["files"].items():
            if hashlib.sha256((here / name).read_bytes()).hexdigest() != expected:
                raise AssertionError("The sealed reproducer changed: " + name)
        return {
            "core_manifest": selected["manifest"],
            "source_files_verified": source_files["file_count"],
            "reproducer_files_verified": len(manifest["files"]),
            "all_source_bytes_and_symlinks_preserved": True,
        }

    def portable_source_check() -> dict:
        for name, expected in manifest["files"].items():
            if hashlib.sha256((here / name).read_bytes()).hexdigest() != expected:
                raise AssertionError("The sealed reproducer changed: " + name)
        return {
            "source_identity": "UNVERIFIED_PORTABLE",
            "requested_core": str(args.core.resolve())
            if args.core is not None
            else "installed_helion",
            "frozen_core_verified": False,
            "reproducer_files_verified": len(manifest["files"]),
            "all_payload_files_preserved": True,
        }

    source_before = (
        source_check() if args.verify_frozen_source else portable_source_check()
    )
    sys.path[:0] = ([str(core)] if core is not None else []) + [str(here)]
    os.environ["HELION_BACKEND"] = "cute"
    os.environ.pop("HELION_AUTOTUNE_EFFORT", None)

    from fused_add_rms_norm import fused_add_rms_norm
    from inputs import Shape
    from inputs import make_case
    import torch
    from validation import reference
    from validation import validate_output

    import helion

    imported_core = Path(helion.__file__).resolve().parent.parent
    if (
        core is not None
        and Path(helion.__file__).resolve() != core / "helion/__init__.py"
    ):
        raise RuntimeError("Imported Helion differs from the explicitly selected core")
    core = imported_core
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    records = json.loads((here / "cases.json").read_text())["cases"]
    record = next(item for item in records if item["case_index"] == args.case)
    config = json.loads((here / record["config"]).read_text())
    winner = json.loads((here / record["original_winner"]).read_text())
    if config != winner["config"]:
        raise ValueError("The fixed config differs from its captured selected winner")
    case = make_case(Shape(**record["shape"]), seed=record["input_seed"])
    tensors = case.tensor_inputs()
    inputs = (*case.kernel_inputs(), case.eps, case.normalize_rounded_residual)

    def input_hashes() -> dict:
        records = {}
        for name, tensor in tensors.items():
            data = (
                tensor.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy()
            )
            records[name] = {
                "shape": list(tensor.shape),
                "stride": list(tensor.stride()),
                "dtype": str(tensor.dtype),
                "storage_bytes": tensor.untyped_storage().nbytes(),
                "sha256": hashlib.sha256(memoryview(data)).hexdigest(),
            }
        return records

    original_input_sources = input_hashes()
    original_input_sha = hashlib.sha256(
        json.dumps(original_input_sources, sort_keys=True).encode()
    ).hexdigest()
    if original_input_sha != record["input_sha256"]:
        raise AssertionError("The canonical input bytes differ from the archived case")
    kernel = helion.kernel(
        fused_add_rms_norm,
        static_shapes=True,
        backend="cute",
        config=helion.Config.from_dict(config),
        cute_structural_policy=helion.CuteStructuralPolicy.from_dict(
            winner["cute_structural_policy"]
        ),
    )

    def snapshot() -> dict:
        return {
            name: tensor.view(torch.uint8).clone() for name, tensor in tensors.items()
        }

    def restore(saved: dict) -> None:
        for name, tensor in tensors.items():
            tensor.view(torch.uint8).copy_(saved[name])

    checks = []

    def check(
        result: tuple[torch.Tensor, torch.Tensor],
        expected: tuple[torch.Tensor, torch.Tensor],
        saved: dict,
        phase: str,
    ) -> None:
        metrics = validate_output(case, result, expected)
        if metrics["l2_limit"] != record["precision_l2_limit"]:
            raise AssertionError("The original global/per-row precision limit changed")
        if not torch.equal(result[1].view(torch.uint8), expected[1].view(torch.uint8)):
            raise AssertionError(
                "Residual storage must equal exact input-dtype cast bits"
            )
        for name, tensor in tensors.items():
            if not torch.equal(tensor.view(torch.uint8), saved[name]):
                raise AssertionError("Input storage changed: " + name)
        checks.append(
            {
                "phase": phase,
                "accuracy": metrics,
                "inputs_immutable_bitwise": True,
                "stored_residual_exact_bitwise": True,
            }
        )
        if args.output is not None:
            partial = args.output.with_suffix(".partial.json")
            partial.write_text(
                json.dumps(
                    {
                        "status": "RUNNING",
                        "case_index": args.case,
                        "numerical_records": len(checks),
                        "checks": checks,
                    },
                    indent=2,
                )
                + "\n"
            )

    def calls(
        expected: tuple[torch.Tensor, torch.Tensor],
        saved: dict,
        phase: str,
        count: int = 3,
    ) -> None:
        for _index in range(count):
            check(kernel(*inputs), expected, saved, phase)

    saved = snapshot()
    expected = reference(case)
    calls(expected, saved, "correctness_eager")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = kernel(*inputs)

    def replays(
        expected: tuple[torch.Tensor, torch.Tensor],
        saved: dict,
        phase: str,
        count: int = 3,
    ) -> None:
        for _index in range(count):
            for tensor in graph_output:
                tensor.fill_(float("nan"))
            graph.replay()
            check(graph_output, expected, saved, phase)

    replays(expected, saved, "correctness_graph")
    for name, tensor in zip(
        ("x", "residual", "weight"), case.kernel_inputs(), strict=True
    ):
        try:
            tensor.neg_()
            changed_saved = snapshot()
            changed_expected = reference(case)
            calls(changed_expected, changed_saved, "changed_" + name + "_eager")
            replays(changed_expected, changed_saved, "changed_" + name + "_graph")
        finally:
            restore(saved)
        replays(expected, saved, "restored_" + name + "_graph", count=1)
    replays(expected, saved, "correctness_restored_graph_inputs")
    calls(expected, saved, "correctness_after_validation")
    if len(checks) != 33:
        raise AssertionError("Expected all 33 original eager/graph/mutation checks")
    final_input_sources = input_hashes()
    if final_input_sources != original_input_sources:
        raise AssertionError("Final input storage differs from canonical storage")
    source_after = (
        source_check() if args.verify_frozen_source else portable_source_check()
    )
    result = {
        "status": "PASS",
        "classification": "FINAL_STANDALONE_PUBLIC_KERNEL_FIXED_CONFIG_CORRECTNESS_ONLY",
        "utc": datetime.now(timezone.utc).isoformat(),
        "case_index": args.case,
        "shape": record["shape"],
        "core": str(core),
        "config": record["config"],
        "selected_origin": record["selected_origin"],
        "autotuning": "Explicit public config; no search requested",
        "precision_l2_limit": record["precision_l2_limit"],
        "numerical_records": len(checks),
        "checks": checks,
        "canonical_archived_program": record["accepted_generated_source"],
        "backend_requirement": record["backend_requirement"],
        "performance_measurement": "NOT_PERFORMED",
        "input_sha256": original_input_sha,
        "final_input_sha256": original_input_sha,
        "input_sources": original_input_sources,
        "final_input_sources": final_input_sources,
        "gpu_uuid": args.gpu_uuid or os.environ.get("CUDA_VISIBLE_DEVICES"),
        "physical_gpu": args.physical_gpu,
        "logical_cuda_device": 0,
        "device_capability": list(torch.cuda.get_device_capability(0)),
        "source_before": source_before,
        "source_after": source_after,
        "source_preserved": args.verify_frozen_source and source_before == source_after,
        "payload_preserved": source_before == source_after,
        "source_identity": "VERIFIED_FROZEN"
        if args.verify_frozen_source
        else "UNVERIFIED_PORTABLE",
        "verify_frozen_source": args.verify_frozen_source,
        "imported_helion_source": str(Path(helion.__file__).resolve()),
        "runner_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "reproducer_manifest_sha256": hashlib.sha256(
            (here / "manifest_v1.json").read_bytes()
        ).hexdigest(),
        "software": {
            name: importlib.metadata.version(name)
            for name in ("torch", "triton", "nvidia-cutlass-dsl")
        },
        "actual_argv": sys.argv,
        "python": sys.executable,
        "environment": {
            name: os.environ.get(name)
            for name in (
                "CUDA_VISIBLE_DEVICES",
                "HELION_BACKEND",
                "HELION_AUTOTUNE_EFFORT",
                "HELION_AUTOTUNE_PRECOMPILE_JOBS",
                "TORCHINDUCTOR_COMPILE_THREADS",
            )
        },
    }
    if args.output is not None:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
