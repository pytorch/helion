"""Run one archived harness cell in its isolated Python environment."""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import traceback
from typing import Any
from typing import cast


def check_execution_path(result: dict[str, Any], required: bool) -> None:
    if (
        required
        and result["accuracy"] == "PASS"
        and result.get("flash_fired") is not True
    ):
        raise RuntimeError(
            "The paper requires the fused CuTe flash path; "
            f"flash_fired={result.get('flash_fired')}, notes={result.get('notes')}"
        )


def run_job(job: dict[str, Any]) -> dict[str, Any]:
    target = job["runtime"]
    compare = importlib.import_module("benchmarks.helion_attn.compare")
    helion = importlib.import_module("helion")
    imported_root = Path(cast("str", helion.__file__)).resolve().parents[1]
    if imported_root != Path(target["helion_root"]).resolve():
        raise RuntimeError(f"Imported Helion from unexpected checkout: {imported_root}")
    harness_file = Path(cast("str", compare.__file__)).resolve()
    if Path(target["harness_root"]).resolve() not in harness_file.parents:
        raise RuntimeError(f"Imported harness from unexpected checkout: {harness_file}")
    case = job["case"]
    variant = compare.V.VARIANTS[case["variant"]]
    shape = variant.shapes[case["shape_index"]]
    actual = {
        "b": shape.b,
        "hq": shape.hq,
        "hkv": shape.hkv,
        "m": shape.m,
        "n": shape.n,
        "d": shape.d,
        "dv": shape.d_v,
        "dtype": str(variant.dtype).replace("torch.", ""),
    }
    parameters = json.loads(json.dumps(shape.params))
    if actual != case["shape"] or parameters != case["parameters"]:
        raise RuntimeError(
            f"Archived harness disagrees with reference case: {case['label']}"
        )
    args = argparse.Namespace(
        variant=case["variant"],
        shape=case["shape_index"],
        impl=job["impl"],
        mode="bwd" if job["mode"] == "backward" else "fwd",
        num_runs=5,
        warmup_ms=500,
        rep_ms=500,
        seed=0,
        force_autotune=int(job["retune"]),
        skip_correctness=0,
        config=job["config"],
        save_config=job["save_config"],
        flex_block_size=None,
    )
    result = compare.run_single(args)
    check_execution_path(
        result, job["record_metadata"]["contract"]["require_fused_flash"]
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    args = parser.parse_args()
    job = json.loads(args.job.read_text())
    if os.environ["CUDA_VISIBLE_DEVICES"] != str(job["physical_gpu"]):
        raise RuntimeError("Worker GPU assignment disagrees with its job manifest")
    try:
        result = run_job(job)
    except Exception as error:
        traceback.print_exc()
        result = {
            "variant": job["case"]["variant"],
            "shape": job["case"]["label"],
            "impl": job["impl"],
            "accuracy": "ERROR",
            "skip_reason": f"{type(error).__name__}: {error}",
        }
    result.update(job["record_metadata"])
    Path(job["result_file"]).write_text(json.dumps(result) + "\n")
    print(json.dumps(result), flush=True)
    if result["accuracy"] in {"ERROR", "FAIL"}:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
