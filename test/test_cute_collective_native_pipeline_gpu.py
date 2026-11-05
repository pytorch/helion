"""Opt-in physical-GPU4 compile/numerics regression; never runs via collection."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if root.name != "helion-pipeline-candidate":
        raise RuntimeError(
            "Component validation must use the isolated pipeline candidate"
        )
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "4":
        raise RuntimeError("This component probe is authorized only on physical GPU4")
    experiment = root.parents[2]
    scripts = experiment / "corpus/scripts"
    sys.path.insert(0, str(scripts))
    from gpu_lock import gpu_lock

    names = (
        "helion/_compiler/cute/collective_matmul.py",
        "helion/_compiler/cute/collective_tcgen05.py",
    )
    before = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names
    }
    out = args.out.resolve()
    if not out.is_relative_to(root.parent) or out.is_relative_to(root):
        raise RuntimeError(
            "Probe evidence must be a new sibling directory under the artifact run"
        )
    out.mkdir(parents=True, exist_ok=False)
    os.environ.update(
        PYTHONDONTWRITEBYTECODE="1",
        TORCHINDUCTOR_CACHE_DIR=str(out / "cache/inductor"),
        HELION_CACHE_DIR=str(out / "cache/helion"),
        CUTE_DSL_CACHE_DIR=str(out / "cache/cute"),
        TRITON_CACHE_DIR=str(out / "cache/triton"),
        CUDA_CACHE_PATH=str(out / "cache/cuda-driver"),
        HELION_FORCE_AUTOTUNE="0",
        HELION_DISALLOW_AUTOTUNING="1",
        HELION_SKIP_CACHE="1",
        HELION_AUTOTUNE_RANDOM_SEED="0",
        TRITON_F32_DEFAULT="tf32",
    )
    report = {
        "status": "running",
        "physical_gpu": 4,
        "source_root": str(root),
        "source_hashes": before,
        "started_at": time.time(),
        "performance_claim": False,
    }
    try:
        with gpu_lock(
            4, label="native-pipeline-v3-component-correctness", timeout_s=600
        ):
            query = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--id=4",
                    "--query-gpu=index,uuid,name,power.limit,memory.used,utilization.gpu",
                    "--format=csv,noheader",
                ],
                text=True,
            ).strip()
            uuid = query.split(",")[1].strip()
            apps = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-compute-apps=gpu_uuid,pid,process_name",
                    "--format=csv,noheader",
                ],
                text=True,
            )
            if any(line.split(",")[0].strip() == uuid for line in apps.splitlines()):
                raise RuntimeError(
                    "GPU4 has a compute process despite exclusive harness lock"
                )
            report["gpu_before"] = query
            import torch

            from test.test_cute_collective_native_pipeline_codegen import (
                _computed_pipeline,
            )
            from test.test_cute_collective_native_pipeline_codegen import _config

            import helion

            if not Path(helion.__file__).resolve().is_relative_to(root / "helion"):
                raise RuntimeError("Wrong Helion source imported")
            if (
                not Path(torch.__file__)
                .resolve()
                .is_relative_to(experiment / "pytorch/torch")
            ):
                raise RuntimeError("Wrong PyTorch source imported")
            if torch.version.cuda is None:
                version = torch._C._cuda_getCompiledVersion()
                torch.version.cuda = f"{version // 1000}.{version % 1000 // 10}"
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
            records = []
            for dtype in (torch.float16, torch.bfloat16):
                torch.manual_seed(0)
                a = torch.randn((256, 768), dtype=dtype, device="cuda")
                b = torch.randn((384, 768), dtype=dtype, device="cuda").t()
                bias = torch.randn((384,), dtype=dtype, device="cuda")
                residual = torch.randn((256, 384), dtype=dtype, device="cuda")
                output = torch.empty_like(residual)
                values = (a, b, bias, residual, output)
                bound = _computed_pipeline.bind(values)
                config = _config(2)
                bound.set_config(config)
                path = bound.get_cached_path()
                if not path or "_b_current" not in Path(path).read_text():
                    raise RuntimeError(
                        "Expected real stage-two generated containers are absent"
                    )
                _computed_pipeline(*values)
                torch.cuda.synchronize()
                expected = (
                    torch.relu(a.float() @ b.float() + bias.float())
                    + residual.float() * 0.5
                ).to(dtype)
                torch.testing.assert_close(output, expected, atol=0.05, rtol=0.01)
                for _ in range(3):
                    _computed_pipeline(*values)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    _computed_pipeline(*values)
                graph.replay()
                torch.cuda.synchronize()
                torch.testing.assert_close(output, expected, atol=0.05, rtol=0.01)
                destination = out / f"generated_{str(dtype).removeprefix('torch.')}.py"
                destination.write_bytes(Path(path).read_bytes())
                records.append(
                    {
                        "dtype": str(dtype),
                        "status": "pass",
                        "compiled": True,
                        "captured": True,
                        "config": dict(bound._require_implicit_config()),
                        "policy": bound.config_envelope(
                            bound._require_implicit_config()
                        ).to_dict(),
                        "source": str(destination),
                        "atol": 0.05,
                        "rtol": 0.01,
                    }
                )
            report.update(status="pass", tests=records)
            report["gpu_after"] = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--id=4",
                    "--query-gpu=index,uuid,name,power.limit,memory.used,utilization.gpu",
                    "--format=csv,noheader",
                ],
                text=True,
            ).strip()
        after = {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in names
        }
        if before != after:
            raise RuntimeError("Candidate sources changed during component test")
        return 0
    except Exception as error:
        report.update(status="error", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report["finished_at"] = time.time()
        (out / "result.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
