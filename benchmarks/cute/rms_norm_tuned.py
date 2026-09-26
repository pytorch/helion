"""[no land] Replay the BF16 RMSNorm hillclimb's 13 winning GB300 configs.

The kernel body is copied from benchmark_rms_norm's
helion_kernel_library/_kernels/rms_norm/rms_norm.py. Explicit compilation
uses the saved configuration and the AOT kernel's static_shapes=False setting.

From the repository root, with Helion and CuTe installed:

    CUDA_VISIBLE_DEVICES=3 python benchmarks/cute/rms_norm_tuned.py --shape 4096x7168
    CUDA_VISIBLE_DEVICES=3 python benchmarks/cute/rms_norm_tuned.py --all --benchmark

Correctness checks only need Helion/PyTorch/CuTe. Timing additionally needs
FlashInfer and CUPTI >= 13, as in repro/breadth/compare_rms_norm.py. The timing
path uses CUDA graphs and cold L2, with no CUDA-event fallback. Saved baseline
ratios describe the recorded 1400 W GB300 run; hardware and clocks affect replay.
This file and its JSON are review artifacts, not production config selection.
"""

from __future__ import annotations

import argparse
import functools
import json
from pathlib import Path
import statistics
import warnings

import torch

import helion
import helion.language as hl


@helion.kernel(backend="cute", static_shapes=False)
def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        acc = x[tile_m, :].to(torch.float32)
        variance = torch.mean(acc * acc, dim=-1)
        inv_rms = torch.rsqrt(variance + eps)
        out[tile_m, :] = (acc * inv_rms[:, None] * weight[:].to(torch.float32)).to(
            x.dtype
        )
    return out


def main() -> None:
    saved = json.loads(
        Path(__file__).with_name("rms_norm_tuned_configs.json").read_text()
    )
    parser = argparse.ArgumentParser(description=__doc__)
    shapes = parser.add_mutually_exclusive_group(required=True)
    shapes.add_argument("--shape", choices=list(saved["configs"]))
    shapes.add_argument("--all", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    results = []
    for shape in saved["configs"] if args.all else [args.shape]:
        entry = saved["configs"][shape]
        m, n = map(int, shape.split("x"))
        torch.manual_seed(0)
        x = torch.randn((m, n), dtype=torch.bfloat16, device="cuda")
        weight = torch.randn(n, dtype=x.dtype, device=x.device)
        inputs = (x, weight, saved["eps"])
        compiled = rms_norm.bind(inputs).compile_config(
            helion.Config(**entry["config"])
        )
        actual = compiled(*inputs)
        expected = torch.nn.functional.rms_norm(x, (n,), weight, eps=saved["eps"])
        torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)
        result = {"shape": shape, "correctness": "PASS", "config": entry["config"]}
        if args.benchmark:
            # Optional benchmark dependency; correctness-only use stays standalone.
            from flashinfer.testing.utils import bench_gpu_time

            run_medians = []
            for _ in range(5):
                with warnings.catch_warnings():
                    warnings.filterwarnings("error", message=".*Falling back to CUDA.*")
                    samples = bench_gpu_time(
                        functools.partial(compiled, *inputs),
                        enable_cupti=True,
                        use_cuda_graph=True,
                        cold_l2_cache=True,
                        dry_run_time_ms=100,
                        repeat_time_ms=300,
                    )
                run_medians.append(statistics.median(samples))
            median_ms = statistics.median(run_medians)
            result.update(
                median_ms=median_ms,
                runs_ms=run_medians,
                baseline_impl=entry["baseline_impl"],
                baseline_ms=entry["baseline_ms"],
                baseline_ratio=entry["baseline_ms"] / median_ms,
                benchmark_timer="cupti",
                use_cuda_graph=True,
                cold_l2_cache=True,
            )
        results.append(result)
        print(json.dumps(result), flush=True)
    if args.output is not None:
        args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
