"""Diagnose the intermittent MI350X layer_norm accuracy failure."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import platform
import shutil
import time
from typing import TYPE_CHECKING
from typing import TypedDict

import examples.layer_norm as layer_norm
import torch
import torch.nn.functional as F
import triton

import helion

if TYPE_CHECKING:
    from collections.abc import Callable

    from helion.runtime.kernel import BoundKernel

M = 4096
N = 2560
EPS = 1e-5
ATOL = 1e-2
RTOL = 1e-2

EXACT_CONFIG: dict[str, object] = {
    "atomic_indexing": [],
    "block_sizes": [4],
    "indexing": [
        "pointer",
        "pointer",
        "pointer",
        "block_ptr",
        "pointer",
        "pointer",
        "block_ptr",
        "pointer",
        "block_ptr",
        "pointer",
        "pointer",
        "block_ptr",
    ],
    "load_cache_modifiers": [".cg", "", ".cg", "", "", ".cg", ".cg", ".cg"],
    "load_eviction_policies": ["", "", "", "", "", "", "", ""],
    "matrix_instr_nonkdim": 0,
    "num_stages": 1,
    "num_warps": 2,
    "pid_type": "flat",
    "range_flattens": [None],
    "range_multi_buffers": [None],
    "range_num_stages": [0],
    "range_unroll_factors": [0],
    "range_warp_specializes": [],
    "reduction_loops": [None],
    "store_cache_modifiers": [".wt", "", "", ""],
    "waves_per_eu": 2,
    "xcd_remap": True,
}

VARIANTS = (
    "exact",
    "no_xcd_remap",
    "no_load_cache_modifiers",
    "no_store_cache_modifiers",
    "waves_per_eu_1",
    "block_size_8",
)


class TensorComparison(TypedDict):
    bad_count: int
    nan_count: int
    inf_count: int
    max_abs: float
    max_abs_location: list[int]
    bad_row_count: int
    first_bad_rows: list[int]
    top_bad_rows: list[dict[str, int]]


class PhaseResult(TypedDict):
    launches: int
    failure_count: int
    worst_bad_count: int
    first_failures: list[dict[str, object]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("fixed", "autotune"), required=True)
    parser.add_argument("--variant", choices=VARIANTS, default="exact")
    parser.add_argument("--iterations", type=int, default=250)
    parser.add_argument("--graph-repeats", type=int, default=128)
    parser.add_argument("--graph-replays", type=int, default=10)
    parser.add_argument("--skip-graph", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fail-on-mismatch", action="store_true")
    return parser.parse_args()


def make_config(variant: str) -> helion.Config:
    values = copy.deepcopy(EXACT_CONFIG)
    if variant == "no_xcd_remap":
        values["xcd_remap"] = False
    elif variant == "no_load_cache_modifiers":
        values["load_cache_modifiers"] = [""] * 8
    elif variant == "no_store_cache_modifiers":
        values["store_cache_modifiers"] = [""] * 4
    elif variant == "waves_per_eu_1":
        values["waves_per_eu"] = 1
    elif variant == "block_size_8":
        values["block_sizes"] = [8]
    return helion.Config.from_dict(values)


def make_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(1337)
    for n in (1024, 1536, 2048, N):
        x = -2.3 + 0.5 * torch.randn((M, n), dtype=torch.float32, device="cuda")
        x.requires_grad_()
        weight = torch.rand(
            (n,), dtype=torch.float32, device="cuda", requires_grad=True
        )
        bias = torch.rand((n,), dtype=torch.float32, device="cuda", requires_grad=True)
    return x, weight, bias


def tensor_comparison(actual: torch.Tensor, expected: torch.Tensor) -> TensorComparison:
    close = torch.isclose(actual, expected, atol=ATOL, rtol=RTOL)
    bad = ~close
    bad_count = int(bad.sum().item())
    diff = (actual - expected).abs()
    finite_diff = torch.nan_to_num(diff, nan=-1.0, posinf=-1.0, neginf=-1.0)
    flat_index = int(finite_diff.argmax().item())
    max_abs = float(finite_diff.flatten()[flat_index].item())
    if actual.ndim == 2:
        location = list(divmod(flat_index, actual.shape[1]))
    else:
        location = [flat_index]
    result: TensorComparison = {
        "bad_count": bad_count,
        "nan_count": int(actual.isnan().sum().item()),
        "inf_count": int(actual.isinf().sum().item()),
        "max_abs": max_abs,
        "max_abs_location": location,
        "bad_row_count": 0,
        "first_bad_rows": [],
        "top_bad_rows": [],
    }
    if actual.ndim == 2 and bad_count:
        row_counts = bad.sum(dim=1)
        bad_rows = torch.nonzero(row_counts, as_tuple=False).flatten()
        top_counts, top_rows = torch.topk(row_counts, min(32, bad_rows.numel()))
        result["bad_row_count"] = int(bad_rows.numel())
        result["first_bad_rows"] = [int(row) for row in bad_rows[:128].cpu().tolist()]
        result["top_bad_rows"] = [
            {"row": int(row), "bad_count": int(count)}
            for row, count in zip(
                top_rows.cpu().tolist(), top_counts.cpu().tolist(), strict=True
            )
        ]
    return result


def record_failure(
    failures: list[dict[str, object]],
    *,
    phase: str,
    iteration: int,
    comparisons: dict[str, TensorComparison],
) -> None:
    if all(result["bad_count"] == 0 for result in comparisons.values()):
        return
    failure: dict[str, object] = {
        "phase": phase,
        "iteration": iteration,
        "comparisons": comparisons,
    }
    if len(failures) < 32:
        failures.append(failure)
    print(f"FAIL {json.dumps(failure, sort_keys=True)}", flush=True)


def compare_raw_outputs(
    out: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    references: dict[str, torch.Tensor],
) -> dict[str, TensorComparison]:
    reconstructed = (references["x"] - mean[:, None]) * rstd[:, None] * references[
        "weight"
    ] + references["bias"]
    return {
        "out_vs_torch": tensor_comparison(out, references["out"]),
        "mean": tensor_comparison(mean, references["mean"]),
        "rstd": tensor_comparison(rstd, references["rstd"]),
        "out_vs_returned_stats": tensor_comparison(out, reconstructed),
    }


def run_raw_stress(
    iterations: int,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    references: dict[str, torch.Tensor],
) -> PhaseResult:
    failures: list[dict[str, object]] = []
    failure_count = 0
    worst_bad_count = 0
    for iteration in range(iterations):
        out, mean, rstd = layer_norm.layer_norm_fwd(x, [N], weight, bias, EPS)
        torch.cuda.synchronize()
        comparisons = compare_raw_outputs(out, mean, rstd, references)
        launch_bad = max(item["bad_count"] for item in comparisons.values())
        if launch_bad:
            failure_count += 1
            worst_bad_count = max(worst_bad_count, launch_bad)
            record_failure(
                failures,
                phase="raw_direct",
                iteration=iteration,
                comparisons=comparisons,
            )
    return {
        "launches": iterations,
        "failure_count": failure_count,
        "worst_bad_count": worst_bad_count,
        "first_failures": failures,
    }


def run_raw_poison_stress(
    iterations: int,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    references: dict[str, torch.Tensor],
) -> PhaseResult:
    failures: list[dict[str, object]] = []
    failure_count = 0
    worst_bad_count = 0
    for iteration in range(iterations):
        probe = layer_norm.layer_norm_fwd(x, [N], weight, bias, EPS)
        probe[0].data.fill_(float("nan"))
        del probe
        out, mean, rstd = layer_norm.layer_norm_fwd(x, [N], weight, bias, EPS)
        torch.cuda.synchronize()
        comparisons = compare_raw_outputs(out, mean, rstd, references)
        launch_bad = max(item["bad_count"] for item in comparisons.values())
        if launch_bad:
            failure_count += 1
            worst_bad_count = max(worst_bad_count, launch_bad)
            record_failure(
                failures,
                phase="raw_poison_after_graph",
                iteration=iteration,
                comparisons=comparisons,
            )
    return {
        "launches": iterations,
        "failure_count": failure_count,
        "worst_bad_count": worst_bad_count,
        "first_failures": failures,
    }


def graph_benchmark(
    fn: Callable[[], object], *, graph_repeats: int, graph_replays: int
) -> dict[str, object]:
    # pyrefly: ignore [missing-attribute]
    cache = triton.runtime.driver.active.get_empty_cache_for_benchmark()
    stream = torch.cuda.Stream()
    start = time.perf_counter()
    with torch.cuda.stream(stream):
        cache.zero_()
        fn()
        for _ in range(5):
            cache.zero_()
            fn()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(graph_repeats):
                cache.zero_()
                fn()
        torch.cuda.synchronize()
        for _ in range(graph_replays):
            graph.replay()
            torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return {
        "captured_calls": graph_repeats,
        "replays": graph_replays,
        "elapsed_seconds": elapsed,
    }


def run_tritonbench_accuracy_stress(
    iterations: int,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> PhaseResult:
    failures: list[dict[str, object]] = []
    failure_count = 0
    worst_bad_count = 0
    for iteration in range(iterations):
        probe = layer_norm.layer_norm(x, [N], weight, bias, EPS)
        probe.data.fill_(float("nan"))
        del probe
        out = layer_norm.layer_norm(x, [N], weight, bias, EPS)
        baseline = F.layer_norm(x, [N], weight, bias, EPS)
        comparison = tensor_comparison(out, baseline)
        bad_count = int(comparison["bad_count"])
        if bad_count:
            failure_count += 1
            worst_bad_count = max(worst_bad_count, bad_count)
            record_failure(
                failures,
                phase="tritonbench_poison_after_graph",
                iteration=iteration,
                comparisons={"out_vs_torch": comparison},
            )
    return {
        "launches": iterations,
        "failure_count": failure_count,
        "worst_bad_count": worst_bad_count,
        "first_failures": failures,
    }


def archive_compiler_artifacts(
    output_dir: Path,
    bound: BoundKernel[object],
    config: helion.Config,
) -> dict[str, object]:
    generated_path_value = bound.get_cached_path(config)
    backend_key = bound.backend_cache_key(config)
    result: dict[str, object] = {
        "backend_cache_key": backend_key,
        "generated_path": generated_path_value,
    }
    artifact_dir = output_dir / "compiler-artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    if generated_path_value is not None:
        generated_path = Path(generated_path_value)
        shutil.copy2(generated_path, artifact_dir / generated_path.name)
    if backend_key is not None:
        cache_root_value = os.environ.get("HELION_CACHE_DIR")
        if cache_root_value is not None:
            matches = [
                path
                for path in Path(cache_root_value).rglob(backend_key)
                if path.is_dir()
            ]
            if matches:
                shutil.copytree(
                    matches[0],
                    artifact_dir / backend_key,
                    dirs_exist_ok=True,
                )
                result["backend_cache_path"] = str(matches[0])
    return result


def environment_info() -> dict[str, object]:
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    return {
        "helion_file": helion.__file__,
        "torch_version": torch.__version__,
        "torch_git_version": torch.version.git_version,
        "torch_hip_version": torch.version.hip,
        "triton_version": triton.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device_count": torch.cuda.device_count(),
        "current_device": torch.cuda.current_device(),
        "device_name": torch.cuda.get_device_name(),
        "device_properties": str(properties),
        "visible_devices": {
            name: os.environ.get(name)
            for name in (
                "CUDA_VISIBLE_DEVICES",
                "HIP_VISIBLE_DEVICES",
                "ROCR_VISIBLE_DEVICES",
            )
        },
    }


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    x, weight, bias = make_inputs()
    references = {
        "x": x.detach().clone(),
        "weight": weight.detach().clone(),
        "bias": bias.detach().clone(),
    }
    references["mean"] = references["x"].mean(dim=-1)
    centered = references["x"] - references["mean"][:, None]
    references["rstd"] = torch.rsqrt((centered * centered).mean(dim=-1) + EPS)
    references["out"] = F.layer_norm(
        references["x"], [N], references["weight"], references["bias"], EPS
    )

    autotune_seconds = None
    if args.mode == "autotune":
        kernel = helion.kernel(
            layer_norm.layer_norm_fwd.fn,
            static_shapes=True,
            force_autotune=True,
        )
    else:
        kernel = helion.kernel(
            layer_norm.layer_norm_fwd.fn,
            config=make_config(args.variant),
            static_shapes=True,
        )
    layer_norm.layer_norm_fwd = kernel

    if args.mode == "autotune":
        start = time.perf_counter()
        layer_norm.layer_norm(x, [N], weight, bias, EPS)
        torch.cuda.synchronize()
        autotune_seconds = time.perf_counter() - start
    else:
        layer_norm.layer_norm_fwd(x, [N], weight, bias, EPS)
        torch.cuda.synchronize()

    bound = kernel.bind((x, [N], weight, bias, EPS))
    config = bound._config
    assert config is not None
    direct_iterations = (
        min(args.iterations, 20) if args.mode == "autotune" else args.iterations
    )
    direct = run_raw_stress(direct_iterations, x, weight, bias, references)
    if args.skip_graph:
        graph: dict[str, object] = {"skipped": True}
    else:
        graph = graph_benchmark(
            lambda: layer_norm.layer_norm(x, [N], weight, bias, EPS),
            graph_repeats=args.graph_repeats,
            graph_replays=args.graph_replays,
        )
    accuracy = run_tritonbench_accuracy_stress(args.iterations, x, weight, bias)
    raw_poison = run_raw_poison_stress(args.iterations, x, weight, bias, references)
    post_graph_raw = run_raw_stress(20, x, weight, bias, references)
    torch.cuda.synchronize()

    summary = {
        "mode": args.mode,
        "requested_variant": args.variant,
        "requested_exact_config": EXACT_CONFIG,
        "selected_config": dict(config),
        "autotune_seconds": autotune_seconds,
        "environment": environment_info(),
        "input_mutation": {
            "x": not torch.equal(x, references["x"]),
            "weight": not torch.equal(weight, references["weight"]),
            "bias": not torch.equal(bias, references["bias"]),
        },
        "direct_before_graph": direct,
        "graph_benchmark": graph,
        "accuracy_after_graph": accuracy,
        "raw_poison_after_graph": raw_poison,
        "raw_after_graph": post_graph_raw,
        "compiler": archive_compiler_artifacts(args.output_dir, bound, config),
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    mismatch_count = sum(
        phase["failure_count"]
        for phase in (direct, accuracy, raw_poison, post_graph_raw)
    )
    print(
        "SUMMARY "
        f"mode={args.mode} variant={args.variant} "
        f"selected={config} mismatching_launches={mismatch_count} "
        f"path={summary_path}",
        flush=True,
    )
    return int(args.fail_on_mismatch and mismatch_count != 0)


if __name__ == "__main__":
    raise SystemExit(main())
