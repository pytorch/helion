# ruff: noqa: ANN401,E402

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import random
import statistics
import sys
import tempfile
from typing import Any
from typing import Mapping
from typing import NamedTuple
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_DEEPGEMM_SELECTED_SOURCE_M_TILE,
)
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_SELECTED_NM_EXPLICIT_STORE_SHAPE,
)

CASE_NAME = "deepgemm_grouped_bf16_nt_production_selected"
SELECTED_BLOCK_M = 256
SELECTED_BLOCK_N = 128
SELECTED_BLOCK_K = 64
SELECTED_SOURCE_M_TILE = TCGEN05_DEEPGEMM_SELECTED_SOURCE_M_TILE
SELECTED_N_ALIGNMENT = TCGEN05_SELECTED_NM_EXPLICIT_STORE_SHAPE[2]
DEFAULT_M_ALIGNMENT = 224
CACHE_ROOT_ENV = "HELION_GROUPED_DEEPGEMM_BENCH_CACHE_ROOT"
DG_JIT_CACHE_ENV = "DG_JIT_CACHE_DIR"
CACHE_ENV_DIRS = {
    "HELION_CACHE_DIR": "helion",
    "CUTE_DSL_CACHE_DIR": "cute_dsl",
    "TORCHINDUCTOR_CACHE_DIR": "torchinductor",
    "TRITON_CACHE_DIR": "triton",
}
ENV_SNAPSHOT_KEYS = (
    "PATH",
    "CUDA_VISIBLE_DEVICES",
    "CUDA_HOME",
    "HELION_BACKEND",
    "HELION_CUTE_MMA_IMPL",
    *CACHE_ENV_DIRS,
    DG_JIT_CACHE_ENV,
)
EXTRA_ENV_SNAPSHOT_KEYS = ("CPATH", "CPLUS_INCLUDE_PATH")
DEEPGEMM_PADDING_SENTINEL = 13.0
HELION_KERNEL_SOURCE = "production-selected"
HELION_KERNEL_SOURCE_KIND = "production_selected"
HELION_KERNEL_SOURCE_NOTE = (
    "Production selected-segment route exercised by benchmarks.cute.deepgemm_m_grouped."
)


class OfficialShape(NamedTuple):
    row_index: int
    groups: int
    expected_m_per_group: int
    n: int
    k: int

    @property
    def label(self) -> str:
        return f"{self.groups}/{self.expected_m_per_group}/{self.n}/{self.k}"


OFFICIAL_SHAPES = (
    OfficialShape(0, 4, 8192, 6144, 7168),
    OfficialShape(1, 4, 8192, 7168, 3072),
    OfficialShape(2, 4, 8192, 4096, 4096),
    OfficialShape(3, 4, 8192, 4096, 2048),
    OfficialShape(4, 8, 4096, 6144, 7168),
    OfficialShape(5, 8, 4096, 7168, 3072),
    OfficialShape(6, 8, 4096, 4096, 4096),
    OfficialShape(7, 8, 4096, 4096, 2048),
)


class BenchmarkSetupError(RuntimeError):
    pass


class BenchmarkCorrectnessError(RuntimeError):
    pass


class HelionKernelBinding(NamedTuple):
    call: Any
    work_tile_metadata: torch.Tensor
    details: dict[str, Any]


class DeepGemmMGroupedBf16GemmNtContiguousRouteReport(NamedTuple):
    route: str
    layout_has_valid_prefix_tiles: bool
    use_generated_segment_kernel: bool
    use_selected_segment_kernel: bool
    selected_segment_work_tile_metadata: torch.Tensor | None

    @property
    def selected_metadata_nonempty(self) -> bool:
        metadata = self.selected_segment_work_tile_metadata
        return metadata is not None and int(metadata.size(0)) > 0

    def as_dict(self) -> dict[str, object]:
        return {
            "layout_has_valid_prefix_tiles": self.layout_has_valid_prefix_tiles,
            "use_generated_segment_kernel": self.use_generated_segment_kernel,
            "use_selected_segment_kernel": self.use_selected_segment_kernel,
            "selected_metadata_nonempty": self.selected_metadata_nonempty,
            "route": self.route,
        }


def parse_rows(value: str, row_count: int = len(OFFICIAL_SHAPES)) -> list[int]:
    if value.strip().lower() == "all":
        return list(range(row_count))
    selected: set[int] = set()
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if start > end:
                raise argparse.ArgumentTypeError(f"invalid row range {part}")
            selected.update(range(start, end + 1))
        else:
            selected.add(int(part))
    rows = sorted(selected)
    invalid = [row for row in rows if row < 0 or row >= row_count]
    if invalid:
        raise argparse.ArgumentTypeError(f"row indices out of range: {invalid}")
    return rows


def align(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def shape_to_dict(shape: OfficialShape) -> dict[str, int | str]:
    return {
        "row_index": shape.row_index,
        "groups": shape.groups,
        "expected_m_per_group": shape.expected_m_per_group,
        "n": shape.n,
        "k": shape.k,
        "label": shape.label,
    }


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, (float, int, str, bool)) or value is None:
        return value
    return str(value)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n")


def summarize_us(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"samples_us": []}
    ordered = sorted(values)
    return {
        "median_us": statistics.median(values),
        "mean_us": statistics.mean(values),
        "min_us": min(values),
        "max_us": max(values),
        "p20_us": ordered[max(0, int(0.2 * len(ordered)) - 1)],
        "p80_us": ordered[min(len(ordered) - 1, int(0.8 * len(ordered)))],
        "samples_us": list(values),
    }


def add_tflops(
    timing: dict[str, Any],
    *,
    valid_flops: int,
    aligned_flops: int,
) -> None:
    median_us = timing.get("median_us")
    if isinstance(median_us, (float, int)) and median_us > 0:
        timing["valid_tflops"] = valid_flops / (median_us * 1e-6) / 1e12
        timing["aligned_tflops"] = aligned_flops / (median_us * 1e-6) / 1e12


def make_official_case(
    torch_module: Any,
    shape: OfficialShape,
    device: Any,
    *,
    m_alignment: int,
) -> tuple[Any, Any, Any, Any, list[dict[str, int]]]:
    alignment = int(m_alignment)
    random.seed(0)
    torch_module.manual_seed(0)
    torch_module.cuda.manual_seed(0)
    actual_ms = [
        int(shape.expected_m_per_group * random.uniform(0.7, 1.3))
        for _ in range(shape.groups)
    ]
    aligned_ms = [align(actual_m, alignment) for actual_m in actual_ms]
    m_total = sum(aligned_ms)

    a = torch_module.randn((m_total, shape.k), device=device, dtype=torch.bfloat16)
    b = torch_module.randn(
        (shape.groups, shape.n, shape.k),
        device=device,
        dtype=torch.bfloat16,
    )
    layout = torch_module.empty(m_total, device=device, dtype=torch.int32)
    ref = torch_module.empty((m_total, shape.n), device=device, dtype=torch.bfloat16)

    per_group: list[dict[str, int]] = []
    start = 0
    for group, (actual_m, aligned_m) in enumerate(
        zip(actual_ms, aligned_ms, strict=True)
    ):
        actual_end = start + actual_m
        aligned_end = start + aligned_m
        layout[start:actual_end] = group
        layout[actual_end:aligned_end] = -1
        a[actual_end:aligned_end] = 0
        ref[start:aligned_end] = a[start:aligned_end] @ b[group].t()
        per_group.append(
            {
                "group": group,
                "start": start,
                "actual_m": actual_m,
                "aligned_m": aligned_m,
                "padding_m": aligned_m - actual_m,
            }
        )
        start = aligned_end
    return a.contiguous(), b.contiguous(), layout.contiguous(), ref, per_group


def calc_diff(torch_module: Any, x: Any, y: Any) -> float:
    if int(x.numel()) == 0:
        return 0.0
    x64 = x.double()
    y64 = y.double()
    denominator = (x64 * x64 + y64 * y64).sum()
    if float(denominator.item()) == 0.0:
        return 0.0
    sim = 2 * (x64 * y64).sum() / denominator
    return float((1 - sim).item())


def max_abs(torch_module: Any, x: Any, y: Any, mask: Any) -> float:
    if not bool(torch_module.any(mask).item()):
        return 0.0
    return float((x[mask].float() - y[mask].float()).abs().max().item())


def correctness_metrics(
    torch_module: Any,
    out: Any,
    ref: Any,
    grouped_layout: Any,
    *,
    max_diff: float,
    padding_atol: float,
) -> dict[str, Any]:
    valid_mask = grouped_layout >= 0
    padding_mask = grouped_layout < 0
    zeros = torch_module.zeros_like(out)
    metrics = {
        "valid_rows": int(torch_module.count_nonzero(valid_mask).item()),
        "padding_rows": int(torch_module.count_nonzero(padding_mask).item()),
        "calc_diff_all": calc_diff(torch_module, out, ref),
        "calc_diff_valid": calc_diff(torch_module, out[valid_mask], ref[valid_mask]),
        "calc_diff_padding": calc_diff(
            torch_module,
            out[padding_mask],
            ref[padding_mask],
        ),
        "max_abs_valid": max_abs(torch_module, out, ref, valid_mask),
        "max_abs_padding_vs_ref": max_abs(torch_module, out, ref, padding_mask),
        "max_abs_padding_vs_zero": max_abs(torch_module, out, zeros, padding_mask),
    }
    metrics["ok"] = (
        metrics["calc_diff_valid"] <= max_diff
        and metrics["calc_diff_padding"] <= max_diff
        and metrics["max_abs_padding_vs_zero"] <= padding_atol
    )
    return metrics


def valid_only_correctness_metrics(
    torch_module: Any,
    out: Any,
    ref: Any,
    grouped_layout: Any,
    *,
    max_diff: float,
) -> dict[str, Any]:
    valid_mask = grouped_layout >= 0
    metrics = {
        "scope": "valid_rows_only",
        "padding_checked_for_zero": False,
        "padding_policy": "reported_separately",
        "valid_rows": int(torch_module.count_nonzero(valid_mask).item()),
        "padding_rows": int(torch_module.count_nonzero(grouped_layout < 0).item()),
        "calc_diff_all_observed": calc_diff(torch_module, out, ref),
        "calc_diff_valid": calc_diff(torch_module, out[valid_mask], ref[valid_mask]),
        "max_abs_valid": max_abs(torch_module, out, ref, valid_mask),
    }
    metrics["ok"] = metrics["calc_diff_valid"] <= max_diff
    return metrics


def padding_zero_report(
    torch_module: Any,
    out: Any,
    ref: Any,
    grouped_layout: Any,
    *,
    padding_atol: float,
) -> dict[str, Any]:
    padding_mask = grouped_layout < 0
    zeros = torch_module.zeros_like(out)
    metrics = {
        "scope": "padding_rows_only",
        "included_in_correctness_ok": False,
        "padding_checked_for_zero": True,
        "padding_rows": int(torch_module.count_nonzero(padding_mask).item()),
        "calc_diff_padding": calc_diff(
            torch_module,
            out[padding_mask],
            ref[padding_mask],
        ),
        "max_abs_padding_vs_ref": max_abs(torch_module, out, ref, padding_mask),
        "max_abs_padding_vs_zero": max_abs(torch_module, out, zeros, padding_mask),
        "padding_atol": padding_atol,
    }
    metrics["zero_padding_ok"] = metrics["max_abs_padding_vs_zero"] <= padding_atol
    return metrics


def correctness_block_ok(block: Mapping[str, Any]) -> bool:
    return all(
        bool(item.get("ok"))
        for key, item in block.items()
        if key.endswith("_correctness") and isinstance(item, Mapping)
    )


def max_correctness_metric(block: Mapping[str, Any]) -> float:
    max_metric = 0.0
    for key, item in block.items():
        if key.endswith("_correctness") and isinstance(item, Mapping):
            max_metric = max(
                max_metric,
                float(item.get("calc_diff_valid", 0.0)),
                float(item.get("calc_diff_padding", 0.0)),
                float(item.get("max_abs_padding_vs_zero", 0.0)),
            )
    return max_metric


def capture_graph(
    torch_module: Any,
    fn: Any,
    *,
    compile_warmups: int,
) -> tuple[Any, Any, list[Any]]:
    held: list[Any] = []
    for _ in range(compile_warmups):
        held.append(fn())
    torch_module.cuda.synchronize()
    graph = torch_module.cuda.CUDAGraph()
    with torch_module.cuda.graph(graph):
        captured = fn()
    held.append(captured)
    torch_module.cuda.synchronize()
    return graph, captured, held


def verify_graph_output(
    torch_module: Any,
    graph: Any,
    captured: Any,
    ref: Any,
    grouped_layout: Any,
    *,
    max_diff: float,
    padding_atol: float,
) -> dict[str, Any]:
    padding_mask = grouped_layout < 0
    if bool(torch_module.any(padding_mask).item()):
        captured[padding_mask] = DEEPGEMM_PADDING_SENTINEL
    graph.replay()
    torch_module.cuda.synchronize()
    return correctness_metrics(
        torch_module,
        captured,
        ref,
        grouped_layout,
        max_diff=max_diff,
        padding_atol=padding_atol,
    )


def positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {text}")
    return value


def nonnegative_int(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError(f"expected a non-negative integer, got {text}")
    return value


def optional_path_from_env(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value) if value else None


def call_int(fn: Any) -> int:
    return int(fn())


def import_deepgemm(root: Path) -> Any:
    root = root.expanduser().resolve()
    if not root.exists():
        raise BenchmarkSetupError(f"DeepGEMM root does not exist: {root}")
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    deep_gemm = importlib.import_module("deep_gemm")
    if not hasattr(deep_gemm, "m_grouped_bf16_gemm_nt_contiguous"):
        raise BenchmarkSetupError(
            "DeepGEMM module is missing m_grouped_bf16_gemm_nt_contiguous"
        )
    return deep_gemm


def import_benchmark_deepgemm_m_grouped() -> Any:
    return importlib.import_module("benchmarks.cute.deepgemm_m_grouped")


def configure_deepgemm(deep_gemm: Any, root: Path, m_alignment: int) -> dict[str, Any]:
    info: dict[str, Any] = {
        "root": root.expanduser().resolve(),
        "file": getattr(deep_gemm, "__file__", None),
        "requested_mk_alignment": int(m_alignment),
    }
    set_alignment = getattr(deep_gemm, "set_mk_alignment_for_contiguous_layout", None)
    if callable(set_alignment):
        set_alignment(int(m_alignment))
        info["mk_alignment_set"] = True
    else:
        info["mk_alignment_set"] = False

    get_alignment = getattr(deep_gemm, "get_mk_alignment_for_contiguous_layout", None)
    if callable(get_alignment):
        info["mk_alignment"] = call_int(get_alignment)

    get_theoretical = getattr(
        deep_gemm,
        "get_theoretical_mk_alignment_for_contiguous_layout",
        None,
    )
    if callable(get_theoretical):
        info["theoretical_mk_alignment"] = call_int(get_theoretical)
    return info


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark the production selected-segment Helion path on the "
            "official DeepGEMM M-grouped BF16 NT rows."
        ),
    )
    parser.add_argument("--rows", default="all", help="Rows such as all, 0,1, or 0-3")
    parser.add_argument("--samples", type=positive_int, default=7)
    parser.add_argument("--iters", type=positive_int, default=10)
    parser.add_argument("--warmups", type=nonnegative_int, default=3)
    parser.add_argument("--compile-warmups", type=nonnegative_int, default=2)
    parser.add_argument(
        "--device",
        default=os.environ.get("HELION_BENCH_DEVICE", "cuda"),
        help="Torch CUDA device string, for example cuda or cuda:0.",
    )
    parser.add_argument(
        "--m-alignment",
        type=positive_int,
        default=DEFAULT_M_ALIGNMENT,
        help="M alignment for selected-only rows.",
    )
    parser.add_argument(
        "--compare-deepgemm",
        action="store_true",
        help="Also compare against DeepGEMM with graph/event kernel timing.",
    )
    parser.add_argument(
        "--deepgemm-root",
        type=Path,
        default=None,
        help="Root to prepend to sys.path for importing deep_gemm in comparison mode.",
    )
    parser.add_argument("--max-diff", type=float, default=1e-3)
    parser.add_argument("--padding-atol", type=float, default=0.0)
    parser.add_argument("--min-capability-major", type=int, default=10)
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=None,
        help="Directory for JSON/log/cache artifacts.",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=optional_path_from_env(CACHE_ROOT_ENV),
        help=f"Cache root override; also read from {CACHE_ROOT_ENV}.",
    )
    parser.add_argument(
        "--json-output",
        "--output",
        dest="json_output",
        type=Path,
        default=None,
        help="JSON output path. Defaults under --artifact-dir.",
    )
    return parser


def work_tile_metadata_summary(work_tile_metadata: torch.Tensor) -> dict[str, Any]:
    values = work_tile_metadata.detach().cpu().tolist()
    return {
        "shape": list(work_tile_metadata.shape),
        "dtype": str(work_tile_metadata.dtype),
        "device": str(work_tile_metadata.device),
        "rows": int(work_tile_metadata.size(0)),
        "values": values,
    }


def deepgemm_m_grouped_bf16_gemm_nt_contiguous_route_report(
    a_packed: torch.Tensor,
    b_grouped: torch.Tensor,
    grouped_layout: torch.Tensor,
) -> DeepGemmMGroupedBf16GemmNtContiguousRouteReport:
    """Report the internal route selected by the benchmarked API call."""
    from benchmarks.cute import deepgemm_m_grouped

    layout_has_valid_prefix_tiles = (
        deepgemm_m_grouped._validate_deepgemm_m_grouped_bf16_gemm_nt_contiguous_args(
            a_packed,
            b_grouped,
            grouped_layout,
        )
    )
    selected_work_tile_metadata = deepgemm_m_grouped._deepgemm_m_grouped_bf16_gemm_nt_contiguous_selected_segment_metadata_tensor(
        a_packed,
        b_grouped,
        grouped_layout,
    )
    use_selected_segment_kernel = deepgemm_m_grouped._deepgemm_m_grouped_bf16_gemm_nt_contiguous_use_selected_segment_kernel(
        a_packed,
        b_grouped,
        grouped_layout,
    )
    use_generated_segment_kernel = deepgemm_m_grouped._deepgemm_m_grouped_bf16_gemm_nt_contiguous_use_generated_segment_kernel(
        a_packed,
        b_grouped,
        grouped_layout,
        layout_has_valid_prefix_tiles=layout_has_valid_prefix_tiles,
    )
    selected_metadata_nonempty = (
        selected_work_tile_metadata is not None
        and int(selected_work_tile_metadata.size(0)) > 0
    )
    if (
        use_generated_segment_kernel
        and use_selected_segment_kernel
        and selected_metadata_nonempty
    ):
        route = "generated_selected_segment"
    elif use_generated_segment_kernel:
        route = "generated_segment"
    else:
        route = "generic"
    return DeepGemmMGroupedBf16GemmNtContiguousRouteReport(
        route=route,
        layout_has_valid_prefix_tiles=layout_has_valid_prefix_tiles,
        use_generated_segment_kernel=use_generated_segment_kernel,
        use_selected_segment_kernel=use_selected_segment_kernel,
        selected_segment_work_tile_metadata=selected_work_tile_metadata,
    )


def production_selected_route_assertions(route_report: Any) -> dict[str, object]:
    work_tile_metadata = route_report.selected_segment_work_tile_metadata
    return {
        "layout_has_valid_prefix_tiles": route_report.layout_has_valid_prefix_tiles,
        "use_generated_segment_kernel": route_report.use_generated_segment_kernel,
        "use_selected_segment_kernel": route_report.use_selected_segment_kernel,
        "selected_metadata_nonempty": (
            work_tile_metadata is not None and int(work_tile_metadata.size(0)) > 0
        ),
        "route": route_report.route,
    }


def check_production_selected_route(
    route_assertions: dict[str, object],
    *,
    context: str,
) -> None:
    if route_assertions["route"] != "generated_selected_segment":
        raise BenchmarkSetupError(
            f"production-selected mode requires route generated_selected_segment "
            f"{context}; got {route_assertions}"
        )


def check_retained_production_selected_route(
    route_assertions: dict[str, object],
) -> None:
    if route_assertions["route"] != "generated_selected_segment":
        raise BenchmarkSetupError(
            "production-selected API did not retain route "
            f"generated_selected_segment after timed API call; got {route_assertions}"
        )


def production_selected_kernel_binding(
    a_packed: torch.Tensor,
    b_grouped: torch.Tensor,
    grouped_layout: torch.Tensor,
) -> HelionKernelBinding:
    deepgemm_m_grouped = import_benchmark_deepgemm_m_grouped()
    route_report = deepgemm_m_grouped_bf16_gemm_nt_contiguous_route_report(
        a_packed,
        b_grouped,
        grouped_layout,
    )
    route_assertions = production_selected_route_assertions(route_report)
    check_production_selected_route(
        route_assertions,
        context="before timed API call",
    )

    work_tile_metadata = route_report.selected_segment_work_tile_metadata
    if work_tile_metadata is None:
        raise BenchmarkSetupError(
            "production-selected route did not provide selected-segment metadata"
        )

    def call_timed_api() -> torch.Tensor:
        result = deepgemm_m_grouped.deepgemm_m_grouped_bf16_gemm_nt_contiguous(
            a_packed,
            b_grouped,
            grouped_layout,
        )
        post_call_route = deepgemm_m_grouped_bf16_gemm_nt_contiguous_route_report(
            a_packed,
            b_grouped,
            grouped_layout,
        )
        post_call_route_assertions = production_selected_route_assertions(
            post_call_route
        )
        details["route_assertions_after_last_call"] = post_call_route_assertions
        check_retained_production_selected_route(post_call_route_assertions)
        return result

    timed_api = (
        "benchmarks.cute.deepgemm_m_grouped.deepgemm_m_grouped_bf16_gemm_nt_contiguous"
    )
    details = {
        "helion_kernel_source": HELION_KERNEL_SOURCE,
        "route": route_assertions["route"],
        "route_assertions": route_assertions,
        "route_retention_check": "after_each_timed_api_call",
        "work_tile_metadata": work_tile_metadata_summary(work_tile_metadata),
        "kernel": timed_api,
        "timed_api": timed_api,
    }
    return HelionKernelBinding(
        call=call_timed_api,
        work_tile_metadata=work_tile_metadata,
        details=details,
    )


def graph_timing(
    *,
    graph: Any,
    samples: int,
    iters: int,
    warmups: int,
    valid_flops: int,
    aligned_flops: int,
) -> dict[str, Any]:
    torch.cuda.synchronize()
    for _ in range(warmups):
        graph.replay()
    torch.cuda.synchronize()

    times: list[float] = []
    stream = torch.cuda.current_stream()
    for _ in range(samples):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(stream)
        for _ in range(iters):
            graph.replay()
        end.record(stream)
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000.0 / iters)

    summary = summarize_us(times)
    add_tflops(summary, valid_flops=valid_flops, aligned_flops=aligned_flops)
    summary["samples"] = samples
    summary["iters"] = iters
    summary["warmups"] = warmups
    summary["method"] = "cuda_graph_replay_cuda_event_elapsed_time"
    return summary


def mark_padding_rows(
    torch_module: Any,
    out: Any,
    grouped_layout: Any,
    value: float,
) -> None:
    padding_mask = grouped_layout < 0
    if bool(torch_module.any(padding_mask).item()):
        out[padding_mask] = value


def deepgemm_comparison_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ratios = [
        float(row["ratio_helion_over_deepgemm"])
        for row in rows
        if "ratio_helion_over_deepgemm" in row
    ]
    if not ratios:
        return {"rows": 0}
    return {
        "rows": len(ratios),
        "median_ratio_helion_over_deepgemm": statistics.median(ratios),
        "mean_ratio_helion_over_deepgemm": statistics.mean(ratios),
        "min_ratio_helion_over_deepgemm": min(ratios),
        "max_ratio_helion_over_deepgemm": max(ratios),
        "helion_faster_rows": sum(ratio < 1.0 for ratio in ratios),
        "deepgemm_faster_rows": sum(ratio > 1.0 for ratio in ratios),
    }


def run_shape(
    args: argparse.Namespace,
    shape: OfficialShape,
    *,
    device: torch.device,
    deep_gemm: Any | None = None,
) -> dict[str, Any]:
    a_packed, b_grouped, layout, ref, per_group = make_official_case(
        torch,
        shape,
        device,
        m_alignment=args.m_alignment,
    )
    valid_m = sum(item["actual_m"] for item in per_group)
    aligned_m = int(layout.numel())
    valid_flops = 2 * valid_m * shape.n * shape.k
    aligned_flops = 2 * aligned_m * shape.n * shape.k
    helion_binding = production_selected_kernel_binding(
        a_packed,
        b_grouped,
        layout,
    )
    metadata = helion_binding.work_tile_metadata
    metadata_kind = helion_binding.details.get("metadata_kind", "work_tile_metadata")
    row: dict[str, Any] = {
        "row_index": shape.row_index,
        "shape": shape_to_dict(shape),
        "helion_kernel_source": HELION_KERNEL_SOURCE,
        "helion_kernel_source_kind": HELION_KERNEL_SOURCE_KIND,
        "helion_kernel_source_note": HELION_KERNEL_SOURCE_NOTE,
        "m_valid_total": valid_m,
        "m_aligned_total": aligned_m,
        "m_padding_total": aligned_m - valid_m,
        "m_alignment": args.m_alignment,
        "per_group": per_group,
        "metadata_kind": metadata_kind,
        "metadata_rows": int(metadata.size(0)),
        "selected_block_m": SELECTED_BLOCK_M,
        "selected_block_n": SELECTED_BLOCK_N,
        "selected_block_k": SELECTED_BLOCK_K,
        "selected_source_m_tile": SELECTED_SOURCE_M_TILE,
        "selected_n_alignment": SELECTED_N_ALIGNMENT,
        "selected_ab_stages": 7,
        "valid_flops": valid_flops,
        "aligned_flops": aligned_flops,
    }
    row["work_tile_metadata_rows"] = int(metadata.size(0))
    if helion_binding.details:
        row["helion_kernel_details"] = helion_binding.details

    def selected_fn() -> torch.Tensor:
        return helion_binding.call()

    selected_eager = selected_fn()
    torch.cuda.synchronize()
    correctness: dict[str, Any] = {
        "selected_eager_correctness": correctness_metrics(
            torch,
            selected_eager,
            ref,
            layout,
            max_diff=args.max_diff,
            padding_atol=args.padding_atol,
        ),
    }
    selected_graph, selected_captured, selected_held = capture_graph(
        torch,
        selected_fn,
        compile_warmups=args.compile_warmups,
    )
    correctness["selected_graph_correctness"] = verify_graph_output(
        torch,
        selected_graph,
        selected_captured,
        ref,
        layout,
        max_diff=args.max_diff,
        padding_atol=args.padding_atol,
    )
    deepgemm_held: list[Any] = []
    deepgemm_graph = None
    deepgemm_captured = None
    if deep_gemm is not None:
        deep_gemm_module = deep_gemm
        deepgemm_out = torch.empty(
            (aligned_m, shape.n),
            device=device,
            dtype=torch.bfloat16,
        )

        def deepgemm_fn() -> torch.Tensor:
            deep_gemm_module.m_grouped_bf16_gemm_nt_contiguous(
                a_packed,
                b_grouped,
                deepgemm_out,
                layout,
            )
            return deepgemm_out

        mark_padding_rows(
            torch,
            deepgemm_out,
            layout,
            DEEPGEMM_PADDING_SENTINEL,
        )
        deepgemm_eager = deepgemm_fn()
        torch.cuda.synchronize()
        correctness["deepgemm_eager_valid_only_correctness"] = (
            valid_only_correctness_metrics(
                torch,
                deepgemm_eager,
                ref,
                layout,
                max_diff=args.max_diff,
            )
        )
        correctness["deepgemm_eager_padding_report"] = padding_zero_report(
            torch,
            deepgemm_eager,
            ref,
            layout,
            padding_atol=args.padding_atol,
        )
        deepgemm_graph, deepgemm_captured, deepgemm_held = capture_graph(
            torch,
            deepgemm_fn,
            compile_warmups=args.compile_warmups,
        )
        mark_padding_rows(
            torch,
            deepgemm_captured,
            layout,
            DEEPGEMM_PADDING_SENTINEL,
        )
        deepgemm_graph.replay()
        torch.cuda.synchronize()
        correctness["deepgemm_graph_valid_only_correctness"] = (
            valid_only_correctness_metrics(
                torch,
                deepgemm_captured,
                ref,
                layout,
                max_diff=args.max_diff,
            )
        )
        correctness["deepgemm_graph_padding_report"] = padding_zero_report(
            torch,
            deepgemm_captured,
            ref,
            layout,
            padding_atol=args.padding_atol,
        )

    correctness["ok"] = correctness_block_ok(correctness)
    correctness["max_abs_metric"] = max_correctness_metric(correctness)
    row["correctness"] = correctness
    if not correctness["ok"]:
        raise BenchmarkCorrectnessError(
            f"correctness failed for row {shape.row_index} ({shape.label})"
        )

    helion_graph_timing = graph_timing(
        graph=selected_graph,
        samples=args.samples,
        iters=args.iters,
        warmups=args.warmups,
        valid_flops=valid_flops,
        aligned_flops=aligned_flops,
    )
    row["helion_graph_timing"] = helion_graph_timing
    row["selected_graph_timing"] = helion_graph_timing
    if deep_gemm is not None:
        assert deepgemm_graph is not None
        assert deepgemm_captured is not None
        row["helion_selected_graph_timing"] = helion_graph_timing
        row["deepgemm_graph_timing"] = graph_timing(
            graph=deepgemm_graph,
            samples=args.samples,
            iters=args.iters,
            warmups=args.warmups,
            valid_flops=valid_flops,
            aligned_flops=aligned_flops,
        )
        row["ratio_helion_over_deepgemm"] = float(
            row["helion_graph_timing"]["median_us"]
        ) / float(row["deepgemm_graph_timing"]["median_us"])
        row["deepgemm_timing_note"] = (
            "DeepGEMM was timed with preallocated output; "
            "padding reports are observational and "
            "are not included in the valid-only DeepGEMM correctness result."
        )
    row["status"] = "ok"
    selected_held.clear()
    deepgemm_held.clear()
    return row


def configure_env(args: argparse.Namespace, artifact_dir: Path) -> dict[str, str]:
    os.environ["HELION_BACKEND"] = "cute"
    os.environ["HELION_CUTE_MMA_IMPL"] = "tcgen05"
    cache_root = args.cache_root or artifact_dir / "caches"
    for env_name, subdir in CACHE_ENV_DIRS.items():
        os.environ[env_name] = str(cache_root / subdir)
    if args.compare_deepgemm:
        os.environ[DG_JIT_CACHE_ENV] = str(cache_root / "deepgemm_jit")
    return {
        key: os.environ.get(key, "")
        for key in (*ENV_SNAPSHOT_KEYS, *EXTRA_ENV_SNAPSHOT_KEYS)
    }


def run(args: argparse.Namespace) -> int:
    selected_rows = parse_rows(args.rows)
    artifact_dir = args.artifact_dir or Path(
        tempfile.mkdtemp(prefix="helion_selected_deepgemm_bench_")
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = artifact_dir.resolve()
    json_output = (
        args.json_output.resolve()
        if args.json_output is not None
        else artifact_dir / "selected_results.json"
    )
    output: dict[str, Any] = {
        "case": CASE_NAME,
        "artifact_dir": artifact_dir,
        "repo_root": REPO_ROOT,
        "official_shapes": [shape_to_dict(shape) for shape in OFFICIAL_SHAPES],
        "selected_rows": selected_rows,
        "helion_kernel_source": HELION_KERNEL_SOURCE,
        "helion_kernel_source_kind": HELION_KERNEL_SOURCE_KIND,
        "helion_kernel_source_note": HELION_KERNEL_SOURCE_NOTE,
        "settings": {
            "rows": args.rows,
            "samples": args.samples,
            "iters": args.iters,
            "warmups": args.warmups,
            "compile_warmups": args.compile_warmups,
            "device": args.device,
            "m_alignment": args.m_alignment,
            "compare_deepgemm": args.compare_deepgemm,
            "helion_kernel_source": HELION_KERNEL_SOURCE,
            "helion_kernel_source_kind": HELION_KERNEL_SOURCE_KIND,
            "deepgemm_root": args.deepgemm_root,
            "cache_root": args.cache_root,
            "max_diff": args.max_diff,
            "padding_atol": args.padding_atol,
        },
        "rows": [],
    }
    try:
        output["env"] = configure_env(args, artifact_dir)

        if not torch.cuda.is_available():
            raise BenchmarkSetupError("CUDA is not available")
        requested_device = torch.device(args.device)
        if requested_device.type != "cuda":
            raise BenchmarkSetupError(f"expected a CUDA device, got {args.device}")
        if requested_device.index is not None:
            torch.cuda.set_device(requested_device)
        device = torch.device("cuda", torch.cuda.current_device())
        capability = torch.cuda.get_device_capability(device)
        output.update(
            {
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "device": str(device),
                "device_name": torch.cuda.get_device_name(device),
                "capability": capability,
            }
        )
        if int(capability[0]) < args.min_capability_major:
            raise BenchmarkSetupError(
                "unsupported GPU for production selected benchmark: "
                f"capability {capability}, need major >= {args.min_capability_major}"
            )

        output["m_alignment_effective"] = args.m_alignment
        deep_gemm = None
        if args.compare_deepgemm:
            if args.deepgemm_root is None:
                raise BenchmarkSetupError("--compare-deepgemm requires --deepgemm-root")
            deep_gemm = import_deepgemm(args.deepgemm_root)
            output["deepgemm"] = configure_deepgemm(
                deep_gemm,
                args.deepgemm_root,
                args.m_alignment,
            )

        with torch.cuda.device(device):
            for row_index in selected_rows:
                shape = OFFICIAL_SHAPES[row_index]
                print(
                    "ROW "
                    f"{row_index}: G={shape.groups} "
                    f"M={shape.expected_m_per_group} "
                    f"N={shape.n} K={shape.k}",
                    flush=True,
                )
                row = run_shape(
                    args,
                    shape,
                    device=device,
                    deep_gemm=deep_gemm,
                )
                output["rows"].append(row)
                if args.compare_deepgemm:
                    output["deepgemm_comparison_summary"] = deepgemm_comparison_summary(
                        output["rows"]
                    )
                print(json.dumps(json_ready(row), sort_keys=True), flush=True)
                torch.cuda.empty_cache()

        output["status"] = "ok"
        write_json(json_output, output)
        return 0
    except BenchmarkCorrectnessError as exc:
        output["status"] = "correctness_failed"
        output["error"] = str(exc)
        write_json(json_output, output)
        print(str(exc), file=sys.stderr)
        return 4
    except BenchmarkSetupError as exc:
        output["status"] = "unsupported"
        output["error"] = str(exc)
        write_json(json_output, output)
        print(str(exc), file=sys.stderr)
        return 2
    except Exception:
        import traceback

        output["status"] = "error"
        output["traceback"] = traceback.format_exc()
        write_json(json_output, output)
        print(output["traceback"], file=sys.stderr)
        return 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
