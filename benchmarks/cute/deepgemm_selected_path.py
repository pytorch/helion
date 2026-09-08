# ruff: noqa: ANN401,E402

"""Reproduce the Helion/DeepGEMM grouped BF16 NT comparison."""

from __future__ import annotations

import argparse
from hashlib import sha256
import importlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.cute.grouped_gemm_workloads import DEEPGEMM_COMMIT
from benchmarks.cute.grouped_gemm_workloads import DEEPGEMM_CUTLASS_COMMIT
from benchmarks.cute.grouped_gemm_workloads import DEEPGEMM_M_ALIGNMENT as M_ALIGNMENT
from benchmarks.cute.grouped_gemm_workloads import OFFICIAL_SHAPES
from benchmarks.cute.grouped_gemm_workloads import OfficialShape
from benchmarks.cute.grouped_gemm_workloads import official_actual_ms
import torch

import helion
import helion.language as hl
import helion.runtime as helion_runtime

# Pin the historical DeepGEMM-selected BK64 schedule. This benchmark must not
# silently change when compiler defaults evolve.
DEEPGEMM_SELECTED_TILE_M = 256
DEEPGEMM_SELECTED_TILE_N = 128
DEEPGEMM_SELECTED_TILE_K = 64
DEFAULT_L2_FLUSH_BYTES = 8_000_000_000
CACHE_DIRS = {
    "CUDA_CACHE_PATH": "cuda",
    "CUTE_DSL_CACHE_DIR": "cute_dsl",
    "DG_JIT_CACHE_DIR": "deepgemm_jit",
    "HELION_CACHE_DIR": "helion",
    "TORCHINDUCTOR_CACHE_DIR": "torchinductor",
    "TRITON_CACHE_DIR": "triton",
    "TORCH_EXTENSIONS_DIR": "torch_extensions",
    "XDG_CACHE_HOME": "xdg",
}


def align(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def parse_rows(value: str) -> list[int]:
    if value.strip().lower() == "all":
        return list(range(len(OFFICIAL_SHAPES)))
    rows: set[int] = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start_text, end_text = item.split("-", 1)
            start, end = int(start_text), int(end_text)
            if start > end:
                raise argparse.ArgumentTypeError(f"invalid row range {item}")
            rows.update(range(start, end + 1))
        else:
            rows.add(int(item))
    result = sorted(rows)
    invalid = [row for row in result if row not in range(len(OFFICIAL_SHAPES))]
    if not result or invalid:
        message = (
            f"row indices out of range: {invalid}" if invalid else "no rows selected"
        )
        raise argparse.ArgumentTypeError(message)
    return result


def selected_key(
    a_packed: torch.Tensor,
    b_grouped: torch.Tensor,
    worklist: torch.Tensor,
) -> tuple[int, ...]:
    return (*a_packed.shape, *b_grouped.shape, int(worklist.size(0)))


def selected_kernel_body(
    A_packed: torch.Tensor,
    B_grouped: torch.Tensor,
    work_tile_metadata: torch.Tensor,
) -> torch.Tensor:
    """
    DeepGEMM-selected generated segment fallback for BF16 NT.

    ``work_tile_metadata`` is int32 ``[W, 4]`` with compact rows
    ``(group, group_start, actual_m, aligned_m)``. The selected N,M generated
    path derives each source-M chunk's valid and store extents on device from
    the grouped scheduler M tile.
    """
    M_total_aligned, K = A_packed.shape
    _G, N, K2 = B_grouped.shape
    assert K == K2, "K dimension mismatch between A_packed and B_grouped"
    assert work_tile_metadata.size(1) == 4

    block_m = hl.register_block_size(DEEPGEMM_SELECTED_TILE_M)
    block_n = hl.register_block_size(DEEPGEMM_SELECTED_TILE_N)
    block_k = hl.register_block_size(DEEPGEMM_SELECTED_TILE_K)
    out = torch.empty(
        M_total_aligned,
        N,
        dtype=A_packed.dtype,
        device=A_packed.device,
    )

    for work_tile, tile_m, tile_n in hl.tile(
        [work_tile_metadata.size(0), DEEPGEMM_SELECTED_TILE_M, N],
        block_size=[1, block_m, block_n],
    ):
        work_id = work_tile.begin
        group_id = work_tile_metadata[work_id, 0]
        global_m_start = work_tile_metadata[work_id, 1]
        valid_m = work_tile_metadata[work_id, 2]
        store_m = work_tile_metadata[work_id, 3]
        local_m = tile_m.index
        row_index = global_m_start + local_m
        valid_rows = local_m < valid_m
        store_rows = local_m < store_m
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(K, block_size=block_k):
            a_blk = hl.load(
                A_packed,
                [row_index, tile_k],
                extra_mask=valid_rows[:, None],  # pyrefly: ignore[bad-index]
            )
            acc = torch.addmm(
                acc,
                a_blk,
                B_grouped[group_id, tile_n, tile_k].T,
            )
        hl.store(
            out,
            [row_index, tile_n],
            acc.to(out.dtype),
            extra_mask=store_rows[:, None],  # pyrefly: ignore[bad-index]
        )

    return out


def selected_config() -> helion.Config:
    return helion.Config(
        block_sizes=[
            DEEPGEMM_SELECTED_TILE_M,
            DEEPGEMM_SELECTED_TILE_N,
            DEEPGEMM_SELECTED_TILE_K,
        ],
        l2_groupings=[1],
        loop_orders=[[0, 1, 2]],
        num_stages=7,
        num_warps=8,
        pid_type="persistent_interleaved",
        tcgen05_cluster_m=2,
        tcgen05_cluster_n=1,
        tcgen05_ab_stages=7,
        tcgen05_acc_stages=2,
        tcgen05_c_stages=2,
        tcgen05_num_epi_warps=4,
        tcgen05_grouped_mode="worklist_nm",
        tcgen05_grouped_worklist_source_m_tile=M_ALIGNMENT,
    )


def make_case(
    shape: OfficialShape,
    actual_ms: Sequence[int],
    device: torch.device,
    m_alignment: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    aligned_ms = [align(value, m_alignment) for value in actual_ms]
    m_total = sum(aligned_ms)
    a = torch.randn((m_total, shape.k), device=device, dtype=torch.bfloat16)
    b = torch.randn(
        (shape.groups, shape.n, shape.k), device=device, dtype=torch.bfloat16
    )
    layout = torch.empty(m_total, device=device, dtype=torch.int32)
    ref = torch.empty((m_total, shape.n), device=device, dtype=torch.bfloat16)
    rows: list[tuple[int, int, int, int]] = []
    start = 0
    for group, (actual_m, aligned_m) in enumerate(
        zip(actual_ms, aligned_ms, strict=True)
    ):
        actual_end, aligned_end = start + actual_m, start + aligned_m
        layout[start:actual_end] = group
        layout[actual_end:aligned_end] = -1
        a[actual_end:aligned_end] = 0
        ref[start:aligned_end] = a[start:aligned_end] @ b[group].T
        rows.append((group, start, actual_m, aligned_m))
        start = aligned_end
    worklist = torch.tensor(rows, device=device, dtype=torch.int32)
    return a.contiguous(), b.contiguous(), layout, ref, worklist


def calc_diff(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual64, expected64 = actual.double(), expected.double()
    denominator = (actual64.square() + expected64.square()).sum()
    if float(denominator) == 0.0:
        return 0.0
    return float((1 - 2 * (actual64 * expected64).sum() / denominator).item())


def correctness(
    out: torch.Tensor,
    ref: torch.Tensor,
    layout: torch.Tensor,
    max_diff: float,
    padding_atol: float,
    *,
    require_zero_padding: bool,
) -> dict[str, Any]:
    valid, padding = layout >= 0, layout < 0
    valid_diff = calc_diff(out[valid], ref[valid])
    padding_max = float(out[padding].float().abs().max()) if padding.any() else 0.0
    zero_padding_ok = padding_max <= padding_atol
    return {
        "valid_rows": int(valid.sum().item()),
        "padding_rows": int(padding.sum().item()),
        "calc_diff_valid": valid_diff,
        "max_abs_padding_vs_zero": padding_max,
        "require_zero_padding": require_zero_padding,
        "zero_padding_ok": zero_padding_ok,
        "ok": valid_diff <= max_diff and (zero_padding_ok or not require_zero_padding),
    }


def capture_and_check(
    fn: Any,
    ref: torch.Tensor,
    layout: torch.Tensor,
    args: argparse.Namespace,
    *,
    require_zero_padding: bool,
    track_cute: bool = False,
) -> tuple[Any, list[Any], dict[str, Any]]:
    held = [fn() for _ in range(args.compile_warmups + 1)]
    torch.cuda.synchronize()
    if track_cute:
        with helion_runtime.cute_cuda_graph() as graph:
            captured = fn()
    else:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = fn()
    held.append(captured)
    captured[layout < 0] = 13.0
    graph.replay()
    torch.cuda.synchronize()
    check = correctness(
        captured,
        ref,
        layout,
        args.max_diff,
        args.padding_atol,
        require_zero_padding=require_zero_padding,
    )
    return graph, held, check


def thermal_warmup(duration_ms: int) -> None:
    if duration_ms <= 0:
        return
    value = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    end = time.monotonic() + duration_ms / 1000
    while time.monotonic() < end:
        for _ in range(50):
            value = value @ value
        torch.cuda.synchronize()


def graph_timings(
    graphs: Sequence[Any],
    args: argparse.Namespace,
    flops: int,
) -> list[dict[str, Any]]:
    graph_count = len(graphs)
    for sample in range(args.warmups):
        order = (
            range(graph_count) if sample % 2 == 0 else range(graph_count - 1, -1, -1)
        )
        for index in order:
            graphs[index].replay()
    torch.cuda.synchronize()
    thermal_warmup(args.thermal_warmup_ms)
    l2_flush = (
        torch.empty(args.l2_flush_bytes // 4, dtype=torch.int, device="cuda")
        if args.l2_flush_bytes
        else None
    )
    values: list[list[float]] = [[] for _ in graphs]
    for sample in range(args.samples):
        order = (
            range(graph_count) if sample % 2 == 0 else range(graph_count - 1, -1, -1)
        )
        events: list[tuple[int, list[tuple[Any, Any]]]] = []
        for index in order:
            pairs = []
            for _ in range(args.iters):
                if l2_flush is not None:
                    l2_flush.zero_()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                graphs[index].replay()
                end.record()
                pairs.append((start, end))
            events.append((index, pairs))
        torch.cuda.synchronize()
        for index, pairs in events:
            values[index].append(
                statistics.mean(start.elapsed_time(end) * 1000 for start, end in pairs)
            )
    medians = [statistics.median(item) for item in values]
    return [
        {
            "median_us": median,
            "valid_tflops": flops / (median * 1e-6) / 1e12,
            "samples_us": samples,
        }
        for median, samples in zip(medians, values, strict=True)
    ]


def git_output(root: Path, *args: str) -> str:
    command = ["git", "-C", str(root), *args]
    return subprocess.check_output(command, text=True).rstrip("\n")


def git_provenance(root: Path) -> dict[str, Any]:
    status = git_output(root, "status", "--porcelain=v1", "--untracked-files=all")
    cutlass = root / "third-party" / "cutlass"
    return {
        "head": git_output(root, "rev-parse", "HEAD"),
        "dirty": bool(status),
        "status": status.splitlines(),
        "cutlass_head": git_output(cutlass, "rev-parse", "HEAD"),
        "cutlass_submodule_status": git_output(
            root, "submodule", "status", "--", "third-party/cutlass"
        ),
    }


def import_deepgemm(root: Path, m_alignment: int) -> tuple[Any, dict[str, Any]]:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise RuntimeError(f"DeepGEMM root does not exist: {root}")
    sys.path.insert(0, str(root))
    module = importlib.import_module("deep_gemm")
    if module.__file__ is None:
        raise RuntimeError("deep_gemm has no module origin")
    origin = Path(module.__file__).resolve()
    if not origin.is_relative_to(root):
        raise RuntimeError(f"deep_gemm resolved outside {root}: {origin}")
    module.set_mk_alignment_for_contiguous_layout(m_alignment)
    effective = int(module.get_mk_alignment_for_contiguous_layout())
    if effective != m_alignment:
        raise RuntimeError(f"DeepGEMM alignment is {effective}, expected {m_alignment}")
    provenance = git_provenance(root)
    if provenance["head"] != DEEPGEMM_COMMIT:
        raise RuntimeError(
            f"DeepGEMM HEAD is {provenance['head']}, expected {DEEPGEMM_COMMIT}"
        )
    if provenance["cutlass_head"] != DEEPGEMM_CUTLASS_COMMIT:
        raise RuntimeError(
            "DeepGEMM CUTLASS HEAD is "
            f"{provenance['cutlass_head']}, expected {DEEPGEMM_CUTLASS_COMMIT}"
        )
    unexpected_status = set(provenance["status"]) - {"?? deep_gemm/include/fmt"}
    if unexpected_status:
        raise RuntimeError(
            f"DeepGEMM checkout has unexpected changes: {unexpected_status}"
        )
    provenance.update({"module_origin": origin, "mk_alignment": effective})
    return module, provenance


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", default="all")
    parser.add_argument("--compare-deepgemm", action="store_true")
    parser.add_argument("--deepgemm-root", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--compile-warmups", type=int, default=2)
    parser.add_argument("--thermal-warmup-ms", type=int, default=10000)
    parser.add_argument("--l2-flush-bytes", type=int, default=DEFAULT_L2_FLUSH_BYTES)
    parser.add_argument("--m-alignment", type=int, default=M_ALIGNMENT)
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--json-output", "--output", dest="json_output", type=Path)
    parser.add_argument("--max-diff", type=float, default=1e-3)
    parser.add_argument("--padding-atol", type=float, default=0.0)
    return parser


def run_shape(
    args: argparse.Namespace,
    shape: OfficialShape,
    actual_ms: Sequence[int],
    device: torch.device,
    kernel: helion.Kernel,
    deep_gemm: Any | None,
) -> dict[str, Any]:
    a, b, layout, ref, worklist = make_case(shape, actual_ms, device, args.m_alignment)
    kernel_args = (a, b, worklist)
    bound = kernel.bind(kernel_args)
    bound.env.config_spec.cute_tcgen05_search_enabled = True

    def helion_fn() -> torch.Tensor:
        return bound(*kernel_args)

    helion_graph, held, helion_check = capture_and_check(
        helion_fn,
        ref,
        layout,
        args,
        require_zero_padding=True,
        track_cute=True,
    )
    checks = {"helion_graph": helion_check}
    deep_graph: Any | None = None
    deep_held: list[Any] = []
    if deep_gemm is not None:
        deep_module = deep_gemm
        deep_out = torch.empty_like(ref)

        def deep_fn() -> torch.Tensor:
            deep_module.m_grouped_bf16_gemm_nt_contiguous(a, b, deep_out, layout)
            return deep_out

        deep_graph, deep_held, checks["deepgemm_graph"] = capture_and_check(
            deep_fn,
            ref,
            layout,
            args,
            require_zero_padding=False,
        )
    if not all(check["ok"] for check in checks.values()):
        raise RuntimeError(f"correctness failed for official row {shape.row_index}")
    valid_m = sum(actual_ms)
    valid_flops = 2 * valid_m * shape.n * shape.k
    graphs = (helion_graph,) if deep_graph is None else (helion_graph, deep_graph)
    timings = graph_timings(graphs, args, valid_flops)
    helion_timing = timings[0]
    deep_timing = timings[1] if deep_graph is not None else None
    row: dict[str, Any] = {
        "row_index": shape.row_index,
        "shape": shape._asdict(),
        "actual_ms": list(actual_ms),
        "work_tile_metadata": worklist.cpu().tolist(),
        "selected_config": dict(kernel.configs[0]),
        "correctness": checks,
        "helion_graph_timing": helion_timing,
    }
    if deep_timing is not None:
        row["deepgemm_graph_timing"] = deep_timing
        row["ratio_helion_over_deepgemm"] = (
            helion_timing["median_us"] / deep_timing["median_us"]
        )
    return row


def run(args: argparse.Namespace) -> int:
    selected_rows = parse_rows(args.rows)
    artifact_dir = (args.artifact_dir or Path(tempfile.mkdtemp())).resolve()
    output_path = (args.json_output or artifact_dir / "selected_results.json").resolve()
    cache_root = (args.cache_root or artifact_dir / "caches").resolve()
    if cache_root.exists() and (not cache_root.is_dir() or any(cache_root.iterdir())):
        raise ValueError(f"refusing to reuse nonempty cache directory: {cache_root}")
    output: dict[str, Any] = {
        "artifact_dir": artifact_dir,
        "json_output": output_path,
        "official_shapes": [shape._asdict() for shape in OFFICIAL_SHAPES],
        "selected_rows": selected_rows,
        "rng_policy": "seed 0; one DeepGEMM-compatible stream across official rows",
        "timing_methodology": (
            "paired cold-L2 CUDA-event graph timings; each sample is the mean of "
            "consecutive replays; first implementation alternates per sample"
            if args.compare_deepgemm
            else "single-implementation cold-L2 CUDA-event graph timings"
        ),
        "settings": vars(args),
        "helion": {
            "head": git_output(REPO_ROOT, "rev-parse", "HEAD"),
            "dirty": bool(git_output(REPO_ROOT, "status", "--porcelain=v1")),
            "benchmark_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "rows": [],
    }
    os.environ["HELION_BACKEND"] = "cute"
    os.environ["HELION_CUTE_MMA_IMPL"] = "tcgen05"
    for name, subdir in CACHE_DIRS.items():
        os.environ[name] = str(cache_root / subdir)
    output["environment"] = {
        key: value
        for key, value in os.environ.items()
        if key.startswith("HELION_")
        or key in CACHE_DIRS
        or key.endswith(("CUDA_HOME", "CUDA_VISIBLE_DEVICES"))
    }
    if (
        min(args.samples, args.iters) <= 0
        or min(
            args.warmups,
            args.compile_warmups,
            args.thermal_warmup_ms,
            args.l2_flush_bytes,
        )
        < 0
    ):
        raise ValueError(
            "samples/iters must be positive; warmups and L2 flush must be non-negative"
        )
    if args.compare_deepgemm and args.samples % 2:
        raise ValueError("--samples must be even when comparing DeepGEMM")
    if args.l2_flush_bytes % 4:
        raise ValueError("--l2-flush-bytes must be divisible by 4")
    if args.m_alignment <= 0 or args.m_alignment % M_ALIGNMENT:
        raise ValueError(f"--m-alignment must be a positive multiple of {M_ALIGNMENT}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    requested = torch.device(args.device)
    if requested.type != "cuda":
        raise RuntimeError(f"expected CUDA device, got {requested}")
    if requested.index is not None:
        torch.cuda.set_device(requested)
    device = torch.device("cuda", torch.cuda.current_device())
    capability = torch.cuda.get_device_capability(device)
    if capability[0] < 10:
        raise RuntimeError(f"compute capability {capability} is unsupported")
    output.update(
        device=str(device),
        device_name=torch.cuda.get_device_name(device),
        capability=capability,
        torch=torch.__version__,
        torch_cuda=torch.version.cuda,
    )
    deep_gemm = None
    if args.compare_deepgemm:
        if args.deepgemm_root is None:
            raise RuntimeError("--compare-deepgemm requires --deepgemm-root")
        deep_gemm, output["deepgemm"] = import_deepgemm(
            args.deepgemm_root, args.m_alignment
        )
    torch.manual_seed(0)
    actual_ms = official_actual_ms()
    kernel = helion.kernel(
        static_shapes=False, key=selected_key, config=selected_config()
    )(selected_kernel_body)
    with torch.cuda.device(device):
        for row_index in selected_rows:
            row = run_shape(
                args,
                OFFICIAL_SHAPES[row_index],
                actual_ms[row_index],
                device,
                kernel,
                deep_gemm,
            )
            output["rows"].append(row)
            print(json.dumps(row, default=str, sort_keys=True), flush=True)
            torch.cuda.empty_cache()
    ratios = [row.get("ratio_helion_over_deepgemm") for row in output["rows"]]
    ratios = [float(value) for value in ratios if value is not None]
    if ratios:
        output["deepgemm_comparison_summary"] = {
            "geomean_ratio_helion_over_deepgemm": statistics.geometric_mean(ratios),
            "helion_faster_rows": sum(value < 1 for value in ratios),
            "rows": len(ratios),
        }
    output["status"] = "ok"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, default=str, indent=2, sort_keys=True) + "\n"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
