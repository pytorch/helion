# ruff: noqa: ANN401
"""Reproduce the Helion versus CUTLASS CuTeDSL grouped-GEMM comparison.

The comparison is intentionally narrow: FP16 NT grouped GEMM, a 128x64 MMA
tile, a 1x1 cluster, and the seven published workloads below.  Every workload
runs in a fresh subprocess with fresh compiler caches.  Both implementations
consume the same A/B tensors, write separate outputs, and are timed through
CUDA graph replay.  The CUTLASS graph includes the pinned-host pointer-table
upload performed by the published reference wrapper before each kernel launch.

Use ``grouped_gemm.py`` from NVIDIA/cutlass commit
``db1c288993354c88e551c40c19a8fb93a774a241``::

    CUDA_VISIBLE_DEVICES=0 python benchmarks/cute/compare_grouped_gemm_backends.py \
      --cutlass-source /path/to/CUTLASS/examples/python/CuTeDSL/blackwell/grouped_gemm.py \
      --out-dir grouped_gemm_cutlass_results --stream-subprocesses
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import linecache
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from types import ModuleType
from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Mapping
    from collections.abc import Sequence

    import torch

hl: Any

REPO_ROOT = Path(__file__).resolve().parents[2]
CUTLASS_COMMIT = "db1c288993354c88e551c40c19a8fb93a774a241"
CUTLASS_SHA256 = "05b74a05682c024557d83284e32f973ed5be4f0d1a1a12c72fe7824c29f7e94f"
HELION_CUTE_MMA_IMPL = "tcgen05"
DEFAULT_OUT_DIR = Path("grouped_gemm_cutlass_results")
CTA_M, CTA_N, CTA_K = 128, 64, 64


@dataclass(frozen=True)
class Case:
    name: str
    problems: tuple[tuple[int, int, int, int], ...]
    reserved_sms: int
    ab_stages: int | None

    @property
    def shape_label(self) -> str:
        shapes = ", ".join(
            f"g{i}: {m}x{n}x{k}" for i, (m, n, k, _l) in enumerate(self.problems)
        )
        return f"G{len(self.problems)} [{shapes}]"


def _doc_case(name: str, m1: int, n3: int, *, reserved_sms: int = 0) -> Case:
    return Case(
        name,
        (
            (8192, 1280, 32, 1),
            (m1, 384, 1536, 1),
            (640, 1280, 16, 1),
            (640, n3, 16, 1),
        ),
        reserved_sms=reserved_sms,
        ab_stages=4,
    )


CASES = (
    Case(
        "default_small_parity",
        ((128, 128, 128, 1), (512, 128, 128, 1), (128, 256, 128, 1)),
        reserved_sms=0,
        ab_stages=None,
    ),
    _doc_case("doc_no_mn_tail", 128, 128),
    _doc_case("doc_no_mn_g3_192_extra_full", 128, 192),
    _doc_case("doc_original", 16, 160, reserved_sms=3),
    _doc_case("doc_mtail_g1_g3_192_extra_full", 16, 192, reserved_sms=3),
    _doc_case("doc_mtail_g1_only", 16, 128),
    _doc_case("doc_ntail_g3_160", 128, 160),
)
CASES_BY_NAME = {case.name: case for case in CASES}


@dataclass
class PreparedLaunch:
    call: Callable[[], object]
    owners: tuple[object, ...]

    def __call__(self) -> object:
        return self.call()


def _make_helion_kernel() -> tuple[Any, Any]:
    global hl, torch
    import torch

    import helion
    import helion.language as hl

    @helion.kernel(backend="cute")
    def grouped_gemm(
        a_placeholder: Any,
        b_placeholder: Any,
        layout: Any,
        n_sizes: Any,
        k_sizes: Any,
        out_placeholder: Any,
        direct_pointers: Any,
        direct_strides: Any,
    ) -> Any:
        m, max_k = a_placeholder.size()
        _groups, max_n, _k = b_placeholder.size()
        for tile_m, tile_n in hl.tile([m, max_n]):
            group_id = layout[tile_m.begin]
            safe_group_id = torch.where(group_id >= 0, group_id, 0)
            valid_rows = layout[tile_m] == safe_group_id
            valid_cols = tile_n.index < n_sizes[safe_group_id]
            valid = valid_rows[:, None] & valid_cols[None, :]  # pyrefly: ignore[bad-index]
            group_k = k_sizes[safe_group_id]
            acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
            for tile_k in hl.tile(max_k):  # pyrefly: ignore[bad-assignment]
                valid_k = (tile_k.index < group_k)[None, :]  # pyrefly: ignore[bad-index]
                a_tile = a_placeholder[tile_m, tile_k]
                b_tile = b_placeholder[safe_group_id, tile_n, tile_k]
                a_tile = torch.where(valid_k, a_tile, torch.zeros_like(a_tile))
                b_tile = torch.where(valid_k, b_tile, torch.zeros_like(b_tile))
                acc = torch.addmm(acc, a_tile, b_tile.T)
            old = out_placeholder[tile_m, tile_n]
            out_placeholder[tile_m, tile_n] = torch.where(
                valid, acc.to(out_placeholder.dtype), old
            )
        return out_placeholder

    return grouped_gemm, helion


def _placeholder(shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
    import torch

    base = torch.empty(max(1, shape[-1]), device=device, dtype=torch.float16)
    return torch.as_strided(base, shape, (0,) * (len(shape) - 1) + (1,))


def _prepare_helion(
    case: Case,
    group_a: tuple[torch.Tensor, ...],
    group_b: tuple[torch.Tensor, ...],
    outputs: tuple[torch.Tensor, ...],
) -> tuple[PreparedLaunch, dict[str, object]]:
    import torch

    kernel, helion = _make_helion_kernel()
    problems = case.problems
    device = group_a[0].device
    aligned_m = tuple((m + CTA_M - 1) // CTA_M * CTA_M for m, _n, _k, _l in problems)
    padded_m = sum(aligned_m)
    max_n = max(n for _m, n, _k, _l in problems)
    max_k = max(k for _m, _n, k, _l in problems)
    layout = torch.empty(padded_m, device=device, dtype=torch.int32)
    cursor = 0
    for group, ((m, _n, _k, _l), padded) in enumerate(
        zip(problems, aligned_m, strict=True)
    ):
        layout[cursor : cursor + m].fill_(group)
        layout[cursor + m : cursor + padded].fill_(-1)
        cursor += padded
    n_sizes = torch.tensor([p[1] for p in problems], device=device, dtype=torch.int32)
    k_sizes = torch.tensor([p[2] for p in problems], device=device, dtype=torch.int32)
    direct_pointers = torch.tensor(
        [
            (a.data_ptr(), b.data_ptr(), out.data_ptr())
            for a, b, out in zip(group_a, group_b, outputs, strict=True)
        ],
        device=device,
        dtype=torch.int64,
    )
    direct_strides = torch.tensor(
        [
            (tuple(a.stride()), tuple(b.stride()), tuple(out.stride()))
            for a, b, out in zip(group_a, group_b, outputs, strict=True)
        ],
        device=device,
        dtype=torch.int32,
    )
    kernel_args = (
        _placeholder((padded_m, max_k), device),
        _placeholder((len(problems), max_n, max_k), device),
        layout,
        n_sizes,
        k_sizes,
        _placeholder((padded_m, max_n), device),
        direct_pointers,
        direct_strides,
    )
    config = helion.Config(
        block_sizes=[CTA_M, CTA_N, CTA_K],
        l2_groupings=[1],
        loop_orders=[[0, 1]],
        num_stages=2,
        num_warps=8,
        pid_type="persistent_interleaved",
        tcgen05_cluster_m=1,
        tcgen05_cluster_n=1,
        tcgen05_acc_stages=2,
        tcgen05_c_stages=2,
        tcgen05_num_epi_warps=4,
        tcgen05_grouped_mode="direct",
        tcgen05_grouped_external_direct_pointers="direct_pointers",
        tcgen05_grouped_external_direct_strides="direct_strides",
        **(
            {"tcgen05_grouped_static_reserved_sms": case.reserved_sms}
            if case.reserved_sms
            else {}
        ),
    )
    if case.ab_stages is not None:
        config.config["tcgen05_ab_stages"] = case.ab_stages
    bound = kernel.bind(kernel_args)
    bound.env.config_spec.cute_tcgen05_search_enabled = True
    bound.set_config(config)

    def launch() -> object:
        with torch.cuda.device(device):
            return bound(*kernel_args)

    owners: tuple[object, ...] = (*group_a, *group_b, *outputs, *kernel_args, bound)
    return PreparedLaunch(launch, owners), {
        "block_sizes": [CTA_M, CTA_N, CTA_K],
        "cluster_shape_mn": [1, 1],
        "reserved_sms": case.reserved_sms,
        "ab_stages": case.ab_stages,
    }


def _load_cutlass_source(source: Path) -> tuple[ModuleType, dict[str, str]]:
    """Hash and execute the retained bytes under an immutable synthetic name."""
    path = source.expanduser().resolve(strict=True)
    source_bytes = path.read_bytes()
    digest = hashlib.sha256(source_bytes).hexdigest()
    if digest != CUTLASS_SHA256:
        raise ValueError(
            f"CUTLASS source SHA256 mismatch: expected {CUTLASS_SHA256} from "
            f"commit {CUTLASS_COMMIT}, got {digest}"
        )
    module_name = f"_helion_cutlass_grouped_gemm_{CUTLASS_COMMIT[:12]}"
    filename = f"<helion-cutlass-grouped-gemm-{digest}.py>"
    source_text = source_bytes.decode("utf-8")
    linecache.cache[filename] = (
        len(source_bytes),
        None,
        source_text.splitlines(keepends=True),
        filename,
    )
    module = ModuleType(module_name)
    module.__file__ = filename
    sys.modules[module_name] = module
    try:
        exec(compile(source_bytes, filename, "exec"), vars(module))
    except BaseException:
        sys.modules.pop(module_name, None)
        linecache.cache.pop(filename, None)
        raise
    return module, {
        "source_path": str(path),
        "source_sha256": digest,
        "expected_cutlass_commit": CUTLASS_COMMIT,
        "cutlass_dsl_version": importlib.metadata.version("nvidia-cutlass-dsl"),
    }


def _prepare_cutlass(
    module: ModuleType,
    problems: tuple[tuple[int, int, int, int], ...],
    group_a: tuple[torch.Tensor, ...],
    group_b: tuple[torch.Tensor, ...],
    outputs: tuple[torch.Tensor, ...],
) -> PreparedLaunch:
    import cutlass
    import cutlass.cute as cute
    import cutlass.torch as cutlass_torch
    import cutlass.utils as cutlass_utils
    import torch

    from helion.runtime import _ensure_cute_dsl_arch_env

    _ensure_cute_dsl_arch_env(())
    initial = tuple(
        module.create_tensor_and_stride(1, 8, 8, False, cutlass.Float16)[2]
        for _ in range(3)
    )
    dims, dims_torch = cutlass_torch.cute_tensor_like(
        torch.tensor(problems, dtype=torch.int32),
        cutlass.Int32,
        is_dynamic_layout=False,
        assumed_align=16,
    )
    strides_data = [
        (tuple(a.stride()), tuple(b.stride()), tuple(out.stride()))
        for a, b, out in zip(group_a, group_b, outputs, strict=True)
    ]
    strides, strides_torch = cutlass_torch.cute_tensor_like(
        torch.tensor(strides_data, dtype=torch.int32),
        cutlass.Int32,
        is_dynamic_layout=False,
        assumed_align=16,
    )
    device = group_a[0].device
    pointers_host = torch.tensor(
        [
            (a.data_ptr(), b.data_ptr(), out.data_ptr())
            for a, b, out in zip(group_a, group_b, outputs, strict=True)
        ],
        dtype=torch.int64,
        pin_memory=True,
    )
    pointers_torch = torch.empty_like(pointers_host, device=device)
    pointers = cutlass_torch.from_dlpack(pointers_torch, assumed_align=16)
    pointers.element_type = cutlass.Int64
    hardware = cutlass_utils.HardwareInfo()
    max_active_clusters = hardware.get_max_active_clusters(1)
    tensormap_torch = torch.empty(
        (
            max_active_clusters,
            module.GroupedGemmKernel.num_tensormaps,
            module.GroupedGemmKernel.bytes_per_tensormap // 8,
        ),
        device=device,
        dtype=torch.int64,
    )
    tensormap = cutlass_torch.from_dlpack(tensormap_torch, assumed_align=16)
    tensormap.element_type = cutlass.Int64
    kernel = module.GroupedGemmKernel(
        cutlass.Float32,
        False,
        (CTA_M, CTA_N),
        (1, 1),
        cutlass_utils.TensorMapUpdateMode.SMEM,
    )
    total_clusters = sum(
        ((m + CTA_M - 1) // CTA_M) * ((n + CTA_N - 1) // CTA_N)
        for m, n, _k, _l in problems
    )
    compiled = cute.compile(
        kernel,
        *initial,
        len(problems),
        dims,
        strides,
        pointers,
        total_clusters,
        tensormap,
        max_active_clusters,
        cutlass_torch.default_stream(),
        options="--opt-level 2",
    )

    def launch() -> object:
        pointers_torch.copy_(pointers_host, non_blocking=True)
        return compiled(
            *initial,
            dims,
            strides,
            pointers,
            tensormap,
            cutlass_torch.current_stream(),
        )

    owners: tuple[object, ...] = (
        compiled,
        *initial,
        dims,
        strides,
        pointers,
        tensormap,
        dims_torch,
        strides_torch,
        pointers_host,
        pointers_torch,
        tensormap_torch,
        *group_a,
        *group_b,
        *outputs,
    )
    return PreparedLaunch(launch, owners)


def _make_inputs(
    problems: Sequence[tuple[int, int, int, int]], device: torch.device
) -> tuple[
    tuple[torch.Tensor, ...], tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]
]:
    import torch

    group_a = tuple(
        torch.randn(m, k, device=device, dtype=torch.float16)
        for m, _n, k, _l in problems
    )
    group_b = tuple(
        torch.randn(n, k, device=device, dtype=torch.float16)
        for _m, n, k, _l in problems
    )
    expected = tuple(
        (a.float() @ b.float().T).half() for a, b in zip(group_a, group_b, strict=True)
    )
    return group_a, group_b, expected


def _outputs(
    group_a: Sequence[torch.Tensor], group_b: Sequence[torch.Tensor]
) -> tuple[torch.Tensor, ...]:
    import torch

    return tuple(
        torch.empty(a.size(0), b.size(0), device=a.device, dtype=torch.float16)
        for a, b in zip(group_a, group_b, strict=True)
    )


def _capture(
    launch: PreparedLaunch, warmups: int, *, track_cute: bool = False
) -> torch.cuda.CUDAGraph:
    import torch

    import helion.runtime as helion_runtime

    for _ in range(warmups):
        launch()
    torch.cuda.synchronize()
    if track_cute:
        with helion_runtime.cute_cuda_graph() as graph:
            launch()
    else:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            launch()
    torch.cuda.synchronize()
    return graph


def _correctness(
    outputs: Sequence[torch.Tensor], expected: Sequence[torch.Tensor]
) -> dict[str, object]:
    import torch

    max_abs = 0.0
    for actual, reference in zip(outputs, expected, strict=True):
        max_abs = max(max_abs, float((actual.float() - reference.float()).abs().max()))
        torch.testing.assert_close(actual, reference, rtol=2e-2, atol=2e-2)
    return {"ok": True, "max_abs": max_abs}


def _thermal_warmup(duration_ms: int) -> None:
    import torch

    if duration_ms <= 0:
        return
    value = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    end = time.monotonic() + duration_ms / 1000
    while time.monotonic() < end:
        for _ in range(50):
            value = value @ value
        torch.cuda.synchronize()


def _do_bench(fn: Callable[[], object], args: argparse.Namespace) -> float:
    from triton.testing import do_bench

    value: Any = do_bench(
        fn, warmup=args.warmup_ms, rep=args.rep_ms, return_mode="mean"
    )
    return float(value[0] if isinstance(value, tuple) else value)


def _bench_pair(
    replays: Mapping[str, Callable[[], object]], args: argparse.Namespace
) -> dict[str, dict[str, Any]]:
    import torch

    if args.cache_warmup_calls:
        for fn in replays.values():
            for _ in range(args.cache_warmup_calls):
                fn()
        torch.cuda.synchronize()
    _thermal_warmup(args.thermal_warmup_ms)
    names = tuple(replays)
    samples = {name: [] for name in names}
    for run in range(args.num_runs):
        order = names if run % 2 == 0 else names[::-1]
        for name in order:
            samples[name].append(_do_bench(replays[name], args))
    method = (
        "one shared cache/thermal warmup phase, then paired external "
        "triton.testing.do_bench CUDA-event runs alternating Helion/CUTLASS "
        "and CUTLASS/Helion"
    )
    return {
        name: {
            "median_ms": statistics.median(values),
            "runs_ms": values,
            "method": method,
        }
        for name, values in samples.items()
    }


def _configure_worker(case_dir: Path) -> dict[str, str]:
    cache_names = (
        "HELION_CACHE_DIR",
        "CUTE_DSL_CACHE_DIR",
        "TORCHINDUCTOR_CACHE_DIR",
        "TRITON_CACHE_DIR",
        "CUDA_CACHE_PATH",
    )
    os.environ.update(
        {
            "HELION_BACKEND": "cute",
            "HELION_CUTE_MMA_IMPL": HELION_CUTE_MMA_IMPL,
            "CUTE_DSL_KEEP": "ir,ptx,cubin",
            "CUTE_DSL_DUMP_DIR": str(case_dir / "dump"),
        }
    )
    (case_dir / "dump").mkdir(parents=True, exist_ok=True)
    for name in cache_names:
        path = case_dir / "cache" / name.lower()
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
    keys = (
        "CUDA_VISIBLE_DEVICES",
        "HELION_BACKEND",
        "HELION_CUTE_MMA_IMPL",
        *cache_names,
    )
    return {key: os.environ.get(key, "") for key in keys}


def _run_case(args: argparse.Namespace) -> int:
    import torch

    if not args.case or len(args.case) != 1:
        raise ValueError("--run-case requires exactly one --case")
    case = CASES_BY_NAME[args.case[0]]
    case_dir = args.out_dir / "cutlass" / case.name
    if case_dir.exists() and any(case_dir.iterdir()):
        raise ValueError(f"refusing to reuse nonempty case directory: {case_dir}")
    case_dir.mkdir(parents=True, exist_ok=True)
    env = _configure_worker(case_dir)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(0)
    torch.manual_seed(args.seed)
    device = torch.device("cuda", 0)
    cutlass_module, provenance = _load_cutlass_source(args.cutlass_source)
    group_a, group_b, expected = _make_inputs(case.problems, device)
    helion_outputs = _outputs(group_a, group_b)
    cutlass_outputs = _outputs(group_a, group_b)

    helion_launch, helion_config = _prepare_helion(
        case, group_a, group_b, helion_outputs
    )
    cutlass_launch = _prepare_cutlass(
        cutlass_module, case.problems, group_a, group_b, cutlass_outputs
    )
    helion_graph = _capture(helion_launch, args.compile_warmups, track_cute=True)
    cutlass_graph = _capture(cutlass_launch, args.compile_warmups)

    helion_graph.replay()
    cutlass_graph.replay()
    torch.cuda.synchronize()
    correctness = {
        "helion_retained": _correctness(helion_outputs, expected),
        "cutlass": _correctness(cutlass_outputs, expected),
    }
    replay = {
        "helion_retained": helion_graph.replay,
        "cutlass": cutlass_graph.replay,
    }
    timings = _bench_pair(replay, args)
    ratio = float(timings["helion_retained"]["median_ms"]) / float(
        timings["cutlass"]["median_ms"]
    )
    result = {
        "comparison": "cutlass_cutedsl_blackwell_grouped_gemm",
        "case": case.name,
        "shape_label": case.shape_label,
        "problem_sizes": [list(problem) for problem in case.problems],
        "device_name": torch.cuda.get_device_name(device),
        "capability": list(torch.cuda.get_device_capability(device)),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cutlass_reference": provenance,
        "helion_config": helion_config,
        "settings": {
            "seed": args.seed,
            "compile_warmups": args.compile_warmups,
            "num_runs": args.num_runs,
            "warmup_ms": args.warmup_ms,
            "rep_ms": args.rep_ms,
            "cache_warmup_calls": args.cache_warmup_calls,
            "thermal_warmup_ms": args.thermal_warmup_ms,
            "fresh_subprocess_per_case": True,
            "cutlass_graph_includes_pointer_table_upload": True,
            "env": env,
        },
        "correctness": correctness,
        "timings": timings,
        "ratios_to_cutlass": {"helion_retained": ratio},
    }
    output = case_dir / "result.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"case": case.name, "result": str(output), "ratio": ratio}))
    return 0


def _geomean(values: Sequence[float]) -> float | None:
    return math.exp(sum(map(math.log, values)) / len(values)) if values else None


def _git(*args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=REPO_ROOT, text=True).strip()


def _build_summary(
    args: argparse.Namespace,
    cases: Sequence[Case],
    commands: Sequence[Mapping[str, object]],
    failures: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    rows = []
    for case in cases:
        path = args.out_dir / "cutlass" / case.name / "result.json"
        if not path.exists():
            rows.append({"case": case.name, "status": "missing", "path": str(path)})
            continue
        result = json.loads(path.read_text())
        rows.append(
            {
                "case": case.name,
                "shape_label": case.shape_label,
                "problem_sizes": result["problem_sizes"],
                "status": "ok",
                "timings": result["timings"],
                "retained_over_cutlass": result["ratios_to_cutlass"].get(
                    "helion_retained"
                ),
            }
        )
    ratios = [
        float(value)
        for row in rows
        if isinstance((value := row.get("retained_over_cutlass")), int | float)
    ]
    return {
        "artifact_dir": str(args.out_dir),
        "top_level_command": " ".join(sys.argv),
        "cutlass_source": str(args.cutlass_source),
        "cutlass_commit": CUTLASS_COMMIT,
        "cutlass_sha256": CUTLASS_SHA256,
        "helion_cute_mma_impl": HELION_CUTE_MMA_IMPL,
        "helion_git": {
            "head": _git("rev-parse", "HEAD"),
            "status_short": _git("status", "--short"),
            "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "methodology": (
            "fresh subprocess/case; common inputs; paired external do_bench CUDA "
            "graph replays; one shared cache/thermal warmup phase; alternate "
            "Helion/CUTLASS then CUTLASS/Helion; CUTLASS graph includes "
            "pinned-host pointer-table upload"
        ),
        "rows": rows,
        "wins": sum(ratio < 1 for ratio in ratios),
        "total": len(ratios),
        "geomean_helion_over_cutlass": _geomean(ratios),
        "commands": list(commands),
        "failures": list(failures),
    }


def _write_summary(args: argparse.Namespace, summary: Mapping[str, object]) -> None:
    json_path = args.out_dir / "summary.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"summary": str(json_path)}))


def _positive(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return value


def _non_negative(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("expected a non-negative integer")
    return value


def _positive_even(text: str) -> int:
    value = _positive(text)
    if value % 2:
        raise argparse.ArgumentTypeError("expected a positive even integer")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--list-cases", action="store_true")
    mode.add_argument("--run-case", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--case", choices=CASES_BY_NAME, action="append")
    parser.add_argument("--cutlass-source", type=Path)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--cuda-visible-devices")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compile-warmups", type=_positive, default=2)
    parser.add_argument("--num-runs", type=_positive_even, default=6)
    parser.add_argument("--warmup-ms", type=_non_negative, default=1000)
    parser.add_argument("--rep-ms", type=_positive, default=500)
    parser.add_argument("--cache-warmup-calls", type=_non_negative, default=5)
    parser.add_argument("--thermal-warmup-ms", type=_non_negative, default=10000)
    parser.add_argument("--stream-subprocesses", action="store_true")
    parser.add_argument(
        "--json", action="store_true", help="JSON output for --list-cases"
    )
    return parser


def _selected_cases(args: argparse.Namespace) -> tuple[Case, ...]:
    return tuple(CASES_BY_NAME[name] for name in args.case) if args.case else CASES


def _worker_command(args: argparse.Namespace, case: Case) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--run-case",
        "--case",
        case.name,
    ]
    forwarded = (
        ("cutlass-source", args.cutlass_source),
        ("out-dir", args.out_dir),
        ("seed", args.seed),
        ("compile-warmups", args.compile_warmups),
        ("num-runs", args.num_runs),
        ("warmup-ms", args.warmup_ms),
        ("rep-ms", args.rep_ms),
        ("cache-warmup-calls", args.cache_warmup_calls),
        ("thermal-warmup-ms", args.thermal_warmup_ms),
    )
    for name, value in forwarded:
        if value is not None:
            command.extend((f"--{name}", str(value)))
    return command


def _run_all(args: argparse.Namespace) -> int:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cases = _selected_cases(args)
    env = os.environ.copy()
    repo_path = str(REPO_ROOT)
    env["PYTHONPATH"] = repo_path + os.pathsep + env.get("PYTHONPATH", "")
    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    commands = []
    failures = []
    for case in cases:
        command = _worker_command(args, case)
        print(json.dumps({"starting": case.name}), flush=True)
        proc = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        stdout_path = args.out_dir / f"cutlass_{case.name}.stdout.txt"
        stderr_path = args.out_dir / f"cutlass_{case.name}.stderr.txt"
        stdout_path.write_text(proc.stdout)
        stderr_path.write_text(proc.stderr)
        if args.stream_subprocesses:
            print(proc.stdout, end="")
            print(proc.stderr, end="", file=sys.stderr)
        record = {
            "case": case.name,
            "cmd": command,
            "returncode": proc.returncode,
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
        }
        commands.append(record)
        if proc.returncode:
            failures.append(record)
            if not args.stream_subprocesses:
                print(proc.stderr, file=sys.stderr)
    _write_summary(args, _build_summary(args, cases, commands, failures))
    return int(bool(failures))


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    args.out_dir = args.out_dir.expanduser().resolve()
    if args.list_cases:
        cases = _selected_cases(args)
        payload = [
            {
                "name": case.name,
                "shape_label": case.shape_label,
                "problems": case.problems,
            }
            for case in cases
        ]
        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            for case in cases:
                print(f"{case.name}: {case.shape_label}")
        return 0
    if args.cutlass_source is None:
        parser.error("--cutlass-source is required")
    args.cutlass_source = args.cutlass_source.expanduser().resolve()
    return _run_case(args) if args.run_case else _run_all(args)


if __name__ == "__main__":
    raise SystemExit(main())
