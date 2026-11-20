"""Launch the MXFP8 study and write the latency-by-shape plot.

The study times one packed-scale GEMM body at three Helion commits:

* ``8781adf88`` parent Helion-Triton, before the packed lowering
* ``71a07049f`` packed descriptor / UTCCP lowering
* ``9192f5909`` split-K and swapped operands, plus Helion-CuTe

Each revision is checked out in a worktree, reinstalled with
``pip install -e --no-deps``, and then the kernels are launched. Triton uses
that commit's ``autotune_effort="none"`` seed. CuTe uses the saved SM103
configs. ``--tune quick`` (or ``full``) adds a Helion-Triton series that runs
Helion's autotuner at the final commit; the seed is part of the search space.
Timing: FlashInfer CUPTI timestamps, a CUDA graph, a cold L2 cache, 10 s of
thermal warmup, and the median of five runs.

Every latency in the CSV and PNG comes from this run; nothing is hardcoded.
"""

from __future__ import annotations

import argparse
import csv
import functools
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import TYPE_CHECKING

from plot_mxfp8_latency_by_shape import write_latency_plot
import torch
import torch.nn.functional as F

import helion
from helion.autotuner import PowerOfTwoFragment
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Callable

_REPO = Path(__file__).resolve().parents[2]
_ARTIFACTS = _REPO / "artifacts" / "mxfp8_dense_gemm_repro"
# Parent, packed lowering, then split-K + operand swap. CuTe is timed on the
# last tree; its MXFP8 sources match the packed-lowering commit.
_REVISIONS = (
    ("parent", "8781adf88d9c88924127fcd65c53ae07ed7bc2f4", ("triton",)),
    ("packed", "71a07049f1e328f743920c02ffb7c3cf074a0b3e", ("triton",)),
    (
        "optimized",
        "9192f5909cf11a372522e71d9566b59e561230d5",
        ("triton", "cute"),
    ),
)
# (series, revision label, backend)
_SERIES = (
    ("Parent Helion-Triton", "parent", "triton"),
    ("Packed lowering Helion-Triton", "packed", "triton"),
    ("Optimized Helion-Triton", "optimized", "triton"),
    ("Helion-CuTe", "optimized", "cute"),
)
_TUNED_SERIES = "Tuned Helion-Triton"
_SHAPES_MN = (
    (1, 1792, 5120),
    (8, 8192, 1280),
    (16, 2048, 4096),
    (32, 5120, 2048),
    (128, 1792, 5120),
    (1024, 8192, 1280),
)


def mxfp8_dense_gemm_body(
    a: torch.Tensor,
    weights: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
) -> torch.Tensor:
    """Return BF16 ``dequant(a) @ dequant(weights).T`` with FP32 accumulation."""
    m, k = a.shape
    n, weight_k = weights.shape
    assert a.dtype == weights.dtype == torch.float8_e4m3fn
    assert scale_a.dtype in (torch.uint8, torch.float8_e8m0fnu, torch.int32)
    assert scale_b.dtype in (torch.uint8, torch.float8_e8m0fnu, torch.int32)
    assert a.device == weights.device == scale_a.device == scale_b.device
    assert a.is_contiguous() and weights.is_contiguous()
    assert m > 0 and n > 0 and k > 0 and k % 32 == 0 and k == weight_k
    packed_a = hl.constexpr(scale_a.dtype == torch.int32)
    packed_b = hl.constexpr(scale_b.dtype == torch.int32)
    if packed_a:
        assert scale_a.shape == (m, (k // 32 + 3) // 4)
        assert scale_a.stride(0) == 1 and scale_a.stride(1) >= m
        assert scale_a.stride(1) % 4 == 0
    else:
        assert scale_a.is_contiguous() and scale_a.shape == (m, k // 32)
        scale_a = scale_a.view(torch.uint8)
    if packed_b:
        assert scale_b.shape == (n, (k // 32 + 3) // 4)
        assert scale_b.stride(0) == 1 and scale_b.stride(1) >= n
        assert scale_b.stride(1) % 4 == 0
    else:
        assert scale_b.is_contiguous() and scale_b.shape == (n, k // 32)
        scale_b = scale_b.view(torch.uint8)
    hl.specialize(m)
    hl.specialize(n)
    hl.specialize(k)
    out = torch.empty((m, n), dtype=torch.bfloat16, device=a.device)
    block_m = hl.register_block_size(16, 128)
    block_n = hl.register_block_size(16, 256)
    block_groups = hl.register_block_size(2, 32)
    split_k = hl.register_tunable("split_k", PowerOfTwoFragment(1, 16, 1))
    groups_per_split = (k // 32 + split_k - 1) // split_k
    partial = torch.empty((split_k, m, n), dtype=torch.float32, device=a.device)
    flat_out = out.view(m * n)
    flat_partial = partial.view(split_k * m * n)
    for task_m, task_n, part in hl.grid(
        [(m + block_m - 1) // block_m, (n + block_n - 1) // block_n, split_k]
    ):
        rows = task_m * block_m + hl.arange(block_m)
        cols = task_n * block_n + hl.arange(block_n)
        acc = hl.zeros([block_m, block_n], dtype=torch.float32)
        group_end = min((part + 1) * groups_per_split, k // 32)
        for groups in hl.tile(
            part * groups_per_split, group_end, block_size=block_groups
        ):
            columns = groups.begin * 32 + hl.arange(block_groups * 32)
            lhs = hl.load(
                a,
                [rows, columns],
                extra_mask=(rows[:, None] < m) & (columns[None, :] < group_end * 32),
            )
            rhs = hl.load(
                weights,
                [cols, columns],
                extra_mask=(cols[:, None] < n) & (columns[None, :] < group_end * 32),
            )
            if packed_a:
                lhs_words = hl.load(
                    scale_a,
                    [rows, groups.index // 4],
                    extra_mask=(rows[:, None] < m)
                    & (groups.index[None, :] < group_end),
                )
                shift = ((groups.index % 4) * 8)[None, :]
                lhs_scale = ((lhs_words >> shift) & 255).to(torch.uint8)
            else:
                lhs_scale = hl.load(
                    scale_a, [rows, groups.index], extra_mask=rows[:, None] < m
                )
            if packed_b:
                rhs_words = hl.load(
                    scale_b,
                    [cols, groups.index // 4],
                    extra_mask=(cols[:, None] < n)
                    & (groups.index[None, :] < group_end),
                )
                shift = ((groups.index % 4) * 8)[None, :]
                rhs_scale = ((rhs_words >> shift) & 255).to(torch.uint8)
            else:
                rhs_scale = hl.load(
                    scale_b, [cols, groups.index], extra_mask=cols[:, None] < n
                )
            acc = hl.dot_scaled(
                lhs, lhs_scale, "e4m3", rhs.T, rhs_scale, "e4m3", acc=acc
            )
        if split_k == 1:
            hl.store(
                flat_out,
                [rows[:, None] * n + cols[None, :]],
                acc.to(out.dtype),
                extra_mask=(rows[:, None] < m) & (cols[None, :] < n),
            )
        else:
            hl.store(
                flat_partial,
                [(part * m + rows[:, None]) * n + cols[None, :]],
                acc,
                extra_mask=(rows[:, None] < m) & (cols[None, :] < n),
            )
    if split_k > 1:
        return partial.sum(dim=0).to(torch.bfloat16)
    return out


_COMMON = {
    "block_sizes": [16, 16, 16],
    "loop_orders": [[0, 1, 2]],
    "load_eviction_policies": ["", "", "", ""],
    "cute_store_policy": "default",
    "cute_pdl": False,
    "cute_vector_widths": [1, 1, 1, 1, 1, 1],
    "cute_lane_layouts": ["blocked"] * 6,
    "cute_collective_mma": False,
    "cute_collective_compute": "warp",
    "cute_collective_native_seeded": False,
    "cute_collective_tmem_seed": False,
    "cute_collective_tmem_a": False,
    "cute_collective_stages": 1,
    "cute_collective_copy": "scalar",
    "cute_collective_recipe": "scalar",
    "cute_collective_operand_packets": False,
    "cute_collective_epilogue": "scalar",
    "cute_gathered_mma_n": 256,
    "cute_gathered_mma_stages": 3,
    "cute_register_chain": False,
    "cute_scaled_mma": True,
    "cute_scaled_fused_scales": True,
    "cute_scaled_sf_tma": True,
    "cute_scaled_sf_inplace": True,
    "split_k": 1,
}


def _scaled(
    tile_m: int,
    tile_n: int,
    tile_k: int,
    stages: int,
    *,
    cluster_m: int = 1,
    persistent: bool = True,
    swap_ab: bool = True,
    compact_m: bool = True,
    epilogue: str = "default",
) -> dict[str, object]:
    return _COMMON | {
        "cute_scaled_tile_m": tile_m,
        "cute_scaled_tile_n": tile_n,
        "cute_scaled_tile_k": tile_k,
        "cute_scaled_stages": stages,
        "cute_scaled_cluster_m": cluster_m,
        "cute_scaled_persistent": persistent,
        "cute_scaled_swap_ab": swap_ab,
        "cute_scaled_compact_m": compact_m,
        "cute_scaled_epilogue_mode": epilogue,
    }


CUTE_CONFIGS: dict[tuple[int, int, int], dict[str, object]] = {
    (1, 1792, 5120): _scaled(32, 16, 256, 8, persistent=False),
    (8, 8192, 1280): _scaled(64, 8, 256, 5),
    (16, 2048, 4096): _scaled(32, 16, 256, 6),
    (32, 5120, 2048): _scaled(64, 32, 256, 6),
    (128, 1792, 5120): _scaled(64, 32, 256, 8, swap_ab=False),
    (1024, 8192, 1280): _scaled(
        128,
        224,
        128,
        6,
        cluster_m=2,
        swap_ab=False,
        compact_m=False,
        epilogue="tma_prefetch_2",
    ),
}


def pack_scales(scales: torch.Tensor) -> torch.Tensor:
    """E8M0 bytes ``[R, K/32]`` to MN-major int32 ``[R, ceil(K/128)]``."""
    packed = F.pad(scales.view(torch.uint8), (0, (-scales.shape[1]) % 4))
    packed = packed.contiguous().view(torch.int32)
    aligned_rows = (packed.shape[0] + 3) // 4 * 4
    storage = torch.zeros(
        (packed.shape[1], aligned_rows), device=scales.device, dtype=torch.int32
    )
    result = storage.T[: packed.shape[0]]
    result.copy_(packed)
    return result


def make_inputs(
    m: int, n: int, k: int, *, seed: int = 0, device: str = "cuda"
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normal/3 values and E8M0 exponents 125..128, matching the recorded run."""
    generator = torch.Generator(device=device).manual_seed(seed)

    def normal(shape: tuple[int, int]) -> torch.Tensor:
        return torch.randn(shape, device=device, generator=generator)

    a = (normal((m, k)) / 3).to(torch.float8_e4m3fn)
    weights = (normal((n, k)) / 3).to(torch.float8_e4m3fn)
    scale_a, scale_b = (
        torch.randint(125, 129, (rows, k // 32), device=device, generator=generator).to(
            torch.uint8
        )
        for rows in (m, n)
    )
    return a, weights, pack_scales(scale_a), pack_scales(scale_b)


def mxfp8_dense_gemm_reference(
    a: torch.Tensor,
    weights: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
) -> torch.Tensor:
    """FP64 dequantization and matmul, returned as BF16."""

    def dequantize(data: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        groups = torch.arange(data.shape[1] // 32, device=scales.device)
        words = scales[:, groups // 4]
        encoded = ((words >> ((groups % 4) * 8)[None, :]) & 255).to(torch.uint8)
        exponents = encoded.to(torch.int32) - 127
        factors = torch.exp2(exponents.to(torch.float64)).repeat_interleave(32, dim=1)
        factors = torch.where(
            encoded.repeat_interleave(32, dim=1) == 255, float("nan"), factors
        )
        return data.to(torch.float64) * factors

    product = dequantize(a, scale_a) @ dequantize(weights, scale_b).T
    return product.bfloat16()


@functools.cache
def _triton_kernel(effort: str) -> helion.Kernel:
    return helion.kernel(
        mxfp8_dense_gemm_body,
        backend="triton",
        static_shapes=True,
        autotune_effort=effort,
        ignore_warnings=[helion.exc.TensorOperationInWrapper],
    )


@functools.cache
def _cute_kernel(m: int, n: int, k: int) -> helion.Kernel:
    return helion.kernel(
        mxfp8_dense_gemm_body,
        config=helion.Config(**CUTE_CONFIGS[m, n, k]),
        backend="cute",
        static_shapes=True,
        ignore_warnings=[helion.exc.TensorOperationInWrapper],
    )


def _launch(
    backend: str, effort: str, inputs: tuple[torch.Tensor, ...]
) -> torch.Tensor:
    if backend == "triton":
        return _triton_kernel(effort)(*inputs)
    m, k = inputs[0].shape
    n = inputs[1].shape[0]
    return _cute_kernel(m, n, k)(*inputs)


def thermal_warmup(seconds: float) -> None:
    left = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    end = time.time() + seconds
    while time.time() < end:
        for _ in range(50):
            left @ left
        torch.cuda.synchronize()


def steady_us(fn: Callable[[], object], *, warmup_s: float) -> float:
    """Median microseconds from five CUPTI CUDA-graph runs with a cold L2."""
    from flashinfer.testing.utils import bench_gpu_time

    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    thermal_warmup(warmup_s)
    runs = [
        statistics.median(
            bench_gpu_time(
                fn,
                enable_cupti=True,
                use_cuda_graph=True,
                cold_l2_cache=True,
                dry_run_time_ms=100,
                repeat_time_ms=500,
            )
        )
        for _ in range(5)
    ]
    return statistics.median(runs) * 1000


def _require_source(source_root: Path, expect_sha: str) -> None:
    installed = Path(helion.__file__).resolve()
    root = source_root.resolve()
    if root not in installed.parents:
        raise SystemExit(f"installed helion is {installed}, expected it under {root}")
    head = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != expect_sha:
        raise SystemExit(f"{root} is {head[:12]}, expected {expect_sha[:12]}")
    print(f"helion {head[:12]} from {installed}", flush=True)


def measure(args: argparse.Namespace) -> None:
    _require_source(args.source_root, args.expect_sha)
    import flashinfer.testing.utils as flashinfer_utils

    if not hasattr(flashinfer_utils, "bench_gpu_time"):
        raise SystemExit("flashinfer.testing.utils.bench_gpu_time is required")
    print(f"device: {torch.cuda.get_device_name()}", flush=True)
    results = args.results
    results.parent.mkdir(parents=True, exist_ok=True)
    for m, n, k in _SHAPES_MN:
        inputs = make_inputs(m, n, k)
        # The first call compiles, and autotunes when the effort is not "none".
        torch.testing.assert_close(
            _launch(args.measure, args.effort, inputs),
            mxfp8_dense_gemm_reference(*inputs),
            atol=0.02,
            rtol=0.016,
        )
        measured = steady_us(
            lambda bound=inputs: _launch(args.measure, args.effort, bound),
            warmup_s=args.warmup_s,
        )
        print(f"{args.series}: M={m} N={n} K={k} {measured:.2f} us", flush=True)
        row = {"series": args.series, "m": m, "n": n, "k": k, "us": measured}
        with results.open("a") as handle:
            handle.write(json.dumps(row) + "\n")


def _geomean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _run(cmd: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=cwd)


def _install(source: Path) -> None:
    _run([sys.executable, "-m", "pip", "install", "-e", str(source), "--no-deps"])


def _ensure_worktree(dest: Path, sha: str) -> None:
    if dest.exists():
        head = subprocess.check_output(
            ["git", "-C", str(dest), "rev-parse", "HEAD"], text=True
        ).strip()
        if head == sha:
            return
        _run(["git", "-C", str(_REPO), "worktree", "remove", "--force", str(dest)])
    dest.parent.mkdir(parents=True, exist_ok=True)
    _run(["git", "-C", str(_REPO), "worktree", "add", "--detach", str(dest), sha])


def _write_csv(results: Path, series: list[str], csv_path: Path) -> None:
    rows = [json.loads(line) for line in results.read_text().splitlines()]
    latest = {(r["series"], r["m"], r["n"], r["k"]): r["us"] for r in rows}
    print(f"{'series':<32} {'geomean_us':>10}")
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["series", "m", "n", "k", "us"])
        for name in series:
            values = []
            for m, n, k in _SHAPES_MN:
                if (name, m, n, k) not in latest:
                    raise SystemExit(
                        f"missing measurement for {name} M={m} N={n} K={k}"
                    )
                values.append(latest[name, m, n, k])
                writer.writerow([name, m, n, k, f"{values[-1]:.4f}"])
            print(f"{name:<32} {_geomean(values):10.2f}")


def reproduce(args: argparse.Namespace) -> None:
    """Rebuild Helion at each study commit and launch that commit's kernels."""
    results = _ARTIFACTS / "mxfp8_latency_by_shape.jsonl"
    if results.exists():
        results.unlink()
    shas = {label: sha for label, sha, _backends in _REVISIONS}
    # (series, revision label, backend, autotune effort), grouped per revision.
    runs = [(name, label, backend, "none") for name, label, backend in _SERIES]
    if args.tune != "none":
        runs.insert(3, (_TUNED_SERIES, "optimized", "triton", args.tune))
    try:
        for label, sha, _backends in _REVISIONS:
            worktree = _ARTIFACTS / "worktrees" / label
            _ensure_worktree(worktree, sha)
            _install(worktree)
            for name, run_label, backend, effort in runs:
                if run_label != label:
                    continue
                _run(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--measure",
                        backend,
                        "--series",
                        name,
                        "--effort",
                        effort,
                        "--label",
                        label,
                        "--source-root",
                        str(worktree),
                        "--expect-sha",
                        shas[label],
                        "--results",
                        str(results),
                        "--warmup-s",
                        str(args.warmup_s),
                    ]
                )
    finally:
        _install(_REPO)
    csv_path = _ARTIFACTS / "mxfp8_latency_by_shape.csv"
    _write_csv(results, [name for name, *_rest in runs], csv_path)
    write_latency_plot(csv_path, _ARTIFACTS / "mxfp8_latency_by_shape.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tune",
        choices=("none", "quick", "full"),
        default="none",
        help="also autotune Helion-Triton at the final commit",
    )
    parser.add_argument("--warmup-s", type=float, default=10.0)
    parser.add_argument("--measure", choices=("triton", "cute"), help=argparse.SUPPRESS)
    parser.add_argument("--series", help=argparse.SUPPRESS)
    parser.add_argument("--effort", default="none", help=argparse.SUPPRESS)
    parser.add_argument(
        "--label", choices=tuple(item[0] for item in _REVISIONS), help=argparse.SUPPRESS
    )
    parser.add_argument("--source-root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--expect-sha", help=argparse.SUPPRESS)
    parser.add_argument("--results", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.measure is None:
        reproduce(args)
        return
    if not (args.series and args.source_root and args.expect_sha and args.results):
        parser.error("--measure is an internal mode used by the reproducer")
    measure(args)


if __name__ == "__main__":
    main()
