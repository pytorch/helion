"""Local preview for KernelGen problem 18 (BF16 grouped GEMM).

The dashboard contract is::

    x:           [8192, 8192]      bfloat16
    w:           [8, 4096, 8192]   bfloat16
    split_sizes: [8]               int32 CUDA tensor
    output:      [8192, 4096]      bfloat16

This harness reproduces the balanced, skewed, and adversarial split regimes.
The Helion kernel consumes the three problem inputs directly.  It builds a
compact grouped work schedule in-device, so a timed invocation is one Helion
kernel launch and does not need a prefix-sum or metadata-preparation launch.

Example::

    source .venv/bin/activate
    python benchmarks/cute/kernelgen_problem18.py --backend cute
"""

from __future__ import annotations

import argparse
import json
import statistics
from typing import TYPE_CHECKING

import torch

import helion
import helion.language as hl
import helion.runtime as helion_runtime

if TYPE_CHECKING:
    from collections.abc import Callable

PROBLEM_M = 8192
PROBLEM_N = 4096
PROBLEM_K = 8192
PROBLEM_GROUPS = 8
PEAK_COMPUTE_FLOPS_PER_SECOND = 2.5e15
PEAK_BANDWIDTH_BYTES_PER_SECOND = 8e12
REGIMES = ("balanced", "skewed", "adversarial")
BLOCK_M = 128
BLOCK_N = 128
BLOCK_K = 128

TRITON_CONFIG = helion.Config(
    block_sizes=[BLOCK_M, BLOCK_N, 64],
    l2_groupings=[1],
    loop_orders=[[0, 1, 2]],
    num_stages=3,
    num_warps=4,
    pid_type="flat",
)

CUTE_CONFIG = helion.Config(
    block_sizes=[256, BLOCK_N, BLOCK_K],
    l2_groupings=[1],
    loop_orders=[[0, 1, 2]],
    num_stages=7,
    num_warps=8,
    pid_type="persistent_interleaved",
    tcgen05_cluster_m=2,
    tcgen05_cluster_n=1,
    tcgen05_ab_stages=3,
    tcgen05_acc_stages=2,
    tcgen05_c_stages=2,
    tcgen05_num_epi_warps=4,
    tcgen05_grouped_mode="worklist_nm",
    tcgen05_grouped_worklist_source_m_tile=256,
)

OPTIMIZED_CORE_CONFIG = helion.Config(
    block_sizes=[128, 128, 64],
    l2_groupings=[1],
    loop_orders=[[0, 1]],
    num_stages=2,
    num_warps=8,
    pid_type="persistent_interleaved",
    tcgen05_cluster_m=1,
    tcgen05_cluster_n=1,
    tcgen05_ab_stages=3,
    tcgen05_acc_stages=2,
    tcgen05_c_stages=2,
    tcgen05_num_epi_warps=4,
    tcgen05_grouped_mode="direct",
    tcgen05_grouped_external_direct_pointers="direct_pointers",
    tcgen05_grouped_external_direct_strides="direct_strides",
)


def grouped_gemm_body(
    x: torch.Tensor,
    w: torch.Tensor,
    split_sizes: torch.Tensor,
) -> torch.Tensor:
    """Compute packed grouped GEMM from device ``split_sizes`` metadata."""
    total_m, k = x.size()
    groups, n, w_k = w.size()
    assert k == w_k
    assert groups == PROBLEM_GROUPS
    assert split_sizes.size(0) == PROBLEM_GROUPS

    out = torch.empty((total_m, n), dtype=x.dtype, device=x.device)
    block_m = hl.register_block_size(256)
    block_n = hl.register_block_size(BLOCK_N)
    block_k = hl.register_block_size(BLOCK_K)

    for group_tile, tile_m, tile_n in hl.tile(
        [groups, total_m, n],
        block_size=[1, block_m, block_n],
    ):
        # The size-one group tile is reduced to a scalar so it can index both
        # the split metadata and the rank-3 RHS tensor.
        group = group_tile.index.sum()
        group_m = split_sizes[group]
        s0 = split_sizes[0]
        s1 = split_sizes[1]
        s2 = split_sizes[2]
        s3 = split_sizes[3]
        s4 = split_sizes[4]
        s5 = split_sizes[5]
        s6 = split_sizes[6]
        group_start = (
            torch.where(group > 0, s0, 0)
            + torch.where(group > 1, s1, 0)
            + torch.where(group > 2, s2, 0)
            + torch.where(group > 3, s3, 0)
            + torch.where(group > 4, s4, 0)
            + torch.where(group > 5, s5, 0)
            + torch.where(group > 6, s6, 0)
        )
        local_m = tile_m.index
        row_index = group_start + local_m
        valid_rows = local_m < group_m
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k, block_size=block_k):
            a = hl.load(
                x,
                [row_index, tile_k],
                extra_mask=valid_rows[:, None],  # pyrefly: ignore[bad-index]
            )
            b = w[group, tile_n, tile_k]
            acc = torch.addmm(acc, a, b.T)
        hl.store(
            out,
            [row_index, tile_n],
            acc.to(out.dtype),
            extra_mask=valid_rows[:, None],  # pyrefly: ignore[bad-index]
        )
    return out


def optimized_core_body(
    a_placeholder: torch.Tensor,
    b_placeholder: torch.Tensor,
    layout: torch.Tensor,
    n_sizes: torch.Tensor,
    k_sizes: torch.Tensor,
    out_placeholder: torch.Tensor,
    direct_pointers: torch.Tensor,
    direct_strides: torch.Tensor,
) -> torch.Tensor:
    """Grouped tcgen05 core using host-prepared direct-pointer metadata."""
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
        for tile_k in hl.tile(max_k):
            valid_k = (tile_k.index < group_k)[None, :]  # pyrefly: ignore[bad-index]
            a = a_placeholder[tile_m, tile_k]
            b = b_placeholder[safe_group_id, tile_n, tile_k]
            a = torch.where(valid_k, a, torch.zeros_like(a))
            b = torch.where(valid_k, b, torch.zeros_like(b))
            acc = torch.addmm(acc, a, b.T)
        old = out_placeholder[tile_m, tile_n]
        out_placeholder[tile_m, tile_n] = torch.where(
            valid,
            acc.to(out_placeholder.dtype),
            old,
        )
    return out_placeholder


def make_kernel(backend: str) -> helion.Kernel[torch.Tensor]:
    return helion.kernel(
        grouped_gemm_body,
        backend=backend,
        static_shapes=True,
        autotune_effort="none",
    )


def make_optimized_core_kernel() -> helion.Kernel[torch.Tensor]:
    return helion.kernel(
        optimized_core_body,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
    )


def placeholder(
    shape: tuple[int, ...],
    device: torch.device,
) -> torch.Tensor:
    base = torch.empty(max(1, shape[-1]), device=device, dtype=torch.bfloat16)
    return torch.as_strided(base, shape, (0,) * (len(shape) - 1) + (1,))


def prepare_optimized_core(
    kernel: helion.Kernel[torch.Tensor],
    x: torch.Tensor,
    w: torch.Tensor,
    split_sizes: torch.Tensor,
) -> tuple[Callable[[], torch.Tensor], torch.Tensor, tuple[object, ...]]:
    """Prepare an optimized core launch; host preparation is not timed."""
    sizes = [int(value) for value in split_sizes.cpu().tolist()]
    device = x.device
    n, k = int(w.size(1)), int(w.size(2))
    output = torch.empty((x.size(0), n), dtype=x.dtype, device=device)
    group_a: list[torch.Tensor] = []
    group_b: list[torch.Tensor] = []
    group_out: list[torch.Tensor] = []
    layout_parts: list[tuple[int, int]] = []
    scratch: list[torch.Tensor] = []
    offset = 0
    for group, size in enumerate(sizes):
        if size:
            a = x[offset : offset + size]
            out = output[offset : offset + size]
            work_size = size
            offset += size
        else:
            a = torch.zeros((1, k), dtype=x.dtype, device=device)
            out = torch.empty((1, n), dtype=x.dtype, device=device)
            scratch.extend((a, out))
            work_size = 1
        group_a.append(a)
        group_b.append(w[group])
        group_out.append(out)
        padded = (work_size + BLOCK_M - 1) // BLOCK_M * BLOCK_M
        layout_parts.append((work_size, padded))

    padded_m = sum(padded for _work_size, padded in layout_parts)
    layout = torch.full((padded_m,), -1, dtype=torch.int32, device=device)
    cursor = 0
    for group, (work_size, padded) in enumerate(layout_parts):
        layout[cursor : cursor + work_size].fill_(group)
        cursor += padded

    n_sizes = torch.full((PROBLEM_GROUPS,), n, dtype=torch.int32, device=device)
    k_sizes = torch.full((PROBLEM_GROUPS,), k, dtype=torch.int32, device=device)
    direct_pointers = torch.tensor(
        [
            (a.data_ptr(), b.data_ptr(), out.data_ptr())
            for a, b, out in zip(group_a, group_b, group_out, strict=True)
        ],
        dtype=torch.int64,
        device=device,
    )
    direct_strides = torch.tensor(
        [
            (tuple(a.stride()), tuple(b.stride()), tuple(out.stride()))
            for a, b, out in zip(group_a, group_b, group_out, strict=True)
        ],
        dtype=torch.int32,
        device=device,
    )
    kernel_args = (
        placeholder((padded_m, k), device),
        placeholder((PROBLEM_GROUPS, n, k), device),
        layout,
        n_sizes,
        k_sizes,
        placeholder((padded_m, n), device),
        direct_pointers,
        direct_strides,
    )
    bound = kernel.bind(kernel_args)
    bound.env.config_spec.cute_tcgen05_search_enabled = True
    bound.set_config(OPTIMIZED_CORE_CONFIG)

    def launch() -> torch.Tensor:
        return bound(*kernel_args)

    owners: tuple[object, ...] = (
        bound,
        *kernel_args,
        *group_a,
        *group_b,
        *group_out,
        *scratch,
    )
    return launch, output, owners


def make_split_sizes(
    total_m: int,
    groups: int,
    regime: str,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Reproduce KernelGen's grouped_split_sizes helper."""
    if regime == "balanced":
        quotient, remainder = divmod(total_m, groups)
        sizes = torch.full((groups,), quotient, dtype=torch.int64, device=device)
        sizes[:remainder] += 1
    elif regime == "skewed":
        concentration = torch.full((groups,), 0.5, device=device)
        proportions = torch.distributions.Dirichlet(concentration).sample()
        sizes = torch.floor(proportions * total_m).to(torch.int64)
        sizes[torch.argmax(sizes)] += total_m - sizes.sum()
    elif regime == "adversarial":
        sizes = torch.zeros(groups, dtype=torch.int64, device=device)
        permutation = torch.randperm(groups, device=device)
        hot = permutation[0]
        empty_groups = max(1, groups // 3)
        active = permutation[1 + empty_groups :]
        hot_size = (total_m * 6) // 10
        sizes[hot] = hot_size
        quotient, remainder = divmod(total_m - hot_size, int(active.numel()))
        sizes[active] = quotient
        sizes[active[:remainder]] += 1
    else:
        raise ValueError(f"unknown split regime: {regime}")
    return sizes.to(torch.int32)


def roofline(m: int, n: int, k: int, groups: int) -> dict[str, float | int]:
    flops = 2 * m * n * k
    io_bytes = 2 * (m * k + groups * n * k + m * n) + 4 * groups
    compute_ms = flops / PEAK_COMPUTE_FLOPS_PER_SECOND * 1000
    bandwidth_ms = io_bytes / PEAK_BANDWIDTH_BYTES_PER_SECOND * 1000
    return {
        "flops": flops,
        "io_bytes": io_bytes,
        "compute_ms": compute_ms,
        "bandwidth_ms": bandwidth_ms,
        "roofline_ms": max(compute_ms, bandwidth_ms),
    }


def check_correctness(
    actual: torch.Tensor,
    x: torch.Tensor,
    w: torch.Tensor,
    split_sizes: torch.Tensor,
    *,
    rtol: float,
    atol: float,
) -> dict[str, float | bool]:
    sizes = [int(value) for value in split_sizes.cpu().tolist()]
    chunks = torch.split(x, sizes, dim=0)
    offset = 0
    max_abs = 0.0
    max_rel = 0.0
    for chunk, weight in zip(chunks, w, strict=True):
        rows = int(chunk.size(0))
        if rows == 0:
            continue
        expected = chunk @ weight.T
        result = actual[offset : offset + rows]
        difference = (result.float() - expected.float()).abs()
        max_abs = max(max_abs, float(difference.max().item()))
        denominator = expected.float().abs().clamp_min(atol)
        max_rel = max(max_rel, float((difference / denominator).max().item()))
        torch.testing.assert_close(result, expected, rtol=rtol, atol=atol)
        offset += rows
    if offset != x.size(0):
        raise AssertionError(f"checked {offset} rows, expected {x.size(0)}")
    return {"ok": True, "max_abs": max_abs, "max_rel": max_rel}


def capture(
    launch: Callable[[], torch.Tensor],
    backend: str,
) -> tuple[Callable[[], object], torch.Tensor, object]:
    if backend == "cute":
        with helion_runtime.cute_cuda_graph() as graph:
            output = launch()
    else:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = launch()
    return graph.replay, output, graph


def measure_ms(
    replay: Callable[[], object],
    *,
    warmups: int,
    samples: int,
    iterations: int,
    l2_flush: torch.Tensor | None,
) -> list[float]:
    for _ in range(warmups):
        replay()
    torch.cuda.synchronize()

    values: list[float] = []
    for _ in range(samples):
        if l2_flush is not None:
            l2_flush.zero_()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            replay()
        end.record()
        end.synchronize()
        values.append(float(start.elapsed_time(end)) / iterations)
    return values


def make_exact_launch(
    bound: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    w: torch.Tensor,
    split_sizes: torch.Tensor,
) -> Callable[[], torch.Tensor]:
    def launch() -> torch.Tensor:
        return bound(x, w, split_sizes)

    return launch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cute", "triton"), default="cute")
    parser.add_argument(
        "--mode",
        choices=("exact-contract", "optimized-core"),
        default="exact-contract",
        help=(
            "exact-contract consumes x/w/split_sizes in one kernel; optimized-core "
            "times only a host-prepared direct-metadata grouped kernel"
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--m", type=int, default=PROBLEM_M)
    parser.add_argument("--n", type=int, default=PROBLEM_N)
    parser.add_argument("--k", type=int, default=PROBLEM_K)
    parser.add_argument("--regimes", nargs="+", choices=REGIMES, default=REGIMES)
    parser.add_argument("--seed-base", type=int, default=42)
    parser.add_argument("--compile-warmups", type=int, default=2)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--l2-flush-mb", type=int, default=0)
    parser.add_argument("--rtol", type=float, default=0.015625)
    parser.add_argument("--atol", type=float, default=0.015625)
    parser.add_argument("--correctness-only", action="store_true")
    parser.add_argument(
        "--cuda-graph",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("problem 18 requires an NVIDIA CUDA device")
    if args.m <= 0 or args.n <= 0 or args.k <= 0:
        raise ValueError("M, N, and K must be positive")
    if args.mode == "optimized-core" and args.backend != "cute":
        raise ValueError("optimized-core is available only with --backend cute")

    kernel = make_kernel(args.backend)
    optimized_core_kernel = make_optimized_core_kernel()
    roof = roofline(args.m, args.n, args.k, PROBLEM_GROUPS)
    l2_flush = (
        torch.empty(
            args.l2_flush_mb * 1024 * 1024 // 4,
            dtype=torch.int32,
            device=device,
        )
        if args.l2_flush_mb > 0
        else None
    )
    rows: list[dict[str, object]] = []
    bound = None

    for trial, regime in enumerate(args.regimes):
        torch.manual_seed(args.seed_base + trial)
        split_sizes = make_split_sizes(
            args.m,
            PROBLEM_GROUPS,
            regime,
            device=device,
        )
        x = torch.randn((args.m, args.k), device=device, dtype=torch.bfloat16)
        w = torch.randn(
            (PROBLEM_GROUPS, args.n, args.k),
            device=device,
            dtype=torch.bfloat16,
        )
        owners: tuple[object, ...] = ()
        if args.mode == "optimized-core":
            launch, output, owners = prepare_optimized_core(
                optimized_core_kernel,
                x,
                w,
                split_sizes,
            )
            launch()
        else:
            if bound is None:
                bound = kernel.bind((x, w, split_sizes))
                if args.backend == "cute":
                    bound.env.config_spec.cute_tcgen05_search_enabled = True
                    bound.set_config(CUTE_CONFIG)
                else:
                    bound.set_config(TRITON_CONFIG)
            launch = make_exact_launch(bound, x, w, split_sizes)
            output = launch()
        for _ in range(args.compile_warmups):
            launch()
        torch.cuda.synchronize()
        correctness = check_correctness(
            output,
            x,
            w,
            split_sizes,
            rtol=args.rtol,
            atol=args.atol,
        )
        row: dict[str, object] = {
            "trial": trial,
            "regime": regime,
            "mode": args.mode,
            "split_sizes": [int(value) for value in split_sizes.cpu().tolist()],
            "correctness": correctness,
        }

        if not args.correctness_only:
            graph_owner = None
            if args.cuda_graph:
                replay, captured_output, graph_owner = capture(launch, args.backend)
                owners = (*owners, graph_owner)
                replay()
                torch.cuda.synchronize()
                check_correctness(
                    output if args.mode == "optimized-core" else captured_output,
                    x,
                    w,
                    split_sizes,
                    rtol=args.rtol,
                    atol=args.atol,
                )
            else:
                replay = launch
            samples_ms = measure_ms(
                replay,
                warmups=args.warmups,
                samples=args.samples,
                iterations=args.iterations,
                l2_flush=l2_flush,
            )
            median_ms = statistics.median(samples_ms)
            row["timing"] = {
                "median_ms": median_ms,
                "min_ms": min(samples_ms),
                "samples_ms": samples_ms,
                "tflops": int(roof["flops"]) / (median_ms * 1e9),
                "estimated_dashboard_roofline_pct": (
                    float(roof["roofline_ms"]) / median_ms * 100
                ),
                "cuda_graph": args.cuda_graph,
                "l2_flush_mb": args.l2_flush_mb,
            }
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    result = {
        "backend": args.backend,
        "mode": args.mode,
        "device": torch.cuda.get_device_name(device),
        "shape": {
            "m": args.m,
            "n": args.n,
            "k": args.k,
            "groups": PROBLEM_GROUPS,
        },
        "roofline": roof,
        "config": (
            OPTIMIZED_CORE_CONFIG.config
            if args.mode == "optimized-core"
            else (
                CUTE_CONFIG.config if args.backend == "cute" else TRITON_CONFIG.config
            )
        ),
        "rows": rows,
        "note": (
            "The roofline percentage uses problem-18 peak constants. Exact dashboard "
            "results additionally depend on Crucible clock pinning and timing policy. "
            "optimized-core excludes host metadata preparation and is not a "
            "submission-compatible end-to-end measurement."
        ),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
