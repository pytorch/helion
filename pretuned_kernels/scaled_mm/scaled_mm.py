"""FP8 scaled matrix multiply (``c = scale_a * (a @ b) * scale_b + bias``).

Pretuned for the small-token-count (decode) FP8 GEMM shapes used by vLLM
serving.  The kernel and its checked-in heuristic are ported from
https://github.com/vllm-project/vllm/pull/46522.
"""

from __future__ import annotations

import math

import torch

import helion
import helion.experimental
import helion.language as hl


@helion.experimental.aot_kernel(
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def scaled_mm(
    c: torch.Tensor,  # [M, N]
    a: torch.Tensor,  # [M, K]
    b: torch.Tensor,  # [K, N]
    scale_a: torch.Tensor,  # [1]/[1, 1]/[M]/[M, 1]
    scale_b: torch.Tensor,  # [1]/[1, 1]/[N]/[N, 1]
    bias: torch.Tensor | None = None,  # [N]
) -> None:
    M, K = a.shape
    N = b.shape[1]
    hl.specialize(K)
    hl.specialize(N)

    assert N > 0 and K > 0 and M > 0
    assert b.shape[0] == K
    assert a.dtype == b.dtype
    assert a.stride(1) == 1
    assert b.stride(0) == 1

    scale_a = scale_a.reshape(-1, 1) if scale_a.dim() <= 1 else scale_a
    scale_b = scale_b.reshape(-1, 1) if scale_b.dim() <= 1 else scale_b

    assert scale_a.dtype == scale_b.dtype and scale_a.is_floating_point()
    assert scale_a.shape[1] == 1 and (scale_a.shape[0] == 1 or scale_a.shape[0] == M)
    assert scale_b.shape[1] == 1 and (scale_b.shape[0] == 1 or scale_b.shape[0] == N)
    hl.specialize(scale_b.shape[1])

    out_dtype = c.dtype
    assert out_dtype.is_floating_point
    if bias is not None:
        assert bias.numel() == N and bias.dtype == out_dtype

    acc_dtype = torch.float32 if a.is_floating_point() else torch.int32

    for tile_m, tile_n in hl.tile([M, N]):
        acc = hl.zeros([tile_m, tile_n], acc_dtype)
        for tile_k in hl.tile(K):
            acc = hl.dot(
                a[tile_m, tile_k],
                b[tile_k, tile_n],
                acc=acc,
                out_dtype=acc_dtype,
            )

        acc = acc.to(torch.float32)

        if scale_a.shape[0] == M:
            scale_a_blk = scale_a[tile_m, :]
        else:
            scale_a_blk = scale_a[0, 0].expand(tile_m.block_size, 1)
        acc = scale_a_blk * acc

        if scale_b.shape[0] == N:
            scale_b_blk = scale_b[tile_n, :]
        else:
            scale_b_blk = scale_b[0, 0].expand(tile_n.block_size, 1)
        acc = scale_b_blk.T * acc

        c_blk = acc.to(out_dtype)

        if bias is not None:
            c_blk += bias[tile_n]

        c[tile_m, tile_n] = c_blk


def _scaled_mm_torch(
    c: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> None:
    out = torch._scaled_mm(
        a,
        b,
        scale_a=scale_a,
        scale_b=scale_b,
        bias=bias,
        out_dtype=c.dtype,
    )
    c.copy_(out)


# Optional vLLM CUTLASS baseline (the production FP8 GEMM this is benchmarked
# against). The pretuned test env has only torch + helion (guarded import); the
# nightly benchmark workflow installs vLLM, so main() then compares against it.
try:
    from vllm import _custom_ops as _vllm_ops

    _HAS_VLLM = hasattr(_vllm_ops, "cutlass_scaled_mm")
except ImportError:
    _vllm_ops = None
    _HAS_VLLM = False


def _scaled_mm_cutlass(
    c: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> None:
    """vLLM CUTLASS baseline (ops.cutlass_scaled_mm)."""
    out = _vllm_ops.cutlass_scaled_mm(a, b, scale_a, scale_b, c.dtype, bias)
    c.copy_(out)


def _baselines() -> list[tuple[str, object]]:
    """Baselines main() benchmarks helion against.

    torch._scaled_mm is always available; vLLM's CUTLASS kernel is added when
    vLLM is installed (the nightly benchmark env). The SUMMARY speedup is helion
    vs the best (fastest) available baseline.

    No torch.compile baseline here: the torch reference is a single opaque GEMM
    (torch._scaled_mm -> cuBLAS/cutlass), which torch.compile just re-dispatches
    to the same kernel -- a redundant, not-faster baseline.
    """
    baselines: list[tuple[str, object]] = [("torch", _scaled_mm_torch)]
    if _HAS_VLLM:
        baselines.append(("cutlass", _scaled_mm_cutlass))
    return baselines


def use_cudagraph() -> bool:
    """Whether main() benchmarks under CUDA graphs (read by pretuned_kernels/run.py).

    True: main() times these tiny decode GEMMs with do_bench_cudagraph (how vLLM
    invokes the kernel), which removes per-call host launch overhead.
    """
    return True


def main(verbose: bool = True) -> dict:
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from _bench import bench_cudagraph

    def _p(*args: object) -> None:
        if verbose:
            print(*args)

    # (K, N) weight shapes pulled from the vLLM Qwen3 FP8 sweep (TP=1).
    kn_shapes = [
        (2048, 4096),
        (2048, 2048),
        (2048, 12288),
        (6144, 2048),
        (4096, 6144),
        (4096, 4096),
        (4096, 24576),
        (12288, 4096),
        (5120, 10240),
        (5120, 5120),
        (5120, 51200),
        (25600, 5120),
    ]
    m_sizes = [16, 64]
    shapes = [(M, K, N) for M in m_sizes for (K, N) in kn_shapes]

    fp8_dtype = torch.float8_e4m3fn
    out_dtype = torch.bfloat16

    baselines = _baselines()
    names = [n for n, _ in baselines]
    _p(
        f"GPU: {torch.cuda.get_device_name()} "
        f"(baselines: {', '.join(names)}; SUMMARY vs best/fastest baseline)"
    )
    base_hdr = "  ".join(f"{n + ' (us)':>12s}" for n in names)
    _p(
        f"{'M':>6s}  {'K':>6s}  {'N':>6s}  {'helion (us)':>12s}  "
        f"{base_hdr}  {'speedup':>8s}"
    )
    _p("-" * (45 + 14 * len(names)))

    speedups_by_base: dict[str, list[float]] = {n: [] for n in names}
    best_speedups: list[float] = []  # helion vs the fastest baseline each shape
    helion_wins = 0
    best_speedup = 0.0
    best_shape = (0, 0, 0)
    for M, K, N in shapes:
        scale = 1.0 / math.sqrt(K)
        a = (scale * (0.5 + torch.rand(M, K, device="cuda"))).to(fp8_dtype)
        b = (scale * (0.5 + torch.rand(N, K, device="cuda"))).to(fp8_dtype).t()
        c = torch.empty((M, N), dtype=out_dtype, device="cuda")
        scale_a = torch.rand((1, 1), dtype=torch.float32, device="cuda") + 0.5
        scale_b = torch.rand((1, 1), dtype=torch.float32, device="cuda") + 0.5
        bias = 0.5 * (torch.rand(N, dtype=out_dtype, device="cuda") - 0.5)

        scaled_mm(c, a, b, scale_a, scale_b, bias)  # warmup / compile
        # Benchmark under CUDA graphs (how vLLM invokes the kernel): removes
        # per-call host launch overhead so timing reflects GPU work. Clears the
        # L2 cache each iteration (see _bench.bench_cudagraph).
        ms_helion = bench_cudagraph(
            lambda c=c, a=a, b=b, sa=scale_a, sb=scale_b, bias=bias: scaled_mm(
                c, a, b, sa, sb, bias
            ),
            rep=100,
        )
        base_ms: dict[str, float] = {}
        for name, fn in baselines:
            base_ms[name] = bench_cudagraph(
                lambda fn=fn, c=c, a=a, b=b, sa=scale_a, sb=scale_b, bias=bias: fn(
                    c, a, b, sa, sb, bias
                ),
                rep=100,
            )
            speedups_by_base[name].append(
                base_ms[name] / ms_helion if ms_helion > 0 else float("nan")
            )
        best_base = min(base_ms, key=base_ms.get)  # fastest baseline this shape
        speedup = base_ms[best_base] / ms_helion if ms_helion > 0 else float("nan")
        best_speedups.append(speedup)
        if speedup > 1.0:
            helion_wins += 1
        if speedup > best_speedup:
            best_speedup = speedup
            best_shape = (M, K, N)
        base_cols = "  ".join(f"{base_ms[n] * 1000:>12.2f}" for n in names)
        _p(
            f"{M:>6d}  {K:>6d}  {N:>6d}  {ms_helion * 1000:>12.2f}  "
            f"{base_cols}  {speedup:>7.2f}x  (vs {best_base})"
        )

    def _geomean(vals: list[float]) -> float:
        pos = [s for s in vals if s > 0]
        return math.exp(sum(math.log(s) for s in pos) / max(len(pos), 1))

    # Per-baseline breakdown (helion speedup over each specific baseline),
    # consumed by pretuned_kernels/run.py for the dashboard's per-kernel dropdown.
    per_baseline = {
        name: {
            "wins": sum(1 for s in speedups_by_base[name] if s > 1.0),
            "total": len(speedups_by_base[name]),
            "geomean": round(_geomean(speedups_by_base[name]), 4),
            "best_speedup": round(max(speedups_by_base[name], default=0.0), 4),
        }
        for name in names
    }
    for name in names:
        m = per_baseline[name]
        _p(
            f"vs {name}: wins={m['wins']}/{m['total']} "
            f"geomean={m['geomean']:.3f}x best={m['best_speedup']:.2f}x"
        )
    geomean = _geomean(best_speedups)
    _p(
        f"\nHelion faster on {helion_wins}/{len(shapes)} shapes vs the best "
        f"baseline; geomean speedup {geomean:.3f}x; "
        f"best speedup {best_speedup:.2f}x at (M, K, N)={best_shape}."
    )
    # Metrics vs the best (fastest) baseline per shape + per-baseline breakdown,
    # returned to the caller (pretuned_kernels/run.py).
    return {
        "helion_wins": helion_wins,
        "total": len(shapes),
        "geomean": round(geomean, 4),
        "best_speedup": round(best_speedup, 4),
        "baselines": per_baseline,
    }


if __name__ == "__main__":
    main()
