"""FP8 scaled matrix multiply with the swapAB trick.

Same contract as ``pretuned_kernels/scaled_mm`` (``c = scale_a * (a @ b) *
scale_b + bias``) but the inner MMA computes ``B @ A`` and transposes the
accumulator in the epilogue.  For the small-token-count (decode) shapes vLLM
serves, ``M`` is tiny (1..64) while ``N`` is large; landing the small ``M``
dimension on the wide MMA axis (the "swapAB" trick) can improve tensor-core
utilization.

Ported from the vLLM fork
https://github.com/vllm-project/vllm/compare/main...xiaohongchen1991:vllm:scaled_mm_swap_ab
and RE-TUNED for this body (its own ``_helion_aot_scaled_mm_swap_ab_cuda_sm90``
heuristic, seeded from the non-swap configs).  Benchmarked side-by-side with the
non-swap ``pretuned_kernels/scaled_mm`` kernel.
"""

from __future__ import annotations

import math

import torch
import triton.testing as tt

import helion
import helion.experimental
import helion.language as hl


# Prefer tritonbench's cache-clearing cudagraph bench: it zeroes a >L2-sized
# buffer before each kernel invocation inside the graph (and subtracts the
# clear cost), so every timed run starts from a COLD L2. That is the realistic
# regime for these tiny decode GEMMs -- in serving, the weights/activations are
# not already L2-resident. Plain triton.testing.do_bench_cudagraph leaves data
# hot in L2 and under-reports time. Fall back to plain do_bench_cudagraph only
# if tritonbench is not on the path.
try:
    from tritonbench.components.do_bench.run import (
        _do_bench_cudagraph_with_cache_clear as _cudagraph_cache_clear,
    )

    def _bench_cudagraph(call) -> float:
        return _cudagraph_cache_clear(
            call, rep=100, return_mode="median", skip_cache_clearing=False
        )

    _CLEARS_L2 = True
except Exception:  # tritonbench unavailable

    def _bench_cudagraph(call) -> float:
        return tt.do_bench_cudagraph(call, rep=100, return_mode="median")

    _CLEARS_L2 = False


@helion.experimental.aot_kernel(
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def scaled_mm_swap_ab(
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
        # swapAB: accumulate [N, M] = B_blk @ A_blk, then transpose in the
        # epilogue so the small M dimension rides the wide MMA axis.
        acc = hl.zeros([tile_n, tile_m], acc_dtype)
        for tile_k in hl.tile(K):
            a_blk = hl.load(a, [tile_m.index[None, :], tile_k.index[:, None]])
            b_blk = hl.load(b, [tile_k.index[None, :], tile_n.index[:, None]])
            acc = hl.dot(
                b_blk,
                a_blk,
                acc=acc,
                out_dtype=acc_dtype,
            )

        acc = acc.t()
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


# Optional non-swap helion kernel: when the sibling pretuned scaled_mm module is
# importable, main() also reports swapAB vs the non-swap helion body (both use
# their own pretuned configs) -- the head-to-head the swapAB trick is about.
try:
    import importlib.util
    from pathlib import Path

    _nonswap_path = (
        Path(__file__).resolve().parent.parent / "scaled_mm" / "scaled_mm.py"
    )
    _spec = importlib.util.spec_from_file_location(
        "_pretuned_scaled_mm_nonswap", _nonswap_path
    )
    assert _spec is not None and _spec.loader is not None
    _nonswap_mod = importlib.util.module_from_spec(_spec)
    import sys as _sys

    _sys.modules["_pretuned_scaled_mm_nonswap"] = _nonswap_mod
    _spec.loader.exec_module(_nonswap_mod)
    _scaled_mm_nonswap = _nonswap_mod.scaled_mm
    _HAS_NONSWAP = True
except Exception:
    _scaled_mm_nonswap = None
    _HAS_NONSWAP = False


# Optional vLLM CUTLASS baseline (guarded like the non-swap kernel).
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
    out = _vllm_ops.cutlass_scaled_mm(a, b, scale_a, scale_b, c.dtype, bias)
    c.copy_(out)


def _baselines() -> list[tuple[str, object]]:
    """Baselines main() benchmarks the swapAB helion kernel against.

    Always includes torch._scaled_mm.  Adds the non-swap helion body (the direct
    swapAB-vs-no-swap comparison) when the sibling module imports, and vLLM's
    CUTLASS kernel when vLLM is installed.  The SUMMARY speedup is swapAB helion
    vs the best (fastest) available baseline per shape.
    """
    baselines: list[tuple[str, object]] = [("torch", _scaled_mm_torch)]
    if _HAS_NONSWAP:
        baselines.append(("helion_noswap", _scaled_mm_nonswap))
    if _HAS_VLLM:
        baselines.append(("cutlass", _scaled_mm_cutlass))
    return baselines


def use_cudagraph() -> bool:
    """Benchmark under CUDA graphs (how vLLM invokes the kernel); read by run.py."""
    return True


def main(verbose: bool = True) -> dict:
    def _p(*args: object) -> None:
        if verbose:
            print(*args)

    # Same (K, N) weight shapes as pretuned_kernels/scaled_mm (vLLM Qwen3 FP8).
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
    timer = (
        "cudagraph + L2-clear (tritonbench)" if _CLEARS_L2 else "cudagraph (hot L2)"
    )
    _p(
        f"GPU: {torch.cuda.get_device_name()} "
        f"(baselines: {', '.join(names)}; SUMMARY vs best/fastest baseline; "
        f"timer: {timer})"
    )
    base_hdr = "  ".join(f"{n + ' (us)':>14s}" for n in names)
    _p(
        f"{'M':>6s}  {'K':>6s}  {'N':>6s}  {'swapAB (us)':>12s}  "
        f"{base_hdr}  {'speedup':>8s}"
    )
    _p("-" * (45 + 16 * len(names)))

    speedups_by_base: dict[str, list[float]] = {n: [] for n in names}
    best_speedups: list[float] = []
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

        scaled_mm_swap_ab(c, a, b, scale_a, scale_b, bias)  # warmup / compile
        ms_helion = _bench_cudagraph(
            lambda c=c, a=a, b=b, sa=scale_a, sb=scale_b, bias=bias: (
                scaled_mm_swap_ab(c, a, b, sa, sb, bias)
            )
        )
        base_ms: dict[str, float] = {}
        for name, fn in baselines:
            base_ms[name] = _bench_cudagraph(
                lambda fn=fn, c=c, a=a, b=b, sa=scale_a, sb=scale_b, bias=bias: fn(
                    c, a, b, sa, sb, bias
                )
            )
            speedups_by_base[name].append(
                base_ms[name] / ms_helion if ms_helion > 0 else float("nan")
            )
        best_base = min(base_ms, key=base_ms.get)
        speedup = base_ms[best_base] / ms_helion if ms_helion > 0 else float("nan")
        best_speedups.append(speedup)
        if speedup > 1.0:
            helion_wins += 1
        if speedup > best_speedup:
            best_speedup = speedup
            best_shape = (M, K, N)
        base_cols = "  ".join(f"{base_ms[n] * 1000:>14.2f}" for n in names)
        _p(
            f"{M:>6d}  {K:>6d}  {N:>6d}  {ms_helion * 1000:>12.2f}  "
            f"{base_cols}  {speedup:>7.2f}x  (vs {best_base})"
        )

    def _geomean(vals: list[float]) -> float:
        pos = [s for s in vals if s > 0]
        return math.exp(sum(math.log(s) for s in pos) / max(len(pos), 1))

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
        f"\nswapAB helion faster on {helion_wins}/{len(shapes)} shapes vs the best "
        f"baseline; geomean speedup {geomean:.3f}x; "
        f"best speedup {best_speedup:.2f}x at (M, K, N)={best_shape}."
    )
    return {
        "helion_wins": helion_wins,
        "total": len(shapes),
        "geomean": round(geomean, 4),
        "best_speedup": round(best_speedup, 4),
        "baselines": per_baseline,
    }


if __name__ == "__main__":
    main()
