"""RowWise-scaled FP8 GEMM for the B200 CuTe (tcgen05) backend.

``out = (scale_a * (x @ y)) * scale_b`` with a per-row ``scale_a`` and a
per-column ``scale_b``, accumulating in FP32 and returning BF16.

The kernel is ported from https://github.com/pytorch/helion/pull/2788/
(``examples/fp8_gemm.py``).  It runs on Helion's CuTe (tcgen05) backend and is
pretuned for the skinny-M (decode / small-batch) and small decoder-layer FP8
W8A8 serving shapes that back the nightly B200 CuTe benchmark dashboard.

Two kernels ship:

* :func:`scale_mm_cute` tiles over both M and N.
* :func:`scale_mm_cute_skinny_m` keeps the full (small) M resident and tiles
  only over N, which the CuTe backend runs faster for tiny M.

:func:`_scale_mm_cute` dispatches to the skinny-M kernel when ``M <= 16``.
"""

from __future__ import annotations

import math

import torch
import triton.testing as tt

import helion.experimental
import helion.language as hl

# M at or below this uses the skinny-M kernel (mirrors examples/fp8_gemm.py).
_SKINNY_M_MAX = 16


@helion.experimental.aot_kernel(backend="cute", static_shapes=True)
def scale_mm_cute(
    x: torch.Tensor,  # [M, K] fp8
    y: torch.Tensor,  # [K, N] fp8
    scale_a: torch.Tensor,  # [M, N] per-row scale broadcast to [M, N]
    scale_b: torch.Tensor,  # [N] per-column scale
) -> torch.Tensor:
    """RowWise-scaled FP8 GEMM (tiled over M and N). Returns BF16 [M, N]."""
    m, k = x.size()
    k2, n = y.size()
    assert k == k2, f"size mismatch {k} != {k2}"
    out = torch.empty([m, n], dtype=torch.bfloat16, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = hl.dot(x[tile_m, tile_k], y[tile_k, tile_n], acc=acc)
        # RowWise scale in the epilogue (per-row scale_a x per-column scale_b).
        acc = acc * scale_a[tile_m, tile_n] * scale_b[tile_n]
        out[tile_m, tile_n] = acc.to(torch.bfloat16)
    return out


@helion.experimental.aot_kernel(backend="cute", static_shapes=True)
def scale_mm_cute_skinny_m(
    x: torch.Tensor,  # [M, K] fp8
    y: torch.Tensor,  # [K, N] fp8
    scale_a: torch.Tensor,  # [M, N] per-row scale broadcast to [M, N]
    scale_b: torch.Tensor,  # [N] per-column scale
) -> torch.Tensor:
    """Skinny-M variant of :func:`scale_mm_cute` for the CuTe (tcgen05) backend.

    Keeps the full (small) M dimension resident and tiles only over N, which the
    CuTe backend runs faster than the [M, N]-tiled kernel when M is tiny (decode
    / small-batch). Same RowWise-scale layout and BF16 output.
    """
    m, k = x.size()
    k2, n = y.size()
    assert k == k2, f"size mismatch {k} != {k2}"
    out = torch.empty([m, n], dtype=torch.bfloat16, device=x.device)
    for tile_n in hl.tile(n):
        acc = hl.zeros([m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = hl.dot(x[:, tile_k], y[tile_k, tile_n], acc=acc)
        acc = acc * scale_a[:, tile_n] * scale_b[tile_n]
        out[:, tile_n] = acc.to(torch.bfloat16)
    return out


def _scale_mm_cute(
    x: torch.Tensor,
    y: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
) -> torch.Tensor:
    """Dispatch to the skinny-M kernel for small M, else the [M, N]-tiled one."""
    if x.size(0) <= _SKINNY_M_MAX:
        return scale_mm_cute_skinny_m(x, y, scale_a, scale_b)
    return scale_mm_cute(x, y, scale_a, scale_b)


def _scale_mm_torch(
    x: torch.Tensor,
    y: torch.Tensor,
    scale_a: torch.Tensor,  # [M, N] broadcast view of the per-row scale
    scale_b: torch.Tensor,  # [N] per-column scale
) -> torch.Tensor:
    """RowWise baseline via torch._scaled_mm (per-row [M, 1] / per-col [1, N])."""
    sa = scale_a[:, :1].contiguous()
    sb = scale_b.reshape(1, -1)
    # torch._scaled_mm requires a column-major second operand.
    if y.stride(0) == 1 and y.stride(1) > 1:
        y_col_major = y
    else:
        y_col_major = y.T.contiguous().T
    return torch._scaled_mm(
        x, y_col_major, sa, sb, use_fast_accum=False, out_dtype=torch.bfloat16
    )


# Optional vLLM CUTLASS baseline (the production FP8 GEMM this is benchmarked
# against). The pretuned test env has only torch + helion (guarded import); the
# nightly benchmark workflow installs vLLM, so main() then compares against it.
try:
    from vllm import _custom_ops as _vllm_ops

    _HAS_VLLM = hasattr(_vllm_ops, "cutlass_scaled_mm")
except ImportError:
    _vllm_ops = None
    _HAS_VLLM = False


def _scale_mm_cutlass(
    x: torch.Tensor,
    y: torch.Tensor,
    scale_a: torch.Tensor,  # [M, N] broadcast view of the per-row scale
    scale_b: torch.Tensor,  # [N] per-column scale
) -> torch.Tensor:
    """vLLM CUTLASS RowWise baseline (ops.cutlass_scaled_mm)."""
    sa = scale_a[:, :1].contiguous()
    sb = scale_b.reshape(-1, 1).contiguous()
    return _vllm_ops.cutlass_scaled_mm(x, y, sa, sb, torch.bfloat16, None)


def _baselines() -> list[tuple[str, object]]:
    """Baselines main() benchmarks helion against.

    torch._scaled_mm is always available; vLLM's CUTLASS kernel is added when
    vLLM is installed (the nightly benchmark env). The SUMMARY speedup is helion
    vs the best (fastest) available baseline.
    """
    baselines: list[tuple[str, object]] = [("torch", _scale_mm_torch)]
    if _HAS_VLLM:
        baselines.append(("cutlass", _scale_mm_cutlass))
    return baselines


def use_cudagraph() -> bool:
    """Whether main() benchmarks under CUDA graphs (read by pretuned_kernels/run.py).

    True: main() times these decode / small-batch GEMMs with do_bench_cudagraph
    (how vLLM invokes the kernel), which removes per-call host launch overhead.
    """
    return True


# Skinny-M (decode / small-batch) + small decoder-layer FP8 W8A8 serving shapes
# that back the nightly B200 CuTe dashboard (benchmarks/run.py, PR #2788).
SHAPES = [  # (M, K, N)
    (1, 4096, 4096),
    (4096, 4096, 4096),
    (1, 4096, 256),
    (512, 2048, 4096),
    (512, 2048, 2048),
]


def _make_inputs(
    m: int, k: int, n: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    scale = 1.0 / math.sqrt(k)
    x = (scale * torch.randn(m, k, device="cuda")).to(torch.float8_e4m3fn)
    # y in [k, n] with a column-major (k contiguous) layout, as torch._scaled_mm
    # and cutlass_scaled_mm want for the second operand.
    y = (scale * torch.randn(k, n, device="cuda")).to(torch.float8_e4m3fn)
    y = y.T.contiguous().T
    scale_a = (torch.rand(m, 1, device="cuda") + 0.5).expand(m, n)
    scale_b = torch.rand(n, device="cuda") + 0.5
    return x, y, scale_a, scale_b


def main(verbose: bool = True) -> dict:
    def _p(*args: object) -> None:
        if verbose:
            print(*args)

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
    for M, K, N in SHAPES:
        x, y, scale_a, scale_b = _make_inputs(M, K, N)

        _scale_mm_cute(x, y, scale_a, scale_b)  # warmup / compile
        ms_helion = tt.do_bench_cudagraph(
            lambda x=x, y=y, sa=scale_a, sb=scale_b: _scale_mm_cute(x, y, sa, sb),
            rep=100,
            return_mode="median",
        )
        base_ms: dict[str, float] = {}
        for name, fn in baselines:
            base_ms[name] = tt.do_bench_cudagraph(
                lambda fn=fn, x=x, y=y, sa=scale_a, sb=scale_b: fn(x, y, sa, sb),
                rep=100,
                return_mode="median",
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
        f"\nHelion faster on {helion_wins}/{len(SHAPES)} shapes vs the best "
        f"baseline; geomean speedup {geomean:.3f}x; "
        f"best speedup {best_speedup:.2f}x at (M, K, N)={best_shape}."
    )
    return {
        "helion_wins": helion_wins,
        "total": len(SHAPES),
        "geomean": round(geomean, 4),
        "best_speedup": round(best_speedup, 4),
        "baselines": per_baseline,
    }


if __name__ == "__main__":
    main()
