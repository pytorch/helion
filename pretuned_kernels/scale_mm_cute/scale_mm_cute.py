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

:func:`_scale_mm_cute` dispatches to the skinny-M kernel when ``M <= 1``.
"""

from __future__ import annotations

import math

import torch

import helion.experimental
import helion.language as hl

# M at or below this uses the skinny-M kernel (mirrors examples/fp8_gemm.py).
_SKINNY_M_MAX = 1


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

    True: main() times these decode / small-batch GEMMs under CUDA graphs (how
    vLLM invokes the kernel), which removes per-call host launch overhead. The
    shared _bench loop clears the L2 cache before every replay (cold L2).
    """
    return True


# Skinny-M (decode / small-batch) + small decoder-layer FP8 W8A8 serving shapes
# that back the nightly B200 CuTe dashboard (benchmarks/run.py, PR #2788).
# The M=64 rows mirror the vLLM Qwen3 FP8 serving (K, N) weight shapes at a
# small-batch token count.
SHAPES = [  # (M, K, N)
    (1, 4096, 4096),
    (4096, 4096, 4096),
    (1, 4096, 256),
    (512, 2048, 4096),
    (512, 2048, 2048),
    (64, 2048, 4096),
    (64, 2048, 2048),
    (64, 2048, 12288),
    (64, 6144, 2048),
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
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from _bench import run_sweep

    baselines = _baselines()

    def make_calls(shape: tuple[int, int, int]) -> tuple:
        M, K, N = shape
        x, y, scale_a, scale_b = _make_inputs(M, K, N)

        def helion_call() -> None:
            _scale_mm_cute(x, y, scale_a, scale_b)

        base_calls = [
            (name, (lambda fn=fn: fn(x, y, scale_a, scale_b))) for name, fn in baselines
        ]
        return helion_call, base_calls, f"{M:>6d}  {K:>6d}  {N:>6d}"

    # run_sweep benchmarks under CUDA graphs (how these decode / small-batch GEMMs
    # are invoked; removes per-call host launch overhead) with L2 cache clearing
    # before every replay (see _bench) -- the fp8 configs are tuned for this
    # cold-L2 regime (PR #2821: the 2-CTA A-multicast win only shows cold).
    return run_sweep(
        SHAPES,
        make_calls,
        use_cudagraph=use_cudagraph(),
        verbose=verbose,
        shape_header=f"{'M':>6s}  {'K':>6s}  {'N':>6s}",
    )


if __name__ == "__main__":
    main()
