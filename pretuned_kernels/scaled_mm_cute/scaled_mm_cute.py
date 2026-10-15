"""FP8 scaled GEMM for the B200 and GB300 CuTe (tcgen05) backends.

B200's row-wise kernels retain the serving-shape schedules and scale views
from the original ``scale_mm_cute`` preset. GB300 adds scalar/token activation
scales and scalar/channel weight scales for five shapes in all four modes.

* :func:`scaled_mm_cute`, :func:`scaled_mm_cute_skinny_m`, and
  :func:`scaled_mm_cute_swap_ab` retain the B200 implementations.
* :func:`scaled_mm_cute_sm103` tiles over M and N for the GB300 schedules.
* :func:`scaled_mm_transposed_cute` computes ``out.T = b.T @ a.T`` for GB300's
  ``(M, K, N) = (256, 4096, 2048)`` shape using transposed input views.
* :func:`scaled_mm_skinny_cute` keeps the single row resident for GB300 decode.

:func:`scaled_mm` chooses the architecture's kernel family and dispatches by
shape. It accepts scalar/vector scales and legacy row-broadcast activation
scale views without materializing scale tensors or transposing input storage.
"""

from __future__ import annotations

import math
import os
import sys

import torch

import helion
import helion.language as hl

# Only single-token decode uses the scalar skinny-M kernel. Wider small-M
# shapes use the swapped tensor-core kernel below.
_SKINNY_M_MAX = 1
_SMALL_M_SWAP_SHAPES = {
    (m, k, n)
    for m in (2, 8, 16, 32)
    for k, n in (
        (4096, 4096),
        (4096, 256),
        (2048, 4096),
        (4096, 6144),
        (2048, 12288),
        (5120, 5120),
        (6144, 2048),
    )
}
_M64_SWAP_SHAPES = {
    (64, 4096, 24576),
    (64, 5120, 51200),
    (64, 25600, 5120),
}
_SWAP_AB_SHAPES = _SMALL_M_SWAP_SHAPES | _M64_SWAP_SHAPES
_M512_OPT_SHAPE = (512, 2048, 4096)


# Architecture-specific measured sweeps. SHAPES exposes their logical union.
B200_SHAPES = [  # (M, K, N)
    (1, 4096, 4096),
    (2, 4096, 4096),
    (2, 4096, 256),
    (2, 2048, 4096),
    (2, 4096, 6144),
    (8, 4096, 4096),
    (8, 4096, 256),
    (8, 2048, 4096),
    (8, 4096, 6144),
    (16, 4096, 4096),
    (16, 4096, 256),
    (16, 2048, 4096),
    (16, 4096, 6144),
    (32, 4096, 4096),
    (32, 4096, 256),
    (32, 2048, 4096),
    (32, 4096, 6144),
    *sorted(
        (m, k, n)
        for m, k, n in _SMALL_M_SWAP_SHAPES
        if (k, n) in {(2048, 12288), (5120, 5120), (6144, 2048)}
    ),
    (4096, 4096, 4096),
    (1, 4096, 256),
    (512, 2048, 4096),
    (512, 2048, 2048),
    (64, 2048, 4096),
    (64, 2048, 2048),
    (64, 2048, 12288),
    (64, 6144, 2048),
    (64, 4096, 6144),
    (64, 4096, 4096),
    (64, 4096, 24576),
    (64, 12288, 4096),
    (64, 5120, 10240),
    (64, 5120, 5120),
    (64, 5120, 51200),
    (64, 25600, 5120),
]

GB300_SHAPES = (  # (M, K, N)
    (1, 4096, 4096),
    (64, 4096, 4096),
    (256, 4096, 2048),
    (128, 2880, 5120),
    (4096, 4096, 2880),
)
SHAPES = tuple(dict.fromkeys((*B200_SHAPES, *GB300_SHAPES)))
SCALE_MODES = ("tensor-tensor", "tensor-channel", "token-tensor", "token-channel")
CASES = tuple((*shape, mode) for mode in SCALE_MODES for shape in GB300_SHAPES)


@helion.aot_kernel(backend="cute", static_shapes=True)
def scaled_mm_cute(
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
    use_grouped_scale = hl.constexpr(m == 512 and k == 2048 and n == 4096)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = hl.dot(x[tile_m, tile_k], y[tile_k, tile_n], acc=acc)
        if use_grouped_scale:
            # This shape gate exists because it is currently the only pretuned
            # shape using the four-CTA schedule, which overlaps the independent
            # scale product with its MMA tail. Keep the sequential form elsewhere:
            # grouping spills at the 255-register limit, e.g. 4096^3 schedule.
            tile_scale = scale_a[tile_m, tile_n] * scale_b[tile_n]
            out[tile_m, tile_n] = (acc * tile_scale).to(torch.bfloat16)
        else:
            # RowWise scale in the epilogue (per-row scale_a x per-column scale_b).
            acc = acc * scale_a[tile_m, tile_n] * scale_b[tile_n]
            out[tile_m, tile_n] = acc.to(torch.bfloat16)
    return out


@helion.aot_kernel(backend="cute", static_shapes=True)
def scaled_mm_cute_skinny_m(
    x: torch.Tensor,  # [M, K] fp8
    y: torch.Tensor,  # [K, N] fp8
    scale_a: torch.Tensor,  # [M, N] per-row scale broadcast to [M, N]
    scale_b: torch.Tensor,  # [N] per-column scale
) -> torch.Tensor:
    """Skinny-M variant of :func:`scaled_mm_cute` for the CuTe (tcgen05) backend.

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


@helion.aot_kernel(backend="cute", static_shapes=True)
def scaled_mm_cute_swap_ab(
    x: torch.Tensor,
    y: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
) -> torch.Tensor:
    """Small-M FP8 GEMM with M/N swapped for efficient tensor-core tiles."""
    m, k = x.size()
    k2, n = y.size()
    assert k == k2, f"size mismatch {k} != {k2}"
    out = torch.empty([m, n], dtype=torch.bfloat16, device=x.device)
    out_t = out.T
    scale_a_t = scale_a.unsqueeze(0)
    scale_b_t = scale_b.unsqueeze(-1).expand(n, m)
    for tile_n, tile_m in hl.tile([n, m]):
        acc = hl.zeros([tile_n, tile_m], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = hl.dot(
                y[tile_k, tile_n].T,
                x[tile_m, tile_k].T,
                acc=acc,
            )
        out_t[tile_n, tile_m] = (
            acc * scale_a_t[tile_n, tile_m] * scale_b_t[tile_n, tile_m]
        ).to(torch.bfloat16)
    return out


def _scaled_mm_cute(
    x: torch.Tensor,
    y: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
) -> torch.Tensor:
    """Dispatch each pretuned shape to its specialized CuTe kernel."""
    if x.size(0) <= _SKINNY_M_MAX:
        return scaled_mm_cute_skinny_m(x, y, scale_a, scale_b)
    shape = (x.size(0), x.size(1), y.size(1))
    if shape in _SWAP_AB_SHAPES:
        return scaled_mm_cute_swap_ab(x, y, scale_a[:, 0], scale_b)
    return scaled_mm_cute(x, y, scale_a, scale_b)


@helion.aot_kernel(
    backend="cute",
    static_shapes=True,
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def scaled_mm_cute_sm103(
    a: torch.Tensor, b: torch.Tensor, scale_a: torch.Tensor, scale_b: torch.Tensor
) -> torch.Tensor:
    m, k = a.shape
    n = b.shape[1]
    assert b.shape[0] == k and a.dtype == b.dtype == torch.float8_e4m3fn
    assert a.stride(1) == 1 and b.stride(0) == 1
    assert scale_a.ndim <= 1 and scale_b.ndim <= 1
    assert scale_a.numel() in (1, m) and scale_b.numel() in (1, n)
    tensor_scale_a = hl.constexpr(scale_a.ndim == 0)
    tensor_scale_b = hl.constexpr(scale_b.ndim == 0)
    sa = scale_a.reshape(-1, 1).expand(m, n)
    out = torch.empty((m, n), device=a.device, dtype=torch.bfloat16)
    for rows, cols in hl.tile([m, n]):
        acc = hl.zeros([rows, cols], dtype=torch.float32)
        for inner in hl.tile(k):
            acc = hl.dot(a[rows, inner], b[inner, cols], acc=acc)
        if tensor_scale_a:
            tile_scale_a = scale_a[()]
        else:
            tile_scale_a = sa[rows, cols]
        if tensor_scale_b:
            tile_scale_b = scale_b[()]
        else:
            tile_scale_b = scale_b[cols]
        out[rows, cols] = (acc * tile_scale_a * tile_scale_b).to(torch.bfloat16)
    return out


@helion.aot_kernel(
    backend="cute",
    static_shapes=True,
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def scaled_mm_transposed_cute(
    b_t: torch.Tensor,
    a_t: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
) -> torch.Tensor:
    """Compute ``out.T = b.T @ a.T`` and return contiguous logical ``out``.

    The operands are already transposed views: ``b_t[N, K]`` and ``a_t[K, M]``.
    Scales keep their original meanings: ``scale_a`` indexes logical M, and
    ``scale_b`` indexes logical N. Multiplication retains ``acc * scale_a *
    scale_b`` order even though the operands and tile axes are swapped.
    """
    n, k = b_t.shape
    m = a_t.shape[1]
    assert a_t.shape[0] == k and a_t.dtype == b_t.dtype == torch.float8_e4m3fn
    assert b_t.stride(1) == 1 and a_t.stride(0) == 1
    assert scale_a.ndim <= 1 and scale_b.ndim <= 1
    assert scale_a.numel() in (1, m) and scale_b.numel() in (1, n)
    tensor_scale_a = hl.constexpr(scale_a.ndim == 0)
    tensor_scale_b = hl.constexpr(scale_b.ndim == 0)
    sb_t = scale_b.reshape(-1, 1).expand(n, m)
    out = torch.empty((m, n), device=a_t.device, dtype=torch.bfloat16)
    out_t = out.T
    for cols, rows in hl.tile([n, m]):
        acc = hl.zeros([cols, rows], dtype=torch.float32)
        for inner in hl.tile(k):
            acc = hl.dot(b_t[cols, inner], a_t[inner, rows], acc=acc)
        if tensor_scale_a:
            tile_scale_a = scale_a[()]
        else:
            tile_scale_a = scale_a[rows]
        if tensor_scale_b:
            tile_scale_b = scale_b[()]
        else:
            tile_scale_b = sb_t[cols, rows]
        out_t[cols, rows] = (acc * tile_scale_a * tile_scale_b).to(torch.bfloat16)
    return out


@helion.aot_kernel(backend="cute", static_shapes=True)
def scaled_mm_skinny_cute(
    a: torch.Tensor, b: torch.Tensor, scale_a: torch.Tensor, scale_b: torch.Tensor
) -> torch.Tensor:
    m, k = a.shape
    n = b.shape[1]
    assert m == 1 and b.shape[0] == k
    assert a.dtype == b.dtype == torch.float8_e4m3fn
    assert a.stride(1) == 1 and b.stride(0) == 1
    assert scale_a.ndim == 0 and scale_b.ndim <= 1
    assert scale_b.numel() in (1, n)
    tensor_scale_b = hl.constexpr(scale_b.ndim == 0)
    out = torch.empty((m, n), device=a.device, dtype=torch.bfloat16)
    for cols in hl.tile(n):
        acc = hl.zeros([m, cols], dtype=torch.float32)
        for inner in hl.tile(k):
            acc = hl.dot(a[:, inner], b[inner, cols], acc=acc)
        acc = acc * scale_a[()]
        if tensor_scale_b:
            acc = acc * scale_b[()]
        else:
            acc = acc * scale_b[cols]
        out[:, cols] = acc.bfloat16()
    return out


def _device_capability(device: torch.device) -> tuple[int, int]:
    """Return the CUDA architecture used to choose the recorded kernel family."""
    return torch.cuda.get_device_capability(device)


def _normalize_scale_a(scale: torch.Tensor, m: int, n: int) -> torch.Tensor:
    """Reduce activation scales to a scalar or M-vector using views only."""
    if scale.numel() == 1:
        return scale.reshape(())
    if scale.ndim == 2 and tuple(scale.shape) == (m, n):
        assert n == 1 or scale.stride(1) == 0, "scale_a must broadcast across N"
        scale = scale[:, 0]
    else:
        scale = scale.squeeze()
    if scale.numel() == 1:
        return scale.reshape(())
    assert scale.ndim == 1 and scale.numel() == m
    return scale


def _normalize_scale_b(scale: torch.Tensor, n: int) -> torch.Tensor:
    """Reduce weight scales to a scalar or N-vector using views only."""
    if scale.numel() == 1:
        return scale.reshape(())
    scale = scale.squeeze()
    assert scale.ndim == 1 and scale.numel() == n
    return scale


def scaled_mm(
    a: torch.Tensor, b: torch.Tensor, scale_a: torch.Tensor, scale_b: torch.Tensor
) -> torch.Tensor:
    """Normalize scale views and dispatch to the architecture and shape's kernel."""
    assert a.ndim == b.ndim == 2 and a.shape[1] == b.shape[0]
    assert a.dtype == b.dtype == torch.float8_e4m3fn
    assert scale_a.dtype == scale_b.dtype == torch.float32
    m, k = a.shape
    n = b.shape[1]
    sa = _normalize_scale_a(scale_a, m, n)
    sb = _normalize_scale_b(scale_b, n)
    capability = _device_capability(a.device)
    if capability == (10, 0):
        # Preserve the row-broadcast views used by the measured B200 kernels.
        sa_rows = sa.reshape(-1, 1).expand(m, n)
        sb_cols = sb.reshape(-1).expand(n)
        return _scaled_mm_cute(a, b, sa_rows, sb_cols)
    if capability != (10, 3):
        raise ValueError(f"scaled_mm_cute has no pretuned kernels for SM{capability}")
    if m == 1:
        return scaled_mm_skinny_cute(a, b, sa, sb)
    if (m, k, n) == (256, 4096, 2048):
        return scaled_mm_transposed_cute(b.T, a.T, sa, sb)
    return scaled_mm_cute_sm103(a, b, sa, sb)


def _make_inputs_sm100(
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


def _make_inputs(
    m: int, k: int, n: int, mode: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mode_a, mode_b = mode.split("-")
    generator = torch.Generator(device="cuda").manual_seed(0)
    a = (torch.randn((m, k), device="cuda", generator=generator) / 3).to(
        torch.float8_e4m3fn
    )
    weights = (torch.randn((n, k), device="cuda", generator=generator) / 3).to(
        torch.float8_e4m3fn
    )
    scale_a = 0.25 + torch.rand(
        (m if mode_a == "token" else 1,), device="cuda", generator=generator
    )
    scale_b = 0.25 + torch.rand(
        (n if mode_b == "channel" else 1,), device="cuda", generator=generator
    )
    return a, weights.T, scale_a, scale_b


def _scaled_mm_torch_sm100(
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


def _scaled_mm_torch(
    a: torch.Tensor, b: torch.Tensor, scale_a: torch.Tensor, scale_b: torch.Tensor
) -> torch.Tensor:
    """torch._scaled_mm baseline for all four scale modes.

    torch._scaled_mm accepts either two scalar scales or row-wise ``[M, 1]`` and
    ``[1, N]`` scales. A mixed mode (tensor-channel or token-tensor) therefore
    expands its scalar side on every call, so that copy is part of the timed
    baseline, matching a caller that keeps the scale on the device.
    """
    sa = scale_a.reshape(-1, 1)
    sb = scale_b.reshape(1, -1)
    if sa.numel() != 1 or sb.numel() != 1:
        sa = sa.expand(a.shape[0], 1).contiguous()
        sb = sb.expand(1, b.shape[1]).contiguous()
    return torch._scaled_mm(
        a,
        b,
        scale_a=sa,
        scale_b=sb,
        use_fast_accum=False,
        out_dtype=torch.bfloat16,
    )


# Optional vLLM CUTLASS baseline for the serving benchmark environments.
try:
    from vllm import _custom_ops as _vllm_ops

    _HAS_VLLM = hasattr(_vllm_ops, "cutlass_scaled_mm")
except ImportError:
    _vllm_ops = None
    _HAS_VLLM = False


def _scaled_mm_cutlass_sm100(
    x: torch.Tensor,
    y: torch.Tensor,
    scale_a: torch.Tensor,  # [M, N] broadcast view of the per-row scale
    scale_b: torch.Tensor,  # [N] per-column scale
) -> torch.Tensor:
    """vLLM CUTLASS RowWise baseline (ops.cutlass_scaled_mm)."""
    sa = scale_a[:, :1].contiguous()
    sb = scale_b.reshape(-1, 1).contiguous()
    return _vllm_ops.cutlass_scaled_mm(x, y, sa, sb, torch.bfloat16, None)


def _scaled_mm_cutlass(
    a: torch.Tensor, b: torch.Tensor, scale_a: torch.Tensor, scale_b: torch.Tensor
) -> torch.Tensor:
    return _vllm_ops.cutlass_scaled_mm(a, b, scale_a, scale_b, torch.bfloat16, None)


def _baselines_sm100() -> list[tuple[str, object]]:
    """Baselines main() benchmarks helion against.

    torch._scaled_mm is always available; vLLM's CUTLASS kernel is added when
    vLLM is installed (the nightly benchmark env). The SUMMARY speedup is helion
    vs the best (fastest) available baseline.
    """
    baselines: list[tuple[str, object]] = [("torch", _scaled_mm_torch_sm100)]
    if _HAS_VLLM:
        baselines.append(("cutlass", _scaled_mm_cutlass_sm100))
    return baselines


def _baselines_sm103() -> list[tuple[str, object]]:
    baselines: list[tuple[str, object]] = [("torch", _scaled_mm_torch)]
    if _HAS_VLLM:
        baselines.append(("cutlass", _scaled_mm_cutlass))
    return baselines


def use_cudagraph() -> bool:
    """Use the CUDA-graph timing regime recorded for both architecture sweeps."""
    return True


def _main_sm100(verbose: bool = True) -> dict:
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from _bench import run_sweep

    baselines = _baselines_sm100()

    def make_calls(shape: tuple[int, int, int]) -> tuple:
        M, K, N = shape
        x, y, scale_a, scale_b = _make_inputs_sm100(M, K, N)

        def helion_call() -> None:
            _scaled_mm_cute(x, y, scale_a, scale_b)

        base_calls = [
            (name, (lambda fn=fn: fn(x, y, scale_a, scale_b))) for name, fn in baselines
        ]
        return helion_call, base_calls, f"{M:>6d}  {K:>6d}  {N:>6d}"

    # run_sweep benchmarks under CUDA graphs (how these decode / small-batch GEMMs
    # are invoked; removes per-call host launch overhead) with L2 cache clearing
    # before every replay (see _bench) -- the fp8 configs are tuned for this
    # cold-L2 regime (PR #2821: the 2-CTA A-multicast win only shows cold).
    return run_sweep(
        B200_SHAPES,
        make_calls,
        use_cudagraph=use_cudagraph(),
        verbose=verbose,
        shape_header=f"{'M':>6s}  {'K':>6s}  {'N':>6s}",
    )


def _main_sm103(verbose: bool = True) -> dict:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from _bench import run_sweep

    baselines = _baselines_sm103()

    def make_calls(case: tuple[int, int, int, str]) -> tuple:
        m, k, n, mode = case
        a, b, scale_a, scale_b = _make_inputs(m, k, n, mode)

        def helion_call() -> torch.Tensor:
            return scaled_mm(a, b, scale_a, scale_b)

        base_calls = [
            (name, (lambda fn=fn: fn(a, b, scale_a, scale_b))) for name, fn in baselines
        ]
        return helion_call, base_calls, f"{m:>6d}  {k:>6d}  {n:>6d}  {mode:>14s}"

    return run_sweep(
        CASES,
        make_calls,
        use_cudagraph=use_cudagraph(),
        verbose=verbose,
        shape_header=f"{'M':>6s}  {'K':>6s}  {'N':>6s}  {'scale mode':>14s}",
    )


def main(verbose: bool = True) -> dict:
    """Benchmark the current architecture's recorded shape and scale-mode sweep."""
    capability = _device_capability(torch.device("cuda"))
    if capability == (10, 0):
        return _main_sm100(verbose=verbose)
    if capability == (10, 3):
        return _main_sm103(verbose=verbose)
    raise ValueError(f"scaled_mm_cute has no pretuned kernels for SM{capability}")


if __name__ == "__main__":
    main()
