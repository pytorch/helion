"""Low-latency NVFP4 GEMV for decode (batch-size-1) inference on Blackwell --
Helion CuTe backend.

The CuTe counterpart of ``pretuned_kernels/nvfp4_gemv`` (the Triton variant):
same two decode regimes, same NVFP4 weight layout (packed E2M1 bytes with
per-16-value E4M3 block scales in PyTorch's SWIZZLE_32_4_4 layout), but backed by
actual Helion DSL kernels compiled with ``backend="cute"``:

* :func:`_nvfp4_gemv_fp4in` -- NVFP4 weight * NVFP4 activation (W4A4).
* :func:`_nvfp4_gemv_bf16in` -- NVFP4 weight * BF16 activation (W4A16).

The kernels load each 16-value group as one aligned 64-bit word, decode FP4 (and
BF16 activations) to FP16 with CuTe inline PTX, and accumulate in FP32. Programs
compute two output rows for ``N <= 8192`` and four rows for wider outputs, which
balances register pressure against activation reuse. Checked-in B200 CuTe
configs select the reduction tile size. The implementation goes through
Helion's normal compilation, configuration, caching, and launch path; there is
no direct ``@cute.kernel`` or ``default_cute_launcher`` shim.

Benchmarked against the production vLLM CUTLASS NVFP4 GEMM
(``ops.cutlass_scaled_fp4_mm`` -- the NVFP4 analog of ``cutlass_scaled_mm``,
which has no dedicated GEMV, so decode is served by the M=1 GEMM) and
``torch.compile`` of the NVFP4 dequant reference. The eager dequant reference is
used only for a one-shot correctness check per shape (it is orders of magnitude
slower than the kernel, so it is not a timed baseline).
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import TYPE_CHECKING
from typing import cast

import torch

import helion
from helion._testing import import_path
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Callable

_EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"
_ex = import_path(_EXAMPLES_DIR / "nvfp4_gemv.py")
make_fp8_scales = _ex.make_fp8_scales
reference_nvfp4_gemv_fp4in = _ex.reference_nvfp4_gemv_fp4in
reference_nvfp4_gemv_bf16in = _ex.reference_nvfp4_gemv_bf16in
swizzled_scale_offsets = _ex.swizzled_scale_offsets

# torch.compile of the NVFP4 dequant references -- a speedup-comparison baseline
# only (correctness is checked against the eager reference).
compiled_reference_nvfp4_gemv_fp4in = torch.compile(reference_nvfp4_gemv_fp4in)
compiled_reference_nvfp4_gemv_bf16in = torch.compile(reference_nvfp4_gemv_bf16in)


def _scaled_row_partial(
    row: torch.SymInt,
    tile_g: hl.Tile,
    weight_bytes: torch.Tensor,
    weight_scale: torch.Tensor,
    x: tuple[torch.Tensor, ...],
    x_scale_value: torch.Tensor | None,
    groups: int,
    rows: int,
) -> torch.Tensor:
    row_index = cast("int", row)
    group_mask = tile_g.index < groups
    row_mask = row_index < rows
    weight = hl.load_float4_e2m1fn_x16_to_float16(
        weight_bytes,
        row_index * groups + tile_g.index,
        extra_mask=group_mask & row_mask,
    )
    contribution = hl.zeros([tile_g], dtype=torch.float16)
    for i in hl.static_range(16):
        contribution = contribution + weight[i] * x[i]
    scale_offsets = swizzled_scale_offsets(row_index, tile_g.index, groups)
    scale = hl.load(
        weight_scale,
        [scale_offsets],
        extra_mask=group_mask & row_mask,
    ).to(torch.float32)
    if x_scale_value is not None:
        scale = scale * x_scale_value
    return (contribution.to(torch.float32) * scale).sum()


@helion.aot_kernel(backend="cute", static_shapes=True)
def nvfp4_gemv_bf16in_rows2_kernel(
    weight_bytes: torch.Tensor,
    x_values: torch.Tensor,
    weight_scale: torch.Tensor,
    out: torch.Tensor,
    alpha: float = 1.0,
) -> torch.Tensor:
    """Helion CuTe W4A16 GEMV with coalesced FP4 decode."""
    M, K_bytes = weight_bytes.shape
    K_groups = K_bytes // 8
    block_g = hl.register_block_size(16, K_groups)
    for program in hl.grid((M + 1) // 2):
        row0 = program * 2
        row1 = row0 + 1
        acc0 = hl.zeros([], dtype=torch.float32)
        acc1 = hl.zeros([], dtype=torch.float32)
        for tile_g in hl.tile(K_groups, block_size=block_g):
            group_mask = tile_g.index < K_groups
            x = hl.load_bfloat16_x16_to_float16(
                x_values,
                tile_g.index,
                extra_mask=group_mask,
            )
            acc0 = acc0 + _scaled_row_partial(
                row0, tile_g, weight_bytes, weight_scale, x, None, K_groups, M
            )
            acc1 = acc1 + _scaled_row_partial(
                row1, tile_g, weight_bytes, weight_scale, x, None, K_groups, M
            )
        hl.store(
            out,
            [row0],
            (acc0 * alpha).to(torch.bfloat16),
            extra_mask=row0 < M,
        )
        hl.store(
            out,
            [row1],
            (acc1 * alpha).to(torch.bfloat16),
            extra_mask=row1 < M,
        )
    return out


@helion.aot_kernel(backend="cute", static_shapes=True)
def nvfp4_gemv_fp4in_rows2_kernel(
    weight_bytes: torch.Tensor,
    x_bytes: torch.Tensor,
    weight_scale: torch.Tensor,
    x_scale: torch.Tensor,
    out: torch.Tensor,
    alpha: float = 1.0,
) -> torch.Tensor:
    """Helion CuTe W4A4 GEMV with coalesced FP4 decode."""
    M, K_bytes = weight_bytes.shape
    K_groups = K_bytes // 8
    block_g = hl.register_block_size(16, K_groups)
    for program in hl.grid((M + 1) // 2):
        row0 = program * 2
        row1 = row0 + 1
        acc0 = hl.zeros([], dtype=torch.float32)
        acc1 = hl.zeros([], dtype=torch.float32)
        for tile_g in hl.tile(K_groups, block_size=block_g):
            group_mask = tile_g.index < K_groups
            x = hl.load_float4_e2m1fn_x16_to_float16(
                x_bytes,
                tile_g.index,
                extra_mask=group_mask,
            )
            x_scale_offsets = swizzled_scale_offsets(
                tile_g.index * 0, tile_g.index, K_groups
            )
            x_scale_value = hl.load(
                x_scale,
                [x_scale_offsets],
                extra_mask=group_mask,
            ).to(torch.float32)
            acc0 = acc0 + _scaled_row_partial(
                row0,
                tile_g,
                weight_bytes,
                weight_scale,
                x,
                x_scale_value,
                K_groups,
                M,
            )
            acc1 = acc1 + _scaled_row_partial(
                row1,
                tile_g,
                weight_bytes,
                weight_scale,
                x,
                x_scale_value,
                K_groups,
                M,
            )
        hl.store(
            out,
            [row0],
            (acc0 * alpha).to(torch.bfloat16),
            extra_mask=row0 < M,
        )
        hl.store(
            out,
            [row1],
            (acc1 * alpha).to(torch.bfloat16),
            extra_mask=row1 < M,
        )
    return out


@helion.aot_kernel(backend="cute", static_shapes=True)
def nvfp4_gemv_bf16in_rows4_kernel(
    weight_bytes: torch.Tensor,
    x_values: torch.Tensor,
    weight_scale: torch.Tensor,
    out: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    M, K_bytes = weight_bytes.shape
    K_groups = K_bytes // 8
    block_g = hl.register_block_size(16, K_groups)
    for program in hl.grid((M + 3) // 4):
        row0 = program * 4
        row1 = row0 + 1
        row2 = row0 + 2
        row3 = row0 + 3
        acc0 = hl.zeros([], dtype=torch.float32)
        acc1 = hl.zeros([], dtype=torch.float32)
        acc2 = hl.zeros([], dtype=torch.float32)
        acc3 = hl.zeros([], dtype=torch.float32)
        for tile_g in hl.tile(K_groups, block_size=block_g):
            group_mask = tile_g.index < K_groups
            x = hl.load_bfloat16_x16_to_float16(
                x_values,
                tile_g.index,
                extra_mask=group_mask,
            )
            acc0 = acc0 + _scaled_row_partial(
                row0, tile_g, weight_bytes, weight_scale, x, None, K_groups, M
            )
            acc1 = acc1 + _scaled_row_partial(
                row1, tile_g, weight_bytes, weight_scale, x, None, K_groups, M
            )
            acc2 = acc2 + _scaled_row_partial(
                row2, tile_g, weight_bytes, weight_scale, x, None, K_groups, M
            )
            acc3 = acc3 + _scaled_row_partial(
                row3, tile_g, weight_bytes, weight_scale, x, None, K_groups, M
            )
        hl.store(
            out,
            [row0],
            (acc0 * alpha).to(torch.bfloat16),
            extra_mask=row0 < M,
        )
        hl.store(
            out,
            [row1],
            (acc1 * alpha).to(torch.bfloat16),
            extra_mask=row1 < M,
        )
        hl.store(
            out,
            [row2],
            (acc2 * alpha).to(torch.bfloat16),
            extra_mask=row2 < M,
        )
        hl.store(
            out,
            [row3],
            (acc3 * alpha).to(torch.bfloat16),
            extra_mask=row3 < M,
        )
    return out


@helion.aot_kernel(backend="cute", static_shapes=True)
def nvfp4_gemv_fp4in_rows4_kernel(
    weight_bytes: torch.Tensor,
    x_bytes: torch.Tensor,
    weight_scale: torch.Tensor,
    x_scale: torch.Tensor,
    out: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    M, K_bytes = weight_bytes.shape
    K_groups = K_bytes // 8
    block_g = hl.register_block_size(16, K_groups)
    for program in hl.grid((M + 3) // 4):
        row0 = program * 4
        row1 = row0 + 1
        row2 = row0 + 2
        row3 = row0 + 3
        acc0 = hl.zeros([], dtype=torch.float32)
        acc1 = hl.zeros([], dtype=torch.float32)
        acc2 = hl.zeros([], dtype=torch.float32)
        acc3 = hl.zeros([], dtype=torch.float32)
        for tile_g in hl.tile(K_groups, block_size=block_g):
            group_mask = tile_g.index < K_groups
            x = hl.load_float4_e2m1fn_x16_to_float16(
                x_bytes,
                tile_g.index,
                extra_mask=group_mask,
            )
            x_scale_offsets = swizzled_scale_offsets(
                tile_g.index * 0, tile_g.index, K_groups
            )
            x_scale_value = hl.load(
                x_scale,
                [x_scale_offsets],
                extra_mask=group_mask,
            ).to(torch.float32)
            acc0 = acc0 + _scaled_row_partial(
                row0,
                tile_g,
                weight_bytes,
                weight_scale,
                x,
                x_scale_value,
                K_groups,
                M,
            )
            acc1 = acc1 + _scaled_row_partial(
                row1,
                tile_g,
                weight_bytes,
                weight_scale,
                x,
                x_scale_value,
                K_groups,
                M,
            )
            acc2 = acc2 + _scaled_row_partial(
                row2,
                tile_g,
                weight_bytes,
                weight_scale,
                x,
                x_scale_value,
                K_groups,
                M,
            )
            acc3 = acc3 + _scaled_row_partial(
                row3,
                tile_g,
                weight_bytes,
                weight_scale,
                x,
                x_scale_value,
                K_groups,
                M,
            )
        hl.store(
            out,
            [row0],
            (acc0 * alpha).to(torch.bfloat16),
            extra_mask=row0 < M,
        )
        hl.store(
            out,
            [row1],
            (acc1 * alpha).to(torch.bfloat16),
            extra_mask=row1 < M,
        )
        hl.store(
            out,
            [row2],
            (acc2 * alpha).to(torch.bfloat16),
            extra_mask=row2 < M,
        )
        hl.store(
            out,
            [row3],
            (acc3 * alpha).to(torch.bfloat16),
            extra_mask=row3 < M,
        )
    return out


def _as_fp4x2(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype is torch.float4_e2m1fn_x2:
        return tensor
    if tensor.dtype is torch.uint8:
        return tensor.view(torch.float4_e2m1fn_x2)
    raise TypeError(f"expected uint8 or float4_e2m1fn_x2 tensor, got {tensor.dtype}")


def _nvfp4_gemv_fp4in(
    weight_packed: torch.Tensor,
    x_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    x_scale: torch.Tensor,
    alpha: float = 1.0,
) -> torch.Tensor:
    """Run the Helion CuTe W4A4 kernel."""
    _ex._check_contiguous("weight_packed", weight_packed)
    _ex._check_contiguous("x_packed", x_packed)
    weight_fp4x2 = _as_fp4x2(weight_packed)
    x_fp4x2 = _as_fp4x2(x_packed)
    weight_bytes = weight_fp4x2.view(torch.uint8)
    x_bytes = x_fp4x2.view(torch.uint8)
    _ex._check_fp4_weight_storage("weight_packed", weight_bytes)
    _ex._check_numel("x_packed", x_bytes, weight_bytes.shape[1])
    groups = weight_bytes.shape[1] // 8
    _ex._check_swizzled_scales(
        "weight_scale", weight_scale, weight_bytes.shape[0], groups
    )
    _ex._check_swizzled_scales("x_scale", x_scale, 1, groups)
    out = torch.empty(
        weight_bytes.shape[0], dtype=torch.bfloat16, device=weight_bytes.device
    )
    kernel = (
        nvfp4_gemv_fp4in_rows4_kernel
        if weight_bytes.shape[0] > 8192
        else nvfp4_gemv_fp4in_rows2_kernel
    )
    return kernel(
        weight_bytes,
        x_bytes,
        weight_scale.reshape(-1),
        x_scale.reshape(-1),
        out,
        alpha,
    )


def _nvfp4_gemv_bf16in(
    weight_packed: torch.Tensor,
    x_bf16: torch.Tensor,
    weight_scale: torch.Tensor,
    alpha: float = 1.0,
) -> torch.Tensor:
    """Run the Helion CuTe W4A16 kernel."""
    _ex._check_contiguous("weight_packed", weight_packed)
    _ex._check_contiguous("x_bf16", x_bf16)
    weight_fp4x2 = _as_fp4x2(weight_packed)
    weight_bytes = weight_fp4x2.view(torch.uint8)
    _ex._check_fp4_weight_storage("weight_packed", weight_bytes)
    _ex._check_numel("x_bf16", x_bf16, weight_bytes.shape[1] * 2)
    groups = weight_bytes.shape[1] // 8
    _ex._check_swizzled_scales(
        "weight_scale", weight_scale, weight_bytes.shape[0], groups
    )
    out = torch.empty(
        weight_bytes.shape[0], dtype=torch.bfloat16, device=weight_bytes.device
    )
    kernel = (
        nvfp4_gemv_bf16in_rows4_kernel
        if weight_bytes.shape[0] > 8192
        else nvfp4_gemv_bf16in_rows2_kernel
    )
    return kernel(
        weight_bytes,
        x_bf16.view(groups, 16),
        weight_scale.reshape(-1),
        out,
        alpha,
    )


def _check(
    got: torch.Tensor, expected: torch.Tensor, variant: str, n: int, k: int
) -> None:
    """Correctness check vs the eager dequant reference (run once, never timed)."""
    torch.testing.assert_close(
        got.float(),
        expected.float(),
        atol=4.0,
        rtol=2e-1,
        msg=lambda m: f"{variant} N={n} K={k} mismatch vs reference:\n{m}",
    )


# Optional vLLM CUTLASS NVFP4 baseline (see the Triton variant for details).
try:
    from vllm import _custom_ops as _vllm_ops

    _HAS_VLLM = hasattr(torch.ops._C, "cutlass_scaled_fp4_mm")
    FLOAT4_E2M1_MAX = 6.0
    FLOAT8_E4M3_MAX = float(torch.finfo(torch.float8_e4m3fn).max)
except ImportError:
    _vllm_ops = None
    _HAS_VLLM = False


def _vllm_quant_weight(weight_bf16: torch.Tensor) -> tuple:
    amax = weight_bf16.abs().max().to(torch.float32)
    global_scale = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / amax
    weight_fp4, weight_sf = _vllm_ops.scaled_fp4_quant(weight_bf16, global_scale)
    return weight_fp4, weight_sf, global_scale


def _make_vllm_fp4in_call(
    n: int, k: int, device: torch.device
) -> Callable[[], torch.Tensor]:
    """vLLM W4A4 decode baseline: activation pre-quantized, then the CUTLASS GEMM."""
    weight_bf16 = torch.randn(n, k, device=device, dtype=torch.bfloat16)
    x_bf16 = torch.randn(1, k, device=device, dtype=torch.bfloat16)
    weight_fp4, weight_sf, weight_gs = _vllm_quant_weight(weight_bf16)
    x_gs = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / x_bf16.abs().max().to(torch.float32)
    alpha = 1.0 / (x_gs * weight_gs)
    x_fp4, x_sf = _vllm_ops.scaled_fp4_quant(x_bf16, x_gs)

    def run() -> torch.Tensor:
        return _vllm_ops.cutlass_scaled_fp4_mm(
            x_fp4, weight_fp4, x_sf, weight_sf, alpha, torch.bfloat16
        )

    return run


def _make_vllm_bf16in_call(
    n: int, k: int, device: torch.device
) -> Callable[[], torch.Tensor]:
    """vLLM W4A16 decode baseline: quantize the BF16 activation on the fly, GEMM."""
    weight_bf16 = torch.randn(n, k, device=device, dtype=torch.bfloat16)
    x_bf16 = torch.randn(1, k, device=device, dtype=torch.bfloat16)
    weight_fp4, weight_sf, weight_gs = _vllm_quant_weight(weight_bf16)
    x_gs = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / x_bf16.abs().max().to(torch.float32)
    alpha = 1.0 / (x_gs * weight_gs)

    def run() -> torch.Tensor:
        x_fp4, x_sf = _vllm_ops.scaled_fp4_quant(x_bf16, x_gs)
        return _vllm_ops.cutlass_scaled_fp4_mm(
            x_fp4, weight_fp4, x_sf, weight_sf, alpha, torch.bfloat16
        )

    return run


def use_cudagraph() -> bool:
    """Whether main() benchmarks under CUDA graphs (read by pretuned_kernels/run.py).

    True: decode (M=1) GEMVs are invoked one row at a time, where per-call host
    launch overhead dominates -- exactly how vLLM issues them. CUDA graphs remove
    that overhead; the shared _bench loop clears the L2 cache before every replay.
    """
    return True


# Decode (M=1) NVFP4 GEMV weight shapes (N=output features, K=reduction dim) for
# common projections. These match the Triton variant's shape coverage.
SHAPES = [  # (N, K)
    (4096, 4096),  # Llama-3-8B o_proj (square)
    (6144, 4096),  # Llama-3-8B qkv_proj (q4096 + kv1024*2)
    (28672, 4096),  # Llama-3-8B gate_up_proj (2 * 14336)
    (4096, 14336),  # Llama-3-8B down_proj
    (5120, 5120),  # 13B-class attention o_proj
    (15360, 5120),  # 13B-class qkv_proj (3 * 5120)
    (8192, 8192),  # 70B-class square projection (nvfp4_backend_comparison "o")
    (10240, 8192),  # wide fused projection
    (8192, 28672),  # nvfp4_backend_comparison "down": K_bytes=14336
]

_VARIANTS = ("fp4in", "bf16in")


def _make_fp4in_inputs(n: int, k: int) -> tuple:
    device = torch.device("cuda")
    k_bytes = k // 2
    weight = torch.randint(0, 256, (n, k_bytes), dtype=torch.uint8, device=device).view(
        torch.float4_e2m1fn_x2
    )
    x = torch.randint(0, 256, (k_bytes,), dtype=torch.uint8, device=device).view(
        torch.float4_e2m1fn_x2
    )
    weight_scale = make_fp8_scales((n, k_bytes // 8), device)
    x_scale = make_fp8_scales((k_bytes // 8,), device)
    return weight, x, weight_scale, x_scale


def _make_bf16in_inputs(n: int, k: int) -> tuple:
    device = torch.device("cuda")
    k_bytes = k // 2
    weight = torch.randint(0, 256, (n, k_bytes), dtype=torch.uint8, device=device).view(
        torch.float4_e2m1fn_x2
    )
    x = torch.randn(k, dtype=torch.bfloat16, device=device)
    weight_scale = make_fp8_scales((n, k_bytes // 8), device)
    return weight, x, weight_scale


def main(verbose: bool = True) -> dict:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from _bench import run_sweep

    device = torch.device("cuda")

    def make_calls(entry: tuple[str, int, int]) -> tuple:
        variant, n, k = entry
        if variant == "fp4in":
            weight, x, w_scale, x_scale = _make_fp4in_inputs(n, k)

            def helion_call() -> torch.Tensor:
                return _nvfp4_gemv_fp4in(weight, x, w_scale, x_scale)

            _check(
                helion_call(),
                reference_nvfp4_gemv_fp4in(weight, x, w_scale, x_scale),
                variant,
                n,
                k,
            )

            base_calls: list[tuple[str, Callable[[], torch.Tensor]]] = [
                (
                    "torch_compile",
                    lambda: compiled_reference_nvfp4_gemv_fp4in(
                        weight, x, w_scale, x_scale
                    ),
                ),
            ]
            if _HAS_VLLM:
                base_calls.append(("cutlass", _make_vllm_fp4in_call(n, k, device)))
        else:
            weight, x, w_scale = _make_bf16in_inputs(n, k)

            def helion_call() -> torch.Tensor:
                return _nvfp4_gemv_bf16in(weight, x, w_scale)

            _check(
                helion_call(),
                reference_nvfp4_gemv_bf16in(weight, x, w_scale),
                variant,
                n,
                k,
            )

            base_calls = [
                (
                    "torch_compile",
                    lambda: compiled_reference_nvfp4_gemv_bf16in(weight, x, w_scale),
                ),
            ]
            if _HAS_VLLM:
                base_calls.append(("cutlass", _make_vllm_bf16in_call(n, k, device)))
        return helion_call, base_calls, f"{variant:>6s}  {n:>6d}  {k:>6d}"

    entries = [(v, n, k) for v in _VARIANTS for (n, k) in SHAPES]
    return run_sweep(
        entries,
        make_calls,
        use_cudagraph=use_cudagraph(),
        verbose=verbose,
        shape_header=f"{'kind':>6s}  {'N':>6s}  {'K':>6s}",
    )


if __name__ == "__main__":
    main()
