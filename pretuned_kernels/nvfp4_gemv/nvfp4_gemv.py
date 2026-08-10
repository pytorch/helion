"""Low-latency NVFP4 GEMV for decode (batch-size-1) inference on Blackwell.

Computes ``out = (weight @ x) * alpha`` for a weight matrix stored as packed
NVFP4 (E2M1) bytes with per-16-value E4M3 block scales in PyTorch's
SWIZZLE_32_4_4 layout. Two variants ship, matching the two decode regimes:

* :func:`_nvfp4_gemv_fp4in` -- NVFP4 weight * NVFP4 activation (W4A4).
* :func:`_nvfp4_gemv_bf16in` -- NVFP4 weight * BF16 activation (W4A16).

These are the coalesced-load FP16-decode Triton GEMV kernels: the weight (and, for W4A4, the
activation) NVFP4 bytes are loaded and unpacked with
``hl.load_float4_e2m1fn_x16_to_float16``, the per-group dot runs in fp16, and
the E4M3 block scale is applied in fp32. They tile over M (a vector accumulator,
``out[tile_m]``), so ``block_m > 1`` is correct and reused across rows.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch

from helion._testing import import_path
import helion.experimental
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Callable

# Reuse the example's pure-python helpers (scale-swizzle offsets, fp8-scale
# builder, dequant references) so this module stays a thin, benchmark-only layer.
_EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"
_ex = import_path(_EXAMPLES_DIR / "nvfp4_gemv.py")
swizzled_scale_offsets = _ex.swizzled_scale_offsets
make_fp8_scales = _ex.make_fp8_scales
reference_nvfp4_gemv_fp4in = _ex.reference_nvfp4_gemv_fp4in
reference_nvfp4_gemv_bf16in = _ex.reference_nvfp4_gemv_bf16in

# torch.compile of the NVFP4 dequant references -- a speedup-comparison baseline
# only (correctness is checked against the eager reference). Compiled once at
# module scope so the compile cache is shared across the shape sweep.
compiled_reference_nvfp4_gemv_fp4in = torch.compile(reference_nvfp4_gemv_fp4in)
compiled_reference_nvfp4_gemv_bf16in = torch.compile(reference_nvfp4_gemv_bf16in)


def _check(
    got: torch.Tensor, expected: torch.Tensor, variant: str, n: int, k: int
) -> None:
    """Correctness check vs the eager dequant reference (run once, never timed).

    The eager reference is orders of magnitude slower than the kernel, so it is
    used only to validate output -- not as a timed baseline. Tolerances match
    the example's FP16-decode path (fp4/fp8 quantization dominates the error).
    """
    torch.testing.assert_close(
        got.float(),
        expected.float(),
        atol=4.0,
        rtol=2e-1,
        msg=lambda m: f"{variant} N={n} K={k} mismatch vs reference:\n{m}",
    )


# --------------------------------------------------------------------------- #
# FP16 scale decode helper (portable inline PTX; lowers on the Triton backend).
# --------------------------------------------------------------------------- #
def _e4m3_byte_to_f32(scale_byte: torch.Tensor) -> torch.Tensor:
    """Decode one E4M3 scale byte to fp32."""
    return hl.inline_asm_elementwise(
        """
        {
          .reg .b16 sc, scale_lo, scale_hi;
          .reg .b32 scale_h2;
          mov.b32 {sc, _}, $1;
          cvt.rn.f16x2.e4m3x2 scale_h2, sc;
          mov.b32 {scale_lo, scale_hi}, scale_h2;
          cvt.f32.f16 $0, scale_lo;
        }
        """,
        "=f,r",
        [scale_byte],
        dtype=torch.float32,
        is_pure=True,
        pack=1,
    )


# --------------------------------------------------------------------------- #
# Kernels: multi-row (block_m > 1 valid) FP16-decode NVFP4 GEMV.
# --------------------------------------------------------------------------- #
@helion.experimental.aot_kernel(backend="triton", static_shapes=True)
def nvfp4_gemv_bf16in_kernel(
    weight_bytes: torch.Tensor,  # (M, K_bytes) uint8 packed NVFP4 weight
    x_values: torch.Tensor,  # (K_groups, 16) bf16 activation
    weight_scale_bytes: torch.Tensor,  # int8 view of SWIZZLE_32_4_4 E4M3 scales
    out: torch.Tensor,  # (M,) bf16 output
    alpha: float = 1.0,
) -> torch.Tensor:
    """W4A16 NVFP4 GEMV: fp16 weight decode * (bf16 x cast to fp16), fp32 scale.

    Each scale group is loaded as one coalesced 64-bit word containing 16 packed
    FP4 values. The BF16 activation is loaded as a contiguous ``(block_g, 16)``
    tile before its lanes are selected. K tiles accumulate into a per-row fp32
    ``acc``.
    """
    M, K_bytes = weight_bytes.shape
    K_groups = K_bytes // 8
    block_m = hl.register_block_size(1, 16)
    block_g = hl.register_block_size(K_groups)
    f16 = torch.float16
    for tile_m in hl.tile(M, block_size=block_m):
        acc = hl.zeros([tile_m], dtype=torch.float32)
        for tile_g in hl.tile(K_groups, block_size=block_g):
            group_offsets = tile_m.index[:, None] * K_groups + tile_g.index[None, :]
            weight_mask = (tile_m.index[:, None] < M) & (
                tile_g.index[None, :] < K_groups
            )
            w = hl.load_float4_e2m1fn_x16_to_float16(
                weight_bytes,
                group_offsets,
                extra_mask=weight_mask,
            )
            xt = x_values[tile_g, :].to(f16)
            lane = hl.arange(16)[None, :]
            contrib = hl.zeros([block_m, block_g], dtype=f16)
            for i in hl.static_range(16):
                xi = torch.where(lane == i, xt, 0.0).sum(-1)[None, :]
                contrib = contrib + w[i] * xi
            scale_offsets = swizzled_scale_offsets(
                tile_m.index[:, None], tile_g.index[None, :], K_groups
            )
            scale = _e4m3_byte_to_f32(weight_scale_bytes[scale_offsets])
            acc = acc + (contrib.to(torch.float32) * scale).sum(-1)
        out[tile_m] = (acc * alpha).to(torch.bfloat16)
    return out


@helion.experimental.aot_kernel(backend="triton", static_shapes=True)
def nvfp4_gemv_fp4in_kernel(
    weight_bytes: torch.Tensor,  # (M, K_bytes) uint8 packed NVFP4 weight
    x_bytes: torch.Tensor,  # (K_bytes,) uint8 packed NVFP4 activation
    weight_scale_bytes: torch.Tensor,  # int8 view of SWIZZLE_32_4_4 E4M3 scales
    x_scale_bytes: torch.Tensor,  # int8 view of SWIZZLE_32_4_4 E4M3 x scales
    out: torch.Tensor,  # (M,) bf16 output
    alpha: float = 1.0,
) -> torch.Tensor:
    """W4A4 NVFP4 GEMV: coalesced FP4 loads, fp16 group dot, fp32 scale."""
    M, K_bytes = weight_bytes.shape
    K_groups = K_bytes // 8
    scale_cols = K_groups
    block_m = hl.register_block_size(1, 16)
    block_g = hl.register_block_size(K_groups)
    for tile_m in hl.tile(M, block_size=block_m):
        acc = hl.zeros([tile_m], dtype=torch.float32)
        for tile_g in hl.tile(K_groups, block_size=block_g):
            group_offsets = tile_m.index[:, None] * K_groups + tile_g.index[None, :]
            group_mask = tile_g.index < K_groups
            weight_mask = (tile_m.index[:, None] < M) & group_mask[None, :]
            w = hl.load_float4_e2m1fn_x16_to_float16(
                weight_bytes,
                group_offsets,
                extra_mask=weight_mask,
            )
            x = hl.load_float4_e2m1fn_x16_to_float16(
                x_bytes,
                tile_g.index,
                extra_mask=group_mask,
            )
            prod = hl.zeros([block_m, block_g], dtype=torch.float16)
            for i in hl.static_range(16):
                prod = prod + w[i] * x[i][None, :]
            g = tile_g.index
            wso = swizzled_scale_offsets(tile_m.index[:, None], g[None, :], scale_cols)
            xso = swizzled_scale_offsets(g * 0, g, scale_cols)
            scale = _e4m3_byte_to_f32(weight_scale_bytes[wso]) * _e4m3_byte_to_f32(
                x_scale_bytes[xso]
            )
            acc = acc + (prod.to(torch.float32) * scale).sum(-1)
        out[tile_m] = (acc * alpha).to(torch.bfloat16)
    return out


def _nvfp4_gemv_fp4in(
    weight_packed: torch.Tensor,  # [N, K // 2] packed NVFP4 (uint8 / fp4x2)
    x_packed: torch.Tensor,  # [K // 2] packed NVFP4 activation
    weight_scale: torch.Tensor,  # SWIZZLE_32_4_4 E4M3 weight scales
    x_scale: torch.Tensor,  # SWIZZLE_32_4_4 E4M3 activation scales
    alpha: float = 1.0,
) -> torch.Tensor:
    """Host wrapper for the coalesced-load W4A4 kernel."""
    weight_bytes = weight_packed.view(torch.uint8)
    n = weight_bytes.shape[0]
    out = torch.empty(n, dtype=torch.bfloat16, device=weight_bytes.device)
    return nvfp4_gemv_fp4in_kernel(
        weight_bytes,
        x_packed.view(torch.uint8),
        weight_scale.reshape(-1).view(torch.int8),
        x_scale.reshape(-1).view(torch.int8),
        out,
        alpha,
    )


def _nvfp4_gemv_bf16in(
    weight_packed: torch.Tensor,  # [N, K // 2] packed NVFP4 (uint8 / fp4x2)
    x_bf16: torch.Tensor,  # [K] BF16 activation
    weight_scale: torch.Tensor,  # SWIZZLE_32_4_4 E4M3 weight scales
    alpha: float = 1.0,
) -> torch.Tensor:
    """Host wrapper for the coalesced-load W4A16 kernel."""
    weight_bytes = weight_packed.view(torch.uint8)
    n, k_bytes = weight_bytes.shape
    g = k_bytes // 8
    out = torch.empty(n, dtype=torch.bfloat16, device=weight_bytes.device)
    return nvfp4_gemv_bf16in_kernel(
        weight_bytes,
        x_bf16.view(g, 16),
        weight_scale.reshape(-1).view(torch.int8),
        out,
        alpha,
    )


# Optional vLLM CUTLASS NVFP4 baseline: the production kernel this is benchmarked
# against. ``cutlass_scaled_fp4_mm`` is the NVFP4 analog of ``cutlass_scaled_mm``
# and has no dedicated GEMV, so decode is served by the M=1 GEMM. The pretuned
# test env has only torch + helion (guarded import); the nightly benchmark
# workflow builds vLLM's ``_C`` extension, so main() then compares against it.
try:
    from vllm import _custom_ops as _vllm_ops

    _HAS_VLLM = hasattr(torch.ops._C, "cutlass_scaled_fp4_mm")
    FLOAT4_E2M1_MAX = 6.0
    FLOAT8_E4M3_MAX = float(torch.finfo(torch.float8_e4m3fn).max)
except ImportError:
    _vllm_ops = None
    _HAS_VLLM = False


def _vllm_quant_weight(weight_bf16: torch.Tensor) -> tuple:
    """Quantize a logical BF16 weight to vLLM NVFP4 (packed bytes + swizzled sf)."""
    amax = weight_bf16.abs().max().to(torch.float32)
    global_scale = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / amax
    weight_fp4, weight_sf = _vllm_ops.scaled_fp4_quant(weight_bf16, global_scale)
    return weight_fp4, weight_sf, global_scale


def _make_vllm_fp4in_call(
    n: int, k: int, device: torch.device
) -> Callable[[], torch.Tensor]:
    """vLLM W4A4 decode baseline: activation pre-quantized, then the CUTLASS GEMM.

    Mirrors vLLM's ``benchmark_nvfp4_gemm`` (``no_a_quant`` path) with M=1 -- the
    decode GEMV that ``cutlass_scaled_fp4_mm`` serves (it has no GEMV kernel).
    """
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
    """vLLM W4A16 decode baseline: quantize the BF16 activation on the fly, GEMM.

    This is vLLM's real NVFP4 linear decode path (``CutlassNvFp4LinearKernel``):
    ``scaled_fp4_quant(x)`` then ``cutlass_scaled_fp4_mm`` -- there is no native
    W4A16 NVFP4 GEMM, so the BF16 activation is dynamically quantized to NVFP4.
    """
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

    True: these are decode (M=1) GEMVs invoked one row at a time, where per-call
    host launch overhead dominates -- exactly how vLLM issues them. CUDA graphs
    remove that overhead; the shared _bench loop clears the L2 cache before every
    replay (cold L2), matching how the weight is streamed cold from DRAM.
    """
    return True


# Decode (M=1) NVFP4 GEMV weight shapes (N=output features, K=reduction dim), the
# attention / MLP projections of common NVFP4-served models (Llama-3, Qwen). K is
# a multiple of 16 (NVFP4 block size) and N a multiple of 128 (CUTLASS alignment).
SHAPES = [  # (N, K)
    (4096, 4096),  # Llama-3-8B o_proj (square)
    (6144, 4096),  # Llama-3-8B qkv_proj (q4096 + kv1024*2)
    (28672, 4096),  # Llama-3-8B gate_up_proj (2 * 14336)
    (4096, 14336),  # Llama-3-8B down_proj
    (5120, 5120),  # 13B-class attention o_proj
    (15360, 5120),  # 13B-class qkv_proj (3 * 5120)
    (8192, 8192),  # 70B-class square projection (nvfp4_backend_comparison "o", bf16in)
    (10240, 8192),  # wide fused projection
    (8192, 28672),  # nvfp4_backend_comparison "down" (fp4in): K_bytes=14336
]

# The two GEMV regimes benchmarked per shape: NVFP4 activation (W4A4) and BF16
# activation (W4A16). Each is compared against its matching vLLM baseline.
_VARIANTS = ("fp4in", "bf16in")


def _make_fp4in_inputs(n: int, k: int) -> tuple:
    """Random helion W4A4 inputs (values are irrelevant to timing)."""
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
    """Random helion W4A16 inputs (values are irrelevant to timing)."""
    device = torch.device("cuda")
    k_bytes = k // 2
    weight = torch.randint(0, 256, (n, k_bytes), dtype=torch.uint8, device=device).view(
        torch.float4_e2m1fn_x2
    )
    x = torch.randn(k, dtype=torch.bfloat16, device=device)
    weight_scale = make_fp8_scales((n, k_bytes // 8), device)
    return weight, x, weight_scale


def main(verbose: bool = True) -> dict:
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from _bench import run_sweep

    device = torch.device("cuda")

    def make_calls(entry: tuple[str, int, int]) -> tuple:
        variant, n, k = entry
        if variant == "fp4in":
            weight, x, w_scale, x_scale = _make_fp4in_inputs(n, k)

            def helion_call() -> torch.Tensor:
                return _nvfp4_gemv_fp4in(weight, x, w_scale, x_scale)

            # Correctness against the eager dequant reference (checked once, not
            # timed -- the eager reference is orders of magnitude slower).
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
