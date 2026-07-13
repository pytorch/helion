"""Low-latency NVFP4 GEMV for decode (batch-size-1) inference on Blackwell.

Computes ``out = (weight @ x) * alpha`` for a weight matrix stored as packed
NVFP4 (E2M1) bytes with per-16-value E4M3 block scales in PyTorch's
SWIZZLE_32_4_4 layout. Two variants ship, matching the two decode regimes:

* :func:`_nvfp4_gemv_fp4in` -- NVFP4 weight * NVFP4 activation (W4A4).
* :func:`_nvfp4_gemv_bf16in` -- NVFP4 weight * BF16 activation (W4A16).

The kernels are the Triton NVFP4 GEMV from https://github.com/pytorch/helion/pull/2738
(``examples/nvfp4_gemv.py``); this module re-exposes them as pretuned
``aot_kernel``s and benchmarks them against the production vLLM CUTLASS NVFP4
GEMM (``ops.cutlass_scaled_fp4_mm`` -- the NVFP4 analog of ``cutlass_scaled_mm``,
which has no dedicated GEMV path, so decode is served by the M=1 GEMM).

The DSL bodies are copied from the PR (the source of truth for correctness,
tested in ``test/test_examples.py::test_nvfp4_gemv``); the shared pure-python
scale-swizzle and dequant-reference helpers are imported from the example so
this module stays a thin, benchmark-only wrapper.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from typing import cast

import torch

from helion._testing import import_path
import helion.experimental
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Callable

# The PR kernel is the source of truth; reuse its pure-python helpers (scale
# swizzle offsets, fp8-scale builder, dequant references) instead of copying.
_EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"
_ex = import_path(_EXAMPLES_DIR / "nvfp4_gemv.py")
swizzled_scale_offsets = _ex.swizzled_scale_offsets
make_fp8_scales = _ex.make_fp8_scales
reference_nvfp4_gemv_fp4in = _ex.reference_nvfp4_gemv_fp4in
reference_nvfp4_gemv_bf16in = _ex.reference_nvfp4_gemv_bf16in


@helion.experimental.aot_kernel(backend="triton", static_shapes=True)
def nvfp4_gemv_fp4in_kernel(
    weight_fp4x2: torch.Tensor,  # [N, K // 16, 8] packed NVFP4 weight bytes
    x_fp4x2: torch.Tensor,  # [K // 16, 8] packed NVFP4 activation bytes
    weight_scale: torch.Tensor,  # SWIZZLE_32_4_4 E4M3 weight block scales
    x_scale: torch.Tensor,  # SWIZZLE_32_4_4 E4M3 activation block scales
    out: torch.Tensor,  # [N] BF16 output
    alpha: float = 1.0,
) -> torch.Tensor:
    """W4A4 NVFP4 GEMV (copied from PR #2738 ``_nvfp4_gemv_fp4in_body``)."""
    M, K_groups, _ = weight_fp4x2.shape
    block_m = hl.register_block_size(1, 8)
    block_k = hl.register_block_size(16, K_groups)

    for tile_m in hl.tile(M, block_size=block_m):
        row = tile_m.begin
        acc = hl.zeros([], dtype=torch.float32)
        for tile_k in hl.tile(K_groups, block_size=block_k):
            contrib = hl.zeros([tile_k], dtype=torch.float32)
            for byte in hl.static_range(8):
                weight_lo, weight_hi = hl.float4_e2m1fn_x2_to_float32(
                    weight_fp4x2[row, tile_k, byte]
                )
                x_lo, x_hi = hl.float4_e2m1fn_x2_to_float32(x_fp4x2[tile_k, byte])
                contrib = contrib + weight_lo * x_lo + weight_hi * x_hi
            weight_scale_offsets = swizzled_scale_offsets(
                cast("int", row), tile_k.index, K_groups
            )
            x_scale_offsets = swizzled_scale_offsets(
                tile_k.index * 0, tile_k.index, K_groups
            )
            scale = hl.load(
                weight_scale, [weight_scale_offsets], extra_mask=tile_k.index < K_groups
            ).to(torch.float32)
            scale = scale * hl.load(
                x_scale, [x_scale_offsets], extra_mask=tile_k.index < K_groups
            ).to(torch.float32)
            acc = acc + (contrib * scale).sum()
        out[row] = (acc * alpha).to(torch.bfloat16)
    return out


@helion.experimental.aot_kernel(backend="triton", static_shapes=True)
def nvfp4_gemv_bf16in_kernel(
    weight_fp4x2: torch.Tensor,  # [N, K // 16, 8] packed NVFP4 weight bytes
    x_values: torch.Tensor,  # [K // 16, 16] BF16 activation
    weight_scale: torch.Tensor,  # SWIZZLE_32_4_4 E4M3 weight block scales
    out: torch.Tensor,  # [N] BF16 output
    alpha: float = 1.0,
) -> torch.Tensor:
    """W4A16 NVFP4 GEMV (copied from PR #2738 ``_nvfp4_gemv_bf16in_body``)."""
    M, K_groups, _ = weight_fp4x2.shape
    block_m = hl.register_block_size(1, 8)
    block_k = hl.register_block_size(16, K_groups)

    for tile_m in hl.tile(M, block_size=block_m):
        row = tile_m.begin
        acc = hl.zeros([], dtype=torch.float32)
        for tile_k in hl.tile(K_groups, block_size=block_k):
            contrib = hl.zeros([tile_k], dtype=torch.float32)
            for byte in hl.static_range(8):
                weight_lo, weight_hi = hl.float4_e2m1fn_x2_to_float32(
                    weight_fp4x2[row, tile_k, byte]
                )
                contrib = contrib + weight_lo * x_values[tile_k, byte * 2].to(
                    torch.float32
                )
                contrib = contrib + weight_hi * x_values[tile_k, byte * 2 + 1].to(
                    torch.float32
                )
            scale_offsets = swizzled_scale_offsets(
                cast("int", row), tile_k.index, K_groups
            )
            scale = hl.load(
                weight_scale, [scale_offsets], extra_mask=tile_k.index < K_groups
            ).to(torch.float32)
            acc = acc + (contrib * scale).sum()
        out[row] = (acc * alpha).to(torch.bfloat16)
    return out


def _nvfp4_gemv_fp4in(
    weight_packed: torch.Tensor,  # [N, K // 2] packed NVFP4 (uint8 / fp4x2)
    x_packed: torch.Tensor,  # [K // 2] packed NVFP4 activation
    weight_scale: torch.Tensor,  # SWIZZLE_32_4_4 E4M3 weight scales
    x_scale: torch.Tensor,  # SWIZZLE_32_4_4 E4M3 activation scales
    alpha: float = 1.0,
) -> torch.Tensor:
    """Host wrapper: reshape to grouped byte layout and run the W4A4 kernel."""
    weight_bytes = weight_packed.view(torch.uint8)
    n, k_bytes = weight_bytes.shape
    out = torch.empty(n, dtype=torch.bfloat16, device=weight_bytes.device)
    return nvfp4_gemv_fp4in_kernel(
        weight_packed.view(torch.float4_e2m1fn_x2).view(n, k_bytes // 8, 8),
        x_packed.view(torch.float4_e2m1fn_x2).view(k_bytes // 8, 8),
        weight_scale.reshape(-1),
        x_scale.reshape(-1),
        out,
        alpha,
    )


def _nvfp4_gemv_bf16in(
    weight_packed: torch.Tensor,  # [N, K // 2] packed NVFP4 (uint8 / fp4x2)
    x_bf16: torch.Tensor,  # [K] BF16 activation
    weight_scale: torch.Tensor,  # SWIZZLE_32_4_4 E4M3 weight scales
    alpha: float = 1.0,
) -> torch.Tensor:
    """Host wrapper: reshape to grouped byte layout and run the W4A16 kernel."""
    weight_bytes = weight_packed.view(torch.uint8)
    n, k_bytes = weight_bytes.shape
    out = torch.empty(n, dtype=torch.bfloat16, device=weight_bytes.device)
    return nvfp4_gemv_bf16in_kernel(
        weight_packed.view(torch.float4_e2m1fn_x2).view(n, k_bytes // 8, 8),
        x_bf16.view(k_bytes // 8, 16),
        weight_scale.reshape(-1),
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
    (8192, 8192),  # 70B-class square projection
    (10240, 8192),  # wide fused projection
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

            base_calls: list[tuple[str, Callable[[], torch.Tensor]]] = [
                (
                    "torch",
                    lambda: reference_nvfp4_gemv_fp4in(weight, x, w_scale, x_scale),
                )
            ]
            if _HAS_VLLM:
                base_calls.append(("cutlass", _make_vllm_fp4in_call(n, k, device)))
        else:
            weight, x, w_scale = _make_bf16in_inputs(n, k)

            def helion_call() -> torch.Tensor:
                return _nvfp4_gemv_bf16in(weight, x, w_scale)

            base_calls = [
                ("torch", lambda: reference_nvfp4_gemv_bf16in(weight, x, w_scale))
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
