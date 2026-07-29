"""Low-latency NVFP4 GEMV for decode (batch-size-1) inference on Blackwell --
Helion CuTe (tcgen05) backend.

The CuTe counterpart of ``pretuned_kernels/nvfp4_gemv`` (the Triton variant):
same two decode regimes, same NVFP4 weight layout (packed E2M1 bytes with
per-16-value E4M3 block scales in PyTorch's SWIZZLE_32_4_4 layout), but backed by
actual Helion DSL kernels compiled with ``backend="cute"``:

* :func:`_nvfp4_gemv_fp4in` -- NVFP4 weight * NVFP4 activation (W4A4).
* :func:`_nvfp4_gemv_bf16in` -- NVFP4 weight * BF16 activation (W4A16).

The kernels use the portable FP32-decode bodies from ``examples/nvfp4_gemv.py``
and checked-in B200 CuTe configs. They go through Helion's normal compilation,
configuration, caching, and launch path; there is no direct ``@cute.kernel`` or
``default_cute_launcher`` shim.

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


@helion.aot_kernel(backend="cute", static_shapes=True)
def nvfp4_gemv_bf16in_kernel(
    weight_fp4x2: torch.Tensor,
    x_values: torch.Tensor,
    weight_scale: torch.Tensor,
    out: torch.Tensor,
    alpha: float = 1.0,
) -> torch.Tensor:
    """Helion CuTe W4A16 GEMV with FP32 accumulation."""
    M, K_groups, _ = weight_fp4x2.shape
    block_m = hl.register_block_size(1, 1)
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
                weight_scale,
                [scale_offsets],
                extra_mask=tile_k.index < K_groups,
            ).to(torch.float32)
            acc = acc + (contrib * scale).sum()
        out[row] = (acc * alpha).to(torch.bfloat16)
    return out


@helion.aot_kernel(backend="cute", static_shapes=True)
def nvfp4_gemv_fp4in_kernel(
    weight_fp4x2: torch.Tensor,
    x_fp4x2: torch.Tensor,
    weight_scale: torch.Tensor,
    x_scale: torch.Tensor,
    out: torch.Tensor,
    alpha: float = 1.0,
) -> torch.Tensor:
    """Helion CuTe W4A4 GEMV with FP32 accumulation."""
    M, K_groups, _ = weight_fp4x2.shape
    block_m = hl.register_block_size(1, 1)
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
                weight_scale,
                [weight_scale_offsets],
                extra_mask=tile_k.index < K_groups,
            ).to(torch.float32)
            scale = scale * hl.load(
                x_scale,
                [x_scale_offsets],
                extra_mask=tile_k.index < K_groups,
            ).to(torch.float32)
            acc = acc + (contrib * scale).sum()
        out[row] = (acc * alpha).to(torch.bfloat16)
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
    return nvfp4_gemv_fp4in_kernel(
        weight_fp4x2.view(weight_bytes.shape[0], groups, 8),
        x_fp4x2.view(groups, 8),
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
    return nvfp4_gemv_bf16in_kernel(
        weight_fp4x2.view(weight_bytes.shape[0], groups, 8),
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
