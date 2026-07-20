"""Low-latency NVFP4 GEMV for decode (batch-size-1) inference on Blackwell --
Helion CuTe (tcgen05) backend.

The CuTe counterpart of ``pretuned_kernels/nvfp4_gemv`` (the Triton variant):
same two decode regimes, same NVFP4 weight layout (packed E2M1 bytes with
per-16-value E4M3 block scales in PyTorch's SWIZZLE_32_4_4 layout), but backed by
the hand-written ``@cute.kernel`` inline-PTX fast paths from
https://github.com/pytorch/helion/pull/2738 (``examples/nvfp4_gemv.py``):

* :func:`_nvfp4_gemv_fp4in` -- NVFP4 weight * NVFP4 activation (W4A4).
* :func:`_nvfp4_gemv_bf16in` -- NVFP4 weight * BF16 activation (W4A16).

Unlike the Triton variant, these are hand-PTX kernels launched directly (fixed
block/grid via ``default_cute_launcher``), so there are no tunable configs -- the
sm100 marker heuristic exists only so ``pretuned_kernels/run.py`` gates the
kernel to B200. The example only compiles its CuTe fast paths when imported under
``HELION_BACKEND=cute``, so this module force-imports a private copy under that
backend (restoring the env afterwards, so a same-process Triton kernel run is
unaffected).

Benchmarked against the production vLLM CUTLASS NVFP4 GEMM
(``ops.cutlass_scaled_fp4_mm`` -- the NVFP4 analog of ``cutlass_scaled_mm``,
which has no dedicated GEMV, so decode is served by the M=1 GEMM) and
``torch.compile`` of the NVFP4 dequant reference. The eager dequant reference is
used only for a one-shot correctness check per shape (it is orders of magnitude
slower than the kernel, so it is not a timed baseline).
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Callable

_EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"


def _load_cute_example() -> object:
    """Import a private copy of examples/nvfp4_gemv.py under HELION_BACKEND=cute.

    The example gates compilation of its hand-PTX CuTe fast paths on the backend
    selected *at import time*, so it must be imported fresh under the cute
    backend (a distinct module name from the Triton variant's import). The env is
    restored afterwards so a same-process Triton kernel run still sees its own
    backend.
    """
    prev = os.environ.get("HELION_BACKEND")
    os.environ["HELION_BACKEND"] = "cute"
    try:
        module_name = "_pretuned_nvfp4_gemv_cute_example"
        spec = importlib.util.spec_from_file_location(
            module_name, _EXAMPLES_DIR / "nvfp4_gemv.py"
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        if prev is None:
            os.environ.pop("HELION_BACKEND", None)
        else:
            os.environ["HELION_BACKEND"] = prev


_ex = _load_cute_example()
make_fp8_scales = _ex.make_fp8_scales
reference_nvfp4_gemv_fp4in = _ex.reference_nvfp4_gemv_fp4in
reference_nvfp4_gemv_bf16in = _ex.reference_nvfp4_gemv_bf16in

# torch.compile of the NVFP4 dequant references -- a speedup-comparison baseline
# only (correctness is checked against the eager reference).
compiled_reference_nvfp4_gemv_fp4in = torch.compile(reference_nvfp4_gemv_fp4in)
compiled_reference_nvfp4_gemv_bf16in = torch.compile(reference_nvfp4_gemv_bf16in)


def _nvfp4_gemv_fp4in(
    weight_packed: torch.Tensor,
    x_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    x_scale: torch.Tensor,
    alpha: float = 1.0,
) -> torch.Tensor:
    """W4A4 NVFP4 GEMV via the CuTe hand-PTX fast path."""
    return _ex.nvfp4_gemv_fp4in(
        weight_packed, x_packed, weight_scale, x_scale, alpha, backend="cute"
    )


def _nvfp4_gemv_bf16in(
    weight_packed: torch.Tensor,
    x_bf16: torch.Tensor,
    weight_scale: torch.Tensor,
    alpha: float = 1.0,
) -> torch.Tensor:
    """W4A16 NVFP4 GEMV via the CuTe hand-PTX fast path."""
    return _ex.nvfp4_gemv_bf16in(
        weight_packed, x_bf16, weight_scale, alpha, backend="cute"
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


# The CuTe fast path requires k_bytes % 2048 == 0 (see _can_use_fast_cute_path in
# the example), so every K here is a multiple of 4096 elements. Decode (M=1) NVFP4
# GEMV weight shapes (N=output features, K=reduction dim) for common projections.
SHAPES = [  # (N, K)
    (4096, 4096),  # Llama-3-8B o_proj (square)
    (6144, 4096),  # Llama-3-8B qkv_proj (q4096 + kv1024*2)
    (28672, 4096),  # Llama-3-8B gate_up_proj (2 * 14336)
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
