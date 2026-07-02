"""SiLU-and-mul fused with FP8 (e4m3) dynamic quantization.

Computes ``out = (silu(x[..., :d]) * x[..., d:]) / scale`` cast to float8_e4m3,
where the input's last dim is ``2 * d``. Ported from vLLM's Helion
``silu_mul_fp8`` kernel (the checked-in heuristic is converted from vLLM's
per-hardware config JSON).
"""

from __future__ import annotations

import torch

import helion
import helion.experimental
import helion.language as hl

# Optional vLLM baseline: the production kernel this is benchmarked against. The
# pretuned test env has only torch + helion (guarded import); the nightly
# benchmark workflow installs vLLM, so main() then compares against the real op.
try:
    import vllm  # noqa: F401  (registers torch.ops._C.*)

    _HAS_VLLM = hasattr(torch.ops._C, "silu_and_mul_quant")
except ImportError:
    _HAS_VLLM = False


@helion.experimental.aot_kernel(
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def silu_mul_fp8(input: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:  # noqa: A002
    original_shape = input.shape
    two_d = hl.specialize(original_shape[-1])
    d = two_d // 2
    output_shape = (*original_shape[:-1], d)

    input_2d = input.view(-1, original_shape[-1])
    m = input_2d.shape[0]

    out = torch.empty((m, d), device=input.device, dtype=torch.float8_e4m3fn)

    input_part_a = input_2d[:, :d]
    input_part_b = input_2d[:, d:]

    assert scale.numel() == 1, "Scale must be a scalar Tensor"

    for tile_m, tile_n in hl.tile([m, d]):
        a_vals = input_part_a[tile_m, tile_n]
        silu_result = torch.nn.functional.silu(a_vals)
        b_vals = input_part_b[tile_m, tile_n]
        result = silu_result * b_vals
        result_f32 = result.to(torch.float32)
        scale_val = hl.load(scale, [0])
        inv_scale = 1.0 / scale_val
        result_scaled = result_f32 * inv_scale
        out[tile_m, tile_n] = result_scaled.to(out.dtype)

    return out.view(output_shape)


def _silu_mul_fp8_torch(input: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:  # noqa: A002
    """Torch-native reference: same math as the Helion kernel."""
    d = input.shape[-1] // 2
    a = input[..., :d]
    b = input[..., d:]
    result = (torch.nn.functional.silu(a) * b).to(torch.float32)
    return (result / scale).to(torch.float8_e4m3fn)


def _silu_mul_fp8_vllm(input: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:  # noqa: A002
    """vLLM compiled baseline (torch.ops._C.silu_and_mul_quant)."""
    d = input.shape[-1] // 2
    out = torch.empty(
        (*input.shape[:-1], d), dtype=torch.float8_e4m3fn, device=input.device
    )
    torch.ops._C.silu_and_mul_quant(out, input, scale)
    return out


def _baselines() -> list[tuple[str, object]]:
    """Baselines main() benchmarks against (torch always; vLLM when installed).

    ``torch_compile`` is ``torch.compile`` of the torch reference -- a
    speedup-comparison baseline only (not checked for accuracy).
    """
    out: list[tuple[str, object]] = [
        ("torch", _silu_mul_fp8_torch),
        ("torch_compile", torch.compile(_silu_mul_fp8_torch)),
    ]
    if _HAS_VLLM:
        out.append(("vllm", _silu_mul_fp8_vllm))
    return out


def use_cudagraph() -> bool:
    """Whether main() benchmarks under CUDA graphs (read by pretuned_kernels/run.py)."""
    return True


def correctness_check() -> None:
    """Assert the Helion kernel matches the torch reference (used by the tests)."""
    torch.manual_seed(0)
    x = torch.randn(16, 2 * 2048, device="cuda", dtype=torch.bfloat16)
    scale = torch.tensor([1.3], device="cuda", dtype=torch.float32)
    got = silu_mul_fp8(x, scale).float()
    ref = _silu_mul_fp8_torch(x, scale).float()
    torch.testing.assert_close(got, ref, rtol=0.2, atol=0.2)


def main(verbose: bool = True) -> dict:
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from _bench import run_sweep

    intermediate_sizes = [2048, 2880, 4096, 8192, 11008, 14336]
    num_tokens_list = [1, 8, 32, 64, 128, 256]
    shapes = [(t, i) for i in intermediate_sizes for t in num_tokens_list]
    baselines = _baselines()

    def make_calls(shape: tuple[int, int]) -> tuple:
        num_tokens, intermediate = shape
        x = torch.randn(
            num_tokens, 2 * intermediate, device="cuda", dtype=torch.bfloat16
        )
        scale = torch.tensor([1.0], device="cuda", dtype=torch.float32)

        def helion_call() -> torch.Tensor:
            return silu_mul_fp8(x, scale)

        base_calls = [(n, (lambda fn=fn: fn(x, scale))) for n, fn in baselines]
        return helion_call, base_calls, f"{num_tokens:>7d}  {intermediate:>6d}"

    return run_sweep(
        shapes,
        make_calls,
        use_cudagraph=use_cudagraph(),
        verbose=verbose,
        shape_header=f"{'tokens':>7s}  {'inter':>6s}",
    )


if __name__ == "__main__":
    main()
