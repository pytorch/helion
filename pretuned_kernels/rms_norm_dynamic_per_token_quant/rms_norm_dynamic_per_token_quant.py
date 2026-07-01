"""RMSNorm fused with dynamic per-token FP8 (e4m3) quantization.

For each token row, computes RMSNorm with a learned ``weight`` (optionally
adding a ``residual`` first), then dynamically quantizes the normalized row to
float8_e4m3 with a per-token scale derived from the row's max-abs value. Ported
from vLLM's Helion ``rms_norm_dynamic_per_token_quant`` kernel (the checked-in
heuristic is converted from vLLM's per-hardware config JSON).
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

    _HAS_VLLM = hasattr(torch.ops._C, "rms_norm_dynamic_per_token_quant")
except ImportError:
    _HAS_VLLM = False

# fp8 e4m3 quant traits (matches vLLM get_fp8_min_max / min_scaling_factor on
# non-fnuz NVIDIA platforms).
_FP8_DTYPE = torch.float8_e4m3fn
_FP8_MIN = float(torch.finfo(_FP8_DTYPE).min)
_FP8_MAX = float(torch.finfo(_FP8_DTYPE).max)
_MIN_SCALING_FACTOR = 1.0 / (_FP8_MAX * 512.0)

# int8 quant traits (mirrors vLLM get_int8_min_max / get_int8_min_scaling_factor).
_INT8_MIN = int(torch.iinfo(torch.int8).min)
_INT8_MAX = int(torch.iinfo(torch.int8).max)
_INT8_MIN_SCALING_FACTOR = 1.0 / (_INT8_MAX * 512.0)


@helion.experimental.aot_kernel(
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def rms_norm_dynamic_per_token_quant(
    result: torch.Tensor,  # [num_tokens, hidden_size]
    input: torch.Tensor,  # [num_tokens, hidden_size]  # noqa: A002
    weight: torch.Tensor,  # [hidden_size]
    scale: torch.Tensor,  # [num_tokens, 1]
    epsilon: float,
    scale_ub: torch.Tensor | None = None,  # []
    residual: torch.Tensor | None = None,  # [num_tokens, hidden_size]
) -> None:
    # This code assumes batch_dim and num_tokens are flattened
    assert input.ndim == 2
    num_tokens, hidden_size = input.shape
    hidden_size = hl.specialize(hidden_size)

    fp8_dtype = _FP8_DTYPE
    assert result.dtype in [fp8_dtype, torch.int8]
    assert result.is_contiguous() and input.is_contiguous()

    if scale_ub is not None:
        assert result.dtype == fp8_dtype
        assert scale_ub.dtype == torch.float32

    assert input.dtype == weight.dtype
    assert scale.shape[0] == num_tokens
    assert scale.dtype == torch.float32

    if residual is not None:
        assert residual.dtype == input.dtype

    quant_dtype = result.dtype
    qtype_traits_min: int | float
    qtype_traits_max: int | float
    if quant_dtype == torch.int8:
        qtype_traits_min, qtype_traits_max = _INT8_MIN, _INT8_MAX
        min_scaling_factor = _INT8_MIN_SCALING_FACTOR
    else:
        qtype_traits_min, qtype_traits_max = _FP8_MIN, _FP8_MAX
        min_scaling_factor = _MIN_SCALING_FACTOR

    # NOTE: vLLM's body wraps this in float(...); inside Helion's traced wrapper
    # that builtin call triggers a data-dependent guard on the epsilon symfloat,
    # so use the (already-float) constant directly.
    qtype_max = qtype_traits_max

    for tile_m in hl.tile(num_tokens, block_size=1):
        rms = hl.zeros([tile_m], dtype=torch.float32)
        for tile_n in hl.tile(hidden_size):
            x_blk = input[tile_m, tile_n].to(torch.float32)
            if residual is not None:
                x_blk = x_blk + residual[tile_m, tile_n]
            rms = rms + x_blk.pow(2).sum(dim=-1)

        rms = torch.rsqrt(rms * (1.0 / hidden_size) + epsilon)
        s_blk = hl.zeros([tile_m], dtype=torch.float32)

        for tile_n in hl.tile(hidden_size):
            x_blk = input[tile_m, tile_n].to(torch.float32)
            if residual is not None:
                x_blk = x_blk + residual[tile_m, tile_n]
            x_blk = (x_blk * rms[:, None]).to(input.dtype) * weight[None, tile_n]
            tmp_blk = torch.amax(torch.abs(x_blk), dim=-1).to(torch.float32)
            s_blk = torch.maximum(s_blk, tmp_blk)

        if scale_ub is not None:
            scale_ub_s = hl.load(scale_ub, [])
            s_blk = s_blk.clamp(max=scale_ub_s)
        s_blk = s_blk * (1.0 / qtype_max)
        s_blk = s_blk.clamp(min=min_scaling_factor)
        scale[tile_m, 0] = s_blk

        for tile_n in hl.tile(hidden_size):
            x_blk = input[tile_m, tile_n].to(torch.float32)
            if residual is not None:
                x_blk = x_blk + residual[tile_m, tile_n]
                residual[tile_m, tile_n] = x_blk.to(residual.dtype)
            x_blk = (x_blk * rms[:, None]).to(input.dtype) * weight[None, tile_n]
            if quant_dtype == torch.int8:
                s_inv_blk = 1.0 / s_blk[:, None]
                y_blk = x_blk * s_inv_blk
                y_blk = y_blk.round()
            else:
                y_blk = x_blk / s_blk[:, None]

            result[tile_m, tile_n] = y_blk.clamp(qtype_traits_min, qtype_traits_max).to(
                result.dtype
            )


def _rms_norm_dynamic_per_token_quant_torch(
    result: torch.Tensor,
    input: torch.Tensor,  # noqa: A002
    weight: torch.Tensor,
    scale: torch.Tensor,
    epsilon: float,
    scale_ub: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
) -> None:
    """Torch-native reference: RMSNorm(weight) then per-token dynamic fp8 quant.

    Mutates ``result``, ``scale`` (and ``residual`` if given) in place, matching
    the Helion kernel's outputs.
    """
    num_tokens, hidden_size = input.shape
    x = input.to(torch.float32)
    if residual is not None:
        x = x + residual.to(torch.float32)
        residual.copy_(x.to(residual.dtype))

    rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + epsilon)
    # Match the kernel: cast normalized value to input dtype before weight mul.
    normed = (x * rms).to(input.dtype) * weight[None, :]

    s = torch.amax(torch.abs(normed.to(torch.float32)), dim=-1, keepdim=True)
    if scale_ub is not None:
        s = s.clamp(max=scale_ub)
    s = s * (1.0 / _FP8_MAX)
    s = s.clamp(min=_MIN_SCALING_FACTOR)
    scale.copy_(s)

    y = normed.to(torch.float32) / s
    result.copy_(y.clamp(_FP8_MIN, _FP8_MAX).to(result.dtype))


def _rms_norm_dynamic_per_token_quant_vllm(
    result: torch.Tensor,
    input: torch.Tensor,  # noqa: A002
    weight: torch.Tensor,
    scale: torch.Tensor,
    epsilon: float,
    scale_ub: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
) -> None:
    """vLLM compiled baseline (torch.ops._C.rms_norm_dynamic_per_token_quant)."""
    torch.ops._C.rms_norm_dynamic_per_token_quant(
        result, input, weight, scale, epsilon, scale_ub, residual
    )


def _baselines() -> list[tuple[str, object]]:
    """Baselines main() benchmarks against (torch always; vLLM when installed).

    ``torch_compile`` is ``torch.compile`` of the torch reference -- a
    speedup-comparison baseline only (not checked for accuracy).
    """
    out: list[tuple[str, object]] = [
        ("torch", _rms_norm_dynamic_per_token_quant_torch),
        (
            "torch_compile",
            torch.compile(_rms_norm_dynamic_per_token_quant_torch),
        ),
    ]
    if _HAS_VLLM:
        out.append(("vllm", _rms_norm_dynamic_per_token_quant_vllm))
    return out


def use_cudagraph() -> bool:
    """Whether main() benchmarks under CUDA graphs (read by pretuned_kernels/run.py)."""
    return True


def correctness_check() -> None:
    """Assert the Helion kernel matches the torch reference (used by the tests)."""
    torch.manual_seed(0)
    num_tokens, hidden_size = 16, 4096
    x_in = torch.randn(num_tokens, hidden_size, device="cuda", dtype=torch.bfloat16)
    weight = torch.normal(
        mean=1.0, std=1.0, size=(hidden_size,), dtype=x_in.dtype, device="cuda"
    )
    result = torch.empty_like(x_in, dtype=torch.float8_e4m3fn)
    scale = torch.empty((num_tokens, 1), device="cuda", dtype=torch.float32)
    result_ref = torch.empty_like(result)
    scale_ref = torch.empty_like(scale)
    rms_norm_dynamic_per_token_quant(result, x_in, weight, scale, 1e-6)
    _rms_norm_dynamic_per_token_quant_torch(result_ref, x_in, weight, scale_ref, 1e-6)
    torch.testing.assert_close(scale, scale_ref, rtol=2e-2, atol=1e-4)
    torch.testing.assert_close(result.float(), result_ref.float(), rtol=0.2, atol=0.2)


def main(verbose: bool = True) -> dict:
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from _bench import run_sweep

    # Representative sweep from the vLLM input_generator / pick_config grid.
    hidden_sizes = [2048, 4096, 5120]
    num_tokens_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    shapes = [(h, t) for h in hidden_sizes for t in num_tokens_list]
    epsilon = 1e-6
    baselines = _baselines()

    def make_calls(shape: tuple[int, int]) -> tuple:
        hidden_size, num_tokens = shape
        x_in = torch.randn(num_tokens, hidden_size, device="cuda", dtype=torch.bfloat16)
        weight = torch.normal(
            mean=1.0, std=1.0, size=(hidden_size,), dtype=x_in.dtype, device="cuda"
        )
        result = torch.empty_like(x_in, dtype=_FP8_DTYPE)
        scale = torch.empty((num_tokens, 1), device="cuda", dtype=torch.float32)
        result_ref = torch.empty_like(result)
        scale_ref = torch.empty_like(scale)

        def helion_call() -> None:
            rms_norm_dynamic_per_token_quant(result, x_in, weight, scale, epsilon)

        base_calls = [
            (
                n,
                (lambda fn=fn: fn(result_ref, x_in, weight, scale_ref, epsilon)),
            )
            for n, fn in baselines
        ]
        return helion_call, base_calls, f"{hidden_size:>7d}  {num_tokens:>7d}"

    return run_sweep(
        shapes,
        make_calls,
        use_cudagraph=use_cudagraph(),
        verbose=verbose,
        shape_header=f"{'hidden':>7s}  {'tokens':>7s}",
    )


if __name__ == "__main__":
    main()
