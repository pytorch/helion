"""Dynamic per-token scaled FP8 (e4m3) quantization.

For each token (row) of a ``[num_tokens, hidden_size]`` input, computes a
dynamic scale from the row's absolute max, then quantizes the row into
float8_e4m3. The scale is ``amax / fp8_max`` (clamped to a floor, optionally
clamped from above by ``scale_ub``), and the output is ``input / scale`` cast
to float8_e4m3. Ported from vLLM's Helion ``dynamic_per_token_scaled_fp8_quant``
kernel (the checked-in heuristic is converted from vLLM's per-hardware config
JSON).
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

    _HAS_VLLM = hasattr(torch.ops._C, "dynamic_per_token_scaled_fp8_quant")
except ImportError:
    _HAS_VLLM = False


@helion.experimental.aot_kernel(
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def dynamic_per_token_scaled_fp8_quant(
    result: torch.Tensor,  # [num_tokens, hidden_size]
    input: torch.Tensor,  # [num_tokens, hidden_size]  # noqa: A002
    scale: torch.Tensor,  # [num_tokens, 1]
    scale_ub: torch.Tensor | None = None,  # scalar tensor
) -> None:
    # This code assumes batch_dim and num_tokens are flattened
    assert input.ndim == 2
    num_tokens, hidden_size = input.shape
    hl.specialize(hidden_size)

    assert result.shape == input.shape
    assert scale.shape[0] == num_tokens
    assert scale.dtype == torch.float32
    assert input.stride()[-1] == 1
    assert result.stride()[-1] == 1

    # float8_e4m3fn range (matches vLLM get_fp8_min_max() on non-fnuz platforms).
    fp8_min, fp8_max = -448.0, 448.0
    min_scaling_factor = 1.0 / (fp8_max * 512.0)

    for tile_m in hl.tile(num_tokens, block_size=1):
        s_blk = hl.zeros([tile_m], dtype=torch.float32)
        for tile_n in hl.tile(hidden_size):
            x_blk = input[tile_m, tile_n].to(dtype=torch.float32)
            tmp_blk = torch.amax(torch.abs(x_blk), dim=-1)
            s_blk = torch.maximum(s_blk, tmp_blk)

        if scale_ub is not None:
            scale_ub_s = hl.load(scale_ub, [])
            s_blk = s_blk.clamp(max=scale_ub_s)
        s_blk = s_blk * (1.0 / fp8_max)
        s_blk = s_blk.clamp(min=min_scaling_factor)
        scale[tile_m, 0] = s_blk

        for tile_n in hl.tile(hidden_size):
            x_blk = input[tile_m, tile_n].to(torch.float32)
            y_blk = x_blk * (1.0 / s_blk[:, None])

            result[tile_m, tile_n] = y_blk.clamp(fp8_min, fp8_max).to(result.dtype)


def _dynamic_per_token_scaled_fp8_quant_torch(
    result: torch.Tensor,
    input: torch.Tensor,  # noqa: A002
    scale: torch.Tensor,
    scale_ub: torch.Tensor | None = None,
) -> None:
    """Torch-native reference: same math as the Helion kernel (in-place)."""
    fp8_min, fp8_max = (
        torch.finfo(torch.float8_e4m3fn).min,
        torch.finfo(torch.float8_e4m3fn).max,
    )
    min_scaling_factor = 1.0 / (fp8_max * 512.0)

    x = input.to(torch.float32)
    s = torch.amax(torch.abs(x), dim=-1)  # [num_tokens]
    if scale_ub is not None:
        s = s.clamp(max=scale_ub)
    s = s * (1.0 / fp8_max)
    s = s.clamp(min=min_scaling_factor)
    scale.copy_(s[:, None])

    y = x * (1.0 / s[:, None])
    result.copy_(y.clamp(fp8_min, fp8_max).to(result.dtype))


def _dynamic_per_token_scaled_fp8_quant_vllm(
    result: torch.Tensor,
    input: torch.Tensor,  # noqa: A002
    scale: torch.Tensor,
    scale_ub: torch.Tensor | None = None,
) -> None:
    """vLLM compiled baseline (torch.ops._C.dynamic_per_token_scaled_fp8_quant)."""
    torch.ops._C.dynamic_per_token_scaled_fp8_quant(result, input, scale, scale_ub)


def _baselines() -> list[tuple[str, object]]:
    """Baselines main() benchmarks against (torch always; vLLM when installed).

    ``torch_compile`` is ``torch.compile`` of the torch reference -- a
    speedup-comparison baseline only (not checked for accuracy).
    """
    out: list[tuple[str, object]] = [
        ("torch", _dynamic_per_token_scaled_fp8_quant_torch),
        (
            "torch_compile",
            torch.compile(_dynamic_per_token_scaled_fp8_quant_torch),
        ),
    ]
    if _HAS_VLLM:
        out.append(("vllm", _dynamic_per_token_scaled_fp8_quant_vllm))
    return out


def use_cudagraph() -> bool:
    """Whether main() benchmarks under CUDA graphs (read by pretuned_kernels/run.py)."""
    return True


def correctness_check() -> None:
    """Assert the Helion kernel matches the torch reference (used by the tests)."""
    torch.manual_seed(0)
    x = torch.randn(16, 4096, device="cuda", dtype=torch.bfloat16)
    result = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    scale = torch.empty((16, 1), device="cuda", dtype=torch.float32)
    result_ref = torch.empty_like(result)
    scale_ref = torch.empty_like(scale)
    dynamic_per_token_scaled_fp8_quant(result, x, scale)
    _dynamic_per_token_scaled_fp8_quant_torch(result_ref, x, scale_ref)
    torch.testing.assert_close(scale, scale_ref, rtol=1e-2, atol=1e-4)
    torch.testing.assert_close(result.float(), result_ref.float(), rtol=0.2, atol=0.2)


def main(verbose: bool = True) -> dict:
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from _bench import run_sweep

    # hidden_size / num_tokens drawn from the vLLM input generator.
    hidden_sizes = [2048, 4096, 5120]
    num_tokens_list = [1, 4, 16, 64, 256, 1024, 2048, 4096]
    shapes = [(t, h) for h in hidden_sizes for t in num_tokens_list]
    baselines = _baselines()

    def make_calls(shape: tuple) -> tuple:
        num_tokens, hidden_size = shape
        x = torch.randn(num_tokens, hidden_size, device="cuda", dtype=torch.bfloat16)
        result = torch.empty_like(x, dtype=torch.float8_e4m3fn)
        scale = torch.empty((num_tokens, 1), device="cuda", dtype=torch.float32)
        result_ref = torch.empty_like(result)
        scale_ref = torch.empty_like(scale)

        def helion_call() -> None:
            dynamic_per_token_scaled_fp8_quant(result, x, scale)

        base_calls = [
            (n, (lambda fn=fn: fn(result_ref, x, scale_ref))) for n, fn in baselines
        ]
        return helion_call, base_calls, f"{num_tokens:>7d}  {hidden_size:>6d}"

    return run_sweep(
        shapes,
        make_calls,
        use_cudagraph=use_cudagraph(),
        verbose=verbose,
        shape_header=f"{'tokens':>7s}  {'hidden':>6s}",
    )


if __name__ == "__main__":
    main()
