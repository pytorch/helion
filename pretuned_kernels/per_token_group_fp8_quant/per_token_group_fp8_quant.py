"""Per-token-group FP8 (e4m3) dynamic quantization.

Quantizes a ``[num_tokens, hidden_size]`` input into float8_e4m3 by splitting
each row into contiguous groups of ``group_size`` elements, computing a per-group
amax-based scale, and dividing. Writes the quantized values into ``output_q`` and
the per-group scales into ``output_s`` (both mutated in place). Ported from
vLLM's Helion ``per_token_group_fp8_quant`` kernel (the checked-in heuristics are
converted from vLLM's per-hardware config JSON).
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

    _HAS_VLLM = hasattr(torch.ops._C, "per_token_group_fp8_quant")
except ImportError:
    _HAS_VLLM = False


@helion.experimental.aot_kernel(
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def per_token_group_fp8_quant(
    input: torch.Tensor,  # [num_tokens, hidden_size]  # noqa: A002
    output_q: torch.Tensor,  # [num_tokens, hidden_size]
    output_s: torch.Tensor,  # [num_tokens, groups_per_row]
    group_size: int,
    eps: float,
    fp8_min: float,
    fp8_max: float,
    scale_ue8m0: bool,
    # Unused dummy args
    # Kept for consistency with existing kernel interface
    dummy_is_scale_transposed: bool = False,
    dummy_is_tma_aligned: bool = False,
) -> None:
    # This code assumes batch_dim and num_tokens are flattened
    assert input.ndim == 2
    num_tokens, hidden_size = input.shape
    hl.specialize(hidden_size)
    hl.specialize(group_size)

    groups_per_row = output_s.shape[1]
    hl.specialize(groups_per_row)
    assert hidden_size % group_size == 0 and hidden_size // group_size == groups_per_row
    assert output_s.ndim == 2 and output_s.dtype == torch.float32

    input = input.view(num_tokens, -1, group_size)  # noqa: A001
    output_q = output_q.view(num_tokens, -1, group_size)
    for tile_m, tile_gn, tile_n in hl.tile(
        [num_tokens, groups_per_row, group_size], block_size=[1, None, group_size]
    ):
        x_blk = input[tile_m, tile_gn, tile_n]
        y_s_blk = torch.clamp(torch.amax(torch.abs(x_blk), dim=-1), min=eps)
        y_s_blk = y_s_blk / fp8_max

        if scale_ue8m0:
            y_s_blk = torch.exp2(torch.ceil(torch.log2(y_s_blk)))

        y_q_blk = torch.clamp(x_blk / y_s_blk[:, :, None], fp8_min, fp8_max).to(
            output_q.dtype
        )

        output_s[tile_m, tile_gn] = y_s_blk
        output_q[tile_m, tile_gn, tile_n] = y_q_blk


def _per_token_group_fp8_quant_torch(
    input: torch.Tensor,  # noqa: A002
    output_q: torch.Tensor,
    output_s: torch.Tensor,
    group_size: int,
    eps: float,
    fp8_min: float,
    fp8_max: float,
    scale_ue8m0: bool,
    dummy_is_scale_transposed: bool = False,
    dummy_is_tma_aligned: bool = False,
) -> None:
    """Torch-native reference: same math as the Helion kernel (mutates outputs)."""
    num_tokens, hidden_size = input.shape
    x = input.view(num_tokens, -1, group_size).to(torch.float32)
    y_s = torch.clamp(torch.amax(torch.abs(x), dim=-1), min=eps) / fp8_max
    if scale_ue8m0:
        y_s = torch.exp2(torch.ceil(torch.log2(y_s)))
    y_q = torch.clamp(x / y_s[:, :, None], fp8_min, fp8_max).to(output_q.dtype)
    output_q.view(num_tokens, -1, group_size).copy_(y_q)
    output_s.copy_(y_s)


def _per_token_group_fp8_quant_vllm(
    input: torch.Tensor,  # noqa: A002
    output_q: torch.Tensor,
    output_s: torch.Tensor,
    group_size: int,
    eps: float,
    fp8_min: float,
    fp8_max: float,
    scale_ue8m0: bool,
    dummy_is_scale_transposed: bool = False,
    dummy_is_tma_aligned: bool = False,
) -> None:
    """vLLM compiled baseline (torch.ops._C.per_token_group_fp8_quant)."""
    torch.ops._C.per_token_group_fp8_quant(
        input,
        output_q,
        output_s,
        group_size,
        eps,
        fp8_min,
        fp8_max,
        scale_ue8m0,
        dummy_is_scale_transposed,
        dummy_is_tma_aligned,
    )


def _baselines() -> list[tuple[str, object]]:
    """Baselines main() benchmarks against (torch always; vLLM when installed).

    ``torch_compile`` is ``torch.compile`` of the torch reference -- a
    speedup-comparison baseline only (not checked for accuracy).
    """
    out: list[tuple[str, object]] = [
        ("torch", _per_token_group_fp8_quant_torch),
        ("torch_compile", torch.compile(_per_token_group_fp8_quant_torch)),
    ]
    if _HAS_VLLM:
        out.append(("vllm", _per_token_group_fp8_quant_vllm))
    return out


def use_cudagraph() -> bool:
    """Whether main() benchmarks under CUDA graphs (read by pretuned_kernels/run.py)."""
    return True


def _bench_shapes() -> list[tuple[int, int]]:
    """Shapes main() benchmarks: (num_tokens, hidden_size) pairs."""
    hidden_sizes = [2048, 4096, 5120]
    num_tokens_list = [1, 4, 16, 64, 256, 1024, 2048, 8192]
    return [(t, h) for h in hidden_sizes for t in num_tokens_list]


def correctness_check() -> None:
    """Assert the Helion kernel matches the torch reference (used by the tests)."""
    torch.manual_seed(0)
    group_size = 128
    const = (group_size, 1e-10, -448.0, 448.0, False)
    for num_tokens, hidden_size in _bench_shapes():
        inp = torch.randn(num_tokens, hidden_size, device="cuda", dtype=torch.bfloat16)
        groups = hidden_size // group_size
        oq = torch.empty_like(inp, dtype=torch.float8_e4m3fn)
        os = torch.empty((num_tokens, groups), device="cuda", dtype=torch.float32)
        oq_ref = torch.empty_like(oq)
        os_ref = torch.empty_like(os)
        per_token_group_fp8_quant(inp, oq, os, *const)
        _per_token_group_fp8_quant_torch(inp, oq_ref, os_ref, *const)
        torch.testing.assert_close(os, os_ref, rtol=1e-2, atol=1e-6)
        torch.testing.assert_close(oq.float(), oq_ref.float(), rtol=0.2, atol=0.2)


def main(verbose: bool = True) -> dict:
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from _bench import run_sweep

    fp8_dtype = torch.float8_e4m3fn
    fp8_min, fp8_max = -448.0, 448.0
    eps = 1e-10
    group_size = 128
    scale_ue8m0 = False

    shapes = _bench_shapes()
    baselines = _baselines()

    def make_calls(shape: tuple) -> tuple:
        num_tokens, hidden_size = shape
        input = torch.randn(  # noqa: A001
            num_tokens, hidden_size, device="cuda", dtype=torch.bfloat16
        )
        groups_per_row = hidden_size // group_size
        output_q = torch.empty_like(input, dtype=fp8_dtype)
        output_s = torch.empty(
            (num_tokens, groups_per_row), device="cuda", dtype=torch.float32
        )
        output_q_ref = torch.empty_like(output_q)
        output_s_ref = torch.empty_like(output_s)

        args = (
            input,
            output_q,
            output_s,
            group_size,
            eps,
            fp8_min,
            fp8_max,
            scale_ue8m0,
        )
        ref_args = (
            input,
            output_q_ref,
            output_s_ref,
            group_size,
            eps,
            fp8_min,
            fp8_max,
            scale_ue8m0,
        )

        def helion_call() -> None:
            per_token_group_fp8_quant(*args)

        base_calls = [(n, (lambda fn=fn: fn(*ref_args))) for n, fn in baselines]
        return (
            helion_call,
            base_calls,
            f"{num_tokens:>7d}  {hidden_size:>6d}  {group_size:>6d}",
        )

    return run_sweep(
        shapes,
        make_calls,
        use_cudagraph=use_cudagraph(),
        verbose=verbose,
        shape_header=f"{'tokens':>7s}  {'hidden':>6s}  {'group':>6s}",
    )


if __name__ == "__main__":
    # Verify numerics across every benchmarked shape before timing.
    correctness_check()
    main()
