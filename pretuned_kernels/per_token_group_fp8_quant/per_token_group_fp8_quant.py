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
import triton
import triton.language as tl

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


@triton.jit
def _per_token_group_quant_fp8(
    # Pointers to inputs and output
    y_ptr: tl.tensor,
    y_q_ptr: tl.tensor,
    y_s_ptr: tl.tensor,
    group_size: int,
    # Num columns of y
    y_num_columns: int,
    y_row_stride: int,
    # Avoid to divide zero
    eps: float,
    # Information for float8
    fp8_min: tl.constexpr,
    fp8_max: tl.constexpr,
    use_ue8m0: tl.constexpr,
    # Meta-parameters
    BLOCK: tl.constexpr,
) -> None:
    """A Triton-accelerated function to perform per-token-group
    quantization on a tensor.
    This function converts the tensor values into float8 values.
    """
    groups_per_row = y_num_columns // group_size

    # Map the program id to the row of X and Y it should compute.
    g_id = tl.program_id(0)
    row = g_id // groups_per_row
    row_g_id = g_id % groups_per_row

    # Ensure offset calculations use int64 to prevent overflow
    y_ptr_offset = (row.to(tl.int64) * y_row_stride) + (
        row_g_id.to(tl.int64) * group_size
    )
    y_ptr += y_ptr_offset

    y_q_ptr_offset = g_id.to(tl.int64) * group_size
    y_q_ptr += y_q_ptr_offset
    y_s_ptr += g_id

    cols = tl.arange(0, BLOCK)  # N <= BLOCK
    mask = cols < group_size

    y = tl.load(y_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    # Quant
    # Use multiply-by-reciprocal instead of division to match PyTorch's
    # tensor/scalar division precision (GPU fast-division for constexpr
    # divisors can introduce 1-ULP error that flips FP8 quantization at
    # representable-value boundaries).
    _absmax = tl.maximum(tl.max(tl.abs(y)), eps)
    scale_raw = _absmax * (1.0 / fp8_max)
    y_s = tl.math.exp2(tl.ceil(tl.log2(scale_raw))) if use_ue8m0 else scale_raw
    y_q = tl.clamp(y / y_s, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)

    tl.store(y_q_ptr + cols, y_q, mask=mask)
    tl.store(y_s_ptr, y_s)


def _per_token_group_fp8_quant_triton(
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
    """vLLM's Triton per-token-group FP8 quant kernel (mutates outputs)."""
    num_tokens, hidden_size = input.shape
    num_groups = num_tokens * (hidden_size // group_size)
    block = triton.next_power_of_2(group_size)
    _per_token_group_quant_fp8[(num_groups,)](
        input,
        output_q,
        output_s,
        group_size,
        hidden_size,
        input.stride(0),
        eps,
        fp8_min=fp8_min,
        fp8_max=fp8_max,
        use_ue8m0=scale_ue8m0,
        BLOCK=block,
    )


def _baselines() -> list[tuple[str, object]]:
    """Baselines main() benchmarks against (torch always; vLLM when installed).

    ``torch_compile`` is ``torch.compile`` of the torch reference -- a
    speedup-comparison baseline only (not checked for accuracy).
    """
    out: list[tuple[str, object]] = [
        ("torch", _per_token_group_fp8_quant_torch),
        ("torch_compile", torch.compile(_per_token_group_fp8_quant_torch)),
        ("triton", _per_token_group_fp8_quant_triton),
    ]
    if _HAS_VLLM:
        out.append(("vllm", _per_token_group_fp8_quant_vllm))
    return out


def use_cudagraph() -> bool:
    """Whether main() benchmarks under CUDA graphs (read by pretuned_kernels/run.py)."""
    return True


def correctness_check() -> None:
    """Assert the Helion kernel matches the torch reference (used by the tests)."""
    torch.manual_seed(0)
    num_tokens, hidden_size, group_size = 16, 4096, 128
    inp = torch.randn(num_tokens, hidden_size, device="cuda", dtype=torch.bfloat16)
    groups = hidden_size // group_size
    oq = torch.empty_like(inp, dtype=torch.float8_e4m3fn)
    os = torch.empty((num_tokens, groups), device="cuda", dtype=torch.float32)
    oq_ref = torch.empty_like(oq)
    os_ref = torch.empty_like(os)
    const = (group_size, 1e-10, -448.0, 448.0, False)
    per_token_group_fp8_quant(inp, oq, os, *const)
    _per_token_group_fp8_quant_torch(inp, oq_ref, os_ref, *const)
    torch.testing.assert_close(os, os_ref, rtol=1e-2, atol=1e-6)
    torch.testing.assert_close(oq.float(), oq_ref.float(), rtol=0.2, atol=0.2)


def main(
    verbose: bool = True,
    cudagraph: bool | None = None,
    limit: int | None = None,
    walltime: bool = False,
    serial: bool = False,
) -> dict:
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from _bench import run_sweep

    fp8_dtype = torch.float8_e4m3fn
    fp8_min, fp8_max = -448.0, 448.0
    eps = 1e-10
    group_size = 128
    scale_ue8m0 = False

    hidden_sizes = [2048, 4096, 5120]
    num_tokens_list = [1, 4, 16, 64, 256, 1024, 2048, 8192]
    shapes = [(t, h) for h in hidden_sizes for t in num_tokens_list]
    if limit is not None:
        shapes = shapes[:limit]
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
        use_cudagraph=use_cudagraph() if cudagraph is None else cudagraph,
        verbose=verbose,
        shape_header=f"{'tokens':>7s}  {'hidden':>6s}  {'group':>6s}",
        walltime=walltime,
        serial=serial,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-cudagraph",
        action="store_true",
        help="benchmark with plain do_bench instead of CUDA graphs",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="only run the first N shapes of the sweep",
    )
    parser.add_argument(
        "--walltime",
        action="store_true",
        help="time full wall-clock per call (host dispatch + GPU) instead of "
        "GPU events; exposes host launch overhead that do_bench hides",
    )
    parser.add_argument(
        "--serial",
        action="store_true",
        help="time host dispatch + GPU per call with a sync after every call, "
        "so CPU and GPU time are summed (not overlapped as --walltime allows); "
        "still clears L2 between runs. Takes precedence over --walltime",
    )
    cli_args = parser.parse_args()
    main(
        cudagraph=not cli_args.no_cudagraph,
        limit=cli_args.limit,
        walltime=cli_args.walltime,
        serial=cli_args.serial,
    )
