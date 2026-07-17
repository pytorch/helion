"""SiLU-and-mul fused with per-block FP8 (e4m3) dynamic quantization.

Computes ``y = silu(input[..., :d]) * input[..., d:]`` and quantizes it to
float8_e4m3 per group of ``group_size`` columns: for each group, ``scale =
amax(|y|) / fp8_max`` (optionally clamped by ``scale_ub``), ``out = (y /
scale)`` cast to float8. ``out`` is ``[num_tokens, intermediate_size]`` and
``scales`` is ``[num_tokens, intermediate_size // group_size]``. Both are
mutated in place. Ported from vLLM's Helion ``silu_and_mul_per_block_quant``
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

    _HAS_VLLM = hasattr(torch.ops._C, "silu_and_mul_per_block_quant")
except ImportError:
    _HAS_VLLM = False


@helion.experimental.aot_kernel(
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def silu_and_mul_per_block_quant(
    out: torch.Tensor,  # [num_tokens, intermediate_size]
    input: torch.Tensor,  # [num_tokens, 2 * intermediate_size]  # noqa: A002
    scales: torch.Tensor,  # [num_tokens, groups_per_row]
    group_size: int,
    scale_ub: torch.Tensor | None = None,  # scalar tensor
    is_scale_transposed: bool = False,  # dummy
) -> None:
    # This code assumes batch_dim and num_tokens are flattened
    assert input.ndim == 2
    num_tokens, two_intermediate_size = input.shape
    hl.specialize(two_intermediate_size)

    assert two_intermediate_size % 2 == 0
    intermediate_size = two_intermediate_size // 2

    assert out.shape[0] == num_tokens
    assert out.shape[1] == intermediate_size
    fp8_dtype = torch.float8_e4m3fn
    assert out.dtype in [fp8_dtype, torch.int8]

    if scale_ub is not None:
        assert out.dtype == fp8_dtype
        assert scale_ub.dtype == torch.float32

    assert scales.ndim == 2 and scales.dtype == torch.float32

    assert scales.shape[0] == num_tokens
    groups_per_row = scales.shape[1]
    hl.specialize(groups_per_row)
    assert (
        intermediate_size % group_size == 0
        and intermediate_size // group_size == groups_per_row
    )

    assert group_size in [64, 128]
    hl.specialize(group_size)

    assert input.stride()[-1] == 1
    assert out.stride()[-1] == 1

    quant_dtype = out.dtype
    qtype_traits_min: int | float
    qtype_traits_max: int | float
    # Quant range constants inlined as literals (vLLM gets these from
    # get_int8_min_max() / get_int8_min_scaling_factor() / get_fp8_min_max())
    # so the kernel body only references torch / hl. int8: iinfo(int8).min/.max,
    # finfo(float32).eps; fp8: finfo(float8_e4m3fn).min/.max == -448.0/448.0.
    if quant_dtype == torch.int8:
        qtype_traits_min, qtype_traits_max = -128, 127
        min_scaling_factor = 1.1920928955078125e-07
    else:
        qtype_traits_min, qtype_traits_max = -448.0, 448.0
        min_scaling_factor = 1.0 / (qtype_traits_max * 512.0)

    qtype_max = float(qtype_traits_max)

    input = input.view(num_tokens, -1, group_size)  # noqa: A001
    out = out.view(num_tokens, -1, group_size)

    for tile_m, tile_gn, tile_n in hl.tile(
        [num_tokens, groups_per_row, group_size], block_size=[1, None, group_size]
    ):
        x_a_blk = input[tile_m, tile_gn, tile_n].to(torch.float32)
        x_b_blk = hl.load(
            input,
            [tile_m, tile_gn.index + groups_per_row, tile_n],
            extra_mask=(tile_gn.index + groups_per_row < 2 * groups_per_row)[
                None, :, None
            ],
        ).to(torch.float32)
        x_blk = x_a_blk * torch.sigmoid(x_a_blk) * x_b_blk
        s_blk = torch.amax(torch.abs(x_blk), dim=-1).to(torch.float32)

        if scale_ub is not None:
            scale_ub_s = hl.load(scale_ub, [])
            s_blk = s_blk.clamp(max=scale_ub_s)
        s_blk = s_blk * (1.0 / qtype_max)
        s_blk = s_blk.clamp(min=min_scaling_factor)

        scales[tile_m, tile_gn] = s_blk
        if quant_dtype == torch.int8:
            y_blk = (x_blk * (1.0 / s_blk[:, :, None])).round()
        else:
            y_blk = x_blk / s_blk[:, :, None]

        out[tile_m, tile_gn, tile_n] = y_blk.clamp(
            qtype_traits_min, qtype_traits_max
        ).to(out.dtype)


def _silu_and_mul_per_block_quant_torch(
    out: torch.Tensor,
    input: torch.Tensor,  # noqa: A002
    scales: torch.Tensor,
    group_size: int,
    scale_ub: torch.Tensor | None = None,
    is_scale_transposed: bool = False,
) -> None:
    """Torch-native reference: same math as the Helion kernel (writes in place)."""
    num_tokens, two_intermediate_size = input.shape
    intermediate_size = two_intermediate_size // 2
    groups_per_row = intermediate_size // group_size

    qtype_traits_max = torch.finfo(torch.float8_e4m3fn).max
    min_scaling_factor = 1.0 / (qtype_traits_max * 512.0)

    a = input[:, :intermediate_size].to(torch.float32)
    b = input[:, intermediate_size:].to(torch.float32)
    y = a * torch.sigmoid(a) * b  # [num_tokens, intermediate_size]

    y_g = y.view(num_tokens, groups_per_row, group_size)
    s = torch.amax(torch.abs(y_g), dim=-1)  # [num_tokens, groups_per_row]
    if scale_ub is not None:
        s = s.clamp(max=scale_ub.item())
    s = (s * (1.0 / qtype_traits_max)).clamp(min=min_scaling_factor)

    q = (y_g / s[:, :, None]).clamp(-qtype_traits_max, qtype_traits_max)
    scales.copy_(s)
    out.copy_(q.view(num_tokens, intermediate_size).to(out.dtype))


def _silu_and_mul_per_block_quant_vllm(
    out: torch.Tensor,
    input: torch.Tensor,  # noqa: A002
    scales: torch.Tensor,
    group_size: int,
    scale_ub: torch.Tensor | None = None,
    is_scale_transposed: bool = False,
) -> None:
    """vLLM compiled baseline (torch.ops._C.silu_and_mul_per_block_quant)."""
    torch.ops._C.silu_and_mul_per_block_quant(
        out, input, scales, group_size, scale_ub, is_scale_transposed
    )


def _baselines() -> list[tuple[str, object]]:
    """Baselines main() benchmarks against (torch always; vLLM when installed).

    ``torch_compile`` is ``torch.compile`` of the torch reference -- a
    speedup-comparison baseline only (not checked for accuracy).
    """
    out: list[tuple[str, object]] = [
        ("torch", _silu_and_mul_per_block_quant_torch),
        ("torch_compile", torch.compile(_silu_and_mul_per_block_quant_torch)),
    ]
    if _HAS_VLLM:
        out.append(("vllm", _silu_and_mul_per_block_quant_vllm))
    return out


def use_cudagraph() -> bool:
    """Whether main() benchmarks under CUDA graphs (read by pretuned_kernels/run.py)."""
    return True


def _bench_shapes() -> list[tuple[int, int]]:
    """The (num_tokens, intermediate_size) shapes main() benchmarks."""
    intermediate_sizes = [6144, 12288, 25600]
    num_tokens_list = [1, 8, 32, 64, 128, 256, 1024, 4096]
    return [(t, i) for i in intermediate_sizes for t in num_tokens_list]


def correctness_check() -> None:
    """Assert the Helion kernel matches the torch reference (used by the tests)."""
    group_size = 128
    torch.manual_seed(0)
    for num_tokens, intermediate in _bench_shapes():
        x = torch.randn(
            num_tokens, 2 * intermediate, device="cuda", dtype=torch.bfloat16
        )
        out = torch.empty(
            num_tokens, intermediate, device="cuda", dtype=torch.float8_e4m3fn
        )
        scales = torch.empty(
            num_tokens, intermediate // group_size, device="cuda", dtype=torch.float32
        )
        out_ref = torch.empty_like(out)
        scales_ref = torch.empty_like(scales)
        silu_and_mul_per_block_quant(out, x, scales, group_size)
        _silu_and_mul_per_block_quant_torch(out_ref, x, scales_ref, group_size)
        torch.testing.assert_close(scales, scales_ref, rtol=1e-2, atol=1e-4)
        torch.testing.assert_close(out.float(), out_ref.float(), rtol=0.2, atol=0.2)


def main(verbose: bool = True) -> dict:
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from _bench import run_sweep

    group_size = 128
    shapes = _bench_shapes()
    baselines = _baselines()

    def make_calls(shape: tuple[int, int]) -> tuple:
        num_tokens, intermediate = shape
        x = torch.randn(
            num_tokens, 2 * intermediate, device="cuda", dtype=torch.bfloat16
        )
        out = torch.empty(
            num_tokens, intermediate, device="cuda", dtype=torch.float8_e4m3fn
        )
        scales = torch.empty(
            num_tokens, intermediate // group_size, device="cuda", dtype=torch.float32
        )
        # Separate ref tensors so the baselines don't clobber helion's in-place
        # outputs (the kernel mutates out and scales).
        out_ref = torch.empty_like(out)
        scales_ref = torch.empty_like(scales)

        def helion_call() -> None:
            silu_and_mul_per_block_quant(out, x, scales, group_size)

        base_calls = [
            (n, (lambda fn=fn: fn(out_ref, x, scales_ref, group_size)))
            for n, fn in baselines
        ]
        return helion_call, base_calls, f"{num_tokens:>7d}  {intermediate:>6d}"

    return run_sweep(
        shapes,
        make_calls,
        use_cudagraph=use_cudagraph(),
        verbose=verbose,
        shape_header=f"{'tokens':>7s}  {'inter':>6s}",
    )


if __name__ == "__main__":
    # Verify numerics across every benchmarked shape before timing.
    correctness_check()
    main()
