"""RMSNorm fused with per-block (group) FP8 dynamic quantization.

Applies RMSNorm (optionally fused with a residual add) to ``input`` and then
quantizes the normalized result into ``result`` (float8_e4m3fn) using a separate
scale per ``group_size``-wide block along the hidden dimension. The per-block
scales are written to ``scale``; when ``residual`` is given, the residual buffer
is updated in place with ``input + residual``.

Ported from vLLM's Helion ``rms_norm_per_block_quant`` kernel (the checked-in
heuristic is converted from vLLM's per-hardware config JSON). The kernel mutates
``result``, ``scale`` and ``residual`` in place and returns ``None``.
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl

# float8_e4m3fn numeric traits, used by the torch-native reference below. The
# kernel body inlines these as literals directly (module globals can't be
# referenced from a traced kernel that is imported without a sys.modules entry).
_FP8_DTYPE = torch.float8_e4m3fn
_FP8_MIN = float(torch.finfo(_FP8_DTYPE).min)
_FP8_MAX = float(torch.finfo(_FP8_DTYPE).max)

# Optional vLLM baseline: the production kernel this is benchmarked against. The
# pretuned test env has only torch + helion (guarded import); the nightly
# benchmark workflow installs vLLM, so main() then compares against the real op.
try:
    import vllm  # noqa: F401  (registers torch.ops._C.*)

    _HAS_VLLM = hasattr(torch.ops._C, "rms_norm_per_block_quant")
except ImportError:
    _HAS_VLLM = False


@helion.aot_kernel(
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def rms_norm_per_block_quant(
    result: torch.Tensor,  # [num_tokens, hidden_size]
    input: torch.Tensor,  # [num_tokens, hidden_size]  # noqa: A002
    weight: torch.Tensor,  # [hidden_size]
    scale: torch.Tensor,  # [num_tokens, groups_per_row]
    epsilon: float,
    scale_ub: torch.Tensor | None,  # []
    residual: torch.Tensor | None,  # [num_tokens, hidden_size]
    group_size: int,
    is_scale_transposed: bool,  # dummy
) -> None:
    # This code assumes batch_dim and num_tokens are flattened
    assert input.ndim == 2
    num_tokens, hidden_size = input.shape
    hl.specialize(hidden_size)
    hl.specialize(group_size)

    groups_per_row = scale.shape[1]
    hl.specialize(groups_per_row)
    assert hidden_size % group_size == 0 and hidden_size // group_size == groups_per_row
    assert scale.shape[0] == num_tokens
    assert scale.dtype == torch.float32
    if scale.stride(1) > 1:
        assert is_scale_transposed

    fp8_dtype = torch.float8_e4m3fn
    assert result.dtype in [fp8_dtype, torch.int8]
    assert result.is_contiguous() and input.is_contiguous()

    if scale_ub is not None:
        assert result.dtype == fp8_dtype
        assert scale_ub.dtype == torch.float32

    assert input.dtype == weight.dtype

    if residual is not None:
        assert residual.dtype == input.dtype

    assert group_size in [64, 128]

    quant_dtype = result.dtype
    qtype_traits_min: int | float
    qtype_traits_max: int | float
    if quant_dtype == torch.int8:
        # torch.iinfo(torch.int8) min/max, torch.finfo(torch.float32).eps.
        qtype_traits_min, qtype_traits_max = -128, 127
        min_scaling_factor = 1.1920928955078125e-07
    else:
        # torch.finfo(torch.float8_e4m3fn) min/max.
        qtype_traits_min, qtype_traits_max = -448.0, 448.0
        min_scaling_factor = 1.0 / (qtype_traits_max * 512.0)

    qtype_max = qtype_traits_max

    for tile_m in hl.tile(num_tokens, block_size=1):
        rms = hl.zeros([tile_m], dtype=torch.float32)
        for tile_n in hl.tile(hidden_size):
            x_blk = input[tile_m, tile_n].to(torch.float32)
            if residual is not None:
                x_blk = x_blk + residual[tile_m, tile_n]
            rms = rms + x_blk.pow(2).sum(dim=-1)

        rms = torch.rsqrt(rms * (1.0 / hidden_size) + epsilon)

        m_idx = tile_m.begin + hl.arange(tile_m.block_size)
        m_blk = m_idx[:, None, None]
        for tile_gn, tile_n in hl.tile(
            [groups_per_row, group_size], block_size=[None, group_size]
        ):
            gn_idx = tile_gn.index
            n_offset = tile_n.index
            n_idx = gn_idx[:, None] * group_size + n_offset[None, :]
            n_blk = n_idx[None, :, :]
            mask = (gn_idx < groups_per_row)[None, :, None]

            x_blk = hl.load(input, [m_blk, n_blk], extra_mask=mask).to(
                dtype=torch.float32
            )
            if residual is not None:
                r_blk = hl.load(residual, [m_blk, n_blk], extra_mask=mask)
                x_blk = x_blk + r_blk

            w_blk = hl.load(weight, [n_blk], extra_mask=mask)
            x_norm_blk = (x_blk * rms[:, None, None]).to(input.dtype) * w_blk
            s_blk = torch.amax(torch.abs(x_norm_blk), dim=-1).to(torch.float32)

            if scale_ub is not None:
                scale_ub_s = hl.load(scale_ub, [])
                s_blk = s_blk.clamp(max=scale_ub_s)

            s_blk = s_blk * (1.0 / qtype_max)
            s_blk = s_blk.clamp(min=min_scaling_factor)

            scale[tile_m, tile_gn] = s_blk

            if quant_dtype == torch.int8:
                y_blk = (x_norm_blk * (1.0 / s_blk[:, :, None])).round()
            else:
                y_blk = x_norm_blk / s_blk[:, :, None]

            y_blk = y_blk.clamp(qtype_traits_min, qtype_traits_max).to(result.dtype)
            hl.store(result, [m_blk, n_blk], y_blk, extra_mask=mask)

            if residual is not None:
                hl.store(
                    residual, [m_blk, n_blk], x_blk.to(residual.dtype), extra_mask=mask
                )


def _rms_norm_per_block_quant_torch(
    result: torch.Tensor,
    input: torch.Tensor,  # noqa: A002
    weight: torch.Tensor,
    scale: torch.Tensor,
    epsilon: float,
    scale_ub: torch.Tensor | None,
    residual: torch.Tensor | None,
    group_size: int,
    is_scale_transposed: bool,
) -> None:
    """Torch-native reference: RMSNorm (optionally + residual) then per-group fp8 quant.

    Computes ``y = rmsnorm(input + residual) * weight``, then quantizes each
    ``group_size``-wide block to float8_e4m3fn with a per-block amax scale; writes
    ``result`` (fp8), ``scale`` (fp32), and updates ``residual`` in place.
    """
    num_tokens, hidden_size = input.shape
    groups_per_row = hidden_size // group_size

    qtype_max = _FP8_MAX
    min_scaling_factor = 1.0 / (_FP8_MAX * 512.0)

    x = input.to(torch.float32)
    if residual is not None:
        x = x + residual.to(torch.float32)
        residual.copy_(x.to(residual.dtype))

    rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + epsilon)
    x_norm = (x * rms).to(input.dtype) * weight
    x_norm = x_norm.to(torch.float32)

    # [num_tokens, groups_per_row, group_size]
    x_grp = x_norm.view(num_tokens, groups_per_row, group_size)
    s = x_grp.abs().amax(dim=-1)
    if scale_ub is not None:
        s = s.clamp(max=scale_ub)
    s = (s * (1.0 / qtype_max)).clamp(min=min_scaling_factor)
    scale.copy_(s)

    y = x_grp / s[:, :, None]
    y = y.clamp(_FP8_MIN, _FP8_MAX).view(num_tokens, hidden_size)
    result.copy_(y.to(result.dtype))


def _rms_norm_per_block_quant_vllm(
    result: torch.Tensor,
    input: torch.Tensor,  # noqa: A002
    weight: torch.Tensor,
    scale: torch.Tensor,
    epsilon: float,
    scale_ub: torch.Tensor | None,
    residual: torch.Tensor | None,
    group_size: int,
    is_scale_transposed: bool,
) -> None:
    """vLLM compiled baseline (torch.ops._C.rms_norm_per_block_quant)."""
    torch.ops._C.rms_norm_per_block_quant(
        result,
        input,
        weight,
        scale,
        epsilon,
        scale_ub,
        residual,
        group_size,
        is_scale_transposed,
    )


def _baselines() -> list[tuple[str, object]]:
    """Baselines main() benchmarks against (torch always; vLLM when installed).

    ``torch_compile`` is ``torch.compile`` of the torch reference -- a
    speedup-comparison baseline only (not checked for accuracy).
    """
    out: list[tuple[str, object]] = [
        ("torch", _rms_norm_per_block_quant_torch),
        ("torch_compile", torch.compile(_rms_norm_per_block_quant_torch)),
    ]
    if _HAS_VLLM:
        out.append(("vllm", _rms_norm_per_block_quant_vllm))
    return out


def use_cudagraph() -> bool:
    """Whether main() benchmarks under CUDA graphs (read by pretuned_kernels/run.py)."""
    return True


def _make_inputs(
    num_tokens: int, hidden_size: int, group_size: int
) -> tuple[torch.Tensor, ...]:
    in_dtype = torch.bfloat16
    out_dtype = torch.float8_e4m3fn
    inp = torch.randn(num_tokens, hidden_size, device="cuda", dtype=in_dtype)
    result = torch.empty(inp.shape, device=inp.device, dtype=out_dtype)
    scale = torch.empty(
        (num_tokens, hidden_size // group_size),
        device=inp.device,
        dtype=torch.float32,
    )
    scale_ub = torch.mean(inp).to(torch.float32)
    residual = torch.randn_like(inp)
    weight = torch.normal(
        mean=1.0, std=1.0, size=(hidden_size,), dtype=in_dtype, device=inp.device
    )
    epsilon = 1e-6
    return (result, inp, weight, scale, epsilon, scale_ub, residual, group_size, False)


def _bench_shapes() -> list[tuple[int, int, int]]:
    """The (hidden_size, group_size, num_tokens) shapes main() benchmarks."""
    hidden_size_list = [2048, 4096, 5120]
    group_size_list = [128]
    num_tokens_list = [1, 8, 32, 64, 128, 256, 1024, 4096]
    return [
        (h, g, t)
        for h in hidden_size_list
        for g in group_size_list
        for t in num_tokens_list
    ]


def correctness_check() -> None:
    """Assert the Helion kernel matches the torch reference (used by the tests)."""
    torch.manual_seed(0)
    for hidden_size, group_size, num_tokens in _bench_shapes():
        args = _make_inputs(num_tokens, hidden_size, group_size)
        (
            result,
            inp,
            weight,
            scale,
            epsilon,
            scale_ub,
            residual,
            group_size,
            is_transposed,
        ) = args
        result_ref = torch.empty_like(result)
        scale_ref = torch.empty_like(scale)
        residual_ref = residual.clone()
        rms_norm_per_block_quant(*args)
        _rms_norm_per_block_quant_torch(
            result_ref,
            inp,
            weight,
            scale_ref,
            epsilon,
            scale_ub,
            residual_ref,
            group_size,
            is_transposed,
        )
        torch.testing.assert_close(scale, scale_ref, rtol=2e-2, atol=1e-4)
        torch.testing.assert_close(
            result.float(), result_ref.float(), rtol=0.2, atol=0.2
        )
        torch.testing.assert_close(
            residual.float(), residual_ref.float(), rtol=2e-2, atol=2e-2
        )


def main(verbose: bool = True) -> dict:
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from _bench import run_sweep

    shapes = _bench_shapes()
    baselines = _baselines()

    def make_calls(shape: tuple[int, int, int]) -> tuple:
        hidden_size, group_size, num_tokens = shape
        helion_args = _make_inputs(num_tokens, hidden_size, group_size)
        result, inp, weight, scale, epsilon, scale_ub, residual, gs, is_transposed = (
            helion_args
        )
        # Parallel ref-arg set: separate result/scale/residual outputs sharing the
        # same inputs, so the baselines don't clobber the helion output buffers
        # (residual is mutated in place).
        ref_args = (
            torch.empty_like(result),
            inp,
            weight,
            torch.empty_like(scale),
            epsilon,
            scale_ub,
            residual.clone() if residual is not None else None,
            gs,
            is_transposed,
        )

        def helion_call() -> None:
            rms_norm_per_block_quant(*helion_args)

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
