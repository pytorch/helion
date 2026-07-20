"""Per-token-group FP8 (e4m3) dynamic quantization with UE8M0 packed scales.

Like ``per_token_group_fp8_quant`` but writes the per-group scale *exponents*
packed 4-per-int32 into a TMA-aligned ``output_s_packed`` tensor (the DeepGEMM
UE8M0 layout), so Helion covers the DeepGEMM path. Ported from vLLM's Helion
``per_token_group_fp8_quant_packed`` kernel
(https://github.com/xiaohongchen1991/vllm/tree/per_token_group_fp8_quant_packed).
"""

from __future__ import annotations

import torch

import helion
import helion.experimental
import helion.language as hl

# Optional vLLM baseline: the production kernel this is benchmarked against.
try:
    import vllm  # noqa: F401  (registers torch.ops._C.*)

    _HAS_VLLM = hasattr(torch.ops._C, "per_token_group_fp8_quant_packed")
except ImportError:
    _HAS_VLLM = False


@helion.experimental.aot_kernel(
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
    static_shapes=True,
)
def per_token_group_fp8_quant_packed(
    input: torch.Tensor,  # [num_tokens, hidden_size]  # noqa: A002
    output_q: torch.Tensor,  # [output_q_num_tokens, hidden_size]
    output_s_packed: torch.Tensor,  # [num_tokens, (groups_per_row+3)//4], int32
    group_size: int,
    eps: float,
    min_8bit: float,
    max_8bit: float,
) -> None:
    assert input.ndim == 2
    num_tokens, hidden_size = input.shape
    hl.specialize(hidden_size)
    hl.specialize(group_size)

    assert group_size == 128
    assert input.dtype in (torch.float16, torch.bfloat16)
    assert input.is_contiguous()
    assert output_q.is_contiguous()
    assert hidden_size % group_size == 0
    groups_per_row = hidden_size // group_size

    k_num_packed = output_s_packed.shape[-1]
    tma_aligned_num_tokens = output_s_packed.stride(1)
    padded_groups_per_row = k_num_packed * 4
    hl.specialize(k_num_packed)
    hl.specialize(padded_groups_per_row)

    assert output_s_packed.shape == (num_tokens, (groups_per_row + 3) // 4)
    assert output_s_packed.stride() == (1, ((num_tokens + 3) // 4) * 4)
    assert output_s_packed.dtype == torch.int32

    output_q_num_tokens = output_q.shape[0]
    assert output_q.shape == (output_q_num_tokens, hidden_size)
    assert output_q_num_tokens >= num_tokens

    # zero out padding part
    if tma_aligned_num_tokens > num_tokens and k_num_packed > 1:
        torch.as_strided(
            output_s_packed,
            size=(k_num_packed - 1, tma_aligned_num_tokens - num_tokens),
            stride=(tma_aligned_num_tokens, 1),
            storage_offset=num_tokens,
        ).zero_()

    if output_q_num_tokens > num_tokens:
        output_q[num_tokens:output_q_num_tokens, :].zero_()

    input = input.view(num_tokens, -1, group_size)  # noqa: A001
    output_q = output_q.view(output_q_num_tokens, -1, group_size)

    for tile_m, tile_gn, tile_n in hl.tile(
        [num_tokens, k_num_packed, group_size], block_size=[1, None, group_size]
    ):
        packed_s_blk = hl.zeros([tile_m, tile_gn], dtype=torch.int32)
        # Hoist all 4 group loads before any dependent compute so their global
        # memory latency overlaps (4 loads in flight) instead of being exposed
        # one-at-a-time. Small-token shapes are memory-latency bound, so this
        # cuts the dominant long-scoreboard stall.
        tile_g0 = tile_gn.index * 4 + 0
        tile_g1 = tile_gn.index * 4 + 1
        tile_g2 = tile_gn.index * 4 + 2
        tile_g3 = tile_gn.index * 4 + 3
        tile_gs = [tile_g0, tile_g1, tile_g2, tile_g3]
        mask_g0 = tile_g0 < groups_per_row
        mask_g1 = tile_g1 < groups_per_row
        mask_g2 = tile_g2 < groups_per_row
        mask_g3 = tile_g3 < groups_per_row
        masks_g = [mask_g0, mask_g1, mask_g2, mask_g3]
        x_blk0 = hl.load(
            input, [tile_m, tile_g0, tile_n], extra_mask=mask_g0[None, :, None]
        ).to(torch.float32)
        x_blk1 = hl.load(
            input, [tile_m, tile_g1, tile_n], extra_mask=mask_g1[None, :, None]
        ).to(torch.float32)
        x_blk2 = hl.load(
            input, [tile_m, tile_g2, tile_n], extra_mask=mask_g2[None, :, None]
        ).to(torch.float32)
        x_blk3 = hl.load(
            input, [tile_m, tile_g3, tile_n], extra_mask=mask_g3[None, :, None]
        ).to(torch.float32)
        x_blks = [x_blk0, x_blk1, x_blk2, x_blk3]

        for i in hl.static_range(4):
            mask_g = masks_g[i]
            x_blk = x_blks[i]

            y_s_blk = torch.clamp(torch.amax(torch.abs(x_blk), dim=-1), min=eps)
            y_s_blk = y_s_blk / max_8bit
            y_s_blk = torch.clamp(y_s_blk, min=1.0e-10)

            exp_unbiased_blk = torch.ceil(torch.log2(y_s_blk))
            exp_byte_blk = exp_unbiased_blk.to(torch.int32) + 127
            exp_byte_blk = torch.where(mask_g[None, :], exp_byte_blk, 0)
            packed_s_blk = packed_s_blk | exp_byte_blk << (i * 8)

            y_s_blk = torch.exp2(exp_unbiased_blk.to(torch.float32))
            y_q_blk = torch.clamp(x_blk / y_s_blk[:, :, None], min_8bit, max_8bit).to(
                output_q.dtype
            )

            hl.store(
                output_q,
                [tile_m, tile_gs[i], tile_n],
                y_q_blk,
                extra_mask=mask_g[None, :, None],
            )

        output_s_packed[tile_m, tile_gn] = packed_s_blk


def _packed_torch(
    input: torch.Tensor,  # noqa: A002
    output_q: torch.Tensor,
    output_s_packed: torch.Tensor,
    group_size: int,
    eps: float,
    min_8bit: float,
    max_8bit: float,
) -> None:
    """Torch-native reference: same math + UE8M0 packing as the Helion kernel."""
    num_tokens, hidden_size = input.shape
    groups_per_row = hidden_size // group_size
    k_num_packed = (groups_per_row + 3) // 4
    x = input.view(num_tokens, groups_per_row, group_size).to(torch.float32)
    y_s = torch.clamp(torch.amax(torch.abs(x), dim=-1), min=eps) / max_8bit
    y_s = torch.clamp(y_s, min=1.0e-10)
    exp_unbiased = torch.ceil(torch.log2(y_s))
    exp_byte = exp_unbiased.to(torch.int32) + 127
    scale = torch.exp2(exp_unbiased)
    y_q = torch.clamp(x / scale[:, :, None], min_8bit, max_8bit).to(output_q.dtype)
    output_q.view(num_tokens, groups_per_row, group_size).copy_(y_q)
    padded = torch.zeros(
        num_tokens, k_num_packed * 4, dtype=torch.int32, device=input.device
    )
    padded[:, :groups_per_row] = exp_byte
    pv = padded.view(num_tokens, k_num_packed, 4)
    packed = (
        pv[:, :, 0] | (pv[:, :, 1] << 8) | (pv[:, :, 2] << 16) | (pv[:, :, 3] << 24)
    )
    output_s_packed.copy_(packed)


def _packed_vllm(
    input: torch.Tensor,  # noqa: A002
    output_q: torch.Tensor,
    output_s_packed: torch.Tensor,
    group_size: int,
    eps: float,
    min_8bit: float,
    max_8bit: float,
) -> None:
    """vLLM compiled baseline (torch.ops._C.per_token_group_fp8_quant_packed)."""
    torch.ops._C.per_token_group_fp8_quant_packed(
        input, output_q, output_s_packed, group_size, eps, min_8bit, max_8bit
    )


def _baselines() -> list[tuple[str, object]]:
    out: list[tuple[str, object]] = [
        ("torch", _packed_torch),
        ("torch_compile", torch.compile(_packed_torch)),
    ]
    if _HAS_VLLM:
        out.append(("vllm", _packed_vllm))
    return out


def use_cudagraph() -> bool:
    return True


def _make_outputs(
    num_tokens: int, hidden_size: int, group_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    fp8_dtype = torch.float8_e4m3fn
    output_q = torch.empty(num_tokens, hidden_size, device="cuda", dtype=fp8_dtype)
    groups_per_row = hidden_size // group_size
    k_num_packed = (groups_per_row + 3) // 4
    tma_aligned = ((num_tokens + 3) // 4) * 4
    output_s_packed = torch.empty_strided(
        (num_tokens, k_num_packed), (1, tma_aligned), device="cuda", dtype=torch.int32
    )
    return output_q, output_s_packed


def _bench_shapes() -> list[tuple[int, int]]:
    """Shapes main() benchmarks: (num_tokens, hidden_size) pairs."""
    hidden_sizes = [2048, 4096, 5120]
    num_tokens_list = [4, 16, 64, 256, 1024, 2048, 8192]
    return [(t, h) for h in hidden_sizes for t in num_tokens_list]


def correctness_check() -> None:
    """Assert the Helion kernel matches the torch reference (used by the tests)."""
    torch.manual_seed(0)
    group_size = 128
    const = (group_size, 1e-10, -448.0, 448.0)
    for num_tokens, hidden_size in _bench_shapes():
        inp = (
            torch.randn(num_tokens, hidden_size, device="cuda", dtype=torch.bfloat16)
            * 8
        )
        oq, osp = _make_outputs(num_tokens, hidden_size, group_size)
        oq_ref, osp_ref = _make_outputs(num_tokens, hidden_size, group_size)
        per_token_group_fp8_quant_packed(inp, oq, osp, *const)
        _packed_torch(inp, oq_ref, osp_ref, *const)
        torch.testing.assert_close(osp, osp_ref)
        torch.testing.assert_close(oq.float(), oq_ref.float(), rtol=0.2, atol=0.2)


def main(
    verbose: bool = True,
    cudagraph: bool | None = None,
    limit: int | None = None,
) -> dict:
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from _bench import run_sweep

    eps = 1e-10
    fp8_min, fp8_max = -448.0, 448.0
    group_size = 128

    shapes = _bench_shapes()
    if limit is not None:
        shapes = shapes[:limit]
    baselines = _baselines()

    def make_calls(shape: tuple) -> tuple:
        num_tokens, hidden_size = shape
        input = (  # noqa: A001
            torch.randn(num_tokens, hidden_size, device="cuda", dtype=torch.bfloat16)
            * 8
        )
        oq, osp = _make_outputs(num_tokens, hidden_size, group_size)
        oq_ref, osp_ref = _make_outputs(num_tokens, hidden_size, group_size)
        args = (input, oq, osp, group_size, eps, fp8_min, fp8_max)
        ref_args = (input, oq_ref, osp_ref, group_size, eps, fp8_min, fp8_max)

        def helion_call() -> None:
            per_token_group_fp8_quant_packed(*args)

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
    )


if __name__ == "__main__":
    import argparse

    # Verify numerics across every benchmarked shape before timing.
    correctness_check()

    parser = argparse.ArgumentParser()
    parser.add_argument("--no-cudagraph", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    cli_args = parser.parse_args()
    main(
        cudagraph=not cli_args.no_cudagraph,
        limit=cli_args.limit,
    )
