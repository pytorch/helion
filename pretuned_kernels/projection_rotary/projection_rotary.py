"""BF16 projection with a fused adjacent-pair rotary epilogue.

The projection accumulates in FP32, adds a per-head bias, and rotates adjacent
output values with a half-packed ``[sin..., cos...]`` table.  The checked-in
B200 config keeps the epilogue in the tcgen05 register fragment instead of
materializing the projection before the rotation.
"""

from __future__ import annotations

import math

import torch

import helion
import helion.language as hl


@helion.aot_kernel(backend="cute", static_shapes=True)
def projection_rotary(
    x: torch.Tensor,
    weight: torch.Tensor,
    table: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """Return ``projection(x, weight)`` with fused bias and rotary mixing."""
    m, k = x.size()
    heads, weight_k, head_dim = weight.size()
    assert k == weight_k
    assert head_dim % 2 == 0
    out = torch.empty(
        [heads, m, head_dim],
        dtype=x.dtype,
        device=x.device,
    )
    for tile_h, tile_m, tile_d in hl.tile([heads, m, head_dim]):
        acc = hl.zeros([tile_h, tile_m, tile_d], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = hl.dot(
                x[tile_m, tile_k],
                weight[tile_h, tile_k, tile_d],
                acc=acc,
            )
        acc = acc + bias[tile_h.index[:, None], tile_d.index[None, :]][:, None, :]

        # ``table`` is half-packed along D. Recover the global adjacent-pair
        # index so every D tile addresses the same [sin..., cos...] layout.
        pair_index = hl.split(tile_d.index.view(tile_d.block_size // 2, 2))[0]
        pair_index = pair_index // 2
        sin = table[tile_m.index[:, None], pair_index[None, :]]
        cos = table[
            tile_m.index[:, None],
            head_dim // 2 + pair_index[None, :],
        ]

        pairs = acc.view(
            tile_h.block_size,
            tile_m.block_size,
            tile_d.block_size // 2,
            2,
        )
        left, right = hl.split(pairs)
        rotated = hl.join(left * cos - right * sin, right * cos + left * sin)
        out[tile_h, tile_m, tile_d] = rotated.view(
            tile_h.block_size,
            tile_m.block_size,
            tile_d.block_size,
        ).to(x.dtype)
    return out


SHAPES = [(1024, 4096, 32, 128)]  # (M, K, heads, head_dim)


def _make_inputs(
    m: int,
    k: int,
    heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x = (torch.randn(m, k, device="cuda", dtype=torch.bfloat16) / math.sqrt(k)).to(
        torch.bfloat16
    )
    weight = torch.randn(
        heads,
        k,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    angles = torch.randn(m, head_dim // 2, device="cuda", dtype=torch.float32)
    table = torch.cat((angles.sin(), angles.cos()), dim=-1).to(torch.bfloat16)
    bias = torch.randn(
        heads,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    return x, weight, table, bias


def _apply_rotary(
    projected: torch.Tensor,
    table: torch.Tensor,
    bias: torch.Tensor,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    heads, m, head_dim = projected.shape
    pairs = (projected + bias[:, None, :]).view(heads, m, head_dim // 2, 2)
    sin, cos = table.view(m, 2, head_dim // 2).unbind(dim=1)
    left, right = pairs.unbind(dim=-1)
    return (
        torch.stack(
            (left * cos - right * sin, right * cos + left * sin),
            dim=-1,
        )
        .view(heads, m, head_dim)
        .to(output_dtype)
    )


def _projection_rotary_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    table: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """Numerical reference matching the kernel's FP32 accumulation boundary."""
    projected = torch.einsum("mk,hkd->hmd", x.float(), weight.float())
    return _apply_rotary(projected, table.float(), bias.float(), x.dtype)


def _projection_rotary_eager(
    x: torch.Tensor,
    weight: torch.Tensor,
    table: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """Typical eager composition, which materializes a BF16 projection."""
    projected = torch.einsum("mk,hkd->hmd", x, weight)
    return _apply_rotary(projected, table, bias, x.dtype)


def use_cudagraph() -> bool:
    """Benchmark the inference composition under CUDA graphs with cold L2."""
    return True


def correctness_check() -> None:
    """Check the pretuned fragment epilogue against FP32 accumulation."""
    torch.manual_seed(0)
    args = _make_inputs(128, 128, 2, 128)
    actual = projection_rotary(*args)
    expected = _projection_rotary_reference(*args)
    torch.testing.assert_close(actual, expected, atol=0.1, rtol=2e-2)


def main(verbose: bool = True) -> dict:
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from _bench import run_sweep  # pyrefly: ignore[missing-import]

    def make_calls(shape: tuple[int, int, int, int]) -> tuple:
        m, k, heads, head_dim = shape
        args = _make_inputs(m, k, heads, head_dim)

        def helion_call() -> torch.Tensor:
            return projection_rotary(*args)

        def eager_call() -> torch.Tensor:
            return _projection_rotary_eager(*args)

        return (
            helion_call,
            [("eager", eager_call)],
            f"{m:>5d}  {k:>5d}  {heads:>5d}  {head_dim:>5d}",
        )

    return run_sweep(
        SHAPES,
        make_calls,
        use_cudagraph=use_cudagraph(),
        verbose=verbose,
        shape_header=f"{'M':>5s}  {'K':>5s}  {'heads':>5s}  {'D':>5s}",
    )


if __name__ == "__main__":
    correctness_check()
    main()
