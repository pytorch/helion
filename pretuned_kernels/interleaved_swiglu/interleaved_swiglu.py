"""BF16 projection with a fused interleaved SwiGLU epilogue.

The projection stores gate/value columns as adjacent register pairs.  The
checked-in B200 config applies ``silu(gate) * value`` directly to those tcgen05
fragment registers and writes the compacted output without materializing the
packed projection.
"""

from __future__ import annotations

import math

import torch

import helion
import helion.language as hl


@helion.aot_kernel(backend="cute", static_shapes=True, fast_math=True)
def interleaved_swiglu(
    x: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Project interleaved gate/value columns and return compact SwiGLU output."""
    m, k = x.size()
    heads, weight_k, packed_dim = weight.size()
    assert k == weight_k
    assert packed_dim % 2 == 0
    out = torch.empty(
        [heads, m, packed_dim // 2],
        dtype=x.dtype,
        device=x.device,
    )
    for tile_h, tile_m, tile_packed_d in hl.tile([heads, m, packed_dim]):
        acc = hl.zeros([tile_h, tile_m, tile_packed_d], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = hl.dot(
                x[tile_m, tile_k],
                weight[tile_h, tile_k, tile_packed_d],
                acc=acc,
            )
        pairs = acc.view(
            tile_h.block_size,
            tile_m.block_size,
            tile_packed_d.block_size // 2,
            2,
        )
        gate, value = hl.split(pairs)
        output_d = hl.split(tile_packed_d.index.view(tile_packed_d.block_size // 2, 2))[
            0
        ]
        output_d = output_d // 2
        out[
            tile_h.index[:, None, None],
            tile_m.index[None, :, None],
            output_d[None, None, :],
        ] = (gate * torch.sigmoid(gate) * value).to(x.dtype)
    return out


SHAPES = [(1024, 4096, 1, 11008)]  # (M, K, heads, packed_dim)


def _make_inputs(
    m: int,
    k: int,
    heads: int,
    packed_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = (torch.randn(m, k, device="cuda", dtype=torch.bfloat16) / math.sqrt(k)).to(
        torch.bfloat16
    )
    weight = torch.randn(
        heads,
        k,
        packed_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    return x, weight


def _apply_swiglu(projected: torch.Tensor, output_dtype: torch.dtype) -> torch.Tensor:
    heads, m, packed_dim = projected.shape
    gate, value = projected.view(heads, m, packed_dim // 2, 2).unbind(dim=-1)
    return (torch.nn.functional.silu(gate) * value).to(output_dtype)


def _interleaved_swiglu_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Numerical reference matching the kernel's FP32 accumulation boundary."""
    projected = torch.einsum("mk,hkd->hmd", x.float(), weight.float())
    return _apply_swiglu(projected, x.dtype)


def _interleaved_swiglu_eager(
    x: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Typical eager composition, which materializes a BF16 projection."""
    projected = torch.einsum("mk,hkd->hmd", x, weight)
    return _apply_swiglu(projected, x.dtype)


def use_cudagraph() -> bool:
    """Benchmark the inference composition under CUDA graphs with cold L2."""
    return True


def correctness_check() -> None:
    """Check the compact fragment store against FP32 accumulation."""
    torch.manual_seed(0)
    args = _make_inputs(128, 128, 2, 128)
    actual = interleaved_swiglu(*args)
    expected = _interleaved_swiglu_reference(*args)
    torch.testing.assert_close(actual, expected, atol=0.1, rtol=2e-2)


def main(verbose: bool = True) -> dict:
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from _bench import run_sweep  # pyrefly: ignore[missing-import]

    def make_calls(shape: tuple[int, int, int, int]) -> tuple:
        m, k, heads, packed_dim = shape
        args = _make_inputs(m, k, heads, packed_dim)

        def helion_call() -> torch.Tensor:
            return interleaved_swiglu(*args)

        def eager_call() -> torch.Tensor:
            return _interleaved_swiglu_eager(*args)

        return (
            helion_call,
            [("eager", eager_call)],
            f"{m:>5d}  {k:>5d}  {heads:>5d}  {packed_dim:>8d}",
        )

    return run_sweep(
        SHAPES,
        make_calls,
        use_cudagraph=use_cudagraph(),
        verbose=verbose,
        shape_header=f"{'M':>5s}  {'K':>5s}  {'heads':>5s}  {'packed D':>8s}",
    )


if __name__ == "__main__":
    correctness_check()
    main()
