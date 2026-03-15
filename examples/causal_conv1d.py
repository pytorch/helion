"""
Causal Depthwise Conv1d
======================

This example implements a causal depthwise 1D convolution kernel in Helion.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

import helion
from helion._testing import DEVICE
from helion._testing import HALF_DTYPE
from helion._testing import run_example
import helion.language as hl


@helion.kernel()
def causal_conv1d(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    """
    Args:
        x: [batch, channels, seqlen]
        weight: [channels, width]
        bias: [channels]
    Returns:
        [batch, channels, seqlen]
    """
    batch, channels, seqlen = x.shape
    width = hl.specialize(weight.size(1))
    out = torch.empty_like(x)

    for tile_b, tile_d, tile_s in hl.tile([batch, channels, seqlen]):
        seq_idx = tile_s.index
        in_bounds = seq_idx < seqlen
        acc = hl.zeros([tile_d, tile_s], dtype=torch.float32) + bias[tile_d][:, None]

        if width == 4:
            x0 = hl.load(
                x,
                [tile_b.begin, tile_d, seq_idx - 3],
                extra_mask=(in_bounds & (seq_idx >= 3))[None, :],
            )
            x1 = hl.load(
                x,
                [tile_b.begin, tile_d, seq_idx - 2],
                extra_mask=(in_bounds & (seq_idx >= 2))[None, :],
            )
            x2 = hl.load(
                x,
                [tile_b.begin, tile_d, seq_idx - 1],
                extra_mask=(in_bounds & (seq_idx >= 1))[None, :],
            )
            x3 = hl.load(
                x,
                [tile_b.begin, tile_d, seq_idx],
                extra_mask=in_bounds[None, :],
            )
            acc = acc + x0 * weight[tile_d, 0][:, None]
            acc = acc + x1 * weight[tile_d, 1][:, None]
            acc = acc + x2 * weight[tile_d, 2][:, None]
            acc = acc + x3 * weight[tile_d, 3][:, None]
        else:
            for tap in range(width):
                shift = width - 1 - tap
                x_tap = hl.load(
                    x,
                    [tile_b.begin, tile_d, seq_idx - shift],
                    extra_mask=(in_bounds & (seq_idx >= shift))[None, :],
                )
                acc = acc + x_tap * weight[tile_d, tap][:, None]

        out[tile_b, tile_d, tile_s] = acc[None, :, :].to(out.dtype)
    return out


def ref_causal_conv1d(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    width = weight.shape[1]
    x_padded = F.pad(x, (width - 1, 0))
    return F.conv1d(x_padded, weight.unsqueeze(1), bias=bias, groups=x.shape[1])


def test(batch: int, channels: int, seqlen: int, width: int) -> None:
    x = torch.randn(batch, channels, seqlen, dtype=HALF_DTYPE, device=DEVICE)
    weight = torch.randn(channels, width, dtype=HALF_DTYPE, device=DEVICE)
    bias = torch.randn(channels, dtype=torch.float32, device=DEVICE)
    run_example(causal_conv1d, ref_causal_conv1d, (x, weight, bias), atol=1e-2)


def main() -> None:
    test(1, 64, 256, 4)
    test(1, 64, 256, 8)


if __name__ == "__main__":
    main()
