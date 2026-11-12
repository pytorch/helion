"""better-benchmark repro mean_c9e1a8113328 (timm nfnet_l0 / dm_nfnet_f0).

Inductor emits one looped reduction, ``triton_red_fused_mean_0``: the
squeeze-and-excitation pool of the NFNet blocks, ``mean(x, [2, 3],
keepdim=True)`` of a channels-last bf16 ``[N, C, H, W]`` activation. Each
``(n, c)`` output reads its ``H * W`` spatial values (``C`` elements apart in
memory), upcasts them to fp32, sums them and divides by ``H * W``; the mean is
stored in bf16 as the contiguous ``[N, C, 1, 1]`` output. Nothing else is
written. Inductor flattens ``(n, c)`` into one parallel index; the port tiles
``n`` and ``c`` and reduces ``x[tile_n, tile_c, :, :]`` over ``[2, 3]``, which
Helion merges into one reduction over ``x.flatten(2, 3)`` (a view: the same
``H * W`` values at stride ``C``).
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(static_shapes=True)
def spatial_mean(x: torch.Tensor) -> torch.Tensor:
    # x: channels-last [N, C, H, W].
    batch, channels, _, _ = x.shape
    out = torch.empty([batch, channels, 1, 1], dtype=x.dtype, device=x.device)
    for tile_n, tile_c in hl.tile([batch, channels]):
        values = x[tile_n, tile_c, :, :].to(torch.float32)
        out[tile_n, tile_c, :, :] = torch.mean(values, dim=[2, 3], keepdim=True).to(
            out.dtype
        )
    return out


def forward(inputs: list) -> tuple[torch.Tensor, ...]:
    (x,) = inputs[:1]
    return (spatial_mean(x),)
