"""better-benchmark repro pointwise_1df91f562302 (timm dm_nfnet_f0).

Inductor emits one pointwise kernel,
``triton_poi_fused_add_convert_element_type_erf_mul_0``: the scaled exact
(erf) GELU that opens an NFNet block,
``(x * 0.5) * (erf(x * 0.7071067811865476) + 1) * 1.7015043497085571``, of a
channels-last bf16 ``[N, C, H, W]`` activation, stored in bf16 in the same
layout. Everything is computed in fp32: the graph rounds the GELU to bf16
before the ``1.7015...`` scale (NFNet's ``gamma``), but Inductor does not
materialize that rounding (no ``emulate_precision_casts``), so the scale
multiplies the fp32 GELU and only the product is rounded at the store.

Input and output are dense, so Inductor indexes both as one flat dimension in
storage order (one bf16 load and one bf16 store per element). The port tiles
the same flat view: the host permutes the input's dims into storage order (a
view), the kernel writes a flat output, and the host permutes it back to the
input's strides.
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(static_shapes=True)
def scaled_gelu_erf(x_flat: torch.Tensor) -> torch.Tensor:
    n = x_flat.size(0)
    out = torch.empty([n], dtype=x_flat.dtype, device=x_flat.device)
    for tile_n in hl.tile(n):
        v = x_flat[tile_n].to(torch.float32)
        half = v * 0.5
        cdf = torch.erf(v * 0.7071067811865476) + 1
        gelu = half * cdf
        out[tile_n] = (gelu * 1.7015043497085571).to(out.dtype)
    return out


def forward(inputs: list) -> tuple[torch.Tensor, ...]:
    (arg0_1,) = inputs[:1]
    # Dims from outermost to innermost in memory (NHWC for channels-last).
    order = sorted(range(arg0_1.dim()), key=arg0_1.stride, reverse=True)
    storage = arg0_1.permute(order)
    out = scaled_gelu_erf(storage.view(-1))
    inverse = [order.index(d) for d in range(arg0_1.dim())]
    return (out.view(storage.shape).permute(inverse),)
