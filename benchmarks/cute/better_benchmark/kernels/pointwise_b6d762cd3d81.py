"""better-benchmark repro pointwise_b6d762cd3d81 (torchbench/infer+train nvidia_deeprecommender).

Inductor emits one pointwise kernel,
``triton_poi_fused_convert_element_type_expm1_gt_mul_where_0``: the decomposed
SELU of a bf16 activation, ``where(x > 0, x * 1.0507009873554805,
expm1(x * 1.0) * 1.7580993408473766)`` (scale and scale * alpha), computed in
fp32 and stored in bf16 (Inductor drops the graph's ``* 1.0``; the port keeps
it, an exact identity). The input is contiguous, so Inductor indexes it as one
flat dimension of ``rows * cols`` elements (one load and one store per
element); the port tiles the same flat view.
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(static_shapes=True)
def selu(x: torch.Tensor) -> torch.Tensor:
    flat = x.view(-1)
    n = flat.size(0)
    out = torch.empty([n], dtype=x.dtype, device=x.device)
    for tile_n in hl.tile(n):
        v = flat[tile_n].to(torch.float32)
        positive = v * 1.0507009873554805
        negative = torch.expm1(v * 1.0) * 1.7580993408473766
        out[tile_n] = torch.where(v > 0, positive, negative).to(out.dtype)
    return out


def forward(inputs: list) -> tuple[torch.Tensor, ...]:
    (arg0_1,) = inputs[:1]
    out = selu(arg0_1)
    return (out.view(arg0_1.shape),)
