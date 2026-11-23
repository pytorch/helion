"""better-benchmark repro pointwise_0872313c7e7a (torchbench/infer+train dcgan).

Inductor emits one pointwise kernel,
``triton_poi_fused_convert_element_type_gt_mul_where_0``: the decomposed
LeakyReLU(0.2) of a bf16 activation, ``where(x > 0, x, x * 0.2)``, computed in
fp32 and stored in bf16. The input (``[batch, channels, height, width]``) is
contiguous, so Inductor indexes it as one flat dimension of
``batch * channels * height * width`` elements (one load and one store per
element); the port tiles the same flat view.
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(static_shapes=True)
def leaky_relu(x: torch.Tensor) -> torch.Tensor:
    flat = x.view(-1)
    n = flat.size(0)
    out = torch.empty([n], dtype=x.dtype, device=x.device)
    for tile_n in hl.tile(n):
        v = flat[tile_n].to(torch.float32)
        out[tile_n] = torch.where(v > 0, v, v * 0.2).to(out.dtype)
    return out


def forward(inputs: list) -> tuple[torch.Tensor, ...]:
    (arg0_1,) = inputs[:1]
    out = leaky_relu(arg0_1)
    return (out.view(arg0_1.shape),)
