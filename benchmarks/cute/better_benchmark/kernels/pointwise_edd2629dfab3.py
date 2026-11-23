"""better-benchmark repro pointwise_edd2629dfab3 (hf MobileBertForMaskedLM and others).

Inductor emits one pointwise kernel, ``triton_poi_fused_relu_view_0``: the ReLU
of a bf16 activation, ``maximum(0, x)`` (NaN-propagating), computed in fp32 and
stored in bf16. The graph's views
(``[rows, cols] -> [batch, seq, cols] -> [rows, cols]``) are free and the input
is contiguous, so Inductor indexes it as one flat dimension of ``rows * cols``
elements (one load and one store per element); the port tiles the same flat
view.
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(static_shapes=True)
def relu(x: torch.Tensor) -> torch.Tensor:
    flat = x.view(-1)
    n = flat.size(0)
    out = torch.empty([n], dtype=x.dtype, device=x.device)
    for tile_n in hl.tile(n):
        v = flat[tile_n].to(torch.float32)
        out[tile_n] = torch.relu(v).to(out.dtype)
    return out


def forward(inputs: list) -> tuple[torch.Tensor, ...]:
    arg0_1 = inputs[0]
    out_shape = inputs[2]  # _shape_param_1 = [rows, cols]
    out = relu(arg0_1)
    return (out.view(out_shape),)
