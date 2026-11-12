"""better-benchmark repro pointwise_826ea0b5f6e4 (hf BertForMaskedLM and others).

Inductor emits one pointwise kernel,
``triton_poi_fused_add_convert_element_type_erf_mul_view_0``: the exact (erf)
GELU of a bf16 activation, ``(x * 0.5) * (erf(x * 0.7071067811865476) + 1)``,
computed in fp32 and stored in bf16. The graph's views
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
def gelu_erf(x: torch.Tensor) -> torch.Tensor:
    flat = x.view(-1)
    n = flat.size(0)
    out = torch.empty([n], dtype=x.dtype, device=x.device)
    for tile_n in hl.tile(n):
        v = flat[tile_n].to(torch.float32)
        half = v * 0.5
        cdf = torch.erf(v * 0.7071067811865476) + 1
        out[tile_n] = (half * cdf).to(out.dtype)
    return out


def forward(inputs: list) -> tuple[torch.Tensor, ...]:
    arg0_1 = inputs[0]
    out_shape = inputs[2]  # _shape_param_1 = [rows, cols]
    out = gelu_erf(arg0_1)
    return (out.view(out_shape),)
