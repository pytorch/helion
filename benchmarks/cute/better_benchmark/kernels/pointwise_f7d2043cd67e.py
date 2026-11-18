"""better-benchmark repro pointwise_f7d2043cd67e (hf/infer/google/gemma-2-2b).

Inductor emits one pointwise kernel, ``triton_poi_fused_div_mul_tanh_view_0``:
Gemma's final logit soft-cap of a bf16 activation, ``tanh(x / 30.0) * 30.0``,
computed in fp32 and stored in bf16. The divisor is a literal in the graph, so
Inductor (and Helion on both backends) multiplies by its reciprocal,
``x * 0.03333333333333333``. The graph's view
(``[rows, cols] -> [1, rows, cols]``) is free and the input is contiguous, so
Inductor indexes it as one flat dimension of ``rows * cols`` elements (one load
and one store per element); the port tiles the same flat view.
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(static_shapes=True)
def tanh_soft_cap(x: torch.Tensor) -> torch.Tensor:
    flat = x.view(-1)
    n = flat.size(0)
    out = torch.empty([n], dtype=x.dtype, device=x.device)
    for tile_n in hl.tile(n):
        v = flat[tile_n].to(torch.float32)
        capped = torch.tanh(v / 30.0) * 30.0
        out[tile_n] = capped.to(out.dtype)
    return out


def forward(inputs: list) -> tuple[torch.Tensor, ...]:
    arg0_1 = inputs[0]
    out_shape = inputs[1]  # _shape_param_0 = [1, rows, cols]
    out = tanh_soft_cap(arg0_1)
    return (out.view(out_shape),)
