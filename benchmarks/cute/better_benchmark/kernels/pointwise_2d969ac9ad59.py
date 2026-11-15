"""better-benchmark repro pointwise_2d969ac9ad59 (hf GPTNeoForCausalLM and others).

Inductor emits one pointwise kernel, ``triton_poi_fused_add_mul_pow_tanh_view_0``:
the tanh-approximate GELU of a bf16 activation,
``(x * 0.5) * (tanh((x + x**3 * 0.044715) * 0.7978845608028654) + 1)``,
computed in fp32 and stored in bf16 (Inductor, and Helion on both backends,
expand ``pow(x, 3.0)`` into ``(x * x) * x``). The graph's views
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
def gelu_tanh(x: torch.Tensor) -> torch.Tensor:
    flat = x.view(-1)
    n = flat.size(0)
    out = torch.empty([n], dtype=x.dtype, device=x.device)
    for tile_n in hl.tile(n):
        v = flat[tile_n].to(torch.float32)
        half = v * 0.5
        cube = torch.pow(v, 3.0)
        inner = (v + cube * 0.044715) * 0.7978845608028654
        out[tile_n] = (half * (torch.tanh(inner) + 1.0)).to(out.dtype)
    return out


def forward(inputs: list) -> tuple[torch.Tensor, ...]:
    arg0_1 = inputs[0]
    out_shape = inputs[2]  # _shape_param_1 = [rows, cols]
    out = gelu_tanh(arg0_1)
    return (out.view(out_shape),)
