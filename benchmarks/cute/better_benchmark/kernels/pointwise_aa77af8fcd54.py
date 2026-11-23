"""better-benchmark repro pointwise_aa77af8fcd54 (hf gemma-2-2b, gemma-3-4b-it).

Inductor emits one pointwise kernel,
``triton_poi_fused_add_convert_element_type_mul_tanh_view_0``: the GeGLU gate
of an MLP, ``gelu_tanh(gate) * up`` with the tanh-approximate GELU written out
as ``(gate * 0.5) * (tanh((gate + gate * gate * gate * 0.044715) *
0.7978845608028654) + 1)``, on two bf16 activations, computed in fp32 and
stored in bf16. Like Inductor (no ``emulate_precision_casts``), the graph's
bf16 rounding of the GELU before the multiply stays in fp32. The graph's views
(``[rows, cols] -> [1, rows, cols] -> [rows, cols]``) are free and both inputs
are contiguous, so Inductor indexes them as one flat dimension of
``rows * cols`` elements (two loads and one store per element); the port tiles
the same flat views.
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(static_shapes=True)
def gelu_tanh_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    gate_flat = gate.view(-1)
    up_flat = up.view(-1)
    n = gate_flat.size(0)
    out = torch.empty([n], dtype=gate.dtype, device=gate.device)
    for tile_n in hl.tile(n):
        g = gate_flat[tile_n].to(torch.float32)
        u = up_flat[tile_n].to(torch.float32)
        half = g * 0.5
        cube = g * g * g
        inner = (g + cube * 0.044715) * 0.7978845608028654
        gelu = half * (torch.tanh(inner) + 1)
        out[tile_n] = (gelu * u).to(out.dtype)
    return out


def forward(inputs: list) -> tuple[torch.Tensor, ...]:
    arg0_1, arg1_1 = inputs[:2]
    out_shape = inputs[4]  # _shape_param_2 = [rows, cols]
    out = gelu_tanh_mul(arg0_1, arg1_1)
    return (out.view(out_shape),)
