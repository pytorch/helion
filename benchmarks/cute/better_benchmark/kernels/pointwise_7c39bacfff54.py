"""better-benchmark repro pointwise_7c39bacfff54 (hf Qwen3-0.6B, Mistral-7B-Instruct).

Inductor emits one pointwise kernel,
``triton_poi_fused_add_convert_element_type_div_exp_mul_neg_view_0``: the
SwiGLU gate of an MLP, ``silu(gate) * up`` with the SiLU written out as
``gate / (exp(-gate) + 1)``, on two bf16 activations, computed in fp32 and
stored in bf16. Like Inductor (no ``emulate_precision_casts``), the graph's
bf16 rounding of the SiLU before the multiply stays in fp32. The graph's views
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
def silu_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    gate_flat = gate.view(-1)
    up_flat = up.view(-1)
    n = gate_flat.size(0)
    out = torch.empty([n], dtype=gate.dtype, device=gate.device)
    for tile_n in hl.tile(n):
        g = gate_flat[tile_n].to(torch.float32)
        u = up_flat[tile_n].to(torch.float32)
        silu = g / (torch.exp(-g) + 1)
        out[tile_n] = (silu * u).to(out.dtype)
    return out


def forward(inputs: list) -> tuple[torch.Tensor, ...]:
    arg0_1, arg1_1 = inputs[:2]
    out_shape = inputs[4]  # _shape_param_2 = [rows, cols]
    out = silu_mul(arg0_1, arg1_1)
    return (out.view(out_shape),)
