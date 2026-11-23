"""better-benchmark repro pointwise_8bb40810003c (hf GPTJForCausalLM and others).

Inductor emits one pointwise kernel,
``triton_poi_fused_convert_element_type_0``: the bf16 cast of an fp32 weight
matrix (the autocast copy of a linear layer's weight). The input is
contiguous, so Inductor indexes it as one flat dimension of ``rows * cols``
elements (one fp32 load and one bf16 store per element); the port tiles the
same flat view. The graph's two permutes are free: Inductor returns the
transposed view of the buffer and then the buffer itself, and the port
returns the same two views of its output.
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(static_shapes=True)
def cast_bf16(x: torch.Tensor) -> torch.Tensor:
    flat = x.view(-1)
    n = flat.size(0)
    out = torch.empty([n], dtype=torch.bfloat16, device=x.device)
    for tile_n in hl.tile(n):
        out[tile_n] = flat[tile_n].to(torch.float32).to(out.dtype)
    return out


def forward(inputs: list) -> tuple[torch.Tensor, ...]:
    arg0_1 = inputs[0]
    convert_element_type = cast_bf16(arg0_1).view(arg0_1.shape)
    permute = convert_element_type.permute(1, 0)
    permute_1 = permute.permute(1, 0)
    return (permute, permute_1)
