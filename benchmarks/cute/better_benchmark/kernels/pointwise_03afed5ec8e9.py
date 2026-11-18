"""better-benchmark repro pointwise_03afed5ec8e9 (hf GPTNeoForCausalLM and others).

Inductor emits one pointwise kernel,
``triton_poi_fused_clone_convert_element_type_permute_view_0``: the contiguous
fp32 copy of an attention projection split into heads,
``x.view(batch, seq, heads, head_dim).permute(0, 2, 1, 3).float().clone()``, of
a contiguous bf16 ``[batch * seq, heads * head_dim]`` activation. The graph's
views, the permute and the expand (to the shape the tensor already has) are
free; the kernel runs over the contiguous ``[batch, heads, seq, head_dim]``
output as one flat dimension, decomposes the index into the four coordinates
and gathers the element at ``d + head_dim * h + hidden * s + seq * hidden * b``
of the input (one bf16 load, widened to fp32, and one fp32 store per element).
The port indexes the same permuted (and expanded) view of the input directly,
tiling its four dimensions, and writes the contiguous fp32 output, which the
host views as ``[batch * heads, seq, head_dim]`` like the graph.
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(static_shapes=True)
def split_heads_cast_copy(x: torch.Tensor) -> torch.Tensor:
    # x: [batch, heads, seq, head_dim], the permuted view of the activation.
    batch, heads, seq, head_dim = x.shape
    out = torch.empty(
        [batch, heads, seq, head_dim], dtype=torch.float32, device=x.device
    )
    for tile_b, tile_h, tile_s, tile_d in hl.tile([batch, heads, seq, head_dim]):
        out[tile_b, tile_h, tile_s, tile_d] = x[tile_b, tile_h, tile_s, tile_d].to(
            torch.float32
        )
    return out


def forward(inputs: list) -> tuple[torch.Tensor, ...]:
    arg0_1 = inputs[0]
    hidden_shape = inputs[1]  # _shape_param_0 = [batch, seq, hidden]
    heads_shape = inputs[2]  # _shape_param_1 = [batch, seq, heads, head_dim]
    expand_shape = inputs[3]  # _shape_param_2 = [batch, heads, seq, head_dim]
    out_shape = inputs[4]  # _shape_param_3 = [batch * heads, seq, head_dim]
    permute = arg0_1.view(hidden_shape).view(heads_shape).permute(0, 2, 1, 3)
    out = split_heads_cast_copy(permute.expand(expand_shape))
    return (out.view(out_shape),)
