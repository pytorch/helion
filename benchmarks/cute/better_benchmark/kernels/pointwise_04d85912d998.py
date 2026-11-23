"""better-benchmark repro pointwise_04d85912d998 (hf GPTJForCausalLM and others).

Inductor emits one pointwise kernel,
``triton_poi_fused_convert_element_type_0``: the fp32 cast of a bf16 tensor
(linear- and convolution-weight shapes, 2-D to 4-D). The output keeps the
input's layout (contiguous at most shape points, channels-last at some of the
4-D ones), and both are dense, so Inductor indexes input and output as one
flat dimension in storage order (one bf16 load and one fp32 store per
element). The port tiles the same flat view: the host permutes the input's
dims into storage order (a view), the kernel writes a flat output, and the
host permutes it back to the input's strides.
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(static_shapes=True)
def cast_fp32(x_flat: torch.Tensor) -> torch.Tensor:
    n = x_flat.size(0)
    out = torch.empty([n], dtype=torch.float32, device=x_flat.device)
    for tile_n in hl.tile(n):
        out[tile_n] = x_flat[tile_n].to(torch.float32)
    return out


def forward(inputs: list) -> tuple[torch.Tensor, ...]:
    (arg0_1,) = inputs[:1]
    # Dims from outermost to innermost in memory (NHWC for channels-last).
    order = sorted(range(arg0_1.dim()), key=arg0_1.stride, reverse=True)
    storage = arg0_1.permute(order)
    out = cast_fp32(storage.view(-1))
    inverse = [order.index(d) for d in range(arg0_1.dim())]
    return (out.view(storage.shape).permute(inverse),)
