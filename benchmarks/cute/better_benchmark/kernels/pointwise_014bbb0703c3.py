"""better-benchmark repro pointwise_014bbb0703c3 (timm visformer_small, hf whisper-tiny).

Inductor emits one pointwise kernel,
``triton_poi_fused_add_convert_element_type_erf_mul_0``: the exact (erf) GELU
of a bf16 activation, ``(x * 0.5) * (erf(x * 0.7071067811865476) + 1)``,
computed in fp32 and stored in bf16. The output keeps the input's layout
(channels-last for the visformer points, contiguous for whisper), and both are
dense, so Inductor indexes input and output as one flat dimension in storage
order (one load and one store per element). The port tiles the same flat view:
the host permutes the input's dims into storage order (a view), the kernel
writes a flat output, and the host permutes it back to the input's strides.
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(static_shapes=True)
def gelu_erf(x_flat: torch.Tensor) -> torch.Tensor:
    n = x_flat.size(0)
    out = torch.empty([n], dtype=x_flat.dtype, device=x_flat.device)
    for tile_n in hl.tile(n):
        v = x_flat[tile_n].to(torch.float32)
        half = v * 0.5
        cdf = torch.erf(v * 0.7071067811865476) + 1
        out[tile_n] = (half * cdf).to(out.dtype)
    return out


def forward(inputs: list) -> tuple[torch.Tensor, ...]:
    (arg0_1,) = inputs[:1]
    # Dims from outermost to innermost in memory (NHWC for channels-last).
    order = sorted(range(arg0_1.dim()), key=arg0_1.stride, reverse=True)
    storage = arg0_1.permute(order)
    out = gelu_erf(storage.view(-1))
    inverse = [order.index(d) for d in range(arg0_1.dim())]
    return (out.view(storage.shape).permute(inverse),)
