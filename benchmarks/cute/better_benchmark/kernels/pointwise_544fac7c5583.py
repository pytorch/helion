"""better-benchmark repro pointwise_544fac7c5583 (timm visformer_small).

Inductor emits one pointwise kernel,
``triton_poi_fused_add_convert_element_type_erf_exp_mul_0``: the backward of
the exact (erf) GELU, the incoming bf16 gradient times the GELU's derivative
at the bf16 activation ``x``,
``grad * ((erf(x * 0.7071067811865476) + 1) * 0.5
+ x * (exp((x * x) * -0.5) * 0.3989422804014327))``,
computed in fp32 and stored in bf16. Both inputs and the output are
channels-last ``[N, C, H, W]`` with the same strides, and all three are dense,
so Inductor indexes them as one flat dimension in storage order (two loads and
one store per element). The port tiles the same flat view: the host permutes
the inputs' dims into storage order (views), the kernel writes a flat output,
and the host permutes it back to the inputs' strides.
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(static_shapes=True)
def gelu_erf_backward(grad_flat: torch.Tensor, x_flat: torch.Tensor) -> torch.Tensor:
    n = x_flat.size(0)
    out = torch.empty([n], dtype=torch.bfloat16, device=x_flat.device)
    for tile_n in hl.tile(n):
        grad = grad_flat[tile_n].to(torch.float32)
        v = x_flat[tile_n].to(torch.float32)
        cdf = (torch.erf(v * 0.7071067811865476) + 1) * 0.5
        pdf = torch.exp((v * v) * -0.5) * 0.3989422804014327
        out[tile_n] = (grad * (cdf + v * pdf)).to(out.dtype)
    return out


def forward(inputs: list) -> tuple[torch.Tensor, ...]:
    arg0_1, arg1_1 = inputs[:2]
    # Dims from outermost to innermost in memory (NHWC for channels-last). The
    # activation shares the gradient's layout at every shape point; view(-1)
    # raises, before the launch, for one that is not dense in that order.
    order = sorted(range(arg0_1.dim()), key=arg0_1.stride, reverse=True)
    grad = arg0_1.permute(order)
    out = gelu_erf_backward(grad.view(-1), arg1_1.permute(order).view(-1))
    inverse = [order.index(d) for d in range(arg0_1.dim())]
    return (out.view(grad.shape).permute(inverse),)
