"""better-benchmark repro sum_4fd6e4019857 (genai SoftmaxBackward, static).

Inductor emits one looped reduction,
``triton_red_fused_convert_element_type_div_exp_expand_fma_mul_neg_sub_sum_0``:
a softmax backward that recomputes the probabilities ``p = exp(x - max) / sum``
from the bf16 logits ``[M, N]`` and the saved fp32 row max and row sum
``[M, 1]``, with the upstream gradient a 0-d bf16 scalar ``g`` expanded to
``[M, N]``. The first loop over N sums ``g * p`` per row; the second reloads
the logits, recomputes ``p`` and stores ``fma(-p, sum, g * p)`` in bf16. Like
Inductor, ``p`` stays in fp32 (the graph's bf16 round trip of ``p`` is
elided) and the row sum is not an output.
"""

from __future__ import annotations

import torch
import torch._inductor.inductor_prims  # noqa: F401  (registers prims.fma)

import helion
import helion.language as hl


@helion.kernel(static_shapes=True)
def softmax_recompute_backward(
    logits: torch.Tensor,
    row_max: torch.Tensor,
    row_sum: torch.Tensor,
    grad_scale: torch.Tensor,
) -> torch.Tensor:
    # logits: [M, N] bf16; row_max, row_sum: [M, 1] fp32; grad_scale: 0-d bf16.
    m, n = logits.shape
    out = torch.empty([m, n], dtype=logits.dtype, device=logits.device)
    for tile_m in hl.tile(m):
        g = grad_scale[()].to(torch.float32)
        # The statistics as ``[tile_m]`` loads, read once per row as in
        # Inductor (``[tile_m, :]`` slices are reloaded in every step of the
        # second loop).
        tile_max = row_max[tile_m, 0]
        tile_sum = row_sum[tile_m, 0]
        probs = (
            torch.exp(logits[tile_m, :].to(torch.float32) - tile_max[:, None])
            / tile_sum[:, None]
        )
        scaled = g * probs
        total = torch.sum(scaled, dim=-1, keepdim=True)
        grad = torch.ops.prims.fma(-probs, total, scaled)
        out[tile_m, :] = grad.to(out.dtype)
    return out


def forward(inputs: list) -> tuple[torch.Tensor, ...]:
    logits, row_max, row_sum, grad_scale = inputs[:4]
    # _shape_param_0 = [M, N] expands grad_scale; the kernel broadcasts it.
    return (softmax_recompute_backward(logits, row_max, row_sum, grad_scale),)
