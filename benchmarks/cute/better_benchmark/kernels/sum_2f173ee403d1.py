"""better-benchmark repro sum_2f173ee403d1 (hf/train AlbertForMaskedLM, BlenderbotForConditionalGeneration).

Inductor emits one persistent reduction,
``triton_per_fused_convert_element_type_fma_mul_neg_sum_view_0``: the softmax
backward of the attention probabilities. Per row of the last dim it loads the
bf16 gradient of the probabilities (``[B*H, S, S]``, viewed as
``[B, H, S, S]``) and the bf16 probabilities (``[B, H, S, S]``), forms
``product = grad * probs`` in fp32, sums it over the row and stores
``fma(-probs, sum, product)`` in bf16, returned as a ``[B*H, S, S]`` view. The
row sum (``sum_1``) is not an output.

Like Inductor's kernel, which indexes both inputs as ``[B*H*S, S]``, the port
runs on contiguous two-dimensional views.
"""

from __future__ import annotations

import torch
import torch._inductor.inductor_prims  # noqa: F401  (registers prims.fma)

import helion
import helion.language as hl


@helion.kernel(static_shapes=True)
def softmax_backward(grad: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
    # grad, probs: [rows, n] bf16 (contiguous views of the [B*H, S, S] and
    # [B, H, S, S] inputs).
    rows, n = probs.shape
    out = torch.empty([rows, n], dtype=probs.dtype, device=probs.device)
    for tile_r in hl.tile(rows):
        p = probs[tile_r, :].to(torch.float32)
        product = grad[tile_r, :].to(torch.float32) * p
        total = torch.sum(product, dim=-1, keepdim=True)
        out[tile_r, :] = torch.ops.prims.fma(-p, total, product).to(out.dtype)
    return out


def forward(inputs: list) -> tuple[torch.Tensor, ...]:
    grad, probs = inputs[:2]
    out_shape = inputs[3]  # _shape_param_1 = [B*H, S, S]
    n = probs.size(-1)
    out = softmax_backward(grad.view(-1, n), probs.view(-1, n))
    return (out.view(out_shape),)
