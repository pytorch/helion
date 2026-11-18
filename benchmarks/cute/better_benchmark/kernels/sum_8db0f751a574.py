"""better-benchmark repro sum_8db0f751a574 (hf/train BertForMaskedLM, RobertaForCausalLM, DistilBert, Electra, M2M100, MobileBert).

Inductor emits one persistent reduction,
``triton_per_fused_convert_element_type_fma_mul_neg_sum_view_0``: the
attention-probability backward of a training step, through dropout and the
softmax. Per row of the last dim it loads the bf16 gradient of the dropped
probabilities (``[B*H, S, S]``, viewed as ``[B, H, S, S]``), the bool dropout
mask and the bf16 probabilities (both ``[B, H, S, S]``), forms
``product = grad * (keep * 1.1111111111111112) * probs`` in fp32, sums it over
the row and stores ``fma(-probs, sum, product)`` in bf16, returned as a
``[B*H, S, S]`` view. The row sum (``sum_1``) is not an output.

As in Inductor's code (no ``emulate_precision_casts``), the graph's bf16
dropout scaling (``mul``, ``mul_1``) stays in fp32; only the output is
rounded. Like Inductor's kernel, which indexes all three inputs as
``[B*H*S, S]``, the port runs on contiguous two-dimensional views.
"""

from __future__ import annotations

import torch
import torch._inductor.inductor_prims  # noqa: F401  (registers prims.fma)

import helion
import helion.language as hl


@helion.kernel(static_shapes=True)
def dropout_softmax_backward(
    grad: torch.Tensor,
    keep: torch.Tensor,
    probs: torch.Tensor,
) -> torch.Tensor:
    # grad, probs: [rows, n] bf16; keep: [rows, n] bool (contiguous views of
    # the [B*H, S, S] and [B, H, S, S] inputs).
    rows, n = probs.shape
    out = torch.empty([rows, n], dtype=probs.dtype, device=probs.device)
    for tile_r in hl.tile(rows):
        scaled_grad = grad[tile_r, :].to(torch.float32) * (
            keep[tile_r, :].to(torch.float32) * 1.1111111111111112
        )
        p = probs[tile_r, :].to(torch.float32)
        product = scaled_grad * p
        total = torch.sum(product, dim=-1, keepdim=True)
        out[tile_r, :] = torch.ops.prims.fma(-p, total, product).to(out.dtype)
    return out


def forward(inputs: list) -> tuple[torch.Tensor, ...]:
    grad, keep, probs = inputs[:3]
    out_shape = inputs[4]  # _shape_param_1 = [B*H, S, S]
    n = probs.size(-1)
    out = dropout_softmax_backward(
        grad.view(-1, n), keep.view(-1, n), probs.view(-1, n)
    )
    return (out.view(out_shape),)
