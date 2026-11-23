"""better-benchmark repro amax_sum_7f67e161bd21 (hf/infer/DebertaV2ForMaskedLM).

Inductor emits one persistent reduction,
``triton_per_fused_convert_element_type_div_exp_full_prepare_softmax_online_sub_view_where_0``:
the attention softmax of DeBERTa-v2's disentangled self-attention, whose
``masked_fill`` of the scores uses a mask that is constant in the graph (a
``full`` of ``False``, ``[B, 1, Q, K]``). Per row of the bf16 scores
``[B*H, Q, K]`` (viewed as ``[B, H, Q, K]``) it selects
``where(False, bf16 min, x)`` in fp32, takes the row max, the sum of
``exp(x - max)`` and stores ``exp(x - max) / sum`` in bf16. The mask is a
constant inside Inductor's kernel too (``tl.full([1, 1], False, tl.int1)``),
never a buffer, so the kernel's only traffic is one load and one store per
element; the port keeps the ``where`` with a constant ``False`` mask for the
backend to fold, as Triton does for Inductor. Inductor flattens the
contiguous problem to ``B*H*Q`` rows of ``K``; the port does the same with a
host-side view and returns the result as the ``[B*H, Q, K]`` view.
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(static_shapes=True)
def masked_softmax(scores: torch.Tensor) -> torch.Tensor:
    # scores: [B*H*Q, K] bf16, the row view of the contiguous input.
    rows, cols = scores.shape
    out = torch.empty([rows, cols], dtype=scores.dtype, device=scores.device)
    for tile_r in hl.tile(rows):
        x = scores[tile_r, :].to(torch.float32)
        # where(full(False), full([], -3.3895313892515355e+38), view): the
        # graph's constant mask and the bf16 min it would fill in. The mask
        # takes the row block's shape from x, so K stays a reduction dim
        # (hl.full([tile_r, cols], ...) would pin K as a literal factory
        # size and drop the looped-reduction choice).
        masked = torch.zeros_like(x, dtype=torch.bool)
        x = torch.where(masked, -3.3895313892515355e38, x)
        row_max = torch.amax(x, dim=-1, keepdim=True)
        unnormalized = torch.exp(x - row_max)
        row_sum = torch.sum(unnormalized, dim=-1, keepdim=True)
        probs = unnormalized / row_sum
        out[tile_r, :] = probs.to(out.dtype)
    return out


def forward(inputs: list) -> tuple[torch.Tensor, ...]:
    scores = inputs[0]
    view_shape = inputs[2]  # _shape_param_1 = [-1, H, Q, K]
    out_shape = inputs[3]  # _shape_param_2 = [-1, Q, K]
    out = masked_softmax(scores.view(-1, view_shape[-1]))
    return (out.view(out_shape),)
