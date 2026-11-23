"""better-benchmark repro amax_sum_fa4cc85fe5ad (hf/infer T5, T5Small and MT5).

Inductor emits one persistent reduction,
``triton_per_fused_convert_element_type_div_exp_prepare_softmax_online_sub_view_0``:
the T5 attention softmax without a position bias. Per row of the bf16 scores
``[B*H, Q, K]`` (viewed as ``[B, H, Q, K]``) it converts to fp32, takes the
row max, the sum of ``exp(x - max)`` and stores ``exp(x - max) / sum`` in
bf16: one load and one store per element. Inductor flattens the contiguous
problem to ``B*H*Q`` rows of ``K``; the port does the same with a host-side
view and returns the result as the ``[B*H, Q, K]`` view (the graph's
``expand`` is a no-op).
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(static_shapes=True)
def row_softmax(scores: torch.Tensor) -> torch.Tensor:
    # scores: [B*H*Q, K] bf16, the row view of the contiguous input.
    rows, cols = scores.shape
    out = torch.empty([rows, cols], dtype=scores.dtype, device=scores.device)
    for tile_r in hl.tile(rows):
        x = scores[tile_r, :].to(torch.float32)
        row_max = torch.amax(x, dim=-1, keepdim=True)
        unnormalized = torch.exp(x - row_max)
        row_sum = torch.sum(unnormalized, dim=-1, keepdim=True)
        probs = unnormalized / row_sum
        out[tile_r, :] = probs.to(out.dtype)
    return out


def forward(inputs: list) -> tuple[torch.Tensor, ...]:
    scores = inputs[0]
    view_shape = inputs[1]  # _shape_param_0 = [B, H, Q, K]
    out_shape = inputs[3]  # _shape_param_2 = [B*H, Q, K]
    out = row_softmax(scores.view(-1, view_shape[-1]))
    return (out.view(out_shape),)
