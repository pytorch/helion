"""better-benchmark repro amax_sum_any_d7a4553f749f (hf/train/AlbertForMaskedLM).

Inductor emits one persistent reduction,
``triton_per_fused_any_convert_element_type_div_eq_exp_logical_not_prepare_softmax_online_sub_view_where_0``:
the attention probabilities of SDPA's math path, a decomposed
``_safe_softmax``. Per row of the bf16 scores ``[B*H, Q, K]`` (viewed as
``[B, H, Q, K]``) it takes an fp32 softmax over K (row max, sum of
``exp(x - max)``, true division) and, in the same pass, ``all(x == -inf)``
written as ``logical_not(any(logical_not(x == -inf)))``; fully masked rows
take the matching row of the second bf16 ``[B, H, Q, K]`` input (the
decomposition's ``zeros_like``, an input of the repro) instead of the
softmax. Like Inductor, both ``where`` branches stay in fp32 until the single
bf16 store. Inductor flattens the contiguous problem to ``B*H*Q`` rows of
``K``; the port does the same with host-side views and returns the result as
``[B, H, Q, K]`` and as its ``[B*H, Q, K]`` view.
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(static_shapes=True)
def safe_softmax(scores: torch.Tensor, fallback: torch.Tensor) -> torch.Tensor:
    # scores, fallback: [B*H*Q, K] bf16, row views of the two contiguous inputs.
    rows, cols = scores.shape
    out = torch.empty([rows, cols], dtype=scores.dtype, device=scores.device)
    for tile_r in hl.tile(rows):
        x = scores[tile_r, :].to(torch.float32)
        live = torch.logical_not(x == float("-inf"))
        any_live = torch.any(live, dim=-1, keepdim=True)
        row_max = torch.amax(x, dim=-1, keepdim=True)
        unnormalized = torch.exp(x - row_max)
        row_sum = torch.sum(unnormalized, dim=-1, keepdim=True)
        probs = unnormalized / row_sum
        other = fallback[tile_r, :].to(torch.float32)
        result = torch.where(torch.logical_not(any_live), other, probs)
        out[tile_r, :] = result.to(out.dtype)
    return out


def forward(inputs: list) -> tuple[torch.Tensor, ...]:
    scores, fallback = inputs[:2]
    view_shape = inputs[2]  # _shape_param_0 = [B, H, Q, K]
    out_shape = inputs[4]  # _shape_param_2 = [B*H, Q, K]
    k_len = view_shape[-1]
    out = safe_softmax(scores.view(-1, k_len), fallback.view(-1, k_len))
    return out.view(view_shape), out.view(out_shape)
