"""better-benchmark repro any_amax_sum_e232f7a3d0d8 (hf/infer BertForMaskedLM, RobertaForCausalLM).

Inductor emits one persistent reduction,
``triton_per_fused_any_convert_element_type_div_eq_exp_full_logical_not_prepare_softmax_online_sub_view_where_0``:
the attention probabilities of SDPA's math path in HF inference, a decomposed
``_safe_softmax``. Per row of the bf16 scores ``[B*H, Q, K]`` (viewed as
``[B, H, Q, K]``) it reads the scores once and takes an fp32 softmax over K
(row max, sum of ``exp(x - max)``, true division) and, in the same pass,
``all(x == -inf)`` written as ``logical_not(any(logical_not(x == -inf)))``;
fully masked rows store zero (the decomposition's bf16 ``full(0)``, which
Inductor folds into a constant) instead of the softmax. Like Inductor, the
``where`` runs in fp32 and the result is cast to bf16 once at the store.
Inductor flattens the contiguous problem to ``B*H*Q`` rows of ``K``; the port
does the same with a host-side view and returns the ``[B, H, Q, K]`` result
as its contiguous ``[B*H, Q, K]`` view (the graph's ``expand`` is a no-op).
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(static_shapes=True)
def safe_softmax(scores: torch.Tensor) -> torch.Tensor:
    # scores: [rows, K] bf16, a contiguous view of the [B*H, Q, K] input.
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
        result = torch.where(torch.logical_not(any_live), 0.0, probs)
        out[tile_r, :] = result.to(out.dtype)
    return out


def forward(inputs: list) -> tuple[torch.Tensor, ...]:
    scores = inputs[0]
    view_shape = inputs[1]  # _shape_param_0 = [B, H, Q, K]
    out_shape = inputs[4]  # _shape_param_3 = [B*H, Q, K]
    k_len = view_shape[-1]
    out = safe_softmax(scores.view(-1, k_len))
    return (out.view(out_shape),)
