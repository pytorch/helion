"""better-benchmark repro amax_sum_20ccc18dda60 (hf/infer/TrOCRForCausalLM).

Inductor emits one persistent reduction,
``triton_per_fused_add_convert_element_type_div_exp_prepare_softmax_online_sub_view_0``:
TrOCR's decoder attention softmax with its additive attention mask. Per row
of the bf16 scores ``[B*H, Q, K]`` (viewed as ``[B, H, Q, K]``) it adds the
bf16 mask ``[B, 1, Q, K]`` in fp32 (no bf16 rounding of the graph's ``add``),
takes the row max, the sum of ``exp(x - max)`` and stores
``exp(x - max) / sum`` in bf16. The mask broadcasts over the heads:
Inductor's flat row index ``x`` reads it at ``(x // (H*Q), x % Q)``; the port
tiles ``[B, H, Q]`` and broadcasts the load instead. The output
``[B, H, Q, K]`` is returned as the contiguous view ``[B*H, Q, K]``.
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(static_shapes=True)
def mask_add_softmax(scores: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
    # scores: [B, H, Q, K] bf16, a view of the [B*H, Q, K] input; attn_mask:
    # [B, Q, K] bf16, a view of the [B, 1, Q, K] input.
    batch, heads, q_len, k_len = scores.shape
    out = torch.empty(
        [batch, heads, q_len, k_len], dtype=scores.dtype, device=scores.device
    )
    for tile_b, tile_h, tile_q in hl.tile([batch, heads, q_len]):
        x = (
            scores[tile_b, tile_h, tile_q, :].to(torch.float32)
            + attn_mask[tile_b, tile_q, :].to(torch.float32)[:, None, :, :]
        )
        row_max = torch.amax(x, dim=-1, keepdim=True)
        unnormalized = torch.exp(x - row_max)
        row_sum = torch.sum(unnormalized, dim=-1, keepdim=True)
        probs = unnormalized / row_sum
        out[tile_b, tile_h, tile_q, :] = probs.to(out.dtype)
    return out


def forward(inputs: list) -> tuple[torch.Tensor, ...]:
    scores, attn_mask = inputs[:2]
    view_shape = inputs[2]  # _shape_param_0 = [B, H, Q, K]
    out_shape = inputs[3]  # _shape_param_1 = [B*H, Q, K]
    batch, _, q_len, k_len = view_shape
    out = mask_add_softmax(scores.view(view_shape), attn_mask.view(batch, q_len, k_len))
    return (out.view(out_shape),)
