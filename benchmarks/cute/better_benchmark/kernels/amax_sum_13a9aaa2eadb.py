"""better-benchmark repro amax_sum_13a9aaa2eadb (hf/infer GPT-Neo).

Inductor emits one persistent reduction,
``triton_per_fused_add_convert_element_type_div_exp_full_prepare_softmax_online_slice_sub_view_where_0``:
the attention softmax of GPTNeoForCausalLM and
GPTNeoForSequenceClassification, with its causal mask and additive attention
mask. Per row of the fp32 scores ``[B*H, Q, K]`` (viewed as ``[B, H, Q, K]``)
it selects ``where(causal, x, fp32 min)`` with the bool causal mask (the
``[1, 1, 2048, 2048]`` buffer sliced to ``[1, 1, Q, K]``, so rows 2048 apart
in memory), adds the bf16 attention mask ``[B, 1, Q, K]`` in fp32, takes the
row max, the sum of ``exp(x - max)`` and stores ``exp(x - max) / sum`` in
bf16. Both masks broadcast over the heads (and the causal mask over the
batch): Inductor's flat row index ``x`` reads the causal mask at ``x % Q``
and the attention mask at ``(x // (H*Q), x % Q)``; the port tiles
``[B, H, Q]`` and broadcasts the two loads instead. The output
``[B, H, Q, K]`` is returned as the contiguous view ``[B*H, Q, K]`` (the
graph's ``expand`` is a no-op).
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(static_shapes=True)
def causal_masked_softmax(
    scores: torch.Tensor, causal: torch.Tensor, attn_mask: torch.Tensor
) -> torch.Tensor:
    # scores: [B, H, Q, K] fp32, a view of the [B*H, Q, K] input; causal:
    # [Q, K] bool, the strided slice of the causal buffer; attn_mask:
    # [B, Q, K] bf16, a view of the [B, 1, Q, K] input.
    batch, heads, q_len, k_len = scores.shape
    out = torch.empty(
        [batch, heads, q_len, k_len], dtype=torch.bfloat16, device=scores.device
    )
    for tile_b, tile_h, tile_q in hl.tile([batch, heads, q_len]):
        keep = causal[tile_q, :][None, None, :, :]
        # where(slice_2, view, full([], -3.4028234663852886e+38)): the fp32 min.
        x = torch.where(keep, scores[tile_b, tile_h, tile_q, :], -3.4028234663852886e38)
        x = x + attn_mask[tile_b, tile_q, :].to(torch.float32)[:, None, :, :]
        row_max = torch.amax(x, dim=-1, keepdim=True)
        unnormalized = torch.exp(x - row_max)
        row_sum = torch.sum(unnormalized, dim=-1, keepdim=True)
        probs = unnormalized / row_sum
        out[tile_b, tile_h, tile_q, :] = probs.to(out.dtype)
    return out


def forward(inputs: list) -> tuple[torch.Tensor, ...]:
    causal, scores, attn_mask = inputs[:3]
    view_shape = inputs[3]  # _shape_param_0 = [B, H, Q, K]
    out_shape = inputs[5]  # _shape_param_2 = [B*H, Q, K]
    batch, _, q_len, k_len = view_shape
    # slice_1, slice_2: the first Q rows and K columns of the causal buffer,
    # whose two leading dims (like the attention mask's head dim) have size 1.
    causal = causal.view(causal.shape[-2:])[:q_len, :k_len]
    out = causal_masked_softmax(
        scores.view(view_shape), causal, attn_mask.view(batch, q_len, k_len)
    )
    return (out.view(out_shape),)
