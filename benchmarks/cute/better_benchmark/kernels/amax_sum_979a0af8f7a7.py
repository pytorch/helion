"""better-benchmark repro amax_sum_979a0af8f7a7 (hf T5 / MT5 inference).

Inductor emits one looped reduction,
``triton_red_fused_add_convert_element_type_div_exp_prepare_softmax_online_sub_view_0``:
the T5 attention softmax with its additive position bias. The bf16 scores
``[B*H, Q, K]`` (viewed as ``[B, H, Q, K]``) plus the bf16 bias ``[B, H, Q, K]``
(heads innermost in memory, strides ``[H*Q*K, 1, K*H, H]``) are added in fp32,
and a two-pass online softmax runs over K blocks of one size (both loops
share an ``hl.register_block_size``, like Inductor's ``R0_BLOCK``): the first
loop keeps running (max, sum of ``exp(x - max)``) pairs, which Inductor holds
per lane of the block (``online_softmax_combine`` on every element, then
``online_softmax_reduce`` across the lanes after the loop) and the port per
row (each block's max, then the sum rescaled to it: one ``exp`` per element
either way); the second loop reloads both inputs and stores
``exp(x - max) / sum`` in bf16. The output ``[B, H, Q, K]`` is returned as
the contiguous view ``[B*H, Q, K]``.

The per-row pair costs two cross-lane reductions per K block that
Inductor's kernel does not do. Inductor's per-lane pair would be a loop
carry with a dim on the K loop's own block; the CuTe backend holds a loop
carry as one scalar per thread and refuses that unless each thread holds a
single element of the block (``helion/_compiler/cute/lane_merged_carries.py``),
so both backends keep the per-row pair and run one algorithm.
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(static_shapes=True)
def bias_add_online_softmax(scores: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    # scores: [B, H, Q, K], a view of the [B*H, Q, K] input; bias: [B, H, Q, K].
    batch, heads, q_len, k_len = bias.shape
    out = torch.empty(
        [batch, heads, q_len, k_len], dtype=scores.dtype, device=scores.device
    )
    block_k = hl.register_block_size(k_len)
    for tile_b, tile_h, tile_q in hl.tile([batch, heads, q_len]):
        row_max = hl.full([tile_b, tile_h, tile_q], float("-inf"), dtype=torch.float32)
        row_sum = hl.zeros([tile_b, tile_h, tile_q], dtype=torch.float32)
        for tile_k in hl.tile(k_len, block_size=block_k):
            x = scores[tile_b, tile_h, tile_q, tile_k].to(torch.float32) + bias[
                tile_b, tile_h, tile_q, tile_k
            ].to(torch.float32)
            new_max = torch.maximum(row_max, torch.amax(x, dim=-1))
            # A row that is still all -inf shifts by 0 instead of -inf
            # (exp(-inf - -inf) = nan) and keeps a zero sum, where Inductor's
            # combine counts its elements; its probabilities are NaN either
            # way, as in eager.
            shift = torch.where(new_max == float("-inf"), 0.0, new_max)
            row_sum = row_sum * torch.exp(row_max - shift) + torch.sum(
                torch.exp(x - shift[:, :, :, None]), dim=-1
            )
            row_max = new_max
        for tile_k in hl.tile(k_len, block_size=block_k):
            x = scores[tile_b, tile_h, tile_q, tile_k].to(torch.float32) + bias[
                tile_b, tile_h, tile_q, tile_k
            ].to(torch.float32)
            probs = torch.exp(x - row_max[:, :, :, None]) / row_sum[:, :, :, None]
            out[tile_b, tile_h, tile_q, tile_k] = probs.to(out.dtype)
    return out


def forward(inputs: list) -> tuple[torch.Tensor, ...]:
    scores, bias = inputs[:2]
    view_shape = inputs[2]  # _shape_param_0 = [B, H, Q, K]
    out_shape = inputs[4]  # _shape_param_2 = [B*H, Q, K]
    out = bias_add_online_softmax(scores.view(view_shape), bias)
    return (out.view(out_shape),)
