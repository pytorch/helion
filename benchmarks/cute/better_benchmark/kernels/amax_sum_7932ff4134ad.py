"""better-benchmark repro amax_sum_7932ff4134ad (hf T5 / T5Small inference).

The T5 encoder self-attention softmax with its bidirectional
relative-position bias built in the graph. Inductor emits two kernels:

- ``triton_poi_fused_abs_add_convert_element_type_div_embedding_full_gt_iota_log_lt_minimum_mul_permute_sub_unsqueeze_where_0``,
  a pointwise kernel over the ``B*Q*K*H`` elements of the returned bf16 bias
  ``add_4`` ``[B, H, Q, K]`` (heads innermost in memory, strides
  ``[H*Q*K, 1, K*H, H]``). Per element it computes the signed offset
  ``k - q`` from two ``iota``s and its T5 bucket: 16 for the keys after the
  query, plus the distance ``|k - q|`` itself below 8, else
  ``min(int(log(distance / 8) / log(16) * 8) + 8, 15)``. It gathers that row
  of the bf16 embedding table ``arg1_1`` ``[32, H]`` and adds the graph's
  ``full(0.0)`` in fp32. The table is the kernel's only load.
- ``triton_red_fused_add_convert_element_type_div_exp_prepare_softmax_online_sub_view_1``,
  a looped reduction: the bf16 scores ``arg0_1`` ``[B*H, Q, K]`` (viewed as
  ``[B, H, Q, K]``) plus the bias buffer are added in fp32, and a two-pass
  online softmax runs over K blocks of one size (both loops share an
  ``hl.register_block_size``, like Inductor's ``R0_BLOCK``): the first loop
  keeps running (max, sum of ``exp(x - max)``) pairs, which Inductor holds
  per lane of the block (``online_softmax_combine`` on every element, then
  ``online_softmax_reduce`` across the lanes after the loop) and the port per
  row (each block's max, then the sum rescaled to it: one ``exp`` per element
  either way); the second loop reloads both inputs and stores
  ``exp(x - max) / sum`` in bf16. The output ``[B, H, Q, K]`` is returned as
  the contiguous view ``[B*H, Q, K]`` (the graph's ``expand`` is a no-op).

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
def relative_position_bias(rel_bias: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    # rel_bias: [32, H] bf16 embedding table; out: [B, Q, K, H] bf16, written
    # in place: the memory-order view of the returned [B, H, Q, K] bias.
    batch, q_len, k_len, heads = out.shape
    for tile_b, tile_q, tile_k, tile_h in hl.tile([batch, q_len, k_len, heads]):
        q = tile_q.index[:, None]
        k = tile_k.index[None, :]
        # sub(iota[None, :], iota[:, None]): the key's signed offset from the
        # query (the graph's two integer "+ 0"s are folded by Inductor too).
        offset = k - q
        # mul(convert_element_type(gt(sub, 0), int64), 16): the keys after the
        # query use the upper half of the table.
        side = (offset > 0).to(torch.int64) * 16
        distance = torch.abs(offset)
        # where(distance < 8, distance, minimum(log-spaced bucket, full(15)))
        log_bucket = (
            torch.log(distance.to(torch.float32) / 8) / 2.772588722239781 * 8
        ).to(torch.int64) + 8
        log_bucket = torch.minimum(log_bucket, torch.full_like(log_bucket, 15))
        bucket = side + torch.where(distance < 8, distance, log_bucket)
        # embedding(arg1_1, bucket): [tile_q, tile_k, tile_h], already in the
        # permuted output's memory order.
        bias = rel_bias[bucket, tile_h].to(torch.float32)
        # add(unsqueeze(permute(embedding)), full([B, 1, Q, K], 0.0))
        value = bias + 0.0
        out[tile_b, tile_q, tile_k, tile_h] = value[None, :, :, :].to(out.dtype)
    return out


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
    scores, rel_bias = inputs[:2]
    view_shape = inputs[2]  # _shape_param_0 = [B, H, Q, K]
    q_len, k_len = inputs[3]  # _shape_param_1 = [Q, K]
    batch = inputs[4][0]  # _shape_param_2 = [B, 1, Q, K]
    out_shape = inputs[6]  # _shape_param_4 = [B*H, Q, K]
    heads = rel_bias.shape[1]
    storage = torch.empty(
        [batch, q_len, k_len, heads], dtype=rel_bias.dtype, device=rel_bias.device
    )
    # add_4: [B, H, Q, K] with heads innermost in memory.
    add_4 = relative_position_bias(rel_bias, storage).permute(0, 3, 1, 2)
    out = bias_add_online_softmax(scores.view(view_shape), add_4)
    return add_4, out.view(out_shape)
