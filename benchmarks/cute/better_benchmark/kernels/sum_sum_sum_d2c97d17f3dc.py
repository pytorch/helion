"""better-benchmark repro sum_sum_sum_d2c97d17f3dc (hf T5ForConditionalGeneration / T5Small training).

T5 training backward for the decoder and the encoder stack: the self-attention
softmax backward (through dropout) of the layer that owns the relative
position bias, fused with the backward of that bias embedding
(``nn.Embedding(num_buckets, H)``), whose ``[num_buckets, H]`` gradient
accumulates the batch-summed position-bias gradients of five other inputs
(the other layers) plus this softmax gradient. Inductor emits six kernel
launches:

- ``triton_red_fused_add_convert_element_type_div_exp_expand_fma_full_le_mul_neg_permute_sub_sum_unsqueeze_view_where_0``:
  decoder self-attention softmax backward over K (looped reduction). The first
  loop recomputes the probabilities ``exp(scores + (bias + mask) - max) / sum``
  (bias ``[Q, K, H]`` permuted to ``[H, Q, K]``, causal mask
  ``where(pos[k] <= pos[q], fill, -3.4028234663852886e38)``), stores them to an
  fp32 ``[B*H, Q, K]`` workspace and sums ``grad * (keep * 1/0.9) * probs``; the
  second loop reloads the probabilities from the workspace and the gradient
  and dropout mask from their inputs and stores
  ``fma(-probs, sum, grad * (keep * 1/0.9) * probs)`` in bf16.
- ``triton_poi_fused__unsafe_masked_index_put_accumulate_full_permute_squeeze_1``:
  zero-fills the fp32 ``[num_buckets, H]`` bias gradient.
- ``triton_per_fused__unsafe_masked_index_put_accumulate_add_convert_element_type_full_permute_squeeze_sum_2``:
  persistent reduction over the batch of the five incoming bias gradients plus
  the softmax gradient (summed in fp32 in graph order); each ``(h, q, k)``
  total is atomically added to ``out[bucket[q, k], h]`` with the bucket
  clamped to ``[-num_buckets, num_buckets - 1]`` and wrapped if negative.
- ``triton_red_fused_add_convert_element_type_div_exp_fma_full_mul_neg_permute_sub_sum_unsqueeze_view_3``:
  encoder self-attention softmax backward (no mask; the bias plus a zero
  ``full``). Its first loop stores ``grad * (keep * 1/0.9) * probs`` to the
  fp32 workspace and sums it; the second loop recomputes the probabilities
  from the scores and bias, reloads the product from the workspace and
  stores ``fma(-probs, sum, product)`` in bf16.
- The zero-fill and the bias-gradient scatter again for the second block.

As in Inductor's code, the scores' bf16 round trip and the bf16 dropout
scaling stay in fp32.
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(static_shapes=True)
def causal_softmax_dropout_bwd(
    grad: torch.Tensor,
    keep: torch.Tensor,
    positions: torch.Tensor,
    fill: torch.Tensor,
    scores: torch.Tensor,
    bias: torch.Tensor,
    row_max: torch.Tensor,
    row_sum: torch.Tensor,
) -> torch.Tensor:
    # grad, keep, scores: [B*H, Q, K]; positions: [K]; fill: 0-d;
    # bias: [H, Q, K] (a permuted view); row_max, row_sum: [B*H, Q].
    bh, q_len, k_len = scores.shape
    heads = bias.size(0)
    probs_ws = torch.empty(
        [bh, q_len, k_len], dtype=torch.float32, device=scores.device
    )
    out = torch.empty([bh, q_len, k_len], dtype=scores.dtype, device=scores.device)
    for tile_bh, tile_q in hl.tile([bh, q_len]):
        head = tile_bh.index % heads
        visible = positions[None, :] <= positions[tile_q][:, None]
        mask_bias = torch.where(visible, fill[()], -3.4028234663852886e38)
        logits = scores[tile_bh, tile_q, :].to(torch.float32) + (
            bias[head, tile_q, :] + mask_bias[None, :, :]
        )
        probs = (
            torch.exp(logits - row_max[tile_bh, tile_q][:, :, None])
            / row_sum[tile_bh, tile_q][:, :, None]
        )
        probs_ws[tile_bh, tile_q, :] = probs
        scaled_grad = grad[tile_bh, tile_q, :].to(torch.float32) * (
            keep[tile_bh, tile_q, :].to(torch.float32) * 1.1111111111111112
        )
        total = torch.sum(scaled_grad * probs, dim=-1, keepdim=True)
        # Like Inductor's second loop: the probabilities come back from the
        # workspace, the scaled gradient from grad and keep.
        probs = probs_ws[tile_bh, tile_q, :]
        out[tile_bh, tile_q, :] = torch.ops.prims.fma(
            -probs, total, scaled_grad * probs
        ).to(out.dtype)
    return out


@helion.kernel(static_shapes=True)
def softmax_dropout_bwd(
    grad: torch.Tensor,
    keep: torch.Tensor,
    scores: torch.Tensor,
    bias: torch.Tensor,
    row_max: torch.Tensor,
    row_sum: torch.Tensor,
) -> torch.Tensor:
    # grad, keep, scores: [B*H, Q, K]; bias: [H, Q, K] (a permuted view);
    # row_max, row_sum: [B*H, Q].
    bh, q_len, k_len = scores.shape
    heads = bias.size(0)
    product_ws = torch.empty(
        [bh, q_len, k_len], dtype=torch.float32, device=scores.device
    )
    out = torch.empty([bh, q_len, k_len], dtype=scores.dtype, device=scores.device)
    for tile_bh, tile_q in hl.tile([bh, q_len]):
        head = tile_bh.index % heads
        scaled_grad = grad[tile_bh, tile_q, :].to(torch.float32) * (
            keep[tile_bh, tile_q, :].to(torch.float32) * 1.1111111111111112
        )
        logits = scores[tile_bh, tile_q, :].to(torch.float32) + (
            bias[head, tile_q, :] + 0.0
        )
        probs = (
            torch.exp(logits - row_max[tile_bh, tile_q][:, :, None])
            / row_sum[tile_bh, tile_q][:, :, None]
        )
        product = scaled_grad * probs
        product_ws[tile_bh, tile_q, :] = product
        total = torch.sum(product, dim=-1, keepdim=True)
        # Like Inductor's second loop: the probabilities are recomputed from
        # scores and bias, the product comes back from the workspace.
        out[tile_bh, tile_q, :] = torch.ops.prims.fma(
            -probs, total, product_ws[tile_bh, tile_q, :]
        ).to(out.dtype)
    return out


@helion.kernel(static_shapes=True)
def zero_fill(out: torch.Tensor) -> torch.Tensor:
    rows, cols = out.shape
    for tile_r, tile_c in hl.tile([rows, cols]):
        out[tile_r, tile_c] = hl.zeros([tile_r, tile_c], dtype=out.dtype)
    return out


@helion.kernel(static_shapes=True)
def relative_bias_grad_scatter(
    grad0: torch.Tensor,
    grad1: torch.Tensor,
    grad2: torch.Tensor,
    grad3: torch.Tensor,
    grad4: torch.Tensor,
    grad_scores: torch.Tensor,
    buckets: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    # grad0..grad4, grad_scores: [B, H, Q, K] bf16; buckets: [Q, K] int64;
    # out: [num_buckets, H] fp32, zero-filled.
    batch, heads, q_len, k_len = grad0.shape
    num_buckets = out.size(0)
    for tile_h, tile_q, tile_k in hl.tile([heads, q_len, k_len]):
        acc = grad0[:, tile_h, tile_q, tile_k].to(torch.float32) + grad1[
            :, tile_h, tile_q, tile_k
        ].to(torch.float32)
        acc = acc + grad2[:, tile_h, tile_q, tile_k].to(torch.float32)
        acc = acc + grad3[:, tile_h, tile_q, tile_k].to(torch.float32)
        acc = acc + grad4[:, tile_h, tile_q, tile_k].to(torch.float32)
        acc = acc + grad_scores[:, tile_h, tile_q, tile_k].to(torch.float32)
        total = torch.sum(acc, dim=0)
        bucket = torch.clamp(buckets[tile_q, tile_k], -num_buckets, num_buckets - 1)
        bucket = torch.where(bucket < 0, bucket + num_buckets, bucket)
        # out[bucket[q, k], h] += total[h, q, k]; unsqueeze rather than
        # tile_h.index[:, None, None], whose slice allocates an unused
        # reduction dim that CuTe maps onto redundant threads.
        head = tile_h.index.unsqueeze(1).unsqueeze(2)
        hl.atomic_add(out, [bucket.unsqueeze(0), head], total)
    return out


def forward(inputs: list) -> tuple[torch.Tensor, ...]:
    dec_grads = inputs[0:5]
    dec_grad_probs, dec_keep, positions, fill, dec_scores = inputs[5:10]
    dec_bias, dec_max, dec_sum, dec_buckets = inputs[10:14]
    enc_grads = inputs[14:19]
    enc_grad_probs, enc_keep, enc_scores, enc_bias, enc_max, enc_sum = inputs[19:25]
    enc_buckets = inputs[25]
    batch, heads, q_len, k_len = inputs[26]  # _shape_param_0
    bias_grad_shape = inputs[33]  # _shape_param_7 = [num_buckets, H]
    bh = batch * heads

    dec_grad_scores = causal_softmax_dropout_bwd(
        dec_grad_probs,
        dec_keep.view(bh, q_len, k_len),
        positions.view(k_len),
        fill,
        dec_scores,
        dec_bias.permute(2, 0, 1),
        dec_max.view(bh, q_len),
        dec_sum.view(bh, q_len),
    )
    dec_bias_grad = zero_fill(
        torch.empty(bias_grad_shape, dtype=torch.float32, device=fill.device)
    )
    relative_bias_grad_scatter(
        *dec_grads,
        dec_grad_scores.view(batch, heads, q_len, k_len),
        dec_buckets,
        dec_bias_grad,
    )
    enc_grad_scores = softmax_dropout_bwd(
        enc_grad_probs,
        enc_keep.view(bh, q_len, k_len),
        enc_scores,
        enc_bias.permute(2, 0, 1),
        enc_max.view(bh, q_len),
        enc_sum.view(bh, q_len),
    )
    enc_bias_grad = zero_fill(
        torch.empty(bias_grad_shape, dtype=torch.float32, device=fill.device)
    )
    relative_bias_grad_scatter(
        *enc_grads,
        enc_grad_scores.view(batch, heads, q_len, k_len),
        enc_buckets,
        enc_bias_grad,
    )
    return dec_grad_scores, dec_bias_grad, enc_grad_scores, enc_bias_grad
