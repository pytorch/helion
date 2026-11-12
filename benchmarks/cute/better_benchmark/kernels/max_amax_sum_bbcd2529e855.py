"""better-benchmark repro max_amax_sum_bbcd2529e855 (hf/infer/openai/gpt-oss-20b).

The attention softmax with per-head sinks of gpt-oss, with the causal mask
built in the graph: two ``iota``s give ``where(k <= q, 0, bf16 min)``, a bf16
``[1, 1, Q, K]`` mask that is also returned. The bf16 scores ``arg0_1``
``[H, Q, K]`` are scaled by 0.125 and the mask is added; the sink logit of each
head (``arg1_1`` ``[H]``, expanded to ``[1, H, Q, 1]``) is concatenated as a
K+1-th column, the row is softmaxed (``max``, ``sub``, fp32 ``amax``, ``sub``,
``exp``, ``sum``, ``div``) and the sink column is sliced off again before the
bf16 result is cloned to a contiguous ``[H, Q, K]`` output. Inductor emits
three kernels:

- ``triton_poi_fused_add_expand_full_le_unsqueeze_where_0``, a pointwise
  kernel that writes the bf16 ``[Q, K]`` mask (the returned ``where``); it
  reads nothing.
- ``triton_per_fused_add_cat_convert_element_type_expand_max_mul_prepare_softmax_online_sub_view_1``,
  a persistent reduction over the K+1 concatenated columns of each of the
  ``H*Q`` rows that reads the scores, the mask buffer and the sinks. It
  computes the scaled-and-masked scores in fp32 (no bf16 rounding of the
  graph's ``mul``/``add``/``sub``), the row ``max`` (stored as a bf16
  ``[H, Q]`` buffer), ``amax(x - max)`` and ``sum(exp(x - max - amax))`` (both
  from the unrounded fp32 max, stored as fp32 ``[H, Q]`` buffers).
- ``triton_poi_fused_add_cat_clone_convert_element_type_div_exp_expand_mul_slice_sub_view_2``,
  a pointwise kernel over the ``H*Q*K`` sliced outputs that recomputes the
  scaled-and-masked scores from the scores and the mask buffer and stores
  ``exp(x - bf16 max - amax) / sum`` in bf16. Its ``cat`` branch for the sink
  column is a load of ``arg1_1`` masked by ``k >= K``, which never executes
  inside the slice, so the port's third kernel does not take the sinks.

Helion reduces over tensor dimensions, so the second kernel reduces the K
score columns of each row and folds in the sink column as one more element of
each reduction (``max``/``amax`` with ``torch.maximum``, the sum by adding its
``exp`` term); the loads and stores are the same as Inductor's.
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(static_shapes=True)
def causal_mask(mask: torch.Tensor) -> torch.Tensor:
    # mask: [Q, K] bf16, written in place: 0 where k <= q, else bf16 min.
    q_len, k_len = mask.shape
    for tile_q, tile_k in hl.tile([q_len, k_len]):
        keep = tile_k.index[None, :] <= tile_q.index[:, None]
        # full([], 0.0) and full([], -3.3895313892515355e+38), the bf16 min.
        mask[tile_q, tile_k] = torch.where(keep, 0.0, -3.3895313892515355e38).to(
            mask.dtype
        )
    return mask


@helion.kernel(static_shapes=True)
def sink_softmax_stats(
    scores: torch.Tensor, mask: torch.Tensor, sinks: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # scores: [H, Q, K] bf16; mask: [Q, K] bf16; sinks: [H] bf16.
    heads, q_len, k_len = scores.shape
    row_max = torch.empty([heads, q_len], dtype=scores.dtype, device=scores.device)
    row_amax = torch.empty([heads, q_len], dtype=torch.float32, device=scores.device)
    row_sum = torch.empty([heads, q_len], dtype=torch.float32, device=scores.device)
    for tile_h, tile_q in hl.tile([heads, q_len]):
        x = (
            scores[tile_h, tile_q, :].to(torch.float32) * 0.125
            + mask[tile_q, :].to(torch.float32)[None, :, :]
        )
        # The concatenated sink column of each row: [tile_h, 1].
        sink = sinks[tile_h].to(torch.float32)[:, None]
        # max(cat([x, sink], -1), -1)
        peak = torch.maximum(torch.amax(x, dim=-1), sink)
        row_max[tile_h, tile_q] = peak.to(row_max.dtype)
        # sub(cat, max) and its amax
        shifted = x - peak[:, :, None]
        shifted_sink = sink - peak
        top = torch.maximum(torch.amax(shifted, dim=-1), shifted_sink)
        row_amax[tile_h, tile_q] = top
        # sum(exp(sub - amax), -1)
        total = torch.sum(torch.exp(shifted - top[:, :, None]), dim=-1) + torch.exp(
            shifted_sink - top
        )
        row_sum[tile_h, tile_q] = total
    return row_max, row_amax, row_sum


@helion.kernel(static_shapes=True)
def sink_softmax_normalize(
    scores: torch.Tensor,
    mask: torch.Tensor,
    row_max: torch.Tensor,
    row_amax: torch.Tensor,
    row_sum: torch.Tensor,
) -> torch.Tensor:
    # scores: [H, Q, K] bf16; mask: [Q, K] bf16; row_max: [H, Q] bf16;
    # row_amax, row_sum: [H, Q] fp32.
    heads, q_len, k_len = scores.shape
    out = torch.empty([heads, q_len, k_len], dtype=scores.dtype, device=scores.device)
    for tile_h, tile_q, tile_k in hl.tile([heads, q_len, k_len]):
        x = (
            scores[tile_h, tile_q, tile_k].to(torch.float32) * 0.125
            + mask[tile_q, tile_k].to(torch.float32)[None, :, :]
        )
        shifted = x - row_max[tile_h, tile_q].to(torch.float32)[:, :, None]
        e = torch.exp(shifted - row_amax[tile_h, tile_q][:, :, None])
        probs = e / row_sum[tile_h, tile_q][:, :, None]
        out[tile_h, tile_q, tile_k] = probs.to(out.dtype)
    return out


def forward(inputs: list) -> tuple[torch.Tensor, ...]:
    arg0_1, arg1_1 = inputs[:2]
    out_shape = inputs[6]  # _shape_param_4 = [H, Q, K]
    heads, q_len, k_len = arg0_1.shape
    where = torch.empty(
        [1, 1, q_len, k_len], dtype=torch.bfloat16, device=arg0_1.device
    )
    mask = causal_mask(where.view(q_len, k_len))
    row_max, row_amax, row_sum = sink_softmax_stats(arg0_1, mask, arg1_1)
    out = sink_softmax_normalize(arg0_1, mask, row_max, row_amax, row_sum)
    return where, out.view(out_shape)
