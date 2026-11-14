"""better-benchmark repro amax_sum_sum_42988b64e7f9 (hf/infer/DistillGPT2).

GPT-2's causal-LM loss in the shifted-label form: the int64 labels ``[B, S]``
are padded with -100 on the right and shifted by one (``labels[b, s + 1]``,
-100 at the last position), and ``cross_entropy`` with ``ignore_index=-100``
and mean reduction runs over all ``B * S`` rows of the bf16 logits
``[B, S, V]``, whose row stride is padded (``V`` rounded up to a multiple of
8). Inductor emits two looped reductions:

- ``triton_red_fused_convert_element_type_prepare_softmax_online_view_0``: an
  online softmax over the vocabulary of each row, one pass over V blocks that
  folds each block into the running row (max, sum) the way Inductor's
  ``online_softmax_reduce_scalar_combine`` does: reduce the block to its own
  max and its sum of ``exp(x - block_max)``, then combine with the running
  state, rescaling whichever side has the smaller max by
  ``exp(-|max_a - max_b|)``. The final state is stored to two fp32
  ``[B * S]`` buffers.
- ``triton_red_fused_clone_constant_pad_nd_convert_element_type_div_full_gather_log_ne_neg_slice_squeeze_sub_sum_unsqueeze_view_where_1``:
  one program looping over the ``B * S`` rows: the shifted label, a gather of
  the target's logit, ``-((x_t - max) - log(sum))`` for the non-ignored rows
  summed in fp32 together with the int64 count of non-ignored rows, and the
  quotient stored as the 0-d fp32 loss.

The gather reads row ``r``'s target logit at ``r * row_stride + target`` of a
1-D view of the logits' storage, the flat offset Inductor's kernel computes.
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(static_shapes=True)
def online_softmax_stats(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # logits: [B * S, V] view of the bf16 [B, S, V] input (padded row stride).
    rows, vocab = logits.shape
    row_max = torch.empty([rows], dtype=torch.float32, device=logits.device)
    row_sum = torch.empty([rows], dtype=torch.float32, device=logits.device)
    for tile_r in hl.tile(rows):
        run_max = hl.full([tile_r], float("-inf"), dtype=torch.float32)
        run_sum = hl.zeros([tile_r], dtype=torch.float32)
        for tile_v in hl.tile(vocab):
            x = logits[tile_r, tile_v].to(torch.float32)
            # online_softmax_reduce: the block's max and sum of exp(x - max);
            # an all -inf block uses 0 instead of -inf - -inf = nan.
            block_max = torch.amax(x, dim=-1)
            delta = torch.where(
                block_max[:, None] == float("-inf"), 0.0, x - block_max[:, None]
            )
            block_sum = torch.sum(torch.exp(delta), dim=-1)
            # online_softmax_combine_with_sum
            new_max = torch.maximum(run_max, block_max)
            scale = torch.exp(-torch.abs(run_max - block_max))
            run_scale = torch.where(run_max == new_max, 1.0, scale)
            block_scale = torch.where(block_max == new_max, 1.0, scale)
            new_sum = run_sum * run_scale + block_sum * block_scale
            run_sum = torch.where(new_max == float("inf"), float("nan"), new_sum)
            run_max = new_max
        row_max[tile_r] = run_max
        row_sum[tile_r] = run_sum
    return row_max, row_sum


@helion.kernel(static_shapes=True)
def shifted_nll_loss_mean(
    flat_labels: torch.Tensor,
    flat_logits: torch.Tensor,
    row_max: torch.Tensor,
    row_sum: torch.Tensor,
    seq: hl.constexpr,
    row_stride: hl.constexpr,
) -> torch.Tensor:
    # flat_labels: the int64 [B, S] labels viewed as [B * S]; flat_logits: the
    # bf16 logits' storage viewed as 1-D (row r starts at r * row_stride);
    # row_max, row_sum: the online softmax statistics, [B * S] fp32.
    loss = torch.empty([], dtype=torch.float32, device=row_max.device)
    for _ in hl.grid(1):
        stat_max = row_max[:]
        stat_sum = row_sum[:]
        row = hl.arange(stat_max.size(0))
        # constant_pad_nd(labels, [0, 1], -100)[:, 1:] at row = b * seq + s:
        # labels[b, s + 1] (flat row + 1), or -100 at the last position.
        has_next = row % seq + 1 < seq
        next_label = hl.load(flat_labels, [row + 1], extra_mask=has_next)
        target = torch.where(has_next, next_label, -100)
        valid = target != -100
        index = torch.where(valid, target, 0)
        picked = hl.load(flat_logits, [row * row_stride + index]).to(torch.float32)
        nll = -((picked - stat_max) - torch.log(stat_sum))
        total = torch.sum(torch.where(valid, nll, 0.0))
        count = torch.sum(valid.to(torch.int64))
        loss[()] = total / count.to(torch.float32)
    return loss


def forward(inputs: list) -> tuple[torch.Tensor, ...]:
    labels, logits = inputs[:2]
    view_shape = inputs[2]  # _shape_param_0 = [-1, V]
    batch, seq = labels.shape
    vocab = logits.size(-1)
    logits_rows = logits.view(view_shape)
    if logits_rows.stride(1) != 1:
        # The gather below reads row r's element v at r * row_stride + v.
        raise ValueError(f"logits rows of stride {logits_rows.stride()}")
    row_stride = logits_rows.stride(0)
    row_max, row_sum = online_softmax_stats(logits_rows)
    flat_logits = logits.as_strided(
        [(batch * seq - 1) * row_stride + vocab], [1], logits.storage_offset()
    )
    loss = shifted_nll_loss_mean(
        labels.view(batch * seq), flat_logits, row_max, row_sum, seq, row_stride
    )
    return (loss,)
