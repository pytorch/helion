"""better-benchmark repro amax_sum_sum_671a76fab586 (genai CrossEntropyBackward, static).

The pieces of a cross-entropy loss over bf16 logits ``[M, N]`` with int64
labels ``[M]`` and ignore index -100: the row ``amax`` and the row
``log(sum(exp(x - amax)))`` of the logits in fp32 (both returned as fp32
``[M, 1]``), and the bf16 sum over the non-ignored rows of the negated
log-softmax at each row's label. Inductor emits two looped reductions:

- ``triton_red_fused_convert_element_type_log_prepare_softmax_online_0``: an
  online softmax (``prepare_softmax_online``) over each row in fp32, one pass
  over N blocks that folds each block into the running row (max, sum) the way
  Inductor's ``online_softmax_reduce_scalar_combine`` does: reduce the block
  to its own max and its sum of ``exp(x - block_max)``, then combine with the
  running state, rescaling whichever side has the smaller max by
  ``exp(-|max_a - max_b|)``. The final max is stored as the ``amax`` output
  and the log of the final sum as the ``log`` output.
- ``triton_red_fused_convert_element_type_full_gather_ne_neg_squeeze_sub_sum_unsqueeze_where_1``:
  one program reducing over all ``M`` rows: gathers each row's logit at its
  label (``where(label != -100, label, 0)``), forms
  ``-((logit - max) - log_sum)`` from the two statistics (like Inductor, the
  log-softmax stays fp32 instead of being rounded to bf16 before the
  gather), zeroes the ignored rows, sums the losses in fp32 and stores the
  total as the 0-d bf16 ``sum_2``.
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(static_shapes=True)
def online_log_softmax_stats(
    logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, vocab = logits.shape
    row_max = torch.empty([rows], dtype=torch.float32, device=logits.device)
    row_log_sum = torch.empty([rows], dtype=torch.float32, device=logits.device)
    for tile_r in hl.tile(rows):
        running_max = hl.full([tile_r], float("-inf"), dtype=torch.float32)
        running_sum = hl.zeros([tile_r], dtype=torch.float32)
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
            new_max = torch.maximum(running_max, block_max)
            scale = torch.exp(-torch.abs(running_max - block_max))
            running_scale = torch.where(running_max == new_max, 1.0, scale)
            block_scale = torch.where(block_max == new_max, 1.0, scale)
            new_sum = running_sum * running_scale + block_sum * block_scale
            running_sum = torch.where(new_max == float("inf"), float("nan"), new_sum)
            running_max = new_max
        row_max[tile_r] = running_max
        row_log_sum[tile_r] = torch.log(running_sum)
    return row_max, row_log_sum


@helion.kernel(static_shapes=True)
def masked_nll_sum(
    labels: torch.Tensor,
    logits: torch.Tensor,
    row_max: torch.Tensor,
    row_log_sum: torch.Tensor,
) -> torch.Tensor:
    # labels (i64), row_max, row_log_sum (fp32): [1, M] views; logits: [M, N].
    one = labels.size(0)
    out = torch.empty([one], dtype=logits.dtype, device=logits.device)
    for tile_o in hl.tile(one):
        label = labels[tile_o, :]
        valid = label != -100
        target = torch.where(valid, label, 0)
        # Row r's logit at its target: the [1, M] row and target index
        # tensors broadcast together (a per-row gather, not a product).
        row = hl.arange(label.size(1))
        picked = logits[row[None, :], target].to(torch.float32)
        logp = (picked - row_max[tile_o, :]) - row_log_sum[tile_o, :]
        nll = torch.where(valid, -logp, 0.0)
        out[tile_o] = torch.sum(nll, dim=-1).to(out.dtype)
    return out


def forward(inputs: list) -> tuple[torch.Tensor, ...]:
    logits, labels = inputs[:2]
    rows = logits.size(0)
    row_max, row_log_sum = online_log_softmax_stats(logits)
    total = masked_nll_sum(
        labels.view(1, rows),
        logits,
        row_max.view(1, rows),
        row_log_sum.view(1, rows),
    )
    return row_max.view(rows, 1), row_log_sum.view(rows, 1), total.view(())
