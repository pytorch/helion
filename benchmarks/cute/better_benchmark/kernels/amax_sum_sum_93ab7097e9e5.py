"""better-benchmark repro amax_sum_sum_93ab7097e9e5 (hf/infer/BertForMaskedLM).

BERT's masked-LM loss (also LayoutLMForMaskedLM and YituTechConvBert at this
shape point): cross entropy with ``ignore_index=-100`` and mean reduction over
the bf16 logits ``[B*S, V+6]`` sliced to the vocabulary ``V`` (the returned
``[B, S, V]`` view keeps the padded row stride). Inductor emits two looped
reductions:

- ``triton_red_fused_convert_element_type_prepare_softmax_online_slice_view_0``:
  an online softmax (``prepare_softmax_online``) over the vocabulary of each
  row in fp32, one pass over V blocks that folds each block into the running
  row (max, sum) the way Inductor's ``online_softmax_reduce_scalar_combine``
  does: reduce the block to its own max and its sum of ``exp(x - block_max)``,
  then combine with the running state, rescaling whichever side has the
  smaller max by ``exp(-|max_a - max_b|)``. The final state is stored to two
  fp32 ``[B*S]`` workspaces.
- ``triton_red_fused_convert_element_type_div_full_gather_log_ne_neg_slice_squeeze_sub_sum_unsqueeze_view_where_1``:
  one program reducing over all ``B*S`` rows: gathers each row's logit at its
  label (``where(label != -100, label, 0)``), forms
  ``-((logit - max) - log(sum))`` for the valid rows (like Inductor, the
  log-softmax stays fp32 instead of being rounded to bf16 before the gather),
  sums the losses in fp32 and the valid labels in int64, and stores
  ``loss_sum / count`` as the 0-d bf16 loss.
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(static_shapes=True)
def online_softmax_stats(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # logits: [B*S, V], the sliced view of the padded [B*S, V+6] logits.
    rows, vocab = logits.shape
    row_max = torch.empty([rows], dtype=torch.float32, device=logits.device)
    row_sum = torch.empty([rows], dtype=torch.float32, device=logits.device)
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
        row_sum[tile_r] = running_sum
    return row_max, row_sum


@helion.kernel(static_shapes=True)
def masked_nll_mean(
    labels: torch.Tensor,
    logits: torch.Tensor,
    row_max: torch.Tensor,
    row_sum: torch.Tensor,
) -> torch.Tensor:
    # labels (i64), row_max, row_sum (fp32): [1, B*S] views; logits: [B*S, V].
    one = labels.size(0)
    loss = torch.empty([one], dtype=logits.dtype, device=logits.device)
    for tile_o in hl.tile(one):
        label = labels[tile_o, :]
        valid = label != -100
        target = torch.where(valid, label, 0)
        # Row r's logit at its target: the [1, B*S] row and target index
        # tensors broadcast together (a per-row gather, not a product).
        row = hl.arange(label.size(1))
        picked = logits[row[None, :], target].to(torch.float32)
        logp = (picked - row_max[tile_o, :]) - torch.log(row_sum[tile_o, :])
        nll = torch.where(valid, -logp, 0.0)
        loss_sum = torch.sum(nll, dim=-1)
        count = torch.sum(valid.to(torch.int64), dim=-1)
        loss[tile_o] = (loss_sum / count.to(torch.float32)).to(loss.dtype)
    return loss


def forward(inputs: list) -> tuple[torch.Tensor, ...]:
    labels, padded_logits = inputs[:2]
    view_shape = inputs[2]  # _shape_param_0 = [B, S, V]
    matrix_shape = inputs[3]  # _shape_param_1 = [B*S, V]
    logits_view = padded_logits[:, :-6].view(view_shape)
    logits = logits_view.view(matrix_shape)
    row_max, row_sum = online_softmax_stats(logits)
    rows = logits.size(0)
    loss = masked_nll_mean(
        labels.view(1, rows), logits, row_max.view(1, rows), row_sum.view(1, rows)
    )
    return logits_view, loss.view(())
