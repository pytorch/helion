"""better-benchmark repro amax_sum_sum_81caaa5dd022 (hf/train/PLBartForCausalLM).

PLBart's causal-LM loss in training: cross entropy with ignore index -100 over
the bf16 logits ``[B*T, V+3]`` sliced to the vocabulary ``V`` (the returned
``[B, T, V]`` view keeps the padded row stride), averaged over the non-ignored
labels. Besides the loss, the graph returns what the backward pass needs: the
row ``amax`` and the row ``log(sum(exp(x - amax)))`` of the logits (both fp32
``[B*T, 1]``) and the fp32 count of non-ignored labels. Inductor emits two
looped reductions:

- ``triton_red_fused_convert_element_type_log_prepare_softmax_online_slice_view_0``:
  per row, one pass over V blocks of the sliced logits in fp32 running an
  online softmax (``prepare_softmax_online``) that folds each block into the
  running row (max, sum) the way Inductor's
  ``online_softmax_reduce_scalar_combine`` does: reduce the block to its own
  max and its sum of ``exp(x - block_max)``, then combine with the running
  state, rescaling whichever side has the smaller max by
  ``exp(-|max_a - max_b|)``. The final max is stored as the ``amax`` output
  and the log of the final sum as the ``log`` output.
- ``triton_red_fused_convert_element_type_div_full_gather_ne_neg_slice_squeeze_sub_sum_unsqueeze_view_where_1``:
  one program reducing over all ``B*T`` rows: each row's logit gathered at its
  label (``where(label != -100, label, 0)``), ``-((x_t - max) - log_sum)``
  from the two statistics (like Inductor, the log-softmax stays fp32 instead
  of being rounded to bf16 before the gather), ignored rows zeroed, the losses
  summed in fp32 and the non-ignored labels counted in int64. The count
  converted to fp32 and ``loss_sum / count`` are stored as the two 0-d fp32
  outputs.

The gather indexes the sliced logits with two ``[1, B*T]`` index tensors (the
row and its target), which broadcast together to one element per row.
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(static_shapes=True)
def online_log_softmax_stats(
    logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # logits: [B*T, V] bf16, the sliced view of the padded [B*T, V+3] logits.
    rows, vocab = logits.shape
    row_max = torch.empty([rows], dtype=torch.float32, device=logits.device)
    row_log_sum = torch.empty([rows], dtype=torch.float32, device=logits.device)
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
        row_log_sum[tile_r] = torch.log(run_sum)
    return row_max, row_log_sum


@helion.kernel(static_shapes=True)
def masked_nll_count_mean(
    labels: torch.Tensor,
    logits: torch.Tensor,
    row_max: torch.Tensor,
    row_log_sum: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # labels (int64), row_max, row_log_sum (fp32): [1, B*T] views; logits: the
    # [B*T, V] sliced bf16 logits.
    one = labels.size(0)
    count_out = torch.empty([one], dtype=torch.float32, device=logits.device)
    loss = torch.empty([one], dtype=torch.float32, device=logits.device)
    for tile_o in hl.tile(one):
        label = labels[tile_o, :]
        valid = label != -100
        target = torch.where(valid, label, 0)
        # Row r's logit at its target: the [1, B*T] row and target index
        # tensors broadcast together (a per-row gather, not a product).
        row = hl.arange(label.size(1))
        picked = logits[row[None, :], target].to(torch.float32)
        logp = (picked - row_max[tile_o, :]) - row_log_sum[tile_o, :]
        nll = torch.where(valid, -logp, 0.0)
        total = torch.sum(nll, dim=-1)
        count = torch.sum(valid.to(torch.int64), dim=-1).to(torch.float32)
        count_out[tile_o] = count
        loss[tile_o] = total / count
    return count_out, loss


def forward(inputs: list) -> tuple[torch.Tensor, ...]:
    padded_logits, labels = inputs[:2]
    view_shape = inputs[2]  # _shape_param_0 = [B, T, V]
    matrix_shape = inputs[3]  # _shape_param_1 = [B*T, V]
    logits_view = padded_logits[:, :-3].view(view_shape)
    logits = logits_view.view(matrix_shape)
    rows = logits.size(0)
    row_max, row_log_sum = online_log_softmax_stats(logits)
    count, loss = masked_nll_count_mean(
        labels.view(1, rows),
        logits,
        row_max.view(1, rows),
        row_log_sum.view(1, rows),
    )
    return (
        logits_view,
        row_max.view(rows, 1),
        row_log_sum.view(rows, 1),
        count.view(()),
        loss.view(()),
    )
