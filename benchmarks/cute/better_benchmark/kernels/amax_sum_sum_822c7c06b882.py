"""better-benchmark repro amax_sum_sum_822c7c06b882 (hf/infer/BartForCausalLM).

The causal-LM loss of BART-style decoders (MBart and Longformer share this
shape point): cross entropy with ignore index -100 over the bf16 logits
``[B*T, V+7]`` sliced to the vocabulary ``V`` (the returned ``[B, T, V]`` view
keeps the padded row stride), averaged over the non-ignored labels. Inductor
emits two kernels:

- ``triton_red_fused_convert_element_type_prepare_softmax_online_slice_view_0``:
  per row, one looped pass over V blocks of the sliced logits in fp32 running
  an online softmax (``prepare_softmax_online``) that folds each block into
  the running row (max, sum) the way Inductor's
  ``online_softmax_reduce_scalar_combine`` does: reduce the block to its own
  max and its sum of ``exp(x - block_max)``, then combine with the running
  state, rescaling whichever side has the smaller max by
  ``exp(-|max_a - max_b|)``. The final state is stored to two fp32 ``[B*T]``
  workspaces.
- ``triton_red_fused_convert_element_type_div_full_gather_log_ne_neg_slice_squeeze_sub_sum_unsqueeze_view_where_1``:
  one program reducing over all ``B*T`` rows: each row's logit gathered at its
  label (``where(label != -100, label, 0)``), ``-((x_t - max) - log(sum))``
  (like Inductor, the log-softmax stays fp32 instead of being rounded to bf16
  before the gather), ignored rows zeroed, the losses summed in fp32 and the
  non-ignored labels counted in int64, and ``loss_sum / count`` stored as the
  0-d bf16 loss.

The gather indexes the sliced logits with two ``[1, B*T]`` index tensors (the
row and its target), which broadcast together to one element per row.
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(static_shapes=True)
def online_softmax_stats(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # logits: [B*T, V] bf16, the sliced view of the padded [B*T, V+7] logits.
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
def masked_nll_mean(
    labels: torch.Tensor,
    logits: torch.Tensor,
    row_max: torch.Tensor,
    row_sum: torch.Tensor,
) -> torch.Tensor:
    # labels (int64), row_max, row_sum (fp32): [1, B*T] views; logits: the
    # [B*T, V] sliced bf16 logits.
    one = labels.size(0)
    loss = torch.empty([one], dtype=logits.dtype, device=logits.device)
    for tile_o in hl.tile(one):
        label = labels[tile_o, :]
        valid = label != -100
        target = torch.where(valid, label, 0)
        row = hl.arange(label.size(1))
        picked = logits[row[None, :], target].to(torch.float32)
        logp = (picked - row_max[tile_o, :]) - torch.log(row_sum[tile_o, :])
        nll = torch.where(valid, -logp, 0.0)
        total = torch.sum(nll, dim=-1)
        count = torch.sum(valid.to(torch.int64), dim=-1)
        loss[tile_o] = (total / count.to(torch.float32)).to(loss.dtype)
    return loss


def forward(inputs: list) -> tuple[torch.Tensor, ...]:
    labels, padded_logits = inputs[:2]
    view_shape = inputs[2]  # _shape_param_0 = [B, T, V]
    matrix_shape = inputs[3]  # _shape_param_1 = [B*T, V]
    logits_view = padded_logits[:, :-7].view(view_shape)
    logits = logits_view.view(matrix_shape)
    rows = logits.size(0)
    row_max, row_sum = online_softmax_stats(logits)
    loss = masked_nll_mean(
        labels.view(1, rows), logits, row_max.view(1, rows), row_sum.view(1, rows)
    )
    return logits_view, loss.view(())
