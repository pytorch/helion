"""better-benchmark repro amax_sum_sum_f58f591dd576 (hf/train/XGLMForCausalLM).

The causal-LM loss of XGLM, OPT and MegatronBert in training, in the
shifted-label form: the int64 labels ``[B, S]`` are padded with -100 on the
right and shifted by one (``padded[b, s + 1]``), and ``cross_entropy`` with
``ignore_index=-100`` and mean reduction runs over all ``B * S`` rows of the
bf16 logits ``[B * S, V]``, which are also returned as a ``[B, S, V]`` view.
Besides the loss, the graph returns what the backward pass needs: the padded
labels (int64 ``[B, S + 1]``), the row ``amax`` and the row
``log(sum(exp(x - amax)))`` of the logits (both fp32 ``[B * S, 1]``) and the
fp32 count of non-ignored labels. Inductor emits three kernels:

- ``triton_red_fused_convert_element_type_log_prepare_softmax_online_view_0``:
  per row, one pass over V blocks of the logits in fp32 running an online
  softmax (``prepare_softmax_online``) that folds each block into the running
  row (max, sum) the way Inductor's ``online_softmax_reduce_scalar_combine``
  does: reduce the block to its own max and its sum of ``exp(x - block_max)``,
  then combine with the running state, rescaling whichever side has the
  smaller max by ``exp(-|max_a - max_b|)``. The final max is stored as the
  ``amax`` output and the log of the final sum as the ``log`` output.
- ``triton_poi_fused_constant_pad_nd_1``: a pointwise kernel over the
  ``[B, S + 1]`` padded labels, a masked load of the label for the first S
  columns and -100 in the padded one.
- ``triton_red_fused_clone_convert_element_type_div_full_gather_ne_neg_slice_squeeze_sub_sum_unsqueeze_view_where_2``:
  one program reducing over all ``B * S`` rows: the shifted label read back
  from the padded buffer (``padded[r // S, 1 + r % S]``), the row's logit
  gathered at it (``where(label != -100, label, 0)``),
  ``-((x_t - max) - log_sum)`` from the two statistics (like Inductor, the
  log-softmax stays fp32 instead of being rounded to bf16 before the
  gather), ignored rows zeroed, the losses summed in fp32 and the non-ignored
  labels counted in int64. The count converted to fp32 and
  ``loss_sum / count`` are stored as the two 0-d fp32 outputs.

Both gathers of the last kernel read 1-D views of their buffers at the flat
offsets Inductor's kernel computes: the padded labels at
``1 + (S + 1) * (r // S) + r % S`` and row ``r``'s target logit at
``r * row_stride + target`` (two 1-D index tensors would index their
Cartesian product in Helion).
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(static_shapes=True)
def online_log_softmax_stats(
    logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # logits: [B * S, V] bf16.
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
def pad_labels(labels: torch.Tensor) -> torch.Tensor:
    # constant_pad_nd(labels, [0, 1], -100): one ignored label after each row.
    batch, seq = labels.shape
    out = torch.empty([batch, seq + 1], dtype=labels.dtype, device=labels.device)
    for tile_b, tile_s in hl.tile([batch, seq + 1]):
        in_bounds = (tile_s.index < seq)[None, :]
        values = hl.load(labels, [tile_b, tile_s], extra_mask=in_bounds)
        out[tile_b, tile_s] = torch.where(in_bounds, values, -100)
    return out


@helion.kernel(static_shapes=True)
def shifted_nll_count_mean(
    flat_padded: torch.Tensor,
    flat_logits: torch.Tensor,
    row_max: torch.Tensor,
    row_log_sum: torch.Tensor,
    seq: hl.constexpr,
    row_stride: hl.constexpr,
) -> tuple[torch.Tensor, torch.Tensor]:
    # flat_padded: the int64 [B, S + 1] padded labels viewed as [B * (S + 1)];
    # flat_logits: the bf16 logits' storage viewed as 1-D (row r starts at
    # r * row_stride); row_max, row_log_sum: the online softmax statistics,
    # [B * S] fp32.
    count_out = torch.empty([], dtype=torch.float32, device=row_max.device)
    loss = torch.empty([], dtype=torch.float32, device=row_max.device)
    for _ in hl.grid(1):
        stat_max = row_max[:]
        stat_log_sum = row_log_sum[:]
        row = hl.arange(stat_max.size(0))
        # padded[:, 1:] at row = b * seq + s: padded[b, s + 1].
        target = hl.load(flat_padded, [1 + (seq + 1) * (row // seq) + row % seq])
        valid = target != -100
        index = torch.where(valid, target, 0)
        picked = hl.load(flat_logits, [row * row_stride + index]).to(torch.float32)
        nll = -((picked - stat_max) - stat_log_sum)
        total = torch.sum(torch.where(valid, nll, 0.0))
        count = torch.sum(valid.to(torch.int64)).to(torch.float32)
        count_out[()] = count
        loss[()] = total / count
    return count_out, loss


def forward(inputs: list) -> tuple[torch.Tensor, ...]:
    logits, labels = inputs[:2]
    view_shape = inputs[2]  # _shape_param_0 = [B, S, V]
    matrix_shape = inputs[3]  # _shape_param_1 = [-1, V]
    batch, seq = labels.shape
    logits_view = logits.view(view_shape)
    logits_rows = logits_view.view(matrix_shape)
    rows, vocab = logits_rows.shape
    if logits_rows.stride(1) != 1:
        # The gather below reads row r's element v at r * row_stride + v.
        raise ValueError(f"logits rows of stride {logits_rows.stride()}")
    row_stride = logits_rows.stride(0)
    row_max, row_log_sum = online_log_softmax_stats(logits_rows)
    padded = pad_labels(labels)
    flat_logits = logits_rows.as_strided(
        [(rows - 1) * row_stride + vocab], [1], logits_rows.storage_offset()
    )
    count, loss = shifted_nll_count_mean(
        padded.view(batch * (seq + 1)),
        flat_logits,
        row_max,
        row_log_sum,
        seq,
        row_stride,
    )
    return (
        logits_view,
        padded,
        row_max.view(rows, 1),
        row_log_sum.view(rows, 1),
        count,
        loss,
    )
