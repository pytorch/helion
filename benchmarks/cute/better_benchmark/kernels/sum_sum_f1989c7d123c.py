"""better-benchmark repro sum_sum_f1989c7d123c (hf/train/GoogleFnet).

The masked-LM head's cross-entropy (``log_softmax`` + ``nll_loss``) backward
of FNet training, accumulated into an existing logits gradient. The summed
gradient is returned as ``[B*S, V]`` (``V = 32000``), as its transposed view
``[V, B*S]`` and as its column sums over the rows (the LM-head bias
gradient); everything is fp32. Inductor emits two kernels:

- ``triton_red_fused_add_div_eq_exp_expand_full_iota_mul_ne_scalar_tensor_sub_sum_unsqueeze_view_where_0``:
  a looped reduction over the vocabulary per token row. Each of the ``B*S``
  rows takes its label (ignore index -100 maps to 0) and the upstream scale
  ``total_loss / total_weight`` (zeroed for ignored rows). The NLL gradient
  ``where(label == iota(V), -1.0, 0.0) * scale`` is pure index math: the
  first loop over the vocab sums it per row without loading anything, and the
  second recomputes it, loads the logits and the incoming gradient, and
  stores ``grad + (nll_grad - exp(logits - max - log_sum_exp) * sum)`` to the
  ``[B, S, V]`` gradient, the buffer both gradient outputs view.
- ``triton_red_fused_add_div_eq_exp_expand_full_iota_mul_ne_scalar_tensor_sub_sum_unsqueeze_view_where_1``:
  a looped reduction over the rows per vocabulary column, the column sums of
  that gradient.

This is sum_sum_85ee747617a6's graph (BERT masked-LM) without the bf16 casts
and without the padded copy of the gradient.

The class iota is sized from the logits row (``hl.arange(x.size(1))``) so it
shares the row's reduction dim and takes each chunk's positions when the
reduction is looped.
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(static_shapes=True)
def cross_entropy_bwd_accumulate(
    labels: torch.Tensor,
    total_loss: torch.Tensor,
    total_weight: torch.Tensor,
    grad: torch.Tensor,
    logits: torch.Tensor,
    row_max: torch.Tensor,
    log_sum_exp: torch.Tensor,
) -> torch.Tensor:
    # labels: [B*S] int64; total_loss, total_weight: 0-d fp32; grad, logits:
    # [B*S, V] fp32; row_max, log_sum_exp: [B*S, 1] fp32.
    rows, vocab = logits.shape
    out = torch.empty([rows, vocab], dtype=grad.dtype, device=grad.device)
    for tile_r in hl.tile(rows):
        label = labels[tile_r]
        valid = label != -100
        target = torch.where(valid, label, 0)
        scale = torch.where(valid, total_loss[()] / total_weight[()], 0.0)
        x = logits[tile_r, :]
        cols = hl.arange(x.size(1), dtype=torch.int64)
        nll_grad = (
            torch.where(target[:, None] == cols[None, :], -1.0, 0.0) * scale[:, None]
        )
        nll_sum = torch.sum(nll_grad, dim=-1, keepdim=True)
        shifted = x - row_max[tile_r, 0][:, None] - log_sum_exp[tile_r, 0][:, None]
        y = nll_grad - torch.exp(shifted) * nll_sum
        out[tile_r, :] = grad[tile_r, :] + y
    return out


@helion.kernel(static_shapes=True)
def column_sum(grad: torch.Tensor) -> torch.Tensor:
    # grad: [B*S, V] fp32; returns its per-column sum over the rows, [V].
    rows, vocab = grad.shape
    out = torch.empty([vocab], dtype=grad.dtype, device=grad.device)
    for tile_v in hl.tile(vocab):
        out[tile_v] = torch.sum(grad[:, tile_v], dim=0)
    return out


def forward(inputs: list) -> tuple[torch.Tensor, ...]:
    total_loss, total_weight, labels, logits, row_max, log_sum_exp, grad = inputs[:7]
    logits_shape = inputs[9]  # _shape_param_2 = [-1, V]
    rows, vocab = inputs[11]  # _shape_param_4 = [B*S, V]
    sum_shape = inputs[12]  # _shape_param_5 = [V]
    grad_sum = cross_entropy_bwd_accumulate(
        labels.view(rows),
        total_loss,
        total_weight,
        grad.view(rows, vocab),
        logits.view(logits_shape),
        row_max,
        log_sum_exp,
    )
    bias_grad = column_sum(grad_sum)
    return grad_sum, grad_sum.permute(1, 0), bias_grad.view(sum_shape)
