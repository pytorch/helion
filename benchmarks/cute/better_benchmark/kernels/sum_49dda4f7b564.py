"""better-benchmark repro sum_49dda4f7b564 (hf XGLMForCausalLM training).

Inductor emits one looped reduction,
``triton_red_fused_add_clone_convert_element_type_div_eq_exp_expand_full_iota_mul_ne_scalar_tensor_slice_sub_sum_unsqueeze_view_where_0``:
the cross-entropy (``log_softmax`` + ``nll_loss``) backward over the vocab,
accumulated into an existing bf16 logits gradient. Each of the ``B*S`` rows
takes its shifted label ``labels[b, s + 1]`` (read straight from the
``[B, S + 1]`` labels at ``b * (S + 1) + s + 1``; ignore index -100 maps to 0)
and the upstream scale ``total_loss / total_weight`` (zeroed for ignored
rows). The NLL gradient ``where(label == iota(V), -1.0, 0.0) * scale`` is pure
index math: the first loop over the vocab sums it per row without loading
anything, and the second recomputes it, reloads the bf16 logits and the bf16
incoming gradient, and stores
``grad + (nll_grad - exp(logits - max - log_sum_exp) * sum)`` in bf16 (fp32
math throughout, as Inductor, so the gradient is not rounded to bf16 before the
add). The ``[B*S, V]`` result is returned with its transposed view.
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(static_shapes=True)
def cross_entropy_bwd_accumulate(
    total_loss: torch.Tensor,
    total_weight: torch.Tensor,
    labels: torch.Tensor,
    logits: torch.Tensor,
    row_max: torch.Tensor,
    log_sum_exp: torch.Tensor,
    grad: torch.Tensor,
) -> torch.Tensor:
    # total_loss, total_weight: 0-d fp32; labels: [B, S + 1] int64;
    # logits, grad: [B*S, V] bf16 views; row_max, log_sum_exp: [B*S, 1] fp32.
    rows, vocab = logits.shape
    label_stride = labels.size(1)
    seq = label_stride - 1
    flat_labels = labels.view(-1)
    out = torch.empty([rows, vocab], dtype=grad.dtype, device=grad.device)
    for tile_r in hl.tile(rows):
        row = tile_r.index
        label = flat_labels[row // seq * label_stride + row % seq + 1]
        valid = label != -100
        target = torch.where(valid, label, 0)
        scale = torch.where(valid, total_loss[()] / total_weight[()], 0.0)
        row_shift = row_max[tile_r, 0]
        row_lse = log_sum_exp[tile_r, 0]
        logit_row = logits[tile_r, :]
        # The vocab iota over the row's reduction dim (it takes each chunk's
        # positions when the reduction is looped).
        cols = hl.arange(logit_row.size(1), dtype=torch.int64)
        nll_grad = (
            torch.where(target[:, None] == cols[None, :], -1.0, 0.0) * scale[:, None]
        )
        nll_sum = torch.sum(nll_grad, dim=-1, keepdim=True)
        shifted = logit_row.to(torch.float32) - row_shift[:, None] - row_lse[:, None]
        y = nll_grad - torch.exp(shifted) * nll_sum
        out[tile_r, :] = (grad[tile_r, :].to(torch.float32) + y).to(out.dtype)
    return out


def forward(inputs: list) -> tuple[torch.Tensor, ...]:
    total_loss, total_weight, labels, logits, row_max, log_sum_exp, grad = inputs[:7]
    logits_shape = inputs[9]  # _shape_param_2 = [-1, V]
    out_shape = inputs[11]  # _shape_param_4 = [B*S, V]
    out = cross_entropy_bwd_accumulate(
        total_loss,
        total_weight,
        labels,
        logits.view(logits_shape),
        row_max,
        log_sum_exp,
        grad.view(out_shape),
    )
    return out, out.permute(1, 0)
