"""better-benchmark repro sum_561404c238b3 (hf M2M100ForConditionalGeneration training).

Inductor emits one looped reduction,
``triton_red_fused_add_convert_element_type_div_eq_exp_expand_full_iota_mul_ne_scalar_tensor_sub_sum_unsqueeze_view_where_0``:
the LM head's cross-entropy (``log_softmax`` + ``nll_loss``) backward over the
vocabulary, accumulated into an existing bf16 logits gradient. Each of the
``B*S`` token rows takes its int64 label (ignore index -100 maps to 0) and the
upstream scale ``grad_loss / total_weight`` (zeroed for ignored rows). The NLL
gradient ``grad_logp = where(label == iota(V), -1.0, 0.0) * scale`` is pure
index math: the first loop over the vocabulary sums it per row without loading
any row data, and the second recomputes it, loads the bf16 logits and the bf16
incoming gradient and stores
``grad_in + (grad_logp - exp(logits - row_max - log_sum_exp) * sum(grad_logp))``
in bf16 (``row_max`` and ``log_sum_exp`` are the fp32 ``[B*S, 1]`` log-softmax
shifts). As in Inductor, the graph's bf16 round trips of intermediates (the
NLL gradient, the shifted logits, the log-softmax gradient before the add)
stay in fp32. The ``[B*S, V]`` result is returned with its transposed view.

The port reduces over a full slice, like sum_b2873b4c9052 and
sum_49dda4f7b564: a looped config rolls it into the same two loops, with the
logits and the incoming gradient loaded only in the second one, and the
autotuner may also pick a persistent reduction.
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(static_shapes=True)
def nll_log_softmax_backward_add(
    grad_loss: torch.Tensor,
    total_weight: torch.Tensor,
    labels: torch.Tensor,
    logits: torch.Tensor,
    row_max: torch.Tensor,
    log_sum_exp: torch.Tensor,
    grad_in: torch.Tensor,
) -> torch.Tensor:
    # grad_loss, total_weight: 0-d fp32; labels: [rows] int64; logits, grad_in:
    # [rows, vocab] bf16; row_max, log_sum_exp: [rows, 1] fp32.
    rows, vocab = logits.shape
    out = torch.empty([rows, vocab], dtype=grad_in.dtype, device=grad_in.device)
    for tile_r in hl.tile(rows):
        label = labels[tile_r]
        valid = label != -100
        target = torch.where(valid, label, 0)
        scale = torch.where(valid, grad_loss[()] / total_weight[()], 0.0)
        # The shifts as ``[tile_r]`` loads, read once per row as in Inductor
        # (``[tile_r, :]`` slices are reloaded in every step of the second loop).
        row_shift = row_max[tile_r, 0]
        row_lse = log_sum_exp[tile_r, 0]
        x = logits[tile_r, :].to(torch.float32)
        # Sized from the slice, the iota names the row's reduction dim, so a
        # looped reduction gives it each chunk's positions (``hl.arange(vocab)``
        # of the host int would keep the reduction persistent).
        cols = hl.arange(x.size(1), dtype=torch.int64)
        grad_logp = (
            torch.where(target[:, None] == cols[None, :], -1.0, 0.0) * scale[:, None]
        )
        grad_sum = torch.sum(grad_logp, dim=-1, keepdim=True)
        logp = x - row_shift[:, None] - row_lse[:, None]
        grad_x = grad_logp - torch.exp(logp) * grad_sum
        out[tile_r, :] = (grad_in[tile_r, :].to(torch.float32) + grad_x).to(out.dtype)
    return out


def forward(inputs: list) -> tuple[torch.Tensor, ...]:
    grad_loss, total_weight, labels, logits, row_max, log_sum_exp, grad_in = inputs[:7]
    logits_shape = inputs[9]  # _shape_param_2 = [-1, V]
    out_shape = inputs[11]  # _shape_param_4 = [B*S, V]
    out = nll_log_softmax_backward_add(
        grad_loss,
        total_weight,
        labels.view(-1),
        logits.view(logits_shape),
        row_max,
        log_sum_exp,
        grad_in.view(out_shape),
    )
    return out, out.permute(1, 0)
