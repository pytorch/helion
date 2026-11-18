"""better-benchmark repro sum_1bd9fae13cde (hf DistillGPT2 training).

Inductor emits one looped reduction,
``triton_red_fused_clone_convert_element_type_div_eq_exp_expand_full_iota_mul_ne_scalar_tensor_slice_sub_sum_unsqueeze_view_where_0``:
the causal-LM cross-entropy (``log_softmax`` + ``nll_loss``) backward over the
vocab. Each of the ``B*S`` rows takes its shifted label ``labels[b, s + 1]``
(ignore index -100 maps to 0) and the upstream scale ``total_loss /
total_weight`` (zeroed for ignored rows). The NLL gradient
``where(label == iota(V), -1.0, 0.0) * scale`` is pure index math: the first
loop over the vocab sums it per row without loading anything, and the second
recomputes it, loads the bf16 logits (their rows padded to a stride of 50264)
and stores ``nll_grad - exp(logits - max - log_sum_exp) * sum`` in bf16 (fp32
math throughout, as Inductor). The ``[B*S, V]`` result is returned as the
contiguous ``[B, S, V]`` logits gradient; the row sum is not an output.

This is sum_0aced6470e1e's graph (OPT) without the incoming gradient that one
accumulates into. Inductor never materializes the graph's contiguous clone of
the shifted label slice; it reads row ``r``'s label at
``1 + row_stride * (r // S) + r % S`` of the ``[B, S + 1]`` labels' storage.
The port reads the same offset of a 1-D view of that storage (two tile-derived
index tensors, ``labels[r // S, r % S + 1]``, would index the Cartesian product
in Helion, not the pairs).

The port sums over a full slice, which a looped config rolls into the same two
loops, with the logits loaded only in the second one, and the autotuner may
also pick a persistent reduction.
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(static_shapes=True)
def cross_entropy_bwd(
    total_loss: torch.Tensor,
    total_weight: torch.Tensor,
    flat_labels: torch.Tensor,
    logits: torch.Tensor,
    row_max: torch.Tensor,
    log_sum_exp: torch.Tensor,
    seq: hl.constexpr,
    label_stride: hl.constexpr,
) -> torch.Tensor:
    # total_loss, total_weight: 0-d fp32; flat_labels: the int64 [B, S + 1]
    # labels' storage viewed as 1-D (row b starts at b * label_stride);
    # logits: [B*S, V] bf16 view (padded row stride); row_max, log_sum_exp:
    # [B*S, 1] fp32.
    rows, vocab = logits.shape
    out = torch.empty([rows, vocab], dtype=logits.dtype, device=logits.device)
    for tile_r in hl.tile(rows):
        row = tile_r.index
        label = flat_labels[row // seq * label_stride + row % seq + 1]
        valid = label != -100
        target = torch.where(valid, label, 0)
        scale = torch.where(valid, total_loss[()] / total_weight[()], 0.0)
        # The statistics as ``[tile_r]`` loads, read once per row as in
        # Inductor (``[tile_r, :]`` slices are reloaded in every step of the
        # second loop).
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
        out[tile_r, :] = (nll_grad - torch.exp(shifted) * nll_sum).to(out.dtype)
    return out


def forward(inputs: list) -> tuple[torch.Tensor, ...]:
    total_loss, total_weight, labels, logits, row_max, log_sum_exp = inputs[:6]
    logits_shape = inputs[8]  # _shape_param_2 = [-1, V]
    out_shape = inputs[9]  # _shape_param_3 = [B, S, V]
    batch, label_len = labels.shape
    label_stride = labels.stride(0)
    if labels.stride(1) != 1:
        # The kernel reads labels[b, s + 1] at b * label_stride + s + 1.
        raise ValueError(f"labels of stride {labels.stride()}")
    flat_labels = labels.as_strided(
        [(batch - 1) * label_stride + label_len], [1], labels.storage_offset()
    )
    out = cross_entropy_bwd(
        total_loss,
        total_weight,
        flat_labels,
        logits.view(logits_shape),
        row_max,
        log_sum_exp,
        label_len - 1,
        label_stride,
    )
    return (out.view(out_shape),)
