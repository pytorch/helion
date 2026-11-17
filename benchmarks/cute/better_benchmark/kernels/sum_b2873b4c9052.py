"""better-benchmark repro sum_b2873b4c9052 (genai CrossEntropyBackward, static).

Inductor emits one looped reduction,
``triton_red_fused_convert_element_type_eq_exp_expand_full_iota_mul_ne_sub_sum_unsqueeze_view_where_0``:
the cross-entropy backward through a log-softmax for bf16 logits ``[N, V]``.
Per row, the one-hot gradient ``where(target == iota(V), -1.0, 0.0) * scale``
(``target = where(label != -100, label, 0)`` from the int64 labels, ``scale``
the 0-d bf16 upstream gradient, zeroed for ignored rows) is summed over V; the
first loop over V blocks does only that index math and reads no row data. The
second loop recomputes the one-hot term, loads the logits and stores
``onehot - exp(logits - shift0 - shift1) * sum`` in bf16 (``shift0`` and
``shift1`` are the fp32 ``[N, 1]`` log-softmax shifts). Like Inductor (which
runs without ``emulate_precision_casts``), the port skips the graph's bf16
round trip of ``logits - shift0 - shift1`` before the ``exp``.

The port sums over a full slice, which a looped config rolls into the same two
loops. The logits load and its ``exp`` go only into the second loop, the one
that reads them, so the first loop loads no row data, as in Inductor.
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(static_shapes=True)
def cross_entropy_backward(
    logits: torch.Tensor,
    shift0: torch.Tensor,
    shift1: torch.Tensor,
    labels: torch.Tensor,
    grad: torch.Tensor,
) -> torch.Tensor:
    # logits: [N, V] bf16; shift0, shift1: [N, 1] fp32; labels: [N] int64;
    # grad: 0-d bf16, expanded over the rows.
    rows, vocab = logits.shape
    out = torch.empty([rows, vocab], dtype=logits.dtype, device=logits.device)
    for tile_r in hl.tile(rows):
        label = labels[tile_r]
        valid = label != -100
        target = torch.where(valid, label, 0)
        scale = torch.where(valid, grad[()].to(torch.float32), 0.0)
        # The shifts as ``[tile_r]`` loads, read once per row as in Inductor
        # (``[tile_r, :]`` slices are reloaded in every step of the second loop).
        row_shift0 = shift0[tile_r, 0]
        row_shift1 = shift1[tile_r, 0]
        x = logits[tile_r, :].to(torch.float32)
        # Sized from the slice, the iota names the row's reduction dim, so a
        # looped reduction gives it each chunk's positions (``hl.arange(vocab)``
        # of the literal size would keep the reduction persistent).
        cols = hl.arange(x.size(1), dtype=torch.int64)
        onehot = (
            torch.where(target[:, None] == cols[None, :], -1.0, 0.0) * scale[:, None]
        )
        total = torch.sum(onehot, dim=-1, keepdim=True)
        shifted = x - row_shift0[:, None] - row_shift1[:, None]
        out[tile_r, :] = (onehot - torch.exp(shifted) * total).to(out.dtype)
    return out


def forward(inputs: list) -> tuple[torch.Tensor, ...]:
    logits, shift0, shift1, labels, grad = inputs[:5]
    out = cross_entropy_backward(logits, shift0, shift1, labels, grad)
    return (out,)
