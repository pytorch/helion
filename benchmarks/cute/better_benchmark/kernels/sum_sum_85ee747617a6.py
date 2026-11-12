"""better-benchmark repro sum_sum_85ee747617a6 (hf/train BertForMaskedLM et al.).

The masked-LM head's cross-entropy backward (BERT, DistilBERT, LayoutLM and
ConvBERT masked-LM training). Inductor emits three kernels:

- ``triton_red_fused_add_convert_element_type_div_eq_exp_expand_full_iota_mul_ne_scalar_tensor_sub_sum_unsqueeze_view_where_0``:
  a looped reduction over the vocabulary per token. The NLL backward builds
  ``grad_nll = where(target == iota, -1.0, 0.0) * where(valid, num / den, 0.0)``
  (``valid = label != -100``, ``target = where(valid, label, 0)``) and the
  first loop sums it per row; the second loop recomputes ``grad_nll``, forms
  the log-softmax backward ``grad_nll - exp(logits - amax - logsumexp) * sum``
  (the bf16 logits rows are padded to a stride of ``vocab + 6``), adds the
  incoming bf16 gradient and stores the bf16 logits gradient ``[B*S, vocab]``.
- ``triton_red_fused_convert_element_type_sum_view_1``: the decoder bias
  gradient, the fp32 column sums of the logits gradient over all tokens.
- ``triton_poi_fused_constant_pad_nd_view_2``: the logits gradient
  zero-padded on the right of the vocabulary dim (``constant_pad_nd`` with
  ``[0, 6, 0, 0]``).

The logits gradient is also returned transposed, as a view. Like Inductor,
the math stays in fp32 between loads and stores: the graph's intermediate
bf16 round trips (``convert_element_type`` to bf16 and back) are not
materialized, including the one on the fp32 bias gradient.

The class iota is sized from the logits row (``hl.arange(x.size(1))``) so it
shares the row's reduction dim and the reduction can roll:
``hl.arange(vocab)`` with the host int would keep the reduction persistent
(the roller keeps the dim of a free range persistent).
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(static_shapes=True)
def cross_entropy_backward_add(
    labels: torch.Tensor,
    numerator: torch.Tensor,
    denominator: torch.Tensor,
    grad_in: torch.Tensor,
    logits: torch.Tensor,
    amax: torch.Tensor,
    logsumexp: torch.Tensor,
) -> torch.Tensor:
    # labels, amax, logsumexp: [rows]; numerator, denominator: 0-d;
    # grad_in, logits: [rows, vocab] (logits rows strided by the padded width).
    rows, vocab = logits.shape
    out = torch.empty([rows, vocab], dtype=grad_in.dtype, device=grad_in.device)
    for tile_r in hl.tile(rows):
        label = labels[tile_r]
        valid = label != -100
        target = torch.where(valid, label, 0)
        scale = torch.where(valid, numerator[()] / denominator[()], 0.0)
        x = logits[tile_r, :].to(torch.float32)
        # Sized from the row, so a looped reduction gives the iota each
        # chunk's positions.
        classes = hl.arange(x.size(1), dtype=torch.int64)
        grad_nll = (
            torch.where(target[:, None] == classes[None, :], -1.0, 0.0) * scale[:, None]
        )
        grad_sum = torch.sum(grad_nll, dim=-1, keepdim=True)
        log_probs = x - amax[tile_r][:, None] - logsumexp[tile_r][:, None]
        grad_logits = grad_nll - torch.exp(log_probs) * grad_sum
        out[tile_r, :] = (grad_in[tile_r, :].to(torch.float32) + grad_logits).to(
            out.dtype
        )
    return out


@helion.kernel(static_shapes=True)
def bias_grad_column_sum(grad: torch.Tensor) -> torch.Tensor:
    rows, vocab = grad.shape
    out = torch.empty([vocab], dtype=torch.float32, device=grad.device)
    for tile_v in hl.tile(vocab):
        out[tile_v] = torch.sum(grad[:, tile_v].to(torch.float32), dim=0)
    return out


@helion.kernel(static_shapes=True)
def pad_last_dim(src: torch.Tensor, pad_right: hl.constexpr) -> torch.Tensor:
    rows, cols = src.shape
    out = torch.empty([rows, cols + pad_right], dtype=src.dtype, device=src.device)
    for tile_r, tile_c in hl.tile([rows, cols + pad_right]):
        in_bounds = (tile_c.index < cols)[None, :]
        values = hl.load(src, [tile_r, tile_c], extra_mask=in_bounds)
        out[tile_r, tile_c] = torch.where(in_bounds, values, 0.0)
    return out


def forward(inputs: list) -> tuple[torch.Tensor, ...]:
    numerator, denominator, labels, logits, amax, logsumexp, grad_in = inputs[:7]
    pad = inputs[12]  # _shape_param_5 = [0, 6, 0, 0]: constant_pad_nd widths
    if pad[0] != 0 or any(pad[2:]):
        raise ValueError(f"this port pads only the right of the last dim, not {pad}")
    batch, seq, vocab = grad_in.shape
    rows = batch * seq
    grad = cross_entropy_backward_add(
        labels.view(rows),
        numerator,
        denominator,
        grad_in.view(rows, vocab),
        logits.view(rows, vocab),
        amax.view(rows),
        logsumexp.view(rows),
    )
    bias_grad = bias_grad_column_sum(grad)
    padded = pad_last_dim(grad, pad[1])
    return padded, grad.permute(1, 0), bias_grad
