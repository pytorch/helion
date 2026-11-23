"""better-benchmark repro sum_b7a98f3c6228 (torchbench/train+infer nvidia_deeprecommender).

The SELU backward of the autoencoder's widest activation (``[batch, items]``
bf16, 197951 items): ``where(x <= 0, grad * 1.7580993408473766 * exp(x),
grad * 1.0507009873554805)`` (scale * alpha and scale) from the SELU input
``x`` and the incoming gradient. The result is returned zero-padded by one
item (to 197952, a multiple of 8) for the layer's matmuls, as
``[batch, items + 1]`` and transposed as ``[items + 1, batch]``, together with
its fp32 column sums over the batch (the layer's bias gradient). Inductor
emits three kernels:

- ``triton_red_fused_convert_element_type_exp_le_mul_sum_view_where_0``: a
  looped reduction over the batch per item column. It loads the bf16 SELU
  input (row stride ``items + 1``) and the bf16 incoming gradient, computes
  the gradient in fp32, stores it in bf16 to the first ``items`` columns of
  the padded output and sums it per column in fp32.
- ``triton_poi_fused_constant_pad_nd_convert_element_type_exp_le_mul_where_1``:
  zero-fills the trailing column of the padded output (a ``[batch, 1]`` alias
  with row stride ``items + 1``, which Inductor indexes as one dimension of
  ``batch`` elements). It loads nothing.
- ``triton_poi_fused_constant_pad_nd_permute_2``: a 2-D grid over
  ``[items + 1, batch]`` that loads the gradient back from the padded
  output's first ``items`` columns, transposed (masked to the first ``items``
  rows, the masked lanes reading as the padding's zero, widened to fp32), and
  stores it in bf16 to the contiguous transposed output.

Like Inductor, the math stays in fp32 between loads and stores: the graph
rounds the gradient to bf16 before summing it and rounds the fp32 column sums
to bf16 and back, and neither round trip is materialized (the sums add the
unrounded fp32 gradient). Inductor drops the graph's two ``* 1``; the port
keeps them, exact identities.

The port allocates the padded output on the host and passes each kernel the
alias Inductor passes it; the transposed kernel takes the permuted view of
its alias, the graph's ``permute``.
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl

# The graph's literal pads: ``constant_pad_nd(grad, [0, 1, 0, 0])`` and
# ``constant_pad_nd(grad.permute(1, 0), [0, 0, 0, 1])``.
PAD = 1


@helion.kernel(static_shapes=True)
def selu_backward_pad_column_sum(
    x: torch.Tensor, grad_out: torch.Tensor, padded: torch.Tensor
) -> torch.Tensor:
    # x: [batch, items] bf16, the SELU input (row stride items + 1); grad_out:
    # [batch, items] bf16; padded: [batch, items], the unpadded columns of the
    # padded output (row stride items + 1).
    batch, items = x.shape
    column_sum = torch.empty([items], dtype=torch.float32, device=x.device)
    for tile_i in hl.tile(items):
        v = x[:, tile_i].to(torch.float32)
        g = grad_out[:, tile_i].to(torch.float32)
        negative = g * 1 * 1.7580993408473766 * torch.exp(v * 1)
        positive = g * 1.0507009873554805
        grad_in = torch.where(v <= 0, negative, positive)
        padded[:, tile_i] = grad_in.to(padded.dtype)
        # The graph sums the bf16 gradient and rounds the sum to bf16 and
        # back; Inductor sums the fp32 gradient and keeps the sum in fp32.
        column_sum[tile_i] = torch.sum(grad_in, dim=0)
    return column_sum


@helion.kernel(static_shapes=True)
def zero_fill_padded_column(out: torch.Tensor) -> torch.Tensor:
    # out: [batch] bf16, the padded column of the padded output (stride
    # items + 1).
    (batch,) = out.shape
    for tile_b in hl.tile(batch):
        out[tile_b] = hl.zeros([tile_b], dtype=out.dtype)
    return out


@helion.kernel(static_shapes=True)
def pad_transposed(grad_t: torch.Tensor, pad: hl.constexpr) -> torch.Tensor:
    # grad_t: [items, batch] bf16, the permuted view of the padded output's
    # unpadded columns (strides 1 and items + pad); returns it zero-padded to
    # [items + pad, batch].
    items, batch = grad_t.shape
    out = torch.empty([items + pad, batch], dtype=grad_t.dtype, device=grad_t.device)
    for tile_i, tile_b in hl.tile([items + pad, batch]):
        in_items = (tile_i.index < items)[:, None]
        v = hl.load(grad_t, [tile_i, tile_b], extra_mask=in_items).to(torch.float32)
        out[tile_i, tile_b] = torch.where(in_items, v, 0.0).to(out.dtype)
    return out


def forward(inputs: list) -> tuple[torch.Tensor, ...]:
    grad_out, x = inputs[:2]
    sum_shape = inputs[2]  # _shape_param_0 = [items]
    batch, items = grad_out.shape
    padded = torch.empty(
        [batch, items + PAD], dtype=grad_out.dtype, device=grad_out.device
    )
    grad_in = padded[:, :items]
    column_sum = selu_backward_pad_column_sum(x, grad_out, grad_in)
    zero_fill_padded_column(padded[:, items])
    padded_t = pad_transposed(grad_in.permute(1, 0), PAD)
    return padded, padded_t, column_sum.view(sum_shape)
