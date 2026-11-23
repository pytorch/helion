"""better-benchmark repro sum_sum_2aec3d12d3d5 (hf RobertaForCausalLM training).

The causal-LM head's cross-entropy (``log_softmax`` + ``nll_loss``) backward,
accumulated into an existing bf16 logits gradient. The summed gradient
``[B*S, V]`` is returned zero-padded along the vocabulary (``V = 50265`` to
``V + 7``, a multiple of 8) for the LM-head matmuls, as ``[B*S, V + 7]`` and
transposed as ``[V + 7, B*S]``, together with its fp32 column sums over the
rows (the LM-head bias gradient). Inductor emits four kernels:

- ``triton_red_fused_add_clone_convert_element_type_div_eq_exp_expand_full_iota_mul_ne_scalar_tensor_slice_sub_sum_unsqueeze_view_where_0``:
  sum_0aced6470e1e's looped reduction over the vocabulary per token row. Each
  of the ``B*S`` rows takes its shifted label ``labels[b, s + 1]`` (ignore
  index -100 maps to 0) and the upstream scale ``total_loss / total_weight``
  (zeroed for ignored rows). The NLL gradient
  ``where(label == iota(V), -1.0, 0.0) * scale`` is pure index math: the
  first loop over the vocab sums it per row without loading anything, and the
  second recomputes it, loads the bf16 logits (row stride ``V + 7``) and the
  bf16 incoming gradient, and stores
  ``grad + (nll_grad - exp(logits - max - log_sum_exp) * sum)`` in bf16 to a
  temporary.
- ``triton_red_fused_add_clone_constant_pad_nd_convert_element_type_div_eq_exp_expand_full_iota_mul_ne_permute_scalar_tensor_slice_sub_sum_unsqueeze_view_where_1``:
  a looped reduction over the rows per vocabulary column. It loads the
  temporary once and stores it twice, to the first ``V`` columns of the
  padded output and transposed to the first ``V`` rows of the transposed
  output, while summing it in fp32 per column.
- ``triton_poi_fused_add_clone_constant_pad_nd_convert_element_type_div_eq_exp_expand_full_iota_mul_ne_scalar_tensor_slice_sub_unsqueeze_view_where_2``:
  zero-fills the 7 trailing columns of the padded output (a ``[B*S, 7]``
  alias with row stride ``V + 7``; Inductor indexes it as one flat dimension
  and recovers the row and column with a division and a remainder). It loads
  nothing.
- ``triton_poi_fused_add_clone_constant_pad_nd_convert_element_type_div_eq_exp_expand_full_iota_mul_ne_permute_scalar_tensor_slice_sub_unsqueeze_view_where_3``:
  zero-fills the 7 trailing rows of the transposed output, a contiguous
  ``[7, B*S]`` alias that Inductor indexes as one flat dimension.

Like Inductor, the math stays in fp32 between loads and stores: the graph's
intermediate bf16 round trips (``convert_element_type`` to bf16 and back) are
not materialized, so the gradient is not rounded to bf16 before the add and
the fp32 column sums are not rounded to bf16.

The port allocates both padded outputs on the host and passes each kernel the
alias Inductor passes it. The temporary between the first two kernels has
Inductor's layout as well: comprehensive padding rounds its row stride up to
128 bytes (``V = 50265`` to 50304 bf16 elements), so every row the first
kernel stores starts aligned.

Inductor never materializes the graph's contiguous clone of the shifted label
slice; it reads row ``r``'s label at ``1 + row_stride * (r // S) + r % S`` of
the labels' storage. The port reads the same offset of a 1-D view of that
storage (two tile-derived index tensors, ``labels[r // S, r % S + 1]``, would
index the Cartesian product in Helion, not the pairs).
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl

# Inductor's comprehensive padding (``padding_alignment_bytes`` and
# ``padding_stride_threshold``): a temporary's stride above 1024 elements is
# rounded up to a multiple of 128 bytes.
PADDING_ALIGNMENT_BYTES = 128
PADDING_STRIDE_THRESHOLD = 1024


@helion.kernel(static_shapes=True)
def cross_entropy_bwd_accumulate(
    total_loss: torch.Tensor,
    total_weight: torch.Tensor,
    flat_labels: torch.Tensor,
    logits: torch.Tensor,
    row_max: torch.Tensor,
    log_sum_exp: torch.Tensor,
    grad: torch.Tensor,
    out: torch.Tensor,
    seq: hl.constexpr,
    label_stride: hl.constexpr,
) -> torch.Tensor:
    # total_loss, total_weight: 0-d fp32; flat_labels: the int64 [B, S + 1]
    # labels' storage viewed as 1-D (row b starts at b * label_stride);
    # logits, grad: [B*S, V] bf16 views (logits with a padded row stride);
    # row_max, log_sum_exp: [B*S, 1] fp32; out: [B*S, V] bf16, the temporary
    # (padded row stride).
    rows, vocab = logits.shape
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


@helion.kernel(static_shapes=True)
def pad_transpose_column_sum(
    grad: torch.Tensor, padded: torch.Tensor, padded_t: torch.Tensor
) -> torch.Tensor:
    # grad: [rows, vocab] bf16, the temporary (padded row stride); padded:
    # [rows, vocab], the unpadded columns of the padded output (row stride
    # vocab + pad); padded_t: [vocab, rows], the unpadded rows of the
    # transposed padded output.
    rows, vocab = grad.shape
    column_sum = torch.empty([vocab], dtype=torch.float32, device=grad.device)
    for tile_v in hl.tile(vocab):
        x = grad[:, tile_v].to(torch.float32)
        padded[:, tile_v] = x.to(padded.dtype)
        padded_t[tile_v, :] = x.permute(1, 0).to(padded_t.dtype)
        # The graph rounds this sum to bf16 and back; Inductor keeps it in
        # fp32.
        column_sum[tile_v] = torch.sum(x, dim=0)
    return column_sum


@helion.kernel(static_shapes=True)
def zero_fill_padded_columns(out: torch.Tensor) -> torch.Tensor:
    # out: [rows, pad] bf16, the padded columns of the padded output (row
    # stride vocab + pad).
    rows, cols = out.shape
    for tile_r, tile_c in hl.tile([rows, cols]):
        out[tile_r, tile_c] = hl.zeros([tile_r, tile_c], dtype=out.dtype)
    return out


@helion.kernel(static_shapes=True)
def zero_fill_padded_rows(out: torch.Tensor) -> torch.Tensor:
    # out: [pad * rows] bf16, the padded rows of the transposed padded output
    # viewed as 1-D (they are contiguous).
    (size,) = out.shape
    for tile in hl.tile(size):
        out[tile] = hl.zeros([tile], dtype=out.dtype)
    return out


def forward(inputs: list) -> tuple[torch.Tensor, ...]:
    total_loss, total_weight, labels, logits, row_max, log_sum_exp, grad = inputs[:7]
    logits_shape = inputs[9]  # _shape_param_2 = [-1, V]
    rows, vocab = inputs[11]  # _shape_param_4 = [B*S, V]
    pad = inputs[12]  # _shape_param_5 = [0, pad, 0, 0]: dim 1
    pad_t = inputs[13]  # _shape_param_6 = [0, 0, 0, pad]: the permuted dim 0
    sum_shape = inputs[14]  # _shape_param_7 = [V]
    if list(pad) != [0, pad[1], 0, 0] or list(pad_t) != [0, 0, 0, pad[1]]:
        raise ValueError(f"unexpected pads {pad} and {pad_t}")
    padded_vocab = vocab + pad[1]
    batch, label_len = labels.shape
    label_stride = labels.stride(0)
    if labels.stride(1) != 1:
        # The kernel reads labels[b, s + 1] at b * label_stride + s + 1.
        raise ValueError(f"labels of stride {labels.stride()}")
    flat_labels = labels.as_strided(
        [(batch - 1) * label_stride + label_len], [1], labels.storage_offset()
    )
    # The temporary's row stride, as Inductor pads it.
    align = PADDING_ALIGNMENT_BYTES // grad.dtype.itemsize
    row_stride = vocab
    if row_stride > PADDING_STRIDE_THRESHOLD:
        row_stride = (row_stride + align - 1) // align * align
    grad_sum = torch.empty([rows, row_stride], dtype=grad.dtype, device=grad.device)[
        :, :vocab
    ]
    cross_entropy_bwd_accumulate(
        total_loss,
        total_weight,
        flat_labels,
        logits.view(logits_shape),
        row_max,
        log_sum_exp,
        grad.view(rows, vocab),
        grad_sum,
        label_len - 1,
        label_stride,
    )
    padded = torch.empty([rows, padded_vocab], dtype=grad.dtype, device=grad.device)
    padded_t = torch.empty([padded_vocab, rows], dtype=grad.dtype, device=grad.device)
    column_sum = pad_transpose_column_sum(
        grad_sum, padded[:, :vocab], padded_t[:vocab, :]
    )
    zero_fill_padded_columns(padded[:, vocab:])
    zero_fill_padded_rows(padded_t[vocab:, :].view(-1))
    return padded, padded_t, column_sum.view(sum_shape)
