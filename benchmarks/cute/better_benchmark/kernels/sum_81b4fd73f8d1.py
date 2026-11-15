"""better-benchmark repro sum_81b4fd73f8d1 (hf BartForCausalLM / MBartForCausalLM).

The training LM-head cross-entropy backward: the NLL gradient fed back
through ``log_softmax`` and added to an incoming bf16 logits gradient, then
zero-padded along the vocabulary (``V`` to ``V + 7``, a multiple of 8) for the
LM-head matmuls, once transposed. Inductor emits two kernels:

- ``triton_red_fused_add_convert_element_type_div_eq_exp_expand_full_iota_mul_ne_scalar_tensor_sub_sum_unsqueeze_view_where_0``:
  a looped reduction over the vocabulary per token row. The NLL gradient
  ``grad = where(iota == target, -1, 0) * where(target != -100, grad_out /
  count, 0)`` (``target`` set to 0 where ignored) is built from the iota
  alone; the first loop sums it, the second rebuilds it, loads the logits
  (row stride ``V + 7``, a view of the padded forward logits) and the
  incoming gradient and stores
  ``logits_grad + (grad - exp(logits - row_max - log_sum_exp) * sum(grad))``
  in bf16. As in Inductor, the graph's bf16 round trips of intermediates
  stay in fp32.
- ``triton_poi_fused_constant_pad_nd_permute_view_1``: reads that bf16
  gradient once and writes it zero-padded both as ``[rows, V + 7]`` and
  transposed as ``[V + 7, rows]``.

The iota is ``hl.arange(x.size(1))`` over the logits row ``x``: it names the
row's reduction dim, so the reduction can roll and the autotuner chooses
looped or persistent, and the logits are still read once per element. Sized
from the host int, ``hl.arange(V)`` would keep the reduction persistent. In a
looped reduction the iota holds the column positions of each chunk, so both
loops compare the target with the columns of their current chunk; in a
persistent one it spans the whole row.
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(static_shapes=True)
def nll_log_softmax_backward_add(
    target: torch.Tensor,
    grad_out: torch.Tensor,
    count: torch.Tensor,
    logits_grad: torch.Tensor,
    logits: torch.Tensor,
    row_max: torch.Tensor,
    log_sum_exp: torch.Tensor,
) -> torch.Tensor:
    # target: [rows] int64; grad_out, count: 0-d fp32; logits_grad, logits:
    # [rows, vocab] bf16; row_max, log_sum_exp: [rows, 1] fp32.
    rows, vocab = logits.shape
    out = torch.empty([rows, vocab], dtype=logits_grad.dtype, device=logits.device)
    for tile_r in hl.tile(rows):
        label = target[tile_r]
        valid = label != -100
        label = torch.where(valid, label, 0)
        scale = torch.where(valid, grad_out[()] / count[()], 0.0)
        x = logits[tile_r, :].to(torch.float32)
        cols = hl.arange(x.size(1), dtype=torch.int64)
        onehot = torch.where(cols[None, :] == label[:, None], -1.0, 0.0)
        grad = onehot * scale[:, None]
        grad_sum = torch.sum(grad, dim=-1, keepdim=True)
        shifted = x - row_max[tile_r, 0][:, None]
        shifted = shifted - log_sum_exp[tile_r, 0][:, None]
        result = grad - torch.exp(shifted) * grad_sum
        total = logits_grad[tile_r, :].to(torch.float32) + result
        out[tile_r, :] = total.to(out.dtype)
    return out


@helion.kernel(static_shapes=True)
def pad_vocab_and_transpose(
    grad: torch.Tensor, pad: hl.constexpr
) -> tuple[torch.Tensor, torch.Tensor]:
    # grad: [rows, vocab]; writes it zero-padded to vocab + pad columns, both
    # transposed ([vocab + pad, rows]) and not ([rows, vocab + pad]).
    rows, vocab = grad.shape
    padded = vocab + pad
    out_t = torch.empty([padded, rows], dtype=grad.dtype, device=grad.device)
    out = torch.empty([rows, padded], dtype=grad.dtype, device=grad.device)
    for tile_r, tile_v in hl.tile([rows, padded]):
        in_range = (tile_v.index < vocab)[None, :]
        value = hl.load(grad, [tile_r, tile_v], extra_mask=in_range)
        value = torch.where(in_range, value, 0.0)
        out[tile_r, tile_v] = value
        out_t[tile_v, tile_r] = value.permute(1, 0)
    return out_t, out


def forward(inputs: list) -> tuple[torch.Tensor, ...]:
    grad_out, count, target, logits, row_max, log_sum_exp, logits_grad = inputs[:7]
    rows, vocab = inputs[9]  # _shape_param_2 = [rows, vocab]
    pad_t = inputs[12]  # _shape_param_5 = [0, 0, 0, pad]: the permuted dim 0
    pad = inputs[13]  # _shape_param_6 = [0, pad, 0, 0]: dim 1
    if list(pad_t) != [0, 0, 0, pad[1]] or list(pad) != [0, pad[1], 0, 0]:
        raise ValueError(f"unexpected pads {pad_t} and {pad}")
    grad = nll_log_softmax_backward_add(
        target.view(rows),
        grad_out,
        count,
        logits_grad.view(rows, vocab),
        logits.view(rows, vocab),
        row_max,
        log_sum_exp,
    )
    return pad_vocab_and_transpose(grad, pad[1])
