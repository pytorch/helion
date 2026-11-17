"""better-benchmark repro sum_6119d597d188 (hf PLBartForCausalLM training).

The LM head's cross-entropy backward. The NLL backward builds the gradient of
the log-probabilities, ``grad_logp = where(label == iota(V), -1.0, 0.0) *
where(valid, grad_loss / total_weight, 0.0)`` (``valid = target != -100``,
``label`` the target with ignored entries replaced by 0), and the log-softmax
backward turns it into ``grad_logp - exp(logp) * sum(grad_logp)``, with the
log-probabilities recomputed as ``logits - amax - logsumexp`` (bf16 logits
whose rows are padded to a stride of 50008, fp32 ``[rows, 1]`` statistics).
That gradient is added to another bf16 gradient, and the sum is padded from
V = 50005 to 50008 vocabulary entries twice: transposed (``[50008, rows]``) and
as is (``[rows, 50008]``). Inductor emits two kernels:

- ``triton_red_fused_add_convert_element_type_div_eq_exp_expand_full_iota_mul_ne_scalar_tensor_sub_sum_unsqueeze_view_where_0``:
  a looped reduction over row blocks. A first loop over the vocabulary sums
  ``grad_logp``, computed from the label and the vocabulary index alone (it
  loads nothing); a second loop recomputes ``grad_logp``, reads the logits and
  the other gradient and stores the summed gradient ``[rows, V]`` in bf16. As
  in Inductor, the graph's intermediate bf16 round trips stay in fp32.
- ``triton_poi_fused_constant_pad_nd_permute_view_1``: a pointwise kernel over
  ``[rows, 50008]`` that loads that gradient (masked to the first V columns,
  zero elsewhere) and stores it to both padded outputs.

The first kernel reduces over full slices, like sum_b2873b4c9052: a looped
config rolls them into the same two loops, with the logits and the other
gradient loaded only in the second one, and the autotuner may also pick a
persistent reduction.
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(static_shapes=True)
def nll_log_softmax_backward_add(
    target: torch.Tensor,
    grad_loss: torch.Tensor,
    total_weight: torch.Tensor,
    grad_in: torch.Tensor,
    logits: torch.Tensor,
    amax: torch.Tensor,
    logsumexp: torch.Tensor,
    ignore_index: hl.constexpr,
) -> torch.Tensor:
    # target: [rows] int64; grad_loss, total_weight: 0-d fp32; grad_in, logits:
    # [rows, vocab] bf16 (logits with a padded row stride); amax, logsumexp:
    # [rows, 1] fp32.
    rows, vocab = grad_in.shape
    out = torch.empty([rows, vocab], dtype=grad_in.dtype, device=grad_in.device)
    for tile_r in hl.tile(rows):
        label = target[tile_r]
        valid = label != ignore_index
        label = torch.where(valid, label, 0)
        scale = torch.where(valid, grad_loss[()] / total_weight[()], 0.0)
        # The statistics as ``[tile_r]`` loads, read once per row as in
        # Inductor (``[tile_r, :]`` slices are reloaded in every step of the
        # second loop).
        row_amax = amax[tile_r, 0]
        row_logsumexp = logsumexp[tile_r, 0]
        x = logits[tile_r, :].to(torch.float32)
        # Sized from the slice, the iota names the row's reduction dim, so a
        # looped reduction gives it each chunk's positions (``hl.arange(vocab)``
        # of the host int would keep the reduction persistent).
        cols = hl.arange(x.size(1), dtype=torch.int64)
        grad_logp = (
            torch.where(label[:, None] == cols[None, :], -1.0, 0.0) * scale[:, None]
        )
        grad_sum = torch.sum(grad_logp, dim=-1, keepdim=True)
        logp = x - row_amax[:, None] - row_logsumexp[:, None]
        grad_x = grad_logp - torch.exp(logp) * grad_sum
        out[tile_r, :] = (grad_in[tile_r, :].to(torch.float32) + grad_x).to(out.dtype)
    return out


@helion.kernel(static_shapes=True)
def pad_vocab_and_transpose(
    grad: torch.Tensor, padded_vocab: hl.constexpr
) -> tuple[torch.Tensor, torch.Tensor]:
    # grad: [rows, vocab] bf16, zero-padded to padded_vocab columns and stored
    # transposed ([padded_vocab, rows]) and as is ([rows, padded_vocab]).
    rows, vocab = grad.shape
    grad_t = torch.empty([padded_vocab, rows], dtype=grad.dtype, device=grad.device)
    grad_padded = torch.empty(
        [rows, padded_vocab], dtype=grad.dtype, device=grad.device
    )
    for tile_r, tile_v in hl.tile([rows, padded_vocab]):
        in_vocab = tile_v.index < vocab
        x = hl.load(grad, [tile_r, tile_v], extra_mask=in_vocab[None, :]).to(
            torch.float32
        )
        x = torch.where(in_vocab[None, :], x, 0.0)
        grad_t[tile_v, tile_r] = x.permute(1, 0).to(grad_t.dtype)
        grad_padded[tile_r, tile_v] = x.to(grad_padded.dtype)
    return grad_t, grad_padded


def forward(inputs: list) -> tuple[torch.Tensor, ...]:
    grad_loss, total_weight, target, logits, amax, logsumexp, grad_in = inputs[:7]
    rows, vocab = inputs[9]  # _shape_param_2 = [B*S, V], the logits view
    pad_t = inputs[12]  # _shape_param_5 = [0, 0, 0, pad]: the permuted dim 0
    pad = inputs[13]  # _shape_param_6 = [0, pad, 0, 0]: dim 1
    if list(pad_t) != [0, 0, 0, pad[1]] or list(pad) != [0, pad[1], 0, 0]:
        raise ValueError(f"unexpected pads {pad_t} and {pad}")
    padded_vocab = vocab + pad[1]
    grad = nll_log_softmax_backward_add(
        target.view(rows),
        grad_loss,
        total_weight,
        grad_in.view(rows, vocab),
        logits.view(rows, vocab),
        amax,
        logsumexp,
        -100,
    )
    grad_t, grad_padded = pad_vocab_and_transpose(grad, padded_vocab)
    return grad_t, grad_padded
