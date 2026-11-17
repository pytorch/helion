"""better-benchmark repro amax_sum_any_30cded63a2f9 (hf/train BertForMaskedLM, RobertaForCausalLM).

Inductor emits one persistent reduction,
``triton_per_fused_any_convert_element_type_div_eq_exp_gt_inductor_lookup_seed_inductor_random_logical_not_mul_prepare_softmax_online_sub_view_where_0``:
the attention probabilities of a BERT training step, with dropout. Per row of
the bf16 scores ``[B*H, S, S]`` (viewed as ``[B, H, S, S]``) it takes an fp32
softmax over the last dim (row max, sum of ``exp(x - max)``, true division)
and, in the same pass, ``any(x != -inf)``; fully masked rows take the matching
row of the second bf16 ``[B, H, S, S]`` input instead of the softmax, stored in
bf16 (``where``; the bf16 cast of the probabilities stays fp32 until that
store, as in Inductor). Dropout draws ``tl.rand(seed, flat_index)`` with the
seed ``seeds[16]`` (``inductor_lookup_seed``) and keeps elements whose sample
exceeds 0.1 (compared at bf16 precision, so against 0.1 rounded to bf16, with
the sample left in fp32 as Inductor does), storing the mask (``gt``) and
``mask * where * 1.1111111111111112`` in bf16, returned as a ``[B*H, S, S]``
view and its transpose.

``hl.rand([tile, n], seed)`` uses the row-major flat index as its Philox
offset, like Inductor, and ``cute_rng_stream="word0"`` keeps Inductor's
stream on both backends: one Philox4x32-10 evaluation per element, keeping
word 0, so the mask is bit-identical to Inductor's. (Word 0 is the Triton
backend's only stream. The CuTe default, ``auto``, draws
``Philox(seed, i // 4)[i % 4]`` and lets the autotuner share one Philox
evaluation across four elements (``cute_rng_packet``): a quarter of
Inductor's RNG work, which is most of this kernel's arithmetic.) Eager
PyTorch draws ``inductor_random`` from its global generator instead, so the
checks compare the mask and the dropout outputs with Inductor's compiled
outputs (``common.AnchoredGate``), the mask exactly.
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(static_shapes=True, cute_rng_stream="word0")
def safe_softmax_dropout(
    scores: torch.Tensor,
    fallback: torch.Tensor,
    seeds: torch.Tensor,
    seed_index: hl.constexpr,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # scores, fallback: [rows, n] bf16 (contiguous views of the [B, H, S, S]
    # inputs); seeds: Inductor's int64 seed table.
    rows, n = scores.shape
    where_out = torch.empty([rows, n], dtype=fallback.dtype, device=scores.device)
    keep_out = torch.empty([rows, n], dtype=torch.bool, device=scores.device)
    dropped_out = torch.empty([rows, n], dtype=fallback.dtype, device=scores.device)
    for tile_r in hl.tile(rows):
        x = scores[tile_r, :].to(torch.float32)
        live = torch.logical_not(x == float("-inf"))
        any_live = torch.any(live, dim=-1, keepdim=True)
        row_max = torch.amax(x, dim=-1, keepdim=True)
        unnormalized = torch.exp(x - row_max)
        probs = unnormalized / torch.sum(unnormalized, dim=-1, keepdim=True)
        other = fallback[tile_r, :].to(torch.float32)
        result = torch.where(torch.logical_not(any_live), other, probs)
        where_out[tile_r, :] = result.to(where_out.dtype)
        rand = hl.rand([tile_r, n], seed=seeds[seed_index])
        # gt(convert_element_type(rand, bf16), 0.1) runs at bf16 precision:
        # 0.1 rounds to 0.10009765625; like Inductor, the sample stays fp32.
        keep = rand > 0.10009765625
        keep_out[tile_r, :] = keep
        dropped = keep.to(torch.float32) * result * 1.1111111111111112
        dropped_out[tile_r, :] = dropped.to(dropped_out.dtype)
    return where_out, keep_out, dropped_out


def forward(inputs: list) -> tuple[torch.Tensor, ...]:
    scores, fallback, seeds = inputs[:3]
    view_shape = inputs[3]  # _shape_param_0 = [B, H, S, S]
    out_shape = inputs[6]  # _shape_param_3 = [B*H, S, S]
    n = view_shape[-1]
    where, keep, dropped = safe_softmax_dropout(
        scores.view(-1, n), fallback.view(-1, n), seeds, 16
    )
    dropped = dropped.view(out_shape)
    return (
        where.view(view_shape),
        keep.view(view_shape),
        dropped,
        dropped.permute(0, 2, 1),
    )
