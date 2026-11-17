"""better-benchmark repro amax_amax_any_80cdfb3610b6 (hf/train LayoutLMForMaskedLM).

Inductor emits one persistent reduction,
``triton_per_fused_abs_amax_any_convert_element_type_div_eq_exp_gt_inductor_lookup_seed_inductor_random_logical_not_mul_ne_sub_sum_view_where_0``:
the attention probabilities of a LayoutLM training step, with dropout. Per row
of the bf16 scores ``[B*H, S, S]`` (viewed as ``[B, H, S, S]``) it reads the
scores once and computes four row reductions: the max of the scores
(``amax``), the max of the scores scaled by 0.125 (``amax_1``; the graph's
bf16 ``mul`` stays in fp32, as Inductor leaves it), and whether every scaled
score is finite (``logical_not(any(logical_not(eq & ne)))``, stored as
``logical_not_1``). Each element then takes
``where(all_finite, (x - amax) * 0.125, x * 0.125 - amax_1)``, and the row
sums its ``exp`` (``sum_1``). Dropout draws ``tl.rand(seed, flat_index)``
with the seed ``seeds[10]`` (``inductor_lookup_seed``) and keeps elements
whose sample exceeds 0.1 (compared at bf16 precision, so against 0.1 rounded
to bf16, with the sample left in fp32 as Inductor does), storing the mask
(``gt``) and ``mask * exp / sum_1 * 1.1111111111111112`` in bf16 (the bf16
casts of the probabilities and of the masked product stay fp32 until that
store, as in Inductor). The dropout output is returned as a ``[B*H, S, S]``
view and its transpose; the four row results as ``[B, H, S, 1]``.

``hl.rand([tile, n], seed)`` uses the row-major flat index as its Philox
offset, like Inductor, and ``cute_rng_stream="word0"`` keeps Inductor's
stream on both backends (one Philox4x32-10 evaluation per element, keeping
word 0; see PORTING.md), so the mask is bit-identical to Inductor's. Eager
PyTorch draws ``inductor_random`` from its global generator instead, so the
checks compare the mask and the dropout outputs with Inductor's compiled
outputs (``common.AnchoredGate``), the mask exactly.
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(static_shapes=True, cute_rng_stream="word0")
def scaled_softmax_dropout(
    scores: torch.Tensor,
    seeds: torch.Tensor,
    seed_index: hl.constexpr,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    # scores: [rows, n] bf16 (a contiguous view of the [B, H, S, S] scores).
    rows, n = scores.shape
    amax_out = torch.empty([rows], dtype=torch.float32, device=scores.device)
    scaled_amax_out = torch.empty([rows], dtype=torch.float32, device=scores.device)
    all_finite_out = torch.empty([rows], dtype=torch.bool, device=scores.device)
    sum_out = torch.empty([rows], dtype=torch.float32, device=scores.device)
    keep_out = torch.empty([rows, n], dtype=torch.bool, device=scores.device)
    dropped_out = torch.empty([rows, n], dtype=scores.dtype, device=scores.device)
    for tile_r in hl.tile(rows):
        x = scores[tile_r, :].to(torch.float32)
        scaled = x * 0.125
        finite = (scaled == scaled) & (torch.abs(scaled) != float("inf"))
        any_nonfinite = torch.any(torch.logical_not(finite), dim=-1)
        all_finite = torch.logical_not(any_nonfinite)
        amax = torch.amax(x, dim=-1)
        scaled_amax = torch.amax(scaled, dim=-1)
        shifted = torch.where(
            all_finite[:, None],
            (x - amax[:, None]) * 0.125,
            scaled - scaled_amax[:, None],
        )
        e = torch.exp(shifted)
        denom = torch.sum(e, dim=-1)
        rand = hl.rand([tile_r, n], seed=seeds[seed_index])
        # gt(convert_element_type(rand, bf16), 0.1) runs at bf16 precision:
        # 0.1 rounds to 0.10009765625; like Inductor, the sample stays fp32.
        keep = rand > 0.10009765625
        dropped = keep.to(torch.float32) * (e / denom[:, None]) * 1.1111111111111112
        amax_out[tile_r] = amax
        scaled_amax_out[tile_r] = scaled_amax
        all_finite_out[tile_r] = all_finite
        sum_out[tile_r] = denom
        keep_out[tile_r, :] = keep
        dropped_out[tile_r, :] = dropped.to(dropped_out.dtype)
    return amax_out, scaled_amax_out, all_finite_out, sum_out, keep_out, dropped_out


def forward(inputs: list) -> tuple[torch.Tensor, ...]:
    scores, seeds = inputs[:2]
    view_shape = inputs[2]  # _shape_param_0 = [B, H, S, S]
    out_shape = inputs[5]  # _shape_param_3 = [B*H, S, S]
    n = view_shape[-1]
    row_shape = [*view_shape[:-1], 1]
    amax, scaled_amax, all_finite, denom, keep, dropped = scaled_softmax_dropout(
        scores.view(-1, n), seeds, 10
    )
    dropped = dropped.view(out_shape)
    return (
        amax.view(row_shape),
        scaled_amax.view(row_shape),
        all_finite.view(row_shape),
        denom.view(row_shape),
        keep.view(view_shape),
        dropped,
        dropped.permute(0, 2, 1),
    )
