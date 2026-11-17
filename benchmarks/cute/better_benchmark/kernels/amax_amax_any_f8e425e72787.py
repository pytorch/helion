"""better-benchmark repro amax_amax_any_f8e425e72787 (timm/train visformer_small).

The attention softmax of a Visformer attention block (6 heads of dim 64,
S = 196 tokens) between two matmuls padded from S to S + 4 = 200: the scores
arrive padded and the probabilities leave zero-padded. Inductor emits two
kernels:

- ``triton_per_fused_abs_amax_any_convert_element_type_div_eq_exp_logical_not_mul_ne_slice_sub_sum_view_where_0``:
  a persistent reduction over the rows of the bf16 scores. The input
  ``[B*H, S + 4, S + 4]`` is sliced to ``[B*H, S, S]`` (``slice_1``,
  ``slice_2``: a strided view, rows ``S + 4`` apart) and viewed as
  ``[B, H, S, S]``. On each row ``x`` (fp32: the graph's bf16 ``* 0.125``
  is not rounded to bf16, as in Inductor, and ``* 1`` folds away) it computes
  four reductions: whether every scaled score ``x * 0.125`` is finite
  (``logical_not(any(logical_not(eq & ne)))``, stored as ``logical_not_1``),
  ``amax(x)``, ``amax(x * 0.125)`` and, after
  ``where(all_finite, (x - amax(x)) * 0.125, x * 0.125 - amax(x * 0.125))``,
  the sum of its ``exp`` (``sum_1``). The four row results are stored as
  ``[B, H, S, 1]`` and the probabilities ``exp / sum`` in bf16 as
  ``[B*H, S, S]``.
- ``triton_poi_fused_constant_pad_nd_convert_element_type_div_exp_mul_slice_sub_view_where_1``:
  a pointwise kernel over the padded ``[B*H, S + 4, S + 4]`` output
  (``constant_pad_nd`` with ``[0, 4, 0, 4, 0, 0]``) that loads the
  probabilities, masked to the first S rows and columns, and stores zero
  elsewhere.

The probabilities are also returned transposed (``permute(0, 2, 1)``), as a
view. Inductor pads the probabilities' per-head stride from ``S * S`` to a
multiple of 64 elements; the port keeps eager's contiguous layout, so the
transposed view has eager's strides.
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(static_shapes=True)
def scaled_softmax_stats(
    scores: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # scores: [B*H, S, S] bf16, the strided slice of the padded input.
    groups, q_len, k_len = scores.shape
    amax_out = torch.empty([groups, q_len], dtype=torch.float32, device=scores.device)
    scaled_amax_out = torch.empty(
        [groups, q_len], dtype=torch.float32, device=scores.device
    )
    all_finite_out = torch.empty(
        [groups, q_len], dtype=torch.bool, device=scores.device
    )
    sum_out = torch.empty([groups, q_len], dtype=torch.float32, device=scores.device)
    probs_out = torch.empty(
        [groups, q_len, k_len], dtype=scores.dtype, device=scores.device
    )
    for tile_g, tile_q in hl.tile([groups, q_len]):
        x = scores[tile_g, tile_q, :].to(torch.float32)
        scaled = x * 0.125
        finite = (scaled == scaled) & (torch.abs(scaled) != float("inf"))
        any_nonfinite = torch.any(torch.logical_not(finite), dim=-1)
        all_finite = torch.logical_not(any_nonfinite)
        amax = torch.amax(x, dim=-1)
        scaled_amax = torch.amax(scaled, dim=-1)
        shifted = torch.where(
            all_finite[:, :, None],
            (x - amax[:, :, None]) * 0.125,
            scaled - scaled_amax[:, :, None],
        )
        e = torch.exp(shifted)
        denom = torch.sum(e, dim=-1)
        all_finite_out[tile_g, tile_q] = all_finite
        amax_out[tile_g, tile_q] = amax
        scaled_amax_out[tile_g, tile_q] = scaled_amax
        sum_out[tile_g, tile_q] = denom
        probs_out[tile_g, tile_q, :] = (e / denom[:, :, None]).to(probs_out.dtype)
    return amax_out, scaled_amax_out, all_finite_out, sum_out, probs_out


@helion.kernel(static_shapes=True)
def pad_rows_and_cols(
    probs: torch.Tensor, pad_rows: hl.constexpr, pad_cols: hl.constexpr
) -> torch.Tensor:
    # probs: [B*H, S, S] bf16; the output pads both trailing dims with zeros.
    groups, rows, cols = probs.shape
    out = torch.empty(
        [groups, rows + pad_rows, cols + pad_cols],
        dtype=probs.dtype,
        device=probs.device,
    )
    for tile_g, tile_r, tile_c in hl.tile([groups, rows + pad_rows, cols + pad_cols]):
        in_bounds = (tile_r.index < rows)[:, None] & (tile_c.index < cols)[None, :]
        values = hl.load(
            probs, [tile_g, tile_r, tile_c], extra_mask=in_bounds[None, :, :]
        ).to(torch.float32)
        out[tile_g, tile_r, tile_c] = torch.where(
            in_bounds[None, :, :], values, 0.0
        ).to(out.dtype)
    return out


def forward(inputs: list) -> tuple[torch.Tensor, ...]:
    arg0_1 = inputs[0]
    view_shape = inputs[1]  # _shape_param_0 = [B, H, S, S]
    pad = inputs[4]  # _shape_param_3 = [0, 4, 0, 4, 0, 0]: constant_pad_nd widths
    if pad[0] != 0 or pad[2] != 0 or any(pad[4:]):
        raise ValueError(f"this port pads only the end of the last two dims: {pad}")
    # slice_1, slice_2: drop the last 4 rows and columns of each head. The
    # kernel reads that strided [B*H, S, S] slice; the graph's view to
    # [B, H, S, S] only splits B*H.
    scores = arg0_1[:, :-4, :-4]
    amax, scaled_amax, all_finite, denom, probs = scaled_softmax_stats(scores)
    padded = pad_rows_and_cols(probs, pad[3], pad[1])
    row_shape = [*view_shape[:-1], 1]
    return (
        amax.view(row_shape),
        scaled_amax.view(row_shape),
        all_finite.view(row_shape),
        denom.view(row_shape),
        padded,
        probs.permute(0, 2, 1),
    )
