"""better-benchmark repro amax_sum_821fb95bd167 (timm/infer/swin_base_patch4_window7_224).

Swin's window attention softmax (the blocks without a shift mask). The bf16
scores arrive zero-padded, ``[B*H, P, P]`` with ``P = 56``, and are sliced to
the window's ``Q = K = 49`` tokens and viewed as ``[B, H, Q, K]``. The
relative position bias is added: a gather of the bf16 ``[169, H]`` table by
the int64 ``[Q, K]`` position index (``table[index[q, k], h]``; the graph's
view, permute and clone fold into the gather). A softmax over K in fp32
follows, stored in bf16 and zero-padded back to ``[B*H, P, P]``. Inductor
emits two kernels:

- ``triton_per_fused_add_clone_convert_element_type_index_permute_prepare_softmax_online_slice_unsqueeze_view_0``:
  a persistent reduction over the ``B*H*Q`` rows. It loads the scores, the
  position index and the gathered bias, adds them in fp32 (no bf16 rounding
  of the graph's ``add``) and stores the row max and the row sum of
  ``exp(x - max)`` to two fp32 ``[B, H, Q]`` workspaces.
- ``triton_poi_fused_add_clone_constant_pad_nd_convert_element_type_div_exp_index_permute_slice_sub_unsqueeze_view_1``:
  a pointwise kernel over the padded output. Inside the window it reloads the
  same three inputs and the two statistics (every load masked to the window),
  recomputes ``x`` and stores ``exp(x - max) / sum`` in bf16; the padding
  stores zero.

Inductor indexes both kernels by one flat dimension and recovers the head and
the row with divisions and remainders; the port tiles ``[B, H, Q]`` and
``[B, H, P, P]`` and broadcasts the loads that do not depend on a dim. The
output is returned as the contiguous ``[B*H, P, P]`` view.
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(static_shapes=True)
def window_softmax_stats(
    scores: torch.Tensor, rel_index: torch.Tensor, rel_table: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    # scores: [B, H, Q, K] bf16, the window slice of the padded [B*H, P, P]
    # input (rows P apart in memory); rel_index: [Q, K] int64; rel_table:
    # [T, H] bf16.
    batch, heads, q_len, k_len = scores.shape
    table_rows = rel_table.size(0)
    row_max = torch.empty(
        [batch, heads, q_len], dtype=torch.float32, device=scores.device
    )
    row_sum = torch.empty(
        [batch, heads, q_len], dtype=torch.float32, device=scores.device
    )
    for tile_b, tile_h, tile_q in hl.tile([batch, heads, q_len]):
        # index(rel_table, [rel_index.view(-1)]), viewed [Q, K, H] and permuted
        # to [H, Q, K]; aten.index wraps a negative index.
        position = rel_index[tile_q, :]
        position = torch.where(position < 0, position + table_rows, position)
        bias = rel_table[position[None, :, :], tile_h.index[:, None, None]]
        x = (
            scores[tile_b, tile_h, tile_q, :].to(torch.float32)
            + bias.to(torch.float32)[None, :, :, :]
        )
        x_max = torch.amax(x, dim=-1)
        x_sum = torch.sum(torch.exp(x - x_max[:, :, :, None]), dim=-1)
        row_max[tile_b, tile_h, tile_q] = x_max
        row_sum[tile_b, tile_h, tile_q] = x_sum
    return row_max, row_sum


@helion.kernel(static_shapes=True)
def window_softmax_padded(
    scores: torch.Tensor,
    rel_index: torch.Tensor,
    rel_table: torch.Tensor,
    row_max: torch.Tensor,
    row_sum: torch.Tensor,
    pad_q: hl.constexpr,
    pad_k: hl.constexpr,
) -> torch.Tensor:
    # scores, rel_index, rel_table: as in window_softmax_stats; row_max,
    # row_sum: [B, H, Q] fp32. Writes
    # constant_pad_nd(softmax, [0, pad_k, 0, pad_q]).
    batch, heads, q_len, k_len = scores.shape
    table_rows = rel_table.size(0)
    out_q = q_len + pad_q
    out_k = k_len + pad_k
    out = torch.empty(
        [batch, heads, out_q, out_k], dtype=scores.dtype, device=scores.device
    )
    for tile_b, tile_h, tile_q, tile_k in hl.tile([batch, heads, out_q, out_k]):
        in_rows = tile_q.index < q_len
        inside = in_rows[:, None] & (tile_k.index < k_len)[None, :]
        score = hl.load(
            scores,
            [tile_b, tile_h, tile_q, tile_k],
            extra_mask=inside[None, None, :, :],
        ).to(torch.float32)
        position = hl.load(rel_index, [tile_q, tile_k], extra_mask=inside)
        position = torch.where(position < 0, position + table_rows, position)
        bias = hl.load(
            rel_table,
            [position[None, :, :], tile_h.index[:, None, None]],
            extra_mask=inside[None, :, :],
        ).to(torch.float32)
        x = score + bias[None, :, :, :]
        x_max = hl.load(
            row_max, [tile_b, tile_h, tile_q], extra_mask=in_rows[None, None, :]
        )
        x_sum = hl.load(
            row_sum, [tile_b, tile_h, tile_q], extra_mask=in_rows[None, None, :]
        )
        probs = torch.exp(x - x_max[:, :, :, None]) / x_sum[:, :, :, None]
        out[tile_b, tile_h, tile_q, tile_k] = torch.where(
            inside[None, None, :, :], probs, 0.0
        ).to(out.dtype)
    return out


def forward(inputs: list) -> tuple[torch.Tensor, ...]:
    padded_scores, rel_index, rel_table = inputs[:3]
    view_shape = inputs[3]  # _shape_param_0 = [B, H, Q, K]
    pad = inputs[7]  # _shape_param_4 = [0, pad_k, 0, pad_q, 0, 0]
    if list(pad) != [0, pad[1], 0, pad[3], 0, 0]:
        raise ValueError(f"unexpected pad {pad}")
    batch, heads, q_len, k_len = view_shape
    # slice_1, slice_2: the graph's literal 0:-7 on both window dims.
    scores = padded_scores[:, :-7, :-7].view(view_shape)
    row_max, row_sum = window_softmax_stats(scores, rel_index, rel_table)
    out = window_softmax_padded(
        scores, rel_index, rel_table, row_max, row_sum, pad[3], pad[1]
    )
    # view_3 and constant_pad_nd: [B*H, Q + pad_q, K + pad_k].
    return (out.view(batch * heads, q_len + pad[3], k_len + pad[1]),)
