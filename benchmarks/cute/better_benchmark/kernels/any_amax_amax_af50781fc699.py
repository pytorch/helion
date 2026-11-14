"""better-benchmark repro any_amax_amax_af50781fc699 (hf/infer XLNetLMHeadModel).

Inductor emits one persistent reduction,
``triton_per_fused__to_copy_abs_add_all_amax_any_convert_element_type_div_eq_exp_index_iota_logical_not_mul_ne_permute_slice_sub_sum_view_where_0``:
the attention probabilities of XLNet's relative attention. Per row of the
``[B, H, Q, K]`` scores it adds the bf16 content scores (``arg0_1``
``[B*H, Q, K]`` through no-op view/permute/view aliases) and the
relative-shifted position scores: ``arg1_1`` ``[B*H, Q, 2Q]``, aliased as
``[B, H, Q, 2Q]``, is viewed as ``[B, H, 2Q, Q]``, its first row dropped
(``slice``), viewed back as ``[B, H, Q, 2Q - 1]`` and indexed by
``iota(K)`` on the last dim, so element ``(g, q, k)`` (``g = b * H + h``)
reads ``arg1_1`` at flat offset ``Q + k + (2Q - 1) * q + 2Q * Q * g``.
Inductor turns that chain into a strided load; the port builds the same
chain on the host and passes the kernel the strided ``[B*H, Q, K]`` view.
On the sum ``x`` (fp32: the graph's bf16 ``add`` and ``* 0.125`` are not
rounded to bf16, and ``+ 0`` and ``* 1`` fold away, as in Inductor) the row
computes four reductions: whether every scaled score ``x * 0.125`` is finite
(``logical_not(any(logical_not(eq & ne)))``), ``amax(x)``,
``amax(x * 0.125)`` and, after
``where(all_finite, (x - amax(x)) * 0.125, x * 0.125 - amax(x * 0.125))``,
the sum of its ``exp``. The probabilities ``exp / sum`` are stored in bf16
as the contiguous ``[B*H, Q, K]`` output.

Inductor's pattern pass rewrites the graph's ``x * 0.125 - amax(x * 0.125)``
branch into a second ``where`` on the same finiteness test, which adds a
duplicate ``amax(x * 1)``. That inner ``where`` never changes the result
(both use the same condition), so the port follows the repro graph's single
``where``.
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(static_shapes=True)
def relative_scaled_softmax(
    content: torch.Tensor, position: torch.Tensor
) -> torch.Tensor:
    # content: [B*H, Q, K] bf16 (contiguous); position: [B*H, Q, K] bf16, the
    # relative-shifted view of the position scores (strides [2Q*Q, 2Q-1, 1],
    # storage offset Q).
    groups, q_len, k_len = content.shape
    out = torch.empty(
        [groups, q_len, k_len], dtype=content.dtype, device=content.device
    )
    for tile_g, tile_q in hl.tile([groups, q_len]):
        x = content[tile_g, tile_q, :].to(torch.float32) + position[
            tile_g, tile_q, :
        ].to(torch.float32)
        scaled = x * 0.125
        finite = (scaled == scaled) & (torch.abs(scaled) != float("inf"))
        any_nonfinite = torch.any(torch.logical_not(finite), dim=-1, keepdim=True)
        all_finite = torch.logical_not(any_nonfinite)
        amax = torch.amax(x, dim=-1, keepdim=True)
        scaled_amax = torch.amax(scaled, dim=-1, keepdim=True)
        shifted = torch.where(all_finite, (x - amax) * 0.125, scaled - scaled_amax)
        e = torch.exp(shifted)
        denom = torch.sum(e, dim=-1, keepdim=True)
        out[tile_g, tile_q, :] = (e / denom).to(out.dtype)
    return out


def forward(inputs: list) -> tuple[torch.Tensor, ...]:
    arg0_1, arg1_1 = inputs[:2]
    # _shape_param_0..6: [B, H, Q, 1, K], [B, H, Q, K], [B, H, Q, 1, 2Q],
    # [B, H, Q, 2Q], [B, H, 2Q, Q], [B, H, Q, 2Q - 1], [B*H, Q, K].
    shape0, shape1, shape2, shape3, shape4, shape5, out_shape = inputs[2:9]
    content = arg0_1.view(shape0).permute(0, 1, 2, 4, 3).view(shape1)
    position = arg1_1.view(shape2).permute(0, 1, 2, 4, 3).view(shape3)
    # The relative shift (view_4, slice_1, view_5), then index(iota(K)).
    k_len = shape1[-1]
    position = position.view(shape4)[:, :, 1:].view(shape5)[..., :k_len]
    groups, q_len, _ = out_shape
    out = relative_scaled_softmax(
        content.view(groups, q_len, k_len), position.view(groups, q_len, k_len)
    )
    return (out.view(out_shape),)
