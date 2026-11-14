"""better-benchmark repro mean_var_236af0aff0ee (torchbench/train BERT_pytorch).

Inductor emits one persistent reduction,
``triton_per_fused_add_convert_element_type_div_gt_inductor_lookup_seed_inductor_random_mean_mul_sqrt_sub_var_view_0``:
a transformer block's residual add of its dropped-out sublayer output, the
block's output dropout, and the next LayerNorm (BERT_pytorch's hand-written
one: unbiased std, eps added to the std). Per row of the ``[B*S, H]``
activations it draws two dropout masks with ``tl.rand(seed, flat_index)``:
``gt`` from ``seeds[54]`` (``inductor_lookup_seed``), compared with 0.1 at
bf16 precision (against 0.1 rounded to bf16, with the sample left in fp32 as
Inductor does), and ``gt_1`` from ``seeds[55]``, compared with 0.1 in fp32.
It forms
``mul_3 = gt_1 * (residual + gt * x * 1.1111111111111112) * 1.1111111111111112``
(``x`` the bf16 sublayer output ``[B*S, H]`` viewed as ``[B, S, H]``,
``residual`` fp32 ``[B, S, H]``; the graph's bf16 dropout product stays fp32,
as Inductor leaves it), then the row mean, the centered row ``sub``, the
unbiased standard deviation ``sqrt(var(mul_3, correction=1))`` and the affine
normalize ``weight * sub / (std + 1e-6) + bias`` stored in bf16 and returned
as a ``[B*S, H]`` view. Both masks, ``mul_3``, ``std`` (``[B, S, 1]``) and
``sub`` are returned as well.

``hl.rand([tile, n], seed)`` uses the row-major flat index as its Philox
offset, like Inductor, and ``cute_rng_stream="word0"`` keeps Inductor's
stream on both backends (one Philox4x32-10 evaluation per element, keeping
word 0; see PORTING.md), so both masks are bit-identical to Inductor's.
Eager PyTorch draws ``inductor_random`` from its global generator instead,
so the checks compare all six outputs, which depend on the masks, with
Inductor's compiled outputs (``common.AnchoredGate``), the masks exactly.
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(static_shapes=True, cute_rng_stream="word0")
def dual_dropout_residual_layernorm(
    x: torch.Tensor,
    seeds: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    seed_index: hl.constexpr,
    seed_index_1: hl.constexpr,
    unbiased_count: int,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    # x: [rows, n] bf16; residual: [rows, n] fp32 (a contiguous view of the
    # [B, S, H] input); weight, bias: [n] fp32; seeds: Inductor's int64 seed
    # table. ``unbiased_count`` (n - 1, var's correction=1) stays a runtime
    # scalar: a constant divisor would become a reciprocal multiply, while
    # Inductor truly divides the squared-deviation sum by n - 1 (and
    # torch.mean truly divides by n).
    rows, n = x.shape
    keep_out = torch.empty([rows, n], dtype=torch.bool, device=x.device)
    keep_1_out = torch.empty([rows, n], dtype=torch.bool, device=x.device)
    dropped_out = torch.empty([rows, n], dtype=residual.dtype, device=x.device)
    std_out = torch.empty([rows], dtype=residual.dtype, device=x.device)
    centered_out = torch.empty([rows, n], dtype=residual.dtype, device=x.device)
    norm_out = torch.empty([rows, n], dtype=x.dtype, device=x.device)
    for tile_r in hl.tile(rows):
        # Both seeds are read before the first draw rather than each in its
        # draw: CuTe declined the natural order at the default config when
        # this port was made (a lane-invariant load between per-lane
        # statements of the variance's consume pass), and reading them first
        # changes no result (PORTING.md, "Leave the backend alone").
        seed = seeds[seed_index]
        seed_1 = seeds[seed_index_1]
        # gt_1: the fp32 sample of seeds[55] against 0.1.
        keep_1 = hl.rand([tile_r, n], seed=seed_1) > 0.1
        # gt(convert_element_type(rand, bf16), 0.1) runs at bf16 precision:
        # 0.1 rounds to 0.10009765625; like Inductor, the sample stays fp32.
        keep = hl.rand([tile_r, n], seed=seed) > 0.10009765625
        scaled = (
            keep.to(torch.float32) * x[tile_r, :].to(torch.float32) * 1.1111111111111112
        )
        added = residual[tile_r, :].to(torch.float32) + scaled
        dropped = keep_1.to(torch.float32) * added * 1.1111111111111112
        mean = torch.mean(dropped, dim=-1, keepdim=True)
        centered = dropped - mean
        var = torch.sum(centered * centered, dim=-1) / unbiased_count
        std = torch.sqrt(var)
        y = weight[None, :].to(torch.float32) * centered / (
            std[:, None] + 1e-06
        ) + bias[None, :].to(torch.float32)
        keep_out[tile_r, :] = keep
        keep_1_out[tile_r, :] = keep_1
        dropped_out[tile_r, :] = dropped
        std_out[tile_r] = std
        centered_out[tile_r, :] = centered
        norm_out[tile_r, :] = y.to(norm_out.dtype)
    return keep_out, keep_1_out, dropped_out, std_out, centered_out, norm_out


def forward(inputs: list) -> tuple[torch.Tensor, ...]:
    x, seeds, residual, weight, bias = inputs[:5]
    view_shape = inputs[5]  # _shape_param_0 = [B, S, H]
    out_shape = inputs[8]  # _shape_param_3 = [B*S, H]
    n = view_shape[-1]
    keep, keep_1, dropped, std, centered, norm = dual_dropout_residual_layernorm(
        x.view(-1, n), seeds, residual.view(-1, n), weight, bias, 54, 55, n - 1
    )
    return (
        keep.view(view_shape),
        keep_1.view(view_shape),
        dropped.view(view_shape),
        std.view([*view_shape[:-1], 1]),
        centered.view(view_shape),
        norm.view(out_shape),
    )
