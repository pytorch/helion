"""better-benchmark repro var_mean_3a2d56cdea00 (hf/infer/GPTNeoForCausalLM).

At the ported shape point (f2c837cd, hidden 2048) Inductor emits one looped
reduction,
``triton_red_fused_add_convert_element_type_mul_rsqrt_sub_var_mean_view_0``:
the GPT block's residual add (a ``[batch * seq, hidden]`` projection output
viewed to ``[batch, seq, hidden]`` plus the stream of that shape), returned in
bf16, feeding a LayerNorm over the hidden dim. The first loop over the hidden
dim loads both bf16 inputs, adds them in fp32, stores the sum in bf16 and
folds the fp32 sum (Inductor does not round it to bf16 before the statistics)
into the row's population ``var_mean``. The second loop reloads the stored
bf16 sum, so the normalize starts from the rounded value, and stores
``(sum - mean) * rsqrt(var + 1e-5) * weight + bias`` in bf16 as a
``[batch * seq, hidden]`` view. The kernel writes only the two bf16 outputs;
the statistics stay in registers.

The port reduces over a full slice, like the Welford partials of
var_mean_42fad1ece813: the row's mean and the mean of its squared deviations
from that mean, both divided by the count as Inductor's Welford statistics
are. A looped config rolls the reductions into loops over the hidden dim, and
the autotuner may also pick a persistent reduction.

At the two hidden 768 shape points (63bebcf6, 1cea4d76) Inductor emits a
persistent reduction instead, which normalizes the fp32 sum without reading
it back; the port follows the looped form.
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(static_shapes=True)
def residual_add_layernorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # x, residual: [rows, hidden] bf16; weight, bias: [hidden] bf16.
    rows, hidden = x.shape
    add_out = torch.empty([rows, hidden], dtype=x.dtype, device=x.device)
    norm_out = torch.empty([rows, hidden], dtype=x.dtype, device=x.device)
    for tile_r in hl.tile(rows):
        summed = x[tile_r, :].to(torch.float32) + residual[tile_r, :].to(torch.float32)
        add_out[tile_r, :] = summed.to(add_out.dtype)
        # torch.mean divides by the count, as Inductor does (a constant
        # divisor would become a reciprocal multiply).
        mean = torch.mean(summed, dim=-1, keepdim=True)
        centered = summed - mean
        var = torch.mean(centered * centered, dim=-1, keepdim=True)
        # Like Inductor's second loop: the sum comes back from the bf16
        # output, the statistics are those of the unrounded fp32 sum.
        stored = add_out[tile_r, :].to(torch.float32)
        y = (stored - mean) * torch.rsqrt(var + 1e-5)
        y = y * weight[None, :].to(torch.float32) + bias[None, :].to(torch.float32)
        norm_out[tile_r, :] = y.to(norm_out.dtype)
    return add_out, norm_out


def forward(inputs: list) -> tuple[torch.Tensor, ...]:
    x, residual, weight, bias = inputs[:4]
    out_shape = inputs[5]  # _shape_param_1 = [batch * seq, hidden]
    batch, seq, hidden = residual.shape
    add_out, norm_out = residual_add_layernorm(
        x, residual.view(batch * seq, hidden), weight, bias
    )
    return add_out.view(batch, seq, hidden), norm_out.view(out_shape)
