"""better-benchmark repro var_mean_e45f42fbfc2b (hf GoogleFnet inference).

Inductor emits one persistent reduction,
``triton_per_fused_add_mul_rsqrt_select_sub_var_mean_0``: a residual add of
the hidden states ``[B, S, C]`` and component 0 of a ``[B, S, C, 2]`` tensor
(``select`` on the last dim, a stride-2 load), followed by a LayerNorm over C
(fp32 statistics, eps 1e-12, affine). Only the normalized result is stored;
the graph returns it as ``[B, S, C]`` and as its ``[B*S, C]`` view.
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(static_shapes=True)
def select_add_layernorm(
    hidden: torch.Tensor,
    pairs: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    # hidden: [B*S, C]; pairs: [B*S, C, 2], views of the graph's inputs.
    rows, channels = hidden.shape
    out = torch.empty([rows, channels], dtype=hidden.dtype, device=hidden.device)
    for tile_r in hl.tile(rows):
        x = hidden[tile_r, :].to(torch.float32) + pairs[tile_r, :, 0].to(torch.float32)
        # torch.mean divides by the count, as Inductor does (a constant
        # divisor would become a reciprocal multiply).
        mean = torch.mean(x, dim=-1, keepdim=True)
        centered = x - mean
        var = torch.mean(centered * centered, dim=-1, keepdim=True)
        y = centered * torch.rsqrt(var + 1e-12)
        y = y * weight[None, :].to(torch.float32) + bias[None, :].to(torch.float32)
        out[tile_r, :] = y.to(out.dtype)
    return out


def forward(inputs: list) -> tuple[torch.Tensor, ...]:
    pairs, hidden, weight, bias = inputs[:4]
    flat_shape = inputs[4]  # _shape_param_0 = [B*S, C]
    batch, seq, channels = hidden.shape
    rows = batch * seq
    out = select_add_layernorm(
        hidden.view(rows, channels), pairs.view(rows, channels, 2), weight, bias
    )
    add_2 = out.view(batch, seq, channels)
    return add_2, add_2.view(flat_shape)
