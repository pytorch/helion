"""better-benchmark repro var_mean_b19e82f2a47b (hf/infer/GPTNeoForCausalLM).

Inductor emits one looped reduction,
``triton_red_fused_add_convert_element_type_mul_rsqrt_sub_var_mean_view_0``:
the transformer block's residual add (the ``[batch, seq, hidden]`` stream plus
a ``[batch * seq, hidden]`` projection output viewed to the same shape),
returned in bf16, followed by a LayerNorm over the hidden dim. At the ported
shape point (f2c837cd, hidden 2048; likewise at the two hidden 2560 points)
the reduction is looped rather than persistent, with two passes over each row:

- the first loads both bf16 inputs, adds them in fp32, stores the sum in bf16
  and accumulates the row's population ``var_mean`` of the fp32 sum (Inductor
  does not round it to bf16 before the statistics);
- the second reloads the stored bf16 sum (not the two inputs) with the bf16
  weight and bias and stores
  ``(sum - mean) * rsqrt(var + 1e-5) * weight + bias`` in bf16, returned as
  three ``[batch * seq, hidden]`` views of one buffer.

So the statistics come from the fp32 sum and the normalized values from the
bf16-rounded one, and the kernel reads the two inputs, its own first output,
the weight and the bias, and writes only the two bf16 outputs. The port keeps
that: the row statistics are reductions over the full row of the fp32 sum
(``mean = sum(x) / hidden``, ``var = sum((x - mean)**2) / hidden``; Inductor
accumulates the same two numbers with a Welford update), which leaves
persistent versus looped to the autotuner, and the normalize reads the sum
back from the output it was stored to.

At the shape points with hidden <= 1024 Inductor emits a persistent reduction
instead, which normalizes the fp32 sum without reading it back; the port
follows the looped form.
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(static_shapes=True)
def residual_add_layernorm(
    residual: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # residual, x: [batch * seq, hidden] bf16; weight, bias: [hidden] bf16.
    rows, hidden = x.shape
    add_out = torch.empty([rows, hidden], dtype=x.dtype, device=x.device)
    norm_out = torch.empty([rows, hidden], dtype=x.dtype, device=x.device)
    for tile_r in hl.tile(rows):
        summed = residual[tile_r, :].to(torch.float32) + x[tile_r, :].to(torch.float32)
        add_out[tile_r, :] = summed.to(add_out.dtype)
        # torch.mean divides by the count, as Inductor does (a constant
        # divisor would become a reciprocal multiply).
        mean = torch.mean(summed, dim=-1, keepdim=True)
        centered = summed - mean
        var = torch.mean(centered * centered, dim=-1, keepdim=True)
        # Inductor's second loop normalizes the bf16 sum it stored, read back
        # from the output buffer.
        stored = add_out[tile_r, :].to(torch.float32)
        y = (stored - mean) * torch.rsqrt(var + 1e-5)
        y = y * weight[None, :].to(torch.float32) + bias[None, :].to(torch.float32)
        norm_out[tile_r, :] = y.to(norm_out.dtype)
    return add_out, norm_out


def forward(inputs: list) -> tuple[torch.Tensor, ...]:
    x, residual, weight, bias = inputs[:4]
    # _shape_param_1.._shape_param_3 = [batch * seq, hidden]
    shape_1, shape_2, shape_3 = inputs[5:8]
    batch, seq, hidden = residual.shape
    add_out, norm_out = residual_add_layernorm(
        residual.view(batch * seq, hidden), x, weight, bias
    )
    return (
        add_out.view(batch, seq, hidden),
        norm_out.view(shape_1),
        norm_out.view(shape_2),
        norm_out.view(shape_3),
    )
