"""better-benchmark repro mean_var_mean_f541715270de (timm beit_base_patch16_224 inference).

BEiT's head: the last block's layer-scaled residual add, the mean over the
patch tokens (the class token is sliced off) and the ``fc_norm`` LayerNorm.
Inductor emits two kernels:

- ``triton_red_fused_add_mean_mul_slice_view_0``: a looped reduction per
  (batch, channel) over the ``T - 1`` patch tokens of ``residual + gamma *
  mlp`` (``arg2_1`` ``[B, T, C]``; ``arg0_1`` ``[B*T, C]`` viewed as
  ``[B, T, C]``; ``gamma`` ``[C]``), reading tokens ``1..T-1`` and storing
  the fp32 token sum to a ``[B, C]`` workspace.
- ``triton_per_fused_add_convert_element_type_mean_mul_rsqrt_slice_sub_var_mean_view_1``:
  a persistent reduction per workspace row: the pooled mean ``sum / (T - 1)``
  and a LayerNorm over C (two-pass mean and population variance, both true
  divisions by C; eps 1e-6; affine) stored in bf16.

Like Inductor (no ``emulate_precision_casts``), the graph's bf16 roundings of
``gamma * mlp``, of the residual add and of the pooled mean stay in fp32.
The class-token slice is a host view of both ``[B, T, C]`` inputs (storage
offset ``C``), so the first kernel reads the same elements as Inductor's.

Where Inductor's output equals the gate's (bf16) reference bit for bit, the
threshold is the gate's 1e-5 floor, below one bf16 ulp: a config whose
summation order differs (on either backend) can round an output next to a
rounding midpoint one bf16 ulp the other way, which the gate's one-ulp
allowance accepts.
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(static_shapes=True)
def layer_scale_residual_token_sum(
    mlp: torch.Tensor, gamma: torch.Tensor, residual: torch.Tensor
) -> torch.Tensor:
    # mlp, residual: [B, T - 1, C] bf16 views without the class token.
    batch, _, channels = residual.shape
    token_sum = torch.empty(
        [batch, channels], dtype=torch.float32, device=residual.device
    )
    for tile_b, tile_c in hl.tile([batch, channels]):
        scale = gamma[tile_c].to(torch.float32)[None, None, :]
        x = residual[tile_b, :, tile_c].to(torch.float32) + scale * mlp[
            tile_b, :, tile_c
        ].to(torch.float32)
        token_sum[tile_b, tile_c] = torch.sum(x, dim=1)
    return token_sum


@helion.kernel(static_shapes=True)
def pooled_mean_layernorm(
    token_sum: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    tokens: int,
) -> torch.Tensor:
    # token_sum: [B, C] fp32; weight, bias: [C] bf16. ``tokens`` stays a
    # runtime scalar: a constant divisor would become a reciprocal multiply,
    # while Inductor truly divides the sum by the token count (and torch.mean
    # divides by C).
    rows, channels = token_sum.shape
    out = torch.empty([rows, channels], dtype=weight.dtype, device=token_sum.device)
    for tile_m in hl.tile(rows):
        pooled = token_sum[tile_m, :] / tokens
        mean = torch.mean(pooled, dim=-1, keepdim=True)
        centered = pooled - mean
        var = torch.mean(centered * centered, dim=-1, keepdim=True)
        y = centered * torch.rsqrt(var + 1e-6)
        y = y * weight[None, :].to(torch.float32) + bias[None, :].to(torch.float32)
        out[tile_m, :] = y.to(out.dtype)
    return out


def forward(inputs: list) -> tuple[torch.Tensor, ...]:
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1 = inputs[:5]
    shape = inputs[5]  # _shape_param_0 = [B, T, C]
    mlp = arg0_1.view(shape)
    token_sum = layer_scale_residual_token_sum(mlp[:, 1:, :], arg1_1, arg2_1[:, 1:, :])
    out = pooled_mean_layernorm(token_sum, arg3_1, arg4_1, shape[1] - 1)
    return (out,)
