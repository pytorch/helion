"""better-benchmark repro var_mean_004f9a9ea198 (timm swin_base_patch4_window7_224).

Inductor emits one persistent reduction,
``triton_per_fused_add_clone_convert_element_type_fmod_index_iota_mul_permute_rsqrt_sub_var_mean_view_0``:
the Swin window reverse (view/permute/clone of ``[B*16*49, C]`` windows back to
``[B, 28, 28, C]``) with the cyclic shift (``index`` by ``(iota + 25) % 28`` on
both spatial dims) gathered into a residual add, returned in bf16, followed by
a LayerNorm over C (fp32 statistics, eps 1e-5, affine) stored in bf16.
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(static_shapes=True)
def swin_reverse_roll_add_layernorm(
    windows: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    window: hl.constexpr,
    shift: hl.constexpr,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, height, width, channels = residual.shape
    rows = batch * height * width
    flat_residual = residual.view(rows, channels)
    add_out = torch.empty(
        [rows, channels], dtype=residual.dtype, device=residual.device
    )
    norm_out = torch.empty(
        [rows, channels], dtype=residual.dtype, device=residual.device
    )
    for tile_r in hl.tile(rows):
        row = tile_r.index
        b = row // (height * width)
        h = (row // width) % height
        w = row % width
        src_h = (h + shift) % height
        src_w = (w + shift) % width
        src = (
            (
                (b * (height // window) + src_h // window) * (width // window)
                + src_w // window
            )
            * window
            + src_h % window
        ) * window + src_w % window
        x = flat_residual[tile_r, :].to(torch.float32) + windows[src, :].to(
            torch.float32
        )
        add_out[tile_r, :] = x.to(residual.dtype)
        # torch.mean divides by the count, as Inductor does (a constant
        # divisor would become a reciprocal multiply).
        mean = torch.mean(x, dim=-1, keepdim=True)
        centered = x - mean
        var = torch.mean(centered * centered, dim=-1, keepdim=True)
        y = centered * torch.rsqrt(var + 1e-5)
        y = y * weight[None, :].to(torch.float32) + bias[None, :].to(torch.float32)
        norm_out[tile_r, :] = y.to(residual.dtype)
    return add_out, norm_out


def forward(inputs: list) -> tuple[torch.Tensor, ...]:
    windows, residual, weight, bias = inputs[:4]
    window = inputs[5][1]  # _shape_param_1 = [-1, 7, 7, C]
    batch, height, width, channels = residual.shape
    add_out, norm_out = swin_reverse_roll_add_layernorm(
        windows, residual, weight, bias, window, 25
    )
    return add_out.view(batch, height * width, channels), norm_out
