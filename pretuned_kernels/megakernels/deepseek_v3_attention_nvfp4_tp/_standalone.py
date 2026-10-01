"""Matched standalone Helion kernels for the TP4 attention boundary."""

from __future__ import annotations

from pretuned_kernels.megakernels.deepseek_v3_attention_nvfp4_tp._common import (
    K_GROUP_BLOCK,
)
from pretuned_kernels.megakernels.deepseek_v3_attention_nvfp4_tp._common import (
    OUTPUT_BLOCK,
)
from pretuned_kernels.nvfp4_gemv import nvfp4_gemv as nvfp4
import torch

import helion
import helion.language as hl


def _attention_o_proj_local(
    weight_bytes: torch.Tensor,
    activation_bytes: torch.Tensor,
    weight_scale_bytes: torch.Tensor,
    activation_scale_bytes: torch.Tensor,
    alpha: float,
    symmetric_output: torch.Tensor,
    group_name: hl.ProcessGroupName,
) -> torch.Tensor:
    """Matched rank-local projection without communication or normalization."""
    hl.specialize(weight_bytes.size(0))
    hl.specialize(weight_bytes.size(1))
    hl.specialize(activation_bytes.size(0))
    hl.specialize(weight_scale_bytes.size(0))
    hl.specialize(activation_scale_bytes.size(0))
    hl.specialize(symmetric_output.size(0))
    hl.specialize(symmetric_output.size(1))
    hl.specialize(weight_bytes.stride())
    hl.specialize(activation_bytes.stride())
    hl.specialize(weight_scale_bytes.stride())
    hl.specialize(activation_scale_bytes.stride())
    hl.specialize(symmetric_output.stride())

    output_features, packed_k = weight_bytes.size()
    assert activation_bytes.size(0) == packed_k
    k_groups = packed_k // 8
    output_block = hl.register_block_size(1, 16)
    group_block = hl.register_block_size(k_groups)

    for tile_n in hl.tile(output_features, block_size=output_block):
        accumulator = hl.zeros([tile_n], dtype=torch.float32)
        for tile_g in hl.tile(k_groups, block_size=group_block):
            group_offsets = tile_n.index[:, None] * k_groups + tile_g.index[None, :]
            group_mask = tile_g.index < k_groups
            weight_mask = (tile_n.index[:, None] < output_features) & group_mask[
                None, :
            ]
            weight = hl.load_float4_e2m1fn_x16_to_float16(
                weight_bytes, group_offsets, extra_mask=weight_mask
            )
            activation = hl.load_float4_e2m1fn_x16_to_float16(
                activation_bytes, tile_g.index, extra_mask=group_mask
            )
            contribution = hl.zeros([output_block, group_block], dtype=torch.float16)
            for lane in hl.static_range(16):
                contribution = contribution + weight[lane] * activation[lane][None, :]
            weight_scale_offsets = nvfp4.swizzled_scale_offsets(
                tile_n.index[:, None], tile_g.index[None, :], k_groups
            )
            activation_scale_offsets = nvfp4.swizzled_scale_offsets(
                tile_g.index * 0, tile_g.index, k_groups
            )
            scale = nvfp4._e4m3_byte_to_f32(weight_scale_bytes[weight_scale_offsets])
            scale = (
                scale
                * nvfp4._e4m3_byte_to_f32(
                    activation_scale_bytes[activation_scale_offsets]
                )[None, :]
            )
            accumulator = accumulator + (contribution.to(torch.float32) * scale).sum(-1)
        symmetric_output[0, tile_n] = (accumulator * alpha).to(torch.bfloat16)
    return symmetric_output


attention_o_proj_local = helion.kernel(
    _attention_o_proj_local,
    config=helion.Config(
        block_sizes=[OUTPUT_BLOCK, K_GROUP_BLOCK],
        num_stages=3,
        num_warps=2,
        pid_type="flat",
        range_multi_buffers=[None, True],
    ),
    static_shapes=False,
    backend="triton",
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)


@helion.kernel(
    config=helion.Config(block_sizes=[1], num_warps=8, pid_type="flat"),
    static_shapes=False,
    backend="triton",
)
def residual_rms_norm(
    reduced: torch.Tensor,
    residual: torch.Tensor,
    gamma: torch.Tensor,
    group_name: hl.ProcessGroupName,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match FlashInfer's residual-add plus FP32-accumulating RMSNorm tail."""
    hl.specialize(reduced.size(0))
    hl.specialize(reduced.size(1))
    hl.specialize(reduced.stride())
    hl.specialize(residual.size(0))
    hl.specialize(residual.size(1))
    hl.specialize(residual.stride())
    hl.specialize(gamma.size(0))
    hl.specialize(gamma.stride())

    rows, output_features = reduced.size()
    assert residual.size() == reduced.size()
    assert gamma.size(0) == output_features
    norm_out = torch.empty_like(reduced)
    residual_out = torch.empty_like(reduced)
    for row in hl.tile(rows):
        value = reduced[row, :].to(torch.float32) + residual[row, :].to(torch.float32)
        residual_out[row, :] = value.to(torch.bfloat16)
        square_sum = torch.sum(value * value, dim=-1)
        inv_rms = torch.rsqrt(square_sum * (1.0 / output_features) + 1e-6)
        norm_out[row, :] = value.to(torch.bfloat16) * inv_rms[:, None] * gamma[None, :]
    return norm_out, residual_out
