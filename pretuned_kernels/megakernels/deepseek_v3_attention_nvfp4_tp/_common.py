"""Shared source and fixtures for the DeepSeek-V3 TP4 attention boundary."""

from __future__ import annotations

from pretuned_kernels.nvfp4_gemv import nvfp4_gemv as nvfp4
import torch

import helion.language as hl

WORLD_SIZE = 4
TOKENS = 1
TOTAL_ATTENTION_WIDTH = 128 * 128
LOCAL_K = TOTAL_ATTENTION_WIDTH // WORLD_SIZE
OUTPUT_FEATURES = 7168
OUTPUT_BLOCK = 8
K_GROUP_BLOCK = 256
COMMUNICATION_BLOCK = 256
NORM_BLOCK = 1024
RMS_EPS = 1e-6
SIGNAL_PAD_BYTES = 32 * 1024
FP4_MAX = 6.0
FP8_MAX = float(torch.finfo(torch.float8_e4m3fn).max)


def attention_boundary_source(
    weight_bytes: torch.Tensor,
    activation_bytes: torch.Tensor,
    weight_scale_bytes: torch.Tensor,
    activation_scale_bytes: torch.Tensor,
    alpha: float,
    symmetric_output: torch.Tensor,
    residual: torch.Tensor,
    gamma: torch.Tensor,
    group_name: hl.ProcessGroupName,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse the rank-local projection, TP4 sum, residual add, and RMSNorm."""
    # Model geometry and layouts are fixed; tensor contents remain runtime data.
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
    assert symmetric_output.size(0) == TOKENS
    assert symmetric_output.size(1) == output_features == OUTPUT_FEATURES
    k_groups = packed_k // 8
    output_block = hl.register_block_size(1, 16)
    group_block = hl.register_block_size(k_groups)

    hl.specialize(residual.size(0))
    hl.specialize(residual.size(1))
    hl.specialize(residual.stride())
    hl.specialize(gamma.size(0))
    hl.specialize(gamma.stride())
    assert residual.size() == symmetric_output.size()
    assert gamma.size(0) == output_features
    assert symmetric_output.stride(1) == 1
    assert symmetric_output.storage_offset() == 0

    post_groups = (output_features + COMMUNICATION_BLOCK - 1) // COMMUNICATION_BLOCK
    # Each projection tile stores one slice of a dense 1-D view, which the
    # compiler pushes to every rank with its readiness tag in band.
    local_output = torch.as_strided(
        symmetric_output, (output_features,), (1,), storage_offset=0
    )
    peer_outputs = torch.ops.symm_mem.get_remote_tensors(local_output, group_name)
    assert len(peer_outputs) == WORLD_SIZE
    unrounded = torch.empty(
        (TOKENS, output_features), dtype=torch.float32, device=symmetric_output.device
    )
    rms_partials = torch.empty(
        (TOKENS, post_groups), dtype=torch.float32, device=symmetric_output.device
    )
    residual_out = torch.empty_like(residual)
    norm_out = torch.empty_like(residual)

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
        local_output[tile_n] = (accumulator * alpha).to(torch.bfloat16)

    for communication_n in hl.tile(output_features, block_size=COMMUNICATION_BLOCK):
        # Canonical rank order gives every rank bit-identical accumulation.
        total = hl.zeros([communication_n], dtype=torch.float32)
        for peer_output in peer_outputs:
            total = total + peer_output[communication_n].to(torch.float32)
        values = total + residual[0, communication_n].to(torch.float32)
        unrounded[0, communication_n] = values
        residual_out[0, communication_n] = values.to(torch.bfloat16)
        rms_partials[0, communication_n.id] = torch.sum(values * values, dim=-1)

    for norm_n in hl.tile(output_features, block_size=NORM_BLOCK):
        square_sum = torch.sum(rms_partials[0, :], dim=-1)
        inv_rms = torch.rsqrt(square_sum * (1.0 / output_features) + RMS_EPS)
        normalized = (unrounded[0, norm_n] * inv_rms).to(torch.bfloat16)
        norm_out[0, norm_n] = normalized * gamma[norm_n]

    return norm_out, residual_out


def quantize_inputs(rank: int) -> tuple[torch.Tensor, ...]:
    """Create deterministic rank-local tensors in vLLM's NVFP4 layouts."""
    from vllm import _custom_ops as vllm_ops

    generator = torch.Generator(device="cuda")
    generator.manual_seed(20260926 + rank)
    activation = torch.randn(
        (TOKENS, LOCAL_K),
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    weight = (
        torch.randn(
            (OUTPUT_FEATURES, LOCAL_K),
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        / 16
    )
    activation_global_scale = FP8_MAX * FP4_MAX / activation.abs().max().float()
    weight_global_scale = FP8_MAX * FP4_MAX / weight.abs().max().float()
    activation_fp4, activation_scale = vllm_ops.scaled_fp4_quant(
        activation, activation_global_scale
    )
    weight_fp4, weight_scale = vllm_ops.scaled_fp4_quant(weight, weight_global_scale)
    alpha = (1.0 / (activation_global_scale * weight_global_scale)).float()
    return activation_fp4, activation_scale, weight_fp4, weight_scale, alpha
