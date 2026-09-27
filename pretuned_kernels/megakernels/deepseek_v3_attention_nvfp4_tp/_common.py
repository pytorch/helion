# ruff: noqa: ANN001, ANN202
# pyrefly: ignore-errors
"""Shared source and fixtures for the DeepSeek-V3 TP4 attention boundary."""

from __future__ import annotations

from pretuned_kernels.nvfp4_gemv import nvfp4_gemv as nvfp4
import torch
import triton
import triton.language as tl

import helion.language as hl
from helion.runtime.triton.dist_utils import _load_volatile_u32
from helion.runtime.triton.dist_utils import _store_relaxed_sys_u32

WORLD_SIZE = 4
TOKENS = 1
TOTAL_ATTENTION_WIDTH = 128 * 128
LOCAL_K = TOTAL_ATTENTION_WIDTH // WORLD_SIZE
OUTPUT_FEATURES = 7168
OUTPUT_BLOCK = 8
K_GROUP_BLOCK = 256
COMMUNICATION_BLOCK = 1024
NORM_BLOCK = 1024
RING_SLOTS = 3
RMS_EPS = 1e-6
SIGNAL_PAD_BYTES = 32 * 1024
FP4_MAX = 6.0
FP8_MAX = float(torch.finfo(torch.float8_e4m3fn).max)


@triton.jit
def _publish_chunk(
    destination_workspace,
    epoch,
    ring_slot,
    source_rank,
    world_size: tl.constexpr,
    workspace_stride,
    communication_block: tl.constexpr,
    begin,
    value,
):
    packed = value.to(tl.uint16, bitcast=True).to(tl.uint32) | (
        epoch.to(tl.uint32) << 16
    )
    indices = begin + tl.arange(0, communication_block)
    pointer = (
        destination_workspace
        + (ring_slot * world_size + source_rank) * workspace_stride
        + indices
    ).to(tl.pointer_type(tl.uint32))
    _store_relaxed_sys_u32(pointer, packed)


@triton.jit
def _pull_chunk(
    local_workspace,
    epoch,
    ring_slot,
    source_rank,
    world_size: tl.constexpr,
    workspace_stride,
    communication_block: tl.constexpr,
    begin,
):
    indices = begin + tl.arange(0, communication_block)
    pointer = (
        local_workspace
        + (ring_slot * world_size + source_rank) * workspace_stride
        + indices
    ).to(tl.pointer_type(tl.uint32))
    packed = _load_volatile_u32(pointer)
    expected = epoch.to(tl.uint16)
    ready = tl.sum((packed >> 16).to(tl.uint16) == expected, 0)
    while ready != communication_block:
        packed = _load_volatile_u32(pointer)
        ready = tl.sum((packed >> 16).to(tl.uint16) == expected, 0)
    value = (packed & 0xFFFF).to(tl.uint16).to(tl.bfloat16, bitcast=True).to(tl.float32)
    _store_relaxed_sys_u32(pointer, tl.zeros((communication_block,), tl.uint32))
    return value


def attention_boundary_source(
    weight_bytes: torch.Tensor,
    activation_bytes: torch.Tensor,
    weight_scale_bytes: torch.Tensor,
    activation_scale_bytes: torch.Tensor,
    alpha: float,
    symmetric_output: torch.Tensor,
    handoff_workspace: torch.Tensor,
    ring_state: torch.Tensor,
    residual: torch.Tensor,
    gamma: torch.Tensor,
    rank: hl.constexpr,
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

    hl.specialize(handoff_workspace.size(0))
    hl.specialize(handoff_workspace.size(1))
    hl.specialize(handoff_workspace.size(2))
    hl.specialize(handoff_workspace.stride())
    hl.specialize(ring_state.size(0))
    hl.specialize(ring_state.stride())
    hl.specialize(residual.size(0))
    hl.specialize(residual.size(1))
    hl.specialize(residual.stride())
    hl.specialize(gamma.size(0))
    hl.specialize(gamma.stride())
    assert residual.size() == symmetric_output.size()
    assert gamma.size(0) == output_features
    assert handoff_workspace.dtype == torch.uint32
    assert handoff_workspace.size(0) == RING_SLOTS
    assert handoff_workspace.size(1) == WORLD_SIZE
    assert handoff_workspace.size(2) >= output_features
    assert handoff_workspace.stride(2) == 1
    assert ring_state.dtype == torch.int32 and ring_state.size(0) == 2

    post_groups = (output_features + COMMUNICATION_BLOCK - 1) // COMMUNICATION_BLOCK
    workspace_stride = handoff_workspace.size(2)
    destination_workspaces = torch.ops.symm_mem.get_remote_tensors(
        handoff_workspace, group_name
    )
    assert len(destination_workspaces) == WORLD_SIZE
    unrounded = torch.empty(
        (TOKENS, output_features), dtype=torch.float32, device=symmetric_output.device
    )
    rms_partials = torch.empty(
        (TOKENS, post_groups), dtype=torch.float32, device=symmetric_output.device
    )
    advance_tokens = torch.empty(
        (post_groups,), dtype=torch.int32, device=symmetric_output.device
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
        symmetric_output[0, tile_n] = (accumulator * alpha).to(torch.bfloat16)

    for communication_n in hl.tile(output_features, block_size=COMMUNICATION_BLOCK):
        local_value = symmetric_output[0, communication_n]
        epoch = ring_state[0]
        ring_slot = ring_state[1]
        for destination_rank, destination_workspace in enumerate(
            destination_workspaces
        ):
            if destination_rank != rank:
                hl.triton_kernel(
                    _publish_chunk,
                    args=(
                        destination_workspace,
                        epoch,
                        ring_slot,
                        rank,
                        WORLD_SIZE,
                        workspace_stride,
                        COMMUNICATION_BLOCK,
                        communication_n.begin,
                        local_value,
                    ),
                    output_like=None,
                )

        # Canonical rank order gives every rank bit-identical accumulation.
        total = hl.zeros([communication_n], dtype=torch.float32)
        for source_rank in range(WORLD_SIZE):
            if source_rank == rank:
                total = total + local_value.to(torch.float32)
            else:
                source_value = hl.triton_kernel(
                    _pull_chunk,
                    args=(
                        handoff_workspace,
                        epoch,
                        ring_slot,
                        source_rank,
                        WORLD_SIZE,
                        workspace_stride,
                        COMMUNICATION_BLOCK,
                        communication_n.begin,
                    ),
                    output_like=hl.zeros([communication_n], dtype=torch.float32),
                )
                total = total + source_value

        values = total + residual[0, communication_n].to(torch.float32)
        unrounded[0, communication_n] = values
        residual_out[0, communication_n] = values.to(torch.bfloat16)
        rms_partials[0, communication_n.id] = torch.sum(values * values, dim=-1)
        advance_tokens[communication_n.id] = 1

    for norm_n in hl.tile(output_features, block_size=NORM_BLOCK):
        square_sum = torch.sum(rms_partials[0, :], dim=-1)
        inv_rms = torch.rsqrt(square_sum * (1.0 / output_features) + RMS_EPS)
        normalized = (unrounded[0, norm_n] * inv_rms).to(torch.bfloat16)
        norm_out[0, norm_n] = normalized * gamma[norm_n]

    for _advance in hl.grid(1):
        completed = torch.sum(advance_tokens[:])
        epoch_step = completed - post_groups + 1
        ring_state[0] = (ring_state[0] - 1 + epoch_step) % 65535 + 1
        ring_state[1] = (ring_state[1] + epoch_step) % RING_SLOTS

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
