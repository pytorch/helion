# ruff: noqa: ANN001, ANN202
# pyrefly: ignore-errors
"""Helion GPT-OSS native-MXFP4 MoE kernels for batch size one.

The four PDL-chained kernels mirror FlashInfer's monolithic TRT-LLM launch
boundaries: routing, GEMM1 plus OAI SwiGLU, GEMM2 plus bias, and weighted finalization.
The GEMMs consume the post-load FlashInfer weight, bias, and scale layouts
directly.  MXFP4 arithmetic uses native ``hl.dot_scaled`` tensor-core ops.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING

from pretuned_kernels.megakernels._pdl import launch_dependent
from pretuned_kernels.megakernels._pdl import signal_dependents
from pretuned_kernels.megakernels._pdl import wait_and_launch_dependents
import torch

import helion
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Callable


CONFIGS: dict[str, dict[str, object]] = {
    "routing": {
        "block_sizes": [],
        "load_eviction_policies": ["last", "first", "first", "last", "first"],
        "num_warps": 2,
        "num_stages": 8,
        "pid_type": "flat",
        "indexing": [
            "tensor_descriptor",
            "tensor_descriptor",
            "tensor_descriptor",
            "tensor_descriptor",
            "tensor_descriptor",
            "pointer",
            "tensor_descriptor",
        ],
        "atomic_indexing": [],
        "range_unroll_factors": [0],
        "range_warp_specializes": [None],
        "range_num_stages": [0],
        "range_multi_buffers": [None],
        "range_flattens": [None],
        "reduction_loops": [None],
    },
    "gemm1_swiglu_oai": {
        "block_sizes": [16, 32],
        "loop_orders": [[0, 1]],
        "l2_groupings": [1],
        "load_eviction_policies": ["", "", "", "", ""],
        "num_warps": 1,
        "num_stages": 1,
        "pid_type": "flat",
        "indexing": ["pointer"] * 6,
        "atomic_indexing": [],
        "range_unroll_factors": [0, 0],
        "range_warp_specializes": [None, None],
        "range_num_stages": [0, 0],
        "range_multi_buffers": [None, None],
        "range_flattens": [None, None],
    },
    "gemm2": {
        "block_sizes": [8, 32],
        "loop_orders": [[0, 1]],
        "l2_groupings": [1],
        "load_eviction_policies": ["", "", "", "", ""],
        "num_warps": 1,
        "num_stages": 2,
        "pid_type": "flat",
        "indexing": ["pointer"] * 6,
        "atomic_indexing": [],
        "range_unroll_factors": [0, 0],
        "range_warp_specializes": [None, None],
        "range_num_stages": [0, 0],
        "range_multi_buffers": [None, None],
        "range_flattens": [None, None],
    },
    "finalize": {
        "block_sizes": [128],
        "load_eviction_policies": ["last", "first", "last", ""],
        "num_warps": 8,
        "num_stages": 1,
        "pid_type": "persistent_interleaved",
        "num_sm_multiplier": 8,
        "maxnreg": 256,
        "indexing": [
            "tensor_descriptor",
            "tensor_descriptor",
            "tensor_descriptor",
            "pointer",
            "pointer",
        ],
        "atomic_indexing": [],
        "range_unroll_factors": [1],
        "range_warp_specializes": [None],
        "range_num_stages": [4],
        "range_multi_buffers": [None],
        "range_flattens": [True],
        "reduction_loops": [None],
    },
}


@dataclass(frozen=True)
class GptOssMoeShape:
    hidden: int = 3072
    output_hidden: int = 2880
    intermediate: int = 2944
    experts: int = 128
    top_k: int = 4


def _trtllm_row(logical_row):
    """Map a logical output row to TRT-LLM's 32-row epilogue shuffle."""
    lane = logical_row & 31
    return logical_row - lane + (lane & 3) * 8 + (lane >> 2)


def _trtllm_scale_offset(row, col, rows: int, cols: int):
    """Offset in FlashInfer's per-expert SWIZZLE_128_4_4 scale layout."""
    col_tiles = (cols + 3) // 4
    return (
        ((row >> 7) * col_tiles + (col >> 2)) * 512
        + (row & 31) * 16
        + ((row & 127) >> 5) * 4
        + (col & 3)
    )


def _e8m0_byte_to_f32(scale_byte: torch.Tensor) -> torch.Tensor:
    """Decode an unsigned E8M0 exponent byte without an SFU operation."""
    return (scale_byte.to(torch.int32) << 23).view(dtype=torch.float32)


@helion.kernel(
    static_shapes=True,
    autotune_effort="full",
    backend="triton",
)
def mxfp4_top4_routing(
    routing_logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-4 followed by softmax over only the selected logits.

    Four explicit argmax reductions avoid sorting all 128 experts.  This is
    equivalent to top-k for the distinct router logits used in inference and
    is substantially better suited to the B=1 case.
    """
    tokens, experts = routing_logits.size()
    top_k = 4
    hl.specialize(experts)
    weights = torch.empty(
        (tokens, top_k), dtype=torch.bfloat16, device=routing_logits.device
    )
    ids = torch.empty((tokens, top_k), dtype=torch.int32, device=routing_logits.device)
    for tile_t in hl.tile(tokens, block_size=1):
        signal_dependents()
        logits = routing_logits[tile_t, :].to(torch.float32)
        expert_index = hl.arange(experts)
        first_value = torch.amax(logits, dim=-1)
        first_id = torch.argmax(logits, dim=-1)
        remaining = torch.where(
            expert_index[None, :] == first_id[:, None],
            float("-inf"),
            logits,
        )
        second_value = torch.amax(remaining, dim=-1)
        second_id = torch.argmax(remaining, dim=-1)
        remaining = torch.where(
            expert_index[None, :] == second_id[:, None],
            float("-inf"),
            remaining,
        )
        third_value = torch.amax(remaining, dim=-1)
        third_id = torch.argmax(remaining, dim=-1)
        remaining = torch.where(
            expert_index[None, :] == third_id[:, None],
            float("-inf"),
            remaining,
        )
        fourth_value = torch.amax(remaining, dim=-1)
        fourth_id = torch.argmax(remaining, dim=-1)
        second_probability = torch.exp(second_value - first_value)
        third_probability = torch.exp(third_value - first_value)
        fourth_probability = torch.exp(fourth_value - first_value)
        denominator = 1.0 + second_probability + third_probability + fourth_probability
        route_slot = hl.arange(top_k)
        probabilities = torch.where(
            route_slot[None, :] == 0,
            (1.0 / denominator)[:, None],
            torch.where(
                route_slot[None, :] == 1,
                (second_probability / denominator)[:, None],
                torch.where(
                    route_slot[None, :] == 2,
                    (third_probability / denominator)[:, None],
                    (fourth_probability / denominator)[:, None],
                ),
            ),
        )
        selected_ids = torch.where(
            route_slot[None, :] == 0,
            first_id[:, None],
            torch.where(
                route_slot[None, :] == 1,
                second_id[:, None],
                torch.where(
                    route_slot[None, :] == 2,
                    third_id[:, None],
                    fourth_id[:, None],
                ),
            ),
        )
        weights[tile_t, route_slot] = probabilities.to(torch.bfloat16)
        ids[tile_t, route_slot] = selected_ids.to(torch.int32)
    return weights, ids


@helion.kernel(
    static_shapes=True,
    autotune_effort="full",
    backend="triton",
)
def mxfp4_moe_gemm1_swiglu_oai(
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w13_scale_bytes: torch.Tensor,
    w13_bias: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    """Selected-expert MXFP4 GEMM1 with fused FP32 bias and OAI SwiGLU."""
    tokens, hidden = hidden_states.size()
    experts, twice_intermediate, packed_hidden = w13.size()
    top_k = topk_ids.size(1)
    intermediate = twice_intermediate // 2
    scale_k = hidden // 32
    assert tokens == 1
    assert packed_hidden * 2 == hidden
    hl.specialize(experts)
    hl.specialize(top_k)
    hl.specialize(intermediate)
    output = torch.empty(
        (top_k, intermediate),
        dtype=torch.bfloat16,
        device=hidden_states.device,
    )
    flat_weight = w13.view(experts * twice_intermediate, packed_hidden)
    flat_bias = w13_bias.view(experts * twice_intermediate)
    flat_scale = w13_scale_bytes.view(-1)
    block_physical_row = hl.register_block_size(16, 128)
    block_scale_k = hl.register_block_size(1, 8)
    for tile_slot, tile_physical_row in hl.tile(
        [top_k, twice_intermediate], block_size=[1, block_physical_row]
    ):
        slot = tile_slot.begin
        expert = topk_ids[0, slot]
        expert_row = expert * twice_intermediate + tile_physical_row.index
        accumulator = hl.zeros([1, tile_physical_row], dtype=torch.float32)

        for tile_scale_k in hl.tile(scale_k, block_size=block_scale_k):
            byte_lane = hl.arange(16)
            weight_byte = tile_scale_k.index[:, None] * 16 + byte_lane[None, :]
            weight = hl.load(
                flat_weight,
                [
                    expert_row[:, None, None],
                    weight_byte[None, :, :],
                ],
            ).reshape(tile_physical_row, block_scale_k * 16)

            value_lane = hl.arange(32)
            hidden_index = tile_scale_k.index[:, None] * 32 + value_lane[None, :]
            activation = hl.load(hidden_states, [0, hidden_index]).reshape(
                1, block_scale_k * 32
            )
            activation_scale = hl.full([1, block_scale_k], 127, dtype=torch.uint8)
            scale_offset = expert * twice_intermediate * scale_k + _trtllm_scale_offset(
                tile_physical_row.index[:, None],
                tile_scale_k.index[None, :],
                twice_intermediate,
                scale_k,
            )
            scale = hl.load(flat_scale, [scale_offset])
            accumulator = hl.dot_scaled(
                activation,
                activation_scale,
                "bf16",
                weight.T,
                scale,
                "e2m1",
                acc=accumulator,
                out_dtype=torch.float32,
            )

        preactivation = accumulator.reshape(tile_physical_row)
        preactivation += flat_bias[expert_row]

        # Each contiguous 16-row half of a shuffle group contains eight
        # [up, gate] pairs.  Form the pairs locally, then scatter the eight
        # activations back to their logical output rows.  This formulation also
        # permits the GEMV-friendly 16-row tile in the autotuner search space.
        pairs = preactivation.reshape(block_physical_row // 16, 2, 8).permute(0, 2, 1)
        up_value, gate_value = hl.split(pairs)
        gate_value = torch.clamp(gate_value, max=7.0)
        up_value = torch.clamp(up_value, min=-7.0, max=7.0)
        activated = (up_value + 1.0) * gate_value * torch.sigmoid(1.702 * gate_value)
        chunk = hl.arange(block_physical_row // 16)[:, None]
        pair_lane = hl.arange(8)[None, :]
        global_chunk = tile_physical_row.begin // 16 + chunk
        output_row = (global_chunk // 2) * 16 + pair_lane * 2 + (global_chunk & 1)
        output[slot, output_row.reshape(block_physical_row // 2)] = activated.reshape(
            block_physical_row // 2
        ).to(torch.bfloat16)
    return output


@helion.kernel(
    static_shapes=True,
    autotune_effort="full",
    backend="triton",
)
def mxfp4_moe_gemm2(
    activation: torch.Tensor,
    w2: torch.Tensor,
    w2_scale_bytes: torch.Tensor,
    w2_bias: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    """Selected-expert MXFP4 GEMM2 with fused FP32 bias."""
    top_k, intermediate = activation.size()
    experts, hidden, packed_intermediate = w2.size()
    scale_k = intermediate // 32
    assert packed_intermediate * 2 == intermediate
    hl.specialize(experts)
    hl.specialize(top_k)
    hl.specialize(hidden)
    output = torch.empty(
        (top_k, hidden), dtype=torch.bfloat16, device=activation.device
    )
    flat_weight = w2.view(experts * hidden, packed_intermediate)
    flat_bias = w2_bias.view(experts * hidden)
    flat_scale = w2_scale_bytes.view(-1)
    block_physical_row = hl.register_block_size(16, 128)
    block_scale_k = hl.register_block_size(1, 8)
    for tile_slot, tile_physical_row in hl.tile(
        [top_k, hidden], block_size=[1, block_physical_row]
    ):
        slot = tile_slot.begin
        expert = topk_ids[0, slot]
        expert_row = expert * hidden + tile_physical_row.index
        accumulator = hl.zeros([1, tile_physical_row], dtype=torch.float32)
        for tile_scale_k in hl.tile(scale_k, block_size=block_scale_k):
            byte_lane = hl.arange(16)
            weight_byte = tile_scale_k.index[:, None] * 16 + byte_lane[None, :]
            weight = hl.load(
                flat_weight,
                [
                    expert_row[:, None, None],
                    weight_byte[None, :, :],
                ],
            ).reshape(tile_physical_row, block_scale_k * 16)
            value_lane = hl.arange(32)
            activation_index = tile_scale_k.index[:, None] * 32 + value_lane[None, :]
            values = hl.load(
                activation,
                [slot, activation_index],
            ).reshape(1, block_scale_k * 32)
            scale_offset = expert * hidden * scale_k + _trtllm_scale_offset(
                tile_physical_row.index[:, None],
                tile_scale_k.index[None, :],
                hidden,
                scale_k,
            )
            scale = hl.load(flat_scale, [scale_offset])
            activation_scale = hl.full([1, block_scale_k], 127, dtype=torch.uint8)
            accumulator = hl.dot_scaled(
                values,
                activation_scale,
                "bf16",
                weight.T,
                scale,
                "e2m1",
                acc=accumulator,
                out_dtype=torch.float32,
            )
        result = accumulator.reshape(tile_physical_row) + flat_bias[expert_row]
        physical_row = tile_physical_row.begin + hl.arange(block_physical_row)
        lane = physical_row & 31
        logical_row = physical_row - lane + (lane & 7) * 4 + (lane >> 3)
        output[slot, logical_row] = result.to(torch.bfloat16)
    return output


@helion.kernel(static_shapes=True, autotune_effort="full", backend="triton")
def mxfp4_moe_gemm1_swiglu_oai_decode(
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w13_scale_bytes: torch.Tensor,
    w13_bias: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    """GEMV-specialized MXFP4 GEMM1 using coalesced software FP4 decode."""
    tokens, hidden = hidden_states.size()
    experts, twice_intermediate, packed_hidden = w13.size()
    top_k = topk_ids.size(1)
    intermediate = twice_intermediate // 2
    scale_k = hidden // 32
    assert tokens == 1
    assert packed_hidden * 2 == hidden
    hl.specialize(experts)
    hl.specialize(top_k)
    hl.specialize(intermediate)
    output = torch.empty(
        (top_k, intermediate),
        dtype=torch.bfloat16,
        device=hidden_states.device,
    )
    flat_weight = w13.view(torch.uint8).view(-1)
    flat_bias = w13_bias.view(-1)
    flat_scale = w13_scale_bytes.view(-1)
    block_physical_row = hl.register_block_size(16, 128)
    block_scale_k = hl.register_block_size(1, scale_k)
    for tile_slot, tile_physical_row in hl.tile(
        [top_k, twice_intermediate], block_size=[1, block_physical_row]
    ):
        wait_and_launch_dependents()
        slot = tile_slot.begin
        expert = topk_ids[0, slot]
        expert_row = expert * twice_intermediate + tile_physical_row.index
        accumulator = hl.zeros([tile_physical_row], dtype=torch.float32)
        for tile_scale_k in hl.tile(scale_k, block_size=block_scale_k):
            group_mask = tile_scale_k.index < scale_k
            subgroup = expert_row[:, None] * (scale_k * 2)
            subgroup += tile_scale_k.index[None, :] * 2
            valid = (tile_physical_row.index[:, None] < twice_intermediate) & (
                group_mask[None, :]
            )
            weight_first = hl.load_float4_e2m1fn_x16_to_float16(
                flat_weight,
                subgroup,
                extra_mask=valid,
            )
            weight_second = hl.load_float4_e2m1fn_x16_to_float16(
                flat_weight,
                subgroup + 1,
                extra_mask=valid,
            )
            activation_first = hl.load_bfloat16_x16_to_float16(
                hidden_states,
                tile_scale_k.index * 2,
                extra_mask=group_mask,
            )
            activation_second = hl.load_bfloat16_x16_to_float16(
                hidden_states,
                tile_scale_k.index * 2 + 1,
                extra_mask=group_mask,
            )
            contribution = hl.zeros(
                [block_physical_row, block_scale_k], dtype=torch.float16
            )
            for index in hl.static_range(16):
                contribution += weight_first[index] * activation_first[index][None, :]
                contribution += weight_second[index] * activation_second[index][None, :]
            scale_offset = expert * twice_intermediate * scale_k + _trtllm_scale_offset(
                tile_physical_row.index[:, None],
                tile_scale_k.index[None, :],
                twice_intermediate,
                scale_k,
            )
            scale = _e8m0_byte_to_f32(hl.load(flat_scale, [scale_offset]))
            accumulator += torch.sum(contribution.to(torch.float32) * scale, dim=-1)

        preactivation = accumulator + flat_bias[expert_row]
        pairs = preactivation.reshape(block_physical_row // 16, 2, 8).permute(0, 2, 1)
        up_value, gate_value = hl.split(pairs)
        gate_value = torch.clamp(gate_value, max=7.0)
        up_value = torch.clamp(up_value, min=-7.0, max=7.0)
        activated = (up_value + 1.0) * gate_value * torch.sigmoid(1.702 * gate_value)
        chunk = hl.arange(block_physical_row // 16)[:, None]
        pair_lane = hl.arange(8)[None, :]
        global_chunk = tile_physical_row.begin // 16 + chunk
        output_row = (global_chunk // 2) * 16 + pair_lane * 2 + (global_chunk & 1)
        output[slot, output_row.reshape(block_physical_row // 2)] = activated.reshape(
            block_physical_row // 2
        ).to(torch.bfloat16)
    return output


@helion.kernel(static_shapes=True, autotune_effort="full", backend="triton")
def mxfp4_moe_gemm2_decode(
    activation: torch.Tensor,
    w2: torch.Tensor,
    w2_scale_bytes: torch.Tensor,
    w2_bias: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    """GEMV-specialized MXFP4 GEMM2 using coalesced software FP4 decode."""
    top_k, intermediate = activation.size()
    experts, hidden, packed_intermediate = w2.size()
    scale_k = intermediate // 32
    assert packed_intermediate * 2 == intermediate
    hl.specialize(experts)
    hl.specialize(top_k)
    hl.specialize(hidden)
    output = torch.empty(
        (top_k, hidden), dtype=torch.bfloat16, device=activation.device
    )
    flat_weight = w2.view(torch.uint8).view(-1)
    flat_bias = w2_bias.view(-1)
    flat_scale = w2_scale_bytes.view(-1)
    block_physical_row = hl.register_block_size(8, 128)
    block_scale_k = hl.register_block_size(1, scale_k)
    for tile_slot, tile_physical_row in hl.tile(
        [top_k, hidden], block_size=[1, block_physical_row]
    ):
        wait_and_launch_dependents()
        slot = tile_slot.begin
        expert = topk_ids[0, slot]
        expert_row = expert * hidden + tile_physical_row.index
        accumulator = hl.zeros([tile_physical_row], dtype=torch.float32)
        for tile_scale_k in hl.tile(scale_k, block_size=block_scale_k):
            group_mask = tile_scale_k.index < scale_k
            subgroup = expert_row[:, None] * (scale_k * 2)
            subgroup += tile_scale_k.index[None, :] * 2
            valid = (tile_physical_row.index[:, None] < hidden) & (group_mask[None, :])
            weight_first = hl.load_float4_e2m1fn_x16_to_float16(
                flat_weight,
                subgroup,
                extra_mask=valid,
            )
            weight_second = hl.load_float4_e2m1fn_x16_to_float16(
                flat_weight,
                subgroup + 1,
                extra_mask=valid,
            )
            activation_group = slot * (scale_k * 2) + tile_scale_k.index * 2
            activation_first = hl.load_bfloat16_x16_to_float16(
                activation,
                activation_group,
                extra_mask=group_mask,
            )
            activation_second = hl.load_bfloat16_x16_to_float16(
                activation,
                activation_group + 1,
                extra_mask=group_mask,
            )
            contribution = hl.zeros(
                [block_physical_row, block_scale_k], dtype=torch.float16
            )
            for index in hl.static_range(16):
                contribution += weight_first[index] * activation_first[index][None, :]
                contribution += weight_second[index] * activation_second[index][None, :]
            scale_offset = expert * hidden * scale_k + _trtllm_scale_offset(
                tile_physical_row.index[:, None],
                tile_scale_k.index[None, :],
                hidden,
                scale_k,
            )
            scale = _e8m0_byte_to_f32(hl.load(flat_scale, [scale_offset]))
            accumulator += torch.sum(contribution.to(torch.float32) * scale, dim=-1)

        result = accumulator + flat_bias[expert_row]
        physical_row = tile_physical_row.begin + hl.arange(block_physical_row)
        lane = physical_row & 31
        logical_row = physical_row - lane + (lane & 7) * 4 + (lane >> 3)
        output[slot, logical_row] = result.to(torch.bfloat16)
    return output


@helion.kernel(
    static_shapes=True,
    autotune_effort="full",
    backend="triton",
)
def mxfp4_moe_finalize(
    expert_output: torch.Tensor,
    topk_weights: torch.Tensor,
    output_hidden: int,
) -> torch.Tensor:
    """Inverse permutation, route-weight reduction, and output unpadding."""
    top_k, hidden = expert_output.size()
    output_hidden = hl.specialize(output_hidden)
    assert output_hidden <= hidden
    output = torch.empty(
        (1, output_hidden), dtype=expert_output.dtype, device=expert_output.device
    )
    weights = topk_weights.view(top_k)
    for tile_n in hl.tile(output_hidden):
        wait_and_launch_dependents()
        values = expert_output[:, tile_n].to(torch.float32)
        output[:, tile_n] = torch.sum(
            values * weights[:, None].to(torch.float32),
            dim=0,
            keepdim=True,
        )
    return output


KERNEL_ORDER = ("routing", "gemm1_swiglu_oai", "gemm2", "finalize")


def _row_permutation(rows: int, device: torch.device) -> torch.Tensor:
    logical = torch.arange(rows, device=device)
    lane = logical % 32
    physical = logical - lane + (lane % 4) * 8 + lane // 4
    permutation = torch.empty_like(logical)
    permutation[physical] = logical
    return permutation


def _swizzle_scales(scales: torch.Tensor) -> torch.Tensor:
    rows, cols = scales.shape
    if rows % 128 or cols % 4:
        raise ValueError("TRT-LLM MXFP4 scales require rows % 128 == cols % 4 == 0")
    row = torch.arange(rows, device=scales.device)[:, None]
    col = torch.arange(cols, device=scales.device)[None, :]
    offsets = _trtllm_scale_offset(row, col, rows, cols)
    output = torch.empty(rows * cols, device=scales.device, dtype=scales.dtype)
    output[offsets.reshape(-1)] = scales.reshape(-1)
    return output.reshape(rows, cols)


def _to_production_w13(
    weight: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows = weight.size(0)
    weight = weight.reshape(rows // 2, 2, -1).flip(1).reshape_as(weight)
    scale = scale.reshape(rows // 2, 2, -1).flip(1).reshape_as(scale)
    bias = bias.reshape(rows // 2, 2).flip(1).reshape_as(bias)
    permutation = _row_permutation(rows, weight.device)
    return (
        weight[permutation].contiguous(),
        _swizzle_scales(scale[permutation].contiguous()),
        bias[permutation].contiguous(),
    )


def _to_production_w2(
    weight: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    permutation = _row_permutation(weight.size(0), weight.device)
    return (
        weight[permutation].contiguous(),
        _swizzle_scales(scale[permutation].contiguous()),
        bias[permutation].contiguous(),
    )


def _dequant_e2m1(storage: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    lut = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
        device=storage.device,
    )
    low = lut[(storage & 0xF).long()]
    high = lut[((storage >> 4) & 0xF).long()]
    values = torch.stack((low, high), dim=-1).flatten(-2)
    e8m0 = torch.pow(2.0, scale.float() - 127.0).repeat_interleave(32, dim=-1)
    return values * e8m0


def _allocate(shape: GptOssMoeShape) -> dict[str, torch.Tensor]:
    device = torch.device("cuda")
    torch.manual_seed(1)
    logits = torch.full((1, shape.experts), -8.0, device=device, dtype=torch.bfloat16)
    hidden = torch.randn((1, shape.hidden), device=device, dtype=torch.bfloat16) * 0.1

    w13 = torch.zeros(
        (shape.experts, 2 * shape.intermediate, shape.hidden // 2),
        device=device,
        dtype=torch.uint8,
    )
    w13_scale = torch.empty(
        (shape.experts, 2 * shape.intermediate, shape.hidden // 32),
        device=device,
        dtype=torch.uint8,
    )
    w13_scale.fill_(127)
    w13_bias = torch.zeros(
        (shape.experts, 2 * shape.intermediate),
        device=device,
        dtype=torch.float32,
    )
    w2 = torch.zeros(
        (shape.experts, shape.hidden, shape.intermediate // 2),
        device=device,
        dtype=torch.uint8,
    )
    w2_scale = torch.empty(
        (shape.experts, shape.hidden, shape.intermediate // 32),
        device=device,
        dtype=torch.uint8,
    )
    w2_scale.fill_(127)
    w2_bias = torch.zeros(
        (shape.experts, shape.hidden), device=device, dtype=torch.float32
    )

    logical_w13 = torch.randint(
        0,
        256,
        (shape.top_k, 2 * shape.intermediate, shape.hidden // 2),
        device=device,
        dtype=torch.uint8,
    )
    logical_w13_scale = torch.randint(
        119,
        122,
        (shape.top_k, 2 * shape.intermediate, shape.hidden // 32),
        device=device,
        dtype=torch.uint8,
    )
    logical_w13_bias = (
        torch.randn(
            (shape.top_k, 2 * shape.intermediate),
            device=device,
            dtype=torch.float32,
        )
        * 0.01
    )
    logical_w2 = torch.randint(
        0,
        256,
        (shape.top_k, shape.hidden, shape.intermediate // 2),
        device=device,
        dtype=torch.uint8,
    )
    logical_w2_scale = torch.randint(
        119,
        122,
        (shape.top_k, shape.hidden, shape.intermediate // 32),
        device=device,
        dtype=torch.uint8,
    )
    logical_w2_bias = (
        torch.randn((shape.top_k, shape.hidden), device=device, dtype=torch.float32)
        * 0.01
    )

    tensors = {
        "logits": logits,
        "hidden": hidden,
        "w13": w13.view(torch.float4_e2m1fn_x2),
        "w13_scale": w13_scale.view(torch.float8_e4m3fn),
        "w13_bias": w13_bias,
        "w2": w2.view(torch.float4_e2m1fn_x2),
        "w2_scale": w2_scale.view(torch.float8_e4m3fn),
        "w2_bias": w2_bias,
        "logical_w13": logical_w13,
        "logical_w13_scale": logical_w13_scale,
        "logical_w13_bias": logical_w13_bias,
        "logical_w2": logical_w2,
        "logical_w2_scale": logical_w2_scale,
        "logical_w2_bias": logical_w2_bias,
    }
    set_selected_experts(tensors, (3, 17, 64, 101))
    return tensors


def set_selected_experts(
    tensors: dict[str, torch.Tensor],
    selected_ids: tuple[int, ...],
    selected_logits: tuple[float, ...] = (4.0, 3.0, 2.0, 1.0),
) -> None:
    """Move the four logical experts to runtime-selected physical IDs."""
    if len(selected_ids) != 4 or len(set(selected_ids)) != 4:
        raise ValueError("selected expert IDs must contain four distinct values")
    if len(selected_logits) != 4:
        raise ValueError("selected logits must contain four values")
    if any(left <= right for left, right in pairwise(selected_logits)):
        raise ValueError("selected logits must be strictly descending")
    experts = tensors["w13"].size(0)
    if any(expert < 0 or expert >= experts for expert in selected_ids):
        raise ValueError(f"selected expert IDs must be in [0, {experts})")

    ids = torch.tensor(selected_ids, device="cuda", dtype=torch.int64)
    tensors["logits"].fill_(-8.0)
    tensors["logits"][0, ids] = torch.tensor(
        selected_logits, device="cuda", dtype=torch.bfloat16
    )
    w13 = tensors["w13"].view(torch.uint8)
    w13_scale = tensors["w13_scale"].view(torch.uint8)
    w2 = tensors["w2"].view(torch.uint8)
    w2_scale = tensors["w2_scale"].view(torch.uint8)
    for slot, expert in enumerate(selected_ids):
        physical_w13, physical_s13, physical_b13 = _to_production_w13(
            tensors["logical_w13"][slot],
            tensors["logical_w13_scale"][slot],
            tensors["logical_w13_bias"][slot],
        )
        physical_w2, physical_s2, physical_b2 = _to_production_w2(
            tensors["logical_w2"][slot],
            tensors["logical_w2_scale"][slot],
            tensors["logical_w2_bias"][slot],
        )
        w13[expert].copy_(physical_w13)
        w13_scale[expert].copy_(physical_s13)
        tensors["w13_bias"][expert].copy_(physical_b13)
        w2[expert].copy_(physical_w2)
        w2_scale[expert].copy_(physical_s2)
        tensors["w2_bias"][expert].copy_(physical_b2)


def _reference(
    tensors: dict[str, torch.Tensor],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    values, ids = torch.topk(tensors["logits"].float(), 4, dim=-1)
    weights = torch.softmax(values, dim=-1).to(torch.bfloat16)
    w13 = _dequant_e2m1(tensors["logical_w13"], tensors["logical_w13_scale"])
    preact = torch.einsum("koi,bi->ko", w13, tensors["hidden"].float())
    preact += tensors["logical_w13_bias"]
    gate = torch.clamp(preact[:, 0::2], max=7.0)
    up = torch.clamp(preact[:, 1::2], min=-7.0, max=7.0)
    activation = ((up + 1.0) * gate * torch.sigmoid(1.702 * gate)).to(torch.bfloat16)
    w2 = _dequant_e2m1(tensors["logical_w2"], tensors["logical_w2_scale"])
    expert_output = torch.einsum("koi,ki->ko", w2, activation.float())
    expert_output += tensors["logical_w2_bias"]
    expert_output = expert_output.to(torch.bfloat16)
    output = torch.sum(
        expert_output.float() * weights.view(-1, 1).float(), dim=0, keepdim=True
    ).to(torch.bfloat16)
    return weights, ids.to(torch.int32), activation, expert_output, output


def _assert_close(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(actual.float(), expected.float(), atol=1.0, rtol=0.1)
    error = float((actual.float() - expected.float()).abs().max().item())
    print(f"correctness {name} max_abs={error:.6f}", flush=True)


def _compile(kernel, args, config_name: str):
    bound = kernel.bind(args)
    config = helion.Config.from_dict(CONFIGS[config_name])
    bound.config_spec.normalize(config.config)
    return bound.compile_config(config)


def build(
    tensors: dict[str, torch.Tensor], shape: GptOssMoeShape
) -> tuple[Callable[[], tuple[torch.Tensor, ...]], tuple[torch.Tensor, ...]]:
    """Compile the independently tuned four-launch graph and return its closure."""
    routing_args = (tensors["logits"],)
    routing = _compile(mxfp4_top4_routing, routing_args, "routing")
    weights, ids = routing(*routing_args)

    gemm1_args = (
        tensors["hidden"],
        tensors["w13"],
        tensors["w13_scale"].view(torch.uint8),
        tensors["w13_bias"],
        ids,
    )
    gemm1 = _compile(
        mxfp4_moe_gemm1_swiglu_oai_decode,
        gemm1_args,
        "gemm1_swiglu_oai",
    )
    activation = launch_dependent(gemm1, *gemm1_args)

    gemm2_args = (
        activation,
        tensors["w2"],
        tensors["w2_scale"].view(torch.uint8),
        tensors["w2_bias"],
        ids,
    )
    gemm2 = _compile(mxfp4_moe_gemm2_decode, gemm2_args, "gemm2")
    expert_output = launch_dependent(gemm2, *gemm2_args)

    finalize_args = (expert_output, weights, shape.output_hidden)
    finalize = _compile(mxfp4_moe_finalize, finalize_args, "finalize")
    output = launch_dependent(finalize, *finalize_args)

    def launch() -> tuple[torch.Tensor, ...]:
        local_weights, local_ids = routing(*routing_args)
        local_activation = launch_dependent(
            gemm1,
            tensors["hidden"],
            tensors["w13"],
            tensors["w13_scale"].view(torch.uint8),
            tensors["w13_bias"],
            local_ids,
        )
        local_expert_output = launch_dependent(
            gemm2,
            local_activation,
            tensors["w2"],
            tensors["w2_scale"].view(torch.uint8),
            tensors["w2_bias"],
            local_ids,
        )
        local_output = launch_dependent(
            finalize,
            local_expert_output,
            local_weights,
            shape.output_hidden,
        )
        return (
            local_output,
            local_weights,
            local_ids,
            local_activation,
            local_expert_output,
        )

    outputs = (output, weights, ids, activation, expert_output)
    return launch, outputs
