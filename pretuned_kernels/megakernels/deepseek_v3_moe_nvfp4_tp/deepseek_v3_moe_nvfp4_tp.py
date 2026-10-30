"""DeepSeek-V3 B1 NVFP4 MoE with a fused TP4 all-reduce on B200.

This is tensor parallelism, not token expert parallelism: every rank receives
the same token and routing result, evaluates a distinct intermediate shard of
the selected experts, and sums the rank-local weighted outputs.  The block ends
like vLLM's and SGLang's fused all-reduce: the summed output is added to the
residual stream and normalized (RMSNorm), all in the same kernel.  The baseline
is vLLM's production path: its FlashInfer MoE call, then FlashInfer's fused
all-reduce + residual add + RMSNorm.

Run the benchmark with four local ranks::

    NVSHMEM_DISABLE_CUDA_VMM=1 \\
      python -m torch.distributed.run --standalone --nproc-per-node=4 \\
      pretuned_kernels/megakernels/deepseek_v3_moe_nvfp4_tp/deepseek_v3_moe_nvfp4_tp.py
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

from pretuned_kernels.megakernels._distributed import benchmark
from pretuned_kernels.megakernels._distributed import initialize
from pretuned_kernels.megakernels.deepseek_v3_moe_nvfp4_tp import _setup
from pretuned_kernels.megakernels.deepseek_v3_moe_nvfp4_tp._common import FP4_MAX
from pretuned_kernels.megakernels.deepseek_v3_moe_nvfp4_tp._common import MMA_N
from pretuned_kernels.megakernels.deepseek_v3_moe_nvfp4_tp._common import (
    first_mma_column,
)
from pretuned_kernels.megakernels.deepseek_v3_moe_nvfp4_tp._common import fp4_nibble
from pretuned_kernels.megakernels.deepseek_v3_moe_nvfp4_tp._common import routed_slots
from pretuned_kernels.megakernels.deepseek_v3_moe_nvfp4_tp._common import routing_keys
import torch
import torch.distributed as dist

import helion
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Callable


WORLD_SIZE = 4
W13_SPLIT_K = 7
COMMUNICATION_N = 64
# The block ends like SGLang's fused all-reduce: residual add, then RMSNorm
# (DeepSeek-V3 rms_norm_eps).
NORM_EPS = 1e-6
# Arrivals of the communication tasks, which wait for each other's partial sums
# of squares; never reset, every launch adds one per task.
COUNTER_SHAPE = (1,)
NUM_SM_MULTIPLIER = 1
SIGNAL_PAD_BYTES = 32 * 1024
# Bytes of one routed w2 weight row (intermediate // 2 FP4 values) and of its
# block scales.
W2_ROW_BYTES = 256
W2_SCALE_ROW_BYTES = 32


def _deepseek_v3_moe_nvfp4_tp(
    hidden_input: torch.Tensor,
    router_weight: torch.Tensor,
    correction_bias: torch.Tensor,
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    shared_w13: torch.Tensor,
    shared_w13_scale: torch.Tensor,
    shared_w2: torch.Tensor,
    shared_w2_scale: torch.Tensor,
    alpha1: torch.Tensor,
    alpha2: torch.Tensor,
    shared_alpha1: torch.Tensor,
    shared_alpha2: torch.Tensor,
    input_global_scale: torch.Tensor,
    activation_global_scale: torch.Tensor,
    top_k: int,
    num_groups: int,
    topk_groups: int,
    routed_scale: float,
    w13_split_k: int,
    norm_counter: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    symmetric_output: torch.Tensor,
    symmetric_shared: torch.Tensor,
    group_name: hl.ProcessGroupName,
) -> tuple[torch.Tensor, ...]:
    # The model geometry and layouts are part of this pretuned kernel's fixed
    # contract. Tensor contents, including routing decisions, remain runtime
    # data even though their metadata is specialized explicitly.
    hl.specialize(hidden_input.size(0))
    hl.specialize(hidden_input.size(1))
    hl.specialize(router_weight.size(0))
    hl.specialize(router_weight.size(1))
    hl.specialize(correction_bias.size(0))
    hl.specialize(w13.size(0))
    hl.specialize(w13.size(1))
    hl.specialize(w13.size(2))
    hl.specialize(w13_scale.size(0))
    hl.specialize(w13_scale.size(1))
    hl.specialize(w13_scale.size(2))
    hl.specialize(w2.size(0))
    hl.specialize(w2.size(1))
    hl.specialize(w2.size(2))
    hl.specialize(w2_scale.size(0))
    hl.specialize(w2_scale.size(1))
    hl.specialize(w2_scale.size(2))
    hl.specialize(shared_w13.size(0))
    hl.specialize(shared_w13.size(1))
    hl.specialize(shared_w13_scale.size(0))
    hl.specialize(shared_w13_scale.size(1))
    hl.specialize(shared_w2.size(0))
    hl.specialize(shared_w2.size(1))
    hl.specialize(shared_w2_scale.size(0))
    hl.specialize(shared_w2_scale.size(1))
    hl.specialize(alpha1.size(0))
    hl.specialize(alpha2.size(0))
    hl.specialize(shared_alpha1.size(0))
    hl.specialize(shared_alpha2.size(0))
    hl.specialize(input_global_scale.size(0))
    hl.specialize(activation_global_scale.size(0))

    hl.specialize(hidden_input.stride(0))
    hl.specialize(hidden_input.stride(1))
    hl.specialize(router_weight.stride(0))
    hl.specialize(router_weight.stride(1))
    hl.specialize(correction_bias.stride(0))
    hl.specialize(w13.stride(0))
    hl.specialize(w13.stride(1))
    hl.specialize(w13.stride(2))
    hl.specialize(w13_scale.stride(0))
    hl.specialize(w13_scale.stride(1))
    hl.specialize(w13_scale.stride(2))
    hl.specialize(w2.stride(0))
    hl.specialize(w2.stride(1))
    hl.specialize(w2.stride(2))
    hl.specialize(w2_scale.stride(0))
    hl.specialize(w2_scale.stride(1))
    hl.specialize(w2_scale.stride(2))
    hl.specialize(shared_w13.stride(0))
    hl.specialize(shared_w13.stride(1))
    hl.specialize(shared_w13_scale.stride(0))
    hl.specialize(shared_w13_scale.stride(1))
    hl.specialize(shared_w2.stride(0))
    hl.specialize(shared_w2.stride(1))
    hl.specialize(shared_w2_scale.stride(0))
    hl.specialize(shared_w2_scale.stride(1))
    hl.specialize(alpha1.stride(0))
    hl.specialize(alpha2.stride(0))
    hl.specialize(shared_alpha1.stride(0))
    hl.specialize(shared_alpha2.stride(0))
    hl.specialize(input_global_scale.stride(0))
    hl.specialize(activation_global_scale.stride(0))

    rows, hidden_size = hidden_input.shape
    experts, router_hidden = router_weight.shape
    assert hidden_size == router_hidden
    router_logits = torch.empty(
        (rows, experts), dtype=torch.bfloat16, device=hidden_input.device
    )
    router_expert_block = hl.register_block_size(2, 32)
    # The router GEMV is split along the hidden dimension so it spreads over many
    # CTAs; the routing task sums the partials.
    router_split_k = 7
    router_k_block = hidden_size // router_split_k
    router_partial = torch.empty(
        (router_split_k, experts), dtype=torch.float32, device=hidden_input.device
    )
    input_quant_groups = hidden_size // 16
    input_hidden_q = torch.empty(
        (rows, hidden_size // 2), dtype=torch.uint8, device=hidden_input.device
    )
    input_hidden_q_groups = input_hidden_q.view(rows, input_quant_groups, 8)
    input_hidden_scale = torch.empty(
        (rows, input_quant_groups),
        dtype=torch.float8_e4m3fn,
        device=hidden_input.device,
    )
    input_quant_group_block = hl.register_block_size(1, 32)
    hidden_q = input_hidden_q
    hidden_scale_bytes = input_hidden_scale
    logits = router_logits
    w13_scale_bytes = w13_scale
    w2_scale_bytes = w2_scale
    topk_batch, topk_num_experts = logits.size()
    top_k = hl.specialize(top_k)
    num_groups = hl.specialize(num_groups)
    topk_groups = hl.specialize(topk_groups)
    w13_split_k = hl.specialize(w13_split_k)
    assert top_k == 8 and topk_groups == 4
    hl.specialize(topk_batch)
    hl.specialize(topk_num_experts)
    assert topk_num_experts & (topk_num_experts - 1) == 0
    topk_weights = torch.empty(
        (topk_batch, top_k), dtype=torch.float32, device=logits.device
    )
    topk_ids = torch.empty((topk_batch, top_k), dtype=torch.int32, device=logits.device)
    tokens, packed_hidden = hidden_q.size()
    experts, twice_intermediate, weight_packed_hidden = w13.size()
    intermediate = twice_intermediate // 2
    hidden_groups = packed_hidden // 8
    assert hidden_groups % w13_split_k == 0
    activation_groups = intermediate // 16
    assert tokens == 1 and packed_hidden == weight_packed_hidden
    hl.specialize(experts)
    hl.specialize(intermediate)
    activation_q = torch.empty(
        (top_k, intermediate // 2), dtype=torch.uint8, device=hidden_q.device
    )
    activation_scale = torch.empty(
        (top_k, activation_groups), dtype=torch.float8_e4m3fn, device=hidden_q.device
    )
    activation_q_groups = activation_q.view(top_k, activation_groups, 8)
    w13_tma = w13.view(experts * (twice_intermediate // 128), 128, packed_hidden)
    flat_w13_scale = w13_scale_bytes.view(experts * twice_intermediate, hidden_groups)
    w2_top_k, packed_intermediate = activation_q.size()
    w2_experts, hidden, weight_packed_intermediate = w2.size()
    w2_groups = packed_intermediate // 8
    assert packed_intermediate == weight_packed_intermediate
    assert packed_intermediate == W2_ROW_BYTES and w2_groups == W2_SCALE_ROW_BYTES
    w2_prefetch_rows = hidden // (activation_groups // 4 * w13_split_k)
    assert w2_prefetch_rows * (activation_groups // 4 * w13_split_k) == hidden
    hl.specialize(w2_experts)
    hl.specialize(w2_top_k)
    hl.specialize(hidden)
    flat_w2_scale = w2_scale_bytes.view(w2_experts * hidden, w2_groups)
    shared_twice_intermediate, shared_weight_packed_hidden = shared_w13.shape
    shared_hidden, shared_weight_packed_intermediate = shared_w2.shape
    assert (
        shared_twice_intermediate == twice_intermediate
        and shared_weight_packed_hidden == packed_hidden
        and shared_hidden == hidden
        and shared_weight_packed_intermediate == packed_intermediate
    )
    flat_shared_w13_scale = shared_w13_scale.view(twice_intermediate, hidden_groups)
    flat_shared_w2 = shared_w2.view(hidden, packed_intermediate)
    flat_shared_w2_scale = shared_w2_scale.view(hidden, w2_groups)
    w2_block_row = hl.register_block_size(128, 256)
    # Each routed w2 task and each shared w2 task writes its rows to symmetric
    # memory once, so every rank's communication tasks receive them as soon as
    # they are written and add up each rank's output there.
    hl.specialize(symmetric_output.size())
    hl.specialize(symmetric_output.stride())
    hl.specialize(symmetric_shared.size())
    hl.specialize(symmetric_shared.stride())
    assert symmetric_output.size() == (w2_top_k, hidden)
    assert symmetric_shared.size() == (1, hidden)
    assert symmetric_output.stride() == (hidden, 1)
    assert symmetric_shared.stride() == (hidden, 1)
    assert symmetric_output.storage_offset() == 0
    assert symmetric_shared.storage_offset() == 0
    expert_output = torch.as_strided(
        symmetric_output, (w2_top_k, hidden), (hidden, 1), storage_offset=0
    )
    shared_expert_output = torch.as_strided(
        symmetric_shared, (1, hidden), (hidden, 1), storage_offset=0
    )
    output = torch.empty((1, hidden), dtype=torch.bfloat16, device=w2.device)
    hl.specialize(residual.size())
    hl.specialize(residual.stride())
    hl.specialize(norm_weight.size(0))
    hl.specialize(norm_weight.stride(0))
    hl.specialize(norm_counter.size(0))
    assert residual.size() == (1, hidden) and norm_weight.size(0) == hidden
    residual_output = torch.empty((1, hidden), dtype=torch.bfloat16, device=w2.device)
    # A literal: a module-level int reaches the kernel as a runtime scalar.
    communication_n = 64
    assert communication_n == COMMUNICATION_N
    norm_tasks = hidden // communication_n
    norm_partial = torch.empty((norm_tasks,), dtype=torch.float32, device=w2.device)
    # The FP32 pre-norm sums, read back by the normalizing loop.
    norm_input = torch.empty((1, hidden), dtype=torch.float32, device=w2.device)
    remote_expert_outputs = torch.ops.symm_mem.get_remote_tensors(
        expert_output, group_name
    )
    remote_shared_outputs = torch.ops.symm_mem.get_remote_tensors(
        shared_expert_output, group_name
    )
    shared_w13_preactivation = torch.empty(
        (twice_intermediate,), dtype=torch.bfloat16, device=w13.device
    )
    shared_activation_q = torch.empty(
        (1, intermediate // 2), dtype=torch.uint8, device=hidden_q.device
    )
    shared_activation_scale = torch.empty(
        (1, activation_groups), dtype=torch.float8_e4m3fn, device=hidden_q.device
    )
    shared_activation_q_groups = shared_activation_q.view(1, activation_groups, 8)
    shared_activation_q_flat = shared_activation_q.view(intermediate // 2)
    w13_partial = torch.empty(
        (top_k, activation_groups // 4, w13_split_k, 128),
        dtype=torch.float32,
        device=hidden_q.device,
    )
    # alpha1 of each w13 task's expert and the activation's global scale, so the
    # finishing tasks read them from L2 with the partials rather than chase
    # topk_ids into cold parameters once the partials are in.
    w13_scales = torch.empty(
        (top_k, activation_groups // 4, w13_split_k, 2),
        dtype=torch.float32,
        device=hidden_q.device,
    )
    w13_groups_per_split = hidden_groups // w13_split_k
    shared_w13_tma = shared_w13.view(twice_intermediate // 128, 128, packed_hidden)
    shared_w13_partial = torch.empty(
        (activation_groups // 4, w13_split_k, 128),
        dtype=torch.float32,
        device=hidden_q.device,
    )
    for tile_expert, tile_split in hl.tile(
        [experts, router_split_k], block_size=[router_expert_block, 1]
    ):
        router_k = tile_split.begin * router_k_block + hl.arange(router_k_block)
        router_x = hidden_input[0, router_k].to(torch.float32)
        router_w = router_weight[tile_expert, router_k].to(torch.float32)
        router_partial[tile_split.begin, tile_expert] = torch.sum(
            router_w * router_x[None, :], dim=1
        )
    for tile_row, tile_group in hl.tile(
        [rows, input_quant_groups], block_size=[1, input_quant_group_block]
    ):
        packed_lane = hl.arange(8)[None, :]
        group_offsets = tile_group.index[:, None] * 16
        low_values = hidden_input[tile_row, group_offsets + packed_lane * 2].to(
            torch.float32
        )
        high_values = hidden_input[tile_row, group_offsets + packed_lane * 2 + 1].to(
            torch.float32
        )
        global_scale = torch.sum(input_global_scale[:])
        block_scale_f32 = (
            torch.maximum(
                torch.amax(torch.abs(low_values), dim=-1),
                torch.amax(torch.abs(high_values), dim=-1),
            )
            * global_scale
            / FP4_MAX
        )
        block_scale = block_scale_f32.to(torch.float8_e4m3fn)
        actual_scale = block_scale.to(torch.float32)
        divisor = torch.where(actual_scale > 0, actual_scale, 1.0)
        low = fp4_nibble(low_values * global_scale / divisor[:, :, None])
        high = fp4_nibble(high_values * global_scale / divisor[:, :, None])
        packed = (low | high << 4).to(torch.uint8)
        input_hidden_q_groups[tile_row, tile_group, :] = packed
        input_hidden_scale[tile_row, tile_group] = block_scale
    # Shared expert gate/up: the routed experts' split-K block-scaled MMA on the
    # shared weights.  It needs no routing, so it runs from the start of the launch.
    for shared_w13_tile_group, shared_w13_tile_split in hl.tile(
        [activation_groups, w13_split_k], block_size=[4, 1]
    ):
        shared_w13_split = shared_w13_tile_split.begin
        shared_w13_row_block = shared_w13_tile_group.begin // 4
        shared_w13_physical_row = (
            shared_w13_tile_group.begin * 32 + hl.arange(128)
        ).to(torch.int64)
        # The shared expert's w2 weights are known from the start: pull them
        # into L2 now, while DRAM idles during routing, so the shared w2 tasks
        # do not compete with the routed w13 stream for it later.  Each of the
        # activation_groups // 4 * w13_split_k tasks takes w2_prefetch_rows rows.
        shared_w2_prefetch_row = (
            shared_w13_row_block * w13_split_k + shared_w13_split
        ) * w2_prefetch_rows + hl.arange(w2_prefetch_rows).to(torch.int64)
        hl.prefetch(flat_shared_w2, [shared_w2_prefetch_row, slice(None)])
        hl.prefetch(flat_shared_w2_scale, [shared_w2_prefetch_row, slice(None)])
        shared_w13_accumulator = hl.zeros([128, MMA_N], dtype=torch.float32)
        for shared_w13_local_group in hl.tile(w13_groups_per_split, block_size=32):
            shared_w13_group_begin = (
                shared_w13_split * w13_groups_per_split + shared_w13_local_group.begin
            )
            shared_w13_group_index = (
                shared_w13_split * w13_groups_per_split + shared_w13_local_group.index
            )
            shared_w13_packed_index = (
                shared_w13_group_begin * 8
                + hl.arange(shared_w13_local_group.block_size * 8)
            ).to(torch.int64)
            shared_w13_lhs = shared_w13_tma[
                shared_w13_row_block, :, shared_w13_packed_index
            ]
            shared_w13_lhs_scale = flat_shared_w13_scale[
                shared_w13_physical_row[:, None], shared_w13_group_index[None, :]
            ]
            shared_w13_hidden_bytes = hidden_q[0, shared_w13_packed_index]
            shared_w13_rhs = shared_w13_hidden_bytes[:, None].expand(
                shared_w13_hidden_bytes.size(0), MMA_N
            )
            shared_w13_hidden_scale = hidden_scale_bytes[0, shared_w13_group_index]
            shared_w13_rhs_scale = shared_w13_hidden_scale[None, :].expand(
                MMA_N, shared_w13_local_group.block_size
            )
            shared_w13_accumulator = hl.dot_scaled(
                shared_w13_lhs,
                shared_w13_lhs_scale,
                "e2m1",
                shared_w13_rhs,
                shared_w13_rhs_scale,
                "e2m1",
                acc=shared_w13_accumulator,
                out_dtype=torch.float32,
            )
        shared_w13_partial[shared_w13_row_block, shared_w13_split, :] = (
            first_mma_column(shared_w13_accumulator)
        )
    for _topk_program in hl.grid(1):
        topk_logits = torch.sum(router_partial[:, :], dim=0).to(torch.bfloat16)
        router_logits[0, :] = topk_logits
        topk_scores, topk_slot, topk_expert_indices = routed_slots(
            topk_logits,
            correction_bias[:],
            topk_num_experts,
            num_groups,
            topk_groups,
            top_k,
        )
        topk_chosen = topk_slot >= 0
        topk_weight = torch.where(topk_chosen, topk_scores, 0.0)
        topk_denominator = torch.sum(topk_weight)
        hl.store(topk_ids, [0, topk_slot], topk_expert_indices, extra_mask=topk_chosen)
        hl.store(
            topk_weights,
            [0, topk_slot],
            topk_weight / topk_denominator * routed_scale,
            extra_mask=topk_chosen,
        )
    for w13_tile_slot, w13_tile_output_group, w13_tile_split in hl.tile(
        [top_k, activation_groups, w13_split_k], block_size=[1, 4, 1]
    ):
        w13_slot = w13_tile_slot.begin
        w13_split = w13_tile_split.begin
        # Route here instead of waiting for the top-k root: the same picks,
        # computed while this task would otherwise spin on its result.
        _w13_scores, w13_keys, _w13_expert_indices = routing_keys(
            torch.sum(router_partial[:, :], dim=0).to(torch.bfloat16),
            correction_bias[:],
            topk_num_experts,
            num_groups,
            topk_groups,
        )
        # Only picks up to this task's slot: later ones cannot change its expert,
        # which a key holds in its low bits (the expert count is a power of two).
        w13_pick = hl.full([], w13_slot, dtype=torch.int32)
        w13_chosen = torch.amax(w13_keys)
        for w13_earlier in hl.static_range(1, top_k):
            if w13_pick >= w13_earlier:
                w13_keys = torch.where(
                    w13_keys == w13_chosen, -9223372036854775807, w13_keys
                )
                w13_chosen = torch.amax(w13_keys)
        w13_expert = (topk_num_experts - 1 - (w13_chosen & (topk_num_experts - 1))).to(
            torch.int32
        )
        w13_expert_address = w13_expert.to(torch.int64)
        w13_scale_pair = hl.arange(2)
        w13_task_scales = torch.where(
            w13_scale_pair == 0,
            alpha1[w13_expert],
            torch.sum(activation_global_scale[:]),
        )
        w13_tma_row = (
            w13_expert * (twice_intermediate // 128) + w13_tile_output_group.begin // 4
        )
        w13_physical_row_index = (w13_tile_output_group.begin * 32 + hl.arange(128)).to(
            torch.int64
        )
        w13_weight_row = (
            w13_expert_address * twice_intermediate + w13_physical_row_index
        )
        w13_accumulator = hl.zeros([128, MMA_N], dtype=torch.float32)
        for w13_local_group in hl.tile(w13_groups_per_split, block_size=32):
            w13_group_begin = w13_split * w13_groups_per_split + w13_local_group.begin
            w13_group_index = w13_split * w13_groups_per_split + w13_local_group.index
            w13_packed_index = (
                w13_group_begin * 8 + hl.arange(w13_local_group.block_size * 8)
            ).to(torch.int64)
            w13_lhs = w13_tma[w13_tma_row, :, w13_packed_index]
            w13_lhs_scale = flat_w13_scale[
                w13_weight_row[:, None], w13_group_index[None, :]
            ]
            w13_hidden_bytes = hidden_q[0, w13_packed_index]
            w13_rhs = w13_hidden_bytes[:, None].expand(w13_hidden_bytes.size(0), MMA_N)
            w13_hidden_scale = hidden_scale_bytes[0, w13_group_index]
            w13_rhs_scale = w13_hidden_scale[None, :].expand(
                MMA_N, w13_local_group.block_size
            )
            w13_accumulator = hl.dot_scaled(
                w13_lhs,
                w13_lhs_scale,
                "e2m1",
                w13_rhs,
                w13_rhs_scale,
                "e2m1",
                acc=w13_accumulator,
                out_dtype=torch.float32,
            )
        w13_partial[
            w13_slot,
            w13_tile_output_group.begin // 4,
            w13_split,
            :,
        ] = first_mma_column(w13_accumulator)
        w13_scales[
            w13_slot, w13_tile_output_group.begin // 4, w13_split, w13_scale_pair
        ] = w13_task_scales
    for w13_tile_slot, w13_tile_output_group in hl.tile(
        [top_k, activation_groups], block_size=[1, 4]
    ):
        w13_slot = w13_tile_slot.begin
        w13_finish_scales = w13_scales[
            w13_slot, w13_tile_output_group.begin // 4, 0, hl.arange(2)
        ]
        w13_first_scale = torch.sum(w13_finish_scales * (hl.arange(2) == 0))
        w13_global_scale = torch.sum(w13_finish_scales * (hl.arange(2) == 1))
        w13_preactivation = (
            w13_partial[
                w13_slot,
                w13_tile_output_group.begin // 4,
                :,
                hl.arange(128),
            ]
            .sum(dim=0)
            .reshape(64, 2)
        )
        w13_pair = hl.arange(2)
        w13_gate = torch.sum(w13_preactivation * (w13_pair[None, :] == 0), dim=-1)
        w13_up = torch.sum(w13_preactivation * (w13_pair[None, :] == 1), dim=-1)
        w13_gate = w13_gate * w13_first_scale
        w13_up = w13_up * w13_first_scale
        w13_activated = w13_gate * torch.sigmoid(w13_gate) * w13_up
        w13_activated_groups = w13_activated.reshape(4, 16)
        w13_block_scale_f32 = (
            torch.amax(torch.abs(w13_activated_groups), dim=-1)
            * w13_global_scale
            / FP4_MAX
        )
        w13_block_scale = w13_block_scale_f32.to(torch.float8_e4m3fn)
        w13_actual_scale = w13_block_scale.to(torch.float32)
        w13_divisor = torch.where(w13_actual_scale > 0, w13_actual_scale, 1.0)
        w13_scaled = w13_activated_groups * w13_global_scale / w13_divisor[:, None]
        w13_nibbles = fp4_nibble(w13_scaled)
        w13_low, w13_high = hl.split(w13_nibbles.reshape(4, 8, 2))
        w13_packed = w13_low | w13_high << 4
        activation_q_groups[w13_slot, w13_tile_output_group, :] = w13_packed.reshape(
            4, 8
        ).to(torch.uint8)
        activation_scale[w13_slot, w13_tile_output_group] = w13_block_scale
    for shared_activation_tile_group in hl.tile(activation_groups, block_size=4):
        shared_preactivation = (
            shared_w13_partial[
                shared_activation_tile_group.begin // 4, :, hl.arange(128)
            ]
            .sum(dim=0)
            .reshape(64, 2)
        )
        shared_pair = hl.arange(2)
        shared_first_scale = torch.sum(shared_alpha1[:])
        shared_w13_preactivation[
            shared_activation_tile_group.begin * 32 + hl.arange(128)
        ] = (shared_preactivation.reshape(128) * shared_first_scale).to(torch.bfloat16)
        # The shared gate/up output is rounded to BF16, as the shared GEMM's output is.
        shared_gate = (
            (
                torch.sum(shared_preactivation * (shared_pair[None, :] == 0), dim=-1)
                * shared_first_scale
            )
            .to(torch.bfloat16)
            .to(torch.float32)
        )
        shared_up = (
            (
                torch.sum(shared_preactivation * (shared_pair[None, :] == 1), dim=-1)
                * shared_first_scale
            )
            .to(torch.bfloat16)
            .to(torch.float32)
        )
        shared_activated = (
            (shared_gate * torch.sigmoid(shared_gate) * shared_up)
            .to(torch.bfloat16)
            .to(torch.float32)
            .reshape(4, 16)
        )
        shared_global_scale = torch.sum(activation_global_scale[:])
        shared_block_scale_f32 = (
            torch.amax(torch.abs(shared_activated), dim=-1)
            * shared_global_scale
            / FP4_MAX
        )
        shared_block_scale = shared_block_scale_f32.to(torch.float8_e4m3fn)
        shared_actual_scale = shared_block_scale.to(torch.float32)
        shared_divisor = torch.where(shared_actual_scale > 0, shared_actual_scale, 1.0)
        shared_scaled = shared_activated * shared_global_scale / shared_divisor[:, None]
        shared_nibbles = fp4_nibble(shared_scaled)
        shared_low, shared_high = hl.split(shared_nibbles.reshape(4, 8, 2))
        shared_packed = shared_low | shared_high << 4
        shared_activation_q_groups[0, shared_activation_tile_group, :] = (
            shared_packed.reshape(4, 8).to(torch.uint8)
        )
        shared_activation_scale[0, shared_activation_tile_group] = shared_block_scale
    for w2_tile_slot, w2_tile_row in hl.tile(
        [w2_top_k, hidden], block_size=[1, w2_block_row]
    ):
        w2_slot = w2_tile_slot.begin
        w2_expert = topk_ids[0, w2_slot]
        w2_expert_address = w2_expert.to(torch.int64)
        w2_row_index = w2_tile_row.index.to(torch.int32)
        w2_weight_row = w2_expert_address * hidden + w2_row_index
        # Load before the K loop, which starts by waiting for the activation.
        w2_second_scale = alpha2[w2_expert]
        # The weights and their scales do not depend on the activation: load
        # them all (two K halves) before the loop below, which starts by
        # waiting for the activation, so they arrive during that wait.
        w2_half = w2_groups // 2
        w2_lhs_low = w2[w2_expert, w2_tile_row, 0 : w2_half * 8]
        w2_lhs_high = w2[w2_expert, w2_tile_row, w2_half * 8 : w2_groups * 8]
        w2_lhs_scale_low = flat_w2_scale[
            w2_weight_row[:, None], hl.arange(w2_half)[None, :]
        ]
        w2_lhs_scale_high = flat_w2_scale[
            w2_weight_row[:, None], w2_half + hl.arange(w2_half)[None, :]
        ]
        w2_accumulator = hl.zeros([w2_block_row, MMA_N], dtype=torch.float32)
        for _w2_all_groups in hl.tile(w2_groups, block_size=w2_groups):
            w2_activation_low = activation_q[w2_slot, hl.arange(w2_half * 8)]
            w2_activation_high = activation_q[
                w2_slot, w2_half * 8 + hl.arange(w2_half * 8)
            ]
            w2_activation_scale_low = activation_scale[w2_slot, hl.arange(w2_half)]
            w2_activation_scale_high = activation_scale[
                w2_slot, w2_half + hl.arange(w2_half)
            ]
            w2_accumulator = hl.dot_scaled(
                w2_lhs_low,
                w2_lhs_scale_low,
                "e2m1",
                w2_activation_low[:, None].expand(w2_half * 8, MMA_N),
                w2_activation_scale_low[None, :].expand(MMA_N, w2_half),
                "e2m1",
                acc=w2_accumulator,
                out_dtype=torch.float32,
            )
            w2_accumulator = hl.dot_scaled(
                w2_lhs_high,
                w2_lhs_scale_high,
                "e2m1",
                w2_activation_high[:, None].expand(w2_half * 8, MMA_N),
                w2_activation_scale_high[None, :].expand(MMA_N, w2_half),
                "e2m1",
                acc=w2_accumulator,
                out_dtype=torch.float32,
            )
        expert_output[w2_tile_slot, w2_tile_row] = (
            (first_mma_column(w2_accumulator) * w2_second_scale)
            .reshape(1, w2_block_row)
            .to(torch.bfloat16)
        )
    for shared_w2_tile_row in hl.tile(hidden, block_size=w2_block_row):
        shared_w2_second_scale = torch.sum(shared_alpha2[:])
        shared_w2_accumulator = hl.zeros([w2_block_row, MMA_N], dtype=torch.float32)
        for shared_w2_tile_group in hl.tile(w2_groups, block_size=16):
            shared_w2_packed_index = (
                shared_w2_tile_group.begin * 8
                + hl.arange(shared_w2_tile_group.block_size * 8)
            ).to(torch.int64)
            shared_w2_lhs = flat_shared_w2[shared_w2_tile_row, shared_w2_packed_index]
            shared_w2_lhs_scale = flat_shared_w2_scale[
                shared_w2_tile_row.index[:, None], shared_w2_tile_group.index[None, :]
            ]
            shared_w2_activation_bytes = shared_activation_q_flat[
                shared_w2_packed_index
            ]
            shared_w2_rhs = shared_w2_activation_bytes[:, None].expand(
                shared_w2_activation_bytes.size(0), MMA_N
            )
            shared_w2_activation_scale = shared_activation_scale[
                0, shared_w2_tile_group
            ]
            shared_w2_rhs_scale = shared_w2_activation_scale[None, :].expand(
                MMA_N, shared_w2_tile_group.block_size
            )
            shared_w2_accumulator = hl.dot_scaled(
                shared_w2_lhs,
                shared_w2_lhs_scale,
                "e2m1",
                shared_w2_rhs,
                shared_w2_rhs_scale,
                "e2m1",
                acc=shared_w2_accumulator,
                out_dtype=torch.float32,
            )
        shared_expert_output[0, shared_w2_tile_row] = (
            first_mma_column(shared_w2_accumulator) * shared_w2_second_scale
        ).to(torch.bfloat16)
    for communication_tile_n in hl.tile(hidden, block_size=communication_n):
        # Load the residual before waiting on the peers, so its DRAM latency
        # overlaps the wait rather than following it.
        communication_residual = residual[0, communication_tile_n].to(torch.float32)
        communication_slots = hl.arange(top_k)
        communication_weights = (
            hl.load(topk_weights, [0, communication_slots])
            .to(torch.bfloat16)
            .to(torch.float32)
        )
        communication_total = hl.zeros([communication_tile_n], dtype=torch.float32)
        for remote_expert_output, remote_shared_output in zip(
            remote_expert_outputs, remote_shared_outputs, strict=True
        ):
            # This peer's MoE output, rounded as the peer itself would round it.
            communication_routed = torch.sum(
                remote_expert_output[communication_slots, communication_tile_n].to(
                    torch.float32
                )
                * communication_weights[:, None],
                dim=0,
                keepdim=True,
            ).to(torch.bfloat16)
            communication_shared = remote_shared_output[
                hl.arange(1), communication_tile_n
            ]
            communication_total = communication_total + (
                (communication_routed + communication_shared)
                .reshape(-1)
                .to(torch.bfloat16)
                .to(torch.float32)
            )
        communication_sum = communication_total + communication_residual
        residual_output[0, communication_tile_n] = communication_sum.to(torch.bfloat16)
        norm_partial[communication_tile_n.begin // communication_n] = torch.sum(
            communication_sum * communication_sum
        )
        norm_input[0, communication_tile_n] = communication_sum
    # Every normalizing task needs all partial sums of squares, so Helion puts a
    # root barrier between the two loops.
    for norm_tile_n in hl.tile(hidden, block_size=communication_n):
        norm_scale = torch.rsqrt(torch.sum(norm_partial[:]) / hidden + NORM_EPS)
        output[0, norm_tile_n] = (
            norm_input[0, norm_tile_n]
            * norm_scale
            * norm_weight[norm_tile_n].to(torch.float32)
        ).to(torch.bfloat16)
    return (
        output,
        router_logits,
        topk_weights,
        topk_ids,
        input_hidden_q,
        input_hidden_scale,
        activation_q,
        activation_scale,
        expert_output,
        shared_w13_preactivation,
        shared_activation_q,
        shared_activation_scale,
        shared_expert_output,
        residual_output,
    )


def _config() -> helion.Config:
    range_num_stages = [0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 2, 0, 1, 0, 0]
    range_multi_buffers = [
        None,
        None,
        None,
        True,
        None,
        None,
        False,
        None,
        None,
        None,
        True,
        None,
        True,
        None,
        None,
    ]
    range_flattens = [
        None,
        None,
        None,
        False,
        None,
        None,
        False,
        None,
        None,
        None,
        False,
        None,
        False,
        None,
        None,
    ]
    return helion.Config(
        block_sizes=[4, 32, 128],
        cross_loop_pipeline="dynamic",
        host_tensor_descriptors=True,
        indexing=[
            "tensor_descriptor" if index in (22, 24, 40, 41, 44, 45) else "pointer"
            for index in range(80)
        ],
        maxnreg=None,
        num_sm_multiplier=NUM_SM_MULTIPLIER,
        num_stages=1,
        num_warps=4,
        pid_type="persistent_blocked",
        range_flattens=range_flattens,
        range_multi_buffers=range_multi_buffers,
        range_num_stages=range_num_stages,
    )


deepseek_v3_moe_nvfp4_tp = helion.kernel(
    _deepseek_v3_moe_nvfp4_tp,
    config=_config(),
    static_shapes=False,
    backend="triton",
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)


def _rms_norm_reference(
    summed: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """(normalized output, residual sum) of the block's epilogue, in FP32."""
    total = summed.float() + residual.float()
    scale = torch.rsqrt(total.square().mean(-1, keepdim=True) + NORM_EPS)
    return (total * scale * weight.float()).to(torch.bfloat16), total.to(torch.bfloat16)


def _check_similar(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    error = _setup._similarity_error(actual, expected)
    if not math.isfinite(error) or error > 1e-3:
        raise AssertionError(f"{name} similarity error {error:.6g} exceeds 1e-3")


def _make_fused_all_reduce(
    rank: int, group: dist.ProcessGroup, hidden: int
) -> tuple[
    Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], None], str
]:
    """vLLM's fused all-reduce + residual add + RMSNorm (FlashInfer), as its
    AllReduceFusionPass calls it for a decode token: the default backend, PDL,
    FP32 accumulation and the one-shot kernel.  Returns it and its backend."""
    import flashinfer.comm as flashinfer_comm
    from flashinfer.comm.mnnvl import TorchDistBackend

    workspace = None
    for backend in ("mnnvl", "trtllm"):
        try:
            workspace = flashinfer_comm.create_allreduce_fusion_workspace(
                backend=backend,
                world_size=WORLD_SIZE,
                rank=rank,
                max_token_num=1024,
                hidden_dim=hidden,
                dtype=torch.bfloat16,
                comm_backend=TorchDistBackend(group=group),
                group=group,
            )
        except Exception:  # mnnvl needs NVSwitch multicast; vLLM falls back too.
            continue
        if backend != "mnnvl" or getattr(workspace, "mc_ptr", 0):
            break
        workspace.destroy()
        workspace = None
    if workspace is None:
        raise RuntimeError("no FlashInfer all-reduce fusion workspace")
    use_oneshot = None if workspace.backend == "mnnvl" else True

    def fused(
        moe_output: torch.Tensor,
        residual: torch.Tensor,
        norm_weight: torch.Tensor,
        norm_output: torch.Tensor,
    ) -> None:
        # With a separate norm output, the residual sum overwrites the
        # all-reduce input and the residual itself is left unchanged.
        flashinfer_comm.allreduce_fusion(
            input=moe_output,
            workspace=workspace,
            pattern=flashinfer_comm.AllReduceFusionPattern.kARResidualRMSNorm,
            launch_with_pdl=True,
            trigger_completion_at_end=True,
            residual_out=moe_output,
            norm_out=norm_output,
            residual_in=residual,
            rms_gamma=norm_weight,
            rms_eps=NORM_EPS,
            use_oneshot=use_oneshot,
            fp32_acc=True,
        )

    return fused, workspace.backend


def _run(
    rank: int,
    local_rank: int,
    group: dist.ProcessGroup,
    *,
    verbose: bool,
) -> dict[str, Any]:
    import torch.distributed._symmetric_memory as symm_mem

    device = torch.device("cuda", local_rank)
    shape = _setup.Shape(intermediate=2048 // WORLD_SIZE)
    tensors = _setup._allocate(shape)
    symmetric_output = symm_mem.empty(
        (shape.top_k, shape.hidden), dtype=torch.bfloat16, device=device
    )
    symm_mem.rendezvous(symmetric_output, group.group_name)
    symmetric_shared = symm_mem.empty(
        (1, shape.hidden), dtype=torch.bfloat16, device=device
    )
    symm_mem.rendezvous(symmetric_shared, group.group_name)
    # Every rank holds the same residual stream and norm weight.
    generator = torch.Generator(device=device)
    generator.manual_seed(5)
    residual = torch.randn(
        1, shape.hidden, device=device, dtype=torch.bfloat16, generator=generator
    )
    norm_weight = (
        1 + 0.1 * torch.randn(shape.hidden, device=device, generator=generator)
    ).to(torch.bfloat16)
    norm_counter = torch.zeros(COUNTER_SHAPE, dtype=torch.int32, device=device)
    # Model each rank as a distinct TP shard while preserving replicated input
    # and routing.  Scaling the local output factors is enough to make peer
    # mixups observable without changing the packed NVFP4 layouts.
    rank_output_scale = 1.0 + rank / 8.0
    tensors["alpha2"].mul_(rank_output_scale)
    tensors["shared_alpha2"].mul_(rank_output_scale)
    args = (
        *_setup._kernel_args(tensors, shape),
        W13_SPLIT_K,
        norm_counter,
        residual,
        norm_weight,
        symmetric_output,
        symmetric_shared,
        group.group_name,
    )
    raw_vllm_call, backend, routing_replay = _setup._make_vllm_call(tensors, shape)
    vllm_call = cast("Callable[[], torch.Tensor]", raw_vllm_call)
    fused_all_reduce, all_reduce_backend = _make_fused_all_reduce(
        rank, group, shape.hidden
    )
    production_output = torch.empty_like(residual)
    case_metrics: dict[str, dict[str, tuple[float, float]]] = {}
    dispatch_signature = None

    for label, selected_ids in _setup.ROUTING_CASES:
        _setup._set_routing_case(tensors, selected_ids)

        def helion_call() -> tuple[torch.Tensor, ...]:
            return cast("tuple[torch.Tensor, ...]", deepseek_v3_moe_nvfp4_tp(*args))

        def production() -> torch.Tensor:
            fused_all_reduce(vllm_call(), residual, norm_weight, production_output)
            return production_output

        vllm_local = vllm_call().clone()
        gathered_local = [torch.empty_like(vllm_local) for _ in range(WORLD_SIZE)]
        dist.all_gather(gathered_local, vllm_local)
        if any(
            torch.equal(left, right)
            for index, left in enumerate(gathered_local)
            for right in gathered_local[index + 1 :]
        ):
            raise AssertionError("rank-local MoE outputs must differ")
        summed = vllm_local.float()
        dist.all_reduce(summed)
        expected, expected_residual = _rms_norm_reference(summed, residual, norm_weight)
        actual = helion_call()
        current_signature = _setup._dispatch_cache_signature(deepseek_v3_moe_nvfp4_tp)
        if dispatch_signature is None:
            dispatch_signature = current_signature
        elif current_signature != dispatch_signature:
            raise AssertionError("runtime routing triggered a recompilation")
        production_actual = production().clone()
        torch.cuda.synchronize()
        dist.barrier()
        torch.testing.assert_close(
            actual[3], routing_replay.to(torch.int32), rtol=0, atol=0
        )
        # Helion and FlashInfer quantize and accumulate differently, so their
        # outputs agree in aggregate rather than elementwise.
        _check_similar("helion output", actual[0], expected)
        _check_similar("helion residual sum", actual[-1], expected_residual)
        _check_similar("vLLM output", production_actual, expected)

        launches: dict[str, Callable[[], object]] = {
            "distributed_helion": helion_call,
            f"vllm_{backend}_{all_reduce_backend}_all_reduce_norm": production,
        }

        def validate_replay(
            name: str, value: object, expected: torch.Tensor = expected
        ) -> None:
            output = (
                cast("tuple[torch.Tensor, ...]", value)[0]
                if name == "distributed_helion"
                else cast("torch.Tensor", value)
            )
            _check_similar(f"{name} replay", output, expected)

        results = benchmark(launches, validate=validate_replay)
        case_metrics[label] = results
        if verbose and rank == 0:
            print(f"routing={label} experts={selected_ids}")
            for name, (warm, cold) in results.items():
                print(f"{name:>44s}: {warm:7.2f} us warm, {cold:7.2f} us cold L2")

    speedups = []
    for results in case_metrics.values():
        helion_cold = results["distributed_helion"][1]
        production_name = next(name for name in results if name.startswith("vllm_"))
        speedups.append(results[production_name][1] / helion_cold)
    return {
        "helion_wins": sum(speedup > 1 for speedup in speedups),
        "total": len(speedups),
        "geomean": math.prod(speedups) ** (1.0 / len(speedups)),
        "best_speedup": max(speedups),
        "results": case_metrics,
    }


@torch.inference_mode()
def main(verbose: bool = True) -> dict[str, Any]:
    """Run TP4 routing correctness and the paired cold-L2 comparison."""
    rank, local_rank, group = initialize(
        kernel_name="deepseek_v3_moe_nvfp4_tp",
        world_size=WORLD_SIZE,
        signal_pad_bytes=SIGNAL_PAD_BYTES,
    )
    try:
        return _run(rank, local_rank, group, verbose=verbose)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
