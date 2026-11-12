# ruff: noqa: ANN001, ANN201
# pyrefly: ignore-errors
"""DeepSeek-V3 NVFP4 MoE with a fused TP4 all-reduce, for T tokens (decode, or
speculative verify).

``deepseek_v3_moe_nvfp4_verify`` on this rank's intermediate shard (each routed
task serves one distinct expert of the window for every token): its per-token
output goes to symmetric memory, and the block then ends like
``deepseek_v3_moe_nvfp4_tp``: every rank sums the token's outputs of all ranks,
adds the residual and applies RMSNorm, in the same kernel.
"""

from __future__ import annotations

from pretuned_kernels.megakernels._distributed import initialize
from pretuned_kernels.megakernels.deepseek_v3_moe_nvfp4._common import FP4_MAX
from pretuned_kernels.megakernels.deepseek_v3_moe_nvfp4._common import distinct_table
from pretuned_kernels.megakernels.deepseek_v3_moe_nvfp4._common import expert_of
from pretuned_kernels.megakernels.deepseek_v3_moe_nvfp4._common import fp4_nibble
from pretuned_kernels.megakernels.deepseek_v3_moe_nvfp4._common import scale_tile
from pretuned_kernels.megakernels.deepseek_v3_moe_nvfp4._common import top_k_picks
from pretuned_kernels.megakernels.deepseek_v3_moe_nvfp4.deepseek_v3_moe_nvfp4 import (
    Shape,
)
from pretuned_kernels.megakernels.deepseek_v3_moe_nvfp4.deepseek_v3_moe_nvfp4_verify import (
    VERIFY_ROUTING_CASES,
)
from pretuned_kernels.megakernels.deepseek_v3_moe_nvfp4.deepseek_v3_moe_nvfp4_verify import (
    _allocate_verify,
)
from pretuned_kernels.megakernels.deepseek_v3_moe_nvfp4.deepseek_v3_moe_nvfp4_verify import (
    _set_verify_routing,
)
from pretuned_kernels.megakernels.deepseek_v3_moe_nvfp4.deepseek_v3_moe_nvfp4_verify import (
    _verify_args,
)
from pretuned_kernels.megakernels.deepseek_v3_moe_nvfp4.deepseek_v3_moe_nvfp4_verify import (
    deepseek_v3_moe_nvfp4_verify,
)
from pretuned_kernels.megakernels.deepseek_v3_moe_nvfp4_tp._common import routing_keys
from pretuned_kernels.megakernels.deepseek_v3_moe_nvfp4_tp.deepseek_v3_moe_nvfp4_tp import (
    NORM_EPS,
)
from pretuned_kernels.megakernels.deepseek_v3_moe_nvfp4_tp.deepseek_v3_moe_nvfp4_tp import (
    SIGNAL_PAD_BYTES,
)
from pretuned_kernels.megakernels.deepseek_v3_moe_nvfp4_tp.deepseek_v3_moe_nvfp4_tp import (
    WORLD_SIZE,
)
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

import helion
import helion.language as hl

# The GEMM operands' views (router, shared and routed W13/W2 weights, scales and
# activations), as load/store slots in codegen order; every other access is a
# pointer.
_TMA_SLOTS = (
    *(0, 1, 8, 9, 10, 11, 23, 24, 25, 26, 32, 33),
    *(34, 35, 38, 39, 40, 41, 47, 48, 49, 50),
)


@helion.kernel(
    static_shapes=False,
    backend="gluon",
    triton_do_not_specialize=False,
    # One persistent CTA per SM with a 4-deep TMA ring under dynamic dispatch:
    # the routed W13/W2 roots (live counts set by the routing) claim their
    # tasks past a fixed first wave.
    config=helion.Config(
        block_sizes=[16, 128, 32, 16, 16, 16, 16, 16, 512],
        num_warps=4,
        pid_type="persistent_blocked",
        cross_loop_pipeline="dynamic",
        num_sm_multiplier=1,
        num_stages=4,
        range_warp_specializes=[True] * 16,
        indexing=[
            "tensor_descriptor" if index in _TMA_SLOTS else "pointer"
            for index in range(128)
        ],
    ),
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def deepseek_v3_moe_nvfp4_tp_verify(
    hidden_input,
    router_weight,
    correction_bias,
    w13,
    w13_scale,
    w2,
    w2_scale,
    shared_w13,
    shared_w13_scale,
    shared_w2,
    shared_w2_scale,
    alpha1,
    alpha2,
    shared_alpha1,
    shared_alpha2,
    input_global_scale,
    activation_global_scale,
    top_k: int,
    num_groups: int,
    topk_groups: int,
    routed_scale: float,
    residual,
    norm_weight,
    symmetric_partial,
    group_name: hl.ProcessGroupName,
):
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
    hl.specialize(w13_scale.size(3))
    hl.specialize(w2.size(0))
    hl.specialize(w2.size(1))
    hl.specialize(w2.size(2))
    hl.specialize(w2_scale.size(0))
    hl.specialize(w2_scale.size(1))
    hl.specialize(w2_scale.size(2))
    hl.specialize(w2_scale.size(3))
    hl.specialize(shared_w13.size(0))
    hl.specialize(shared_w13.size(1))
    hl.specialize(shared_w13_scale.size(0))
    hl.specialize(shared_w13_scale.size(1))
    hl.specialize(shared_w13_scale.size(2))
    hl.specialize(shared_w2.size(0))
    hl.specialize(shared_w2.size(1))
    hl.specialize(shared_w2_scale.size(0))
    hl.specialize(shared_w2_scale.size(1))
    hl.specialize(shared_w2_scale.size(2))
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
    hl.specialize(w13_scale.stride(3))
    hl.specialize(w2.stride(0))
    hl.specialize(w2.stride(1))
    hl.specialize(w2.stride(2))
    hl.specialize(w2_scale.stride(0))
    hl.specialize(w2_scale.stride(1))
    hl.specialize(w2_scale.stride(2))
    hl.specialize(w2_scale.stride(3))
    hl.specialize(shared_w13.stride(0))
    hl.specialize(shared_w13.stride(1))
    hl.specialize(shared_w13_scale.stride(0))
    hl.specialize(shared_w13_scale.stride(1))
    hl.specialize(shared_w13_scale.stride(2))
    hl.specialize(shared_w2.stride(0))
    hl.specialize(shared_w2.stride(1))
    hl.specialize(shared_w2_scale.stride(0))
    hl.specialize(shared_w2_scale.stride(1))
    hl.specialize(shared_w2_scale.stride(2))
    hl.specialize(alpha1.stride(0))
    hl.specialize(alpha2.stride(0))
    hl.specialize(shared_alpha1.stride(0))
    hl.specialize(shared_alpha2.stride(0))
    hl.specialize(input_global_scale.stride(0))
    hl.specialize(activation_global_scale.stride(0))
    top_k = hl.specialize(top_k)
    num_groups = hl.specialize(num_groups)
    topk_groups = hl.specialize(topk_groups)
    hl.specialize(residual.size())
    hl.specialize(residual.stride())
    hl.specialize(norm_weight.size(0))
    hl.specialize(norm_weight.stride(0))
    hl.specialize(symmetric_partial.size())
    hl.specialize(symmetric_partial.stride())

    rows, hidden = hidden_input.shape
    experts = router_weight.size(0)
    twice_intermediate, packed_hidden = w13.shape[1:]
    intermediate = twice_intermediate // 2
    packed_intermediate = intermediate // 2
    hidden_groups = hidden // 16
    activation_groups = intermediate // 16
    assert top_k == 8 and topk_groups == 4 and experts <= 256
    # The window's distinct routed experts: at most every token's top-k.
    slots = rows * top_k
    # K steps of 512 FP4 values: 256 packed bytes and 32 scales per row.
    w13_steps = packed_hidden // 256
    w2_steps = packed_intermediate // 256
    # A task's 128 weight rows give 64 SwiGLU outputs: 4 quantization groups.
    w13_tiles = twice_intermediate // 128

    # 28 chunks of 256: two 32 KB steps per task.
    router_chunks = 28
    router_chunk_k = hidden // router_chunks
    router_weight_chunked = router_weight.view(experts, router_chunks, router_chunk_k)
    hidden_chunked = hidden_input.view(rows, router_chunks, router_chunk_k)
    router_partials = torch.empty(
        (rows, router_chunks, experts), dtype=torch.float32, device=hidden_input.device
    )
    router_logits = torch.empty(
        (rows, experts), dtype=torch.bfloat16, device=hidden_input.device
    )
    input_q = torch.empty(
        (rows, packed_hidden), dtype=torch.uint8, device=hidden_input.device
    )
    input_q_groups = input_q.view(rows, hidden_groups, 8)
    input_q_steps = input_q.view(rows, w13_steps, 256)
    input_scale = torch.empty(
        (rows, hidden_groups), dtype=torch.float8_e4m3fn, device=hidden_input.device
    )
    input_scale_steps = input_scale.view(rows, w13_steps, 32)
    topk_weights = torch.empty(
        (rows, top_k), dtype=torch.float32, device=hidden_input.device
    )
    topk_ids = torch.empty((rows, top_k), dtype=torch.int32, device=hidden_input.device)
    routes = topk_ids.view(slots)

    w13_view = w13.view(experts, twice_intermediate, w13_steps, 256)
    # The routed W13's first K half and its FP32 partials.
    w13_half = w13_steps // 2
    w13_partials = torch.empty(
        (slots, rows, twice_intermediate), dtype=torch.float32, device=w13.device
    )
    shared_w13_view = shared_w13.view(twice_intermediate, w13_steps, 256)
    w2_view = w2.view(experts, hidden, w2_steps, 256)
    shared_w2_view = shared_w2.view(hidden, w2_steps, 256)

    activation_q = torch.empty(
        (slots, rows, packed_intermediate), dtype=torch.uint8, device=w13.device
    )
    activation_q_groups = activation_q.view(slots, rows, w13_tiles, 4, 8)
    activation_q_steps = activation_q.view(slots, rows, w2_steps, 256)
    activation_scale = torch.empty(
        (slots, rows, activation_groups), dtype=torch.float8_e4m3fn, device=w13.device
    )
    activation_scale_groups = activation_scale.view(slots, rows, w13_tiles, 4)
    activation_scale_steps = activation_scale.view(slots, rows, w2_steps, 32)
    # Each token's eight routed expert rows, in top-k order.
    routed_output = torch.empty(
        (rows, top_k, hidden), dtype=torch.bfloat16, device=w2.device
    )
    routed_rows = routed_output.view(rows * top_k, hidden)
    shared_w13_preactivation = torch.empty(
        (rows, twice_intermediate), dtype=torch.bfloat16, device=w13.device
    )
    shared_activation_q = torch.empty(
        (rows, packed_intermediate), dtype=torch.uint8, device=w13.device
    )
    shared_activation_q_groups = shared_activation_q.view(rows, w13_tiles, 4, 8)
    shared_activation_q_steps = shared_activation_q.view(rows, w2_steps, 256)
    shared_activation_scale = torch.empty(
        (rows, activation_groups), dtype=torch.float8_e4m3fn, device=w13.device
    )
    shared_activation_scale_groups = shared_activation_scale.view(rows, w13_tiles, 4)
    shared_activation_scale_steps = shared_activation_scale.view(rows, w2_steps, 32)
    shared_expert_output = torch.empty(
        (rows, hidden), dtype=torch.bfloat16, device=w2.device
    )
    # This rank's MoE output in symmetric memory: every rank's communication
    # tasks read it as soon as it is written.
    assert symmetric_partial.size() == (rows, hidden)
    remote_partials = torch.ops.symm_mem.get_remote_tensors(
        symmetric_partial, group_name
    )
    output = torch.empty((rows, hidden), dtype=torch.bfloat16, device=w2.device)
    residual_output = torch.empty_like(output)
    # A literal: a module-level int reaches the kernel as a runtime scalar.
    communication_n = 512
    norm_tasks = hidden // communication_n
    norm_partial = torch.empty(
        (rows, norm_tasks), dtype=torch.float32, device=w2.device
    )
    # The FP32 pre-norm sums, read back by the normalizing loop.
    norm_input = torch.empty((rows, hidden), dtype=torch.float32, device=w2.device)

    # Router projection: K chunks of the 256 x 7168 BF16 weight, summed in order
    # by the routing root.
    for router_tile_e, router_tile_c, router_tile_t in hl.tile(
        [experts, router_chunks, rows], block_size=[128, 1, None]
    ):
        router_acc = hl.zeros([router_tile_e, router_tile_t], dtype=torch.float32)
        for router_tile_k in hl.tile(router_chunk_k):
            router_acc = hl.dot(
                router_weight_chunked[
                    router_tile_e, router_tile_c.begin, router_tile_k
                ],
                hidden_chunked[router_tile_t, router_tile_c.begin, router_tile_k].T,
                acc=router_acc,
            )
        router_partials[router_tile_t, router_tile_c, router_tile_e] = router_acc.T[
            :, None, :
        ]

    for quant_tile_row, quant_tile_group in hl.tile(
        [rows, hidden_groups], block_size=[1, None]
    ):
        packed_lane = hl.arange(8)[None, :]
        group_offsets = quant_tile_group.index[:, None] * 16
        low_values = hidden_input[quant_tile_row, group_offsets + packed_lane * 2].to(
            torch.float32
        )
        high_values = hidden_input[
            quant_tile_row, group_offsets + packed_lane * 2 + 1
        ].to(torch.float32)
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
        input_q_groups[quant_tile_row, quant_tile_group, :] = (low | high << 4).to(
            torch.uint8
        )
        input_scale[quant_tile_row, quant_tile_group] = block_scale

    # Shared W13, its SwiGLU and NVFP4 quantization (gate/up rows interleaved).
    for shared_w13_tile_n, shared_w13_tile_t in hl.tile(
        [twice_intermediate, rows], block_size=[128, None]
    ):
        shared_w13_acc = hl.zeros(
            [shared_w13_tile_n, shared_w13_tile_t], dtype=torch.float32
        )
        for shared_w13_tile_s in hl.tile(w13_steps, block_size=1):
            shared_w13_acc = hl.dot_scaled(
                shared_w13_view[shared_w13_tile_n, shared_w13_tile_s.begin, :],
                scale_tile(
                    shared_w13_scale[
                        shared_w13_tile_n.begin // 128 * w13_steps
                        + shared_w13_tile_s.begin,
                        :,
                        :,
                    ]
                ),
                "e2m1",
                input_q_steps[shared_w13_tile_t, shared_w13_tile_s.begin, :].T,
                input_scale_steps[shared_w13_tile_t, shared_w13_tile_s.begin, :],
                "e2m1",
                acc=shared_w13_acc,
                out_dtype=torch.float32,
            )
        shared_tokens = shared_w13_acc.size(1)
        shared_preactivation = (shared_w13_acc.T * torch.sum(shared_alpha1[:])).to(
            torch.bfloat16
        )
        shared_w13_preactivation[shared_w13_tile_t, shared_w13_tile_n] = (
            shared_preactivation
        )
        shared_gate, shared_up = hl.split(
            shared_preactivation.to(torch.float32).reshape(shared_tokens, 64, 2)
        )
        shared_activated = (
            (shared_gate * torch.sigmoid(shared_gate) * shared_up)
            .to(torch.bfloat16)
            .to(torch.float32)
            .reshape(shared_tokens, 4, 16)
        )
        shared_global_scale = torch.sum(activation_global_scale[:])
        shared_block_scale = (
            torch.amax(torch.abs(shared_activated), dim=-1)
            * shared_global_scale
            / FP4_MAX
        ).to(torch.float8_e4m3fn)
        shared_actual_scale = shared_block_scale.to(torch.float32)
        shared_divisor = torch.where(shared_actual_scale > 0, shared_actual_scale, 1.0)
        shared_low, shared_high = hl.split(
            fp4_nibble(
                shared_activated * shared_global_scale / shared_divisor[:, :, None]
            ).reshape(shared_tokens, 4, 8, 2)
        )
        shared_tile = shared_w13_tile_n.begin // 128
        shared_activation_q_groups[shared_w13_tile_t, shared_tile, :, :] = (
            shared_low | shared_high << 4
        ).to(torch.uint8)
        shared_activation_scale_groups[shared_w13_tile_t, shared_tile, :] = (
            shared_block_scale
        )

    # Routing, one token per task.
    for topk_tile_t in hl.tile(rows, block_size=1):
        topk_token = topk_tile_t.begin
        # The bias loads under the partials' sum.
        topk_bias = correction_bias[:]
        logits = torch.sum(router_partials[topk_token, :, :], dim=0).to(torch.bfloat16)
        router_logits[topk_tile_t, :] = logits[None, :]
        topk_scores, topk_keys, topk_expert_indices = routing_keys(
            logits, topk_bias, experts, num_groups, topk_groups
        )
        topk_id, topk_weight, topk_denominator = top_k_picks(
            topk_scores, topk_keys, topk_expert_indices, experts
        )
        topk_ids[topk_tile_t, :] = topk_id[None, :]
        topk_weights[topk_tile_t, :] = (topk_weight / topk_denominator * routed_scale)[
            None, :
        ]

    # Routed W13 per (128 rows, slot), slot j serving the window's j-th
    # distinct expert for every token, in two K halves: the first half's FP32
    # partial, then the second half with its SwiGLU and NVFP4 quantization.
    for partial_w13_tile_n, partial_w13_tile_t, partial_w13_tile_j in hl.tile(
        [twice_intermediate, rows, slots], block_size=[128, None, 1]
    ):
        partial_w13_words, partial_w13_before, partial_w13_distinct = distinct_table(
            routes[:]
        )
        if partial_w13_tile_j.begin < partial_w13_distinct:
            partial_w13_expert = expert_of(
                partial_w13_words, partial_w13_before, partial_w13_tile_j.begin
            )
            partial_w13_acc = hl.zeros(
                [partial_w13_tile_n, partial_w13_tile_t], dtype=torch.float32
            )
            for partial_w13_tile_s in hl.tile(w13_half, block_size=1):
                partial_w13_acc = hl.dot_scaled(
                    w13_view[
                        partial_w13_expert,
                        partial_w13_tile_n,
                        partial_w13_tile_s.begin,
                        :,
                    ],
                    scale_tile(
                        w13_scale[
                            partial_w13_expert,
                            partial_w13_tile_n.begin // 128 * w13_steps
                            + partial_w13_tile_s.begin,
                            :,
                            :,
                        ]
                    ),
                    "e2m1",
                    input_q_steps[partial_w13_tile_t, partial_w13_tile_s.begin, :].T,
                    input_scale_steps[partial_w13_tile_t, partial_w13_tile_s.begin, :],
                    "e2m1",
                    acc=partial_w13_acc,
                    out_dtype=torch.float32,
                )
            w13_partials[
                partial_w13_tile_j.begin, partial_w13_tile_t, partial_w13_tile_n
            ] = partial_w13_acc.T
    for w13_tile_n, w13_tile_t, w13_tile_j in hl.tile(
        [twice_intermediate, rows, slots], block_size=[128, None, 1]
    ):
        w13_words, w13_before, w13_distinct = distinct_table(routes[:])
        if w13_tile_j.begin < w13_distinct:
            w13_expert = expert_of(w13_words, w13_before, w13_tile_j.begin)
            # The epilogue's operands load under the K loop.
            w13_partial = w13_partials[w13_tile_j.begin, w13_tile_t, w13_tile_n]
            w13_alpha = alpha1[w13_expert]
            w13_global_scale = torch.sum(activation_global_scale[:])
            w13_acc = hl.zeros([w13_tile_n, w13_tile_t], dtype=torch.float32)
            for w13_tile_s in hl.tile(w13_half, w13_steps, block_size=1):
                w13_acc = hl.dot_scaled(
                    w13_view[w13_expert, w13_tile_n, w13_tile_s.begin, :],
                    scale_tile(
                        w13_scale[
                            w13_expert,
                            w13_tile_n.begin // 128 * w13_steps + w13_tile_s.begin,
                            :,
                            :,
                        ]
                    ),
                    "e2m1",
                    input_q_steps[w13_tile_t, w13_tile_s.begin, :].T,
                    input_scale_steps[w13_tile_t, w13_tile_s.begin, :],
                    "e2m1",
                    acc=w13_acc,
                    out_dtype=torch.float32,
                )
            w13_tokens = w13_acc.size(1)
            w13_preactivation = (w13_acc.T + w13_partial) * w13_alpha
            w13_gate, w13_up = hl.split(w13_preactivation.reshape(w13_tokens, 64, 2))
            w13_activated = (w13_gate * torch.sigmoid(w13_gate) * w13_up).reshape(
                w13_tokens, 4, 16
            )
            w13_block_scale = (
                torch.amax(torch.abs(w13_activated), dim=-1)
                * w13_global_scale
                / FP4_MAX
            ).to(torch.float8_e4m3fn)
            w13_actual_scale = w13_block_scale.to(torch.float32)
            w13_divisor = torch.where(w13_actual_scale > 0, w13_actual_scale, 1.0)
            w13_low, w13_high = hl.split(
                fp4_nibble(
                    w13_activated * w13_global_scale / w13_divisor[:, :, None]
                ).reshape(w13_tokens, 4, 8, 2)
            )
            w13_tile = w13_tile_n.begin // 128
            activation_q_groups[w13_tile_j.begin, w13_tile_t, w13_tile, :, :] = (
                w13_low | w13_high << 4
            ).to(torch.uint8)
            activation_scale_groups[w13_tile_j.begin, w13_tile_t, w13_tile, :] = (
                w13_block_scale
            )

    # The shared expert's W2 before the routed one: it is ready long before.
    for shared_w2_tile_n, shared_w2_tile_t in hl.tile(
        [hidden, rows], block_size=[128, None]
    ):
        shared_w2_acc = hl.zeros(
            [shared_w2_tile_n, shared_w2_tile_t], dtype=torch.float32
        )
        for shared_w2_tile_s in hl.tile(w2_steps, block_size=1):
            shared_w2_acc = hl.dot_scaled(
                shared_w2_view[shared_w2_tile_n, shared_w2_tile_s.begin, :],
                scale_tile(
                    shared_w2_scale[
                        shared_w2_tile_n.begin // 128 * w2_steps
                        + shared_w2_tile_s.begin,
                        :,
                        :,
                    ]
                ),
                "e2m1",
                shared_activation_q_steps[
                    shared_w2_tile_t, shared_w2_tile_s.begin, :
                ].T,
                shared_activation_scale_steps[
                    shared_w2_tile_t, shared_w2_tile_s.begin, :
                ],
                "e2m1",
                acc=shared_w2_acc,
                out_dtype=torch.float32,
            )
        shared_expert_output[shared_w2_tile_t, shared_w2_tile_n] = (
            shared_w2_acc.T * torch.sum(shared_alpha2[:])
        ).to(torch.bfloat16)

    for w2_tile_n, w2_tile_t, w2_tile_f in hl.tile(
        [hidden, rows, slots], block_size=[128, None, 1]
    ):
        w2_words, w2_before, w2_distinct = distinct_table(routes[:])
        if w2_tile_f.begin < w2_distinct:
            w2_expert = expert_of(w2_words, w2_before, w2_tile_f.begin)
            # The epilogue's operands load under the K loop: the scale, and
            # each token's pick of this expert (if any), which names its row.
            w2_alpha = alpha2[w2_expert]
            w2_hits = topk_ids[w2_tile_t, :] == w2_expert
            w2_acc = hl.zeros([w2_tile_n, w2_tile_t], dtype=torch.float32)
            for w2_tile_s in hl.tile(w2_steps, block_size=1):
                w2_acc = hl.dot_scaled(
                    w2_view[w2_expert, w2_tile_n, w2_tile_s.begin, :],
                    scale_tile(
                        w2_scale[
                            w2_expert,
                            w2_tile_n.begin // 128 * w2_steps + w2_tile_s.begin,
                            :,
                            :,
                        ]
                    ),
                    "e2m1",
                    activation_q_steps[
                        w2_tile_f.begin, w2_tile_t, w2_tile_s.begin, :
                    ].T,
                    activation_scale_steps[
                        w2_tile_f.begin, w2_tile_t, w2_tile_s.begin, :
                    ],
                    "e2m1",
                    acc=w2_acc,
                    out_dtype=torch.float32,
                )
            w2_pick = torch.sum(
                torch.where(w2_hits, hl.arange(top_k)[None, :], 0), dim=1
            )
            w2_used = (torch.sum(w2_hits.to(torch.int32), dim=1) > 0) & (
                w2_tile_t.index < rows
            )
            hl.store(
                routed_rows,
                [w2_tile_t.index * top_k + w2_pick, w2_tile_n],
                (w2_acc.T * w2_alpha).to(torch.bfloat16),
                extra_mask=w2_used[:, None],
            )

    # Each token's eight expert rows, weighted in top-k order, plus the shared.
    for partial_tile_t, partial_tile_n in hl.tile([rows, hidden], block_size=[1, None]):
        partial_token = partial_tile_t.begin
        partial_routed_values = routed_output[partial_token, :, partial_tile_n].to(
            torch.float32
        )
        partial_routed_weights = (
            topk_weights[partial_token, :].to(torch.bfloat16).to(torch.float32)
        )
        partial_routed_output = torch.sum(
            partial_routed_values * partial_routed_weights[:, None], dim=0
        ).to(torch.bfloat16)
        symmetric_partial[partial_token, partial_tile_n] = (
            partial_routed_output + shared_expert_output[partial_token, partial_tile_n]
        ).to(torch.bfloat16)

    # Each token's sum over ranks (in rank order), then the residual.
    for comm_tile_t, comm_tile_n in hl.tile(
        [rows, hidden], block_size=[1, communication_n]
    ):
        comm_token = comm_tile_t.begin
        # Load the residual before waiting on the peers.
        comm_residual = residual[comm_token, comm_tile_n].to(torch.float32)
        comm_total = hl.zeros([comm_tile_n], dtype=torch.float32)
        for remote_partial in remote_partials:
            comm_total = comm_total + remote_partial[comm_token, comm_tile_n].to(
                torch.float32
            )
        comm_sum = comm_total + comm_residual
        residual_output[comm_token, comm_tile_n] = comm_sum.to(torch.bfloat16)
        norm_partial[comm_token, comm_tile_n.begin // communication_n] = torch.sum(
            comm_sum * comm_sum
        )
        norm_input[comm_token, comm_tile_n] = comm_sum
    for norm_tile_t, norm_tile_n in hl.tile(
        [rows, hidden], block_size=[1, communication_n]
    ):
        norm_token = norm_tile_t.begin
        norm_scale = torch.rsqrt(
            torch.sum(norm_partial[norm_token, :]) / hidden + NORM_EPS
        )
        output[norm_token, norm_tile_n] = (
            norm_input[norm_token, norm_tile_n]
            * norm_scale
            * norm_weight[norm_tile_n].to(torch.float32)
        ).to(torch.bfloat16)
    return (
        output,
        router_logits,
        topk_weights,
        topk_ids,
        input_q,
        input_scale,
        activation_q,
        activation_scale,
        routed_output,
        shared_w13_preactivation,
        shared_activation_q,
        shared_activation_scale,
        shared_expert_output,
        residual_output,
    )


def correctness_check(window_sizes: tuple[int, ...] = (8, 16)) -> None:
    """Check every rank against the single-GPU verify kernel run on each rank's
    shard, for every routing case and window size: the routing bit for bit, the
    residual sum (the shards' outputs summed in rank order, plus the residual)
    and the RMSNorm output to rounding.  Runs on four ranks::

        torchrun --nproc-per-node=4 -m \\
            pretuned_kernels.megakernels.deepseek_v3_moe_nvfp4_tp.deepseek_v3_moe_nvfp4_tp_verify
    """
    rank, local_rank, group = initialize(
        kernel_name="deepseek_v3_moe_nvfp4_tp_verify",
        world_size=WORLD_SIZE,
        signal_pad_bytes=SIGNAL_PAD_BYTES,
    )
    try:
        device = torch.device("cuda", local_rank)
        shape = Shape(intermediate=2048 // WORLD_SIZE)
        for tokens in window_sizes:
            tensors = _allocate_verify(shape, tokens)
            # Distinct shards: scale each rank's down projections.
            tensors["alpha2"].mul_(1.0 + rank / 8.0)
            tensors["shared_alpha2"].mul_(1.0 + rank / 8.0)
            generator = torch.Generator(device=device)
            generator.manual_seed(5)
            residual = torch.randn(
                tokens,
                shape.hidden,
                device=device,
                dtype=torch.bfloat16,
                generator=generator,
            )
            norm_weight = (
                1 + 0.1 * torch.randn(shape.hidden, device=device, generator=generator)
            ).to(torch.bfloat16)
            partial = symm_mem.empty(
                (tokens, shape.hidden), dtype=torch.bfloat16, device=device
            )
            symm_mem.rendezvous(partial, group.group_name)
            for case in VERIFY_ROUTING_CASES:
                _set_verify_routing(tensors, case)
                args = _verify_args(tensors, shape)
                result = deepseek_v3_moe_nvfp4_tp_verify(
                    *args, residual, norm_weight, partial, group.group_name
                )
                single = deepseek_v3_moe_nvfp4_verify(*args)
                shards = [torch.empty_like(single[0]) for _ in range(WORLD_SIZE)]
                dist.all_gather(shards, single[0])
                total = torch.zeros_like(single[0], dtype=torch.float32)
                for shard in shards:
                    total = total + shard.float()
                total = total + residual.float()
                scale = torch.rsqrt(total.square().mean(-1, keepdim=True) + NORM_EPS)
                normed = (total * scale * norm_weight.float()).to(torch.bfloat16)
                # The kernel sums the routed W13 in two K halves, the single-GPU
                # kernel in one chain.
                for name, got, expected, tolerance in (
                    ("topk_ids", result[3], single[3], 0.0),
                    ("residual_output", result[-1], total.to(torch.bfloat16), 2e-2),
                    ("output", result[0], normed, 2e-2),
                ):
                    torch.testing.assert_close(
                        got,
                        expected,
                        rtol=tolerance,
                        atol=tolerance,
                        msg=lambda m, t=tokens, c=case, n=name: (
                            f"rank {rank} T={t} {c} {n}: {m}"
                        ),
                    )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    correctness_check()
