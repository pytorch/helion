# ruff: noqa: ANN001, ANN201
# pyrefly: ignore-errors
"""DeepSeek-V3 NVFP4 MoE megakernel for the T tokens of a speculative-decoding
verify step.

``deepseek_v3_moe_nvfp4``'s boundary and inputs (router projection, grouped
routing, NVFP4 input quantization, routed and shared experts, final
accumulation), with every expert GEMM a swap-AB tcgen05 MMA over TMA-shaped
weight views: K steps of 512 FP4 values whose E4M3 scales are stored swizzled for
tcgen05 (``swizzle_scales``), the token dimension padded to the MMA's N by its
block size.  Each token is routed on its own: a routed task serves one distinct
expert of the window (``slots`` = T * top_k bounds their count) for every token
at once, and the final reduction gathers each token's eight expert rows.
``correctness_check`` compares each token's results with ``deepseek_v3_moe_nvfp4``
run on that token alone.
"""

from __future__ import annotations

from pretuned_kernels.megakernels.deepseek_v3_moe_nvfp4._common import FP4_MAX
from pretuned_kernels.megakernels.deepseek_v3_moe_nvfp4._common import distinct_table
from pretuned_kernels.megakernels.deepseek_v3_moe_nvfp4._common import expert_of
from pretuned_kernels.megakernels.deepseek_v3_moe_nvfp4._common import fp4_nibble
from pretuned_kernels.megakernels.deepseek_v3_moe_nvfp4._common import scale_tile
from pretuned_kernels.megakernels.deepseek_v3_moe_nvfp4._common import swizzle_scales
from pretuned_kernels.megakernels.deepseek_v3_moe_nvfp4.deepseek_v3_moe_nvfp4 import (
    OUTPUT_NAMES,
)
from pretuned_kernels.megakernels.deepseek_v3_moe_nvfp4.deepseek_v3_moe_nvfp4 import (
    ROUTING_LOGITS,
)
from pretuned_kernels.megakernels.deepseek_v3_moe_nvfp4.deepseek_v3_moe_nvfp4 import (
    Shape,
)
from pretuned_kernels.megakernels.deepseek_v3_moe_nvfp4.deepseek_v3_moe_nvfp4 import (
    _kernel_args,
)
from pretuned_kernels.megakernels.deepseek_v3_moe_nvfp4.deepseek_v3_moe_nvfp4 import (
    deepseek_v3_moe_nvfp4,
)
from pretuned_kernels.megakernels.deepseek_v3_moe_nvfp4_tp._common import routing_keys
from pretuned_kernels.megakernels.deepseek_v3_moe_nvfp4_tp._common import top_k_picks
import torch

import helion
import helion.language as hl

# The GEMM operands' views (router, shared and routed W13/W2 weights, scales and
# activations), as load/store slots in codegen order; every other access is a
# pointer.
_TMA_SLOTS = (
    *(0, 1, 8, 9, 10, 11, 25, 26, 27, 28),
    *(31, 32, 33, 34, 40, 41, 42, 43),
)


@helion.kernel(
    static_shapes=True,
    backend="gluon",
    triton_do_not_specialize=False,
    # One persistent CTA per SM with a 4-deep TMA ring under dynamic dispatch:
    # the routed W13/W2 roots (live counts set by the routing) claim their
    # tasks past a fixed first wave; the final reduction takes 512 columns.
    config=helion.Config(
        block_sizes=[16, 128, 64, 16, 16, 16, 16, 512],
        num_warps=4,
        pid_type="persistent_blocked",
        cross_loop_pipeline="dynamic",
        num_sm_multiplier=1,
        num_stages=4,
        range_warp_specializes=[
            *(None, True, None, None, True, None, None),
            *(True, None, True, None, True, None),
        ],
        indexing=[
            "tensor_descriptor" if index in _TMA_SLOTS else "pointer"
            for index in range(64)
        ],
    ),
)
def deepseek_v3_moe_nvfp4_verify(
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
):

    top_k = hl.specialize(top_k)
    num_groups = hl.specialize(num_groups)
    topk_groups = hl.specialize(topk_groups)

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
    # distinct_table reads every route at once, padded to a power of two: a
    # padding lane would read expert 0.
    assert slots & (slots - 1) == 0, "the token count must be a power of two"
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
    output = torch.empty((rows, hidden), dtype=torch.bfloat16, device=w2.device)

    # Router projection: K chunks of the 256 x 7168 BF16 weight, summed by one
    # (deterministic) reduction in the routing root.
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

    # Routed W13 with its SwiGLU and NVFP4 quantization, per (128 rows, slot):
    # slot j serves the window's j-th distinct expert for every token.  Slots
    # are the slowest axis: the distinct count bounds the live tasks.
    for w13_tile_n, w13_tile_t, w13_tile_j in hl.tile(
        [twice_intermediate, rows, slots], block_size=[128, None, 1]
    ):
        w13_words, _, w13_before, w13_distinct = distinct_table(routes[:])
        if w13_tile_j.begin < w13_distinct:
            w13_expert = expert_of(w13_words, w13_before, w13_tile_j.begin)
            # The epilogue's operands load under the K loop.
            w13_alpha = alpha1[w13_expert]
            w13_global_scale = torch.sum(activation_global_scale[:])
            w13_acc = hl.zeros([w13_tile_n, w13_tile_t], dtype=torch.float32)
            for w13_tile_s in hl.tile(w13_steps, block_size=1):
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
            w13_preactivation = w13_acc.T * w13_alpha
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
        w2_words, _, w2_before, w2_distinct = distinct_table(routes[:])
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
    for final_tile_t, final_tile_n in hl.tile([rows, hidden], block_size=[1, None]):
        final_token = final_tile_t.begin
        # The shared expert's output first: it is ready long before the routed
        # rows (a task waits for what it reads in the order it reads it).
        final_shared = shared_expert_output[final_token, final_tile_n]
        final_routed_values = routed_output[final_token, :, final_tile_n].to(
            torch.float32
        )
        final_routed_weights = (
            topk_weights[final_token, :].to(torch.bfloat16).to(torch.float32)
        )
        final_routed_output = torch.sum(
            final_routed_values * final_routed_weights[:, None], dim=0
        ).to(torch.bfloat16)
        output[final_token, final_tile_n] = (final_routed_output + final_shared).to(
            torch.bfloat16
        )
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
    )


# Per-token routing draws for T tokens: same8 routes every token to one set of
# eight experts; pool picks from 16 experts in four groups; spread gives each
# token its own two experts in each of four groups; decoy is same8 with two
# strong experts in a fifth group (_DECOYS): their group's top-two sum beats the
# weakest chosen group's but for their correction bias, and the first beats the
# weakest chosen expert, so only the top-4-group mask on biased scores keeps
# them out.
VERIFY_ROUTING_CASES = ("same8", "pool", "spread", "decoy")
# The decoys' logits (no other case routes to them; their bias is -0.15).
_DECOYS = {186: 9.0, 189: 1.5}


def _route(case: str, token: int) -> list[int]:
    if case in ("same8", "decoy"):
        return [3, 17, 35, 50, 77, 95, 113, 127]
    if case == "pool":
        pool = [g * 32 + e for g in (1, 3, 4, 6) for e in (2, 9, 20, 27)]
        return [pool[(5 * token + 3 * k) % 16] for k in range(8)]
    groups = [(token + 2 * k) % 8 for k in range(4)]
    return [g * 32 + (5 * token + 17 * e) % 32 for g in groups for e in range(2)]


def _allocate_verify(
    shape: Shape,
    tokens: int,
    seed: int = 17,
    init: bool = True,
    device: torch.device | str = "cuda",
) -> dict[str, torch.Tensor]:
    """The verify kernels' inputs for T tokens, every expert initialized
    (``init=False``: uninitialized; binding and code generation need only
    their shapes)."""
    torch.manual_seed(seed)
    e, inter, h = shape.num_experts, shape.intermediate, shape.hidden
    e4m3 = torch.float8_e4m3fn

    def pattern(rows: int, groups: int, base: int, a: int, b: int) -> torch.Tensor:
        r = torch.arange(rows, device=device)[:, None]
        g = torch.arange(groups, device=device)[None, :]
        return torch.exp2((base + (r + a * g) % 3).float()) * (
            1.0 + ((b * r + 7 * g) % 8).float() / 8.0
        )

    def rolled(p: torch.Tensor, count: int) -> torch.Tensor:
        # Convert before repeating: an FP32 [experts, rows, groups] temporary
        # would be four times the scales' size.
        variants = torch.stack([torch.roll(p, shifts=s, dims=1) for s in range(3)])
        variants = variants.to(e4m3).view(torch.uint8)
        return variants[torch.arange(count, device=device) % 3].view(e4m3)

    def scales(
        count: int, rows: int, groups: int, base: int, a: int, b: int
    ) -> torch.Tensor:
        if not init:
            return torch.empty(count, rows, groups, device=device, dtype=e4m3)
        return rolled(pattern(rows, groups, base, a, b), count)

    def codes(*size: int) -> torch.Tensor:
        weight = torch.empty(size, device=device, dtype=torch.uint8)
        return weight.random_(0, 256) if init else weight

    def interleave(weight: torch.Tensor) -> torch.Tensor:
        # In place, one expert at a time: a copy of the routed W13 would double
        # its 3.5 GiB.
        n, rows2, k = weight.shape
        for expert in range(n if init else 0):
            weight[expert] = (
                weight[expert].view(2, rows2 // 2, k).transpose(0, 1).reshape(rows2, k)
            )
        return weight

    w13 = codes(e, 2 * inter, h // 2)
    w13_scale = scales(e, 2 * inter, h // 16, -9, 3, 5)
    w2 = codes(e, h, inter // 2)
    w2_scale = scales(e, h, inter // 16, -7, 5, 3)
    shared_w13 = codes(1, 2 * inter, h // 2)
    shared_w13_scale = scales(2, 2 * inter, h // 16, -9, 3, 5)[1:]
    shared_w2 = codes(h, inter // 2)
    shared_w2_scale = scales(2, h, inter // 16, -7, 5, 3)[1]
    activation_global_scale = torch.full((1,), 16.0, device=device)
    correction_bias = (((torch.arange(e, device=device) * 7) % 17) - 8).float() * 0.001
    correction_bias[list(_DECOYS)] = -0.15
    randn = torch.randn if init else torch.empty
    return {
        "hidden": randn(tokens, h, device=device, dtype=torch.bfloat16),
        "router_weight": torch.empty(e, h, device=device, dtype=torch.bfloat16),
        "correction_bias": correction_bias,
        "native_w13": interleave(w13),
        "native_w13_scale": interleave(w13_scale.view(torch.uint8)).view(e4m3),
        "w2": w2,
        "w2_scale": w2_scale,
        "native_shared_w13": interleave(shared_w13)[0],
        "native_shared_w13_scale": interleave(shared_w13_scale.view(torch.uint8)).view(
            e4m3
        )[0],
        "shared_w2": shared_w2,
        "shared_w2_scale": shared_w2_scale,
        # Per expert, so a kernel reading another expert's alphas fails.
        "alpha1": 0.5 + torch.rand(e, device=device),
        "alpha2": (0.5 + torch.rand(e, device=device)) / 16.0,
        "shared_alpha1": torch.ones(1, device=device),
        "shared_alpha2": torch.full((1,), 1.0 / 16.0, device=device),
        "input_global_scale": torch.ones(1, device=device),
        "activation_global_scale": activation_global_scale,
    }


def _set_verify_routing(tensors: dict[str, torch.Tensor], case: str) -> None:
    """Router weights whose logits route each token to ``_route(case, token)``."""
    hidden = tensors["hidden"].double()
    tokens = hidden.size(0)
    target = torch.full(
        (tokens, tensors["router_weight"].size(0)),
        -12.0,
        dtype=torch.float64,
        device=hidden.device,
    )
    for t in range(tokens):
        for expert, value in zip(_route(case, t), ROUTING_LOGITS, strict=True):
            target[t, expert] = value
    for expert, value in _DECOYS.items() if case == "decoy" else ():
        target[:, expert] = value
    weight = target.T @ torch.linalg.solve(hidden @ hidden.T, hidden)
    tensors["router_weight"].copy_(weight.to(torch.bfloat16))


def _verify_args(tensors: dict[str, torch.Tensor], shape: Shape) -> tuple:
    """The kernel arguments: ``_kernel_args``'s, weight scales swizzled."""
    args = list(_kernel_args(tensors, shape))
    for index in (4, 6, 8, 10):
        args[index] = swizzle_scales(args[index])
    return tuple(args)


# The outputs both kernels keep per token, as (index, tolerance): all but the
# routed activations, which are per distinct expert here and per pick in
# ``deepseek_v3_moe_nvfp4``, and the shared expert's quantized activation (FP4
# codes and scales of a preactivation that may differ by rounding, so a code
# may flip; the shared expert's output is compared).  The input quantization is
# elementwise, so it must match bit for bit; the GEMMs are computed
# differently, so the rest may differ by rounding.
_PER_TOKEN_OUTPUTS = (
    (0, 2e-2),
    (1, 2e-2),
    (2, 1e-3),
    (4, 0.0),
    (5, 0.0),
    (8, 2e-2),
    (9, 2e-2),
    (12, 2e-2),
)


def correctness_check(tokens: int = 8) -> None:
    """Check every token of the window against ``deepseek_v3_moe_nvfp4`` run on
    that token alone, for every routing case."""
    shape = Shape()
    tensors = _allocate_verify(shape, tokens)
    for case in VERIFY_ROUTING_CASES:
        _set_verify_routing(tensors, case)
        window = deepseek_v3_moe_nvfp4_verify(*_verify_args(tensors, shape))
        for token in range(tokens):
            single = {**tensors, "hidden": tensors["hidden"][token : token + 1]}
            alone = deepseek_v3_moe_nvfp4(*_kernel_args(single, shape))
            picked = window[3][token].tolist()
            if sorted(picked) != sorted(_route(case, token)) or picked != (
                alone[3][0].tolist()
            ):
                raise AssertionError(f"{case} token {token} routed to {picked}")
            for index, tolerance in _PER_TOKEN_OUTPUTS:
                got = window[index][token]
                # The batch-one kernel's outputs have a token dimension of one
                # or none (its shared W13 preactivation and expert rows).
                expected = alone[index].view_as(got)
                torch.testing.assert_close(
                    got,
                    expected,
                    rtol=tolerance,
                    atol=tolerance,
                    msg=lambda m, c=case, t=token, i=index: (
                        f"{c} token {t} {OUTPUT_NAMES[i]}: {m}"
                    ),
                )


if __name__ == "__main__":
    for window in (8, 16):
        correctness_check(window)
