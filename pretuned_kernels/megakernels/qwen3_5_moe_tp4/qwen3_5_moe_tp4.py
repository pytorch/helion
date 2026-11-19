# ruff: noqa: ANN001, ANN201, ANN202
# pyrefly: ignore-errors
"""Qwen3.5-397B-A17B NVFP4 MoE for a TP4 speculative-verify window (T <= 8
tokens), with the tensor-parallel all-reduce, the residual add and the next
layer's input RMSNorm fused into the same persistent Gluon megakernel.

Each rank holds every one of the 512 routed experts at a quarter of the
intermediate (256), in FlashInfer's trtllm-gen layout, read in place:

- W13 [E, 512, 2048] packed E2M1: physical row ``p`` of a 32-row block is
  up (rows [0, 8) and [16, 24)) or gate (the others) of intermediate
  ``j = block / 2 + 2 (p % 8) + (p % 32) // 16``.
- W2 [E, 4096, 128]: physical row ``p`` of a 32-row block is hidden
  ``block + (p % 8) * 4 + (p % 32) // 8``.
- Scales: 128x4-interleaved E4M3 (``swizzle_scales``) per 128-row tile.

The shared expert is static per-tensor FP8 (gate rows [0, 256), up rows
[256, 512) of ``shared_w13``) with a sigmoid gate; the router is BF16 with a
softmax over the top-10 logits (trtllm-gen's Renormalize routing).

The roots:

- router: FP32 partials of 64 experts x one of 8 K chunks (512) per task; routing
  (one token per task) sums them, rounds to BF16 and takes the top 10 by packed
  (value, index) keys: the top 10 of each group of 128 experts (one int32 max
  reduction per round, a warp redux), then the top 10 of those 40. It also packs
  the token's ids 3 to an int32 word for the routed tasks' duplicate test.
- input quant: the input's NVFP4 (routed experts) and static FP8 (shared
  expert) copies.
- routed W13, one task per (top-k pair, row half): two 128-row tiles of the
  pair's expert over the full K, SwiGLU and the NVFP4 activation of every token
  of the window that picked the expert. A pair whose expert an earlier pair of
  the window has writes zeros: the window's dedup runs inside the task (one
  load of the packed ids, no dispatch root, no W13 K split).
- routed W2, one task per (top-k pair, hidden half) of a pair that owns its
  expert: 256-row tiles, each physical row stored at its hidden column.
- shared expert: K-split gate / up, SiLU, the FP8 down projection, and its
  sigmoid gate in a root of its own (off the routing task).
- combine: each token's top-k rows weighted in pick order plus the gated shared
  output, this rank's partial in symmetric memory.
- all-reduce: the ranks' partials summed in rank order (FP32) plus the residual,
  the BF16 sum's row partial sums of squares, then the RMSNorm.

``hl.prefetch_code()`` at entry brings the code toward L2 (the previous launch's
weight stream evicted it). The small loads and stores (the routing's, the
routed tasks' packed ids and activation stores, the shared outputs and the
tail's) are pointer accesses (``_POINTER_OPS``): issued together, without a
round trip through the TMA ring each.

Besides the next layer's input and residual, it returns the router logits, the
top-k weights and ids, the routed rows, the shared output and the NVFP4 input
and activations for checks; activation rows are top-k pairs (``t * top_k + k``,
the expert's first pair of the window holds them).
"""

from __future__ import annotations

from pretuned_kernels.megakernels.qwen3_5_moe_tp4._common import FP4_MAX
from pretuned_kernels.megakernels.qwen3_5_moe_tp4._common import FP8_MAX
from pretuned_kernels.megakernels.qwen3_5_moe_tp4._common import NORM_EPS
from pretuned_kernels.megakernels.qwen3_5_moe_tp4._common import PICKED32
from pretuned_kernels.megakernels.qwen3_5_moe_tp4._common import bf16_key
from pretuned_kernels.megakernels.qwen3_5_moe_tp4._common import fp4_nibble
from pretuned_kernels.megakernels.qwen3_5_moe_tp4._common import key_value
from pretuned_kernels.megakernels.qwen3_5_moe_tp4._common import scale_tile
from pretuned_kernels.megakernels.qwen3_5_moe_tp4._common import scale_tiles_16
import torch

import helion
import helion.language as hl


def imax(a, b):
    """int32 max: a reduction with this one combiner lowers to redux.sync across a warp."""
    return torch.maximum(a, b)


# Config.indexing slots (one per load / store, in the order Helion visits them: a
# loop's ops at the loop's place in its root) lowered as pointer accesses; every
# other memory op is a tensor descriptor (TMA ring / bulk store).
_MEMORY_OPS = 72
_POINTER_OPS = {
    2: "router: partials",
    13: "shared gate / up: partials",
    16: "shared gate: token rows",
    17: "shared gate: padding rows",
    18: "routing: router partials",
    19: "routing: logits",
    20: "routing: top-k ids",
    21: "routing: top-k weights",
    22: "routing: packed ids",
    23: "routing: padding ids",
    24: "routing: padding packed ids",
    29: "shared activation (FP8)",
    31: "W13: packed ids (duplicate test)",
    33: "W13: a duplicate pair's activation",
    34: "W13: a duplicate pair's activation scales",
    40: "W13: activation",
    41: "W13: activation scales",
    46: "shared down: output",
    48: "W2: packed ids (duplicate test)",
    56: "combine: routed rows",
    57: "combine: top-k weights",
    58: "combine: shared output",
    60: "all-reduce: residual",
    65: "all-reduce: residual output",
    66: "all-reduce: row partial sums",
    67: "all-reduce: norm input",
    68: "norm: row partial sums",
    69: "norm: input",
    71: "norm: output",
}


def _indexing():
    return [
        "pointer" if op in _POINTER_OPS else "tensor_descriptor"
        for op in range(_MEMORY_OPS)
    ]


@helion.kernel(
    static_shapes=False,
    backend="gluon",
    triton_do_not_specialize=False,
    # One persistent CTA per SM with a 4-deep TMA ring under dynamic dispatch.
    config=helion.Config(
        block_sizes=[16, 128, 32, 16, 128, 256, 16, 16, 128, 16, 512],
        num_warps=4,
        pid_type="persistent_blocked",
        cross_loop_pipeline="dynamic",
        num_sm_multiplier=1,
        num_stages=4,
        range_warp_specializes=[True] * 16,
        indexing=_indexing(),
    ),
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def qwen3_5_moe_tp4(
    hidden_input,  # [T, 4096] bf16, the post-attention RMSNorm output
    router_weight,  # [512, 4096] bf16
    w13,  # [512, 512, 2048] uint8 (trtllm-gen layout)
    w13_scale,  # [512, 4 * 8, 16, 256] e4m3 (swizzled)
    w2,  # [512, 4096, 128] uint8 (trtllm-gen layout)
    w2_scale,  # [512, 32, 8, 256] e4m3 (swizzled, 128-row tiles)
    alpha1,  # [512] f32
    alpha2,  # [512] f32
    input_global_scale,  # [1] f32
    activation_global_scale,  # [1] f32
    shared_w13,  # [512, 4096] e4m3: gate [0, 256), up [256, 512)
    shared_w2,  # [4096, 256] e4m3
    shared_scales,  # [4] f32: 1/in1, in1*w1, 1/in2, in2*w2
    shared_gate_weight,  # [4096] bf16
    top_k: int,
    residual,  # [T, 4096] bf16
    norm_weight,  # [4096] bf16, the next layer's (1 + w)
    symmetric_partial,  # [T, 4096] bf16 symmetric memory
    group_name: hl.ProcessGroupName,
):
    hl.specialize(hidden_input.size())
    hl.specialize(hidden_input.stride())
    hl.specialize(router_weight.size())
    hl.specialize(router_weight.stride())
    hl.specialize(w13.size())
    hl.specialize(w13.stride())
    hl.specialize(w13_scale.size())
    hl.specialize(w13_scale.stride())
    hl.specialize(w2.size())
    hl.specialize(w2.stride())
    hl.specialize(w2_scale.size())
    hl.specialize(w2_scale.stride())
    hl.specialize(alpha1.size())
    hl.specialize(alpha1.stride())
    hl.specialize(alpha2.size())
    hl.specialize(alpha2.stride())
    hl.specialize(input_global_scale.size())
    hl.specialize(input_global_scale.stride())
    hl.specialize(activation_global_scale.size())
    hl.specialize(activation_global_scale.stride())
    hl.specialize(shared_w13.size())
    hl.specialize(shared_w13.stride())
    hl.specialize(shared_w2.size())
    hl.specialize(shared_w2.stride())
    hl.specialize(shared_scales.size())
    hl.specialize(shared_scales.stride())
    hl.specialize(shared_gate_weight.size())
    hl.specialize(shared_gate_weight.stride())
    hl.specialize(residual.size())
    hl.specialize(residual.stride())
    hl.specialize(norm_weight.size())
    hl.specialize(norm_weight.stride())
    hl.specialize(symmetric_partial.size())
    hl.specialize(symmetric_partial.stride())
    top_k = hl.specialize(top_k)

    rows, hidden = hidden_input.shape
    # A literal: a module-level int reaches the kernel as a runtime scalar.
    pick_lanes_n = 16
    experts = router_weight.size(0)
    twice_intermediate, packed_hidden = w13.shape[1:]
    intermediate = twice_intermediate // 2
    packed_intermediate = intermediate // 2
    hidden_groups = hidden // 16
    activation_groups = intermediate // 16
    shared_twice = shared_w13.size(0)
    shared_intermediate = shared_twice // 2
    assert experts == 512 and top_k <= 16
    assert intermediate == 256 and packed_intermediate == 128
    # Token rows padded to a power of two: the routing table's rows.
    assert rows <= 8
    padded_rows = 8
    slots = rows * top_k
    # K steps of 512 FP4 values: 256 packed bytes and 32 scales per row.
    w13_steps = packed_hidden // 256
    w13_tiles = twice_intermediate // 128
    w2_tiles = hidden // 256
    w2_rows = w2.view(experts, w2_tiles, 256, packed_intermediate)
    w2_scale_rows = w2_scale.view(experts, w2_tiles, 16, 256)

    router_chunks = 8
    router_chunk_k = hidden // router_chunks
    router_weight_chunked = router_weight.view(experts, router_chunks, router_chunk_k)
    hidden_chunked = hidden_input.view(rows, router_chunks, router_chunk_k)
    router_partials = torch.empty(
        (rows, router_chunks, experts), dtype=torch.float32, device=hidden_input.device
    )
    router_logits = torch.empty(
        (rows, experts), dtype=torch.bfloat16, device=hidden_input.device
    )
    topk_groups = 4
    topk_group_n = experts // topk_groups
    topk_segment_n = topk_group_n // topk_groups
    router_partials_wide = router_partials.view(
        rows, router_chunks, topk_groups, topk_group_n
    )
    router_logits_wide = router_logits.view(rows, topk_groups, topk_group_n)
    input_q = torch.empty(
        (rows, packed_hidden), dtype=torch.uint8, device=hidden_input.device
    )
    input_q_groups = input_q.view(rows, hidden_groups, 8)
    input_q_steps = input_q.view(rows, w13_steps, 256)
    input_scale = torch.empty(
        (rows, hidden_groups), dtype=torch.float8_e4m3fn, device=hidden_input.device
    )
    input_scale_steps = input_scale.view(rows, w13_steps, 32)
    input_fp8 = torch.empty(
        (rows, hidden), dtype=torch.float8_e4m3fn, device=hidden_input.device
    )
    topk_weights = torch.empty(
        (padded_rows, pick_lanes_n), dtype=torch.float32, device=hidden_input.device
    )
    topk_ids = torch.empty(
        (padded_rows, pick_lanes_n), dtype=torch.int32, device=hidden_input.device
    )
    # Each token's top-k ids, 3 to an int32 word (10-bit fields): word t * 4 + k
    # holds picks 3k .. 3k + 2.
    word_picks = 3
    token_words = 4
    assert top_k <= word_picks * token_words and experts <= 1024  # 10-bit fields
    route_words = torch.empty(
        (padded_rows * token_words,), dtype=torch.int32, device=hidden_input.device
    )
    route_words_rows = route_words.view(padded_rows, token_words)
    shared_gate = torch.empty(
        (padded_rows,), dtype=torch.float32, device=hidden_input.device
    )

    w13_view = w13.view(experts, twice_intermediate, w13_steps, 256)
    activation_q = torch.empty(
        (slots, rows, packed_intermediate), dtype=torch.uint8, device=w13.device
    )
    activation_q_groups = activation_q.view(slots, rows, w13_tiles, 4, 8)
    activation_scale = torch.empty(
        (slots, rows, activation_groups), dtype=torch.float8_e4m3fn, device=w13.device
    )
    activation_scale_groups = activation_scale.view(slots, rows, w13_tiles, 4)
    routed_output = torch.empty(
        (rows, pick_lanes_n, hidden), dtype=torch.bfloat16, device=w2.device
    )
    routed_rows = routed_output.view(rows * pick_lanes_n, hidden)

    # Shared expert: K-split gate/up partials, then SiLU and the FP8 down input.
    shared_chunks = 8
    shared_chunk_k = hidden // shared_chunks
    shared_partials = torch.empty(
        (shared_chunks, rows, shared_twice), dtype=torch.float32, device=w13.device
    )
    shared_w13_chunked = shared_w13.view(shared_twice, shared_chunks, shared_chunk_k)
    input_fp8_chunked = input_fp8.view(rows, shared_chunks, shared_chunk_k)
    shared_act = torch.empty(
        (rows, shared_intermediate), dtype=torch.float8_e4m3fn, device=w13.device
    )
    shared_output = torch.empty((rows, hidden), dtype=torch.bfloat16, device=w2.device)

    assert symmetric_partial.size() == (rows, hidden)
    remote_partials = torch.ops.symm_mem.get_remote_tensors(
        symmetric_partial, group_name
    )
    output = torch.empty((rows, hidden), dtype=torch.bfloat16, device=w2.device)
    residual_output = torch.empty_like(output)
    communication_n = 512
    norm_tasks = hidden // communication_n
    norm_partial = torch.empty(
        (rows, norm_tasks), dtype=torch.float32, device=w2.device
    )
    norm_input = torch.empty((rows, hidden), dtype=torch.float32, device=w2.device)

    # The next launch may start early: weights the previous launch does not
    # write stream before the grid wait.
    hl.pdl_wait(
        independent=(
            router_weight,
            w13,
            w13_scale,
            w2,
            w2_scale,
            alpha1,
            alpha2,
            input_global_scale,
            activation_global_scale,
            shared_w13,
            shared_w2,
            shared_scales,
            shared_gate_weight,
            norm_weight,
        )
    )
    # The code toward L2 first: the previous launch's weight stream evicted it.
    hl.prefetch_code()

    # Router projection: K chunks of the 512 x 4096 BF16 weight.
    for router_tile_e, router_tile_c, router_tile_t in hl.tile(
        [experts, router_chunks, rows], block_size=[64, 1, None]
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

    # The input's NVFP4 (routed experts) and static FP8 (shared expert) copies.
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
        inv_in1 = shared_scales[0]
        input_fp8[quant_tile_row, group_offsets + packed_lane * 2] = torch.clamp(
            low_values * inv_in1, -FP8_MAX, FP8_MAX
        ).to(torch.float8_e4m3fn)
        input_fp8[quant_tile_row, group_offsets + packed_lane * 2 + 1] = torch.clamp(
            high_values * inv_in1, -FP8_MAX, FP8_MAX
        ).to(torch.float8_e4m3fn)

    # Shared gate/up: FP8 K chunks.
    for sp_tile_n, sp_tile_c, sp_tile_t in hl.tile(
        [shared_twice, shared_chunks, rows], block_size=[128, 1, None]
    ):
        sp_acc = hl.zeros([sp_tile_n, sp_tile_t], dtype=torch.float32)
        for sp_tile_k in hl.tile(shared_chunk_k):
            sp_acc = hl.dot(
                shared_w13_chunked[sp_tile_n, sp_tile_c.begin, sp_tile_k],
                input_fp8_chunked[sp_tile_t, sp_tile_c.begin, sp_tile_k].T,
                acc=sp_acc,
            )
        shared_partials[sp_tile_c.begin, sp_tile_t, sp_tile_n] = sp_acc.T

    # The shared expert's sigmoid gate, one token per task (off the routing task).
    for sg_tile_t in hl.tile(padded_rows, block_size=1):
        if sg_tile_t.begin < rows:
            sg_logit = torch.sum(
                hidden_input[sg_tile_t.begin, :].to(torch.float32)
                * shared_gate_weight[:].to(torch.float32)
            )
            shared_gate[sg_tile_t] = torch.sigmoid(sg_logit)[None]
        else:
            shared_gate[sg_tile_t] = hl.zeros([1], dtype=torch.float32)

    # Routing, one token per task (padded rows write sentinel picks and zero
    # packed words, which no earlier pair reads).
    for topk_tile_t in hl.tile(padded_rows, block_size=1):
        topk_token = topk_tile_t.begin
        picks = hl.arange(pick_lanes_n)
        if topk_token < rows:
            logits = torch.sum(router_partials_wide[topk_token, :, :, :], dim=0).to(
                torch.bfloat16
            )
            router_logits_wide[topk_tile_t, :, :] = logits[None, :, :]
            tk_groups = hl.arange(topk_groups)
            tk_slots = hl.arange(topk_group_n)
            expert_indices = (tk_groups[:, None] * topk_group_n + tk_slots[None, :]).to(
                torch.int32
            )
            keys = bf16_key(logits, expert_indices)
            tk_lists = torch.full_like(keys, PICKED32)
            for rnd in hl.static_range(top_k):
                tk_group_max = hl.reduce(imax, keys, dim=1)
                keys = torch.where(keys == tk_group_max[:, None], PICKED32, keys)
                tk_lists = torch.where(
                    tk_slots[None, :] == tk_groups[:, None] * topk_segment_n + rnd,
                    tk_group_max[:, None],
                    tk_lists,
                )
            tk_cands = hl.reduce(imax, tk_lists, dim=0)
            tops = hl.full([pick_lanes_n], PICKED32, dtype=torch.int32)
            for rnd in hl.static_range(top_k):
                tk_top = hl.reduce(imax, tk_cands, dim=0)
                tk_cands = torch.where(tk_cands == tk_top, PICKED32, tk_cands)
                tops = torch.where(picks == rnd, tk_top, tops)
            ids = torch.where(picks < top_k, 65535 - (tops & 65535), experts)
            top_values = key_value(tops)
            top_max = torch.amax(torch.where(picks < top_k, top_values, float("-inf")))
            exps = torch.where(picks < top_k, torch.exp(top_values - top_max), 0.0)
            total = torch.sum(exps)
            topk_ids[topk_tile_t, :] = ids[None, :]
            topk_weights[topk_tile_t, :] = (exps / total)[None, :]
            rw_words = hl.arange(token_words)
            rw_fields = torch.where(
                ((picks[None, :] // word_picks) == rw_words[:, None])
                & (picks[None, :] < word_picks * token_words),
                (ids & 1023)[None, :] << (10 * (picks[None, :] % word_picks)),
                0,
            )
            route_words_rows[topk_tile_t, :] = torch.sum(rw_fields, dim=1)[None, :]
        else:
            topk_ids[topk_tile_t, :] = hl.full(
                [1, pick_lanes_n], experts, dtype=torch.int32
            )
            route_words_rows[topk_tile_t, :] = hl.zeros(
                [1, token_words], dtype=torch.int32
            )

    # Shared SiLU(gate) * up and its static FP8 copy.
    for sa_tile_t, sa_tile_j in hl.tile(
        [rows, shared_intermediate], block_size=[1, None]
    ):
        sa_gate = torch.sum(shared_partials[:, sa_tile_t, sa_tile_j], dim=0)
        sa_up = torch.sum(
            shared_partials[:, sa_tile_t, sa_tile_j.index + shared_intermediate], dim=0
        )
        alpha_s1 = shared_scales[1]
        g = sa_gate * alpha_s1
        u = sa_up * alpha_s1
        act = (g * torch.sigmoid(g) * u).to(torch.bfloat16).to(torch.float32)
        shared_act[sa_tile_t, sa_tile_j] = torch.clamp(
            act * shared_scales[2], -FP8_MAX, FP8_MAX
        ).to(torch.float8_e4m3fn)

    # Routed W13 per (top-k pair, row half): the pair's expert, unless an earlier
    # pair of the window has it (then zeros), as two 128-row tiles over full K,
    # then SwiGLU and the NVFP4 activation of the expert's tokens.
    for p_pair, p_half, p_t in hl.tile([slots, 2, rows], block_size=[1, 1, None]):
        p_token = p_pair.begin // top_k
        p_pick = p_pair.begin % top_k
        p_expert = topk_ids[p_token, p_pick]
        p_words = route_words[:]
        p_w = hl.arange(padded_rows * token_words)
        p_hit = hl.zeros([padded_rows * token_words], dtype=torch.int32)
        for p_j in hl.static_range(word_picks):
            p_pk = (p_w % token_words) * word_picks + p_j
            p_before = (p_pk < top_k) & (
                (p_w // token_words) * pick_lanes_n + p_pk
                < p_token * pick_lanes_n + p_pick
            )
            p_same = (((p_words >> (10 * p_j)) & 1023) == p_expert) & p_before
            p_hit = p_hit | p_same.to(torch.int32)
        p_dup = hl.reduce(imax, p_hit, dim=0) > 0
        p_global_scale = torch.sum(activation_global_scale[:])
        for p_r in hl.tile(2, block_size=1):
            p_tile = p_half.begin * 2 + p_r.begin
            if p_dup:
                activation_q_groups[p_pair.begin, p_t, p_tile, :, :] = hl.zeros(
                    [p_t, 4, 8], dtype=torch.uint8
                )
                activation_scale_groups[p_pair.begin, p_t, p_tile, :] = hl.zeros(
                    [p_t, 4], dtype=torch.float8_e4m3fn
                )
            else:
                p_alpha = alpha1[p_expert]
                p_acc = hl.zeros([128, p_t], dtype=torch.float32)
                for p_s in hl.tile(w13_steps, block_size=1):
                    p_acc = hl.dot_scaled(
                        w13_view[p_expert, p_tile * 128 + hl.arange(128), p_s.begin, :],
                        scale_tile(
                            w13_scale[p_expert, p_tile * w13_steps + p_s.begin, :, :]
                        ),
                        "e2m1",
                        input_q_steps[p_t, p_s.begin, :].T,
                        input_scale_steps[p_t, p_s.begin, :],
                        "e2m1",
                        acc=p_acc,
                        out_dtype=torch.float32,
                    )
                p_tokens = p_acc.size(1)
                # [T, 4 blocks, half, up/gate, m] -> [T, 4, m, half, up/gate]
                p_pre = (
                    (p_acc.T * p_alpha)
                    .reshape(p_tokens, 4, 2, 2, 8)
                    .permute(0, 1, 4, 2, 3)
                )
                p_up, p_gate = hl.split(p_pre)
                p_act = p_gate * torch.sigmoid(p_gate) * p_up
                p_block_scale = (
                    torch.amax(torch.abs(p_act.reshape(p_tokens, 4, 16)), dim=-1)
                    * p_global_scale
                    / FP4_MAX
                ).to(torch.float8_e4m3fn)
                p_actual = p_block_scale.to(torch.float32)
                p_divisor = torch.where(p_actual > 0, p_actual, 1.0)
                p_low, p_high = hl.split(
                    fp4_nibble(p_act * p_global_scale / p_divisor[:, :, None, None])
                )
                activation_q_groups[p_pair.begin, p_t, p_tile, :, :] = (
                    p_low | p_high << 4
                ).to(torch.uint8)
                activation_scale_groups[p_pair.begin, p_t, p_tile, :] = p_block_scale

    # Shared down projection (FP8).
    for sd_tile_n, sd_tile_t in hl.tile([hidden, rows], block_size=[128, None]):
        sd_acc = hl.zeros([sd_tile_n, sd_tile_t], dtype=torch.float32)
        for sd_tile_k in hl.tile(shared_intermediate):
            sd_acc = hl.dot(
                shared_w2[sd_tile_n, sd_tile_k],
                shared_act[sd_tile_t, sd_tile_k].T,
                acc=sd_acc,
            )
        shared_output[sd_tile_t, sd_tile_n] = (
            (sd_acc.T * shared_scales[3]).to(torch.bfloat16).to(torch.float32)
            * shared_gate[sd_tile_t][:, None]
        ).to(torch.bfloat16)

    # Routed W2 per (top-k pair, hidden half) of a pair that owns its expert:
    # 8 tiles of 256 physical hidden rows, K = 256 (one MMA step each), for
    # every token of the window that picked the expert.
    for q_pair, q_half, q_t in hl.tile([slots, 2, rows], block_size=[1, 1, None]):
        q_token = q_pair.begin // top_k
        q_pick = q_pair.begin % top_k
        q_expert = topk_ids[q_token, q_pick]
        q_words = route_words[:]
        q_w = hl.arange(padded_rows * token_words)
        q_hit = hl.zeros([padded_rows * token_words], dtype=torch.int32)
        for q_j in hl.static_range(word_picks):
            q_pk = (q_w % token_words) * word_picks + q_j
            q_before = (q_pk < top_k) & (
                (q_w // token_words) * pick_lanes_n + q_pk
                < q_token * pick_lanes_n + q_pick
            )
            q_same = (((q_words >> (10 * q_j)) & 1023) == q_expert) & q_before
            q_hit = q_hit | q_same.to(torch.int32)
        q_dup = hl.reduce(imax, q_hit, dim=0) > 0
        if not q_dup:
            q_alpha = alpha2[q_expert]
            q_hits = topk_ids[q_t, :] == q_expert
            q_where = torch.sum(
                torch.where(q_hits, hl.arange(pick_lanes_n)[None, :], 0), dim=1
            )
            q_used = (torch.sum(q_hits.to(torch.int32), dim=1) > 0) & (q_t.index < rows)
            for q_n in hl.tile(8, block_size=1):
                q_row_tile = q_half.begin * 8 + q_n.begin
                q_acc = hl.zeros([256, q_t], dtype=torch.float32)
                for _q_k in hl.tile(1, block_size=1):
                    q_acc = hl.dot_scaled(
                        w2_rows[q_expert, q_row_tile, :, :],
                        scale_tiles_16(w2_scale_rows[q_expert, q_row_tile, :, :]),
                        "e2m1",
                        activation_q[q_pair.begin, q_t, :].T,
                        activation_scale[q_pair.begin, q_t, :],
                        "e2m1",
                        acc=q_acc,
                        out_dtype=torch.float32,
                    )
                # Physical row b*32 + 8 c + m holds hidden b*32 + 4 m + c: store each row
                # at its hidden column (no register permute; a warp's 32 rows still
                # cover one 32-column block).
                q_values = (q_acc.T * q_alpha).to(torch.bfloat16)
                q_phys = hl.arange(256)
                q_hidden = (q_phys // 32) * 32 + (q_phys % 8) * 4 + (q_phys % 32) // 8
                hl.store(
                    routed_rows,
                    [
                        (q_t.index * pick_lanes_n + q_where)[:, None],
                        (q_row_tile * 256 + q_hidden)[None, :],
                    ],
                    q_values,
                    extra_mask=q_used[:, None],
                )

    # Each token's top-k rows, weighted in pick order, plus the gated shared.
    for partial_tile_t, partial_tile_n in hl.tile([rows, hidden], block_size=[1, None]):
        partial_token = partial_tile_t.begin
        pick_lanes = hl.arange(pick_lanes_n)
        partial_values = routed_output[partial_token, :, partial_tile_n].to(
            torch.float32
        )
        partial_weights = (
            topk_weights[partial_token, :].to(torch.bfloat16).to(torch.float32)
        )
        partial_weights = torch.where(pick_lanes < top_k, partial_weights, 0.0)
        partial_values = torch.where((pick_lanes < top_k)[:, None], partial_values, 0.0)
        partial_routed = torch.sum(partial_values * partial_weights[:, None], dim=0)
        symmetric_partial[partial_token, partial_tile_n] = (
            partial_routed
            + shared_output[partial_token, partial_tile_n].to(torch.float32)
        ).to(torch.bfloat16)

    hl.pdl_launch_dependents()
    # Each token's sum over ranks (in rank order), then the residual.
    for comm_tile_t, comm_tile_n in hl.tile(
        [rows, hidden], block_size=[1, communication_n]
    ):
        comm_token = comm_tile_t.begin
        comm_residual = residual[comm_token, comm_tile_n].to(torch.float32)
        comm_total = hl.zeros([comm_tile_n], dtype=torch.float32)
        for remote_partial in remote_partials:
            comm_total = comm_total + remote_partial[comm_token, comm_tile_n].to(
                torch.float32
            )
        comm_sum = comm_total + comm_residual
        residual_output[comm_token, comm_tile_n] = comm_sum.to(torch.bfloat16)
        comm_sum = comm_sum.to(torch.bfloat16).to(torch.float32)
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
        residual_output,
        router_logits,
        topk_weights,
        topk_ids,
        routed_output,
        shared_output,
        input_q,
        input_scale,
        activation_q,
        activation_scale,
    )
