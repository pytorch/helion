# ruff: noqa: ANN001, ANN201, ANN202
# pyrefly: ignore-errors
"""GLM-5.3 (GlmMoeDsa) NVFP4 MoE for a TP4 speculative-verify window (T <= 8
tokens), with the tensor-parallel all-reduce, the residual add and the next
layer's input RMSNorm fused into the same persistent Gluon megakernel.

The structure is ``qwen3_5_moe_tp4``'s; what differs is the model:

- 256 routed experts at a quarter of the intermediate (512) per rank, in
  FlashInfer's trtllm-gen layout, read in place (``qwen3_5_moe_tp4``'s
  docstring has the row maps): W13 [E, 1024, 3072], W2 [E, 6144, 256].
- Routing: DeepSeek-V3's noaux_tc with one group (``_route``, in
  ``stable_key``'s full-FP32 order): FP32 logits of the BF16 router (SGLang's
  ``MoEGate``), sigmoid scores, the top 8 of score + correction bias, their
  scores renormalized times the routed scale (2.5).
- The shared expert is BF16 (gate rows [0, 512), up rows [512, 1024) of
  ``shared_w13``) with no gate: K-split gate / up, SiLU in BF16, BF16 down.
- W2's K (512) is one MMA step: 128-row tiles (``scale_tile``), eight per task.

The roots: router partials (64 experts x one of 16 K chunks of 384), input
quant (NVFP4, 128 groups a task), routing (one token a task; the picks also as
a bitmap of the 256 experts, 8 words), shared gate / up, shared activation,
shared down (its weights stream while the routing runs), routed W13 (128-row
tile x distinct-expert slot, full K, SwiGLU, NVFP4 activation), routed W2
(sixth x slot), combine into this rank's symmetric partial, the all-reduce with
the residual and the norm. The routing chain (router -> routing -> W13) heads
the kernel: W13's weights wait for it, so the shared expert fills the HBM
meanwhile.

Split-K partials and the all-reduce are summed in a fixed order and the only
atomics are the compiler's int32 counters: the outputs are run-to-run
deterministic.

``correctness_check()`` compares every rank with a torch reference on the
logical weights (``_setup``) at the served window; ``main()`` runs it, then
times the kernel alone with cold weights; ``--check`` runs only the check::

    NVSHMEM_DISABLE_CUDA_VMM=1 PYTHONPATH=. \\
      python -m torch.distributed.run --standalone --nproc-per-node=4 \\
      pretuned_kernels/megakernels/glm5_3_tp4/glm5_3_moe_tp4.py [--check]
"""

from __future__ import annotations

import operator
import os
import sys
from typing import Any

from pretuned_kernels.megakernels._qwen3_5_tp4 import all_reduce_norm
from pretuned_kernels.megakernels._qwen3_5_tp4 import graph_time
from pretuned_kernels.megakernels._qwen3_5_tp4 import report
from pretuned_kernels.megakernels._qwen3_5_tp4 import run_main
from pretuned_kernels.megakernels._qwen3_5_tp4 import say
from pretuned_kernels.megakernels._qwen3_5_tp4 import symmetric_partials
from pretuned_kernels.megakernels._qwen3_5_tp4 import verdict
from pretuned_kernels.megakernels.deepseek_v3_moe_nvfp4._common import FP4_MAX
from pretuned_kernels.megakernels.deepseek_v3_moe_nvfp4._common import expert_of
from pretuned_kernels.megakernels.deepseek_v3_moe_nvfp4._common import fp4_nibble
from pretuned_kernels.megakernels.deepseek_v3_moe_nvfp4._common import popcount
from pretuned_kernels.megakernels.deepseek_v3_moe_nvfp4._common import scale_tile
from pretuned_kernels.megakernels.glm5_3_tp4 import _setup
from pretuned_kernels.megakernels.glm5_3_tp4._setup import NORM_EPS
from pretuned_kernels.megakernels.qwen3_5_moe_tp4._common import pick_bits
import torch
import torch.distributed as dist

import helion
import helion.language as hl

# Config.indexing slots (one per load / store, in the order Helion visits them)
# lowered as pointer accesses; every other memory op is a tensor descriptor.
# ``qwen3_5_moe_tp4``'s choice root for root, picked from the bound kernel's
# ``config_spec.memory_op_facts``; re-derive the table after editing the roots.
# Numbered for 4 ranks: the all-reduce loads each rank's partial (slots 45-48).
_MEMORY_OPS = 57
_POINTER_OPS = {
    2: "store router_partials",  # router
    # routing: bias, router partials, logits, top-k ids and weights, bitmap
    8: "load correction_bias",
    9: "load router_partials",
    10: "store router_logits",
    11: "store topk_ids",
    12: "store topk_weights",
    13: "store token_experts",
    16: "store shared_partials",  # shared gate / up
    19: "store shared_act",  # shared activation
    22: "store shared_output",  # shared down
    23: "load token_experts",  # W13: expert bitmaps, activation and its scales
    30: "store activation_q_groups",
    31: "store activation_scale_groups",
    32: "load token_experts",  # W2
    40: "load shared_output",  # combine: shared output, routed rows, top-k weights
    41: "load routed_output",
    42: "load topk_weights",
    44: "load residual",  # all-reduce
    49: "store residual_output",
    50: "store norm_partial",
    51: "store norm_input",
    52: "load norm_weight",  # norm
    53: "load norm_partial",
    54: "load norm_input",
    55: "store output",
    56: "load None",  # an int32 [64] load Helion adds to the quant root
}


def _indexing():
    return [
        "pointer" if op in _POINTER_OPS else "tensor_descriptor"
        for op in range(_MEMORY_OPS)
    ]


def _route(logits, bias, live, experts, routed_scale):
    """A token's top-8 from its FP32 router logits [experts] (noaux_tc with one
    group): the sigmoid scores, the picks the largest score + bias in full FP32
    (``stable_key``'s order: ties to the lower expert), one int32 max of the
    order-preserving value bits and one min of the expert at it per pick; the
    picks' scores over their sum (one reduction) times ``routed_scale``. Its ids
    in pick order (``experts`` where ``live`` fails), weights and the picks as a
    bitmap of the experts."""
    scores = torch.sigmoid(logits)
    bits = (scores + bias).view(torch.int32)
    values = torch.where(bits < 0, -2147483648 - bits, bits)
    indices = hl.arange(experts).to(torch.int32)
    picks = hl.arange(8)
    ids = hl.full([8], 0, dtype=torch.int32)
    for pick in hl.static_range(8):
        best = torch.amax(values)
        best_id = torch.amin(torch.where(values == best, indices, experts))
        values = torch.where(indices == best_id, -2147483648, values)
        ids = torch.where(picks == pick, best_id, ids)
    # Every pick's score in one reduction (one-hot sums: exact).
    weights = torch.sum(
        torch.where(indices[None, :] == ids[:, None], scores[None, :], 0.0), dim=1
    )
    total = torch.sum(weights)
    ids = torch.where(live, ids, experts)
    return ids, weights / total * routed_scale, pick_bits(ids, live, experts // 32)


CONFIG = {
    # W13 and W2 take the window's rows (T <= 8) as the MMA's 8 columns.
    "block_sizes": [16, 128, 128, 16, 128, 256, 16, 128, 8, 8, 512],
    "num_warps": 4,
    "pid_type": "persistent_blocked",
    "cross_loop_pipeline": "dynamic",
    "num_sm_multiplier": 1,
    "num_stages": 4,
    "range_warp_specializes": [True] * 17,
    "indexing": _indexing(),
    "cross_loop_inband_protocol": ["lamport"],
    # Over DSMEM: a token's norm input and row partial sums (six of them, in
    # clusters of six).
    "cross_loop_dsmem": [False] * 14 + [True, True],
}


@helion.kernel(
    static_shapes=False,
    backend="gluon",
    triton_do_not_specialize=False,
    config=helion.Config(**CONFIG),
    ignore_warnings=[helion.exc.TensorOperationInWrapper, helion.exc.FastMathUnused],
)
def glm5_3_moe_tp4(
    hidden_input,  # [T, 6144] bf16, the post-attention RMSNorm output
    router_weight,  # [256, 6144] bf16
    correction_bias,  # [256] f32
    w13,  # [256, 1024, 3072] uint8 (trtllm-gen layout)
    w13_scale,  # [256, 8 * 12, 16, 256] e4m3 (swizzled)
    w2,  # [256, 6144, 256] uint8 (trtllm-gen layout)
    w2_scale,  # [256, 48, 16, 256] e4m3 (swizzled, 128-row tiles)
    alpha1,  # [256] f32
    alpha2,  # [256] f32
    input_global_scale,  # [1] f32
    activation_global_scale,  # [1] f32
    shared_w13,  # [1024, 6144] bf16: gate [0, 512), up [512, 1024)
    shared_w2,  # [6144, 512] bf16
    top_k: int,
    routed_scale: float,
    residual,  # [T, 6144] bf16
    norm_weight,  # [6144] bf16, the next layer's
    symmetric_partial,  # [T, 6144] bf16 symmetric memory
    group_name: hl.ProcessGroupName,
):
    hl.specialize(hidden_input.size())
    hl.specialize(hidden_input.stride())
    hl.specialize(router_weight.size())
    hl.specialize(router_weight.stride())
    hl.specialize(correction_bias.size())
    hl.specialize(correction_bias.stride())
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
    hl.specialize(residual.size())
    hl.specialize(residual.stride())
    hl.specialize(norm_weight.size())
    hl.specialize(norm_weight.stride())
    hl.specialize(symmetric_partial.size())
    hl.specialize(symmetric_partial.stride())
    top_k = hl.specialize(top_k)

    rows, hidden = hidden_input.shape
    experts = router_weight.size(0)
    twice_intermediate, packed_hidden = w13.shape[1:]
    intermediate = twice_intermediate // 2
    packed_intermediate = intermediate // 2
    hidden_groups = hidden // 16
    activation_groups = intermediate // 16
    shared_twice = shared_w13.size(0)
    shared_intermediate = shared_twice // 2
    # Eight picks; the 8-word expert bitmaps.
    assert experts == 256 and top_k == 8
    assert intermediate == 512 and packed_intermediate == 256
    # Token rows padded to a power of two: the routing table's rows.
    assert rows <= 8
    padded_rows = 8
    slots = rows * top_k
    # K steps of 512 FP4 values: 256 packed bytes and 32 scales per row.
    w13_steps = packed_hidden // 256
    w13_tiles = twice_intermediate // 128
    # W2: 128-row tiles of one K step, a sixth of them per task.
    w2_tiles = hidden // 128
    w2_parts = 6
    w2_part_tiles = w2_tiles // w2_parts
    w2_rows = w2.view(experts, w2_tiles, 128, packed_intermediate)

    # 16 K chunks: the router's 64-task stream ends first (it heads the routing
    # chain), the routing sums 16 partials.
    router_chunks = 16
    router_chunk_k = hidden // router_chunks
    router_weight_chunked = router_weight.view(experts, router_chunks, router_chunk_k)
    router_x = hidden_input.view(rows, router_chunks, router_chunk_k)
    # padded_rows rows: the routing root's tasks (one token each) cover all 8
    # rows and read the rows past ``rows``, which no task writes (their results
    # are guarded off).
    router_partials = torch.empty(
        (padded_rows, router_chunks, experts),
        dtype=torch.float32,
        device=hidden_input.device,
    )
    router_logits = torch.empty(
        (rows, experts), dtype=torch.float32, device=hidden_input.device
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
        (padded_rows, top_k), dtype=torch.float32, device=hidden_input.device
    )
    topk_ids = torch.empty(
        (padded_rows, top_k), dtype=torch.int32, device=hidden_input.device
    )
    # Each token's picks as a bitmap of the experts (bit b of word w: expert 32
    # w + b).
    expert_words = experts // 32
    token_experts = torch.empty(
        (padded_rows, expert_words), dtype=torch.int32, device=hidden_input.device
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
        (rows, top_k, hidden), dtype=torch.bfloat16, device=w2.device
    )
    routed_rows = routed_output.view(rows * top_k, hidden)

    # Shared expert: K-split gate/up partials, then SiLU and the BF16 down
    # input.
    shared_chunks = 8
    shared_chunk_k = hidden // shared_chunks
    shared_partials = torch.empty(
        (shared_chunks, rows, shared_twice), dtype=torch.float32, device=w13.device
    )
    shared_w13_chunked = shared_w13.view(shared_twice, shared_chunks, shared_chunk_k)
    hidden_chunked = hidden_input.view(rows, shared_chunks, shared_chunk_k)
    shared_act = torch.empty(
        (rows, shared_intermediate), dtype=torch.bfloat16, device=w13.device
    )
    shared_output = torch.empty((rows, hidden), dtype=torch.bfloat16, device=w2.device)

    assert symmetric_partial.size() == (rows, hidden)
    remote_partials = torch.ops.symm_mem.get_remote_tensors(
        symmetric_partial, group_name
    )
    output = torch.empty((rows, hidden), dtype=torch.bfloat16, device=w2.device)
    residual_output = torch.empty_like(output)
    communication_n = 1024
    norm_tasks = hidden // communication_n
    norm_partial = torch.empty(
        (rows, norm_tasks), dtype=torch.float32, device=w2.device
    )
    norm_input = torch.empty((rows, hidden), dtype=torch.float32, device=w2.device)

    # Programmatic dependent launch: wait for the previous kernel at entry.
    hl.pdl_wait()
    # The code toward L2 first: the previous launch's weight stream evicted it.
    hl.prefetch_code()

    # Router projection: K chunks of the 256 x 6144 BF16 weight.
    for router_tile_e, router_tile_c, router_tile_t in hl.tile(
        [experts, router_chunks, rows], block_size=[64, 1, None]
    ):
        router_acc = hl.zeros([router_tile_e, router_tile_t], dtype=torch.float32)
        for router_tile_k in hl.tile(router_chunk_k):
            router_acc = hl.dot(
                router_weight_chunked[
                    router_tile_e, router_tile_c.begin, router_tile_k
                ],
                router_x[router_tile_t, router_tile_c.begin, router_tile_k].T,
                acc=router_acc,
            )
        router_partials[router_tile_t, router_tile_c, router_tile_e] = router_acc.T[
            :, None, :
        ]

    # The input's NVFP4 copy (routed experts).
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

    # Routing, one token a task (padded rows pick no experts).
    for topk_tile_t in hl.tile(padded_rows, block_size=1):
        topk_token = topk_tile_t.begin
        # The bias loads under the partials' sum.
        topk_bias = correction_bias[:]
        topk_logits = torch.sum(router_partials[topk_token, :, :], dim=0)
        if topk_token < rows:
            router_logits[topk_token, :] = topk_logits
        topk_id, topk_weight, topk_union = _route(
            topk_logits,
            topk_bias,
            hl.arange(top_k) * 0 + topk_token < rows,
            experts,
            routed_scale,
        )
        topk_ids[topk_tile_t, :] = topk_id[None, :]
        topk_weights[topk_tile_t, :] = topk_weight[None, :]
        token_experts[topk_tile_t, :] = topk_union[None, :]

    # Shared gate/up: BF16 K chunks.
    for sp_tile_n, sp_tile_c, sp_tile_t in hl.tile(
        [shared_twice, shared_chunks, rows], block_size=[128, 1, None]
    ):
        sp_acc = hl.zeros([sp_tile_n, sp_tile_t], dtype=torch.float32)
        for sp_tile_k in hl.tile(shared_chunk_k):
            sp_acc = hl.dot(
                shared_w13_chunked[sp_tile_n, sp_tile_c.begin, sp_tile_k],
                hidden_chunked[sp_tile_t, sp_tile_c.begin, sp_tile_k].T,
                acc=sp_acc,
            )
        shared_partials[sp_tile_c.begin, sp_tile_t, sp_tile_n] = sp_acc.T

    # Shared SiLU(gate) * up in BF16 (SGLang's BF16 GEMM output and SiluAndMul).
    for sa_tile_t, sa_tile_j in hl.tile(
        [rows, shared_intermediate], block_size=[1, None]
    ):
        sa_gate = torch.sum(shared_partials[:, sa_tile_t, sa_tile_j], dim=0)
        sa_up = torch.sum(
            shared_partials[:, sa_tile_t, sa_tile_j.index + shared_intermediate], dim=0
        )
        g = sa_gate.to(torch.bfloat16).to(torch.float32)
        u = sa_up.to(torch.bfloat16).to(torch.float32)
        shared_act[sa_tile_t, sa_tile_j] = (g * torch.sigmoid(g) * u).to(torch.bfloat16)

    # Shared down projection (BF16), ahead of the routed experts: its weights
    # stream while the routing runs; 64-row tasks (96) end sooner once the
    # activation lands, so their workers join W13 earlier.
    for sd_tile_n, sd_tile_t in hl.tile([hidden, rows], block_size=[64, None]):
        sd_acc = hl.zeros([sd_tile_n, sd_tile_t], dtype=torch.float32)
        for sd_tile_k in hl.tile(shared_intermediate):
            sd_acc = hl.dot(
                shared_w2[sd_tile_n, sd_tile_k],
                shared_act[sd_tile_t, sd_tile_k].T,
                acc=sd_acc,
            )
        shared_output[sd_tile_t, sd_tile_n] = sd_acc.T.to(torch.bfloat16)

    # Routed W13 per (row tile, distinct expert): the window's j-th distinct
    # expert (ascending ids), one 128-row tile over the full K, then SwiGLU and
    # the NVFP4 activation of the tile's 64 intermediates for every token of the
    # window.
    for p_tile, p_t, p_j in hl.tile([w13_tiles, rows, slots], block_size=[1, None, 1]):
        p_words = hl.reduce(operator.or_, token_experts[:, :], dim=0)
        p_counts = popcount(p_words)
        p_before = hl.cumsum(p_counts, dim=0) - p_counts
        p_distinct = torch.sum(p_counts)
        p_global_scale = torch.sum(activation_global_scale[:])
        if p_j.begin < p_distinct:
            p_expert = expert_of(
                p_words, p_before, p_j.begin, lanes=expert_words, counts=p_counts
            )
            p_alpha = alpha1[p_expert]
            p_acc = hl.zeros([128, p_t], dtype=torch.float32)
            for p_s in hl.tile(w13_steps, block_size=1):
                p_acc = hl.dot_scaled(
                    w13_view[
                        p_expert, p_tile.begin * 128 + hl.arange(128), p_s.begin, :
                    ],
                    scale_tile(
                        w13_scale[p_expert, p_tile.begin * w13_steps + p_s.begin, :, :]
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
                (p_acc.T * p_alpha).reshape(p_tokens, 4, 2, 2, 8).permute(0, 1, 4, 2, 3)
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
            activation_q_groups[p_j.begin, p_t, p_tile.begin, :, :] = (
                p_low | p_high << 4
            ).to(torch.uint8)
            activation_scale_groups[p_j.begin, p_t, p_tile.begin, :] = p_block_scale

    # Routed W2 per (sixth, distinct expert): 128-row tiles of one K step, for
    # every token of the window that picked the expert, stored in the token's
    # row of its pick.
    for q_part, q_t, q_j in hl.tile([w2_parts, rows, slots], block_size=[1, None, 1]):
        q_words = hl.reduce(operator.or_, token_experts[:, :], dim=0)
        q_counts = popcount(q_words)
        q_before = hl.cumsum(q_counts, dim=0) - q_counts
        q_distinct = torch.sum(q_counts)
        if q_j.begin < q_distinct:
            q_expert = expert_of(
                q_words, q_before, q_j.begin, lanes=expert_words, counts=q_counts
            )
            q_alpha = alpha2[q_expert]
            q_hits = topk_ids[q_t, :] == q_expert
            q_where = torch.sum(
                torch.where(q_hits, hl.arange(top_k)[None, :], 0), dim=1
            )
            q_used = (torch.sum(q_hits.to(torch.int32), dim=1) > 0) & (q_t.index < rows)
            for q_n in hl.tile(w2_part_tiles, block_size=1):
                q_row_tile = q_part.begin * w2_part_tiles + q_n.begin
                q_acc = hl.zeros([128, q_t], dtype=torch.float32)
                for _q_k in hl.tile(1, block_size=1):
                    q_acc = hl.dot_scaled(
                        w2_rows[q_expert, q_row_tile, :, :],
                        scale_tile(w2_scale[q_expert, q_row_tile, :, :]),
                        "e2m1",
                        activation_q[q_j.begin, q_t, :].T,
                        activation_scale[q_j.begin, q_t, :],
                        "e2m1",
                        acc=q_acc,
                        out_dtype=torch.float32,
                    )
                # Physical row b*32 + 8 c + m holds hidden b*32 + 4 m + c.
                q_values = (q_acc.T * q_alpha).to(torch.bfloat16)
                q_phys = hl.arange(128)
                q_hidden = (q_phys // 32) * 32 + (q_phys % 8) * 4 + (q_phys % 32) // 8
                hl.store(
                    routed_rows,
                    [
                        (q_t.index * top_k + q_where)[:, None],
                        (q_row_tile * 128 + q_hidden)[None, :],
                    ],
                    q_values,
                    extra_mask=q_used[:, None],
                )

    # Each token's top-k rows, weighted in pick order, plus the shared output.
    for partial_tile_t, partial_tile_n in hl.tile([rows, hidden], block_size=[1, None]):
        partial_token = partial_tile_t.begin
        # The shared expert's output first: it is ready long before the routed
        # rows.
        partial_shared = shared_output[partial_token, partial_tile_n].to(torch.float32)
        partial_values = routed_output[partial_token, :, partial_tile_n].to(
            torch.float32
        )
        partial_weights = (
            topk_weights[partial_token, :].to(torch.bfloat16).to(torch.float32)
        )
        partial_routed = torch.sum(partial_values * partial_weights[:, None], dim=0)
        symmetric_partial[partial_token, partial_tile_n] = (
            partial_routed + partial_shared
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
        norm_partial[comm_token, comm_tile_n.id] = torch.sum(comm_sum * comm_sum)
        norm_input[comm_token, comm_tile_n] = comm_sum
    for norm_tile_t, norm_tile_n in hl.tile(
        [rows, hidden], block_size=[1, communication_n]
    ):
        norm_token = norm_tile_t.begin
        # The weights first: they do not wait for the all-reduce.
        norm_weights = norm_weight[norm_tile_n].to(torch.float32)
        norm_scale = torch.rsqrt(
            torch.sum(norm_partial[norm_token, :]) / hidden + NORM_EPS
        )
        output[norm_token, norm_tile_n] = (
            norm_input[norm_token, norm_tile_n] * norm_scale * norm_weights
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


# The served verify window (speculative-num-draft-tokens: EAGLE 5/1/6).
TOKENS = (6,)


def correctness_check(group: dist.ProcessGroup, verbose: bool = True) -> None:
    """Every rank against ``_setup.moe_reference`` on its own weights at the
    served window, for a "spread" window (independent tokens), a "clustered" one
    (every token the same row: one expert set) and a spread window one row
    shorter: the top-k ids as sets, the NVFP4 input and activations against
    their exact quantization, the residual output and the normed output within
    2e-2 relative RMS, and a second launch bitwise equal (``qwen3_5_moe_tp4``'s
    check)."""
    rank = group.rank()
    device = torch.device("cuda", torch.cuda.current_device())
    shape = _setup.MoeShape()
    weights = _setup.moe_weights(shape, seed=100 + rank, device=device)
    weight_args = _setup.moe_weight_args(weights, shape)
    ok = True
    for window in TOKENS:
        g = torch.Generator(device=device)
        g.manual_seed(7)
        cases = {
            f"spread, T={window}": torch.randn(
                window, shape.hidden, generator=g, device=device
            ),
            f"clustered, T={window}": torch.randn(
                1, shape.hidden, generator=g, device=device
            ).repeat(window, 1),
        }
        residual = torch.randn(window, shape.hidden, generator=g, device=device).to(
            torch.bfloat16
        )
        norm_weight = (
            1 + 0.1 * torch.randn(shape.hidden, generator=g, device=device)
        ).to(torch.bfloat16)
        short = window - 1
        cases[f"spread, T={short}"] = torch.randn(
            short, shape.hidden, generator=g, device=device
        )
        partials = {
            tokens: symmetric_partials(tokens, shape.hidden, group.group_name, device)
            for tokens in (window, short)
        }
        for name, xf in cases.items():
            tokens = xf.size(0)
            x, res = xf.to(torch.bfloat16), residual[:tokens]
            out = glm5_3_moe_tp4(
                x, *weight_args, res, norm_weight, partials[tokens][0], group.group_name
            )
            torch.cuda.synchronize()
            ids = out[4][:tokens, : shape.top_k]
            distinct = len(set(ids.flatten().tolist()))
            partial, ref_ids, quant = _setup.moe_reference(
                x,
                weights,
                (out[7], out[8], out[9][:distinct], out[10][:distinct]),
                shape,
            )
            rows = []
            for label, (got, exact) in quant.items():
                ok &= report(rows, label, got, exact, 2e-2)
            same_ids = torch.equal(ids.sort(-1).values, ref_ids.sort(-1).values)
            rows.append(
                f"  {'top-k ids':<22} {'equal' if same_ids else 'DIFFER  <-- FAIL'}"
            )
            ok &= same_ids
            output, residual_output = all_reduce_norm(
                partial, res, norm_weight, NORM_EPS
            )
            ok &= report(rows, "residual_out", out[1], residual_output, 2e-2)
            ok &= report(rows, "output", out[0], output, 2e-2)
            again = glm5_3_moe_tp4(
                x, *weight_args, res, norm_weight, partials[tokens][1], group.group_name
            )
            deterministic = torch.equal(again[0], out[0])
            rows.append(
                f"  {'determinism':<22} "
                f"{'bitwise' if deterministic else 'DIFFERS  <-- FAIL'}"
            )
            ok &= deterministic
            if verbose or not ok:
                say(f"rank{rank} {name}\n" + "\n".join(rows))
    verdict(ok, group, "glm5_3_moe_tp4")


# Distinct experts over the served window: mean 31.8 (p10 25, p50 31, p90 40).
DISTINCT = 32


def benchmark(group: dist.ProcessGroup, verbose: bool = True) -> dict[str, Any]:
    """Cold-weight timing at the served window (``qwen3_5_moe_tp4.benchmark``):
    ``LAYERS`` distinct weight copies (default 8, about 1.4 GB each; even) back
    to back, ``REPS`` times (default 4) in one CUDA graph. Each copy's router
    routes the window to the served mean of distinct experts (``DISTINCT``;
    the env variable of that name sets another count)."""
    rank = group.rank()
    device = torch.device("cuda", torch.cuda.current_device())
    layers = int(os.environ.get("LAYERS", "8"))
    assert layers % 2 == 0, "LAYERS must be even"
    reps = int(os.environ.get("REPS", "4"))
    shape = _setup.MoeShape()
    weights = [
        _setup.moe_benchmark_weights(i, rank, device, shape) for i in range(layers)
    ]
    torch.cuda.empty_cache()
    norm_weight = torch.ones(shape.hidden, device=device, dtype=torch.bfloat16)
    results = {}
    for tokens in TOKENS:
        distinct = int(os.environ.get("DISTINCT", DISTINCT))
        g = torch.Generator(device=device)
        g.manual_seed(3)
        x = torch.randn(tokens, shape.hidden, generator=g, device=device).to(
            torch.bfloat16
        )
        args = []
        for i, w in enumerate(weights):
            w["router"] = _setup.window_router(x, distinct, shape, seed=i)
            args.append(_setup.moe_weight_args(w, shape))
        residual = torch.randn(tokens, shape.hidden, generator=g, device=device).to(
            torch.bfloat16
        )
        partials = symmetric_partials(tokens, shape.hidden, group.group_name, device)

        def launch(
            i: int, x=x, args=args, residual=residual, partials=partials
        ) -> tuple[torch.Tensor, ...]:
            return glm5_3_moe_tp4(
                x, *args[i], residual, norm_weight, partials[i % 2], group.group_name
            )

        def step(launch=launch) -> None:
            for i in range(layers):
                launch(i)

        median, minimum = graph_time(step, layers, reps)
        routed = [
            len(set(launch(i)[4][:tokens, : shape.top_k].flatten().tolist()))
            for i in range(layers)
        ]
        # Bytes a launch streams: the router, the shared expert, and W13, W2 and
        # their scales for each distinct expert.
        E, H, N = shape.num_experts, shape.hidden, shape.intermediate
        expert_bytes = N * H + 2 * N * H // 16 + H * N // 2 + H * N // 16
        shared_bytes = 2 * 3 * shape.shared_intermediate * H
        megabytes = (distinct * expert_bytes + 2 * E * H + shared_bytes) / 1e6
        if verbose:
            say(
                f"rank{rank} T={tokens} distinct={'/'.join(map(str, routed))} "
                f"us/launch med {median:.2f} min {minimum:.2f} "
                f"(~{megabytes:.0f} MB -> {megabytes / median:.2f} TB/s)",
            )
        results[tokens] = {"latency_us": median, "min_latency_us": minimum}
    return results


def main(verbose: bool = True, check_only: bool = False) -> dict[str, Any]:
    """The four-rank check, then the cold-weight timing (``check_only``: the
    check alone)."""
    return run_main("glm5_3_moe_tp4", correctness_check, benchmark, verbose, check_only)


if __name__ == "__main__":
    main(check_only="--check" in sys.argv)
