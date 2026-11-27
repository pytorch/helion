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
  (two tokens per task) sums them, rounds to BF16 and takes the top 10 by packed
  (value, index) keys: the top 10 of each group of 128 experts (one int32 max
  reduction per round, a warp redux), then the top 10 of those 40. It also
  writes each token's picks as a bitmap of the 512 experts (16 int32 words).
- input quant: the input's NVFP4 (routed experts) and static FP8 (shared
  expert) copies.
- routed W13, one task per (128-row tile, slot j): the window's j-th distinct
  expert (ascending ids, from the union of the tokens' bitmaps) over the full
  K, SwiGLU and the NVFP4 activation of every token of the window; slots past
  the window's distinct experts find none (no dispatch root, no W13 K split).
- routed W2, one task per (hidden quarter, slot j): 256-row tiles, each
  physical row stored at its hidden column in the row of each token's pick.
- shared expert: K-split gate / up, SiLU, the FP8 down projection, and its
  sigmoid gate in a root of its own (off the routing task).
- combine: each token's top-k rows weighted in pick order plus the gated shared
  output, this rank's partial in symmetric memory.
- all-reduce: the ranks' partials summed in rank order (FP32) plus the residual,
  the BF16 sum's row partial sums of squares, then the RMSNorm.

``hl.prefetch_code()`` at entry brings the code toward L2 (the previous launch's
weight stream evicted it). The small loads and stores (the routing's, the
routed tasks' expert bitmaps and activation stores, the shared outputs and the
tail's) are pointer accesses (``_POINTER_OPS``): issued together, without a
round trip through the TMA ring each.

Besides the next layer's input and residual, it returns the router logits, the
top-k weights and ids, the routed rows, the shared output and the NVFP4 input
and activations for checks; activation slot j holds the window's j-th distinct
expert's rows.

``correctness_check()`` compares every rank with a torch reference on the
logical weights (``_setup``); ``main()`` runs it, then times the kernel alone
with cold weights; ``--check`` runs only the check::

    NVSHMEM_DISABLE_CUDA_VMM=1 PYTHONPATH=. \\
      python -m torch.distributed.run --standalone --nproc-per-node=4 \\
      pretuned_kernels/megakernels/qwen3_5_moe_tp4/qwen3_5_moe_tp4.py [--check]
"""

from __future__ import annotations

import operator
import os
import sys
from typing import Any

from pretuned_kernels.megakernels._qwen3_5_tp4 import FP8_MAX
from pretuned_kernels.megakernels._qwen3_5_tp4 import NORM_EPS
from pretuned_kernels.megakernels._qwen3_5_tp4 import all_reduce_norm
from pretuned_kernels.megakernels._qwen3_5_tp4 import graph_time
from pretuned_kernels.megakernels._qwen3_5_tp4 import report
from pretuned_kernels.megakernels._qwen3_5_tp4 import run_main
from pretuned_kernels.megakernels._qwen3_5_tp4 import say
from pretuned_kernels.megakernels._qwen3_5_tp4 import symmetric_partials
from pretuned_kernels.megakernels.deepseek_v3_moe_nvfp4._common import FP4_MAX
from pretuned_kernels.megakernels.deepseek_v3_moe_nvfp4._common import fp4_nibble
from pretuned_kernels.megakernels.deepseek_v3_moe_nvfp4._common import popcount
from pretuned_kernels.megakernels.deepseek_v3_moe_nvfp4._common import scale_tile
from pretuned_kernels.megakernels.qwen3_5_moe_tp4 import _setup
from pretuned_kernels.megakernels.qwen3_5_moe_tp4._common import expert_of
from pretuned_kernels.megakernels.qwen3_5_moe_tp4._common import route
from pretuned_kernels.megakernels.qwen3_5_moe_tp4._common import scale_tiles_16
import torch
import torch.distributed as dist

import helion
import helion.language as hl

# Config.indexing slots (one per load / store, in the order Helion visits them: a
# loop's ops at the loop's place in its root) lowered as pointer accesses; every
# other memory op is a tensor descriptor (TMA ring / bulk store). Picked by hand
# from the bound kernel's ``config_spec.memory_op_facts``: each slot's load or
# store and tensor (test_verify_megakernels_codegen checks them), by root. Slot
# numbers: an added, removed or reordered memory op moves the later ones, so
# re-derive the table after editing the roots.
_MEMORY_OPS = 70
_POINTER_OPS = {
    2: "store router_partials",  # router
    # routing (each task's two tokens): router partials, logits, top-k ids and
    # weights; the expert bitmaps
    11: "load router_partials_wide",
    12: "load router_partials_wide",
    13: "store router_logits_wide",
    14: "store router_logits_wide",
    15: "store topk_ids",
    16: "store topk_ids",
    17: "store topk_weights",
    18: "store topk_weights",
    19: "store token_experts",
    22: "store shared_partials",  # shared gate / up
    25: "store shared_gate",  # shared gate: token rows
    26: "store shared_gate",  # shared gate: padding rows
    31: "store shared_act",  # shared activation (FP8)
    32: "load token_experts",  # W13: expert bitmaps, activation and its scales
    39: "store activation_q_groups",
    40: "store activation_scale_groups",
    45: "store shared_output",  # shared down
    46: "load token_experts",  # W2
    54: "load shared_output",  # combine: shared output, routed rows, top-k weights
    55: "load routed_output",
    56: "load topk_weights",
    58: "load residual",  # all-reduce
    63: "store residual_output",
    64: "store norm_partial",
    65: "store norm_input",
    66: "load norm_weight",  # norm
    67: "load norm_partial",
    68: "load norm_input",
    69: "store output",
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
    # W13's and W2's token tiles (block sizes 6 and 9) are the window's <= 8 rows:
    # 8-column block-scaled MMAs.
    config=helion.Config(
        block_sizes=[16, 128, 32, 16, 128, 256, 8, 16, 128, 8, 512],
        num_warps=4,
        pid_type="persistent_blocked",
        cross_loop_pipeline="dynamic",
        num_sm_multiplier=1,
        num_stages=4,
        range_warp_specializes=[True] * 16,
        indexing=_indexing(),
        cross_loop_inband_protocol=["lamport"],
        # Over DSMEM: W13's expert bitmaps (every cluster routes) and a token's
        # norm input and row partial sums.
        cross_loop_dsmem=[False] * 7 + [True] + [False] * 9 + [True, True],
    ),
    ignore_warnings=[helion.exc.TensorOperationInWrapper, helion.exc.FastMathUnused],
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
    w2_part_tiles = w2_tiles // 4
    w2_rows = w2.view(experts, w2_tiles, 256, packed_intermediate)
    w2_scale_rows = w2_scale.view(experts, w2_tiles, 16, 256)

    router_chunks = 8
    router_chunk_k = hidden // router_chunks
    router_weight_chunked = router_weight.view(experts, router_chunks, router_chunk_k)
    hidden_chunked = hidden_input.view(rows, router_chunks, router_chunk_k)
    # padded_rows rows: the routing tasks (token pairs) also read the rows past
    # ``rows``, which no task writes (their results are guarded off).
    router_partials = torch.empty(
        (padded_rows, router_chunks, experts),
        dtype=torch.float32,
        device=hidden_input.device,
    )
    router_logits = torch.empty(
        (rows, experts), dtype=torch.bfloat16, device=hidden_input.device
    )
    topk_groups = 4
    topk_group_n = experts // topk_groups
    topk_segment_n = topk_group_n // topk_groups
    router_partials_wide = router_partials.view(
        padded_rows, router_chunks, topk_groups, topk_group_n
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
    # Each token's picks as a bitmap of the experts (bit b of word w: expert 32 w + b).
    expert_words = experts // 32
    assert expert_words == 16
    token_experts = torch.empty(
        (padded_rows, expert_words), dtype=torch.int32, device=hidden_input.device
    )
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

    # Routing, two tokens a task (padded rows pick no experts): every launch
    # cluster runs all four tasks, a rank each, for W13's copy of the bitmaps in
    # its shared memory.
    for topk_tile_t in hl.tile(padded_rows, block_size=2):
        first = topk_tile_t.begin
        picks = hl.arange(pick_lanes_n)
        # Both tokens' partials load before either top-k.
        first_parts = router_partials_wide[first, :, :, :]
        second_parts = router_partials_wide[first + 1, :, :, :]
        first_logits = torch.sum(first_parts, dim=0).to(torch.bfloat16)
        if first < rows:
            router_logits_wide[first, :, :] = first_logits
        first_ids, first_weights, first_union = route(
            first_logits,
            picks * 0 + first < rows,
            top_k,
            experts,
            picks,
            topk_groups,
            topk_segment_n,
        )
        second_logits = torch.sum(second_parts, dim=0).to(torch.bfloat16)
        if first + 1 < rows:
            router_logits_wide[first + 1, :, :] = second_logits
        second_ids, second_weights, second_union = route(
            second_logits,
            picks * 0 + first + 1 < rows,
            top_k,
            experts,
            picks,
            topk_groups,
            topk_segment_n,
        )
        topk_ids[first, :] = first_ids
        topk_ids[first + 1, :] = second_ids
        topk_weights[first, :] = first_weights
        topk_weights[first + 1, :] = second_weights
        token_experts[topk_tile_t, :] = torch.where(
            hl.arange(2)[:, None] == 0, first_union[None, :], second_union[None, :]
        )

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

    # Routed W13 per (row tile, distinct expert): the window's j-th distinct expert
    # (ascending ids), one 128-row tile over the full K, then SwiGLU and the NVFP4
    # activation of the tile's 64 intermediates for every token of the window.
    # Tiles first: an expert's four tiles go to four workers, and the distinct
    # experts fill the leading tasks (the others find no expert).
    for p_tile, p_t, p_j in hl.tile([w13_tiles, rows, slots], block_size=[1, None, 1]):
        p_words = hl.reduce(operator.or_, token_experts[:, :], dim=0)
        p_counts = popcount(p_words)
        p_before = hl.cumsum(p_counts, dim=0) - p_counts
        p_distinct = torch.sum(p_counts)
        p_global_scale = torch.sum(activation_global_scale[:])
        if p_j.begin < p_distinct:
            p_expert = expert_of(p_words, p_counts, p_before, p_j.begin)
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

    # Routed W2 per (quarter, distinct expert): 4 tiles of 256 physical hidden rows,
    # K = 256 (one MMA step each), for every token of the window that picked the
    # expert, stored in the token's row of its pick.
    for q_part, q_t, q_j in hl.tile([4, rows, slots], block_size=[1, None, 1]):
        q_words = hl.reduce(operator.or_, token_experts[:, :], dim=0)
        q_counts = popcount(q_words)
        q_before = hl.cumsum(q_counts, dim=0) - q_counts
        q_distinct = torch.sum(q_counts)
        if q_j.begin < q_distinct:
            q_expert = expert_of(q_words, q_counts, q_before, q_j.begin)
            q_alpha = alpha2[q_expert]
            q_hits = topk_ids[q_t, :] == q_expert
            q_where = torch.sum(
                torch.where(q_hits, hl.arange(pick_lanes_n)[None, :], 0), dim=1
            )
            q_used = (torch.sum(q_hits.to(torch.int32), dim=1) > 0) & (q_t.index < rows)
            for q_n in hl.tile(w2_part_tiles, block_size=1):
                q_row_tile = q_part.begin * w2_part_tiles + q_n.begin
                q_acc = hl.zeros([256, q_t], dtype=torch.float32)
                for _q_k in hl.tile(1, block_size=1):
                    q_acc = hl.dot_scaled(
                        w2_rows[q_expert, q_row_tile, :, :],
                        scale_tiles_16(w2_scale_rows[q_expert, q_row_tile, :, :]),
                        "e2m1",
                        activation_q[q_j.begin, q_t, :].T,
                        activation_scale[q_j.begin, q_t, :],
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
        # The shared expert's output first: it is ready long before the routed
        # rows (a task waits for what it reads in the order it reads it).
        partial_shared = shared_output[partial_token, partial_tile_n].to(torch.float32)
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


# The verify window the checks and timings use (speculative-num-draft-tokens).
TOKENS = 7


def correctness_check(group: dist.ProcessGroup, verbose: bool = True) -> None:
    """Every rank against ``_setup.reference_partial`` on its own weights, for a
    "spread" window (independent tokens), a "clustered" one (every token the
    same row: one expert set) and a shorter spread one (routing tasks whose
    first token is padding): the top-k ids as sets, the residual output (the
    ranks' reference partials summed in rank order, plus the residual) and the
    normed output within 2e-2 relative RMS, and a second launch bitwise equal.
    Each stage is judged on its own: the kernel's NVFP4 input and activations
    against the exact quantization (of the input, and of the reference's SwiGLU
    of W13 on the kernel's NVFP4 input), within 2e-2 relative RMS; each pick's
    routed W2 row (W2 on the kernel's activations, matched to the reference's
    by expert) and the gated shared output before the all-reduce, within 2e-2
    relative RMS (the shared output's FP8 activation is the reference's own, so
    a rounding tie can flip there); and the outputs with each GEMM reading the
    kernel's copy of its input."""
    rank = group.rank()
    device = torch.device("cuda", torch.cuda.current_device())
    shape = _setup.Shape()
    weights = _setup.make_weights(shape, seed=100 + rank, device=device)
    g = torch.Generator(device=device)
    g.manual_seed(7)
    cases = {
        "spread": torch.randn(TOKENS, shape.hidden, generator=g, device=device),
        "clustered": torch.randn(1, shape.hidden, generator=g, device=device).repeat(
            TOKENS, 1
        ),
    }
    residual = torch.randn(TOKENS, shape.hidden, generator=g, device=device).to(
        torch.bfloat16
    )
    norm_weight = (1 + 0.1 * torch.randn(shape.hidden, generator=g, device=device)).to(
        torch.bfloat16
    )
    cases["spread, T=5"] = torch.randn(5, shape.hidden, generator=g, device=device)
    partials = {
        tokens: symmetric_partials(tokens, shape.hidden, group.group_name, device)
        for tokens in (TOKENS, 5)
    }
    weight_args = _setup.kernel_weight_args(weights, shape)
    ok = True
    for name, xf in cases.items():
        tokens = xf.size(0)
        x, res = xf.to(torch.bfloat16), residual[:tokens]
        out = qwen3_5_moe_tp4(
            x, *weight_args, res, norm_weight, partials[tokens][0], group.group_name
        )
        torch.cuda.synchronize()
        ids = out[4][:tokens, : shape.top_k]
        distinct = len(set(ids.flatten().tolist()))
        partial, ref_ids, quant, stages = _setup.reference_partial(
            x, weights, (out[7], out[8], out[9][:distinct], out[10][:distinct]), shape
        )
        rows = []
        for label, (got, exact) in quant.items():
            ok &= report(rows, label, got, exact, 2e-2)
        same_ids = torch.equal(ids.sort(-1).values, ref_ids.sort(-1).values)
        rows.append(
            f"  {'top-k ids':<22} {'equal' if same_ids else 'DIFFER  <-- FAIL'}"
        )
        ok &= same_ids
        if same_ids:
            # The kernel's pick lanes follow its own top-k order: match by expert.
            lane = ids.unsqueeze(-1) == ref_ids.unsqueeze(1)  # [T, K kernel, K ref]
            expected = torch.einsum("tkr,trh->tkh", lane.float(), stages["routed rows"])
            got_rows = out[5][:tokens, : shape.top_k]
            ok &= report(rows, "routed rows", got_rows, expected, 2e-2)
            # Per W2 task (a pick's hidden quarter): an error in one task is a
            # 1/(4 K T) share of the tensor-wide error, so judge each on its own.
            quarters = (got_rows.float() - expected).unflatten(-1, (4, -1))
            worst = (
                quarters.norm(dim=-1) / expected.unflatten(-1, (4, -1)).norm(dim=-1)
            ).max()
            task_ok = bool(worst <= 2e-2)
            rows.append(
                f"  {'routed W2 tasks':<22} worst rel {float(worst):.2e}"
                + ("" if task_ok else "  <-- FAIL")
            )
            ok &= task_ok
        ok &= report(
            rows, "shared output", out[6][:tokens], stages["shared output"], 2e-2
        )
        output, residual_output = all_reduce_norm(partial, res, norm_weight, NORM_EPS)
        ok &= report(rows, "residual_out", out[1], residual_output, 2e-2)
        ok &= report(rows, "output", out[0], output, 2e-2)
        again = qwen3_5_moe_tp4(
            x, *weight_args, res, norm_weight, partials[tokens][1], group.group_name
        )
        deterministic = torch.equal(again[0], out[0])
        rows.append(
            f"  {'determinism':<22} {'bitwise' if deterministic else 'DIFFERS  <-- FAIL'}"
        )
        ok &= deterministic
        if verbose or not ok:
            say(f"rank{rank} {name}\n" + "\n".join(rows))
    if not ok:
        raise AssertionError(f"rank {rank}: qwen3_5_moe_tp4 differs from the reference")


def benchmark(group: dist.ProcessGroup, verbose: bool = True) -> dict[str, Any]:
    """Cold-weight timing: ``LAYERS`` distinct weight copies (default 8, about
    0.8 GB each; an even count, as consecutive launches alternate two
    partials) run back to back, ``REPS`` times (default 4) in one CUDA graph,
    so each launch streams its experts from HBM as consecutive layers do; the
    previous launch's all-reduce keeps the ranks in step. Microseconds per
    launch (median and minimum of 10 replays) and each layer's distinct experts
    of the window. The routing: ``ROUTER_SUPPORT`` = S (default 0: off)
    concentrates each layer's router on about S experts
    (``_setup.benchmark_weights``); ``XCORR`` = m (default 0) gives the tokens
    sqrt(m) of a common direction, so they share experts (m = 0: about 63
    distinct experts of 70 picks, 0.6: about 43, 0.7: about 36)."""
    rank = group.rank()
    device = torch.device("cuda", torch.cuda.current_device())
    layers = int(os.environ.get("LAYERS", "8"))
    assert layers % 2 == 0, "LAYERS must be even"
    reps = int(os.environ.get("REPS", "4"))
    support = int(os.environ.get("ROUTER_SUPPORT", "0"))
    xcorr = float(os.environ.get("XCORR", "0"))
    shape = _setup.Shape()
    args = []
    for i in range(layers):
        weights = _setup.benchmark_weights(i, rank, device, support, shape)
        args.append(_setup.kernel_weight_args(weights, shape))
    torch.cuda.empty_cache()
    g = torch.Generator(device=device)
    g.manual_seed(3)
    common = torch.randn(1, shape.hidden, generator=g, device=device)
    own = torch.randn(TOKENS, shape.hidden, generator=g, device=device)
    x = (xcorr**0.5 * common + (1 - xcorr) ** 0.5 * own).to(torch.bfloat16)
    residual = torch.randn(TOKENS, shape.hidden, generator=g, device=device).to(
        torch.bfloat16
    )
    norm_weight = torch.ones(shape.hidden, device=device, dtype=torch.bfloat16)
    partials = symmetric_partials(TOKENS, shape.hidden, group.group_name, device)

    def launch(i: int) -> tuple[torch.Tensor, ...]:
        # The same input every layer, which each layer's router routes its own way.
        return qwen3_5_moe_tp4(
            x, *args[i], residual, norm_weight, partials[i % 2], group.group_name
        )

    def step() -> None:
        for i in range(layers):
            launch(i)

    median, minimum = graph_time(step, layers, reps)
    distinct = [
        len(set(launch(i)[4][:TOKENS, : shape.top_k].flatten().tolist()))
        for i in range(layers)
    ]
    mean_distinct = sum(distinct) / layers
    # Bytes a launch streams on average: the router, the shared expert, and W13,
    # W2 and their scales for each distinct expert.
    E, H, N = shape.num_experts, shape.hidden, shape.intermediate
    expert_bytes = N * H + 2 * N * H // 16 + H * N // 2 + H * N // 16
    shared_bytes = 3 * shape.shared_intermediate * H
    megabytes = (mean_distinct * expert_bytes + 2 * E * H + shared_bytes) / 1e6
    if verbose:
        say(
            f"rank{rank} T={TOKENS} distinct={'/'.join(map(str, distinct))} "
            f"us/launch med {median:.2f} min {minimum:.2f} "
            f"(~{megabytes:.0f} MB -> {megabytes / median:.2f} TB/s)",
        )
    return {"latency_us": median, "min_latency_us": minimum, "distinct": mean_distinct}


def main(verbose: bool = True, check_only: bool = False) -> dict[str, Any]:
    """The four-rank check, then the cold-weight timing (``check_only``: the
    check alone)."""
    return run_main(
        "qwen3_5_moe_tp4", correctness_check, benchmark, verbose, check_only
    )


if __name__ == "__main__":
    main(check_only="--check" in sys.argv)
