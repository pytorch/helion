# ruff: noqa: ANN001, ANN201, ANN202
# pyrefly: ignore-errors
"""A Qwen3.5-397B-A17B TP4 speculative-verify layer boundary as one persistent
Gluon megakernel: a layer's NVFP4 MoE (``qwen3_5_moe_tp4``: routing, routed and
shared experts, the tensor-parallel all-reduce, the residual and the next
layer's input RMSNorm), then the next layer's Gated-DeltaNet mixer
(``qwen3_5_gdn_mixer_tp4``: in_proj to out_proj, its all-reduce, the residual
and the following RMSNorm). One launch replaces the MoE's and the mixer's: the
mixer's weights start streaming while the MoE's tail runs, and no launch
boundary sits between the MoE's norm and the mixer's in_proj. PDL as in the
standalone kernels: ``hl.pdl_wait()`` at entry, ``hl.pdl_launch_dependents()``
before the mixer's all-reduce.

The two halves are copies of the standalone kernels' roots (layouts, numerics
and the all-reduce code as documented there): the MoE's verbatim, the mixer's
with an ``mx_`` prefix on its names (a fix to one copy belongs in the other).
They differ in:

- ``hl.prefetch_code()`` before the mixer half, whose code the MoE's weight
  stream evicted from L2 (none at entry).
- the mixer's in_proj input: a quant root writes it (the standalone mixer
  quantizes in in_proj's operand path), recomputing the MoE's RMSNorm from the
  all-reduce's row partial sums and norm input, so in_proj waits on the
  all-reduce, not on the norm root (one hop less at the boundary); the norm
  root still writes the MoE's output (ba's input, and a return).
- no z root: the gated norm sums the z rows' in_proj partials and takes their
  SiLU itself (and writes qkvz's z columns).
- no 2-CTA core clusters: the core writes only its BF16 output, which the gated
  norm reloads (the standalone core also sends each head's output as FP32 to
  the gated norm's CTA over DSMEM).
- the gated norm and out_proj issue their loads in compute order (the
  standalone mixer issues them ahead of the waits).
- pointer accesses (``_POINTER_OPS``): the standalone MoE's, and in the mixer
  half (the standalone mixer has none) the quant root, ba's and in_proj's
  partial stores and the all-reduce / norm tail.
- block size 10: the MoE's combine root runs 256-column tasks, twice as many
  as the standalone MoE's 512-column ones.

The outputs are run-to-run deterministic, and on the seeded check they equal
the two standalone kernels' bit for bit (the MoE's output, residual and top-k
ids, the mixer's output and residual, and the updated mixer state). Equality on
every input is not guaranteed: two reductions add in another FP32 order. The
combine's [16 picks, 256 columns] tile gets another register layout than the
standalone's 512-column one (warps along the picks, so each thread pre-sums 4
picks, not 8, before the cross-warp step), and the gated norm's sum of squares
runs over its reloaded BF16 rows, not the FP32 DSMEM receive's layout. Either
can round a BF16 output differently.

Arguments: the MoE's, then the mixer's without its hidden_states and residual
(the MoE's outputs); mixer arguments whose names the MoE also has carry
``LAYER_PREFIX`` (``merged_args``). Returns the mixer's outputs, then the MoE's
(``LAYER_FIRST_RETURNS`` of them).

``correctness_check()`` compares every rank's outputs and state with the two
standalone kernels', bitwise, on its seeded inputs; ``main()`` runs it, then
times the merged kernel against the two standalone launches; ``--check`` runs
only the check::

    NVSHMEM_DISABLE_CUDA_VMM=1 PYTHONPATH=. \\
      python -m torch.distributed.run --standalone --nproc-per-node=4 \\
      pretuned_kernels/megakernels/qwen3_5_moe_tp4/qwen3_5_layer_tp4.py [--check]
"""

from __future__ import annotations

import operator
import os
import sys
from typing import Any

from pretuned_kernels.megakernels._qwen3_5_tp4 import FP8_MAX
from pretuned_kernels.megakernels._qwen3_5_tp4 import NORM_EPS
from pretuned_kernels.megakernels._qwen3_5_tp4 import graph_time
from pretuned_kernels.megakernels._qwen3_5_tp4 import run_main
from pretuned_kernels.megakernels._qwen3_5_tp4 import same
from pretuned_kernels.megakernels._qwen3_5_tp4 import say
from pretuned_kernels.megakernels._qwen3_5_tp4 import symmetric_partials
from pretuned_kernels.megakernels.deepseek_v3_moe_nvfp4._common import FP4_MAX
from pretuned_kernels.megakernels.deepseek_v3_moe_nvfp4._common import fp4_nibble
from pretuned_kernels.megakernels.deepseek_v3_moe_nvfp4._common import popcount
from pretuned_kernels.megakernels.deepseek_v3_moe_nvfp4._common import scale_tile
from pretuned_kernels.megakernels.qwen3_5_gdn_tp4 import _setup as ms
from pretuned_kernels.megakernels.qwen3_5_gdn_tp4._common import conv_silu
from pretuned_kernels.megakernels.qwen3_5_gdn_tp4._common import finite
from pretuned_kernels.megakernels.qwen3_5_gdn_tp4._common import proj_rows
from pretuned_kernels.megakernels.qwen3_5_gdn_tp4._common import store_conv_state
from pretuned_kernels.megakernels.qwen3_5_gdn_tp4._common import store_conv_window
from pretuned_kernels.megakernels.qwen3_5_gdn_tp4.qwen3_5_gdn_mixer_tp4 import (
    qwen3_5_gdn_mixer_tp4,
)
from pretuned_kernels.megakernels.qwen3_5_moe_tp4 import _setup
from pretuned_kernels.megakernels.qwen3_5_moe_tp4._common import expert_of
from pretuned_kernels.megakernels.qwen3_5_moe_tp4._common import route
from pretuned_kernels.megakernels.qwen3_5_moe_tp4._common import scale_tiles_16
from pretuned_kernels.megakernels.qwen3_5_moe_tp4.qwen3_5_moe_tp4 import (
    _POINTER_OPS as _MOE_POINTER_OPS,
)
from pretuned_kernels.megakernels.qwen3_5_moe_tp4.qwen3_5_moe_tp4 import qwen3_5_moe_tp4
import torch
import torch.distributed as dist

import helion
import helion.language as hl

LAYER_PREFIX = "mx_"
LAYER_FIRST_RETURNS = 11


# Config.indexing slots lowered as pointer accesses (see ``qwen3_5_moe_tp4``):
# the MoE half's are the standalone MoE's (the same roots in the same order),
# then the mixer half's.
_MEMORY_OPS = 155
_POINTER_OPS = {
    **_MOE_POINTER_OPS,
    # mixer quant: row partial sums, norm input and weight; in_proj's FP8 input
    70: "load norm_partial",
    71: "load norm_input_chunked",
    72: "load norm_weight_chunked",
    74: "store mx_ip_input",
    77: "store mx_ba_partials",  # mixer ba
    80: "store mx_qkv_partials",  # mixer in_proj q|k|v
    83: "store mx_z_partials",  # mixer in_proj z
    143: "load residual_output",  # mixer all-reduce
    148: "store mx_residual_output",
    149: "store mx_norm_partial",
    150: "store mx_norm_input",
    151: "load mx_norm_partial",  # mixer norm
    152: "load mx_norm_input",
    154: "store mx_output",
}


def _indexing():
    return [
        "pointer" if op in _POINTER_OPS else "tensor_descriptor"
        for op in range(_MEMORY_OPS)
    ]


@helion.kernel(
    static_shapes=False,
    # The core's causal products in FP32, as SGLang's verify core sums them (TF32 gives
    # 10x its error against the FP32 reference).
    dot_precision="ieee",
    backend="gluon",
    triton_do_not_specialize=False,
    # One persistent CTA per SM with a 4-deep TMA ring under dynamic dispatch; the
    # MoE half's 11 block sizes (the standalone MoE's but the combine's 256),
    # then the mixer's 6.
    config=helion.Config(
        block_sizes=[
            16,
            128,
            32,
            16,
            128,
            256,
            8,
            16,
            128,
            8,
            256,
            16,
            256,
            16,
            256,
            16,
            512,
        ],
        num_warps=4,
        pid_type="persistent_blocked",
        cross_loop_pipeline="dynamic",
        num_sm_multiplier=1,
        num_stages=4,
        range_warp_specializes=[True] * 16 + [False] * 2 + [True] * 12 + [False] * 2,
        indexing=_indexing(),
        cross_loop_inband_protocol=["lamport", "tagged"],
        # Over DSMEM, as the standalone MoE: W13's expert bitmaps (every cluster
        # routes) and a token's norm input and row partial sums.
        cross_loop_dsmem=[False] * 7
        + [True]
        + [False] * 10
        + [True, False, True]
        + [False] * 21,
    ),
    ignore_warnings=[helion.exc.TensorOperationInWrapper, helion.exc.FastMathUnused],
)
def qwen3_5_layer_tp4(
    hidden_input,
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
    top_k,
    residual,
    norm_weight,
    symmetric_partial,
    inproj_weight,
    inproj_scales,
    ba_weight,
    conv_weight,
    conv_state,
    conv_window,
    A_log,
    dt_bias,
    h0,
    d_cache,
    k_cache,
    g_cache,
    dlow_cache,
    klow_cache,
    state_index,
    window_index,
    replay_index,
    write_pos,
    cache_base,
    gate_norm_weight,
    out_proj_weight,
    out_proj_scales,
    mx_norm_weight,
    mx_symmetric_partial,
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
    assert rows <= 8
    padded_rows = 8
    slots = rows * top_k
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
    hl.specialize(inproj_weight.size())
    hl.specialize(inproj_weight.stride())
    hl.specialize(inproj_scales.size())
    hl.specialize(inproj_scales.stride())
    hl.specialize(ba_weight.size())
    hl.specialize(ba_weight.stride())
    hl.specialize(conv_weight.size())
    hl.specialize(conv_weight.stride())
    hl.specialize(conv_state.size())
    hl.specialize(conv_state.stride())
    hl.specialize(conv_window.size())
    hl.specialize(conv_window.stride())
    hl.specialize(A_log.size())
    hl.specialize(A_log.stride())
    hl.specialize(dt_bias.size())
    hl.specialize(dt_bias.stride())
    hl.specialize(h0.size())
    hl.specialize(h0.stride())
    hl.specialize(d_cache.size())
    hl.specialize(d_cache.stride())
    hl.specialize(k_cache.size())
    hl.specialize(k_cache.stride())
    hl.specialize(g_cache.size())
    hl.specialize(g_cache.stride())
    hl.specialize(dlow_cache.size())
    hl.specialize(dlow_cache.stride())
    hl.specialize(klow_cache.size())
    hl.specialize(klow_cache.stride())
    hl.specialize(state_index.size())
    hl.specialize(state_index.stride())
    hl.specialize(window_index.size())
    hl.specialize(window_index.stride())
    hl.specialize(replay_index.size())
    hl.specialize(replay_index.stride())
    hl.specialize(write_pos.size())
    hl.specialize(write_pos.stride())
    hl.specialize(cache_base.size())
    hl.specialize(cache_base.stride())
    hl.specialize(gate_norm_weight.size())
    hl.specialize(gate_norm_weight.stride())
    hl.specialize(out_proj_weight.size())
    hl.specialize(out_proj_weight.stride())
    hl.specialize(out_proj_scales.size())
    hl.specialize(out_proj_scales.stride())
    hl.specialize(mx_norm_weight.size())
    hl.specialize(mx_norm_weight.stride())
    hl.specialize(mx_symmetric_partial.size())
    hl.specialize(mx_symmetric_partial.stride())

    mx_tokens, mx_hidden_dim = output.shape
    mx_conv_dim, mx_conv_width = conv_weight.shape
    mx_state_len = conv_state.size(2)
    mx_window_cols = conv_window.size(2)
    mx_heads, mx_ring, mx_head_k = k_cache.shape[1:]
    mx_v_heads, mx_head_v = d_cache.size(1), d_cache.size(3)
    mx_group = mx_v_heads // mx_heads
    mx_key_dim = mx_heads * mx_head_k
    mx_value_dim = mx_v_heads * mx_head_v
    # Literals: a module-level int reaches the kernel as a runtime scalar.
    mx_lanes_n = 8  # window token lanes (T <= 8)
    mx_history_n = 16  # ring lanes (L <= 16)
    # The core's K in one step (2 steps of 64: +2%): the checkpoint / history keys
    # load whole rows.
    assert mx_head_k == 128
    mx_conv_block = 128
    mx_ba_slice_n = 256  # in_proj_ba's K slice (one partial each)
    # (T >= 3: the post-window conv state is the window's last 3 tokens)
    assert (
        mx_state_len <= mx_tokens <= mx_lanes_n
        and mx_ring <= mx_history_n
        and mx_ring & (mx_ring - 1) == 0
    )
    assert mx_conv_width == 4 and mx_state_len == 3 and mx_window_cols == mx_tokens + 2
    assert mx_conv_dim == 2 * mx_key_dim + mx_value_dim
    assert inproj_weight.shape == (mx_conv_dim + mx_value_dim, mx_hidden_dim)
    assert ba_weight.shape == (2 * mx_v_heads, mx_hidden_dim) and mx_head_k == mx_head_v
    mx_hidden = residual_output.size(1)
    assert out_proj_weight.shape == (mx_hidden, mx_value_dim)
    assert mx_symmetric_partial.size() == (mx_tokens, mx_hidden)
    mx_scale = mx_head_k**-0.5
    mx_dev = residual_output.device

    # Flat views: gathers index rows of 2-D views (no scalar between index tensors).
    mx_state2 = conv_state.view(conv_state.size(0) * mx_conv_dim, mx_state_len)
    mx_window2 = conv_window.view(conv_window.size(0) * mx_conv_dim, mx_window_cols)
    mx_h0_2 = h0.view(h0.size(0) * mx_v_heads * mx_head_v, mx_head_k)
    mx_kc2 = k_cache.view(k_cache.size(0) * mx_heads * mx_ring, mx_head_k)
    mx_klow2 = klow_cache.view(k_cache.size(0) * mx_heads * mx_ring, mx_head_k)
    mx_dc2 = d_cache.view(d_cache.size(0) * mx_v_heads * mx_ring, mx_head_v)
    mx_dlow2 = dlow_cache.view(d_cache.size(0) * mx_v_heads * mx_ring, mx_head_v)
    mx_g1 = g_cache.view(g_cache.size(0) * mx_v_heads * mx_ring)
    # in_proj's static FP8 input (written by the quant root).
    mx_hidden_fp8 = torch.empty(
        (mx_tokens, mx_hidden_dim), dtype=torch.float8_e4m3fn, device=mx_dev
    )

    # in_proj: in_proj_ba partials per K slice, FP32 in_proj_qkvz partials of 4 K
    # chunks.
    mx_ba_slices = 16
    assert mx_hidden_dim == mx_ba_slices * mx_ba_slice_n
    mx_ba_partials = torch.empty(
        (mx_ba_slices, mx_tokens, 2 * mx_v_heads), dtype=torch.float32, device=mx_dev
    )
    mx_ip_chunks = 4
    mx_qkvz_rows = mx_conv_dim + mx_value_dim
    mx_ip_weight = inproj_weight.view(
        mx_qkvz_rows, mx_ip_chunks, mx_hidden_dim // mx_ip_chunks
    )
    mx_ip_input = mx_hidden_fp8.view(
        mx_tokens, mx_ip_chunks, mx_hidden_dim // mx_ip_chunks
    )
    norm_input_chunked = norm_input.view(
        mx_tokens, mx_ip_chunks, mx_hidden_dim // mx_ip_chunks
    )
    norm_weight_chunked = norm_weight.view(mx_ip_chunks, mx_hidden_dim // mx_ip_chunks)
    mx_ba_partials2 = mx_ba_partials.view(mx_ba_slices * mx_tokens, 2 * mx_v_heads)
    mx_qkv_partials = torch.empty(
        (mx_ip_chunks, mx_tokens, mx_conv_dim), dtype=torch.float32, device=mx_dev
    )
    # (the z columns of the qkvz rows only)
    mx_z_partials = torch.empty(
        (mx_ip_chunks, mx_tokens, mx_qkvz_rows), dtype=torch.float32, device=mx_dev
    )

    # Checks: the in_proj and conv outputs as SGLang materializes them.
    mx_qkvz = torch.empty(
        (mx_tokens, mx_conv_dim + mx_value_dim), dtype=torch.bfloat16, device=mx_dev
    )
    mx_ba = torch.empty(
        (mx_tokens, 2 * mx_v_heads), dtype=torch.bfloat16, device=mx_dev
    )
    mx_mixed_qkv = torch.empty(
        (mx_tokens, mx_conv_dim), dtype=torch.bfloat16, device=mx_dev
    )
    mx_core_out = torch.empty(
        (mx_tokens, mx_value_dim), dtype=torch.bfloat16, device=mx_dev
    )
    mx_gated_out = torch.empty(
        (mx_tokens, mx_value_dim), dtype=torch.bfloat16, device=mx_dev
    )
    mx_attn_fp8 = torch.empty(
        (mx_tokens, mx_value_dim), dtype=torch.float8_e4m3fn, device=mx_dev
    )

    mx_remote_partials = torch.ops.symm_mem.get_remote_tensors(
        mx_symmetric_partial, group_name
    )
    mx_output = torch.empty((mx_tokens, mx_hidden), dtype=torch.bfloat16, device=mx_dev)
    mx_residual_output = torch.empty_like(mx_output)
    mx_communication_n = 512
    mx_norm_tasks = mx_hidden // mx_communication_n
    mx_norm_partial = torch.empty(
        (mx_tokens, mx_norm_tasks), dtype=torch.float32, device=mx_dev
    )
    mx_norm_input = torch.empty(
        (mx_tokens, mx_hidden), dtype=torch.float32, device=mx_dev
    )

    # Programmatic dependent launch: wait for the previous kernel at entry.
    hl.pdl_wait()

    # The MoE. Router projection: K chunks of the 512 x 4096 BF16 weight.
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

    # Shared gate / up: FP8 K chunks.
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
    # The shared expert's sigmoid gate, one token per task.
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

    # Each token's top-k rows, weighted in pick order, plus the gated shared output.
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
    # The next layer's input RMSNorm (ba's input, and a return).
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
    # The next layer's GDN mixer: its code toward L2 first (the MoE's weight stream
    # evicted it).
    hl.prefetch_code()

    # in_proj's static FP8 input: the MoE's norm recomputed from the all-reduce's
    # outputs (the norm root still writes ``output``, ba's input), so the quant
    # waits on the all-reduce, not on the norm (one hop less at the boundary).
    for mx_qt_c, mx_qt_k, mx_qt_t in hl.tile(
        [mx_ip_chunks, mx_hidden_dim // mx_ip_chunks, mx_tokens],
        block_size=[1, 128, 16],
    ):
        mx_qt_scale = torch.rsqrt(
            torch.sum(norm_partial[mx_qt_t, :], dim=-1) / hidden + NORM_EPS
        )
        mx_qt_x = (
            norm_input_chunked[mx_qt_t, mx_qt_c.begin, mx_qt_k]
            * mx_qt_scale[:, None]
            * norm_weight_chunked[mx_qt_c.begin, mx_qt_k].to(torch.float32)[None, :]
        ).to(torch.bfloat16)
        mx_ip_input[mx_qt_t, mx_qt_c.begin, mx_qt_k] = torch.clamp(
            mx_qt_x.to(torch.float32) * (1.0 / inproj_scales[0]), -FP8_MAX, FP8_MAX
        ).to(torch.float8_e4m3fn)

    # in_proj_ba's partial per K slice (summed by the core), in steps of 64
    # columns (4 KB weight boxes on the ring).
    for mx_bp_k, mx_bp_t in hl.tile(
        [mx_hidden_dim, mx_tokens], block_size=[mx_ba_slice_n, 16]
    ):
        mx_bp_acc = hl.zeros([mx_bp_t, 2 * mx_v_heads], dtype=torch.float32)
        for mx_bp_kk in hl.tile(mx_bp_k.begin, mx_bp_k.end, block_size=64):
            mx_bp_acc = hl.dot(
                output[mx_bp_t, mx_bp_kk], ba_weight[:, mx_bp_kk].T, acc=mx_bp_acc
            )
        mx_ba_partials[mx_bp_k.begin // mx_ba_slice_n, mx_bp_t, :] = mx_bp_acc

    # in_proj_qkvz, q|k|v rows: FP8 K chunks of 128 rows.
    for mx_iq_n, mx_iq_c, mx_iq_t in hl.tile(
        [mx_conv_dim, mx_ip_chunks, mx_tokens], block_size=[128, 1, None]
    ):
        mx_iq_acc = hl.zeros([mx_iq_n, mx_iq_t], dtype=torch.float32)
        for mx_iq_k in hl.tile(mx_hidden_dim // mx_ip_chunks):
            mx_iq_acc = hl.dot(
                mx_ip_weight[mx_iq_n, mx_iq_c.begin, mx_iq_k],
                mx_ip_input[mx_iq_t, mx_iq_c.begin, mx_iq_k].T,
                acc=mx_iq_acc,
            )
        mx_qkv_partials[mx_iq_c.begin, mx_iq_t, mx_iq_n] = mx_iq_acc.T

    # in_proj_qkvz, z rows (read by the gated norm only), the same way.
    for mx_iz_n, mx_iz_c, mx_iz_t in hl.tile(
        [mx_conv_dim, 0, 0],
        [mx_qkvz_rows, mx_ip_chunks, mx_tokens],
        block_size=[128, 1, None],
    ):
        mx_iz_acc = hl.zeros([mx_iz_n, mx_iz_t], dtype=torch.float32)
        for mx_iz_k in hl.tile(mx_hidden_dim // mx_ip_chunks):
            mx_iz_acc = hl.dot(
                mx_ip_weight[mx_iz_n, mx_iz_c.begin, mx_iz_k],
                mx_ip_input[mx_iz_t, mx_iz_c.begin, mx_iz_k].T,
                acc=mx_iz_acc,
            )
        mx_z_partials[mx_iz_c.begin, mx_iz_t, mx_iz_n] = mx_iz_acc.T

    # The in_proj rows (the K chunks' partials in chunk order, BF16) and the causal
    # conv, once per channel block: the conv output, the verify window rows and
    # the new conv state of the block's channels.
    for mx_cv_c in hl.tile(mx_conv_dim, block_size=mx_conv_block):
        mx_cv_lanes = hl.arange(mx_lanes_n)
        mx_cv_live = mx_cv_lanes < mx_tokens
        mx_cv_lane = torch.where(mx_cv_live, mx_cv_lanes, 0)
        mx_cv_channels = mx_cv_c.index
        mx_cv_row0 = state_index[0] * mx_conv_dim
        mx_cv_x = proj_rows(
            mx_qkv_partials,
            mx_ip_chunks,
            inproj_scales[1],
            mx_cv_lanes,
            mx_cv_live,
            mx_cv_channels,
        )
        mx_cv_out = conv_silu(
            mx_cv_x,
            mx_state2,
            conv_weight,
            mx_cv_row0,
            mx_cv_channels,
            mx_cv_lanes,
            mx_cv_live,
        )
        hl.store(
            mx_mixed_qkv,
            [mx_cv_lane[:, None], mx_cv_channels[None, :]],
            mx_cv_out.to(torch.bfloat16),
            extra_mask=mx_cv_live[:, None],
        )
        hl.store(
            mx_qkvz,
            [mx_cv_lane[:, None], mx_cv_channels[None, :]],
            mx_cv_x.to(torch.bfloat16),
            extra_mask=mx_cv_live[:, None],
        )
        store_conv_window(
            mx_cv_x,
            mx_state2,
            mx_window2,
            mx_cv_row0,
            window_index[0] * mx_conv_dim,
            mx_cv_channels,
            mx_cv_lanes,
            mx_cv_live,
        )
        store_conv_state(
            mx_cv_x,
            mx_state2,
            mx_cv_row0,
            mx_cv_channels,
            mx_cv_lanes,
            mx_cv_live,
            mx_tokens,
            mx_state_len,
        )

    # ReplaySSM circular verify core, one task per (value head = key
    # head h, group member g; 64 value columns): half a head per task is ~1.2 us
    # faster than whole heads with the gated norm in the task epilogue.
    for mx_c_h, mx_c_g, mx_v_tile in hl.tile(
        [mx_heads, mx_group, mx_head_v], block_size=[1, 1, 64]
    ):
        mx_h = mx_c_h.begin
        mx_hv = mx_h * mx_group + mx_c_g.begin
        mx_lanes = hl.arange(mx_lanes_n)
        mx_live = mx_lanes < mx_tokens
        mx_lane = torch.where(mx_live, mx_lanes, 0)
        mx_slot = state_index[0]
        mx_rslot = replay_index[0]
        # This task commits its key head's ring keys.
        mx_owns_qk = (mx_c_g.begin == 0) & (mx_v_tile.begin == 0)
        mx_kcols = hl.arange(mx_head_k)
        mx_q_channels = mx_h * mx_head_k + mx_kcols
        mx_k_channels = mx_key_dim + mx_q_channels
        mx_v_channels = 2 * mx_key_dim + mx_hv * mx_head_v + mx_v_tile.index
        mx_wp = write_pos[mx_rslot]
        mx_base = cache_base[mx_rslot]
        mx_hist = hl.arange(mx_history_n)
        mx_hist_live = mx_hist < mx_wp
        mx_phys_c = (mx_base + mx_hist) & (mx_ring - 1)
        mx_phys_s = (mx_base + mx_wp + mx_lanes) & (mx_ring - 1)
        mx_spec_store = mx_live & ((mx_wp + mx_lanes) < mx_ring)

        # The committed history's log-decays and update vectors (no producer in
        # this launch), issued first.
        mx_g_hist = torch.where(
            mx_hist_live,
            hl.load(
                mx_g1,
                [(mx_rslot * mx_v_heads + mx_hv) * mx_ring + mx_phys_c],
                extra_mask=mx_hist_live,
            ),
            0.0,
        )
        mx_d_hist = (
            hl.load(
                mx_dc2,
                [
                    ((mx_rslot * mx_v_heads + mx_hv) * mx_ring + mx_phys_c)[:, None],
                    mx_v_tile.index[None, :],
                ],
                extra_mask=mx_hist_live[:, None],
            )
            .to(torch.float32)
            .T
        )

        # The checkpoint rows and the committed history keys (no producer in this
        # launch; one K step covers the whole head).
        mx_h0_all = hl.load(
            mx_h0_2,
            [
                ((mx_slot * mx_v_heads + mx_hv) * mx_head_v + mx_v_tile.index)[:, None],
                mx_kcols[None, :],
            ],
        )
        mx_k_hist_all = hl.load(
            mx_kc2,
            [
                ((mx_rslot * mx_heads + mx_h) * mx_ring + mx_phys_c)[:, None],
                mx_kcols[None, :],
            ],
            extra_mask=mx_hist_live[:, None],
        )

        # The window's q / k / v rows (the conv's output), issued with the gates' loads.
        mx_q_full = hl.load(
            mx_mixed_qkv,
            [mx_lane[:, None], mx_q_channels[None, :]],
            extra_mask=mx_live[:, None],
        ).to(torch.float32)
        mx_k_full = hl.load(
            mx_mixed_qkv,
            [mx_lane[:, None], mx_k_channels[None, :]],
            extra_mask=mx_live[:, None],
        ).to(torch.float32)
        mx_v_s = (
            hl.load(
                mx_mixed_qkv,
                [mx_lane[:, None], mx_v_channels[None, :]],
                extra_mask=mx_live[:, None],
            )
            .to(torch.float32)
            .T
        )
        # Gates (in_proj_ba: its K slices' partials summed, BF16), beta, the window's
        # cumulative decay.
        mx_ba_rows = (hl.arange(mx_ba_slices) * mx_tokens)[:, None] + mx_lane[None, :]
        mx_b_s = torch.sum(
            hl.load(mx_ba_partials2, [mx_ba_rows, mx_hv], extra_mask=mx_live[None, :]),
            dim=0,
        ).to(torch.bfloat16)
        mx_a_s = torch.sum(
            hl.load(
                mx_ba_partials2,
                [mx_ba_rows, mx_v_heads + mx_hv],
                extra_mask=mx_live[None, :],
            ),
            dim=0,
        ).to(torch.bfloat16)
        mx_b_s = mx_b_s.to(torch.float32)
        mx_a_s = mx_a_s.to(torch.float32)
        mx_x = mx_a_s + dt_bias[mx_hv].to(torch.float32)
        mx_softplus = torch.where(mx_x <= 20.0, torch.log(1.0 + torch.exp(mx_x)), mx_x)
        mx_g_s = torch.where(mx_live, -torch.exp(A_log[mx_hv]) * mx_softplus, 0.0)
        mx_beta_s = torch.where(mx_live, 1.0 / (1.0 + torch.exp(-mx_b_s)), 0.0)
        mx_cum_g = torch.cumsum(mx_g_s, dim=0)
        mx_exp_g = torch.exp(mx_cum_g)

        mx_q_rnorm = torch.where(
            mx_live,
            1.0 / torch.sqrt(torch.sum(mx_q_full * mx_q_full, dim=1) + 1e-6),
            0.0,
        )
        mx_k_rnorm = torch.where(
            mx_live,
            1.0 / torch.sqrt(torch.sum(mx_k_full * mx_k_full, dim=1) + 1e-6),
            0.0,
        )

        mx_hw_q = hl.zeros([mx_v_tile, mx_lanes_n], dtype=torch.float32)
        mx_hw_k = hl.zeros([mx_v_tile, mx_lanes_n], dtype=torch.float32)
        mx_scores_q = hl.zeros([mx_history_n, mx_lanes_n], dtype=torch.float32)
        mx_scores_k = hl.zeros([mx_history_n, mx_lanes_n], dtype=torch.float32)
        mx_kk_mat = hl.zeros([mx_lanes_n, mx_lanes_n], dtype=torch.float32)
        mx_kq_mat = hl.zeros([mx_lanes_n, mx_lanes_n], dtype=torch.float32)
        # One K step covers the whole head: the products use the q / k rows loaded
        # for the norms.
        mx_q_t = (mx_q_full * (mx_q_rnorm * mx_scale)[:, None]).to(torch.bfloat16)
        mx_k_prec = mx_k_full * mx_k_rnorm[:, None]
        mx_k_t = mx_k_prec.to(torch.bfloat16)
        mx_k_r = (mx_k_prec - mx_k_t.to(torch.float32)).to(torch.bfloat16)
        # The key products on the raw BF16 key rows (exact operands), scaled by the
        # rows' inverse norms after (one product each instead of the normalized key's
        # high and low parts); the ring still stores the normalized key's parts.
        mx_k_raw = mx_k_full.to(torch.bfloat16)
        mx_kk_mat = hl.dot(mx_k_raw, mx_k_raw.T, acc=mx_kk_mat) * (
            mx_k_rnorm[:, None] * mx_k_rnorm[None, :]
        )
        mx_kq_mat = hl.dot(mx_k_raw, mx_q_t.T, acc=mx_kq_mat) * mx_k_rnorm[:, None]
        # Checkpoint projections (the non-window contribution).
        mx_hw_q = hl.dot(mx_h0_all, mx_q_t.T, acc=mx_hw_q)
        mx_hw_k = hl.dot(mx_h0_all, mx_k_raw.T, acc=mx_hw_k) * mx_k_rnorm[None, :]
        # Committed history keys.
        mx_scores_q = hl.dot(mx_k_hist_all, mx_q_t.T, acc=mx_scores_q)
        mx_scores_k = hl.dot(mx_k_hist_all, mx_k_t.T, acc=mx_scores_k)
        if mx_owns_qk:
            hl.store(
                mx_kc2,
                [
                    ((mx_rslot * mx_heads + mx_h) * mx_ring + mx_phys_s)[:, None],
                    mx_kcols[None, :],
                ],
                mx_k_t,
                extra_mask=mx_spec_store[:, None],
            )
            hl.store(
                mx_klow2,
                [
                    ((mx_rslot * mx_heads + mx_h) * mx_ring + mx_phys_s)[:, None],
                    mx_kcols[None, :],
                ],
                mx_k_r,
                extra_mask=mx_spec_store[:, None],
            )

        # History replay: decay the committed d records to the window start.
        mx_g_prefix = torch.cumsum(mx_g_hist, dim=0)
        mx_g_total = torch.sum(mx_g_hist, dim=0)
        mx_replay_decay = torch.where(
            mx_hist_live, torch.exp(mx_g_total - mx_g_prefix), 0.0
        )
        mx_d_scaled = (mx_d_hist * mx_replay_decay[None, :]).to(torch.bfloat16)
        mx_total_decay = torch.exp(mx_g_total)
        mx_hw_q = mx_total_decay * mx_hw_q + hl.dot(
            mx_d_scaled, mx_scores_q.to(torch.bfloat16)
        )
        mx_hw_k = mx_total_decay * mx_hw_k + hl.dot(
            mx_d_scaled, mx_scores_k.to(torch.bfloat16)
        )

        # T^T = ((I + A)^-1)^T for the strictly lower A (A^8 = 0): the sum of
        # N^k with N = -A^T, as (I + N)(I + N^2)(I + N^4); A^T built directly (the
        # key products are symmetric).
        mx_upper = (
            (mx_lanes[:, None] < mx_lanes[None, :])
            & mx_live[:, None]
            & mx_live[None, :]
        )
        mx_n1 = finite(
            torch.where(
                mx_upper,
                -(
                    mx_beta_s[None, :]
                    * torch.exp(mx_cum_g[None, :] - mx_cum_g[:, None])
                    * mx_kk_mat
                ),
                0.0,
            )
        )
        mx_eye = (mx_lanes[:, None] == mx_lanes[None, :]).to(torch.float32)
        mx_n2 = hl.dot(mx_n1, mx_n1)
        mx_n4 = hl.dot(mx_n2, mx_n2)
        mx_tt = hl.dot(hl.dot(mx_eye + mx_n1, mx_eye + mx_n2), mx_eye + mx_n4)

        # R and D = R T^T (causal).
        mx_r_mat = finite(mx_beta_s[None, :] * (mx_v_s - mx_exp_g[None, :] * mx_hw_k))
        mx_d_spec = hl.dot(mx_r_mat, mx_tt)

        # Outputs.
        mx_causal = (
            (mx_lanes[:, None] <= mx_lanes[None, :])
            & mx_live[:, None]
            & mx_live[None, :]
        )
        mx_f_mat = torch.where(
            mx_causal, torch.exp(mx_cum_g[None, :] - mx_cum_g[:, None]) * mx_kq_mat, 0.0
        )
        mx_o_tile = mx_exp_g[None, :] * mx_hw_q + hl.dot(mx_d_spec, mx_f_mat)
        mx_out_cols = (mx_hv * mx_head_v + mx_v_tile.index)[None, :]
        mx_o_bf = mx_o_tile.T.to(torch.bfloat16)
        hl.store(
            mx_core_out,
            [mx_lane[:, None], mx_out_cols],
            mx_o_bf,
            extra_mask=mx_live[:, None],
        )

        # This window's records at the circular spec positions.
        mx_d_hi = mx_d_spec.to(torch.bfloat16)
        hl.store(
            mx_dc2,
            [
                ((mx_rslot * mx_v_heads + mx_hv) * mx_ring + mx_phys_s)[:, None],
                mx_v_tile.index[None, :],
            ],
            mx_d_hi.T,
            extra_mask=mx_spec_store[:, None],
        )
        hl.store(
            mx_dlow2,
            [
                ((mx_rslot * mx_v_heads + mx_hv) * mx_ring + mx_phys_s)[:, None],
                mx_v_tile.index[None, :],
            ],
            (mx_d_spec - mx_d_hi.to(torch.float32)).to(torch.bfloat16).T,
            extra_mask=mx_spec_store[:, None],
        )
        if mx_v_tile.begin == 0:
            hl.store(
                mx_ba, [mx_lane, mx_hv], mx_b_s.to(torch.bfloat16), extra_mask=mx_live
            )
            hl.store(
                mx_ba,
                [mx_lane, mx_v_heads + mx_hv],
                mx_a_s.to(torch.bfloat16),
                extra_mask=mx_live,
            )
            hl.store(
                mx_g1,
                [(mx_rslot * mx_v_heads + mx_hv) * mx_ring + mx_phys_s],
                mx_g_s,
                extra_mask=mx_spec_store,
            )

    # Gated RMSNorm over each value head (norm_before_gate; the z rows' in_proj
    # output, BF16, and its SiLU gate summed here and written to qkvz's z columns),
    # BF16 out, then out_proj's static FP8 input quant.
    for mx_gn_h, mx_gn_g in hl.tile([mx_heads, mx_group], block_size=[1, 1]):
        mx_gn_hv = mx_gn_h.begin * mx_group + mx_gn_g.begin
        mx_gn_lanes = hl.arange(mx_lanes_n)
        mx_gn_live = mx_gn_lanes < mx_tokens
        mx_gn_lane = torch.where(mx_gn_live, mx_gn_lanes, 0)
        mx_gn_channels = mx_gn_hv * mx_head_v + hl.arange(mx_head_v)
        mx_gn_cols = mx_gn_channels[None, :]
        # The core's BF16 output, reloaded.
        mx_o_f = hl.load(
            mx_core_out,
            [mx_gn_lane[:, None], mx_gn_cols],
            extra_mask=mx_gn_live[:, None],
        ).to(torch.float32)
        mx_rstd = torch.rsqrt(torch.sum(mx_o_f * mx_o_f, dim=1) / mx_head_v + NORM_EPS)
        mx_z_x = proj_rows(
            mx_z_partials,
            mx_ip_chunks,
            inproj_scales[1],
            mx_gn_lanes,
            mx_gn_live,
            mx_conv_dim + mx_gn_channels,
        )
        mx_gated = (
            mx_o_f * mx_rstd[:, None] * gate_norm_weight[:].to(torch.float32)[None, :]
        ) * (mx_z_x * torch.sigmoid(mx_z_x))
        mx_gated_bf = mx_gated.to(torch.bfloat16)
        hl.store(
            mx_gated_out,
            [mx_gn_lane[:, None], mx_gn_cols],
            mx_gated_bf,
            extra_mask=mx_gn_live[:, None],
        )
        hl.store(
            mx_attn_fp8,
            [mx_gn_lane[:, None], mx_gn_cols],
            torch.clamp(
                mx_gated_bf.to(torch.float32) * (1.0 / out_proj_scales[0]),
                -FP8_MAX,
                FP8_MAX,
            ).to(torch.float8_e4m3fn),
            extra_mask=mx_gn_live[:, None],
        )
        hl.store(
            mx_qkvz,
            [mx_gn_lane[:, None], (mx_conv_dim + mx_gn_channels)[None, :]],
            mx_z_x.to(torch.bfloat16),
            extra_mask=mx_gn_live[:, None],
        )

    # out_proj over the whole K in one task of 64 output rows, its BF16 result
    # straight into the symmetric partial.
    for mx_op_tile_n, mx_op_tile_t in hl.tile(
        [mx_hidden, mx_tokens], block_size=[64, None]
    ):
        mx_op_acc = hl.zeros([mx_op_tile_n, mx_op_tile_t], dtype=torch.float32)
        for mx_op_tile_k in hl.tile(mx_value_dim):
            mx_op_acc = hl.dot(
                out_proj_weight[mx_op_tile_n, mx_op_tile_k],
                mx_attn_fp8[mx_op_tile_t, mx_op_tile_k].T,
                acc=mx_op_acc,
            )
        mx_symmetric_partial[mx_op_tile_t, mx_op_tile_n] = (
            mx_op_acc.T * out_proj_scales[1]
        ).to(torch.bfloat16)

    hl.pdl_launch_dependents()
    # Each token's sum over ranks (in rank order), then the residual.
    for mx_comm_tile_t, mx_comm_tile_n in hl.tile(
        [mx_tokens, mx_hidden], block_size=[1, mx_communication_n]
    ):
        mx_comm_token = mx_comm_tile_t.begin
        mx_comm_residual = residual_output[mx_comm_token, mx_comm_tile_n].to(
            torch.float32
        )
        mx_comm_total = hl.zeros([mx_comm_tile_n], dtype=torch.float32)
        for mx_remote_partial in mx_remote_partials:
            mx_comm_total = mx_comm_total + mx_remote_partial[
                mx_comm_token, mx_comm_tile_n
            ].to(torch.float32)
        mx_comm_sum = mx_comm_total + mx_comm_residual
        mx_residual_output[mx_comm_token, mx_comm_tile_n] = mx_comm_sum.to(
            torch.bfloat16
        )
        mx_comm_sum = mx_comm_sum.to(torch.bfloat16).to(torch.float32)
        mx_norm_partial[mx_comm_token, mx_comm_tile_n.begin // mx_communication_n] = (
            torch.sum(mx_comm_sum * mx_comm_sum)
        )
        mx_norm_input[mx_comm_token, mx_comm_tile_n] = mx_comm_sum
    for mx_norm_tile_t, mx_norm_tile_n in hl.tile(
        [mx_tokens, mx_hidden], block_size=[1, mx_communication_n]
    ):
        mx_norm_token = mx_norm_tile_t.begin
        mx_norm_scale = torch.rsqrt(
            torch.sum(mx_norm_partial[mx_norm_token, :]) / mx_hidden + NORM_EPS
        )
        mx_output[mx_norm_token, mx_norm_tile_n] = (
            mx_norm_input[mx_norm_token, mx_norm_tile_n]
            * mx_norm_scale
            * mx_norm_weight[mx_norm_tile_n].to(torch.float32)
        ).to(torch.bfloat16)
    return (
        mx_output,
        mx_residual_output,
        mx_core_out,
        mx_gated_out,
        mx_mixed_qkv,
        mx_attn_fp8,
        mx_qkvz,
        mx_ba,
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


def merged_args(moe_args: dict[str, object], mixer_args: dict[str, object]) -> list:
    """The merged kernel's positional arguments from the standalone MoE's and
    mixer's by name: the mixer's hidden_states and residual are dropped (the
    MoE's outputs feed them) and its names that the MoE also has carry
    ``LAYER_PREFIX``."""
    args = dict(moe_args)
    for name, value in mixer_args.items():
        if name in ("hidden_states", "residual"):
            continue
        if name in moe_args and name != "group_name":
            name = LAYER_PREFIX + name
        args[name] = value
    return [args[name] for name in qwen3_5_layer_tp4.signature.parameters]


class _Layers:
    """``count`` distinct layer boundaries on this rank (MoE weights, mixer
    weights and state) and their launches: the two standalone kernels back to
    back, or the merged kernel. The MoE's input and residual and the
    TP-replicated weights (router, shared expert gate, norm weights) are the
    same on every rank, so every rank routes alike; ``router_support`` and
    ``xcorr`` shape the routing as ``qwen3_5_moe_tp4.benchmark``'s
    ``ROUTER_SUPPORT`` and ``XCORR`` do."""

    def __init__(
        self,
        group: dist.ProcessGroup,
        count: int,
        router_support: int = 0,
        xcorr: float = 0.0,
    ) -> None:
        rank = group.rank()
        device = torch.device("cuda", torch.cuda.current_device())
        self.group_name = group.group_name
        self.shape = _setup.Shape()
        hidden = self.shape.hidden
        self.mixer_layers, self.states, self.moe_weights = [], [], []
        for i in range(count):
            layer = ms.make_layer(1000 * i + rank, device)
            g = torch.Generator(device=device)
            g.manual_seed(1000 * i + 555)
            layer["norm_weight"] = (
                1 + 0.1 * torch.randn(hidden, generator=g, device=device)
            ).to(torch.bfloat16)
            self.mixer_layers.append(layer)
            self.states.append(ms.make_state(2000 * i + rank, 2, 5, device))
            weights = _setup.benchmark_weights(
                i, rank, device, router_support, self.shape
            )
            self.moe_weights.append(_setup.kernel_weight_args(weights, self.shape))
        torch.cuda.empty_cache()
        self.x = ms.make_inputs(4242, device)  # replicated across the ranks
        if xcorr:
            g = torch.Generator(device=device)
            g.manual_seed(4343)
            common = torch.randn(1, hidden, generator=g, device=device)
            own = torch.randn(ms.T, hidden, generator=g, device=device)
            self.x["hidden"] = (xcorr**0.5 * common + (1 - xcorr) ** 0.5 * own).to(
                torch.bfloat16
            )
        g = torch.Generator(device=device)
        g.manual_seed(5)
        self.moe_norm_weight = (
            1 + 0.1 * torch.randn(hidden, generator=g, device=device)
        ).to(torch.bfloat16)
        self.mixer_partials = symmetric_partials(ms.T, hidden, self.group_name, device)
        self.moe_partials = symmetric_partials(ms.T, hidden, self.group_name, device)

    def _mixer_args(self, i, hidden=None, residual=None) -> dict[str, object]:
        values = ms.kernel_args(
            self.mixer_layers[i],
            self.states[i],
            self.x,
            self.mixer_partials[i % 2],
            self.group_name,
        )
        args = dict(
            zip(qwen3_5_gdn_mixer_tp4.signature.parameters, values, strict=True)
        )
        if hidden is not None:
            args["hidden_states"], args["residual"] = hidden, residual
        return args

    def _moe_args(self, i) -> dict[str, object]:
        values = (
            self.x["hidden"],
            *self.moe_weights[i],
            self.x["residual"],
            self.moe_norm_weight,
            self.moe_partials[i % 2],
            self.group_name,
        )
        return dict(zip(qwen3_5_moe_tp4.signature.parameters, values, strict=True))

    def separate(self, i: int) -> tuple[tuple, tuple]:
        """MoE(i) on the layer input, then mixer(i) on the MoE's output."""
        moe = qwen3_5_moe_tp4(*self._moe_args(i).values())
        mixer = qwen3_5_gdn_mixer_tp4(*self._mixer_args(i, moe[0], moe[1]).values())
        return moe, mixer

    def merged(self, i: int) -> tuple[tuple, tuple]:
        """The merged kernel on the same inputs: (MoE outputs, mixer outputs)."""
        out = qwen3_5_layer_tp4(*merged_args(self._moe_args(i), self._mixer_args(i)))
        k = len(out) - LAYER_FIRST_RETURNS
        return out[k:], out[:k]

    def distinct(self, moe: tuple) -> int:
        return len(set(moe[4][: ms.T, : self.shape.top_k].flatten().tolist()))


def correctness_check(group: dist.ProcessGroup, verbose: bool = True) -> None:
    """One layer boundary, from the same mixer state for each path: the merged
    kernel's MoE output, residual and top-k ids, its mixer output and residual,
    and the updated mixer state must equal the standalone kernels' bitwise (on
    these seeded inputs: the module docstring has the two sums that may differ on
    others)."""
    rank = group.rank()
    layers = _Layers(group, 1)
    state = layers.states[0]
    initial = {k: v.clone() for k, v in state.items()}
    moe_s, mixer_s = layers.separate(0)
    torch.cuda.synchronize()
    state_s = {k: v.clone() for k, v in state.items()}
    for k, v in state.items():
        v.copy_(initial[k])
    dist.barrier()
    moe_m, mixer_m = layers.merged(0)
    torch.cuda.synchronize()
    assert len(moe_m) == len(moe_s) and len(mixer_m) == len(mixer_s)
    checks = [
        ("moe output", moe_m[0], moe_s[0]),
        ("moe residual_out", moe_m[1], moe_s[1]),
        ("moe topk_ids", moe_m[4], moe_s[4]),
        ("mixer output", mixer_m[0], mixer_s[0]),
        ("mixer residual_out", mixer_m[1], mixer_s[1]),
    ]
    checks += [
        (f"state {k}", state[k], state_s[k])
        for k in state_s
        if state[k].dtype != torch.int32
    ]
    rows, ok = [], True
    for label, merged, separate in checks:
        exact = same(merged, separate)
        err = (merged.float() - separate.float()).abs().max().item()
        rows.append(
            f"  {label:<22} {'bitwise' if exact else f'DIFF max abs {err:.3g}  <-- FAIL'}"
        )
        ok &= exact
    if verbose or not ok:
        say(
            f"rank{rank} distinct={layers.distinct(moe_s)}\n" + "\n".join(rows),
        )
    if not ok:
        raise AssertionError(
            f"rank {rank}: qwen3_5_layer_tp4 differs from the standalone kernels"
        )


def benchmark(group: dist.ProcessGroup, verbose: bool = True) -> dict[str, Any]:
    """``LAYERS`` distinct layer boundaries (default 8: MoE weights, mixer
    weights and state; an even count, as consecutive launches alternate two
    partials), ``REPS`` steps (default 4) in one CUDA graph:
    microseconds per layer (median and minimum of 10 replays) of the merged
    kernel (one launch) and of the two standalone kernels (two launches).
    ``ROUTER_SUPPORT`` and ``XCORR`` shape the routing as in
    ``qwen3_5_moe_tp4.benchmark``."""
    rank = group.rank()
    count = int(os.environ.get("LAYERS", "8"))
    assert count % 2 == 0, "LAYERS must be even"
    reps = int(os.environ.get("REPS", "4"))
    layers = _Layers(
        group,
        count,
        router_support=int(os.environ.get("ROUTER_SUPPORT", "0")),
        xcorr=float(os.environ.get("XCORR", "0")),
    )

    def steps(launch):
        return lambda: [launch(i) for i in range(count)]

    t_separate = graph_time(steps(layers.separate), count, reps)
    t_merged = graph_time(steps(layers.merged), count, reps)
    distinct = "/".join(str(layers.distinct(layers.merged(i)[0])) for i in range(count))
    if verbose:
        say(
            f"rank{rank} layers={count} distinct={distinct} us/layer: separate med "
            f"{t_separate[0]:.2f} min {t_separate[1]:.2f} | merged med "
            f"{t_merged[0]:.2f} min {t_merged[1]:.2f} "
            f"({100 * (t_merged[0] / t_separate[0] - 1):+.1f}%)",
        )
    return {"latency_us": t_merged[0], "separate_latency_us": t_separate[0]}


def main(verbose: bool = True, check_only: bool = False) -> dict[str, Any]:
    """The four-rank check, then the timing against the standalone kernels
    (``check_only``: the check alone)."""
    return run_main(
        "qwen3_5_layer_tp4", correctness_check, benchmark, verbose, check_only
    )


if __name__ == "__main__":
    main(check_only="--check" in sys.argv)
