# ruff: noqa: ANN001, ANN201, ANN202
# pyrefly: ignore-errors
"""A Qwen3.5-397B-A17B TP4 speculative-verify layer boundary as one persistent
Gluon megakernel: a layer's NVFP4 MoE (``qwen3_5_moe_tp4``: routing, routed and
shared experts, the tensor-parallel all-reduce, the residual and the next
layer's input RMSNorm), then the next layer's Gated-DeltaNet mixer
(``qwen3_5_gdn_mixer_tp4``: in_proj to out_proj, its all-reduce, the residual
and the following RMSNorm). One launch replaces the MoE's and the mixer's: the
mixer's weights start streaming while the MoE's tail runs, and no launch
boundary sits between the MoE's norm and the mixer's in_proj.

The two halves are the standalone kernels' roots (layouts, numerics and the
all-reduce code as documented there), with these differences:

- dedup: the last top-k pair of an expert serves it (the standalone MoE: the
  first; both test the window's packed ids). The first pairs' tasks land on the
  low lanes, the workers that also run the router, the input quant and the
  shared down projection.
- the mixer's in_proj input: its quant root recomputes the MoE's RMSNorm from
  the all-reduce's row partial sums and norm input, so in_proj waits on the
  all-reduce, not on the norm root (one hop less at the boundary); the norm
  root still writes the MoE's output (ba's input, and a return).
- ``hl.prefetch_code()`` before the mixer half, whose code the MoE's weight
  stream evicted from L2 (none at entry).
- pointer accesses (``_POINTER_OPS``): the standalone MoE's and the routing's
  partials load; in the mixer half (the standalone mixer has none) the quant
  root, the in_proj partial stores and the all-reduce / norm tail.

The outputs equal the two standalone kernels' bit for bit (``layer_bench
--check``): the MoE's output, residual and top-k ids, the mixer's output and
residual, and the updated mixer state.

Arguments: the MoE's, then the mixer's without its hidden_states and residual
(the MoE's outputs); mixer arguments whose names the MoE also has carry
``LAYER_PREFIX``. Returns the mixer's outputs, then the MoE's
(``LAYER_FIRST_RETURNS`` of them).
"""

from __future__ import annotations

from pretuned_kernels.megakernels.qwen3_5_gdn_tp4._common import conv_silu
from pretuned_kernels.megakernels.qwen3_5_gdn_tp4._common import proj_rows
from pretuned_kernels.megakernels.qwen3_5_gdn_tp4._common import store_conv_state
from pretuned_kernels.megakernels.qwen3_5_gdn_tp4._common import store_conv_window
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

# The separate kernels (layer_bench / layer_compile): the MoE in this package, the
# mixer in qwen3_5_gdn_tp4.
LAYER_MOE = "qwen3_5_moe_tp4"
LAYER_MIXER = "qwen3_5_gdn_mixer_tp4"
LAYER_PREFIX = "mx_"
LAYER_FIRST_RETURNS = 11


def imax(a, b):
    """int32 max: a reduction with this one combiner lowers to redux.sync across a warp."""
    return torch.maximum(a, b)


# Config.indexing slots (one per load / store, in the order Helion visits them: a
# loop's ops at the loop's place in its root) lowered as pointer accesses; every
# other memory op is a tensor descriptor (TMA ring / bulk store).
_MEMORY_OPS = 157
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
    72: "mixer quant: row partial sums",
    73: "mixer quant: norm input",
    74: "mixer quant: norm weight",
    76: "mixer quant: in_proj's FP8 input",
    79: "mixer ba: partials",
    82: "mixer in_proj q|k|v: partials",
    85: "mixer in_proj z: partials",
    145: "mixer all-reduce: residual",
    150: "mixer all-reduce: residual output",
    151: "mixer all-reduce: row partial sums",
    152: "mixer all-reduce: norm input",
    153: "mixer norm: row partial sums",
    154: "mixer norm: input",
    156: "mixer norm: output",
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
    # The mixer core's small FP32 products exact (FMA dots).
    dot_precision="ieee",
    # One persistent CTA per SM with a 4-deep TMA ring under dynamic dispatch; the
    # MoE's 11 block sizes, then the mixer's 6.
    config=helion.Config(
        block_sizes=[
            16,
            128,
            32,
            16,
            128,
            256,
            16,
            16,
            128,
            16,
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
        range_warp_specializes=[True] * 16 + [False] * 3 + [True] * 13 + [False],
        indexing=_indexing(),
    ),
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
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
    mx_v_heads, mx_head_v = (d_cache.size(1), d_cache.size(3))
    mx_group = mx_v_heads // mx_heads
    mx_key_dim = mx_heads * mx_head_k
    mx_value_dim = mx_v_heads * mx_head_v
    mx_lanes_n = 8
    mx_history_n = 16
    mx_k_chunk = 128
    assert mx_head_k == mx_k_chunk
    mx_conv_block = 128
    mx_quant_n = 256
    assert (
        mx_tokens <= mx_lanes_n
        and mx_ring <= mx_history_n
        and (mx_ring & mx_ring - 1 == 0)
    )
    assert (
        mx_conv_width == 4 and mx_state_len == 3 and (mx_window_cols == mx_tokens + 2)
    )
    assert mx_conv_dim == 2 * mx_key_dim + mx_value_dim
    assert inproj_weight.shape == (mx_conv_dim + mx_value_dim, mx_hidden_dim)
    assert ba_weight.shape == (2 * mx_v_heads, mx_hidden_dim) and mx_head_k == mx_head_v
    mx_hidden = residual_output.size(1)
    assert out_proj_weight.shape == (mx_hidden, mx_value_dim)
    assert mx_symmetric_partial.size() == (mx_tokens, mx_hidden)
    mx_scale = mx_head_k ** (-0.5)
    mx_dev = residual_output.device
    mx_state2 = conv_state.view(conv_state.size(0) * mx_conv_dim, mx_state_len)
    mx_window2 = conv_window.view(conv_window.size(0) * mx_conv_dim, mx_window_cols)
    mx_h0_2 = h0.view(h0.size(0) * mx_v_heads * mx_head_v, mx_head_k)
    mx_kc2 = k_cache.view(k_cache.size(0) * mx_heads * mx_ring, mx_head_k)
    mx_klow2 = klow_cache.view(k_cache.size(0) * mx_heads * mx_ring, mx_head_k)
    mx_dc2 = d_cache.view(d_cache.size(0) * mx_v_heads * mx_ring, mx_head_v)
    mx_dlow2 = dlow_cache.view(d_cache.size(0) * mx_v_heads * mx_ring, mx_head_v)
    mx_g1 = g_cache.view(g_cache.size(0) * mx_v_heads * mx_ring)
    mx_hidden_fp8 = torch.empty(
        (mx_tokens, mx_hidden_dim), dtype=torch.float8_e4m3fn, device=mx_dev
    )
    mx_quant_chunks = 16
    assert mx_hidden_dim == mx_quant_chunks * mx_quant_n
    mx_ba_partials = torch.empty(
        (mx_quant_chunks, mx_tokens, 2 * mx_v_heads), dtype=torch.float32, device=mx_dev
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
    mx_ba_partials2 = mx_ba_partials.view(mx_quant_chunks * mx_tokens, 2 * mx_v_heads)
    mx_qkv_partials = torch.empty(
        (mx_ip_chunks, mx_tokens, mx_conv_dim), dtype=torch.float32, device=mx_dev
    )
    mx_z_partials = torch.empty(
        (mx_ip_chunks, mx_tokens, mx_qkvz_rows), dtype=torch.float32, device=mx_dev
    )
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
    # Routing, one token per task: the warp-local top 10 (padded rows write
    # sentinel picks and zero packed words, which the dedup masks).
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
    # Routed W13 per (top-k pair, row half): a pair whose expert a later pair of
    # the window has writes zeros.
    for p_pair, p_half, p_t in hl.tile([slots, 2, rows], block_size=[1, 1, None]):
        p_token = p_pair.begin // top_k
        p_pick = p_pair.begin % top_k
        p_expert = topk_ids[p_token, p_pick]
        # A later pick of the window (padded tokens' words are zeros, not picks).
        p_words = route_words[:]
        p_w = hl.arange(padded_rows * token_words)
        p_hit = hl.zeros([padded_rows * token_words], dtype=torch.int32)
        for p_j in hl.static_range(word_picks):
            p_pk = (p_w % token_words) * word_picks + p_j
            p_later = (
                (p_pk < top_k)
                & (p_w // token_words < rows)
                & (
                    (p_w // token_words) * pick_lanes_n + p_pk
                    > p_token * pick_lanes_n + p_pick
                )
            )
            p_same = (((p_words >> (10 * p_j)) & 1023) == p_expert) & p_later
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
    # Routed W2 per (top-k pair, hidden half) of the pair that serves its expert.
    for q_pair, q_half, q_t in hl.tile([slots, 2, rows], block_size=[1, 1, None]):
        q_token = q_pair.begin // top_k
        q_pick = q_pair.begin % top_k
        q_expert = topk_ids[q_token, q_pick]
        # A later pick of the window (padded tokens' words are zeros, not picks).
        q_words = route_words[:]
        q_w = hl.arange(padded_rows * token_words)
        q_hit = hl.zeros([padded_rows * token_words], dtype=torch.int32)
        for q_j in hl.static_range(word_picks):
            q_pk = (q_w % token_words) * word_picks + q_j
            q_later = (
                (q_pk < top_k)
                & (q_w // token_words < rows)
                & (
                    (q_w // token_words) * pick_lanes_n + q_pk
                    > q_token * pick_lanes_n + q_pick
                )
            )
            q_same = (((q_words >> (10 * q_j)) & 1023) == q_expert) & q_later
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
                q_values = (q_acc.T * q_alpha).to(torch.bfloat16)
                q_phys = hl.arange(256)
                q_hidden = q_phys // 32 * 32 + q_phys % 8 * 4 + q_phys % 32 // 8
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
        norm_partial[comm_token, comm_tile_n.begin // communication_n] = torch.sum(
            comm_sum * comm_sum
        )
        norm_input[comm_token, comm_tile_n] = comm_sum
    # The next layer's input RMSNorm (ba's input, and a return).
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
    # The next layer's GDN mixer: its code toward L2 first (the MoE's weight stream
    # evicted it).
    hl.prefetch_code()
    for mx_qt_c, mx_qt_k, mx_qt_t in hl.tile(
        [mx_ip_chunks, mx_hidden_dim // mx_ip_chunks, mx_tokens],
        block_size=[1, 128, 16],
    ):
        # in_proj's static FP8 input: the MoE's norm recomputed from the all-reduce's
        # outputs (the norm root still writes ``output``, ba's input), so the quant
        # waits on the all-reduce, not on the norm (one hop less at the boundary).
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
    # in_proj_ba's partial per K slice (summed by the core).
    for mx_bp_k, mx_bp_t in hl.tile(
        [mx_hidden_dim, mx_tokens], block_size=[mx_quant_n, 16]
    ):
        mx_bp_acc = hl.zeros([mx_bp_t, 2 * mx_v_heads], dtype=torch.float32)
        for mx_bp_kk in hl.tile(mx_bp_k.begin, mx_bp_k.end, block_size=64):
            mx_bp_acc = hl.dot(
                output[mx_bp_t, mx_bp_kk], ba_weight[:, mx_bp_kk].T, acc=mx_bp_acc
            )
        mx_ba_partials[mx_bp_k.begin // mx_quant_n, mx_bp_t, :] = mx_bp_acc
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
    # The in_proj rows and the causal conv, once per channel block.
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
    # ReplaySSM circular verify core, one task per (value head, 64 value columns).
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
        mx_owns_qk = (mx_c_g.begin == 0) & (mx_v_tile.begin == 0)
        mx_kcols = hl.arange(mx_head_k)
        mx_q_channels = mx_h * mx_head_k + mx_kcols
        mx_k_channels = mx_key_dim + mx_q_channels
        mx_v_channels = 2 * mx_key_dim + mx_hv * mx_head_v + mx_v_tile.index
        mx_wp = write_pos[mx_rslot]
        mx_base = cache_base[mx_rslot]
        mx_hist = hl.arange(mx_history_n)
        mx_hist_live = mx_hist < mx_wp
        mx_phys_c = mx_base + mx_hist & mx_ring - 1
        mx_phys_s = mx_base + mx_wp + mx_lanes & mx_ring - 1
        mx_spec_store = mx_live & (mx_wp + mx_lanes < mx_ring)
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
        mx_kcols0 = hl.arange(mx_head_k)
        mx_h0_all = hl.load(
            mx_h0_2,
            [
                ((mx_slot * mx_v_heads + mx_hv) * mx_head_v + mx_v_tile.index)[:, None],
                mx_kcols0[None, :],
            ],
        )
        mx_k_hist_all = hl.load(
            mx_kc2,
            [
                ((mx_rslot * mx_heads + mx_h) * mx_ring + mx_phys_c)[:, None],
                mx_kcols0[None, :],
            ],
            extra_mask=mx_hist_live[:, None],
        )
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
        mx_ba_rows = (hl.arange(mx_quant_chunks) * mx_tokens)[:, None] + mx_lane[
            None, :
        ]
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
            1.0 / torch.sqrt(torch.sum(mx_q_full * mx_q_full, dim=1) + 1e-06),
            0.0,
        )
        mx_k_rnorm = torch.where(
            mx_live,
            1.0 / torch.sqrt(torch.sum(mx_k_full * mx_k_full, dim=1) + 1e-06),
            0.0,
        )
        mx_hw_q = hl.zeros([mx_v_tile, mx_lanes_n], dtype=torch.float32)
        mx_hw_k = hl.zeros([mx_v_tile, mx_lanes_n], dtype=torch.float32)
        mx_scores_q = hl.zeros([mx_history_n, mx_lanes_n], dtype=torch.float32)
        mx_scores_k = hl.zeros([mx_history_n, mx_lanes_n], dtype=torch.float32)
        mx_kk_mat = hl.zeros([mx_lanes_n, mx_lanes_n], dtype=torch.float32)
        mx_kq_mat = hl.zeros([mx_lanes_n, mx_lanes_n], dtype=torch.float32)
        mx_cols = mx_kcols
        mx_q_c = mx_q_full
        mx_k_c = mx_k_full
        mx_q_t = (mx_q_c * (mx_q_rnorm * mx_scale)[:, None]).to(torch.bfloat16)
        mx_k_prec = mx_k_c * mx_k_rnorm[:, None]
        mx_k_t = mx_k_prec.to(torch.bfloat16)
        mx_k_r = (mx_k_prec - mx_k_t.to(torch.float32)).to(torch.bfloat16)
        mx_k_raw = mx_k_c.to(torch.bfloat16)
        mx_kk_mat = hl.dot(mx_k_raw, mx_k_raw.T, acc=mx_kk_mat) * (
            mx_k_rnorm[:, None] * mx_k_rnorm[None, :]
        )
        mx_kq_mat = hl.dot(mx_k_raw, mx_q_t.T, acc=mx_kq_mat) * mx_k_rnorm[:, None]
        mx_h0_c = mx_h0_all
        mx_hw_q = hl.dot(mx_h0_c, mx_q_t.T, acc=mx_hw_q)
        mx_hw_k = hl.dot(mx_h0_c, mx_k_raw.T, acc=mx_hw_k) * mx_k_rnorm[None, :]
        mx_k_hist = mx_k_hist_all
        mx_scores_q = hl.dot(mx_k_hist, mx_q_t.T, acc=mx_scores_q)
        mx_scores_k = hl.dot(mx_k_hist, mx_k_t.T, acc=mx_scores_k)
        if mx_owns_qk:
            hl.store(
                mx_kc2,
                [
                    ((mx_rslot * mx_heads + mx_h) * mx_ring + mx_phys_s)[:, None],
                    mx_cols[None, :],
                ],
                mx_k_t,
                extra_mask=mx_spec_store[:, None],
            )
            hl.store(
                mx_klow2,
                [
                    ((mx_rslot * mx_heads + mx_h) * mx_ring + mx_phys_s)[:, None],
                    mx_cols[None, :],
                ],
                mx_k_r,
                extra_mask=mx_spec_store[:, None],
            )
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
        mx_upper = (
            (mx_lanes[:, None] < mx_lanes[None, :])
            & mx_live[:, None]
            & mx_live[None, :]
        )
        mx_n1 = torch.where(
            mx_upper,
            -(
                mx_beta_s[None, :]
                * torch.exp(mx_cum_g[None, :] - mx_cum_g[:, None])
                * mx_kk_mat
            ),
            0.0,
        )
        mx_eye = (mx_lanes[:, None] == mx_lanes[None, :]).to(torch.float32)
        mx_n2 = hl.dot(mx_n1, mx_n1)
        mx_n4 = hl.dot(mx_n2, mx_n2)
        mx_tt = hl.dot(hl.dot(mx_eye + mx_n1, mx_eye + mx_n2), mx_eye + mx_n4)
        mx_r_mat = mx_beta_s[None, :] * (mx_v_s - mx_exp_g[None, :] * mx_hw_k)
        mx_d_spec = hl.dot(mx_r_mat, mx_tt)
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
    # Gated RMSNorm per value head, then out_proj's static FP8 input quant.
    for mx_gn_h, mx_gn_g in hl.tile([mx_heads, mx_group], block_size=[1, 1]):
        mx_gn_hv = mx_gn_h.begin * mx_group + mx_gn_g.begin
        mx_gn_lanes = hl.arange(mx_lanes_n)
        mx_gn_live = mx_gn_lanes < mx_tokens
        mx_gn_lane = torch.where(mx_gn_live, mx_gn_lanes, 0)
        mx_gn_channels = mx_gn_hv * mx_head_v + hl.arange(mx_head_v)
        mx_gn_cols = mx_gn_channels[None, :]
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
            mx_o_f
            * mx_rstd[:, None]
            * gate_norm_weight[:].to(torch.float32)[None, :]
            * (mx_z_x * torch.sigmoid(mx_z_x))
        )
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
    # out_proj, one task per 64 output rows over the whole K, into the symmetric
    # partial.
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
    # The following layer's input RMSNorm.
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
