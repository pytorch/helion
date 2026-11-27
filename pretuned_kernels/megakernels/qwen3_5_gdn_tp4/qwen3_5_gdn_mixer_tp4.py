# ruff: noqa: ANN001, ANN201
# pyrefly: ignore-errors
"""Qwen3.5 Gated-DeltaNet mixer for a TP4 speculative-verify window (one request),
from the layer's input RMSNorm output to the next RMSNorm, as one persistent Gluon
megakernel.

Per rank: 4 key heads and 16 value heads of 128, 3 <= T <= 8 window tokens. The
roots reproduce SGLang's target-verify path:

- ba: ``in_proj_ba`` (BF16) as partials of 16 K slices.
- in_proj: ``in_proj_qkvz`` (ModelOpt per-tensor FP8 [5120, 4096], the physical
  layout of SGLang's transposed ``weight``) as FP32 partials of 4 K chunks of 128
  rows: the q|k|v rows (96 tasks) and, in a root of their own that only the z
  gate waits for, the z rows (64 tasks). Its static FP8 input (``clamp(x * (1 /
  input_scale))`` -> E4M3) is quantized in the operand path, each K step's tile
  written to its ring slot by the body. Partials sum in chunk order,
  * input_scale * weight_scale, and round to BF16 as SGLang's GEMM output does.
- conv: ``causal_conv1d_update`` (width 4, SiLU, no bias), one task per 128 of the
  3072 q|k|v channels (each channel's state has one owner). Token t sees
  ``[x_{t-3}, x_{t-2}, x_{t-1}, x_t]`` with ``x_{-3..-1}`` = the slot's
  ``conv_state``; products round to BF16 and add in FP32 (Triton's bf16 * bf16
  then ``acc +=``). ``conv_state`` becomes the post-window state ``[x_{T-3},
  x_{T-2}, x_{T-1}]`` in place, and the window row ``[state[1], state[2], x_0 ..
  x_{T-1}]`` goes to the deduplicated ``intermediate_conv_window`` (MambaPool's
  physical ``[slots + 1, 3072, T + 2]`` layout, step t = columns t..t+2), so
  SGLang's commit (``fused_conv_window_scatter_with_mask``) still restores the
  accepted state.
- z gate: the z rows' in_proj output (BF16, as for q|k|v) and its SiLU, the
  gated norm's gate, per value head while the core runs.
- core: the ReplaySSM circular verify, one task per (value head, 64 value
  columns), the key head's normalized q / k and the ring records written as
  SGLang's kernel writes them. Its small causal FP32 products (``T = (I +
  A)^-1``, ``D = R T^T``, ``D F``) are dots whose operands ``N`` and ``R`` have
  their non-finite entries zeroed (``finite``; ``F`` is selected causal), so the
  zero terms of a non-finite later draft stay zero and cannot reach an earlier
  token's output or records; that draft's own output stays non-finite.
- gated norm: SGLang's RMSNormGated (``x * rsqrt(mean(x^2) + eps) * w * silu(z)``,
  BF16 out) per value head, then ``out_proj``'s static FP8 input quant
  (``clamp(y * (1 / input_scale))`` -> E4M3).
- out_proj: ModelOpt per-tensor FP8 ``[4096, 2048]``, one task per 64 output rows
  over the whole K (its weights fill the ring before its input is ready):
  ``bf16(acc * input_scale * weight_scale)`` straight into the symmetric partial.
- the tensor-parallel all-reduce (rank order, FP32) + residual, and the next
  GemmaRMSNorm, as SGLang's CuTe DSL ``kARResidualRMSNorm`` fusion computes them:
  the residual output and the sum of squares use the BF16-rounded sum, and the
  norm weight is the pre-folded ``1 + w`` (``gemma_weight``).

``hl.prefetch_code()`` after the PDL wait brings the code toward L2: in a decoder
layer the MoE's weight stream evicted it (SGLang serving, 4 x GB300: -30..-50 us per
verify step with both mixers; the standalone timing alone does not show it).

Besides the next layer's input and residual, it returns the intermediate
activations SGLang materializes (core output, gated norm, conv output, FP8
out_proj input, qkvz, ba) for checks.

``correctness_check()`` compares every stage with SGLang's kernels, the
projections and the all-reduce + norm with torch (``_setup``); ``main()`` runs
it, then times the kernel against SGLang's stock sequence; ``--check`` runs only
the check. Both need SGLang with its kernels' module paths of the 2026-10-20
operator regroup (sgl-project/sglang#45112; checked at 175ab80)::

    NVSHMEM_DISABLE_CUDA_VMM=1 PYTHONPATH=. \\
      python -m torch.distributed.run --standalone --nproc-per-node=4 \\
      pretuned_kernels/megakernels/qwen3_5_gdn_tp4/qwen3_5_gdn_mixer_tp4.py [--check]
"""

from __future__ import annotations

from itertools import starmap
import os
import sys
from typing import Any

from pretuned_kernels.megakernels._qwen3_5_tp4 import FP8_MAX
from pretuned_kernels.megakernels._qwen3_5_tp4 import NORM_EPS
from pretuned_kernels.megakernels._qwen3_5_tp4 import all_reduce_norm
from pretuned_kernels.megakernels._qwen3_5_tp4 import bytes_equal
from pretuned_kernels.megakernels._qwen3_5_tp4 import graph_time
from pretuned_kernels.megakernels._qwen3_5_tp4 import report
from pretuned_kernels.megakernels._qwen3_5_tp4 import run_main
from pretuned_kernels.megakernels._qwen3_5_tp4 import same
from pretuned_kernels.megakernels._qwen3_5_tp4 import say
from pretuned_kernels.megakernels._qwen3_5_tp4 import sglang_quant
from pretuned_kernels.megakernels._qwen3_5_tp4 import symmetric_partials
from pretuned_kernels.megakernels.qwen3_5_gdn_tp4 import _setup as ms
from pretuned_kernels.megakernels.qwen3_5_gdn_tp4._common import conv_silu
from pretuned_kernels.megakernels.qwen3_5_gdn_tp4._common import finite
from pretuned_kernels.megakernels.qwen3_5_gdn_tp4._common import proj_rows
from pretuned_kernels.megakernels.qwen3_5_gdn_tp4._common import store_conv_state
from pretuned_kernels.megakernels.qwen3_5_gdn_tp4._common import store_conv_window
import torch
import torch.distributed as dist

import helion
import helion.language as hl


@helion.kernel(
    static_shapes=False,
    # The core's causal products in FP32, as SGLang's verify core sums them (TF32 gives
    # 10x its error against the FP32 reference).
    dot_precision="ieee",
    backend="gluon",
    triton_do_not_specialize=False,
    # One persistent CTA per SM under dynamic dispatch; the GEMVs' weights stream
    # through a 4-deep TMA ring before their activation waits (3 stages: +6%),
    # warp specialization gives them their own loader and MMA warps.
    config=helion.Config(
        block_sizes=[16, 256, 16, 256, 16, 512],
        num_warps=4,
        pid_type="persistent_blocked",
        cross_loop_pipeline="dynamic",
        num_sm_multiplier=1,
        num_stages=4,
        range_warp_specializes=[True] * 12,
        indexing="tensor_descriptor",
        # The core's two half-head tasks run on one 2-CTA cluster and send the
        # head's output into the gated norm's CTA (its rank 0) over DSMEM.
        cross_loop_dsmem=[False] * 14 + [True] + [False] * 3,
    ),
    ignore_warnings=[helion.exc.TensorOperationInWrapper, helion.exc.FastMathUnused],
)
def qwen3_5_gdn_mixer_tp4(
    hidden_states,  # [T, 4096] bf16: the layer's input RMSNorm output
    inproj_weight,  # [5120, 4096] e4m3: in_proj_qkvz (row = output, K contiguous)
    inproj_scales,  # [2] f32: input_scale, input_scale * weight_scale
    ba_weight,  # [32, 4096] bf16: in_proj_ba (b | a rows)
    conv_weight,  # [3072, 4] bf16 (column 3 = current token)
    conv_state,  # [slots, 3072, 3] bf16, updated in place
    conv_window,  # [wslots, 3072, T + 2] bf16: dedup verify windows (physical)
    A_log,  # [16] f32
    dt_bias,  # [16] bf16 or f32
    h0,  # [slots, 16, 128, 128] bf16 checkpoint (read only)
    d_cache,  # [rslots, 16, L, 128] bf16
    k_cache,  # [rslots, 4, L, 128] bf16
    g_cache,  # [rslots, 16, L] f32
    dlow_cache,  # [rslots, 16, L, 128] bf16
    klow_cache,  # [rslots, 4, L, 128] bf16
    state_index,  # [1] int32: the request's mamba slot (conv state, checkpoint)
    window_index,  # [1] int32: the request's conv window row
    replay_index,  # [1] int32: the request's ring / cursor slot
    write_pos,  # [rslots] int32
    cache_base,  # [rslots] int32
    gate_norm_weight,  # [128] bf16 (RMSNormGated, no +1)
    out_proj_weight,  # [4096, 2048] e4m3 (row = output, K contiguous)
    out_proj_scales,  # [2] f32: input_scale, input_scale * weight_scale
    residual,  # [T, 4096] bf16
    norm_weight,  # [4096] bf16, the next norm's (1 + w)
    symmetric_partial,  # [T, 4096] bf16 symmetric memory
    group_name: hl.ProcessGroupName,
):
    hl.specialize(hidden_states.size())
    hl.specialize(hidden_states.stride())
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
    hl.specialize(residual.size())
    hl.specialize(residual.stride())
    hl.specialize(norm_weight.size())
    hl.specialize(norm_weight.stride())
    hl.specialize(symmetric_partial.size())
    hl.specialize(symmetric_partial.stride())

    tokens, hidden_dim = hidden_states.shape
    conv_dim, conv_width = conv_weight.shape
    state_len = conv_state.size(2)
    window_cols = conv_window.size(2)
    heads, ring, head_k = k_cache.shape[1:]
    v_heads, head_v = d_cache.size(1), d_cache.size(3)
    group = v_heads // heads
    key_dim = heads * head_k
    value_dim = v_heads * head_v
    # Literals: a module-level int reaches the kernel as a runtime scalar.
    lanes_n = 8  # window token lanes (T <= 8)
    history_n = 16  # ring lanes (L <= 16)
    # The core's K in one step (2 steps of 64: +2%): the checkpoint / history keys
    # load whole rows.
    assert head_k == 128
    conv_block = 128
    ba_slice_n = 256  # in_proj_ba's K slice (one partial each)
    # (T >= 3: the post-window conv state is the window's last 3 tokens)
    assert state_len <= tokens <= lanes_n and ring <= history_n
    assert ring & (ring - 1) == 0
    assert conv_width == 4 and state_len == 3 and window_cols == tokens + 2
    assert conv_dim == 2 * key_dim + value_dim
    assert inproj_weight.shape == (conv_dim + value_dim, hidden_dim)
    assert ba_weight.shape == (2 * v_heads, hidden_dim) and head_k == head_v
    hidden = residual.size(1)
    assert out_proj_weight.shape == (hidden, value_dim)
    assert symmetric_partial.size() == (tokens, hidden)
    scale = head_k**-0.5
    dev = residual.device

    # Flat views: gathers index rows of 2-D views (no scalar between index tensors).
    state2 = conv_state.view(conv_state.size(0) * conv_dim, state_len)
    window2 = conv_window.view(conv_window.size(0) * conv_dim, window_cols)
    h0_2 = h0.view(h0.size(0) * v_heads * head_v, head_k)
    kc2 = k_cache.view(k_cache.size(0) * heads * ring, head_k)
    klow2 = klow_cache.view(k_cache.size(0) * heads * ring, head_k)
    dc2 = d_cache.view(d_cache.size(0) * v_heads * ring, head_v)
    dlow2 = dlow_cache.view(d_cache.size(0) * v_heads * ring, head_v)
    g1 = g_cache.view(g_cache.size(0) * v_heads * ring)

    # in_proj: in_proj_ba partials per K slice, FP32 in_proj_qkvz partials of 4 K
    # chunks.
    ba_slices = 16
    assert hidden_dim == ba_slices * ba_slice_n
    ba_partials = torch.empty(
        (ba_slices, tokens, 2 * v_heads), dtype=torch.float32, device=dev
    )
    ip_chunks = 4
    qkvz_rows = conv_dim + value_dim
    ip_weight = inproj_weight.view(qkvz_rows, ip_chunks, hidden_dim // ip_chunks)
    hidden_chunked = hidden_states.view(tokens, ip_chunks, hidden_dim // ip_chunks)
    ba_partials2 = ba_partials.view(ba_slices * tokens, 2 * v_heads)
    qkv_partials = torch.empty(
        (ip_chunks, tokens, conv_dim), dtype=torch.float32, device=dev
    )
    # (the z columns of the qkvz rows only)
    z_partials = torch.empty(
        (ip_chunks, tokens, qkvz_rows), dtype=torch.float32, device=dev
    )

    # Checks: the in_proj and conv outputs as SGLang materializes them.
    qkvz = torch.empty((tokens, conv_dim + value_dim), dtype=torch.bfloat16, device=dev)
    ba = torch.empty((tokens, 2 * v_heads), dtype=torch.bfloat16, device=dev)
    mixed_qkv = torch.empty((tokens, conv_dim), dtype=torch.bfloat16, device=dev)
    core_out = torch.empty((tokens, value_dim), dtype=torch.bfloat16, device=dev)
    gated_out = torch.empty((tokens, value_dim), dtype=torch.bfloat16, device=dev)
    attn_fp8 = torch.empty((tokens, value_dim), dtype=torch.float8_e4m3fn, device=dev)
    # SiLU(z) (FP32) per value channel, the gated norm's gate (every window lane:
    # the gated norm reads whole lanes on the ring, lanes past the window are never
    # stored).
    z_gate = torch.empty((lanes_n, value_dim), dtype=torch.float32, device=dev)
    z_gate4 = z_gate.view(lanes_n, heads, group, head_v)
    # The core output per value head as the gated norm reads it: FP32, all lanes.
    o_part4 = torch.empty(
        (lanes_n, heads, group, head_v), dtype=torch.float32, device=dev
    )
    # (each core task's half of a head: one element of the split dimension)
    o_part5 = o_part4.view(lanes_n, heads, group, 2, head_v // 2)

    remote_partials = torch.ops.symm_mem.get_remote_tensors(
        symmetric_partial, group_name
    )
    output = torch.empty((tokens, hidden), dtype=torch.bfloat16, device=dev)
    residual_output = torch.empty_like(output)
    communication_n = 512
    norm_tasks = hidden // communication_n
    norm_partial = torch.empty((tokens, norm_tasks), dtype=torch.float32, device=dev)
    norm_input = torch.empty((tokens, hidden), dtype=torch.float32, device=dev)

    # Programmatic dependent launch: wait for the previous kernel at entry.
    hl.pdl_wait()
    # The code toward L2 first: in a decoder layer the MoE's weight stream evicted it.
    hl.prefetch_code()

    # in_proj_ba's partial per K slice (summed by the core), in steps of 64
    # columns (4 KB weight boxes on the ring).
    for bp_k, bp_t in hl.tile([hidden_dim, tokens], block_size=[ba_slice_n, 16]):
        bp_acc = hl.zeros([bp_t, 2 * v_heads], dtype=torch.float32)
        for bp_kk in hl.tile(bp_k.begin, bp_k.end, block_size=64):
            bp_acc = hl.dot(
                hidden_states[bp_t, bp_kk], ba_weight[:, bp_kk].T, acc=bp_acc
            )
        ba_partials[bp_k.begin // ba_slice_n, bp_t, :] = bp_acc

    # in_proj_qkvz, q|k|v rows: FP8 K chunks of 128 rows. Its static FP8 input is
    # quantized in the operand path (``clamp(x * (1 / input_scale))`` -> E4M3).
    for iq_n, iq_c, iq_t in hl.tile(
        [conv_dim, ip_chunks, tokens], block_size=[128, 1, None]
    ):
        iq_inv = 1.0 / inproj_scales[0]
        iq_acc = hl.zeros([iq_n, iq_t], dtype=torch.float32)
        for iq_k in hl.tile(hidden_dim // ip_chunks):
            iq_q = torch.clamp(
                hidden_chunked[iq_t, iq_c.begin, iq_k].to(torch.float32) * iq_inv,
                -FP8_MAX,
                FP8_MAX,
            ).to(torch.float8_e4m3fn)
            iq_acc = hl.dot(ip_weight[iq_n, iq_c.begin, iq_k], iq_q.T, acc=iq_acc)
        qkv_partials[iq_c.begin, iq_t, iq_n] = iq_acc.T

    # in_proj_qkvz, z rows (read by the z gate only), the same way.
    for iz_n, iz_c, iz_t in hl.tile(
        [conv_dim, 0, 0], [qkvz_rows, ip_chunks, tokens], block_size=[128, 1, None]
    ):
        iz_inv = 1.0 / inproj_scales[0]
        iz_acc = hl.zeros([iz_n, iz_t], dtype=torch.float32)
        for iz_k in hl.tile(hidden_dim // ip_chunks):
            iz_q = torch.clamp(
                hidden_chunked[iz_t, iz_c.begin, iz_k].to(torch.float32) * iz_inv,
                -FP8_MAX,
                FP8_MAX,
            ).to(torch.float8_e4m3fn)
            iz_acc = hl.dot(ip_weight[iz_n, iz_c.begin, iz_k], iz_q.T, acc=iz_acc)
        z_partials[iz_c.begin, iz_t, iz_n] = iz_acc.T

    # The in_proj rows (the K chunks' partials in chunk order, BF16) and the causal
    # conv, once per channel block: the conv output, the verify window rows and
    # the new conv state of the block's channels.
    for cv_c in hl.tile(conv_dim, block_size=conv_block):
        cv_lanes = hl.arange(lanes_n)
        cv_live = cv_lanes < tokens
        cv_lane = torch.where(cv_live, cv_lanes, 0)
        cv_channels = cv_c.index
        cv_row0 = state_index[0] * conv_dim
        cv_x = proj_rows(
            qkv_partials, ip_chunks, inproj_scales[1], cv_lanes, cv_live, cv_channels
        )
        cv_out = conv_silu(
            cv_x, state2, conv_weight, cv_row0, cv_channels, cv_lanes, cv_live
        )
        hl.store(
            mixed_qkv,
            [cv_lane[:, None], cv_channels[None, :]],
            cv_out.to(torch.bfloat16),
            extra_mask=cv_live[:, None],
        )
        hl.store(
            qkvz,
            [cv_lane[:, None], cv_channels[None, :]],
            cv_x.to(torch.bfloat16),
            extra_mask=cv_live[:, None],
        )
        store_conv_window(
            cv_x,
            state2,
            window2,
            cv_row0,
            window_index[0] * conv_dim,
            cv_channels,
            cv_lanes,
            cv_live,
        )
        store_conv_state(
            cv_x, state2, cv_row0, cv_channels, cv_lanes, cv_live, tokens, state_len
        )

    # The z rows' in_proj output (the K chunks' partials in chunk order, BF16) and
    # its SiLU gate, once per value head, while the core runs.
    for zs_h in hl.tile(v_heads, block_size=1):
        zs_lanes = hl.arange(lanes_n)
        zs_live = zs_lanes < tokens
        zs_lane = torch.where(zs_live, zs_lanes, 0)
        zs_channels = zs_h.begin * head_v + hl.arange(head_v)
        zs_x = proj_rows(
            z_partials,
            ip_chunks,
            inproj_scales[1],
            zs_lanes,
            zs_live,
            conv_dim + zs_channels,
        )
        hl.store(
            qkvz,
            [zs_lane[:, None], (conv_dim + zs_channels)[None, :]],
            zs_x.to(torch.bfloat16),
            extra_mask=zs_live[:, None],
        )
        hl.store(
            z_gate,
            [zs_lane[:, None], zs_channels[None, :]],
            zs_x * torch.sigmoid(zs_x),
            extra_mask=zs_live[:, None],
        )

    # ReplaySSM circular verify core, one task per (value head = key
    # head h, group member g; 64 value columns): half a head per task is ~1.2 us
    # faster than whole heads with the gated norm in the task epilogue.
    for c_h, c_g, v_tile in hl.tile([heads, group, head_v], block_size=[1, 1, 64]):
        h = c_h.begin
        hv = h * group + c_g.begin
        lanes = hl.arange(lanes_n)
        live = lanes < tokens
        lane = torch.where(live, lanes, 0)
        slot = state_index[0]
        rslot = replay_index[0]
        # This task commits its key head's ring keys.
        owns_qk = (c_g.begin == 0) & (v_tile.begin == 0)
        kcols = hl.arange(head_k)
        q_channels = h * head_k + kcols
        k_channels = key_dim + q_channels
        v_channels = 2 * key_dim + hv * head_v + v_tile.index
        wp = write_pos[rslot]
        base = cache_base[rslot]
        hist = hl.arange(history_n)
        hist_live = hist < wp
        phys_c = (base + hist) & (ring - 1)
        phys_s = (base + wp + lanes) & (ring - 1)
        spec_store = live & ((wp + lanes) < ring)

        # The committed history's log-decays and update vectors (no producer in
        # this launch), issued first.
        g_hist = torch.where(
            hist_live,
            hl.load(g1, [(rslot * v_heads + hv) * ring + phys_c], extra_mask=hist_live),
            0.0,
        )
        d_hist = (
            hl.load(
                dc2,
                [
                    ((rslot * v_heads + hv) * ring + phys_c)[:, None],
                    v_tile.index[None, :],
                ],
                extra_mask=hist_live[:, None],
            )
            .to(torch.float32)
            .T
        )

        # The checkpoint rows and the committed history keys (no producer in this
        # launch; one K step covers the whole head).
        h0_all = hl.load(
            h0_2,
            [((slot * v_heads + hv) * head_v + v_tile.index)[:, None], kcols[None, :]],
        )
        k_hist_all = hl.load(
            kc2,
            [((rslot * heads + h) * ring + phys_c)[:, None], kcols[None, :]],
            extra_mask=hist_live[:, None],
        )

        # The window's q / k / v rows (the conv's output), issued with the gates' loads.
        q_full = hl.load(
            mixed_qkv, [lane[:, None], q_channels[None, :]], extra_mask=live[:, None]
        ).to(torch.float32)
        k_full = hl.load(
            mixed_qkv, [lane[:, None], k_channels[None, :]], extra_mask=live[:, None]
        ).to(torch.float32)
        v_s = (
            hl.load(
                mixed_qkv,
                [lane[:, None], v_channels[None, :]],
                extra_mask=live[:, None],
            )
            .to(torch.float32)
            .T
        )
        # Gates (in_proj_ba: its K slices' partials summed, BF16), beta, the window's
        # cumulative decay.
        ba_rows = (hl.arange(ba_slices) * tokens)[:, None] + lane[None, :]
        b_s = torch.sum(
            hl.load(ba_partials2, [ba_rows, hv], extra_mask=live[None, :]), dim=0
        ).to(torch.bfloat16)
        a_s = torch.sum(
            hl.load(ba_partials2, [ba_rows, v_heads + hv], extra_mask=live[None, :]),
            dim=0,
        ).to(torch.bfloat16)
        b_s = b_s.to(torch.float32)
        a_s = a_s.to(torch.float32)
        x = a_s + dt_bias[hv].to(torch.float32)
        softplus = torch.where(x <= 20.0, torch.log(1.0 + torch.exp(x)), x)
        g_s = torch.where(live, -torch.exp(A_log[hv]) * softplus, 0.0)
        beta_s = torch.where(live, 1.0 / (1.0 + torch.exp(-b_s)), 0.0)
        cum_g = torch.cumsum(g_s, dim=0)
        exp_g = torch.exp(cum_g)

        q_rnorm = torch.where(
            live, 1.0 / torch.sqrt(torch.sum(q_full * q_full, dim=1) + 1e-6), 0.0
        )
        k_rnorm = torch.where(
            live, 1.0 / torch.sqrt(torch.sum(k_full * k_full, dim=1) + 1e-6), 0.0
        )

        hw_q = hl.zeros([v_tile, lanes_n], dtype=torch.float32)
        hw_k = hl.zeros([v_tile, lanes_n], dtype=torch.float32)
        scores_q = hl.zeros([history_n, lanes_n], dtype=torch.float32)
        scores_k = hl.zeros([history_n, lanes_n], dtype=torch.float32)
        kk_mat = hl.zeros([lanes_n, lanes_n], dtype=torch.float32)
        kq_mat = hl.zeros([lanes_n, lanes_n], dtype=torch.float32)
        # One K step covers the whole head: the products use the q / k rows loaded
        # for the norms.
        q_t = (q_full * (q_rnorm * scale)[:, None]).to(torch.bfloat16)
        k_prec = k_full * k_rnorm[:, None]
        k_t = k_prec.to(torch.bfloat16)
        k_r = (k_prec - k_t.to(torch.float32)).to(torch.bfloat16)
        # The key products on the raw BF16 key rows (exact operands), scaled by the
        # rows' inverse norms after (one product each instead of the normalized key's
        # high and low parts); the ring still stores the normalized key's parts.
        k_raw = k_full.to(torch.bfloat16)
        kk_mat = hl.dot(k_raw, k_raw.T, acc=kk_mat) * (
            k_rnorm[:, None] * k_rnorm[None, :]
        )
        kq_mat = hl.dot(k_raw, q_t.T, acc=kq_mat) * k_rnorm[:, None]
        # Checkpoint projections (the non-window contribution).
        hw_q = hl.dot(h0_all, q_t.T, acc=hw_q)
        hw_k = hl.dot(h0_all, k_raw.T, acc=hw_k) * k_rnorm[None, :]
        # Committed history keys.
        scores_q = hl.dot(k_hist_all, q_t.T, acc=scores_q)
        scores_k = hl.dot(k_hist_all, k_t.T, acc=scores_k)
        if owns_qk:
            hl.store(
                kc2,
                [((rslot * heads + h) * ring + phys_s)[:, None], kcols[None, :]],
                k_t,
                extra_mask=spec_store[:, None],
            )
            hl.store(
                klow2,
                [((rslot * heads + h) * ring + phys_s)[:, None], kcols[None, :]],
                k_r,
                extra_mask=spec_store[:, None],
            )

        # History replay: decay the committed d records to the window start.
        g_prefix = torch.cumsum(g_hist, dim=0)
        g_total = torch.sum(g_hist, dim=0)
        replay_decay = torch.where(hist_live, torch.exp(g_total - g_prefix), 0.0)
        d_scaled = (d_hist * replay_decay[None, :]).to(torch.bfloat16)
        total_decay = torch.exp(g_total)
        hw_q = total_decay * hw_q + hl.dot(d_scaled, scores_q.to(torch.bfloat16))
        hw_k = total_decay * hw_k + hl.dot(d_scaled, scores_k.to(torch.bfloat16))

        # T^T = ((I + A)^-1)^T for the strictly lower A (A^8 = 0): the sum of
        # N^k with N = -A^T, as (I + N)(I + N^2)(I + N^4); A^T built directly (the
        # key products are symmetric).
        upper = (lanes[:, None] < lanes[None, :]) & live[:, None] & live[None, :]
        n1 = finite(
            torch.where(
                upper,
                -(
                    beta_s[None, :]
                    * torch.exp(cum_g[None, :] - cum_g[:, None])
                    * kk_mat
                ),
                0.0,
            )
        )
        eye = (lanes[:, None] == lanes[None, :]).to(torch.float32)
        n2 = hl.dot(n1, n1)
        n4 = hl.dot(n2, n2)
        tt = hl.dot(hl.dot(eye + n1, eye + n2), eye + n4)

        # R and D = R T^T (causal).
        r_mat = finite(beta_s[None, :] * (v_s - exp_g[None, :] * hw_k))
        d_spec = hl.dot(r_mat, tt)

        # Outputs.
        causal = (lanes[:, None] <= lanes[None, :]) & live[:, None] & live[None, :]
        f_mat = torch.where(
            causal, torch.exp(cum_g[None, :] - cum_g[:, None]) * kq_mat, 0.0
        )
        o_tile = exp_g[None, :] * hw_q + hl.dot(d_spec, f_mat)
        out_cols = (hv * head_v + v_tile.index)[None, :]
        o_bf = o_tile.T.to(torch.bfloat16)
        hl.store(core_out, [lane[:, None], out_cols], o_bf, extra_mask=live[:, None])
        # The gated norm's input: the BF16 output as FP32, every lane (a 32-bit
        # unmasked slab of the head, its two halves' tasks the producers).
        o_part5[:, c_h.begin, c_g.begin, v_tile.id, :] = o_bf.to(torch.float32)

        # This window's records at the circular spec positions.
        d_hi = d_spec.to(torch.bfloat16)
        hl.store(
            dc2,
            [((rslot * v_heads + hv) * ring + phys_s)[:, None], v_tile.index[None, :]],
            d_hi.T,
            extra_mask=spec_store[:, None],
        )
        hl.store(
            dlow2,
            [((rslot * v_heads + hv) * ring + phys_s)[:, None], v_tile.index[None, :]],
            (d_spec - d_hi.to(torch.float32)).to(torch.bfloat16).T,
            extra_mask=spec_store[:, None],
        )
        if v_tile.begin == 0:
            hl.store(ba, [lane, hv], b_s.to(torch.bfloat16), extra_mask=live)
            hl.store(ba, [lane, v_heads + hv], a_s.to(torch.bfloat16), extra_mask=live)
            hl.store(
                g1, [(rslot * v_heads + hv) * ring + phys_s], g_s, extra_mask=spec_store
            )

    # Gated RMSNorm over each value head (norm_before_gate, SiLU gate from the z
    # gate root), BF16 out, then out_proj's static FP8 input quant.
    for gn_h, gn_g in hl.tile([heads, group], block_size=[1, 1]):
        gn_hv = gn_h.begin * group + gn_g.begin
        gn_lanes = hl.arange(lanes_n)
        gn_live = gn_lanes < tokens
        gn_lane = torch.where(gn_live, gn_lanes, 0)
        gn_channels = gn_hv * head_v + hl.arange(head_v)
        gn_cols = gn_channels[None, :]
        # Every global load first: they issue before the wait for the head's core
        # output (none moves past that wait).
        z_g = z_gate4[:, gn_h.begin, gn_g.begin, :]
        gn_w = gate_norm_weight[:].to(torch.float32)
        gn_inv = 1.0 / out_proj_scales[0]
        o_f = o_part4[:, gn_h.begin, gn_g.begin, :]
        rstd = torch.rsqrt(torch.sum(o_f * o_f, dim=1) / head_v + NORM_EPS)
        gated = (o_f * rstd[:, None] * gn_w[None, :]) * z_g
        gated_bf = gated.to(torch.bfloat16)
        hl.store(
            gated_out,
            [gn_lane[:, None], gn_cols],
            gated_bf,
            extra_mask=gn_live[:, None],
        )
        hl.store(
            attn_fp8,
            [gn_lane[:, None], gn_cols],
            torch.clamp(gated_bf.to(torch.float32) * gn_inv, -FP8_MAX, FP8_MAX).to(
                torch.float8_e4m3fn
            ),
            extra_mask=gn_live[:, None],
        )

    # out_proj over the whole K in one task of 64 output rows (the weights fill the ring
    # before its input is ready), its BF16 result straight into the symmetric partial.
    for op_tile_n, op_tile_t in hl.tile([hidden, tokens], block_size=[64, None]):
        # (the output scale loads before the K loop, not after its MMAs' wait)
        op_scale = out_proj_scales[1]
        op_acc = hl.zeros([op_tile_n, op_tile_t], dtype=torch.float32)
        for op_tile_k in hl.tile(value_dim):
            op_acc = hl.dot(
                out_proj_weight[op_tile_n, op_tile_k],
                attn_fp8[op_tile_t, op_tile_k].T,
                acc=op_acc,
            )
        symmetric_partial[op_tile_t, op_tile_n] = (op_acc.T * op_scale).to(
            torch.bfloat16
        )

    hl.pdl_launch_dependents()
    # Each token's sum over ranks (in rank order), then the residual.
    for comm_tile_t, comm_tile_n in hl.tile(
        [tokens, hidden], block_size=[1, communication_n]
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
        [tokens, hidden], block_size=[1, communication_n]
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
        core_out,
        gated_out,
        mixed_qkv,
        attn_fp8,
        qkvz,
        ba,
    )


# (write_pos, cache_base) of the request's ring slot in each checked case.
CASES = ((0, 0), (2, 5), (3, 14), (9, 11))


def _clone(d: dict) -> dict:
    return {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in d.items()}


def correctness_check(group: dist.ProcessGroup, verbose: bool = True) -> None:
    """Every stage against SGLang's target-verify path, each reference fed with
    the megakernel's previous stage (so each stage's error is its own): torch's
    FP8 in_proj_qkvz (static input quant, FP32 accumulate) and BF16 in_proj_ba,
    ``causal_conv1d_update`` (dedup window view, conv state and window bitwise),
    ``gdn_replayssm_spec_decode(launch_mode="verify")`` (output and rings),
    ``rms_norm_gated``, ``static_quant_fp8``, then torch for the FP8 out_proj
    and the all-reduce (FP32 sum in rank order, as the served CuTe DSL fusion
    sums), + residual and GemmaRMSNorm. The "chain" rows run SGLang's whole
    sequence from the inputs. A second launch on the same inputs and state must
    be bitwise equal."""
    rank = group.rank()
    device = torch.device("cuda", torch.cuda.current_device())
    partials = symmetric_partials(ms.T, ms.HIDDEN, group.group_name, device)

    def run(layer, state, x, partial) -> tuple:
        out = qwen3_5_gdn_mixer_tp4(
            *ms.kernel_args(layer, state, x, partial, group.group_name)
        )
        torch.cuda.synchronize()
        return out, partial.clone()

    layer = ms.make_layer(100 + rank, device)
    ok = True
    for ci, (wp, base) in enumerate(CASES):
        x = ms.make_inputs(300 + 10 * ci + rank, device)
        # The residual is replicated across TP ranks.
        x["residual"] = ms.make_inputs(999 + ci, device)["residual"]
        state0 = ms.make_state(200 + 10 * ci + rank, wp, base, device)
        sk = _clone(state0)
        outs, partial = run(layer, sk, x, partials[0])
        output, residual_out, core, gated, mixed, q8, qkvz, ba = outs
        rows = []
        # in_proj (from the hidden input); later stages read the kernel's.
        qkvz_ref, ba_ref = ms.torch_in_proj(x["hidden"], layer)
        ok &= report(rows, "in_proj qkvz", qkvz, qkvz_ref, 1e-2)
        ok &= report(rows, "in_proj ba", ba, ba_ref, 1e-2)
        # conv (from the in_proj output)
        sr = _clone(state0)
        mixed_ref = ms.sglang_conv(qkvz, layer, sr)
        ok &= report(rows, "conv mixed_qkv", mixed, mixed_ref, 1e-2)
        for name in ("conv_state", "conv_window"):
            ok &= report(rows, name, sk[name], sr[name], 0, bitwise=True)
        # core (on the kernel's conv output)
        sc = _clone(state0)
        core_ref = ms.sglang_core(mixed, ba, layer, sc)
        ok &= report(rows, "core out", core, core_ref, 1e-2)
        ok &= report(rows, "  g_cache", sk["g_cache"], sc["g_cache"], 1e-2)
        # The rings keep each FP32 record as a BF16 pair (hi, lo = bf16(x - hi)).
        # When the FP32 value differs by an ulp (exact vs approximate exp / div)
        # and crosses a BF16 rounding boundary, one BF16 ulp moves between the
        # parts: hi is off by an ulp and lo by about minus that ulp, a large
        # relative error in lo alone while hi + lo, the value the replay uses,
        # stays within FP32 rounding. So the pair is judged on hi and on hi + lo;
        # lo alone is reported, not judged.
        for hi, lo in (("d_cache", "dlow_cache"), ("k_cache", "klow_cache")):
            ok &= report(rows, f"  {hi}", sk[hi], sc[hi], 1e-2)
            report(rows, f"  {lo} (info)", sk[lo], sc[lo], float("inf"))
            ok &= report(
                rows,
                f"  {hi}+lo",
                sk[hi].float() + sk[lo].float(),
                sc[hi].float() + sc[lo].float(),
                1e-2,
            )
        # gated norm (on the kernel's core output), FP8 input quant (on its norm)
        gated_ref = ms.sglang_gated_norm(core, qkvz, layer)
        ok &= report(rows, "gated norm", gated, gated_ref, 1e-2)
        ok &= bytes_equal(
            rows, "fp8 input quant", q8, sglang_quant(gated, layer["input_scale"])
        )
        # out_proj (on the kernel's FP8 input), AR + residual + norm (on the
        # kernel's partials, gathered)
        partial_ref = ms.torch_out_proj(q8, layer)
        ok &= report(rows, "out_proj partial", partial, partial_ref, 1e-2)
        out_ref, res_ref = all_reduce_norm(
            partial, x["residual"], layer["norm_weight"], NORM_EPS
        )
        ok &= report(rows, "residual_out", residual_out, res_ref, 1e-2)
        ok &= report(rows, "normed output", output, out_ref, 1e-2)
        # the whole SGLang chain from the inputs
        sch = _clone(state0)
        m_ch = ms.sglang_conv(qkvz_ref, layer, sch)
        c_ch = ms.sglang_core(m_ch, ba_ref, layer, sch)
        g_ch = ms.sglang_gated_norm(c_ch, qkvz_ref, layer)
        p_ch = ms.torch_out_proj(sglang_quant(g_ch, layer["input_scale"]), layer)
        o_ch, r_ch = all_reduce_norm(
            p_ch, x["residual"], layer["norm_weight"], NORM_EPS
        )
        ok &= report(rows, "chain core out", core, c_ch, 2e-2)
        ok &= report(rows, "chain partial", partial, p_ch, 2e-2)
        ok &= report(rows, "chain residual_out", residual_out, r_ch, 2e-2)
        ok &= report(rows, "chain normed output", output, o_ch, 2e-2)
        # determinism
        s2 = _clone(state0)
        out2, partial2 = run(layer, s2, x, partials[1])
        deterministic = all(
            starmap(same, zip((*outs, partial), (*out2, partial2), strict=True))
        ) and all(same(sk[n], s2[n]) for n in sk)
        rows.append(
            f"  {'determinism':<22} {'bitwise' if deterministic else 'DIFFERS  <-- FAIL'}"
        )
        ok &= deterministic
        if verbose or not ok:
            say(f"rank{rank} wp={wp} base={base}\n" + "\n".join(rows))
    if not ok:
        raise AssertionError(f"rank {rank}: qwen3_5_gdn_mixer_tp4 differs from SGLang")


def benchmark(group: dist.ProcessGroup, verbose: bool = True) -> dict[str, Any]:
    """``LAYERS`` distinct layers (default 16: weights and state; an even count,
    as consecutive launches alternate two partials), each launched once a step,
    ``REPS`` steps (default 4) in one CUDA graph: microseconds per layer (median
    and minimum of 10 replays) for the megakernel (with the all-reduce, residual
    and norm) and for SGLang's stock sequence (the FP8 in_proj_qkvz and BF16
    in_proj_ba GEMMs, qkvzba split, causal_conv1d_update, qkv split, ReplaySSM
    verify, gated RMSNorm, static FP8 quant + FlashInfer bmm_fp8) without the
    all-reduce, residual and norm."""
    from sglang.kernels.ops.attention.fla.layernorm_gated import rms_norm_gated
    from sglang.kernels.ops.attention.linear.gdn_replayssm_spec_decode import (
        gdn_replayssm_spec_decode,
    )
    from sglang.kernels.ops.attention.triton_gdn_fused_proj import (
        fused_qkv_split_gdn_prefill,
    )
    from sglang.kernels.ops.attention.triton_gdn_fused_proj import (
        fused_qkvzba_split_reshape_cat_contiguous,
    )
    from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_update
    from sglang.srt.layers.quantization.fp8_utils import apply_fp8_linear_bmm_flashinfer

    rank = group.rank()
    dev = torch.device("cuda", torch.cuda.current_device())
    n_layers = int(os.environ.get("LAYERS", "16"))
    assert n_layers % 2 == 0, "LAYERS must be even"
    reps = int(os.environ.get("REPS", "4"))
    partials = symmetric_partials(ms.T, ms.HIDDEN, group.group_name, dev)
    layers = [ms.make_layer(1000 * i + rank, dev) for i in range(n_layers)]
    states = [ms.make_state(2000 * i + rank, 2, 5, dev) for i in range(n_layers)]
    x = ms.make_inputs(77 + rank, dev)
    x["residual"] = ms.make_inputs(4242, dev)["residual"]
    qsl = torch.tensor([0, ms.T], dtype=torch.int32, device=dev)
    flush = torch.zeros(ms.RSLOTS, dtype=torch.int8, device=dev)
    win_views = [ms.window_view(s["conv_window"]) for s in states]

    def ours() -> None:
        for i in range(n_layers):
            qwen3_5_gdn_mixer_tp4(
                *ms.kernel_args(
                    layers[i], states[i], x, partials[i % 2], group.group_name
                )
            )

    def stock() -> None:
        for i in range(n_layers):
            lw, st = layers[i], states[i]
            qkvz = apply_fp8_linear_bmm_flashinfer(
                x["hidden"],
                lw["inproj_stored"],
                lw["inproj_weight_scale"],
                lw["inproj_input_scale"],
                pre_quant_output_dtype=torch.bfloat16,
            )
            ba = torch.nn.functional.linear(x["hidden"], lw["ba_weight"])
            mixed_qkv, z, b, a = fused_qkvzba_split_reshape_cat_contiguous(
                qkvz, ba, ms.H, ms.HV, ms.K, ms.V
            )
            conv = causal_conv1d_update(
                mixed_qkv.view(1, ms.T, -1).transpose(1, 2),
                st["conv_state"],
                lw["conv_weight"],
                None,
                "silu",
                conv_state_indices=st["state_index"],
                intermediate_conv_window=win_views[i],
                intermediate_state_indices=st["window_index"],
            )
            mixed = conv.transpose(1, 2).reshape(ms.T, -1)
            q, k, v = fused_qkv_split_gdn_prefill(
                mixed, ms.H, ms.H, ms.HV, ms.K, ms.K, ms.V
            )
            core = torch.empty(ms.T, ms.HV, ms.V, dtype=torch.bfloat16, device=dev)
            gdn_replayssm_spec_decode(
                q=q.reshape(ms.T, ms.H, ms.K),
                k=k.reshape(ms.T, ms.H, ms.K),
                v=v.reshape(ms.T, ms.HV, ms.V),
                a=a,
                b=b,
                A_log=lw["A_log"],
                dt_bias=lw["dt_bias"],
                checkpoint_state=st["h0"],
                d_cache=st["d_cache"],
                k_cache=st["k_cache"],
                g_cache=st["g_cache"],
                rawv_cache=st["dlow_cache"],
                rawk_cache=st["klow_cache"],
                beta_cache=None,
                out=core,
                query_start_loc=qsl,
                ssm_state_indices=st["state_index"],
                replay_indices=st["replay_index"],
                write_pos=st["write_pos"],
                cache_base=st["cache_base"],
                is_flush=flush,
                max_cache_len=ms.L,
                max_spec_len=ms.T,
                scale=ms.K**-0.5,
                use_qk_l2norm_in_kernel=True,
                null_block_id=-1,
                launch_mode="verify",
            )
            y = rms_norm_gated(
                x=core.reshape(-1, ms.V),
                weight=lw["gate_norm_weight"],
                bias=None,
                z=z.reshape(-1, ms.V),
                eps=1e-6,
                group_size=None,
                norm_before_gate=True,
                is_rms_norm=True,
                activation="swish",
            ).reshape(ms.T, ms.VALUE_DIM)
            apply_fp8_linear_bmm_flashinfer(
                y,
                lw["out_proj_stored"],
                lw["weight_scale"],
                lw["input_scale"],
                pre_quant_output_dtype=torch.bfloat16,
            )

    t_ours = graph_time(ours, n_layers, reps)
    t_stock = graph_time(stock, n_layers, reps, collective=False)
    if verbose:
        say(
            f"rank{rank} T={ms.T} layers={n_layers} us/layer: megakernel med "
            f"{t_ours[0]:.2f} min {t_ours[1]:.2f} (incl. AR+residual+norm) | SGLang "
            f"stock med {t_stock[0]:.2f} min {t_stock[1]:.2f} (excl. AR+residual+norm)",
        )
    return {"latency_us": t_ours[0], "sglang_latency_us": t_stock[0]}


def main(verbose: bool = True, check_only: bool = False) -> dict[str, Any]:
    """The four-rank check, then the timing against SGLang's stock sequence
    (``check_only``: the check alone)."""
    return run_main(
        "qwen3_5_gdn_mixer_tp4", correctness_check, benchmark, verbose, check_only
    )


if __name__ == "__main__":
    main(check_only="--check" in sys.argv)
