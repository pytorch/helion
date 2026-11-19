# ruff: noqa: ANN001, ANN201
# pyrefly: ignore-errors
"""Qwen3.5 Gated-DeltaNet mixer for a TP4 speculative-verify window (one request),
from the layer's input RMSNorm output to the next RMSNorm, as one persistent Gluon
megakernel.

Per rank: 4 key heads and 16 value heads of 128, T <= 8 window tokens. The roots
reproduce SGLang's target-verify path:

- ba: ``in_proj_ba`` (BF16) as partials of 16 K slices.
- in_proj: ``in_proj_qkvz`` (ModelOpt per-tensor FP8 [5120, 4096], the physical
  layout of SGLang's transposed ``weight``) as FP32 partials of 4 K chunks of 128
  rows: the q|k|v rows (96 tasks) and, in a root of their own that only the
  gated norm waits for, the z rows (64 tasks). Its static FP8 input
  (``clamp(x / input_scale)`` -> E4M3) is quantized in the operand path, each K
  step's tile written to its ring slot by the body. Partials sum in chunk order,
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
- core: the ReplaySSM circular verify, one task per (value head, 64 value
  columns), the key head's normalized q / k and the ring records written as
  SGLang's kernel writes them; its small FP32 products (``T = (I + A)^-1``,
  ``D = R T^T``, ``D F``) are exact FMA dots (``dot_precision="ieee"``).
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

Besides the next layer's input and residual, it returns the intermediate
activations SGLang materializes (core output, gated norm, conv output, FP8
out_proj input, qkvz, ba) for checks.
"""

from __future__ import annotations

from pretuned_kernels.megakernels.qwen3_5_gdn_tp4._common import conv_silu
from pretuned_kernels.megakernels.qwen3_5_gdn_tp4._common import proj_rows
from pretuned_kernels.megakernels.qwen3_5_gdn_tp4._common import store_conv_state
from pretuned_kernels.megakernels.qwen3_5_gdn_tp4._common import store_conv_window
import torch

import helion
import helion.language as hl

FP8_MAX = 448.0
NORM_EPS = 1e-6


@helion.kernel(
    static_shapes=False,
    backend="gluon",
    triton_do_not_specialize=False,
    # The core's small FP32 products (T, D, D F) exact (FMA dots); the BF16 / FP8
    # dots are unaffected.
    dot_precision="ieee",
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
        cross_loop_dsmem=[False, True, False, False, False],
    ),
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
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
    k_chunk = 128  # the core's K in one step (2 steps of 64: +2%)
    assert head_k == k_chunk  # (the checkpoint / history keys load whole rows)
    conv_block = 128
    quant_n = 256  # the quant root's K chunk
    assert tokens <= lanes_n and ring <= history_n and ring & (ring - 1) == 0
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
    quant_chunks = 16
    assert hidden_dim == quant_chunks * quant_n
    ba_partials = torch.empty(
        (quant_chunks, tokens, 2 * v_heads), dtype=torch.float32, device=dev
    )
    ip_chunks = 4
    qkvz_rows = conv_dim + value_dim
    ip_weight = inproj_weight.view(qkvz_rows, ip_chunks, hidden_dim // ip_chunks)
    hidden_chunked = hidden_states.view(tokens, ip_chunks, hidden_dim // ip_chunks)
    ba_partials2 = ba_partials.view(quant_chunks * tokens, 2 * v_heads)
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
    # The core output per value head as the gated norm reads it: FP32, all lanes.
    # SiLU(z) (FP32) per value channel, the gated norm's gate.
    # (every window lane: the gated norm reads whole lanes on the ring, lanes past the window
    # are never stored)
    z_gate = torch.empty((lanes_n, value_dim), dtype=torch.float32, device=dev)
    z_gate4 = z_gate.view(lanes_n, heads, group, head_v)
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

    # The next launch may start early: weights the previous launch does not
    # write stream before the grid wait.
    hl.pdl_wait(independent=(inproj_weight, ba_weight, out_proj_weight))

    # in_proj_ba's partial per K slice (summed by the core), in steps of 64
    # columns (4 KB weight boxes on the ring).
    for bp_k, bp_t in hl.tile([hidden_dim, tokens], block_size=[quant_n, 16]):
        bp_acc = hl.zeros([bp_t, 2 * v_heads], dtype=torch.float32)
        for bp_kk in hl.tile(bp_k.begin, bp_k.end, block_size=64):
            bp_acc = hl.dot(
                hidden_states[bp_t, bp_kk], ba_weight[:, bp_kk].T, acc=bp_acc
            )
        ba_partials[bp_k.begin // quant_n, bp_t, :] = bp_acc

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

    # in_proj_qkvz, z rows (read by the core's epilogue only), the same way.
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
        # launch; the K loop is one step of the whole head).
        kcols0 = hl.arange(head_k)
        h0_all = hl.load(
            h0_2,
            [((slot * v_heads + hv) * head_v + v_tile.index)[:, None], kcols0[None, :]],
        )
        k_hist_all = hl.load(
            kc2,
            [((rslot * heads + h) * ring + phys_c)[:, None], kcols0[None, :]],
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
        # Gates (in_proj_ba: the quant chunks' partials, BF16), beta, the window's
        # cumulative decay.
        ba_rows = (hl.arange(quant_chunks) * tokens)[:, None] + lane[None, :]
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
        # The K loop is one step of the whole head: its body runs once, on the q / k
        # rows loaded for the norms.
        cols = kcols
        q_c = q_full
        k_c = k_full
        q_t = (q_c * (q_rnorm * scale)[:, None]).to(torch.bfloat16)
        k_prec = k_c * k_rnorm[:, None]
        k_t = k_prec.to(torch.bfloat16)
        k_r = (k_prec - k_t.to(torch.float32)).to(torch.bfloat16)
        # The key products on the raw BF16 key rows (exact operands), scaled by the
        # rows' inverse norms after (one product each instead of the normalized key's
        # high and low parts); the ring still stores the normalized key's parts.
        k_raw = k_c.to(torch.bfloat16)
        kk_mat = hl.dot(k_raw, k_raw.T, acc=kk_mat) * (
            k_rnorm[:, None] * k_rnorm[None, :]
        )
        kq_mat = hl.dot(k_raw, q_t.T, acc=kq_mat) * k_rnorm[:, None]
        # Checkpoint projections (the non-window contribution).
        h0_c = h0_all
        hw_q = hl.dot(h0_c, q_t.T, acc=hw_q)
        hw_k = hl.dot(h0_c, k_raw.T, acc=hw_k) * k_rnorm[None, :]
        # Committed history keys.
        k_hist = k_hist_all
        scores_q = hl.dot(k_hist, q_t.T, acc=scores_q)
        scores_k = hl.dot(k_hist, k_t.T, acc=scores_k)
        if owns_qk:
            hl.store(
                kc2,
                [((rslot * heads + h) * ring + phys_s)[:, None], cols[None, :]],
                k_t,
                extra_mask=spec_store[:, None],
            )
            hl.store(
                klow2,
                [((rslot * heads + h) * ring + phys_s)[:, None], cols[None, :]],
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
        n1 = torch.where(
            upper,
            -(beta_s[None, :] * torch.exp(cum_g[None, :] - cum_g[:, None]) * kk_mat),
            0.0,
        )
        eye = (lanes[:, None] == lanes[None, :]).to(torch.float32)
        n2 = hl.dot(n1, n1)
        n4 = hl.dot(n2, n2)
        tt = hl.dot(hl.dot(eye + n1, eye + n2), eye + n4)

        # R and D = R T^T (causal).
        r_mat = beta_s[None, :] * (v_s - exp_g[None, :] * hw_k)
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

    # Gated RMSNorm over each value head (norm_before_gate, SiLU gate; z from the
    # in_proj partials), BF16 out, then out_proj's static FP8 input quant.
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
