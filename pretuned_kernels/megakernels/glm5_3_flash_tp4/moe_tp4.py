# ruff: noqa: ANN001, ANN201, ANN202
# pyrefly: ignore-errors
"""GLM-5.3 Flash NVFP4 MoE sublayer for a TP4 speculative-verify window (T <= 8
tokens) as one Gluon megakernel, with the TP all-reduce and the FFN -> next
attention mHC boundary folded into its tail (``_mhc``): from the FFN's input
(its hc_pre output) to the next layer's attention input, R', post' and comb'.

The model (``_setup.moe_reference`` has the PyTorch reference):

- router: BF16 [288, 4096] -> FP32 logits; sigmoid scores, the top 8 of score +
  FP32 correction bias (noaux_tc, one group; ties to the lower expert), their
  scores over their sum times 2.5;
- 288 routed experts at a quarter of the intermediate (512) per rank, NVFP4
  (group 16), in FlashInfer's trtllm-gen layout, read in place (SGLang's
  ``flashinfer_trtllm`` runner): W13 [288, 1024, 2048] (rows [up; gate],
  gated-act interleave + epilogue shuffle), W2 [288, 4096, 256] (shuffled rows),
  128x4-interleaved E4M3 block scales per 128-row tile and 512-wide K step; the
  SwiGLU clamps (swiglu_limit 10: gate <= 10, |up| <= 10, FlashInfer's gatedSilu
  with gemm1_clamp_limit) after the per-expert GEMM1 alpha;
- one NVFP4 shared expert (its own input / activation global scales and
  alphas): W13 in the routed trtllm-gen layout (a copy ``prepared_weights``
  builds once), W2 in SGLang's row order.

Roots, in launch order (one persistent CTA per SM, TMA ring, dynamic dispatch):
  1. router partials: 128 experts x one of 16 K chunks of 256 (48 tasks, the
     last expert tile masked);
  2. input quant: the routed and the shared NVFP4 copies of the input;
  3. routing: one token a task (8 tasks), the 16 partials summed in order, the
     top 8 over 512 lanes (288 experts), two picks per reduction; each token's
     picks also as a bitmap of 16 int32 words (``token_experts``);
  4. shared W13 (8 tiles of 128 rows, full K) with the clamped SwiGLU and its
     NVFP4 activation;
  5. routed W13 per (128-row tile, distinct-expert slot j): the window's j-th
     distinct expert (ascending), full K, clamped SwiGLU, NVFP4 activation of
     every window token; slots past the distinct count find none;
  6. shared W2 (32 tiles of 128 rows, K = 512) -> shared output;
  7. routed W2 per (eighth of the hidden, slot j): 4 tiles of 128 rows, K = 512
     (one MMA step), each row stored at its hidden column in the row of each
     token's pick, in a part-major scratch [eighth, pick, token, 512];
  8. combine per eighth of the hidden (it waits for that part's W2 tasks only):
     each token's 8 pick rows weighted (FP32) plus the shared output -> this
     rank's partial in symmetric memory;
  9-11. the mHC boundary (``_mhc``): the ranks' partials summed in rank order,
     hc_post, the next layer's hc_attn_pre and input RMSNorm -> li, R', post',
     comb'.

Every split sum is in a fixed order and the all-reduce in rank order (no
atomics): deterministic run to run.

``correctness_check()`` compares every rank with the torch reference (``_setup``) on
synthetic weights; ``main()`` runs it, then times the kernel with cold weights::

    NVSHMEM_DISABLE_CUDA_VMM=1 PYTHONPATH=. \\
      python -m torch.distributed.run --standalone --nproc-per-node=4 \\
      pretuned_kernels/megakernels/glm5_3_flash_tp4/moe_tp4.py [--check]
"""

from __future__ import annotations

import functools
import operator
import os
import sys
from typing import Any

from pretuned_kernels.megakernels._qwen3_5_tp4 import graph_time
from pretuned_kernels.megakernels._qwen3_5_tp4 import report
from pretuned_kernels.megakernels._qwen3_5_tp4 import run_main
from pretuned_kernels.megakernels._qwen3_5_tp4 import say
from pretuned_kernels.megakernels._qwen3_5_tp4 import symmetric_buffer
from pretuned_kernels.megakernels._qwen3_5_tp4 import verdict
from pretuned_kernels.megakernels._ring_config import ring_config_fields
from pretuned_kernels.megakernels.deepseek_v3_moe_nvfp4._common import popcount
from pretuned_kernels.megakernels.deepseek_v3_moe_nvfp4._common import scale_tile
from pretuned_kernels.megakernels.glm5_3_flash_tp4 import _mhc
from pretuned_kernels.megakernels.glm5_3_flash_tp4 import _setup
from pretuned_kernels.megakernels.glm5_3_flash_tp4._common import clamped_swiglu_nvfp4
from pretuned_kernels.megakernels.glm5_3_flash_tp4._common import nvfp4_pairs
from pretuned_kernels.megakernels.glm5_3_flash_tp4._common import route
from pretuned_kernels.megakernels.glm5_3_flash_tp4._common import swiglu_nvfp4
from pretuned_kernels.megakernels.glm5_3_flash_tp4._common import up_gate
from pretuned_kernels.megakernels.glm5_3_tp4._setup import trtllm_gen_w13
from pretuned_kernels.megakernels.glm5_3_tp4._setup import window_router
from pretuned_kernels.megakernels.qwen3_5_moe_tp4._common import expert_of
import torch
import torch.distributed as dist

import helion
import helion.language as hl

LANES = 8  # window token lanes (T <= 8)
TOP_K = 8
ROUTE_LANES = 512  # the routing's expert lanes (a power of two >= 288)
WORDS = ROUTE_LANES // 32  # int32 words of a token's expert bitmap
ROUTER_CHUNKS = 16  # the router's K splits
W2_PARTS = 8  # routed W2 tasks per expert (4 tiles of 128 rows each)


@helion.kernel(
    static_shapes=True,
    backend="gluon",
    triton_do_not_specialize=False,
    ignore_warnings=[helion.exc.TensorOperationInWrapper, helion.exc.FastMathUnused],
)
def glmf_moe_tp4(
    x,  # [T, 4096] bf16: the FFN input (its hc_pre output), plain loads (input quant)
    xr,  # [T, 16, 256] bf16: the same rows as the router's K chunks (a distinct view)
    router_weight,  # [288, 16, 256] bf16
    bias,  # [288] fp32: e_score_correction_bias
    w13,  # [288, 1024, 2048] uint8 (trtllm-gen layout)
    w13_scale,  # [288, 8 * 8, 16, 256] e4m3 (per 128-row tile and K step)
    w2,  # [288, 4096, 256] uint8 (trtllm-gen layout)
    w2_scale,  # [288, 32, 16, 256] e4m3
    alpha1,  # [288] fp32: GEMM1 alpha (gate == up)
    alpha2,  # [288] fp32: GEMM2 alpha
    clamp1,  # [288] fp32: the SwiGLU clamp on GEMM1's raw accumulators (10 / alpha1)
    sw13,  # [1024, 2048] uint8: shared W13 in the routed trtllm-gen layout
    sw13_scale,  # [8 * 8, 16, 256] e4m3
    sw2,  # [4096, 256] uint8: shared W2 (rows in hidden order)
    sw2_scale,  # [32, 16, 256] e4m3
    # [6] fp32: routed input global, shared input global, routed activation
    # global, shared alpha1, shared alpha2, shared activation global
    scales,
    # scratch
    router_partials,  # [8, 16, 512] fp32
    input_q,  # [T, 2048] uint8: routed NVFP4 input
    input_scale,  # [T, 256] e4m3
    s_input_q,  # [T, 2048] uint8: shared NVFP4 input
    s_input_scale,  # [T, 256] e4m3
    topk_ids,  # [8, 8] int32
    topk_weights,  # [8, 8] fp32
    token_experts,  # [8, 16] int32
    activation_q,  # [8 T, T, 256] uint8: slot j = the j-th distinct expert
    activation_scale,  # [8 T, T, 32] e4m3
    shared_act_q,  # [T, 256] uint8
    shared_act_scale,  # [T, 32] e4m3
    routed_output,  # [8, 8, T, 512] bf16: each pick's W2 row, by (eighth of the hidden, pick)
    shared_output,  # [T, 4096] bf16
    symmetric_partial,  # [T, 4096] bf16 symmetric memory: this rank's MoE partial
    # the mHC boundary after the FFN (``_mhc``): its residual streams and mixes
    # in, the next layer's hc_attn fn / base / scale and input norm, outputs
    r,  # [T, 4, 4096] bf16
    post,  # [T, 4] fp32
    comb,  # [T, 4, 4] fp32
    fn_pad,  # [32, 4, 4096] bf16
    hc_scale,  # [3] fp32
    base_pre,  # [4] fp32
    base_post,  # [4] fp32
    base_comb,  # [4, 4] fp32
    norm_w,  # [4096] bf16: the next layer's input_layernorm
    mix_part,  # [S, 8, 32] fp32 scratch (Sinkhorn's mixes)
    mix_t,  # [8, 32, S] fp32 scratch: the same, token-major (finalize's warp-local sums)
    gram_part,  # [8, 4, 4, S] fp32 scratch, token-major
    sq_part,  # [8, S] fp32 scratch, token-major
    li,  # [T, 4096] bf16 out: the next attention's input
    r_new,  # [T, 4, 4096] bf16 out
    post_out,  # [T, 4] fp32 out
    comb_out,  # [T, 4, 4] fp32 out (``comb_out_raw``: logits)
    group_name: hl.ProcessGroupName,
    comb_in_raw: hl.constexpr,  # comb holds the previous boundary's logits: Sinkhorn them here
    comb_out_raw: hl.constexpr,  # comb_out gets this boundary's logits (the next MK takes them raw)
) -> None:
    tokens, hidden = x.shape
    experts = w13.size(0)
    twice_intermediate, packed_hidden = w13.shape[1:]
    intermediate = twice_intermediate // 2
    packed_intermediate = intermediate // 2
    hidden_groups = hidden // 16
    router_chunks, router_k = xr.shape[1:]
    # Literals: a module-level int reaches the kernel as a runtime scalar.
    lanes = 512
    words = 16
    top_k = 8
    padded_rows = 8
    w2_parts = 8
    assert (
        experts <= lanes
        and topk_ids.size(1) == top_k
        and token_experts.size(1) == words
    )
    assert tokens <= padded_rows and intermediate == 512 and packed_intermediate == 256
    assert routed_output.size() == (w2_parts, top_k, tokens, 512)
    assert router_partials.size() == (padded_rows, router_chunks, lanes)
    slots = tokens * top_k
    # K steps of 512 FP4 values: 256 packed bytes and 32 scales per row.
    w13_steps = packed_hidden // 256
    w13_tiles = twice_intermediate // 128
    w2_tiles = hidden // 128
    w2_part_tiles = w2_tiles // w2_parts
    w13_view = w13.view(experts, twice_intermediate, w13_steps, 256)
    w2_rows = w2.view(experts, w2_tiles, 128, packed_intermediate)
    sw13_view = sw13.view(twice_intermediate, w13_steps, 256)
    sw2_rows = sw2.view(w2_tiles, 128, packed_intermediate)
    input_q_groups = input_q.view(tokens, hidden_groups, 8)
    input_q_steps = input_q.view(tokens, w13_steps, 256)
    input_scale_steps = input_scale.view(tokens, w13_steps, 32)
    s_input_q_groups = s_input_q.view(tokens, hidden_groups, 8)
    s_input_q_steps = s_input_q.view(tokens, w13_steps, 256)
    s_input_scale_steps = s_input_scale.view(tokens, w13_steps, 32)
    activation_q_groups = activation_q.view(slots, tokens, w13_tiles, 4, 8)
    activation_scale_groups = activation_scale.view(slots, tokens, w13_tiles, 4)
    shared_act_q_groups = shared_act_q.view(tokens, w13_tiles, 4, 8)
    shared_act_scale_groups = shared_act_scale.view(tokens, w13_tiles, 4)
    part_columns = hidden // w2_parts
    routed_rows = routed_output.view(w2_parts, top_k * tokens, part_columns)
    remote_partials = torch.ops.symm_mem.get_remote_tensors(
        symmetric_partial, group_name
    )

    hl.pdl_wait()
    # The code toward L2 first: the previous launch's weight stream evicted it.
    hl.prefetch_code()

    # 1. router partials: 128 experts x one K chunk (the last expert tile
    # masked)
    for rt_e, rt_c, rt_t in hl.tile(
        [experts, router_chunks, tokens], block_size=[128, 1, 16]
    ):
        rt_acc = hl.zeros([rt_e, rt_t], dtype=torch.float32)
        for rt_k in hl.tile(router_k, block_size=128):
            rt_acc = hl.dot(
                router_weight[rt_e, rt_c.begin, rt_k],
                xr[rt_t, rt_c.begin, rt_k].T,
                acc=rt_acc,
            )
        router_partials[rt_t, rt_c, rt_e] = rt_acc.T[:, None, :]

    # 2. the input's NVFP4 copies: routed experts and shared expert (own global
    # scales)
    for qt_row, qt_grp in hl.tile([tokens, hidden_groups], block_size=[1, 128]):
        qt_lane = hl.arange(8)[None, :]
        qt_offsets = qt_grp.index[:, None] * 16
        qt_low = x[qt_row, qt_offsets + qt_lane * 2].to(torch.float32)
        qt_high = x[qt_row, qt_offsets + qt_lane * 2 + 1].to(torch.float32)
        qt_amax = torch.maximum(
            torch.amax(torch.abs(qt_low), dim=-1),
            torch.amax(torch.abs(qt_high), dim=-1),
        )
        qt_q, qt_bs = nvfp4_pairs(qt_low, qt_high, qt_amax, hl.load(scales, [0]))
        qt_sq, qt_sbs = nvfp4_pairs(qt_low, qt_high, qt_amax, hl.load(scales, [1]))
        input_q_groups[qt_row, qt_grp, :] = qt_q
        input_scale[qt_row, qt_grp] = qt_bs
        s_input_q_groups[qt_row, qt_grp, :] = qt_sq
        s_input_scale[qt_row, qt_grp] = qt_sbs

    # 3. routing, one token a task (padding rows pick no experts)
    for rk_t in hl.tile(padded_rows, block_size=1):
        rk_tok = rk_t.begin
        rk_e = hl.arange(lanes)
        rk_c = hl.arange(router_chunks)
        rk_live = rk_e < experts
        rk_bias = hl.load(bias, [rk_e], extra_mask=rk_live)
        rk_p = hl.load(
            router_partials,
            [rk_tok + rk_c[:, None] * 0, rk_c[:, None], rk_e[None, :]],
            extra_mask=rk_live[None, :],
        )
        rk_id, rk_w, rk_u = route(
            torch.sum(rk_p, dim=0),
            rk_bias,
            hl.arange(top_k) * 0 + rk_tok < tokens,
            experts,
            lanes,
            2.5,  # the routed scaling factor
        )
        topk_ids[rk_t, :] = rk_id[None, :]
        topk_weights[rk_t, :] = rk_w[None, :]
        token_experts[rk_t, :] = rk_u[None, :]

    # 4. shared W13 (a routed W13 task of a fixed expert): clamped SwiGLU, NVFP4
    for s_tile, s_t in hl.tile([w13_tiles, tokens], block_size=[1, 8]):
        s_alpha = hl.load(scales, [3])
        s_gs = hl.load(scales, [5])
        s_acc = hl.zeros([128, s_t], dtype=torch.float32)
        for s_s in hl.tile(w13_steps, block_size=1):
            s_acc = hl.dot_scaled(
                sw13_view[s_tile.begin * 128 + hl.arange(128), s_s.begin, :],
                scale_tile(sw13_scale[s_tile.begin * w13_steps + s_s.begin, :, :]),
                "e2m1",
                s_input_q_steps[s_t, s_s.begin, :].T,
                s_input_scale_steps[s_t, s_s.begin, :],
                "e2m1",
                acc=s_acc,
                out_dtype=torch.float32,
            )
        # SGLang's shared expert clamps its GEMM output (after the alpha)
        s_packed, s_bs = clamped_swiglu_nvfp4(s_acc, s_alpha, s_gs)
        shared_act_q_groups[s_t, s_tile.begin, :, :] = s_packed
        shared_act_scale_groups[s_t, s_tile.begin, :] = s_bs

    # 5. routed W13 per (row tile, distinct expert): the window's j-th distinct
    # expert (ascending ids), one 128-row tile over the full K, then the clamped
    # SwiGLU and NVFP4 activation of its 64 intermediates for every window
    # token. Tiles first: an expert's 8 tiles go to 8 workers, the live slots
    # lead.
    for p_tile, p_t, p_j in hl.tile([w13_tiles, tokens, slots], block_size=[1, 8, 1]):
        p_words = hl.reduce(operator.or_, token_experts[:, :], dim=0)
        p_counts = popcount(p_words)
        p_before = hl.cumsum(p_counts, dim=0) - p_counts
        p_distinct = torch.sum(p_counts)
        p_gs = hl.load(scales, [2])
        if p_j.begin < p_distinct:
            p_expert = expert_of(p_words, p_counts, p_before, p_j.begin)
            p_alpha = alpha1[p_expert]
            p_limit = clamp1[p_expert]
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
            # FlashInfer's gatedSilu: the raw accumulators clamped (gate <=
            # limit, |up| <= limit), then the alpha
            p_up, p_gate = up_gate(p_acc)
            p_packed, p_bs = swiglu_nvfp4(
                torch.minimum(torch.maximum(p_up, -p_limit), p_limit) * p_alpha,
                torch.minimum(p_gate, p_limit) * p_alpha,
                p_gs,
            )
            activation_q_groups[p_j.begin, p_t, p_tile.begin, :, :] = p_packed
            activation_scale_groups[p_j.begin, p_t, p_tile.begin, :] = p_bs

    # 6. shared W2: 128-row tiles, K = 512 (one MMA step); after the routed W13
    # in claim order, so that its tasks do not hold workers waiting for the
    # shared activation
    for s2_tile, s2_t in hl.tile([w2_tiles, tokens], block_size=[1, 8]):
        s2_alpha = hl.load(scales, [4])
        s2_acc = hl.zeros([128, s2_t], dtype=torch.float32)
        for _s2_k in hl.tile(1, block_size=1):
            s2_acc = hl.dot_scaled(
                sw2_rows[s2_tile.begin, :, :],
                scale_tile(sw2_scale[s2_tile.begin, :, :]),
                "e2m1",
                shared_act_q[s2_t, :].T,
                shared_act_scale[s2_t, :],
                "e2m1",
                acc=s2_acc,
                out_dtype=torch.float32,
            )
        shared_output[s2_t, s2_tile.begin * 128 + hl.arange(128)] = (
            s2_acc.T * s2_alpha
        ).to(torch.bfloat16)

    # 7. routed W2 per (eighth, distinct expert): 128-row tiles of one K step
    # for every window token that picked the expert, stored in the token's row
    # of its pick (physical row b*32 + 8 c + m holds hidden b*32 + 4 m + c).
    for q_part, q_t, q_j in hl.tile([w2_parts, tokens, slots], block_size=[1, 8, 1]):
        q_words = hl.reduce(operator.or_, token_experts[:, :], dim=0)
        q_counts = popcount(q_words)
        q_before = hl.cumsum(q_counts, dim=0) - q_counts
        q_distinct = torch.sum(q_counts)
        if q_j.begin < q_distinct:
            q_expert = expert_of(q_words, q_counts, q_before, q_j.begin)
            q_alpha = alpha2[q_expert]
            q_hits = topk_ids[q_t, :] == q_expert
            q_where = torch.sum(
                torch.where(q_hits, hl.arange(top_k)[None, :], 0), dim=1
            )
            q_used = (torch.sum(q_hits.to(torch.int32), dim=1) > 0) & (
                q_t.index < tokens
            )
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
                q_values = (q_acc.T * q_alpha).to(torch.bfloat16)
                q_phys = hl.arange(128)
                q_hidden = (q_phys // 32) * 32 + (q_phys % 8) * 4 + (q_phys % 32) // 8
                hl.store(
                    routed_rows,
                    [
                        q_part.begin,
                        (q_where * tokens + q_t.index)[:, None],
                        (q_n.begin * 128 + q_hidden)[None, :],
                    ],
                    q_values,
                    extra_mask=q_used[:, None],
                )

    # The tail's code toward L2 again: the experts' weight stream evicted it
    # (runs where workers reach the combine, in the ring's idle MMA partition).
    hl.prefetch_code()
    # 8. combine per eighth of the hidden (a W2 part: the routed scratch is
    # part-major, so a task waits only for its part's W2 tasks, not for every
    # one): each token's pick rows weighted in pick order (FP32 weights) plus
    # the shared output, this rank's partial
    for c_t, c_n in hl.tile([tokens, hidden], block_size=[8, 512]):
        # the shared output first: it is ready long before the routed rows
        c_shared = shared_output[c_t, c_n].to(torch.float32)
        c_columns = c_n.index - c_n.begin
        c_routed = hl.zeros([c_t, c_n], dtype=torch.float32)
        for c_k in hl.static_range(8):  # top_k picks, in order
            c_routed = c_routed + topk_weights[c_t, c_k][:, None] * routed_output[
                c_n.id, c_k, c_t, c_columns
            ].to(torch.float32)
        symmetric_partial[c_t, c_n] = (c_routed + c_shared).to(torch.bfloat16)

    hl.pdl_launch_dependents()
    # 9-11. the mHC boundary (``_mhc``): the all-reduce of the ranks' partials,
    # hc_post, the next layer's hc_attn_pre -> its input, R', post', comb'
    for tt, tc in hl.tile([tokens, hidden], block_size=[8, 128]):
        # a raw comb made final per chunk task (redundant across chunks, a few
        # hundred FLOPs) before its all-reduce wait: no root nor barrier of its
        # own
        cc = comb[tt, :, :]
        if comb_in_raw:
            cc = _mhc.sinkhorn_init(cc)
            for _cs_it in hl.tile(19, block_size=1):  # ITERS - 1
                cc = _mhc.sinkhorn_step(cc)
        _mhc.chunk(
            remote_partials,
            tt,
            tc,
            r,
            post,
            cc,
            fn_pad,
            r_new,
            mix_part,
            gram_part,
            sq_part,
            mix_t=mix_t,
            comb_tile=True,
        )
    for tb, tn in hl.tile([tokens, hidden], block_size=[8, 512]):
        _mhc.finalize(
            tb,
            tn,
            hidden,
            r_new,
            norm_w,
            mix_part,
            gram_part,
            sq_part,
            hc_scale,
            base_pre,
            li,
            mix_t=mix_t,
        )
    for ts in hl.tile(tokens, block_size=2):
        c = _mhc.comb_logits(
            ts,
            hidden,
            mix_part,
            sq_part,
            hc_scale,
            base_post,
            base_comb,
            post_out,
        )
        if not comb_out_raw:
            c = _mhc.sinkhorn_init(c)
            # ITERS - 1 (a literal: a module int reaches the kernel as a runtime scalar)
            for _it in hl.tile(19, block_size=1):
                c = _mhc.sinkhorn_step(c)
        comb_out[ts, :, :] = c


# The dot operands streamed through the TMA ring; every other access is a
# pointer.
RING_LOADS = frozenset(
    [
        "router_weight",
        "xr",
        "w13_view",
        "w13_scale",
        "input_q_steps",
        "input_scale_steps",
        "w2_rows",
        "w2_scale",
        "activation_q",
        "activation_scale",
        "sw13_view",
        "sw13_scale",
        "s_input_q_steps",
        "s_input_scale_steps",
        "sw2_rows",
        "sw2_scale",
        "shared_act_q",
        "shared_act_scale",
    ]
)


def mk_config(bound) -> helion.Config:
    """The dot operands of ``RING_LOADS`` through the TMA ring, every other
    access by pointer; one persistent CTA per SM claiming tasks root by root;
    the ring roots and their loops warp-specialized (``ring_config_fields``'
    ``"root_loops"``); the all-reduce in band (Lamport). Every positional list
    is derived from the bound kernel."""
    fields = ring_config_fields(bound, RING_LOADS, warp_specialize="root_loops")
    return helion.Config(
        indexing=fields["indexing"],
        pid_type="persistent_blocked",
        num_sm_multiplier=1,
        num_stages=4,
        num_warps=4,
        cross_loop_pipeline="dynamic",
        range_warp_specializes=fields["range_warp_specializes"],
        cross_loop_inband_protocol=["lamport"],
    )


@functools.cache
def scratch(device: torch.device, tokens: int, hidden: int) -> dict[str, torch.Tensor]:
    """The kernel's scratch buffers on ``device``, one set for every call."""
    f32, u8, e4m3 = torch.float32, torch.uint8, torch.float8_e4m3fn

    def z(*shape: int, dtype=f32):
        return torch.zeros(*shape, device=device, dtype=dtype)

    slots = tokens * TOP_K
    return {
        "router_partials": z(LANES, ROUTER_CHUNKS, ROUTE_LANES),
        "input_q": z(tokens, hidden // 2, dtype=u8),
        "input_scale": z(tokens, hidden // 16, dtype=e4m3),
        "s_input_q": z(tokens, hidden // 2, dtype=u8),
        "s_input_scale": z(tokens, hidden // 16, dtype=e4m3),
        "topk_ids": z(LANES, TOP_K, dtype=torch.int32),
        "topk_weights": z(LANES, TOP_K),
        "token_experts": z(LANES, WORDS, dtype=torch.int32),
        "activation_q": z(slots, tokens, 256, dtype=u8),
        "activation_scale": z(slots, tokens, 32, dtype=e4m3),
        "shared_act_q": z(tokens, 256, dtype=u8),
        "shared_act_scale": z(tokens, 32, dtype=e4m3),
        "routed_output": z(
            W2_PARTS, TOP_K, tokens, hidden // W2_PARTS, dtype=torch.bfloat16
        ),
        "shared_output": z(tokens, hidden, dtype=torch.bfloat16),
    }


def symmetric_partial_spec(tokens: int) -> tuple[tuple[int, ...], torch.dtype]:
    """The (shape, dtype) of the symmetric buffer (rendezvoused on the TP group)
    the kernel writes this rank's MoE partial into and reads the peers' from."""
    return (tokens, _setup.HIDDEN), torch.bfloat16


def kernel_args(
    x,
    pw,
    symmetric_partial,
    mhc,
    out,
    group_name,
    comb_in_raw=False,
    comb_out_raw=False,
) -> tuple:
    """``glmf_moe_tp4``'s arguments: the input ``x`` [T, 4096] (contiguous BF16, T
    <= 8; two views), the prepared weights ``pw`` (``prepared_weights``), the
    module's scratch, ``symmetric_partial`` (``symmetric_partial_spec``), the mHC
    boundary's ``mhc`` (``_mhc.boundary_operands``) and ``out`` = (li, r_new,
    post_out, comb_out) (``_mhc.new_outputs``). ``comb_in_raw``: mhc's comb
    holds the previous boundary's logits (``_mhc.comb_logits``), else the final
    comb; ``comb_out_raw``: comb_out gets this boundary's logits, else the final
    comb (each mode pair compiles a kernel of its own). The scratch (this
    module's and ``_mhc``'s, cached per device and T) may hold anything at
    entry: no other input."""
    tokens, hidden = x.shape
    sc = scratch(x.device, tokens, hidden)
    # a distinct view object for the router's ring loads: the memory ops of one
    # tensor object share its name
    xr = x.view(tokens, ROUTER_CHUNKS, hidden // ROUTER_CHUNKS)
    return (
        x,
        xr,
        pw["router"],
        pw["bias"],
        pw["w13"],
        pw["w13s"],
        pw["w2"],
        pw["w2s"],
        pw["alpha1"],
        pw["alpha2"],
        pw["clamp1"],
        pw["sw13"],
        pw["sw13s"],
        pw["sw2"],
        pw["sw2s"],
        pw["scales"],
        *(sc[n] for n in SCRATCH_ORDER),
        symmetric_partial,
        *_mhc.tail_args(mhc, x.device, hidden),
        *out,
        group_name,
        bool(comb_in_raw),
        bool(comb_out_raw),
    )


SCRATCH_ORDER = (
    "router_partials",
    "input_q",
    "input_scale",
    "s_input_q",
    "s_input_scale",
    "topk_ids",
    "topk_weights",
    "token_experts",
    "activation_q",
    "activation_scale",
    "shared_act_q",
    "shared_act_scale",
    "routed_output",
    "shared_output",
)


def compile_kernel(args: tuple):
    """``glmf_moe_tp4`` compiled for ``args`` (once per binding)."""
    return _setup.compiled(glmf_moe_tp4, mk_config, args)


def prepared_weights(w: dict) -> dict:
    """The kernel's operands of one layer's per-rank weights (SGLang's processed
    MoE tensors; ``_setup.moe_weights`` names them): views of the routed experts
    in their trtllm-gen layout (read in place), the shared expert's W13 rebuilt in
    that layout from its logical copy (once, 2.4 MB a layer and rank), its W2 and
    128x4-interleaved scales as SGLang holds them; the per-layer scalars in one
    vector (``scales``). One alpha serves GEMM1's gate and up halves."""
    experts, hidden = w["w13"].shape[0], w["w2"].shape[1]
    assert torch.equal(w["g1_alphas"], w["g1_alphas_up"]), "untied gate / up alphas"
    sw13, sw13s = trtllm_gen_w13(w["sw13_log"][None], w["sw13s_log"][None])
    scales = (
        torch.stack(
            [
                w["w13_input_scale_quant"].reshape(()),
                w["s13_input_scale_inv"].reshape(()),
                w["w2_input_scale_quant"].reshape(()),
                w["s13_alpha"].reshape(()),
                w["s2_alpha"].reshape(()),
                w["s2_input_scale_inv"].reshape(()),
            ]
        )
        .float()
        .contiguous()
    )
    return {
        "router": w["router"].view(experts, ROUTER_CHUNKS, hidden // ROUTER_CHUNKS),
        "bias": w["bias"],
        "w13": w["w13"],
        "w13s": w["w13s"].view(experts, -1, 16, 256),
        "w2": w["w2"],
        "w2s": w["w2s"].view(experts, hidden // 128, 16, 256),
        "alpha1": w["g1_alphas"],
        "alpha2": w["g2_alphas"],
        "clamp1": w["gemm1_clamp_limit"],
        "sw13": sw13[0],
        "sw13s": sw13s[0].view(-1, 16, 256),
        "sw2": w["sw2"],
        "sw2s": w["sw2s_sw"].view(hidden // 128, 16, 256),
        "scales": scales,
    }


# ---------------------------- check and timing (torch.distributed.run, 4 ranks)

# The served verify window's distinct experts: p10 / p50 / p90 (mean 31.6).
DISTINCT = (23, 31, 41)


def _window(distinct: int, seed: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """A window [T, 4096] and a router that routes it to ``distinct`` experts
    (``glm5_3_tp4._setup.window_router``)."""
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(_setup.T, _setup.HIDDEN, generator=g, device=device)
    x = x.to(torch.bfloat16)
    return x, window_router(x, distinct, _setup.MOE_SHAPE, seed=seed)


def correctness_check(group: dist.ProcessGroup, verbose: bool = True) -> None:
    """Every rank its own weights (``_setup.moe_weights``) and windows routed to
    the served distinct-expert counts (``DISTINCT``), one with 30x the GEMM1
    alphas (it drives the SwiGLU clamps) and one with raw combs in and out:
    against ``_setup.moe_reference`` (each GEMM on the kernel's NVFP4 copies) the
    top-8 ids as sets, the NVFP4 copies, the shared output, each pick's W2 row and
    this rank's partial; li, R', post', comb' against the boundary of the ranks'
    reference partials summed in rank order (``_setup.report_boundary``). Then
    every case again back to back (no host synchronization) and once more with
    NaN scratch and outputs: the same bits."""
    rank = group.rank()
    device = torch.device("cuda", torch.cuda.current_device())
    T, H = _setup.T, _setup.HIDDEN
    t = _setup.boundary_inputs(seed=11, device=device)
    partial = symmetric_buffer(*symmetric_partial_spec(T), group.group_name, device)
    cases = [(f"D={d}", d, 1.0, False) for d in DISTINCT]
    cases += [
        ("D=31, GEMM1 alphas x30", 31, 30.0, False),
        ("D=31, raw combs", 31, 1.0, True),
    ]
    weights = _setup.moe_weights(seed=100 + rank, device=device)
    launches, ok = [], True
    for i, (name, distinct, factor, raw) in enumerate(cases):
        if raw:  # the comb as logits
            t = dict(t, comb=_setup.comb_logits(seed=5, device=device))
        x, router = _window(distinct, 10 * i + rank, device)
        alphas = weights["g1_alphas"] * factor
        w = dict(weights, router=router, g1_alphas=alphas, g1_alphas_up=alphas)
        w["gemm1_clamp_limit"] = _setup.SWIGLU_LIMIT / alphas
        out = _mhc.new_outputs(t["r"])
        mhc = _mhc.boundary_operands(t)
        args = kernel_args(
            x, prepared_weights(w), partial, mhc, out, group.group_name, raw, raw
        )
        kernel = compile_kernel(args)
        kernel(*args)
        torch.cuda.synchronize()
        sc = {k: v.clone() for k, v in scratch(device, T, H).items()}
        reference, ids, y, shared, quant = _setup.moe_reference(x, w, sc)
        got_ids = sc["topk_ids"][:T].long()
        rows = [f"  routed experts {len(set(got_ids.flatten().tolist()))}"]
        same_ids = torch.equal(got_ids.sort(-1).values, ids.sort(-1).values)
        rows.append(
            f"  {'top-k ids':<22} {'equal' if same_ids else 'DIFFER  <-- FAIL'}"
        )
        ok &= same_ids
        for label, (got, exact) in quant.items():
            ok &= report(rows, label, got, exact, 2e-2)
        ok &= report(rows, "shared output", sc["shared_output"], shared, 2e-2)
        # (picks matched by expert: near-equal keys may order differently)
        routed = sc["routed_output"].permute(2, 1, 0, 3).reshape(T, TOP_K, H)
        order = got_ids.argsort(-1)[..., None].expand(-1, -1, H)
        expected = y.gather(1, ids.argsort(-1)[..., None].expand(-1, -1, H))
        ok &= report(rows, "routed rows", routed.gather(1, order), expected, 1e-2)
        ok &= report(rows, "partial", partial, reference, 2e-2)
        ok &= _setup.report_boundary(rows, out, reference, t, group, raw)
        launches.append((functools.partial(kernel, *args), out))
        if verbose or not ok:
            say(f"rank{rank} {name}\n" + "\n".join(rows))
    poisoned = [
        *launches[-1][1],
        *scratch(device, T, H).values(),
        *_mhc.scratch(device, H).values(),
    ]
    back_to_back, nan = _setup.repeat_bitwise(launches, poisoned)
    ok &= back_to_back and nan
    if verbose or not ok:
        say(
            f"rank{rank} repeats: back to back {'bitwise' if back_to_back else 'DIFFER  <-- FAIL'}, "
            f"NaN scratch {'bitwise' if nan else 'DIFFERS  <-- FAIL'}"
        )
    verdict(ok, group, "glmf_moe_tp4")


def benchmark(group: dist.ProcessGroup, verbose: bool = True) -> dict[str, Any]:
    """Cold-weight timing per served distinct-expert count (``DISTINCT``):
    ``LAYERS`` distinct weight copies (default 8, about 1 GB each) back to back,
    ``REPS`` times (default 4) in one CUDA graph, each copy's router routing the
    window to that many experts."""
    rank = group.rank()
    device = torch.device("cuda", torch.cuda.current_device())
    layers = int(os.environ.get("LAYERS", "8"))
    reps = int(os.environ.get("REPS", "4"))
    T = _setup.T
    t = _setup.boundary_inputs(device=device)
    mhc, out = _mhc.boundary_operands(t), _mhc.new_outputs(t["r"])
    partial = symmetric_buffer(*symmetric_partial_spec(T), group.group_name, device)
    weights = [
        prepared_weights(_setup.moe_weights(seed=1000 * i + rank, device=device))
        for i in range(layers)
    ]
    torch.cuda.empty_cache()
    results = {}
    for distinct in DISTINCT:
        launches = []
        for i, pw in enumerate(weights):
            x, router = _window(distinct, i, device)
            pw = dict(pw, router=router.view(pw["router"].shape))
            args = kernel_args(x, pw, partial, mhc, out, group.group_name)
            launches.append(functools.partial(compile_kernel(args), *args))

        def step(launches=launches) -> None:
            for launch in launches:
                launch()

        median, minimum = graph_time(step, layers, reps)
        if verbose:
            say(f"rank{rank} D={distinct} us/launch med {median:.2f} min {minimum:.2f}")
        results[distinct] = {"latency_us": median, "min_latency_us": minimum}
    return results


def main(verbose: bool = True, check_only: bool = False) -> dict[str, Any]:
    """The four-rank check, then the cold-weight timing (``check_only``: the
    check alone)."""
    return run_main("glmf_moe_tp4", correctness_check, benchmark, verbose, check_only)


if __name__ == "__main__":
    main(check_only="--check" in sys.argv)
