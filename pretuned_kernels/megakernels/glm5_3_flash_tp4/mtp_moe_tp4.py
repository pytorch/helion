# ruff: noqa: ANN001, ANN201, ANN202
# pyrefly: ignore-errors
"""GLM-5.3 Flash MTP draft MoE at TP4 (BF16 experts: the NextN layer is unquantized) for
the draft's one row (a decode step, or the draft extend's selected row), with the
tensor-parallel all-reduce, the residual add and the draft's final norm
(shared_head.norm) fused into one persistent Gluon megakernel. Replaces SGLang's router
tiny GEMM, FlashInfer's trtllm_bf16_moe (in-kernel DeepSeekV3 routing, FC1, SwiGLU, FC2,
finalize), the BF16 shared expert (gate_up GEMM, silu_and_mul_clamp, down GEMM), the
shared add, the custom all-reduce and the fused add + RMSNorm.

The model (SGLang 8fdaa8787b; DeepseekV2MoE with reduce_results False, the
flashinfer_trtllm runner):

- router: BF16 [288, 4096] -> FP32 logits; sigmoid scores, the top 8 of score + FP32
  correction bias (noaux_tc, one group; ties to the lower expert), their scores over
  their sum times 2.5 (``_common.route``);
- 288 routed experts at a quarter of the intermediate (512) per rank, BF16, in
  FlashInfer's TRT-LLM BF16 block layout, read in place (SGLang's
  ``_apply_trtllm_bf16_block_layout``): W13 [288, 64, 1024, 64] (K blocks of 64;
  canonical rows [up 512 | gate 512] (the loader swaps w1 / w3 for trtllm), the
  gated-GEMM row shuffle of 128-row tiles: physical row 32 b + 16 h + 8 u + m of tile t
  holds intermediate 64 t + 16 b + 2 m + h, up for u = 0, gate for u = 1), W2 [288, 8,
  4096, 64] (physical row 32 b + 8 c + m of a 128-row tile holds hidden 32 b + 4 m + c);
  SiLU(gate) * up with NO clamp (SGLang passes no swiglu_limit to trtllm_bf16_moe);
- one BF16 shared expert: gate_up [gate 512 | up 512, 4096] rounded to BF16, then
  silu_and_mul_clamp (gate <= 10, |up| <= 10), down [4096, 512] rounded to BF16;
- routed (BF16 finalize: the picks' rows weighted in pick order by BF16 weights) +=
  shared in BF16, the all-reduce (FP32 in rank order, BF16), then shared_head.norm with
  the residual (SGLang's JIT fused add + RMSNorm: BF16 sum, bf16((x * w) * r)).

Roots, in launch order (one persistent CTA per SM, TMA ring, dynamic dispatch):
  1. router partials: 128 experts x one of 16 K chunks of 256 (48 tasks);
  2. routing (one task): the 16 partials summed in order, the top 8 over 512 lanes;
  3. shared W13 per (128-row tile, K chunk of 512): FP32 partials (64 tasks);
  4. shared activation per 128 intermediates: the chunk sums, BF16, clamped, SiLU * up;
  5. shared W2 per 128-row tile (32 tasks) -> shared output (BF16) (the shared expert
     fills the routing's latency);
  6-11. the routed experts in two halves of the picks (a: 0-3, b: 4-7; ``SPLIT``),
     each with its own scratch, in the claim order W13 a, SwiGLU a, W13 b, W2 a, SwiGLU b,
     W2 b: half a's W2 tasks are ready work for the workers half b's last W13 wave leaves
     idle (one routed pass would make every W2 task wait for every SwiGLU task: a one-wave
     producer gets a root barrier); only half b's SwiGLU and W2 trail the weight stream.
     W13 per (pick j, 128-row tile, K chunk of 512): the j-th pick's expert, FP32 partials
     (256 tasks a half); SwiGLU per (pick, tile): the chunk sums in order, SiLU(gate) * up,
     BF16; W2 per (pick, 128-row tile) (128 tasks a half): the full K (512), BF16 rows into
     a part-major scratch [eighth of the hidden, pick, 512];
  12. combine per eighth of the hidden: the picks' rows in pick order with BF16 weights (FP32), BF16, plus the shared output (BF16)
     -> this rank's partial in symmetric memory;
  13. all-reduce + residual per (token, 128 columns): the ranks' partials in rank order
     (FP32), BF16, plus the residual, BF16; sum of squares;
  14. shared_head.norm per (token, 512 columns) -> the draft hidden state.

Every split sum is in a fixed order and the all-reduce in rank order (no atomics):
deterministic run to run. The all-reduce reads only partials root 12 wrote in the same
launch.

``correctness_check()`` compares every rank with the torch reference (``_setup``) on
synthetic weights; ``main()`` runs it, then times the kernel with cold weights::

    NVSHMEM_DISABLE_CUDA_VMM=1 PYTHONPATH=. \\
      python -m torch.distributed.run --standalone --nproc-per-node=4 \\
      pretuned_kernels/megakernels/glm5_3_flash_tp4/mtp_moe_tp4.py [--check]
"""

from __future__ import annotations

import functools
import os
import sys
from typing import Any

from pretuned_kernels.megakernels._qwen3_5_tp4 import all_reduce_norm
from pretuned_kernels.megakernels._qwen3_5_tp4 import graph_time
from pretuned_kernels.megakernels._qwen3_5_tp4 import report
from pretuned_kernels.megakernels._qwen3_5_tp4 import run_main
from pretuned_kernels.megakernels._qwen3_5_tp4 import same
from pretuned_kernels.megakernels._qwen3_5_tp4 import say
from pretuned_kernels.megakernels._qwen3_5_tp4 import symmetric_buffer
from pretuned_kernels.megakernels._qwen3_5_tp4 import verdict
from pretuned_kernels.megakernels._ring_config import ring_config_fields
from pretuned_kernels.megakernels.glm5_3_flash_tp4 import _setup
from pretuned_kernels.megakernels.glm5_3_flash_tp4._common import route
import torch
import torch.distributed as dist

import helion
import helion.language as hl

HIDDEN = _setup.HIDDEN
TOP_K = 8
ROUTE_LANES = 512
ROUTER_CHUNKS = 16
W13_CHUNKS = 8  # K chunks of W13 (512 = 8 blocks of 64 each)
W2_PARTS = 8  # eighths of the hidden (the combine's tasks)
SPLIT = 4  # picks of the first routed half
RING_LOADS = frozenset(
    [
        "router_weight",
        "xr",
        "sw13",
        "xs",
        "w13",
        "xk",
        "sw2",
        "shared_act",
        "w2",
        "act_blocks_a",
        "act_blocks_b",
    ]
)


@helion.kernel(
    static_shapes=True,
    backend="gluon",
    triton_do_not_specialize=False,
    ignore_warnings=[helion.exc.TensorOperationInWrapper, helion.exc.FastMathUnused],
)
def glmf_mtp_moe_tp4(
    xr,  # [T, 16, 256] bf16: the MoE input (post_attention_layernorm) as the router's K chunks (a view)
    xs,  # [T, 8, 512] bf16: the same rows as the shared W13's K chunks (a distinct view)
    xk,  # [T, 8, 8, 64] bf16: the same rows as the routed W13's (chunk, K block) (a distinct view)
    router_weight,  # [288, 16, 256] bf16 (a view)
    bias,  # [288] fp32: e_score_correction_bias
    w13,  # [288, 8, 8, 1024, 64] bf16: TRT-LLM BF16 block layout [E, 64 K blocks, 1024, 64] in K chunks (a view)
    w2,  # [288, 8, 32, 128, 64] bf16: [E, 8 K blocks, 4096, 64] in 128-row tiles (a view)
    sw13,  # [1024, 8, 512] bf16: shared gate | up in K chunks (a view)
    sw2,  # [4096, 512] bf16: shared down
    residual,  # [T, 4096] bf16
    norm_w,  # [4096] bf16: shared_head.norm
    # scratch
    router_partials,  # [T, 16, 512] fp32
    topk_ids,  # [T, 8] int32
    topk_weights,  # [T, 8] fp32
    s_parts,  # [8, T, 1024] fp32: shared W13 K-chunk partials
    w13_parts_a,  # [4, 8, T, 8, 128] fp32: routed W13 partials of picks 0-3 [slot, tile, token, chunk, row]
    w13_parts_b,  # [4, 8, T, 8, 128] fp32: picks 4-7
    shared_act,  # [T, 512] bf16
    routed_act_a,  # [4, T, 512] bf16: pick j's activation (picks 0-3)
    routed_act_b,  # [4, T, 512] bf16: picks 4-7
    shared_output,  # [T, 4096] bf16
    routed_output_a,  # [8, 4, T, 512] bf16: each pick's W2 row, part-major (eighth of the hidden, pick, token)
    routed_output_b,  # [8, 4, T, 512] bf16: picks 4-7
    symmetric_partial,  # [T, 4096] bf16 symmetric memory: this rank's MoE partial
    norm_partial,  # [T, 32] fp32
    norm_input,  # [T, 4096] fp32
    output,  # [T, 4096] bf16 out: the draft hidden state (shared_head.norm output)
    group_name: hl.ProcessGroupName,
) -> None:
    tokens, router_chunks, router_k = xr.shape
    experts = router_weight.size(0)
    w13_chunks, chunk_blocks, twice_intermediate = w13.shape[1:4]
    intermediate = twice_intermediate // 2
    w2_blocks, w2_tiles = w2.shape[1:3]
    hidden = w2_tiles * 128
    # Literals: a module-level int reaches the kernel as a runtime scalar.
    lanes = 512
    top_k = 8
    w2_parts = 8
    eps = 1e-5
    assert (
        tokens == 1
        and experts <= lanes
        and intermediate == 512
        and w2_blocks * 64 == intermediate
    )
    assert routed_output_a.size() == (
        w2_parts,
        w13_parts_a.size(0),
        tokens,
        hidden // w2_parts,
    )
    split = w13_parts_a.size(0)  # picks of half a
    rest = top_k - split
    w13_tiles = twice_intermediate // 128
    part_tiles = w2_tiles // w2_parts
    act_blocks_a = routed_act_a.view(split, tokens, w2_blocks, 64)
    act_blocks_b = routed_act_b.view(rest, tokens, w2_blocks, 64)
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

    # 2. routing (one token)
    for rk_t in hl.tile(tokens, block_size=1):
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
        rk_id, rk_w, _rk_u = route(
            torch.sum(rk_p, dim=0),
            rk_bias,
            hl.arange(top_k) * 0 + rk_tok < tokens,
            experts,
            lanes,
            2.5,
        )
        topk_ids[rk_t, :] = rk_id[None, :]
        topk_weights[rk_t, :] = rk_w[None, :]

    # 3. shared W13 per (128-row tile, K chunk): FP32 partials (its weights
    # stream while the routing runs)
    for sp_n, sp_c, sp_t in hl.tile(
        [twice_intermediate, w13_chunks, tokens], block_size=[128, 1, 16]
    ):
        sp_acc = hl.zeros([sp_n, sp_t], dtype=torch.float32)
        for sp_k in hl.tile(sw13.size(2), block_size=64):
            sp_acc = hl.dot(
                sw13[sp_n, sp_c.begin, sp_k], xs[sp_t, sp_c.begin, sp_k].T, acc=sp_acc
            )
        s_parts[sp_c.begin, sp_t, sp_n] = sp_acc.T

    # 4. shared activation per 128 intermediates: gate / up chunk sums (BF16:
    # the GEMM's output), clamped, bf16(silu(gate) * up) (SGLang's
    # silu_and_mul_clamp)
    for sa_t, sa_j in hl.tile([tokens, intermediate], block_size=[1, 128]):
        sa_tok = sa_t.begin
        sa_gate = s_parts[0, sa_tok, sa_j]
        sa_up = s_parts[0, sa_tok, sa_j.index + intermediate]
        for sa_c in hl.static_range(1, w13_chunks):
            sa_gate = sa_gate + s_parts[sa_c, sa_tok, sa_j]
            sa_up = sa_up + s_parts[sa_c, sa_tok, sa_j.index + intermediate]
        sa_g = torch.clamp(sa_gate.to(torch.bfloat16).to(torch.float32), max=10.0)
        sa_u = torch.clamp(
            sa_up.to(torch.bfloat16).to(torch.float32), min=-10.0, max=10.0
        )
        shared_act[sa_tok, sa_j] = (sa_g * torch.sigmoid(sa_g) * sa_u).to(
            torch.bfloat16
        )

    # 5. shared W2 per 128-row tile, K = 512 -> BF16 (the shared expert runs
    # while the routing does)
    for sd_n, sd_t in hl.tile([hidden, tokens], block_size=[128, 16]):
        sd_acc = hl.zeros([sd_n, sd_t], dtype=torch.float32)
        for sd_k in hl.tile(intermediate, block_size=64):
            sd_acc = hl.dot(sw2[sd_n, sd_k], shared_act[sd_t, sd_k].T, acc=sd_acc)
        shared_output[sd_t, sd_n] = sd_acc.T.to(torch.bfloat16)

    # The routed experts in two halves of the slots (a: picks 0-3, b: picks 4-7;
    # their own scratch, so no false dependency orders them), so the claim order
    # interleaves them: half a's SwiGLU is claimed behind its W13 tasks and half
    # a's W2 behind half b's W13 (ready work for the workers the last W13 wave
    # leaves idle); only half b's SwiGLU and W2 trail the stream. 6 / 8. routed
    # W13 per (slot, 128-row tile, K chunk): the slot's expert, FP32 partials of
    # the physical rows
    for pa_j, pa_tile, pa_c, pa_t in hl.tile(
        [split, w13_tiles, w13_chunks, tokens], block_size=[1, 1, 1, 16]
    ):
        pa_expert = topk_ids[0, pa_j.begin]
        pa_acc = hl.zeros([128, pa_t], dtype=torch.float32)
        for pa_kb in hl.tile(chunk_blocks, block_size=1):
            pa_acc = hl.dot(
                w13[
                    pa_expert,
                    pa_c.begin,
                    pa_kb.begin,
                    pa_tile.begin * 128 + hl.arange(128),
                    :,
                ],
                xk[pa_t, pa_c.begin, pa_kb.begin, :].T,
                acc=pa_acc,
            )
        w13_parts_a[pa_j.begin, pa_tile.begin, pa_t, pa_c.begin, :] = pa_acc.T
    # 7 / 10. routed SwiGLU per (slot, tile): the chunk sums in order,
    # SiLU(gate) * up in FP32, BF16, at the tile's 64 intermediates
    for ga_j, ga_tile in hl.tile([split, w13_tiles], block_size=[1, 1]):
        ga_pre = w13_parts_a[ga_j.begin, ga_tile.begin, 0, 0, :]
        for ga_c in hl.static_range(1, w13_chunks):
            ga_pre = ga_pre + w13_parts_a[ga_j.begin, ga_tile.begin, 0, ga_c, :]
        # [block 4, half 2, up/gate 2, m 8] -> [block, m, half, up/gate]
        ga_up, ga_gate = hl.split(ga_pre.reshape(4, 2, 2, 8).permute(0, 3, 1, 2))
        ga_act = (ga_gate * torch.sigmoid(ga_gate) * ga_up).reshape(64)
        routed_act_a[ga_j.begin, 0, ga_tile.begin * 64 + hl.arange(64)] = ga_act.to(
            torch.bfloat16
        )
    for pb_j, pb_tile, pb_c, pb_t in hl.tile(
        [rest, w13_tiles, w13_chunks, tokens], block_size=[1, 1, 1, 16]
    ):
        pb_expert = topk_ids[0, split + pb_j.begin]
        pb_acc = hl.zeros([128, pb_t], dtype=torch.float32)
        for pb_kb in hl.tile(chunk_blocks, block_size=1):
            pb_acc = hl.dot(
                w13[
                    pb_expert,
                    pb_c.begin,
                    pb_kb.begin,
                    pb_tile.begin * 128 + hl.arange(128),
                    :,
                ],
                xk[pb_t, pb_c.begin, pb_kb.begin, :].T,
                acc=pb_acc,
            )
        w13_parts_b[pb_j.begin, pb_tile.begin, pb_t, pb_c.begin, :] = pb_acc.T
    # 9 / 11. routed W2 per (slot, 128-row tile): the full K, BF16, at the
    # hidden columns of the tile's physical rows, part-major
    for qa_j, qa_n, qa_t in hl.tile([split, w2_tiles, tokens], block_size=[1, 1, 16]):
        qa_expert = topk_ids[0, qa_j.begin]
        qa_acc = hl.zeros([128, qa_t], dtype=torch.float32)
        for qa_kb in hl.tile(w2_blocks, block_size=1):
            qa_acc = hl.dot(
                w2[qa_expert, qa_kb.begin, qa_n.begin, :, :],
                act_blocks_a[qa_j.begin, qa_t, qa_kb.begin, :].T,
                acc=qa_acc,
            )
        # Physical row 32 b + 8 c + m holds hidden 32 b + 4 m + c.
        qa_phys = hl.arange(128)
        qa_hidden = (qa_phys // 32) * 32 + (qa_phys % 8) * 4 + (qa_phys % 32) // 8
        routed_output_a[
            qa_n.begin // part_tiles,
            qa_j.begin,
            qa_t,
            (qa_n.begin % part_tiles) * 128 + qa_hidden,
        ] = qa_acc.T.to(torch.bfloat16)
    for gb_j, gb_tile in hl.tile([rest, w13_tiles], block_size=[1, 1]):
        gb_pre = w13_parts_b[gb_j.begin, gb_tile.begin, 0, 0, :]
        for gb_c in hl.static_range(1, w13_chunks):
            gb_pre = gb_pre + w13_parts_b[gb_j.begin, gb_tile.begin, 0, gb_c, :]
        # [block 4, half 2, up/gate 2, m 8] -> [block, m, half, up/gate]
        gb_up, gb_gate = hl.split(gb_pre.reshape(4, 2, 2, 8).permute(0, 3, 1, 2))
        gb_act = (gb_gate * torch.sigmoid(gb_gate) * gb_up).reshape(64)
        routed_act_b[gb_j.begin, 0, gb_tile.begin * 64 + hl.arange(64)] = gb_act.to(
            torch.bfloat16
        )
    for qb_j, qb_n, qb_t in hl.tile([rest, w2_tiles, tokens], block_size=[1, 1, 16]):
        qb_expert = topk_ids[0, split + qb_j.begin]
        qb_acc = hl.zeros([128, qb_t], dtype=torch.float32)
        for qb_kb in hl.tile(w2_blocks, block_size=1):
            qb_acc = hl.dot(
                w2[qb_expert, qb_kb.begin, qb_n.begin, :, :],
                act_blocks_b[qb_j.begin, qb_t, qb_kb.begin, :].T,
                acc=qb_acc,
            )
        # Physical row 32 b + 8 c + m holds hidden 32 b + 4 m + c.
        qb_phys = hl.arange(128)
        qb_hidden = (qb_phys // 32) * 32 + (qb_phys % 8) * 4 + (qb_phys % 32) // 8
        routed_output_b[
            qb_n.begin // part_tiles,
            qb_j.begin,
            qb_t,
            (qb_n.begin % part_tiles) * 128 + qb_hidden,
        ] = qb_acc.T.to(torch.bfloat16)

    # The tail's code toward L2 again: the experts' weight stream evicted it.
    hl.prefetch_code()
    # 12. combine per eighth of the hidden: the picks' rows in pick order, BF16
    # weights, FP32 -> BF16 (the finalize), + the shared output in BF16 -> this
    # rank's partial
    for c_t, c_n in hl.tile([tokens, hidden], block_size=[1, 512]):
        c_tok = c_t.begin
        c_cols = c_n.index - c_n.begin
        c_routed = hl.zeros([c_n], dtype=torch.float32)
        # top_k picks, in order: half a's, then half b's
        for c_k in hl.static_range(split):
            c_w = topk_weights[c_tok, c_k].to(torch.bfloat16).to(torch.float32)
            c_routed = c_routed + c_w * routed_output_a[c_n.id, c_k, c_tok, c_cols].to(
                torch.float32
            )
        for c_k in hl.static_range(rest):
            c_w = topk_weights[c_tok, split + c_k].to(torch.bfloat16).to(torch.float32)
            c_routed = c_routed + c_w * routed_output_b[c_n.id, c_k, c_tok, c_cols].to(
                torch.float32
            )
        c_sum = c_routed.to(torch.bfloat16).to(torch.float32) + shared_output[
            c_tok, c_n
        ].to(torch.float32)
        symmetric_partial[c_tok, c_n] = c_sum.to(torch.bfloat16)

    hl.pdl_launch_dependents()
    # 13. all-reduce + residual per (token, 128 columns): rank order (FP32),
    # BF16, + residual, BF16.
    for ar_t, ar_n in hl.tile([tokens, hidden], block_size=[1, 128]):
        ar_tok = ar_t.begin
        ar_sum = hl.zeros([ar_n], dtype=torch.float32)
        for ar_r in hl.static_range(len(remote_partials)):
            ar_sum = ar_sum + remote_partials[ar_r][ar_tok, ar_n].to(torch.float32)
        ar_x = ar_sum.to(torch.bfloat16).to(torch.float32) + residual[ar_tok, ar_n].to(
            torch.float32
        )
        ar_f = ar_x.to(torch.bfloat16).to(torch.float32)
        norm_input[ar_tok, ar_n] = ar_f
        norm_partial[ar_tok, ar_n.id] = torch.sum(ar_f * ar_f)
    # 14. shared_head.norm per (token, 512 columns): bf16((x * w) * r).
    for nm_t, nm_n in hl.tile([tokens, hidden], block_size=[1, 512]):
        nm_tok = nm_t.begin
        nm_scale = torch.rsqrt(torch.sum(norm_partial[nm_tok, :]) / hidden + eps)
        output[nm_tok, nm_n] = (
            (norm_input[nm_tok, nm_n] * norm_w[nm_n].to(torch.float32)) * nm_scale
        ).to(torch.bfloat16)


def mk_config(bound) -> helion.Config:
    """The dot operands of ``RING_LOADS`` through the TMA ring, every other
    access by pointer; one persistent CTA per SM claiming tasks root by root;
    the ring roots and their loops warp-specialized (``ring_config_fields``'
    ``"root_loops"``); the all-reduce in band (Lamport)."""
    default = bound.config_spec.default_config().config
    fields = ring_config_fields(bound, RING_LOADS, warp_specialize="root_loops")
    return helion.Config(
        indexing=fields["indexing"],
        pid_type="persistent_blocked",
        num_sm_multiplier=1,
        num_stages=6,
        num_warps=4,
        cross_loop_pipeline="dynamic",
        range_warp_specializes=fields["range_warp_specializes"],
        cross_loop_inband_protocol=["lamport"]
        * len(default.get("cross_loop_inband_protocol", [])),
    )


@functools.cache
def scratch(device: torch.device, tokens: int) -> dict[str, torch.Tensor]:
    """The kernel's scratch buffers on ``device``, one set for every call."""
    f32, bf16 = torch.float32, torch.bfloat16
    n = _setup.MOE_SHAPE.intermediate
    halves = (("a", SPLIT), ("b", TOP_K - SPLIT))

    def z(*shape: int, dtype=f32):
        return torch.zeros(*shape, device=device, dtype=dtype)

    return {
        "router_partials": z(tokens, ROUTER_CHUNKS, ROUTE_LANES),
        "topk_ids": z(tokens, TOP_K, dtype=torch.int32),
        "topk_weights": z(tokens, TOP_K),
        "s_parts": z(W13_CHUNKS, tokens, 2 * n),
        **{
            f"w13_parts_{h}": z(k, 2 * n // 128, tokens, W13_CHUNKS, 128)
            for h, k in halves
        },
        "shared_act": z(tokens, n, dtype=bf16),
        **{f"routed_act_{h}": z(k, tokens, n, dtype=bf16) for h, k in halves},
        "shared_output": z(tokens, HIDDEN, dtype=bf16),
        "norm_partial": z(tokens, HIDDEN // 128),
        "norm_input": z(tokens, HIDDEN),
        **{
            f"routed_output_{h}": z(W2_PARTS, k, tokens, HIDDEN // W2_PARTS, dtype=bf16)
            for h, k in halves
        },
    }


def symmetric_partial_spec(tokens: int) -> tuple[tuple[int, ...], torch.dtype]:
    """The (shape, dtype) of the symmetric buffer of this rank's MoE partial."""
    return (tokens, HIDDEN), torch.bfloat16


def kernel_args(
    x, w: dict, residual, norm_w, symmetric_partial, output, group_name
) -> tuple:
    """``glmf_mtp_moe_tp4``'s arguments: x [T, 4096] BF16 (the MoE input), ``w`` the
    weights (``_setup.mtp_moe_weights``' names: router [288, 4096], bias [288] FP32,
    w13 [288, 64, 1024, 64], w2 [288, 8, 4096, 64] in the TRT-LLM block layout, sw13
    [1024, 4096], sw2 [4096, 512]), residual [T, 4096] BF16, shared_head.norm's
    weight; writes ``output`` [T, 4096] BF16."""
    tokens = x.shape[0]
    assert x.is_contiguous()
    sc = scratch(x.device, tokens)
    e = w["w13"].shape[0]
    kb = w["w13"].shape[1]
    return (
        x.view(tokens, ROUTER_CHUNKS, -1),
        x.view(tokens, W13_CHUNKS, -1),
        x.view(tokens, W13_CHUNKS, kb // W13_CHUNKS, 64),
        w["router"].view(e, ROUTER_CHUNKS, -1),
        w["bias"],
        w["w13"].view(e, W13_CHUNKS, kb // W13_CHUNKS, w["w13"].shape[2], 64),
        w["w2"].view(e, w["w2"].shape[1], w["w2"].shape[2] // 128, 128, 64),
        w["sw13"].view(w["sw13"].shape[0], W13_CHUNKS, -1),
        w["sw2"],
        residual,
        norm_w,
        sc["router_partials"],
        sc["topk_ids"],
        sc["topk_weights"],
        sc["s_parts"],
        sc["w13_parts_a"],
        sc["w13_parts_b"],
        sc["shared_act"],
        sc["routed_act_a"],
        sc["routed_act_b"],
        sc["shared_output"],
        sc["routed_output_a"],
        sc["routed_output_b"],
        symmetric_partial,
        sc["norm_partial"],
        sc["norm_input"],
        output,
        group_name,
    )


def compile_kernel(args: tuple):
    """``glmf_mtp_moe_tp4`` compiled for ``args`` (once per binding)."""
    return _setup.compiled(glmf_mtp_moe_tp4, mk_config, args)


# ---------------------------- check and timing (torch.distributed.run, 4 ranks)


def _case(w: dict, seed: int, partial, group_name) -> tuple:
    """A draft row, a residual of the MoE output's scale (so that the check sees the
    MoE's bits through the norm), a norm weight, the output, the kernel's
    arguments."""
    device = w["bias"].device
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(1, HIDDEN, generator=g, device=device).bfloat16()
    residual = (0.02 * torch.randn(1, HIDDEN, generator=g, device=device)).bfloat16()
    norm_w = (1 + 0.1 * torch.randn(HIDDEN, generator=g, device=device)).bfloat16()
    out = torch.empty(1, HIDDEN, device=device, dtype=torch.bfloat16)
    return (
        x,
        residual,
        norm_w,
        out,
        kernel_args(x, w, residual, norm_w, partial, out, group_name),
    )


def correctness_check(group: dist.ProcessGroup, verbose: bool = True) -> None:
    """Rank-dependent weights (``_setup.mtp_moe_weights``) and rows; four
    back-to-back launches with changing inputs (no host synchronization), each
    against ``_setup.mtp_moe_reference``: the top-8 ids as sets, this rank's
    partial, and the output (the all-reduce of every rank's reference partial in
    rank order, the residual add, the RMSNorm; ``_qwen3_5_tp4.all_reduce_norm``);
    then the last again with NaN scratch: the same bits."""
    rank = group.rank()
    device = torch.device("cuda", torch.cuda.current_device())
    w = _setup.mtp_moe_weights(seed=rank, device=device)
    partial = symmetric_buffer(*symmetric_partial_spec(1), group.group_name, device)
    cases = [_case(w, 100 + seed, partial, group.group_name) for seed in range(4)]
    torch.cuda.synchronize()
    dist.barrier()
    ids, partials = [], []
    for case in cases:
        compile_kernel(case[-1])(*case[-1])
        ids.append(scratch(device, 1)["topk_ids"].clone())
        partials.append(partial.clone())
    torch.cuda.synchronize()
    rows, ok = [], True
    for i, (x, residual, norm_w, out, _) in enumerate(cases):
        reference, ref_ids = _setup.mtp_moe_reference(x, w)
        same_ids = torch.equal(ids[i].long().sort(-1).values, ref_ids.sort(-1).values)
        rows.append(
            f"  case {i} top-k ids {'equal' if same_ids else 'DIFFER  <-- FAIL'}"
        )
        ok &= same_ids
        ok &= report(rows, f"case {i} partial", partials[i], reference, 1e-2)
        normed, _ = all_reduce_norm(reference, residual, norm_w, _setup.NORM_EPS)
        ok &= report(rows, f"case {i} output", out, normed, 1e-2)
    *_, out, args = cases[-1]
    first = out.clone()
    _setup.poison([out, *scratch(device, 1).values()])
    dist.barrier()
    compile_kernel(args)(*args)
    torch.cuda.synchronize()
    repeat = same(out, first)
    rows.append(
        f"  {'NaN scratch repeat':<22} {'bitwise' if repeat else 'DIFFERS  <-- FAIL'}"
    )
    ok &= repeat
    if verbose or not ok:
        say(f"rank{rank} T=1\n" + "\n".join(rows))
    verdict(ok, group, "glmf_mtp_moe_tp4")


def benchmark(group: dist.ProcessGroup, verbose: bool = True) -> dict[str, Any]:
    """Cold-weight timing: ``LAYERS`` distinct weight copies (default 2, 3.6 GB
    each) back to back, ``REPS`` times (default 8) in one CUDA graph."""
    rank = group.rank()
    device = torch.device("cuda", torch.cuda.current_device())
    layers = int(os.environ.get("LAYERS", "2"))
    reps = int(os.environ.get("REPS", "8"))
    partial = symmetric_buffer(*symmetric_partial_spec(1), group.group_name, device)
    args = []
    for i in range(layers):
        w = _setup.mtp_moe_weights(seed=1000 * i + rank, device=device)
        del w["canon"]
        args.append(_case(w, i, partial, group.group_name)[-1])
    torch.cuda.empty_cache()

    def step() -> None:
        for a in args:
            compile_kernel(a)(*a)

    median, minimum = graph_time(step, layers, reps)
    if verbose:
        say(f"rank{rank} T=1 us/launch med {median:.2f} min {minimum:.2f}")
    return {1: {"latency_us": median, "min_latency_us": minimum}}


def main(verbose: bool = True, check_only: bool = False) -> dict[str, Any]:
    """The four-rank check, then the cold-weight timing (``check_only``: the
    check alone)."""
    return run_main(
        "glmf_mtp_moe_tp4", correctness_check, benchmark, verbose, check_only
    )


if __name__ == "__main__":
    main(check_only="--check" in sys.argv)
