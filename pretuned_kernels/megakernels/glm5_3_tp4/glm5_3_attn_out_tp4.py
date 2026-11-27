# ruff: noqa: ANN001, ANN201, ANN202
# pyrefly: ignore-errors
"""GLM-5.3 (GlmMoeDsa) attention output side for a TP4 speculative-verify
window (one request, T <= 8 tokens), from the FP8 MLA queries and the indexer's
top-k to the MoE's input, as one persistent Gluon megakernel.

Per rank: 16 heads. The roots:

- splits: per (token, split of 64 top-k picks; 128 past 4 rows), the sparse
  MLA over the picked slots of the raw FP8 MLA cache (``_setup``) with swapped
  operands (slots as the MMA rows, heads as its columns): QK over the 576 dims,
  the softmax (log2 domain), P (E4M3, times 256; 448 for 128 picks) V over the
  512 latent dims; BF16 partials and the log2-sum-exp2 per head (FP32): stock's
  trtllm-gen sparse MLA stores its multi-CTA KV-split partials so too.
- merge: per (token, 32 latent columns), the splits combined in split order,
  rounded to BF16 (the trtllm kernel's output).
- value up-projection: per (head, 128 value rows), W_UV over the 512 latent
  dims, BF16 (SGLang's bmm).
- o_proj: BF16 [6144, 4096] in three K chunks (5, 5 and 6 heads), a root each:
  the first two store FP32 partials, the last chunk's task adds them to its own
  in chunk order and publishes the BF16 sum (SGLang's BF16 o_proj output) into
  this rank's symmetric partial.
- the all-reduce (rank order, FP32) + residual and the post-attention RMSNorm
  (``qwen3_5_moe_tp4``'s tail).

It serves the target's verify window and the MTP draft's attention (one row a
decode step: 32-pick splits; the window in the draft extend). The splits are
merged and the partials summed in a fixed order (no float atomics): the outputs
are run-to-run deterministic.

``correctness_check()`` compares every rank with the torch reference
(``_setup``) at every served window; ``main()`` runs it, then times the kernel
with cold weights::

    NVSHMEM_DISABLE_CUDA_VMM=1 PYTHONPATH=. \\
      python -m torch.distributed.run --standalone --nproc-per-node=4 \\
      pretuned_kernels/megakernels/glm5_3_tp4/glm5_3_attn_out_tp4.py [--check]
"""

from __future__ import annotations

import math
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
from pretuned_kernels.megakernels.glm5_3_tp4 import _setup
from pretuned_kernels.megakernels.glm5_3_tp4._setup import NORM_EPS
import torch
import torch.distributed as dist

import helion
import helion.language as hl

# The weight streams ride the TMA ring (``_ring_config.ring_config_fields(bound,
# RING_LOADS)``: W_UV and o_proj, with their activation operands); every other
# memory op, the gathered cache rows among them, is a pointer access. Numbered
# for 4 ranks (the all-reduce loads each rank's partial after every ring slot).
RING_LOADS = ("w_uv", "attn", "op_w0", "op_x0", "op_w1", "op_x1", "op_w2", "op_x2")
_RING_INDEXING = (10, 11, 13, 14, 16, 17, 19, 20)
_MEMORY_OPS = 40


def _indexing():
    return [
        "tensor_descriptor" if op in _RING_INDEXING else "pointer"
        for op in range(_MEMORY_OPS)
    ]


CONFIG = {
    "block_sizes": [16, 64, 16, 64, 16, 64, 16, 64],
    "indexing": _indexing(),
    "range_warp_specializes": [None] * 2 + [True, None] * 4 + [None] * 2,
    # W_UV's and o_proj's last weight tiles stream into tensor memory ahead of
    # their tasks' waits.
    "cross_loop_tmem_prefetch": [True] * 4,
    "num_warps": 4,
    "pid_type": "persistent_blocked",
    "cross_loop_pipeline": "dynamic",
    "num_sm_multiplier": 1,
    "num_stages": 6,
    "cross_loop_inband_protocol": ["lamport"],
}


@helion.kernel(
    static_shapes=False,
    backend="gluon",
    triton_do_not_specialize=False,
    config=helion.Config(**CONFIG),
    ignore_warnings=[helion.exc.TensorOperationInWrapper, helion.exc.FastMathUnused],
)
def glm5_3_attn_out_tp4(
    q,  # [T, 16, 576] e4m3: the absorbed queries (latent 512 | rope 64)
    kv_cache,  # [slots, 576] e4m3: one layer's raw MLA cache
    topk,  # [T, 2048] int32: each row's picked slots (-1: none)
    w_uv,  # [16, 256, 512] bf16: W_UV per head (K contiguous)
    o_proj,  # [6144, 4096] bf16
    scale_log2: float,  # softmax scale * log2(e)
    residual,  # [T, 6144] bf16
    norm_weight,  # [6144] bf16: post_attention_layernorm
    symmetric_partial,  # [T, 6144] bf16 symmetric memory
    group_name: hl.ProcessGroupName,
):
    hl.specialize(q.size())
    hl.specialize(q.stride())
    hl.specialize(kv_cache.stride())
    hl.specialize(topk.size())
    hl.specialize(topk.stride())
    hl.specialize(w_uv.size())
    hl.specialize(w_uv.stride())
    hl.specialize(o_proj.size())
    hl.specialize(o_proj.stride())
    hl.specialize(residual.size())
    hl.specialize(residual.stride())
    hl.specialize(norm_weight.size())
    hl.specialize(norm_weight.stride())
    hl.specialize(symmetric_partial.size())
    hl.specialize(symmetric_partial.stride())

    rows, heads, qk_dim = q.shape
    top_k = topk.size(1)
    v_head, kv_lora = w_uv.shape[1:]
    hidden, q_width = o_proj.shape
    rope_dim = qk_dim - kv_lora
    assert rows <= 8 and heads == 16 and qk_dim == 576 and kv_lora == 512
    assert q_width == heads * v_head
    dev = residual.device
    # Literals: a module-level int reaches the kernel as a runtime scalar.
    # Splits of 64 picks (32 a token) up to 4 tokens; past 4, 128 picks (16 a
    # token): the window's split tasks stay within one wave of workers. One
    # token (the MTP draft's decode step): 32 picks (64 splits), its gathers
    # latency bound (W=1: 26.8 us vs 31.3 at 64, 33.2 at 128). P's E4M3 scale:
    # the format's maximum for the longer splits (their smaller probabilities
    # keep more mantissa bits). (Arithmetic on the specialized row count: a
    # branch merges into a symbol.)
    longer = rows // 5
    split_n = 64 * (1 + longer) // (1 + 2 // (rows + 1))
    splits = top_k // split_n
    p_scale = 256.0 + 192.0 * longer
    assert splits * split_n == top_k
    # Latent-major split outputs: a split task stores its [512, 16] O^T as is.
    att_ws = torch.empty(
        (splits, rows, kv_lora, heads), dtype=torch.bfloat16, device=dev
    )
    lse_ws = torch.empty((splits, rows, heads), dtype=torch.float32, device=dev)
    attn = torch.empty((rows, heads, kv_lora), dtype=torch.bfloat16, device=dev)
    attn_v = torch.empty((rows, heads, v_head), dtype=torch.bfloat16, device=dev)
    # o_proj in three K chunks of 5, 5 and 6 heads: 3 x 48 tasks fit one per
    # worker (4 x 48 left 40 of them a second task, after the attention).
    op_chunks = 3
    op_k1 = 5 * v_head
    op_k2 = 10 * v_head
    attn_v_rows = attn_v.view(rows, q_width)
    op_w0 = o_proj[:, :op_k1]
    op_w1 = o_proj[:, op_k1:op_k2]
    op_w2 = o_proj[:, op_k2:]
    op_x0 = attn_v_rows[:, :op_k1]
    op_x1 = attn_v_rows[:, op_k1:op_k2]
    op_x2 = attn_v_rows[:, op_k2:]
    # The first two chunks' partials (the last chunk's task adds them).
    op_partials = torch.empty(
        (op_chunks - 1, rows, hidden), dtype=torch.float32, device=dev
    )

    remote_partials = torch.ops.symm_mem.get_remote_tensors(
        symmetric_partial, group_name
    )
    output = torch.empty((rows, hidden), dtype=torch.bfloat16, device=dev)
    residual_output = torch.empty_like(output)
    communication_n = 1024
    norm_tasks = hidden // communication_n
    norm_partial = torch.empty((rows, norm_tasks), dtype=torch.float32, device=dev)
    norm_input = torch.empty((rows, hidden), dtype=torch.float32, device=dev)

    hl.pdl_wait()
    hl.prefetch_code()

    # Sparse MLA splits: 64 picks of one token's top-k, one block (no online
    # rescaling). Swapped operands (the picked slots as the MMA's rows, the 16
    # heads as its columns): S^T = K q^T over the latent and the rope dims,
    # O^T = V^T P^T.
    for sp_t, sp_s in hl.tile([rows, splits], block_size=[1, 1]):
        sp_token = sp_t.begin
        sp_q_lat = q[sp_token, :, :kv_lora]
        sp_q_rope = q[sp_token, :, kv_lora:]
        slot = topk[sp_token, sp_s.begin * split_n + hl.arange(split_n)]
        valid = slot >= 0
        safe = torch.where(valid, slot, 0)
        key_lat = hl.load(kv_cache, [safe[:, None], hl.arange(kv_lora)[None, :]])
        key_rope = hl.load(
            kv_cache, [safe[:, None], (kv_lora + hl.arange(rope_dim))[None, :]]
        )
        scores = hl.dot(key_lat, sp_q_lat.T, out_dtype=torch.float32)
        scores = hl.dot(key_rope, sp_q_rope.T, acc=scores, out_dtype=torch.float32)
        scores = torch.where(valid[:, None], scores * scale_log2, float("-inf"))
        sp_max = torch.amax(scores, dim=0)
        safe_max = torch.where(sp_max == float("-inf"), 0.0, sp_max)
        prob = torch.exp2(scores - safe_max[None, :])
        sp_sum = torch.sum(prob, dim=0)
        # P * p_scale in E4M3 against the FP8 latent (divided out below).
        acc = hl.dot(
            key_lat.T, (prob * p_scale).to(torch.float8_e4m3fn), out_dtype=torch.float32
        )
        seen = sp_sum > 0.0
        safe_sum = torch.where(seen, sp_sum, 1.0)
        att_ws[sp_s.begin, sp_token, :, :] = (acc / (p_scale * safe_sum[None, :])).to(
            torch.bfloat16
        )
        lse_ws[sp_s.begin, sp_token, :] = torch.where(
            seen, safe_max + torch.log2(safe_sum), float("-inf")
        )

    # Merge per (token, 32 latent columns): the splits in split order.
    for mg_t, mg_c in hl.tile([rows, kv_lora], block_size=[1, 32]):
        mg_token = mg_t.begin
        mg_lse = lse_ws[:, mg_token, :]
        mg_best = torch.amax(mg_lse, dim=0)
        mg_best = torch.where(mg_best == float("-inf"), 0.0, mg_best)
        mg_w = torch.exp2(mg_lse - mg_best[None, :])
        mg_vals = att_ws[:, mg_token, mg_c, :].to(torch.float32)
        mg_out = torch.sum(mg_w[:, None, :] * mg_vals, dim=0)
        mg_den = torch.sum(mg_w, dim=0)
        attn[mg_token, :, mg_c] = (
            mg_out / torch.where(mg_den > 0, mg_den, 1.0)[None, :]
        ).T.to(torch.bfloat16)

    # Value up-projection per (head, 128 value rows).
    for uv_h, uv_n, uv_t in hl.tile([heads, v_head, rows], block_size=[1, 128, None]):
        uv_acc = hl.zeros([uv_n, uv_t], dtype=torch.float32)
        for uv_k in hl.tile(kv_lora):
            uv_acc = hl.dot(
                w_uv[uv_h.begin, uv_n, uv_k],
                attn[uv_t, uv_h.begin, uv_k].T,
                acc=uv_acc,
            )
        attn_v[uv_t, uv_h.begin, uv_n] = uv_acc.T.to(torch.bfloat16)

    # o_proj: the three K chunks, a root each.
    for o0_n, o0_t in hl.tile([hidden, rows], block_size=[128, None]):
        # The first chunk's tasks run during the attention: warm L2 with their
        # rows of the later chunks (streamed after the value up-projection).
        hl.prefetch(op_w1, [o0_n, slice(None)])
        hl.prefetch(op_w2, [o0_n, slice(None)])
        o0_acc = hl.zeros([o0_n, o0_t], dtype=torch.float32)
        for o0_k in hl.tile(op_k1):
            o0_acc = hl.dot(op_w0[o0_n, o0_k], op_x0[o0_t, o0_k].T, acc=o0_acc)
        op_partials[0, o0_t, o0_n] = o0_acc.T
    for o1_n, o1_t in hl.tile([hidden, rows], block_size=[128, None]):
        o1_acc = hl.zeros([o1_n, o1_t], dtype=torch.float32)
        for o1_k in hl.tile(op_k2 - op_k1):
            o1_acc = hl.dot(op_w1[o1_n, o1_k], op_x1[o1_t, o1_k].T, acc=o1_acc)
        op_partials[1, o1_t, o1_n] = o1_acc.T
    for o2_n, o2_t in hl.tile([hidden, rows], block_size=[128, None]):
        o2_acc = hl.zeros([o2_n, o2_t], dtype=torch.float32)
        for o2_k in hl.tile(q_width - op_k2):
            o2_acc = hl.dot(op_w2[o2_n, o2_k], op_x2[o2_t, o2_k].T, acc=o2_acc)
        # Publish from the last chunk's task: the BF16 sum of the chunks in
        # order (SGLang's BF16 o_proj output).
        o2_sum = op_partials[0, o2_t, o2_n] + op_partials[1, o2_t, o2_n] + o2_acc.T
        symmetric_partial[o2_t, o2_n] = o2_sum.to(torch.bfloat16)

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
        norm_weights = norm_weight[norm_tile_n].to(torch.float32)
        norm_scale = torch.rsqrt(
            torch.sum(norm_partial[norm_token, :]) / hidden + NORM_EPS
        )
        output[norm_token, norm_tile_n] = (
            norm_input[norm_token, norm_tile_n] * norm_scale * norm_weights
        ).to(torch.bfloat16)
    return output, residual_output, attn, attn_v


# The served windows: the target's verify window (EAGLE 5/1/6; the MTP draft
# extend's is 6 too) and the MTP draft's decode step.
TOKENS = (6, 1)
SCALE_LOG2 = _setup.SOFTMAX_SCALE * math.log2(math.e)


def correctness_check(group: dist.ProcessGroup, verbose: bool = True) -> None:
    """Every rank against ``_setup.out_reference`` per served window, for a
    short context (every slot picked, the rest -1) and a long one (2048 random
    picks): the sparse MLA, the value rows, the residual output (the ranks'
    reference partials in rank order plus the residual) and the normed output
    within 2e-2 relative RMS (P in E4M3 over 128-pick splits above 4 rows:
    ~1.0e-2 at the short context), and a second launch bitwise equal."""
    rank = group.rank()
    device = torch.device("cuda", torch.cuda.current_device())
    shape = _setup.AttnShape()
    w = _setup.out_weights(shape, seed=100 + rank, device=device)
    ok = True
    for tokens in TOKENS:
        g = torch.Generator(device=device)
        g.manual_seed(7)
        residual = torch.randn(tokens, shape.hidden, generator=g, device=device).to(
            torch.bfloat16
        )
        norm_weight = (
            1 + 0.1 * torch.randn(shape.hidden, generator=g, device=device)
        ).to(torch.bfloat16)
        partials = symmetric_partials(tokens, shape.hidden, group.group_name, device)
        for context in (300, 5000):
            c = _setup.make_cache(tokens, context, shape, seed=11 + rank, device=device)
            args = (
                c["q"],
                c["kv_cache"],
                c["topk"],
                w["w_uv"],
                w["o_proj"],
                SCALE_LOG2,
            )
            out = glm5_3_attn_out_tp4(
                *args, residual, norm_weight, partials[0], group.group_name
            )
            torch.cuda.synchronize()
            attn, v, partial = _setup.out_reference(
                c["q"], c["kv_cache"], c["topk"], w, shape
            )
            rows = []
            ok &= report(rows, "sparse MLA", out[2], attn, 2e-2)
            ok &= report(rows, "value rows", out[3], v, 2e-2)
            output, residual_output = all_reduce_norm(
                partial, residual, norm_weight, NORM_EPS
            )
            ok &= report(rows, "residual_out", out[1], residual_output, 2e-2)
            ok &= report(rows, "output", out[0], output, 2e-2)
            again = glm5_3_attn_out_tp4(
                *args, residual, norm_weight, partials[1], group.group_name
            )
            deterministic = torch.equal(again[0], out[0])
            rows.append(
                f"  {'determinism':<22} "
                f"{'bitwise' if deterministic else 'DIFFERS  <-- FAIL'}"
            )
            ok &= deterministic
            if verbose or not ok:
                say(f"rank{rank} T={tokens} context {context}\n" + "\n".join(rows))
    verdict(ok, group, "glm5_3_attn_out_tp4")


def benchmark(group: dist.ProcessGroup, verbose: bool = True) -> dict[str, Any]:
    """Cold-weight timing per served window (``qwen3_5_moe_tp4.benchmark``):
    ``LAYERS`` distinct weight copies and caches (default 8; even) back to back,
    ``REPS`` times (default 4) in one CUDA graph; ``CONTEXT`` committed slots
    (default 8192)."""
    rank = group.rank()
    device = torch.device("cuda", torch.cuda.current_device())
    layers = int(os.environ.get("LAYERS", "8"))
    reps = int(os.environ.get("REPS", "4"))
    context = int(os.environ.get("CONTEXT", "8192"))
    shape = _setup.AttnShape()
    weights = [
        _setup.out_weights(shape, seed=1000 * i + rank, device=device)
        for i in range(layers)
    ]
    megabytes = (
        2 * shape.hidden * shape.heads * shape.v_head
        + 2 * shape.heads * shape.v_head * shape.kv_lora
    ) / 1e6
    results = {}
    for tokens in TOKENS:
        args = []
        for i, w in enumerate(weights):
            c = _setup.make_cache(tokens, context, shape, seed=i, device=device)
            args.append(
                (c["q"], c["kv_cache"], c["topk"], w["w_uv"], w["o_proj"], SCALE_LOG2)
            )
        g = torch.Generator(device=device)
        g.manual_seed(3)
        residual = torch.randn(tokens, shape.hidden, generator=g, device=device).to(
            torch.bfloat16
        )
        norm_weight = torch.ones(shape.hidden, device=device, dtype=torch.bfloat16)
        partials = symmetric_partials(tokens, shape.hidden, group.group_name, device)

        def step(
            args=args, residual=residual, norm_weight=norm_weight, partials=partials
        ) -> None:
            for i in range(layers):
                glm5_3_attn_out_tp4(
                    *args[i], residual, norm_weight, partials[i % 2], group.group_name
                )

        median, minimum = graph_time(step, layers, reps)
        if verbose:
            say(
                f"rank{rank} T={tokens} context={context} us/launch med {median:.2f} "
                f"min {minimum:.2f} (~{megabytes:.0f} MB weights -> "
                f"{megabytes / median:.2f} TB/s)"
            )
        results[tokens] = {"latency_us": median, "min_latency_us": minimum}
    return results


def main(verbose: bool = True, check_only: bool = False) -> dict[str, Any]:
    """The four-rank check, then the cold-weight timing (``check_only``: the
    check alone)."""
    return run_main(
        "glm5_3_attn_out_tp4", correctness_check, benchmark, verbose, check_only
    )


if __name__ == "__main__":
    main(check_only="--check" in sys.argv)
