# ruff: noqa: ANN001, ANN201
# pyrefly: ignore-errors
"""GLM-5.3 Flash MTP draft attention output side as one Gluon megakernel: a TP4 rank's 16
heads for one request (T <= 8 rows; it serves the draft decode step's one row), from
the FP8 absorbed queries and the DSA top-k to the MoE's input.
The draft layer has no mHC: its attention -> FFN boundary is the all-reduce, the residual
add and post_attention_layernorm. Replaces SGLang's trtllm-gen sparse MLA (+ its top-k
padding / metadata launches), the W_UV bmm, o_proj, the custom all-reduce and the fused
add + RMSNorm.

Roots 1-4 follow ``mla_out_tp4`` (the target's out MK, whose tail is the mHC boundary):
  1. splits: per (token, split of the top-k picks: 32 for one row (latency-bound gathers;
     GLM-5.3 draft: 26.8 vs 31.3 us at 64), 64 up to 4 rows, 128 past), the sparse MLA over
     the picked slots of the raw FP8 cache (512 B a token, NoPE) with swapped operands:
     S^T = K q^T, a one-block softmax (log2 domain), O^T = K^T P^T with P in E4M3 x 256
     (x 448 for 128 picks); BF16 [512, 16] partials and the per-head log2-sum-exp2
     (FP32), as stock's trtllm-gen sparse MLA stores its multi-CTA KV-split partials.
  2. merge: per (token, 8 latent columns), the splits in split order, BF16 (the trtllm
     kernel's output).
  3. W_UV per (head, 128 value rows), BF16 (SGLang's bmm).
  4. o_proj [4096, 4096] per (128 rows, K chunk of 4 heads): FP32 partials into this
     rank's symmetric buffer [4, T, 4096] (no publish hop: the all-reduce sums every
     rank's chunks).
  5. all-reduce + residual per (token, 128 columns): the ranks' chunks summed in rank
     then chunk order (FP32), rounded to BF16 (the all-reduce's output dtype), plus the
     residual, rounded to BF16 (SGLang's fused add): the new residual, its sum of squares.
  6. post_attention_layernorm per (token, 512 columns): bf16((x * w) * r) (SGLang's JIT
     RMSNorm order) -> the MoE's input.

Every split sum runs in a fixed order (no atomics): deterministic run to run. The
all-reduce reads only partials root 4 wrote in the same launch.

``correctness_check()`` compares every rank with the torch reference (``_setup``) on
synthetic weights; ``main()`` runs it, then times the kernel with cold weights::

    NVSHMEM_DISABLE_CUDA_VMM=1 PYTHONPATH=. \\
      python -m torch.distributed.run --standalone --nproc-per-node=4 \\
      pretuned_kernels/megakernels/glm5_3_flash_tp4/mtp_attn_out_tp4.py [--check]
"""

from __future__ import annotations

import functools
from itertools import starmap
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
from pretuned_kernels.megakernels.glm5_3_flash_tp4 import mla_out_tp4
from pretuned_kernels.megakernels.glm5_3_tp4._setup import make_cache
from pretuned_kernels.megakernels.glm5_3_tp4._setup import out_reference
from pretuned_kernels.megakernels.glm5_3_tp4._setup import out_weights
import torch
import torch.distributed as dist

import helion
import helion.language as hl

O_CHUNKS = 4  # o_proj's K chunks (FP32 partials each: the all-reduce sums them)
RING_LOADS = frozenset(["w_uv", "attn", "w_o", "attn_v_chunks"])


def splits_for(tokens: int, width: int = mla_out_tp4.TOPK_WIDTH) -> tuple[int, int]:
    """(picks a split, splits a token): 32 picks for one row, 64 up to 4 rows, 128 past 4."""
    n = 64 * (1 + tokens // 5) // (1 + 2 // (tokens + 1))
    return n, (width + n - 1) // n


@helion.kernel(
    static_shapes=False,
    backend="gluon",
    triton_do_not_specialize=False,
    ignore_warnings=[helion.exc.TensorOperationInWrapper, helion.exc.FastMathUnused],
)
def glmf_mtp_attn_out_tp4(
    q,  # [T, 16, 512] e4m3: the absorbed queries
    kv_cache,  # [slots, 512] e4m3: the draft layer's raw MLA cache
    topk,  # [T, W] int32: each row's picked slots (-1: none)
    w_uv,  # [16, 256, 512] bf16: W_UV per head (K contiguous)
    scale_log2: float,  # softmax scale * log2(e)
    att_ws,  # [S, T, 512, 16] bf16 scratch: split outputs (latent-major)
    lse_ws,  # [S, T, 16] fp32 scratch: split log2-sum-exp2
    attn,  # [T, 16, 512] bf16 scratch: the sparse MLA
    attn_v,  # [T, 16, 256] bf16 scratch: the value rows
    w_o,  # [4096, K, 4096 / K] bf16: o_proj in K chunks (a view)
    symmetric_partial,  # [K, T, 4096] fp32 symmetric memory: this rank's o_proj K-chunk partials
    residual,  # [T, 4096] bf16: the residual (the eh output)
    norm_w,  # [4096] bf16: post_attention_layernorm
    norm_partial,  # [T, 4096 / 128] fp32 scratch
    norm_input,  # [T, 4096] fp32 scratch
    residual_out,  # [T, 4096] bf16 out: the new residual
    output,  # [T, 4096] bf16 out: the MoE's input
    group_name: hl.ProcessGroupName,
) -> None:
    hl.specialize(q.size())
    hl.specialize(q.stride())
    hl.specialize(kv_cache.stride())
    hl.specialize(topk.size())
    hl.specialize(topk.stride())
    hl.specialize(w_uv.size())
    hl.specialize(w_uv.stride())
    hl.specialize(att_ws.size())
    hl.specialize(att_ws.stride())
    hl.specialize(lse_ws.size())
    hl.specialize(lse_ws.stride())
    hl.specialize(attn.size())
    hl.specialize(attn.stride())
    hl.specialize(attn_v.size())
    hl.specialize(attn_v.stride())
    hl.specialize(w_o.size())
    hl.specialize(w_o.stride())
    hl.specialize(symmetric_partial.size())
    hl.specialize(symmetric_partial.stride())
    hl.specialize(residual.size())
    hl.specialize(residual.stride())
    hl.specialize(norm_w.size())
    hl.specialize(norm_w.stride())
    hl.specialize(norm_partial.size())
    hl.specialize(norm_partial.stride())
    hl.specialize(norm_input.size())
    hl.specialize(norm_input.stride())
    hl.specialize(residual_out.size())
    hl.specialize(residual_out.stride())
    hl.specialize(output.size())
    hl.specialize(output.stride())

    rows, heads, kv_lora = q.shape
    top_k = topk.size(1)
    v_head = w_uv.size(1)
    hidden, o_chunks, o_k = w_o.shape
    q_width = o_chunks * o_k
    assert rows <= 8 and heads == 16 and kv_lora == 512 and q_width == heads * v_head
    # Literals: a module-level int reaches the kernel as a runtime scalar.
    # (Arithmetic on the specialized row count: a branch merges into a symbol.)
    eps = 1e-5
    longer = rows // 5
    split_n = 64 * (1 + longer) // (1 + 2 // (rows + 1))
    splits = (top_k + split_n - 1) // split_n
    # P's E4M3 scale: the format's maximum for the longer splits (their smaller
    # probabilities keep more mantissa bits).
    p_scale = 256.0 + 192.0 * longer
    assert att_ws.size(0) == splits
    attn_v_chunks = attn_v.view(rows, o_chunks, o_k)
    remote_partials = torch.ops.symm_mem.get_remote_tensors(
        symmetric_partial, group_name
    )

    hl.pdl_wait()
    # The code toward L2 first: in a decoder layer the previous weight stream
    # evicted it.
    hl.prefetch_code()

    # 1. sparse MLA splits per (token, split).
    for sp_t, sp_s in hl.tile([rows, splits], block_size=[1, 1]):
        mla_out_tp4.attention_split(
            q,
            kv_cache,
            topk,
            att_ws,
            lse_ws,
            sp_t.begin,
            sp_s.begin,
            split_n,
            (top_k, kv_lora),
            scale_log2,
            p_scale,
        )

    # 2. merge per (token, 8 latent columns): the splits in split order (64
    # tasks: 32 columns made each task hold all splits x 32 x 16 heads, ~0.9 us
    # slower).
    for mg_t, mg_c in hl.tile([rows, kv_lora], block_size=[1, 8]):
        mla_out_tp4.merge_splits(lse_ws, att_ws, attn, mg_t.begin, mg_c)

    # 3. value up-projection per (head, 128 value rows).
    for uv_h, uv_n, uv_t in hl.tile([heads, v_head, rows], block_size=[1, 128, 16]):
        uv_acc = hl.zeros([uv_n, uv_t], dtype=torch.float32)
        for uv_k in hl.tile(kv_lora, block_size=64):
            uv_acc = hl.dot(
                w_uv[uv_h.begin, uv_n, uv_k], attn[uv_t, uv_h.begin, uv_k].T, acc=uv_acc
            )
        attn_v[uv_t, uv_h.begin, uv_n] = uv_acc.T.to(torch.bfloat16)

    # 4. o_proj per (128 rows, K chunk): FP32 partials into this rank's
    # symmetric buffer (the all-reduce sums every rank's chunks: no publish
    # hop).
    for o_n, o_c, o_t in hl.tile([hidden, o_chunks, rows], block_size=[128, 1, 16]):
        o_acc = hl.zeros([o_n, o_t], dtype=torch.float32)
        for o_kk in hl.tile(o_k, block_size=64):
            o_acc = hl.dot(
                w_o[o_n, o_c.begin, o_kk],
                attn_v_chunks[o_t, o_c.begin, o_kk].T,
                acc=o_acc,
            )
        symmetric_partial[o_c.begin, o_t, o_n] = o_acc.T

    hl.pdl_launch_dependents()
    # 5. all-reduce + residual per (token, 128 columns): rank then chunk order
    # (FP32), BF16, + residual, BF16.
    for ar_t, ar_n in hl.tile([rows, hidden], block_size=[1, 128]):
        ar_token = ar_t.begin
        ar_sum = hl.zeros([ar_n], dtype=torch.float32)
        for ar_r in hl.static_range(len(remote_partials)):
            for ar_c in hl.static_range(o_chunks):
                ar_sum = ar_sum + remote_partials[ar_r][ar_c, ar_token, ar_n]
        ar_x = ar_sum.to(torch.bfloat16).to(torch.float32) + residual[
            ar_token, ar_n
        ].to(torch.float32)
        ar_new = ar_x.to(torch.bfloat16)
        residual_out[ar_token, ar_n] = ar_new
        ar_f = ar_new.to(torch.float32)
        norm_input[ar_token, ar_n] = ar_f
        norm_partial[ar_token, ar_n.id] = torch.sum(ar_f * ar_f)
    # 6. post_attention_layernorm per (token, 512 columns): bf16((x * w) * r).
    for nm_t, nm_n in hl.tile([rows, hidden], block_size=[1, 512]):
        nm_token = nm_t.begin
        nm_scale = torch.rsqrt(torch.sum(norm_partial[nm_token, :]) / hidden + eps)
        output[nm_token, nm_n] = (
            (norm_input[nm_token, nm_n] * norm_w[nm_n].to(torch.float32)) * nm_scale
        ).to(torch.bfloat16)


def mk_config(bound) -> helion.Config:
    """The weight streams (W_UV, o_proj) and their activation operands through the TMA
    ring, every other access by pointer (the key gathers among them); the dot roots
    warp-specialized; W_UV's weight tiles stream into tensor memory ahead of its
    tasks' waits (T=1, cold: 2% faster; o_proj's tail, the other candidate, adds
    nothing); Lamport in-band all-reduce transport."""
    fields = ring_config_fields(bound, RING_LOADS)
    default = bound.config_spec.default_config().config
    return helion.Config(
        indexing=fields["indexing"],
        pid_type="persistent_blocked",
        num_sm_multiplier=1,
        num_stages=6,
        num_warps=4,
        cross_loop_pipeline="dynamic",
        range_warp_specializes=fields["range_warp_specializes"],
        cross_loop_tmem_prefetch=[True, False],
        cross_loop_inband_protocol=["lamport"]
        * len(default.get("cross_loop_inband_protocol", [])),
    )


@functools.cache
def scratch(device: torch.device, tokens: int) -> dict[str, torch.Tensor]:
    """The kernel's scratch on ``device``."""
    _, splits = splits_for(tokens)
    heads, latent, v_head = (
        _setup.ATTN_SHAPE.heads,
        _setup.ATTN_SHAPE.kv_lora,
        _setup.ATTN_SHAPE.v_head,
    )
    f32, bf16, H = torch.float32, torch.bfloat16, _setup.HIDDEN
    return {
        "att_ws": torch.zeros(splits, tokens, latent, heads, device=device, dtype=bf16),
        "lse_ws": torch.zeros(splits, tokens, heads, device=device, dtype=f32),
        "attn": torch.zeros(tokens, heads, latent, device=device, dtype=bf16),
        "attn_v": torch.zeros(tokens, heads, v_head, device=device, dtype=bf16),
        "norm_partial": torch.zeros(tokens, H // 128, device=device, dtype=f32),
        "norm_input": torch.zeros(tokens, H, device=device, dtype=f32),
    }


def symmetric_partial_spec(tokens: int) -> tuple[tuple[int, ...], torch.dtype]:
    """The (shape, dtype) of the symmetric buffer the kernel's o_proj writes."""
    return (O_CHUNKS, tokens, _setup.HIDDEN), torch.float32


def kernel_args(
    q8,
    kv_cache,
    topk,
    w_uv,
    o_proj,
    symmetric_partial,
    residual,
    norm_w,
    out,
    group_name,
) -> tuple:
    """``glmf_mtp_attn_out_tp4``'s arguments: q8 [T, 16, 512] E4M3, kv_cache [slots, 512]
    E4M3, topk [T, W] int32 cache slots (-1: none), W_UV [16, 256, 512], o_proj [4096,
    16 x 256], ``symmetric_partial`` per ``symmetric_partial_spec``, residual [T, 4096]
    BF16, post_attention_layernorm's weight; ``out`` = (residual_out, output) [T, 4096]
    BF16."""
    tokens = q8.shape[0]
    sc = scratch(q8.device, tokens)
    return (
        q8,
        kv_cache,
        topk,
        w_uv,
        _setup.SCALE_LOG2,
        sc["att_ws"],
        sc["lse_ws"],
        sc["attn"],
        sc["attn_v"],
        o_proj.view(o_proj.shape[0], O_CHUNKS, o_proj.shape[1] // O_CHUNKS),
        symmetric_partial,
        residual,
        norm_w,
        sc["norm_partial"],
        sc["norm_input"],
        out[0],
        out[1],
        group_name,
    )


def compile_kernel(args: tuple):
    """``glmf_mtp_attn_out_tp4`` compiled for ``args`` (once per binding)."""
    return _setup.compiled(glmf_mtp_attn_out_tp4, mk_config, args)


# ---------------------------- check and timing (torch.distributed.run, 4 ranks)

# The MTP draft's decode step (one row; its extend runs SGLang's attention).
TOKENS = (1,)


def _case(
    rank: int, tokens: int, context: int, seed: int, partial, group_name
) -> tuple:
    """Rank-dependent weights and cache (``glm5_3_tp4._setup``'s at
    ``_setup.ATTN_SHAPE``), a residual and a norm weight: (weights, cache,
    residual, norm weight, outputs, kernel arguments)."""
    device = torch.device("cuda", torch.cuda.current_device())
    w = out_weights(_setup.ATTN_SHAPE, seed=seed + rank, device=device)
    c = make_cache(tokens, context, _setup.ATTN_SHAPE, seed=seed + rank, device=device)
    g = torch.Generator(device=device).manual_seed(seed + 1000)
    residual = (
        2 * torch.randn(tokens, _setup.HIDDEN, generator=g, device=device)
    ).bfloat16()
    norm_w = (
        1 + 0.1 * torch.randn(_setup.HIDDEN, generator=g, device=device)
    ).bfloat16()
    out = tuple(
        torch.empty(tokens, _setup.HIDDEN, device=device, dtype=torch.bfloat16)
        for _ in range(2)
    )
    args = kernel_args(
        c["q"],
        c["kv_cache"],
        c["topk"],
        w["w_uv"],
        w["o_proj"],
        partial,
        residual,
        norm_w,
        out,
        group_name,
    )
    return w, c, residual, norm_w, out, args


def correctness_check(group: dist.ProcessGroup, verbose: bool = True) -> None:
    """Each rank its own weights' slice and cache (``TOKENS``): a short
    context (every pool picked), a long one (512 random pools and the open pool)
    and a middle one, launched back to back (no host synchronization); the
    residual output and the normed output against ``out_reference``'s partials
    all-reduced in rank order plus the residual, then post_attention_layernorm
    (``_qwen3_5_tp4.all_reduce_norm``); then the last again with NaN scratch and
    outputs: the same bits."""
    rank = group.rank()
    device = torch.device("cuda", torch.cuda.current_device())
    ok = True
    for tokens in TOKENS:
        partial = symmetric_buffer(
            *symmetric_partial_spec(tokens), group.group_name, device
        )
        contexts = (700, 9000, 3000)
        cases = [
            _case(rank, tokens, context, seed, partial, group.group_name)
            for seed, context in enumerate(contexts, start=11)
        ]
        torch.cuda.synchronize()
        dist.barrier()
        for case in cases:
            compile_kernel(case[-1])(*case[-1])
        torch.cuda.synchronize()
        rows = []
        for (w, c, residual, norm_w, out, _), context in zip(
            cases, contexts, strict=True
        ):
            _, _, reference = out_reference(
                c["q"], c["kv_cache"], c["topk"], w, _setup.ATTN_SHAPE
            )
            normed, residual_out = all_reduce_norm(
                reference, residual, norm_w, _setup.NORM_EPS
            )
            ok &= report(
                rows, f"context {context} residual", out[0], residual_out, 2e-2
            )
            ok &= report(rows, f"context {context} output", out[1], normed, 2e-2)
        *_, out, args = cases[-1]
        first = [o.clone() for o in out]
        _setup.poison([*out, *scratch(device, tokens).values()])
        dist.barrier()
        compile_kernel(args)(*args)
        torch.cuda.synchronize()
        repeat = all(starmap(same, zip(out, first, strict=True)))
        rows.append(
            f"  {'NaN scratch repeat':<22} {'bitwise' if repeat else 'DIFFERS  <-- FAIL'}"
        )
        ok &= repeat
        if verbose or not ok:
            say(f"rank{rank} T={tokens}\n" + "\n".join(rows))
    verdict(ok, group, "glmf_mtp_attn_out_tp4")


def benchmark(group: dist.ProcessGroup, verbose: bool = True) -> dict[str, Any]:
    """Cold-weight timing (``TOKENS``): ``LAYERS`` distinct weight copies and caches
    (default 8) back to back, ``REPS`` times (default 4) in one CUDA graph;
    ``CONTEXT`` committed tokens (default 8192)."""
    rank = group.rank()
    device = torch.device("cuda", torch.cuda.current_device())
    layers = int(os.environ.get("LAYERS", "8"))
    reps = int(os.environ.get("REPS", "4"))
    context = int(os.environ.get("CONTEXT", "8192"))
    results = {}
    for tokens in TOKENS:
        partial = symmetric_buffer(
            *symmetric_partial_spec(tokens), group.group_name, device
        )
        args = [
            _case(rank, tokens, context, 1000 * i, partial, group.group_name)[-1]
            for i in range(layers)
        ]

        def step(args=args) -> None:
            for a in args:
                compile_kernel(a)(*a)

        median, minimum = graph_time(step, layers, reps)
        if verbose:
            say(
                f"rank{rank} T={tokens} context={context} us/launch med {median:.2f} "
                f"min {minimum:.2f}"
            )
        results[tokens] = {"latency_us": median, "min_latency_us": minimum}
    return results


def main(verbose: bool = True, check_only: bool = False) -> dict[str, Any]:
    """The four-rank check, then the cold-weight timing (``check_only``: the
    check alone)."""
    return run_main(
        "glmf_mtp_attn_out_tp4", correctness_check, benchmark, verbose, check_only
    )


if __name__ == "__main__":
    main(check_only="--check" in sys.argv)
