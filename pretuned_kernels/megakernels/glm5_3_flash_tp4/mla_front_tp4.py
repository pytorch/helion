# ruff: noqa: ANN001, ANN201
# pyrefly: ignore-errors
"""GLM-5.3 Flash MLA prologue as one Gluon megakernel: a TP4 rank's 16 heads for one
request of T <= 8 rows, from the layer input to the FP8 absorbed queries, q_lora (the
DSA indexer's input) and the window's MLA cache rows
(``glm5_3_tp4._setup.front_reference``; SGLang: dsv3_fused_a_gemm, two RMSNorms,
q_b, the W_UK bmm, FP8 casts, set_mla_kv_buffer). NoPE: no RoPE anywhere. It
serves the MTP draft's decode step (one row; the target's MLA layers run
``mla_frontidx_tp4``). The split-K partials are summed in a fixed order (no
atomics): the outputs are run-to-run deterministic.

Roots, in launch order (one persistent CTA per SM, TMA ring for the weights):
  1. a: the fused q_a | kv_a projection (BF16 [2048, 4096], replicated as SGLang
     holds it) per (128 rows, K chunk of 512) (128 tasks): FP32 partials.
  2. q norm per token: q_a = the chunks' sum (chunk order) rounded to BF16, RMSNorm
     (BF16): q_lora.
  3. kv per token: kv_a likewise, RMSNorm, BF16, E4M3 into the window's cache slot.
  4. q_b (BF16 [16 x 256, 1536]) per (128 rows, K chunk) (FP32 partials).
  5. absorb per (head, 128 latent rows): W_UK over the head's 256 nope dims (q_b's
     partials summed in order, BF16), BF16, E4M3.

``correctness_check()`` compares every rank with the torch reference (``_setup``) on
synthetic weights; ``main()`` runs it, then times the kernel with cold weights::

    NVSHMEM_DISABLE_CUDA_VMM=1 PYTHONPATH=. \\
      python -m torch.distributed.run --standalone --nproc-per-node=4 \\
      pretuned_kernels/megakernels/glm5_3_flash_tp4/mla_front_tp4.py [--check]
"""

from __future__ import annotations

import functools
from itertools import starmap
import os
import sys
from typing import Any

from pretuned_kernels.megakernels._qwen3_5_tp4 import graph_time
from pretuned_kernels.megakernels._qwen3_5_tp4 import report
from pretuned_kernels.megakernels._qwen3_5_tp4 import run_main
from pretuned_kernels.megakernels._qwen3_5_tp4 import same
from pretuned_kernels.megakernels._qwen3_5_tp4 import say
from pretuned_kernels.megakernels._qwen3_5_tp4 import verdict
from pretuned_kernels.megakernels._ring_config import ring_config_fields
from pretuned_kernels.megakernels.glm5_3_flash_tp4 import _setup
from pretuned_kernels.megakernels.glm5_3_tp4._setup import front_reference
from pretuned_kernels.megakernels.glm5_3_tp4._setup import front_weights
import torch
import torch.distributed as dist

import helion
import helion.language as hl

LANES = 8
A_CHUNKS = 8  # K chunks of the fused a projection (512 each)
B_CHUNKS = 4  # K chunks of q_b


@helion.kernel(
    static_shapes=False,
    backend="gluon",
    triton_do_not_specialize=False,
    ignore_warnings=[helion.exc.TensorOperationInWrapper, helion.exc.FastMathUnused],
)
def glmf_mla_front_tp4(
    x_a,  # [T, A, 4096 / A] bf16: the layer input in K chunks (a view)
    w_a,  # [2048, A, 4096 / A] bf16: q_a (1536) | kv_a (512) rows in K chunks (a view)
    q_a_norm,  # [1536] bf16
    kv_a_norm,  # [512] bf16
    w_qb,  # [16 x 256, B, 1536 / B] bf16: q_b in K chunks (a view)
    w_uk,  # [16, 512, 256] bf16: W_UK per head (K contiguous)
    cache_loc,  # [T] int64: the window's MLA cache slots
    kv_cache,  # [slots, 512] e4m3: one layer's raw MLA cache (written at cache_loc)
    a_parts,  # [A, 8, 2048] fp32 scratch
    q_lora,  # [T, 1536] bf16 out (written as [T, B, 1536 / B] below)
    qb_parts,  # [B, 8, 16, 256] fp32 scratch
    q8,  # [T, 16, 512] e4m3 out
) -> None:
    hl.specialize(x_a.size())
    hl.specialize(x_a.stride())
    hl.specialize(w_a.size())
    hl.specialize(w_a.stride())
    hl.specialize(q_a_norm.size())
    hl.specialize(q_a_norm.stride())
    hl.specialize(kv_a_norm.size())
    hl.specialize(kv_a_norm.stride())
    hl.specialize(w_qb.size())
    hl.specialize(w_qb.stride())
    hl.specialize(w_uk.size())
    hl.specialize(w_uk.stride())
    hl.specialize(cache_loc.size())
    hl.specialize(cache_loc.stride())
    hl.specialize(kv_cache.stride())
    hl.specialize(a_parts.size())
    hl.specialize(a_parts.stride())
    hl.specialize(q_lora.size())
    hl.specialize(q_lora.stride())
    hl.specialize(qb_parts.size())
    hl.specialize(qb_parts.stride())
    hl.specialize(q8.size())
    hl.specialize(q8.stride())

    rows, a_chunks, a_k = x_a.shape
    a_rows = w_a.size(0)
    q_lora_n = q_a_norm.size(0)
    kv_lora = kv_a_norm.size(0)
    heads, _, nope = w_uk.shape
    qb_rows, b_chunks, b_k = w_qb.shape
    assert rows <= 8 and a_rows == q_lora_n + kv_lora and qb_rows == heads * nope
    assert b_chunks * b_k == q_lora_n
    # Literals: a module-level int reaches the kernel as a runtime scalar.
    eps = 1e-5
    q_lora_chunked = q_lora.view(rows, b_chunks, b_k)

    hl.pdl_wait()
    hl.prefetch_code()

    # 1. the fused a projection: FP32 partials of its K chunks.
    for a_n, a_c, a_t in hl.tile([a_rows, a_chunks, rows], block_size=[128, 1, 16]):
        a_acc = hl.zeros([a_n, a_t], dtype=torch.float32)
        for a_kk in hl.tile(a_k, block_size=64):
            a_acc = hl.dot(
                w_a[a_n, a_c.begin, a_kk], x_a[a_t, a_c.begin, a_kk].T, acc=a_acc
            )
        a_parts[a_c.begin, a_t, a_n] = a_acc.T

    # 2. q norm per token: q_a (BF16), RMSNorm -> q_lora.
    for qn_t in hl.tile(rows, block_size=1):
        qn_tok = qn_t.begin
        qn_w = q_a_norm[:].to(torch.float32)
        qn_x = torch.sum(a_parts[:, qn_tok, :q_lora_n], dim=0)
        qn_x = qn_x.to(torch.bfloat16).to(torch.float32)
        qn_inv = torch.rsqrt(torch.sum(qn_x * qn_x) / q_lora_n + eps)
        q_lora[qn_tok, :] = (qn_x * qn_inv * qn_w).to(torch.bfloat16)

    # 3. kv per token: kv_a (BF16), RMSNorm (BF16), E4M3 into the cache slot.
    for kn_t in hl.tile(rows, block_size=1):
        kn_tok = kn_t.begin
        kn_w = kv_a_norm[:].to(torch.float32)
        kn_slot = cache_loc[kn_tok]
        kn_x = torch.sum(a_parts[:, kn_tok, q_lora_n:], dim=0)
        kn_x = kn_x.to(torch.bfloat16).to(torch.float32)
        kn_inv = torch.rsqrt(torch.sum(kn_x * kn_x) / kv_lora + eps)
        kv_cache[kn_slot, :] = (
            (kn_x * kn_inv * kn_w).to(torch.bfloat16).to(torch.float8_e4m3fn)
        )

    # 4. q_b: FP32 partials of its K chunks of q_lora (chunks first: a head's
    # tasks are consecutive, so its absorb starts early).
    for qb_c, qb_n, qb_t in hl.tile([b_chunks, qb_rows, rows], block_size=[1, 128, 16]):
        qb_acc = hl.zeros([qb_n, qb_t], dtype=torch.float32)
        for qb_k in hl.tile(b_k, block_size=64):
            qb_acc = hl.dot(
                w_qb[qb_n, qb_c.begin, qb_k],
                q_lora_chunked[qb_t, qb_c.begin, qb_k].T,
                acc=qb_acc,
            )
        qb_parts[qb_c.begin, qb_t, qb_n.begin // nope, qb_n.index % nope] = qb_acc.T

    hl.pdl_launch_dependents()
    # 5. absorb per (head, 128 latent rows): W_UK over the head's BF16 nope
    # dims.
    for ab_h, ab_l, ab_t in hl.tile([heads, kv_lora, rows], block_size=[1, 128, 16]):
        ab_acc = hl.zeros([ab_l, ab_t], dtype=torch.float32)
        for ab_k in hl.tile(nope, block_size=64):
            ab_q = qb_parts[0, ab_t, ab_h.begin, ab_k]
            for ab_c in hl.static_range(1, b_chunks):
                ab_q = ab_q + qb_parts[ab_c, ab_t, ab_h.begin, ab_k]
            ab_acc = hl.dot(
                w_uk[ab_h.begin, ab_l, ab_k], ab_q.to(torch.bfloat16).T, acc=ab_acc
            )
        q8[ab_t, ab_h.begin, ab_l] = ab_acc.T.to(torch.bfloat16).to(torch.float8_e4m3fn)


RING_LOADS = frozenset(["w_a", "x_a", "w_qb", "q_lora_chunked", "w_uk"])


def mk_config(bound) -> helion.Config:
    """The weights and their activation operands through the TMA ring; the dot roots
    warp-specialized; q_b's weight tiles stream into tensor memory ahead of its
    tasks' waits (T=1, cold: 13.4 vs 14.5 us; the absorb's W_UK loop, the other
    candidate, takes none: its dot issues from the body)."""
    fields = ring_config_fields(bound, RING_LOADS)
    return helion.Config(
        indexing=fields["indexing"],
        pid_type="persistent_blocked",
        num_sm_multiplier=1,
        num_stages=4,
        num_warps=4,
        cross_loop_pipeline="dynamic",
        range_warp_specializes=fields["range_warp_specializes"],
        cross_loop_tmem_prefetch=[True, False],
    )


@functools.cache
def scratch(device: torch.device, tokens: int) -> dict[str, torch.Tensor]:
    """The kernel's scratch and outputs (q_lora, q8) on ``device``."""
    s, f32 = _setup.ATTN_SHAPE, torch.float32
    return {
        "a_parts": torch.zeros(
            A_CHUNKS, LANES, s.q_lora + s.kv_lora, device=device, dtype=f32
        ),
        "qb_parts": torch.zeros(
            B_CHUNKS, LANES, s.heads, s.nope, device=device, dtype=f32
        ),
        "q_lora": torch.zeros(tokens, s.q_lora, device=device, dtype=torch.bfloat16),
        "q8": torch.zeros(
            tokens, s.heads, s.kv_lora, device=device, dtype=torch.float8_e4m3fn
        ),
    }


def prepared_weights(w: dict) -> dict:
    """The layer's weights as the kernel reads them: SGLang's fused q_a | kv_a rows
    and q_b in K chunks (views), W_UK per head K contiguous."""
    return {
        "w_a": w["qkv_a"].view(w["qkv_a"].size(0), A_CHUNKS, -1),
        "q_a_norm": w["q_a_norm"],
        "kv_a_norm": w["kv_a_norm"],
        "w_qb": w["q_b"].view(w["q_b"].size(0), B_CHUNKS, -1),
        "w_uk": w["w_uk"],
    }


def kernel_args(x, pw, cache_loc, kv_cache) -> tuple:
    """``glmf_mla_front_tp4``'s arguments: the attention input ``x`` [T, 4096] BF16
    (contiguous), ``pw`` (``prepared_weights``), the window's MLA cache slots
    ``cache_loc`` [T] int64 and ``kv_cache`` [slots, 512] E4M3. Outputs in
    ``scratch``: q_lora [T, 1536] BF16, q8 [T, 16, 512] E4M3."""
    tokens = x.shape[0]
    sc = scratch(x.device, tokens)
    return (
        x.view(tokens, A_CHUNKS, -1),
        pw["w_a"],
        pw["q_a_norm"],
        pw["kv_a_norm"],
        pw["w_qb"],
        pw["w_uk"],
        cache_loc,
        kv_cache,
        sc["a_parts"],
        sc["q_lora"],
        sc["qb_parts"],
        sc["q8"],
    )


def compile_kernel(args: tuple):
    """``glmf_mla_front_tp4`` compiled for ``args`` (once per binding)."""
    return _setup.compiled(glmf_mla_front_tp4, mk_config, args)


# ---------- check and timing (torch.distributed.run, 4 ranks; no communication)

# The MTP draft's decode step (one row; its extend runs SGLang's attention).
TOKENS = (1,)
SLOTS = 4096  # the checks' MLA cache slots


def _inputs(layer: int, tokens: int, device) -> tuple:
    """Layer ``layer``'s weights (``glm5_3_tp4._setup.front_weights`` at
    ``_setup.ATTN_SHAPE``), an input window, a zero MLA cache and the window's
    slots."""
    w = front_weights(_setup.ATTN_SHAPE, seed=1000 * layer, device=device)
    g = torch.Generator(device=device).manual_seed(50 + layer)
    x = torch.randn(tokens, _setup.HIDDEN, generator=g, device=device).to(
        torch.bfloat16
    )
    kv_cache = torch.zeros(
        SLOTS, _setup.ATTN_SHAPE.kv_lora, dtype=torch.float8_e4m3fn, device=device
    )
    cache_loc = torch.randperm(SLOTS, generator=g, device=device)[:tokens]
    return w, x, kv_cache, cache_loc


def correctness_check(group: dist.ProcessGroup, verbose: bool = True) -> None:
    """Every rank its own layer per window (``TOKENS``) against
    ``glm5_3_tp4._setup.front_reference``: q_lora, the FP8 queries and the
    window's cache rows within 2e-2 relative RMS, the other cache rows untouched;
    a second launch with NaN scratch bitwise equal."""
    rank = group.rank()
    device = torch.device("cuda", torch.cuda.current_device())
    ok = True
    for tokens in TOKENS:
        w, x, kv_cache, cache_loc = _inputs(rank, tokens, device)
        args = kernel_args(x, prepared_weights(w), cache_loc, kv_cache)
        kernel = compile_kernel(args)
        kernel(*args)
        torch.cuda.synchronize()
        q_lora, q8, kv8 = front_reference(x, w, None, None, _setup.ATTN_SHAPE)
        sc = scratch(device, tokens)
        rows = []
        ok &= report(rows, "q_lora", sc["q_lora"], q_lora, 2e-2)
        ok &= report(rows, "q (FP8)", sc["q8"].float(), q8.float(), 2e-2)
        ok &= report(
            rows, "cache rows (FP8)", kv_cache[cache_loc].float(), kv8.float(), 2e-2
        )
        untouched = torch.ones(SLOTS, dtype=torch.bool, device=device)
        untouched[cache_loc] = False
        clean = not kv_cache[untouched].view(torch.uint8).any()
        rows.append(
            f"  {'other cache rows':<22} {'untouched' if clean else 'WRITTEN  <-- FAIL'}"
        )
        ok &= clean
        first = [sc["q_lora"].clone(), sc["q8"].clone(), kv_cache.clone()]
        _setup.poison(sc.values())
        kernel(*args)
        torch.cuda.synchronize()
        again = (sc["q_lora"], sc["q8"], kv_cache)
        repeat = all(starmap(same, zip(again, first, strict=True)))
        rows.append(
            f"  {'NaN scratch repeat':<22} {'bitwise' if repeat else 'DIFFERS  <-- FAIL'}"
        )
        ok &= repeat
        if verbose or not ok:
            say(f"rank{rank} T={tokens}\n" + "\n".join(rows))
    verdict(ok, group, "glmf_mla_front_tp4")


def benchmark(group: dist.ProcessGroup, verbose: bool = True) -> dict[str, Any]:
    """Cold-weight timing per window: ``LAYERS`` distinct weight copies (default 8)
    back to back, ``REPS`` times (default 4) in one CUDA graph."""
    rank = group.rank()
    device = torch.device("cuda", torch.cuda.current_device())
    layers = int(os.environ.get("LAYERS", "8"))
    reps = int(os.environ.get("REPS", "4"))
    results = {}
    for tokens in TOKENS:
        args = []
        for i in range(layers):
            w, x, kv_cache, cache_loc = _inputs(i, tokens, device)
            args.append(kernel_args(x, prepared_weights(w), cache_loc, kv_cache))
        kernel = compile_kernel(args[0])

        def step(args=args, kernel=kernel) -> None:
            for a in args:
                kernel(*a)

        median, minimum = graph_time(step, layers, reps)
        if verbose:
            say(f"rank{rank} T={tokens} us/launch med {median:.2f} min {minimum:.2f}")
        results[tokens] = {"latency_us": median, "min_latency_us": minimum}
    return results


def main(verbose: bool = True, check_only: bool = False) -> dict[str, Any]:
    """The check on every rank, then the cold-weight timing (``check_only``: the
    check alone)."""
    return run_main(
        "glmf_mla_front_tp4", correctness_check, benchmark, verbose, check_only
    )


if __name__ == "__main__":
    main(check_only="--check" in sys.argv)
