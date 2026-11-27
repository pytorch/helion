# ruff: noqa: ANN001, ANN201, ANN202
# pyrefly: ignore-errors
"""GLM-5.3 (GlmMoeDsa) MLA prologue for a TP4 speculative-verify window (one
request, T <= 8 tokens), from the layer's input RMSNorm output to the FP8
queries and the window's MLA cache rows, as one persistent Gluon megakernel.

Per rank: 16 heads. The roots:

- q_a: SGLang replicates the fused q_a / kv_a projection; here each rank
  computes a quarter of the q_a rows (BF16 [512, 6144]) as FP32 partials of 24
  K chunks and publishes their chunk-order sum, rounded to BF16 (the GEMM
  output), in symmetric memory; the kv_a | k_pe rows (BF16 [576, 6144]) stay
  replicated, FP32 partials of 6 K chunks in 64-row tasks.
- q norm: per token, every rank's q_a in rank order, the q_a RMSNorm (BF16):
  ``q_lora``, the indexer's input.
- q_b: BF16 [16 * 256, 2048] as FP32 partials of 4 K chunks.
- absorb: per (head, 128 latent rows), W_UK over the head's 192 nope dims of
  q (the q_b partials summed, BF16), BF16, then E4M3.
- q rope: per token, the 64 pe dims of each head's q (BF16), interleaved RoPE,
  E4M3.
- kv: per token, the kv_a rows (BF16) through the kv_a RMSNorm (BF16) and the
  RoPE'd k_pe, E4M3 into the window's MLA cache slot.

It serves the target's verify window and the MTP draft's attention (one row a
decode step, the window in the draft extend). The split-K partials are summed
in a fixed order and q_a's in rank order (no float atomics): the outputs are
run-to-run deterministic.

``correctness_check()`` compares every rank with the torch reference
(``_setup``) at every served window; ``main()`` runs it, then times the kernel
with cold weights::

    NVSHMEM_DISABLE_CUDA_VMM=1 PYTHONPATH=. \\
      python -m torch.distributed.run --standalone --nproc-per-node=4 \\
      pretuned_kernels/megakernels/glm5_3_tp4/glm5_3_attn_front_tp4.py [--check]
"""

from __future__ import annotations

import os
import sys
from typing import Any

from pretuned_kernels.megakernels._qwen3_5_tp4 import graph_time
from pretuned_kernels.megakernels._qwen3_5_tp4 import report
from pretuned_kernels.megakernels._qwen3_5_tp4 import run_main
from pretuned_kernels.megakernels._qwen3_5_tp4 import say
from pretuned_kernels.megakernels._qwen3_5_tp4 import symmetric_partials
from pretuned_kernels.megakernels._qwen3_5_tp4 import verdict
from pretuned_kernels.megakernels.glm5_3_tp4 import _setup
import torch
import torch.distributed as dist

import helion
import helion.language as hl

# The weight streams ride the TMA ring with their activation operands
# (``_ring_config.ring_config_fields(bound, RING_LOADS)``); every other memory
# op is a pointer access. Memory-op slots in source order, for the 4 ranks: the
# q_a and kv_a roots (0-5); the publish's 6 * 4 chunk loads and its store; the q
# norm's 4 loads, then 4 (load, norm weight, store) triples; q_b (weight,
# activation, store), its absorb's 4 partials, W_UK; 33 more after it.
RING_LOADS = (
    "qa_w",
    "x_qa_chunked",
    "kv_w",
    "x_chunked",
    "qb_w",
    "q_lora_chunked",
    "w_uk",
)
_WORLD = 4
_QB = 7 + 10 * _WORLD
_RING_INDEXING = (0, 1, 3, 4, _QB, _QB + 1, _QB + 7)
_MEMORY_OPS = _QB + 41

CONFIG = {
    "block_sizes": [16, 64, 16, 64, 16, 64, 16, 64],
    "num_warps": 4,
    "pid_type": "persistent_blocked",
    "cross_loop_pipeline": "dynamic",
    "num_sm_multiplier": 1,
    "num_stages": 4,
    "indexing": [
        "tensor_descriptor" if op in _RING_INDEXING else "pointer"
        for op in range(_MEMORY_OPS)
    ],
    "range_warp_specializes": [True, None] * 2
    + [None] * 2
    + [True, None] * 2
    + [None] * 2,
    "cross_loop_inband_protocol": ["lamport"],
    # q_b's weight tiles stream into tensor memory ahead of its tasks' waits
    # (the absorb's W_UK loop takes none: its dot issues from the body).
    "cross_loop_tmem_prefetch": [True, False],
}


def rope_interleaved(x, cos, sin, sin_first=False):
    """Interleaved RoPE of ``x`` [n, 64] (pairs 2i, 2i + 1 by frequency i), FP32.
    ``sin_first``: the odd lanes as x0 sin + x1 cos (the indexer's order; with
    FMA contraction the order changes the bits)."""
    n = x.size(0)
    x0, x1 = hl.split(x.reshape(n, 32, 2))
    even = x0 * cos - x1 * sin
    odd = x0 * sin + x1 * cos if sin_first else x1 * cos + x0 * sin
    return hl.join(even, odd).reshape(n, 64)


@helion.kernel(
    static_shapes=False,
    backend="gluon",
    triton_do_not_specialize=False,
    config=helion.Config(**CONFIG),
    ignore_warnings=[helion.exc.TensorOperationInWrapper, helion.exc.FastMathUnused],
)
def glm5_3_attn_front_tp4(
    hidden_states,  # [T, 6144] bf16: the layer's input RMSNorm output
    q_a_local,  # [2048 / world, 6144] bf16: this rank's q_a rows
    kv_a,  # [576, 6144] bf16: kv_a (512) | k_pe (64), replicated
    q_a_norm,  # [2048] bf16
    kv_a_norm,  # [512] bf16
    q_b,  # [4096, 2048] bf16: per head nope (192) | pe (64)
    w_uk,  # [16, 512, 192] bf16: W_UK per head (K contiguous)
    cos_sin,  # [max_position, 64] f32: cos (32) | sin (32)
    positions,  # [T] int64
    cache_loc,  # [T] int64: the window's MLA cache slots
    kv_cache,  # [slots, 576] e4m3: one layer's raw MLA cache
    symmetric_q_a,  # [T, 2048 / world] bf16 symmetric memory: this rank's q_a
    group_name: hl.ProcessGroupName,
):
    hl.specialize(hidden_states.size())
    hl.specialize(hidden_states.stride())
    hl.specialize(q_a_local.size())
    hl.specialize(q_a_local.stride())
    hl.specialize(kv_a.size())
    hl.specialize(kv_a.stride())
    hl.specialize(symmetric_q_a.size())
    hl.specialize(symmetric_q_a.stride())
    hl.specialize(q_a_norm.size())
    hl.specialize(kv_a_norm.size())
    hl.specialize(q_b.size())
    hl.specialize(q_b.stride())
    hl.specialize(w_uk.size())
    hl.specialize(w_uk.stride())
    hl.specialize(cos_sin.stride())
    hl.specialize(positions.size())
    hl.specialize(cache_loc.size())
    hl.specialize(kv_cache.stride())

    rows, hidden = hidden_states.shape
    q_lora_n = q_a_norm.size(0)
    kv_lora = kv_a_norm.size(0)
    heads, _, nope = w_uk.shape
    qk_head = q_b.size(0) // heads
    rope = qk_head - nope
    assert rows <= 8 and heads == 16 and kv_lora == 512 and nope == 192 and rope == 64
    dev = hidden_states.device
    # Literals: a module-level int reaches the kernel as a runtime scalar.
    eps = 1e-5
    a_chunks = 6
    a_chunk_k = hidden // a_chunks
    x_chunked = hidden_states.view(rows, a_chunks, a_chunk_k)
    # This rank's q_a rows in more K chunks (fewer rows, as many tasks).
    q_local = q_a_local.size(0)
    qa_chunks = a_chunks * (q_lora_n // q_local)
    qa_chunk_k = hidden // qa_chunks
    x_qa_chunked = hidden_states.view(rows, qa_chunks, qa_chunk_k)
    qa_w = q_a_local.view(q_local, qa_chunks, qa_chunk_k)
    kv_n = kv_lora + rope
    kv_w = kv_a.view(kv_n, a_chunks, a_chunk_k)
    qa_parts = torch.empty((qa_chunks, rows, q_local), dtype=torch.float32, device=dev)
    remote_q_a = torch.ops.symm_mem.get_remote_tensors(symmetric_q_a, group_name)
    kv_parts = torch.empty((a_chunks, rows, kv_n), dtype=torch.float32, device=dev)
    q_lora = torch.empty((rows, q_lora_n), dtype=torch.bfloat16, device=dev)
    b_chunks = 4
    b_chunk_k = q_lora_n // b_chunks
    q_lora_chunked = q_lora.view(rows, b_chunks, b_chunk_k)
    qb_w = q_b.view(heads * qk_head, b_chunks, b_chunk_k)
    qb_parts = torch.empty(
        (b_chunks, rows, heads, qk_head), dtype=torch.float32, device=dev
    )
    q8 = torch.empty(
        (rows, heads, kv_lora + rope), dtype=torch.float8_e4m3fn, device=dev
    )

    hl.pdl_wait()
    hl.prefetch_code()

    # This rank's q_a rows: FP32 partials of its K chunks.
    for qa_n, qa_c, qa_t in hl.tile(
        [q_local, qa_chunks, rows], block_size=[128, 1, None]
    ):
        qa_acc = hl.zeros([qa_n, qa_t], dtype=torch.float32)
        for qa_k in hl.tile(qa_chunk_k):
            qa_acc = hl.dot(
                qa_w[qa_n, qa_c.begin, qa_k],
                x_qa_chunked[qa_t, qa_c.begin, qa_k].T,
                acc=qa_acc,
            )
        qa_parts[qa_c.begin, qa_t, qa_n] = qa_acc.T

    # kv_a | k_pe rows.
    for kv_r, kv_c, kv_t in hl.tile([kv_n, a_chunks, rows], block_size=[64, 1, None]):
        kv_acc = hl.zeros([kv_r, kv_t], dtype=torch.float32)
        for kv_k in hl.tile(a_chunk_k):
            kv_acc = hl.dot(
                kv_w[kv_r, kv_c.begin, kv_k],
                x_chunked[kv_t, kv_c.begin, kv_k].T,
                acc=kv_acc,
            )
        kv_parts[kv_c.begin, kv_t, kv_r] = kv_acc.T

    # Publish per (token, 128 columns): this rank's q_a, the chunk-order sum
    # rounded to BF16 (the GEMM output), into its symmetric buffer.
    for qs_t, qs_n in hl.tile([rows, q_local], block_size=[1, 128]):
        qs_x = qa_parts[0, qs_t.begin, qs_n]
        for qs_a in hl.static_range(1, qa_chunks):
            qs_x = qs_x + qa_parts[qs_a, qs_t.begin, qs_n]
        symmetric_q_a[qs_t.begin, qs_n] = qs_x.to(torch.bfloat16)

    # q norm per token: every rank's q_a (in rank order), the q_a RMSNorm
    # (BF16): q_lora, the indexer's input.
    for ql_t in hl.tile(rows, block_size=1):
        ql_token = ql_t.begin
        ql_sumsq = hl.zeros([1], dtype=torch.float32)
        for ql_remote in remote_q_a:
            ql_x = ql_remote[ql_token, :].to(torch.float32)
            ql_sumsq = ql_sumsq + torch.sum(ql_x * ql_x)
        ql_inv = torch.rsqrt(ql_sumsq / q_lora_n + eps)
        for ql_r, ql_remote in enumerate(remote_q_a):
            ql_cols = ql_r * q_local + hl.arange(q_local)
            q_lora[ql_token, ql_cols] = (
                ql_remote[ql_token, :].to(torch.float32)
                * ql_inv
                * q_a_norm[ql_cols].to(torch.float32)
            ).to(torch.bfloat16)

    # q_b: FP32 partials of 4 K chunks of q_lora. K chunks first: a head's tasks
    # are consecutive, so its absorb starts early.
    for qb_c, qb_n, qb_t in hl.tile(
        [b_chunks, heads * qk_head, rows], block_size=[1, 128, None]
    ):
        # The task's weight slab toward L2 ahead of q_lora (the ring holds only
        # its first steps): it streams under the q_a gather.
        hl.prefetch(qb_w, [qb_n, qb_c.begin, slice(None)])
        qb_acc = hl.zeros([qb_n, qb_t], dtype=torch.float32)
        for qb_k in hl.tile(b_chunk_k):
            qb_acc = hl.dot(
                qb_w[qb_n, qb_c.begin, qb_k],
                q_lora_chunked[qb_t, qb_c.begin, qb_k].T,
                acc=qb_acc,
            )
        qb_parts[qb_c.begin, qb_t, qb_n.begin // qk_head, qb_n.index % qk_head] = (
            qb_acc.T
        )

    hl.pdl_launch_dependents()
    # Absorb per (head, 128 latent rows): W_UK over the head's BF16 nope dims.
    for ab_h, ab_l, ab_t in hl.tile([heads, kv_lora, rows], block_size=[1, 128, None]):
        ab_acc = hl.zeros([ab_l, ab_t], dtype=torch.float32)
        for ab_k in hl.tile(nope):
            ab_q = qb_parts[0, ab_t, ab_h.begin, ab_k]
            for ab_c in hl.static_range(1, b_chunks):
                ab_q = ab_q + qb_parts[ab_c, ab_t, ab_h.begin, ab_k]
            ab_acc = hl.dot(
                w_uk[ab_h.begin, ab_l, ab_k], ab_q.to(torch.bfloat16).T, acc=ab_acc
            )
        q8[ab_t, ab_h.begin, ab_l] = ab_acc.T.to(torch.bfloat16).to(torch.float8_e4m3fn)

    # q rope per token: every head's pe dims.
    for qr_t in hl.tile(rows, block_size=1):
        qr_token = qr_t.begin
        qr_pos = positions[qr_token]
        qr_cos = cos_sin[qr_pos, : rope // 2]
        qr_sin = cos_sin[qr_pos, rope // 2 :]
        qr_pe = qb_parts[0, qr_token, :, nope:]
        for qr_c in hl.static_range(1, b_chunks):
            qr_pe = qr_pe + qb_parts[qr_c, qr_token, :, nope:]
        qr_pe = qr_pe.to(torch.bfloat16).to(torch.float32)
        q8[qr_token, :, kv_lora:] = rope_interleaved(
            qr_pe, qr_cos[None, :], qr_sin[None, :]
        ).to(torch.float8_e4m3fn)

    # kv per token: the normed latent and the RoPE'd k_pe into the cache slot.
    for kc_t in hl.tile(rows, block_size=1):
        kc_token = kc_t.begin
        kc_w = kv_a_norm[:].to(torch.float32)
        kc_pos = positions[kc_token]
        kc_slot = cache_loc[kc_token]
        kc_cos = cos_sin[kc_pos, : rope // 2]
        kc_sin = cos_sin[kc_pos, rope // 2 :]
        kc_x = kv_parts[0, kc_token, :kv_lora]
        kc_pe = kv_parts[0, kc_token, kv_lora:]
        for kc_c in hl.static_range(1, a_chunks):
            kc_x = kc_x + kv_parts[kc_c, kc_token, :kv_lora]
            kc_pe = kc_pe + kv_parts[kc_c, kc_token, kv_lora:]
        kc_x = kc_x.to(torch.bfloat16).to(torch.float32)
        kc_inv = torch.rsqrt(torch.sum(kc_x * kc_x) / kv_lora + eps)
        kv_cache[kc_slot, :kv_lora] = (
            (kc_x * kc_inv * kc_w).to(torch.bfloat16).to(torch.float8_e4m3fn)
        )
        kc_pe = kc_pe.to(torch.bfloat16).to(torch.float32)
        kv_cache[kc_slot, kv_lora:] = (
            rope_interleaved(kc_pe[None, :], kc_cos[None, :], kc_sin[None, :])
            .reshape(rope)
            .to(torch.float8_e4m3fn)
        )
    return q_lora, q8


# The served windows: the target's verify window (EAGLE 5/1/6; the MTP draft
# extend's is 6 too) and the MTP draft's decode step.
TOKENS = (6, 1)


def _inputs(shape, layer: int, rank: int, tokens: int, device, context: int = 1000):
    """Layer ``layer``'s (replicated) weights with this rank's q_a rows, an input
    window and an MLA cache."""
    w = _setup.front_weights(shape, seed=1000 * layer, device=device)
    q_local = shape.q_lora // _WORLD
    w["q_a_local"] = w["qkv_a"][rank * q_local : (rank + 1) * q_local].contiguous()
    w["kv_a"] = w["qkv_a"][shape.q_lora :].contiguous()
    g = torch.Generator(device=device)
    g.manual_seed(50 + layer)
    x = torch.randn(tokens, shape.hidden, generator=g, device=device).to(torch.bfloat16)
    cs = _setup.cos_sin_cache(32768, device=device)
    positions = torch.arange(context, context + tokens, device=device)
    kv_cache = torch.zeros(
        (4096, shape.kv_lora + shape.rope), dtype=torch.float8_e4m3fn, device=device
    )
    cache_loc = torch.randperm(4096, generator=g, device=device)[:tokens]
    return w, x, cs, positions, kv_cache, cache_loc


def _args(w, x, cs, positions, kv_cache, cache_loc, partial, group_name):
    return (
        x,
        w["q_a_local"],
        w["kv_a"],
        w["q_a_norm"],
        w["kv_a_norm"],
        w["q_b"],
        w["w_uk"],
        cs,
        positions,
        cache_loc,
        kv_cache,
        partial,
        group_name,
    )


def correctness_check(group: dist.ProcessGroup, verbose: bool = True) -> None:
    """Every rank against ``_setup.front_reference`` per served window: q_lora,
    the FP8 queries and the window's cache rows within 2e-2 relative RMS, and a
    second launch bitwise equal."""
    rank = group.rank()
    device = torch.device("cuda", torch.cuda.current_device())
    shape = _setup.AttnShape()
    ok = True
    for tokens in TOKENS:
        w, x, cs, positions, kv_cache, cache_loc = _inputs(
            shape, 0, rank, tokens, device
        )
        partials = symmetric_partials(
            tokens, shape.q_lora // _WORLD, group.group_name, device
        )
        q_lora, q8 = glm5_3_attn_front_tp4(
            *_args(
                w, x, cs, positions, kv_cache, cache_loc, partials[0], group.group_name
            )
        )
        torch.cuda.synchronize()
        ref_lora, ref_q8, ref_kv8 = _setup.front_reference(x, w, positions, cs, shape)
        rows = []
        ok &= report(rows, "q_lora", q_lora, ref_lora, 2e-2)
        ok &= report(rows, "q (FP8)", q8.float(), ref_q8.float(), 2e-2)
        ok &= report(
            rows,
            "cache rows (FP8)",
            kv_cache[cache_loc].float(),
            ref_kv8.float(),
            2e-2,
        )
        again = glm5_3_attn_front_tp4(
            *_args(
                w, x, cs, positions, kv_cache, cache_loc, partials[1], group.group_name
            )
        )
        deterministic = torch.equal(again[1].view(torch.uint8), q8.view(torch.uint8))
        rows.append(
            f"  {'determinism':<22} {'bitwise' if deterministic else 'DIFFERS  <-- FAIL'}"
        )
        ok &= deterministic
        if verbose or not ok:
            say(f"rank{rank} glm5_3_attn_front_tp4 T={tokens}\n" + "\n".join(rows))
    verdict(ok, group, "glm5_3_attn_front_tp4")


def benchmark(group: dist.ProcessGroup, verbose: bool = True) -> dict[str, Any]:
    """Cold-weight timing per served window: ``LAYERS`` distinct weight copies
    (default 8, even) back to back, ``REPS`` times (default 4) in one CUDA
    graph."""
    rank = group.rank()
    device = torch.device("cuda", torch.cuda.current_device())
    layers = int(os.environ.get("LAYERS", "8"))
    reps = int(os.environ.get("REPS", "4"))
    shape = _setup.AttnShape()
    megabytes = (
        2
        * (
            (shape.q_lora // _WORLD + shape.kv_lora + shape.rope) * shape.hidden
            + shape.heads * (shape.nope + shape.rope) * shape.q_lora
            + shape.heads * shape.kv_lora * shape.nope
        )
        / 1e6
    )
    results = {}
    for tokens in TOKENS:
        partials = symmetric_partials(
            tokens, shape.q_lora // _WORLD, group.group_name, device
        )
        args = [
            _args(
                *_inputs(shape, i, rank, tokens, device),
                partials[i % 2],
                group.group_name,
            )
            for i in range(layers)
        ]

        def step(args=args) -> None:
            for a in args:
                glm5_3_attn_front_tp4(*a)

        median, minimum = graph_time(step, layers, reps)
        if verbose:
            say(
                f"rank{rank} T={tokens} us/launch med {median:.2f} min {minimum:.2f} "
                f"(~{megabytes:.0f} MB weights -> {megabytes / median:.2f} TB/s)"
            )
        results[tokens] = {"latency_us": median, "min_latency_us": minimum}
    return results


def main(verbose: bool = True, check_only: bool = False) -> dict[str, Any]:
    """The four-rank check, then the cold-weight timing (``check_only``: the
    check alone)."""
    return run_main(
        "glm5_3_attn_front_tp4", correctness_check, benchmark, verbose, check_only
    )


if __name__ == "__main__":
    main(check_only="--check" in sys.argv)
