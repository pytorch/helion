# ruff: noqa: ANN001, ANN201, ANN202
# pyrefly: ignore-errors
"""GLM-5.3 MTP draft input at TP4 (one request: 1 row a draft decode step, 6 a
draft extend), from the token ids and the previous hidden states to the
decoder layer's normed input and residual, as one persistent Gluon megakernel.
Replaces the stock vocab-parallel embedding + its all-reduce, fused_eh_norm,
the replicated eh_proj GEMM (BF16 [6144, 12288], 151 MB a rank) and the
decoder's input RMSNorm.

The roots:

- e norm / h norm: per token, the embedding row (the replicated table:
  SGLANG_ENABLE_EMBED_REPLICATION) through enorm, the previous hidden state
  through hnorm, in FP32 rounded once to BF16 (stock fused_eh_norm): ``eh_in``.
- eh: this rank's 6144 / world eh_proj output rows (column parallel, a row
  slice of the weight, 37.7 MB at TP4) over the whole K = 12288, as FP32
  partials of 12 K chunks.
- publish: per (token, 128 columns), the chunk-order sum rounded to BF16 (the
  GEMM's output) into this rank's symmetric slice.
- gather + norm: per token, every rank's slice in rank order: the residual (the
  eh output), and the decoder's input RMSNorm of it.

Column parallel, eh_proj keeps the stock GEMM's single FP32 accumulation and
BF16 rounding per output; its K chunks are summed in order and the slices
gathered in rank order (no atomics): the outputs are run-to-run deterministic
and the same on every rank.

``correctness_check()`` compares with a torch reference of the stock chain
(``_setup.eh_reference``) at both served windows; ``main()`` runs it, then
times the kernel against the stock chain (torch ops standing in for SGLang's
kernels) with cold weights::

    NVSHMEM_DISABLE_CUDA_VMM=1 PYTHONPATH=. \\
      python -m torch.distributed.run --standalone --nproc-per-node=4 \\
      pretuned_kernels/megakernels/glm5_3_tp4/glm5_3_draft_eh_tp4.py [--check]
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
from pretuned_kernels.megakernels.glm5_3_tp4._setup import HIDDEN
from pretuned_kernels.megakernels.glm5_3_tp4._setup import VOCAB
import torch
import torch.distributed as dist

import helion
import helion.language as hl

_WORLD = 4
# The served windows: the MTP draft's decode step and its extend.
TOKENS = (1, 6)

# The eh weight stream rides the TMA ring with its activation operand
# (``_ring_config.ring_config_fields(bound, RING_LOADS)``); every other memory
# op is a pointer access. Every ring slot precedes the gather's 4 memory ops per
# rank.
RING_LOADS = ("eh_w", "eh_in_chunked")
_RING_INDEXING = (14, 15)
_MEMORY_OPS = 37 + 4 * _WORLD

CONFIG = {
    "block_sizes": [16, 64],
    "num_warps": 4,
    "pid_type": "persistent_blocked",
    "cross_loop_pipeline": "dynamic",
    "num_sm_multiplier": 1,
    "num_stages": 4,
    "indexing": [
        "tensor_descriptor" if op in _RING_INDEXING else "pointer"
        for op in range(_MEMORY_OPS)
    ],
    "range_warp_specializes": [None, None, None, None, True, None, None, None, None],
}


def glm5_3_draft_eh(
    input_ids,  # [T] int64
    embed,  # [V, 6144] bf16: the replicated embedding table
    prev_hidden,  # [T, 6144] bf16: the target's (draft extend) or previous step's hidden states
    enorm,  # [6144] bf16
    hnorm,  # [6144] bf16
    eh_local,  # [6144 / world, 12288] bf16: this rank's eh_proj output rows
    input_norm,  # [6144] bf16: the decoder's input_layernorm
    symmetric_eh,  # [T, 6144 / world] bf16 symmetric memory: this rank's eh columns
    group_name: hl.ProcessGroupName,
):
    hl.specialize(input_ids.size())
    hl.specialize(embed.size())
    hl.specialize(embed.stride())
    hl.specialize(prev_hidden.size())
    hl.specialize(prev_hidden.stride())
    hl.specialize(enorm.size())
    hl.specialize(hnorm.size())
    hl.specialize(eh_local.size())
    hl.specialize(eh_local.stride())
    hl.specialize(input_norm.size())
    hl.specialize(symmetric_eh.size())
    hl.specialize(symmetric_eh.stride())

    rows, hidden = prev_hidden.shape
    n_local, eh_k = eh_local.shape
    assert rows <= 8 and eh_k == 2 * hidden
    dev = prev_hidden.device
    # Literals: a module-level int reaches the kernel as a runtime scalar.
    eps = 1e-5
    norm_n = 1024
    norm_blocks = hidden // norm_n
    chunks = 12
    chunk_k = eh_k // chunks
    gather_n = 512
    gather_blocks = n_local // gather_n
    eh_in = torch.empty((rows, eh_k), dtype=torch.bfloat16, device=dev)
    eh_in_halves = eh_in.view(rows, 2, hidden)
    eh_in_chunked = eh_in.view(rows, chunks, chunk_k)
    eh_w = eh_local.view(n_local, chunks, chunk_k)
    eh_parts = torch.empty((chunks, rows, n_local), dtype=torch.float32, device=dev)
    remote_eh = torch.ops.symm_mem.get_remote_tensors(symmetric_eh, group_name)
    residual_out = torch.empty((rows, hidden), dtype=torch.bfloat16, device=dev)
    output = torch.empty((rows, hidden), dtype=torch.bfloat16, device=dev)
    # Sums of squares per (token, source, 1024-column block); the gathered rows
    # (FP32) and theirs per (token, rank, 512-column block).
    norm_ss = torch.empty((rows, 2, norm_blocks), dtype=torch.float32, device=dev)
    world = len(remote_eh)
    gather_parts = world * gather_blocks
    norm_partial = torch.empty((rows, gather_parts), dtype=torch.float32, device=dev)
    norm_input = torch.empty((rows, hidden), dtype=torch.float32, device=dev)

    hl.pdl_wait()
    hl.prefetch_code()

    # Sums of squares per (token, 1024-column block): the embedding row's, the
    # previous hidden state's.
    for es_t, es_b in hl.tile([rows, norm_blocks], block_size=[1, 1]):
        es_tok = input_ids[es_t]
        es_x = hl.load(
            embed, [es_tok[:, None], (es_b.begin * norm_n + hl.arange(norm_n))[None, :]]
        ).to(torch.float32)
        norm_ss[es_t, 0, es_b] = torch.sum(es_x * es_x, dim=1)[:, None]
    for hs_t, hs_b in hl.tile([rows, norm_blocks], block_size=[1, 1]):
        hs_x = prev_hidden[hs_t, hs_b.begin * norm_n + hl.arange(norm_n)].to(
            torch.float32
        )
        norm_ss[hs_t, 1, hs_b] = torch.sum(hs_x * hs_x, dim=1)[:, None]

    # e norm / h norm per (token, block): enorm / hnorm in FP32, rounded once
    # (stock fused_eh_norm).
    for en_t, en_b in hl.tile([rows, norm_blocks], block_size=[1, 1]):
        en_tok = input_ids[en_t]
        en_inv = torch.rsqrt(torch.sum(norm_ss[en_t, 0, :], dim=1) / hidden + eps)
        en_cols = en_b.begin * norm_n + hl.arange(norm_n)
        en_x = hl.load(embed, [en_tok[:, None], en_cols[None, :]]).to(torch.float32)
        eh_in_halves[en_t, 0, en_cols] = (
            en_x * en_inv[:, None] * enorm[en_cols].to(torch.float32)[None, :]
        ).to(torch.bfloat16)
    for hn_t, hn_b in hl.tile([rows, norm_blocks], block_size=[1, 1]):
        hn_inv = torch.rsqrt(torch.sum(norm_ss[hn_t, 1, :], dim=1) / hidden + eps)
        hn_cols = hn_b.begin * norm_n + hl.arange(norm_n)
        hn_x = prev_hidden[hn_t, hn_cols].to(torch.float32)
        eh_in_halves[hn_t, 1, hn_cols] = (
            hn_x * hn_inv[:, None] * hnorm[hn_cols].to(torch.float32)[None, :]
        ).to(torch.bfloat16)

    # eh: this rank's output rows, FP32 partials of the K chunks.
    for eh_n, eh_c, eh_t in hl.tile([n_local, chunks, rows], block_size=[128, 1, None]):
        eh_acc = hl.zeros([eh_n, eh_t], dtype=torch.float32)
        for eh_kk in hl.tile(chunk_k):
            eh_acc = hl.dot(
                eh_w[eh_n, eh_c.begin, eh_kk],
                eh_in_chunked[eh_t, eh_c.begin, eh_kk].T,
                acc=eh_acc,
            )
        eh_parts[eh_c.begin, eh_t, eh_n] = eh_acc.T

    # Publish per (token, 128 columns): the chunk-order sum rounded to BF16.
    for pb_t, pb_n in hl.tile([rows, n_local], block_size=[1, 128]):
        pb_x = eh_parts[0, pb_t.begin, pb_n]
        for pb_c in hl.static_range(1, chunks):
            pb_x = pb_x + eh_parts[pb_c, pb_t.begin, pb_n]
        symmetric_eh[pb_t.begin, pb_n] = pb_x.to(torch.bfloat16)

    hl.pdl_launch_dependents()
    # Gather per (token, 512-column block of a slice): every rank's block (one
    # read each), the residual, its FP32 copy and sum of squares.
    for cm_t, cm_b in hl.tile([rows, gather_blocks], block_size=[1, 1]):
        cm_local = cm_b.begin * gather_n + hl.arange(gather_n)
        for cm_r, cm_remote in enumerate(remote_eh):
            cm_x = cm_remote[cm_t, cm_local]
            cm_cols = cm_r * n_local + cm_local
            residual_out[cm_t, cm_cols] = cm_x
            cm_f = cm_x.to(torch.float32)
            norm_input[cm_t, cm_cols] = cm_f
            norm_partial[cm_t, cm_r * gather_blocks + cm_b.begin] = torch.sum(
                cm_f * cm_f, dim=1
            )
    # The decoder's input RMSNorm per (token, 512 columns).
    for nm_t, nm_n in hl.tile([rows, hidden], block_size=[1, gather_n]):
        nm_token = nm_t.begin
        nm_weights = input_norm[nm_n].to(torch.float32)
        nm_scale = torch.rsqrt(torch.sum(norm_partial[nm_token, :]) / hidden + eps)
        output[nm_token, nm_n] = (
            norm_input[nm_token, nm_n] * nm_scale * nm_weights
        ).to(torch.bfloat16)
    return output, residual_out


glm5_3_draft_eh_tp4 = helion.kernel(
    glm5_3_draft_eh,
    static_shapes=False,
    backend="gluon",
    triton_do_not_specialize=False,
    config=helion.Config(**CONFIG),
    ignore_warnings=[helion.exc.TensorOperationInWrapper, helion.exc.FastMathUnused],
)


def _args(ids, prev, w, rank, sym, group_name):
    n_local = HIDDEN // _WORLD
    return (
        ids,
        w["embed"],
        prev,
        w["enorm"],
        w["hnorm"],
        w["eh"][rank * n_local : (rank + 1) * n_local],
        w["input_norm"],
        sym,
        group_name,
    )


def correctness_check(group: dist.ProcessGroup, verbose: bool = True) -> None:
    """Per served window and two inputs: the residual (the eh output) and the
    normed output within 1e-2 relative RMS of ``_setup.eh_reference``, a second
    launch bitwise equal and every rank's outputs the same bits."""
    rank, world = group.rank(), group.size()
    device = torch.device("cuda", torch.cuda.current_device())
    w = _setup.eh_weights(device)
    ok = True
    for tokens in TOKENS:
        sym = symmetric_partials(tokens, HIDDEN // world, group.group_name, device)
        for seed in (1, 2):
            ids, prev = _setup.eh_inputs(tokens, device, seed)
            out, res = glm5_3_draft_eh_tp4(
                *_args(ids, prev, w, rank, sym[0], group.group_name)
            )
            torch.cuda.synchronize()
            ref_out, ref_res = _setup.eh_reference(ids, prev, w)
            rows = []
            ok &= report(rows, "residual (eh out)", res, ref_res, 1e-2)
            ok &= report(rows, "normed output", out, ref_out, 1e-2)
            again = glm5_3_draft_eh_tp4(
                *_args(ids, prev, w, rank, sym[1], group.group_name)
            )
            same = torch.equal(again[0], out) and torch.equal(again[1], res)
            peers = [torch.empty_like(res) for _ in range(world)]
            dist.all_gather(peers, res, group=group)
            agree = all(torch.equal(p, res) for p in peers)
            rows.extend(
                [
                    f"  {'determinism':<22} {'bitwise' if same else 'DIFFERS  <-- FAIL'}",
                    f"  {'ranks agree':<22} {'bitwise' if agree else 'DIFFER  <-- FAIL'}",
                ]
            )
            ok &= same and agree
            if verbose or not ok:
                say(f"rank{rank} T={tokens} seed {seed}\n" + "\n".join(rows))
    verdict(ok, group, "glm5_3_draft_eh_tp4")


def _stock_step(ids, prev, w, group):
    """The stock chain at TP: the vocab-parallel embedding (masked gather +
    all-reduce), fused_eh_norm, the replicated eh_proj GEMM and the input
    RMSNorm (torch ops standing in for SGLang's kernels)."""
    rank, world = group.rank(), group.size()
    shard = VOCAB // world
    local = w["embed"][rank * shard : (rank + 1) * shard]

    def call():
        mask = (ids >= rank * shard) & (ids < (rank + 1) * shard)
        e = local[(ids - rank * shard).clamp(0, shard - 1)] * mask[:, None]
        dist.all_reduce(e, group=group)
        eh_in = torch.cat(
            [_setup.rms(e, w["enorm"]), _setup.rms(prev, w["hnorm"])], dim=-1
        )
        r = torch.mm(eh_in, w["eh"].T)
        return _setup.rms(r, w["input_norm"]), r

    return call


def benchmark(group: dist.ProcessGroup, verbose: bool = True) -> dict[str, Any]:
    """Cold-weight timing per served window against the stock chain:
    ``LAYERS`` distinct eh weight copies back to back (default 4), ``REPS``
    times (default 4) in one CUDA graph."""
    rank, world = group.rank(), group.size()
    device = torch.device("cuda", torch.cuda.current_device())
    layers = int(os.environ.get("LAYERS", "4"))
    reps = int(os.environ.get("REPS", "4"))
    weights = [_setup.eh_weights(device, seed=10 + i) for i in range(layers)]
    megabytes = 2 * (HIDDEN // world) * 2 * HIDDEN / 1e6
    results = {}
    for tokens in TOKENS:
        ids, prev = _setup.eh_inputs(tokens, device)
        sym = symmetric_partials(tokens, HIDDEN // world, group.group_name, device)
        args = [
            _args(ids, prev, w, rank, sym[i % 2], group.group_name)
            for i, w in enumerate(weights)
        ]

        def step(args=args) -> None:
            for a in args:
                glm5_3_draft_eh_tp4(*a)

        median, minimum = graph_time(step, layers, reps)
        stocks = [_stock_step(ids, prev, w, group) for w in weights]

        def stock_step(stocks=stocks) -> None:
            for s in stocks:
                s()

        stock_median, _ = graph_time(stock_step, layers, reps)
        if verbose and rank == 0:
            say(
                f"T={tokens}: helion us/launch med {median:.2f} min {minimum:.2f} "
                f"(~{megabytes:.0f} MB -> {megabytes / median:.2f} TB/s) | stock "
                f"chain med {stock_median:.2f}"
            )
        results[tokens] = {"latency_us": median, "stock_us": stock_median}
    return results


@torch.inference_mode()
def main(verbose: bool = True, check_only: bool = False) -> dict[str, Any]:
    """The four-rank check, then the timing (``check_only``: the check alone)."""
    return run_main(
        "glm5_3_draft_eh_tp4", correctness_check, benchmark, verbose, check_only
    )


if __name__ == "__main__":
    main(check_only="--check" in sys.argv)
