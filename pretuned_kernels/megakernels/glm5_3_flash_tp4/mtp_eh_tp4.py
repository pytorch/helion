# ruff: noqa: ANN001, ANN201
# pyrefly: ignore-errors
"""GLM-5.3 Flash MTP draft input at TP4 (one request: 1 row a draft decode step, 6 the
draft extend), from the token ids and the previous hidden states to the decoder layer's
normed input and residual, as one persistent Gluon megakernel. Replaces the stock
vocab-parallel embedding + its all-reduce, fused_eh_norm, the replicated eh_proj GEMM
(BF16 [4096, 8192], 67 MB a rank) and the decoder's input RMSNorm. GLM-5.3's
``glm5_3_tp4/glm5_3_draft_eh_tp4.py`` with the embedding kept vocab parallel (no
replicated table: the target's configuration is unchanged) and the eh GEMV in K
chunks on the TMA ring.

The roots:

- e publish: per (token, 512 columns), when the token is in this rank's vocab shard, its embedding row
  (stock: the masked local lookup) through enorm in FP32 rounded once to BF16 (stock
  fused_eh_norm: bf16(x * r * w)), else zeros, into this rank's symmetric slot.
- h norm: per token, hnorm of the previous hidden state likewise.
- e gather: per (token, 128 columns), the ranks' slots summed (stock: the embedding
  all-reduce; one rank holds the row, so the sum is exact).
- eh (two roots: the h half, then the e half of K): this rank's 4096 / world eh_proj
  output rows (column parallel, a row slice of the weight, 16.8 MB at TP4) over the whole
  K = 8192, as FP32 partials of K chunks (the h half runs while the e row crosses the
  ranks).
- publish: per (token, 128 columns), the chunk-order sum rounded to BF16 (the GEMM's
  output) into this rank's symmetric slice.
- gather: per (token, 128 columns of every rank's slice), the ranks' columns in rank
  order: the residual (the eh output) and the block's sum of squares. (Narrow tasks: one
  task polling every rank's whole slice waited ~10 us past the last publish at 4 ranks.)
- norm: per (token, 512 columns), the decoder's input RMSNorm of the residual (SGLang's
  JIT RMSNorm: bf16((x * w) * r)).

Every cross-rank read reads a slot a root of the same launch wrote. The K chunks are
summed in order and the slices gathered in rank order (no atomics): the outputs are
run-to-run deterministic and the same on every rank.

``correctness_check()`` compares with a torch reference of the stock chain
(``glm5_3_tp4._setup.eh_reference``: tokens on every rank's shard, back-to-back
launches); ``main()`` runs it, then times the kernel with cold weights::

    NVSHMEM_DISABLE_CUDA_VMM=1 PYTHONPATH=. \\
      python -m torch.distributed.run --standalone --nproc-per-node=4 \\
      pretuned_kernels/megakernels/glm5_3_flash_tp4/mtp_eh_tp4.py [--check]
"""

from __future__ import annotations

import functools
import os
import sys
from typing import Any

from pretuned_kernels.megakernels._qwen3_5_tp4 import graph_time
from pretuned_kernels.megakernels._qwen3_5_tp4 import report
from pretuned_kernels.megakernels._qwen3_5_tp4 import run_main
from pretuned_kernels.megakernels._qwen3_5_tp4 import same
from pretuned_kernels.megakernels._qwen3_5_tp4 import say
from pretuned_kernels.megakernels._qwen3_5_tp4 import symmetric_buffer
from pretuned_kernels.megakernels._qwen3_5_tp4 import verdict
from pretuned_kernels.megakernels._ring_config import ring_config_fields
from pretuned_kernels.megakernels.glm5_3_flash_tp4 import _setup
from pretuned_kernels.megakernels.glm5_3_tp4._setup import VOCAB
from pretuned_kernels.megakernels.glm5_3_tp4._setup import eh_inputs
from pretuned_kernels.megakernels.glm5_3_tp4._setup import eh_reference
from pretuned_kernels.megakernels.glm5_3_tp4._setup import eh_weights
import torch
import torch.distributed as dist

import helion
import helion.language as hl

HIDDEN = _setup.HIDDEN
# K chunks of each half (e, h) of the eh GEMV (the host views the weight as
# [rows, 2, CHUNKS, 4096 / CHUNKS]).
CHUNKS = 8
RING_LOADS = ("eh_w", "e_in", "h_in")


def glmf_mtp_eh(
    input_ids,  # [T] int64
    embed,  # [V / W, 4096] bf16: this rank's vocab shard of the embedding table
    prev_hidden,  # [T, 4096] bf16: the target's (draft extend) or the previous step's hidden states
    enorm,  # [4096] bf16
    hnorm,  # [4096] bf16
    eh_w,  # [4096 / W, 2, chunks, 4096 / chunks] bf16: this rank's eh_proj output rows, [e | h] K halves (a view)
    input_norm,  # [4096] bf16: the decoder's input_layernorm
    e_in,  # [T, chunks, 4096 / chunks] bf16 scratch: enorm(e) (eh_proj's first K half)
    h_in,  # [T, chunks, 4096 / chunks] bf16 scratch: hnorm(h)
    e_parts,  # [chunks, T, 4096 / W] fp32 scratch
    h_parts,  # [chunks, T, 4096 / W] fp32 scratch
    symmetric_e,  # [T, 4096] bf16 symmetric: this rank's masked embedding rows
    symmetric_eh,  # [T, 4096 / W] bf16 symmetric: this rank's eh columns
    residual_out,  # [T, 4096] bf16 out: the eh output (the decoder's residual)
    output,  # [T, 4096] bf16 out: input_layernorm(residual)
    group_name: hl.ProcessGroupName,
    rank: hl.constexpr,
):
    hl.specialize(input_ids.size())
    hl.specialize(embed.size())
    hl.specialize(embed.stride())
    hl.specialize(prev_hidden.size())
    hl.specialize(prev_hidden.stride())
    hl.specialize(enorm.size())
    hl.specialize(hnorm.size())
    hl.specialize(eh_w.size())
    hl.specialize(eh_w.stride())
    hl.specialize(input_norm.size())
    hl.specialize(e_in.size())
    hl.specialize(e_in.stride())
    hl.specialize(h_in.size())
    hl.specialize(h_in.stride())
    hl.specialize(e_parts.size())
    hl.specialize(e_parts.stride())
    hl.specialize(h_parts.size())
    hl.specialize(h_parts.stride())
    hl.specialize(symmetric_e.size())
    hl.specialize(symmetric_e.stride())
    hl.specialize(symmetric_eh.size())
    hl.specialize(symmetric_eh.stride())
    hl.specialize(residual_out.size())
    hl.specialize(residual_out.stride())
    hl.specialize(output.size())
    hl.specialize(output.stride())

    rows, hidden = prev_hidden.shape
    shard = embed.size(0)
    n_local, _, chunks, chunk_k = eh_w.shape
    assert rows <= 8 and chunks * chunk_k == hidden
    # Literals: a module-level int reaches the kernel as a runtime scalar.
    eps = 1e-5
    e_flat = e_in.view(rows, hidden)
    h_flat = h_in.view(rows, hidden)
    remote_e = torch.ops.symm_mem.get_remote_tensors(symmetric_e, group_name)
    remote_eh = torch.ops.symm_mem.get_remote_tensors(symmetric_eh, group_name)

    hl.pdl_wait()
    hl.prefetch_code()

    # e publish per token: when this rank's shard holds the token, its embedding
    # row (stock: the masked local lookup) through enorm in FP32 rounded once to
    # BF16 (stock fused_eh_norm: bf16(x * r * w)), else zeros, into this rank's
    # symmetric slot. (Per 512 columns: eight tasks push a row's columns; each
    # reads the whole row for its sum of squares.)
    for ep_t, ep_n in hl.tile([rows, hidden], block_size=[1, 512]):
        ep_local = input_ids[ep_t] - rank * shard
        ep_mine = (ep_local >= 0) & (ep_local < shard)
        ep_row = torch.where(ep_mine, ep_local, 0)
        ep_all = hl.arange(hidden)
        ep_full = hl.load(
            embed, [ep_row[:, None], ep_all[None, :]], extra_mask=ep_mine[:, None]
        ).to(torch.float32)
        ep_inv = torch.rsqrt(torch.sum(ep_full * ep_full, dim=1) / hidden + eps)
        ep_x = hl.load(
            embed, [ep_row[:, None], ep_n.index[None, :]], extra_mask=ep_mine[:, None]
        ).to(torch.float32)
        ep_y = ep_x * ep_inv[:, None] * enorm[ep_n].to(torch.float32)[None, :]
        symmetric_e[ep_t, ep_n] = torch.where(ep_mine[:, None], ep_y, 0.0).to(
            torch.bfloat16
        )

    # h norm per (token, 512 columns): hnorm likewise (each task reads the whole
    # row for its sum of squares; the h GEMV's K chunk c waits for its one
    # task).
    for hn_t, hn_n in hl.tile([rows, hidden], block_size=[1, 512]):
        hn_all = hl.arange(hidden)
        hn_full = prev_hidden[hn_t, hn_all].to(torch.float32)
        hn_inv = torch.rsqrt(torch.sum(hn_full * hn_full, dim=1) / hidden + eps)
        hn_x = prev_hidden[hn_t, hn_n].to(torch.float32)
        h_flat[hn_t, hn_n] = (
            hn_x * hn_inv[:, None] * hnorm[hn_n].to(torch.float32)[None, :]
        ).to(torch.bfloat16)

    # e gather per (token, 512 columns): the ranks' slots summed (stock: the
    # embedding all-reduce; one rank holds the row, so the sum is exact).
    # (Remote reads indexed by the column tile itself: the dependency analysis
    # proves every word is polled, so the words travel in band as Lamport words;
    # the token by .begin, see below.)
    for eg_t, eg_n in hl.tile([rows, hidden], block_size=[1, 512]):
        eg_x = hl.zeros([eg_n], dtype=torch.float32)
        for eg_r in hl.static_range(len(remote_e)):
            eg_x = eg_x + remote_e[eg_r][eg_t.begin, eg_n].to(torch.float32)
        e_flat[eg_t.begin, eg_n] = eg_x.to(torch.bfloat16)

    # eh: this rank's output rows, FP32 partials of the K chunks. The h half
    # first (its operand is ready before the e row has crossed the ranks), then
    # the e half. (At T=1 TP4 the roots so far are 8 + 8 + 8 + 64 + 64 = 152
    # tasks, one wave: the e half's weight streams from the launch. A root that
    # would wrap onto busy workers is placed W-aligned behind the earlier roots'
    # tasks, so 32 e-gather tasks delayed the e half ~3 us.)
    for ha_c, ha_n, ha_t in hl.tile([chunks, n_local, rows], block_size=[1, 128, 16]):
        ha_acc = hl.zeros([ha_n, ha_t], dtype=torch.float32)
        for ha_kk in hl.tile(chunk_k, block_size=256):
            ha_acc = hl.dot(
                eh_w[ha_n, 1, ha_c.begin, ha_kk],
                h_in[ha_t, ha_c.begin, ha_kk].T,
                acc=ha_acc,
            )
        h_parts[ha_c.begin, ha_t, ha_n] = ha_acc.T
    for ea_c, ea_n, ea_t in hl.tile([chunks, n_local, rows], block_size=[1, 128, 16]):
        ea_acc = hl.zeros([ea_n, ea_t], dtype=torch.float32)
        for ea_kk in hl.tile(chunk_k, block_size=256):
            ea_acc = hl.dot(
                eh_w[ea_n, 0, ea_c.begin, ea_kk],
                e_in[ea_t, ea_c.begin, ea_kk].T,
                acc=ea_acc,
            )
        e_parts[ea_c.begin, ea_t, ea_n] = ea_acc.T

    # Publish per (token, 128 columns): the chunk-order sum (K order: the e
    # half's chunks, then the h half's) rounded to BF16.
    for pb_t, pb_n in hl.tile([rows, n_local], block_size=[1, 128]):
        pb_x = e_parts[0, pb_t.begin, pb_n]
        for pb_c in hl.static_range(1, chunks):
            pb_x = pb_x + e_parts[pb_c, pb_t.begin, pb_n]
        for pb_d in hl.static_range(chunks):
            pb_x = pb_x + h_parts[pb_d, pb_t.begin, pb_n]
        symmetric_eh[pb_t.begin, pb_n] = pb_x.to(torch.bfloat16)

    hl.pdl_launch_dependents()
    # Gather + the decoder's input RMSNorm per (token, 128 columns of every
    # rank's slice): the row's sum of squares over every rank's whole slice
    # (rank order, so every task gets the same scale; a full slice, so the
    # analysis proves every word polled and the words stay Lamport), then each
    # rank's columns of the task: the residual (BF16) and bf16((x * w) * r)
    # (SGLang's JIT RMSNorm order). (A separate norm root was one more local
    # hop.) In-band reads index the token by a scalar (.begin): a block-1 token
    # tile on the size-1 token dim of T=1 broadcasts, which polls element by
    # element (one serialized spin per element: ~2 us for 8 per thread) instead
    # of the CTA-wide word poll.
    for ga_t, ga_n in hl.tile([rows, n_local], block_size=[1, 128]):
        ga_ss = hl.zeros([], dtype=torch.float32)
        for ga_q in hl.static_range(len(remote_eh)):
            ga_y = remote_eh[ga_q][ga_t.begin, :].to(torch.float32)
            ga_ss = ga_ss + torch.sum(ga_y * ga_y)
        ga_scale = torch.rsqrt(ga_ss / hidden + eps)
        for ga_p in hl.static_range(len(remote_eh)):
            ga_x = remote_eh[ga_p][ga_t.begin, ga_n].to(torch.float32)
            ga_cols = ga_p * n_local + ga_n.index
            residual_out[ga_t.begin, ga_cols] = ga_x.to(torch.bfloat16)
            output[ga_t.begin, ga_cols] = (
                (ga_x * input_norm[ga_cols].to(torch.float32)) * ga_scale
            ).to(torch.bfloat16)


glmf_mtp_eh_tp4 = helion.kernel(
    glmf_mtp_eh,
    static_shapes=False,
    backend="gluon",
    triton_do_not_specialize=False,
    ignore_warnings=[helion.exc.TensorOperationInWrapper, helion.exc.FastMathUnused],
)


def mk_config(bound) -> helion.Config:
    """The eh weight stream and its activation operand on the warp-specialized TMA ring,
    every other access by pointer; dynamic dispatch, Lamport in-band transport."""
    fields = ring_config_fields(bound, RING_LOADS)
    default = bound.config_spec.default_config().config
    return helion.Config(
        block_sizes=default["block_sizes"],
        indexing=fields["indexing"],
        pid_type="persistent_blocked",
        num_sm_multiplier=1,
        num_stages=3,
        num_warps=4,
        cross_loop_pipeline="dynamic",
        range_warp_specializes=fields["range_warp_specializes"],
        cross_loop_inband_protocol=["lamport"]
        * len(default.get("cross_loop_inband_protocol", [])),
    )


@functools.cache
def scratch(device: torch.device, tokens: int, n_local: int) -> dict[str, torch.Tensor]:
    """The kernel's scratch buffers on ``device``, one set for every call."""
    f32, bf16, k = torch.float32, torch.bfloat16, HIDDEN // CHUNKS
    return {
        "e_in": torch.zeros(tokens, CHUNKS, k, device=device, dtype=bf16),
        "h_in": torch.zeros(tokens, CHUNKS, k, device=device, dtype=bf16),
        "e_parts": torch.zeros(CHUNKS, tokens, n_local, device=device, dtype=f32),
        "h_parts": torch.zeros(CHUNKS, tokens, n_local, device=device, dtype=f32),
    }


# Symmetric buffers live for the process (torch finds a handle by address).
@functools.cache
def symmetric_buffers(
    tokens: int, n_local: int, group_name: str, device
) -> list[tuple]:
    """Two (e rows [T, 4096], eh columns [T, 4096 / W]) BF16 symmetric pairs for
    consecutive launches to alternate between."""
    return [
        tuple(
            symmetric_buffer((tokens, width), torch.bfloat16, group_name, device)
            for width in (HIDDEN, n_local)
        )
        for _ in range(2)
    ]


def kernel_args(
    ids,
    embed_shard,
    prev,
    enorm,
    hnorm,
    eh_local,
    input_norm,
    sym,
    out,
    group_name,
    rank,
) -> tuple:
    """``glmf_mtp_eh_tp4``'s arguments: ``embed_shard`` this rank's vocab shard of the
    embedding [V / W, 4096], ``eh_local`` its [4096 / W, 8192] eh_proj rows, ``sym``
    one ``symmetric_buffers`` pair, ``out`` = (residual, normed) [T, 4096] BF16."""
    tokens = prev.shape[0]
    n_local = eh_local.shape[0]
    sc = scratch(prev.device, tokens, n_local)
    return (
        ids,
        embed_shard,
        prev,
        enorm,
        hnorm,
        eh_local.view(n_local, 2, CHUNKS, -1),
        input_norm,
        sc["e_in"],
        sc["h_in"],
        sc["e_parts"],
        sc["h_parts"],
        sym[0],
        sym[1],
        out[0],
        out[1],
        group_name,
        rank,
    )


def compile_kernel(args: tuple):
    """``glmf_mtp_eh_tp4`` compiled for ``args`` (once per binding)."""
    return _setup.compiled(glmf_mtp_eh_tp4, mk_config, args)


# ---------------------------- check and timing (torch.distributed.run, 4 ranks)

# The MTP draft's decode step (one row) and its extend (the verify window).
TOKENS = (1, 6)


def _args(ids, prev, w, rank, world, sym, group_name, out) -> tuple:
    """``kernel_args`` of the replicated weights ``w`` (``glm5_3_tp4._setup.eh_weights``)
    at ``rank`` of ``world``."""
    n_local, shard = HIDDEN // world, w["embed"].shape[0] // world
    return kernel_args(
        ids,
        w["embed"][rank * shard : (rank + 1) * shard],
        prev,
        w["enorm"],
        w["hnorm"],
        w["eh"][rank * n_local : (rank + 1) * n_local],
        w["input_norm"],
        sym,
        out,
        group_name,
        rank,
    )


def _outputs(tokens: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    return tuple(
        torch.empty(tokens, HIDDEN, device=device, dtype=torch.bfloat16)
        for _ in range(2)
    )


def correctness_check(group: dist.ProcessGroup, verbose: bool = True) -> None:
    """Against ``glm5_3_tp4._setup.eh_reference`` (the stock chain): tokens on every
    rank's vocab shard (its edges included), six back-to-back launches with
    changing tokens and hidden states (no host synchronization), the residual and
    the normed output within 1e-2 relative RMS and identical on every rank; then a
    repeat bitwise."""
    rank, world = group.rank(), group.size()
    device = torch.device("cuda", torch.cuda.current_device())
    w = eh_weights(device, hidden=HIDDEN)
    shard = VOCAB // world
    ok = True
    for tokens in TOKENS:
        sym = symmetric_buffers(tokens, HIDDEN // world, group.group_name, device)
        cases = []
        for seed in range(6):
            ids, prev = eh_inputs(tokens, device, seed + 1, hidden=HIDDEN)
            ids[0] = (seed % world) * shard + (0 if seed < world else shard - 1)
            cases.append((ids, prev, _outputs(tokens, device)))
        torch.cuda.synchronize()
        dist.barrier()
        for i, (ids, prev, out) in enumerate(cases):
            args = _args(ids, prev, w, rank, world, sym[i % 2], group.group_name, out)
            compile_kernel(args)(*args)
        torch.cuda.synchronize()
        rows = []
        for i, (ids, prev, out) in enumerate(cases):
            normed, residual = eh_reference(ids, prev, w)
            ok &= report(rows, f"case {i} residual", out[0], residual, 1e-2)
            ok &= report(rows, f"case {i} normed", out[1], normed, 1e-2)
            peers = [torch.empty_like(out[0]) for _ in range(world)]
            dist.all_gather(peers, out[0], group=group)
            agree = all(torch.equal(p, out[0]) for p in peers)
            rows.append(f"  case {i} ranks {'agree' if agree else 'DIFFER  <-- FAIL'}")
            ok &= agree
        ids, prev, out = cases[-1]
        again = _outputs(tokens, device)
        args = _args(ids, prev, w, rank, world, sym[0], group.group_name, again)
        compile_kernel(args)(*args)
        torch.cuda.synchronize()
        repeat = same(again[0], out[0]) and same(again[1], out[1])
        rows.append(f"  {'repeat':<22} {'bitwise' if repeat else 'DIFFERS  <-- FAIL'}")
        ok &= repeat
        if verbose or not ok:
            say(f"rank{rank} T={tokens}\n" + "\n".join(rows))
    verdict(ok, group, "glmf_mtp_eh_tp4")


def benchmark(group: dist.ProcessGroup, verbose: bool = True) -> dict[str, Any]:
    """Cold-weight timing per window: ``LAYERS`` distinct weight copies (default 4)
    back to back, ``REPS`` times (default 4) in one CUDA graph."""
    rank, world = group.rank(), group.size()
    device = torch.device("cuda", torch.cuda.current_device())
    layers = int(os.environ.get("LAYERS", "4"))
    reps = int(os.environ.get("REPS", "4"))
    weights = [eh_weights(device, seed=10 + i, hidden=HIDDEN) for i in range(layers)]
    results = {}
    for tokens in TOKENS:
        ids, prev = eh_inputs(tokens, device, hidden=HIDDEN)
        sym = symmetric_buffers(tokens, HIDDEN // world, group.group_name, device)
        out = _outputs(tokens, device)
        args = [
            _args(ids, prev, w, rank, world, sym[i % 2], group.group_name, out)
            for i, w in enumerate(weights)
        ]

        def step(args=args) -> None:
            for a in args:
                compile_kernel(a)(*a)

        median, minimum = graph_time(step, layers, reps)
        if verbose:
            say(f"rank{rank} T={tokens} us/launch med {median:.2f} min {minimum:.2f}")
        results[tokens] = {"latency_us": median, "min_latency_us": minimum}
    return results


def main(verbose: bool = True, check_only: bool = False) -> dict[str, Any]:
    """The four-rank check, then the cold-weight timing (``check_only``: the
    check alone)."""
    return run_main(
        "glmf_mtp_eh_tp4", correctness_check, benchmark, verbose, check_only
    )


if __name__ == "__main__":
    main(check_only="--check" in sys.argv)
