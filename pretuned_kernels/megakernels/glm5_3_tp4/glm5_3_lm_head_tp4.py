# ruff: noqa: ANN001, ANN202
# pyrefly: ignore-errors
"""GLM-5.3 greedy logits at TP4 as one Helion megakernel launch: the
vocab-parallel lm head GEMV (BF16 operands, FP32 accumulation, BF16 logits as
stock's GEMM rounds them), optionally the all-gather of the vocab shards, and
the greedy argmax across the ranks. It serves the MTP draft's per-step top-1
(one row; SGLang's ``draft_topk1_postprocess``: NaN as -1e30, the first max on
ties), replacing the stock lm head GEMM, NCCL all-gather, logits copy and top-1
launches. ``muse_spark/lm_head.py``
without the logit scale and softcap, its G root split into K chunks on the TMA
ring.

Roots (rank r):

    G  lm head block x K chunk -> FP32 partials                               [V / W / BV x 3]
    M  512 vocab rows -> the chunks' sum rounded to BF16: this rank's logits
       (symm, only when the step needs them), the block's max and first
       argmax                                                                 [V / W / 512]
    A  every peer's shard block -> full logits, rank-major vocab, unless the
       step only needs the greedy tokens                                      [V / W / BA]
    R  this rank's block maxima -> its per-token max and first argmax (symm)  [T]
    F  every rank's max and argmax -> greedy token, the first max as argmax   [T]

The greedy argmax crosses ranks once per token: M's block maxima stay local
and only R's per-token words travel, in band, to F. The chunks are summed and
the maxima taken in a fixed order (no atomics): the outputs are run-to-run
deterministic and the same on every rank.

Calls and graph replays must occur in identical collective order across ranks
on one ordered CUDA stream. ``main()`` checks against FP64 and torch.argmax,
then times against the stock path (cuBLAS GEMM, NCCL all-gather, argmax)::

    NVSHMEM_DISABLE_CUDA_VMM=1 PYTHONPATH=. \\
      python -m torch.distributed.run --standalone --nproc-per-node=4 \\
      pretuned_kernels/megakernels/glm5_3_tp4/glm5_3_lm_head_tp4.py [--check]
"""

from __future__ import annotations

import dataclasses
import functools
import math
import sys
from typing import Any

from pretuned_kernels.megakernels._distributed import benchmark as paired_timing
from pretuned_kernels.megakernels._qwen3_5_tp4 import run_main
from pretuned_kernels.megakernels._qwen3_5_tp4 import say
from pretuned_kernels.megakernels._qwen3_5_tp4 import verdict
from pretuned_kernels.megakernels.glm5_3_tp4._setup import VOCAB
from pretuned_kernels.megakernels.glm5_3_tp4._setup import check_greedy
from pretuned_kernels.megakernels.glm5_3_tp4._setup import lm_head_inputs
from pretuned_kernels.megakernels.glm5_3_tp4._setup import stock_lm_head
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

import helion
import helion.language as hl

# The MTP draft's top-1 (one row).
TOKENS = (1,)
G_BLOCK = 64
M_BLOCK = 512

# The G weight stream rides the TMA ring with its activation operand
# (``_ring_config.ring_config_fields(bound, RING_LOADS)``); every other memory
# op is a pointer access. G is the first root, so its two loads are slots 0 and
# 1 (of 35 at 4 ranks).
RING_LOADS = ("w_chunked", "x_chunked")
_MEMORY_OPS = 35


def glm5_3_lm_head(
    x: torch.Tensor,  # [T, K] bf16, final-norm output
    w: torch.Tensor,  # [V / W, K] bf16, this rank's lm head shard
    shard: torch.Tensor,  # [T, V / W] fp32 symm, whole G blocks
    logits: torch.Tensor,  # [T, W, V / W] fp32
    block_max: torch.Tensor,  # [T, ceil(V / W / 512)] fp32
    block_arg: torch.Tensor,  # [T, ceil(V / W / 512)] int32, global vocab ids
    rank_max: torch.Tensor,  # [T] fp32 symm, this rank's best logit
    rank_arg: torch.Tensor,  # [T] int32 symm, its first argmax (global vocab id)
    need_logits: torch.Tensor,  # [1] int32, 0 when the step only reads greedy_ids (equal on every rank)
    greedy_ids: torch.Tensor,  # [T] int64 (the draft's topk_index rows)
    group_name: hl.ProcessGroupName,
    rank: hl.constexpr,
    world: hl.constexpr,
) -> torch.Tensor:
    hl.specialize(x.size())
    hl.specialize(x.stride())
    hl.specialize(w.size())
    hl.specialize(w.stride())
    hl.specialize(shard.size())
    hl.specialize(shard.stride())
    hl.specialize(logits.size())
    hl.specialize(logits.stride())
    hl.specialize(block_max.size())
    hl.specialize(block_max.stride())
    hl.specialize(block_arg.size())
    hl.specialize(block_arg.stride())
    hl.specialize(rank_max.size())
    hl.specialize(rank_arg.size())
    hl.specialize(need_logits.size())
    hl.specialize(greedy_ids.size())

    tokens = x.size(0)
    vocab = w.size(0)
    k_blk = hl.register_block_size(w.size(1))
    remote = torch.ops.symm_mem.get_remote_tensors(shard, group_name)
    remote_max = torch.ops.symm_mem.get_remote_tensors(rank_max, group_name)
    remote_arg = torch.ops.symm_mem.get_remote_tensors(rank_arg, group_name)

    # K chunks: 3 x 605 G tasks a TP4 rank (whole-K 64-row tasks: 1.3 waves of
    # CTAs, ran as 2; on the ring 2 / 4 / 6 chunks are within 1.3 us).
    chunks = 3
    chunk_k = w.size(1) // chunks
    w_chunked = w.view(vocab, chunks, chunk_k)
    x_chunked = x.view(tokens, chunks, chunk_k)
    parts = torch.empty(
        (chunks, tokens, shard.size(1)), dtype=torch.float32, device=x.device
    )

    hl.pdl_wait()
    # G. Stock: the lm head GEMM, FP32 partials of the K chunks. Unmasked rows
    # (the shard is whole G blocks, ``make_workspaces``): a TMA ring operand.
    # (The token rows as a tile: a window that is not a power of two pads to the
    # block.)
    for g_v, g_c, g_t in hl.tile([vocab, chunks, tokens], block_size=[64, 1, None]):
        acc = hl.zeros([g_v, g_t], dtype=torch.float32)
        for g_k in hl.tile(chunk_k, block_size=k_blk):
            acc = hl.dot(
                w_chunked[g_v, g_c.begin, g_k],
                x_chunked[g_t, g_c.begin, g_k].permute(1, 0),
                acc=acc,
            )
        parts[g_c.begin, g_t, g_v] = acc.permute(1, 0)

    # M. Per 512 vocab rows: the chunk-order sum rounded to BF16 (stock: the
    # GEMM's BF16 logits, upcast exactly), the shard words, and the block's max
    # and first argmax. A literal block size: Helion reduces (the block argmax)
    # only over literal blocks.
    for m_v, m_t in hl.tile([shard.size(1), tokens], block_size=[512, None]):
        m_sum = parts[0, m_t, m_v]
        for m_c in hl.static_range(1, chunks):
            m_sum = m_sum + parts[m_c, m_t, m_v]
        out = m_sum.to(torch.bfloat16).to(torch.float32)
        # Gated: a greedy-only step pushes no shard words (A reads none).
        hl.store(
            shard,
            [m_t, m_v],
            out,
            extra_mask=(need_logits[hl.arange(1)] != 0)[:, None],
        )
        # Stock draft top-1 (draft_topk1_postprocess): NaN as -1e30, the first
        # max on ties.
        vocab_id = rank * vocab + m_v.index
        live = torch.where(out == out, out, -1e30)
        live = torch.where((m_v.index < vocab)[None, :], live, float("-inf"))
        best = torch.amax(live, dim=1)
        hit = live == best[:, None]
        block_max[m_t, m_v.id] = best
        block_arg[m_t, m_v.id] = torch.amin(
            torch.where(hit, vocab_id[None, :], 2**31 - 1), dim=1
        )

    # A. Stock: the vocab-shard all-gather (NCCL) and the logits copy. (Literal
    # block sizes: a module-global one would be symbolic.)
    for a_t, a_v in hl.tile([tokens, shard.size(1)], block_size=[None, 128]):
        keep = (a_v.index < vocab)[None, :] & (need_logits[hl.arange(1)] != 0)[:, None]
        for peer in hl.static_range(world):
            hl.store(
                logits,
                [a_t, peer, a_v],
                hl.load(remote[peer], [a_t, a_v], extra_mask=keep),
                extra_mask=keep,
            )

    # R. This rank's argmax over its block maxima (local reads only).
    for r_t in hl.tile(tokens, block_size=1):
        r_max = block_max[r_t, :]
        r_best = torch.amax(r_max, dim=1)
        r_hit = r_max == r_best[:, None]
        rank_max[r_t] = r_best
        rank_arg[r_t] = torch.amin(
            torch.where(r_hit, block_arg[r_t, :], 2**31 - 1), dim=1
        )

    # F. Stock greedy: the argmax over every rank's best, in rank (= vocab id)
    # order.
    for f_t in hl.tile(tokens, block_size=1):
        peer_bests = [m[f_t] for m in remote_max]
        peer_ids = [a[f_t] for a in remote_arg]
        top = peer_bests[0]
        top_id = peer_ids[0]
        # (From peer 0: its own entry never replaces itself; a one-rank group
        # keeps a nonempty loop.)
        for peer in hl.static_range(world):
            peer_best = peer_bests[peer]
            peer_id = peer_ids[peer]
            # Peers in rank order hold increasing ids: a tie keeps the earlier
            # rank.
            take = peer_best > top
            top = torch.where(take, peer_best, top)
            top_id = torch.where(take, peer_id, top_id)
        greedy_ids[f_t] = top_id.to(torch.int64)
    # Trigger at exit: dependents parked in their PDL wait would hold SM slots
    # the G tasks still need.
    hl.pdl_launch_dependents()
    return logits


# Dynamic dispatch, the G K loop on the warp-specialized TMA ring. GB300, one
# TP4 rank's shard on one GPU (T=1, warm, logits off): K blocks of 256 at 4
# stages 77.8 us; 256 x 3 80.2, 256 x 5 78.6, 512 x 2 78.3, 128 x 6 79.7,
# 128 x 8 78.8, 256 x 4 at 8 warps 78.2, at 2 warps 97.9 (256 x 6, 512 x 3,
# 128 x 12 exceed shared memory). Pointer loads (a cp.async pipe a CTA per G
# task) at their best, 256 x 3: 83.1. (Stock: the cuBLAS GEMM + gather + argmax
# 93.8.)
CONFIG = helion.Config(
    block_sizes=[256, 16, 16, 16],
    cross_loop_pipeline="dynamic",
    indexing=[
        "tensor_descriptor" if op < 2 else "pointer" for op in range(_MEMORY_OPS)
    ],
    range_warp_specializes=[True, None, None, None, None, None],
    num_stages=4,
    num_warps=4,
)


glm5_3_lm_head_tp4 = helion.kernel(
    glm5_3_lm_head,
    config=CONFIG,
    static_shapes=True,
    backend="gluon",
    ignore_warnings=[helion.exc.TensorOperationInWrapper, helion.exc.FastMathUnused],
)


@dataclasses.dataclass(frozen=True)
class LmHeadWorkspaces:
    """One window's buffers: the vocab shard and the greedy argmax's block and
    rank maxima (symmetric where peers read them)."""

    shard: torch.Tensor  # [T, V / W] fp32 symm, whole G blocks
    block_max: torch.Tensor  # [T, ceil(V / W / M)] fp32
    block_arg: torch.Tensor  # [T, ceil(V / W / M)] int32
    rank_max: torch.Tensor  # [T] fp32 symm
    rank_arg: torch.Tensor  # [T] int32 symm
    need_logits: torch.Tensor  # [1] int32: 0 skips the logits (greedy tokens only)
    greedy_ids: torch.Tensor  # [T] int64


# Symmetric buffers live for the process: torch finds a symmetric-memory handle
# by address, so a larger buffer at a freed smaller one's address gets its stale
# handle.
@functools.cache
def make_workspaces(
    tokens: int, group_name: str, world: int, device: torch.device | None = None
) -> LmHeadWorkspaces:
    """The group's buffers (its vocab shard: the vocabulary over its ranks) on
    ``device`` (default: the current CUDA device)."""
    # The G root reads whole 64-row blocks (unmasked TMA): 605 of them a rank.
    assert VOCAB % (world * G_BLOCK) == 0, world
    rows = VOCAB // world
    blocks = math.ceil(rows / M_BLOCK)
    device = device or torch.device("cuda", torch.cuda.current_device())
    workspaces = LmHeadWorkspaces(
        shard=symm_mem.empty(tokens, rows, dtype=torch.float32, device=device),
        block_max=torch.empty(tokens, blocks, dtype=torch.float32, device=device),
        block_arg=torch.empty(tokens, blocks, dtype=torch.int32, device=device),
        rank_max=symm_mem.empty(tokens, dtype=torch.float32, device=device),
        rank_arg=symm_mem.empty(tokens, dtype=torch.int32, device=device),
        need_logits=torch.ones(1, dtype=torch.int32, device=device),
        greedy_ids=torch.empty(tokens, dtype=torch.int64, device=device),
    )
    for buffer in (workspaces.shard, workspaces.rank_max, workspaces.rank_arg):
        symm_mem.rendezvous(buffer, group_name)
    return workspaces


def launch(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    workspaces: LmHeadWorkspaces,
    out: torch.Tensor,
    group_name: str,
    rank: int,
    world: int,
) -> torch.Tensor:
    """Full FP32 logits [T, vocab] into ``out`` (unless ``workspaces.need_logits``
    is 0) and the greedy tokens into ``workspaces.greedy_ids``, on every rank
    (``weight``: this rank's [vocab / W, H] shard)."""
    tokens = hidden.size(0)
    glm5_3_lm_head_tp4(
        hidden,
        weight,
        workspaces.shard,
        out.view(tokens, world, -1),
        workspaces.block_max,
        workspaces.block_arg,
        workspaces.rank_max,
        workspaces.rank_arg,
        workspaces.need_logits,
        workspaces.greedy_ids,
        group_name,
        rank,
        world,
    )
    return out


def correctness_check(group: dist.ProcessGroup, verbose: bool = True) -> None:
    """Fresh inputs per call against FP64 (and on a tie-heavy grid); a
    greedy-only call (need_logits 0) writes the same tokens and no logits."""
    rank, world = group.rank(), group.size()
    failures = []
    for tokens in TOKENS:
        ws = make_workspaces(tokens, group.group_name, world)
        for seed, ties in ((21, False), (22, False), (23, True), (24, True)):
            hidden, weight = lm_head_inputs(seed, rank, tokens, ties)
            out = torch.full((tokens, VOCAB), float("nan"), device="cuda")
            ws.need_logits.fill_(1)
            launch(hidden, weight, ws, out, group.group_name, rank, world)
            failures += check_greedy(out, hidden, weight, ws.greedy_ids, group, ties)
            greedy = ws.greedy_ids.clone()
            skipped = torch.full_like(out, float("nan"))
            ws.need_logits.fill_(0)
            ws.greedy_ids.fill_(-1)  # so a launch that writes nothing fails
            launch(hidden, weight, ws, skipped, group.group_name, rank, world)
            if not torch.equal(ws.greedy_ids, greedy):
                failures.append("greedy-only call: tokens differ")
            if not bool(torch.isnan(skipped).all()):
                failures.append("greedy-only call: logits written")
            ws.need_logits.fill_(1)
    if failures or (verbose and rank == 0):
        say(f"rank{rank} glm5_3_lm_head_tp4: " + ("; ".join(failures) or "passed"))
    verdict(not failures, group, "glm5_3_lm_head_tp4")


def benchmark(group: dist.ProcessGroup, verbose: bool = True) -> dict[str, Any]:
    """Paired max-rank warm and cold-L2 medians against the stock path, per
    window, with and without the logits (``_distributed.benchmark``)."""
    rank, world = group.rank(), group.size()
    results = {}
    for tokens in TOKENS:
        hidden, weight = lm_head_inputs(31, rank, tokens)
        ws = make_workspaces(tokens, group.group_name, world)
        out = torch.empty(tokens, VOCAB, device="cuda")
        for need in (1, 0):

            def helion_call(hidden=hidden, weight=weight, ws=ws, out=out):
                return launch(hidden, weight, ws, out, group.group_name, rank, world)

            def validate(name, logits, hidden=hidden, weight=weight, ws=ws, need=need):
                if name == "helion":
                    logits = logits if need else None
                    failures = check_greedy(
                        logits, hidden, weight, ws.greedy_ids, group, False
                    )
                    verdict(not failures, group, "glm5_3_lm_head_tp4")

            ws.need_logits.fill_(need)
            timing = paired_timing(
                {"helion": helion_call, "stock": stock_lm_head(hidden, weight, group)},
                validate=validate,
            )
            results[(tokens, need)] = timing
            if verbose and rank == 0:
                h, s = timing["helion"], timing["stock"]
                print(
                    f"T={tokens} logits={'on ' if need else 'off'}: helion "
                    f"{h[0]:7.2f} warm {h[1]:7.2f} cold | stock {s[0]:7.2f} warm "
                    f"{s[1]:7.2f} cold | cold speedup {s[1] / h[1]:.3f}x",
                    flush=True,
                )
        ws.need_logits.fill_(1)
    return results


@torch.inference_mode()
def main(verbose: bool = True, check_only: bool = False) -> dict[str, Any]:
    """The four-rank check, then the timing (``check_only``: the check alone)."""
    return run_main(
        "glm5_3_lm_head_tp4", correctness_check, benchmark, verbose, check_only
    )


if __name__ == "__main__":
    main(check_only="--check" in sys.argv)
