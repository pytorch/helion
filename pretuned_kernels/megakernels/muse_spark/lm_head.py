"""Muse Spark spec-decode verify logits (8 or 16 rows) at TP4 or TP8 as one
Helion megakernel launch (dynamic dispatch, one CTA per G task): the
vocab-parallel lm head GEMV, the logit scale and softcap, the all-gather of the
vocab shards and the greedy argmax. Replaces the stock FP32-output GEMM, logit
scale, NCCL all-gather, logits copy, softcap and argmax launches.

The shard is padded to whole G blocks, so every block lands whole and its words
travel tagged to every rank as they are produced, when the step needs the
logits (the store is gated on need_logits, so a greedy-only step pushes none);
A polls them locally and drops the padding.

Roots (rank r):

    G  lm head block -> capped FP32 logits of this rank's vocab shard (symm),
       and its per-token max and first argmax (local)                         [V'/BV]
    A  every peer's shard block -> full logits, rank-major vocab, unless the
       step only needs the greedy tokens                                      [V'/BA]
    R  this rank's block maxima -> its per-token max and first argmax (symm)  [T]
    F  every rank's max and argmax -> greedy token, the first max as argmax   [T]

The softcap is elementwise, so capping each shard before the gather matches the
stock cap of the gathered logits.

The greedy argmax crosses ranks once per token: G's block maxima stay local (R
waits for them behind a local root barrier) and only R's per-token words travel,
inband, to F. Peer counters on the block maxima would make every G task publish
with system-scope release atomics, which serialize GPU-wide.

Calls and graph replays must occur in identical collective order across ranks
on one ordered CUDA stream. ``main()`` compares against the sglang default
(cuBLAS FP32 GEMM, NCCL all-gather, softcap, argmax) at TP4; ``--check`` runs
only the FP64 correctness check::

    NVSHMEM_DISABLE_CUDA_VMM=1 PYTHONPATH=. \
      python -m torch.distributed.run --standalone --nproc-per-node=4 \
      pretuned_kernels/megakernels/muse_spark/lm_head.py [--check]
"""

from __future__ import annotations

import dataclasses
import functools
import math
import sys
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

from pretuned_kernels.megakernels._distributed import benchmark
from pretuned_kernels.megakernels._distributed import initialize
from pretuned_kernels.megakernels.muse_spark._common import HIDDEN
from pretuned_kernels.megakernels.muse_spark._common import SIGNAL_PAD_BYTES
from pretuned_kernels.megakernels.muse_spark._common import VERIFY_TOKENS
from pretuned_kernels.megakernels.muse_spark._common import WORLD_SIZE
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

import helion
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Callable

# The served 202048-token vocabulary; its shard (TP4: 50512 rows, TP8: 25256) is
# not a whole number of G blocks, so the padded tail is exercised.
VOCAB = 202048
VOCAB_SHARD = VOCAB // WORLD_SIZE
# Compile-time logit scale and softcap; these place the logits at the cap's knee.
LOGIT_SCALE = 0.75
LOGIT_CAP = 20.0
G_BLOCK = 64


def _muse_lm_head_verify_megakernel(
    x: torch.Tensor,  # [T, K] bf16, final-norm output
    w: torch.Tensor,  # [V / W, K] bf16, this rank's lm head shard
    shard: torch.Tensor,  # [T, V'] fp32 symm, V / W padded to G blocks
    logits: torch.Tensor,  # [T, W, V / W] fp32
    block_max: torch.Tensor,  # [T, V' / BV] fp32
    block_arg: torch.Tensor,  # [T, V' / BV] int32, global vocab ids
    rank_max: torch.Tensor,  # [T] fp32 symm, this rank's best capped logit
    rank_arg: torch.Tensor,  # [T] int32 symm, its first argmax (global vocab id)
    need_logits: torch.Tensor,  # [1] int32, 0 when the step only reads greedy_ids (equal on every rank)
    greedy_ids: torch.Tensor,  # [T] int32
    group_name: hl.ProcessGroupName,
    rank: hl.constexpr,
    world: hl.constexpr,
    scale: hl.constexpr,
    cap: hl.constexpr,
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

    hl.pdl_wait()
    # G. Stock: the FP32 lm head GEMM (_compute_lm_head), logit scale and softcap.
    # A literal block size: Helion reduces (the block argmax) only over literal blocks.
    for g_v in hl.tile(shard.size(1), block_size=64):
        acc = hl.zeros([g_v, tokens], dtype=torch.float32)
        rows = (g_v.index < vocab)[:, None]
        for g_k in hl.tile(w.size(1), block_size=k_blk):
            acc = hl.dot(
                hl.load(w, [g_v, g_k], extra_mask=rows),
                x[:, g_k].permute(1, 0),
                acc=acc,
            )
        capped = torch.tanh(acc.permute(1, 0) * scale / cap) * cap
        # Gated: a greedy-only step pushes no shard words (A reads none).
        hl.store(
            shard,
            [slice(None), g_v],
            capped,
            extra_mask=(need_logits[hl.arange(1)] != 0)[:, None],
        )
        # Stock greedy: torch.argmax over the full logits, the first max on ties.
        vocab_id = rank * vocab + g_v.index
        live = torch.where((g_v.index < vocab)[None, :], capped, float("-inf"))
        best = torch.amax(live, dim=1)
        # torch.argmax takes NaN as the max: a NaN row's first NaN, like a tie's first max.
        hit = (live == best[:, None]) | (torch.isnan(live) & torch.isnan(best)[:, None])
        block_max[:, g_v.id] = best
        block_arg[:, g_v.id] = torch.amin(
            torch.where(hit, vocab_id[None, :], 2**31 - 1), dim=1
        )

    # A. Stock: the vocab-shard all-gather (NCCL) and the logits copy. (Literal block sizes: a module-global one
    # would be symbolic.)
    for a_t, a_v in hl.tile([tokens, shard.size(1)], block_size=[tokens, 128]):
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
        r_hit = (r_max == r_best[:, None]) | (
            torch.isnan(r_max) & torch.isnan(r_best)[:, None]
        )
        rank_max[r_t] = r_best
        rank_arg[r_t] = torch.amin(
            torch.where(r_hit, block_arg[r_t, :], 2**31 - 1), dim=1
        )

    # F. Stock greedy verify: the argmax over every rank's best, in rank (= vocab id) order.
    for f_t in hl.tile(tokens, block_size=1):
        peer_bests = [m[f_t] for m in remote_max]
        peer_ids = [a[f_t] for a in remote_arg]
        top = peer_bests[0]
        top_id = peer_ids[0]
        for peer in hl.static_range(1, world):
            peer_best = peer_bests[peer]
            peer_id = peer_ids[peer]
            peer_nan, top_nan = torch.isnan(peer_best), torch.isnan(top)
            # Peers in rank order hold increasing ids: a tie keeps the earlier rank.
            take = (peer_best > top) | (peer_nan & ~top_nan)
            top = torch.where(take, peer_best, top)
            top_id = torch.where(take, peer_id, top_id)
        greedy_ids[f_t] = top_id
    # Trigger at exit: dependents parked in their PDL wait would hold SM slots
    # the G tasks still need.
    hl.pdl_launch_dependents()
    return logits


# Dynamic dispatch, one CTA per G task, as many resident per SM as fit (3 at 5 stages), its K loop a
# cp.async pipe of num_stages - 1 slots. Pointer indexing keeps that pipe for whole-block weight loads too (a
# tensor-descriptor load would issue unpipelined), and no TMA load means no async-proxy fences around the
# tasks. GB300: 5 stages (3 CTAs resident per SM) beat 3, 4 and 6 at both shards, logits on or off; the gluon
# ring (TMA weight ring on persistent CTAs, 3-10 stages) is 1.2-1.6x slower.
CONFIG = helion.Config(
    block_sizes=[128],
    cross_loop_pipeline="dynamic",
    indexing="pointer",
    num_stages=5,
    num_warps=4,
)


muse_spark_lm_head = helion.kernel(
    _muse_lm_head_verify_megakernel,
    config=CONFIG,
    static_shapes=True,
    backend="gluon",
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)


@dataclasses.dataclass(frozen=True)
class LmHeadWorkspaces:
    """One verify window's buffers: the padded shard and the greedy argmax's
    block and rank maxima (symmetric where peers read them)."""

    shard: torch.Tensor  # [T, V'] fp32 symm, V / W padded to whole G blocks
    block_max: torch.Tensor  # [T, V' / G] fp32
    block_arg: torch.Tensor  # [T, V' / G] int32
    rank_max: torch.Tensor  # [T] fp32 symm
    rank_arg: torch.Tensor  # [T] int32 symm
    need_logits: torch.Tensor  # [1] int32: 0 skips the logits (greedy tokens only)
    greedy_ids: torch.Tensor  # [T] int32


# Symmetric buffers live for the process: torch finds a symmetric-memory handle by
# address, so a larger buffer at a freed smaller one's address gets its stale handle.
@functools.cache
def make_workspaces(
    tokens: int, group: dist.ProcessGroup, device: torch.device | None = None
) -> LmHeadWorkspaces:
    """The group's buffers (its vocab shard: the vocabulary over its ranks) on
    ``device`` (default: the current CUDA device)."""
    padded = math.ceil(VOCAB // group.size() / G_BLOCK) * G_BLOCK
    blocks = padded // G_BLOCK
    device = device or torch.device("cuda", torch.cuda.current_device())
    workspaces = LmHeadWorkspaces(
        shard=symm_mem.empty(tokens, padded, dtype=torch.float32, device=device),
        block_max=torch.empty(tokens, blocks, dtype=torch.float32, device=device),
        block_arg=torch.empty(tokens, blocks, dtype=torch.int32, device=device),
        rank_max=symm_mem.empty(tokens, dtype=torch.float32, device=device),
        rank_arg=symm_mem.empty(tokens, dtype=torch.int32, device=device),
        need_logits=torch.ones(1, dtype=torch.int32, device=device),
        greedy_ids=torch.empty(tokens, dtype=torch.int32, device=device),
    )
    for buffer in (workspaces.shard, workspaces.rank_max, workspaces.rank_arg):
        symm_mem.rendezvous(buffer, group.group_name)
    return workspaces


def launch(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    workspaces: LmHeadWorkspaces,
    out: torch.Tensor,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    """Full capped FP32 logits [T, vocab] into ``out`` (unless
    ``workspaces.need_logits`` is 0) and the greedy tokens into
    ``workspaces.greedy_ids``, on every rank of ``group`` (``weight``: this
    rank's [vocab / W, H] shard)."""
    tokens, world = hidden.size(0), group.size()
    muse_spark_lm_head(
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
        group.group_name,
        group.rank(),
        world,
        LOGIT_SCALE,
        LOGIT_CAP,
    )
    return out


def make_inputs(seed: int, rank: int, tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Replicated final-norm rows and this rank's LM head shard."""
    x = torch.Generator(device="cuda").manual_seed(seed)
    g = torch.Generator(device="cuda").manual_seed(seed * 16 + rank)
    hidden = torch.randn(tokens, HIDDEN, device="cuda", generator=x).bfloat16()
    weight = 0.15 * torch.randn(VOCAB_SHARD, HIDDEN, device="cuda", generator=g)
    return hidden, weight.bfloat16()


def reference_shard(hidden: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """FP64 capped logits of one rank's vocab shard."""
    logits = (hidden.double() @ weight.double().T) * LOGIT_SCALE
    return LOGIT_CAP * torch.tanh(logits / LOGIT_CAP)


def _check(
    logits: torch.Tensor,
    hidden: torch.Tensor,
    weight: torch.Tensor,
    greedy_ids: torch.Tensor | None = None,
) -> None:
    """This rank's columns match FP64, every rank holds identical logits, and
    the greedy tokens, when given, are their first argmax (torch.argmax)."""
    rank, tokens = dist.get_rank(), hidden.size(0)
    mine = logits.view(tokens, WORLD_SIZE, VOCAB_SHARD)[:, rank].double()
    # BF16 inputs with FP32 accumulation over K = 8192: ~1e-4 absolute.
    error = float((mine - reference_shard(hidden, weight)).abs().max())
    if error > 1e-3:
        raise AssertionError(f"rank {rank}: logits vs FP64 max abs {error:.3e}")
    peers = [torch.empty_like(logits) for _ in range(WORLD_SIZE)]
    dist.all_gather(peers, logits)
    if not all(torch.equal(peer, logits) for peer in peers):
        raise AssertionError(f"rank {rank}: ranks disagree on the gathered logits")
    if greedy_ids is not None and not torch.equal(
        greedy_ids.long(), torch.argmax(logits, dim=-1)
    ):
        raise AssertionError(f"rank {rank}: greedy tokens are not the logits' argmax")


def correctness_check(group: dist.ProcessGroup) -> None:
    """Fresh inputs per call against FP64, so a stale shard block shows up; a
    greedy-only call (need_logits 0) writes the same tokens and no logits."""
    for tokens in VERIFY_TOKENS:
        workspaces = make_workspaces(tokens, group)
        for seed in range(21, 24):
            hidden, weight = make_inputs(seed, group.rank(), tokens)
            out = torch.full((tokens, VOCAB), float("nan"), device="cuda")
            workspaces.need_logits.fill_(1)
            launch(hidden, weight, workspaces, out, group)
            _check(out, hidden, weight, workspaces.greedy_ids)
            greedy = workspaces.greedy_ids.clone()
            skipped = torch.full_like(out, float("nan"))
            workspaces.need_logits.fill_(0)
            workspaces.greedy_ids.fill_(-1)  # so a launch that writes nothing fails
            launch(hidden, weight, workspaces, skipped, group)
            if not torch.equal(workspaces.greedy_ids, greedy):
                raise AssertionError("greedy-only call: tokens differ")
            if not bool(torch.isnan(skipped).all()):
                raise AssertionError("greedy-only call: logits written")
            workspaces.need_logits.fill_(1)
        dist.barrier()


def use_cudagraph() -> bool:
    return True


def _sglang_default(
    hidden: torch.Tensor, weight: torch.Tensor, group: dist.ProcessGroup
) -> Callable[[], torch.Tensor]:
    """sglang's verify logits and greedy tokens: FP32-output GEMM and scale, NCCL
    all-gather of the vocab shards, rank-major copy, softcap and argmax. None of
    these launch with PDL, in serving either (cuBLAS, NCCL and PyTorch kernels)."""
    tokens = hidden.size(0)
    gathered = torch.empty(WORLD_SIZE, tokens, VOCAB_SHARD, device="cuda")
    greedy_ids = torch.empty(tokens, dtype=torch.int64, device="cuda")

    def call() -> torch.Tensor:
        logits = torch.mm(hidden, weight.T, out_dtype=torch.float32)
        logits.mul_(LOGIT_SCALE)
        dist.all_gather_into_tensor(gathered, logits, group=group)
        full = gathered.permute(1, 0, 2).reshape(tokens, VOCAB)
        capped = LOGIT_CAP * torch.tanh(full / LOGIT_CAP)
        torch.argmax(capped, dim=-1, out=greedy_ids)
        return capped

    return call


def _run(
    group: dist.ProcessGroup, *, verbose: bool, check_only: bool
) -> dict[str, Any]:
    correctness_check(group)
    if check_only:
        return {}
    rank = group.rank()
    speedups = []
    for tokens in VERIFY_TOKENS:
        hidden, weight = make_inputs(31, rank, tokens)
        workspaces = make_workspaces(tokens, group)
        out = torch.empty(tokens, VOCAB, device="cuda")

        def helion_call(
            hidden: torch.Tensor = hidden,
            weight: torch.Tensor = weight,
            workspaces: LmHeadWorkspaces = workspaces,
            out: torch.Tensor = out,
        ) -> torch.Tensor:
            return launch(hidden, weight, workspaces, out, group)

        def validate(
            name: str,
            logits: object,
            hidden: torch.Tensor = hidden,
            weight: torch.Tensor = weight,
            greedy_ids: torch.Tensor = workspaces.greedy_ids,
        ) -> None:
            _check(
                cast("torch.Tensor", logits),
                hidden,
                weight,
                greedy_ids if name == "helion" else None,
            )

        results = benchmark(
            {
                "helion": helion_call,
                "sglang_default": _sglang_default(hidden, weight, group),
            },
            validate=validate,
        )
        speedup = results["sglang_default"][1] / results["helion"][1]
        speedups.append(speedup)
        if verbose and rank == 0:
            print(f"T={tokens}")
            for name, (warm, cold) in results.items():
                print(f"{name:>16s}: {warm:7.2f} us warm, {cold:7.2f} us cold L2")
            print(f"{'speedup':>16s}: {speedup:.3f}x (cold L2)")
    geomean = math.prod(speedups) ** (1 / len(speedups))
    return {
        "helion_wins": sum(s > 1 for s in speedups),
        "total": len(speedups),
        "geomean": geomean,
        "best_speedup": max(speedups),
    }


@torch.inference_mode()
def main(verbose: bool = True, check_only: bool = False) -> dict[str, Any]:
    """Run the TP4 FP64 check, then cold-L2 timings against the sglang default."""
    _rank, _local_rank, group = initialize(
        kernel_name="muse_spark_lm_head",
        world_size=WORLD_SIZE,
        signal_pad_bytes=SIGNAL_PAD_BYTES,
    )
    try:
        return _run(group, verbose=verbose, check_only=check_only)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main(check_only="--check" in sys.argv)
