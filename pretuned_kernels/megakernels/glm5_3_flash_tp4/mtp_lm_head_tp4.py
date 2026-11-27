# ruff: noqa: ANN001, ANN201, ANN202
# pyrefly: ignore-errors
"""GLM-5.3 Flash greedy logits at TP4 as one Helion megakernel launch: the
vocab-parallel lm head GEMV (FP32 sums of BF16 operands, rounded to BF16 like
the stock GEMM's output), optionally the all-gather of the vocab shards, and the
greedy argmax across the ranks. Serves the MTP draft: its per-step top-1 (one
row, greedy only) and the draft extend's selected row (with its logits); the
target's verify logits stay SGLang's. Replaces the stock lm head GEMM, the logits
all-gather, the FP32 copy and the draft's top-1
(``draft_topk1_postprocess``) launches, and the draft step's position advance.
GLM-5.3's ``glm5_3_tp4/glm5_3_lm_head_tp4.py`` at hidden 4096, with whole-K G
tasks taking the block argmax in their epilogue (one root instead of G + M: 52.4
vs 55.0 us at one TP4 shard, T=1) and the config derived from the bound kernel.

Roots (rank r):

    G  per 64 vocab rows (whole K): the FP32 sum rounded to BF16, this
       rank's shard words (symm, only when the step needs the logits), the
       block's max and first argmax                                       [V / W / 64]
    A  every peer's shard block -> full FP32 logits, rank-major vocab,
       unless the step only needs the greedy tokens                       [V / W / 128]
    R  this rank's block maxima -> its per-token max and first argmax (symm) [T]
    F  every rank's max and argmax -> greedy token, the first max as argmax [T]

The greedy argmax crosses ranks once per token: G's block maxima stay local and
only R's per-token words travel, in band, to F (written by R in this launch:
no cross-launch symmetric reads). The maxima are taken in a fixed order (no
atomics): the outputs are run-to-run deterministic and the same on every rank.

Stock top-1 (sglang kernels/ops/speculative/topk1.py): FP32 view of the BF16
logits, NaN as -1e30, the first index of the max (Triton argmax per 8192-split,
then the first split holding the max) = the lowest vocab id among ties. The
kernel's token is that rule over its own BF16-rounded logits; the logits differ
from cuBLAS's only by the FP32 summation order.

Calls and graph replays must occur in identical collective order across ranks
on one ordered CUDA stream. ``main()`` checks against FP64 and torch.argmax
(random and tie-heavy integer inputs, back-to-back launches with rank-dependent
weights), then times against the stock path::

    NVSHMEM_DISABLE_CUDA_VMM=1 PYTHONPATH=. \\
      python -m torch.distributed.run --standalone --nproc-per-node=4 \\
      pretuned_kernels/megakernels/glm5_3_flash_tp4/mtp_lm_head_tp4.py [--check]
"""

from __future__ import annotations

import dataclasses
import functools
import sys
from typing import Any

from pretuned_kernels.megakernels._distributed import benchmark as paired_timing
from pretuned_kernels.megakernels._qwen3_5_tp4 import run_main
from pretuned_kernels.megakernels._qwen3_5_tp4 import say
from pretuned_kernels.megakernels._qwen3_5_tp4 import symmetric_buffer
from pretuned_kernels.megakernels._qwen3_5_tp4 import verdict
from pretuned_kernels.megakernels._ring_config import ring_config_fields
from pretuned_kernels.megakernels.glm5_3_flash_tp4 import _setup
from pretuned_kernels.megakernels.glm5_3_tp4._setup import VOCAB
from pretuned_kernels.megakernels.glm5_3_tp4._setup import check_greedy
from pretuned_kernels.megakernels.glm5_3_tp4._setup import lm_head_inputs
from pretuned_kernels.megakernels.glm5_3_tp4._setup import stock_lm_head
import torch
import torch.distributed as dist

import helion
import helion.language as hl

G_BLOCK = 64
# The G weight stream rides the TMA ring with its activation operand; every
# other memory op is a pointer access.
RING_LOADS = ("w", "x")


def glmf_mtp_lm_head(
    x: torch.Tensor,  # [T, K] bf16, final-norm output
    w: torch.Tensor,  # [V / W, K] bf16, this rank's lm head shard
    shard: torch.Tensor,  # [T, V / W] fp32 symm, whole G blocks
    logits: torch.Tensor,  # [T, W, V / W] fp32
    block_max: torch.Tensor,  # [T, V / W / 64] fp32
    block_arg: torch.Tensor,  # [T, V / W / 64] int32, global vocab ids
    rank_max: torch.Tensor,  # [T] fp32 symm, this rank's best logit
    rank_arg: torch.Tensor,  # [T] fp32 symm, its first argmax (global vocab id; exact: ids < 2**24)
    need_logits: torch.Tensor,  # [1] int32, 0 when the step only reads greedy_ids (equal on every rank)
    greedy_ids: torch.Tensor,  # [T] int64 (the draft's topk_index rows)
    positions: torch.Tensor,  # [T] int64, += 1 (a greedy draft step's position advance, else a workspace dummy)
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
    hl.specialize(positions.size())

    tokens = x.size(0)
    vocab = w.size(0)
    k_blk = hl.register_block_size(w.size(1))
    remote = torch.ops.symm_mem.get_remote_tensors(shard, group_name)
    remote_max = torch.ops.symm_mem.get_remote_tensors(rank_max, group_name)
    remote_arg = torch.ops.symm_mem.get_remote_tensors(rank_arg, group_name)

    hl.pdl_wait()
    # G. Stock: the lm head GEMM (BF16 logits, upcast exactly to FP32) and the
    # draft top-1's block maxima. Per 64 vocab rows: the FP32 sum rounded to
    # BF16, the shard words and the block's max and first argmax. Unmasked rows
    # (the shard is whole G blocks, ``make_workspaces``): a TMA ring operand.
    # (The token rows as a tile: a window that is not a power of two pads to the
    # block. A literal row block: Helion reduces (the block argmax) only over
    # literal blocks.)
    for g_v, g_t in hl.tile([vocab, tokens], block_size=[64, None]):
        acc = hl.zeros([g_v, g_t], dtype=torch.float32)
        for g_k in hl.tile(w.size(1), block_size=k_blk):
            acc = hl.dot(w[g_v, g_k], x[g_t, g_k].permute(1, 0), acc=acc)
        g_out = acc.permute(1, 0).to(torch.bfloat16).to(torch.float32)
        # Gated: a greedy-only step pushes no shard words (A reads none).
        hl.store(
            shard,
            [g_t, g_v],
            g_out,
            extra_mask=(need_logits[hl.arange(1)] != 0)[:, None],
        )
        # Stock draft top-1 (draft_topk1_postprocess): NaN as -1e30, the first
        # max on ties.
        g_id = rank * vocab + g_v.index
        g_live = torch.where(g_out == g_out, g_out, -1e30)
        g_best = torch.amax(g_live, dim=1)
        g_hit = g_live == g_best[:, None]
        block_max[g_t, g_v.id] = g_best
        block_arg[g_t, g_v.id] = torch.amin(
            torch.where(g_hit, g_id[None, :], 2**31 - 1), dim=1
        )

    # A. Stock: the vocab-shard all-gather and the FP32 logits copy. (Literal
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
        # (An FP32 word like the max: vocab ids are exact in FP32.)
        rank_arg[r_t] = torch.amin(
            torch.where(r_hit, block_arg[r_t, :], 2**31 - 1), dim=1
        ).to(torch.float32)

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
        positions[f_t] = positions[f_t] + 1
    # Trigger at exit: dependents parked in their PDL wait would hold SM slots
    # the G tasks still need.
    hl.pdl_launch_dependents()
    return logits


glmf_mtp_lm_head_tp4 = helion.kernel(
    glmf_mtp_lm_head,
    static_shapes=True,
    backend="gluon",
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)


def mk_config(bound) -> helion.Config:
    """G's weight stream and its activation operand on the warp-specialized TMA ring
    (``ring_config_fields`` of the bound kernel), every other access by pointer;
    dynamic dispatch. The block sizes and stages are GLM-5.3's sweep at K 6144 (K
    blocks of 256 at 4 stages best), not swept again at Flash's K 4096."""
    fields = ring_config_fields(bound, RING_LOADS)
    default = bound.config_spec.default_config().config
    # [G's K block, then the token tiles]: the window padded to 16 rows (BF16
    # MMA N >= 16).
    return helion.Config(
        block_sizes=[256] + [16] * (len(default["block_sizes"]) - 1),
        cross_loop_pipeline="dynamic",
        indexing=fields["indexing"],
        range_warp_specializes=fields["range_warp_specializes"],
        num_stages=4,
        num_warps=4,
    )


@dataclasses.dataclass(frozen=True)
class LmHeadWorkspaces:
    """One window's buffers: the vocab shard and the greedy argmax's block and
    rank maxima (symmetric where peers read them)."""

    shard: torch.Tensor  # [T, V / W] fp32 symm, whole G blocks
    block_max: torch.Tensor  # [T, V / W / 64] fp32
    block_arg: torch.Tensor  # [T, V / W / 64] int32
    rank_max: torch.Tensor  # [T] fp32 symm
    rank_arg: torch.Tensor  # [T] fp32 symm (exact vocab ids)
    need_logits: torch.Tensor  # [1] int32: 0 skips the logits (greedy tokens only)
    greedy_ids: torch.Tensor  # [T] int64
    positions: (
        torch.Tensor
    )  # [T] int64: the advance's target when the caller passes none


# Symmetric buffers live for the process: torch finds a symmetric-memory handle
# by address, so a larger buffer at a freed smaller one's address gets its stale
# handle.
@functools.cache
def make_workspaces(
    tokens: int,
    group_name: str,
    world: int,
    device: torch.device | None = None,
) -> LmHeadWorkspaces:
    """The group's buffers (its vocab shard: the vocabulary over its ranks) on
    ``device`` (default: the current CUDA device)."""
    # The G root reads whole 64-row blocks (unmasked TMA): 605 of them a TP4
    # rank.
    assert VOCAB % (world * G_BLOCK) == 0, world
    rows = VOCAB // world
    device = device or torch.device("cuda", torch.cuda.current_device())

    def symmetric(*shape: int):
        return symmetric_buffer(shape, torch.float32, group_name, device)

    return LmHeadWorkspaces(
        shard=symmetric(tokens, rows),
        block_max=torch.empty(
            tokens, rows // G_BLOCK, dtype=torch.float32, device=device
        ),
        block_arg=torch.empty(
            tokens, rows // G_BLOCK, dtype=torch.int32, device=device
        ),
        rank_max=symmetric(tokens),
        rank_arg=symmetric(tokens),
        need_logits=torch.ones(1, dtype=torch.int32, device=device),
        greedy_ids=torch.empty(tokens, dtype=torch.int64, device=device),
        positions=torch.zeros(tokens, dtype=torch.int64, device=device),
    )


def kernel_args(
    hidden,
    weight,
    workspaces,
    out,
    group_name,
    rank,
    world,
    positions=None,
    ids_out=None,
) -> tuple:
    """``glmf_mtp_lm_head_tp4``'s arguments. ``positions``: [T] int64 the kernel
    advances by one (a greedy draft step's positions.add_(1) folded in), default a
    workspace dummy; ``ids_out``: [T] int64 the greedy tokens go to (e.g. a column of
    the draft round's tokens), default ``workspaces.greedy_ids``."""
    tokens = hidden.size(0)
    assert hidden.is_contiguous() and weight.is_contiguous()
    positions = workspaces.positions if positions is None else positions
    ids_out = workspaces.greedy_ids if ids_out is None else ids_out
    for t in (positions, ids_out):
        assert t.dtype == torch.int64 and tuple(t.shape) == (tokens,), (
            t.dtype,
            t.shape,
        )
    return (
        hidden,
        weight,
        workspaces.shard,
        out.view(tokens, world, -1),
        workspaces.block_max,
        workspaces.block_arg,
        workspaces.rank_max,
        workspaces.rank_arg,
        workspaces.need_logits,
        ids_out,
        positions,
        group_name,
        rank,
        world,
    )


def compile_kernel(args: tuple):
    """``glmf_mtp_lm_head_tp4`` compiled for ``args`` (once per binding)."""
    return _setup.compiled(glmf_mtp_lm_head_tp4, mk_config, args)


def launch(
    hidden,
    weight,
    workspaces: LmHeadWorkspaces,
    out,
    group_name: str,
    rank: int,
    world: int,
    positions=None,
    ids_out=None,
) -> torch.Tensor:
    """Full FP32 logits [T, vocab] into ``out`` (unless ``workspaces.need_logits``
    is 0) and the greedy tokens into ``workspaces.greedy_ids``, on every rank
    (``weight``: this rank's [vocab / W, H] shard); ``positions`` (optional, [T]
    int64) advanced by one; the tokens into ``ids_out`` ([T] int64) when given."""
    args = kernel_args(
        hidden, weight, workspaces, out, group_name, rank, world, positions, ids_out
    )
    compile_kernel(args)(*args)
    return out


# ---------------------------- check and timing (torch.distributed.run, 4 ranks)

# The draft's one row: its decode step's top-1 and its extend's selected row.
TOKENS = (1,)


def correctness_check(group: dist.ProcessGroup, verbose: bool = True) -> None:
    """Fresh inputs per call against FP64 (and on a tie-heavy grid:
    ``glm5_3_tp4._setup.check_greedy``); a greedy-only call (need_logits 0) writes the
    same tokens and no logits; the tokens into a column of a round's tokens; six
    back-to-back greedy launches with changing rank-dependent weights (no host
    synchronization between them) give each call's tokens and advance the positions
    six times."""
    rank, world = group.rank(), group.size()
    failures = []
    for tokens in TOKENS:
        ws = make_workspaces(tokens, group.group_name, world)
        for seed, ties in ((21, False), (22, False), (23, True), (24, True)):
            hidden, weight = lm_head_inputs(
                seed, rank, tokens, ties, hidden=_setup.HIDDEN
            )
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
            # the tokens into a column of a [T, 4] round (a strided view)
            round_tokens = torch.full((tokens, 4), -1, dtype=torch.int64, device="cuda")
            args = (hidden, weight, ws, skipped, group.group_name, rank, world)
            launch(*args, None, round_tokens[:, 2])
            if not torch.equal(round_tokens[:, 2], greedy) or bool(
                (round_tokens[:, [0, 1, 3]] != -1).any()
            ):
                failures.append("column output: tokens differ or other columns written")
        # Back to back: rank-dependent weights per call, no sync; each call's
        # tokens kept.
        cases = [
            lm_head_inputs(40 + i, rank, tokens, bool(i % 2), hidden=_setup.HIDDEN)
            for i in range(6)
        ]
        out = torch.empty(tokens, VOCAB, device="cuda")
        got = []
        positions = (
            torch.arange(tokens, dtype=torch.int64, device="cuda") * 7 + 100 * rank
        )
        start = positions.clone()
        torch.cuda.synchronize()
        for hidden, weight in cases:
            launch(hidden, weight, ws, out, group.group_name, rank, world, positions)
            got.append(ws.greedy_ids.clone())
        torch.cuda.synchronize()
        if not torch.equal(positions, start + len(cases)):
            failures.append(f"positions {positions.tolist()} advanced wrong")
        for (hidden, weight), ids in zip(cases, got, strict=True):
            failures += check_greedy(None, hidden, weight, ids, group, False)
            full = [
                torch.empty(tokens, weight.size(0), dtype=torch.float64, device="cuda")
                for _ in range(world)
            ]
            dist.all_gather(
                full, (hidden.double() @ weight.double().T).contiguous(), group=group
            )
            exact = torch.cat(full, dim=1)
            # the token's exact logit is within BF16 rounding + summation order
            # of the max
            pick = exact.gather(1, ids.view(-1, 1)).squeeze(1)
            best = exact.max(dim=1).values
            if bool(((best - pick) > best.abs() * 2.0**-7 + 1e-2).any()):
                failures.append("back-to-back token is not a near-max logit")
        ws.need_logits.fill_(1)
    if failures or (verbose and rank == 0):
        say(f"rank{rank} glmf_mtp_lm_head_tp4: " + ("; ".join(failures) or "passed"))
    verdict(not failures, group, "glmf_mtp_lm_head_tp4")


def benchmark(group: dist.ProcessGroup, verbose: bool = True) -> dict[str, Any]:
    """Paired max-rank warm and cold-L2 medians against the stock path
    (``glm5_3_tp4._setup.stock_lm_head``), the logits off (a decode step) and on
    (the extend's selected row) (``_distributed.benchmark``)."""
    rank, world = group.rank(), group.size()
    results = {}
    for tokens in TOKENS:
        hidden, weight = lm_head_inputs(31, rank, tokens, hidden=_setup.HIDDEN)
        ws = make_workspaces(tokens, group.group_name, world)
        out = torch.empty(tokens, VOCAB, device="cuda")
        for need in (0, 1):

            def helion_call(hidden=hidden, weight=weight, ws=ws, out=out):
                return launch(hidden, weight, ws, out, group.group_name, rank, world)

            def validate(name, logits, hidden=hidden, weight=weight, ws=ws, need=need):
                if name == "helion":
                    logits = logits if need else None
                    failures = check_greedy(
                        logits, hidden, weight, ws.greedy_ids, group, False
                    )
                    verdict(not failures, group, "glmf_mtp_lm_head_tp4")

            ws.need_logits.fill_(need)
            timing = paired_timing(
                {"helion": helion_call, "stock": stock_lm_head(hidden, weight, group)},
                validate=validate,
            )
            results[(tokens, need)] = timing
            if verbose and rank == 0:
                h, s = timing["helion"], timing["stock"]
                print(
                    f"T={tokens} logits={'on ' if need else 'off'}: helion {h[0]:7.2f} warm "
                    f"{h[1]:7.2f} cold | stock {s[0]:7.2f} warm {s[1]:7.2f} cold | cold "
                    f"speedup {s[1] / h[1]:.3f}x",
                    flush=True,
                )
        ws.need_logits.fill_(1)
    return results


@torch.inference_mode()
def main(verbose: bool = True, check_only: bool = False) -> dict[str, Any]:
    """The four-rank check, then the timing (``check_only``: the check alone)."""
    return run_main(
        "glmf_mtp_lm_head_tp4", correctness_check, benchmark, verbose, check_only
    )


if __name__ == "__main__":
    main(check_only="--check" in sys.argv)
