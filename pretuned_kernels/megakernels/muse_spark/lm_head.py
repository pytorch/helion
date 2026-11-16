"""Muse Spark spec-decode verify logits for TP4 as one persistent Helion megakernel.

One request of T = 8 or 16 draft tokens: the vocab-parallel LM head GEMV, the
logit scale and softcap, and the all-gather of the vocab shards. It replaces
sglang's FP32-output LM head GEMM, logit scale, NCCL all-gather, logits copy and
softcap launches.

The shard is padded to whole G blocks, so every block lands unmasked and its
words travel tagged to every rank as they are produced; A polls them locally and
drops the padding.

Roots (rank r):

    G  LM head block -> capped FP32 logits of this rank's vocab shard (symm)  [V'/BV]
    A  every peer's shard block -> full logits, rank-major vocab              [V'/BA]

The softcap is elementwise, so capping each shard before the gather matches the
stock cap of the gathered logits.

Calls and graph replays must occur in identical collective order across ranks
on one ordered CUDA stream. ``main()`` compares against the sglang default
(cuBLAS FP32 GEMM, NCCL all-gather, softcap); ``--check`` runs only the FP64
correctness check::

    NVSHMEM_DISABLE_CUDA_VMM=1 PYTHONPATH=. \
      python -m torch.distributed.run --standalone --nproc-per-node=4 \
      pretuned_kernels/megakernels/muse_spark/lm_head.py [--check]
"""

from __future__ import annotations

import functools
import math
import sys
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

from pretuned_kernels.megakernels._distributed import benchmark
from pretuned_kernels.megakernels._distributed import initialize
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

HIDDEN = 8192
# The served 202048-token vocabulary over four ranks: not a whole number of G
# blocks, so the padded tail is exercised.
VOCAB_SHARD = 50512
# Compile-time logit scale and softcap; these place the logits at the cap's knee.
LOGIT_SCALE = 0.75
LOGIT_CAP = 20.0
G_BLOCK = 64
A_BLOCK = 128


def _muse_spark_lm_head_megakernel(
    x: torch.Tensor,  # [T, K] bf16, final-norm output
    w: torch.Tensor,  # [V / W, K] bf16, this rank's lm head shard
    shard: torch.Tensor,  # [T, V'] fp32 symm, V / W padded to G blocks
    logits: torch.Tensor,  # [T, W, V / W] fp32
    group_name: hl.ProcessGroupName,
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

    tokens = x.size(0)
    vocab = w.size(0)
    k_blk = hl.register_block_size(w.size(1))
    remote = torch.ops.symm_mem.get_remote_tensors(shard, group_name)

    hl.pdl_wait()
    # G. Stock: the FP32 lm head GEMM (_compute_lm_head), logit scale and softcap.
    for g_v in hl.tile(shard.size(1), block_size=G_BLOCK):
        acc = hl.zeros([g_v, tokens], dtype=torch.float32)
        rows = (g_v.index < vocab)[:, None]
        # Weights stream once: evict_first keeps the megakernel code in L2.
        for g_k in hl.tile(w.size(1), block_size=k_blk):
            acc = hl.dot(
                hl.load(w, [g_v, g_k], extra_mask=rows, eviction_policy="evict_first"),
                x[:, g_k].permute(1, 0),
                acc=acc,
            )
        shard[:, g_v] = torch.tanh(acc.permute(1, 0) * scale / cap) * cap

    # A. Stock: the vocab-shard all-gather (NCCL) and the logits copy.
    for a_t, a_v in hl.tile([tokens, shard.size(1)], block_size=[tokens, A_BLOCK]):
        keep = (a_v.index < vocab)[None, :]
        for peer in hl.static_range(world):
            hl.store(logits, [a_t, peer, a_v], remote[peer][a_t, a_v], extra_mask=keep)
    # Trigger at exit, unlike the other verify megakernels: dependents parked in
    # their PDL wait would hold SM slots the persistent G tasks still need.
    hl.pdl_launch_dependents()
    return logits


muse_spark_lm_head = helion.kernel(
    _muse_spark_lm_head_megakernel,
    config=helion.Config(
        block_sizes=[128],
        cross_loop_pipeline="dynamic",
        indexing="pointer",
        num_stages=5,
        num_warps=4,
        pid_type="persistent_blocked",
    ),
    static_shapes=True,
    backend="triton",
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)


# Symmetric buffers live for the process: torch finds a symmetric-memory handle by
# address, so a larger buffer at a freed smaller one's address gets its stale handle.
@functools.cache
def make_shard(tokens: int, group: dist.ProcessGroup) -> torch.Tensor:
    """This rank's symmetric FP32 shard buffer, padded to whole G blocks."""
    padded = math.ceil(VOCAB_SHARD / G_BLOCK) * G_BLOCK
    device = torch.device("cuda", torch.cuda.current_device())
    shard = symm_mem.empty(tokens, padded, dtype=torch.float32, device=device)
    symm_mem.rendezvous(shard, group.group_name)
    return shard


def launch(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    shard: torch.Tensor,
    out: torch.Tensor,
    group_name: str,
) -> torch.Tensor:
    """Full capped FP32 logits [T, W * VOCAB_SHARD] into ``out``, on every rank."""
    tokens = hidden.size(0)
    muse_spark_lm_head(
        hidden,
        weight,
        shard,
        out.view(tokens, WORLD_SIZE, -1),
        group_name,
        WORLD_SIZE,
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


def _check(logits: torch.Tensor, hidden: torch.Tensor, weight: torch.Tensor) -> None:
    """This rank's columns match FP64, and every rank holds identical logits."""
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


def correctness_check(group: dist.ProcessGroup) -> None:
    """Fresh inputs per call against FP64, so a stale shard block shows up."""
    for tokens in VERIFY_TOKENS:
        shard = make_shard(tokens, group)
        for seed in range(21, 24):
            hidden, weight = make_inputs(seed, group.rank(), tokens)
            out = torch.full(
                (tokens, WORLD_SIZE * VOCAB_SHARD), float("nan"), device="cuda"
            )
            _check(launch(hidden, weight, shard, out, group.group_name), hidden, weight)
        dist.barrier()


def use_cudagraph() -> bool:
    return True


def _sglang_default(
    hidden: torch.Tensor, weight: torch.Tensor, group: dist.ProcessGroup
) -> Callable[[], torch.Tensor]:
    """sglang's verify logits: FP32-output GEMM and scale, NCCL all-gather of
    the vocab shards, rank-major copy and softcap. None of these launch with PDL,
    in serving either (cuBLAS, NCCL and PyTorch elementwise kernels)."""
    tokens = hidden.size(0)
    gathered = torch.empty(WORLD_SIZE, tokens, VOCAB_SHARD, device="cuda")

    def call() -> torch.Tensor:
        logits = torch.mm(hidden, weight.T, out_dtype=torch.float32)
        logits.mul_(LOGIT_SCALE)
        dist.all_gather_into_tensor(gathered, logits, group=group)
        full = gathered.permute(1, 0, 2).reshape(tokens, WORLD_SIZE * VOCAB_SHARD)
        return LOGIT_CAP * torch.tanh(full / LOGIT_CAP)

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
        shard = make_shard(tokens, group)
        out = torch.empty(tokens, WORLD_SIZE * VOCAB_SHARD, device="cuda")

        def helion_call(
            hidden: torch.Tensor = hidden,
            weight: torch.Tensor = weight,
            shard: torch.Tensor = shard,
            out: torch.Tensor = out,
        ) -> torch.Tensor:
            return launch(hidden, weight, shard, out, group.group_name)

        def validate(
            _name: str,
            logits: object,
            hidden: torch.Tensor = hidden,
            weight: torch.Tensor = weight,
        ) -> None:
            _check(cast("torch.Tensor", logits), hidden, weight)

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
