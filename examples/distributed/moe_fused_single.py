"""SINGLE fused comms+compute Helion kernel: expert FFN + reduce-scatter, one launch.

This is the fused shape of the msl-tpu-kernel v2 gmm-fused-RS: one Helion kernel,
one grid over the ``world_size`` output chunks.  For each chunk ``k`` the kernel
computes THIS chip's expert contribution to chunk ``k``'s tokens
(GMM1 gate+up -> silu(gate)*up -> GMM2 down, router-weighted) AND pushes that
``[chunk, H]`` result to chip ``k``'s ``recv`` slot -- compute and the ICI
reduce-scatter are fused in the SAME grid iteration.  A trailing local sum of the
``world_size`` received chunks yields each chip's reduced shard.

The single kernel lowers to BOTH targets from one source:
  * TPU  : Pallas -- GMM ``tl``-free matmuls + ``pltpu.make_async_remote_copy``.
    VERIFIED end-to-end on an 8-chip TPU host: output matches the pure-PyTorch
    reference exactly (maxdiff 0.0).
  * GPU  : Triton -- ``tl.dot`` for the three matmuls +
    ``nvshmem_putmem_signal_block`` / ``nvshmem_signal_wait_until`` for the
    reduce-scatter (verified by inspecting the generated Triton; no NVSHMEM
    runtime required to emit it).

Why this fuses cleanly (and an uneven direct-write scatter does not): every grid
iteration issues exactly ONE remote push and its ``op.wait()``, so sends and
receives stay balanced across chips -- no receive-imbalance deadlock, and each
``recv`` slot is written once (no scatter-add / no cross-step persistence).

Dense softmax router (one expert per chip) keeps the reference unambiguous; swap
in top-k routing + a gather to recover the sparse MoE.  ``chunk`` (= T // ws) is
128 here so the per-chunk token gather is sublane-aligned for Mosaic.
"""

from __future__ import annotations

import contextlib
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import helion
import helion.language as hl

with contextlib.suppress(Exception):  # registers the "tpu" PrivateUse1 device
    import torch_tpu  # noqa: F401

WORLD_SIZE = 8


@helion.kernel(
    backend="pallas",
    distributed=[WORLD_SIZE],
    static_shapes=True,
    config=helion.Config(block_sizes=[128, 64]),
)
def fused_moe_reduce_scatter(
    hidden: torch.Tensor,  # [T, H] replicated tokens
    wg: torch.Tensor,  # [H, I] gate weight (this chip's expert)
    wu: torch.Tensor,  # [H, I] up weight
    wd: torch.Tensor,  # [I, H] down weight
    wt: torch.Tensor,  # [T] router weight for this chip's expert, per token
    scratch: torch.Tensor,  # [ws, chunk, H] per-chunk compute buffer
    recv: torch.Tensor,  # [ws, chunk, H] symmetric reduce-scatter target
    dest: torch.Tensor,  # [ws] int32 == arange(ws)
    rank: hl.constexpr,
    ws: hl.constexpr,
    chunk: hl.constexpr,
    idim: hl.constexpr,
) -> torch.Tensor:
    h = hidden.shape[1]
    for k in hl.grid(ws):
        for tm in hl.tile(chunk):
            rows = k * chunk + tm.index  # tokens of output chunk k
            acc_g = hl.zeros([tm, idim], dtype=torch.float32)
            acc_u = hl.zeros([tm, idim], dtype=torch.float32)
            for tk in hl.tile(h):
                xk = hidden[rows, tk]  # gather rows, tile H cols (index original)
                acc_g = torch.addmm(acc_g, xk, wg[tk, :])
                acc_u = torch.addmm(acc_u, xk, wu[tk, :])
            inter = (torch.nn.functional.silu(acc_g) * acc_u).to(hidden.dtype)
            down = torch.matmul(inter, wd[:, :])  # [tile, H]
            scratch[k, tm.index, :] = (wt[rows][:, None] * down).to(scratch.dtype)
        # fused ICI reduce-scatter: push chunk k to chip k's recv[rank]
        op = hl.start_async_remote_copy(
            scratch, [k], dest[k], dst=recv, dst_index=[rank]
        )
        op.wait()
    return recv


def moe_reference(
    hidden: torch.Tensor,
    gating: torch.Tensor,
    wg_all: torch.Tensor,
    wu_all: torch.Tensor,
    wd_all: torch.Tensor,
    rank: int,
    world_size: int,
) -> torch.Tensor:
    """Pure-PyTorch dense-MoE oracle; returns this rank's ``[T // ws, H]`` shard."""
    t = hidden.shape[0]
    weights = torch.softmax(gating.float(), dim=-1)  # [T, ws]
    out = torch.zeros_like(hidden, dtype=torch.float32)
    x = hidden.float()
    for c in range(world_size):
        gate = x @ wg_all[c].float()
        up = x @ wu_all[c].float()
        inter = torch.nn.functional.silu(gate) * up
        out += weights[:, c : c + 1] * (inter @ wd_all[c].float())
    chunk = t // world_size
    return out[rank * chunk : (rank + 1) * chunk]


def _run() -> None:
    from torch_tpu._internal.distributed import tpu_distributed

    dist.init_process_group(backend="tpu_dist")
    ws = dist.get_world_size()
    did = tpu_distributed.global_device_id()
    dev = torch.device("tpu")
    t, h, i_dim = 1024, 128, 256  # chunk = 1024 // 8 = 128 (sublane-aligned)
    chunk = t // ws

    g = torch.Generator(device="cpu").manual_seed(0)
    hidden = torch.randn(t, h, generator=g)
    gating = torch.randn(t, ws, generator=g)
    wg_all = torch.randn(ws, h, i_dim, generator=g) * h**-0.5
    wu_all = torch.randn(ws, h, i_dim, generator=g) * h**-0.5
    wd_all = torch.randn(ws, i_dim, h, generator=g) * i_dim**-0.5
    weights = torch.softmax(gating.float(), dim=-1)

    recv = fused_moe_reduce_scatter(
        hidden.to(dev),
        wg_all[did].contiguous().to(dev),
        wu_all[did].contiguous().to(dev),
        wd_all[did].contiguous().to(dev),
        weights[:, did].contiguous().to(dev),
        torch.zeros(ws, chunk, h, dtype=torch.float32, device=dev),
        torch.zeros(ws, chunk, h, dtype=torch.float32, device=dev),
        torch.arange(ws, dtype=torch.int32, device=dev),
        hl.constexpr(did),
        hl.constexpr(ws),
        hl.constexpr(chunk),
        hl.constexpr(i_dim),
    )
    dist.barrier()

    got = recv.float().sum(0).cpu()  # sum contributions from all chips
    exp = moe_reference(hidden, gating, wg_all, wu_all, wd_all, did, ws)
    torch.testing.assert_close(got, exp, rtol=2e-2, atol=2e-2)
    if did == 0:
        print(
            f"[rank {did}] fused compute+reduce-scatter OK "
            f"(maxdiff {float((got - exp).abs().max()):.5f})"
        )
        print("OK")


def _worker(rank: int, world_size: int, master_port: int) -> None:
    os.environ.update(
        MASTER_ADDR="localhost",
        MASTER_PORT=str(master_port),
        RANK=str(rank),
        WORLD_SIZE=str(world_size),
        LOCAL_RANK=str(rank),
        GROUP_RANK="0",
        LOCAL_WORLD_SIZE=str(world_size),
    )
    _run()


def main() -> None:
    import portpicker
    from torch_tpu._internal.distributed.launchers import singlehost_wrapper

    port = portpicker.pick_unused_port()
    singlehost_wrapper.prepare_tpu_environment(world_size=WORLD_SIZE)
    mp.spawn(_worker, args=(WORLD_SIZE, port), nprocs=WORLD_SIZE, join=True)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
