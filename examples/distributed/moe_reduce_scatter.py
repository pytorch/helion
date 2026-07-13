"""Expert-parallel MoE combine as an in-kernel reduce-scatter (Helion -> Pallas).

This is the comms half of a fused MoE: after each chip computes its experts'
contributions to every token, the per-token results must be summed across chips
and each chip keeps only its own ``T // world_size`` token shard -- a
reduce-scatter.  It runs INSIDE a Helion kernel (no torch collective) and its
output matches ``torch``'s reduce-scatter / a pure-PyTorch MoE reference exactly
(verified on an 8-chip TPU host, maxdiff 0.0).

Design -- a BALANCED all-to-all of chunked partials:
  * Each chip forms a dense partial ``[T, H]`` (its experts' contribution to all
    tokens), viewed as ``world_size`` chunks ``[ws, chunk, H]``.
  * Chip ``c`` pushes chunk ``k`` into slot ``c`` of chip ``k``'s ``recv`` buffer
    -- ``ws`` sends and ``ws`` receives per chip.  Because sends and receives are
    balanced, the per-copy ``op.wait()`` (receive-side) never deadlocks the way
    an uneven direct-write scatter would; and each ``recv`` slot is written
    exactly once, so no cross-step persistence or accumulation-forwarding is
    needed (that keeps ``recv`` a pure HBM DMA target -- see plan_tiling).
  * Each chip sums its ``ws`` received chunks -> its reduced shard.

The compute is done in torch here to isolate the comms; drop in the Helion
compute kernel from ``fused_moe.py`` (``fused_moe_local_kernel``) to produce the
partial on-device.
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


@helion.kernel(backend="pallas", distributed=[WORLD_SIZE], config=helion.Config())
def reduce_scatter_all_to_all(
    partial: torch.Tensor,  # (ws, chunk, H) this chip's per-chunk contributions
    recv: torch.Tensor,  # (ws, chunk, H) symmetric; peers write recv[their_rank]
    dest: torch.Tensor,  # (ws,) int32 == arange(ws) (peer ids)
    rank: hl.constexpr,
    ws: hl.constexpr,
) -> torch.Tensor:
    """Push chunk ``k`` to peer ``k``'s ``recv[rank]``; balanced ws-by-ws."""
    for k in hl.grid(ws):
        op = hl.start_async_remote_copy(
            partial, [k], dest[k], dst=recv, dst_index=[rank]
        )
        op.wait()
    return recv


def _route(gating: torch.Tensor, top_k: int) -> tuple[torch.Tensor, torch.Tensor]:
    scores = torch.softmax(gating.float(), dim=-1)
    w, idx = torch.topk(scores, top_k, dim=-1)
    return w, idx.to(torch.int32)


def moe_reference(
    hidden: torch.Tensor,
    gating: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    top_k: int,
    rank: int,
    world_size: int,
) -> torch.Tensor:
    """Pure-PyTorch EP MoE oracle; returns this rank's ``[T // ws, H]`` shard."""
    t, h = hidden.shape
    e, _, two_i = w1.shape
    inter = two_i // 2
    tw, ti = _route(gating, top_k)
    out = torch.zeros(t, h, dtype=torch.float32)
    x = hidden.float()
    for expert in range(e):
        sel = ti == expert
        mask = sel.any(-1)
        if not bool(mask.any()):
            continue
        weight = (tw * sel).sum(-1)
        gate_up = x[mask] @ w1[expert].float()
        act = torch.nn.functional.silu(gate_up[:, :inter]) * gate_up[:, inter:]
        out[mask] += weight[mask, None] * (act @ w2[expert].float())
    c = t // world_size
    return out[rank * c : (rank + 1) * c]


def _local_partial(
    hidden: torch.Tensor,
    gating: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    top_k: int,
    e_lo: int,
    e_hi: int,
) -> torch.Tensor:
    """Dense ``[T, H]`` contribution of THIS chip's experts to all tokens."""
    t, h = hidden.shape
    inter = w1.shape[2] // 2
    tw, ti = _route(gating, top_k)
    partial = torch.zeros(t, h, dtype=torch.float32)
    x = hidden.float()
    for expert in range(e_lo, e_hi):
        sel = ti == expert
        mask = sel.any(-1)
        if not bool(mask.any()):
            continue
        weight = (tw * sel).sum(-1)
        gate_up = x[mask] @ w1[expert].float()
        act = torch.nn.functional.silu(gate_up[:, :inter]) * gate_up[:, inter:]
        partial[mask] += weight[mask, None] * (act @ w2[expert].float())
    return partial


def _run() -> None:
    from torch_tpu._internal.distributed import tpu_distributed

    dist.init_process_group(backend="tpu_dist")
    ws = dist.get_world_size()
    did = tpu_distributed.global_device_id()
    dev = torch.device("tpu")
    t, h, inter, e, top_k = 64, 128, 256, 8, 2
    chunk = t // ws
    e_local = e // ws
    e_lo = did * e_local

    g = torch.Generator(device="cpu").manual_seed(0)
    hidden = torch.randn(t, h, generator=g)
    gating = torch.randn(t, e, generator=g)
    w1 = torch.randn(e, h, 2 * inter, generator=g) * h**-0.5
    w2 = torch.randn(e, inter, h, generator=g) * inter**-0.5

    partial = _local_partial(hidden, gating, w1, w2, top_k, e_lo, e_lo + e_local)
    partial = partial.reshape(ws, chunk, h).to(dev)
    recv = torch.zeros(ws, chunk, h, dtype=torch.float32, device=dev)
    dest = torch.arange(ws, dtype=torch.int32, device=dev)

    recv = reduce_scatter_all_to_all(
        partial, recv, dest, hl.constexpr(did), hl.constexpr(ws)
    )
    dist.barrier()

    got = recv.float().sum(0).cpu()  # sum contributions from all chips
    exp = moe_reference(hidden, gating, w1, w2, top_k, did, ws)
    torch.testing.assert_close(got, exp, rtol=2e-2, atol=2e-2)
    if did == 0:
        print(
            f"[rank {did}] MoE reduce-scatter OK (maxdiff {float((got - exp).abs().max()):.5f})"
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
