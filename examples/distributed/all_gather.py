"""Ring all-gather written in Helion, lowered to a Pallas TPU kernel.

Mirrors the JAX docs example
(``https://docs.jax.dev/en/latest/pallas/tpu/distributed.html#example-all-gather-lax-all-gather``)
using two new pieces::

    hl.start_async_remote_copy(tensor, index, device_id)
        # pushes tensor[index] locally -> tensor[index] on peer device_id
    @helion.kernel(distributed=[world_size])
        # marker that this kernel participates in a distributed group

``device_id`` is a flat integer PE id (LOGICAL / NVSHMEM-style
addressing) — obtained on the host side from ``dist.get_rank()`` and
passed to the kernel as an ``hl.constexpr``.  Because the ring math is
Python-level (rank and world_size are compile-time constants per
process), each rank compiles its own specialized kernel.

Algorithm (identical to the JAX docs example):

    Each rank seeds slot ``rank`` of a symmetric gather buffer of shape
    ``(world_size, per_device, inner)``.  At step ``s`` of a
    ``world_size - 1``-step loop, each rank forwards slot
    ``(rank - s) mod world_size`` to its right neighbour.  After
    ``world_size - 1`` steps every rank has the complete gathered
    output in every slot.

Run on a TPU host with 8 chips::

    source dunfanlu_notes/scripts/init_helion_pallas_tpu.sh
    python examples/distributed/all_gather.py
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import helion
import helion.language as hl


@helion.kernel(backend="pallas", distributed=[8], config=helion.Config())
def ring_all_gather(
    local: torch.Tensor,
    gather: torch.Tensor,
    rank: hl.constexpr,
    world_size: hl.constexpr,
) -> torch.Tensor:
    """Ring all-gather.

    ``gather`` has shape ``(world_size, *local.shape)`` and is filled
    in-place by the kernel.  At step 0 the kernel seeds slot ``rank``
    with ``local`` (the local piece); on each subsequent iteration it
    pushes the current slot to the right neighbour.  After
    ``world_size - 1`` iterations every rank has the full gathered
    tensor.
    """
    right = (rank + 1) % world_size
    for step in hl.grid(world_size - 1):
        # rank and world_size are constexpr; step is a symint.
        if step == 0:
            # Seed slot ``rank`` with this device's local piece.  Pallas
            # allows ``pltpu.make_async_remote_copy`` to address any
            # memory space, so ``gather`` stays in VMEM by default and
            # the assignment is an ordinary slice store.
            gather[rank, :, :] = local
        slot = (rank + world_size - step) % world_size
        ring_step = hl.start_async_remote_copy(gather, [slot], right)
        ring_step.wait()
    return gather


def _run() -> None:
    """One process per TPU chip."""
    from torch_tpu._internal.distributed import tpu_distributed

    device = torch.device("tpu")
    dist.init_process_group(backend="tpu_dist")
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    device_id = tpu_distributed.global_device_id()

    per = 8
    inner = 128

    # This rank's local piece: (per, inner) filled with its device id
    # so the gather result is trivially verifiable.
    local = torch.full(
        (per, inner), float(device_id), dtype=torch.float32, device=device
    )

    # Symmetric gather buffer: (world_size, per, inner).  Every rank
    # allocates the same shape.  The kernel seeds slot ``device_id``
    # from ``local`` in-place; no host-side seeding is required.
    gather = torch.zeros((world_size, per, inner), dtype=torch.float32, device=device)

    # ring_all_gather is a torch-tensor-facing Helion kernel; call it
    # like any torch op.  With this rank's constexpr rank baked in,
    # the kernel emits an 8-chip-mesh Pallas program that DMAs slots
    # around the ring.
    gathered = ring_all_gather(
        local, gather, hl.constexpr(device_id), hl.constexpr(world_size)
    )

    expected = torch.stack(
        [
            torch.full((per, inner), float(d), dtype=torch.float32, device=device)
            for d in range(world_size)
        ],
        dim=0,
    )
    torch.testing.assert_close(gathered.cpu(), expected.cpu())
    if rank == 0:
        print(
            f"[rank {rank}] gathered.shape={tuple(gathered.shape)} "
            f"first row mean={gathered[0].mean().item():.3f} "
            f"last row mean={gathered[-1].mean().item():.3f}"
        )
        print("OK")


def _worker(rank: int, world_size: int, master_port: int) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["GROUP_RANK"] = "0"
    os.environ["LOCAL_WORLD_SIZE"] = str(world_size)
    _run()


def main() -> None:
    import portpicker
    from torch_tpu._internal.distributed.launchers import singlehost_wrapper

    world_size = 8
    master_port = portpicker.pick_unused_port()
    singlehost_wrapper.prepare_tpu_environment(world_size=world_size)
    mp.spawn(_worker, args=(world_size, master_port), nprocs=world_size, join=True)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
