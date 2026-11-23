"""Distributed kernels shared by ``test/test_distributed.py`` (they run on
several GPUs) and ``test/test_gluon_units.py`` (their code generated on CPU
under a fake process group), and that fake group (``fake_group``)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch
import torch.distributed as dist
from torch.testing._internal.distributed.fake_pg import FakeStore

import helion
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture
def fake_group() -> Generator[str, None, None]:
    """A fake process group of 4 ranks, to bind distributed kernels on CPU;
    yields its name."""
    if dist.is_initialized():
        pytest.skip("needs exclusive control of the process group")
    dist.init_process_group(backend="fake", store=FakeStore(), rank=0, world_size=4)
    assert dist.group.WORLD is not None
    try:
        yield dist.group.WORLD.group_name
    finally:
        dist.destroy_process_group()


def remote_views(t: torch.Tensor, group_name: hl.constexpr) -> tuple[torch.Tensor, ...]:
    return torch.ops.symm_mem.get_remote_tensors(t, group_name)


@helion.kernel(autotune_effort="none", static_shapes=True, backend="gluon")
def tagged_ring_exchange_kernel(
    z: torch.Tensor,
    symm: torch.Tensor,
    x: torch.Tensor,
    w: torch.Tensor,
    s: torch.Tensor,
    group_name: hl.ProcessGroupName,
    rank: hl.constexpr,
) -> tuple[torch.Tensor, torch.Tensor]:
    z_peers = remote_views(z, group_name)
    peers = remote_views(symm, group_name)
    out = torch.empty([z.size(0), w.size(1)], dtype=torch.float32, device=z.device)
    total = torch.empty_like(s)
    k = x.size(1)
    # Every rank pushes its columns of z to every rank's copy.
    for tile_m in hl.tile(x.size(0)):
        rows = x[tile_m, :]
        for z_peer in z_peers:
            # pyrefly: ignore [unsupported-operation]
            z_peer[tile_m, rank * k + hl.arange(k)] = rows
    # An inband exchange every rank polls in full: the mirrors' parity credit.
    for tile in hl.tile(s.size(0)):
        symm[tile] = s[tile]
    for tile in hl.tile(s.size(0)):
        acc = hl.zeros([tile], dtype=s.dtype)
        for peer in peers:
            acc = acc + peer[tile]
        total[tile] = acc
    # A consumer whose K loop reads its own copy of z through the ring.
    for tile_m, tile_n in hl.tile([z.size(0), w.size(1)]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(z.size(1)):
            acc = hl.dot(z[tile_m, tile_k], w[tile_k, tile_n], acc=acc)
        out[tile_m, tile_n] = acc
    return out, total


# A ring configuration of tagged_ring_exchange_kernel: z is an MMA operand the
# gated producer fills.
TAGGED_RING = {
    "block_sizes": [64, 64, 64, 128, 64, 64],
    "indexing": "tensor_descriptor",
    "pid_type": "persistent_blocked",
    "num_sm_multiplier": 1,
    "num_stages": 2,
    "num_warps": 4,
}


@helion.kernel(autotune_effort="none", static_shapes=True)
def scatter_after_local_kernel(
    symm: torch.Tensor,
    x: torch.Tensor,
    rows: torch.Tensor,
    count: torch.Tensor,
    group_name: hl.ProcessGroupName,
) -> torch.Tensor:
    """An in-band scatter exchange whose scatter stores rows a local root
    scaled first (``test_distributed.scatter_exchange_kernel``'s scatter)."""
    peers = remote_views(symm, group_name)
    scaled = torch.empty_like(x)
    out = torch.empty(symm.size(), dtype=torch.float32, device=symm.device)
    half = symm.size(1) // 2
    for tile_t in hl.tile(x.size(0), block_size=1):
        scaled[tile_t, :, :] = x[tile_t, :, :] * 2
    for tile_h, tile_t in hl.tile([2, x.size(0)], block_size=[1, 1]):
        if tile_t.begin < count[0]:
            row = rows[tile_t.begin, :]
            cols = tile_h.begin * half + hl.arange(half)
            hl.store(
                symm,
                [row, cols],
                scaled[tile_t.begin, :, cols],
                extra_mask=(row >= 0)[:, None],
            )
    for tile_m, tile_c in hl.tile(symm.size(), block_size=[8, 32]):
        acc = hl.zeros([tile_m, tile_c], dtype=torch.float32)
        for peer in peers:
            acc = acc + peer[tile_m, tile_c].to(torch.float32)
        out[tile_m, tile_c] = acc
    return out


@helion.kernel(autotune_effort="none", static_shapes=True, backend="gluon")
def scatter_ring_kernel(
    symm: torch.Tensor,
    x: torch.Tensor,
    w: torch.Tensor,
    rows: torch.Tensor,
    count: torch.Tensor,
    group_name: hl.ProcessGroupName,
) -> torch.Tensor:
    """``scatter_after_local_kernel`` with a ring matmul in the scatter root:
    its operand from the local root is gated, its weights are not."""
    peers = remote_views(symm, group_name)
    scaled = torch.empty_like(x)
    out = torch.empty(symm.size(), dtype=torch.float32, device=symm.device)
    for tile_t in hl.tile(x.size(0), block_size=1):
        scaled[tile_t, :, :] = x[tile_t, :, :] * 2
    for tile_n, tile_t in hl.tile([w.size(1), x.size(0)], block_size=[None, 1]):
        if tile_t.begin < count[0]:
            row = rows[tile_t.begin, :]
            acc = hl.zeros([row.size(0), tile_n], dtype=torch.float32)
            for tile_k in hl.tile(x.size(2)):
                acc = hl.dot(
                    scaled[tile_t.begin, :, tile_k], w[tile_k, tile_n], acc=acc
                )
            hl.store(
                symm, [row, tile_n], acc.to(symm.dtype), extra_mask=(row >= 0)[:, None]
            )
    for tile_m, tile_c in hl.tile(symm.size(), block_size=[128, 32]):
        total = hl.zeros([tile_m, tile_c], dtype=torch.float32)
        for peer in peers:
            total = total + peer[tile_m, tile_c].to(torch.float32)
        out[tile_m, tile_c] = total
    return out


@helion.kernel(autotune_effort="none", static_shapes=True)
def pipelined_allreduce_kernel(
    symm: torch.Tensor,
    x: torch.Tensor,
    group_name: hl.ProcessGroupName,
    variant: hl.constexpr,
    world: hl.constexpr,
) -> torch.Tensor:
    peers = remote_views(symm, group_name)
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        if variant == "constant":
            symm[tile] = hl.full([tile], 1.0, dtype=x.dtype)
        else:
            symm[tile] = x[tile]
        if variant == "R1":
            symm[tile] = x[tile] + 0
    for tile in hl.tile(x.size(0)):
        acc = hl.zeros([tile], dtype=x.dtype)
        for r in hl.static_range(world):  # pyrefly: ignore[bad-argument-type]
            acc = acc + peers[r][tile]
        out[tile] = acc
        if variant == "print":
            hl.device_print("acc", acc)
        if variant == "loop":
            for inner in hl.tile(tile.begin, tile.end, block_size=64):
                part = hl.zeros([inner], dtype=x.dtype)
                for r in hl.static_range(world):  # pyrefly: ignore[bad-argument-type]
                    part = part + peers[r][inner]
                out[inner] = part
    return out
