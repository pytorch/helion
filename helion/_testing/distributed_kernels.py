"""Distributed kernels shared by ``test/test_distributed.py`` (they run on
several GPUs) and ``test/test_gluon_units.py`` (their code generated on CPU
under a fake process group)."""

from __future__ import annotations

import torch

import helion
import helion.language as hl


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
