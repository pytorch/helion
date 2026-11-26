from __future__ import annotations

from typing import Annotated

import torch

import helion
import helion.language as hl


@helion.kernel(backend="cute", static_shapes=True, config=helion.Config(block_sizes=[]))
def layout_chunks(
    counts: torch.Tensor,
    valid: torch.Tensor,
    rank: Annotated[int, hl.constexpr],
    limit: Annotated[int, hl.constexpr],
    rows: Annotated[int, hl.constexpr] = 0,
    atom_rows: torch.Tensor | None = None,
    assert_status: Annotated[bool, hl.constexpr] = False,
) -> tuple[torch.Tensor, ...]:
    """Compute chunk prefixes and map each padded 128-row atom to expert storage."""
    world = hl.specialize(valid.numel())
    local = hl.specialize(counts.size(1) // world)
    prior = torch.empty(
        (counts.size(0), local), dtype=torch.int32, device=counts.device
    )
    sizes = torch.empty(local, dtype=torch.int32, device=counts.device)
    edges = torch.empty(local + 1, dtype=torch.int64, device=counts.device)
    recv = torch.empty(world, dtype=torch.int32, device=counts.device)
    ok = torch.empty(1, dtype=torch.bool, device=counts.device)
    for dst in hl.grid(world):
        e = hl.arange(local)
        v = counts[:, dst * local + e]
        n = torch.sum(v, dim=0)
        padded = ((n + 127) // 128) * 128
        total = torch.sum(padded, dim=0)
        recv[dst] = total
        if dst == rank:
            prior[:, :] = torch.cumsum(v, dim=0) - v
            sizes[:] = padded
            edges[0] = 0
            edges[e + 1] = torch.cumsum(padded, dim=0).to(torch.int64)
            good = torch.all(valid[:]) & (total <= limit)
            if rows > 0:
                good = good & (torch.max(n) <= rows)
            ok[0] = good
            if atom_rows is not None:
                ends = torch.cumsum(padded, dim=0)
                starts = ends - padded
                for j in hl.static_range((atom_rows.numel() + 31) // 32):
                    atom = j * 32 + hl.arange(32)
                    compact = atom[:, None] * 128
                    inside = (compact >= starts[None, :]) & (compact < ends[None, :])
                    physical = e[None, :] * rows + compact - starts[None, :]
                    base = torch.amax(torch.where(inside, physical, -1), dim=1)
                    hl.store(
                        atom_rows,
                        [atom],
                        torch.where(good, base, -1),
                        extra_mask=atom < atom_rows.numel(),
                    )
            if assert_status:
                torch._assert_async(good, STATUS_MESSAGE)
    return prior, sizes, edges, recv, ok


STATUS_MESSAGE = "Dist-MoE peer token count mismatch or receive capacity exceeded"
