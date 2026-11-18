"""Which loads of a prefetch-only root order later writes (``TileAccess.hint``).

Bound on CPU, where no cross-loop pipeline runs: a kernel whose later root
waits on an earlier one raises ``LoopDependencyError`` at bind time.
"""

from __future__ import annotations

import pytest
import torch

import helion
from helion import exc
from helion._testing import TestCase
from helion._testing import onlyBackends
from helion._testing import prefetch_kernels
from helion._testing.cute_codegen import cpu_bind
import helion.language as hl

pytestmark = pytest.mark.cpu_only


hint_read_roots = helion.kernel(autotune_effort="none", static_shapes=True)(
    prefetch_kernels.hint_read_roots
)


@helion.kernel(autotune_effort="none", static_shapes=True)
def ordered_read_roots(
    x: torch.Tensor, ids: torch.Tensor, z: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """``hint_read_roots`` whose first root also stores what it read."""
    seen = torch.empty_like(ids)
    for tile in hl.tile(ids.size(0), block_size=1):
        row = ids[tile.begin]
        seen[tile] = ids[tile]
        if row >= 0:
            hl.prefetch(x, [row, slice(None)])
    for tile in hl.tile(ids.size(0), block_size=16):
        ids[tile] = z[tile]
    return ids, seen


@helion.kernel(autotune_effort="none", static_shapes=True)
def counted_read_roots(
    x: torch.Tensor, ids: torch.Tensor, z: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """``hint_read_roots`` whose first root also counts its prefetches."""
    count = torch.zeros([1], dtype=torch.int32, device=ids.device)
    for tile in hl.tile(ids.size(0), block_size=1):
        row = ids[tile.begin]
        if row >= 0:
            hl.prefetch(x, [row, slice(None)])
            hl.atomic_add(count, [0], 1)
    for tile in hl.tile(ids.size(0), block_size=16):
        ids[tile] = z[tile]
    return ids, count


@helion.kernel(autotune_effort="none", static_shapes=True)
def polling_read_roots(
    x: torch.Tensor, ids: torch.Tensor, z: torch.Tensor
) -> torch.Tensor:
    """``hint_read_roots`` whose first root spins until its entry is nonzero."""
    for tile in hl.tile(ids.size(0), block_size=1):
        row = ids[tile.begin]
        while row == 0:
            row = ids[tile.begin]
        hl.prefetch(x, [row, slice(None)])
    for tile in hl.tile(ids.size(0), block_size=16):
        ids[tile] = z[tile]
    return ids


@helion.kernel(autotune_effort="none", static_shapes=True)
def indexed_read_roots(
    x: torch.Tensor, ids: torch.Tensor, z: torch.Tensor
) -> torch.Tensor:
    """``hint_read_roots`` whose first root reads a row of ``x`` at its entry."""
    for tile in hl.tile(ids.size(0), block_size=1):
        row = ids[tile.begin]
        if x[row, 0] > 0:
            hl.prefetch(x, [row, slice(None)])
    for tile in hl.tile(ids.size(0), block_size=16):
        ids[tile] = z[tile]
    return ids


@helion.kernel(autotune_effort="none", static_shapes=True)
def masked_read_roots(
    x: torch.Tensor, ids: torch.Tensor, z: torch.Tensor
) -> torch.Tensor:
    """``hint_read_roots`` whose first root reads ``x`` under a mask its entry
    sets."""
    for tile in hl.tile(ids.size(0), block_size=1):
        row = ids[tile.begin]
        first = hl.load(x, [tile.begin, 0], extra_mask=row > 0)
        hl.prefetch(x, [first.to(torch.int32), slice(None)])
    for tile in hl.tile(ids.size(0), block_size=16):
        ids[tile] = z[tile]
    return ids


@helion.kernel(autotune_effort="none", static_shapes=True)
def stepped_read_roots(
    x: torch.Tensor, ids: torch.Tensor, z: torch.Tensor
) -> torch.Tensor:
    """``hint_read_roots`` whose first root prefetches every other row up to its
    entry."""
    for tile in hl.tile(ids.size(0), block_size=1):
        rows = ids[tile.begin]
        for row in hl.grid(0, rows, 2):
            hl.prefetch(x, [row, slice(None)])
    for tile in hl.tile(ids.size(0), block_size=16):
        ids[tile] = z[tile]
    return ids


@helion.kernel(autotune_effort="none", static_shapes=True)
def branched_read_roots(
    x: torch.Tensor, ids: torch.Tensor, z: torch.Tensor
) -> torch.Tensor:
    """``hint_read_roots`` whose first root reads ``x`` (at an index no load
    computes) in the branch its entry takes."""
    for tile in hl.tile(ids.size(0), block_size=1):
        row = ids[tile.begin]
        if row >= 0:
            first = x[tile.begin, 0]
            hl.prefetch(x, [first.to(torch.int32), slice(None)])
    for tile in hl.tile(ids.size(0), block_size=16):
        ids[tile] = z[tile]
    return ids


@helion.kernel(autotune_effort="none", static_shapes=True)
def printing_read_roots(
    x: torch.Tensor, ids: torch.Tensor, z: torch.Tensor
) -> torch.Tensor:
    """``hint_read_roots`` whose first root prints its entry."""
    for tile in hl.tile(ids.size(0), block_size=1):
        row = ids[tile.begin]
        hl.device_print("row", row)
        if row >= 0:
            hl.prefetch(x, [row, slice(None)])
    for tile in hl.tile(ids.size(0), block_size=16):
        ids[tile] = z[tile]
    return ids


def _printing_max(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    hl.device_print("max", a)
    return torch.maximum(a, b)


@helion.kernel(autotune_effort="none", static_shapes=True)
def reducing_read_roots(
    x: torch.Tensor, ids: torch.Tensor, z: torch.Tensor
) -> torch.Tensor:
    """``hint_read_roots`` whose first root prints in a reduction's combine
    function."""
    for _tile in hl.tile(ids.size(0), block_size=1):
        row = hl.reduce(_printing_max, ids[:], dim=0)
        if row >= 0:
            hl.prefetch(x, [row, slice(None)])
    for tile in hl.tile(ids.size(0), block_size=16):
        ids[tile] = z[tile]
    return ids


@helion.kernel(autotune_effort="none", static_shapes=True)
def written_hint_roots(
    x: torch.Tensor, ids: torch.Tensor, z: torch.Tensor
) -> torch.Tensor:
    """``hint_read_roots`` whose list an earlier root writes."""
    for tile in hl.tile(ids.size(0), block_size=16):
        ids[tile] = z[tile]
    for tile in hl.tile(ids.size(0), block_size=1):
        row = ids[tile.begin]
        if row >= 0:
            hl.prefetch(x, [row, slice(None)])
    return ids


@onlyBackends(["triton"])
class TestPrefetchHints(TestCase):
    def test_prefetch_only_root_loads_order_no_later_write(self) -> None:
        ids = torch.arange(32, dtype=torch.int32)
        args = (torch.randn(64, 256), ids, torch.full_like(ids, 7))
        # Its loads only steer prefetch addresses: the later root that rewrites
        # ``ids`` need not wait for them, so the kernel has no dependency.
        bound = cpu_bind(hint_read_roots, args)
        self.assertIsNone(bound.config_spec.cross_loop_pipeline)
        # The rewrite still waits for a root that also stores or atomically
        # updates, whose loads bound a loop, index, mask or guard another load,
        # or that prints (also in a combine function); and a hint root still
        # waits for an earlier root's writes.
        for kernel in (
            ordered_read_roots,
            counted_read_roots,
            polling_read_roots,
            indexed_read_roots,
            masked_read_roots,
            stepped_read_roots,
            branched_read_roots,
            printing_read_roots,
            reducing_read_roots,
            written_hint_roots,
        ):
            with (
                self.subTest(kernel=kernel.fn.__name__),
                self.assertRaises(exc.LoopDependencyError),
            ):
                cpu_bind(kernel, args)
