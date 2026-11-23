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
from helion._testing import skipIfRefEager
from helion._testing import skipIfXPU
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
# Binding for the modelled sm_100 target asks the CUDA allocator whether a
# prefetch can take the tensor form; XPU builds of torch have no CUDA allocator.
@skipIfXPU("the modelled CUDA target queries the CUDA allocator")
class TestPrefetchHints(TestCase):
    @skipIfRefEager("reads compiler state, which ref mode does not build")
    def test_prefetch_only_root_loads_order_no_later_write(self) -> None:
        ids = torch.arange(32, dtype=torch.int32)
        args = (torch.randn(64, 256), ids, torch.full_like(ids, 7))
        # Its loads only steer prefetch addresses: the later root that rewrites
        # ``ids`` need not wait for them, so the kernel has no dependency.
        bound = cpu_bind(hint_read_roots, args)
        self.assertIsNone(bound.config_spec.cross_loop_pipeline)
        # The rewrite still waits for a root that also stores, or whose loads
        # index another load; and a hint root still waits for an earlier
        # root's writes.
        for kernel in (
            ordered_read_roots,
            indexed_read_roots,
            written_hint_roots,
        ):
            with (
                self.subTest(kernel=kernel.fn.__name__),
                self.assertRaises(exc.LoopDependencyError),
            ):
                cpu_bind(kernel, args)
