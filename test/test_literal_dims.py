"""Literal tile dims (``hl.zeros([tile_m, 64])``) beside tiles of that size.

Triton combines tile tensors by position.  CuTe holds every block id at one
coordinate per thread (and per lane iteration) and finds the block of a
literal dim by its size: the reduction block of that extent, never a tile,
whose block carries the loop's extent.  A literal dim that holds a tile's
elements (an element-wise op with a tile, a load through a tile whose block
size became a literal) must therefore stay away from the consumers that
find its block by size, and a literal carry cannot hold a device loop's
tile: each thread carried one scalar through the tile's lanes.  CuTe refuses
those (``cute/literal_dims.py``); literal dims that agree keep working.
"""

from __future__ import annotations

import unittest

import torch

import helion
from helion import exc
from helion._testing import DEVICE
from helion._testing import RefEagerTestBase
from helion._testing import TestCase
from helion._testing import _get_backend
from helion._testing import code_and_output
from helion._testing import onlyBackends
import helion.language as hl


@helion.kernel(static_shapes=True)
def literal_carry_row_sum(x: torch.Tensor) -> torch.Tensor:
    """Column partials of 64-wide tiles in a literal carry, summed after."""
    m, n = x.shape
    out = torch.empty([m], dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, 64], dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=64):
            acc = acc + x[tile_m, tile_n]
        out[tile_m] = acc.sum(1)
    return out


@helion.kernel(static_shapes=True)
def literal_carry_tile_first(x: torch.Tensor) -> torch.Tensor:
    """``literal_carry_row_sum`` with the tile as the first operand."""
    m, n = x.shape
    out = torch.empty([m, 64], dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, 64], dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=64):
            acc = x[tile_m, tile_n] + acc
        out[tile_m, :] = acc
    return out


@helion.kernel(static_shapes=True)
def tile_plus_slice(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """A 64-wide tile plus a ``:`` slice of a 64-wide tensor."""
    m, n = x.shape
    out = torch.empty_like(x)
    for tile_m, tile_n in hl.tile([m, n], block_size=[None, 64]):
        out[tile_m, tile_n] = x[tile_m, tile_n] + y[tile_m, :]
    return out


@helion.kernel(static_shapes=True)
def literal_store_into_tile(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """A literal-wide value stored through a 64-wide tile."""
    m, n = x.shape
    out = torch.empty_like(x)
    for tile_m, tile_n in hl.tile([m, n], block_size=[None, 64]):
        out[tile_m, tile_n] = y[tile_m, :] + hl.zeros([tile_m, 64], dtype=x.dtype)
    return out


@helion.kernel(static_shapes=True)
def literal_carry_of_slices(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """A literal carry updated with ``:`` slices of its width."""
    m, k = x.shape
    out = torch.empty([m, 64], dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, 64], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = acc + y[tile_m, :] * x[tile_m, tile_k].sum(1, keepdim=True)
        out[tile_m, :] = acc
    return out


@helion.kernel(static_shapes=True)
def reshaped_tile_row_sum(x: torch.Tensor) -> torch.Tensor:
    """A 64-wide tile reshaped to a literal width (the tile's block size
    becomes the literal 64), then summed by the literal dim."""
    m, n = x.shape
    out = torch.zeros([m], dtype=torch.float32, device=x.device)
    for tile_m, tile_n in hl.tile([m, n], block_size=[None, 64]):
        rows = x[tile_m, tile_n].reshape([tile_m, 64])
        hl.atomic_add(out, [tile_m], rows.sum(1))
    return out


@helion.kernel(static_shapes=True)
def literal_carry_beside_grid_tile(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """A literal carry holding a 64-wide grid tile through an inner loop."""
    m, n = x.shape
    out = torch.empty_like(x)
    for tile_m, tile_n in hl.tile([m, n], block_size=[None, 64]):
        acc = hl.zeros([tile_m, 64], dtype=torch.float32)
        for tile_k in hl.tile(w.size(0)):
            acc = acc + x[tile_m, tile_n] * w[tile_k].sum()
        out[tile_m, tile_n] = acc
    return out


@helion.kernel(static_shapes=True)
def tile_plus_constants(x: torch.Tensor) -> torch.Tensor:
    """A tile plus literal-wide values that are the same along the dim or an
    iota matched to the tile."""
    m, n = x.shape
    out = torch.empty_like(x)
    for tile_m, tile_n in hl.tile([m, n], block_size=[None, 64]):
        # The literal-wide result of the first sum holds the tile's elements.
        shifted = hl.full([tile_m, 64], 1.0, dtype=x.dtype) + x[tile_m, tile_n]
        out[tile_m, tile_n] = shifted + hl.arange(64)[None, :].to(x.dtype)
    return out


def _ints(*shape: int) -> torch.Tensor:
    # Integer values keep the sums exact in any order.
    return torch.randint(-4, 5, shape, device=DEVICE).to(torch.float32)


@onlyBackends(["triton", "cute"])
class TestLiteralDims(RefEagerTestBase, TestCase):
    def _assert_refused_on_cute(
        self,
        kernel: object,
        args: tuple[torch.Tensor, ...],
        expected: torch.Tensor,
        reason: str = "finds the block of a literal dim by its size",
    ) -> None:
        if _get_backend() == "cute" and not self._in_ref_eager_mode:
            with self.assertRaisesRegex(exc.BackendUnsupported, reason):
                code_and_output(kernel, args)
            return
        _code, result = code_and_output(kernel, args)
        torch.testing.assert_close(result, expected)

    def test_literal_carry_beside_device_loop_tile(self) -> None:
        # CuTe summed every column of a thread's lanes into one scalar and
        # reduced it across the rows.
        x = _ints(96, 640)
        reason = "loop carry .* holds block id 1's tile .* that tile's own loop"
        self._assert_refused_on_cute(literal_carry_row_sum, (x,), x.sum(1), reason)
        self._assert_refused_on_cute(
            literal_carry_tile_first, (x,), x.view(96, 10, 64).sum(1), reason
        )

    def test_literal_slice_beside_tile(self) -> None:
        # The ``:`` slice's reduction block walked all 64 columns beside
        # each element of the tile.
        x, y = _ints(96, 64), _ints(96, 64)
        self._assert_refused_on_cute(tile_plus_slice, (x, y), x + y)
        self._assert_refused_on_cute(literal_store_into_tile, (x, y), y)

    def test_tile_reshaped_to_literal_width(self) -> None:
        # The reshape made the tile's block size the literal 64 everywhere;
        # the sum found the reduction block of 64 for the tile's dim.
        x = _ints(96, 64)
        self._assert_refused_on_cute(reshaped_tile_row_sum, (x,), x.sum(1))

    def test_agreeing_literal_dims(self) -> None:
        x, y = _ints(96, 640), _ints(96, 64)
        _code, result = code_and_output(literal_carry_of_slices, (x, y))
        torch.testing.assert_close(result, y * x.sum(1, keepdim=True))
        # The carry holds the grid tile, which the inner loop keeps.
        w = _ints(48)
        _code, result = code_and_output(literal_carry_beside_grid_tile, (x, w))
        torch.testing.assert_close(result, x * w.sum())
        x = _ints(96, 64)
        _code, result = code_and_output(tile_plus_constants, (x,))
        torch.testing.assert_close(
            result, x + 1.0 + torch.arange(64, device=DEVICE, dtype=x.dtype)
        )


if __name__ == "__main__":
    unittest.main()
