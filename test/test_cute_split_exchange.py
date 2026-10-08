"""Shared-memory ``hl.split`` exchange under CuTe lane loops.

``hl.split`` of a non-load tile stages the tile in shared memory inside the
current lane iteration.  The exchange is only correct when both pair elements
are staged in the same lane iteration that reads them; the layouts below pin
down which combinations of lane layout and pair geometry that admits.
"""

from __future__ import annotations

import unittest

import torch

import helion
from helion import exc
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
import helion.language as hl


@helion.kernel(static_shapes=True)
def halves_tiled(x: torch.Tensor) -> torch.Tensor:
    """Block-local halves of a tiled dim, swapped (``x * 2`` defeats the load fold)."""
    n, d = x.size()
    out = torch.empty_like(x)
    for tile_n, tile_d in hl.tile([n, d]):
        pair = (
            (x[tile_n, tile_d] * 2.0)
            .reshape([tile_n, 2, tile_d.block_size // 2])
            .permute(0, 2, 1)
        )
        lo, hi = hl.split(pair)
        out[tile_n, tile_d] = hl.join(hi, lo).permute(0, 2, 1).reshape([tile_n, tile_d])
    return out


@helion.kernel(static_shapes=True)
def interleaved_tiled(x: torch.Tensor) -> torch.Tensor:
    """Adjacent pairs of a tiled dim, swapped."""
    n, d = x.size()
    out = torch.empty_like(x)
    for tile_n, tile_d in hl.tile([n, d]):
        pair = (x[tile_n, tile_d] * 2.0).reshape([tile_n, tile_d.block_size // 2, 2])
        lo, hi = hl.split(pair)
        out[tile_n, tile_d] = hl.join(hi, lo).reshape([tile_n, tile_d])
    return out


@helion.kernel(static_shapes=True)
def halves_full(x: torch.Tensor) -> torch.Tensor:
    """Halves of a ``:`` dim (the rope layout), swapped."""
    n, d = x.size()
    out = torch.empty_like(x)
    for tile_n in hl.tile(n):
        pair = (x[tile_n, :] * 2.0).reshape([tile_n, 2, d // 2]).permute(0, 2, 1)
        lo, hi = hl.split(pair)
        out[tile_n, :] = hl.join(hi, lo).permute(0, 2, 1).reshape([tile_n, d])
    return out


@helion.kernel(static_shapes=True)
def interleaved_full(x: torch.Tensor) -> torch.Tensor:
    """Adjacent pairs of a ``:`` dim, swapped."""
    n, d = x.size()
    out = torch.empty_like(x)
    for tile_n in hl.tile(n):
        pair = (x[tile_n, :] * 2.0).reshape([tile_n, d // 2, 2])
        lo, hi = hl.split(pair)
        out[tile_n, :] = hl.join(hi, lo).reshape([tile_n, d])
    return out


@helion.kernel(static_shapes=True)
def halves_loaded(x: torch.Tensor) -> torch.Tensor:
    """Block-local halves of a loaded tile, swapped (re-read, no exchange)."""
    n, d = x.size()
    out = torch.empty_like(x)
    for tile_n, tile_d in hl.tile([n, d]):
        pair = x[tile_n, tile_d].reshape([tile_n, 2, tile_d.block_size // 2])
        lo, hi = hl.split(pair.permute(0, 2, 1))
        out[tile_n, tile_d] = hl.join(hi, lo).permute(0, 2, 1).reshape([tile_n, tile_d])
    return out


@helion.kernel(static_shapes=True)
def join_then_split(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """The pair dim of ``hl.join`` belongs to no block."""
    n, d = x.size()
    out = torch.empty_like(x)
    for tile_n, tile_d in hl.tile([n, d]):
        lo, hi = hl.split(hl.join(x[tile_n, tile_d], y[tile_n, tile_d]))
        out[tile_n, tile_d] = hi - lo
    return out


@helion.kernel(static_shapes=True)
def transposed_pairs(x: torch.Tensor) -> torch.Tensor:
    """Pairs of a transposed tile; the permute is a cross-thread shuffle."""
    n, d = x.size()
    out = torch.empty([d, n], dtype=x.dtype, device=x.device)
    for tile_n, tile_d in hl.tile([n, d]):
        pair = (
            x[tile_n, tile_d].permute(1, 0).reshape([tile_d, tile_n.block_size // 2, 2])
        )
        lo, hi = hl.split(pair)
        out[tile_d, tile_n] = hl.join(hi, lo).reshape([tile_d, tile_n])
    return out


def _swapped_halves(x: torch.Tensor, block: int) -> torch.Tensor:
    pairs = (x * 2.0).view(x.shape[0], x.shape[1] // block, 2, block // 2)
    return torch.cat((pairs[:, :, 1:], pairs[:, :, :1]), dim=2).view(x.shape)


def _swapped_interleaved(x: torch.Tensor) -> torch.Tensor:
    pairs = (x * 2.0).view(x.shape[0], x.shape[1] // 2, 2)
    return torch.stack((pairs[..., 1], pairs[..., 0]), dim=-1).view(x.shape)


@onlyBackends(["cute"])
class TestCuteSplitExchange(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.x = torch.randn([128, 64], device=DEVICE)

    def _assert_exchange(
        self, kernel: object, expected: torch.Tensor, **config: object
    ) -> None:
        code, result = code_and_output(kernel, (self.x,), **config)
        self.assertIn("split_smem", code)
        self.assertIn("cute.arch.sync_threads()", code)
        torch.testing.assert_close(result, expected)

    def _assert_rejected(self, kernel: object, **config: object) -> None:
        with self.assertRaisesRegex(
            exc.BackendUnsupported, "different iteration of an enclosing lane loop"
        ):
            code_and_output(kernel, (self.x,), **config)

    def test_halves_blocked_lanes(self) -> None:
        # ``tid * EPT + lane``: the partner ``half`` columns away shares the
        # lane whenever EPT divides ``half`` (the epilogue-subtiling layout).
        self._assert_exchange(
            halves_tiled, _swapped_halves(self.x, 64), block_sizes=[64, 64]
        )

    def test_interleaved_strided_lanes(self) -> None:
        # ``tid + lane * NT``: adjacent columns stay in one lane iteration.
        self._assert_exchange(
            interleaved_tiled,
            _swapped_interleaved(self.x),
            block_sizes=[64, 64],
            cute_lane_layouts=["blocked", "strided"],
        )

    def test_interleaved_synthetic_lanes(self) -> None:
        # ``:`` dims use ``tid + lane * T``; adjacent pairs never cross a lane.
        self._assert_exchange(
            interleaved_full, _swapped_interleaved(self.x), block_size=64
        )

    def test_halves_strided_lanes_rejected(self) -> None:
        # ``tid + lane * NT`` with NT == half: the partner is the next lane
        # iteration, which has not been staged when the first one reads it.
        self._assert_rejected(
            halves_tiled,
            block_sizes=[64, 64],
            cute_lane_layouts=["blocked", "strided"],
        )

    def test_interleaved_blocked_lanes_rejected(self) -> None:
        # ``tid * EPT + lane``: columns ``2c`` and ``2c + 1`` are consecutive
        # lane iterations of one thread.
        self._assert_rejected(interleaved_tiled, block_sizes=[64, 64])

    def test_halves_synthetic_lanes_rejected(self) -> None:
        # The rope layout on a non-load tile (rope itself takes the load fold).
        self._assert_rejected(halves_full, block_size=64)

    def test_partner_past_partial_tile_reads_zero(self) -> None:
        # A loaded tile is split by re-reading each thread's partner element;
        # past the edge of a partial tile the partner is the zero a masked
        # tile load holds there, not the next row's memory.
        x = torch.randn([128, 48], device=DEVICE)
        code, result = code_and_output(halves_loaded, (x,), block_sizes=[32, 32])
        self.assertNotIn("split_smem", code)
        expected = torch.cat(
            (x[:, 16:32], x[:, :16], torch.zeros_like(x[:, 32:48])), dim=1
        )
        torch.testing.assert_close(result, expected)

    def test_unowned_pair_dim_rejected(self) -> None:
        # Selecting data by the coordinate of a dim no thread or lane owns
        # would keep one operand for both pair elements; it must fail loudly.
        y = torch.randn_like(self.x)
        with self.assertRaisesRegex(
            exc.BackendUnsupported, "has no thread or lane owner"
        ):
            code_and_output(join_then_split, (self.x, y), block_sizes=[32, 32])

    def test_shuffled_permute_is_not_re_read(self) -> None:
        # A permute materialized through shared memory holds one element per
        # thread; the load fold cannot re-read it at the partner's coordinate,
        # so the split takes the verified exchange (rejected for this layout)
        # instead of silently returning the thread's own element twice.
        with self.assertRaisesRegex(
            exc.BackendUnsupported,
            "hl.split of a non-load tile: a pair element is produced by a "
            "different iteration",
        ):
            code_and_output(transposed_pairs, (self.x,), block_sizes=[64, 32])

    def test_exchange_over_smem_budget_rejected(self) -> None:
        # A 64 x 256 fp32 tile needs 64 KiB of static shared memory.
        x = torch.randn([128, 256], device=DEVICE)
        with self.assertRaisesRegex(
            exc.BackendUnsupported, r"needs 65536 bytes of shared memory"
        ):
            code_and_output(interleaved_tiled, (x,), block_sizes=[64, 256])


if __name__ == "__main__":
    unittest.main()
