"""Shared-memory ``hl.split`` exchange under CuTe lane loops.

``hl.split`` of a non-load tile stages the tile in shared memory inside the
current lane iteration.  The exchange is only correct when both pair elements
are staged in the same lane iteration that reads them; the layouts below pin
down which combinations of lane layout and pair geometry that admits.

A permute in the split's producer chain is a per-thread relabel (every block
id owns one coordinate per thread), so the thread still holds the element at
its block coordinates and both the load fold and the exchange see the pair
the split-view coordinates describe.  The old position-keyed permute shuffle
broke that invariant and made every accepted layout of ``transposed_pairs``
below return the wrong pairs; those kernels now pin the numerics.
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
def halves_loaded_through_alias(x: torch.Tensor) -> torch.Tensor:
    """``halves_loaded`` read through a host view of ``x`` that the kernel
    then overwrites: the split may not re-read the zeros."""
    y = x.view_as(x)
    n, d = x.size()
    out = torch.empty_like(x)
    for tile_n, tile_d in hl.tile([n, d]):
        pair = y[tile_n, tile_d].reshape([tile_n, 2, tile_d.block_size // 2])
        x[tile_n, tile_d] = hl.zeros([tile_n, tile_d], dtype=x.dtype)
        lo, hi = hl.split(pair.permute(0, 2, 1))
        out[tile_n, tile_d] = hl.join(hi, lo).permute(0, 2, 1).reshape([tile_n, tile_d])
    return out


@helion.kernel(static_shapes=True)
def interleaved_loaded(x: torch.Tensor) -> torch.Tensor:
    """Adjacent pairs of a loaded tile, swapped (re-read, no exchange)."""
    n, d = x.size()
    out = torch.empty_like(x)
    for tile_n, tile_d in hl.tile([n, d]):
        pair = x[tile_n, tile_d].reshape([tile_n, tile_d.block_size // 2, 2])
        lo, hi = hl.split(pair)
        out[tile_n, tile_d] = hl.join(hi, lo).reshape([tile_n, tile_d])
    return out


@helion.kernel(static_shapes=True)
def halves_loaded_through_nested_alias(x: torch.Tensor) -> torch.Tensor:
    """As ``halves_loaded_through_alias``, with the alias the result of an op
    the host alias analysis does not know, bound in a nested ``if``."""
    y = x.clone()
    if x.size(0) > 1:
        y = torch.ops.aten.alias.default(x)
    n, d = x.size()
    out = torch.empty_like(x)
    for tile_n, tile_d in hl.tile([n, d]):
        pair = y[tile_n, tile_d].reshape([tile_n, 2, tile_d.block_size // 2])
        x[tile_n, tile_d] = hl.zeros([tile_n, tile_d], dtype=x.dtype)
        lo, hi = hl.split(pair.permute(0, 2, 1))
        out[tile_n, tile_d] = hl.join(hi, lo).permute(0, 2, 1).reshape([tile_n, tile_d])
    return out


@helion.kernel(static_shapes=True)
def trailing_pairs(x: torch.Tensor) -> torch.Tensor:
    """A trailing size-two dim of a loaded 3-d tile, swapped."""
    m, n, _ = x.size()
    out = torch.empty_like(x)
    for tile_m, tile_n in hl.tile([m, n]):
        lo, hi = hl.split(x[tile_m, tile_n, :])
        out[tile_m, tile_n, :] = hl.join(hi, lo)
    return out


@helion.kernel(static_shapes=True)
def row_pairs_tiled(x: torch.Tensor) -> torch.Tensor:
    """Adjacent rows of a tile, swapped (the pair dim is the row block's)."""
    n, d = x.size()
    out = torch.empty_like(x)
    for tile_n, tile_d in hl.tile([n, d]):
        pair = (
            (x[tile_n, tile_d] * 2.0)
            .reshape([tile_n.block_size // 2, 2, tile_d])
            .permute(0, 2, 1)
        )
        lo, hi = hl.split(pair)
        out[tile_n, tile_d] = hl.join(hi, lo).permute(0, 2, 1).reshape([tile_n, tile_d])
    return out


@helion.kernel(static_shapes=True)
def interleaved_3d(x: torch.Tensor) -> torch.Tensor:
    """Adjacent pairs of the last dim of a 3-d tile, swapped."""
    a, b, c = x.size()
    out = torch.empty_like(x)
    for tile_a, tile_b, tile_c in hl.tile([a, b, c]):
        pair = (x[tile_a, tile_b, tile_c] * 2.0).reshape(
            [tile_a, tile_b, tile_c.block_size // 2, 2]
        )
        lo, hi = hl.split(pair)
        out[tile_a, tile_b, tile_c] = hl.join(hi, lo).reshape([tile_a, tile_b, tile_c])
    return out


@helion.kernel(static_shapes=True)
def pair_slices(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The pair halves of a loaded tile, each stored to a full slice."""
    n, d = x.size()
    lo_out = x.new_empty([n, d // 2])
    hi_out = x.new_empty([n, d // 2])
    for tile_n, tile_d in hl.tile([n, d]):
        pair = x[tile_n, tile_d].reshape([tile_n, tile_d.block_size // 2, 2])
        lo, hi = hl.split(pair)
        lo_out[tile_n, :] = lo
        hi_out[tile_n, :] = hi
    return lo_out, hi_out


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
    """Adjacent pairs of a transposed loaded tile, swapped (re-read fold)."""
    n, d = x.size()
    out = torch.empty([d, n], dtype=x.dtype, device=x.device)
    for tile_n, tile_d in hl.tile([n, d]):
        pair = (
            x[tile_n, tile_d].permute(1, 0).reshape([tile_d, tile_n.block_size // 2, 2])
        )
        lo, hi = hl.split(pair)
        out[tile_d, tile_n] = hl.join(hi, lo).reshape([tile_d, tile_n])
    return out


@helion.kernel(static_shapes=True)
def transposed_pairs_scaled(x: torch.Tensor) -> torch.Tensor:
    """``transposed_pairs`` on a non-load tile (``* 2.0`` after the permute)."""
    n, d = x.size()
    out = torch.empty([d, n], dtype=x.dtype, device=x.device)
    for tile_n, tile_d in hl.tile([n, d]):
        pair = (x[tile_n, tile_d].permute(1, 0) * 2.0).reshape(
            [tile_d, tile_n.block_size // 2, 2]
        )
        lo, hi = hl.split(pair)
        out[tile_d, tile_n] = hl.join(hi, lo).reshape([tile_d, tile_n])
    return out


@helion.kernel(static_shapes=True)
def hoisted_transposed_pairs(x: torch.Tensor, steps: torch.Tensor) -> torch.Tensor:
    """The permute is hoisted out of the loop that splits it (exchange)."""
    n, d = x.size()
    out = torch.empty([d, n], dtype=x.dtype, device=x.device)
    for tile_n, tile_d in hl.tile([n, d]):
        transposed = x[tile_n, tile_d].permute(1, 0)
        acc = hl.zeros([tile_d, tile_n], dtype=x.dtype)
        for _tile_s in hl.tile(steps.size(0)):
            pair = transposed.reshape([tile_d, tile_n.block_size // 2, 2])
            lo, hi = hl.split(pair)
            acc = acc + hl.join(hi, lo).reshape([tile_d, tile_n])
        out[tile_d, tile_n] = acc
    return out


@helion.kernel(static_shapes=True)
def joined_pairs_transposed(x: torch.Tensor) -> torch.Tensor:
    """``hl.join`` of swapped pairs, stored through a permute."""
    n, d = x.size()
    out = torch.empty([d, n], dtype=x.dtype, device=x.device)
    for tile_n, tile_d in hl.tile([n, d]):
        pair = (x[tile_n, tile_d] * 2.0).reshape([tile_n, tile_d.block_size // 2, 2])
        lo, hi = hl.split(pair)
        out[tile_d, tile_n] = hl.join(hi, lo).reshape([tile_n, tile_d]).permute(1, 0)
    return out


@helion.kernel(static_shapes=True)
def joined_pairs_times(x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """The reshape undoing a pair split feeds a pointwise op with a loaded tile."""
    n, d = x.size()
    out = torch.empty_like(x)
    for tile_n, tile_d in hl.tile([n, d]):
        pair = x[tile_n, tile_d].reshape([tile_n, tile_d.block_size // 2, 2])
        lo, hi = hl.split(pair)
        joined = hl.join(hi, lo).reshape([tile_n, tile_d])
        out[tile_n, tile_d] = joined * z[tile_n, tile_d]
    return out


@helion.kernel(static_shapes=True)
def unit_permuted_pairs(x: torch.Tensor) -> torch.Tensor:
    """A permute that only moves a unit dim, over a non-load tile."""
    n, d = x.size()
    out = torch.empty_like(x)
    for tile_n, tile_d in hl.tile([n, d]):
        rows = (x[tile_n, tile_d] * 2.0).unsqueeze(0).permute(1, 0, 2)
        pair = rows.reshape([tile_n, 1, tile_d.block_size // 2, 2])
        lo, hi = hl.split(pair)
        out[tile_n, tile_d] = hl.join(hi, lo).reshape([tile_n, 1, tile_d]).squeeze(1)
    return out


def _swapped_halves(x: torch.Tensor, block: int, scale: float = 2.0) -> torch.Tensor:
    pairs = (x * scale).view(x.shape[0], x.shape[1] // block, 2, block // 2)
    return torch.cat((pairs[:, :, 1:], pairs[:, :, :1]), dim=2).view(x.shape)


def _swapped_interleaved(x: torch.Tensor, scale: float = 2.0) -> torch.Tensor:
    pairs = (x * scale).view(x.shape[0], x.shape[1] // 2, 2)
    return torch.stack((pairs[..., 1], pairs[..., 0]), dim=-1).view(x.shape)


def _swapped_transposed_pairs(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    return _swapped_interleaved(x.t().contiguous(), scale)


# Flattened [n, d] tiles: whole rows, half rows, runs crossing rows (48 and
# 40 wide rows), a partial last tile, and lane loops of either layout.
_FLATTENED_LAYOUTS = (
    {"block_sizes": [2, 64]},
    {"block_sizes": [4, 32]},
    {"block_sizes": [4, 32], "num_threads": [1, 32]},
    {
        "block_sizes": [4, 32],
        "num_threads": [1, 32],
        "cute_lane_layouts": ["strided", "strided"],
    },
)

# ``transposed_pairs`` layouts without lane loops.  The old position-keyed
# permute shuffle returned the wrong pairs for every one of them (and the
# exchange proof rejected [64, 32]); the permute is a per-thread relabel now.
_TRANSPOSED_PAIR_LAYOUTS = ([32, 32], [16, 32], [32, 16], [16, 16], [8, 32])


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

    def _assert_flattened_rejected(
        self, kernel: object, x: torch.Tensor, **config: object
    ) -> None:
        with self.assertRaisesRegex(
            exc.BackendUnsupported, "keeps that block's coordinate only modulo"
        ):
            code_and_output(kernel, (x,), flatten_loops=[True], **config)

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

    def test_loaded_alias_written_later_exchanges(self) -> None:
        # The loaded y may share x's storage (``x.view_as(x)``, or an unknown
        # op's result bound in a nested ``if``), so the store to x makes the
        # load of y written: the split must exchange the loaded values rather
        # than re-read y after the store.
        for kernel in (halves_loaded_through_alias, halves_loaded_through_nested_alias):
            with self.subTest(kernel=kernel.name):
                x = torch.randn([64, 32], device=DEVICE)
                expected = torch.cat(
                    (x[:, 8:16], x[:, :8], x[:, 24:32], x[:, 16:24]), 1
                )
                code, result = code_and_output(kernel, (x,), block_sizes=[8, 16])
                self.assertIn("split_smem", code)
                torch.testing.assert_close(result, expected)
                self.assertEqual(int(torch.count_nonzero(x)), 0)

    def test_unowned_pair_dim_rejected(self) -> None:
        # Selecting data by the coordinate of a dim no thread or lane owns
        # would keep one operand for both pair elements; it must fail loudly.
        y = torch.randn_like(self.x)
        with self.assertRaisesRegex(
            exc.BackendUnsupported, "has no thread or lane owner"
        ):
            code_and_output(join_then_split, (self.x, y), block_sizes=[32, 32])

    def test_transposed_pairs_re_read(self) -> None:
        # The permute relabels the loaded tile, so the fold re-reads each
        # partner at the inverse-permuted coordinate: no exchange, and the
        # lane-loop layout [64, 32] works too (its store stages the final
        # reshape through shared memory by block coordinates).
        expected = _swapped_transposed_pairs(self.x)
        for block_sizes in (*_TRANSPOSED_PAIR_LAYOUTS, [64, 32]):
            with self.subTest(block_sizes=block_sizes):
                code, result = code_and_output(
                    transposed_pairs, (self.x,), block_sizes=block_sizes
                )
                self.assertNotIn("split_smem", code)
                torch.testing.assert_close(result, expected)

    def test_transposed_pairs_exchange(self) -> None:
        # A pointwise op after the permute makes the tile non-load; the
        # exchange stages the thread's element at its block coordinates,
        # which the relabeled permute leaves in place.
        expected = _swapped_transposed_pairs(self.x, 2.0)
        for block_sizes in _TRANSPOSED_PAIR_LAYOUTS:
            with self.subTest(block_sizes=block_sizes):
                self._assert_exchange(
                    transposed_pairs_scaled, expected, block_sizes=block_sizes
                )

    def test_transposed_pairs_exchange_lane_loop_rejected(self) -> None:
        # Under lane loops the interleaved partner of the non-load tile is
        # the next lane iteration of the blocked layout, as for
        # ``interleaved_tiled``; the proof rejects it rather than the permute.
        self._assert_rejected(transposed_pairs_scaled, block_sizes=[64, 32])

    def test_hoisted_transposed_pairs(self) -> None:
        # The permute reaches the split through a loop-body placeholder; the
        # inner loop's exchange still sees the thread's own element.
        steps = torch.empty([4], device=DEVICE)
        expected = _swapped_transposed_pairs(self.x)
        for block_sizes in ([32, 32, 4], [16, 32, 4]):
            with self.subTest(block_sizes=block_sizes):
                code, result = code_and_output(
                    hoisted_transposed_pairs, (self.x, steps), block_sizes=block_sizes
                )
                self.assertIn("split_smem", code)
                self.assertNotIn("rebind_smem", code)
                torch.testing.assert_close(result, expected)

    def test_joined_pairs_stored_transposed(self) -> None:
        # ``hl.join`` selects by the split's minor coordinate; the permute on
        # the way to the store relabels the joined tile for the transposed
        # subscript.
        expected = _swapped_interleaved(self.x).t().contiguous()
        for block_sizes in ([32, 32], [16, 32], [64, 32]):
            with self.subTest(block_sizes=block_sizes):
                code, result = code_and_output(
                    joined_pairs_transposed, (self.x,), block_sizes=block_sizes
                )
                self.assertNotIn("rebind_smem", code)
                torch.testing.assert_close(result, expected)

    def test_joined_pairs_feed_pointwise(self) -> None:
        # The join's coordinates recompose the tile's own, so the merging
        # reshape keeps every element on its thread for any consumer.
        z = torch.randn_like(self.x)
        expected = _swapped_interleaved(self.x, 1.0) * z
        for block_sizes in ([32, 32], [16, 64], [64, 16]):
            with self.subTest(block_sizes=block_sizes):
                _, result = code_and_output(
                    joined_pairs_times, (self.x, z), block_sizes=block_sizes
                )
                torch.testing.assert_close(result, expected)

    def test_unit_permute_exchange(self) -> None:
        # Moving a unit dim reorders no thread dim either; the exchange of
        # the non-load tile stays exact.
        for block_sizes in ([32, 32], [64, 64]):
            with self.subTest(block_sizes=block_sizes):
                self._assert_exchange(
                    unit_permuted_pairs,
                    _swapped_interleaved(self.x),
                    block_sizes=block_sizes,
                    cute_lane_layouts=["blocked", "strided"],
                )

    def test_flattened_tiles_re_read(self) -> None:
        # A flattened tile has no per-block tile base: the fold moves the
        # flat position and decodes the partner's indices from it.  The run
        # crosses rows when the column block does not divide the row, but a
        # pair group that divides the row stays in it: adjacent pairs of the
        # 48 and 40 wide rows are the unflattened tile's.  Halves of a 64 or
        # 32 block there would pair across rows and are refused.
        for shape in ([128, 64], [130, 48], [65, 40]):
            x = torch.randn(shape, device=DEVICE)
            for layout in _FLATTENED_LAYOUTS:
                block = layout["block_sizes"][1]
                halves = None if shape[1] % block else _swapped_halves(x, block, 1.0)
                for kernel, expected in (
                    (interleaved_loaded, _swapped_interleaved(x, 1.0)),
                    (halves_loaded, halves),
                ):
                    with self.subTest(
                        shape=shape, kernel=kernel.fn.__name__, layout=layout
                    ):
                        if expected is None:
                            self._assert_flattened_rejected(kernel, x, **layout)
                            continue
                        code, result = code_and_output(
                            kernel, (x,), flatten_loops=[True], **layout
                        )
                        self.assertNotIn("split_smem", code)
                        torch.testing.assert_close(result, expected)

    def test_flattened_tiles_exchange(self) -> None:
        # The exchange keys a flattened tile by the same coordinates.  Under
        # blocked lanes the interleaved partner is the next lane iteration.
        for shape in ([130, 64], [130, 48]):
            x = torch.randn(shape, device=DEVICE)
            for layout in _FLATTENED_LAYOUTS:
                block = layout["block_sizes"][1]
                halves = None if shape[1] % block else _swapped_halves(x, block)
                for kernel, expected in (
                    (halves_tiled, halves),
                    (interleaved_tiled, _swapped_interleaved(x)),
                ):
                    with self.subTest(
                        shape=shape, kernel=kernel.fn.__name__, layout=layout
                    ):
                        if expected is None:
                            self._assert_flattened_rejected(kernel, x, **layout)
                            continue
                        if (
                            kernel is interleaved_tiled
                            and layout == _FLATTENED_LAYOUTS[2]
                        ):
                            self._assert_rejected(
                                kernel, flatten_loops=[True], **layout
                            )
                            continue
                        code, result = code_and_output(
                            kernel, (x,), flatten_loops=[True], **layout
                        )
                        self.assertIn("split_smem", code)
                        torch.testing.assert_close(result, expected)

    def test_flattened_tiles_follow_the_loop_order(self) -> None:
        # A flattened tile's coordinates are the digits of its run in
        # iteration order.  When the blocks iterated faster than the pair's
        # block span their dims, the run is a rectangle along it and a
        # reordered loop gives the same pairs as the unflattened tile.  Else
        # the digits count across rows: 2 x 4 elements of 64 rows iterated
        # rows fastest are one column, whose column pairs would be rows 0 and
        # 2.  That is refused.
        x = torch.randn([64, 32], device=DEVICE)
        for kernel in (
            halves_loaded,
            interleaved_loaded,
            halves_tiled,
            interleaved_tiled,
            row_pairs_tiled,
        ):
            for block_sizes, loop_order in (
                ([64, 8], [1, 0]),
                ([4, 32], [0, 1]),
                ([2, 4], [1, 0]),
                ([32, 8], [1, 0]),
                ([4, 16], [0, 1]),
            ):
                with self.subTest(
                    kernel=kernel.fn.__name__, block_sizes=block_sizes, order=loop_order
                ):
                    config = {"block_sizes": block_sizes, "loop_orders": [loop_order]}
                    fastest = loop_order[-1]
                    pair = 0 if kernel is row_pairs_tiled else 1
                    if fastest != pair and block_sizes[fastest] != x.size(fastest):
                        self._assert_flattened_rejected(kernel, x, **config)
                        continue
                    _code, expected = code_and_output(
                        kernel, (x,), flatten_loops=[False], **config
                    )
                    _code, result = code_and_output(
                        kernel, (x,), flatten_loops=[True], **config
                    )
                    torch.testing.assert_close(result, expected)

    def test_flattened_3d_loop_orders(self) -> None:
        # The blocks iterated faster than the pair's block must hold as many
        # elements as their dims, not each span its own: block sizes 8 and 4
        # over dims of 4 and 8 hold whole 32-element planes of the last dim.
        # Iterated fastest, the last dim's pairs need only divide its 6.
        x = torch.randn([4, 8, 6], device=DEVICE)
        expected = (x * 2.0).view(4, 8, 3, 2).flip(-1).view(x.shape)
        for loop_order, accepted in (
            ([0, 1, 2], True),
            ([0, 2, 1], False),
            ([1, 0, 2], True),
            ([1, 2, 0], False),
            ([2, 0, 1], True),
            ([2, 1, 0], True),
        ):
            with self.subTest(order=loop_order):
                config = {"block_sizes": [8, 4, 2], "loop_orders": [loop_order]}
                if not accepted:
                    self._assert_flattened_rejected(interleaved_3d, x, **config)
                    continue
                _code, result = code_and_output(
                    interleaved_3d, (x,), flatten_loops=[True], **config
                )
                torch.testing.assert_close(result, expected)

    def test_flattened_trailing_pairs(self) -> None:
        for shape in ([4, 8, 2], [5, 7, 2]):
            x = torch.randn(shape, device=DEVICE)
            for block_sizes in ([2, 8], [4, 4]):
                with self.subTest(shape=shape, block_sizes=block_sizes):
                    code, result = code_and_output(
                        trailing_pairs,
                        (x,),
                        block_sizes=block_sizes,
                        flatten_loops=[True],
                    )
                    self.assertNotIn("split_smem", code)
                    torch.testing.assert_close(result, x.flip(-1))

    def test_sliced_store_of_a_view_fills_the_slice(self) -> None:
        # A view dim stored to a full slice is addressed by its coordinate
        # from the slice's start, unmasked.  Only a view dim as wide as the
        # slice writes it once, so the others are refused: the 4 wide halves
        # of an 8 wide block would leave most of the 16 wide slice unwritten
        # and the 32 wide halves of a 64 wide block would write past it
        # (Triton fails the shape check).
        x = torch.randn([64, 32], device=DEVICE)
        _code, (lo, hi) = code_and_output(pair_slices, (x,), block_sizes=[2, 32])
        torch.testing.assert_close(lo, x[:, 0::2])
        torch.testing.assert_close(hi, x[:, 1::2])
        for block_sizes, extent in (([2, 8], 4), ([2, 64], 32)):
            with (
                self.subTest(block_sizes=block_sizes),
                self.assertRaisesRegex(
                    exc.BackendUnsupported,
                    rf"view dim of {extent} elements fills a slice of 16",
                ),
            ):
                code_and_output(pair_slices, (x,), block_sizes=block_sizes)

    def test_exchange_over_smem_budget_rejected(self) -> None:
        # A 64 x 256 fp32 tile needs 64 KiB of static shared memory.
        x = torch.randn([128, 256], device=DEVICE)
        with self.assertRaisesRegex(
            exc.BackendUnsupported, r"needs 65536 bytes of shared memory"
        ):
            code_and_output(interleaved_tiled, (x,), block_sizes=[64, 256])


if __name__ == "__main__":
    unittest.main()
