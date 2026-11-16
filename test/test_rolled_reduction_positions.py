"""Positions along a reduction dim inside a rolled (looped) reduction.

``torch.tril(x[tile, :])`` compares a column iota over the row's reduction dim
with a row iota.  With ``reduction_loops`` the roller moves the iota into the
chunked loop, where it holds the chunk's global positions (``roffset +
local``): not the chunk-local coordinate (``rindex - roffset``), which would
restart the triangle at every chunk, nor the full-extent ``tl.arange``, which
does not match the chunk's shape.  The same holds for
``hl.arange``/``torch.arange`` over the reduction dim compared with another
index.  ``hl.arange(n)`` of the host int ``n`` (not ``row.size(1)``) stays a
full-extent range outside the loop: it shares the row's reduction dim (under
dynamic shapes it keeps a dim of its own, which the ops between them tie to
the row's), and the row's dim then offers no reduction loop.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import helion
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import _get_backend
from helion._testing import code_and_output
from helion._testing import onlyBackends
from helion._testing import skipIfNotTriton
from helion._testing import skipIfRefEager
from helion._testing import skipIfTileIR
from helion._testing.padded_range_kernels import grid_pick_column
from helion._testing.padded_range_kernels import host_arange_mark
from helion._testing.padded_range_kernels import host_arange_mark_then_row_sums
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Callable


@helion.kernel(static_shapes=True)
def _tril(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile_m in hl.tile(x.size(0)):
        out[tile_m, :] = torch.tril(x[tile_m, :], 1)
    return out


@helion.kernel(static_shapes=True)
def _triu(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile_m in hl.tile(x.size(0)):
        out[tile_m, :] = torch.triu(x[tile_m, :], -2)
    return out


@helion.kernel(static_shapes=True)
def _tril_beside_sum(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.empty_like(x)
    sums = torch.empty([x.size(0)], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(x.size(0)):
        row = x[tile_m, :]
        out[tile_m, :] = torch.tril(row)
        sums[tile_m] = row.sum(-1)
    return out, sums


@helion.kernel(static_shapes=True)
def _tril_transposed(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty([x.size(1), x.size(0)], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(x.size(0)):
        out[:, tile_m] = torch.tril(x[tile_m, :]).T
    return out


@helion.kernel(static_shapes=True)
def _arange_below_row(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile_m in hl.tile(x.size(0)):
        row = x[tile_m, :]
        cols = hl.arange(row.size(1))
        out[tile_m, :] = torch.where(cols[None, :] <= tile_m.index[:, None], row, 0.0)
    return out


@helion.kernel(static_shapes=True)
def _arange_scaled(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile_m in hl.tile(x.size(0)):
        row = x[tile_m, :]
        cols = hl.arange(row.size(1)) * 2 + 1
        out[tile_m, :] = row * cols[None, :].to(row.dtype)
    return out


@helion.kernel(static_shapes=True)
def _arange_at_position(x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile_m in hl.tile(x.size(0)):
        row = x[tile_m, :]
        cols = torch.arange(row.size(1), device=row.device)
        out[tile_m, :] = torch.where(cols[None, :] == pos[tile_m][:, None], row, -row)
    return out


@helion.kernel(static_shapes=True)
def _host_arange_sum(
    x: torch.Tensor, pos: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    m, n = x.shape
    out = torch.empty_like(x)
    sums = torch.empty([m], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        # The range comes before the row's full slice of its extent.  The
        # misses are not zero, so the sum is no one-hot row CuTe folds into
        # closed form (``fold_one_hot_reductions``): it keeps its reduction.
        hit = torch.where(hl.arange(n)[None, :] == pos[tile_m][:, None], -1.0, 0.5)
        sums[tile_m] = hit.sum(-1)
        out[tile_m, :] = x[tile_m, :] + hit
    return out, sums


@helion.kernel(static_shapes=True)
def _host_arange_count(x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    counts = torch.empty([m], dtype=torch.int32, device=x.device)
    for tile_m in hl.tile(m):
        # No full slice: the range alone gives the reduced dim its extent.
        above = hl.arange(n)[None, :] >= pos[tile_m][:, None]
        counts[tile_m] = above.to(torch.int32).sum(-1)
    return counts


_host_arange_mark = helion.kernel(static_shapes=True)(host_arange_mark)
_host_arange_mark_then_row_sums = helion.kernel(static_shapes=True)(
    host_arange_mark_then_row_sums
)
_grid_pick_column = helion.kernel(static_shapes=True)(grid_pick_column)


@helion.kernel(static_shapes=False)
def _host_arange_mark_dynamic(
    x: torch.Tensor, pos: torch.Tensor, n: hl.constexpr
) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile_m in hl.tile(x.size(0)):
        # The row's dim is dynamic: the range's literal dim of 200 does not
        # join it, but the torch.where between them ties their blocks.
        row = x[tile_m, :]
        out[tile_m, :] = torch.where(
            hl.arange(n)[None, :] == pos[tile_m][:, None], -1.0, row
        )
    return out


@helion.kernel(static_shapes=False)
def _factory_beside_dynamic_row(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile_m in hl.tile(x.size(0)):
        # The same with a padded factory dim of 200.
        out[tile_m, :] = x[tile_m, :] + hl.full([tile_m, 200], 1.0, dtype=x.dtype)
    return out


@helion.kernel(static_shapes=True)
def _host_arange_rotate(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        # The range only indexes a load: no subscript of it names the row's dim.
        out[tile_m, :] = x[tile_m, (hl.arange(n) + 1) % n]
    return out


@helion.kernel(static_shapes=True)
def _host_arange_scatter(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        # The range only indexes a store.
        out[tile_m, (hl.arange(n) + 1) % n] = x[tile_m, :]
    return out


@helion.kernel(static_shapes=True)
def _gather_rows(
    x: torch.Tensor, idx: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.empty_like(x)
    sums = torch.empty([x.size(0)], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(x.size(0)):
        picked = torch.gather(x[tile_m, :], 1, idx[tile_m, :])
        out[tile_m, :] = picked
        sums[tile_m] = picked.sum(-1)
    return out, sums


def _per_tile(
    fn: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor, block: int
) -> torch.Tensor:
    """``fn`` applied to each ``block``-row tile: the row iota is tile-local."""
    return torch.cat([fn(chunk) for chunk in x.split(block)])


# (shape, block_m, reduction chunk): several chunks per row, a non-power-of-2
# width, a ragged last chunk.
_LAYOUTS = (
    ((64, 64), 4, 32),
    ((64, 64), 8, 16),
    ((32, 96), 4, 32),
    ((16, 200), 2, 64),
)


@onlyBackends(["triton", "cute"])
class TestRolledReductionPositions(TestCase):
    def _configs(self, block_m: int, chunk: int) -> list[dict[str, object]]:
        config: dict[str, object] = {
            "block_sizes": [block_m],
            "reduction_loops": [chunk],
        }
        if _get_backend() != "cute":
            return [config]
        # Fewer threads than the chunk (a lane loop over it), and one thread
        # walking the rows.
        return [
            config,
            {**config, "num_threads": [0, 8]},
            {**config, "num_threads": [1, 0]},
        ]

    @skipIfRefEager(
        "the row iota of torch.tril on a tile is tile-local, so the result depends"
        " on block_sizes, which ref mode does not apply"
    )
    def test_tril_and_triu(self) -> None:
        for shape, block_m, chunk in _LAYOUTS:
            x = torch.randn(shape, device=DEVICE)
            for config in self._configs(block_m, chunk):
                with self.subTest(shape=shape, config=config):
                    _code, out = code_and_output(_tril, (x,), **config)
                    torch.testing.assert_close(
                        out, _per_tile(lambda t: torch.tril(t, 1), x, block_m)
                    )
                    _code, out = code_and_output(_triu, (x,), **config)
                    torch.testing.assert_close(
                        out, _per_tile(lambda t: torch.triu(t, -2), x, block_m)
                    )

    @skipIfRefEager(
        "the row iota of torch.tril on a tile is tile-local, so the result depends"
        " on block_sizes, which ref mode does not apply"
    )
    def test_tril_beside_a_rolled_sum(self) -> None:
        x = torch.randn([32, 96], device=DEVICE)
        for config in self._configs(4, 32):
            with self.subTest(config=config):
                _code, (out, sums) = code_and_output(_tril_beside_sum, (x,), **config)
                torch.testing.assert_close(out, _per_tile(torch.tril, x, 4))
                torch.testing.assert_close(sums, x.sum(-1), rtol=1e-4, atol=1e-4)

    @skipIfRefEager(
        "the row iota of torch.tril on a tile is tile-local, so the result depends"
        " on block_sizes, which ref mode does not apply"
    )
    def test_tril_transposed_store(self) -> None:
        x = torch.randn([64, 64], device=DEVICE)
        for config in self._configs(8, 16):
            with self.subTest(config=config):
                _code, out = code_and_output(_tril_transposed, (x,), **config)
                torch.testing.assert_close(out, _per_tile(torch.tril, x, 8).T)

    def test_arange_over_the_reduction_dim(self) -> None:
        for shape, block_m, chunk in _LAYOUTS:
            x = torch.randn(shape, device=DEVICE)
            rows = torch.arange(shape[0], device=DEVICE)[:, None]
            cols = torch.arange(shape[1], device=DEVICE)[None, :]
            pos = torch.randint(0, shape[1], (shape[0],), device=DEVICE)
            for config in self._configs(block_m, chunk):
                with self.subTest(shape=shape, config=config):
                    _code, out = code_and_output(_arange_below_row, (x,), **config)
                    torch.testing.assert_close(out, torch.where(cols <= rows, x, 0.0))
                    _code, out = code_and_output(_arange_scaled, (x,), **config)
                    torch.testing.assert_close(out, x * (cols * 2 + 1).float())
                    _code, out = code_and_output(
                        _arange_at_position, (x, pos), **config
                    )
                    torch.testing.assert_close(
                        out, torch.where(cols == pos[:, None], x, -x)
                    )

    @skipIfRefEager("checks the reduction dims and the offered reduction loops")
    def test_host_int_arange_shares_the_row_dim_and_is_not_rolled(self) -> None:
        # The range of 200 is emitted over 256 lanes; its padding lanes stay
        # out of the sums and counts (position 203 matches no column).  A
        # chunked loop would read the full-extent range as the chunk's
        # positions, so the row's dim offers no reduction loop, also when
        # only a pointwise op or a load's or store's index reads the range:
        # its slot, which older versions offered, stays pinned to None.
        x = torch.randn([16, 200], device=DEVICE)
        pos = torch.randint(0, 200, (16,), device=DEVICE)
        pos[0] = 203
        cols = torch.arange(200, device=DEVICE)[None, :]
        hit = torch.where(cols == pos[:, None], -1.0, 0.5)
        for kernel, args, expected in (
            (_host_arange_sum, (x, pos), (x + hit, hit.sum(-1))),
            (
                _host_arange_count,
                (x, pos),
                (cols >= pos[:, None]).sum(-1, dtype=torch.int32),
            ),
            (_host_arange_mark, (x, pos), torch.where(cols == pos[:, None], -1.0, x)),
            (_host_arange_rotate, (x,), x.roll(-1, dims=1)),
            (_host_arange_scatter, (x,), x.roll(1, dims=1)),
        ):
            with self.subTest(kernel=kernel.name):
                bound = kernel.bind(args)
                rdims = [info for info in bound.env.block_sizes if info.reduction]
                self.assertEqual(len(rdims), 1)
                self.assertEqual(
                    [
                        (slot.size_hint, slot.pinned)
                        for slot in bound.config_spec.reduction_loops
                    ],
                    [(200, True)],
                )
                _code, out = code_and_output(kernel, args, block_sizes=[2])
                torch.testing.assert_close(out, expected)

    @skipIfRefEager("normalizes a config saved by an older version")
    def test_saved_config_with_the_row_dims_reduction_loop_loads(self) -> None:
        # The row's dim used to offer a reduction loop beside the range, and
        # only its persistent choices (None, or a chunk of at least 200)
        # compiled on Triton.  A chunk size for the dim saved then, which
        # never compiled, keeps the reduction persistent (test_config_api.py
        # checks how the saved entries load).
        x = torch.randn([16, 200], device=DEVICE)
        pos = torch.randint(0, 200, (16,), device=DEVICE)
        _code, out = code_and_output(
            _host_arange_mark, (x, pos), block_sizes=[2], reduction_loops=[64]
        )
        cols = torch.arange(200, device=DEVICE)[None, :]
        torch.testing.assert_close(out, torch.where(cols == pos[:, None], -1.0, x))

    @skipIfNotTriton("Triton configs are the saved configs that must keep working")
    @skipIfRefEager("normalizes a config saved by an older version")
    def test_saved_config_keeps_each_reduction_loop_on_its_dim(self) -> None:
        # Configs older versions saved for this kernel have an entry for the
        # row beside the range, then for the row sums of y and z.  The first
        # slot is pinned to None now, and the 64 still chunks the sum over y,
        # too wide for a persistent reduction.
        x = torch.randn([16, 200], device=DEVICE)
        pos = torch.randint(0, 200, (16,), device=DEVICE)
        y = torch.randn([2, 2097152], device=DEVICE)
        z = torch.randn([4, 1024], device=DEVICE)
        _code, (out, y_sums, z_sums) = code_and_output(
            _host_arange_mark_then_row_sums,
            (x, pos, y, z),
            block_sizes=[2, 1, 2],
            reduction_loops=[None, 64, None],
        )
        cols = torch.arange(200, device=DEVICE)[None, :]
        torch.testing.assert_close(out, torch.where(cols == pos[:, None], -1.0, x))
        torch.testing.assert_close(y_sums, y.sum(-1), rtol=1e-4, atol=1e-2)
        torch.testing.assert_close(z_sums, z.sum(-1), rtol=1e-4, atol=1e-4)

    def test_grid_load_beside_the_range_subscript(self) -> None:
        # The load over the range shares the subscript's reduction dim of 200,
        # also when typed first, so its mask reaches the load and the sum,
        # which stays correct with a position (203) in the padding lanes
        # (test_config_api.py checks the dim and the mask on CPU).  Typed with
        # the padded 256 beside the subscript's dim, the load made CuTe refuse
        # the sum; with both padded, CuTe summed wrong values.
        x = torch.randn([16, 200], device=DEVICE)
        pos = torch.randint(0, 200, (16,), device=DEVICE)
        pos[0] = 203
        _code, out = code_and_output(_grid_pick_column, (x, pos))
        cols = torch.arange(200, device=DEVICE)[None, :]
        expected = torch.where(cols == pos[:, None], x, 0.0).sum(-1)
        torch.testing.assert_close(out, expected)

    @skipIfRefEager("checks the offered reduction loops")
    def test_literal_dim_tied_to_a_dynamic_row_is_not_rolled(self) -> None:
        # Rolling the row's dim would chunk the row but not the range or the
        # factory tied to it by the op between them.
        x = torch.randn([16, 200], device=DEVICE)
        pos = torch.randint(0, 200, (16,), device=DEVICE)
        pos[0] = 203
        cols = torch.arange(200, device=DEVICE)[None, :]
        for kernel, args, expected, slots in (
            (
                _host_arange_mark_dynamic,
                (x, pos, 200),
                torch.where(cols == pos[:, None], -1.0, x),
                [],
            ),
            # Older versions offered the row's dim a loop.
            (_factory_beside_dynamic_row, (x,), x + 1, [(200, True)]),
        ):
            with self.subTest(kernel=kernel.name):
                bound = kernel.bind(args)
                self.assertEqual(
                    [
                        (slot.size_hint, slot.pinned)
                        for slot in bound.config_spec.reduction_loops
                    ],
                    slots,
                )
                if _get_backend() == "cute":
                    # CuTe has no exchange between the two persistent dims.
                    with self.assertRaises(helion.exc.BackendUnsupported):
                        code_and_output(kernel, args, block_sizes=[2])
                    continue
                _code, out = code_and_output(kernel, args, block_sizes=[2])
                torch.testing.assert_close(out, expected)

    @skipIfTileIR("TileIR does not support gather operation")
    def test_gather_along_the_reduction_dim_is_not_rolled(self) -> None:
        # A gather reads its row at any position of the reduction dim, which
        # one chunk per loop iteration does not hold.
        x = torch.randn([32, 64], device=DEVICE)
        idx = torch.randint(0, 64, (32, 64), device=DEVICE)
        bound = _gather_rows.bind((x, idx))
        self.assertEqual(bound.env.config_spec.reduction_loops.valid_block_ids(), [])
        _code, (out, sums) = code_and_output(_gather_rows, (x, idx), block_sizes=[4])
        expected = torch.gather(x, 1, idx)
        torch.testing.assert_close(out, expected)
        torch.testing.assert_close(sums, expected.sum(-1), rtol=1e-4, atol=1e-4)
