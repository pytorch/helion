"""Positions along a reduction dim inside a rolled (looped) reduction.

``torch.tril(x[tile, :])`` compares a column iota over the row's reduction dim
with a row iota.  With ``reduction_loops`` the roller moves the iota into the
chunked loop, where it holds the chunk's global positions (``roffset +
local``): not the chunk-local coordinate (``rindex - roffset``), which would
restart the triangle at every chunk, nor the full-extent ``tl.arange``, which
does not match the chunk's shape.  The same holds for
``hl.arange``/``torch.arange`` over the reduction dim compared with another
index.
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
from helion._testing import skipIfRefEager
from helion._testing import skipIfTileIR
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
