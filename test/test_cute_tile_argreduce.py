"""CuTe argmax/argmin over a tile dim combine the tile's own threads and lanes.

``out[tm, tn.id] = x[tm, tn].argmax(1)`` reduces the ``tn`` tile, which sits
on a thread axis above the ``tm`` rows (or spans more than one warp), so
consecutive lanes belong to different rows.  The value is reduced with the
grouped (row-strided) reductions and then the lowest index holding it, and
across a lane loop as two dependent lane reductions, like the persistent
strategy's argmax.  NaN wins (the first one) and ties go to the first index,
as in torch.  The index is the position in the tile, as torch returns it for
the tile tensor; ``+ tile.begin`` makes it global.
"""

from __future__ import annotations

import torch

import helion
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
import helion.language as hl


@helion.kernel(backend="cute", static_shapes=True)
def _tile_argmax(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    for tile_m, tile_n in hl.tile([x.size(0), x.size(1)]):
        out[tile_m, tile_n.id] = x[tile_m, tile_n].argmax(1)
    return out


@helion.kernel(backend="cute", static_shapes=True)
def _tile_argmin(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    for tile_m, tile_n in hl.tile([x.size(0), x.size(1)]):
        out[tile_m, tile_n.id] = x[tile_m, tile_n].argmin(1)
    return out


@helion.kernel(backend="cute", static_shapes=True)
def _nested_argmax(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    for tile_m in hl.tile(x.size(0)):
        for tile_n in hl.tile(x.size(1)):
            out[tile_m, tile_n.id] = x[tile_m, tile_n].argmax(1)
    return out


@helion.kernel(backend="cute", static_shapes=True)
def _tile_argmax_rows(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    for tile_m, tile_n in hl.tile([x.size(0), x.size(1)]):
        out[tile_m.id, tile_n] = x[tile_m, tile_n].argmax(0)
    return out


def _values(
    shape: tuple[int, int], dtype: torch.dtype, *, ties: bool = False
) -> torch.Tensor:
    torch.manual_seed(0)
    if ties:
        values = torch.randint(0, 4, shape)
    elif dtype.is_floating_point:
        values = torch.randn(shape)
    elif dtype is torch.uint8:
        values = torch.randint(0, 256, shape)
    else:
        values = torch.randint(-(2**31), 2**31 - 1, shape)
    return values.to(dtype)


def _reference(x: torch.Tensor, block: int, *, argmin: bool) -> torch.Tensor:
    """Per-tile argmax/argmin along dim 1 as positions in the tile (``N`` a
    multiple of ``block`` or padded so padding never wins)."""
    m, n = x.shape
    tiles = (n + block - 1) // block
    wide = x.cpu().double()
    pad = torch.full((m, tiles * block - n), float("inf") if argmin else -float("inf"))
    grouped = torch.cat([wide, pad.double()], 1).view(m, tiles, block)
    return grouped.argmin(2) if argmin else grouped.argmax(2)


# (shape, block_sizes): rows below the reduced tile on one warp, a 64-wide and
# a 128-wide tile (cross-warp), one row per CTA, a ragged last tile.
_LAYOUTS = (
    ((16, 64), [4, 32]),
    ((16, 64), [2, 64]),
    ((32, 128), [1, 128]),
    ((16, 64), [8, 16]),
    ((16, 96), [4, 64]),
)


@onlyBackends(["cute"])
class TestCuteTileArgreduce(TestCase):
    def _check(
        self,
        kernel: object,
        x: torch.Tensor,
        config: dict[str, object],
        *,
        argmin: bool = False,
    ) -> None:
        block = config["block_sizes"][1]  # pyrefly: ignore [bad-index]
        tiles = (x.size(1) + block - 1) // block
        out = torch.zeros(x.size(0), tiles, dtype=torch.int64, device=DEVICE)
        _code, result = code_and_output(kernel, (x.to(DEVICE), out), **config)
        torch.testing.assert_close(
            result.cpu(), _reference(x, block, argmin=argmin), atol=0, rtol=0
        )

    def test_float_layouts(self) -> None:
        for dtype in (torch.float32, torch.bfloat16):
            for shape, block_sizes in _LAYOUTS:
                for kernel, argmin in ((_tile_argmax, False), (_tile_argmin, True)):
                    with self.subTest(
                        dtype=dtype, shape=shape, block_sizes=block_sizes, argmin=argmin
                    ):
                        self._check(
                            kernel,
                            _values(shape, dtype),
                            {"block_sizes": block_sizes},
                            argmin=argmin,
                        )

    def test_integer_layouts_and_ties(self) -> None:
        for dtype in (torch.int32, torch.uint8):
            for shape, block_sizes in _LAYOUTS[:3]:
                for ties in (False, True):
                    with self.subTest(
                        dtype=dtype, shape=shape, block_sizes=block_sizes, ties=ties
                    ):
                        self._check(
                            _tile_argmax,
                            _values(shape, dtype, ties=ties),
                            {"block_sizes": block_sizes},
                        )

    def test_first_nan_wins(self) -> None:
        x = _values((16, 128), torch.float32)
        x[torch.rand(x.shape, generator=torch.Generator().manual_seed(1)) < 0.05] = (
            float("nan")
        )
        for block_sizes in ([4, 32], [1, 128]):
            for kernel, argmin in ((_tile_argmax, False), (_tile_argmin, True)):
                with self.subTest(block_sizes=block_sizes, argmin=argmin):
                    self._check(kernel, x, {"block_sizes": block_sizes}, argmin=argmin)

    def test_lane_looped_tiles(self) -> None:
        x = _values((32, 128), torch.float32, ties=True)
        for kernel in (_tile_argmax, _nested_argmax):
            for config in (
                {"block_sizes": [4, 64], "num_threads": [0, 16]},
                {"block_sizes": [8, 32], "num_threads": [4, 0]},
            ):
                with self.subTest(kernel=kernel.name, config=config):
                    self._check(kernel, x, config)

    def test_reduction_over_the_row_tile(self) -> None:
        x = _values((32, 64), torch.float32)
        for block_sizes in ([8, 16], [4, 32]):
            with self.subTest(block_sizes=block_sizes):
                rows = block_sizes[0]
                out = torch.zeros(32 // rows, 64, dtype=torch.int64, device=DEVICE)
                _code, result = code_and_output(
                    _tile_argmax_rows, (x.to(DEVICE), out), block_sizes=block_sizes
                )
                expected = x.view(32 // rows, rows, 64).argmax(1)
                torch.testing.assert_close(result.cpu(), expected, atol=0, rtol=0)

    def test_vector_lane_loop_is_refused(self) -> None:
        x = _values((16, 128), torch.float32)
        out = torch.zeros(16, 2, dtype=torch.int64, device=DEVICE)
        with self.assertRaisesRegex(helion.exc.BackendUnsupported, "lane loop"):
            code_and_output(
                _nested_argmax,
                (x.to(DEVICE), out),
                block_sizes=[4, 64],
                num_threads=[0, 16],
                cute_vector_widths=[1, 4],
            )
