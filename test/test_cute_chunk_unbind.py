"""``torch.chunk`` / ``torch.unbind`` inside CuTe device loops.

Both lower through reshape, permute and ``hl.split``
(``view_ops._torch_chunk``).  A split of a loaded tile re-reads each half
from memory, so it holds under any thread layout; a computed tile needs both
halves staged in one lane iteration and is refused otherwise.  An unbind of a
``torch.stack`` resolves to the stacked operands.
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
def chunk_store(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    n, d = x.shape
    lo = x.new_empty([n, d // 2])
    hi = torch.empty_like(lo)
    for tile in hl.tile(n):
        a, b = torch.chunk(x[tile, :], 2, dim=-1)
        lo[tile, :] = a
        hi[tile, :] = b
    return lo, hi


@helion.kernel(static_shapes=True)
def chunk_glu(x: torch.Tensor) -> torch.Tensor:
    n, d = x.shape
    out = x.new_empty([n, d // 2])
    for tile in hl.tile(n):
        a, b = torch.chunk(x[tile, :], 2, dim=-1)
        out[tile, :] = torch.sigmoid(a) * b
    return out


@helion.kernel(static_shapes=True)
def chunk_glu_rows(x: torch.Tensor) -> torch.Tensor:
    m, n, d = x.shape
    out = x.new_empty([m, n, d // 2])
    for tile_m, tile_n in hl.tile([m, n]):
        a, b = x[tile_m, tile_n, :].chunk(2, dim=-1)
        out[tile_m, tile_n, :] = a * torch.sigmoid(b)
    return out


@helion.kernel(static_shapes=True)
def unbind_pairs(x: torch.Tensor) -> torch.Tensor:
    n, _ = x.shape
    out = x.new_empty([n])
    for tile in hl.tile(n):
        re, im = torch.unbind(x[tile, :], dim=-1)
        out[tile] = re * re + im * im
    return out


@helion.kernel(static_shapes=True)
def unbind_stacked(x: torch.Tensor) -> torch.Tensor:
    n, _ = x.shape
    out = torch.empty_like(x)
    for tile in hl.tile(n):
        p, q = torch.stack((x[tile, :] * 2.0, x[tile, :] + 1.0), dim=0).unbind(0)
        out[tile, :] = p * q
    return out


@helion.kernel(static_shapes=True)
def chunk_computed(x: torch.Tensor) -> torch.Tensor:
    n, d = x.shape
    out = x.new_empty([n, d // 2])
    for tile in hl.tile(n):
        a, b = torch.chunk(x[tile, :] * 2.0, 2, dim=-1)
        out[tile, :] = a * b
    return out


@helion.kernel(static_shapes=True)
def chunk_bias(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    n, d = x.shape
    out = x.new_empty([n, d // 2])
    for tile in hl.tile(n):
        a, b = torch.chunk(x[tile, :], 2, dim=-1)
        out[tile, :] = a * bias[tile, :] + b
    return out


# One element per thread, synthetic lanes, vectorized and strided lanes.
_ROW_LAYOUTS = (
    {"block_sizes": [32]},
    {"block_sizes": [1]},
    {"block_sizes": [4], "cute_vector_widths": [4, 4, 4]},
    {"block_sizes": [64], "cute_lane_layouts": ["strided", "strided", "strided"]},
)


@onlyBackends(["cute"])
class TestCuteChunkUnbind(TestCase):
    def test_chunk_of_loaded_rows(self) -> None:
        for d in (16, 128):
            x = torch.randn([65, d], device=DEVICE)
            a, b = x.chunk(2, dim=-1)
            for layout in _ROW_LAYOUTS:
                with self.subTest(d=d, layout=layout):
                    code, (lo, hi) = code_and_output(chunk_store, (x,), **layout)
                    self.assertNotIn("split_smem", code)
                    torch.testing.assert_close(lo, a)
                    torch.testing.assert_close(hi, b)
                    _code, result = code_and_output(chunk_glu, (x,), **layout)
                    torch.testing.assert_close(result, torch.sigmoid(a) * b)

    def test_chunk_of_flattened_tiles(self) -> None:
        # Triton cannot permute a rank-compacted tile; CuTe re-reads each half
        # at the positional coordinates of the flattened tile.
        for shape in ([3, 33, 64], [4, 8, 16]):
            x = torch.randn(shape, device=DEVICE)
            a, b = x.chunk(2, dim=-1)
            for block_sizes, flatten in (
                ([2, 16], False),
                ([2, 16], True),
                ([4, 8], True),
            ):
                with self.subTest(
                    shape=shape, block_sizes=block_sizes, flatten=flatten
                ):
                    _code, result = code_and_output(
                        chunk_glu_rows,
                        (x,),
                        block_sizes=block_sizes,
                        flatten_loops=[flatten],
                    )
                    torch.testing.assert_close(result, a * torch.sigmoid(b))

    def test_unbind_of_trailing_pair(self) -> None:
        x = torch.randn([65, 2], device=DEVICE)
        for block_sizes in ([32], [128]):
            with self.subTest(block_sizes=block_sizes):
                _code, result = code_and_output(
                    unbind_pairs, (x,), block_sizes=block_sizes
                )
                torch.testing.assert_close(result, x[:, 0] ** 2 + x[:, 1] ** 2)

    def test_unbind_of_stack_takes_its_operands(self) -> None:
        # The stacked dim has no lane; the split of its unbind is the pair of
        # computed operands, with no exchange.
        for d in (16, 64):
            x = torch.randn([65, d], device=DEVICE)
            with self.subTest(d=d):
                code, result = code_and_output(unbind_stacked, (x,), block_sizes=[32])
                self.assertNotIn("split_smem", code)
                torch.testing.assert_close(result, (x * 2.0) * (x + 1.0))

    def test_chunk_of_computed_row_refused(self) -> None:
        # A row wider than a warp runs 32 threads with synthetic lanes; the
        # halves of a computed row are other lane iterations of one thread.
        for d in (64, 128):
            x = torch.randn([65, d], device=DEVICE)
            with (
                self.subTest(d=d),
                self.assertRaisesRegex(
                    exc.BackendUnsupported,
                    "different iteration of an enclosing lane loop",
                ),
            ):
                code_and_output(chunk_computed, (x,), block_sizes=[32])

    def test_chunk_with_operand_on_another_lane_refused(self) -> None:
        x = torch.randn([65, 64], device=DEVICE)
        bias = torch.randn([65, 32], device=DEVICE)
        with self.assertRaisesRegex(
            exc.BackendUnsupported, "with different lane coordinates"
        ):
            code_and_output(chunk_bias, (x, bias), block_sizes=[32])


if __name__ == "__main__":
    unittest.main()
