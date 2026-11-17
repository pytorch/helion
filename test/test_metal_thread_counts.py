"""How a Metal launch's tiles take their threads (``metal/thread_counts.py``):
the rules themselves, the launch they predict against the one codegen
plans, an explicit ``num_threads`` past its block counted at its block, and
tiles past the three thread axes.  Binds CPU tensors only."""

from __future__ import annotations

from typing import TYPE_CHECKING
import unittest

import pytest
import torch

import helion
from helion._compiler.metal.thread_counts import launch_threads
from helion._compiler.metal.thread_counts import serialize_auto_threads
from helion._compiler.metal.thread_counts import shrink_tile_threads
from helion._testing import skipUnlessBackends
from helion._testing.metal_codegen import launch_dims
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Sequence

# Metal codegen only: other backends' CPU lanes skip the module.
pytestmark = [pytest.mark.cpu_only, skipUnlessBackends(["metal"])]


@helion.kernel(backend="metal", static_shapes=True)
def matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    n = y.size(1)
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = acc
    return out


@helion.kernel(backend="metal", static_shapes=True)
def add_one(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size()):
        out[tile] = x[tile] + 1
    return out


@helion.kernel(backend="metal", static_shapes=True)
def add_one_nested(x: torch.Tensor) -> torch.Tensor:
    a, b, c, d = x.size()
    out = torch.empty_like(x)
    for tile_a, tile_b in hl.tile([a, b]):
        for tile_c, tile_d in hl.tile([c, d]):
            out[tile_a, tile_b, tile_c, tile_d] = x[tile_a, tile_b, tile_c, tile_d] + 1
    return out


def _threaded(dims: Sequence[int]) -> list[int]:
    """The counts over one thread, in order: a tile's threads, or a launch
    axis's, where a tile on one thread takes no axis."""
    return [d for d in dims if d > 1]


class TestThreadCounts(unittest.TestCase):
    def test_rules(self) -> None:
        """A loop's tiles shrink largest first (the first of equals) to the
        budget; a tile reduced after its loop goes to one; nested auto
        tiles shrink to what the rest of the kernel leaves."""
        self.assertEqual(shrink_tile_threads([16, 16], [16, 16], 32), [4, 8])
        self.assertEqual(shrink_tile_threads([16, 16], [16, 16], 64, fixed=2), [4, 8])
        self.assertEqual(
            shrink_tile_threads([16, 16], [16, 16], 32, to_one={0}), [1, 16]
        )
        self.assertEqual(serialize_auto_threads([16], [16], [0], 256, 32), ([2], 32))
        self.assertEqual(serialize_auto_threads([16], [16], [], 256, 32), ([16], 256))

    def test_launch_matches_codegen(self) -> None:
        """``launch_threads`` over the kernel's loops (the grid's M and N,
        then K) gives the threadgroup codegen plans, for explicit and auto
        counts, nested counts kept as given, and counts past their block."""
        x = torch.randn(256, 128, dtype=torch.float16)
        y = torch.randn(128, 256, dtype=torch.float16)
        bound = matmul.bind((x, y))
        index = bound.config_spec.num_threads.block_id_to_index
        loops = [[index(0), index(1)], [index(2)]]
        # Each axis covers a launch rule: auto tiles shrinking to the
        # budget, explicit counts kept or shrunk with the grid, nested
        # explicit counts kept, counts past their block, and a count that
        # does not divide its block.
        for block_sizes, num_threads, num_warps in (
            ([16, 16, 16], [0, 0, 0], 4),
            ([16, 16, 16], [0, 0, 4], 4),
            ([16, 16, 16], [2, 2, 0], 1),
            ([16, 16, 16], [1, 0, 0], 1),
            ([32, 16, 64], [0, 0, 2], 8),
            ([64, 32, 32], [64, 32, 1], 8),
            ([16, 16, 16], [128, 1, 0], 8),
            ([16, 16, 64], [1, 1, 0], 1),
            ([16, 128, 16], [1, 96, 0], 4),
        ):
            with self.subTest(blocks=block_sizes, threads=num_threads):
                config = helion.Config(
                    block_sizes=block_sizes,
                    num_threads=num_threads,
                    num_warps=num_warps,
                )
                predicted = launch_threads(
                    loops,
                    block_sizes,
                    num_threads,
                    min(1024, 32 * num_warps),
                )
                self.assertEqual(
                    _threaded(predicted), _threaded(launch_dims(bound.to_code(config)))
                )

    def test_launch_past_three_axes(self) -> None:
        """While a loop's tiles take more thread axes than Metal's three
        leave it (less the grid's, below a nested loop), its latest auto tile
        runs on one thread: ``launch_threads`` gives the launch codegen plans
        for a 4-D tile and for two tiles nested in a 2-D grid."""
        x = torch.randn(8, 8, 16, 16)
        for kernel, loops, block_sizes, num_threads, num_warps, launched in (
            # Every tile on auto: the last runs on one thread.
            (add_one, [[0, 1, 2, 3]], [2, 4, 4, 4], [0, 0, 0, 0], 4, [2, 4, 4, 1]),
            # The one auto tile runs on one thread, ahead of explicit ones.
            (add_one, [[0, 1, 2, 3]], [16, 4, 4, 16], [0, 2, 8, 8], 32, [1, 2, 4, 8]),
            # The grid takes two axes and leaves the nested loop one: its
            # auto tile runs on one thread, and the one shrunk to the budget
            # keeps its 2.
            (
                add_one_nested,
                [[0, 1], [2, 3]],
                [8, 4, 4, 16],
                [4, 0, 0, 0],
                16,
                [4, 4, 2, 1],
            ),
        ):
            with self.subTest(blocks=block_sizes, threads=num_threads):
                bound = kernel.bind((x,))
                index = bound.config_spec.num_threads.block_id_to_index
                predicted = launch_threads(
                    [[index(b) for b in loop] for loop in loops],
                    block_sizes,
                    num_threads,
                    min(1024, 32 * num_warps),
                )
                self.assertEqual(predicted, launched)
                config = helion.Config(
                    block_sizes=block_sizes,
                    num_threads=num_threads,
                    num_warps=num_warps,
                )
                self.assertEqual(
                    _threaded(launch_dims(bound.to_code(config))), _threaded(launched)
                )

    def test_count_past_block(self) -> None:
        """An explicit count past its block launches its block, and the
        others count it so: N's 32 over a block of 16 leaves K 2 threads,
        one SIMD group, where counting the 32 left K one thread and the
        launch 16 threads, which MPP refuses."""
        x = torch.randn(64, 64, dtype=torch.float16)
        y = torch.randn(64, 64, dtype=torch.float16)
        bound = matmul.bind((x, y))
        code = bound.to_code(
            helion.Config(block_sizes=[16, 16, 16], num_threads=[1, 32, 0], num_warps=1)
        )
        self.assertEqual(launch_dims(code), (16, 2, 1))
        code = bound.to_code(
            helion.Config(
                block_sizes=[16, 16, 16], num_threads=[128, 1, 0], num_warps=8
            )
        )
        self.assertEqual(launch_dims(code), (16, 16, 1))

    def test_count_not_dividing_block(self) -> None:
        """An explicit count that does not divide its block launches the
        largest divisor below it, and the others count it so: N's 96 over a
        block of 128 launches 64 threads and leaves K 2, as 64 does."""
        x = torch.randn(256, 64, dtype=torch.float16)
        y = torch.randn(64, 256, dtype=torch.float16)
        bound = matmul.bind((x, y))
        for count in (96, 64):
            with self.subTest(count=count):
                code = bound.to_code(
                    helion.Config(
                        block_sizes=[16, 128, 16],
                        num_threads=[1, count, 0],
                        num_warps=4,
                    )
                )
                self.assertEqual(launch_dims(code), (64, 2, 1))


if __name__ == "__main__":
    unittest.main()
