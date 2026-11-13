"""How a Metal launch's tiles take their threads (``metal/thread_counts.py``):
the rules themselves, the launch they predict against the one codegen
plans, and an explicit ``num_threads`` past its block counted at its
block.  Binds CPU tensors only."""

from __future__ import annotations

import math
import re
import unittest

import pytest
import torch

import helion
from helion._compiler.metal.thread_counts import launch_threads
from helion._compiler.metal.thread_counts import serialize_auto_threads
from helion._compiler.metal.thread_counts import shrink_tile_threads
from helion._testing import skipUnlessBackends
import helion.language as hl

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


def _launch(code: str) -> tuple[int, ...]:
    """The threadgroup the generated launcher asks for."""
    match = re.search(r"_block_dims=\((\d+), (\d+), (\d+)\)", code)
    assert match is not None, code
    return tuple(int(d) for d in match.groups())


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
        # explicit counts kept, and counts past their block.
        for block_sizes, num_threads, num_warps in (
            ([16, 16, 16], [0, 0, 0], 4),
            ([16, 16, 16], [0, 0, 4], 4),
            ([16, 16, 16], [2, 2, 0], 1),
            ([16, 16, 16], [1, 0, 0], 1),
            ([32, 16, 64], [0, 0, 2], 8),
            ([64, 32, 32], [64, 32, 1], 8),
            ([16, 16, 16], [128, 1, 0], 8),
            ([16, 16, 64], [1, 1, 0], 1),
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
                    math.prod(predicted), math.prod(_launch(bound.to_code(config)))
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
        self.assertEqual(_launch(code), (16, 2, 1))
        code = bound.to_code(
            helion.Config(
                block_sizes=[16, 16, 16], num_threads=[128, 1, 0], num_warps=8
            )
        )
        self.assertEqual(_launch(code), (16, 16, 1))


if __name__ == "__main__":
    unittest.main()
