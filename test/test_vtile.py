from __future__ import annotations

import unittest

import torch

import helion
from helion._testing import DEVICE
from helion._testing import RefEagerTestBase
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
import helion.language as hl


@onlyBackends(["triton"])
class TestVTile(RefEagerTestBase, TestCase):
    def test_vtile_jagged_sum(self):
        @helion.kernel(autotune_effort="none")
        def jagged_row_sum(x_data: torch.Tensor, x_offsets: torch.Tensor) -> torch.Tensor:
            b = x_offsets.size(0) - 1
            out = torch.zeros([b], dtype=x_data.dtype, device=x_data.device)

            for tile_b in hl.tile(b):
                starts = x_offsets[tile_b]
                ends = x_offsets[tile_b.index + 1]
                nnz = ends - starts
                acc = hl.zeros([tile_b], dtype=x_data.dtype)

                for tile_k in hl.vtile(nnz):
                    idx = starts[:, None] + tile_k.index[None, :]
                    acc = acc + x_data[idx].sum(dim=1)

                out[tile_b] = acc
            return out

        lengths = torch.tensor([3, 1, 4, 2], device=DEVICE, dtype=torch.long)
        offsets = torch.cat(
            [
                torch.zeros(1, device=DEVICE, dtype=torch.long),
                torch.cumsum(lengths, dim=0),
            ]
        )
        x = torch.randn(int(offsets[-1].item()), device=DEVICE, dtype=torch.float32)

        def ref(x_data: torch.Tensor, x_offsets: torch.Tensor) -> torch.Tensor:
            b = x_offsets.numel() - 1
            out = torch.zeros([b], dtype=x_data.dtype, device=x_data.device)
            for i in range(b):
                s = int(x_offsets[i].item())
                e = int(x_offsets[i + 1].item())
                out[i] = x_data[s:e].sum()
            return out

        _, result = code_and_output(jagged_row_sum, (x, offsets))
        torch.testing.assert_close(result, ref(x, offsets))

    def test_vtile_cannot_be_outermost_loop(self):
        @helion.kernel(autotune_effort="none")
        def bad_outer_vtile(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
            out = torch.zeros_like(x)
            for tile_i in hl.vtile(lengths):
                out[tile_i] = x[tile_i]
            return out

        x = torch.randn(8, device=DEVICE)
        lengths = torch.tensor([2, 3], device=DEVICE, dtype=torch.long)

        with self.assertRaises(helion.exc.InternalError):
            code_and_output(bad_outer_vtile, (x, lengths))

    def test_vtile_cannot_be_outer_and_scalar(self):
        @helion.kernel(autotune_effort="none")
        def bad_outer_vtile(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
            out = torch.zeros_like(x)
            m, = x.size()
            for tile_i in hl.vtile(m):
                out[tile_i] = x[tile_i]
            return out

        x = torch.randn(8, device=DEVICE)
        lengths = torch.tensor([2, 3], device=DEVICE, dtype=torch.long)

        with self.assertRaises(helion.exc.IncorrectTileUsage):
            code_and_output(bad_outer_vtile, (x, lengths))

    def test_vtile_no_scalar_bound(self):
        @helion.kernel(autotune_effort="none")
        def dense_add_bad_vtile(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, n = x.size()
            out = torch.empty_like(x)
            for tile_m in hl.tile(m):
                for tile_n in hl.vtile(n):
                    out[tile_m, tile_n] = x[tile_m, tile_n] + y[tile_m, tile_n]
            return out

        x = torch.randn([8, 16], device=DEVICE)
        y = torch.randn([8, 16], device=DEVICE)

        with self.assertRaises(helion.exc.IncorrectTileUsage):
            code_and_output(dense_add_bad_vtile, (x, y))

if __name__ == "__main__":
    unittest.main()
