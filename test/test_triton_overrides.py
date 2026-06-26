from __future__ import annotations

import torch

import helion
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import onlyBackends
from helion._testing import skipIfRefEager
import helion.language as hl


@onlyBackends(["triton"])
class TestTritonDivisionSimplification(TestCase):
    @skipIfRefEager("to_triton_code() requires compilation")
    def test_tile_index_divmod_simplified(self):

        @helion.kernel(
            config=helion.Config(block_sizes=[32, 32, 32], num_warps=4),
            static_shapes=True,
        )
        def gather_kernel(
            x: torch.Tensor,
            w: torch.Tensor,
            out: torch.Tensor,
            M_cols: int,
            K_b: int,
        ) -> None:
            M, N = out.size()
            K = w.size(0)
            for tile_m, tile_n in hl.tile([M, N]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(K):
                    row = tile_m.index // M_cols
                    a = tile_k.index // K_b
                    b = tile_k.index % K_b
                    flat_idx = row[:, None] * K + a[None, :] * K_b + b[None, :]
                    a_tile = hl.load(x, [flat_idx])
                    acc = torch.addmm(acc, a_tile, w[tile_k, tile_n])
                out[tile_m, tile_n] = acc

        M_rows, M_cols, K_a, K_b, N = 16, 16, 4, 8, 32
        M, K = M_rows * M_cols, K_a * K_b

        x = torch.randn(M_rows * K, device=DEVICE)
        w = torch.randn(K, N, device=DEVICE)
        out = torch.empty(M, N, device=DEVICE)

        bound = gather_kernel.bind((x, w, out, M_cols, K_b))
        code = bound.to_triton_code()

        self.assertIn(" // ", code)
        self.assertIn(" % ", code)
        self.assertNotIn("signbit", code)
        self.assertNotIn("bitwise_not", code)

        gather_kernel(x, w, out, M_cols, K_b)
        x_2d = x.view(M_rows, K)
        idx = torch.arange(M, device=DEVICE)
        rows = idx // M_cols
        expected = x_2d[rows] @ w
        torch.testing.assert_close(out, expected, rtol=1e-2, atol=1e-2)

    @skipIfRefEager("to_triton_code() requires compilation")
    def test_chained_divmod_simplified(self):

        @helion.kernel(
            config=helion.Config(block_sizes=[32], num_warps=4),
            static_shapes=True,
        )
        def decompose_kernel(
            out_row: torch.Tensor,
            out_col: torch.Tensor,
            W_out: int,
        ) -> None:
            (N,) = out_row.size()
            for tile in hl.tile(N):
                idx = tile.index
                row = idx // W_out
                col = idx % W_out
                out_row[tile] = row
                out_col[tile] = col

        H_out_val = 8
        W_out_val = 4
        total = H_out_val * W_out_val

        out_row = torch.empty(total, device=DEVICE, dtype=torch.int32)
        out_col = torch.empty(total, device=DEVICE, dtype=torch.int32)

        bound = decompose_kernel.bind((out_row, out_col, W_out_val))
        code = bound.to_triton_code()

        self.assertIn(" // ", code)
        self.assertIn(" % ", code)
        self.assertNotIn("signbit", code)

        decompose_kernel(out_row, out_col, W_out_val)
        expected_row = torch.arange(total, device=DEVICE) // W_out_val
        expected_col = torch.arange(total, device=DEVICE) % W_out_val
        torch.testing.assert_close(out_row, expected_row.to(torch.int32))
        torch.testing.assert_close(out_col, expected_col.to(torch.int32))
