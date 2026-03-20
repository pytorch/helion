from __future__ import annotations

import unittest

import torch

import helion
from helion._testing import DEVICE
from helion._testing import RefEagerTestDisabled
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
import helion.language as hl


@onlyBackends(["triton"])
class TestJaggedTile(RefEagerTestDisabled, TestCase):
    def test_jagged_tile_jagged_sum(self):
        @helion.kernel(autotune_effort="none")
        def jagged_row_sum(
            x_data: torch.Tensor, x_offsets: torch.Tensor
        ) -> torch.Tensor:
            b = x_offsets.size(0) - 1
            out = torch.zeros([b], dtype=x_data.dtype, device=x_data.device)

            for tile_b in hl.tile(b):
                starts = x_offsets[tile_b]
                ends = x_offsets[tile_b.index + 1]
                nnz = ends - starts
                acc = hl.zeros([tile_b], dtype=x_data.dtype)

                for tile_k in hl.jagged_tile(nnz):
                    idx = starts[:, None] + tile_k.index[None, :]
                    acc += x_data[idx].sum(dim=1)

                out[tile_b] = acc
            return out

        offsets = torch.tensor([0, 3, 4, 8, 10], device=DEVICE, dtype=torch.long)
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

    def test_jagged_tile_reduction_mask(self):
        @helion.kernel(autotune_effort="none")
        def jagged_row_sum(
            x_data: torch.Tensor, x_offsets: torch.Tensor
        ) -> torch.Tensor:
            b = x_offsets.size(0) - 1
            out = torch.zeros([b], dtype=x_data.dtype, device=x_data.device)

            for tile_b in hl.tile(b):
                starts = x_offsets[tile_b]
                ends = x_offsets[tile_b.index + 1]
                nnz = ends - starts
                acc = hl.zeros([tile_b], dtype=x_data.dtype)

                for tile_k in hl.jagged_tile(nnz):
                    idx = starts[:, None] + tile_k.index[None, :]
                    acc += (x_data[idx] + 1).sum(dim=1)

                out[tile_b] = acc
            return out

        offsets = torch.tensor([0, 3, 4, 8, 10], device=DEVICE, dtype=torch.long)
        x = torch.randn(int(offsets[-1].item()), device=DEVICE, dtype=torch.float32)

        def ref(x_data: torch.Tensor, x_offsets: torch.Tensor) -> torch.Tensor:
            b = x_offsets.numel() - 1
            out = torch.zeros([b], dtype=x_data.dtype, device=x_data.device)
            for i in range(b):
                s = int(x_offsets[i].item())
                e = int(x_offsets[i + 1].item())
                out[i] = (x_data[s:e] + 1).sum()
            return out

        code, result = code_and_output(jagged_row_sum, (x, offsets))
        self.assertIn("tl.where", code)
        torch.testing.assert_close(result, ref(x, offsets))

    def test_jagged_tile_blocksize_1(self):
        @helion.kernel(config={"block_sizes": [32, 1]})
        def jagged_row_sum(
            x_data: torch.Tensor, x_offsets: torch.Tensor
        ) -> torch.Tensor:
            b = x_offsets.size(0) - 1
            out = torch.zeros([b], dtype=x_data.dtype, device=x_data.device)

            for tile_b in hl.tile(b):
                starts = x_offsets[tile_b]
                ends = x_offsets[tile_b.index + 1]
                nnz = ends - starts
                acc = hl.zeros([tile_b], dtype=x_data.dtype)

                for tile_k in hl.jagged_tile(nnz):
                    idx = starts[:, None] + tile_k.index[None, :]
                    acc = acc + x_data[idx].sum(dim=1)

                out[tile_b] = acc
            return out

        offsets = torch.tensor([0, 3, 4, 8, 10], device=DEVICE, dtype=torch.long)
        x = torch.randn(int(offsets[-1].item()), device=DEVICE, dtype=torch.float32)

        def ref(x_data: torch.Tensor, x_offsets: torch.Tensor) -> torch.Tensor:
            b = x_offsets.numel() - 1
            out = torch.zeros([b], dtype=x_data.dtype, device=x_data.device)
            for i in range(b):
                s = int(x_offsets[i].item())
                e = int(x_offsets[i + 1].item())
                out[i] = x_data[s:e].sum()
            return out

        code, result = code_and_output(jagged_row_sum, (x, offsets))
        self.assertIn("mask_1 = indices_1[None, :] < v_2[:, None]", code)
        self.assertIn("mask_0[:, None] & mask_1", code)
        torch.testing.assert_close(result, ref(x, offsets))

    def test_nested_jagged_tile(self):
        @helion.kernel(autotune_effort="none")
        def dense_jagged_mean(
            x: torch.Tensor,
            lengths: torch.Tensor,
            feature_counts: torch.Tensor,
        ) -> torch.Tensor:
            b = x.size(0)
            max_m = x.size(2)
            out = torch.zeros([b, max_m], dtype=x.dtype, device=x.device)

            for tile_b in hl.tile(b):
                row_lengths = lengths[tile_b]
                row_feature_counts = feature_counts[tile_b]

                for tile_m in hl.jagged_tile(row_feature_counts):
                    acc = hl.zeros([tile_b, tile_m], dtype=x.dtype)

                    for tile_k in hl.jagged_tile(row_lengths):
                        acc += x[tile_b, tile_k, tile_m].sum(dim=1)

                    out[tile_b, tile_m] = acc / row_lengths.to(x.dtype)[:, None]

            return out

        lengths = torch.tensor([3, 1, 2], device=DEVICE, dtype=torch.long)
        feature_counts = torch.tensor([2, 4, 3], device=DEVICE, dtype=torch.int32)
        max_k = 4
        max_m = 5
        x = torch.arange(
            lengths.numel() * max_k * max_m,
            device=DEVICE,
            dtype=torch.float32,
        ).view(-1, max_k, max_m)

        def ref(
            x_data: torch.Tensor,
            row_lengths: torch.Tensor,
            row_feature_counts: torch.Tensor,
        ) -> torch.Tensor:
            b = x_data.size(0)
            out = torch.zeros(
                (b, x_data.size(2)),
                dtype=x_data.dtype,
                device=x_data.device,
            )
            for i in range(b):
                k = int(row_lengths[i].item())
                f = int(row_feature_counts[i].item())
                out[i, :f] = x_data[i, :k, :f].mean(dim=0)
            return out

        code, result = code_and_output(dense_jagged_mean, (x, lengths, feature_counts))
        self.assertIn("mask_1 = indices_1[None, :] < row_feature_counts[:, None]", code)
        self.assertIn("mask_2 = indices_2[None, :] < row_lengths_copy_0[:, None]", code)
        self.assertIn(
            "mask_0[:, None, None] & mask_2[:, :, None] & mask_1[:, None, :]", code
        )
        self.assertIn("mask_0[:, None] & mask_1", code)

        torch.testing.assert_close(result, ref(x, lengths, feature_counts))

    def test_jagged_tile_cannot_be_used_without_parent(self):
        @helion.kernel(autotune_effort="none")
        def chained_jagged_mean(
            x: torch.Tensor,
            lengths: torch.Tensor,
            feature_counts: torch.Tensor,
        ) -> torch.Tensor:
            b = x.size(0)
            out = torch.zeros([b], dtype=x.dtype, device=x.device)

            for tile_b in hl.tile(b):
                row_lengths = lengths[tile_b]
                row_acc = hl.zeros([tile_b], dtype=x.dtype)

                for tile_k in hl.jagged_tile(row_lengths):
                    # jagged_tile (tile_k) cannot be used alone; the parent tile (tile_b) must be involved.
                    # Correct code should be :
                    # token_feature_counts = feature_counts[tile_b[:,None]*0 + tile_k[None,:]]
                    token_feature_counts = feature_counts[tile_k]
                    token_acc = hl.zeros([tile_b, tile_k], dtype=x.dtype)

                    for tile_m in hl.jagged_tile(token_feature_counts):
                        token_acc += x[tile_b, tile_k, tile_m].sum(dim=2)

                    row_acc += (
                        token_acc / token_feature_counts[None, :].to(x.dtype)
                    ).sum(dim=1)

                out[tile_b] = row_acc / row_lengths.to(x.dtype)

            return out

        lengths = torch.tensor([3, 1, 2], device=DEVICE, dtype=torch.long)
        feature_counts = torch.tensor([2, 4, 3, 5], device=DEVICE, dtype=torch.int32)
        max_k = 4
        max_m = 5
        x = torch.arange(
            lengths.numel() * max_k * max_m,
            device=DEVICE,
            dtype=torch.float32,
        ).view(-1, max_k, max_m)

        with self.assertRaises(helion.exc.InvalidJaggedTileUsage):
            _, result = code_and_output(
                chained_jagged_mean, (x, lengths, feature_counts)
            )

    def test_jagged_tile_cannot_be_outermost_loop(self):
        @helion.kernel(autotune_effort="none")
        def bad_outer_jagged_tile(
            x: torch.Tensor, lengths: torch.Tensor
        ) -> torch.Tensor:
            out = torch.zeros_like(x)
            for tile_i in hl.jagged_tile(lengths):
                out[tile_i] = x[tile_i]
            return out

        x = torch.randn(8, device=DEVICE)
        lengths = torch.tensor([2, 3], device=DEVICE, dtype=torch.long)

        with self.assertRaises(helion.exc.InvalidJaggedTileUsage):
            code_and_output(bad_outer_jagged_tile, (x, lengths))

    def test_jagged_tile_cannot_be_outer_and_scalar(self):
        @helion.kernel(autotune_effort="none")
        def bad_outer_jagged_tile(
            x: torch.Tensor, lengths: torch.Tensor
        ) -> torch.Tensor:
            out = torch.zeros_like(x)
            (m,) = x.size()
            for tile_i in hl.jagged_tile(m):
                out[tile_i] = x[tile_i]
            return out

        x = torch.randn(8, device=DEVICE)
        lengths = torch.tensor([2, 3], device=DEVICE, dtype=torch.long)

        with self.assertRaises(helion.exc.InvalidJaggedTileUsage):
            code_and_output(bad_outer_jagged_tile, (x, lengths))

    def test_jagged_tile_no_scalar_bound(self):
        @helion.kernel(autotune_effort="none")
        def dense_add_bad_jagged_tile(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, n = x.size()
            out = torch.empty_like(x)
            for tile_m in hl.tile(m):
                for tile_n in hl.jagged_tile(n):
                    out[tile_m, tile_n] = x[tile_m, tile_n] + y[tile_m, tile_n]
            return out

        x = torch.randn([8, 16], device=DEVICE)
        y = torch.randn([8, 16], device=DEVICE)

        with self.assertRaises(helion.exc.InvalidJaggedTileUsage):
            code_and_output(dense_add_bad_jagged_tile, (x, y))


if __name__ == "__main__":
    unittest.main()
