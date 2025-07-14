from __future__ import annotations

import unittest

import torch

import helion
from helion._compat import get_tensor_descriptor_fn_name
from helion._compat import supports_tensor_descriptor
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import code_and_output
import helion.language as hl


@helion.kernel
def broadcast_add_3d(
    x: torch.Tensor, bias1: torch.Tensor, bias2: torch.Tensor
) -> torch.Tensor:
    d0, d1, d2 = x.size()
    out = torch.empty_like(x)
    for tile_l, tile_m, tile_n in hl.tile([d0, d1, d2]):
        # bias1 has shape [1, d1, d2], bias2 has shape [d0, 1, d2]
        out[tile_l, tile_m, tile_n] = (
            x[tile_l, tile_m, tile_n]
            + bias1[tile_l, tile_m, tile_n]
            + bias2[tile_l, tile_m, tile_n]
        )
    return out


class TestIndexing(TestCase):
    def test_arange(self):
        @helion.kernel
        def arange(length: int, device: torch.device) -> torch.Tensor:
            out = torch.empty([length], dtype=torch.int32, device=device)
            for tile in hl.tile(length):
                out[tile] = tile.index
            return out

        code, result = code_and_output(
            arange,
            (100, DEVICE),
            block_size=32,
        )
        torch.testing.assert_close(
            result, torch.arange(0, 100, device=DEVICE, dtype=torch.int32)
        )
        self.assertExpectedJournal(code)

    def test_pairwise_add(self):
        @helion.kernel()
        def pairwise_add(x: torch.Tensor) -> torch.Tensor:
            out = x.new_empty([x.size(0) - 1])
            for tile in hl.tile(out.size(0)):
                out[tile] = x[tile] + x[tile.index + 1]
            return out

        x = torch.randn([500], device=DEVICE)
        code, result = code_and_output(
            pairwise_add,
            (x,),
            block_size=32,
        )
        torch.testing.assert_close(result, x[:-1] + x[1:])
        self.assertExpectedJournal(code)

    def test_mask_store(self):
        @helion.kernel
        def masked_store(x: torch.Tensor) -> torch.Tensor:
            out = torch.zeros_like(x)
            for tile in hl.tile(out.size(0)):
                hl.store(out, [tile], x[tile], extra_mask=(tile.index % 2) == 0)
            return out

        x = torch.randn([200], device=DEVICE)
        code, result = code_and_output(
            masked_store,
            (x,),
            block_size=16,
        )
        torch.testing.assert_close(
            result, torch.where(torch.arange(200, device=DEVICE) % 2 == 0, x, 0)
        )
        self.assertExpectedJournal(code)

    def test_mask_load(self):
        @helion.kernel
        def masked_load(x: torch.Tensor) -> torch.Tensor:
            out = torch.zeros_like(x)
            for tile in hl.tile(out.size(0)):
                out[tile] = hl.load(x, [tile], extra_mask=(tile.index % 2) == 0)
            return out

        x = torch.randn([200], device=DEVICE)
        code, result = code_and_output(
            masked_load,
            (x,),
            block_size=16,
        )
        torch.testing.assert_close(
            result, torch.where(torch.arange(200, device=DEVICE) % 2 == 0, x, 0)
        )
        self.assertExpectedJournal(code)

    def test_tile_begin_end(self):
        @helion.kernel
        def tile_range_copy(x: torch.Tensor) -> torch.Tensor:
            out = torch.zeros_like(x)
            for tile in hl.tile(x.size(0)):
                for inner_tile in hl.tile(tile.begin, tile.end):
                    out[inner_tile] = x[inner_tile]
            return out

        x = torch.randn([100], device=DEVICE)
        code, result = code_and_output(
            tile_range_copy,
            (x,),
            block_size=[32, 16],
        )
        torch.testing.assert_close(result, x)
        code, result = code_and_output(
            tile_range_copy,
            (x,),
            block_size=[1, 1],
        )
        torch.testing.assert_close(result, x)

    def test_tile_block_size(self):
        @helion.kernel
        def test_block_size_access(x: torch.Tensor) -> torch.Tensor:
            out = torch.zeros_like(x, dtype=torch.int32)
            for tile in hl.tile(x.size(0)):
                out[tile] = tile.block_size
            return out

        x = torch.randn([64], device=DEVICE)
        code, result = code_and_output(
            test_block_size_access,
            (x,),
            block_size=16,
        )
        expected = torch.full_like(x, 16, dtype=torch.int32)
        torch.testing.assert_close(result, expected)
        code, result = code_and_output(
            test_block_size_access,
            (x,),
            block_size=1,
        )
        expected = torch.full_like(x, 1, dtype=torch.int32)
        torch.testing.assert_close(result, expected)

    def test_assign_int(self):
        @helion.kernel
        def fn(x: torch.Tensor) -> torch.Tensor:
            for tile in hl.tile(x.size(0)):
                x[tile] = 1
            return x

        x = torch.zeros([200], device=DEVICE)
        expected = torch.ones_like(x)
        code, result = code_and_output(
            fn,
            (x,),
        )
        torch.testing.assert_close(result, expected)

    def test_tile_id(self):
        @helion.kernel
        def test_tile_id_access(x: torch.Tensor) -> torch.Tensor:
            out = torch.zeros_like(x, dtype=torch.int32)
            for tile in hl.tile(x.size(0)):
                out[tile] = tile.id
            return out

        x = torch.randn([64], device=DEVICE)
        code, result = code_and_output(
            test_tile_id_access,
            (x,),
            block_size=16,
        )
        expected = torch.arange(4, device=DEVICE, dtype=torch.int32).repeat_interleave(
            repeats=16
        )
        torch.testing.assert_close(result, expected)
        code, result = code_and_output(
            test_tile_id_access,
            (x,),
            block_size=1,
        )
        expected = torch.arange(64, device=DEVICE, dtype=torch.int32)
        torch.testing.assert_close(result, expected)

    def test_tile_id_1d_indexing(self):
        @helion.kernel
        def test_tile_id_atomic_add(x: torch.Tensor) -> torch.Tensor:
            out = torch.zeros_like(x, dtype=torch.int32)
            for tile_m in hl.tile(x.size(0)):
                hl.atomic_add(out, [tile_m.id], 1)
            return out

        x = torch.randn(64, device=DEVICE)
        code, result = code_and_output(
            test_tile_id_atomic_add,
            (x,),
            block_size=[
                16,
            ],
        )

        expected = torch.zeros(64, device=DEVICE, dtype=torch.int32)
        expected[:4] = 1
        torch.testing.assert_close(result, expected)
        code, result = code_and_output(
            test_tile_id_atomic_add,
            (x,),
            block_size=[
                1,
            ],
        )
        expected = torch.ones(64, device=DEVICE, dtype=torch.int32)
        torch.testing.assert_close(result, expected)

    def test_tile_id_2d_indexing(self):
        @helion.kernel
        def test_tile_id_index_st(x: torch.Tensor) -> torch.Tensor:
            out = torch.zeros_like(x, dtype=torch.int32)
            for tile_m, tile_n in hl.tile(x.size()):
                out[tile_m.id, tile_n.id] = 1
            return out

        x = torch.randn(64, 64, device=DEVICE)
        code, result = code_and_output(
            test_tile_id_index_st,
            (x,),
            block_size=[16, 16],
        )

        expected = torch.zeros(64, 64, device=DEVICE, dtype=torch.int32)
        expected[:4, :4] = 1
        torch.testing.assert_close(result, expected)
        code, result = code_and_output(
            test_tile_id_index_st,
            (x,),
            block_size=[1, 1],
        )
        expected = torch.ones(64, 64, device=DEVICE, dtype=torch.int32)
        torch.testing.assert_close(result, expected)

    def test_atomic_add_symint(self):
        @helion.kernel(config={"block_size": 32})
        def fn(x: torch.Tensor) -> torch.Tensor:
            for tile in hl.tile(x.size(0)):
                hl.atomic_add(x, [tile], tile.block_size + 1)
            return x

        x = torch.zeros([200], device=DEVICE)
        expected = x + 33
        code, result = code_and_output(
            fn,
            (x,),
        )
        torch.testing.assert_close(result, expected)

    def test_arange_tile_block_size(self):
        @helion.kernel(use_default_config=True)
        def arange_from_block_size(x: torch.Tensor) -> torch.Tensor:
            out = torch.zeros([x.size(0)], dtype=torch.int32, device=x.device)
            for tile in hl.tile(x.size(0)):
                # Test the exact pattern requested: torch.arange(tile.block_size, device=x.device)
                out[tile] = torch.arange(tile.block_size, device=x.device)
            return out

        x = torch.randn([64], device=DEVICE)
        code, result = code_and_output(
            arange_from_block_size,
            (x,),
            block_size=16,
        )
        expected = torch.arange(16, dtype=torch.int32, device=DEVICE).repeat(4)
        torch.testing.assert_close(result, expected)

    def test_arange_two_args(self):
        @helion.kernel(use_default_config=True)
        def arange_two_args(x: torch.Tensor) -> torch.Tensor:
            out = torch.zeros([x.size(0)], dtype=torch.int32, device=x.device)
            for tile in hl.tile(x.size(0)):
                # Test the exact pattern requested: torch.arange(tile.begin, tile.begin+tile.block_size, device=x.device)
                out[tile] = torch.arange(
                    tile.begin, tile.begin + tile.block_size, device=x.device
                )
            return out

        x = torch.randn([64], device=DEVICE)
        code, result = code_and_output(
            arange_two_args,
            (x,),
            block_size=16,
        )
        expected = torch.arange(64, dtype=torch.int32, device=DEVICE)
        torch.testing.assert_close(result, expected)

    def test_arange_three_args_step(self):
        @helion.kernel(config={"block_size": 8})
        def arange_three_args_step(x: torch.Tensor) -> torch.Tensor:
            out = torch.zeros([x.size(0) // 2], dtype=torch.int32, device=x.device)
            for tile in hl.tile(x.size(0) // 2):
                # Test the exact pattern requested: torch.arange(start, end, step=2, device=x.device)
                start_idx = tile.begin * 2
                end_idx = (tile.begin + tile.block_size) * 2
                out[tile] = torch.arange(start_idx, end_idx, step=2, device=x.device)
            return out

        x = torch.randn([64], device=DEVICE)
        code, result = code_and_output(
            arange_three_args_step,
            (x,),
        )
        expected = torch.arange(0, 64, step=2, dtype=torch.int32, device=DEVICE)
        torch.testing.assert_close(result, expected)
        self.assertExpectedJournal(code)

    def test_arange_hl_alias(self):
        @helion.kernel(config={"block_size": 8})
        def arange_three_args_step(x: torch.Tensor) -> torch.Tensor:
            out = torch.zeros([x.size(0) // 2], dtype=torch.int32, device=x.device)
            for tile in hl.tile(x.size(0) // 2):
                start_idx = tile.begin * 2
                end_idx = (tile.begin + tile.block_size) * 2
                out[tile] = hl.arange(start_idx, end_idx, step=2)
            return out

        x = torch.randn([64], device=DEVICE)
        code, result = code_and_output(
            arange_three_args_step,
            (x,),
        )
        expected = torch.arange(0, 64, step=2, dtype=torch.int32, device=DEVICE)
        torch.testing.assert_close(result, expected)

    def test_broadcasting_pointer_indexing(self):
        x = torch.randn([16, 24, 32], device=DEVICE)
        bias1 = torch.randn([1, 24, 32], device=DEVICE)
        bias2 = torch.randn([16, 1, 32], device=DEVICE)
        code, result = code_and_output(
            broadcast_add_3d,
            (x, bias1, bias2),
            indexing="pointer",
            block_size=[8, 8, 8],
        )
        expected = x + bias1 + bias2
        torch.testing.assert_close(result, expected)
        self.assertExpectedJournal(code)

    def test_broadcasting_block_ptr_indexing(self):
        x = torch.randn([16, 24, 32], device=DEVICE)
        bias1 = torch.randn([1, 24, 32], device=DEVICE)
        bias2 = torch.randn([16, 1, 32], device=DEVICE)
        code, result = code_and_output(
            broadcast_add_3d,
            (x, bias1, bias2),
            indexing="block_ptr",
            block_size=[8, 8, 8],
        )
        expected = x + bias1 + bias2
        torch.testing.assert_close(result, expected)
        self.assertExpectedJournal(code)

    @unittest.skipIf(not supports_tensor_descriptor(), "TensorDescriptor not supported")
    @unittest.skipIf(
        get_tensor_descriptor_fn_name() == "tl._experimental_make_tensor_descriptor",
        "LLVM ERROR: Illegal shared layout",
    )
    def test_broadcasting_tensor_descriptor_indexing(self):
        x = torch.randn([16, 24, 32], device=DEVICE)
        bias1 = torch.randn([1, 24, 32], device=DEVICE)
        bias2 = torch.randn([16, 1, 32], device=DEVICE)
        code, result = code_and_output(
            broadcast_add_3d,
            (x, bias1, bias2),
            indexing="tensor_descriptor",
            block_size=[8, 8, 8],
        )
        expected = x + bias1 + bias2
        torch.testing.assert_close(result, expected)
        self.assertExpectedJournal(code)


if __name__ == "__main__":
    unittest.main()
