from __future__ import annotations

import unittest

import pytest
import torch

import helion
from helion._compat import get_tensor_descriptor_fn_name
from helion._compat import supports_tensor_descriptor
from helion._testing import DEVICE
from helion._testing import RefEagerTestBase
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import skipIfNormalMode
from helion._testing import skipIfRefEager
from helion._testing import skipIfRocm
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


@helion.kernel
def reduction_sum(x: torch.Tensor) -> torch.Tensor:
    m, _ = x.size()
    out = torch.empty([m], device=x.device, dtype=x.dtype)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile, :].to(torch.float32).sum(-1).to(x.dtype)

    return out


class TestIndexing(RefEagerTestBase, TestCase):
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

    @skipIfRefEager(
        "Test is block size dependent which is not supported in ref eager mode"
    )
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

    @skipIfRefEager(
        "Test is block size dependent which is not supported in ref eager mode"
    )
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

    @skipIfRefEager(
        "Test is block size dependent which is not supported in ref eager mode"
    )
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

    @skipIfRefEager(
        "Test is block size dependent which is not supported in ref eager mode"
    )
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

    @skipIfRefEager(
        "Test is block size dependent which is not supported in ref eager mode"
    )
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

    @skipIfRefEager(
        "Test is block size dependent which is not supported in ref eager mode"
    )
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

    def test_arange_block_size_multiple(self):
        """Test that tile.block_size * constant works in hl.arange"""

        @helion.kernel(use_default_config=True, static_shapes=True)
        def arange_block_size_mul(x: torch.Tensor) -> torch.Tensor:
            out = torch.zeros([x.size(0) * 2], dtype=torch.int32, device=x.device)
            for tile in hl.tile(x.size(0)):
                indices = hl.arange(
                    tile.begin * 2, tile.begin * 2 + tile.block_size * 2
                )
                out[indices] = indices
            return out

        x = torch.randn([64], device=DEVICE)
        code, result = code_and_output(arange_block_size_mul, (x,))

        expected = torch.arange(128, dtype=torch.int32, device=DEVICE)
        torch.testing.assert_close(result, expected)

        self.assertExpectedJournal(code)

    def test_slice_block_size_multiple(self):
        """Test that tile.block_size * constant works as slice bounds"""

        @helion.kernel(use_default_config=True, static_shapes=True)
        def arange_block_size_mul(x: torch.Tensor) -> torch.Tensor:
            out = torch.zeros([x.size(0) * 2], dtype=torch.int32, device=x.device)
            ones = torch.ones_like(out)
            for tile in hl.tile(x.size(0)):
                indices_start = tile.begin * 2
                indices_end = indices_start + tile.block_size * 2
                out[indices_start:indices_end] = ones[indices_start:indices_end]
            return out

        x = torch.randn([64], device=DEVICE)
        code, result = code_and_output(arange_block_size_mul, (x,))

        expected = torch.ones(128, dtype=torch.int32, device=DEVICE)
        torch.testing.assert_close(result, expected)

        self.assertExpectedJournal(code)

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

    @unittest.skipIf(not supports_tensor_descriptor(), "TensorDescriptor not supported")
    @unittest.skipIf(
        get_tensor_descriptor_fn_name() != "tl._experimental_make_tensor_descriptor",
        "Not using experimental tensor descriptor",
    )
    def test_reduction_tensor_descriptor_indexing_block_size(self):
        x = torch.randn([64, 64], dtype=torch.float32, device=DEVICE)

        # Given block_size 4, tensor_descriptor should not actually be used
        # Convert to default pointer indexing
        code, result = code_and_output(
            reduction_sum,
            (x,),
            indexing="tensor_descriptor",
            block_size=[4],
        )

        expected = torch.sum(x, dim=1)
        torch.testing.assert_close(result, expected)
        self.assertExpectedJournal(code)

    @unittest.skipIf(not supports_tensor_descriptor(), "TensorDescriptor not supported")
    @unittest.skipIf(
        get_tensor_descriptor_fn_name() != "tl._experimental_make_tensor_descriptor",
        "Not using experimental tensor descriptor",
    )
    def test_reduction_tensor_descriptor_indexing_reduction_loop(self):
        x = torch.randn([64, 256], dtype=torch.float16, device=DEVICE)

        # Given reduction_loop 2, # of columns not compatible with tensor_descriptor
        # Convert to default pointer indexing
        code, result = code_and_output(
            reduction_sum,
            (x,),
            indexing="tensor_descriptor",
            block_size=[8],
            reduction_loops=[8],
        )

        expected = torch.sum(x, dim=1)
        torch.testing.assert_close(result, expected)
        self.assertExpectedJournal(code)

    def test_2d_slice_index(self):
        """Test both setter from scalar and getter for [:,i]"""

        @helion.kernel(use_default_config=True)
        def kernel(
            src: torch.Tensor, dst: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            N = src.shape[1]
            for i in hl.grid(N):
                dst[:, i] = 1.0  # Test setter with scalar
                src[:, i] = dst[:, i]  # Test getter from dst and setter to src
            return src, dst

        N = 128
        src = torch.zeros([1, N], device=DEVICE)
        dst = torch.zeros([1, N], device=DEVICE)

        src_result, dst_result = kernel(src, dst)

        # Both should be ones after the kernel
        expected_src = torch.ones([1, N], device=DEVICE)
        expected_dst = torch.ones([1, N], device=DEVICE)
        torch.testing.assert_close(src_result, expected_src)
        torch.testing.assert_close(dst_result, expected_dst)

    def test_2d_full_slice(self):
        """Test both setter from scalar and getter for [:,:]"""

        @helion.kernel(use_default_config=True)
        def kernel(
            src: torch.Tensor, dst: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            N = src.shape[1]
            for _ in hl.grid(N):
                dst[:, :] = 1.0  # Test setter with scalar
                src[:, :] = dst[:, :]  # Test getter from dst and setter to src
            return src, dst

        N = 128
        src = torch.zeros([1, N], device=DEVICE)
        dst = torch.zeros([1, N], device=DEVICE)

        src_result, dst_result = kernel(src, dst)

        # Both should be ones after the kernel
        expected_src = torch.ones([1, N], device=DEVICE)
        expected_dst = torch.ones([1, N], device=DEVICE)
        torch.testing.assert_close(src_result, expected_src)
        torch.testing.assert_close(dst_result, expected_dst)

    def test_1d_index(self):
        """Test both setter from scalar and getter for [i]"""

        @helion.kernel(use_default_config=True)
        def kernel(
            src: torch.Tensor, dst: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            N = src.shape[0]
            for i in hl.grid(N):
                dst[i] = 1.0  # Test setter with scalar
                src[i] = dst[i]  # Test getter from dst and setter to src
            return src, dst

        N = 128
        src = torch.zeros([N], device=DEVICE)
        dst = torch.zeros([N], device=DEVICE)

        src_result, dst_result = kernel(src, dst)

        # Both should be ones after the kernel
        expected_src = torch.ones([N], device=DEVICE)
        expected_dst = torch.ones([N], device=DEVICE)
        torch.testing.assert_close(src_result, expected_src)
        torch.testing.assert_close(dst_result, expected_dst)

    def test_1d_full_slice(self):
        """Test both setter from scalar and getter for [:] with multiple scalar types"""

        @helion.kernel(config={"block_size": 128})
        def kernel(
            src_float: torch.Tensor,
            dst_float: torch.Tensor,
            src_int: torch.Tensor,
            dst_int: torch.Tensor,
            src_symint: torch.Tensor,
            dst_symint: torch.Tensor,
        ) -> tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]:
            N = src_float.shape[0]
            for tile in hl.tile(N):
                # Test float scalar
                dst_float[:] = 1.0
                src_float[:] = dst_float[:]

                # Test int scalar
                dst_int[:] = 99
                src_int[:] = dst_int[:]

                # Test SymInt scalar
                dst_symint[:] = tile.block_size
                src_symint[:] = dst_symint[:]

            return (
                src_float,
                dst_float,
                src_int,
                dst_int,
                src_symint,
                dst_symint,
            )

        N = 128
        src_float = torch.zeros([N], device=DEVICE)
        dst_float = torch.zeros([N], device=DEVICE)
        src_int = torch.zeros([N], device=DEVICE)
        dst_int = torch.zeros([N], device=DEVICE)
        src_symint = torch.zeros([N], device=DEVICE)
        dst_symint = torch.zeros([N], device=DEVICE)

        results = kernel(
            src_float,
            dst_float,
            src_int,
            dst_int,
            src_symint,
            dst_symint,
        )

        # Check float results
        expected_float = torch.ones([N], device=DEVICE)
        torch.testing.assert_close(results[0], expected_float)
        torch.testing.assert_close(results[1], expected_float)

        # Check int results
        expected_int = torch.full([N], 99.0, device=DEVICE)
        torch.testing.assert_close(results[2], expected_int)
        torch.testing.assert_close(results[3], expected_int)

        # Check SymInt results
        expected_symint = torch.full([N], 128.0, device=DEVICE)
        torch.testing.assert_close(results[4], expected_symint)
        torch.testing.assert_close(results[5], expected_symint)

    def test_1d_slice_from_indexed_value(self):
        """buf[:] = zeros[i] - Assign slice from indexed value"""

        @helion.kernel(use_default_config=True)
        def kernel(buf: torch.Tensor, zeros: torch.Tensor) -> torch.Tensor:
            N = buf.shape[0]
            for i in hl.grid(N):
                buf[:] = zeros[i]
            return buf

        N = 128
        buf = torch.ones([N], device=DEVICE)
        zeros = torch.zeros([N], device=DEVICE)

        result = kernel(buf.clone(), zeros)
        expected = torch.zeros([N], device=DEVICE)
        torch.testing.assert_close(result, expected)

    @skipIfRocm("failure on rocm")
    @unittest.skip("takes 5+ minutes to run")
    def test_1d_indexed_value_from_slice(self):
        """buf2[i] = buf[:] - Assign slice to indexed value"""

        @helion.kernel
        def getter_kernel(buf: torch.Tensor, buf2: torch.Tensor) -> torch.Tensor:
            N = buf2.shape[0]
            for i in hl.grid(N):
                buf2[i, :] = buf[:]
            return buf2

        N = 128
        buf = torch.rand([N], device=DEVICE)
        buf2 = torch.zeros(
            [N, N], device=DEVICE
        )  # Note: Different shape to accommodate slice assignment

        result = getter_kernel(buf.clone(), buf2.clone())
        expected = buf.expand(N, N).clone()
        torch.testing.assert_close(result, expected)

    def test_1d_index_from_index(self):
        """buf[i] = zeros[i] - Index to index assignment"""

        @helion.kernel(use_default_config=True)
        def kernel(buf: torch.Tensor, zeros: torch.Tensor) -> torch.Tensor:
            N = buf.shape[0]
            for i in hl.grid(N):
                buf[i] = zeros[i]
            return buf

        N = 128
        buf = torch.ones([N], device=DEVICE)
        zeros = torch.zeros([N], device=DEVICE)

        result = kernel(buf.clone(), zeros)
        expected = torch.zeros([N], device=DEVICE)
        torch.testing.assert_close(result, expected)

    def test_mixed_slice_index(self):
        """Test both setter from scalar and getter for [i,:]"""

        @helion.kernel(use_default_config=True)
        def kernel(
            src: torch.Tensor, dst: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            N = src.shape[0]
            for i in hl.grid(N):
                dst[i, :] = 1.0  # Test setter with scalar
                src[i, :] = dst[i, :]  # Test getter from dst and setter to src
            return src, dst

        N = 32
        src = torch.zeros([N, N], device=DEVICE)
        dst = torch.zeros([N, N], device=DEVICE)

        src_result, dst_result = kernel(src, dst)

        # Both should be ones after the kernel
        expected_src = torch.ones([N, N], device=DEVICE)
        expected_dst = torch.ones([N, N], device=DEVICE)
        torch.testing.assert_close(src_result, expected_src)
        torch.testing.assert_close(dst_result, expected_dst)

    def test_strided_slice(self):
        """Test both setter from scalar and getter for strided slices [::2] and [1::3]"""

        @helion.kernel(use_default_config=True)
        def kernel(
            src1: torch.Tensor,
            dst1: torch.Tensor,
            src2: torch.Tensor,
            dst2: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            for _ in hl.grid(1):
                # Test [::2] - every other element starting from 0
                dst1[::2] = 1.0  # Test setter with scalar
                src1[::2] = dst1[::2]  # Test getter from dst and setter to src

                # Test [1::3] - every 3rd element starting from 1
                dst2[1::3] = 2.0  # Test setter with scalar
                src2[1::3] = dst2[1::3]  # Test getter from dst and setter to src
            return src1, dst1, src2, dst2

        N = 128
        src1 = torch.zeros([N], device=DEVICE)
        dst1 = torch.zeros([N], device=DEVICE)
        src2 = torch.zeros([N], device=DEVICE)
        dst2 = torch.zeros([N], device=DEVICE)

        src1_result, dst1_result, src2_result, dst2_result = kernel(
            src1, dst1, src2, dst2
        )

        # Only even indices should be ones for [::2]
        expected_src1 = torch.zeros([N], device=DEVICE)
        expected_src1[::2] = 1.0
        expected_dst1 = expected_src1.clone()
        torch.testing.assert_close(src1_result, expected_src1)
        torch.testing.assert_close(dst1_result, expected_dst1)

        # Elements at indices 1, 4, 7, ... should be twos for [1::3]
        expected_src2 = torch.zeros([N], device=DEVICE)
        expected_src2[1::3] = 2.0
        expected_dst2 = expected_src2.clone()
        torch.testing.assert_close(src2_result, expected_src2)
        torch.testing.assert_close(dst2_result, expected_dst2)

    @skipIfNormalMode("InternalError: Negative indexes")
    def test_negative_indexing(self):
        """Test both setter from scalar and getter for [-1]"""

        @helion.kernel(use_default_config=True)
        def kernel(
            src: torch.Tensor, dst: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            for _ in hl.grid(1):
                dst[-1] = 1.0  # Test setter with scalar
                src[-1] = dst[-1]  # Test getter from dst and setter to src
            return src, dst

        N = 128
        src = torch.zeros([N], device=DEVICE)
        dst = torch.zeros([N], device=DEVICE)

        src_result, dst_result = kernel(src, dst)

        # Only last element should be one
        expected_src = torch.zeros([N], device=DEVICE)
        expected_src[-1] = 1.0
        expected_dst = expected_src.clone()
        torch.testing.assert_close(src_result, expected_src)
        torch.testing.assert_close(dst_result, expected_dst)

    @skipIfNormalMode(
        "RankMismatch: Cannot assign a tensor of rank 2 to a buffer of rank 3"
    )
    def test_ellipsis_indexing(self):
        """Test both setter from scalar and getter for [..., i]"""

        @helion.kernel(use_default_config=True)
        def kernel(
            src: torch.Tensor, dst: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            N = src.shape[-1]
            for i in hl.grid(N):
                dst[..., i] = 1.0  # Test setter with scalar
                src[..., i] = dst[..., i]  # Test getter from dst and setter to src
            return src, dst

        N = 32
        src = torch.zeros([2, 3, N], device=DEVICE)
        dst = torch.zeros([2, 3, N], device=DEVICE)

        src_result, dst_result = kernel(src, dst)

        # All elements should be ones after the kernel
        expected_src = torch.ones([2, 3, N], device=DEVICE)
        expected_dst = torch.ones([2, 3, N], device=DEVICE)
        torch.testing.assert_close(src_result, expected_src)
        torch.testing.assert_close(dst_result, expected_dst)

    @skipIfNormalMode(
        "RankMismatch: Cannot assign a tensor of rank 2 to a buffer of rank 3"
    )
    def test_multi_dim_slice(self):
        """Test both setter from scalar and getter for [:, :, i]"""

        @helion.kernel(use_default_config=True)
        def kernel(
            src: torch.Tensor, dst: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            N = src.shape[-1]
            for i in hl.grid(N):
                dst[:, :, i] = 1.0  # Test setter with scalar
                src[:, :, i] = dst[:, :, i]  # Test getter from dst and setter to src
            return src, dst

        N = 32
        src = torch.zeros([2, 3, N], device=DEVICE)
        dst = torch.zeros([2, 3, N], device=DEVICE)

        src_result, dst_result = kernel(src, dst)

        # All elements should be ones after the kernel
        expected_src = torch.ones([2, 3, N], device=DEVICE)
        expected_dst = torch.ones([2, 3, N], device=DEVICE)
        torch.testing.assert_close(src_result, expected_src)
        torch.testing.assert_close(dst_result, expected_dst)

    @skipIfNormalMode(
        "RankMismatch: Expected ndim=2, but got ndim=1 - tensor value assignment shape mismatch"
    )
    def test_tensor_value(self):
        """Test both setter from tensor value and getter for [i]"""

        @helion.kernel(use_default_config=True)
        def kernel(
            src: torch.Tensor, dst: torch.Tensor, val: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            N = src.shape[0]
            for i in hl.grid(N):
                dst[i] = val  # Test setter with tensor value
                src[i] = dst[i]  # Test getter from dst and setter to src
            return src, dst

        N = 32
        src = torch.zeros([N, 4], device=DEVICE)
        dst = torch.zeros([N, 4], device=DEVICE)
        val = torch.ones([4], device=DEVICE)

        src_result, dst_result = kernel(src, dst, val)

        # All rows should be equal to val
        expected_src = val.expand(N, -1)
        expected_dst = val.expand(N, -1)
        torch.testing.assert_close(src_result, expected_src)
        torch.testing.assert_close(dst_result, expected_dst)

    def test_slice_to_slice(self):
        """buf[:] = zeros[:] - Full slice to slice assignment"""

        @helion.kernel(use_default_config=True)
        def kernel(buf: torch.Tensor, zeros: torch.Tensor) -> torch.Tensor:
            N = buf.shape[0]
            for _ in hl.grid(N):
                buf[:] = zeros[:]
            return buf

        N = 128
        buf = torch.ones([N], device=DEVICE)
        zeros = torch.zeros([N], device=DEVICE)

        result = kernel(buf.clone(), zeros)
        expected = torch.zeros([N], device=DEVICE)
        torch.testing.assert_close(result, expected)

    def test_broadcast(self):
        """Test both setter from scalar and getter for [:, i]"""

        @helion.kernel(use_default_config=True)
        def kernel(
            src: torch.Tensor, dst: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            N = src.shape[1]
            for i in hl.grid(N):
                dst[:, i] = 1.0  # Test setter with scalar (broadcast)
                src[:, i] = dst[:, i]  # Test getter from dst and setter to src
            return src, dst

        N = 32
        src = torch.zeros([N, N], device=DEVICE)
        dst = torch.zeros([N, N], device=DEVICE)

        src_result, dst_result = kernel(src, dst)

        # All elements should be ones after the kernel
        expected_src = torch.ones([N, N], device=DEVICE)
        expected_dst = torch.ones([N, N], device=DEVICE)
        torch.testing.assert_close(src_result, expected_src)
        torch.testing.assert_close(dst_result, expected_dst)

    @skipIfNormalMode("InternalError: Unexpected type <class 'slice'>")
    def test_range_slice(self):
        """Test both setter from scalar and getter for [10:20]"""

        @helion.kernel(use_default_config=True)
        def kernel(
            src: torch.Tensor, dst: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            for _ in hl.grid(1):
                dst[10:20] = 1.0  # Test setter with scalar
                src[10:20] = dst[10:20]  # Test getter from dst and setter to src
            return src, dst

        N = 128
        src = torch.zeros([N], device=DEVICE)
        dst = torch.zeros([N], device=DEVICE)

        src_result, dst_result = kernel(src, dst)

        # Only indices 10:20 should be ones
        expected_src = torch.zeros([N], device=DEVICE)
        expected_src[10:20] = 1.0
        expected_dst = expected_src.clone()
        torch.testing.assert_close(src_result, expected_src)
        torch.testing.assert_close(dst_result, expected_dst)

    @skipIfNormalMode(
        "InternalError: AssertionError in type_propagation.py - slice indexing error"
    )
    def test_range_slice_dynamic(self):
        """Test both [i:i+1] = scalar and [i] = [i:i+1] patterns"""

        @helion.kernel(use_default_config=True)
        def kernel(
            src: torch.Tensor, dst: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            N = src.shape[0]
            for i in hl.grid(N - 1):
                dst[i : i + 1] = 1.0  # Test setter with scalar to slice
                src[i] = dst[i : i + 1]  # Test getter from slice to index
            return src, dst

        N = 128
        src = torch.zeros([N], device=DEVICE)
        dst = torch.zeros([N], device=DEVICE)

        src_result, dst_result = kernel(src, dst)

        # All elements except last should be ones
        expected_src = torch.ones([N], device=DEVICE)
        expected_src[-1] = 0.0  # Last element not modified since loop goes to N-1
        expected_dst = expected_src.clone()

        torch.testing.assert_close(src_result, expected_src)
        torch.testing.assert_close(dst_result, expected_dst)

    def test_advanced_indexing_2d_integer_tensor(self):
        @helion.kernel(static_shapes=True, use_default_config=True)
        def advanced_indexing_kernel(
            x: torch.Tensor, 
            index_i: torch.Tensor, 
            index_j: torch.Tensor
        ) -> torch.Tensor:
            out = torch.empty([index_i.size(0), index_j.size(0)], 
                             device=x.device, dtype=x.dtype)
            
            for tile in hl.tile(4):
                i_indices = index_i[:, :]
                j_indices = index_j[:]
                
                values = x[i_indices, j_indices]
                out[:, :] = values
            
            return out

        x = torch.tensor([[1.0, 2.0, 3.0], 
                          [4.0, 5.0, 6.0], 
                          [7.0, 8.0, 9.0]], device=DEVICE)
        
        index_i = torch.tensor([[0], [1]], device=DEVICE, dtype=torch.long)
        index_j = torch.tensor([2], device=DEVICE, dtype=torch.long)
        
        expected = torch.tensor([[3.0], [6.0]], device=DEVICE)
        
        code, result = code_and_output(
            advanced_indexing_kernel,
            (x, index_i, index_j),
        )
        
        torch.testing.assert_close(result, expected)
        self.assertExpectedJournal(code)
    
    def test_advanced_indexing_broadcast_shapes(self):
        """Test advanced indexing with multiple tensor indices that broadcast"""
        @helion.kernel(static_shapes=True, use_default_config=True)
        def advanced_indexing_broadcast_kernel(
            x: torch.Tensor,  # Shape: [5, 4, 6]
            idx0: torch.Tensor,  # Shape: [3, 1] - will broadcast
            idx1: torch.Tensor,  # Shape: [1, 2] - will broadcast
        ) -> torch.Tensor:
            """
            Test indexing with broadcasting tensor indices.
            idx0 shape [3, 1] and idx1 shape [1, 2] broadcast to [3, 2]
            Result shape should be [3, 2, 6] (broadcast shape + remaining dims)
            """
            # Create output tensor with the expected shape
            out = torch.empty([3, 2, 6], device=x.device, dtype=x.dtype)
            
            for tile in hl.tile(4):
                # Load the index tensors
                idx0_tile = idx0[:, :]
                idx1_tile = idx1[:, :]
                
                # Perform advanced indexing with broadcasting directly on x
                values = x[idx0_tile, idx1_tile, :]
                out[:, :, :] = values
            
            return out
        
        x = torch.randn(5, 4, 6, device=DEVICE)
        
        # Create indices that need to broadcast
        idx0 = torch.tensor([[0], [2], [4]], device=DEVICE, dtype=torch.long)  # Shape: [3, 1]
        idx1 = torch.tensor([[1, 3]], device=DEVICE, dtype=torch.long)  # Shape: [1, 2]
        
        # Expected: indices broadcast to select x[0,1,:], x[0,3,:], x[2,1,:], x[2,3,:], x[4,1,:], x[4,3,:]
        # Resulting in shape [3, 2, 6]
        expected = x[idx0, idx1, :]
        
        code, result = code_and_output(
            advanced_indexing_broadcast_kernel,
            (x, idx0, idx1),
        )
        
        torch.testing.assert_close(result, expected)
        self.assertExpectedJournal(code)
    
    def test_advanced_indexing_7d_integer_tensor(self):
        @helion.kernel(static_shapes=True, use_default_config=True)
        def advanced_indexing_7d_kernel(
            x: torch.Tensor,  # Shape: [3, 2, 5, 4, 3, 6, 2]
            dim0_indices: torch.Tensor,  # Shape: [1, 1]
            dim1_indices: torch.Tensor,  # Shape: [2, 1]
            dim2_indices: torch.Tensor,  # Shape: [3]
            dim3_indices: torch.Tensor,  # Shape: [1, 1, 1]
            dim4_indices: torch.Tensor,  # Shape: [1, 3]
            dim5_indices: torch.Tensor,  # Shape: [4, 1, 1]
            dim6_indices: torch.Tensor   # Shape: [1]
        ) -> torch.Tensor:
            """
            Perform advanced indexing on a 7D tensor with index tensors of different shapes.
            Result shape will be [4, 2, 3] due to broadcasting of all index tensors.
            """
            # Calculate output shape based on broadcasting all index tensors
            # The shapes are:
            # dim0_indices: [1, 1]
            # dim1_indices: [2, 1]
            # dim2_indices: [3]
            # dim3_indices: [1, 1, 1]
            # dim4_indices: [1, 3]
            # dim5_indices: [4, 1, 1]
            # dim6_indices: [1]
            # Broadcasting all together gives shape [4, 2, 3]
            out = torch.empty([4, 2, 3], device=x.device, dtype=x.dtype)
            
            for tile in hl.tile(1):
                # Load index tensors - use full slicing to preserve shapes
                idx0 = dim0_indices[:, :]       # Keep shape [1, 1]
                idx1 = dim1_indices[:, :]       # Keep shape [2, 1]
                idx2 = dim2_indices[:]          # Keep shape [3]
                idx3 = dim3_indices[:, :, :]    # Keep shape [1, 1, 1]
                idx4 = dim4_indices[:, :]       # Keep shape [1, 3]
                idx5 = dim5_indices[:, :, :]    # Keep shape [4, 1, 1]
                idx6 = dim6_indices[:]          # Keep shape [1]
                
                values = x[idx0, idx1, idx2, idx3, idx4, idx5, idx6]
                out[:, :, :] = values
            
            return out
        
        total_elements = 3 * 2 * 5 * 4 * 3 * 6 * 2
        x = torch.arange(total_elements, dtype=torch.float32, device=DEVICE).reshape(3, 2, 5, 4, 3, 6, 2)
        
        # Create index tensors with different shapes that need to broadcast
        dim0_indices = torch.tensor([[2]], device=DEVICE, dtype=torch.long)            # Shape: [1, 1]
        dim1_indices = torch.tensor([[0], [1]], device=DEVICE, dtype=torch.long)       # Shape: [2, 1]
        dim2_indices = torch.tensor([1, 2, 4], device=DEVICE, dtype=torch.long)        # Shape: [3]
        dim3_indices = torch.tensor([[[3]]], device=DEVICE, dtype=torch.long)          # Shape: [1, 1, 1]
        dim4_indices = torch.tensor([[0, 1, 2]], device=DEVICE, dtype=torch.long)      # Shape: [1, 3]
        dim5_indices = torch.tensor([[[0]], [[2]], [[4]], [[5]]], device=DEVICE, dtype=torch.long)  # Shape: [4, 1, 1]
        dim6_indices = torch.tensor([0], device=DEVICE, dtype=torch.long)              # Shape: [1]
        
        # Expected result - PyTorch can handle this directly!
        expected = x[dim0_indices, dim1_indices, dim2_indices, dim3_indices, dim4_indices, dim5_indices, dim6_indices]
        
        code, result = code_and_output(
            advanced_indexing_7d_kernel,
            (x, dim0_indices, dim1_indices, dim2_indices, 
             dim3_indices, dim4_indices, dim5_indices, dim6_indices),
        )
        
        torch.testing.assert_close(result, expected)
        self.assertExpectedJournal(code)
    
    def test_advanced_indexing_mixed_index_types(self):
        """Test mixed indexing with integer tensors, tiles, and dynamic slices in various positions"""
        @helion.kernel(static_shapes=True, use_default_config=True, config={"block_size": 4})
        def mixed_indexing_kernel(
            x: torch.Tensor,           # Shape: [5, 8, 10, 6, 12, 4, 9, 7, 16, 11]
            int_tensor_1: torch.Tensor, # Shape: [3, 2]
            int_tensor_2: torch.Tensor, # Shape: [1, 2]
        ) -> torch.Tensor:
            """
            Harder mixed pattern with two tiles and two dynamic slices:
            x[idx1, :, 0, tile, a:(a+tile.block_size), tile, idx2, :, a:(a+tile.block_size), 0]
            Output dims = broadcast(idx1, idx2) + [x.size(1), bs, bs, bs, x.size(7), bs]
            where bs = tile.block_size (fixed to 4 via config).
            """
            d1 = x.size(1)
            d7 = x.size(7)
            out = torch.zeros([3, 2, d1, 4, 4, 4, d7, 4], device=x.device, dtype=x.dtype)

            for tile in hl.tile(1):
                idx1 = int_tensor_1[:, :]
                idx2 = int_tensor_2[:, :]
                a = tile.begin
                values = x[idx1, :, 0, tile, a : (a + tile.block_size), tile, idx2, :, a : (a + tile.block_size), 0]
                out[:, :, :, :, :, :, :, :] = values

            return out

        # Create test tensor with 10D shape matching indices above
        x = torch.randn(5, 8, 10, 6, 12, 4, 9, 7, 16, 11, device=DEVICE)

        # Create integer tensor indices
        int_tensor_1 = torch.tensor([[0, 1], [2, 3], [4, 0]], device=DEVICE, dtype=torch.long)  # [3, 2]
        int_tensor_2 = torch.tensor([[2, 5]], device=DEVICE, dtype=torch.long)                  # [1, 2]

        # Expected: tile and dynamic slice dims become 0:4; scalars remove dims
        expected = x[int_tensor_1, :, 0, 0:4, 0:4, 0:4, int_tensor_2, :, 0:4, 0]

        code, result = code_and_output(
            mixed_indexing_kernel,
            (x, int_tensor_1, int_tensor_2),
        )

        torch.testing.assert_close(result, expected)
        self.assertExpectedJournal(code)


if __name__ == "__main__":
    unittest.main()
