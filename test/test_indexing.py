from __future__ import annotations

import unittest

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

    def test_colon_slice_reshape(self):
        @helion.kernel(use_default_config=True, static_shapes=True)
        def matmul_bf16_int4(A: Tensor, B: Tensor) -> Tensor:
            M, K = A.shape
            _, N = B.shape

            C = torch.zeros(M, N, dtype=torch.bfloat16, device=A.device)
            block_size_k_packed = hl.register_block_size(K // 2)
            block_size_n = hl.register_block_size(N)

            # Use Helion to tile the computation
            for tile_m in hl.tile(M):
                for tile_n in hl.tile(N, block_size=block_size_n):
                    acc = hl.zeros((tile_m, tile_n), dtype=torch.float32)

                    for tile_k_packed in hl.tile(K // 2, block_size=block_size_k_packed):
                        # Load packed int8 data from B
                        b_tile = B[tile_k_packed, tile_n]  # [BLOCK_SIZE_K//2, BLOCK_SIZE_N]
                        
                        # Convert to uint8 to ensure logical right shift (not arithmetic)
                        # This is needed because int8 >> 4 does sign extension for negative values
                        b_tile_uint8 = b_tile.to(torch.uint8)
                        
                        # Extract low and high 4-bit values
                        b_lo = b_tile_uint8 & 0x0F  # Extract low 4 bits
                        b_hi = (b_tile_uint8 >> 4) & 0x0F  # Extract high 4 bits
                        
                        # Convert to bfloat16
                        b_lo_bf16 = b_lo.to(torch.bfloat16)  # [BLOCK_SIZE_K//2, BLOCK_SIZE_N]
                        b_hi_bf16 = b_hi.to(torch.bfloat16)  # [BLOCK_SIZE_K//2, BLOCK_SIZE_N]
                        
                        # Use torch.stack to interleave the low and high bits
                        # This creates a tensor of shape [BLOCK_SIZE_K//2, 2, BLOCK_SIZE_N]
                        # where [:, 0, :] contains b_lo and [:, 1, :] contains b_hi
                        b_bf16 = torch.stack([b_lo_bf16, b_hi_bf16], dim=1)
                        
                        # Reshape from [BLOCK_SIZE_K//2, 2, BLOCK_SIZE_N] to [BLOCK_SIZE_K, BLOCK_SIZE_N]
                        # The default reshape will interleave correctly: [i, 0, :] then [i, 1, :] for each i
                        b_bf16_reshaped = b_bf16.reshape(tile_k_packed.block_size * 2, tile_n.block_size)
                        
                        # Load corresponding tiles from A (need to load twice the packed tile size)
                        # We need to map tile_k_packed to the corresponding range in A
                        # Use arange to create indices for the second dimension
                        a_tile_begin = tile_k_packed.begin * 2
                        a_tile_len = tile_k_packed.block_size * 2
                        a_tile = A[tile_m, a_tile_begin:(a_tile_begin + a_tile_len)]  # [BLOCK_SIZE_M, BLOCK_SIZE_K]
                        
                        acc = acc + hl.dot(a_tile, b_bf16_reshaped)  # [BLOCK_SIZE_M, BLOCK_SIZE_N]

                    C[tile_m, tile_n] = acc.to(torch.bfloat16)
            
            return C

        # Test with smaller matrices for validation
        M, K, N = 256, 512, 256
        A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
        # Create packed int4 matrix B (K//2 x N)
        # Since we're packing two 4-bit values, we can use values from 0 to 255 
        # but need to use uint8 to store them properly
        B_packed_uint8 = torch.randint(0, 256, (K // 2, N), dtype=torch.uint8, device="cuda")
        # Convert to int8 for the kernel (bit pattern remains the same)
        B_packed = B_packed_uint8.to(torch.int8)
        
        # Reference implementation: unpack B and compute matmul
        B_unpacked = torch.zeros(K, N, dtype=torch.bfloat16, device="cuda")
        for i in range(K // 2):
            # Extract low and high 4-bit values from uint8 representation
            b_lo = (B_packed_uint8[i] & 0x0F).to(torch.bfloat16)
            b_hi = ((B_packed_uint8[i] >> 4) & 0x0F).to(torch.bfloat16)
            # Place in unpacked matrix
            B_unpacked[i * 2] = b_lo
            B_unpacked[i * 2 + 1] = b_hi
        
        # Compute reference result
        expected = torch.matmul(A, B_unpacked)
        
        # Run the kernel
        result = matmul_bf16_int4(A, B_packed)
        
        # Check accuracy with appropriate tolerance for bfloat16
        torch.testing.assert_close(result, expected, rtol=1e-2, atol=1e-2)

if __name__ == "__main__":
    unittest.main()
