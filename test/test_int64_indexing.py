"""Tests for int64 indexing with different indexing strategies.

This module tests that int64 index_dtype works correctly with:
- pointer indexing (should use int64 throughout)
- block_ptr indexing (should fall back to pointer since block_ptr only supports int32)
- tensor_descriptor indexing (should fall back to pointer since tensor_descriptor only supports int32)

The key cases to test:
1. index_dtype=torch.int64 setting
2. Explicit casting of indices to int64
3. Using tile.index with int64
4. Loading indices as int64 tensors
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

import helion
from helion import _compat
from helion._testing import DEVICE
from helion._testing import RefEagerTestBase
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
from helion._testing import skipIfCpu
from helion._testing import skipIfRefEager
from helion._testing import skipIfTileIR
from helion._testing import skipUnlessTensorDescriptor
import helion.language as hl


@onlyBackends(["triton"])
class TestInt64Indexing(RefEagerTestBase, TestCase):
    """Test int64 indexing with different indexing strategies."""

    @skipIfRefEager("Test checks generated code")
    def test_int64_pointer_indexing(self):
        """Test int64 index_dtype with pointer indexing."""

        @helion.kernel(index_dtype=torch.int64)
        def add_kernel_int64(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile_m, tile_n in hl.tile(x.size()):
                out[tile_m, tile_n] = x[tile_m, tile_n] + y[tile_m, tile_n]
            return out

        x = torch.randn(64, 64, device=DEVICE)
        y = torch.randn(64, 64, device=DEVICE)

        code, result = code_and_output(
            add_kernel_int64, (x, y), indexing="pointer", block_size=[16, 16]
        )

        expected = x + y
        torch.testing.assert_close(result, expected)

        # Verify int64 type is used in generated code
        self.assertIn("tl.int64", code)
        self.assertExpectedJournal(code)

    @skipIfRefEager("Test checks generated code")
    @skipIfTileIR("TileIR does not support block_ptr indexing")
    @patch.object(_compat, "_supports_tensor_descriptor", lambda: False)
    def test_int64_block_ptr_falls_back_to_pointer(self):
        """Test that int64 index_dtype causes block_ptr to fall back to pointer indexing.

        Triton's make_block_ptr only supports 32-bit offsets, so when int64 is requested,
        we must fall back to pointer indexing.
        """

        @helion.kernel(index_dtype=torch.int64)
        def add_kernel_int64(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile_m, tile_n in hl.tile(x.size()):
                out[tile_m, tile_n] = x[tile_m, tile_n] + y[tile_m, tile_n]
            return out

        x = torch.randn(64, 64, device=DEVICE)
        y = torch.randn(64, 64, device=DEVICE)

        code, result = code_and_output(
            add_kernel_int64, (x, y), indexing="block_ptr", block_size=[16, 16]
        )

        expected = x + y
        torch.testing.assert_close(result, expected)

        # Verify block_ptr is NOT used (falls back to pointer)
        self.assertNotIn("tl.make_block_ptr", code)
        # Verify int64 type is used
        self.assertIn("tl.int64", code)
        self.assertExpectedJournal(code)

    @skipIfRefEager("Test checks generated code")
    @skipUnlessTensorDescriptor("Tensor descriptor support is required")
    def test_int64_tensor_descriptor_falls_back_to_pointer(self):
        """Test that int64 index_dtype causes tensor_descriptor to fall back to pointer indexing.

        Tensor descriptors only support 32-bit offsets, so when int64 is requested,
        we must fall back to pointer indexing.
        """

        @helion.kernel(index_dtype=torch.int64)
        def add_kernel_int64(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile_m, tile_n in hl.tile(x.size()):
                out[tile_m, tile_n] = x[tile_m, tile_n] + y[tile_m, tile_n]
            return out

        x = torch.randn(64, 64, device=DEVICE)
        y = torch.randn(64, 64, device=DEVICE)

        code, result = code_and_output(
            add_kernel_int64, (x, y), indexing="tensor_descriptor", block_size=[16, 16]
        )

        expected = x + y
        torch.testing.assert_close(result, expected)

        # Verify tensor_descriptor is NOT used (falls back to pointer)
        self.assertNotIn("make_tensor_descriptor", code)
        self.assertNotIn("_experimental_make_tensor_descriptor", code)
        # Verify int64 type is used
        self.assertIn("tl.int64", code)
        self.assertExpectedJournal(code)

    @skipIfRefEager("Test checks generated code")
    @skipIfTileIR("TileIR does not support block_ptr indexing")
    @patch.object(_compat, "_supports_tensor_descriptor", lambda: False)
    def test_int64_block_ptr_with_tile_index_falls_back(self):
        """Test int64 indexing with tile.index pattern falls back from block_ptr."""

        @helion.kernel(index_dtype=torch.int64)
        def pairwise_add_int64(x: torch.Tensor) -> torch.Tensor:
            out = x.new_empty([x.size(0) - 1])
            for tile in hl.tile(out.size(0)):
                out[tile] = x[tile] + x[tile.index + 1]
            return out

        x = torch.randn([500], device=DEVICE)
        code, result = code_and_output(
            pairwise_add_int64, (x,), indexing="block_ptr", block_size=32
        )

        expected = x[:-1] + x[1:]
        torch.testing.assert_close(result, expected)

        # Verify block_ptr is NOT used (falls back to pointer)
        self.assertNotIn("tl.make_block_ptr", code)
        self.assertIn("tl.int64", code)
        self.assertExpectedJournal(code)

    @skipIfRefEager("Test checks generated code")
    @skipUnlessTensorDescriptor("Tensor descriptor support is required")
    @skipIfTileIR(
        "TileIR does not support descriptor with index not multiple of tile size"
    )
    def test_int64_tensor_descriptor_with_tile_index_falls_back(self):
        """Test int64 indexing with tile.index pattern falls back from tensor_descriptor."""

        @helion.kernel(index_dtype=torch.int64)
        def pairwise_add_2d_int64(x: torch.Tensor) -> torch.Tensor:
            M, N = x.size()
            out = x.new_empty(M - 10, N)
            for tile_m in hl.tile(out.size(0)):
                # Use tile + offset pattern
                tile_offset = tile_m + 10
                out[tile_m, :] = x[tile_offset, :]
            return out

        x = torch.randn([128, 64], device=DEVICE)
        code, result = code_and_output(
            pairwise_add_2d_int64, (x,), indexing="tensor_descriptor", block_size=32
        )

        expected = x[10:, :]
        torch.testing.assert_close(result, expected)

        # Verify tensor_descriptor is NOT used (falls back to pointer)
        self.assertNotIn("make_tensor_descriptor", code)
        self.assertIn("tl.int64", code)
        self.assertExpectedJournal(code)

    @skipIfRefEager("Test checks generated code")
    @skipIfTileIR("TileIR does not support block_ptr indexing")
    @patch.object(_compat, "_supports_tensor_descriptor", lambda: False)
    def test_int64_block_ptr_with_explicit_cast(self):
        """Test int64 indexing with explicit .to(torch.int64) cast falls back."""

        @helion.kernel(index_dtype=torch.int64)
        def kernel_with_cast(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile_m, tile_n in hl.tile(x.size()):
                # Use explicit cast
                idx_m = tile_m.index.to(torch.int64)
                idx_n = tile_n.index.to(torch.int64)
                out[idx_m, idx_n] = x[tile_m, tile_n] * 2.0
            return out

        x = torch.randn(64, 64, device=DEVICE)
        code, result = code_and_output(
            kernel_with_cast, (x,), indexing="block_ptr", block_size=[16, 16]
        )

        expected = x * 2.0
        torch.testing.assert_close(result, expected)
        # Falls back to pointer, should still use int64
        self.assertNotIn("tl.make_block_ptr", code)
        self.assertIn("tl.int64", code)
        self.assertExpectedJournal(code)

    @skipIfRefEager("Test checks generated code")
    @skipIfCpu("fails on Triton CPU backend")
    def test_int64_with_loaded_indices(self):
        """Test int64 indexing when indices are loaded from an int64 tensor."""

        @helion.kernel(index_dtype=torch.int64)
        def gather_kernel_int64(
            input_tensor: torch.Tensor,
            index_tensor: torch.Tensor,
        ) -> torch.Tensor:
            N, M = input_tensor.size()
            K = index_tensor.size(1)
            out = torch.empty(
                N, K, dtype=input_tensor.dtype, device=input_tensor.device
            )
            for tile_n, tile_k in hl.tile([N, K]):
                # indices loaded as int64
                indices = index_tensor[tile_n, tile_k]
                out[tile_n, tile_k] = input_tensor[tile_n.index[:, None], indices]
            return out

        N, M, K = 32, 64, 16
        input_tensor = torch.randn(N, M, device=DEVICE)
        index_tensor = torch.randint(0, M, (N, K), device=DEVICE, dtype=torch.int64)

        code, result = code_and_output(
            gather_kernel_int64,
            (input_tensor, index_tensor),
            indexing="pointer",
            block_size=[8, 8],
        )

        expected = torch.gather(input_tensor, 1, index_tensor)
        torch.testing.assert_close(result, expected)
        self.assertIn("tl.int64", code)
        self.assertExpectedJournal(code)

    @skipIfRefEager("Test checks generated code")
    @skipIfTileIR("TileIR does not support block_ptr indexing")
    @patch.object(_compat, "_supports_tensor_descriptor", lambda: False)
    def test_int64_block_ptr_with_reduction_falls_back(self):
        """Test int64 indexing with block_ptr and reduction loops falls back to pointer."""

        @helion.kernel(index_dtype=torch.int64)
        def reduction_sum_int64(x: torch.Tensor) -> torch.Tensor:
            m, _ = x.size()
            out = torch.empty([m], device=x.device, dtype=x.dtype)
            for tile in hl.tile(x.size(0)):
                out[tile] = x[tile, :].sum(-1)
            return out

        x = torch.randn([64, 128], device=DEVICE)

        code, result = code_and_output(
            reduction_sum_int64,
            (x,),
            indexing="block_ptr",
            block_size=[8],
        )

        expected = x.sum(dim=1)
        torch.testing.assert_close(result, expected)
        # Verify block_ptr is NOT used (falls back to pointer)
        self.assertNotIn("tl.make_block_ptr", code)
        self.assertIn("tl.int64", code)
        self.assertExpectedJournal(code)

    @skipIfRefEager("Test checks generated code")
    @skipUnlessTensorDescriptor("Tensor descriptor support is required")
    def test_int64_tensor_descriptor_with_reduction_falls_back(self):
        """Test int64 indexing with tensor_descriptor and reduction loops falls back to pointer."""

        @helion.kernel(index_dtype=torch.int64)
        def reduction_sum_int64(x: torch.Tensor) -> torch.Tensor:
            m, _ = x.size()
            out = torch.empty([m], device=x.device, dtype=x.dtype)
            for tile in hl.tile(x.size(0)):
                out[tile] = x[tile, :].sum(-1)
            return out

        x = torch.randn([64, 128], device=DEVICE)

        code, result = code_and_output(
            reduction_sum_int64,
            (x,),
            indexing="tensor_descriptor",
            block_size=[8],
        )

        expected = x.sum(dim=1)
        torch.testing.assert_close(result, expected)
        # Verify tensor_descriptor is NOT used (falls back to pointer)
        self.assertNotIn("make_tensor_descriptor", code)
        self.assertIn("tl.int64", code)
        self.assertExpectedJournal(code)

    @skipIfRefEager("Test checks generated code")
    @skipIfTileIR("TileIR does not support block_ptr indexing")
    @patch.object(_compat, "_supports_tensor_descriptor", lambda: False)
    def test_int64_block_ptr_matmul_falls_back(self):
        """Test int64 indexing with block_ptr in a matmul pattern falls back to pointer."""

        @helion.kernel(index_dtype=torch.int64)
        def matmul_int64(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, k = x.size()
            k2, n = y.size()
            assert k == k2
            out = torch.empty(
                [m, n],
                dtype=torch.promote_types(x.dtype, y.dtype),
                device=x.device,
            )
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc.to(out.dtype)
            return out

        x = torch.randn((64, 64), device=DEVICE, dtype=torch.float16)
        y = torch.randn((64, 64), device=DEVICE, dtype=torch.float16)

        code, result = code_and_output(
            matmul_int64, (x, y), indexing="block_ptr", block_size=[16, 16, 16]
        )

        expected = torch.matmul(x, y)
        torch.testing.assert_close(result, expected, atol=1e-2, rtol=1e-2)
        # Verify block_ptr is NOT used (falls back to pointer)
        self.assertNotIn("tl.make_block_ptr", code)
        self.assertIn("tl.int64", code)
        self.assertExpectedJournal(code)

    @skipIfRefEager("Test checks generated code")
    @skipUnlessTensorDescriptor("Tensor descriptor support is required")
    def test_int64_tensor_descriptor_matmul_falls_back(self):
        """Test int64 indexing with tensor_descriptor in a matmul pattern falls back to pointer."""

        @helion.kernel(index_dtype=torch.int64)
        def matmul_int64(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, k = x.size()
            k2, n = y.size()
            assert k == k2
            out = torch.empty(
                [m, n],
                dtype=torch.promote_types(x.dtype, y.dtype),
                device=x.device,
            )
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc.to(out.dtype)
            return out

        x = torch.randn((64, 64), device=DEVICE, dtype=torch.float16)
        y = torch.randn((64, 64), device=DEVICE, dtype=torch.float16)

        code, result = code_and_output(
            matmul_int64, (x, y), indexing="tensor_descriptor", block_size=[16, 16, 16]
        )

        expected = torch.matmul(x, y)
        torch.testing.assert_close(result, expected, atol=1e-2, rtol=1e-2)
        # Verify tensor_descriptor is NOT used (falls back to pointer)
        self.assertNotIn("make_tensor_descriptor", code)
        self.assertIn("tl.int64", code)
        self.assertExpectedJournal(code)

    @skipIfRefEager("Test checks generated code")
    @skipIfTileIR("TileIR does not support block_ptr indexing")
    @patch.object(_compat, "_supports_tensor_descriptor", lambda: False)
    def test_int32_block_ptr_still_works(self):
        """Test that int32 (default) still uses block_ptr when requested."""

        @helion.kernel(index_dtype=torch.int32)
        def add_kernel_int32(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile_m, tile_n in hl.tile(x.size()):
                out[tile_m, tile_n] = x[tile_m, tile_n] + y[tile_m, tile_n]
            return out

        x = torch.randn(64, 64, device=DEVICE)
        y = torch.randn(64, 64, device=DEVICE)

        code, result = code_and_output(
            add_kernel_int32, (x, y), indexing="block_ptr", block_size=[16, 16]
        )

        expected = x + y
        torch.testing.assert_close(result, expected)

        # Verify block_ptr IS used with int32
        self.assertIn("tl.make_block_ptr", code)
        # Should NOT have int64 cast
        self.assertNotIn("tl.int64", code)
        self.assertExpectedJournal(code)


if __name__ == "__main__":
    unittest.main()
