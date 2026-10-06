from __future__ import annotations

import operator
import unittest
from unittest.mock import patch

import torch

import helion
from helion._testing import DEVICE
from helion._testing import LONG_INT_TYPE
from helion._testing import RefEagerTestBase
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
from helion._testing import skipIfNotCUDA
from helion._testing import skipIfRefEager
from helion._testing import skipIfRocm
from helion._testing import skipIfTileIR
from helion._testing import skipUnlessTensorDescriptor
from helion._testing import xfailIfPallas
import helion.language as hl
from helion.runtime.settings import _get_backend


@helion.kernel()
def atomic_add_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Test basic atomic_add functionality."""
    for i in hl.tile(x.size(0)):
        hl.atomic_add(x, [i], y[i])
    return x


@helion.kernel(static_shapes=True)
def atomic_add_overlap_kernel(
    x: torch.Tensor, y: torch.Tensor, indices: torch.Tensor
) -> torch.Tensor:
    """Test atomic_add with overlapping indices."""
    for i in hl.tile([y.size(0)]):
        idx = indices[i]
        hl.atomic_add(x, [idx], y[i])
    return x


@helion.kernel()
def atomic_add_2d_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Test atomic_add with 2D indexing."""
    for i, j in hl.tile([y.size(0), y.size(1)]):
        hl.atomic_add(x, [i, j], y[i, j])
    return x


@helion.kernel()
def atomic_add_float_kernel(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Test atomic_add with a float constant value and reading from lookup"""
    for i in hl.tile(indices.size(0)):
        idx = indices[i]
        hl.atomic_add(x, [idx], 2.0)
    return x


@helion.kernel(static_shapes=True)
def split_k_atomic_add_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Split-K matmul with atomic_add into a non-zero output."""
    m, k = x.size()
    k2, n = y.size()
    out = torch.ones([m, n], dtype=x.dtype, device=x.device)
    for tile_m, tile_n, tile_k in hl.tile([m, n, k]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for inner_k in hl.tile(tile_k.begin, tile_k.end):
            acc = torch.addmm(acc, x[tile_m, inner_k], y[inner_k, tile_n])
        hl.atomic_add(out, [tile_m, tile_n], acc.to(x.dtype))
    return out


@helion.kernel()
def atomic_add_f32_into_bf16_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Test atomic_add where value dtype (float32) differs from output (bfloat16)."""
    m, n = x.size()
    out = torch.zeros([m, n], dtype=x.dtype, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        acc = acc + x[tile_m, tile_n].to(torch.float32)
        acc = acc + y[tile_m, tile_n].to(torch.float32)
        hl.atomic_add(out, [tile_m, tile_n], acc)
    return out


@helion.kernel()
def atomic_add_w_tile_attr(x: torch.Tensor) -> torch.Tensor:
    """Test atomic_add where the index is a symbolic int"""
    y = torch.zeros_like(x, device=x.device, dtype=torch.int32)
    for tile in hl.tile(x.size(0)):
        hl.atomic_add(y, [tile.begin], 1)
    return y


@helion.kernel()
def atomic_add_tile_begin_reduce_other_axis(x: torch.Tensor) -> torch.Tensor:
    out = torch.zeros([x.size(0)], device=x.device, dtype=x.dtype)
    for tile_m, tile_n in hl.tile([x.size(0), x.size(1)]):
        hl.atomic_add(out, [tile_m.begin], x[tile_m, tile_n])
    return out


@helion.kernel()
def atomic_add_1d_tensor_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Test atomic_add where the index is a 1D tensor"""
    m, n = x.shape
    n = hl.specialize(n)

    z = torch.zeros([n], dtype=x.dtype, device=x.device)

    for tile_m in hl.tile(m):
        x_tile = x[tile_m, :].to(torch.float32)
        y_tile = y[tile_m, :].to(torch.float32)
        z_vec = torch.sum(x_tile * y_tile, dim=0).to(x.dtype)
        hl.atomic_add(z, [hl.arange(0, n)], z_vec)

    return z


@helion.kernel()
def atomic_add_1d_tensor_offset_kernel(
    x: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    m, n = x.shape
    n = hl.specialize(n)

    z = torch.zeros([n + 16], dtype=x.dtype, device=x.device)

    for tile_m in hl.tile(m):
        x_tile = x[tile_m, :].to(torch.float32)
        y_tile = y[tile_m, :].to(torch.float32)
        z_vec = torch.sum(x_tile * y_tile, dim=0).to(x.dtype)
        hl.atomic_add(z, [hl.arange(16, n + 16)], z_vec)

    return z


@helion.kernel()
def atomic_add_1d_tensor_step_arange_kernel(x: torch.Tensor) -> torch.Tensor:
    n = x.size(0)
    n = hl.specialize(n)

    z = torch.zeros([2 * n], dtype=x.dtype, device=x.device)
    for _tile in hl.tile(1):
        hl.atomic_add(z, [hl.arange(0, 2 * n, step=2)], x)
    return z


@helion.kernel()
def atomic_add_tensor_index_kernel(
    values: torch.Tensor, indices: torch.Tensor
) -> torch.Tensor:
    n = values.size(0)
    out = torch.zeros([n], dtype=values.dtype, device=values.device)
    for tile in hl.tile(n):
        hl.atomic_add(out, [indices[tile]], values[tile])
    return out


# New kernels for other atomics


@helion.kernel()
def atomic_and_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    for i in hl.tile(x.size(0)):
        hl.atomic_and(x, [i], y[i])
    return x


@helion.kernel()
def atomic_or_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    for i in hl.tile(x.size(0)):
        hl.atomic_or(x, [i], y[i])
    return x


@helion.kernel()
def atomic_xor_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    for i in hl.tile(x.size(0)):
        hl.atomic_xor(x, [i], y[i])
    return x


@helion.kernel()
def atomic_xchg_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    for i in hl.tile(x.size(0)):
        hl.atomic_xchg(x, [i], y[i])
    return x


@helion.kernel()
def atomic_max_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    for i in hl.tile(x.size(0)):
        hl.atomic_max(x, [i], y[i])
    return x


@helion.kernel()
def atomic_max_return_kernel(
    x: torch.Tensor, y: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    for i in hl.tile(x.size(0)):
        out[i] = hl.atomic_max(x, [i], y[i])
    return out


@helion.kernel()
def atomic_min_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    for i in hl.tile(x.size(0)):
        hl.atomic_min(x, [i], y[i])
    return x


@helion.kernel()
def atomic_cas_kernel(
    x: torch.Tensor, y: torch.Tensor, expect: torch.Tensor
) -> torch.Tensor:
    for i in hl.tile(x.size(0)):
        hl.atomic_cas(x, [i], expect[i], y[i])
    return x


# 2D kernels for tensor descriptor atomic tests (TD requires ndim >= 2 + static_shapes)


@helion.kernel(static_shapes=True)
def atomic_add_2d_td_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    for i, j in hl.tile([x.size(0), x.size(1)]):
        hl.atomic_add(x, [i, j], y[i, j])
    return x


@helion.kernel(static_shapes=True)
def atomic_and_2d_td_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    for i, j in hl.tile([x.size(0), x.size(1)]):
        hl.atomic_and(x, [i, j], y[i, j])
    return x


@helion.kernel(static_shapes=True)
def atomic_or_2d_td_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    for i, j in hl.tile([x.size(0), x.size(1)]):
        hl.atomic_or(x, [i, j], y[i, j])
    return x


@helion.kernel(static_shapes=True)
def atomic_xor_2d_td_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    for i, j in hl.tile([x.size(0), x.size(1)]):
        hl.atomic_xor(x, [i, j], y[i, j])
    return x


@helion.kernel(static_shapes=True)
def atomic_max_2d_td_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    for i, j in hl.tile([x.size(0), x.size(1)]):
        hl.atomic_max(x, [i, j], y[i, j])
    return x


@helion.kernel(static_shapes=True)
def atomic_min_2d_td_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    for i, j in hl.tile([x.size(0), x.size(1)]):
        hl.atomic_min(x, [i, j], y[i, j])
    return x


@helion.kernel(static_shapes=True)
def atomic_xchg_2d_td_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    for i, j in hl.tile([x.size(0), x.size(1)]):
        hl.atomic_xchg(x, [i, j], y[i, j])
    return x


@onlyBackends(["triton", "cute", "pallas"])
class TestAtomicOperations(RefEagerTestBase, TestCase):
    def test_basic_atomic_add(self):
        x = torch.zeros(10, device=DEVICE)
        y = torch.ones(10, device=DEVICE)
        args = (x, y)

        code, result = code_and_output(
            atomic_add_kernel,
            args,
            block_sizes=[32],
        )

        expected = torch.ones(10, device=DEVICE)
        torch.testing.assert_close(result, expected)
        if _get_backend() == "triton":
            self.assertIn("tl.atomic_add", code)

    @xfailIfPallas("view-backed atomic_add targets are not supported on Pallas")
    def test_basic_atomic_add_strided_target(self):
        x_base = torch.zeros(16, device=DEVICE)
        x = x_base[::2]
        y = torch.ones(8, device=DEVICE)

        code, result = code_and_output(
            atomic_add_kernel,
            (x, y),
            block_sizes=[32],
        )

        expected = torch.ones(8, device=DEVICE)
        torch.testing.assert_close(result, expected)

    @xfailIfPallas("Integer indexing not supported on Pallas")
    def test_atomic_add_1d_tensor(self):
        M, N = 32, 64
        x = torch.randn(M, N, device=DEVICE, dtype=torch.float32)
        y = torch.randn(M, N, device=DEVICE, dtype=torch.float32)
        args = (x, y)

        code, result = code_and_output(
            atomic_add_1d_tensor_kernel,
            args,
            block_sizes=[32],
        )

        expected = (x * y).sum(dim=0)
        torch.testing.assert_close(result, expected)

    def test_atomic_add_tensor_index(self):
        if _get_backend() == "pallas":
            self.skipTest("Pallas/TPU does not support integer tensor indexing")
        values = torch.randn(64, device=DEVICE, dtype=torch.float32)
        indices = torch.randperm(64, device=DEVICE, dtype=LONG_INT_TYPE)

        code, result = code_and_output(
            atomic_add_tensor_index_kernel,
            (values, indices),
            block_sizes=[32],
        )

        expected = torch.zeros(64, device=DEVICE, dtype=values.dtype)
        expected.scatter_add_(0, indices, values)
        torch.testing.assert_close(result, expected)

    def test_atomic_add_1d_tensor_with_offset_arange(self):
        if _get_backend() != "cute":
            self.skipTest("CuTe-specific direct iota scatter regression")

        M, N = 32, 64
        x = torch.randn(M, N, device=DEVICE, dtype=torch.float32)
        y = torch.randn(M, N, device=DEVICE, dtype=torch.float32)

        code, result = code_and_output(
            atomic_add_1d_tensor_offset_kernel,
            (x, y),
            block_sizes=[32],
        )

        expected = torch.zeros(N + 16, device=DEVICE, dtype=x.dtype)
        expected[16:] = (x * y).sum(dim=0)
        torch.testing.assert_close(result, expected)

    def test_atomic_add_1d_tensor_with_step_arange(self):
        if _get_backend() != "cute":
            self.skipTest("CuTe-specific stepped iota scatter regression")

        N = 64
        x = torch.randn(N, device=DEVICE, dtype=torch.float32)

        code, result = code_and_output(
            atomic_add_1d_tensor_step_arange_kernel,
            (x,),
        )

        expected = torch.zeros(2 * N, device=DEVICE, dtype=x.dtype)
        expected[::2] = x
        torch.testing.assert_close(result, expected)

    def test_atomic_add_returns_prev(self):
        @helion.kernel()
        def k(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            prev = torch.empty_like(x)
            for i in hl.tile(x.size(0)):
                old = hl.atomic_add(x, [i], y[i])
                prev[i] = old
            return x, prev

        x = torch.zeros(8, device=DEVICE)
        y = torch.arange(8, device=DEVICE, dtype=torch.float32)
        code, (out, prev) = code_and_output(k, (x, y))
        torch.testing.assert_close(out, y)
        torch.testing.assert_close(prev, torch.zeros_like(x))

    @xfailIfPallas("gather indexing with different-sized tensors unsupported on Pallas")
    def test_overlapping_atomic_add(self):
        # Test with overlapping indices
        x = torch.zeros(5, device=DEVICE)
        y = torch.ones(10, device=DEVICE)
        indices = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2, 3, 4], device=DEVICE)
        args = (x, y, indices)

        code, result = code_and_output(
            atomic_add_overlap_kernel,
            args,
            block_sizes=[32],
        )

        expected = torch.ones(5, device=DEVICE) * 2
        torch.testing.assert_close(result, expected)

    def test_2d_atomic_add(self):
        """Test atomic_add with 2D tensor indexing."""
        x = torch.zeros(3, 4, device=DEVICE)
        y = torch.ones(3, 4, device=DEVICE)
        args = (x, y)

        code, result = code_and_output(
            atomic_add_2d_kernel,
            args,
            block_sizes=[8, 8],
        )

        expected = torch.ones(3, 4, device=DEVICE)
        torch.testing.assert_close(result, expected)

    @onlyBackends(["pallas"])
    def test_atomic_add_f32_into_bf16(self):
        """atomic_add of a float32 value into a bfloat16 output tensor."""
        x = torch.ones(64, 128, device=DEVICE, dtype=torch.bfloat16)
        y = torch.ones(64, 128, device=DEVICE, dtype=torch.bfloat16)
        args = (x, y)

        code, result = code_and_output(
            atomic_add_f32_into_bf16_kernel,
            args,
            block_sizes=[64, 128],
        )

        expected = (x.float() + y.float()).to(torch.bfloat16)
        torch.testing.assert_close(result, expected)

    @onlyBackends(["pallas"])
    def test_split_k_atomic_add_vmem_preload(self):
        """Split-K matmul where output is initialised to ones (not zeros)."""
        m, k, n = 128, 1024, 128
        x = torch.randn(m, k, device=DEVICE, dtype=torch.bfloat16)
        y = torch.randn(k, n, device=DEVICE, dtype=torch.bfloat16)
        args = (x, y)

        code, result = code_and_output(
            split_k_atomic_add_kernel,
            args,
            block_sizes=[128, 128, 1024, 128],
            pallas_loop_type="fori_loop",
        )

        # expected = 1 + x @ y  (ones init + matmul via atomic_add)
        expected = torch.ones(m, n, device=DEVICE, dtype=torch.bfloat16) + (x @ y).to(
            torch.bfloat16
        )
        torch.testing.assert_close(result, expected, atol=0.1, rtol=0.05)

    @onlyBackends(["pallas"])
    def test_pallas_structural_atomic_add_shared_tile(self):
        @helion.kernel(static_shapes=True)
        def pallas_structural_atomic_add_kernel(x: torch.Tensor) -> torch.Tensor:
            """Structurally shared output tile: contributor dim is leading K."""
            k, m, n = x.shape
            out = torch.ones([m, n], dtype=x.dtype, device=x.device)
            for tile_k, tile_m, tile_n in hl.tile([k, m, n]):
                value = torch.sum(x[tile_k, tile_m, tile_n], dim=0)
                hl.atomic_add(out, [tile_m, tile_n], value)
            return out

        x = torch.randn(4, 16, 128, device=DEVICE, dtype=torch.float32)

        for loop_type in ("unroll", "fori_loop", "emit_pipeline"):
            with self.subTest(loop_type=loop_type):
                code, result = code_and_output(
                    pallas_structural_atomic_add_kernel,
                    (x,),
                    block_sizes=[1, 8, 128],
                    pallas_loop_type=loop_type,
                )
                expected = torch.ones(16, 128, device=DEVICE) + x.sum(dim=0)
                torch.testing.assert_close(result, expected)

    def test_structural_atomic_add_two_contributor_dims(self):
        @helion.kernel(static_shapes=True)
        def structural_atomic_add_2_contributor_kernel(
            x: torch.Tensor,
        ) -> torch.Tensor:
            """Two contributor dims both update the same output tile."""
            r, k, m, n = x.shape
            out = torch.ones([m, n], dtype=x.dtype, device=x.device)
            for tile_r, tile_k, tile_m, tile_n in hl.tile([r, k, m, n]):
                value = torch.sum(
                    torch.sum(x[tile_r, tile_k, tile_m, tile_n], dim=0), dim=0
                )
                hl.atomic_add(out, [tile_m, tile_n], value)
            return out

        x = torch.randn(3, 4, 16, 128, device=DEVICE, dtype=torch.float32)

        code, result = code_and_output(
            structural_atomic_add_2_contributor_kernel,
            (x,),
            block_sizes=[1, 1, 8, 128],
        )

        expected = torch.ones(16, 128, device=DEVICE) + x.sum(dim=(0, 1))
        torch.testing.assert_close(result, expected)

    def test_structural_atomic_add_multiple_outputs(self):
        @helion.kernel(static_shapes=True)
        def structural_atomic_add_two_outputs_kernel(
            x: torch.Tensor, y: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            k, m, n = x.shape
            out_x = torch.ones([m, n], dtype=x.dtype, device=x.device)
            out_y = torch.full([m, n], 2.0, dtype=y.dtype, device=y.device)
            for tile_k, tile_m, tile_n in hl.tile([k, m, n]):
                x_value = torch.sum(x[tile_k, tile_m, tile_n], dim=0)
                y_value = torch.sum(y[tile_k, tile_m, tile_n], dim=0)
                hl.atomic_add(out_x, [tile_m, tile_n], x_value)
                hl.atomic_add(out_y, [tile_m, tile_n], y_value)
            return out_x, out_y

        x = torch.randn(4, 16, 128, device=DEVICE, dtype=torch.float32)
        y = torch.randn(4, 16, 128, device=DEVICE, dtype=torch.float32)

        code, (out_x, out_y) = code_and_output(
            structural_atomic_add_two_outputs_kernel,
            (x, y),
            block_sizes=[1, 8, 128],
        )

        torch.testing.assert_close(
            out_x, torch.ones(16, 128, device=DEVICE) + x.sum(dim=0)
        )
        torch.testing.assert_close(
            out_y, torch.full([16, 128], 2.0, device=DEVICE) + y.sum(dim=0)
        )

    def test_structural_atomic_max_min_shared_tile(self):
        @helion.kernel(static_shapes=True)
        def structural_atomic_max_kernel(x: torch.Tensor) -> torch.Tensor:
            out = torch.full(
                [x.size(1), x.size(2)], -1000.0, dtype=x.dtype, device=x.device
            )
            for tile_k, tile_m, tile_n in hl.tile(x.size()):
                value = torch.amax(x[tile_k, tile_m, tile_n], dim=0)
                hl.atomic_max(out, [tile_m, tile_n], value)
            return out

        @helion.kernel(static_shapes=True)
        def structural_atomic_min_kernel(x: torch.Tensor) -> torch.Tensor:
            out = torch.full(
                [x.size(1), x.size(2)], 1000.0, dtype=x.dtype, device=x.device
            )
            for tile_k, tile_m, tile_n in hl.tile(x.size()):
                value = torch.amin(x[tile_k, tile_m, tile_n], dim=0)
                hl.atomic_min(out, [tile_m, tile_n], value)
            return out

        x = torch.randn(4, 16, 128, device=DEVICE, dtype=torch.float32)

        code_max, result_max = code_and_output(
            structural_atomic_max_kernel,
            (x,),
            block_sizes=[1, 8, 128],
        )
        expected_max = torch.maximum(
            torch.full([16, 128], -1000.0, device=DEVICE), x.amax(dim=0)
        )
        torch.testing.assert_close(result_max, expected_max)

        code_min, result_min = code_and_output(
            structural_atomic_min_kernel,
            (x,),
            block_sizes=[1, 8, 128],
        )
        expected_min = torch.minimum(
            torch.full([16, 128], 1000.0, device=DEVICE), x.amin(dim=0)
        )
        torch.testing.assert_close(result_min, expected_min)

    def test_atomic_add_code_generation(self):
        """Test that the generated code contains atomic_add."""
        x = torch.zeros(10, device=DEVICE)
        y = torch.ones(10, device=DEVICE)
        args = (x, y)

        code, result = code_and_output(atomic_add_kernel, args)
        expected = torch.ones(10, device=DEVICE)
        torch.testing.assert_close(result, expected)
        self.assertIn("atomic_add", code)

    @xfailIfPallas("int64 index dtype causes MLIR type mismatch on TPU")
    def test_atomic_add_float(self):
        """Test that atomic_add works with float constants."""
        x = torch.zeros(5, device=DEVICE, dtype=torch.float32)

        indices = torch.tensor([0, 1, 2, 2, 3, 3, 3, 4], device=DEVICE)
        expected = torch.tensor(
            [2.0, 2.0, 4.0, 6.0, 2.0], device=DEVICE, dtype=torch.float32
        )

        args = (x, indices)
        code, result = code_and_output(
            atomic_add_float_kernel,
            args,
            block_sizes=[32],
        )

        torch.testing.assert_close(result, expected)

    def test_atomic_add_invalid_sem(self):
        """Test that atomic_add raises with an invalid sem value."""
        x = torch.zeros(10, device=DEVICE)
        y = torch.ones(10, device=DEVICE)

        @helion.kernel()
        def bad_atomic_add_kernel(x: torch.Tensor, y: torch.Tensor):
            for i in hl.tile(x.size(0)):
                hl.atomic_add(x, [i], y[i], sem="ERROR")
            return x

        with self.assertRaises(helion.exc.InternalError) as ctx:
            code_and_output(
                bad_atomic_add_kernel,
                (x, y),
                block_sizes=[32],
            )
        self.assertIn("Invalid memory semantic 'ERROR'", str(ctx.exception))

    @skipIfRefEager(
        "Test is block size dependent which is not supported in ref eager mode"
    )
    def test_atomic_add_w_tile_attr(self):
        """Test atomic_add where the index is a symbolic int"""
        x = torch.randn(20, device=DEVICE)
        code, result = code_and_output(
            atomic_add_w_tile_attr,
            (x,),
            block_sizes=[2],
        )

        expected = torch.tensor([1, 0], device=DEVICE, dtype=torch.int32).repeat(10)
        torch.testing.assert_close(result, expected)

    @xfailIfPallas(
        "atomic scalar-origin reduction pattern is only validated on GPU backends"
    )
    def test_atomic_add_tile_begin_reduce_other_axis(self):
        if _get_backend() != "cute":
            self.skipTest("CuTe regression coverage")
        x = torch.ones((4, 4), device=DEVICE)
        code, result = code_and_output(
            atomic_add_tile_begin_reduce_other_axis,
            (x,),
            block_sizes=[2, 2],
        )

        # ``x[tile_m, tile_n]`` covers both tile axes, so every element of a
        # tile is added to ``out[tile_m.begin]``: two column tiles of four
        # ones per row tile, whatever the thread mapping of the row axis.
        expected = torch.tensor([8, 0, 8, 0], device=DEVICE, dtype=x.dtype)
        torch.testing.assert_close(result, expected)

    @xfailIfPallas("AtomicOnDeviceTensor error message differs on Pallas")
    @skipIfRefEager("Error only raises in normal mode")
    def test_atomic_add_device_tensor_error(self):
        @helion.kernel(static_shapes=True, autotune_effort="none")
        def kernel(x: torch.Tensor) -> torch.Tensor:
            for tile in hl.tile(x.size(0), block_size=128):
                device_tensor = hl.zeros([tile], dtype=x.dtype)
                hl.atomic_add(device_tensor, [tile], x[tile])
            return x

        x = torch.ones(256, device=DEVICE, dtype=torch.float32)
        error = helion.exc.AtomicOnDeviceTensor
        message = r"hl\.atomic_add\(\)"
        if _get_backend() == "cute":
            # CuTe supports constant CTA-local allocations, but this allocation
            # is tile-dependent and therefore has no supported local owner.
            error = helion.exc.InvalidConfig
            message = (
                "CTA-local atomics require a constant one-dimensional root allocation"
            )
        with self.assertRaisesRegex(error, message):
            kernel(x)

    def test_atomic_and(self):
        x0 = torch.full((8,), 0b1111, device=DEVICE, dtype=torch.int32)
        y = torch.tensor([0b1010] * 8, device=DEVICE, dtype=torch.int32)
        code, result = code_and_output(atomic_and_kernel, (x0.clone(), y))
        expected = torch.full((8,), 0b1111 & 0b1010, device=DEVICE, dtype=torch.int32)
        torch.testing.assert_close(result, expected)
        if _get_backend() == "triton":
            self.assertIn("tl.atomic_and", code)

    def test_atomic_or(self):
        x0 = torch.zeros(8, device=DEVICE, dtype=torch.int32)
        y = torch.tensor([0b1010] * 8, device=DEVICE, dtype=torch.int32)
        code, result = code_and_output(atomic_or_kernel, (x0.clone(), y))
        expected = torch.full((8,), 0b1010, device=DEVICE, dtype=torch.int32)
        torch.testing.assert_close(result, expected)
        if _get_backend() == "triton":
            self.assertIn("tl.atomic_or", code)

    def test_atomic_xor(self):
        x0 = torch.tensor([0b1010] * 8, device=DEVICE, dtype=torch.int32)
        y = torch.tensor([0b1100] * 8, device=DEVICE, dtype=torch.int32)
        code, result = code_and_output(atomic_xor_kernel, (x0.clone(), y))
        expected = torch.full((8,), 0b1010 ^ 0b1100, device=DEVICE, dtype=torch.int32)
        torch.testing.assert_close(result, expected)
        if _get_backend() == "triton":
            self.assertIn("tl.atomic_xor", code)

    @skipIfRocm("ROCm backend currently lacks support for these atomics")
    def test_atomic_xchg(self):
        x0 = torch.zeros(8, device=DEVICE, dtype=torch.int32)
        y = torch.arange(8, device=DEVICE, dtype=torch.int32)
        code, result = code_and_output(atomic_xchg_kernel, (x0.clone(), y))
        torch.testing.assert_close(result, y)
        if _get_backend() == "triton":
            self.assertIn("tl.atomic_xchg", code)

    @skipIfRocm("ROCm backend currently lacks support for these atomics")
    def test_atomic_max(self):
        x = torch.tensor([1, 5, 3, 7], device=DEVICE, dtype=torch.int32)
        y = torch.tensor([4, 2, 9, 1], device=DEVICE, dtype=torch.int32)
        code, result = code_and_output(atomic_max_kernel, (x.clone(), y))
        expected = torch.tensor([4, 5, 9, 7], device=DEVICE, dtype=torch.int32)
        torch.testing.assert_close(result, expected)
        if _get_backend() == "triton":
            self.assertIn("tl.atomic_max", code)

    def test_atomic_max_return_value(self):
        x = torch.tensor([1, 5, 3, 7], device=DEVICE, dtype=torch.int32)
        y = torch.tensor([4, 2, 9, 1], device=DEVICE, dtype=torch.int32)
        out = torch.empty(4, device=DEVICE, dtype=torch.int32)
        _, result = code_and_output(
            atomic_max_return_kernel, (x.clone(), y, out), block_sizes=[4]
        )
        # Return value should be the previous values of x
        expected = torch.tensor([1, 5, 3, 7], device=DEVICE, dtype=torch.int32)
        torch.testing.assert_close(result, expected)

    @skipIfRocm("ROCm backend currently lacks support for these atomics")
    def test_atomic_min(self):
        x = torch.tensor([1, 5, 3, 7], device=DEVICE, dtype=torch.int32)
        y = torch.tensor([4, 2, 9, 1], device=DEVICE, dtype=torch.int32)
        code, result = code_and_output(atomic_min_kernel, (x.clone(), y))
        expected = torch.tensor([1, 2, 3, 1], device=DEVICE, dtype=torch.int32)
        torch.testing.assert_close(result, expected)
        if _get_backend() == "triton":
            self.assertIn("tl.atomic_min", code)

    def test_atomic_cas(self):
        x = torch.tensor([1, 5, 3, 7], device=DEVICE, dtype=torch.int32)
        expect = torch.tensor([1, 6, 3, 0], device=DEVICE, dtype=torch.int32)
        y = torch.tensor([9, 9, 9, 9], device=DEVICE, dtype=torch.int32)
        code, result = code_and_output(atomic_cas_kernel, (x.clone(), y, expect))
        # Only positions where expect matches original x are replaced
        expected = torch.tensor([9, 5, 9, 7], device=DEVICE, dtype=torch.int32)
        torch.testing.assert_close(result, expected)
        if _get_backend() == "triton":
            self.assertIn("tl.atomic_cas", code)

    @onlyBackends("triton")
    @skipIfNotCUDA()
    @skipIfTileIR("TileIR does not legalize tl.debug_barrier")
    @skipIfRefEager("program-level atomic synchronization is codegen-only")
    def test_release_acquire_atomics_sync_program(self):
        """Release and acquire atomics order every thread of the program, not only the issuing one."""
        config = helion.Config(block_sizes=[128], num_warps=4)

        @helion.kernel(config=config, static_shapes=True)
        def release_unused(
            out: torch.Tensor, x: torch.Tensor, count: torch.Tensor, done: torch.Tensor
        ) -> torch.Tensor:
            for tile in hl.tile(x.size(0)):
                out[tile] = x[tile] * 2.0
                hl.atomic_add(count, [0], 1, sem="release")
            return out

        @helion.kernel(config=config, static_shapes=True)
        def acq_rel_unused(
            out: torch.Tensor, x: torch.Tensor, count: torch.Tensor, done: torch.Tensor
        ) -> torch.Tensor:
            for tile in hl.tile(x.size(0)):
                out[tile] = x[tile] * 2.0
                hl.atomic_add(count, [0], 1, sem="acq_rel")
            return out

        @helion.kernel(config=config, static_shapes=True)
        def acq_rel_used(
            out: torch.Tensor, x: torch.Tensor, count: torch.Tensor, done: torch.Tensor
        ) -> torch.Tensor:
            for tile in hl.tile(x.size(0)):
                out[tile] = x[tile] * 2.0
                arrived = hl.atomic_add(count, [0], 1, sem="acq_rel")
                if arrived == 3:
                    done[0] = 1
            return out

        @helion.kernel(config=config, static_shapes=True)
        def cas_acquire_unused(
            out: torch.Tensor, x: torch.Tensor, count: torch.Tensor, done: torch.Tensor
        ) -> torch.Tensor:
            for tile in hl.tile(x.size(0)):
                hl.atomic_cas(done, [0], 0, 1, sem="acquire")
                hl.atomic_add(count, [0], 1, sem="relaxed")
                out[tile] = x[tile] * 2.0
            return out

        @helion.kernel(config=config, static_shapes=True)
        def relaxed(
            out: torch.Tensor, x: torch.Tensor, count: torch.Tensor, done: torch.Tensor
        ) -> torch.Tensor:
            for tile in hl.tile(x.size(0)):
                out[tile] = x[tile] * 2.0
                hl.atomic_add(count, [0], 1, sem="relaxed")
            return out

        # kernel: (atomic call, barrier before it, barrier after it, final done)
        cases = {
            release_unused: ("tl.atomic_add(", True, False, 0),
            acq_rel_unused: ("tl.atomic_add(", True, True, 0),
            # Triton itself broadcasts a used scalar result behind a bar.sync.
            acq_rel_used: ("tl.atomic_add(", True, False, 1),
            cas_acquire_unused: ("tl.atomic_cas(", False, True, 1),
            relaxed: ("tl.atomic_add(", False, False, 0),
        }
        x = torch.randn(512, device=DEVICE)
        for kernel, (call, before, after, final_done) in cases.items():
            out = torch.empty_like(x)
            count = torch.zeros(1, device=DEVICE, dtype=torch.int32)
            done = torch.zeros(1, device=DEVICE, dtype=torch.int32)
            code, result = code_and_output(kernel, (out, x, count, done))
            torch.testing.assert_close(result, x * 2.0)
            self.assertEqual(int(count.item()), 4)
            self.assertEqual(int(done.item()), final_done)
            atomic = code.index(call)
            barrier = code.rfind("tl.debug_barrier()", 0, atomic)
            if before:
                self.assertGreater(barrier, code.rfind("tl.store(", 0, atomic))
            else:
                self.assertEqual(barrier, -1)
            self.assertEqual(code.find("tl.debug_barrier()", atomic) != -1, after)

    @onlyBackends("triton")
    @skipIfRocm("Tensor descriptor not supported on ROCm")
    @skipIfTileIR("TileIR does not legalize tl.debug_barrier")
    @skipUnlessTensorDescriptor("Tensor descriptor support is required")
    @skipIfRefEager("TMA drains are codegen-only")
    def test_release_drains_tma_store_after_tile_index(self):
        """A tile-index read shifts codegen's memory-op slots; the TMA store still drains."""

        @helion.kernel(static_shapes=True, autotune_effort="none")
        def store_then_release(x: torch.Tensor, flag: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile_m, tile_n in hl.tile(x.size(), block_size=[32, 32]):
                rows = tile_m.index[:, None]
                out[tile_m, tile_n] = x[tile_m, tile_n] + rows
                hl.atomic_add(flag, [0], 1, sem="release")
            return out

        x = torch.randn(64, 64, device=DEVICE)
        flag = torch.zeros(1, device=DEVICE, dtype=torch.int32)
        code, out = code_and_output(
            store_then_release,
            (x, flag),
            indexing=["pointer", "tensor_descriptor", "pointer"],
        )
        rows = torch.arange(64, device=DEVICE)[:, None]
        torch.testing.assert_close(out, x + rows)
        self.assertIn("out_desc.store(", code)
        drain = code.find("cp.async.bulk.wait_group 0;")
        self.assertNotEqual(drain, -1)
        self.assertLess(drain, code.index("tl.atomic_add("))

    @onlyBackends("triton")
    @skipIfRocm("Tensor descriptor not supported on ROCm")
    @skipIfTileIR("TileIR does not support descriptor atomics")
    def test_atomic_td_fallbacks(self):
        """Test that tensor_descriptor atomics fall back to pointer when needed."""

        # Return value consumed: should fall back to pointer
        @helion.kernel(
            config=helion.Config(
                block_sizes=[64, 64],
                indexing="tensor_descriptor",
                atomic_indexing="tensor_descriptor",
            ),
            static_shapes=True,
        )
        def atomic_add_td_prev_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.zeros_like(x)
            for i, j in hl.tile([x.size(0), x.size(1)]):
                prev = hl.atomic_add(x, [i, j], y[i, j])
                out[i, j] = prev
            return out

        M, N = 128, 64
        x = torch.zeros(M, N, device=DEVICE, dtype=torch.float32)
        y = torch.ones(M, N, device=DEVICE, dtype=torch.float32)
        code, result = code_and_output(atomic_add_td_prev_kernel, (x, y))
        expected = torch.zeros(M, N, device=DEVICE, dtype=torch.float32)
        torch.testing.assert_close(result, expected)
        self.assertIn("tl.atomic_add", code)
        self.assertNotIn("desc.atomic_add(", code)

        # Non-relaxed sem: should fall back to pointer
        @helion.kernel(
            config=helion.Config(
                block_sizes=[64, 64],
                indexing="tensor_descriptor",
                atomic_indexing="tensor_descriptor",
            ),
            static_shapes=True,
        )
        def atomic_add_td_release_kernel(
            x: torch.Tensor, y: torch.Tensor
        ) -> torch.Tensor:
            for i, j in hl.tile([x.size(0), x.size(1)]):
                hl.atomic_add(x, [i, j], y[i, j], sem="release")
            return x

        x2 = torch.zeros(M, N, device=DEVICE, dtype=torch.float32)
        y2 = torch.ones(M, N, device=DEVICE, dtype=torch.float32)
        with patch(
            "helion._compiler.compile_environment.target_device_capability",
            return_value=(8, 0),
        ):
            code2, result2 = code_and_output(atomic_add_td_release_kernel, (x2, y2))
        expected2 = torch.ones(M, N, device=DEVICE, dtype=torch.float32)
        torch.testing.assert_close(result2, expected2)
        self.assertIn("tl.atomic_add", code2)
        self.assertNotIn("desc.atomic_add(", code2)
        self.assertNotIn("cp.async.bulk.wait_group", code2)

    @onlyBackends("triton")
    @skipIfRocm("Tensor descriptor not supported on ROCm")
    @skipIfTileIR("TileIR does not support descriptor atomics")
    @skipUnlessTensorDescriptor("Tensor descriptor support is required")
    def test_atomic_add_per_op_indexing(self):
        """Test per-op atomic_indexing list: first op pointer, second op tensor_descriptor."""

        @helion.kernel(
            config=helion.Config(
                block_sizes=[64, 64],
                indexing="tensor_descriptor",
                atomic_indexing=["pointer", "tensor_descriptor"],
            ),
            static_shapes=True,
        )
        def two_atomic_adds(
            out1: torch.Tensor, out2: torch.Tensor, val: torch.Tensor
        ) -> torch.Tensor:
            for i, j in hl.tile([out1.size(0), out1.size(1)]):
                hl.atomic_add(out1, [i, j], val[i, j])  # pointer
                hl.atomic_add(out2, [i, j], val[i, j])  # tensor_descriptor
            return out1

        M, N = 128, 64
        out1 = torch.zeros(M, N, device=DEVICE, dtype=torch.float32)
        out2 = torch.zeros(M, N, device=DEVICE, dtype=torch.float32)
        val = torch.ones(M, N, device=DEVICE, dtype=torch.float32)
        code, result = code_and_output(two_atomic_adds, (out1, out2, val))
        expected = torch.ones(M, N, device=DEVICE, dtype=torch.float32)
        torch.testing.assert_close(result, expected)
        torch.testing.assert_close(out2, expected)
        # out1 uses pointer: tl.atomic_add(out1 + ...)
        self.assertIn("tl.atomic_add(out1", code)
        # out2 uses tensor_descriptor: out2_desc.atomic_add(...)
        self.assertIn("out2_desc.atomic_add(", code)
        # out1 should NOT use descriptor, out2 should NOT use pointer
        self.assertNotIn("out1_desc", code)
        self.assertNotIn("tl.atomic_add(out2", code)

    @onlyBackends("triton")
    @skipIfRocm("Tensor descriptor not supported on ROCm")
    @skipIfTileIR("TileIR does not support descriptor atomics")
    @skipUnlessTensorDescriptor("Tensor descriptor support is required")
    def test_atomic_ops_tensor_descriptor(self):
        """Test all TMA-supported atomic ops generate desc.atomic_{op} codegen."""
        M, N = 128, 64
        td_config = {
            "block_sizes": [64, 64],
            "indexing": "tensor_descriptor",
            "atomic_indexing": "tensor_descriptor",
        }
        # (op_name, kernel, x, y, expected)
        cases = [
            (
                "atomic_add",
                atomic_add_2d_td_kernel,
                torch.zeros(M, N, device=DEVICE, dtype=torch.float32),
                torch.ones(M, N, device=DEVICE, dtype=torch.float32),
                torch.ones(M, N, device=DEVICE, dtype=torch.float32),
            ),
            (
                "atomic_and",
                atomic_and_2d_td_kernel,
                torch.full((M, N), 0b1111, device=DEVICE, dtype=torch.int32),
                torch.full((M, N), 0b1010, device=DEVICE, dtype=torch.int32),
                torch.full((M, N), 0b1010, device=DEVICE, dtype=torch.int32),
            ),
            (
                "atomic_or",
                atomic_or_2d_td_kernel,
                torch.zeros(M, N, device=DEVICE, dtype=torch.int32),
                torch.full((M, N), 0b1010, device=DEVICE, dtype=torch.int32),
                torch.full((M, N), 0b1010, device=DEVICE, dtype=torch.int32),
            ),
            (
                "atomic_xor",
                atomic_xor_2d_td_kernel,
                torch.full((M, N), 0b1010, device=DEVICE, dtype=torch.int32),
                torch.full((M, N), 0b1100, device=DEVICE, dtype=torch.int32),
                torch.full((M, N), 0b0110, device=DEVICE, dtype=torch.int32),
            ),
            (
                "atomic_max",
                atomic_max_2d_td_kernel,
                torch.ones(M, N, device=DEVICE, dtype=torch.int32),
                torch.full((M, N), 5, device=DEVICE, dtype=torch.int32),
                torch.full((M, N), 5, device=DEVICE, dtype=torch.int32),
            ),
            (
                "atomic_min",
                atomic_min_2d_td_kernel,
                torch.full((M, N), 10, device=DEVICE, dtype=torch.int32),
                torch.full((M, N), 3, device=DEVICE, dtype=torch.int32),
                torch.full((M, N), 3, device=DEVICE, dtype=torch.int32),
            ),
        ]
        for op_name, kernel, x, y, expected in cases:
            with self.subTest(op=op_name):
                code, result = code_and_output(kernel, (x, y), **td_config)
                torch.testing.assert_close(result, expected)
                self.assertIn(f"desc.{op_name}(", code)
                self.assertNotIn(f"tl.{op_name}", code)

        # xchg is NOT a TMA reduction op — should fall back to pointer
        with self.subTest(op="atomic_xchg_fallback"):
            x = torch.zeros(M, N, device=DEVICE, dtype=torch.int32)
            y = torch.ones(M, N, device=DEVICE, dtype=torch.int32)
            code, result = code_and_output(
                atomic_xchg_2d_td_kernel, (x, y), **td_config
            )
            torch.testing.assert_close(
                result, torch.ones(M, N, device=DEVICE, dtype=torch.int32)
            )
            self.assertIn("tl.atomic_xchg", code)
            self.assertNotIn("desc.atomic_xchg", code)

    @onlyBackends("triton")
    @skipIfRocm("Tensor descriptor not supported on ROCm")
    @skipIfTileIR("TileIR does not support descriptor atomics")
    @skipUnlessTensorDescriptor("Tensor descriptor support is required")
    def test_dynamic_atomic_add_tensor_descriptor(self):
        """Dynamic shape atomics should register TD layout guards."""

        @helion.kernel(
            config=helion.Config(
                block_sizes=[64, 64],
                indexing="tensor_descriptor",
                atomic_indexing="tensor_descriptor",
            ),
            static_shapes=False,
        )
        def atomic_add_dynamic_td(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            for i, j in hl.tile([x.size(0), x.size(1)]):
                hl.atomic_add(x, [i, j], y[i, j])
            return x

        x = torch.zeros(128, 64, device=DEVICE, dtype=torch.float32)
        y = torch.ones(128, 64, device=DEVICE, dtype=torch.float32)
        code, result = code_and_output(atomic_add_dynamic_td, (x, y))
        torch.testing.assert_close(result, torch.ones_like(x))
        self.assertIn("x_desc.atomic_add(", code)
        self.assertNotIn("tl.atomic_add(", code)

    @onlyBackends("triton")
    @skipIfRocm("Tensor descriptor not supported on ROCm")
    @skipIfTileIR("TileIR does not support descriptor atomics")
    @skipUnlessTensorDescriptor("Tensor descriptor support is required")
    def test_atomic_td_scalar_symint(self):
        """Composite scalar SymInts should not prevent descriptor atomics."""

        @helion.kernel(
            config=helion.Config(
                block_sizes=[64, 64],
                indexing="tensor_descriptor",
                atomic_indexing="tensor_descriptor",
            ),
            static_shapes=True,
        )
        def batched_atomic_add(
            x: torch.Tensor, y: torch.Tensor, start: int
        ) -> torch.Tensor:
            B, M, N = x.size()
            for tile_b in hl.tile(B - start, block_size=1):
                for tile_m, tile_n in hl.tile([M, N]):
                    batch = start + tile_b.begin
                    hl.atomic_add(
                        x,
                        [batch, tile_m, tile_n],
                        y[batch, tile_m, tile_n],
                    )
            return x

        x = torch.zeros(4, 64, 64, device=DEVICE, dtype=torch.float32)
        y = torch.ones(4, 64, 64, device=DEVICE, dtype=torch.float32)
        code, result = code_and_output(batched_atomic_add, (x, y, 1))
        expected = torch.zeros_like(x)
        expected[1:] = 1
        torch.testing.assert_close(result, expected)
        self.assertIn("desc.atomic_add(", code)
        self.assertNotIn("tl.atomic_add(", code)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _atomic_logical_axes_2d(out, buckets, values, mode: hl.constexpr):
    for row, col in hl.tile(values.shape):
        if mode == "scatter":
            index = buckets[row, col]
            value = values[row, col]
        elif mode == "broadcast_value":
            index = buckets[row, 0][:, None]
            value = values[row, col]
        elif mode == "broadcast_constant":
            index = buckets[row, 0][:, None]
            value = 1
        else:
            index = buckets[row, 0][:, None]
            value = hl.full([row, col], 1, dtype=values.dtype)
        hl.atomic_add(out, [row.index[:, None], index], value)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _atomic_logical_axes_3d(out, buckets, values, scalar_plane: hl.constexpr):
    for plane, row, col in hl.tile(values.shape):
        index = buckets[plane, row, col]
        value = values[plane, row, col]
        if scalar_plane:
            hl.atomic_add(out, [0, index], value)
        else:
            hl.atomic_add(out, [index], value)
    return out


def _atomic_logical_axes_codegen(kernel, inputs, tiles):
    from unittest.mock import patch

    from test._cute_binding import _cpu_bind
    from test._cute_binding import _forbid_native_compile
    from test._cute_binding import _mock_cuda_unavailable
    from test.cute_population_contracts import _target

    with (
        _mock_cuda_unavailable(),
        _target(),
        _forbid_native_compile(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CPU only")),
    ):
        bound = _cpu_bind(kernel, inputs)
        config = bound.config_spec.default_config()
        config.config["block_sizes"] = tiles
        return bound.to_code(config)


def _execute_atomic_thread_program(source, inputs):
    """Run the actual scalar source for each emitted CTA/thread, with CPU atomics."""
    import ast
    import itertools
    from types import SimpleNamespace

    current = {"block": (0, 0, 0), "thread": (0, 0, 0)}
    atomic_calls = []

    class Pointer:
        def __init__(self, tensor, offset=0):
            self.tensor, self.offset = tensor, int(offset)

        def __add__(self, offset):
            return Pointer(self.tensor, self.offset + int(offset))

        @property
        def llvm_ptr(self):
            return self

        def load(self):
            assert 0 <= self.offset < self.tensor.numel()
            return self.tensor.reshape(-1)[self.offset].item()

        def store(self, value):
            assert 0 <= self.offset < self.tensor.numel()
            self.tensor.reshape(-1)[self.offset] = value

    def atomic_add(pointer, val, sem):
        assert sem == "relaxed"
        atomic_calls.append((pointer.offset, val))
        previous = pointer.load()
        pointer.store(previous + val)
        return previous

    def launcher(function, grid, *arguments, block):
        args = [
            SimpleNamespace(
                iterator=Pointer(arg), layout=SimpleNamespace(stride=arg.stride())
            )
            if isinstance(arg, torch.Tensor)
            else arg
            for arg in arguments
        ]
        for cta in itertools.product(*(range(size) for size in grid)):
            current["block"] = (*cta, *((0,) * (3 - len(cta))))
            for thread in itertools.product(*(range(size) for size in block)):
                current["thread"] = thread
                function(*args)

    tree = ast.parse(source)
    body = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(node, ast.FunctionDef):
            node.decorator_list = []
        body.append(node)
    namespace = {
        "torch": torch,
        "hl": hl,
        "cutlass": SimpleNamespace(Int32=int, Int64=int, Float32=float),
        "cute": SimpleNamespace(
            crd2idx=lambda coords, layout: sum(
                c * s for c, s in zip(coords, layout.stride, strict=True)
            ),
            arch=SimpleNamespace(
                block_idx=lambda: current["block"],
                thread_idx=lambda: current["thread"],
                atomic_add=atomic_add,
            ),
        ),
        "_default_cute_launcher": launcher,
        "_next_power_of_2": lambda value: 1 << (value - 1).bit_length(),
    }
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])),
            "<actual-atomic-thread-program>",
            "exec",
        ),
        namespace,
    )
    wrapper = next(
        node.name for node in reversed(body) if isinstance(node, ast.FunctionDef)
    )
    result = namespace[wrapper](*inputs)
    return result, atomic_calls


class TestAtomicLogicalAxesCPU(unittest.TestCase):
    def test_data_dependent_and_broadcast_atomic_contributions(self):
        for shape, tiles in [((3, 19), [4, 8]), ((5, 7), [2, 4])]:
            for mode in (
                "scatter",
                "broadcast_value",
                "broadcast_constant",
                "broadcast_tensor_constant",
            ):
                for dtype in (torch.int32, torch.float32):
                    with self.subTest(shape=shape, mode=mode, dtype=dtype):
                        values = (
                            torch.arange(1, 1 + shape[0] * shape[1])
                            .reshape(shape)
                            .to(dtype)
                        )
                        buckets = (torch.arange(values.numel()).reshape(shape) % 3).to(
                            torch.int32
                        )
                        out = torch.zeros((shape[0], 3), dtype=dtype)
                        inputs = (out, buckets, values, mode)
                        source = _atomic_logical_axes_codegen(
                            _atomic_logical_axes_2d, inputs, tiles
                        )
                        actual, calls = _execute_atomic_thread_program(source, inputs)
                        expected = torch.zeros_like(out)
                        if mode == "broadcast_constant":
                            contributions = (shape[1] + tiles[1] - 1) // tiles[1]
                            for row in range(shape[0]):
                                expected[row, buckets[row, 0]] = contributions
                            expected_calls = shape[0] * contributions
                        else:
                            for row in range(shape[0]):
                                for col in range(shape[1]):
                                    bucket = (
                                        buckets[row, col]
                                        if mode == "scatter"
                                        else buckets[row, 0]
                                    )
                                    value = (
                                        1
                                        if mode == "broadcast_tensor_constant"
                                        else values[row, col]
                                    )
                                    expected[row, bucket] += value
                            expected_calls = values.numel()
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                        self.assertEqual(len(calls), expected_calls)

    def test_three_dimensional_and_mixed_scalar_tensor_atomic_contributions(self):
        for scalar_plane in (False, True):
            with self.subTest(scalar_plane=scalar_plane):
                shape = (3, 5, 9)
                values = torch.arange(1, 136, dtype=torch.int32).reshape(shape)
                buckets = (values % 7).to(torch.int32)
                out = torch.zeros((1, 7) if scalar_plane else (7,), dtype=values.dtype)
                inputs = (out, buckets, values, scalar_plane)
                source = _atomic_logical_axes_codegen(
                    _atomic_logical_axes_3d, inputs, [2, 4, 4]
                )
                actual, calls = _execute_atomic_thread_program(source, inputs)
                expected = torch.zeros(7, dtype=values.dtype).scatter_add_(
                    0, buckets.flatten().long(), values.flatten()
                )
                torch.testing.assert_close(actual.flatten(), expected, rtol=0, atol=0)
                self.assertEqual(len(calls), values.numel())


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _atomic_logical_axes_ghost(out, copied, values, row_values):
    for row in hl.tile(values.size(0)):
        for col in hl.tile(values.size(1)):
            copied[row, col] = values[row, col]
        hl.atomic_add(out, [row], row_values[row])
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _atomic_logical_axes_reduced(out, values):
    for row, col in hl.tile(values.shape):
        total = values[row, col].sum(-1)
        hl.atomic_add(out, [row], total)
    return out


class TestAtomicLogicalAxisLeadersCPU(unittest.TestCase):
    def test_ghost_axis_contributes_once_after_nested_loop(self):
        values = torch.arange(15, dtype=torch.int32).reshape(3, 5)
        row_values = values.sum(-1).to(torch.int32)
        out = torch.zeros(3, dtype=torch.int32)
        copied = torch.full_like(values, -1)
        inputs = (out, copied, values, row_values)
        source = _atomic_logical_axes_codegen(
            _atomic_logical_axes_ghost, inputs, [2, 4]
        )
        result, calls = _execute_atomic_thread_program(source, inputs)
        torch.testing.assert_close(copied, values, rtol=0, atol=0)
        torch.testing.assert_close(result, row_values, rtol=0, atol=0)
        self.assertEqual(len(calls), 3)


@onlyBackends("cute")
class TestAtomicLogicalAxesNative(TestCase):
    def test_atomic_rank_two_contribution_domain(self):
        for mode in (
            "scatter",
            "broadcast_value",
            "broadcast_constant",
            "broadcast_tensor_constant",
        ):
            with self.subTest(mode=mode):
                values = torch.arange(1, 58, device=DEVICE, dtype=torch.int32).reshape(
                    3, 19
                )
                buckets = values % 7
                out = torch.zeros((3, 7), device=DEVICE, dtype=values.dtype)
                expected = torch.zeros_like(out)
                if mode == "scatter":
                    expected.scatter_add_(1, buckets.long(), values)
                elif mode == "broadcast_value":
                    expected.scatter_add_(
                        1,
                        buckets[:, :1].long(),
                        values.sum(-1, keepdim=True).to(values.dtype),
                    )
                else:
                    expected.scatter_add_(
                        1,
                        buckets[:, :1].long(),
                        torch.full(
                            (3, 1),
                            3 if mode == "broadcast_constant" else 19,
                            device=DEVICE,
                            dtype=values.dtype,
                        ),
                    )
                _, result = code_and_output(
                    _atomic_logical_axes_2d,
                    (out, buckets, values, mode),
                    block_sizes=[4, 8],
                )
                torch.testing.assert_close(result, expected, rtol=0, atol=0)

    def test_atomic_rank_three_contribution_domain(self):
        for scalar_plane in (False, True):
            with self.subTest(scalar_plane=scalar_plane):
                values = torch.arange(1, 136, device=DEVICE, dtype=torch.int32).reshape(
                    3, 5, 9
                )
                buckets = values % 7
                out = torch.zeros(
                    (1, 7) if scalar_plane else (7,), device=DEVICE, dtype=values.dtype
                )
                expected = torch.zeros(
                    7, device=DEVICE, dtype=values.dtype
                ).scatter_add_(0, buckets.flatten().long(), values.flatten())
                _, result = code_and_output(
                    _atomic_logical_axes_3d,
                    (out, buckets, values, scalar_plane),
                    block_sizes=[2, 4, 4],
                )
                torch.testing.assert_close(result.flatten(), expected, rtol=0, atol=0)

    def test_atomic_reduced_value_leaders(self):
        values = torch.arange(15, device=DEVICE, dtype=torch.float32).reshape(3, 5)
        out = torch.zeros(3, device=DEVICE)
        _, result = code_and_output(
            _atomic_logical_axes_reduced, (out, values), block_sizes=[2, 4]
        )
        torch.testing.assert_close(result, values.sum(-1), rtol=0, atol=0)

    def test_atomic_ghost_axis_leaders(self):
        values = torch.arange(15, device=DEVICE, dtype=torch.int32).reshape(3, 5)
        row_values = values.sum(-1).to(torch.int32)
        out = torch.zeros(3, device=DEVICE, dtype=torch.int32)
        copied = torch.full_like(values, -1)
        _, result = code_and_output(
            _atomic_logical_axes_ghost,
            (out, copied, values, row_values),
            block_sizes=[2, 4],
        )
        torch.testing.assert_close(copied, values, rtol=0, atol=0)
        torch.testing.assert_close(result, row_values, rtol=0, atol=0)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _local_atomic_histogram(
    x, bins: hl.constexpr, repeats: hl.constexpr, mode: hl.constexpr
):
    out = torch.empty((x.size(0), bins), dtype=x.dtype, device=x.device)
    for row in hl.grid(x.size(0)):
        local = hl.full([bins], 3, dtype=x.dtype)
        indices = hl.arange(x.size(1)) % bins
        values = x[row, :]
        if mode == "bounds":
            indices = hl.arange(x.size(1)) - 1
        for _iteration in range(repeats):
            if mode == "alias":
                alias = local[:, None]
                hl.atomic_add(alias, [indices, 0], values)
            elif mode == "result":
                previous = hl.atomic_add(local, [indices], values)
                out[row, :] = previous
            elif mode == "release":
                hl.atomic_add(local, [indices], values, sem="release")
            elif mode == "divergent":
                if x[row, 0] > 0:
                    hl.atomic_add(local, [indices], values)
            elif mode == "early_read":
                out[row, :] = local
                hl.atomic_add(local, [indices], values)
            elif mode == "nested":
                for inner in range(2):
                    hl.atomic_add(local, [indices], values * (inner + 1))
            elif mode == "counts":
                hl.atomic_add(local, [indices], 1)
            elif mode == "self_value":
                hl.atomic_add(local, [indices], local)
            else:
                hl.atomic_add(local, [indices], values)
        out[row, :] = local
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _local_atomic_subnormal(x):
    out = torch.empty((x.size(0), 1), dtype=x.dtype, device=x.device)
    for row in hl.grid(x.size(0)):
        local = hl.zeros([1], dtype=x.dtype)
        indices = hl.arange(x.size(1)) * 0
        hl.atomic_add(local, [indices], x[row, :])
        out[row, :] = local
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _local_atomic_flush(x, out):
    for row in hl.grid(x.size(0)):
        local = hl.zeros([17], dtype=x.dtype)
        indices = hl.arange(x.size(1)) % 17
        hl.atomic_add(local, [indices], x[row, :])
        hl.atomic_add(out, [0, hl.arange(17)], local)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _local_atomic_negative(x, mode: hl.constexpr):
    out = torch.empty((x.size(0), 17), device=x.device, dtype=x.dtype)
    for row in hl.grid(x.size(0)):
        local = hl.zeros([17], dtype=x.dtype)
        if mode == "last":
            hl.atomic_add(local, [-1], x[row, 0])
        elif mode == "first":
            hl.atomic_add(local, [-17], x[row, 0])
        elif mode == "invalid":
            hl.atomic_add(local, [17], x[row, 0])
        else:
            hl.atomic_add(local, [hl.arange(x.size(1)) - 17], x[row, :])
        out[row, :] = local
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _local_atomic_negative_flush(x, out):
    for row in hl.grid(x.size(0)):
        local = hl.zeros([17], dtype=x.dtype)
        hl.atomic_add(local, [hl.arange(17)], x[row, :])
        hl.atomic_add(out, [row - x.size(0), hl.arange(17) - 17], local)
    return out


def _simulate_local_atomic_program(source, x, shape):
    """Run complete generated phases; this models coordinates, not GPU races."""
    import ast
    import math
    from types import SimpleNamespace

    import numpy as np

    calls = []

    class Pointer:
        def __init__(self, values, written=None, offset=0):
            self.values, self.written, self.offset = values, written, int(offset)

        def __add__(self, offset):
            return Pointer(self.values, self.written, self.offset + int(offset))

        @property
        def llvm_ptr(self):
            return self

        def load(self):
            assert 0 <= self.offset < self.values.size
            if self.written is not None:
                assert self.written[self.offset], "read before shared initialization"
            return self.values[self.offset]

        def store(self, value):
            assert 0 <= self.offset < self.values.size
            self.values[self.offset] = value
            if self.written is not None:
                self.written[self.offset] = True

    class Shared:
        def __init__(self, dtype, shape):
            self.values = np.empty(math.prod(shape), dtype=dtype)
            self.written = np.zeros(self.values.size, dtype=bool)
            self.iterator = Pointer(self.values, self.written)

        def __getitem__(self, index):
            return (self.iterator + index).load()

        def __setitem__(self, index, value):
            (self.iterator + index).store(value)

    class Allocator:
        def allocate_tensor(self, dtype, layout, byte_alignment):
            assert byte_alignment == 16
            return Shared(dtype, layout)

    def atomic_add(pointer, value, *, sem, scope):
        assert sem == "relaxed" and scope in ("cta", "gpu")
        assert (pointer.written is not None) == (scope == "cta")
        previous = pointer.load()
        pointer.store(previous + value)
        calls.append(pointer.offset)
        return previous

    class SerialPhases(ast.NodeTransformer):
        def visit_For(self, node):
            node = self.generic_visit(node)
            if isinstance(node.target, ast.Name) and node.target.id.startswith(
                "fragment_index"
            ):
                assert isinstance(node.iter, ast.Call) and len(node.iter.args) == 3
                node.iter.args[0] = ast.Constant(0)
                node.iter.args[2] = ast.Constant(1)
            return node

    tree = ast.parse(source)
    function = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name.startswith("_helion_")
    )
    function.decorator_list = []
    function = SerialPhases().visit(function)
    module = ast.fix_missing_locations(
        ast.Module(
            body=[*[n for n in tree.body if isinstance(n, ast.Assign)], function],
            type_ignores=[],
        )
    )
    input_array = x.numpy().reshape(-1)
    output = np.zeros(shape, dtype=input_array.dtype)
    for row in range(x.size(0)):
        environment = {
            "operator": operator,
            "cutlass": SimpleNamespace(
                Int32=np.int32,
                Int64=np.int64,
                Float32=np.float32,
                Boolean=bool,
                utils=SimpleNamespace(SmemAllocator=Allocator),
            ),
            "cute": SimpleNamespace(
                math=SimpleNamespace(
                    min=lambda left, right, propagate_nan=True: (
                        np.minimum(left, right)
                        if propagate_nan
                        else np.fmin(left, right)
                    ),
                    max=lambda left, right, propagate_nan=True: (
                        np.maximum(left, right)
                        if propagate_nan
                        else np.fmax(left, right)
                    ),
                ),
                make_layout=lambda shape: shape,
                arch=SimpleNamespace(
                    thread_idx=lambda: (0, 0, 0),
                    block_idx=lambda row=row: (row, 0, 0),
                    sync_threads=lambda: None,
                    atomic_add=atomic_add,
                ),
            ),
        }
        exec(compile(module, "<local-atomic-generated>", "exec"), environment)
        arguments = {
            "x": SimpleNamespace(iterator=Pointer(input_array)),
            "out": SimpleNamespace(iterator=Pointer(output.reshape(-1))),
        }
        environment[function.name](*(arguments[arg.arg] for arg in function.args.args))
    return torch.from_numpy(output), calls


class TestLocalAtomicCPU(unittest.TestCase):
    def test_local_atomic_generated_counts_repeats_tails(self):
        for rows, width, bins, repeats in (
            (1, 1, 1, 1),
            (3, 65, 17, 3),
            (2, 257, 31, 2),
        ):
            for dtype in (torch.int32, torch.float32):
                with self.subTest(
                    rows=rows, width=width, bins=bins, repeats=repeats, dtype=dtype
                ):
                    x = torch.arange(rows * width, dtype=dtype).reshape(rows, width) % 7
                    source = _atomic_logical_axes_codegen(
                        _local_atomic_histogram, (x, bins, repeats, "plain"), []
                    )
                    actual, calls = _simulate_local_atomic_program(
                        source, x, (rows, bins)
                    )
                    expected = torch.full((rows, bins), 3, dtype=dtype)
                    expected.scatter_add_(
                        1, (torch.arange(width) % bins).expand(rows, -1), x * repeats
                    )
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    self.assertEqual(len(calls), rows * width * repeats)
                    self.assertIn("scope='cta'", source)
                    self.assertIn("block=(128, 1, 1)", source)

    def test_local_atomic_nested_loops_scalar_contributions_and_global_flush(self):
        x = torch.arange(3 * 65, dtype=torch.int32).reshape(3, 65) % 7
        indices = (torch.arange(65) % 17).expand(3, -1)
        for mode in ("nested", "counts"):
            source = _atomic_logical_axes_codegen(
                _local_atomic_histogram, (x, 17, 2, mode), []
            )
            actual, calls = _simulate_local_atomic_program(source, x, (3, 17))
            expected = torch.full((3, 17), 3, dtype=torch.int32)
            expected.scatter_add_(
                1, indices, x * 6 if mode == "nested" else torch.full_like(x, 2)
            )
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertEqual(len(calls), x.numel() * (4 if mode == "nested" else 2))
        out = torch.zeros((1, 17), dtype=x.dtype)
        source = _atomic_logical_axes_codegen(_local_atomic_flush, (x, out), [])
        actual, calls = _simulate_local_atomic_program(source, x, (1, 17))
        expected = torch.zeros_like(out).scatter_add_(
            1, indices.reshape(1, -1), x.reshape(1, -1)
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(len(calls), x.numel() + 3 * 17)

    def test_local_atomic_masks_logical_non_power_of_two_bounds(self):
        x = torch.arange(2 * 65, dtype=torch.int32).reshape(2, 65)
        source = _atomic_logical_axes_codegen(
            _local_atomic_histogram, (x, 17, 2, "bounds"), []
        )
        actual, calls = _simulate_local_atomic_program(source, x, (2, 17))
        expected = 3 + 2 * x[:, 1:18]
        expected[:, -1] += 2 * x[:, 0]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(len(calls), 2 * 18 * 2)

    def test_local_atomic_valid_negative_indices_and_flush(self):
        x = torch.arange(1, 35, dtype=torch.int32).reshape(2, 17)
        for mode in ("last", "first", "tensor"):
            source = _atomic_logical_axes_codegen(_local_atomic_negative, (x, mode), [])
            actual, calls = _simulate_local_atomic_program(source, x, (2, 17))
            expected = torch.zeros_like(x)
            if mode == "tensor":
                expected.copy_(x)
            else:
                expected[:, -1 if mode == "last" else 0] = x[:, 0]
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertEqual(len(calls), x.numel() if mode == "tensor" else 2)
        out = torch.zeros_like(x)
        source = _atomic_logical_axes_codegen(
            _local_atomic_negative_flush, (x, out), []
        )
        actual, calls = _simulate_local_atomic_program(source, x, x.shape)
        torch.testing.assert_close(actual, x, rtol=0, atol=0)
        self.assertEqual(len(calls), 2 * x.numel())

    def test_local_atomic_invalid_static_index_rejects(self):
        from helion import exc

        with self.assertRaisesRegex(exc.InvalidConfig, "static index"):
            _atomic_logical_axes_codegen(
                _local_atomic_negative, (torch.ones((2, 17)), "invalid"), []
            )

    def test_local_atomic_shared_subnormal_arithmetic(self):
        tiny = torch.finfo(torch.float32).tiny
        x = torch.full((2, 2), tiny / 2)
        source = _atomic_logical_axes_codegen(_local_atomic_subnormal, (x,), [])
        actual, calls = _simulate_local_atomic_program(source, x, (2, 1))
        torch.testing.assert_close(actual, torch.full((2, 1), tiny), rtol=0, atol=0)
        self.assertEqual(len(calls), 4)
        # Global FP32 atomic inputs would each flush to zero, unlike shared adds.
        self.assertTrue(torch.all(x.abs() < tiny))

    def test_local_atomic_rejects_unsupported_ownership(self):
        from helion import exc

        for mode in ("alias", "result", "release", "divergent", "early_read"):
            with self.subTest(mode=mode), self.assertRaises(exc.InvalidConfig):
                _atomic_logical_axes_codegen(
                    _local_atomic_histogram, (torch.ones((2, 17)), 17, 2, mode), []
                )

    def test_local_atomic_rejects_self_reading_updates(self):
        from helion import exc

        with self.assertRaisesRegex(exc.InvalidConfig, "do not read"):
            _atomic_logical_axes_codegen(
                _local_atomic_histogram, (torch.ones((2, 32)), 32, 2, "self_value"), []
            )

    def test_local_atomic_capacity_rejection(self):
        from helion import exc

        with self.assertRaisesRegex(exc.InvalidConfig, "shared bytes"):
            _atomic_logical_axes_codegen(
                _local_atomic_histogram, (torch.ones((1, 17)), 131072, 1, "plain"), []
            )


@onlyBackends("cute")
class TestLocalAtomicNative(TestCase):
    def test_local_atomic_counts_repeats_tails(self):
        for dtype in (torch.int32, torch.float32):
            x = torch.arange(3 * 257, device=DEVICE, dtype=dtype).reshape(3, 257) % 7
            _, actual = code_and_output(_local_atomic_histogram, (x, 17, 3, "plain"))
            expected = torch.full((3, 17), 3, dtype=dtype, device=DEVICE)
            expected.scatter_add_(
                1, (torch.arange(257, device=DEVICE) % 17).expand(3, -1), x * 3
            )
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

        x = torch.arange(1, 35, device=DEVICE, dtype=torch.int32).reshape(2, 17)
        for mode in ("last", "first", "tensor"):
            _, actual = code_and_output(_local_atomic_negative, (x, mode))
            expected = torch.zeros_like(x)
            if mode == "tensor":
                expected.copy_(x)
            else:
                expected[:, -1 if mode == "last" else 0] = x[:, 0]
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        out = torch.zeros_like(x)
        _, actual = code_and_output(_local_atomic_negative_flush, (x, out))
        torch.testing.assert_close(actual, x, rtol=0, atol=0)

    def test_local_atomic_nested_and_global_flush(self):
        x = torch.arange(3 * 65, device=DEVICE, dtype=torch.int32).reshape(3, 65) % 7
        _, actual = code_and_output(_local_atomic_histogram, (x, 17, 2, "nested"))
        indices = (torch.arange(65, device=DEVICE) % 17).expand(3, -1)
        expected = torch.full((3, 17), 3, device=DEVICE, dtype=x.dtype).scatter_add_(
            1, indices, x * 6
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        out = torch.zeros((1, 17), device=DEVICE, dtype=x.dtype)
        _, actual = code_and_output(_local_atomic_flush, (x, out))
        expected = torch.zeros_like(out).scatter_add_(
            1, indices.reshape(1, -1), x.reshape(1, -1)
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_local_atomic_preserves_subnormals(self):
        tiny = torch.finfo(torch.float32).tiny
        x = torch.full((2, 2), tiny / 2, device=DEVICE)
        _, actual = code_and_output(_local_atomic_subnormal, (x,))
        torch.testing.assert_close(
            actual, torch.full((2, 1), tiny, device=DEVICE), rtol=0, atol=0
        )


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _register_load_histogram(x, repeats: hl.constexpr, capacity: hl.constexpr):
    out = torch.empty((x.size(0), 17), dtype=torch.float32, device=x.device)
    for row in hl.grid(x.size(0)):
        local = hl.zeros([17], dtype=torch.float32)
        for iteration in range(repeats):
            columns = hl.arange(capacity)
            values = hl.load(x, [row, columns], extra_mask=columns < x.size(1))
            contribution = torch.where(
                columns < x.size(1),
                values.to(torch.float32) + iteration,
                0.0,
            )
            indices = columns % 17
            hl.atomic_add(local, [indices], contribution)
            hl.atomic_add(local, [indices], contribution * 2)
        out[row, :] = local
    return out


def _register_load_codegen(x, enabled, threads=128, repeats=3):
    from unittest.mock import patch

    from test._cute_binding import _cpu_bind
    from test._cute_binding import _forbid_native_compile
    from test._cute_binding import _mock_cuda_unavailable
    from test.cute_population_contracts import _target

    with (
        _mock_cuda_unavailable(),
        _target(),
        _forbid_native_compile(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CPU only")),
    ):
        bound = _cpu_bind(
            _register_load_histogram, (x, repeats, 1 << (x.size(1) - 1).bit_length())
        )
        config = bound.config_spec.default_config()
        config.config.update(
            cute_fragment_register_loads=enabled, cute_fragment_threads=threads
        )
        return bound.to_code(config)


def _simulate_register_load_program(
    source, x, threads, *, host_tensors=None, lane_order=None
):
    """Execute each actual thread as a coroutine, rendezvousing at CTA barriers."""
    import ast
    import operator
    from types import SimpleNamespace

    import numpy as np

    state = {"lane": 0, "row": 0, "shared": [], "barriers": 0}

    class Pointer:
        def __init__(self, values, offset=0):
            self.values, self.offset = values, int(offset)

        def __add__(self, offset):
            return Pointer(self.values, self.offset + int(offset))

        @property
        def llvm_ptr(self):
            return self

        def load(self):
            assert 0 <= self.offset < self.values.numel()
            return self.values.flatten()[self.offset].item()

        def store(self, value):
            assert 0 <= self.offset < self.values.numel()
            self.values.flatten()[self.offset] = float(value)

    class Shared:
        def __init__(self, count):
            self.values = torch.full((count,), float("nan"))
            self.iterator = Pointer(self.values)

        def __getitem__(self, index):
            return (self.iterator + index).load()

        def __setitem__(self, index, value):
            (self.iterator + index).store(value)

    class Allocator:
        def __init__(self):
            self.ordinal = 0

        def allocate_tensor(self, dtype, layout, byte_alignment):
            assert byte_alignment == 16
            ordinal = self.ordinal
            self.ordinal += 1
            if len(state["shared"]) == ordinal:
                state["shared"].append(Shared(layout[0]))
            return state["shared"][ordinal]

    def atomic_add(pointer, value, *, sem, scope):
        assert sem == "relaxed" and scope in ("cta", "gpu")
        old = pointer.load()
        assert not np.isnan(old), "atomic before shared initialization"
        pointer.store(np.float32(np.float32(old) + np.float32(value)))
        return old

    class Barriers(ast.NodeTransformer):
        def visit_Expr(self, node):
            if (
                isinstance(node.value, ast.Call)
                and ast.unparse(node.value.func) == "cute.arch.sync_threads"
            ):
                return ast.copy_location(
                    ast.Expr(ast.Yield(ast.Constant(node.lineno))), node
                )
            return self.generic_visit(node)

    tree = ast.parse(source)
    fn = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name.startswith("_helion_")
    )
    fn.decorator_list = []
    fn = Barriers().visit(fn)
    tree = ast.fix_missing_locations(
        ast.Module(
            body=[*[n for n in tree.body if isinstance(n, ast.Assign)], fn],
            type_ignores=[],
        )
    )
    out = torch.full((x.size(0), 17), -999.0)
    env = {
        "operator": operator,
        "cutlass": SimpleNamespace(
            Int32=np.int32,
            Int64=np.int64,
            Float32=np.float32,
            Float64=np.float64,
            Float16=np.float16,
            BFloat16=lambda value: torch.tensor(float(value)).bfloat16().item(),
            Boolean=bool,
            utils=SimpleNamespace(SmemAllocator=Allocator),
        ),
        "cute": SimpleNamespace(
            make_layout=lambda shape: shape,
            arch=SimpleNamespace(
                thread_idx=lambda: (state["lane"], 0, 0),
                block_idx=lambda: (state["row"], 0, 0),
                atomic_add=atomic_add,
            ),
        ),
    }
    exec(compile(tree, "<actual-register-load-threads>", "exec"), env)
    for row in range(x.size(0)):
        state.update(row=row, shared=[])
        args = {
            "x": SimpleNamespace(iterator=Pointer(x)),
            "out": SimpleNamespace(iterator=Pointer(out)),
        }
        if host_tensors is not None:
            args.update(
                {
                    name: SimpleNamespace(iterator=Pointer(value))
                    for name, value in host_tensors.items()
                }
            )
        order = list(range(threads)) if lane_order is None else lane_order
        assert sorted(order) == list(range(threads))
        lanes = [
            env[fn.name](*(args[arg.arg] for arg in fn.args.args))
            for _ in range(threads)
        ]
        while True:
            reached = []
            for lane in order:
                program = lanes[lane]
                state["lane"] = lane
                try:
                    reached.append(next(program))
                except StopIteration:
                    reached.append(None)
            assert len(set(reached)) == 1, "divergent CTA barriers"
            if reached[0] is None:
                break
            state["barriers"] += 1
    return out, state["barriers"]


class TestFragmentRegisterLoadsCPU(unittest.TestCase):
    def test_register_loads_typing_tails_and_loop_refresh(self):
        for dtype in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
            torch.int32,
            torch.int64,
        ):
            for columns in (17, 65, 128, 129):
                with self.subTest(dtype=dtype, columns=columns):
                    x = (torch.arange(3 * columns).reshape(3, columns) % 11).to(dtype)
                    # One-element lanes follow the reference's accumulation
                    # order. Multi-element fallback uses exact integer sums;
                    # its legal per-thread atomic interleaving differs.
                    if dtype.is_floating_point and columns <= 128:
                        x = (x.double() * 0.1 + 2**-30).to(dtype)
                    elif dtype == torch.int64 and columns <= 128:
                        x += (1 << 24) + 1
                    before = _register_load_codegen(x, False)
                    after = _register_load_codegen(x, True)
                    expected = torch.zeros((3, 17))
                    index = (torch.arange(columns) % 17).expand(3, -1)
                    for iteration in range(3):
                        expected.scatter_add_(1, index, x.float() + iteration)
                        expected.scatter_add_(1, index, (x.float() + iteration) * 2)
                    old, old_barriers = _simulate_register_load_program(before, x, 128)
                    actual, barriers = _simulate_register_load_program(after, x, 128)
                    torch.testing.assert_close(old, expected, rtol=0, atol=0)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    self.assertEqual("fragment_register_load" in after, columns <= 128)
                    self.assertEqual(
                        old_barriers - barriers, 3 * x.size(0) if columns <= 128 else 0
                    )

    def test_register_loads_preserve_shared_subnormals(self):
        x = torch.full((2, 17), torch.finfo(torch.float32).tiny / 2)
        for enabled in (False, True):
            source = _register_load_codegen(x, enabled, repeats=1)
            actual, _barriers = _simulate_register_load_program(source, x, 128)
            expected = torch.zeros((2, 17))
            for iteration in range(1):
                value = x + iteration
                expected += value
                expected += value * 2
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@onlyBackends("cute")
class TestFragmentRegisterLoadsNative(TestCase):
    def test_register_loads_values_and_fallback(self):
        for threads, columns in ((32, 17), (128, 65), (128, 129), (512, 129)):
            with self.subTest(threads=threads, columns=columns):
                x = (
                    torch.arange(3 * columns, device=DEVICE).reshape(3, columns) % 11
                ).float()
                before = x.clone()
                args = (x, 3, 1 << (columns - 1).bit_length())
                reference = torch.zeros((3, 17), device=DEVICE)
                indices = (torch.arange(columns, device=DEVICE) % 17).expand(3, -1)
                for iteration in range(3):
                    reference.scatter_add_(1, indices, (x + iteration) * 3)
                for enabled in (False, True):
                    code, actual = code_and_output(
                        _register_load_histogram,
                        args,
                        cute_fragment_threads=threads,
                        cute_fragment_register_loads=enabled,
                    )
                    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
                    torch.testing.assert_close(x, before, rtol=0, atol=0)
                    self.assertEqual(
                        "fragment_register_load" in code, enabled and args[2] <= threads
                    )

    def test_register_loads_cross_warp_alias_ordering(self):
        original = torch.arange(256, device=DEVICE).reshape(2, 128).float()
        z = torch.full_like(original, 2)
        indices = (torch.arange(128, device=DEVICE) % 17).expand(2, -1)
        expected = torch.zeros((2, 17), device=DEVICE).scatter_add_(
            1, indices, original + z
        )
        for mode in ("dependent_store", "unrelated_atomic"):
            for aliased in (False, True):
                for enabled in (False, True):
                    with self.subTest(mode=mode, aliased=aliased, enabled=enabled):
                        x = original.clone()
                        y = x.view_as(x) if aliased else torch.full_like(x, 500)
                        before_y = y.clone()
                        _code, actual = code_and_output(
                            _register_load_cross_lane_effect,
                            (x, y, z, mode),
                            cute_fragment_register_loads=enabled,
                        )
                        contribution = (
                            original + 1
                            if mode == "dependent_store"
                            else torch.arange(128, device=DEVICE).float().expand(2, -1)
                            + 1000
                        )
                        target = torch.flip(contribution, [1])
                        if mode == "unrelated_atomic":
                            target += before_y
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                        torch.testing.assert_close(y, target, rtol=0, atol=0)
                        torch.testing.assert_close(
                            z, torch.full_like(z, 2), rtol=0, atol=0
                        )
                        if not aliased:
                            torch.testing.assert_close(x, original, rtol=0, atol=0)

    def test_register_loads_index_logical_domain(self):
        for width in (17, 65, 129):
            x = (torch.arange(2 * width, device=DEVICE).reshape(2, width) % 7).int()
            expected = torch.zeros((2, 17), device=DEVICE).scatter_add_(
                1, x.long(), torch.full(x.shape, 6.0, device=DEVICE)
            )
            for enabled in (False, True):
                with self.subTest(width=width, enabled=enabled):
                    before = x.clone()
                    code, actual = code_and_output(
                        _register_load_index_domain,
                        (x,),
                        cute_fragment_register_loads=enabled,
                    )
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    torch.testing.assert_close(x, before, rtol=0, atol=0)
                    self.assertEqual(
                        "fragment_register_load" in code, enabled and width <= 128
                    )

    def test_register_loads_shared_subnormals(self):
        x = torch.full((2, 17), torch.finfo(torch.float32).tiny / 2, device=DEVICE)
        for enabled in (False, True):
            _code, actual = code_and_output(
                _register_load_histogram,
                (x, 1, 32),
                cute_fragment_register_loads=enabled,
            )
            torch.testing.assert_close(actual, x * 3, rtol=0, atol=0)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _register_load_mutation(x):
    out = torch.empty((x.size(0), 17), dtype=torch.float32, device=x.device)
    for row in hl.grid(x.size(0)):
        local = hl.zeros([17], dtype=torch.float32)
        values = x[row, :]
        x[row, :] = values + 1
        indices = hl.arange(x.size(1)) % 17
        hl.atomic_add(local, [indices], values)
        refreshed = x[row, :]
        hl.atomic_add(local, [indices], refreshed)
        out[row, :] = local
    return out


class TestFragmentRegisterLifetimeCPU(unittest.TestCase):
    def test_register_load_snapshots_survive_alias_store_and_reload(self):
        from unittest.mock import patch

        from test._cute_binding import _cpu_bind
        from test._cute_binding import _forbid_native_compile
        from test._cute_binding import _mock_cuda_unavailable
        from test.cute_population_contracts import _target

        x = torch.arange(64).reshape(2, 32).float()
        expected = torch.zeros((2, 17))
        index = (torch.arange(32) % 17).expand(2, -1)
        expected.scatter_add_(1, index, x)
        expected.scatter_add_(1, index, x + 1)
        for enabled in (False, True):
            with (
                _mock_cuda_unavailable(),
                _target(),
                _forbid_native_compile(),
                patch("torch.cuda._lazy_init", side_effect=AssertionError("CPU only")),
            ):
                bound = _cpu_bind(_register_load_mutation, (x,))
                config = bound.config_spec.default_config()
                config.config["cute_fragment_register_loads"] = enabled
                if enabled:
                    self.assertFalse(
                        bound.config_spec.cute_fragment_register_load_root_ids
                    )
                    with self.assertRaisesRegex(
                        helion.exc.InvalidConfig, "lane-private"
                    ):
                        bound.to_code(config)
                    continue
                source = bound.to_code(config)
            actual_input = x.clone()
            actual, _barriers = _simulate_register_load_program(
                source, actual_input, 128
            )
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(actual_input, x + 1, rtol=0, atol=0)
            self.assertEqual("fragment_register_load" in source, enabled)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _register_load_cross_lane_effect(x, y, z, mode: hl.constexpr):
    out = torch.empty((x.size(0), 17), dtype=torch.float32, device=x.device)
    for row in hl.grid(x.size(0)):
        local = hl.zeros([17], dtype=torch.float32)
        values = x[row, :]
        columns = hl.arange(x.size(1))
        if mode == "dependent_store":
            y[row, x.size(1) - 1 - columns] = values + 1
        elif mode == "unrelated_store":
            y[row, x.size(1) - 1 - columns] = columns.to(torch.float32) + 1000
        elif mode == "dependent_atomic":
            hl.atomic_add(y, [row, x.size(1) - 1 - columns], values + 1)
        else:
            hl.atomic_add(
                y,
                [row, x.size(1) - 1 - columns],
                columns.to(torch.float32) + 1000,
            )
        hl.atomic_add(local, [columns % 17], values)
        other = z[row, :]
        hl.atomic_add(local, [columns % 17], other)
        out[row, :] = local
    return out


class TestFragmentRegisterEffectsCPU(unittest.TestCase):
    def test_cross_warp_host_effects_and_aliased_arguments(self):
        from unittest.mock import patch

        from test._cute_binding import _cpu_bind
        from test._cute_binding import _forbid_native_compile
        from test._cute_binding import _mock_cuda_unavailable
        from test.cute_population_contracts import _target

        x = torch.arange(256).reshape(2, 128).float()
        z = torch.full_like(x, 2)
        index = (torch.arange(128) % 17).expand(2, -1)
        expected = torch.zeros((2, 17)).scatter_add_(1, index, x + z)
        orders = [
            list(range(128)),
            [
                lane
                for warp in (2, 0, 3, 1)
                for lane in range(warp * 32, (warp + 1) * 32)
            ],
        ]
        for mode in (
            "dependent_store",
            "unrelated_store",
            "dependent_atomic",
            "unrelated_atomic",
        ):
            for aliased in (False, True):
                y = x.view_as(x) if aliased else torch.full_like(x, 500)
                with self.subTest(mode=mode, aliased=aliased):
                    with (
                        _mock_cuda_unavailable(),
                        _target(),
                        _forbid_native_compile(),
                        patch(
                            "torch.cuda._lazy_init",
                            side_effect=AssertionError("CPU only"),
                        ),
                    ):
                        bound = _cpu_bind(
                            _register_load_cross_lane_effect, (x, y, z, mode)
                        )
                        config = bound.config_spec.default_config()
                        sources = []
                        for enabled in (False, True):
                            config.config["cute_fragment_register_loads"] = enabled
                            sources.append(bound.to_code(config))
                    for source in sources:
                        for order in orders:
                            actual_x = x.clone()
                            actual_y = (
                                actual_x.view_as(actual_x) if aliased else y.clone()
                            )
                            actual, _barriers = _simulate_register_load_program(
                                source,
                                actual_x,
                                128,
                                host_tensors={"y": actual_y, "z": z},
                                lane_order=order,
                            )
                            contribution = (
                                x + 1
                                if mode.startswith("dependent")
                                else torch.arange(128).float().expand(2, -1) + 1000
                            )
                            target = torch.flip(contribution, [1])
                            if mode.endswith("atomic"):
                                target = target + (x if aliased else y)
                            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                            torch.testing.assert_close(actual_y, target, rtol=0, atol=0)
                            if not aliased:
                                torch.testing.assert_close(actual_x, x, rtol=0, atol=0)
                    # The independent z input remains eligible even when x must
                    # retain its materialization barrier. This exercises a real
                    # mixed root, rather than only rejecting the config key.
                    self.assertIn("fragment_register_load", sources[1])

    def test_runtime_alias_matrix_guards_rebinding_and_views(self):
        from unittest.mock import patch

        from test._cute_binding import _forbid_native_compile
        from test._cute_binding import _mock_cuda_unavailable
        from test.cute_population_contracts import _target

        from helion._compiler.cute.local_atomic import _reachable_graphs
        from helion._compiler.cute.register_loads import host_load_is_readonly
        from helion.language import memory_ops

        kernel = helion.kernel(
            _register_load_cross_lane_effect.fn,
            backend="cute",
            static_shapes=True,
            autotune_effort="none",
        )
        storage = torch.arange(257).float()
        x = storage[:256].view(2, 128)
        y = torch.zeros_like(x)
        z = torch.full_like(x, 2)
        with (
            _mock_cuda_unavailable(),
            _target(),
            _forbid_native_compile(),
            patch("torch.cuda._lazy_init", side_effect=AssertionError("CPU only")),
        ):
            disjoint = kernel.bind((x, y, z, "dependent_store"))
            for alias in (
                x,
                x.view_as(x),
                torch.from_dlpack(x),
                storage[1:].view_as(x),
            ):
                with self.subTest(offset=alias.storage_offset(), same=alias is x):
                    overlapping = kernel.bind((x, alias, z, "dependent_store"))
                    self.assertIsNot(disjoint, overlapping)
                    self.assertIs(kernel.bind((x, y, z, "dependent_store")), disjoint)
                    for bound, expected in ((disjoint, True), (overlapping, False)):
                        host = bound.host_function
                        graphs = _reachable_graphs(host.device_ir.graphs)
                        loads = [
                            n
                            for graph in graphs
                            for n in graph.graph.nodes
                            if n.target is memory_ops.load and n.args[0].args == ("x",)
                        ]
                        self.assertTrue(loads)
                        with bound.env, host:
                            for load in loads:
                                self.assertEqual(
                                    host_load_is_readonly(load, bound.env, graphs),
                                    expected,
                                )
                    # Even a previously safe plan cannot consume missing or
                    # contradictory alias facts during later code generation.
                    host = disjoint.host_function
                    graphs = _reachable_graphs(host.device_ir.graphs)
                    load = next(
                        n
                        for g in graphs
                        for n in g.graph.nodes
                        if n.target is memory_ops.load and n.args[0].args == ("x",)
                    )
                    with disjoint.env, host:
                        with patch.object(
                            disjoint.env,
                            "bound_runtime_input_specialization_results",
                            {},
                        ):
                            self.assertFalse(
                                host_load_is_readonly(load, disjoint.env, graphs)
                            )
                        with disjoint.env.use_runtime_arg_values(
                            {"x": x, "y": alias, "z": z}
                        ):
                            self.assertFalse(
                                host_load_is_readonly(load, disjoint.env, graphs)
                            )


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _local_atomic_index_domain(x, bins: hl.constexpr, mode: hl.constexpr):
    out = torch.empty((x.size(0), bins), dtype=x.dtype, device=x.device)
    for row in hl.grid(x.size(0)):
        local = hl.zeros([bins], dtype=x.dtype)
        for iteration in range(3):
            if mode == "load_mask":
                values = hl.load(
                    x,
                    [row, hl.arange(x.size(1))],
                    extra_mask=hl.arange(x.size(1)) < x.size(1) // 2,
                )
            else:
                values = hl.load(x, [row, hl.arange(x.size(1))])
            contribution = values + iteration
            indices = hl.arange(x.size(1)) % bins
            if mode == "negative":
                indices = indices - bins
            elif mode == "clamp":
                indices = torch.clamp(hl.arange(x.size(1)), max=bins - 1)
            if mode == "masked":
                contribution = torch.where(values > 0, contribution, 0)
            if mode == "counts":
                hl.atomic_add(local, [indices], iteration + 1)
            else:
                hl.atomic_add(local, [indices], contribution)
                hl.atomic_add(local, [indices], contribution * 2)
        out[row, :] = local
        if mode == "global":
            hl.atomic_add(out, [row, hl.arange(x.size(1)) % bins], 1)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _local_atomic_broadcast_domain(x, planes: hl.constexpr):
    out = torch.empty((x.size(0), 17), dtype=x.dtype, device=x.device)
    for row in hl.grid(x.size(0)):
        local = hl.zeros([17], dtype=x.dtype)
        # The singleton row coordinate must broadcast. Each physical axis
        # retains its own logical extent, even when both pad to the same size.
        index = (hl.arange(planes)[:, None] + hl.arange(x.size(1))[None, :]) % 17
        for iteration in range(3):
            hl.atomic_add(local, [index], iteration + 1)
        out[row, :] = local
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _local_atomic_loaded_index_domain(x, mode: hl.constexpr):
    out = torch.empty((x.size(0), 17), dtype=x.dtype, device=x.device)
    for row in hl.grid(x.size(0)):
        local = hl.zeros([17], dtype=x.dtype)
        if mode == "factory":
            indices = hl.full([x.size(1)], 1, dtype=torch.int32)
        else:
            indices = x[row, :].to(torch.int32) % 17
        for iteration in range(3):
            if mode == "carry":
                indices = (indices + 1) % 17
            hl.atomic_add(local, [indices], iteration + 1)
        out[row, :] = local
    return out


def _local_atomic_index_domain_reference(x, bins, mode):
    indices = torch.arange(x.size(1))
    indices = indices.clamp(max=bins - 1) if mode == "clamp" else indices % bins
    result = torch.zeros((x.size(0), bins), dtype=x.dtype, device=x.device)
    indices = indices.to(x.device).expand(x.size(0), -1)
    for iteration in range(3):
        values = (
            torch.where(torch.arange(x.size(1), device=x.device) < x.size(1) // 2, x, 0)
            if mode == "load_mask"
            else x
        )
        contribution = values + iteration
        if mode == "masked":
            contribution = torch.where(x > 0, contribution, 0)
        if mode == "counts":
            contribution = torch.full_like(x, iteration + 1)
        else:
            contribution = contribution * 3
        result.scatter_add_(1, indices, contribution)
    if mode == "global":
        result.scatter_add_(1, indices, torch.ones_like(x))
    return result


class TestFragmentAtomicIndexDomainCPU(unittest.TestCase):
    def test_loaded_factory_and_carried_index_domains(self):
        for width in (17, 31, 65):
            for mode in ("loaded", "factory", "carry"):
                with self.subTest(width=width, mode=mode):
                    x = torch.arange(2 * width, dtype=torch.int32).reshape(2, width) % 7
                    source = _atomic_logical_axes_codegen(
                        _local_atomic_loaded_index_domain, (x, mode), []
                    )
                    result, calls = _simulate_local_atomic_program(source, x, (2, 17))
                    indices = torch.ones_like(x) if mode == "factory" else x.clone()
                    expected = torch.zeros((2, 17), dtype=x.dtype)
                    for iteration in range(3):
                        if mode == "carry":
                            indices = (indices + 1) % 17
                        expected.scatter_add_(
                            1, indices.long(), torch.full_like(x, iteration + 1)
                        )
                    torch.testing.assert_close(result, expected, rtol=0, atol=0)
                    self.assertEqual(len(calls), 2 * width * 3)

    def test_logical_tail_contributions_after_arithmetic(self):
        for width, bins in ((1, 1), (17, 17), (31, 17), (33, 31), (65, 17)):
            for dtype in (torch.float32, torch.int32):
                for mode in (
                    "plain",
                    "negative",
                    "clamp",
                    "masked",
                    "load_mask",
                    "counts",
                    "global",
                ):
                    with self.subTest(width=width, bins=bins, dtype=dtype, mode=mode):
                        x = torch.ones((2, width), dtype=dtype)
                        x[:, ::3] = 0
                        source = _atomic_logical_axes_codegen(
                            _local_atomic_index_domain, (x, bins, mode), []
                        )
                        result, calls = _simulate_local_atomic_program(
                            source, x, (2, bins)
                        )
                        torch.testing.assert_close(
                            result,
                            _local_atomic_index_domain_reference(x, bins, mode),
                            rtol=0,
                            atol=0,
                        )
                        self.assertEqual(
                            len(calls),
                            2 * width * (3 if mode == "counts" else 6)
                            + (2 * width if mode == "global" else 0),
                        )

    def test_broadcast_axes_keep_distinct_logical_extents(self):
        for planes, width in ((1, 17), (17, 1), (17, 31), (31, 17), (3, 65)):
            with self.subTest(planes=planes, width=width):
                x = torch.ones((2, width), dtype=torch.int32)
                source = _atomic_logical_axes_codegen(
                    _local_atomic_broadcast_domain, (x, planes), []
                )
                result, calls = _simulate_local_atomic_program(source, x, (2, 17))
                indices = (torch.arange(planes)[:, None] + torch.arange(width)) % 17
                expected = torch.zeros((2, 17), dtype=x.dtype)
                expected.scatter_add_(
                    1,
                    indices.flatten().expand(2, -1),
                    torch.full((2, planes * width), 6, dtype=x.dtype),
                )
                torch.testing.assert_close(result, expected, rtol=0, atol=0)
                self.assertEqual(len(calls), 2 * planes * width * 3)


@onlyBackends("cute")
class TestFragmentAtomicIndexDomainNative(TestCase):
    def test_loaded_index_loop_carry(self):
        x = torch.arange(130, dtype=torch.int32, device=DEVICE).reshape(2, 65) % 7
        _, result = code_and_output(_local_atomic_loaded_index_domain, (x, "carry"))
        expected = torch.zeros((2, 17), dtype=x.dtype, device=DEVICE)
        for iteration in range(3):
            indices = (x + iteration + 1) % 17
            expected.scatter_add_(1, indices.long(), torch.full_like(x, iteration + 1))
        torch.testing.assert_close(result, expected, rtol=0, atol=0)

    def test_logical_tail_after_arithmetic(self):
        for width, mode in (
            (17, "plain"),
            (65, "negative"),
            (31, "masked"),
            (33, "global"),
        ):
            with self.subTest(width=width, mode=mode):
                x = torch.ones((2, width), dtype=torch.float32, device=DEVICE)
                x[:, ::3] = 0
                _, result = code_and_output(_local_atomic_index_domain, (x, 17, mode))
                torch.testing.assert_close(
                    result,
                    _local_atomic_index_domain_reference(x, 17, mode),
                    rtol=0,
                    atol=0,
                )

    def test_broadcast_axes_keep_distinct_extents(self):
        x = torch.ones((2, 31), dtype=torch.int32, device=DEVICE)
        _, result = code_and_output(_local_atomic_broadcast_domain, (x, 17))
        indices = (
            torch.arange(17, device=DEVICE)[:, None] + torch.arange(31, device=DEVICE)
        ) % 17
        expected = torch.zeros((2, 17), dtype=x.dtype, device=DEVICE)
        expected.scatter_add_(
            1,
            indices.flatten().expand(2, -1),
            torch.full((2, 17 * 31), 6, dtype=x.dtype, device=DEVICE),
        )
        torch.testing.assert_close(result, expected, rtol=0, atol=0)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _register_load_index_domain(x):
    out = torch.empty((x.size(0), 17), dtype=torch.float32, device=x.device)
    for row in hl.grid(x.size(0)):
        local = hl.zeros([17], dtype=torch.float32)
        for iteration in range(3):
            columns = hl.arange(x.size(1))[None, :]
            indices = hl.load(x, [row, columns]).to(torch.int32) % 17
            hl.atomic_add(local, [indices], iteration + 1)
        out[row, :] = local
    return out


class TestFragmentRegisterDomainCPU(unittest.TestCase):
    def test_promoted_indices_retain_logical_tail_domain(self):
        from unittest.mock import patch

        from test._cute_binding import _cpu_bind
        from test._cute_binding import _forbid_native_compile
        from test._cute_binding import _mock_cuda_unavailable
        from test.cute_population_contracts import _target

        for width in (17, 31, 65, 129):
            x = (torch.arange(2 * width).reshape(2, width) % 7).int()
            expected = torch.zeros((2, 17)).scatter_add_(
                1, x.long(), torch.full(x.shape, 6.0)
            )
            for enabled in (False, True):
                with self.subTest(width=width, enabled=enabled):
                    with (
                        _mock_cuda_unavailable(),
                        _target(),
                        _forbid_native_compile(),
                        patch(
                            "torch.cuda._lazy_init",
                            side_effect=AssertionError("CPU only"),
                        ),
                    ):
                        bound = _cpu_bind(_register_load_index_domain, (x,))
                        config = bound.config_spec.default_config()
                        config.config["cute_fragment_register_loads"] = enabled
                        source = bound.to_code(config)
                    actual, _barriers = _simulate_register_load_program(source, x, 128)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    self.assertEqual(
                        "fragment_register_load" in source, enabled and width <= 128
                    )


if __name__ == "__main__":
    unittest.main()
