from __future__ import annotations

import operator
import re
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
from helion._testing import skipUnlessCuteAvailable
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


@helion.kernel(static_shapes=True)
def atomic_add_full_slice_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """atomic_add into ``[tile, :]`` with values computed from two loads."""
    out = torch.zeros_like(x)
    for tile_m in hl.tile(x.size(0)):
        hl.atomic_add(out, [tile_m, slice(None)], x[tile_m, :] * 2 + y[tile_m, :])
        hl.atomic_add(out, [tile_m, slice(None)], y[tile_m, :])
    return out


@helion.kernel(static_shapes=True)
def atomic_add_full_slice_reduce_kernel(
    x: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    """Column sums accumulated through a bare ``[:]`` index."""
    m, n = x.shape
    n = hl.specialize(n)
    out = torch.zeros([n], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        hl.atomic_add(out, [slice(None)], torch.sum(x[tile_m, :] * y[tile_m, :], dim=0))
    return out


@helion.kernel(static_shapes=True)
def atomic_max_full_slice_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """atomic_max into ``[tile, :]``."""
    for tile_m in hl.tile(x.size(0)):
        hl.atomic_max(x, [tile_m, slice(None)], y[tile_m, :])
    return x


@helion.kernel(static_shapes=True)
def atomic_add_partial_slice_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """atomic_add into the partial slice ``[tile, 16:]``."""
    out = torch.zeros_like(x)
    for tile_m in hl.tile(x.size(0)):
        hl.atomic_add(out, [tile_m, slice(16, None)], x[tile_m, 16:] + y[tile_m, 16:])
    return out


@helion.kernel(static_shapes=True)
def atomic_add_two_partial_slices_kernel(
    x: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    """Two atomic_adds on slices of different widths in one tile loop."""
    out = torch.zeros_like(x)
    for tile_m in hl.tile(x.size(0)):
        hl.atomic_add(out, [tile_m, slice(16, None)], x[tile_m, 16:] + y[tile_m, 16:])
        hl.atomic_add(out, [tile_m, slice(None, 16)], y[tile_m, :16])
    return out


@helion.kernel(static_shapes=True)
def atomic_add_invariant_inner_tile_kernel(
    x: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    """atomic_add inside an inner tile loop whose tile it does not index.

    The value only depends on the inner tile's block size, so the loop adds
    ``y.size(0) * x`` in total however the inner loop is tiled.
    """
    out = torch.zeros([x.size(0)], device=x.device, dtype=x.dtype)
    for tile_m in hl.tile(x.size(0)):
        for tile_n in hl.tile(y.size(0)):
            hl.atomic_add(out, [tile_m], x[tile_m] * tile_n.block_size)
    return out


@helion.kernel(static_shapes=True)
def atomic_add_invariant_inner_2d_tile_kernel(
    x: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    """atomic_add inside a 2-D inner tile loop whose tiles it does not index."""
    out = torch.zeros([x.size(0)], device=x.device, dtype=x.dtype)
    for tile_m in hl.tile(x.size(0)):
        for tile_i, tile_j in hl.tile([y.size(0), y.size(1)]):
            hl.atomic_add(
                out, [tile_m], x[tile_m] * (tile_i.block_size * tile_j.block_size)
            )
    return out


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

    @onlyBackends(["triton", "cute"])
    def test_atomic_add_full_slice(self):
        x = torch.randn(96, 48, device=DEVICE, dtype=torch.float32)
        y = torch.randn(96, 48, device=DEVICE, dtype=torch.float32)
        code, result = code_and_output(
            atomic_add_full_slice_kernel,
            (x, y),
            block_sizes=[32],
        )
        torch.testing.assert_close(result, 2 * x + 2 * y)
        if _get_backend() == "cute":
            self.assertIn("atomic_add", code)

    @onlyBackends(["triton", "cute"])
    def test_atomic_add_full_slice_reduce(self):
        x = torch.randn(128, 64, device=DEVICE, dtype=torch.float32)
        y = torch.randn(128, 64, device=DEVICE, dtype=torch.float32)
        _, result = code_and_output(
            atomic_add_full_slice_reduce_kernel,
            (x, y),
            block_sizes=[32],
        )
        torch.testing.assert_close(result, (x * y).sum(dim=0), rtol=1e-4, atol=1e-4)

    @onlyBackends(["triton", "cute"])
    def test_atomic_max_full_slice(self):
        x = torch.randn(64, 48, device=DEVICE, dtype=torch.float32)
        y = torch.randn(64, 48, device=DEVICE, dtype=torch.float32)
        _, result = code_and_output(
            atomic_max_full_slice_kernel,
            (x.clone(), y),
            block_sizes=[32],
        )
        torch.testing.assert_close(result, torch.maximum(x, y))

    @onlyBackends(["triton", "cute"])
    def test_atomic_add_partial_slice(self):
        x = torch.randn(96, 48, device=DEVICE, dtype=torch.float32)
        y = torch.randn(96, 48, device=DEVICE, dtype=torch.float32)
        code, result = code_and_output(
            atomic_add_partial_slice_kernel,
            (x, y),
            block_sizes=[32],
        )
        expected = torch.zeros_like(x)
        expected[:, 16:] = x[:, 16:] + y[:, 16:]
        torch.testing.assert_close(result, expected)
        if _get_backend() == "cute":
            self.assertIn("atomic_add", code)

    @onlyBackends(["triton", "cute"])
    def test_atomic_add_two_partial_slices(self):
        x = torch.randn(96, 48, device=DEVICE, dtype=torch.float32)
        y = torch.randn(96, 48, device=DEVICE, dtype=torch.float32)
        code, result = code_and_output(
            atomic_add_two_partial_slices_kernel,
            (x, y),
            block_sizes=[32],
        )
        expected = torch.zeros_like(x)
        expected[:, 16:] = x[:, 16:] + y[:, 16:]
        expected[:, :16] = y[:, :16]
        torch.testing.assert_close(result, expected)
        if _get_backend() == "cute":
            # The 16-wide slice is walked by a per-thread lane loop.  The
            # 32-wide slice's atomic reads none of that loop's coordinates, so
            # it is placed before the loop (or pinned to its first lane) and
            # fires once per thread instead of once per lane.
            atomic_at = code.find("cute.arch.atomic_add(")
            lane_loop_at = code.find("for synthetic_lane_")
            self.assertTrueIfInNormalMode(
                0 <= atomic_at < lane_loop_at
                or re.search(r"if [^\n]*lane_\d+ == 0", code) is not None
            )

    @onlyBackends(["triton", "cute"])
    def test_atomic_add_invariant_in_inner_tile_loop(self):
        x = torch.randn(64, device=DEVICE, dtype=torch.float32)
        y = torch.randn(128, 8, device=DEVICE, dtype=torch.float32)
        configs = [{}]
        if _get_backend() == "cute":
            # Scalar lane loop and outer x constexpr-vector lane partition: the
            # body ignores the inner tile's lanes, so the nest check splices
            # the lane loop out, or pins the atomic to lane 0 if it keeps it.
            configs = [{}, {"cute_vector_widths": [1, 4]}]
        for config in configs:
            code, result = code_and_output(
                atomic_add_invariant_inner_tile_kernel,
                (x, y),
                block_sizes=[32, 64],
                **config,
            )
            torch.testing.assert_close(result, y.size(0) * x)
            if config:
                self.assertTrueIfInNormalMode(
                    re.search(r"for (vec_)?lane_\d+ in", code) is None
                    or re.search(r"if lane_\d+ == 0 and vec_lane_\d+ == 0", code)
                    is not None
                )

    @onlyBackends(["triton", "cute"])
    def test_atomic_add_invariant_in_inner_2d_tile_loop(self):
        x = torch.randn(64, device=DEVICE, dtype=torch.float32)
        y = torch.randn(32, 64, device=DEVICE, dtype=torch.float32)
        code, result = code_and_output(
            atomic_add_invariant_inner_2d_tile_kernel,
            (x, y),
            block_sizes=[32, 8, 8],
        )
        torch.testing.assert_close(result, y.numel() * x)
        if _get_backend() == "cute":
            # Both inner tile blocks are walked by single-thread lane loops
            # that the body never reads: the loops are eliminated, or the
            # atomic is pinned to their first lanes.
            self.assertTrueIfInNormalMode(
                "for lane_" not in code
                or re.search(r"if lane_\d+ == 0 and lane_\d+ == 0", code) is not None
            )

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
            "_cute_python_mod": operator.mod,
            "cutlass": SimpleNamespace(
                Int32=np.int32,
                Uint32=np.uint32,
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
    source,
    x,
    threads,
    *,
    host_tensors=None,
    lane_order=None,
    atomic_events=None,
    atomic_workers=None,
    memory_events=None,
    scalar_args=None,
):
    """Check generated ownership/rendezvous, not GPU memory-model behavior."""
    import ast
    import operator
    from types import SimpleNamespace

    import numpy as np

    state = {"lane": 0, "row": 0, "shared": [], "barriers": 0}

    class Pointer:
        def __init__(self, values, offset=0, initialized=None):
            self.values, self.offset = values, int(offset)
            self.initialized = initialized

        def __add__(self, offset):
            return Pointer(self.values, self.offset + int(offset), self.initialized)

        @property
        def llvm_ptr(self):
            return self

        def load(self):
            assert 0 <= self.offset < self.values.numel()
            if self.initialized is not None:
                assert self.offset in self.initialized, (
                    "shared pointer read before initialization"
                )
            return self.values.flatten()[self.offset].item()

        def store(self, value):
            assert 0 <= self.offset < self.values.numel()
            self.values.flatten()[self.offset] = float(value)
            if self.initialized is not None:
                self.initialized.add(self.offset)
            elif memory_events is not None:
                memory_events.append(
                    (
                        "store",
                        state["row"],
                        state["lane"],
                        state["barriers"],
                        self.offset,
                    )
                )

    class Shared:
        def __init__(self, count, dtype):
            tensor_dtype = {
                np.int32: torch.int32,
                np.int64: torch.int64,
                bool: torch.bool,
            }.get(dtype)
            self.values = (
                torch.zeros(count, dtype=tensor_dtype)
                if tensor_dtype is not None
                else torch.full((count,), float("nan"))
            )
            self.initialized = set()
            self.iterator = Pointer(self.values, initialized=self.initialized)

        def __getitem__(self, index):
            assert int(index) in self.initialized, "shared read before initialization"
            return (self.iterator + index).load()

        def __setitem__(self, index, value):
            (self.iterator + index).store(value)
            self.initialized.add(int(index))

    class Allocator:
        def __init__(self):
            self.ordinal = 0

        def allocate_tensor(self, dtype, layout, byte_alignment):
            assert byte_alignment == 16
            ordinal = self.ordinal
            self.ordinal += 1
            if len(state["shared"]) == ordinal:
                state["shared"].append(Shared(layout[0], dtype))
            return state["shared"][ordinal]

    def atomic_add(pointer, value, *, sem, scope):
        assert sem in ("relaxed", "acquire", "release", "acq_rel")
        assert scope in ("cta", "gpu") and (scope == "gpu" or sem == "relaxed")
        old = pointer.load()
        assert not np.isnan(old), "atomic before shared initialization"
        if pointer.values.dtype == torch.int32:
            pointer.store(np.add(np.int32(old), np.int32(value), dtype=np.int32))
        else:
            pointer.store(np.float32(np.float32(old) + np.float32(value)))
        if atomic_events is not None:
            atomic_events.append((scope, pointer.offset, old, int(value)))
        if atomic_workers is not None:
            atomic_workers.append((scope, state["lane"]))
        if memory_events is not None:
            memory_events.append(
                ("atomic", state["row"], state["lane"], state["barriers"], sem, scope)
            )
        return old

    def fence():
        if memory_events is not None:
            memory_events.append(
                ("fence", state["row"], state["lane"], state["barriers"])
            )

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

        def visit_Call(self, node):
            function = ast.unparse(node.func)
            if function in ("cute.arch.shuffle_sync", "cute.arch.shuffle_sync_bfly"):
                keywords = {item.arg: item.value for item in node.keywords}
                offset = node.args[1] if len(node.args) > 1 else keywords["offset"]
                assert ast.literal_eval(keywords.get("mask", ast.Constant(-1))) in (
                    -1,
                    0xFFFFFFFF,
                )
                assert (
                    ast.literal_eval(keywords.get("mask_and_clamp", ast.Constant(31)))
                    == 31
                )
                return ast.copy_location(
                    ast.Yield(
                        ast.Tuple(
                            [
                                ast.Constant("shuffle"),
                                ast.Constant(node.lineno),
                                ast.Constant(function.endswith("bfly")),
                                node.args[0],
                                offset,
                            ],
                            ast.Load(),
                        )
                    ),
                    node,
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
        "_cute_python_mod": operator.mod,
        "cutlass": SimpleNamespace(
            Int32=np.int32,
            Uint32=np.uint32,
            Int64=np.int64,
            Float32=np.float32,
            Float64=np.float64,
            Float16=np.float16,
            BFloat16=lambda value: torch.tensor(float(value)).bfloat16().item(),
            Boolean=bool,
            utils=SimpleNamespace(SmemAllocator=Allocator),
        ),
        "cute": SimpleNamespace(
            math=SimpleNamespace(min=np.minimum, max=np.maximum),
            make_layout=lambda shape: shape,
            arch=SimpleNamespace(
                thread_idx=lambda: (state["lane"], 0, 0),
                block_idx=lambda: (state["row"], 0, 0),
                atomic_add=atomic_add,
                fence_acq_rel_gpu=fence,
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
        if scalar_args is not None:
            args.update(scalar_args)
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

        def advance(lane, value=None, *, send=False, generators=lanes):
            state["lane"] = lane
            try:
                return generators[lane].send(value) if send else next(generators[lane])
            except StopIteration:
                return None

        reached = [None] * threads
        for lane in order:
            reached[lane] = advance(lane)
        while any(event is not None for event in reached):
            if all(isinstance(event, int) for event in reached):
                assert len(set(reached)) == 1, "divergent CTA barriers"
                state["barriers"] += 1
                for lane in order:
                    reached[lane] = advance(lane)
                continue
            progressed = False
            for base in range(0, threads, 32):
                group = reached[base : base + 32]
                if not all(
                    isinstance(event, tuple) and event[0] == "shuffle"
                    for event in group
                ):
                    continue
                assert len({(event[1], event[2]) for event in group}) == 1, (
                    "divergent warp shuffle"
                )
                values = [event[3] for event in group]
                returned = [
                    values[(lane ^ int(event[4])) if event[2] else int(event[4])]
                    for lane, event in enumerate(group)
                ]
                for lane in order:
                    if base <= lane < base + 32:
                        reached[lane] = advance(lane, returned[lane - base], send=True)
                progressed = True
            assert progressed, "divergent CTA barriers or incomplete warp shuffle"
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


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _fragment_global_fetch_add(
    x: torch.Tensor,
    counter: torch.Tensor,
    alias: torch.Tensor,
    tickets: torch.Tensor,
    reused: torch.Tensor,
    snapshots: torch.Tensor,
    masked: hl.constexpr,
):
    out = torch.empty((x.size(0), 17), dtype=torch.int32, device=x.device)
    for row in hl.grid(x.size(0)):
        local = hl.zeros([17], dtype=torch.int32)
        lane = hl.arange(x.size(1))
        for iteration in range(2):
            before = alias[row]
            indices = row + lane * 0
            if masked:
                indices = torch.where(lane + 1 < x.size(1), indices, counter.size(0))
            previous = hl.atomic_add(counter, [indices], torch.ones_like(indices))
            after = alias[row]
            tickets[row, iteration, :] = previous
            reused[row, iteration, :] = previous * 2 + 3
            snapshots[row, iteration, 0] = before
            snapshots[row, iteration, 1] = after
            hl.atomic_add(local, [previous % 17], x[row, :])
        out[row, :] = local
    return out


def _fragment_fetch_add_codegen(args, enabled, threads=128):
    from test._cute_binding import _cpu_bind
    from test._cute_binding import _forbid_native_compile
    from test._cute_binding import _mock_cuda_unavailable
    from test.cute_population_contracts import _target

    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound = _cpu_bind(_fragment_global_fetch_add, args)
        config = bound.config_spec.default_config()
        config.config.update(
            cute_fragment_register_loads=enabled, cute_fragment_threads=threads
        )
        return bound.to_code(config)


class TestFragmentGlobalFetchAddCPU(unittest.TestCase):
    def test_once_per_element_reuse_alias_snapshots_and_tails(self):
        for width in (1, 17, 65, 129):
            for enabled in (False, True):
                for masked in (False, True):
                    with self.subTest(width=width, enabled=enabled, masked=masked):
                        x = torch.ones((2, width), dtype=torch.int32)
                        counter = torch.zeros(2, dtype=torch.int32)
                        alias = counter.view(2)
                        tickets = torch.full((2, 2, width), -99, dtype=torch.int32)
                        reused = torch.full_like(tickets, -99)
                        snapshots = torch.full((2, 2, 2), -99, dtype=torch.int32)
                        args = (x, counter, alias, tickets, reused, snapshots, masked)
                        source = _fragment_fetch_add_codegen(args, enabled)
                        events = []
                        out, barriers = _simulate_register_load_program(
                            source,
                            x,
                            128,
                            host_tensors={
                                "counter": counter,
                                "alias": alias,
                                "tickets": tickets,
                                "reused": reused,
                                "snapshots": snapshots,
                            },
                            lane_order=list(reversed(range(128))),
                            atomic_events=events,
                        )
                        active = width - int(masked)
                        self.assertGreater(barriers, 0)
                        self.assertTrue(
                            torch.equal(counter, torch.full_like(counter, 2 * active))
                        )
                        global_events = [event for event in events if event[0] == "gpu"]
                        self.assertEqual(len(global_events), 2 * 2 * active)
                        for iteration in range(2):
                            expected = torch.arange(
                                iteration * active, (iteration + 1) * active
                            )
                            for row in range(2):
                                torch.testing.assert_close(
                                    tickets[row, iteration, :active].sort().values,
                                    expected.to(torch.int32),
                                    rtol=0,
                                    atol=0,
                                )
                            self.assertTrue(
                                torch.all(
                                    snapshots[:, iteration, 0] == iteration * active
                                )
                            )
                            self.assertTrue(
                                torch.all(
                                    snapshots[:, iteration, 1]
                                    == (iteration + 1) * active
                                )
                            )
                        if masked:
                            self.assertTrue(torch.all(tickets[:, :, -1] == 0))
                        torch.testing.assert_close(
                            reused, tickets * 2 + 3, rtol=0, atol=0
                        )
                        expected_local = torch.zeros(2, 17, dtype=torch.int32)
                        expected_local.scatter_add_(
                            1,
                            (torch.arange(2 * active) % 17).expand(2, -1),
                            torch.ones((2, 2 * active), dtype=torch.int32),
                        )
                        if masked:
                            expected_local[:, 0] += 2
                        torch.testing.assert_close(
                            out, expected_local.float(), rtol=0, atol=0
                        )
                        torch.testing.assert_close(
                            x, torch.ones_like(x), rtol=0, atol=0
                        )

    def test_generated_missing_or_delayed_shared_initialization_rejects(self):
        import ast

        x = torch.ones((2, 17), dtype=torch.int32)
        counter = torch.zeros(2, dtype=torch.int32)
        tickets = torch.empty((2, 2, 17), dtype=torch.int32)
        reused = torch.empty_like(tickets)
        snapshots = torch.empty((2, 2, 2), dtype=torch.int32)
        source = _fragment_fetch_add_codegen(
            (x, counter, counter.view(2), tickets, reused, snapshots, False), False
        )
        for delayed in (False, True):
            for floating in (False, True):
                with self.subTest(delayed=delayed, floating=floating):
                    tree = ast.parse(source)
                    fn = next(
                        node
                        for node in tree.body
                        if isinstance(node, ast.FunctionDef)
                        and node.name.startswith("_helion_")
                    )
                    first = next(node for node in fn.body if isinstance(node, ast.For))
                    self.assertIn("fragment_buffer[", ast.unparse(first))
                    initialization = ast.parse(ast.unparse(first)).body[0]
                    first.body = [ast.Pass()]
                    if delayed:
                        output_index = next(
                            index
                            for index, node in enumerate(fn.body)
                            if isinstance(node, ast.For)
                            and "out.iterator" in ast.unparse(node)
                        )
                        fn.body[output_index:output_index] = [
                            initialization,
                            ast.parse("cute.arch.sync_threads()").body[0],
                        ]
                    if floating:
                        allocation = next(
                            node
                            for node in fn.body
                            if isinstance(node, ast.Assign)
                            and ast.unparse(node.targets[0]) == "fragment_buffer"
                        )
                        self.assertEqual(
                            ast.unparse(allocation.value.args[0]), "cutlass.Int32"
                        )
                        allocation.value.args[0] = ast.parse(
                            "cutlass.Float32", mode="eval"
                        ).body
                    mutated = ast.unparse(ast.fix_missing_locations(tree))
                    counter.zero_()
                    events = []
                    with self.assertRaisesRegex(
                        AssertionError, "shared pointer read before initialization"
                    ):
                        _simulate_register_load_program(
                            mutated,
                            x,
                            128,
                            host_tensors={
                                "counter": counter,
                                "alias": counter.view(2),
                                "tickets": tickets,
                                "reused": reused,
                                "snapshots": snapshots,
                            },
                            atomic_events=events,
                        )
                    self.assertEqual(sum(event[0] == "cta" for event in events), 0)

    def test_returned_float_atomic_is_not_silently_admitted(self):
        x = torch.ones((2, 17), dtype=torch.int32)
        counter = torch.zeros(2)
        args = (
            x,
            counter,
            counter.view(2),
            torch.empty(2, 2, 17, dtype=torch.int32),
            torch.empty(2, 2, 17, dtype=torch.int32),
            torch.empty(2, 2, 2, dtype=torch.int32),
            False,
        )
        with self.assertRaisesRegex(helion.exc.InvalidConfig, "global int32 target"):
            _fragment_fetch_add_codegen(args, False)


@onlyBackends("cute")
class TestFragmentGlobalFetchAddNative(TestCase):
    def test_fetch_add_reuse_masks_aliases_and_integer_payload(self):
        base = 1 << 24
        for width in (17, 129):
            for enabled in (False, True):
                for masked in (False, True):
                    with self.subTest(width=width, enabled=enabled, masked=masked):
                        x = torch.ones((2, width), dtype=torch.int32, device=DEVICE)
                        counter = torch.full(
                            (2,), base, dtype=torch.int32, device=DEVICE
                        )
                        tickets = torch.empty(
                            (2, 2, width), dtype=torch.int32, device=DEVICE
                        )
                        reused = torch.empty_like(tickets)
                        snapshots = torch.empty(
                            (2, 2, 2), dtype=torch.int32, device=DEVICE
                        )
                        _, actual = code_and_output(
                            _fragment_global_fetch_add,
                            (
                                x,
                                counter,
                                counter.view(2),
                                tickets,
                                reused,
                                snapshots,
                                masked,
                            ),
                            cute_fragment_register_loads=enabled,
                            cute_fragment_threads=128,
                        )
                        active = width - int(masked)
                        torch.testing.assert_close(
                            counter,
                            torch.full_like(counter, base + 2 * active),
                            rtol=0,
                            atol=0,
                        )
                        for iteration in range(2):
                            expected = torch.arange(
                                base + iteration * active,
                                base + (iteration + 1) * active,
                                dtype=torch.int32,
                                device=DEVICE,
                            )
                            for row in range(2):
                                torch.testing.assert_close(
                                    tickets[row, iteration, :active].sort().values,
                                    expected,
                                    rtol=0,
                                    atol=0,
                                )
                            self.assertTrue(
                                torch.all(
                                    snapshots[:, iteration, 0]
                                    == base + iteration * active
                                )
                            )
                            self.assertTrue(
                                torch.all(
                                    snapshots[:, iteration, 1]
                                    == base + (iteration + 1) * active
                                )
                            )
                        if masked:
                            self.assertTrue(torch.all(tickets[:, :, -1] == 0))
                        torch.testing.assert_close(
                            reused, tickets * 2 + 3, rtol=0, atol=0
                        )
                        expected_local = (
                            torch.bincount(
                                torch.arange(base, base + 2 * active, device=DEVICE)
                                % 17,
                                minlength=17,
                            )
                            .int()
                            .expand(2, -1)
                            .clone()
                        )
                        if masked:
                            expected_local[:, 0] += 2
                        torch.testing.assert_close(
                            actual, expected_local, rtol=0, atol=0
                        )
                        torch.testing.assert_close(
                            x, torch.ones_like(x), rtol=0, atol=0
                        )


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _local_histogram_consumers(x, bins: hl.constexpr, mode: hl.constexpr):
    out = torch.empty((x.size(0), bins), dtype=x.dtype, device=x.device)
    for row in hl.grid(x.size(0)):
        histogram = hl.full([bins], 2, dtype=x.dtype)
        indices = hl.arange(x.size(1)) % bins
        for _iteration in range(2):
            hl.atomic_add(histogram, [indices], x[row, :])
        if mode == "scan":
            prefix = hl.cumsum(histogram * 2 + 1, dim=0)
            out[row, :] = histogram + prefix
        elif mode == "reduce":
            out[row, :] = histogram + torch.sum(histogram)
        elif mode == "extrema":
            out[row, :] = (
                histogram + 2 * torch.amax(-histogram) + 3 * torch.amin(histogram)
            )
        elif mode == "prod":
            out[row, :] = histogram + torch.prod(histogram * 0 + 2)
        else:
            out[row, :] = histogram * 2 + 1
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _local_histogram_consumer_negative(x, mode: hl.constexpr):
    out = torch.empty((x.size(0), 17), dtype=x.dtype, device=x.device)
    for row in hl.grid(x.size(0)):
        histogram = hl.zeros([17], dtype=x.dtype)
        indices = hl.arange(x.size(1)) % 17
        if mode == "early_scan":
            value = hl.cumsum(histogram, dim=0)
            hl.atomic_add(histogram, [indices], x[row, :])
        elif mode == "later_update":
            hl.atomic_add(histogram, [indices], x[row, :])
            value = histogram + 1
            hl.atomic_add(histogram, [indices], x[row, :])
        elif mode == "loop_read":
            value = hl.zeros([17], dtype=x.dtype)
            for _iteration in range(2):
                hl.atomic_add(histogram, [indices], x[row, :])
                value = hl.cumsum(histogram, dim=0)
        elif mode == "loop_allocation":
            value = hl.zeros([17], dtype=x.dtype)
            for _iteration in range(2):
                temporary = hl.zeros([17], dtype=x.dtype)
                hl.atomic_add(temporary, [indices], x[row, :])
                value = temporary + 1
        elif mode == "conditional":
            if x[row, 0] > 0:
                hl.atomic_add(histogram, [indices], x[row, :])
            value = histogram + 1
        elif mode == "alias":
            alias = histogram[:, None]
            hl.atomic_add(alias, [indices, 0], x[row, :])
            value = histogram + 1
        else:
            value = histogram + 1
            hl.atomic_add(histogram, [hl.arange(17)], value)
        out[row, :] = value
    return out


def _local_histogram_consumer_codegen(kernel, args, scan="serial", threads=128):
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
        bound = _cpu_bind(kernel, args)
        config = bound.config_spec.default_config()
        config.config.update(cute_fragment_scan=scan, cute_fragment_threads=threads)
        return bound.to_code(config)


def _local_histogram_consumer_reference(x, bins, mode):
    histogram = torch.full((x.size(0), bins), 2, dtype=x.dtype, device=x.device)
    indices = (torch.arange(x.size(1), device=x.device) % bins).expand_as(x)
    for _ in range(2):
        histogram.scatter_add_(1, indices, x)
    if mode == "scan":
        return (histogram + torch.cumsum(histogram * 2 + 1, dim=1)).to(x.dtype)
    if mode == "reduce":
        return (histogram + torch.sum(histogram, dim=1, keepdim=True)).to(x.dtype)
    if mode == "extrema":
        return (
            histogram
            + 2 * torch.amax(-histogram, dim=1, keepdim=True)
            + 3 * torch.amin(histogram, dim=1, keepdim=True)
        ).to(x.dtype)
    if mode == "prod":
        return (histogram + torch.prod(histogram * 0 + 2, dim=1, keepdim=True)).to(
            x.dtype
        )
    return histogram * 2 + 1


class TestLocalHistogramConsumersCPU(unittest.TestCase):
    def test_local_histogram_non_neutral_padding_for_each_reduction(self):
        for dtype in (torch.int32, torch.float32):
            x = torch.ones((1, 65), dtype=dtype)
            for mode in ("reduce", "extrema", "prod"):
                with self.subTest(dtype=dtype, mode=mode):
                    code = _local_histogram_consumer_codegen(
                        _local_histogram_consumers, (x, 17, mode)
                    )
                    actual, calls = _simulate_local_atomic_program(code, x, (1, 17))
                    torch.testing.assert_close(
                        actual,
                        _local_histogram_consumer_reference(x, 17, mode),
                        rtol=0,
                        atol=0,
                    )
                    self.assertEqual(len(calls), 130)

    def test_local_histogram_completed_readers_generated_values(self):
        for dtype in (torch.int32, torch.float32):
            for width, bins in ((1, 1), (65, 17), (257, 256)):
                x = (torch.arange(width).reshape(1, width) % 7).to(dtype)
                for mode in ("scan", "reduce", "pointwise"):
                    with self.subTest(dtype=dtype, width=width, bins=bins, mode=mode):
                        code = _local_histogram_consumer_codegen(
                            _local_histogram_consumers, (x, bins, mode)
                        )
                        actual, calls = _simulate_local_atomic_program(
                            code, x, (1, bins)
                        )
                        torch.testing.assert_close(
                            actual,
                            _local_histogram_consumer_reference(x, bins, mode),
                            rtol=0,
                            atol=0,
                        )
                        self.assertEqual(len(calls), 2 * width)

    def test_local_histogram_consumers_uniform_barriers_and_shared_lifetime(self):
        for width, bins, threads in ((65, 17, 32), (257, 256, 128), (129, 257, 512)):
            x = (torch.arange(width).reshape(1, width) % 7).float()
            expected = _local_histogram_consumer_reference(x, bins, "scan")
            for scan in ("serial", "cooperative"):
                code = _local_histogram_consumer_codegen(
                    _local_histogram_consumers, (x, bins, "scan"), scan, threads
                )
                for order in (list(range(threads)), list(reversed(range(threads)))):
                    with self.subTest(width=width, scan=scan, reverse=order[0] != 0):
                        output = torch.full_like(expected, -999)
                        _simulate_register_load_program(
                            code,
                            x,
                            threads,
                            host_tensors={"out": output},
                            lane_order=order,
                        )
                        torch.testing.assert_close(output, expected, rtol=0, atol=0)

    def test_local_histogram_consumer_missing_update_barrier_is_detected(self):
        import ast

        x = (torch.arange(65).reshape(1, 65) % 7).float()
        code = _local_histogram_consumer_codegen(
            _local_histogram_consumers, (x, 17, "scan"), threads=32
        )
        tree = ast.parse(code)
        removed = 0
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.For, ast.If)):
                continue
            for index in range(len(node.body) - 1, 0, -1):
                previous, current = node.body[index - 1 : index + 1]
                if (
                    isinstance(previous, ast.For)
                    and "cute.arch.atomic_add" in ast.unparse(previous)
                    and ast.unparse(current) == "cute.arch.sync_threads()"
                ):
                    node.body.pop(index)
                    removed += 1
        self.assertEqual(removed, 1)
        output = torch.full((1, 17), -999.0)
        _simulate_register_load_program(
            ast.unparse(tree), x, 32, host_tensors={"out": output}
        )
        self.assertFalse(
            torch.equal(output, _local_histogram_consumer_reference(x, 17, "scan"))
        )

    def test_local_histogram_readers_reject_mutating_or_nonuniform_lifetime(self):
        from helion import exc

        for mode in (
            "early_scan",
            "later_update",
            "loop_read",
            "loop_allocation",
            "conditional",
            "alias",
            "self_derived_update",
        ):
            with self.subTest(mode=mode), self.assertRaises(exc.InvalidConfig):
                _local_histogram_consumer_codegen(
                    _local_histogram_consumer_negative, (torch.ones((1, 65)), mode)
                )

    def test_local_histogram_read_proof_rejects_mutable_operator_schema(self):
        from unittest.mock import patch

        from helion import exc
        from helion._compiler.cute.local_atomic import prove_local_atomics

        inspected = []

        def reject_mutable_consumer(graphs):
            node = next(
                node
                for info in graphs
                for node in info.graph.nodes
                if node.target is torch.ops.aten.mul.Tensor
            )
            original = node.target
            try:
                node.target = torch.ops.aten.mul_.Tensor
                inspected.append(node.target)
                return prove_local_atomics(graphs)
            finally:
                node.target = original

        with (
            patch(
                "helion._compiler.cute.computed_fragment.prove_local_atomics",
                side_effect=reject_mutable_consumer,
            ),
            self.assertRaisesRegex(exc.InvalidConfig, "read-only root consumers"),
        ):
            _local_histogram_consumer_codegen(
                _local_histogram_consumers,
                (torch.ones((1, 65)), 17, "pointwise"),
            )
        self.assertEqual(inspected, [torch.ops.aten.mul_.Tensor])


@onlyBackends("cute")
class TestLocalHistogramConsumersNative(TestCase):
    def test_local_histogram_completed_readers(self):
        for dtype in (torch.int32, torch.float32):
            for width, bins, threads in ((65, 17, 32), (257, 256, 128)):
                x = (torch.arange(width, device=DEVICE).reshape(1, width) % 7).to(dtype)
                before = x.clone()
                for scan in ("serial", "cooperative"):
                    with self.subTest(dtype=dtype, width=width, scan=scan):
                        _, actual = code_and_output(
                            _local_histogram_consumers,
                            (x, bins, "scan"),
                            cute_fragment_scan=scan,
                            cute_fragment_threads=threads,
                        )
                        torch.testing.assert_close(
                            actual,
                            _local_histogram_consumer_reference(x, bins, "scan"),
                            rtol=0,
                            atol=0,
                        )
                        torch.testing.assert_close(x, before, rtol=0, atol=0)
            x = torch.ones((1, 65), device=DEVICE, dtype=dtype)
            before = x.clone()
            for mode in ("reduce", "extrema", "prod"):
                _, actual = code_and_output(_local_histogram_consumers, (x, 17, mode))
                torch.testing.assert_close(
                    actual,
                    _local_histogram_consumer_reference(x, 17, mode),
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(x, before, rtol=0, atol=0)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _fragment_warp_reservations(
    x, counter, alias, tickets, reused, snapshots, masked: hl.constexpr
):
    out = torch.empty((x.size(0), 17), dtype=torch.int32, device=x.device)
    groups = x.size(1) // 32
    for row in hl.grid(x.size(0)):
        local = hl.zeros([17], dtype=torch.int32)
        for iteration in range(2):
            counts = (x[row, :].reshape(groups, 32) + iteration).sum(
                -1, dtype=torch.int32
            )
            before = alias[row]
            group = hl.arange(groups)
            indices = row + group * 0
            if masked:
                indices = torch.where(group + 1 < groups, indices, -counter.size(0) - 1)
            previous = hl.atomic_add(counter, [indices], counts)
            after = alias[row]
            tickets[row, iteration, :] = previous
            reused[row, iteration, :] = previous * 2 + 3
            snapshots[row, iteration, 0] = before
            snapshots[row, iteration, 1] = after
            slots = (previous[:, None] + hl.arange(32)[None, :]).reshape(groups * 32)
            hl.atomic_add(local, [slots % 17], torch.ones_like(slots))
        out[row, :] = local
    return out


def _warp_reservation_args(groups, device="cpu"):
    x = (
        torch.arange(2 * groups * 32, device=device).reshape(2, groups * 32) % 3 + 1
    ).int()
    counter = torch.full((2,), (1 << 24) + 3, dtype=torch.int32, device=device)
    tickets = torch.full((2, 2, groups), -99, dtype=torch.int32, device=device)
    return (
        x,
        counter,
        counter.view(2),
        tickets,
        torch.full_like(tickets, -99),
        torch.full((2, 2, 2), -99, dtype=torch.int32, device=device),
    )


def _warp_reservation_codegen(args, enabled, threads, reduction="serial"):
    from test._cute_binding import _cpu_bind
    from test._cute_binding import _forbid_native_compile
    from test._cute_binding import _mock_cuda_unavailable
    from test.cute_population_contracts import _target

    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound = _cpu_bind(_fragment_warp_reservations, args)
        config = bound.config_spec.default_config()
        config.config.update(
            cute_fragment_threads=threads,
            cute_fragment_warp_results=enabled,
            cute_fragment_reduction=reduction,
        )
        return bound.to_code(config)


def _check_warp_reservations(args, actual, masked):
    x, counter, alias, tickets, reused, snapshots = args
    rows, width = x.shape
    groups = width // 32
    active = groups - int(masked)
    expected_local = torch.zeros((rows, 17), dtype=torch.int32, device=x.device)
    running = torch.full((rows,), (1 << 24) + 3, dtype=torch.int32, device=x.device)
    for iteration in range(2):
        counts = (x.reshape(rows, groups, 32) + iteration).sum(-1, dtype=torch.int32)
        torch.testing.assert_close(snapshots[:, iteration, 0], running, rtol=0, atol=0)
        for row in range(rows):
            order = tickets[row, iteration, :active].argsort()
            expected = (
                running[row]
                + counts[row, :active][order].cumsum(0)
                - counts[row, :active][order]
            )
            torch.testing.assert_close(
                tickets[row, iteration, :active][order].long(), expected, rtol=0, atol=0
            )
        running += counts[:, :active].sum(-1, dtype=torch.int32)
        torch.testing.assert_close(snapshots[:, iteration, 1], running, rtol=0, atol=0)
        slots = (
            tickets[:, iteration, :, None].long() + torch.arange(32, device=x.device)
        ).reshape(rows, width)
        expected_local.scatter_add_(
            1, slots % 17, torch.ones_like(slots, dtype=torch.int32)
        )
    if masked:
        assert torch.all(tickets[:, :, -1] == 0)
    torch.testing.assert_close(reused, tickets * 2 + 3, rtol=0, atol=0)
    torch.testing.assert_close(counter, running, rtol=0, atol=0)
    torch.testing.assert_close(alias, running, rtol=0, atol=0)
    torch.testing.assert_close(actual.to(torch.int32), expected_local, rtol=0, atol=0)


class TestFragmentWarpResultsCPU(unittest.TestCase):
    def test_generated_register_reservations_reuse_masks_and_alias_order(self):
        for groups, threads, reduction, masked in (
            (1, 32, "serial", False),
            (4, 128, "serial", True),
            (4, 128, "warp", False),
            (16, 512, "serial", True),
        ):
            for enabled in (False, True):
                with self.subTest(
                    groups=groups,
                    threads=threads,
                    reduction=reduction,
                    masked=masked,
                    enabled=enabled,
                ):
                    args = _warp_reservation_args(groups)
                    before = args[0].clone()
                    source = _warp_reservation_codegen(
                        (*args, masked), enabled, threads, reduction
                    )
                    self.assertEqual("fragment_warp_atomic" in source, enabled)
                    self.assertEqual("fragment_warp_reduction" in source, enabled)
                    events = []
                    workers = []
                    actual, barriers = _simulate_register_load_program(
                        source,
                        args[0],
                        threads,
                        host_tensors=dict(
                            zip(
                                ("counter", "alias", "tickets", "reused", "snapshots"),
                                args[1:],
                                strict=True,
                            )
                        ),
                        lane_order=list(reversed(range(threads))),
                        atomic_events=events,
                        atomic_workers=workers,
                    )
                    _check_warp_reservations(args, actual, masked)
                    torch.testing.assert_close(args[0], before, rtol=0, atol=0)
                    self.assertEqual(
                        sum(event[0] == "gpu" for event in events),
                        4 * (groups - int(masked)),
                    )
                    self.assertGreater(barriers, 0)
                    self.assertEqual(
                        sorted(worker for scope, worker in workers if scope == "gpu"),
                        sorted(
                            [
                                group * (32 if enabled else 1)
                                for group in range(groups - int(masked))
                            ]
                            * 4
                        ),
                    )

    def test_warp_ownership_broadcast_proof(self):
        from helion._compiler.cute.warp_results import broadcast_preserves_warp

        self.assertTrue(broadcast_preserves_warp((4, 1), (4, 32), 4))
        self.assertTrue(broadcast_preserves_warp((4,), (1, 4), 4))
        self.assertFalse(broadcast_preserves_warp((4,), (32, 4), 4))
        self.assertFalse(broadcast_preserves_warp((1, 4), (32, 4), 4))
        self.assertFalse(broadcast_preserves_warp((4, 1), (4, 64), 4))

    def test_generated_incomplete_warp_shuffle_is_rejected(self):
        import ast

        args = _warp_reservation_args(4)
        source = _warp_reservation_codegen((*args, False), True, 128)
        tree = ast.parse(source)
        branch = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.If)
            and ast.unparse(node.test) == "fragment_thread // 32 < 4"
        )
        branch.test = ast.parse(
            "fragment_thread // 32 < 4 and fragment_thread % 32 == 0", mode="eval"
        ).body
        with self.assertRaisesRegex(AssertionError, "incomplete warp shuffle"):
            _simulate_register_load_program(
                ast.unparse(tree),
                args[0],
                128,
                host_tensors=dict(
                    zip(
                        ("counter", "alias", "tickets", "reused", "snapshots"),
                        args[1:],
                        strict=True,
                    )
                ),
            )


@onlyBackends("cute")
class TestFragmentWarpResultsNative(TestCase):
    def test_warp_reservations_reuse_masks_and_alias_order(self):
        for groups, threads, masked in ((4, 128, True), (16, 512, False)):
            for enabled in (False, True):
                with self.subTest(
                    groups=groups, threads=threads, masked=masked, enabled=enabled
                ):
                    args = _warp_reservation_args(groups, DEVICE)
                    before = args[0].clone()
                    source, actual = code_and_output(
                        _fragment_warp_reservations,
                        (*args, masked),
                        cute_fragment_threads=threads,
                        cute_fragment_warp_results=enabled,
                    )
                    self.assertEqual("fragment_warp_atomic" in source, enabled)
                    _check_warp_reservations(args, actual, masked)
                    torch.testing.assert_close(args[0], before, rtol=0, atol=0)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _fragment_warp_result_decline(x, counter, mode: hl.constexpr):
    out = torch.empty((x.size(0), 17), dtype=torch.int32, device=x.device)
    for row in hl.grid(x.size(0)):
        local = hl.zeros([17], dtype=torch.int32)
        values = x[row, :].reshape(4, x.size(1) // 4)
        if mode == "first_axis":
            counts = values.sum(0, dtype=torch.int32)
        elif mode == "floating":
            counts = values.float().sum(-1).to(torch.int32)
        else:
            counts = values.sum(-1, dtype=torch.int32)
        if mode == "carried":
            previous = torch.zeros_like(counts)
            for _iteration in range(2):
                previous = hl.atomic_add(
                    counter, [row + hl.arange(counts.size(0)) * 0], counts
                )
        else:
            previous = hl.atomic_add(
                counter, [row + hl.arange(counts.size(0)) * 0], counts
            )
        if mode == "cross_warp":
            slots = previous[None, :] + hl.arange(32)[:, None]
        else:
            slots = previous[:, None] + hl.arange(32)[None, :]
        hl.atomic_add(
            local, [slots.reshape(-1) % 17], torch.ones_like(slots.reshape(-1))
        )
        out[row, :] = local
    return out


class TestFragmentWarpResultsConfigCPU(unittest.TestCase):
    def test_scope_layout_and_width_declines_preserve_fallback(self):
        from test._cute_binding import _cpu_bind
        from test._cute_binding import _forbid_native_compile
        from test._cute_binding import _mock_cuda_unavailable
        from test.cute_population_contracts import _target

        with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
            for mode, width in (
                ("first_axis", 128),
                ("floating", 128),
                ("cross_warp", 128),
                ("carried", 128),
                ("short", 64),
            ):
                with self.subTest(mode=mode):
                    args = (
                        torch.ones((2, width), dtype=torch.int32),
                        torch.zeros(2, dtype=torch.int32),
                        mode,
                    )
                    bound = _cpu_bind(_fragment_warp_result_decline, args)
                    self.assertFalse(
                        bound.config_spec.cute_fragment_warp_result_root_ids
                    )
                    config = bound.config_spec.default_config()
                    before = bound.to_code(config)
                    config.config["cute_fragment_warp_results"] = False
                    self.assertEqual(bound.to_code(config), before)
                    config.config["cute_fragment_warp_results"] = True
                    with self.assertRaisesRegex(helion.exc.InvalidConfig, "warp-owned"):
                        bound.to_code(config)

    def test_default_strict_thread_guard_and_population_prefix(self):
        from copy import deepcopy

        from test._cute_binding import _cpu_bind
        from test._cute_binding import _forbid_native_compile
        from test._cute_binding import _mock_cuda_unavailable
        from test.cute_population_contracts import _target
        from test.cute_population_contracts import checked_initial_population

        from helion.autotuner.pattern_search import InitialPopulationStrategy
        from helion.autotuner.pattern_search import PatternSearch

        args = (*_warp_reservation_args(4), False)
        kernel = helion.kernel(
            _fragment_warp_reservations.fn,
            backend="cute",
            static_shapes=True,
            autotune_effort="full",
        )
        with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
            bound = _cpu_bind(kernel, args)
            spec = bound.config_spec
            config = spec.default_config()
            self.assertNotIn("cute_fragment_warp_results", config)
            self.assertTrue(
                all(
                    "cute_fragment_warp_results" not in seed
                    for seed in spec.compiler_seed_configs
                )
            )
            original = bound.to_code(config)
            off = deepcopy(config)
            off.config["cute_fragment_warp_results"] = False
            self.assertEqual(original, bound.to_code(off))
            bad = deepcopy(config)
            bad.config.update(cute_fragment_warp_results=True, cute_fragment_threads=32)
            with self.assertRaisesRegex(helion.exc.InvalidConfig, "warp-owned"):
                bound.to_code(bad)
            groups = spec.compiler_coverage_groups
            group = next(
                group for group in groups if group.key == "cute_fragment_warp_results"
            )
            self.assertEqual(group.legacy, False)
            self.assertEqual(
                [(d.key, d.value) for d in group.dependencies],
                [("cute_fragment_threads", 512)],
            )
            with bound.env:
                search = PatternSearch(
                    bound,
                    args,
                    initial_population=4,
                    initial_population_strategy=InitialPopulationStrategy.FROM_RANDOM,
                )
                population = checked_initial_population(search)
                self.assertGreaterEqual(len(population), 4)
                outcome = next(
                    o
                    for o in search.compiler_coverage_outcomes
                    if o.mechanism == group.mechanism
                )
                self.assertIn(outcome.outcome, ("added", "already_present"))
                self.assertTrue(outcome.effective["cute_fragment_warp_results"])


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _fragment_warp_reservation_bonus(
    x, counter, alias, tickets, reused, snapshots, bonus, masked: hl.constexpr
):
    out = torch.empty((x.size(0), 17), dtype=torch.int32, device=x.device)
    groups = x.size(1) // 32
    for row in hl.grid(x.size(0)):
        local = hl.zeros([17], dtype=torch.int32)
        for iteration in range(2):
            counts = (x[row, :].reshape(groups, 32) + iteration).sum(
                -1, dtype=torch.int32
            )
            before = alias[row]
            group = hl.arange(groups)
            indices = row + group * 0
            if masked:
                indices = torch.where(group + 1 < groups, indices, -counter.size(0) - 1)
            previous = hl.atomic_add(counter, [indices], counts)
            after = alias[row]
            tickets[row, iteration, :] = previous
            reused[row, iteration, :] = previous * 2 + 3 + bonus[row, iteration, :]
            snapshots[row, iteration, 0] = before
            snapshots[row, iteration, 1] = after
            slots = (previous[:, None] + hl.arange(32)[None, :]).reshape(groups * 32)
            hl.atomic_add(local, [slots % 17], torch.ones_like(slots))
        out[row, :] = local
    return out


class TestFragmentWarpResultStorageConflictCPU(unittest.TestCase):
    def test_independent_operand_four_storage_combinations(self):
        from copy import deepcopy

        from test._cute_binding import _cpu_bind
        from test._cute_binding import _forbid_native_compile
        from test._cute_binding import _mock_cuda_unavailable
        from test.cute_population_contracts import _target
        from test.cute_population_contracts import checked_initial_population

        from helion.autotuner.pattern_search import InitialPopulationStrategy
        from helion.autotuner.pattern_search import PatternSearch

        with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
            for registers, warp in (
                (False, False),
                (True, False),
                (False, True),
                (True, True),
            ):
                with self.subTest(registers=registers, warp=warp):
                    args = _warp_reservation_args(4)
                    original_x = args[0].clone()
                    bonus = torch.arange(16, dtype=torch.int32).reshape(2, 2, 4) + 10
                    original_bonus = bonus.clone()
                    full_args = (*args, bonus, False)
                    bound = _cpu_bind(_fragment_warp_reservation_bonus, full_args)
                    config = bound.config_spec.default_config()
                    config.config.update(
                        cute_fragment_threads=128,
                        cute_fragment_warp_results=warp,
                        cute_fragment_register_loads=registers,
                    )
                    if registers and warp:
                        saved = deepcopy(config.config)
                        for repair in (False, True):
                            with (
                                bound.env,
                                self.assertRaisesRegex(
                                    helion.exc.InvalidConfig,
                                    "incompatible physical ownership",
                                ),
                            ):
                                bound.config_spec.normalize(config, _fix_invalid=repair)
                            self.assertEqual(config.config, saved)
                        with self.assertRaisesRegex(
                            helion.exc.InvalidConfig, "incompatible physical ownership"
                        ):
                            bound.to_code(config)
                        with bound.env:
                            generation = bound.config_spec.create_config_generation()
                            with self.assertRaisesRegex(
                                helion.exc.InvalidConfig,
                                "incompatible physical ownership",
                            ):
                                generation.strict_config_pair(config)
                            with self.assertRaisesRegex(
                                helion.exc.InvalidConfig,
                                "incompatible physical ownership",
                            ):
                                generation.canonicalize_flat(generation.flatten(config))
                        continue
                    source = bound.to_code(config)
                    hosts = dict(
                        zip(
                            ("counter", "alias", "tickets", "reused", "snapshots"),
                            args[1:],
                            strict=True,
                        )
                    )
                    hosts["bonus"] = bonus
                    events = []
                    actual, _ = _simulate_register_load_program(
                        source, args[0], 128, host_tensors=hosts, atomic_events=events
                    )
                    torch.testing.assert_close(
                        args[4], args[3] * 2 + 3 + bonus, rtol=0, atol=0
                    )
                    _check_warp_reservations(
                        (*args[:4], args[4] - bonus, args[5]), actual, False
                    )
                    torch.testing.assert_close(args[0], original_x, rtol=0, atol=0)
                    torch.testing.assert_close(bonus, original_bonus, rtol=0, atol=0)
                    self.assertEqual(sum(e[0] == "gpu" for e in events), 16)
                    self.assertEqual("fragment_register_load" in source, registers)
                    self.assertEqual("fragment_warp_atomic" in source, warp)
            kernel = helion.kernel(
                _fragment_warp_reservation_bonus.fn,
                backend="cute",
                static_shapes=True,
                autotune_effort="full",
            )
            bound = _cpu_bind(kernel, full_args)
            with bound.env:
                search = PatternSearch(
                    bound,
                    full_args,
                    initial_population=4,
                    initial_population_strategy=InitialPopulationStrategy.FROM_RANDOM,
                )
                self.assertTrue(search.config_gen.compiler_coverage_enabled)
                rows = checked_initial_population(search)
                configs = [search.config_gen.unflatten(row) for row in rows]
                self.assertTrue(
                    all(
                        not (
                            cfg.get("cute_fragment_register_loads", False)
                            and cfg.get("cute_fragment_warp_results", False)
                        )
                        for cfg in configs
                    )
                )
                for key in (
                    "cute_fragment_register_loads",
                    "cute_fragment_warp_results",
                ):
                    group = next(
                        g
                        for g in bound.config_spec.compiler_coverage_groups
                        if g.key == key
                    )
                    for witness in group.witnesses:
                        requested = witness.carrier
                        requested.config[key] = witness.value
                        _, effective = search.config_gen.strict_config_pair(requested)
                        self.assertFalse(
                            effective.get("cute_fragment_register_loads", False)
                            and effective.get("cute_fragment_warp_results", False)
                        )


class TestFragmentPredicatePreprocessingCPU(unittest.TestCase):
    def test_boolean_prefixes_preserve_short_circuit_order_and_refresh(self):
        import ast
        import itertools
        from types import SimpleNamespace

        from helion._compiler.cute.computed_fragment import FragmentCompiler

        for operation in ("and", "or"):
            for stop in (0, 3, 16, None):
                with self.subTest(operation=operation, stop=stop):
                    compiler = object.__new__(FragmentCompiler)
                    names = itertools.count()
                    compiler.df = SimpleNamespace(
                        new_var=lambda prefix, names=names: f"{prefix}_{next(names)}"
                    )
                    statements = []
                    compiler.emit = statements.append
                    predicates = [f"check({index})" for index in range(17)]
                    # Nested equal operators must not retain the SDK's
                    # exponential repeated-left-operand expansion either.
                    expression = predicates[-1]
                    for item in reversed(predicates[:-1]):
                        expression = f"{item} {operation} ({expression})"
                    lowered = compiler.predicate([expression])
                    events = []

                    state = {"stop": stop}

                    def check(index, events=events, state=state, operation=operation):
                        events.append(index)
                        value = index == state["stop"]
                        return not value if operation == "and" else value

                    expected = eval(expression, {"check": check})
                    expected_events = events.copy()
                    events.clear()
                    namespace = {"check": check}
                    exec("\n".join(statements), namespace)
                    self.assertEqual(namespace[lowered], expected)
                    self.assertEqual(events, expected_events)
                    self.assertTrue(
                        all(
                            len(node.values) <= 2
                            for node in ast.walk(ast.parse("\n".join(statements)))
                            if isinstance(node, ast.BoolOp)
                        )
                    )
                    # A second execution sees new mutable predicate data.
                    state["stop"] = 1
                    events.clear()
                    exec("\n".join(statements), namespace)
                    self.assertEqual(events, [0, 1])
        compiler = object.__new__(FragmentCompiler)
        compiler.emit = lambda source: self.fail("short mask must remain unchanged")
        self.assertEqual(compiler.predicate(["a", "b", "c"]), "a and b and c")

    def test_actual_sdk_preprocesses_reservation_masks(self):
        import importlib.util
        from pathlib import Path
        import subprocess
        import sys
        import tempfile

        if importlib.util.find_spec("cutlass") is None:
            self.skipTest("requires the CuTe DSL AST preprocessor")
        program = r"""
import ast, importlib.util, inspect, sys
from pathlib import Path
sys.path.insert(0, sys.argv[2])
from cutlass.base_dsl.ast_preprocessor import DSLPreprocessor
path = Path(sys.argv[1])
spec = importlib.util.spec_from_file_location("predicate_sdk_case", path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
name = next(node.name for node in ast.parse(path.read_text()).body
            if isinstance(node, ast.FunctionDef) and node.name.startswith("_helion_"))
raw = inspect.unwrap(getattr(module, name))
with DSLPreprocessor(["cutlass"]).get_session() as session:
    tree = session.transform(raw, dict(raw.__globals__))
    compile(tree, str(path), "exec", dont_inherit=True)
assert sum(1 for node in ast.walk(tree)) < 100000
print("SDK_PREPROCESS_PASS")
"""
        for enabled, groups, threads, masked in (
            (False, 4, 128, False),
            (True, 4, 128, False),
            (True, 16, 512, True),
        ):
            with self.subTest(enabled=enabled, groups=groups, masked=masked):
                args = (*_warp_reservation_args(groups), masked)
                source = _warp_reservation_codegen(args, enabled, threads)
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "generated.py"
                    path.write_text(source)
                    result = subprocess.run(
                        [
                            sys.executable,
                            "-c",
                            program,
                            str(path),
                            str(Path(helion.__file__).resolve().parent.parent),
                        ],
                        check=True,
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                self.assertIn("SDK_PREPROCESS_PASS", result.stdout)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _local_atomic_disjoint_epochs(x, mode: hl.constexpr):
    out = torch.empty((x.size(0), 2, x.size(1)), dtype=x.dtype, device=x.device)
    for row in hl.grid(x.size(0)):
        first = hl.zeros([x.size(1)], dtype=x.dtype)
        second = hl.zeros([x.size(1)], dtype=x.dtype)
        for iteration in range(3):
            index = hl.arange(x.size(1))
            if mode == "shift":
                index = (index + iteration) % x.size(1)
            values = x[row, :] + iteration
            hl.atomic_add(first, [index], values)
            hl.atomic_add(second, [index], values * 2)
            if mode == "repeat":
                hl.atomic_add(first, [index], -values)
        out[row, 0, :] = first
        out[row, 1, :] = second
    return out


def _local_atomic_disjoint_reference(x, mode):
    out = torch.zeros((x.size(0), 2, x.size(1)), dtype=x.dtype, device=x.device)
    for iteration in range(3):
        index = torch.arange(x.size(1), device=x.device)
        if mode == "shift":
            index = (index + iteration) % x.size(1)
        for row in range(x.size(0)):
            out[row, 0].scatter_add_(0, index, x[row] + iteration)
            out[row, 1].scatter_add_(0, index, 2 * (x[row] + iteration))
            if mode == "repeat":
                out[row, 0].scatter_add_(0, index, -(x[row] + iteration))
    return out


class TestFragmentAtomicBarriersCPU(unittest.TestCase):
    def test_disjoint_targets_and_repeated_epochs(self):
        from test._cute_binding import _cpu_bind
        from test._cute_binding import _forbid_native_compile
        from test._cute_binding import _mock_cuda_unavailable
        from test.cute_population_contracts import _target

        for width in (17, 65):
            for mode in ("disjoint", "shift", "repeat"):
                with self.subTest(width=width, mode=mode):
                    x = (torch.arange(2 * width).reshape(2, width) % 7).float() * 0.25
                    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
                        bound = _cpu_bind(_local_atomic_disjoint_epochs, (x, mode))
                        code = bound.to_code(bound.config_spec.default_config())
                    expected = _local_atomic_disjoint_reference(x, mode)
                    for order in (list(range(128)), list(reversed(range(128)))):
                        actual = torch.full_like(expected, float("nan"))
                        _unused, barriers = _simulate_register_load_program(
                            code, x, 128, host_tensors={"out": actual}, lane_order=order
                        )
                        self.assertGreater(barriers, 0)
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_mutable_reads_and_conditional_updates_still_reject(self):
        self.assertIn(
            "atomic_add",
            _atomic_logical_axes_codegen(
                _local_atomic_histogram, (torch.ones(2, 17), 17, 3, "ordinary"), []
            ),
        )
        for mode in ("early_read", "self_value", "divergent", "result", "release"):
            with self.subTest(mode=mode), self.assertRaises(helion.exc.InvalidConfig):
                _atomic_logical_axes_codegen(
                    _local_atomic_histogram,
                    (torch.ones(2, 17), 17, 3, mode),
                    [],
                )


@onlyBackends("cute")
class TestFragmentAtomicBarriersNative(TestCase):
    def test_disjoint_targets_and_repeated_epochs(self):
        for width in (17, 65):
            for mode in ("disjoint", "shift", "repeat"):
                with self.subTest(width=width, mode=mode):
                    x = (
                        torch.arange(2 * width, device=DEVICE).reshape(2, width) % 7
                    ).float() * 0.25
                    _source, actual = code_and_output(
                        _local_atomic_disjoint_epochs, (x, mode)
                    )
                    torch.testing.assert_close(
                        actual,
                        _local_atomic_disjoint_reference(x, mode),
                        rtol=0,
                        atol=0,
                    )


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _local_atomic_disjoint_consumers(x, bins: hl.constexpr, mode: hl.constexpr):
    out = torch.empty((x.size(0), bins), dtype=x.dtype, device=x.device)
    for row in hl.grid(x.size(0)):
        first = hl.full([bins], 2, dtype=x.dtype)
        second = hl.full([bins], 3, dtype=x.dtype)
        index = hl.arange(x.size(1)) % bins
        values = x[row, :]
        doubled = values * 2
        hl.atomic_add(first, [index], values)
        hl.atomic_add(second, [index], doubled)
        derived = first * 2 + second + 1
        if mode == "scan":
            out[row, :] = hl.cumsum(derived, dim=0)
        elif mode == "reduce":
            out[row, :] = derived + torch.sum(derived)
        else:
            out[row, :] = derived
    return out


def _local_atomic_disjoint_consumer_reference(x, bins, mode):
    first = torch.full((x.size(0), bins), 2, dtype=x.dtype, device=x.device)
    second = torch.full_like(first, 3)
    indices = (torch.arange(x.size(1), device=x.device) % bins).expand_as(x)
    first.scatter_add_(1, indices, x)
    second.scatter_add_(1, indices, x * 2)
    derived = first * 2 + second + 1
    if mode == "scan":
        return torch.cumsum(derived, dim=1).to(x.dtype)
    if mode == "reduce":
        return (derived + derived.sum(dim=1, keepdim=True)).to(x.dtype)
    return derived


class TestFragmentAtomicConsumerBarriersCPU(unittest.TestCase):
    def test_final_derived_consumers_observe_both_pending_allocations(self):
        for width, bins, threads in ((65, 17, 32), (257, 65, 128), (129, 17, 512)):
            for dtype in (torch.int32, torch.float32):
                x = (torch.arange(width).reshape(1, width) % 7).to(dtype)
                for mode in ("pointwise", "reduce", "scan"):
                    with self.subTest(width=width, dtype=dtype, mode=mode):
                        code = _local_histogram_consumer_codegen(
                            _local_atomic_disjoint_consumers,
                            (x, bins, mode),
                            threads=threads,
                        )
                        expected = _local_atomic_disjoint_consumer_reference(
                            x, bins, mode
                        )
                        for order in (
                            list(range(threads)),
                            list(reversed(range(threads))),
                        ):
                            output = torch.full_like(expected, -999)
                            _simulate_register_load_program(
                                code,
                                x,
                                threads,
                                host_tensors={"out": output},
                                lane_order=order,
                            )
                            torch.testing.assert_close(output, expected, rtol=0, atol=0)

    def test_final_consumer_requires_pending_update_barrier(self):
        import ast

        x = (torch.arange(257).reshape(1, 257) % 7).float()
        code = _local_histogram_consumer_codegen(
            _local_atomic_disjoint_consumers, (x, 17, "pointwise"), threads=32
        )
        tree = ast.parse(code)
        removed = 0
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.For, ast.If)):
                continue
            for index in range(len(node.body) - 1, 0, -1):
                previous, current = node.body[index - 1 : index + 1]
                if (
                    isinstance(previous, ast.For)
                    and "cute.arch.atomic_add" in ast.unparse(previous)
                    and ast.unparse(current) == "cute.arch.sync_threads()"
                ):
                    node.body.pop(index)
                    removed += 1
        self.assertEqual(removed, 1)
        output = torch.full((1, 17), -999.0)
        _simulate_register_load_program(
            ast.unparse(tree), x, 32, host_tensors={"out": output}
        )
        self.assertFalse(
            torch.equal(
                output, _local_atomic_disjoint_consumer_reference(x, 17, "pointwise")
            )
        )

    def test_pending_read_tracks_nested_lazy_fragment_dependencies(self):
        import ast
        from unittest.mock import patch

        from helion._compiler.cute.computed_fragment import Fragment
        from helion._compiler.cute.computed_fragment import FragmentCompiler
        from helion.language import _tracing_ops

        original = FragmentCompiler.node
        wrapped = []

        def identity(value):
            return Fragment(
                value.shape,
                value.dtype,
                value.read,
                dependencies=(value,),
                logical_domain=value.logical_domain,
            )

        def with_lazy_operands(compiler, node, values):
            if node.target in (torch.ops.aten.mul.Tensor, _tracing_ops._mask_to):
                substitutions = {
                    key: identity(identity(value))
                    for key, value in values.items()
                    if key in node.all_input_nodes
                    and isinstance(value, Fragment)
                    and value.storage in compiler.pending_local_atomics
                }
                if substitutions:
                    wrapped.extend(substitutions)
                    values = {**values, **substitutions}
            return original(compiler, node, values)

        x = (torch.arange(257).reshape(1, 257) % 7).float()
        args = (x, 17, "scan")
        ordinary = _local_histogram_consumer_codegen(
            _local_atomic_disjoint_consumers, args, threads=32
        )
        with patch.object(FragmentCompiler, "node", with_lazy_operands):
            lazy = _local_histogram_consumer_codegen(
                _local_atomic_disjoint_consumers, args, threads=32
            )
        self.assertTrue(wrapped)
        # Identity recipes change only dependency representation, so all emitted
        # instructions, including the pre-read barrier, must remain identical.
        self.assertEqual(ast.dump(ast.parse(lazy)), ast.dump(ast.parse(ordinary)))
        output = torch.full((1, 17), -999.0)
        _simulate_register_load_program(lazy, x, 32, host_tensors={"out": output})
        torch.testing.assert_close(
            output,
            _local_atomic_disjoint_consumer_reference(x, 17, "scan"),
            rtol=0,
            atol=0,
        )


@onlyBackends("cute")
class TestFragmentAtomicConsumerBarriersNative(TestCase):
    def test_disjoint_updates_before_final_consumers(self):
        for dtype in (torch.int32, torch.float32):
            x = (torch.arange(257, device=DEVICE).reshape(1, 257) % 7).to(dtype)
            before = x.clone()
            for mode in ("pointwise", "reduce", "scan"):
                with self.subTest(dtype=dtype, mode=mode):
                    _code, output = code_and_output(
                        _local_atomic_disjoint_consumers,
                        (x, 17, mode),
                        cute_fragment_threads=32,
                    )
                    torch.testing.assert_close(
                        output,
                        _local_atomic_disjoint_consumer_reference(x, 17, mode),
                        rtol=0,
                        atol=0,
                    )
                    torch.testing.assert_close(x, before, rtol=0, atol=0)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _fragment_ordered_publication(
    x, counter, payload, observed, tickets, sem: hl.constexpr
):
    out = torch.empty((x.size(0), 17), dtype=torch.int32, device=x.device)
    for row in hl.grid(x.size(0)):
        local = hl.zeros([17], dtype=torch.int32)
        lane = hl.arange(x.size(1))
        hl.atomic_add(local, [lane % 17], x[row, :])
        payload[row, :] = x[row, :] + 1
        previous = hl.atomic_add(counter, [0], 1, sem=sem)
        tickets[row, :] = previous
        for producer in range(x.size(0)):
            # Only the last arrival reads other CTAs' published data. Earlier
            # arrivals do not issue racing loads; no global spin is needed.
            observed[row, producer, :] = hl.load(
                payload,
                [producer, lane],
                extra_mask=previous == x.size(0) - 1,
            )
        out[row, :] = local
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _fragment_ordered_vector(
    x, counter, tickets, sem: hl.constexpr, unused: hl.constexpr
):
    out = torch.empty((x.size(0), 17), dtype=torch.int32, device=x.device)
    for row in hl.grid(x.size(0)):
        local = hl.zeros([17], dtype=torch.int32)
        lane = hl.arange(x.size(1))
        previous = hl.atomic_add(counter, [row + lane * 0], 1, sem=sem)
        if not unused:
            tickets[row, :] = previous
        hl.atomic_add(local, [lane % 17], x[row, :])
        out[row, :] = local
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _fragment_ordered_local(x, sem: hl.constexpr):
    out = torch.empty((x.size(0), 17), dtype=torch.int32, device=x.device)
    for row in hl.grid(x.size(0)):
        local = hl.zeros([17], dtype=torch.int32)
        hl.atomic_add(local, [hl.arange(x.size(1)) % 17], x[row, :], sem=sem)
        out[row, :] = local
    return out


def _fragment_ordered_codegen(kernel, args, threads=128):
    from test._cute_binding import _cpu_bind
    from test._cute_binding import _forbid_native_compile
    from test._cute_binding import _mock_cuda_unavailable
    from test.cute_population_contracts import _target

    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound = _cpu_bind(kernel, args)
        config = bound.config_spec.default_config()
        config.config["cute_fragment_threads"] = threads
        return bound.to_code(config)


def _fragment_publication_args(rows, width, sem, device="cpu"):
    x = torch.arange(rows * width, dtype=torch.int32, device=device).reshape(
        rows, width
    )
    return (
        x,
        torch.zeros(1, dtype=torch.int32, device=device),
        torch.zeros_like(x),
        torch.full((rows, rows, width), -99, dtype=torch.int32, device=device),
        torch.full_like(x, -99),
        sem,
    )


def _check_fragment_publication(args):
    x, counter, payload, observed, tickets, _sem = args
    torch.testing.assert_close(
        counter, torch.full_like(counter, x.size(0)), rtol=0, atol=0
    )
    torch.testing.assert_close(payload, x + 1, rtol=0, atol=0)
    torch.testing.assert_close(
        tickets, tickets[:, :1].expand_as(tickets), rtol=0, atol=0
    )
    torch.testing.assert_close(
        tickets[:, 0].sort().values,
        torch.arange(x.size(0), dtype=torch.int32, device=x.device),
        rtol=0,
        atol=0,
    )
    winner = tickets[:, 0] == x.size(0) - 1
    torch.testing.assert_close(observed[winner][0], x + 1, rtol=0, atol=0)
    torch.testing.assert_close(
        observed[~winner], torch.zeros_like(observed[~winner]), rtol=0, atol=0
    )


class TestFragmentOrderedAtomicsCPU(unittest.TestCase):
    def test_scalar_publication_ownership_barriers_and_fresh_loads(self):
        for sem in ("acquire", "release", "acq_rel"):
            for threads, width in ((32, 17), (128, 129), (512, 65)):
                with self.subTest(sem=sem, threads=threads, width=width):
                    args = _fragment_publication_args(3, width, sem)
                    source = _fragment_ordered_codegen(
                        _fragment_ordered_publication, args, threads
                    )
                    events = []
                    workers = []
                    _simulate_register_load_program(
                        source,
                        args[0],
                        threads,
                        host_tensors=dict(
                            zip(
                                ("counter", "payload", "observed", "tickets"),
                                args[1:5],
                                strict=True,
                            )
                        ),
                        lane_order=list(reversed(range(threads))),
                        atomic_workers=workers,
                        memory_events=events,
                    )
                    _check_fragment_publication(args)
                    self.assertEqual(
                        [lane for scope, lane in workers if scope == "gpu"], [0, 0, 0]
                    )
                    self.assertNotIn("fragment_warp_atomic", source)
                    for row in range(3):
                        arrivals = [
                            e
                            for e in events
                            if e[0] == "atomic" and e[1] == row and e[-1] == "gpu"
                        ]
                        self.assertEqual(len(arrivals), 1)
                        arrival = arrivals[0]
                        self.assertEqual(arrival[4], sem)
                        fences = [
                            e
                            for e in events[: events.index(arrival)]
                            if e[0] == "fence" and e[1] == row
                        ]
                        if sem in ("release", "acq_rel"):
                            self.assertEqual(
                                sorted(e[2] for e in fences), list(range(threads))
                            )
                            self.assertTrue(all(e[3] < arrival[3] for e in fences))
                        else:
                            self.assertFalse(fences)

    def test_vector_results_and_unused_effects(self):
        for sem in ("acquire", "release", "acq_rel"):
            for unused in (False, True):
                with self.subTest(sem=sem, unused=unused):
                    x = torch.ones((2, 17), dtype=torch.int32)
                    counter = torch.zeros(2, dtype=torch.int32)
                    tickets = torch.full_like(x, -99)
                    source = _fragment_ordered_codegen(
                        _fragment_ordered_vector, (x, counter, tickets, sem, unused)
                    )
                    events = []
                    _simulate_register_load_program(
                        source,
                        x,
                        128,
                        host_tensors={"counter": counter, "tickets": tickets},
                        atomic_events=events,
                    )
                    self.assertEqual(sum(e[0] == "gpu" for e in events), 34)
                    torch.testing.assert_close(
                        counter, torch.full_like(counter, 17), rtol=0, atol=0
                    )
                    expected = (
                        torch.full_like(tickets, -99)
                        if unused
                        else torch.arange(17, dtype=torch.int32).expand(2, -1)
                    )
                    torch.testing.assert_close(
                        tickets.sort(-1).values, expected, rtol=0, atol=0
                    )

    def test_local_and_returned_float_scope_stays_rejected(self):
        x = torch.ones((2, 17), dtype=torch.int32)
        for sem in ("acquire", "release", "acq_rel"):
            with self.subTest(sem=sem):
                with self.assertRaisesRegex(helion.exc.InvalidConfig, "relaxed"):
                    _fragment_ordered_codegen(_fragment_ordered_local, (x, sem))
                with self.assertRaisesRegex(
                    helion.exc.InvalidConfig, "global int32 target"
                ):
                    _fragment_ordered_codegen(
                        _fragment_ordered_vector,
                        (x, torch.zeros(2), torch.zeros_like(x), sem, False),
                    )

    @skipUnlessCuteAvailable("requires CuTe DSL")
    def test_sdk_atomic_and_fence_ir_carries_requested_memory_order(self):
        import ast

        import cutlass
        from cutlass._mlir import ir
        from cutlass._mlir.dialects import func
        import cutlass.cute as cute

        for sem in ("relaxed", "acquire", "release", "acq_rel"):
            with self.subTest(sem=sem):
                source = _fragment_ordered_codegen(
                    _fragment_ordered_publication,
                    _fragment_publication_args(1, 17, sem),
                )
                calls = [
                    n
                    for n in ast.walk(ast.parse(source))
                    if isinstance(n, ast.Call)
                    and ast.unparse(n.func) == "cute.arch.atomic_add"
                ]
                global_call = next(
                    n
                    for n in calls
                    if any(
                        k.arg == "scope" and ast.literal_eval(k.value) == "gpu"
                        for k in n.keywords
                    )
                )
                keywords = {
                    k.arg: ast.literal_eval(k.value) for k in global_call.keywords
                }
                self.assertEqual(keywords, {"sem": sem, "scope": "gpu"})
                with ir.Context(), ir.Location.unknown():
                    module = ir.Module.create()
                    with ir.InsertionPoint(module.body):
                        fn = func.FuncOp(
                            "ordered_atomic",
                            (
                                [
                                    ir.Type.parse("!llvm.ptr<1>"),
                                    cutlass.Int32.mlir_type,
                                ],
                                [],
                            ),
                        )
                        block = fn.add_entry_block()
                        with ir.InsertionPoint(block):
                            if sem in ("release", "acq_rel"):
                                cute.arch.fence_acq_rel_gpu()
                                cute.arch.sync_threads()
                            cute.arch.atomic_add(
                                block.arguments[0],
                                cutlass.Int32(block.arguments[1]),
                                **keywords,
                            )
                            cute.arch.sync_threads()
                            func.ReturnOp([])
                    self.assertTrue(module.operation.verify())
                    text = str(module)
                    atomic = next(
                        op
                        for op in block.operations
                        if op.operation.name == "nvvm.atomicrmw"
                    )
                    self.assertEqual(str(atomic.memOrder), f"#nvvm.mem_order<{sem}>")
                    self.assertEqual(str(atomic.syncscope), "#nvvm.mem_scope<gpu>")
                    self.assertEqual(
                        "llvm.fence" in text, sem in ("release", "acq_rel")
                    )


@onlyBackends("cute")
class TestFragmentOrderedAtomicsNative(TestCase):
    def test_multi_cta_publication_and_single_arrival(self):
        # This is a pending native memory-order litmus, not a CPU-model claim.
        for rows, width, threads in ((1, 17, 32), (3, 129, 128), (17, 65, 512)):
            for repeat in range(3):
                with self.subTest(
                    rows=rows, width=width, threads=threads, repeat=repeat
                ):
                    args = _fragment_publication_args(rows, width, "acq_rel", DEVICE)
                    code_and_output(
                        _fragment_ordered_publication,
                        args,
                        cute_fragment_threads=threads,
                    )
                    _check_fragment_publication(args)

    def test_vector_semantics_returned_and_unused(self):
        for sem in ("acquire", "release", "acq_rel"):
            for unused in (False, True):
                with self.subTest(sem=sem, unused=unused):
                    x = torch.ones((2, 129), dtype=torch.int32, device=DEVICE)
                    counter = torch.zeros(2, dtype=torch.int32, device=DEVICE)
                    tickets = torch.full_like(x, -99)
                    code_and_output(
                        _fragment_ordered_vector, (x, counter, tickets, sem, unused)
                    )
                    torch.testing.assert_close(
                        counter, torch.full_like(counter, 129), rtol=0, atol=0
                    )
                    expected = (
                        torch.full_like(tickets, -99)
                        if unused
                        else torch.arange(129, dtype=torch.int32, device=DEVICE).expand(
                            2, -1
                        )
                    )
                    torch.testing.assert_close(
                        tickets.sort(-1).values, expected, rtol=0, atol=0
                    )


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _fragment_terminal_finalizer(
    x,
    partials,
    alias,
    counter,
    tickets,
    result,
    sem: hl.constexpr,
    mode: hl.constexpr,
    reset: hl.constexpr,
):
    for row in hl.grid(x.size(0)):
        local = hl.zeros([x.size(1)], dtype=torch.int32)
        index = hl.arange(x.size(1))
        hl.atomic_add(local, [index], x[row, :])
        hl.atomic_add(partials, [index], local)
        previous = hl.atomic_add(counter, [0], 1, sem=sem)
        tickets[row] = previous
        if previous == x.size(0) - 1:
            selected = alias[:]
            if mode == "scan":
                result[:] = torch.cumsum(selected, 0).to(torch.int32)
            elif mode == "reduce":
                result[:] = selected + selected.sum().to(torch.int32)
            else:
                result[:] = selected * 2
            if reset:
                counter[0] = 0
    return result


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _fragment_terminal_rejected(x, counter, result, mode: hl.constexpr):
    for row in hl.grid(x.size(0)):
        local = hl.zeros([x.size(1)], dtype=torch.int32)
        index = hl.arange(x.size(1))
        hl.atomic_add(local, [index], x[row, :])
        if mode == "vector":
            previous = hl.atomic_add(counter, [index * 0], 1, sem="acq_rel")
        elif mode == "singleton":
            previous = hl.atomic_add(counter, [hl.arange(1)], 1, sem="acq_rel")
        else:
            previous = hl.atomic_add(counter, [0], 1, sem="acq_rel")
        lazy = local + 1
        if mode == "loaded":
            condition = counter[0] == x.size(0)
        else:
            condition = previous == x.size(0) - 1
        if mode == "loop":
            for iteration in range(2):
                if condition:
                    result[:] = x[row, :] + iteration
        else:
            if condition:
                if mode == "local":
                    result[:] = local
                elif mode == "lazy":
                    result[:] = lazy
                elif mode == "update":
                    hl.atomic_add(local, [index], x[row, :])
                elif mode == "global":
                    hl.atomic_add(counter, [0], 1)
                elif mode == "nested":
                    if previous >= 0:
                        result[:] = x[row, :]
                else:
                    result[:] = x[row, :]
            if mode == "later":
                hl.atomic_add(local, [index], x[row, :])
            elif mode == "nonterminal":
                result[:] = x[row, :]
    return result


def _terminal_finalizer_args(rows, width, mode, reset=False, device="cpu"):
    x = torch.arange(rows * width, dtype=torch.int32, device=device).reshape(
        rows, width
    )
    partials = torch.zeros(width, dtype=torch.int32, device=device)
    return (
        x,
        partials,
        partials.view_as(partials),
        torch.zeros(1, dtype=torch.int32, device=device),
        torch.full((rows,), -99, dtype=torch.int32, device=device),
        torch.full_like(partials, -999),
        "acq_rel",
        mode,
        reset,
    )


def _check_terminal_finalizer(args):
    x, partials, alias, counter, tickets, result, _sem, mode, reset = args
    expected = x.sum(0).to(torch.int32)
    torch.testing.assert_close(partials, expected, rtol=0, atol=0)
    torch.testing.assert_close(alias, expected, rtol=0, atol=0)
    if mode == "scan":
        expected = expected.cumsum(0).to(torch.int32)
    elif mode == "reduce":
        expected = expected + expected.sum().to(torch.int32)
    else:
        expected = expected * 2
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        tickets.sort().values,
        torch.arange(x.size(0), dtype=torch.int32, device=x.device),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        counter,
        torch.full_like(counter, 0 if reset else x.size(0)),
        rtol=0,
        atol=0,
    )


class TestFragmentTerminalFinalizerCPU(unittest.TestCase):
    def test_generated_uniform_branch_alias_reads_and_collectives(self):
        for rows, width, threads in ((1, 17, 32), (3, 65, 128), (5, 129, 512)):
            for mode in ("plain", "scan", "reduce"):
                with self.subTest(rows=rows, width=width, threads=threads, mode=mode):
                    args = _terminal_finalizer_args(rows, width, mode, reset=True)
                    code = _fragment_ordered_codegen(
                        _fragment_terminal_finalizer, args, threads
                    )
                    events = []
                    workers = []
                    _simulate_register_load_program(
                        code,
                        args[0],
                        threads,
                        host_tensors=dict(
                            zip(
                                ("partials", "alias", "counter", "tickets", "result"),
                                args[1:6],
                                strict=True,
                            )
                        ),
                        lane_order=list(reversed(range(threads))),
                        atomic_workers=workers,
                        memory_events=events,
                    )
                    _check_terminal_finalizer(args)
                    global_owners = [lane for scope, lane in workers if scope == "gpu"]
                    self.assertEqual(len(global_owners), rows * (width + 1))
                    arrivals = [
                        e for e in events if e[0] == "atomic" and e[4] == "acq_rel"
                    ]
                    self.assertEqual(len(arrivals), rows)
                    self.assertTrue(all(e[2] == 0 for e in arrivals))

    def test_predicate_shared_slot_survives_branch_allocation_pressure(self):
        from unittest.mock import patch

        from helion._compiler.cute.computed_fragment import FragmentCompiler

        original_conditional = FragmentCompiler.conditional
        original_allocate = FragmentCompiler.allocate
        active = set()
        allocations = []
        pooled = []

        def conditional(compiler, node, values):
            active.update(compiler.referenced_buffers([values[node.args[0]]]))
            pooled.extend(
                capacity
                for name, _dtype, capacity in compiler.buffers
                if name in active
            )
            try:
                return original_conditional(compiler, node, values)
            finally:
                active.clear()

        def allocate(compiler, value):
            result = original_allocate(compiler, value)
            if active:
                allocations.append(result.storage)
                self.assertNotIn(result.storage, active)
            return result

        args = _terminal_finalizer_args(3, 129, "reduce")
        with (
            patch.object(FragmentCompiler, "conditional", conditional),
            patch.object(FragmentCompiler, "allocate", allocate),
        ):
            source = _fragment_ordered_codegen(_fragment_terminal_finalizer, args, 128)
        self.assertTrue(allocations)
        self.assertTrue(any(capacity > 1 for capacity in pooled))
        _simulate_register_load_program(
            source,
            args[0],
            128,
            host_tensors=dict(
                zip(
                    ("partials", "alias", "counter", "tickets", "result"),
                    args[1:6],
                    strict=True,
                )
            ),
            lane_order=list(reversed(range(128))),
        )
        _check_terminal_finalizer(args)

    def test_adversarial_control_and_local_lifetime_rejections(self):
        for mode in (
            "vector",
            "singleton",
            "loaded",
            "loop",
            "local",
            "lazy",
            "update",
            "global",
            "nested",
            "later",
            "nonterminal",
        ):
            with self.subTest(mode=mode), self.assertRaises(helion.exc.InvalidConfig):
                _fragment_ordered_codegen(
                    _fragment_terminal_rejected,
                    (
                        torch.ones((3, 17), dtype=torch.int32),
                        torch.zeros(1, dtype=torch.int32),
                        torch.zeros(17, dtype=torch.int32),
                        mode,
                    ),
                )
        for sem in ("relaxed", "release"):
            args = list(_terminal_finalizer_args(3, 17, "plain"))
            args[6] = sem
            with self.subTest(sem=sem), self.assertRaises(helion.exc.InvalidConfig):
                _fragment_ordered_codegen(_fragment_terminal_finalizer, tuple(args))

    def test_emission_rejects_nonshared_or_warp_ticket_ownership(self):
        from dataclasses import replace
        from unittest.mock import patch

        from helion._compiler.cute.computed_fragment import FragmentCompiler

        original = FragmentCompiler.atomic_add
        for mode in ("nonshared", "warp"):

            def atomic_add(compiler, node, values, mode=mode):
                result = original(compiler, node, values)
                if node in compiler.scalar_ordered_tickets:
                    if mode == "nonshared":
                        compiler.scalar_ordered_tickets[node] = replace(
                            result, storage=None
                        )
                    else:
                        compiler.warp_result_nodes[node] = 1
                return result

            args = _terminal_finalizer_args(3, 17, "plain")
            with (
                self.subTest(mode=mode),
                patch.object(FragmentCompiler, "atomic_add", atomic_add),
                self.assertRaisesRegex(
                    helion.exc.InvalidConfig, "shared scalar ticket"
                ),
            ):
                _fragment_ordered_codegen(_fragment_terminal_finalizer, args)


@onlyBackends("cute")
class TestFragmentTerminalFinalizerNative(TestCase):
    def test_multi_cta_finalization_aliases_and_reset(self):
        for rows, width, threads in ((1, 17, 32), (3, 65, 128), (17, 129, 512)):
            for mode in ("plain", "scan", "reduce"):
                with self.subTest(rows=rows, width=width, threads=threads, mode=mode):
                    args = _terminal_finalizer_args(
                        rows, width, mode, reset=True, device=DEVICE
                    )
                    before = args[0].clone()
                    code_and_output(
                        _fragment_terminal_finalizer,
                        args,
                        cute_fragment_threads=threads,
                    )
                    _check_terminal_finalizer(args)
                    torch.testing.assert_close(args[0], before, rtol=0, atol=0)

    def test_initialized_graph_replay_has_one_finalizer_each_time(self):
        args = _terminal_finalizer_args(17, 65, "scan", reset=True, device=DEVICE)
        before = args[0].clone()
        bound = _fragment_terminal_finalizer.bind(args)
        config = bound.config_spec.default_config()
        config.config["cute_fragment_threads"] = 128
        compiled = bound.compile_config(config)
        compiled(*args)
        _check_terminal_finalizer(args)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            args[1].zero_()
            args[3].zero_()
            args[4].fill_(-99)
            args[5].fill_(-999)
            compiled(*args)
        for replay in range(3):
            with self.subTest(replay=replay):
                graph.replay()
                _check_terminal_finalizer(args)
                torch.testing.assert_close(args[0], before, rtol=0, atol=0)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _fragment_terminal_loop_finalizer(
    x,
    partials,
    alias,
    counter,
    tickets,
    result,
    begin: int,
    end: int,
    step: hl.constexpr,
):
    for row in hl.grid(x.size(0)):
        local = hl.zeros([x.size(1)], dtype=x.dtype)
        index = hl.arange(x.size(1))
        hl.atomic_add(local, [index], x[row, :])
        hl.atomic_add(partials, [index], local)
        ticket = hl.atomic_add(counter, [0], 1, sem="acq_rel")
        tickets[row] = ticket
        if ticket == x.size(0) - 1:
            total = hl.full([], 0, dtype=x.dtype)
            previous = hl.full([], -1, dtype=x.dtype)
            mass = hl.full([], 0.0, dtype=torch.float32)
            found = hl.full([], False, dtype=torch.bool)
            for column in range(begin, end, step):
                value = hl.load(alias, [column % x.size(1)])
                previous, total = total, total + value
                mass = mass + previous.to(torch.float32) * 0.5
                found = found | (value > 2)
            result[0] = total.to(torch.float32)
            result[1] = previous.to(torch.float32)
            result[2] = mass
            result[3] = found.to(torch.float32)
            counter[0] = 0
    return result


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _fragment_terminal_loop_rejected(x, counter, result, mode: hl.constexpr):
    for row in hl.grid(x.size(0)):
        local = hl.zeros([x.size(1)], dtype=torch.int32)
        index = hl.arange(x.size(1))
        hl.atomic_add(local, [index], x[row, :])
        ticket = hl.atomic_add(counter, [0], 1, sem="acq_rel")
        if ticket == x.size(0) - 1:
            if mode == "vector":
                total = hl.zeros([x.size(1)], dtype=torch.int32)
            else:
                total = hl.full([], 0, dtype=torch.int32)
            for column in range(3):
                if mode == "store":
                    result[column] = total
                elif mode == "atomic":
                    hl.atomic_add(counter, [0], 1)
                elif mode == "nested":
                    for inner in range(2):
                        total = total + inner
                elif mode == "conditional":
                    if total > 0:
                        total = total + 1
                elif mode == "local":
                    total = total + local.sum(dtype=torch.int32)
                elif mode == "uninitialized":
                    fresh = x[row, column]
                else:
                    total = total + column
            if mode == "uninitialized":
                result[:] = fresh
            else:
                result[:] = total
    return result


def _terminal_loop_args(rows, width, dtype, begin, end, step, device="cpu"):
    x = (torch.arange(rows * width, device=device).reshape(rows, width) % 7).to(dtype)
    if dtype == torch.float32:
        x = x * 0.25
    partials = torch.zeros(width, dtype=dtype, device=device)
    return (
        x,
        partials,
        partials.view_as(partials),
        torch.zeros(1, dtype=torch.int32, device=device),
        torch.full((rows,), -99, dtype=torch.int32, device=device),
        torch.full((4,), -999.0, device=device),
        begin,
        end,
        step,
    )


def _check_terminal_loop(args):
    x, partials, alias, counter, tickets, result, begin, end, step = args
    expected = x.sum(0).to(x.dtype)
    total = torch.zeros((), dtype=x.dtype, device=x.device)
    previous = torch.full((), -1, dtype=x.dtype, device=x.device)
    mass = torch.zeros((), dtype=torch.float32, device=x.device)
    found = torch.full((), False, dtype=torch.bool, device=x.device)
    for column in range(begin, end, step):
        value = expected[column % x.size(1)]
        previous, total = total, total + value
        mass = mass + previous.float() * 0.5
        found = found | (value > 2)
    reference = torch.stack((total.float(), previous.float(), mass, found.float()))
    for actual, wanted in (
        (partials, expected),
        (alias, expected),
        (counter, torch.zeros_like(counter)),
        (
            tickets.sort().values,
            torch.arange(x.size(0), dtype=torch.int32, device=x.device),
        ),
        (result, reference),
    ):
        torch.testing.assert_close(actual, wanted, rtol=0, atol=0)


class TestFragmentTerminalLoopCPU(unittest.TestCase):
    def test_generated_scalar_carries_zero_trips_and_runtime_bounds(self):
        for dtype in (torch.int32, torch.float32):
            for begin, end, step in ((0, 0, 1), (0, 1, 1), (1, 13, 1), (12, 1, 1)):
                for threads in (32, 128):
                    with self.subTest(
                        dtype=dtype, bounds=(begin, end, step), threads=threads
                    ):
                        args = _terminal_loop_args(3, 17, dtype, begin, end, step)
                        code = _fragment_ordered_codegen(
                            _fragment_terminal_loop_finalizer, args, threads
                        )
                        _simulate_register_load_program(
                            code,
                            args[0],
                            threads,
                            host_tensors=dict(
                                zip(
                                    (
                                        "partials",
                                        "alias",
                                        "counter",
                                        "tickets",
                                        "result",
                                    ),
                                    args[1:6],
                                    strict=True,
                                )
                            ),
                            scalar_args={"begin": begin, "end": end},
                            lane_order=list(reversed(range(threads))),
                        )
                        _check_terminal_loop(args)

    def test_one_generated_wrapper_replays_bounds_and_keeps_predicate_live(self):
        from unittest.mock import patch

        from helion._compiler.cute.computed_fragment import FragmentCompiler

        original_loop = FragmentCompiler.loop
        original_allocate = FragmentCompiler.allocate
        active = set()
        allocated = []

        def loop(compiler, node, values):
            active.update(
                compiler.referenced_buffers(compiler.scalar_ordered_tickets.values())
            )
            try:
                return original_loop(compiler, node, values)
            finally:
                active.clear()

        def allocate(compiler, value):
            result = original_allocate(compiler, value)
            if active:
                allocated.append(result.storage)
                self.assertNotIn(result.storage, active)
            return result

        args = _terminal_loop_args(3, 65, torch.float32, 1, 13, 1)
        with (
            patch.object(FragmentCompiler, "loop", loop),
            patch.object(FragmentCompiler, "allocate", allocate),
        ):
            code = _fragment_ordered_codegen(
                _fragment_terminal_loop_finalizer, args, 512
            )
        self.assertTrue(allocated)
        for begin, end in ((0, 0), (0, 1), (1, 13), (7, 4)):
            with self.subTest(begin=begin, end=end):
                args = _terminal_loop_args(3, 65, torch.float32, begin, end, 1)
                _simulate_register_load_program(
                    code,
                    args[0],
                    512,
                    host_tensors=dict(
                        zip(
                            ("partials", "alias", "counter", "tickets", "result"),
                            args[1:6],
                            strict=True,
                        )
                    ),
                    scalar_args={"begin": begin, "end": end},
                    lane_order=list(reversed(range(512))),
                )
                _check_terminal_loop(args)

    def test_loop_keeps_non_scalar_geometry_and_unknown_origins_rejected(self):
        from unittest.mock import patch

        from helion._compiler.cute.computed_fragment import FragmentCompiler
        from helion._compiler.host_function import HostFunction

        for step in (2, -2):
            args = _terminal_loop_args(3, 17, torch.int32, 1, 13, step)
            with (
                self.subTest(step=step),
                self.assertRaisesRegex(
                    helion.exc.InvalidConfig, "uniform scalar loops"
                ),
            ):
                _fragment_ordered_codegen(_fragment_terminal_loop_finalizer, args)
        original = FragmentCompiler.uniform_finalizer_symbols
        removed = []

        def uniform(compiler, nodes, loop_blocks=None):
            entries = HostFunction.current().expr_to_origin
            saved = {
                symbol: entry
                for symbol, entry in entries.items()
                if loop_blocks
                and type(entry.origin).__name__ == "GridOrigin"
                and entry.origin.block_id in loop_blocks
            }
            for symbol in saved:
                removed.append(symbol)
                entries.pop(symbol)
            try:
                return original(compiler, nodes, loop_blocks)
            finally:
                entries.update(saved)

        args = _terminal_loop_args(3, 17, torch.int32, 1, 13, 1)
        with (
            patch.object(FragmentCompiler, "uniform_finalizer_symbols", uniform),
            self.assertRaisesRegex(
                helion.exc.InvalidConfig, "proved uniform scalar origins"
            ),
        ):
            _fragment_ordered_codegen(_fragment_terminal_loop_finalizer, args)
        self.assertTrue(removed)

    def test_loop_rejects_effects_nested_control_and_non_scalar_carries(self):
        for mode in (
            "store",
            "atomic",
            "nested",
            "conditional",
            "vector",
            "local",
        ):
            with self.subTest(mode=mode), self.assertRaises(helion.exc.InvalidConfig):
                _fragment_ordered_codegen(
                    _fragment_terminal_loop_rejected,
                    (
                        torch.ones((3, 17), dtype=torch.int32),
                        torch.zeros(1, dtype=torch.int32),
                        torch.zeros(17, dtype=torch.int32),
                        mode,
                    ),
                )

    def test_frontend_cross_root_rejections_precede_loop_admission(self):
        for mode in ("uninitialized",):
            with (
                self.subTest(mode=mode),
                self.assertRaises(helion.exc.CrossRootDeviceValue),
            ):
                _fragment_ordered_codegen(
                    _fragment_terminal_loop_rejected,
                    (
                        torch.ones((3, 17), dtype=torch.int32),
                        torch.zeros(1, dtype=torch.int32),
                        torch.zeros(17, dtype=torch.int32),
                        mode,
                    ),
                )

    def test_uninitialized_loop_output_is_rejected_at_real_codegen(self):
        from unittest.mock import patch

        from helion._compiler.cute import local_atomic
        from helion.language import _tracing_ops

        original = local_atomic.terminal_loop_symbols
        changed = []

        def proof(node, graphs):
            phi = next(
                user
                for item in node.users
                for user in item.users
                if user.target is _tracing_ops._phi
            )
            changed.append(phi)
            phi.target = _tracing_ops._new_var
            try:
                return original(node, graphs)
            finally:
                phi.target = _tracing_ops._phi

        args = _terminal_loop_args(3, 17, torch.int32, 1, 13, 1)
        with (
            patch.object(local_atomic, "terminal_loop_symbols", proof),
            self.assertRaisesRegex(
                helion.exc.InvalidConfig, "initialized scalar carries"
            ),
        ):
            _fragment_ordered_codegen(_fragment_terminal_loop_finalizer, args)
        self.assertTrue(changed)

    def test_scalar_loop_preserves_wide_runtime_bounds(self):
        import numpy as np

        for dtype in (torch.int32, torch.float32):
            args = _terminal_loop_args(5, 9, dtype, 0, 2, 1)
            source = _fragment_ordered_codegen(
                _fragment_terminal_loop_finalizer, args, 64
            )
            for begin in (
                0,
                2**31 - 1,
                2**31 + 4,
                -(2**31) - 4,
                2**32 + 4,
                -(2**32) - 4,
            ):
                for fresh in (False, True):
                    with self.subTest(dtype=dtype, begin=begin, fresh=fresh):
                        args = _terminal_loop_args(5, 9, dtype, begin, begin + 2, 1)
                        code = (
                            _fragment_ordered_codegen(
                                _fragment_terminal_loop_finalizer, args, 64
                            )
                            if fresh
                            else source
                        )
                        _simulate_register_load_program(
                            code,
                            args[0],
                            64,
                            host_tensors=dict(
                                zip(
                                    (
                                        "partials",
                                        "alias",
                                        "counter",
                                        "tickets",
                                        "result",
                                    ),
                                    args[1:6],
                                    strict=True,
                                )
                            ),
                            scalar_args={
                                "begin": np.int64(begin),
                                "end": np.int64(begin + 2),
                            },
                        )
                        _check_terminal_loop(args)

    @skipUnlessCuteAvailable("requires CuTe DSL")
    def test_terminal_loop_range_uses_actual_sdk_i64_induction(self):
        import ast
        import copy
        import importlib.util
        from pathlib import Path
        import tempfile

        import cutlass
        from cutlass._mlir import ir
        from cutlass._mlir.dialects import func

        from helion.runtime.cute.launcher import _cute_scalar_annotation

        self.assertEqual(_cute_scalar_annotation("int"), "cutlass.Int64")
        args = _terminal_loop_args(5, 9, torch.int32, 2**31 + 4, 2**31 + 6, 1)
        source = _fragment_ordered_codegen(_fragment_terminal_loop_finalizer, args, 64)
        loop = next(
            node
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.For)
            and isinstance(node.target, ast.Name)
            and node.target.id.startswith("fragment_tile")
        )
        self.assertEqual(
            ast.unparse(loop.iter), "range(cutlass.Int64(begin), cutlass.Int64(end), 1)"
        )
        loop = copy.deepcopy(loop)
        loop.body = ast.parse(f"total = total + {loop.target.id}").body
        module_text = (
            "import cutlass\nimport cutlass.cute as cute\n@cute.jit\n"
            "def sdk_loop(begin, end):\n    total = cutlass.Int64(0)\n"
            + "\n".join("    " + line for line in ast.unparse(loop).splitlines())
            + "\n    return total\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "range_width.py"
            path.write_text(module_text)
            spec = importlib.util.spec_from_file_location("terminal_range_width", path)
            assert spec is not None and spec.loader is not None
            sdk = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(sdk)
            with ir.Context(), ir.Location.unknown():
                module = ir.Module.create()
                with ir.InsertionPoint(module.body):
                    fn = func.FuncOp(
                        "range_width",
                        ([cutlass.Int64.mlir_type] * 2, [cutlass.Int64.mlir_type]),
                    )
                    block = fn.add_entry_block()
                    with ir.InsertionPoint(block):
                        value = sdk.sdk_loop(
                            *(cutlass.Int64(arg) for arg in block.arguments)
                        )
                        func.ReturnOp([value.ir_value()])
                self.assertTrue(module.operation.verify())
                loop_op = next(
                    op for op in block.operations if op.operation.name == "scf.for"
                )
                self.assertEqual(str(loop_op.body.arguments[0].type), "i64")
                self.assertNotIn("arith.trunci", str(module))


@onlyBackends("cute")
class TestFragmentTerminalLoopNative(TestCase):
    def test_multi_cta_scalar_loop_carries(self):
        for rows, width, threads in ((1, 17, 32), (3, 65, 128), (17, 129, 512)):
            for dtype in (torch.int32, torch.float32):
                for begin, end, step in ((0, 0, 1), (0, 1, 1), (1, 13, 1)):
                    with self.subTest(
                        shape=(rows, width), dtype=dtype, bounds=(begin, end, step)
                    ):
                        args = _terminal_loop_args(
                            rows, width, dtype, begin, end, step, device=DEVICE
                        )
                        before = args[0].clone()
                        code_and_output(
                            _fragment_terminal_loop_finalizer,
                            args,
                            cute_fragment_threads=threads,
                        )
                        _check_terminal_loop(args)
                        torch.testing.assert_close(args[0], before, rtol=0, atol=0)

    def test_multi_cta_wide_loop_bounds(self):
        for dtype in (torch.int32, torch.float32):
            for begin in (2**31 + 4, -(2**31) - 4, 2**32 + 4):
                with self.subTest(dtype=dtype, begin=begin):
                    args = _terminal_loop_args(
                        5, 9, dtype, begin, begin + 2, 1, device=DEVICE
                    )
                    before = args[0].clone()
                    code_and_output(
                        _fragment_terminal_loop_finalizer,
                        args,
                        cute_fragment_threads=64,
                    )
                    _check_terminal_loop(args)
                    torch.testing.assert_close(args[0], before, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
