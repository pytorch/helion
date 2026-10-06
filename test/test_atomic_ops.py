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
                "CTA-local atomics require a constant one-dimensional root/arm "
                "allocation of int32/float32"
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
        def __init__(self, values, offset=None, initialized=None):
            self.values = values
            self.offset = values.storage_offset() if offset is None else int(offset)
            self.flat = values.as_strided(
                (values.untyped_storage().nbytes() // values.element_size(),),
                (1,),
                storage_offset=0,
            )
            self.initialized = initialized

        def __add__(self, offset):
            return Pointer(self.values, self.offset + int(offset), self.initialized)

        @property
        def llvm_ptr(self):
            return self

        def load(self):
            assert 0 <= self.offset < self.flat.numel()
            if self.initialized is not None:
                assert self.offset in self.initialized, (
                    "shared pointer read before initialization"
                )
            value = self.flat[self.offset].item()
            if self.values.dtype == torch.int32:
                return np.int32(value)
            if self.values.dtype == torch.int64:
                return np.int64(value)
            return value

        def store(self, value):
            assert 0 <= self.offset < self.flat.numel()
            self.flat[self.offset] = (
                float(value) if self.values.dtype.is_floating_point else int(value)
            )
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

    class Registers(Shared):
        def fill(self, value):
            for index in range(self.values.numel()):
                self[index] = value

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
            collectives = {
                "cute.arch.vote_ballot_sync": "ballot",
                "cute.arch.match_sync": "match",
                "cute.arch.warp_redux_sync": "redux",
            }
            if function in collectives:
                kind = collectives[function]
                if kind == "ballot":
                    mask, value = ast.Constant(0xFFFFFFFF), node.args[0]
                elif kind == "match":
                    mask, value = node.args
                else:
                    assert ast.literal_eval(node.args[1]) == "add"
                    value, mask = node.args[0], node.args[2]
                return ast.copy_location(
                    ast.Yield(
                        ast.Tuple(
                            [
                                ast.Constant(kind),
                                ast.Constant(node.lineno),
                                mask,
                                value,
                            ],
                            ast.Load(),
                        )
                    ),
                    node,
                )
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
            range_constexpr=range,
        ),
        "cute": SimpleNamespace(
            make_rmem_tensor=lambda shape, dtype: Registers(shape[0], dtype),
            math=SimpleNamespace(min=np.minimum, max=np.maximum),
            make_layout=lambda shape: shape,
            arch=SimpleNamespace(
                thread_idx=lambda: (state["lane"], 0, 0),
                block_idx=lambda: (state["row"], 0, 0),
                atomic_add=atomic_add,
                lanemask_lt=lambda: np.uint32((1 << (state["lane"] % 32)) - 1),
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
                for event in group:
                    if not isinstance(event, tuple) or event[0] not in (
                        "ballot",
                        "match",
                        "redux",
                    ):
                        continue
                    kind, line, mask, _value = event
                    members = [lane for lane in range(32) if int(mask) & (1 << lane)]
                    assert members, "empty participating mask"
                    if not all(
                        isinstance(group[lane], tuple) and group[lane][:3] == event[:3]
                        for lane in members
                    ):
                        continue
                    values = [group[lane][3] for lane in members]
                    if kind == "ballot":
                        result = np.uint32(
                            sum(1 << lane for lane in members if group[lane][3])
                        )
                        returned = dict.fromkeys(members, result)
                    elif kind == "match":
                        returned = {
                            lane: np.uint32(
                                sum(
                                    1 << other
                                    for other in members
                                    if group[other][3] == group[lane][3]
                                )
                            )
                            for lane in members
                        }
                    else:
                        result = np.add.reduce(
                            np.array(values, dtype=np.int32), dtype=np.int32
                        )
                        returned = dict.fromkeys(members, result)
                    for lane in order:
                        if lane - base in returned:
                            reached[lane] = advance(
                                lane, returned[lane - base], send=True
                            )
                    progressed = True
                    break
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


def _private_loop_codegen(args, enabled, threads=128):
    from test._cute_binding import _cpu_bind
    from test._cute_binding import _forbid_native_compile
    from test._cute_binding import _mock_cuda_unavailable
    from test.cute_population_contracts import _target

    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound = _cpu_bind(_fragment_terminal_loop_finalizer, args)
        config = bound.config_spec.default_config()
        config.config.update(
            cute_fragment_threads=threads,
            cute_fragment_private_scalar_loops=enabled,
        )
        return bound.to_code(config)


class TestFragmentPrivateScalarLoopCPU(unittest.TestCase):
    def test_generated_values_zero_trip_aliases_and_wide_bounds(self):
        for dtype, width, threads in ((torch.int32, 17, 32), (torch.float32, 65, 128)):
            sources = [
                _private_loop_codegen(
                    _terminal_loop_args(3, width, dtype, 0, 2, 1), enabled, threads
                )
                for enabled in (False, True)
            ]
            self.assertNotIn("fragment_private_scalar", sources[0])
            self.assertIn("fragment_private_scalar", sources[1])
            for begin, end in ((0, 0), (7, 4), (-3, 19), (2**31 + 4, 2**31 + 6)):
                observed = []
                for source in sources:
                    args = _terminal_loop_args(3, width, dtype, begin, end, 1)
                    _simulate_register_load_program(
                        source,
                        args[0],
                        threads,
                        host_tensors=dict(
                            zip(
                                ("partials", "alias", "counter", "tickets", "result"),
                                args[1:6],
                                strict=True,
                            )
                        ),
                        scalar_args={"begin": begin, "end": end},
                        lane_order=list(reversed(range(threads))),
                    )
                    _check_terminal_loop(args)
                    observed.append(args)
                for before, after in zip(observed[0], observed[1], strict=True):
                    if isinstance(before, torch.Tensor):
                        self.assertTrue(
                            torch.equal(
                                before.view(torch.uint8), after.view(torch.uint8)
                            )
                        )

    def test_emitted_owner_and_scratch_initialization_fail_closed(self):
        import ast
        from unittest.mock import patch

        from helion._compiler.cute import computed_fragment

        original = computed_fragment.privatize_scalar_loop
        for mutation in (
            "owner",
            "collective",
            "slot",
            "uninitialized",
            "conditional_store",
            "alias",
        ):

            def corrupt(compiler, loop, outgoing, mutation=mutation):
                first = next(n for n in loop.body if isinstance(n, ast.For))
                if mutation == "owner":
                    first.iter.args[0] = ast.Constant(1)
                elif mutation == "collective":
                    first.body.insert(0, ast.parse("cute.arch.sync_threads()").body[0])
                elif mutation == "slot":
                    for node in ast.walk(loop):
                        if isinstance(node, ast.Subscript):
                            node.slice = ast.Constant(1)
                            break
                elif mutation == "conditional_store":
                    target = min(outgoing)
                    first.body.append(
                        ast.parse(f"if True:\n    {target}[0] = {target}[0]").body[0]
                    )
                elif mutation == "alias":
                    target = min(outgoing)
                    first.body.insert(0, ast.parse(f"hidden = {target}").body[0])
                else:
                    scratch = next(
                        name
                        for name, _dtype, _size in compiler.buffers
                        if name not in outgoing
                    )
                    first.body.insert(
                        0, ast.parse(f"{scratch}[0] = {scratch}[0]").body[0]
                    )
                return original(compiler, loop, outgoing)

            with (
                self.subTest(mutation=mutation),
                patch.object(computed_fragment, "privatize_scalar_loop", corrupt),
                self.assertRaisesRegex(helion.exc.InvalidConfig, "thread-zero"),
            ):
                _private_loop_codegen(
                    _terminal_loop_args(3, 17, torch.int32, 0, 2, 1), True
                )

    @skipUnlessCuteAvailable("requires CuTe DSL")
    def test_private_actual_sdk_preprocessor_and_legacy_scope(self):
        import ast
        import importlib.util
        import inspect
        from pathlib import Path
        import sys
        import tempfile
        from unittest.mock import patch

        from cutlass.base_dsl.ast_preprocessor import DSLPreprocessor

        from test._cute_binding import _cpu_bind
        from test._cute_binding import _forbid_native_compile
        from test._cute_binding import _mock_cuda_unavailable
        from test.cute_population_contracts import _target

        args = _terminal_loop_args(3, 17, torch.int32, -3, 19, 1)
        source = _private_loop_codegen(args, True)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "private_scalar_loop.py"
            path.write_text(source)
            spec = importlib.util.spec_from_file_location("private_scalar_sdk", path)
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            with (
                patch.dict(sys.modules, {spec.name: module}),
                patch("torch.cuda._lazy_init", side_effect=AssertionError("CPU only")),
            ):
                spec.loader.exec_module(module)
                name = next(
                    n.name
                    for n in ast.parse(source).body
                    if isinstance(n, ast.FunctionDef) and n.name.startswith("_helion_")
                )
                raw = inspect.unwrap(module.__dict__[name])
                with DSLPreprocessor(["cutlass"]).get_session() as session:
                    tree = session.transform(raw, dict(raw.__globals__))
                    compile(tree, str(path), "exec", dont_inherit=True)
        with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
            args = _terminal_finalizer_args(3, 17, "plain")
            bound = _cpu_bind(_fragment_terminal_finalizer, args)
            self.assertFalse(
                bound.config_spec.cute_fragment_private_scalar_loop_root_ids
            )
            config = bound.config_spec.default_config()
            before = bound.to_code(config)
            config.config["cute_fragment_private_scalar_loops"] = False
            self.assertEqual(before, bound.to_code(config))
            config.config["cute_fragment_private_scalar_loops"] = True
            with self.assertRaisesRegex(
                helion.exc.InvalidConfig, "scalar finalizer loop"
            ):
                bound.to_code(config)

    def test_default_and_population_prefix_remain_legacy(self):
        from copy import deepcopy

        from test._cute_binding import _cpu_bind
        from test._cute_binding import _forbid_native_compile
        from test._cute_binding import _mock_cuda_unavailable
        from test.cute_population_contracts import _target
        from test.cute_population_contracts import checked_initial_population

        from helion.autotuner.pattern_search import InitialPopulationStrategy
        from helion.autotuner.pattern_search import PatternSearch

        args = _terminal_loop_args(3, 17, torch.int32, 0, 2, 1)
        kernel = helion.kernel(
            _fragment_terminal_loop_finalizer.fn,
            backend="cute",
            static_shapes=True,
            autotune_effort="full",
        )
        key = "cute_fragment_private_scalar_loops"
        with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
            bound = _cpu_bind(kernel, args)
            spec = bound.config_spec
            self.assertTrue(spec.cute_fragment_private_scalar_loop_root_ids)
            default = spec.default_config()
            self.assertNotIn(key, default)
            self.assertTrue(all(key not in seed for seed in spec.compiler_seed_configs))
            off = deepcopy(default)
            off.config[key] = False
            self.assertEqual(bound.to_code(default), bound.to_code(off))
            for value in (1, "private", None):
                invalid = deepcopy(default)
                invalid.config[key] = value
                with self.assertRaises(helion.exc.InvalidConfig):
                    bound.to_code(invalid)
            group = next(g for g in spec.compiler_coverage_groups if g.key == key)
            self.assertFalse(group.legacy)
            self.assertEqual(group.dependencies, ())
            with bound.env:
                search = PatternSearch(
                    bound,
                    args,
                    initial_population=4,
                    initial_population_strategy=InitialPopulationStrategy.FROM_RANDOM,
                )
                checked_initial_population(search)
                outcome = next(
                    o
                    for o in search.compiler_coverage_outcomes
                    if o.mechanism == group.mechanism
                )
                self.assertIn(outcome.outcome, ("added", "already_present"))
                self.assertTrue(outcome.effective[key])


@onlyBackends("cute")
class TestFragmentPrivateScalarLoopNative(TestCase):
    def test_scalar_carries_aliases_zero_trip_and_wide_bounds(self):
        for dtype, width, threads in ((torch.int32, 17, 32), (torch.float32, 65, 128)):
            for begin, end in ((0, 0), (-3, 19), (2**31 + 4, 2**31 + 6)):
                with self.subTest(dtype=dtype, width=width, bounds=(begin, end)):
                    args = _terminal_loop_args(
                        3, width, dtype, begin, end, 1, device=DEVICE
                    )
                    before = args[0].clone()
                    code_and_output(
                        _fragment_terminal_loop_finalizer,
                        args,
                        cute_fragment_threads=threads,
                        cute_fragment_private_scalar_loops=True,
                    )
                    _check_terminal_loop(args)
                    torch.testing.assert_close(args[0], before, rtol=0, atol=0)

    def test_readonly_masked_loads(self):
        x = torch.arange(51, dtype=torch.float32, device=DEVICE).reshape(3, 17) * 0.125
        for begin, end in ((-3, 21), (20, 25), (5, 2)):
            with self.subTest(bounds=(begin, end)):
                counter = torch.zeros(1, dtype=torch.int32, device=DEVICE)
                result = torch.full((1,), float("nan"), device=DEVICE)
                before = x.clone()
                code_and_output(
                    _fragment_private_masked_loop,
                    (x, counter, result, begin, end),
                    cute_fragment_threads=128,
                    cute_fragment_private_scalar_loops=True,
                )
                expected = torch.tensor(0.0, dtype=torch.float32, device=DEVICE)
                for column in range(begin, end):
                    if 0 <= column < x.size(1):
                        expected = expected + x[0, column]
                torch.testing.assert_close(result, expected.reshape(1), rtol=0, atol=0)
                torch.testing.assert_close(x, before, rtol=0, atol=0)
                self.assertEqual(int(counter[0]), 3)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _fragment_private_masked_loop(x, counter, result, begin: int, end: int):
    for row in hl.grid(x.size(0)):
        local = hl.zeros([1], dtype=torch.int32)
        hl.atomic_add(local, [0], row)
        ticket = hl.atomic_add(counter, [0], 1, sem="acq_rel")
        if ticket == x.size(0) - 1:
            total = hl.full([], 0.0, dtype=torch.float32)
            for column in range(begin, end):
                value = hl.load(x, [0, column])
                total = total + value
            result[0] = total
    return result


class TestFragmentPrivateMaskedLoopCPU(unittest.TestCase):
    def test_readonly_conditional_global_loads_keep_masks_and_order(self):
        from test._cute_binding import _cpu_bind
        from test._cute_binding import _forbid_native_compile
        from test._cute_binding import _mock_cuda_unavailable
        from test.cute_population_contracts import _target

        x = torch.arange(51, dtype=torch.float32).reshape(3, 17) * 0.125
        for begin, end in ((-3, 21), (20, 25), (5, 2)):
            for enabled in (False, True):
                with self.subTest(bounds=(begin, end), private=enabled):
                    counter = torch.zeros(1, dtype=torch.int32)
                    result = torch.full((1,), float("nan"))
                    args = (x, counter, result, begin, end)
                    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
                        bound = _cpu_bind(_fragment_private_masked_loop, args)
                        config = bound.config_spec.default_config()
                        config.config.update(
                            cute_fragment_threads=32,
                            cute_fragment_private_scalar_loops=enabled,
                        )
                        code = bound.to_code(config)
                    _simulate_register_load_program(
                        code,
                        x,
                        32,
                        host_tensors={"counter": counter, "result": result},
                        scalar_args={"begin": begin, "end": end},
                        lane_order=list(reversed(range(32))),
                    )
                    expected = torch.tensor(0.0, dtype=torch.float32)
                    for column in range(begin, end):
                        if 0 <= column < x.size(1):
                            expected = expected + x[0, column]
                    torch.testing.assert_close(
                        result, expected.reshape(1), rtol=0, atol=0
                    )
                    self.assertEqual(int(counter[0]), 3)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _fragment_aggregated_histogram(
    x, indices, bins: hl.constexpr, initial: hl.constexpr, repeats: hl.constexpr
):
    out = torch.empty((x.size(0), bins), dtype=torch.int32, device=x.device)
    for row in hl.grid(x.size(0)):
        histogram = hl.full([bins], initial, dtype=torch.int32)
        for iteration in range(repeats):
            hl.atomic_add(histogram, [indices[row, :]], x[row, :] + iteration)
        out[row, :] = histogram
    return out


def _aggregation_reference(x, indices, bins, initial, repeats):
    expected = torch.full(
        (x.size(0), bins), initial, dtype=torch.int32, device=x.device
    )
    wrapped = torch.where(indices < 0, indices + bins, indices).long()
    for iteration in range(repeats):
        expected.scatter_add_(
            1, wrapped.reshape(x.size(0), -1), (x + iteration).reshape(x.size(0), -1)
        )
    return expected


def _aggregation_codegen(kernel, args, enabled, threads=128):
    from test._cute_binding import _cpu_bind
    from test._cute_binding import _forbid_native_compile
    from test._cute_binding import _mock_cuda_unavailable
    from test.cute_population_contracts import _target

    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound = _cpu_bind(kernel, args)
        config = bound.config_spec.default_config()
        config.config.update(
            cute_fragment_atomic_aggregation=enabled, cute_fragment_threads=threads
        )
        return bound.to_code(config)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _fragment_aggregation_mixed(x, counter, tickets, sem: hl.constexpr):
    out = torch.empty((x.size(0), 17), dtype=torch.int32, device=x.device)
    floating_out = torch.empty((x.size(0), 17), dtype=torch.float32, device=x.device)
    for row in hl.grid(x.size(0)):
        local = hl.zeros([17], dtype=torch.int32)
        floating = hl.zeros([17], dtype=torch.float32)
        indices = hl.arange(x.size(1)) % 17
        hl.atomic_add(local, [indices], x[row, :])
        hl.atomic_add(floating, [indices], x[row, :].to(torch.float32))
        tickets[row] = hl.atomic_add(counter, [row], 0, sem=sem)
        out[row, :] = local
        floating_out[row, :] = floating
    return out, floating_out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _fragment_aggregation_masked_tail(x):
    out = torch.empty((x.size(0), 17), dtype=torch.int32, device=x.device)
    for row in hl.grid(x.size(0)):
        local = hl.zeros([17], dtype=torch.int32)
        column = hl.arange(x.size(1))
        value = torch.where(column < x.size(1), 1.0, float("nan"))
        hl.atomic_add(local, [column % 17], value)
        out[row, :] = local
    return out


class TestFragmentAtomicAggregationCPU(unittest.TestCase):
    def test_masked_logical_tail_and_index_conversions_stay_guarded(self):
        import warnings

        for mode in ("logical_tail", "invalid_index"):
            for enabled in (False, True):
                with self.subTest(mode=mode, enabled=enabled):
                    x = torch.ones((2, 17), dtype=torch.float32)
                    inputs = {}
                    expected = torch.ones((2, 17), dtype=torch.int32)
                    if mode == "logical_tail":
                        kernel, args = _fragment_aggregation_masked_tail, (x,)
                    else:
                        indices = torch.arange(17, dtype=torch.int32).repeat(2, 1)
                        indices[:, 0] = 17
                        indices[:, 1] = -18
                        x[:, :2] = float("nan")
                        inputs["indices"] = indices
                        expected[:, :2] = 0
                        kernel = _fragment_aggregated_histogram
                        args = (x, indices, 17, 0, 1)
                    code = _aggregation_codegen(kernel, args, enabled, 32)
                    out = torch.full_like(expected, -99)
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", RuntimeWarning)
                        _simulate_register_load_program(
                            code, x, 32, host_tensors={"out": out, **inputs}
                        )
                    torch.testing.assert_close(out, expected, rtol=0, atol=0)

    def test_exact_weights_tails_wrapped_keys_and_multiple_rounds(self):
        import numpy as np

        for shape, threads in (
            ((2, 17), 32),
            ((2, 65), 128),
            ((2, 129), 512),
            ((2, 3, 17), 128),
        ):
            width = torch.empty(shape).numel() // shape[0]
            for mode in ("collision", "unique", "zero", "overflow"):
                with self.subTest(shape=shape, threads=threads, mode=mode):
                    x = (torch.arange(2 * width).reshape(shape) % 11 - 5).int()
                    bins = 257 if mode == "unique" else 17
                    idx = (
                        (torch.arange(width).reshape(shape[1:]) % bins)
                        .expand(shape)
                        .int()
                        .contiguous()
                    )
                    if mode == "collision":
                        idx.zero_()
                    elif mode == "zero":
                        x.zero_()
                    elif mode == "overflow":
                        x.flatten()[::3] = torch.iinfo(torch.int32).max
                        x.flatten()[1::7] = torch.iinfo(torch.int32).min
                    idx = torch.where(
                        torch.arange(width).reshape(shape[1:]) % 2 == 0, idx - bins, idx
                    )
                    args = (x, idx, bins, 2, 2)
                    expected = _aggregation_reference(*args)
                    for enabled in (False, True):
                        code = _aggregation_codegen(
                            _fragment_aggregated_histogram, args, enabled, threads
                        )
                        for order in (
                            list(range(threads)),
                            list(reversed(range(threads))),
                        ):
                            out = torch.full_like(expected, -999)
                            with np.errstate(over="ignore"):
                                _simulate_register_load_program(
                                    code,
                                    x,
                                    threads,
                                    host_tensors={"indices": idx, "out": out},
                                    lane_order=order,
                                )
                            torch.testing.assert_close(out, expected, rtol=0, atol=0)

    def test_collisions_reduce_operations_and_preserve_barriers(self):
        x = torch.ones((1, 65), dtype=torch.int32)
        indices = torch.zeros_like(x)
        expected = _aggregation_reference(x, indices, 17, 0, 1)
        counts, barriers = [], []
        for enabled in (False, True):
            code = _aggregation_codegen(
                _fragment_aggregated_histogram, (x, indices, 17, 0, 1), enabled, 32
            )
            out = torch.full_like(expected, -999)
            events = []
            _, count = _simulate_register_load_program(
                code,
                x,
                32,
                host_tensors={"indices": indices, "out": out},
                atomic_events=events,
            )
            torch.testing.assert_close(out, expected, rtol=0, atol=0)
            counts.append(len(events))
            barriers.append(count)
        self.assertEqual(counts, [65, 3])
        self.assertEqual(barriers[0], barriers[1])

    def test_zero_updates_are_private_only_and_invalid_lifetimes_still_reject(self):
        for mode in (
            "early_scan",
            "later_update",
            "loop_read",
            "loop_allocation",
            "conditional",
            "alias",
            "self_derived_update",
        ):
            with self.subTest(mode=mode), self.assertRaises(helion.exc.InvalidConfig):
                _aggregation_codegen(
                    _local_histogram_consumer_negative,
                    (torch.ones((1, 65), dtype=torch.int32), mode),
                    True,
                )
        with self.assertRaises(helion.exc.InvalidConfig):
            _aggregation_codegen(
                _local_histogram_consumers, (torch.ones((1, 65)), 17, "pointwise"), True
            )

    def test_broadcast_logical_domains_and_wide_contribution_casts(self):
        x = torch.ones((2, 31), dtype=torch.int32)
        indices = (torch.arange(17)[:, None] + torch.arange(31)) % 17
        expected = torch.zeros((2, 17), dtype=torch.int32)
        expected.scatter_add_(
            1,
            indices.flatten().expand(2, -1),
            torch.full((2, 17 * 31), 6, dtype=torch.int32),
        )
        code = _aggregation_codegen(_local_atomic_broadcast_domain, (x, 17), True, 128)
        out = torch.full_like(expected, -999)
        _simulate_register_load_program(code, x, 128, host_tensors={"out": out})
        torch.testing.assert_close(out, expected, rtol=0, atol=0)
        x = torch.full((2, 65), 2**40 + 3, dtype=torch.int64)
        indices = torch.zeros_like(x)
        args = (x, indices, 17, 0, 1)
        expected = _aggregation_reference(x.int(), indices, 17, 0, 1)
        code = _aggregation_codegen(_fragment_aggregated_histogram, args, True, 32)
        out = torch.full_like(expected, -999)
        _simulate_register_load_program(
            code, x, 32, host_tensors={"indices": indices, "out": out}
        )
        torch.testing.assert_close(out, expected, rtol=0, atol=0)

    def test_global_zero_ordered_returns_and_float_atomics_stay_per_element(self):
        for sem in ("relaxed", "acquire", "release", "acq_rel"):
            with self.subTest(sem=sem):
                x = torch.zeros((2, 17), dtype=torch.int32)
                counter = torch.full((2,), 9, dtype=torch.int32)
                tickets = torch.full_like(counter, -1)
                out = torch.full((2, 17), -1, dtype=torch.int32)
                floating_out = torch.full((2, 17), -1.0)
                code = _aggregation_codegen(
                    _fragment_aggregation_mixed, (x, counter, tickets, sem), True, 32
                )
                events = []
                memory = []
                _simulate_register_load_program(
                    code,
                    x,
                    32,
                    host_tensors={
                        "counter": counter,
                        "tickets": tickets,
                        "out": out,
                        "floating_out": floating_out,
                    },
                    atomic_events=events,
                    memory_events=memory,
                )
                self.assertEqual(sum(event[0] == "gpu" for event in events), 2)
                self.assertEqual(sum(event[0] == "cta" for event in events), 34)
                self.assertEqual(
                    [
                        event[4]
                        for event in memory
                        if event[0] == "atomic" and event[-1] == "gpu"
                    ],
                    [sem, sem],
                )
                self.assertTrue(torch.equal(counter, tickets))
                self.assertTrue(torch.equal(out, torch.zeros_like(out)))
                self.assertTrue(
                    torch.equal(floating_out, torch.zeros_like(floating_out))
                )

    def test_model_rejects_fullmask_after_tail_and_duplicate_leaders(self):
        import ast

        x = torch.ones((1, 17), dtype=torch.int32)
        indices = torch.zeros_like(x)
        code = _aggregation_codegen(
            _fragment_aggregated_histogram, (x, indices, 17, 0, 1), True, 32
        )
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and ast.unparse(node.func) == "cute.arch.match_sync"
            ):
                node.args[0] = ast.Constant(0xFFFFFFFF)
        with self.assertRaisesRegex(AssertionError, "incomplete warp"):
            _simulate_register_load_program(
                ast.unparse(tree), x, 32, host_tensors={"indices": indices}
            )
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.If) and "lanemask_lt" in ast.unparse(node.test):
                node.test = ast.Constant(True)
        out = torch.full((1, 17), -1, dtype=torch.int32)
        _simulate_register_load_program(
            ast.unparse(tree), x, 32, host_tensors={"indices": indices, "out": out}
        )
        self.assertFalse(torch.equal(out, _aggregation_reference(x, indices, 17, 0, 1)))

    @skipUnlessCuteAvailable("requires CuTe DSL")
    def test_actual_sdk_staged_match_redux_and_shared_atomic(self):
        import ast
        import importlib.util
        from pathlib import Path
        import tempfile

        import cutlass
        from cutlass._mlir import ir
        from cutlass._mlir.dialects import func
        import cutlass.cute as cute

        x = torch.ones((1, 17), dtype=torch.int32)
        source = _aggregation_codegen(
            _fragment_aggregated_histogram, (x, torch.zeros_like(x), 17, 0, 1), True, 32
        )
        fn = next(
            n
            for n in ast.parse(source).body
            if isinstance(n, ast.FunctionDef) and n.name.startswith("_helion_")
        )
        fn.name = "staged_histogram"
        # The actual generated body stages through the installed CuTe frontend.
        # Pointer arguments and scalar-thread intrinsics remain actual SDK IR.
        fn.decorator_list = [ast.parse("cute.jit", mode="eval").body]
        module = ast.fix_missing_locations(
            ast.Module(
                body=[
                    ast.Import([ast.alias("operator")]),
                    ast.Import([ast.alias("cutlass")]),
                    ast.Import([ast.alias("cutlass.cute", "cute")]),
                    *[n for n in ast.parse(source).body if isinstance(n, ast.Assign)],
                    fn,
                ],
                type_ignores=[],
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sdk_histogram.py"
            path.write_text(ast.unparse(module))
            spec = importlib.util.spec_from_file_location("sdk_histogram", path)
            sdk = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(sdk)
            with ir.Context(), ir.Location.unknown():
                emitted = ir.Module.create()
                with ir.InsertionPoint(emitted.body):
                    entry = func.FuncOp("entry", ([], []))
                    block = entry.add_entry_block()
                    with ir.InsertionPoint(block):
                        pointers = [
                            cute.make_ptr(
                                cutlass.Int32,
                                0,
                                cute.AddressSpace.gmem,
                                assumed_align=16,
                            )
                            for _ in fn.args.args
                        ]
                        tensors = [
                            cute.make_tensor(ptr, cute.make_layout((17,)))
                            for ptr in pointers
                        ]
                        sdk.staged_histogram(*tensors)
                        func.ReturnOp([])
                self.assertTrue(emitted.operation.verify())
                text = str(emitted)
                self.assertIn("nvvm.match.sync", text)
                self.assertIn("nvvm.redux.sync", text)
                self.assertIn("nvvm.vote.sync  ballot", text)
                self.assertIn("nvvm.atomicrmw", text)
                self.assertIn("#nvvm.mem_scope<cta>", text)


@onlyBackends("cute")
class TestFragmentAtomicAggregationNative(TestCase):
    def test_masked_logical_tail_contribution_conversion(self):
        x = torch.ones((2, 17), dtype=torch.float32, device=DEVICE)
        _, actual = code_and_output(
            _fragment_aggregation_masked_tail,
            (x,),
            cute_fragment_atomic_aggregation=True,
        )
        torch.testing.assert_close(
            actual,
            torch.ones((2, 17), dtype=torch.int32, device=DEVICE),
            rtol=0,
            atol=0,
        )

    def test_arbitrary_wrapped_histogram_keys_and_int32_boundaries(self):
        for width, threads, bins in ((17, 32, 17), (65, 128, 17), (129, 512, 257)):
            for mode in ("collision", "weighted", "overflow", "zero"):
                with self.subTest(width=width, threads=threads, mode=mode):
                    x = (
                        torch.arange(2 * width, device=DEVICE).reshape(2, width) % 11
                        - 5
                    ).int()
                    indices = (
                        (torch.arange(width, device=DEVICE) % bins)
                        .expand_as(x)
                        .contiguous()
                        .int()
                    )
                    if mode == "collision":
                        indices.zero_()
                    if mode == "overflow":
                        x[:, ::3] = torch.iinfo(torch.int32).max
                        x[:, 1::7] = torch.iinfo(torch.int32).min
                    if mode == "zero":
                        x.zero_()
                    indices = torch.where(
                        torch.arange(width, device=DEVICE) % 2 == 0,
                        indices - bins,
                        indices,
                    )
                    args = (x, indices, bins, 2, 2)
                    before = (x.clone(), indices.clone())
                    _, actual = code_and_output(
                        _fragment_aggregated_histogram,
                        args,
                        cute_fragment_atomic_aggregation=True,
                        cute_fragment_threads=threads,
                    )
                    torch.testing.assert_close(
                        actual, _aggregation_reference(*args), rtol=0, atol=0
                    )
                    torch.testing.assert_close(x, before[0], rtol=0, atol=0)
                    torch.testing.assert_close(indices, before[1], rtol=0, atol=0)

    def test_mixed_global_zero_semantics_and_floating_updates(self):
        for sem in ("relaxed", "acquire", "release", "acq_rel"):
            with self.subTest(sem=sem):
                x = torch.zeros((2, 17), dtype=torch.int32, device=DEVICE)
                counter = torch.full((2,), 9, dtype=torch.int32, device=DEVICE)
                tickets = torch.full_like(counter, -1)
                _, (out, floating) = code_and_output(
                    _fragment_aggregation_mixed,
                    (x, counter, tickets, sem),
                    cute_fragment_atomic_aggregation=True,
                )
                torch.testing.assert_close(
                    counter, torch.full_like(counter, 9), rtol=0, atol=0
                )
                torch.testing.assert_close(tickets, counter, rtol=0, atol=0)
                torch.testing.assert_close(out, torch.zeros_like(out), rtol=0, atol=0)
                torch.testing.assert_close(
                    floating, torch.zeros_like(floating), rtol=0, atol=0
                )

    def test_broadcast_domains_and_wide_contributions(self):
        x = torch.ones((2, 31), dtype=torch.int32, device=DEVICE)
        indices = (
            torch.arange(17, device=DEVICE)[:, None] + torch.arange(31, device=DEVICE)
        ) % 17
        expected = torch.zeros((2, 17), dtype=torch.int32, device=DEVICE)
        expected.scatter_add_(
            1,
            indices.flatten().expand(2, -1),
            torch.full((2, 17 * 31), 6, dtype=torch.int32, device=DEVICE),
        )
        _, actual = code_and_output(
            _local_atomic_broadcast_domain,
            (x, 17),
            cute_fragment_atomic_aggregation=True,
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        x = torch.full((2, 65), 2**40 + 3, dtype=torch.int64, device=DEVICE)
        indices = torch.zeros_like(x)
        _, actual = code_and_output(
            _fragment_aggregated_histogram,
            (x, indices, 17, 0, 1),
            cute_fragment_atomic_aggregation=True,
        )
        torch.testing.assert_close(
            actual, _aggregation_reference(x.int(), indices, 17, 0, 1), rtol=0, atol=0
        )


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _fragment_local_fetch_add(
    x, tickets, reused, reversed_tickets, mode: hl.constexpr, initial: hl.constexpr
):
    out = torch.empty((x.size(0), 17), dtype=torch.int32, device=x.device)
    for row in hl.grid(x.size(0)):
        local = hl.full([17], initial, dtype=torch.int32)
        lane = hl.arange(x.size(1))
        indices = lane % 17
        if mode == "negative":
            indices = indices - 17
        elif mode == "masked":
            indices = torch.where(lane + 1 < x.size(1), indices, 17)
        elif mode == "bounds":
            indices = lane - 18
        if mode == "epochs":
            hl.atomic_add(local, [indices], x[row, :])
        for iteration in range(2):
            previous = hl.atomic_add(local, [indices], x[row, :])
            if mode == "snapshot":
                hl.atomic_add(local, [indices], x[row, :])
            # The first consumer deliberately crosses physical lane/warp owners.
            reversed_tickets[row, iteration, :] = torch.flip(previous, [0])
            tickets[row, iteration, :] = previous
            reused[row, iteration, :] = previous ^ 85
        out[row, :] = local
    return out


def _local_fetch_args(width, mode, initial=3, value=1, device="cpu"):
    x = torch.full((2, width), value, dtype=torch.float32, device=device)
    outputs = [
        torch.full((2, 2, width), -99, dtype=torch.int32, device=device)
        for _ in range(3)
    ]
    return (x, *outputs, mode, initial)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _fragment_local_scalar_fetch(x, tickets):
    out = torch.empty((x.size(0), 17), dtype=torch.int32, device=x.device)
    for row in hl.grid(x.size(0)):
        local = hl.full([17], 5, dtype=torch.int32)
        for iteration in range(2):
            previous = hl.atomic_add(local, [-1], 1)
            tickets[row, iteration, :] = previous + hl.arange(x.size(1)) * 0
        out[row, :] = local
    return out


def _check_local_fetch(args, actual):
    x, tickets, reused, reversed_tickets, mode, initial = args
    lane = torch.arange(x.size(1), device=x.device)
    indices = lane % 17
    if mode == "negative":
        indices -= 17
    elif mode == "masked":
        indices = torch.where(lane + 1 < x.size(1), indices, 17)
    elif mode == "bounds":
        indices = lane - 18
    indices = torch.where(indices < 0, indices + 17, indices)
    valid = (indices >= 0) & (indices < 17)
    expected = torch.full_like(actual, initial, dtype=torch.int64)
    step = int(x[0, 0].to(torch.int32))
    for bin_index in range(17):
        selected = valid & (indices == bin_index)
        count = int(selected.sum())
        start = initial + (step * count if mode == "epochs" else 0)
        epochs = 4 if mode == "snapshot" else (3 if mode == "epochs" else 2)
        expected[:, bin_index] += epochs * step * count
        for iteration in range(2):
            values = (
                start
                + step
                * (
                    iteration * count * (2 if mode == "snapshot" else 1)
                    + torch.arange(count, device=x.device)
                )
            ).to(torch.int32)
            torch.testing.assert_close(
                tickets[:, iteration, selected].sort(-1).values,
                values.sort().values.expand(x.size(0), -1),
                rtol=0,
                atol=0,
            )
    torch.testing.assert_close(actual, expected.int(), rtol=0, atol=0)
    assert bool((tickets[:, :, ~valid] == 0).all())
    torch.testing.assert_close(reused, tickets ^ 85, rtol=0, atol=0)
    torch.testing.assert_close(reversed_tickets, tickets.flip(-1), rtol=0, atol=0)
    return int(valid.sum()) * x.size(0) * epochs


class TestFragmentLocalFetchAddCPU(unittest.TestCase):
    def test_scalar_result_is_one_arrival_broadcast_to_all_consumers(self):
        for threads in (32, 128, 512):
            with self.subTest(threads=threads):
                x = torch.ones((2, 65))
                tickets = torch.full((2, 2, 65), -99, dtype=torch.int32)
                actual = torch.empty((2, 17), dtype=torch.int32)
                source = _aggregation_codegen(
                    _fragment_local_scalar_fetch, (x, tickets), False, threads
                )
                workers = []
                _simulate_register_load_program(
                    source,
                    x,
                    threads,
                    host_tensors={"tickets": tickets, "out": actual},
                    atomic_workers=workers,
                )
                self.assertEqual(workers, [("cta", 0)] * 4)
                expected = torch.full_like(actual, 5)
                expected[:, -1] = 7
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(
                    tickets,
                    torch.tensor([5, 6]).view(1, 2, 1).expand_as(tickets).int(),
                    rtol=0,
                    atol=0,
                )

    def test_duplicate_bins_snapshots_wrapping_tails_and_loop_reuse(self):
        for width, threads in ((17, 32), (65, 128), (129, 128), (65, 512)):
            for mode in ("plain", "negative", "masked", "bounds", "epochs", "snapshot"):
                with self.subTest(width=width, threads=threads, mode=mode):
                    args = _local_fetch_args(width, mode)
                    source = _aggregation_codegen(
                        _fragment_local_fetch_add, args, False, threads
                    )
                    for reverse in (False, True):
                        actual = torch.empty((2, 17), dtype=torch.int32)
                        events = []
                        _simulate_register_load_program(
                            source,
                            args[0],
                            threads,
                            host_tensors={
                                "out": actual,
                                "tickets": args[1],
                                "reused": args[2],
                                "reversed_tickets": args[3],
                            },
                            lane_order=(
                                list(reversed(range(threads))) if reverse else None
                            ),
                            atomic_events=events,
                        )
                        expected_calls = _check_local_fetch(args, actual)
                        self.assertEqual(len(events), expected_calls)
                        self.assertTrue(all(event[0] == "cta" for event in events))
                    self.assertNotIn("fragment_warp_atomic", source)

    def test_returned_updates_never_aggregate_or_elide_zero(self):
        for initial, value in (
            ((1 << 24) + 5, 0.75),
            ((1 << 24) + 5, 1.75),
            ((1 << 24) + 5, -1.75),
            ((1 << 31) - 2, 1.75),
        ):
            with self.subTest(initial=initial, value=value):
                args = _local_fetch_args(65, "plain", initial, value)
                before = _aggregation_codegen(_fragment_local_fetch_add, args, False)
                with self.assertRaisesRegex(helion.exc.InvalidConfig, "unused"):
                    _aggregation_codegen(_fragment_local_fetch_add, args, True)
                self.assertNotIn("match_sync", before)
                actual = torch.empty((2, 17), dtype=torch.int32)
                events = []
                _simulate_register_load_program(
                    before,
                    args[0],
                    128,
                    host_tensors={
                        "out": actual,
                        "tickets": args[1],
                        "reused": args[2],
                        "reversed_tickets": args[3],
                    },
                    atomic_events=events,
                )
                self.assertEqual(len(events), _check_local_fetch(args, actual))

    def test_unused_aggregation_composes_with_ungrouped_local_returns(self):
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                args = _local_fetch_args(65, "epochs")
                source = _aggregation_codegen(_fragment_local_fetch_add, args, enabled)
                actual = torch.empty((2, 17), dtype=torch.int32)
                _simulate_register_load_program(
                    source,
                    args[0],
                    128,
                    host_tensors={
                        "out": actual,
                        "tickets": args[1],
                        "reused": args[2],
                        "reversed_tickets": args[3],
                    },
                )
                _check_local_fetch(args, actual)
                self.assertEqual("match_sync" in source, enabled)

    def test_missing_result_exchange_barrier_is_observable(self):
        import ast

        args = _local_fetch_args(128, "plain")
        source = _aggregation_codegen(_fragment_local_fetch_add, args, False)
        tree = ast.parse(source)
        removed = []

        class RemoveExchange(ast.NodeTransformer):
            def visit_For(self, node):
                self.generic_visit(node)
                for index, statement in enumerate(node.body[:-1]):
                    if (
                        isinstance(statement, ast.For)
                        and "fragment_atomic_previous" in ast.unparse(statement)
                        and ast.unparse(node.body[index + 1])
                        == "cute.arch.sync_threads()"
                    ):
                        removed.append(node.body.pop(index + 1))
                        break
                return node

        changed = ast.unparse(ast.fix_missing_locations(RemoveExchange().visit(tree)))
        self.assertEqual(len(removed), 1)
        with self.assertRaisesRegex(AssertionError, "shared.*before initialization"):
            _simulate_register_load_program(
                changed,
                args[0],
                128,
                host_tensors={
                    "tickets": args[1],
                    "reused": args[2],
                    "reversed_tickets": args[3],
                },
            )

    def test_local_return_rejects_float_ordering_aliases_and_early_reads(self):
        for dtype, mode in (
            (torch.float32, "result"),
            (torch.int32, "release"),
            (torch.int32, "alias"),
            (torch.int32, "early_read"),
            (torch.int32, "divergent"),
        ):
            with (
                self.subTest(dtype=dtype, mode=mode),
                self.assertRaises(helion.exc.InvalidConfig),
            ):
                _aggregation_codegen(
                    _local_atomic_histogram,
                    (torch.ones((2, 17), dtype=dtype), 17, 2, mode),
                    False,
                )

    @skipUnlessCuteAvailable("requires CuTe DSL")
    def test_sdk_shared_int32_return_is_used_and_has_cta_scope(self):
        import cutlass
        from cutlass._mlir import ir
        from cutlass._mlir.dialects import func
        import cutlass.cute as cute

        with ir.Context(), ir.Location.unknown():
            module = ir.Module.create()
            with ir.InsertionPoint(module.body):
                fn = func.FuncOp(
                    "local_fetch_add",
                    (
                        [ir.Type.parse("!llvm.ptr<3>"), cutlass.Int32.mlir_type],
                        [cutlass.Int32.mlir_type],
                    ),
                )
                block = fn.add_entry_block()
                with ir.InsertionPoint(block):
                    previous = cute.arch.atomic_add(
                        block.arguments[0],
                        cutlass.Int32(block.arguments[1]),
                        sem="relaxed",
                        scope="cta",
                    )
                    cute.arch.sync_threads()
                    func.ReturnOp([previous.ir_value()])
            self.assertTrue(module.operation.verify())
            atomic = next(
                op for op in block.operations if op.operation.name == "nvvm.atomicrmw"
            )
            self.assertEqual(str(atomic.memOrder), "#nvvm.mem_order<relaxed>")
            self.assertEqual(str(atomic.syncscope), "#nvvm.mem_scope<cta>")
            self.assertIn("!llvm.ptr<3>", str(module))
            self.assertIn("return", str(module))

    @skipUnlessCuteAvailable("requires CuTe DSL")
    def test_actual_generated_local_fetch_stages_through_sdk(self):
        import ast
        import importlib.util
        from pathlib import Path
        import tempfile

        import cutlass
        from cutlass._mlir import ir
        from cutlass._mlir.dialects import func
        import cutlass.cute as cute

        source = _aggregation_codegen(
            _fragment_local_fetch_add, _local_fetch_args(65, "masked"), False
        )
        tree = ast.parse(source)
        fn = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name.startswith("_helion_")
        )
        fn.name = "staged_local_fetch"
        fn.decorator_list = [ast.parse("cute.jit", mode="eval").body]
        tree.body = [
            node
            for node in tree.body
            if isinstance(node, (ast.Import, ast.ImportFrom, ast.Assign))
        ] + [fn]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "staged_local_fetch.py"
            path.write_text(ast.unparse(ast.fix_missing_locations(tree)))
            spec = importlib.util.spec_from_file_location("staged_local_fetch", path)
            sdk = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(sdk)
            with ir.Context(), ir.Location.unknown():
                emitted = ir.Module.create()
                with ir.InsertionPoint(emitted.body):
                    entry = func.FuncOp("entry", ([], []))
                    block = entry.add_entry_block()
                    with ir.InsertionPoint(block):
                        tensors = [
                            cute.make_tensor(
                                cute.make_ptr(
                                    cutlass.Float32
                                    if arg.arg == "x"
                                    else cutlass.Int32,
                                    0,
                                    cute.AddressSpace.gmem,
                                    assumed_align=16,
                                ),
                                cute.make_layout((260,)),
                            )
                            for arg in fn.args.args
                        ]
                        sdk.staged_local_fetch(*tensors)
                        func.ReturnOp([])
                self.assertTrue(emitted.operation.verify())
                text = str(emitted)
                self.assertIn("nvvm.atomicrmw", text)
                self.assertIn("#nvvm.mem_scope<cta>", text)
                self.assertIn("nvvm.barrier", text)
                self.assertNotIn("nvvm.match.sync", text)


@onlyBackends("cute")
class TestFragmentLocalFetchAddNative(TestCase):
    def test_scalar_arrival_and_full_cta_broadcast(self):
        for threads in (32, 128, 512):
            with self.subTest(threads=threads):
                x = torch.ones((2, 65), device=DEVICE)
                tickets = torch.full((2, 2, 65), -99, dtype=torch.int32, device=DEVICE)
                _, actual = code_and_output(
                    _fragment_local_scalar_fetch,
                    (x, tickets),
                    cute_fragment_threads=threads,
                )
                expected = torch.full_like(actual, 5)
                expected[:, -1] = 7
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(
                    tickets,
                    torch.tensor([5, 6], device=DEVICE)
                    .view(1, 2, 1)
                    .expand_as(tickets)
                    .int(),
                    rtol=0,
                    atol=0,
                )

    def test_local_tickets_masks_cross_lane_consumers_and_reuse(self):
        for width, threads in ((17, 32), (65, 128), (129, 128), (65, 512)):
            for mode in ("plain", "negative", "masked", "bounds", "epochs", "snapshot"):
                with self.subTest(width=width, threads=threads, mode=mode):
                    args = _local_fetch_args(width, mode, device=DEVICE)
                    before = args[0].clone()
                    _, actual = code_and_output(
                        _fragment_local_fetch_add, args, cute_fragment_threads=threads
                    )
                    _check_local_fetch(args, actual)
                    torch.testing.assert_close(args[0], before, rtol=0, atol=0)

    def test_local_integer_payload_zero_and_signed_wrap(self):
        for initial, value in (
            ((1 << 24) + 5, 0.75),
            ((1 << 24) + 5, -1.75),
            ((1 << 31) - 2, 1.75),
        ):
            with self.subTest(initial=initial, value=value):
                args = _local_fetch_args(65, "plain", initial, value, DEVICE)
                _, actual = code_and_output(_fragment_local_fetch_add, args)
                _check_local_fetch(args, actual)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _fragment_local_fetch_shapes(x, direct, reverse, shifted, vector: hl.constexpr):
    out = torch.empty((x.size(0), 2, 17), dtype=torch.int32, device=x.device)
    for row in hl.grid(x.size(0)):
        local = hl.full([17], 3, dtype=torch.int32)
        routed = hl.zeros([17], dtype=torch.int32)
        lane = hl.arange(x.size(1))
        if vector:
            previous = hl.atomic_add(local, [lane % 17], x[row, :])
        else:
            previous = hl.atomic_add(local, [lane % 17], 1)
        direct[row, :] = previous
        reverse[row, :] = torch.flip(previous, [0])
        shifted[row, :] = (
            previous.reshape([x.size(1), 1]) + lane.unsqueeze(1)
        ).reshape([x.size(1)])
        hl.atomic_add(routed, [previous % 17], 1)
        out[row, 0, :] = local
        out[row, 1, :] = routed
    return out


def _check_local_fetch_shapes(x, direct, reverse, shifted, actual):
    width = x.size(1)
    columns = torch.arange(width, device=x.device)
    expected = 3 + torch.bincount(columns % 17, minlength=17).int()
    torch.testing.assert_close(
        actual[:, 0], expected.expand_as(actual[:, 0]), rtol=0, atol=0
    )
    for bucket in range(17):
        tickets = direct[:, columns % 17 == bucket]
        expected_tickets = torch.arange(3, 3 + tickets.size(1), device=x.device).int()
        torch.testing.assert_close(
            tickets.sort(-1).values, expected_tickets.expand_as(tickets), rtol=0, atol=0
        )
    torch.testing.assert_close(reverse, direct.flip(-1), rtol=0, atol=0)
    torch.testing.assert_close(
        shifted, direct + columns, rtol=0, atol=0, check_dtype=False
    )
    for row in range(x.size(0)):
        torch.testing.assert_close(
            actual[row, 1],
            torch.bincount(direct[row].long() % 17, minlength=17).int(),
            rtol=0,
            atol=0,
        )


class TestFragmentLocalFetchShapeCPU(unittest.TestCase):
    def test_scalar_and_vector_returns_keep_logical_views_and_index_domains(self):
        for width, threads in ((17, 32), (65, 128), (129, 512)):
            for vector in (False, True):
                with self.subTest(width=width, threads=threads, vector=vector):
                    x = torch.ones((2, width))
                    direct = torch.full((2, width), -99, dtype=torch.int32)
                    reverse = torch.full_like(direct, -99)
                    shifted = torch.full_like(direct, -99)
                    source = _aggregation_codegen(
                        _fragment_local_fetch_shapes,
                        (x, direct, reverse, shifted, vector),
                        False,
                        threads,
                    )
                    for descending in (False, True):
                        actual = torch.empty((2, 2, 17), dtype=torch.int32)
                        events = []
                        _simulate_register_load_program(
                            source,
                            x,
                            threads,
                            host_tensors={
                                "direct": direct,
                                "reverse": reverse,
                                "shifted": shifted,
                                "out": actual,
                            },
                            lane_order=list(reversed(range(threads)))
                            if descending
                            else None,
                            atomic_events=events,
                        )
                        self.assertEqual(len(events), 4 * width)
                        _check_local_fetch_shapes(x, direct, reverse, shifted, actual)
                        torch.testing.assert_close(
                            x, torch.ones_like(x), rtol=0, atol=0
                        )


@onlyBackends("cute")
class TestFragmentLocalFetchShapeNative(TestCase):
    def test_mixed_global_and_unused_local_controls(self):
        for sem in ("relaxed", "acq_rel"):
            for unused in (False, True):
                with self.subTest(sem=sem, unused=unused):
                    x = torch.ones((2, 17), dtype=torch.int32, device=DEVICE)
                    counter = torch.zeros(2, dtype=torch.int32, device=DEVICE)
                    tickets = torch.full_like(x, -99)
                    _, actual = code_and_output(
                        _fragment_ordered_vector, (x, counter, tickets, sem, unused)
                    )
                    torch.testing.assert_close(actual, x, rtol=0, atol=0)
                    torch.testing.assert_close(
                        counter, torch.full_like(counter, 17), rtol=0, atol=0
                    )
                    expected = (
                        torch.full_like(tickets, -99)
                        if unused
                        else torch.arange(17, dtype=torch.int32, device=DEVICE)
                        .view(1, 17)
                        .expand_as(tickets)
                    )
                    torch.testing.assert_close(
                        tickets.sort(-1).values, expected, rtol=0, atol=0
                    )
                    torch.testing.assert_close(x, torch.ones_like(x), rtol=0, atol=0)

    def test_scalar_and_vector_returns_keep_logical_views_and_index_domains(self):
        for width, threads in ((17, 32), (65, 128), (129, 512)):
            for vector in (False, True):
                with self.subTest(width=width, threads=threads, vector=vector):
                    x = torch.ones((2, width), device=DEVICE)
                    direct = torch.full(
                        (2, width), -99, dtype=torch.int32, device=DEVICE
                    )
                    reverse = torch.full_like(direct, -99)
                    shifted = torch.full_like(direct, -99)
                    _, actual = code_and_output(
                        _fragment_local_fetch_shapes,
                        (x, direct, reverse, shifted, vector),
                        cute_fragment_threads=threads,
                    )
                    _check_local_fetch_shapes(x, direct, reverse, shifted, actual)
                    torch.testing.assert_close(x, torch.ones_like(x), rtol=0, atol=0)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _fragment_local_registers(x, tickets, reused, scalar: hl.constexpr):
    out = torch.empty((x.size(0), 17), dtype=torch.int32, device=x.device)
    for row in hl.grid(x.size(0)):
        local = hl.zeros([17], dtype=torch.int32)
        lane = hl.arange(helion.next_power_of_2(x.size(1)))
        index = torch.where(lane < x.size(1), lane % 19 - 1, 17)
        if scalar:
            value = 1
        else:
            value = hl.load(x, [row, lane])
        previous = hl.atomic_add(local, [index], value)
        hl.store(tickets, [row, lane], previous)
        hl.store(reused, [row, lane], previous * 3 + 2)
        out[row, :] = local
    return out


def _local_register_codegen(kernel, args, enabled, threads=128):
    from test._cute_binding import _cpu_bind
    from test._cute_binding import _forbid_native_compile
    from test._cute_binding import _mock_cuda_unavailable
    from test.cute_population_contracts import _target

    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound = _cpu_bind(kernel, args)
        config = bound.config_spec.default_config()
        config.config.update(
            cute_fragment_local_atomic_registers=enabled, cute_fragment_threads=threads
        )
        _, config = bound.config_spec.create_config_generation().strict_config_pair(
            config
        )
        return bound.to_code(config)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _fragment_local_register_escape(x, tickets, counter, mode: hl.constexpr):
    for row in hl.grid(x.size(0)):
        local = hl.zeros([17], dtype=torch.int32)
        lane = hl.arange(helion.next_power_of_2(x.size(1)))
        ticket = hl.atomic_add(local, [lane % 17], 1)
        if mode == "flip":
            result = torch.flip(ticket, [0])
        elif mode == "scan":
            result = hl.cumsum(ticket, 0)
        elif mode == "reduce":
            result = ticket + ticket.sum()
        elif mode == "carry":
            result = ticket
            for step in range(2):
                result = result + step
        else:
            hl.atomic_add(counter, [row + lane * 0], ticket)
            result = ticket
        hl.store(tickets, [row, lane], result)
    return tickets


class TestFragmentLocalRegistersCPU(unittest.TestCase):
    def test_tails_wrapped_indices_values_reuse_and_epoch_parity(self):
        for width, threads in (
            (17, 32),
            (65, 32),
            (129, 128),
            (1025, 512),
            (1023, 1024),
            (1025, 1024),
        ):
            for scalar in (False, True):
                with self.subTest(width=width, threads=threads, scalar=scalar):
                    x = ((torch.arange(2 * width).reshape(2, width) % 7) - 3).int()
                    ticket = torch.full_like(x, -99)
                    reused = torch.full_like(x, -99)
                    args = (x, ticket, reused, scalar)
                    sources = [
                        _local_register_codegen(
                            _fragment_local_registers, args, flag, threads
                        )
                        for flag in (False, True)
                    ]
                    self.assertNotIn("fragment_local_tickets", sources[0])
                    self.assertIn("cute.make_rmem_tensor", sources[1])
                    for order in (list(range(threads)), list(reversed(range(threads)))):
                        observed = []
                        for source in sources:
                            ticket.fill_(-99)
                            reused.fill_(-99)
                            events = []
                            output = torch.full((2, 17), -99, dtype=torch.int32)
                            actual, barriers = _simulate_register_load_program(
                                source,
                                x,
                                threads,
                                host_tensors={
                                    "tickets": ticket,
                                    "reused": reused,
                                    "out": output,
                                },
                                lane_order=order,
                                atomic_events=events,
                            )
                            observed.append(
                                (
                                    output.clone(),
                                    ticket.clone(),
                                    reused.clone(),
                                    barriers,
                                    events,
                                )
                            )
                        for a, b in zip(observed[0][:3], observed[1][:3], strict=True):
                            torch.testing.assert_close(a, b, rtol=0, atol=0)
                        self.assertEqual(observed[0][3:], observed[1][3:])
                        indices = torch.arange(width) % 19 - 1
                        indices = torch.where(indices < 0, indices + 17, indices)
                        active = indices < 17
                        expected = torch.zeros((2, 17), dtype=torch.int32)
                        expected.scatter_add_(
                            1,
                            indices[active].expand(2, -1),
                            torch.ones_like(x[:, active]) if scalar else x[:, active],
                        )
                        torch.testing.assert_close(
                            observed[1][0], expected, rtol=0, atol=0
                        )
                        torch.testing.assert_close(
                            observed[1][2], observed[1][1] * 3 + 2, rtol=0, atol=0
                        )

    def test_cross_lane_views_decline_and_slot_budget_is_explicit(self):
        from helion import exc

        x = torch.ones((2, 17))
        out = torch.zeros_like(x, dtype=torch.int32)
        with self.assertRaises(exc.InvalidConfig):
            _local_register_codegen(
                _fragment_local_fetch_shapes,
                (x, out, out.clone(), out.clone(), False),
                True,
                32,
            )
        x = torch.ones((2, 1025), dtype=torch.int32)
        out = torch.zeros_like(x)
        with self.assertRaises(exc.InvalidConfig):
            _local_register_codegen(
                _fragment_local_registers, (x, out, out.clone(), False), True, 32
            )
        source = _local_register_codegen(
            _fragment_local_registers, (x, out, out.clone(), False), True, 64
        )
        self.assertIn("cute.make_rmem_tensor", source)

    def test_masked_nan_and_fractional_contributions_stay_guarded(self):
        x = (torch.arange(130).reshape(2, 65).float() % 5) - 2.5
        x[:, torch.arange(65) % 19 == 18] = float("nan")
        tickets = torch.zeros_like(x, dtype=torch.int32)
        reused = torch.zeros_like(tickets)
        records = []
        for flag in (False, True):
            source = _local_register_codegen(
                _fragment_local_registers, (x, tickets, reused, False), flag, 32
            )
            events = []
            out, barriers = _simulate_register_load_program(
                source,
                x,
                32,
                host_tensors={"tickets": tickets, "reused": reused},
                atomic_events=events,
            )
            records.append(
                (out.clone(), tickets.clone(), reused.clone(), events, barriers)
            )
        for a, b in zip(records[0][:3], records[1][:3], strict=True):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertEqual(records[0][3:], records[1][3:])

    def test_default_and_complete_previous_coverage_prefix(self):
        from copy import deepcopy
        import random
        from unittest.mock import patch

        from test._cute_binding import _cpu_bind
        from test._cute_binding import _forbid_native_compile
        from test._cute_binding import _mock_cuda_unavailable
        from test.cute_population_contracts import _target
        from test.cute_population_contracts import checked_initial_population

        from helion.autotuner.pattern_search import InitialPopulationStrategy
        from helion.autotuner.pattern_search import PatternSearch

        key = "cute_fragment_local_atomic_registers"
        x = torch.ones((2, 65), dtype=torch.int32)
        args = (x, x.clone(), x.clone(), False)
        kernel = helion.kernel(
            _fragment_local_registers.fn,
            backend="cute",
            static_shapes=True,
            autotune_effort="full",
        )
        with (
            _mock_cuda_unavailable(),
            _target(),
            _forbid_native_compile(),
            patch(
                "helion._compiler.autotuner_heuristics.register_fragment_published_scalars_coverage"
            ),
            patch(
                "helion._compiler.autotuner_heuristics.register_fragment_skip_zero_atomics_coverage"
            ),
            patch(
                "helion._compiler.autotuner_heuristics.register_fragment_register_snapshots_coverage"
            ),
        ):
            bound = _cpu_bind(kernel, args)
            spec = bound.config_spec
            default = spec.default_config()
            self.assertNotIn(key, default)
            self.assertTrue(all(key not in seed for seed in spec.compiler_seed_configs))
            off = deepcopy(default)
            off.config[key] = False
            self.assertEqual(bound.to_code(default), bound.to_code(off))
            for invalid in (1, "registers", None):
                bad = deepcopy(default)
                bad.config[key] = invalid
                with self.assertRaises(helion.exc.InvalidConfig):
                    bound.to_code(bad)
            groups = spec.compiler_coverage_groups
            self.assertEqual(groups[-1].key, key)
            self.assertTrue(groups[-1].deferred)
            self.assertFalse(groups[-1].legacy)

            def population():
                with bound.env:
                    search = PatternSearch(
                        bound,
                        args,
                        initial_population=8,
                        initial_population_strategy=InitialPopulationStrategy.FROM_RANDOM,
                    )
                    rows = checked_initial_population(search)
                    return [
                        dict(search.config_gen.canonicalize_flat(row)[1])
                        for row in rows
                    ], random.getstate()

            random.seed(20261003)
            new, state = population()
            with patch(
                "helion._compiler.autotuner_heuristics.register_fragment_local_atomic_registers_coverage"
            ):
                bound = _cpu_bind(kernel, args)
            random.seed(20261003)
            old, old_state = population()
            self.assertEqual(new[: len(old)], old)
            self.assertEqual(state, old_state)
            self.assertTrue(any(row.get(key) is True for row in new))

    def test_nonlocal_consumers_keep_the_existing_shared_path(self):
        x = torch.ones((2, 65), dtype=torch.int32)
        for mode in ("flip", "scan", "reduce", "carry", "global"):
            with self.subTest(mode=mode):
                args = (x, torch.zeros_like(x), torch.zeros(2, dtype=torch.int32), mode)
                baseline = _local_register_codegen(
                    _fragment_local_register_escape, args, False
                )
                self.assertNotIn("fragment_local_tickets", baseline)
                with self.assertRaises(helion.exc.InvalidConfig):
                    _local_register_codegen(_fragment_local_register_escape, args, True)

    @skipUnlessCuteAvailable("requires CuTe DSL")
    def test_actual_sdk_compiles_register_slot_producer_and_consumers(self):
        import ast
        import importlib.util
        from pathlib import Path
        import tempfile

        import cutlass
        from cutlass._mlir import ir
        from cutlass._mlir.dialects import func
        import cutlass.cute as cute

        for width, threads in ((129, 32), (1025, 1024)):
            with self.subTest(width=width, threads=threads):
                x = torch.ones((2, width), dtype=torch.int32)
                code = _local_register_codegen(
                    _fragment_local_registers,
                    (x, x.clone(), x.clone(), False),
                    True,
                    threads,
                )
                tree = ast.parse(code)
                fn = next(
                    n
                    for n in tree.body
                    if isinstance(n, ast.FunctionDef) and n.name.startswith("_helion_")
                )
                fn.name = "staged"
                fn.decorator_list = [ast.parse("cute.jit", mode="eval").body]
                tree.body = [
                    n
                    for n in tree.body
                    if isinstance(n, (ast.Import, ast.ImportFrom, ast.Assign))
                ] + [fn]
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "register_slots.py"
                    path.write_text(ast.unparse(ast.fix_missing_locations(tree)))
                    spec = importlib.util.spec_from_file_location(
                        "register_slots", path
                    )
                    sdk = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(sdk)
                    with ir.Context(), ir.Location.unknown():
                        module = ir.Module.create()
                        with ir.InsertionPoint(module.body):
                            entry = func.FuncOp("entry", ([], []))
                            with ir.InsertionPoint(entry.add_entry_block()):
                                tensors = [
                                    cute.make_tensor(
                                        cute.make_ptr(
                                            cutlass.Int32,
                                            0,
                                            cute.AddressSpace.gmem,
                                            assumed_align=16,
                                        ),
                                        cute.make_layout((2 * width,)),
                                    )
                                    for _arg in fn.args.args
                                ]
                                sdk.staged(*tensors)
                                func.ReturnOp([])
                        self.assertTrue(module.operation.verify())
                        self.assertIn("nvvm.atomicrmw", str(module))
                        self.assertIn("#nvvm.mem_scope<cta>", str(module))
                        self.assertIn("nvvm.barrier", str(module))

    def test_1024_threads_shared_float_subnormals_and_capacity_guard(self):
        from unittest.mock import patch

        from test._cute_binding import _cpu_bind
        from test._cute_binding import _forbid_native_compile
        from test._cute_binding import _mock_cuda_unavailable
        from test.cute_population_contracts import _target

        tiny = torch.finfo(torch.float32).tiny
        x = torch.full((2, 2), tiny / 2)
        source = _fragment_ordered_codegen(_local_atomic_subnormal, (x,), 1024)
        self.assertIn("block=(1024, 1, 1)", source)
        for order in (list(range(1024)), list(reversed(range(1024)))):
            actual = torch.full((2, 1), -999.0)
            _simulate_register_load_program(
                source, x, 1024, lane_order=order, host_tensors={"out": actual}
            )
            torch.testing.assert_close(actual, torch.full((2, 1), tiny), rtol=0, atol=0)
        with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
            bound = _cpu_bind(
                _local_atomic_histogram,
                (torch.ones((1, 17)), 17, 2, "plain"),
            )
            config = bound.config_spec.default_config()
            config.config["cute_fragment_threads"] = 1024
            with (
                patch(
                    "helion._compiler.cute.tcgen05_config.CuteTcgen05Config.per_cta_smem_capacity_bytes",
                    return_value=16,
                ),
                self.assertRaisesRegex(helion.exc.InvalidConfig, "shared bytes"),
            ):
                bound.to_code(config)


@onlyBackends("cute")
class TestFragmentLocalRegistersNative(TestCase):
    def test_register_ticket_slots_tails_and_repeated_consumers(self):
        for width, threads in ((17, 32), (129, 128), (8192, 512), (1025, 1024)):
            for scalar in (False, True):
                with self.subTest(width=width, threads=threads, scalar=scalar):
                    x = torch.ones((2, width), dtype=torch.int32, device=DEVICE)
                    tickets = torch.full_like(x, -99)
                    reused = torch.full_like(x, -99)
                    source, out = code_and_output(
                        _fragment_local_registers,
                        (x, tickets, reused, scalar),
                        cute_fragment_local_atomic_registers=True,
                        cute_fragment_threads=threads,
                    )
                    self.assertIn("fragment_local_tickets", source)
                    indices = torch.arange(width, device=DEVICE) % 19 - 1
                    indices = torch.where(indices < 0, indices + 17, indices)
                    expected = torch.zeros((2, 17), dtype=torch.int32, device=DEVICE)
                    for bucket in range(17):
                        active = indices == bucket
                        count = int(active.sum())
                        expected[:, bucket] = count
                        for row in range(2):
                            torch.testing.assert_close(
                                tickets[row, active].sort().values,
                                torch.arange(count, dtype=torch.int32, device=DEVICE),
                                rtol=0,
                                atol=0,
                            )
                    self.assertTrue(torch.all(tickets[:, indices >= 17] == 0))
                    torch.testing.assert_close(out, expected, rtol=0, atol=0)
                    torch.testing.assert_close(reused, tickets * 3 + 2, rtol=0, atol=0)
                    torch.testing.assert_close(x, torch.ones_like(x), rtol=0, atol=0)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _fragment_local_buffer_capacity(
    x, out, totals, capacity: hl.constexpr, mode: hl.constexpr
):
    for row in hl.grid(x.size(0)):
        count = hl.zeros([1], dtype=torch.int32)
        histogram = hl.zeros([17], dtype=torch.int32)
        compacted = hl.zeros([capacity], dtype=x.dtype)
        column = hl.arange(helion.next_power_of_2(x.size(1)))
        value = hl.load(x, [row, column], extra_mask=column < x.size(1))
        active = (column < x.size(1)) & (value > 0)
        ticket = hl.atomic_add(
            count, [torch.zeros_like(column)], active.to(torch.int32)
        )
        hl.atomic_add(histogram, [column % 17], active.to(torch.int32))
        safe_ticket = torch.where(
            ticket < 0, 0, torch.where(ticket >= capacity, capacity - 1, ticket)
        )
        hl.atomic_add(
            compacted,
            [safe_ticket],
            torch.where(active & (ticket < capacity), value, 0),
        )
        if mode == "scalar":
            total = count.sum()
        else:
            total = histogram.sum()
        totals[row] = total
        if total <= capacity:
            if mode == "pressure":
                prefix = torch.cumsum(compacted, 0)
                out[row, :] = prefix + prefix.sum().to(torch.int32)
            else:
                out[row, :] = compacted
        else:
            # A failed capacity attempt must overwrite the entire output. This
            # fixture deliberately starts with nonzero/stale output contents.
            out[row, :] = -1
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _fragment_local_buffer_rejected(x, out, mode: hl.constexpr):
    for row in hl.grid(x.size(0)):
        local = hl.zeros([17], dtype=torch.int32)
        lane = hl.arange(x.size(1))
        hl.atomic_add(local, [lane % 17], x[row, :])
        total = local.sum()
        if mode == "view":
            captured = torch.flip(local, [0])
        elif mode == "lazy":
            captured = local + 1
        else:
            captured = local
        if mode == "vector":
            condition = local > 0
        elif mode == "singleton":
            condition = local.sum(0, keepdim=True) > 0
        elif mode == "loaded":
            condition = (total > 0) & (x[row, 0] > 0)
        else:
            condition = total > 0
        if mode == "early":
            hl.atomic_add(local, [lane % 17], x[row, :])
        if condition:
            if mode == "atomic":
                hl.atomic_add(local, [lane % 17], x[row, :])
            elif mode == "loop":
                for iteration in range(2):
                    out[row, :] = local + iteration
            elif mode == "nested":
                if total > 1:
                    out[row, :] = local
            elif mode == "phi":
                captured = local * 2
            else:
                out[row, :] = captured
        else:
            out[row, :] = -1
        if mode == "later":
            hl.atomic_add(local, [lane % 17], x[row, :])
        elif mode == "phi":
            out[row, :] = captured
        elif mode == "nonterminal":
            out[row, :] = local
    return out


def _local_buffer_capacity_args(width, capacity, mode, device="cpu", dtype=torch.int32):
    # Empty, exact-capacity and overflowing populations share one compiled CTA
    # program. A zero at every inactive coordinate checks initialized padding.
    x = torch.zeros((4, width), dtype=dtype, device=device)
    for row, count in enumerate(
        (0, min(width, capacity), min(width, capacity + 1), width)
    ):
        x[row, :count] = torch.arange(1, count + 1, dtype=torch.int32, device=device)
    return (
        x,
        torch.full((4, capacity), -777, dtype=dtype, device=device),
        torch.full((4,), -999, dtype=torch.int32, device=device),
        capacity,
        mode,
    )


def _check_local_buffer_capacity(args):
    x, out, totals, capacity, mode = args
    expected_count = (x > 0).sum(-1).to(torch.int32)
    torch.testing.assert_close(totals, expected_count, rtol=0, atol=0)
    for row in range(x.size(0)):
        if expected_count[row] > capacity:
            torch.testing.assert_close(
                out[row], torch.full_like(out[row], -1), rtol=0, atol=0
            )
        elif mode == "pressure":
            # Prefix order can follow either valid atomic schedule. Recover the
            # compacted values; its multiset is the independent invariant.
            count = int(expected_count[row])
            prefix = out[row] - out[row].sum().div(capacity + 1, rounding_mode="trunc")
            recovered = torch.diff(prefix, prepend=prefix.new_zeros(1))
            expected = torch.zeros_like(recovered)
            expected[:count] = x[row][x[row] > 0]
            torch.testing.assert_close(
                recovered.sort().values, expected.sort().values, rtol=0, atol=0
            )
        else:
            expected = torch.zeros_like(out[row])
            selected = x[row][x[row] > 0]
            expected[: selected.numel()] = selected
            torch.testing.assert_close(
                out[row].sort().values, expected.sort().values, rtol=0, atol=0
            )


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _fragment_local_buffer_extrema(x, out, limit: int, minimum: hl.constexpr):
    for row in hl.grid(x.size(0)):
        local = hl.zeros([17], dtype=torch.int32)
        hl.atomic_add(local, [hl.arange(17)], x[row, :])
        if minimum:
            extreme = torch.amin(local)
        else:
            extreme = torch.amax(local)
        if extreme >= limit:
            out[row, :] = local * 2
        else:
            out[row, :] = local - 1
    return out


def _local_buffer_extrema_args(minimum, device="cpu"):
    x = torch.arange(3 * 17, dtype=torch.int32, device=device).reshape(3, 17) - 17
    return x, torch.full_like(x, -999), 0, minimum


def _check_local_buffer_extrema(args):
    x, out, limit, minimum = args
    extreme = x.amin(1) if minimum else x.amax(1)
    expected = torch.where((extreme >= limit)[:, None], x * 2, x - 1)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


class TestFragmentLocalBufferConditionalsCPU(unittest.TestCase):
    def test_float_buffer_and_extrema_uniform_host_bound(self):
        args = _local_buffer_capacity_args(65, 17, "pressure", dtype=torch.float32)
        source = _fragment_ordered_codegen(_fragment_local_buffer_capacity, args)
        _simulate_register_load_program(
            source, args[0], 128, host_tensors={"out": args[1], "totals": args[2]}
        )
        _check_local_buffer_capacity(args)
        for minimum in (False, True):
            args = _local_buffer_extrema_args(minimum)
            source = _fragment_ordered_codegen(_fragment_local_buffer_extrema, args)
            for limit in (-19, 0, 18, 100):
                with self.subTest(minimum=minimum, limit=limit):
                    values = (*args[:2], limit, minimum)
                    _simulate_register_load_program(
                        source,
                        args[0],
                        128,
                        host_tensors={"out": args[1]},
                        scalar_args={"limit": limit},
                    )
                    _check_local_buffer_extrema(values)

    def test_compaction_capacity_tails_and_complete_fallback(self):
        for width, capacity, threads in ((17, 8, 32), (65, 17, 128), (129, 65, 512)):
            for mode in ("scalar", "histogram", "pressure"):
                with self.subTest(
                    width=width, capacity=capacity, threads=threads, mode=mode
                ):
                    args = _local_buffer_capacity_args(width, capacity, mode)
                    source = _fragment_ordered_codegen(
                        _fragment_local_buffer_capacity, args, threads
                    )
                    for order in (list(range(threads)), list(reversed(range(threads)))):
                        args[1].fill_(314159)
                        events = []
                        before = args[0].clone()
                        _simulate_register_load_program(
                            source,
                            args[0],
                            threads,
                            host_tensors={"out": args[1], "totals": args[2]},
                            lane_order=order,
                            memory_events=events,
                        )
                        _check_local_buffer_capacity(args)
                        torch.testing.assert_close(args[0], before, rtol=0, atol=0)
                        self.assertEqual(
                            sum(event[0] == "atomic" for event in events),
                            4 * 3 * helion.next_power_of_2(width),
                        )
                    # Same generated program takes the other branch per row.
                    args[0].copy_(args[0].flip(0))
                    _simulate_register_load_program(
                        source,
                        args[0],
                        threads,
                        host_tensors={"out": args[1], "totals": args[2]},
                    )
                    _check_local_buffer_capacity(args)

    def test_predicate_and_local_captures_stay_live_during_branch_allocation(self):
        from unittest.mock import patch

        from helion._compiler.cute.computed_fragment import FragmentCompiler

        original = FragmentCompiler.allocate
        checks = []

        def allocate(compiler, value):
            held = compiler.referenced_buffers(compiler.held)
            result = original(compiler, value)
            if held:
                self.assertNotIn(result.storage, held)
                checks.append(result.storage)
            return result

        args = _local_buffer_capacity_args(65, 17, "pressure")
        with patch.object(FragmentCompiler, "allocate", allocate):
            source = _fragment_ordered_codegen(_fragment_local_buffer_capacity, args)
        self.assertTrue(checks)
        _simulate_register_load_program(
            source, args[0], 128, host_tensors={"out": args[1], "totals": args[2]}
        )
        _check_local_buffer_capacity(args)

    def test_unsupported_predicates_aliases_mutation_and_reconvergence_reject(self):
        for mode in (
            "vector",
            "singleton",
            "loaded",
            "view",
            "lazy",
            "atomic",
            "loop",
            "nested",
            "early",
            "later",
            "phi",
            "nonterminal",
        ):
            with self.subTest(mode=mode), self.assertRaises(helion.exc.InvalidConfig):
                _fragment_ordered_codegen(
                    _fragment_local_buffer_rejected,
                    (
                        torch.ones((2, 17), dtype=torch.int32),
                        torch.empty((2, 17), dtype=torch.int32),
                        mode,
                    ),
                )

    def test_warp_private_reduction_cannot_control_cta_branch(self):
        from unittest.mock import patch

        from helion._compiler.cute.computed_fragment import FragmentCompiler
        from helion._compiler.cute.local_atomic import local_buffer_conditional_inputs

        original = FragmentCompiler.conditional

        def conditional(compiler, node, values):
            reductions, _symbols = local_buffer_conditional_inputs(
                node, compiler.graphs
            )
            compiler.warp_result_nodes.update(dict.fromkeys(reductions, 1))
            return original(compiler, node, values)

        with (
            patch.object(FragmentCompiler, "conditional", conditional),
            self.assertRaisesRegex(helion.exc.InvalidConfig, "CTA-shared"),
        ):
            _fragment_ordered_codegen(
                _fragment_local_buffer_capacity,
                _local_buffer_capacity_args(17, 8, "histogram"),
            )

    @skipUnlessCuteAvailable("requires CuTe DSL")
    def test_actual_sdk_stages_shared_predicate_and_both_collective_branches(self):
        import ast
        import importlib.util
        from pathlib import Path
        import tempfile

        import cutlass
        from cutlass._mlir import ir
        from cutlass._mlir.dialects import func
        import cutlass.cute as cute

        for mode in ("scalar", "pressure"):
            source = _fragment_ordered_codegen(
                _fragment_local_buffer_capacity,
                _local_buffer_capacity_args(65, 17, mode),
            )
            tree = ast.parse(source)
            fn = next(
                n
                for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name.startswith("_helion_")
            )
            fn.name = "staged_local_branch"
            fn.decorator_list = [ast.parse("cute.jit", mode="eval").body]
            tree.body = [
                n
                for n in tree.body
                if isinstance(n, (ast.Import, ast.ImportFrom, ast.Assign))
            ] + [fn]
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "staged.py"
                path.write_text(ast.unparse(ast.fix_missing_locations(tree)))
                spec = importlib.util.spec_from_file_location(
                    "staged_local_branch", path
                )
                sdk = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(sdk)
                with ir.Context(), ir.Location.unknown():
                    emitted = ir.Module.create()
                    with ir.InsertionPoint(emitted.body):
                        entry = func.FuncOp("entry", ([], []))
                        block = entry.add_entry_block()
                        with ir.InsertionPoint(block):
                            tensors = [
                                cute.make_tensor(
                                    cute.make_ptr(
                                        cutlass.Int32,
                                        0,
                                        cute.AddressSpace.gmem,
                                        assumed_align=16,
                                    ),
                                    cute.make_layout((260,)),
                                )
                                for _arg in fn.args.args
                            ]
                            sdk.staged_local_branch(*tensors)
                            func.ReturnOp([])
                    self.assertTrue(emitted.operation.verify())
                    text = str(emitted)
                    self.assertIn("scf.if", text)
                    self.assertIn("nvvm.barrier", text)
                    self.assertIn("nvvm.atomicrmw", text)


@onlyBackends("cute")
class TestFragmentLocalBufferConditionalsNative(TestCase):
    def test_float_buffer_and_uniform_extrema(self):
        args = _local_buffer_capacity_args(65, 17, "pressure", DEVICE, torch.float32)
        code_and_output(_fragment_local_buffer_capacity, args)
        _check_local_buffer_capacity(args)
        for minimum in (False, True):
            with self.subTest(minimum=minimum):
                args = _local_buffer_extrema_args(minimum, DEVICE)
                code_and_output(_fragment_local_buffer_extrema, args)
                _check_local_buffer_extrema(args)

    def test_complete_compaction_and_overflow_branches(self):
        for width, capacity, threads in ((17, 8, 32), (65, 17, 128), (129, 65, 512)):
            for mode in ("scalar", "histogram", "pressure"):
                with self.subTest(
                    width=width, capacity=capacity, threads=threads, mode=mode
                ):
                    args = _local_buffer_capacity_args(width, capacity, mode, DEVICE)
                    before = args[0].clone()
                    code_and_output(
                        _fragment_local_buffer_capacity,
                        args,
                        cute_fragment_threads=threads,
                    )
                    _check_local_buffer_capacity(args)
                    torch.testing.assert_close(args[0], before, rtol=0, atol=0)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _fragment_branch_refinement(
    x, lengths, out, counts, capacity: hl.constexpr, k: hl.constexpr
):
    for row in hl.grid(x.size(0)):
        count = hl.zeros([1], torch.int32)
        compact_keys = hl.zeros([capacity], torch.int32)
        compact_ids = hl.zeros([capacity], torch.int32)
        col = hl.arange(helion.next_power_of_2(x.size(1)))
        valid = (col < x.size(1)) & (col < lengths[row])
        value = hl.load(x, [row, col], extra_mask=valid)
        active = valid & (value >= 0)
        ticket = hl.atomic_add(count, [torch.zeros_like(col)], active.to(torch.int32))
        address = torch.where(ticket < capacity, ticket, capacity - 1)
        hl.atomic_add(
            compact_keys,
            [address],
            torch.where(active & (ticket < capacity), value + 8, 0),
        )
        hl.atomic_add(
            compact_ids,
            [address],
            torch.where(active & (ticket < capacity), col + 1, 0),
        )
        population = count.sum()
        counts[row] = population
        out[row, :] = -1
        if (population >= k) & (population <= capacity):
            small_hist = hl.zeros([16], torch.int32)
            small_col = hl.arange(helion.next_power_of_2(capacity))
            small_valid = (small_col < population) & (small_col < capacity)
            hl.atomic_add(small_hist, [compact_keys], small_valid.to(torch.int32))
            small_cdf = torch.cumsum(small_hist, 0)
            small_cut = (small_cdf < population - k + 1).to(torch.int32).sum()
            small_above = small_valid & (compact_keys > small_cut)
            small_tie = small_valid & (compact_keys == small_cut)
            small_need = k - small_above.to(torch.int32).sum()
            small_slots = hl.zeros([2], torch.int32)
            small_ticket = hl.atomic_add(
                small_slots,
                [small_tie.to(torch.int32)],
                (small_above | small_tie).to(torch.int32),
            )
            small_dest = torch.where(
                small_above, small_ticket, k - small_need + small_ticket
            )
            hl.store(
                out,
                [row, small_dest],
                compact_ids - 1,
                extra_mask=small_above | (small_tie & (small_ticket < small_need)),
            )
        else:
            full_hist = hl.zeros([16], torch.int32)
            full_col = hl.arange(helion.next_power_of_2(x.size(1)))
            full_valid = (full_col < x.size(1)) & (full_col < lengths[row])
            full_value = hl.load(x, [row, full_col], extra_mask=full_valid)
            full_key = full_value + 8
            hl.atomic_add(full_hist, [full_key], full_valid.to(torch.int32))
            full_cdf = torch.cumsum(full_hist, 0)
            full_count = full_valid.to(torch.int32).sum()
            full_cut = (full_cdf < full_count - k + 1).to(torch.int32).sum()
            full_above = full_valid & (full_key > full_cut)
            full_tie = full_valid & (full_key == full_cut)
            full_need = k - full_above.to(torch.int32).sum()
            full_slots = hl.zeros([2], torch.int32)
            full_ticket = hl.atomic_add(
                full_slots,
                [full_tie.to(torch.int32)],
                (full_above | full_tie).to(torch.int32),
            )
            full_dest = torch.where(
                full_above, full_ticket, k - full_need + full_ticket
            )
            hl.store(
                out,
                [row, full_dest],
                full_col,
                extra_mask=full_above | (full_tie & (full_ticket < full_need)),
            )
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _fragment_branch_atomic_epochs(x, out):
    for row in hl.grid(x.size(0)):
        parent = hl.zeros([17], dtype=torch.int32)
        lane = hl.arange(x.size(1))
        hl.atomic_add(parent, [lane % 17], x[row, :])
        total = parent.sum()
        if total > 0:
            first = hl.zeros([17], dtype=torch.int32)
            second = hl.full([33], 3, dtype=torch.float32)
            hl.atomic_add(first, [lane % 17 - 17], x[row, :])
            out[row, :] = first + parent
            # These final unused epochs must complete in the selected arm.
            hl.atomic_add(second, [lane % 33], x[row, :])
        else:
            other = hl.zeros([17], dtype=torch.int32)
            spare = hl.full([65], 5, dtype=torch.float32)
            hl.atomic_add(other, [lane % 17], x[row, :])
            out[row, :] = other + parent
            hl.atomic_add(spare, [lane % 65], x[row, :])
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _fragment_branch_captured(x, out):
    for row in hl.grid(x.size(0)):
        parent = hl.zeros([17], dtype=torch.int32)
        lane = hl.arange(x.size(1))
        hl.atomic_add(parent, [lane % 17], x[row, :])
        total = parent.sum()
        if total > 0:
            fresh = hl.zeros([17], dtype=torch.int32)
            hl.atomic_add(fresh, [lane % 17], x[row, :])
            hl.atomic_add(parent, [lane % 17], x[row, :])
            out[row, :] = fresh
        else:
            out[row, :] = parent
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _fragment_branch_nested(x, out):
    for row in hl.grid(x.size(0)):
        parent = hl.zeros([17], dtype=torch.int32)
        lane = hl.arange(x.size(1))
        hl.atomic_add(parent, [lane % 17], x[row, :])
        total = parent.sum()
        if total > 0:
            fresh = hl.zeros([17], dtype=torch.int32)
            if total > 1:
                hl.atomic_add(fresh, [lane % 17], x[row, :])
            out[row, :] = fresh
        else:
            out[row, :] = parent
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _fragment_branch_loop(x, out):
    for row in hl.grid(x.size(0)):
        parent = hl.zeros([17], dtype=torch.int32)
        lane = hl.arange(x.size(1))
        hl.atomic_add(parent, [lane % 17], x[row, :])
        total = parent.sum()
        if total > 0:
            fresh = hl.zeros([17], dtype=torch.int32)
            for iteration in range(2):
                hl.atomic_add(fresh, [lane % 17], x[row, :] + iteration)
            out[row, :] = fresh
        else:
            out[row, :] = parent
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _fragment_branch_alias(x, out):
    for row in hl.grid(x.size(0)):
        parent = hl.zeros([17], dtype=torch.int32)
        lane = hl.arange(x.size(1))
        hl.atomic_add(parent, [lane % 17], x[row, :])
        total = parent.sum()
        if total > 0:
            fresh = hl.zeros([17], dtype=torch.int32)
            viewed = fresh[:, None]
            hl.atomic_add(viewed, [lane % 17, 0], x[row, :])
            out[row, :] = fresh
        else:
            out[row, :] = parent
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _fragment_branch_early(x, out):
    for row in hl.grid(x.size(0)):
        parent = hl.zeros([17], dtype=torch.int32)
        lane = hl.arange(x.size(1))
        hl.atomic_add(parent, [lane % 17], x[row, :])
        total = parent.sum()
        if total > 0:
            fresh = hl.zeros([17], dtype=torch.int32)
            hl.atomic_add(fresh, [lane % 17], x[row, :])
            early = fresh + 1
            hl.atomic_add(fresh, [lane % 17], x[row, :])
            out[row, :] = early
        else:
            out[row, :] = parent
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _fragment_branch_escape(x, out):
    for row in hl.grid(x.size(0)):
        parent = hl.zeros([17], dtype=torch.int32)
        lane = hl.arange(x.size(1))
        hl.atomic_add(parent, [lane % 17], x[row, :])
        carried = parent
        if parent.sum() > 0:
            fresh = hl.zeros([17], dtype=torch.int32)
            hl.atomic_add(fresh, [lane % 17], x[row, :])
            carried = fresh
        out[row, :] = carried
    return out


def _branch_refinement_args(width, capacity, device="cpu"):
    k = 4
    x = (torch.arange(8 * width, device=device).reshape(8, width) % 8 - 8).to(
        torch.int32
    )
    for row, population in (
        (1, k),
        (2, capacity),
        (3, capacity + 1),
        (4, width),
        (5, k),
    ):
        x[row, :population] = (
            torch.arange(population, dtype=torch.int32, device=device) % 8
        )
    x[4].fill_(7)
    lengths = torch.tensor(
        [width] * 5 + [width - 3, 0, 2], dtype=torch.int32, device=device
    )
    x[7, :2] = torch.tensor([-8, 7], dtype=torch.int32, device=device)
    for row in range(8):
        x[row, lengths[row] :] = 999
    return (
        x,
        lengths,
        torch.full((8, k), -777, dtype=torch.int32, device=device),
        torch.full((8,), -99, dtype=torch.int32, device=device),
        capacity,
        k,
    )


def _check_branch_refinement(args):
    x, lengths, out, counts, _capacity, k = args
    for row in range(x.size(0)):
        length = int(lengths[row])
        self_count = int((x[row, :length] >= 0).sum())
        assert int(counts[row]) == self_count
        selected = out[row][out[row] >= 0].long()
        assert selected.numel() == min(k, length)
        assert selected.unique().numel() == selected.numel()
        assert bool(torch.all(selected < length))
        assert int((out[row] == -1).sum()) == k - selected.numel()
        reference = x[row, :length].sort(descending=True).values[:k]
        torch.testing.assert_close(
            x[row, selected].sort(descending=True).values, reference, rtol=0, atol=0
        )


class TestFragmentBranchLocalAtomicsCPU(unittest.TestCase):
    def test_exact_compact_refinement_and_full_row_fallback(self):
        for width, capacity, threads in ((17, 8, 32), (65, 17, 128), (129, 33, 512)):
            for scan in ("serial", "cooperative"):
                with self.subTest(
                    width=width, capacity=capacity, threads=threads, scan=scan
                ):
                    args = _branch_refinement_args(width, capacity)
                    source = _local_histogram_consumer_codegen(
                        _fragment_branch_refinement, args, scan, threads
                    )
                    for reverse in (False, True):
                        args[2].fill_(-777)
                        before = [value.clone() for value in args[:2]]
                        _simulate_register_load_program(
                            source,
                            args[0],
                            threads,
                            host_tensors={
                                "lengths": args[1],
                                "out": args[2],
                                "counts": args[3],
                            },
                            lane_order=list(reversed(range(threads)))
                            if reverse
                            else None,
                        )
                        _check_branch_refinement(args)
                        for actual, expected in zip(args[:2], before, strict=True):
                            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    args[0].copy_(args[0].flip(0))
                    args[1].copy_(args[1].flip(0))
                    _simulate_register_load_program(
                        source,
                        args[0],
                        threads,
                        host_tensors={
                            "lengths": args[1],
                            "out": args[2],
                            "counts": args[3],
                        },
                    )
                    _check_branch_refinement(args)

    def test_unused_epochs_finish_inside_each_arm_and_join(self):
        import ast
        from unittest.mock import patch

        from helion._compiler.cute.computed_fragment import FragmentCompiler
        from helion._compiler.device_ir import ElseGraphInfo
        from helion._compiler.device_ir import IfGraphInfo

        original = FragmentCompiler.graph
        entries, exits = [], []

        def graph(compiler, graph, values):
            info = next(info for info in compiler.graphs if info.graph is graph)
            branch = isinstance(info, (IfGraphInfo, ElseGraphInfo))
            if branch:
                entries.append(set(compiler.pending_local_atomics))
                self.assertFalse(compiler.pending_local_atomics)
            result = original(compiler, graph, values)
            if branch:
                exits.append(set(compiler.pending_local_atomics))
            return result

        x = torch.stack((torch.ones(65), -torch.ones(65), torch.zeros(65))).to(
            torch.int32
        )
        out = torch.full((3, 17), -99, dtype=torch.int32)
        with patch.object(FragmentCompiler, "graph", graph):
            source = _fragment_ordered_codegen(
                _fragment_branch_atomic_epochs, (x, out), 128
            )
        self.assertEqual(len(entries), 2)
        self.assertTrue(all(exits))
        tree = ast.parse(source)
        fn = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name.startswith("_helion_")
        )
        branch = next(
            n
            for n in fn.body
            if isinstance(n, ast.If) and isinstance(n.test, ast.Subscript)
        )
        self.assertEqual(ast.unparse(branch.body[-1]), "cute.arch.sync_threads()")
        self.assertEqual(ast.unparse(branch.orelse[-1]), "cute.arch.sync_threads()")
        self.assertEqual(
            ast.unparse(fn.body[fn.body.index(branch) + 1]), "cute.arch.sync_threads()"
        )
        for order in (list(range(128)), list(reversed(range(128)))):
            events = []
            _simulate_register_load_program(
                source,
                x,
                128,
                host_tensors={"out": out},
                lane_order=order,
                memory_events=events,
            )
            expected = torch.zeros((3, 17), dtype=torch.int32)
            expected.scatter_add_(1, (torch.arange(65) % 17).expand(3, -1), x)
            torch.testing.assert_close(out, expected * 2, rtol=0, atol=0)
            self.assertTrue(
                all(event[-1] == "cta" for event in events if event[0] == "atomic")
            )

    def test_captured_mutation_nested_control_alias_and_escape_reject(self):
        for kernel, reason in (
            (_fragment_branch_captured, "terminal uniform"),
            (_fragment_branch_nested, "terminal uniform"),
            (_fragment_branch_loop, "terminal uniform"),
            (_fragment_branch_alias, "terminal uniform"),
            (_fragment_branch_early, "initialization, then updates, then final reads"),
        ):
            with (
                self.subTest(kernel=kernel.fn.__name__),
                self.assertRaisesRegex(helion.exc.InvalidConfig, reason),
            ):
                _fragment_ordered_codegen(
                    kernel,
                    (
                        torch.ones((3, 17), dtype=torch.int32),
                        torch.empty((3, 17), dtype=torch.int32),
                    ),
                )
        with self.assertRaisesRegex(helion.exc.InvalidConfig, "terminal uniform"):
            _fragment_ordered_codegen(
                _fragment_branch_escape,
                (
                    torch.ones((3, 17), dtype=torch.int32),
                    torch.empty((3, 17), dtype=torch.int32),
                ),
            )

    @skipUnlessCuteAvailable("requires CuTe DSL")
    def test_actual_sdk_compact_and_fallback_fresh_histograms(self):
        import ast
        import importlib.util
        from pathlib import Path
        import tempfile

        import cutlass
        from cutlass._mlir import ir
        from cutlass._mlir.dialects import func
        import cutlass.cute as cute

        for kernel, args, scan in (
            (_fragment_branch_refinement, _branch_refinement_args(17, 8), "serial"),
            (
                _fragment_branch_refinement,
                _branch_refinement_args(17, 8),
                "cooperative",
            ),
            (
                _fragment_branch_atomic_epochs,
                (
                    torch.ones((3, 17), dtype=torch.int32),
                    torch.empty((3, 17), dtype=torch.int32),
                ),
                "serial",
            ),
        ):
            source = _local_histogram_consumer_codegen(kernel, args, scan, 128)
            tree = ast.parse(source)
            fn = next(
                n
                for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name.startswith("_helion_")
            )
            fn.name = "staged_branch"
            fn.decorator_list = [ast.parse("cute.jit", mode="eval").body]
            tree.body = [
                n
                for n in tree.body
                if isinstance(n, (ast.Import, ast.ImportFrom, ast.Assign))
            ] + [fn]
            with (
                self.subTest(kernel=kernel.fn.__name__, scan=scan),
                tempfile.TemporaryDirectory() as directory,
            ):
                path = Path(directory) / "staged.py"
                path.write_text(ast.unparse(ast.fix_missing_locations(tree)))
                spec = importlib.util.spec_from_file_location("staged_branch", path)
                sdk = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(sdk)
                with ir.Context(), ir.Location.unknown():
                    emitted = ir.Module.create()
                    with ir.InsertionPoint(emitted.body):
                        entry = func.FuncOp("entry", ([], []))
                        block = entry.add_entry_block()
                        with ir.InsertionPoint(block):
                            tensors = [
                                cute.make_tensor(
                                    cute.make_ptr(
                                        cutlass.Int32,
                                        0,
                                        cute.AddressSpace.gmem,
                                        assumed_align=16,
                                    ),
                                    cute.make_layout((512,)),
                                )
                                for _arg in fn.args.args
                            ]
                            sdk.staged_branch(*tensors)
                            func.ReturnOp([])
                    self.assertTrue(emitted.operation.verify())
                    text = str(emitted)
                    self.assertIn("scf.if", text)
                    self.assertIn("nvvm.atomicrmw", text)
                    self.assertIn("nvvm.barrier", text)


@onlyBackends("cute")
class TestFragmentBranchLocalAtomicsNative(TestCase):
    def test_compact_refinement_and_overflow_fallback(self):
        for width, capacity, threads in ((17, 8, 32), (65, 17, 128), (129, 33, 512)):
            for scan in ("serial", "cooperative"):
                with self.subTest(
                    width=width, capacity=capacity, threads=threads, scan=scan
                ):
                    args = _branch_refinement_args(width, capacity, DEVICE)
                    before = [value.clone() for value in args[:2]]
                    code_and_output(
                        _fragment_branch_refinement,
                        args,
                        cute_fragment_threads=threads,
                        cute_fragment_scan=scan,
                    )
                    _check_branch_refinement(args)
                    for actual, expected in zip(args[:2], before, strict=True):
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_unused_branch_epochs(self):
        x = torch.stack(
            (
                torch.ones(65, device=DEVICE),
                -torch.ones(65, device=DEVICE),
                torch.zeros(65, device=DEVICE),
            )
        ).to(torch.int32)
        out = torch.full((3, 17), -99, dtype=torch.int32, device=DEVICE)
        code_and_output(_fragment_branch_atomic_epochs, (x, out))
        expected = torch.zeros_like(out)
        expected.scatter_add_(
            1, (torch.arange(65, device=DEVICE) % 17).expand(3, -1), x
        )
        torch.testing.assert_close(out, expected * 2, rtol=0, atol=0)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _fragment_register_atomic_consumers(x, out, capacity: hl.constexpr):
    for row in hl.grid(x.size(0)):
        lane = hl.arange(helion.next_power_of_2(x.size(1)))
        value = hl.load(x, [row, lane], extra_mask=lane < x.size(1))
        active = value > 0
        counter = hl.zeros([1], dtype=torch.int32)
        payload = hl.zeros([capacity], dtype=torch.int32)
        identifiers = hl.zeros([capacity], dtype=torch.int32)
        tickets = hl.atomic_add(
            counter, [torch.zeros_like(lane)], active.to(torch.int32)
        )
        fits = active & (tickets < capacity)
        index = torch.where(
            fits,
            torch.where(lane % 2 == 0, tickets - capacity, tickets),
            torch.where(lane % 2 == 0, -capacity - 1, capacity),
        )
        # Invalid lanes deliberately contain an unsafe integer conversion.
        contribution = torch.where(fits, value, float("nan"))
        hl.atomic_add(payload, [index], contribution)
        hl.atomic_add(identifiers, [index], lane.to(torch.int32) + 1)
        # The same immutable ticket is consumed again as a contribution.
        hl.atomic_add(payload, [index], tickets)
        out[row, 0] = counter.sum()
        hl.store(out, [row, hl.arange(capacity) + 1], payload)
        hl.store(out, [row, hl.arange(capacity) + capacity + 1], identifiers)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _fragment_register_atomic_branches(x, out, capacity: hl.constexpr):
    for row in hl.grid(x.size(0)):
        lane = hl.arange(helion.next_power_of_2(x.size(1)))
        value = hl.load(x, [row, lane], extra_mask=lane < x.size(1))
        population = hl.zeros([1], dtype=torch.int32)
        hl.atomic_add(population, [torch.zeros_like(lane)], (value > 0).to(torch.int32))
        if population.sum() <= capacity:
            counter = hl.zeros([1], dtype=torch.int32)
            payload = hl.zeros([capacity], dtype=torch.int32)
            identifiers = hl.zeros([capacity], dtype=torch.int32)
            active = value > 0
            tickets = hl.atomic_add(
                counter, [torch.zeros_like(lane)], active.to(torch.int32)
            )
            index = torch.where(active, tickets, capacity)
            hl.atomic_add(payload, [index], value)
            hl.atomic_add(identifiers, [index], lane.to(torch.int32) + 1)
            hl.atomic_add(payload, [index], tickets)
            out[row, 0] = counter.sum()
            hl.store(out, [row, hl.arange(capacity) + 1], payload)
            hl.store(out, [row, hl.arange(capacity) + capacity + 1], identifiers)
        else:
            counter_fallback = hl.zeros([1], dtype=torch.int32)
            payload_fallback = hl.zeros([capacity], dtype=torch.int32)
            identifiers_fallback = hl.zeros([capacity], dtype=torch.int32)
            # A complete deterministic fallback set, independent of the sample.
            active_fallback = lane < capacity
            tickets_fallback = hl.atomic_add(
                counter_fallback,
                [torch.zeros_like(lane)],
                active_fallback.to(torch.int32),
            )
            index_fallback = torch.where(active_fallback, tickets_fallback, capacity)
            hl.atomic_add(payload_fallback, [index_fallback], value)
            hl.atomic_add(
                identifiers_fallback, [index_fallback], lane.to(torch.int32) + 1
            )
            hl.atomic_add(payload_fallback, [index_fallback], tickets_fallback)
            out[row, 0] = counter_fallback.sum()
            hl.store(out, [row, hl.arange(capacity) + 1], payload_fallback)
            hl.store(
                out, [row, hl.arange(capacity) + capacity + 1], identifiers_fallback
            )
    return out


def _register_consumer_codegen(kernel, args, enabled, aggregate, threads):
    from test._cute_binding import _cpu_bind
    from test._cute_binding import _forbid_native_compile
    from test._cute_binding import _mock_cuda_unavailable
    from test.cute_population_contracts import _target

    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound = _cpu_bind(kernel, args)
        config = bound.config_spec.default_config()
        config.config.update(
            cute_fragment_local_atomic_registers=enabled,
            cute_fragment_atomic_aggregation=aggregate,
            cute_fragment_threads=threads,
        )
        _, config = bound.config_spec.create_config_generation().strict_config_pair(
            config
        )
        return bound.to_code(config)


def _register_consumer_args(width, capacity, device="cpu"):
    x = torch.ones((4, width), device=device, dtype=torch.float32) * 2.75
    x[0].zero_()
    x[1, capacity - 1 :] = -1.25
    x[2, capacity:] = -1.25
    out = torch.full((4, 2 * capacity + 1), -99, dtype=torch.int32, device=device)
    return x, out, capacity


def _check_register_consumers(kernel, args):
    x, out, capacity = args
    branch = kernel is _fragment_register_atomic_branches
    for row in range(x.size(0)):
        selected = torch.nonzero(x[row] > 0).flatten()
        expected_count = selected.numel()
        if branch and expected_count > capacity:
            selected = torch.arange(capacity, device=x.device)
            expected_count = capacity
        assert int(out[row, 0]) == expected_count
        count = min(expected_count, capacity)
        ids = out[row, capacity + 1 : capacity + 1 + count].to(torch.int64) - 1
        assert torch.unique(ids).numel() == count
        assert bool(torch.isin(ids, selected).all())
        torch.testing.assert_close(
            out[row, 1 : 1 + count],
            x[row, ids].to(torch.int32)
            + torch.arange(count, device=x.device, dtype=torch.int32),
            rtol=0,
            atol=0,
        )
        assert bool((out[row, 1 + count : capacity + 1] == 0).all())
        assert bool((out[row, capacity + 1 + count :] == 0).all())


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _fragment_register_atomic_rejected(x, out, mode: hl.constexpr):
    for row in hl.grid(x.size(0)):
        lane = hl.arange(helion.next_power_of_2(x.size(1)))
        counter = hl.zeros([1], dtype=torch.int32)
        ticket = hl.atomic_add(counter, [torch.zeros_like(lane)], 1)
        if mode == "float":
            target = hl.zeros([17], dtype=torch.float32)
            hl.atomic_add(target, [ticket % 17], 1.0)
            out[row, :] = target.to(torch.int32)
        elif mode == "ordered":
            target = hl.zeros([17], dtype=torch.int32)
            hl.atomic_add(target, [ticket % 17], 1, sem="acq_rel")
            out[row, :] = target
        elif mode == "global":
            hl.atomic_add(out, [row, ticket % 17], 1)
        elif mode == "flip":
            target = hl.zeros([17], dtype=torch.int32)
            hl.atomic_add(target, [torch.flip(ticket, [0]) % 17], 1)
            out[row, :] = target
        else:
            target = hl.zeros([17], dtype=torch.int32)
            result = torch.stack((ticket, ticket))
            hl.atomic_add(target, [result % 17], 1)
            out[row, :] = target
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _fragment_atomic_consumer_fusion(x, out, capacity: hl.constexpr):
    for row in hl.grid(x.size(0)):
        lane = hl.arange(helion.next_power_of_2(x.size(1)))
        value = hl.load(x, [row, lane], extra_mask=lane < x.size(1))
        active = value > 0
        counter = hl.zeros([1], dtype=torch.int32)
        payload = hl.zeros([capacity], dtype=torch.int32)
        identifiers = hl.zeros([capacity], dtype=torch.int32)
        tickets = hl.atomic_add(
            counter, [torch.zeros_like(lane)], active.to(torch.int32)
        )
        fits = active & (tickets < capacity)
        index = torch.where(
            fits,
            torch.where(lane % 2 == 0, tickets - capacity, tickets),
            torch.where(lane % 2 == 0, -capacity - 1, capacity),
        )
        # The guard must still dominate this conversion, including padded lanes.
        contribution = torch.where(fits, value, float("nan"))
        hl.atomic_add(payload, [index], contribution)
        hl.atomic_add(identifiers, [index], lane.to(torch.int32) + 1)
        out[row, 0] = counter.sum()
        hl.store(out, [row, hl.arange(capacity) + 1], payload)
        hl.store(out, [row, hl.arange(capacity) + capacity + 1], identifiers)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _fragment_atomic_consumer_exclusions(x, out, mode: hl.constexpr):
    for row in hl.grid(x.size(0)):
        lane = hl.arange(x.size(1))
        counter = hl.zeros([1], dtype=torch.int32)
        sink = hl.zeros([17], dtype=torch.int32)
        second = hl.zeros([17], dtype=torch.int32)
        if mode == "prior_epoch":
            hl.atomic_add(counter, [torch.zeros_like(lane)], 1)
        tickets = hl.atomic_add(counter, [torch.zeros_like(lane)], 1)
        if mode == "late_init":
            late = hl.zeros([17], dtype=torch.int32)
            hl.atomic_add(late, [tickets % 17], 1)
            out[row, :] = late
        elif mode == "observer":
            hl.store(out, [row, lane], tickets, extra_mask=lane < 17)
            hl.atomic_add(sink, [tickets % 17], 1)
            out[row, :] = sink
        elif mode == "flip":
            hl.atomic_add(sink, [torch.flip(tickets, [0]) % 17], 1)
            out[row, :] = sink
        elif mode == "repeat_target":
            hl.atomic_add(sink, [tickets % 17], 1)
            hl.atomic_add(sink, [tickets % 17], 2)
            out[row, :] = sink
        elif mode == "host_store":
            hl.store(out, [row, lane], tickets, extra_mask=lane < 17)
        elif mode == "returned_sink":
            observed = hl.atomic_add(sink, [tickets % 17], 1)
            hl.atomic_add(second, [observed % 17], 1)
            out[row, :] = second
        else:
            hl.atomic_add(sink, [tickets % 17], 1)
            out[row, :] = sink
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _fragment_atomic_consumer_typed(x, out):
    for row in hl.grid(x.size(0)):
        lane = hl.arange(helion.next_power_of_2(x.size(1)))
        valid = lane < x.size(1)
        value = hl.load(x, [row, lane], extra_mask=valid)
        counter = hl.zeros([1], dtype=torch.int32)
        left = hl.full([17], 3, dtype=torch.int32)
        right = hl.full([17], -4, dtype=torch.int32)
        tickets = hl.atomic_add(
            counter, [torch.zeros_like(lane)], valid.to(torch.int32)
        )
        index = torch.where(valid, tickets % 17, 17)
        # Distinct SSA recipes retain their own cast, even with equal scalar
        # spellings. The second recipe must not reuse the first after rebinding.
        current = value.to(torch.int32).to(torch.float32) + 0.75
        contribution = torch.where(tickets >= 0, current, -0.0)
        hl.atomic_add(left, [index], contribution)
        current = value + 0.75
        contribution = torch.where(tickets >= 0, current, 0.0)
        hl.atomic_add(right, [index], contribution)
        hl.store(out, [row, hl.arange(17)], left)
        hl.store(out, [row, hl.arange(17) + 17], right)
    return out


def _atomic_consumer_fusion_codegen(kernel, args, enabled, threads=32, **extra):
    from test._cute_binding import _cpu_bind
    from test._cute_binding import _forbid_native_compile
    from test._cute_binding import _mock_cuda_unavailable
    from test.cute_population_contracts import _target

    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound = _cpu_bind(kernel, args)
        config = bound.config_spec.default_config()
        config.config.update(
            cute_fragment_atomic_consumer_fusion=enabled,
            cute_fragment_threads=threads,
            **extra,
        )
        _, config = bound.config_spec.create_config_generation().strict_config_pair(
            config
        )
        return bound.to_code(config)


def _check_fused_compaction(args):
    x, result, capacity = args
    for row in range(x.size(0)):
        selected = torch.nonzero(x[row] > 0).flatten()
        count = min(selected.numel(), capacity)
        assert int(result[row, 0]) == selected.numel()
        ids = result[row, capacity + 1 : capacity + 1 + count].long() - 1
        assert torch.unique(ids).numel() == count
        assert bool(torch.isin(ids, selected).all())
        torch.testing.assert_close(
            result[row, 1 : 1 + count], x[row, ids].int(), rtol=0, atol=0
        )
        assert bool((result[row, 1 + count : capacity + 1] == 0).all())
        assert bool((result[row, capacity + 1 + count :] == 0).all())


@onlyBackends("cute")
class TestFragmentAtomicConsumerFusionNative(TestCase):
    def test_ticket_payload_pairing_and_zero_updates(self):
        for width, capacity, threads in ((17, 8, 32), (65, 17, 128), (129, 33, 512)):
            for enabled in (False, True):
                with self.subTest(width=width, threads=threads, enabled=enabled):
                    args = _register_consumer_args(width, capacity, DEVICE)
                    before = args[0].clone()
                    code_and_output(
                        _fragment_atomic_consumer_fusion,
                        args,
                        cute_fragment_threads=threads,
                        cute_fragment_atomic_consumer_fusion=enabled,
                    )
                    _check_fused_compaction(args)
                    torch.testing.assert_close(args[0], before, rtol=0, atol=0)

    def test_typed_rebindings_and_initialized_colliding_targets(self):
        x = (
            torch.tensor([2.625, -2.625], device=DEVICE)[:, None]
            .expand(2, 65)
            .contiguous()
        )
        counts = torch.bincount(torch.arange(65, device=DEVICE) % 17, minlength=17)
        expected = torch.cat(
            (
                3 + counts * (x[:, :1].int().float() + 0.75).int(),
                -4 + counts * (x[:, :1] + 0.75).int(),
            ),
            dim=1,
        ).int()
        before = x.clone()
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                out = torch.zeros((2, 34), dtype=torch.int32, device=DEVICE)
                code_and_output(
                    _fragment_atomic_consumer_typed,
                    (x, out),
                    cute_fragment_threads=128,
                    cute_fragment_atomic_consumer_fusion=enabled,
                )
                torch.testing.assert_close(out, expected, rtol=0, atol=0)
                torch.testing.assert_close(x, before, rtol=0, atol=0)


class TestFragmentAtomicConsumerFusionCPU(unittest.TestCase):
    def test_default_strict_domain_and_complete_population_prefix(self):
        from copy import deepcopy
        import random
        from unittest.mock import patch

        from test._cute_binding import _cpu_bind
        from test._cute_binding import _forbid_native_compile
        from test._cute_binding import _mock_cuda_unavailable
        from test.cute_population_contracts import _target
        from test.cute_population_contracts import checked_initial_population

        from helion.autotuner.pattern_search import InitialPopulationStrategy
        from helion.autotuner.pattern_search import PatternSearch

        key = "cute_fragment_atomic_consumer_fusion"
        args = _register_consumer_args(65, 17)
        kernel = helion.kernel(
            _fragment_atomic_consumer_fusion.fn,
            backend="cute",
            static_shapes=True,
            autotune_effort="full",
        )
        with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
            bound = _cpu_bind(kernel, args)
            default = bound.config_spec.default_config()
            self.assertNotIn(key, default)
            off = deepcopy(default)
            off.config[key] = False
            self.assertEqual(bound.to_code(default), bound.to_code(off))
            for value in (1, None, "true"):
                bad = deepcopy(default)
                bad.config[key] = value
                with self.assertRaises(helion.exc.InvalidConfig):
                    bound.to_code(bad)
            group = bound.config_spec.compiler_coverage_groups[-1]
            self.assertEqual(group.key, key)
            self.assertTrue(group.deferred)
            self.assertFalse(group.legacy)

            def population(bound):
                with bound.env:
                    search = PatternSearch(
                        bound,
                        args,
                        initial_population=100,
                        initial_population_strategy=InitialPopulationStrategy.FROM_RANDOM,
                    )
                    rows = checked_initial_population(search)
                    return [
                        dict(search.config_gen.canonicalize_flat(row)[1])
                        for row in rows
                    ], random.getstate()

            for seed in (17, 2026):
                with self.subTest(seed=seed):
                    random.seed(seed)
                    current, state = population(bound)
                    with patch(
                        "helion._compiler.autotuner_heuristics.register_fragment_atomic_consumer_fusion_coverage"
                    ):
                        old_bound = _cpu_bind(kernel, args)
                    random.seed(seed)
                    old, old_state = population(old_bound)
                    self.assertEqual(current[: len(old)], old)
                    self.assertEqual(state, old_state)
                    self.assertTrue(any(row.get(key) is True for row in current))

    @skipUnlessCuteAvailable("requires CuTe DSL")
    def test_actual_sdk_stages_fused_typed_and_masked_consumers(self):
        import ast
        import importlib.util
        from pathlib import Path
        import tempfile

        import cutlass
        from cutlass._mlir import ir
        from cutlass._mlir.dialects import func
        import cutlass.cute as cute

        for kernel, args in (
            (_fragment_atomic_consumer_fusion, _register_consumer_args(65, 17)),
            (
                _fragment_atomic_consumer_typed,
                (torch.ones((2, 65)), torch.zeros((2, 34), dtype=torch.int32)),
            ),
        ):
            with self.subTest(kernel=kernel.fn.__name__):
                source = _atomic_consumer_fusion_codegen(kernel, args, True)
                tree = ast.parse(source)
                fn = next(
                    n
                    for n in tree.body
                    if isinstance(n, ast.FunctionDef) and n.name.startswith("_helion_")
                )
                fn.name = "staged"
                fn.decorator_list = [ast.parse("cute.jit", mode="eval").body]
                tree.body = [
                    n
                    for n in tree.body
                    if isinstance(n, (ast.Import, ast.ImportFrom, ast.Assign))
                ] + [fn]
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "fusion.py"
                    path.write_text(ast.unparse(ast.fix_missing_locations(tree)))
                    spec = importlib.util.spec_from_file_location("fusion_staged", path)
                    sdk = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(sdk)
                    with ir.Context(), ir.Location.unknown():
                        emitted = ir.Module.create()
                        with ir.InsertionPoint(emitted.body):
                            entry = func.FuncOp("entry", ([], []))
                            with ir.InsertionPoint(entry.add_entry_block()):
                                tensors = [
                                    cute.make_tensor(
                                        cute.make_ptr(
                                            cutlass.Float32
                                            if arg.arg == "x"
                                            else cutlass.Int32,
                                            0,
                                            cute.AddressSpace.gmem,
                                            assumed_align=16,
                                        ),
                                        cute.make_layout((512,)),
                                    )
                                    for arg in fn.args.args
                                ]
                                sdk.staged(*tensors)
                                func.ReturnOp([])
                        self.assertTrue(emitted.operation.verify())
                        text = str(emitted)
                        self.assertEqual(text.count("nvvm.atomicrmw"), 3)
                        self.assertIn("#nvvm.mem_scope<cta>", text)
                        self.assertIn("nvvm.barrier", text)

    def test_typed_current_ssa_constants_and_nonzero_initialization(self):
        x = ((torch.arange(130).reshape(2, 65) % 7).float() - 3) * 0.625
        out = torch.zeros((2, 34), dtype=torch.int32)
        for order in (list(range(32)), list(reversed(range(32)))):
            for enabled in (False, True):
                with self.subTest(enabled=enabled, reversed=order[0] != 0):
                    code = _atomic_consumer_fusion_codegen(
                        _fragment_atomic_consumer_typed, (x, out), enabled
                    )
                    _simulate_register_load_program(
                        code, x, 32, host_tensors={"out": out}, lane_order=order
                    )
                    columns = [
                        column for thread in order for column in range(thread, 65, 32)
                    ]
                    slots = torch.empty(65, dtype=torch.int64)
                    slots[torch.tensor(columns)] = torch.arange(65) % 17
                    expected = torch.cat(
                        (torch.full((2, 17), 3), torch.full((2, 17), -4)), dim=1
                    ).int()
                    expected[:, :17].scatter_add_(
                        1, slots.expand(2, -1), (x.int().float() + 0.75).int()
                    )
                    expected[:, 17:].scatter_add_(
                        1, slots.expand(2, -1), (x + 0.75).int()
                    )
                    torch.testing.assert_close(out, expected, rtol=0, atol=0)

    def test_observers_alias_epochs_and_cross_coordinate_consumers_decline(self):
        x = torch.ones((2, 65), dtype=torch.int32)
        out = torch.zeros((2, 17), dtype=torch.int32)
        for mode in (
            "prior_epoch",
            "late_init",
            "observer",
            "flip",
            "repeat_target",
            "host_store",
            "returned_sink",
        ):
            with self.subTest(mode=mode):
                baseline = _atomic_consumer_fusion_codegen(
                    _fragment_atomic_consumer_exclusions, (x, out, mode), False
                )
                self.assertIn("cute.arch.atomic_add", baseline)
                with self.assertRaises(helion.exc.InvalidConfig):
                    _atomic_consumer_fusion_codegen(
                        _fragment_atomic_consumer_exclusions, (x, out, mode), True
                    )

    def test_existing_register_and_aggregation_schedules_remain_unchanged(self):
        args = _register_consumer_args(65, 17)
        for options in (
            {"cute_fragment_local_atomic_registers": True},
            {"cute_fragment_atomic_aggregation": True},
        ):
            with self.subTest(options=options):
                codes = [
                    _atomic_consumer_fusion_codegen(
                        _fragment_atomic_consumer_fusion, args, flag, **options
                    )
                    for flag in (False, True)
                ]
                self.assertEqual(*codes)

    def test_input_register_escape_rejection_is_preserved(self):
        # This base's load-register proof does not admit the indexed atomic
        # consumer chain. Fusion must not broaden that separate owner proof.
        args = _register_consumer_args(65, 17)
        for enabled in (False, True):
            with (
                self.subTest(enabled=enabled),
                self.assertRaises(helion.exc.InvalidConfig),
            ):
                _atomic_consumer_fusion_codegen(
                    _fragment_atomic_consumer_fusion,
                    args,
                    enabled,
                    cute_fragment_register_loads=True,
                )

    def test_compaction_preserves_payload_id_pairing_masks_and_publication(self):
        import ast

        for width, capacity, threads in ((17, 8, 32), (65, 17, 32), (129, 33, 128)):
            with self.subTest(width=width, threads=threads):
                args = _register_consumer_args(width, capacity)
                codes = [
                    _atomic_consumer_fusion_codegen(
                        _fragment_atomic_consumer_fusion, args, flag, threads
                    )
                    for flag in (False, True)
                ]
                loops = [
                    sum(isinstance(node, ast.For) for node in ast.walk(ast.parse(code)))
                    for code in codes
                ]
                self.assertEqual(loops[0] - loops[1], 2)
                self.assertEqual(
                    codes[0].count("cute.arch.sync_threads()"),
                    codes[1].count("cute.arch.sync_threads()"),
                )
                for order in (list(range(threads)), list(reversed(range(threads)))):
                    outputs = []
                    for code in codes:
                        args[1].fill_(-99)
                        before = args[0].clone()
                        events = []
                        _result, barriers = _simulate_register_load_program(
                            code,
                            args[0],
                            threads,
                            host_tensors={"out": args[1]},
                            lane_order=order,
                            atomic_events=events,
                        )
                        result = args[1]
                        _check_fused_compaction(args)
                        torch.testing.assert_close(args[0], before, rtol=0, atol=0)
                        outputs.append((result.clone(), barriers, events))
                    torch.testing.assert_close(
                        outputs[0][0], outputs[1][0], rtol=0, atol=0
                    )
                    self.assertEqual(outputs[0][1], outputs[1][1])
                    # Updates to independent targets may interleave, while all
                    # per-target updates, including zero contributions, remain.
                    self.assertCountEqual(outputs[0][2], outputs[1][2])


class TestFragmentRegisterAtomicConsumersCPU(unittest.TestCase):
    def test_root_and_branch_compaction_events_masks_and_epochs(self):
        for kernel in (
            _fragment_register_atomic_consumers,
            _fragment_register_atomic_branches,
        ):
            for width, capacity, threads in ((17, 8, 32), (65, 17, 32), (129, 33, 128)):
                for aggregate in (False, True):
                    with self.subTest(
                        kernel=kernel.fn.__name__, width=width, aggregate=aggregate
                    ):
                        args = _register_consumer_args(width, capacity)
                        before = args[0].clone()
                        codes = [
                            _register_consumer_codegen(
                                kernel, args, flag, aggregate, threads
                            )
                            for flag in (False, True)
                        ]
                        self.assertNotIn("fragment_local_tickets", codes[0])
                        # One real returned atomic in each lexical arm; terminal
                        # unused consumers must not acquire return snapshots.
                        self.assertEqual(
                            codes[1].count("= cute.make_rmem_tensor"),
                            2 if kernel is _fragment_register_atomic_branches else 1,
                        )
                        for order in (
                            list(range(threads)),
                            list(reversed(range(threads))),
                        ):
                            observed = []
                            for code in codes:
                                args[1].fill_(-99)
                                events = []
                                result, barriers = _simulate_register_load_program(
                                    code,
                                    args[0],
                                    threads,
                                    host_tensors={"out": args[1]},
                                    lane_order=order,
                                    atomic_events=events,
                                )
                                _check_register_consumers(kernel, args)
                                observed.append((result.clone(), barriers, events))
                            torch.testing.assert_close(
                                observed[0][0], observed[1][0], rtol=0, atol=0
                            )
                            self.assertEqual(observed[0][1:], observed[1][1:])
                        torch.testing.assert_close(args[0], before, rtol=0, atol=0)

    def test_atomic_consumers_decline_changes_of_scope_dtype_or_coordinate(self):
        args = (
            torch.ones((2, 65), dtype=torch.int32),
            torch.zeros((2, 17), dtype=torch.int32),
        )
        for mode in ("float", "ordered", "global", "flip", "stack"):
            with self.subTest(mode=mode), self.assertRaises(helion.exc.InvalidConfig):
                _register_consumer_codegen(
                    _fragment_register_atomic_rejected, (*args, mode), True, False, 128
                )

    @skipUnlessCuteAvailable("requires CuTe DSL")
    def test_actual_sdk_register_consumers_and_uniform_aggregate_branches(self):
        import ast
        import importlib.util
        from pathlib import Path
        import tempfile

        import cutlass
        from cutlass._mlir import ir
        from cutlass._mlir.dialects import func
        import cutlass.cute as cute

        for kernel in (
            _fragment_register_atomic_consumers,
            _fragment_register_atomic_branches,
        ):
            for aggregate in (False, True):
                with self.subTest(kernel=kernel.fn.__name__, aggregate=aggregate):
                    source = _register_consumer_codegen(
                        kernel, _register_consumer_args(65, 17), True, aggregate, 32
                    )
                    tree = ast.parse(source)
                    fn = next(
                        n
                        for n in tree.body
                        if isinstance(n, ast.FunctionDef)
                        and n.name.startswith("_helion_")
                    )
                    fn.name = "staged"
                    fn.decorator_list = [ast.parse("cute.jit", mode="eval").body]
                    tree.body = [
                        n
                        for n in tree.body
                        if isinstance(n, (ast.Import, ast.ImportFrom, ast.Assign))
                    ] + [fn]
                    with tempfile.TemporaryDirectory() as directory:
                        path = Path(directory) / "staged.py"
                        path.write_text(ast.unparse(ast.fix_missing_locations(tree)))
                        spec = importlib.util.spec_from_file_location(
                            "register_atomic_staged", path
                        )
                        sdk = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(sdk)
                        with ir.Context(), ir.Location.unknown():
                            emitted = ir.Module.create()
                            with ir.InsertionPoint(emitted.body):
                                entry = func.FuncOp("entry", ([], []))
                                with ir.InsertionPoint(entry.add_entry_block()):
                                    tensors = [
                                        cute.make_tensor(
                                            cute.make_ptr(
                                                cutlass.Float32
                                                if arg.arg == "x"
                                                else cutlass.Int32,
                                                0,
                                                cute.AddressSpace.gmem,
                                                assumed_align=16,
                                            ),
                                            cute.make_layout((512,)),
                                        )
                                        for arg in fn.args.args
                                    ]
                                    sdk.staged(*tensors)
                                    func.ReturnOp([])
                            self.assertTrue(emitted.operation.verify())
                            text = str(emitted)
                            self.assertIn("nvvm.atomicrmw", text)
                            self.assertIn("#nvvm.mem_scope<cta>", text)
                            self.assertIn("nvvm.barrier", text)
                            self.assertEqual("nvvm.match.sync" in text, aggregate)
                            self.assertEqual("nvvm.redux.sync" in text, aggregate)


@onlyBackends("cute")
class TestFragmentRegisterAtomicConsumersNative(TestCase):
    def test_root_and_branch_compaction_return_registers(self):
        for kernel in (
            _fragment_register_atomic_consumers,
            _fragment_register_atomic_branches,
        ):
            for width, capacity, threads in (
                (17, 8, 32),
                (65, 17, 128),
                (8192, 1024, 512),
            ):
                for aggregate in (False, True):
                    with self.subTest(
                        kernel=kernel.fn.__name__, width=width, aggregate=aggregate
                    ):
                        args = _register_consumer_args(width, capacity, DEVICE)
                        before = args[0].clone()
                        code_and_output(
                            kernel,
                            args,
                            cute_fragment_threads=threads,
                            cute_fragment_local_atomic_registers=True,
                            cute_fragment_atomic_aggregation=aggregate,
                        )
                        _check_register_consumers(kernel, args)
                        torch.testing.assert_close(args[0], before, rtol=0, atol=0)


class TestFragmentThreadCoverageCompatibilityCPU(unittest.TestCase):
    def test_previous_population_and_rng_with_dependent_witnesses(self):
        from contextlib import ExitStack
        import random
        from unittest.mock import patch

        from test._cute_binding import _cpu_bind
        from test._cute_binding import _forbid_native_compile
        from test._cute_binding import _mock_cuda_unavailable
        from test.cute_population_contracts import _target
        from test.cute_population_contracts import checked_initial_population

        from helion._compiler.autotuner_heuristics import (
            cute_fragment_local_atomic_registers,
        )
        from helion._compiler.autotuner_heuristics import cute_fragment_threads
        from helion._compiler.autotuner_heuristics import cute_fragment_warp_results
        from helion.autotuner.pattern_search import InitialPopulationStrategy
        from helion.autotuner.pattern_search import PatternSearch

        x = torch.ones((1, 8193), dtype=torch.int32)
        fixtures = (
            (
                _fragment_warp_reservations,
                (*_warp_reservation_args(4), False),
                "cute_fragment_warp_results",
            ),
            (
                _register_load_histogram,
                (torch.ones(2, 65), 3, 128),
                "cute_fragment_register_loads",
            ),
            (
                _fragment_local_registers,
                (x, x.clone(), x.clone(), False),
                "cute_fragment_local_atomic_registers",
            ),
        )

        def capture(kernel, args, key, legacy):
            with ExitStack() as stack:
                if legacy:
                    for module in (
                        cute_fragment_threads,
                        cute_fragment_warp_results,
                        cute_fragment_local_atomic_registers,
                    ):
                        stack.enter_context(
                            patch.object(
                                module,
                                "THREADS",
                                cute_fragment_threads.LEGACY_COVERAGE_THREADS,
                            )
                        )
                bound = _cpu_bind(
                    helion.kernel(
                        kernel.fn,
                        backend="cute",
                        static_shapes=True,
                        autotune_effort="full",
                    ),
                    args,
                )
                spec = bound.config_spec
                group = next(g for g in spec.compiler_coverage_groups if g.key == key)
                self.assertEqual(
                    [(d.key, d.value) for d in group.dependencies],
                    [("cute_fragment_threads", 512)],
                )
                inventory = [
                    (
                        g.key,
                        [(d.key, d.value) for d in g.dependencies],
                        [(dict(w.carrier), w.value) for w in g.witnesses],
                    )
                    for g in spec.compiler_coverage_groups
                ]
                records = []
                for seed in (0, 107):
                    for strategy in (
                        InitialPopulationStrategy.FROM_RANDOM,
                        InitialPopulationStrategy.FROM_BEST_AVAILABLE,
                    ):
                        random.seed(seed)
                        with bound.env:
                            search = PatternSearch(
                                bound,
                                args,
                                initial_population=100,
                                initial_population_strategy=strategy,
                            )
                            flat = checked_initial_population(search)
                            rows = [
                                dict(search.config_gen.canonicalize_flat(row)[1])
                                for row in flat
                            ]
                        self.assertTrue(any(row.get(key) is True for row in rows))
                        self.assertTrue(
                            all(
                                row.get("cute_fragment_threads", 128) != 1024
                                for row in rows
                            )
                        )
                        records.append((rows, random.getstate()))
                return (
                    dict(spec.default_config()),
                    [dict(c) for c in spec.compiler_seed_configs],
                    inventory,
                    records,
                )

        with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
            for kernel, args, key in fixtures:
                with self.subTest(key=key):
                    self.assertEqual(
                        capture(kernel, args, key, True),
                        capture(kernel, args, key, False),
                    )

    def test_new_only_eligibility_can_require_1024_witness(self):
        from test._cute_binding import _cpu_bind
        from test._cute_binding import _forbid_native_compile
        from test._cute_binding import _mock_cuda_unavailable
        from test.cute_population_contracts import _target
        from test.cute_population_contracts import checked_initial_population

        from helion.autotuner.pattern_search import InitialPopulationStrategy
        from helion.autotuner.pattern_search import PatternSearch

        x = torch.ones((1, 16385), dtype=torch.int32)
        fixtures = (
            (
                _fragment_warp_reservations,
                (*_warp_reservation_args(32), False),
                "cute_fragment_warp_results",
            ),
            (
                _fragment_local_registers,
                (x, x.clone(), x.clone(), False),
                "cute_fragment_local_atomic_registers",
            ),
        )
        with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
            for kernel, args, key in fixtures:
                with self.subTest(key=key):
                    bound = _cpu_bind(
                        helion.kernel(
                            kernel.fn,
                            backend="cute",
                            static_shapes=True,
                            autotune_effort="full",
                        ),
                        args,
                    )
                    spec = bound.config_spec
                    group = next(
                        g for g in spec.compiler_coverage_groups if g.key == key
                    )
                    self.assertEqual(
                        [(d.key, d.value) for d in group.dependencies],
                        [("cute_fragment_threads", 1024)],
                    )
                    config = spec.default_config()
                    config.config.update({key: True, "cute_fragment_threads": 512})
                    with self.assertRaises(helion.exc.InvalidConfig):
                        bound.to_code(config)
                    config.config["cute_fragment_threads"] = 1024
                    self.assertIn("block=(1024, 1, 1)", bound.to_code(config))
                    with bound.env:
                        search = PatternSearch(
                            bound,
                            args,
                            initial_population=100,
                            initial_population_strategy=InitialPopulationStrategy.FROM_RANDOM,
                        )
                        rows = [
                            dict(search.config_gen.canonicalize_flat(row)[1])
                            for row in checked_initial_population(search)
                        ]
                    self.assertTrue(
                        any(
                            row.get(key) is True
                            and row.get("cute_fragment_threads") == 1024
                            for row in rows
                        )
                    )


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _fragment_snapshot_branch(x, positive: hl.constexpr):
    out = torch.empty_like(x)
    counts = torch.empty((x.size(0), 17), dtype=torch.int32, device=x.device)
    for row in hl.grid(x.size(0)):
        index = hl.arange(x.size(1))
        values = hl.load(x, [row, index])
        gate = hl.zeros([1], dtype=torch.int32)
        hl.atomic_add(gate, [0], positive)
        if gate.sum() > 0:
            local = hl.zeros([17], dtype=torch.int32)
            hl.atomic_add(local, [index % 17], values.to(torch.int32))
            counts[row, :] = local
            hl.store(out, [row, index], values + 2)
        else:
            other = hl.zeros([17], dtype=torch.int32)
            hl.atomic_add(other, [index % 17], (values + 1).to(torch.int32))
            counts[row, :] = other
            hl.store(out, [row, index], values - 3)
    return out, counts


class TestFragmentReadonlySnapshotAtomicCPU(TestCase):
    def test_uniform_captured_snapshot_epochs_and_aggregation(self):
        from test._cute_binding import _cpu_bind
        from test._cute_binding import _forbid_native_compile
        from test._cute_binding import _mock_cuda_unavailable
        from test.cute_population_contracts import _target

        for positive in (False, True):
            for aggregate in (False, True):
                for threads in (32, 128):
                    with self.subTest(
                        positive=positive, aggregate=aggregate, threads=threads
                    ):
                        x = torch.arange(2 * 65).reshape(2, 65).float() % 7
                        with (
                            _mock_cuda_unavailable(),
                            _target(),
                            _forbid_native_compile(),
                        ):
                            bound = _cpu_bind(_fragment_snapshot_branch, (x, positive))
                            config = bound.config_spec.default_config()
                            config.config.update(
                                cute_fragment_register_snapshots=True,
                                cute_fragment_threads=threads,
                                cute_fragment_atomic_aggregation=aggregate,
                            )
                            code = bound.to_code(config)
                        self.assertIn("fragment_snapshot", code)
                        for order in (
                            list(range(threads)),
                            list(reversed(range(threads))),
                        ):
                            out = torch.full_like(x, -99)
                            counts = torch.full((2, 17), -99, dtype=torch.int32)
                            _simulate_register_load_program(
                                code,
                                x,
                                threads,
                                host_tensors={"out": out, "counts": counts},
                                lane_order=order,
                            )
                            torch.testing.assert_close(
                                out, x + 2 if positive else x - 3, rtol=0, atol=0
                            )
                            expected = torch.zeros_like(counts)
                            expected.scatter_add_(
                                1,
                                (torch.arange(65) % 17).expand(2, -1),
                                (x if positive else x + 1).to(torch.int32),
                            )
                            torch.testing.assert_close(counts, expected, rtol=0, atol=0)


@onlyBackends("cute")
class TestFragmentReadonlySnapshotNative(TestCase):
    def test_uniform_branch_captures(self):
        for positive in (False, True):
            for aggregate in (False, True):
                for threads in (32, 1024):
                    with self.subTest(
                        positive=positive, aggregate=aggregate, threads=threads
                    ):
                        x = torch.arange(130, device=DEVICE).reshape(2, 65).float() % 7
                        before = x.clone()
                        bound = _fragment_snapshot_branch.bind((x, positive))
                        config = bound.config_spec.default_config()
                        config.config.update(
                            cute_fragment_register_snapshots=True,
                            cute_fragment_threads=threads,
                            cute_fragment_atomic_aggregation=aggregate,
                        )
                        out, counts = bound.compile_config(config)(x, positive)
                        torch.testing.assert_close(
                            out, x + 2 if positive else x - 3, rtol=0, atol=0
                        )
                        expected = torch.zeros_like(counts)
                        expected.scatter_add_(
                            1,
                            (torch.arange(65, device=DEVICE) % 17).expand(2, -1),
                            (x if positive else x + 1).to(torch.int32),
                        )
                        torch.testing.assert_close(counts, expected, rtol=0, atol=0)
                        torch.testing.assert_close(x, before, rtol=0, atol=0)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _integer_epoch_histogram(x, repeats: int, bins: hl.constexpr, chunk: hl.constexpr):
    out = torch.empty((x.size(0), bins), dtype=x.dtype, device=x.device)
    for row in hl.grid(x.size(0)):
        histogram = hl.full([bins], 3, dtype=x.dtype)
        for step in range(repeats):
            index = step * chunk + hl.arange(chunk)
            value = hl.load(x, [row, index], extra_mask=index < x.size(1))
            hl.atomic_add(histogram, [index % bins], value)
        out[row, :] = histogram
    return out


def _integer_epoch_codegen(x, repeats, *, enabled, register=True, threads=32):
    from test._cute_binding import _cpu_bind
    from test._cute_binding import _forbid_native_compile
    from test._cute_binding import _mock_cuda_unavailable
    from test.cute_population_contracts import _target

    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound = _cpu_bind(_integer_epoch_histogram, (x, repeats, 17, 32))
        config = bound.config_spec.default_config()
        config.config.update(
            cute_fragment_integer_atomic_epochs=enabled,
            cute_fragment_register_loads=register,
            cute_fragment_threads=threads,
        )
        return bound.to_code(config)


class TestFragmentIntegerAtomicEpochsCPU(unittest.TestCase):
    @skipUnlessCuteAvailable("requires CuTe DSL")
    def test_real_loop_exit_publication_and_zero_trip(self):
        import ast

        for width, repeats, threads in ((65, 3, 32), (65, 0, 32), (33, 2, 128)):
            x = (torch.arange(width).reshape(1, width) % 11 - 5).to(torch.int32)
            expected = torch.full((1, 17), 3, dtype=torch.int32)
            used = min(width, repeats * 32)
            expected.scatter_add_(1, (torch.arange(used) % 17)[None, :], x[:, :used])
            codes = []
            for enabled in (False, True):
                with self.subTest(
                    width=width, repeats=repeats, threads=threads, enabled=enabled
                ):
                    code = _integer_epoch_codegen(
                        x, repeats, enabled=enabled, threads=threads
                    )
                    codes.append(code)
                    for order in (list(range(threads)), list(reversed(range(threads)))):
                        output = torch.full_like(expected, -999)
                        _simulate_register_load_program(
                            code,
                            x,
                            threads,
                            host_tensors={"out": output},
                            scalar_args={"repeats": repeats},
                            lane_order=order,
                        )
                        torch.testing.assert_close(output, expected, rtol=0, atol=0)

            def inside(code):
                return [
                    sum(
                        isinstance(n, ast.Call)
                        and ast.unparse(n.func) == "cute.arch.sync_threads"
                        for n in ast.walk(loop)
                    )
                    for loop in ast.walk(ast.parse(code))
                    if isinstance(loop, ast.For)
                    and isinstance(loop.target, ast.Name)
                    and loop.target.id.startswith("fragment_tile")
                ]

            self.assertEqual(inside(codes[0]), [1])
            self.assertEqual(inside(codes[1]), [0])

    def test_shared_input_staging_keeps_backedge_barrier(self):
        import ast

        x = torch.arange(65, dtype=torch.int32).reshape(1, 65)
        a = _integer_epoch_codegen(x, 3, enabled=False, register=False)
        b = _integer_epoch_codegen(x, 3, enabled=True, register=False)
        self.assertEqual(ast.dump(ast.parse(a)), ast.dump(ast.parse(b)))
        output = torch.full((1, 17), -999, dtype=torch.int32)
        _simulate_register_load_program(
            b, x, 32, host_tensors={"out": output}, scalar_args={"repeats": 3}
        )
        expected = torch.full_like(output, 3)
        expected.scatter_add_(1, (torch.arange(65) % 17)[None, :], x)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)

    def test_emitted_effect_and_alias_proof(self):
        import ast

        from helion._compiler.cute.integer_atomic_epochs import can_defer_integer_epoch

        text = """value = cutlass.Int32(readonly[0])
for index in range(thread, 32, 32):
    if index < limit:
        cute.arch.atomic_add((hist.iterator + index).llvm_ptr, value, sem='relaxed', scope='cta')
"""
        buffers = {"hist": torch.int32, "readonly": torch.int32}

        def valid(t):
            return can_defer_integer_epoch(ast.parse(t).body, buffers, {"hist"})

        self.assertTrue(valid(text))
        mutations = [
            text.replace("readonly[0]", "hist[0]"),
            "readonly[0] = 1\n" + text,
            "alias = hist\n" + text,
            "alias = readonly.iterator\n" + text,
            text.replace("scope='cta'", "scope='gpu'"),
            text.replace("sem='relaxed'", "sem='release'"),
            text.replace("cute.arch.atomic_add", "previous = cute.arch.atomic_add"),
            text + "cute.arch.sync_threads()\n",
            text + "opaque(value)\n",
            text + "host.iterator.store(value)\n",
            text + "if value: return\n",
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                self.assertFalse(valid(mutation))
        for dtype in (torch.float32, torch.int64):
            with self.subTest(dtype=dtype):
                self.assertFalse(
                    can_defer_integer_epoch(
                        ast.parse(text).body, dict(buffers, hist=dtype), {"hist"}
                    )
                )

    def test_wrapping_zero_and_final_barrier_counterexample(self):
        import ast

        x = torch.full((1, 65), torch.iinfo(torch.int32).max, dtype=torch.int32)
        x[:, 1::3] = torch.iinfo(torch.int32).min
        x[:, 2::3] = 0
        expected = torch.full((1, 17), 3, dtype=torch.int32)
        expected.scatter_add_(1, (torch.arange(65) % 17)[None, :], x)
        code = _integer_epoch_codegen(x, 3, enabled=True)
        for order in (list(range(32)), list(reversed(range(32)))):
            output = torch.full_like(expected, -999)
            _simulate_register_load_program(
                code,
                x,
                32,
                host_tensors={"out": output},
                scalar_args={"repeats": 3},
                lane_order=order,
            )
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
        tree = ast.parse(code)
        removed = 0
        for parent in ast.walk(tree):
            for _field, items in ast.iter_fields(parent):
                if not isinstance(items, list):
                    continue
                for index in range(len(items) - 2, -1, -1):
                    loop = items[index]
                    if (
                        isinstance(loop, ast.For)
                        and isinstance(loop.target, ast.Name)
                        and loop.target.id.startswith("fragment_tile")
                    ):
                        self.assertEqual(
                            ast.unparse(items[index + 1]), "cute.arch.sync_threads()"
                        )
                        items.pop(index + 1)
                        removed += 1
        self.assertEqual(removed, 1)
        output = torch.full_like(expected, -999)
        _simulate_register_load_program(
            ast.unparse(tree),
            x,
            32,
            host_tensors={"out": output},
            scalar_args={"repeats": 3},
        )
        self.assertFalse(torch.equal(output, expected))

    def test_default_seed_and_full_population_prefix(self):
        import random
        from unittest.mock import patch

        from test._cute_binding import _cpu_bind
        from test._cute_binding import _forbid_native_compile
        from test._cute_binding import _mock_cuda_unavailable
        from test.cute_population_contracts import _target
        from test.test_compiler_coverage import make_search

        from helion.autotuner.pattern_search import InitialPopulationStrategy

        key = "cute_fragment_integer_atomic_epochs"
        args = (torch.ones(1, 65, dtype=torch.int32), 3, 17, 32)
        with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
            with patch(
                "helion._compiler.autotuner_heuristics.register_fragment_integer_atomic_epochs_coverage"
            ):
                old = _cpu_bind(
                    helion.kernel(
                        _integer_epoch_histogram.fn, backend="cute", static_shapes=True
                    ),
                    args,
                )
            new = _cpu_bind(
                helion.kernel(
                    _integer_epoch_histogram.fn, backend="cute", static_shapes=True
                ),
                args,
            )
            self.assertEqual(
                old.config_spec.default_config(), new.config_spec.default_config()
            )
            self.assertEqual(
                old.config_spec.compiler_seed_configs,
                new.config_spec.compiler_seed_configs,
            )
            import ast

            self.assertEqual(
                ast.dump(ast.parse(old.to_code(old.config_spec.default_config()))),
                ast.dump(ast.parse(new.to_code(new.config_spec.default_config()))),
            )
            for strategy in (
                InitialPopulationStrategy.FROM_RANDOM,
                InitialPopulationStrategy.FROM_BEST_AVAILABLE,
            ):
                for seed in (73, 741, 2031):
                    populations = []
                    states = []
                    for b in (old, new):
                        search = make_search(
                            b.config_spec, count=100, strategy=strategy
                        )
                        random.seed(seed)
                        populations.append(
                            [
                                search.config_gen.unflatten(row)
                                for row in search._generate_initial_population_flat()
                            ]
                        )
                        states.append(random.getstate())
                    self.assertEqual(states[0], states[1])
                    self.assertEqual(populations[1][:-1], populations[0])
                    self.assertIs(populations[1][-1][key], True)
                    code = new.to_code(populations[1][-1])
                    import ast

                    loops = [
                        n
                        for n in ast.walk(ast.parse(code))
                        if isinstance(n, ast.For)
                        and isinstance(n.target, ast.Name)
                        and n.target.id.startswith("fragment_tile")
                    ]
                    self.assertEqual(len(loops), 1)
                    self.assertNotIn("sync_threads", ast.unparse(loops[0]))


class TestFragmentIntegerAtomicEpochsNative(TestCase):
    @skipUnlessCuteAvailable("requires CuTe DSL")
    def test_loop_epochs_tails_zero_trip_and_wrapping(self):
        for width, repeats, threads in ((65, 3, 32), (65, 0, 32), (33, 2, 128)):
            x = (torch.arange(width, device=DEVICE).reshape(1, width) % 11 - 5).int()
            x[:, ::7] = torch.iinfo(torch.int32).max
            expected = torch.full((1, 17), 3, dtype=torch.int32, device=DEVICE)
            used = min(width, repeats * 32)
            expected.scatter_add_(
                1, (torch.arange(used, device=DEVICE) % 17)[None, :], x[:, :used]
            )
            before = x.clone()
            for enabled in (False, True):
                with self.subTest(width=width, repeats=repeats, enabled=enabled):
                    _, actual = code_and_output(
                        _integer_epoch_histogram,
                        (x, repeats, 17, 32),
                        cute_fragment_integer_atomic_epochs=enabled,
                        cute_fragment_register_loads=True,
                        cute_fragment_threads=threads,
                    )
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    torch.testing.assert_close(x, before, rtol=0, atol=0)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _integer_epoch_effect_boundary(x, repeats: int, mode: hl.constexpr):
    out = torch.empty((1, 17), dtype=torch.int32, device=x.device)
    floating_out = torch.empty((1, 17), dtype=torch.float32, device=x.device)
    for row in hl.grid(1):
        histogram = hl.full([17], 3, dtype=torch.int32)
        floating = hl.zeros([17], dtype=torch.float32)
        indices = hl.arange(32) % 17
        hl.atomic_add(histogram, [indices], hl.full([32], 1, dtype=torch.int32))
        for step in range(repeats):
            columns = step * 32 + hl.arange(32)
            value = hl.load(x, [row, columns], extra_mask=columns < x.size(1))
            hl.atomic_add(histogram, [indices], value)
            if mode == "repeat_target":
                hl.atomic_add(histogram, [indices], value)
        for step in range(repeats):
            columns = step * 32 + hl.arange(32)
            value = hl.load(x, [row, columns], extra_mask=columns < x.size(1))
            hl.atomic_add(floating, [indices], value.to(torch.float32))
        out[row, :] = histogram
        floating_out[row, :] = floating
    return out, floating_out


class TestFragmentIntegerAtomicEpochBoundariesCPU(unittest.TestCase):
    def test_incoming_epoch_float_order_and_repeated_target(self):
        import ast

        from test._cute_binding import _cpu_bind
        from test._cute_binding import _forbid_native_compile
        from test._cute_binding import _mock_cuda_unavailable
        from test.cute_population_contracts import _target

        x = (torch.arange(65).reshape(1, 65) % 7 - 3).int()
        for repeats, mode in ((0, "single"), (3, "single"), (3, "repeat_target")):
            with (
                self.subTest(repeats=repeats, mode=mode),
                _mock_cuda_unavailable(),
                _target(),
                _forbid_native_compile(),
            ):
                bound = _cpu_bind(_integer_epoch_effect_boundary, (x, repeats, mode))
                codes = []
                for enabled in (False, True):
                    config = bound.config_spec.default_config()
                    config.config.update(
                        cute_fragment_integer_atomic_epochs=enabled,
                        cute_fragment_register_loads=True,
                        cute_fragment_threads=32,
                    )
                    codes.append(bound.to_code(config))
                loops = [
                    [
                        n
                        for n in ast.walk(ast.parse(code))
                        if isinstance(n, ast.For)
                        and isinstance(n.target, ast.Name)
                        and n.target.id.startswith("fragment_tile")
                    ]
                    for code in codes
                ]
                self.assertEqual(len(loops[0]), 2)
                self.assertEqual(ast.dump(loops[0][1]), ast.dump(loops[1][1]))
                if mode == "repeat_target":
                    self.assertEqual(
                        ast.dump(ast.parse(codes[0])), ast.dump(ast.parse(codes[1]))
                    )
                else:
                    self.assertIn("sync_threads", ast.unparse(loops[0][0]))
                    self.assertNotIn("sync_threads", ast.unparse(loops[1][0]))
                for code in codes:
                    tree = ast.parse(code)
                    fn = next(
                        n
                        for n in tree.body
                        if isinstance(n, ast.FunctionDef)
                        and n.name.startswith("_helion")
                    )
                    index = next(
                        i
                        for i, n in enumerate(fn.body)
                        if isinstance(n, ast.For)
                        and isinstance(n.target, ast.Name)
                        and n.target.id.startswith("fragment_tile")
                    )
                    self.assertEqual(
                        ast.unparse(fn.body[index - 1]), "cute.arch.sync_threads()"
                    )
                    out = torch.full((1, 17), -999, dtype=torch.int32)
                    floating_out = torch.full((1, 17), -999.0)
                    _simulate_register_load_program(
                        code,
                        x,
                        32,
                        host_tensors={"out": out, "floating_out": floating_out},
                        scalar_args={"repeats": repeats},
                    )
                    expected = torch.full_like(out, 3)
                    idx = (torch.arange(32) % 17)[None, :]
                    expected.scatter_add_(
                        1, idx, torch.ones((1, 32), dtype=torch.int32)
                    )
                    floating = torch.zeros_like(floating_out)
                    for step in range(repeats):
                        values = torch.zeros((1, 32), dtype=torch.int32)
                        part = x[:, step * 32 : (step + 1) * 32]
                        values[:, : part.size(1)] = part
                        expected.scatter_add_(
                            1, idx, values * (2 if mode == "repeat_target" else 1)
                        )
                        floating.scatter_add_(1, idx, values.float())
                    torch.testing.assert_close(out, expected, rtol=0, atol=0)
                    torch.testing.assert_close(floating_out, floating, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()


def _skip_zero_codegen(kernel, args, enabled, threads=32, **options):
    from test._cute_binding import _cpu_bind
    from test._cute_binding import _forbid_native_compile
    from test._cute_binding import _mock_cuda_unavailable
    from test.cute_population_contracts import _target

    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound = _cpu_bind(kernel, args)
        config = bound.config_spec.default_config()
        config.config.update(
            cute_fragment_skip_zero_atomics=enabled,
            cute_fragment_threads=threads,
            **options,
        )
        return bound.to_code(config)


def _skip_zero_reference(x, indices, bins, initial, repeats):
    result = torch.full((x.size(0), bins), initial, dtype=torch.int32, device=x.device)
    wrapped = torch.where(indices < 0, indices + bins, indices).long()
    valid = (wrapped >= 0) & (wrapped < bins)
    for iteration in range(repeats):
        # Mask before converting: invalid positions may contain NaNs.
        value = torch.where(valid, x + iteration, 0).to(torch.int32)
        result.scatter_add_(1, wrapped.clamp(0, bins - 1), value)
    return result


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _dead_zero_tickets(x, indices, mode: hl.constexpr, initial: hl.constexpr):
    rows, width = x.shape
    out = torch.empty(x.shape, device=x.device, dtype=torch.int32)
    second = torch.empty(x.shape, device=x.device, dtype=torch.int32)
    totals = torch.empty((rows, 17), device=x.device, dtype=torch.int32)
    for row in hl.grid(rows):
        column = hl.arange(width)
        hl.store(out, [row, column], -99)
        hl.store(second, [row, column], -99)
        valid = column < width - 7
        values = hl.load(x, [row, column], extra_mask=valid)
        index = indices[row, column]
        above = valid & (values > 0)
        tied = valid & (values == 0)
        active = above | tied
        counters = hl.full([17], initial, dtype=torch.int32)
        if mode != "gated":
            # Keeps the pre-existing unused-zero option eligible in negatives.
            hl.atomic_add(counters, [0], 0)
        update = active.to(torch.int32)
        if mode == "numeric":
            update = values.to(torch.int32)
        ticket = hl.atomic_add(counters, [index], update)
        chosen = above | (tied & (ticket < 7))
        if mode == "lost_mask":
            chosen = above | (ticket < 7)
        elif mode == "flip":
            ticket = torch.flip(ticket, [0])
        elif mode == "divide":
            ticket = 17 // (ticket + 1)
        hl.store(out, [row, column], ticket + 3, extra_mask=chosen)
        if mode == "observe":
            hl.store(second, [row, column], ticket)
        else:
            hl.store(second, [row, column], ticket ^ 85, extra_mask=chosen & active)
        hl.store(totals, [row, hl.arange(17)], counters)
    return out, second, totals


class TestFragmentDeadZeroResultsCPU(unittest.TestCase):
    def test_existing_register_snapshot_storage_composes(self):
        x = (torch.arange(256).reshape(2, 128) % 5 - 2).float()
        indices = (torch.arange(128) % 17).repeat(2, 1).int()
        records = []
        for enabled in (False, True):
            code = _skip_zero_codegen(
                _dead_zero_tickets,
                (x, indices, "gated", 3),
                enabled,
                cute_fragment_local_atomic_registers=True,
            )
            self.assertIn("fragment_local_tickets", code)
            hosts = {
                "indices": indices,
                "out": torch.full_like(indices, -99),
                "second": torch.full_like(indices, -99),
                "totals": torch.empty((2, 17), dtype=torch.int32),
            }
            _, barriers = _simulate_register_load_program(
                code, x, 32, host_tensors=hosts, lane_order=list(reversed(range(32)))
            )
            records.append((hosts, barriers))
        for key in ("out", "second", "totals"):
            torch.testing.assert_close(
                records[0][0][key], records[1][0][key], rtol=0, atol=0
            )
        self.assertEqual(records[0][1], records[1][1])

    def test_masked_return_observations_and_coupled_orders(self):
        import numpy as np

        for width, threads in ((32, 32), (128, 128), (256, 512)):
            for initial in (3, 2147483646):
                for all_dead in (False, True):
                    with self.subTest(width=width, initial=initial, all_dead=all_dead):
                        x = (torch.arange(2 * width).reshape(2, width) % 5 - 2).float()
                        if all_dead:
                            x.fill_(-1)
                        x[:, ::11] = float("nan")
                        x[:, -7:] = float("nan")
                        indices = (torch.arange(width) % 17).repeat(2, 1).int()
                        indices[:, ::2] -= 17
                        indices[:, 1::13] = -18
                        before = x.clone()
                        codes = [
                            _skip_zero_codegen(
                                _dead_zero_tickets,
                                (x, indices, "gated", initial),
                                flag,
                                threads,
                            )
                            for flag in (False, True)
                        ]
                        self.assertIn("fragment_atomic_nonzero_value", codes[1])
                        for order in (
                            list(range(threads)),
                            list(reversed(range(threads))),
                        ):
                            records = []
                            for code in codes:
                                hosts = {
                                    "indices": indices,
                                    "out": torch.full_like(indices, -99),
                                    "second": torch.full_like(indices, -99),
                                    "totals": torch.full(
                                        (2, 17), -99, dtype=torch.int32
                                    ),
                                }
                                events = []
                                with np.errstate(over="ignore"):
                                    _, barriers = _simulate_register_load_program(
                                        code,
                                        x,
                                        threads,
                                        host_tensors=hosts,
                                        lane_order=order,
                                        atomic_events=events,
                                    )
                                records.append((hosts, events, barriers))
                            for name in ("out", "second", "totals"):
                                torch.testing.assert_close(
                                    records[0][0][name],
                                    records[1][0][name],
                                    rtol=0,
                                    atol=0,
                                )
                            self.assertEqual(
                                [e for e in records[0][1] if e[-1] != 0], records[1][1]
                            )
                            self.assertEqual(records[0][2], records[1][2])
                            if all_dead:
                                self.assertFalse(records[1][1])
                                self.assertTrue(torch.all(records[1][0]["out"] == -99))
                        torch.testing.assert_close(
                            x, before, rtol=0, atol=0, equal_nan=True
                        )

    def test_observed_remapped_unsafe_and_nonboolean_returns_stay(self):
        for mode in ("observe", "lost_mask", "flip", "divide", "numeric"):
            with self.subTest(mode=mode):
                x = (torch.arange(64).reshape(2, 32) % 5 - 2).float()
                indices = torch.zeros_like(x, dtype=torch.int32)
                code = _skip_zero_codegen(
                    _dead_zero_tickets, (x, indices, mode, 3), True
                )
                # Exactly the unrelated unused scalar atomic gets a guard.
                self.assertEqual(code.count("!= cutlass.Int32(0)"), 1)

    def test_boolean_implication_does_not_assume_ticket_value(self):
        from torch.fx import Graph

        from helion._compiler.cute.dead_zero_atomics import _masks_imply_update

        graph = Graph()

        def value(name):
            node = graph.placeholder(name)
            node.meta["val"] = torch.empty((17,), dtype=torch.bool)
            return node

        def call(target, *args):
            node = graph.call_function(target, args)
            node.meta["val"] = torch.empty((17,), dtype=torch.bool)
            return node

        a, b, ticket_test = value("a"), value("b"), value("ticket_test")
        update = call(torch.ops.aten.bitwise_or.Tensor, a, b)
        chosen = call(
            torch.ops.aten.bitwise_or.Tensor,
            a,
            call(torch.ops.aten.bitwise_and.Tensor, b, ticket_test),
        )
        self.assertTrue(_masks_imply_update(update, [chosen]))
        self.assertFalse(_masks_imply_update(update, [chosen, ticket_test]))
        self.assertFalse(_masks_imply_update(a, [chosen]))
        for i in range(13):
            chosen = call(
                torch.ops.aten.bitwise_and.Tensor, chosen, value(f"unknown{i}")
            )
        self.assertFalse(_masks_imply_update(update, [chosen]))

    def test_fractional_converted_zero_return_remains_observable(self):
        x = torch.full((2, 32), 0.25)
        indices = torch.zeros_like(x, dtype=torch.int32)
        code = _skip_zero_codegen(_dead_zero_tickets, (x, indices, "numeric", 3), True)
        hosts = {
            "indices": indices,
            "out": torch.full_like(indices, -99),
            "second": torch.full_like(indices, -99),
            "totals": torch.full((2, 17), -99, dtype=torch.int32),
        }
        events = []
        _simulate_register_load_program(
            code, x, 32, host_tensors=hosts, atomic_events=events
        )
        self.assertEqual(len(events), 64)
        self.assertTrue(torch.all(hosts["out"][:, :25] == 6))
        self.assertTrue(torch.all(hosts["out"][:, 25:] == -99))
        self.assertTrue(torch.all(hosts["totals"] == 3))

    def test_missing_readonly_proof_and_unmatched_shape_decline(self):
        from unittest.mock import patch

        from test._cute_binding import _cpu_bind
        from test._cute_binding import _forbid_native_compile
        from test._cute_binding import _mock_cuda_unavailable
        from test.cute_population_contracts import _target

        from helion._compiler.cute.dead_zero_atomics import dead_zero_atomic_results

        for width in (17, 32):
            x = torch.ones((2, width))
            args = (x, torch.zeros_like(x, dtype=torch.int32), "gated", 3)
            with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
                if width == 17:
                    with self.assertRaises(helion.exc.TorchOpTracingError):
                        _cpu_bind(_dead_zero_tickets, args)
                    continue
                bound = _cpu_bind(_dead_zero_tickets, args)
                with bound.env, bound.host_function:
                    graphs = bound.host_function.device_ir.graphs
                    self.assertEqual(
                        bool(dead_zero_atomic_results(graphs, bound.env)), width == 32
                    )
                    with patch(
                        "helion._compiler.cute.dead_zero_atomics.host_load_is_readonly",
                        return_value=False,
                    ):
                        self.assertFalse(dead_zero_atomic_results(graphs, bound.env))


class TestFragmentDeadZeroResultsNative(TestCase):
    @onlyBackends(["cute"])
    def test_masked_tickets_shared_and_register_storage(self):
        for width, threads in ((32, 32), (128, 128), (256, 512)):
            x = torch.ones((2, width), device=DEVICE)
            x[:, ::3] = -1
            before = x.clone()
            indices = torch.zeros_like(x, dtype=torch.int32)
            active = (x > 0) & (torch.arange(width, device=DEVICE) < width - 7)
            bound = _dead_zero_tickets.bind((x, indices, "gated", 3))
            for registers in (False, True):
                for enabled in (False, True):
                    config = bound.config_spec.default_config()
                    config.config.update(
                        cute_fragment_threads=threads,
                        cute_fragment_skip_zero_atomics=enabled,
                        cute_fragment_local_atomic_registers=registers,
                    )
                    out, second, totals = bound.compile_config(config)(
                        x, indices, "gated", 3
                    )
                    expected_totals = torch.full_like(totals, 3)
                    expected_totals[:, 0] += active.sum(1).int()
                    torch.testing.assert_close(totals, expected_totals, rtol=0, atol=0)
                    for row in range(2):
                        count = int(active[row].sum())
                        expected = torch.arange(
                            6, 6 + count, device=DEVICE, dtype=torch.int32
                        )
                        torch.testing.assert_close(
                            out[row][active[row]].sort().values,
                            expected,
                            rtol=0,
                            atol=0,
                        )
                    self.assertTrue(torch.all(out[~active] == -99))
                    self.assertTrue(torch.all(second[~active] == -99))
                    torch.testing.assert_close(
                        second[active], (out[active] - 3) ^ 85, rtol=0, atol=0
                    )
                    torch.testing.assert_close(x, before, rtol=0, atol=0)


class TestFragmentSkipZeroAtomicsCPU(unittest.TestCase):
    def test_signed_cast_wrap_epochs_and_zero_events(self):
        import numpy as np

        for dtype, width, threads in (
            (torch.int32, 17, 32),
            (torch.int64, 65, 128),
            (torch.float32, 129, 512),
        ):
            for all_zero in (False, True):
                with self.subTest(dtype=dtype, width=width, zero=all_zero):
                    x = (torch.arange(2 * width).reshape(2, width) % 7 - 3).to(dtype)
                    if dtype == torch.int64:
                        x[:, ::3] += 2**40
                    elif dtype == torch.float32:
                        x += 0.75
                    else:
                        x[:, ::5] = torch.iinfo(torch.int32).max
                        x[:, 1::7] = torch.iinfo(torch.int32).min
                    if all_zero:
                        x.zero_()
                    indices = (torch.arange(width) % 17).repeat(2, 1).int()
                    indices[:, ::2] -= 17
                    args = (x, indices, 17, 3, 2)
                    expected = _skip_zero_reference(*args)
                    codes = [
                        _skip_zero_codegen(
                            _fragment_aggregated_histogram, args, flag, threads
                        )
                        for flag in (False, True)
                    ]
                    for order in (list(range(threads)), list(reversed(range(threads)))):
                        records = []
                        for code in codes:
                            out = torch.full_like(expected, -99)
                            events = []
                            with np.errstate(over="ignore"):
                                _, barriers = _simulate_register_load_program(
                                    code,
                                    x,
                                    threads,
                                    host_tensors={"indices": indices, "out": out},
                                    lane_order=order,
                                    atomic_events=events,
                                )
                            torch.testing.assert_close(out, expected, rtol=0, atol=0)
                            records.append((events, barriers))
                        self.assertEqual(
                            [e for e in records[0][0] if e[-1] != 0], records[1][0]
                        )
                        self.assertEqual(records[0][1], records[1][1])

    def test_masked_nan_never_reaches_conversion(self):
        import warnings

        for mode in ("logical_tail", "invalid_index"):
            for flag in (False, True):
                with self.subTest(mode=mode, flag=flag):
                    x = torch.ones((2, 17), dtype=torch.float32)
                    hosts = {}
                    expected = torch.ones((2, 17), dtype=torch.int32)
                    if mode == "logical_tail":
                        kernel, args = _fragment_aggregation_masked_tail, (x,)
                    else:
                        indices = torch.arange(17).repeat(2, 1).int()
                        indices[:, 0] = 17
                        indices[:, 1] = -18
                        x[:, :2] = float("nan")
                        expected[:, :2] = 0
                        hosts["indices"] = indices
                        kernel, args = (
                            _fragment_aggregated_histogram,
                            (x, indices, 17, 0, 1),
                        )
                    source = _skip_zero_codegen(kernel, args, flag)
                    out = torch.full_like(expected, -99)
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", RuntimeWarning)
                        _simulate_register_load_program(
                            source, x, 32, host_tensors={"out": out, **hosts}
                        )
                    torch.testing.assert_close(out, expected, rtol=0, atol=0)

    def test_global_ordered_and_float_zero_operations_are_retained(self):
        for sem in ("relaxed", "acquire", "release", "acq_rel"):
            with self.subTest(sem=sem):
                x = torch.zeros((2, 17), dtype=torch.int32)
                counter = torch.full((2,), 9, dtype=torch.int32)
                tickets = torch.full_like(counter, -1)
                out = torch.full_like(x, -1)
                floating_out = torch.full_like(x, -1, dtype=torch.float32)
                source = _skip_zero_codegen(
                    _fragment_aggregation_mixed, (x, counter, tickets, sem), True
                )
                events, memory = [], []
                _simulate_register_load_program(
                    source,
                    x,
                    32,
                    host_tensors={
                        "counter": counter,
                        "tickets": tickets,
                        "out": out,
                        "floating_out": floating_out,
                    },
                    atomic_events=events,
                    memory_events=memory,
                )
                self.assertEqual(sum(e[0] == "gpu" for e in events), 2)
                self.assertEqual(sum(e[0] == "cta" for e in events), 34)
                self.assertEqual(
                    [e[4] for e in memory if e[0] == "atomic" and e[-1] == "gpu"],
                    [sem, sem],
                )
                self.assertTrue(torch.equal(counter, tickets))
                self.assertFalse(out.any())
                self.assertFalse(floating_out.any())

    def test_returned_zeros_and_snapshot_consumers_remain(self):
        for flag in (False, True):
            args = _local_fetch_args(17, "epochs", initial=3, value=0)
            source = _skip_zero_codegen(_fragment_local_fetch_add, args, flag)
            out = torch.full((2, 17), -99, dtype=torch.int32)
            events = []
            _simulate_register_load_program(
                source,
                args[0],
                32,
                host_tensors={
                    "tickets": args[1],
                    "reused": args[2],
                    "reversed_tickets": args[3],
                    "out": out,
                },
                atomic_events=events,
            )
            self.assertEqual(len(events), 68 if flag else 102)
            self.assertTrue(torch.all(args[1] == 3))
            self.assertTrue(torch.all(args[2] == (3 ^ 85)))
            self.assertTrue(torch.all(args[3] == 3))
            self.assertTrue(torch.all(out == 3))

    def test_aggregation_composes_and_excluded_roots_reject(self):
        x = torch.zeros((1, 17), dtype=torch.int32)
        args = (x, torch.zeros_like(x), 17, 0, 2)
        sources = [
            _skip_zero_codegen(
                _fragment_aggregated_histogram,
                args,
                flag,
                cute_fragment_atomic_aggregation=True,
            )
            for flag in (False, True)
        ]
        self.assertEqual(sources[0], sources[1])
        for mode in (
            "early_scan",
            "later_update",
            "loop_read",
            "loop_allocation",
            "conditional",
            "alias",
            "self_derived_update",
        ):
            with self.subTest(mode=mode), self.assertRaises(helion.exc.InvalidConfig):
                _skip_zero_codegen(_local_histogram_consumer_negative, (x, mode), True)
        with self.assertRaises(helion.exc.InvalidConfig):
            _skip_zero_codegen(
                _local_histogram_consumers, (x.float(), 17, "pointwise"), True
            )
        args = _local_fetch_args(17, "negative")
        with self.assertRaises(helion.exc.InvalidConfig):
            _skip_zero_codegen(_fragment_local_fetch_add, args, True)

    def test_default_normalization_and_deferred_coverage(self):
        from copy import deepcopy
        from unittest.mock import patch

        from test._cute_binding import _cpu_bind
        from test._cute_binding import _forbid_native_compile
        from test._cute_binding import _mock_cuda_unavailable
        from test.cute_population_contracts import _target

        key = "cute_fragment_skip_zero_atomics"
        x = torch.ones((1, 17), dtype=torch.int32)
        args = (x, torch.zeros_like(x), 17, 0, 1)
        with (
            _mock_cuda_unavailable(),
            _target(),
            _forbid_native_compile(),
            patch(
                "helion._compiler.autotuner_heuristics.register_fragment_integer_atomic_epochs_coverage"
            ),
        ):
            bound = _cpu_bind(_fragment_aggregated_histogram, args)
            default = bound.config_spec.default_config()
            self.assertNotIn(key, default)
            self.assertTrue(
                all(key not in seed for seed in bound.config_spec.compiler_seed_configs)
            )
            off = deepcopy(default)
            off.config[key] = False
            self.assertEqual(bound.to_code(default), bound.to_code(off))
            group = bound.config_spec.compiler_coverage_groups[-1]
            self.assertEqual(group.key, key)
            self.assertTrue(group.deferred)
            self.assertFalse(group.legacy)
            for value in (1, "zero", None):
                config = deepcopy(default)
                config.config[key] = value
                with self.assertRaises(helion.exc.InvalidConfig):
                    bound.to_code(config)


@onlyBackends("cute")
class TestFragmentSkipZeroAtomicsNative(TestCase):
    def test_zero_signed_and_cast_updates(self):
        for dtype, width, mode in (
            (torch.int32, 17, "wrap"),
            (torch.int64, 65, "wide"),
            (torch.float32, 129, "fraction"),
            (torch.int32, 65, "zero"),
        ):
            with self.subTest(dtype=dtype, width=width, mode=mode):
                x = (
                    torch.arange(2 * width, device=DEVICE).reshape(2, width) % 7 - 3
                ).to(dtype)
                if mode == "wrap":
                    x[:, ::3] = torch.iinfo(torch.int32).max
                    x[:, 1::7] = torch.iinfo(torch.int32).min
                elif mode == "wide":
                    x[:, ::3] += 2**40
                elif mode == "fraction":
                    x += 0.75
                else:
                    x.zero_()
                indices = (torch.arange(width, device=DEVICE) % 17).repeat(2, 1).int()
                indices[:, ::2] -= 17
                args = (x, indices, 17, 3, 2)
                expected = _skip_zero_reference(*args)
                before = (x.clone(), indices.clone())
                for enabled in (False, True):
                    _, out = code_and_output(
                        _fragment_aggregated_histogram,
                        args,
                        cute_fragment_skip_zero_atomics=enabled,
                        cute_fragment_threads=32,
                    )
                    torch.testing.assert_close(out, expected, rtol=0, atol=0)
                    torch.testing.assert_close(x, before[0], rtol=0, atol=0)
                    torch.testing.assert_close(indices, before[1], rtol=0, atol=0)

    def test_nan_logical_tail_and_invalid_index(self):
        for mode in ("logical_tail", "invalid_index"):
            x = torch.ones((2, 17), dtype=torch.float32, device=DEVICE)
            if mode == "logical_tail":
                kernel, args = _fragment_aggregation_masked_tail, (x,)
                expected = torch.ones((2, 17), dtype=torch.int32, device=DEVICE)
            else:
                indices = torch.arange(17, device=DEVICE).repeat(2, 1).int()
                indices[:, 0] = 17
                indices[:, 1] = -18
                x[:, :2] = float("nan")
                kernel, args = _fragment_aggregated_histogram, (x, indices, 17, 0, 1)
                expected = _skip_zero_reference(*args)
            before = x.clone()
            for enabled in (False, True):
                _, out = code_and_output(
                    kernel,
                    args,
                    cute_fragment_skip_zero_atomics=enabled,
                    cute_fragment_threads=32,
                )
                torch.testing.assert_close(out, expected, rtol=0, atol=0)
                torch.testing.assert_close(x, before, rtol=0, atol=0, equal_nan=True)

    def test_global_ordered_float_and_local_returned_zeros(self):
        for sem in ("relaxed", "acquire", "release", "acq_rel"):
            x = torch.zeros((2, 17), dtype=torch.int32, device=DEVICE)
            counter = torch.full((2,), 9, dtype=torch.int32, device=DEVICE)
            tickets = torch.full_like(counter, -1)
            _, (out, floating) = code_and_output(
                _fragment_aggregation_mixed,
                (x, counter, tickets, sem),
                cute_fragment_skip_zero_atomics=True,
                cute_fragment_threads=32,
            )
            torch.testing.assert_close(
                counter, torch.full_like(counter, 9), rtol=0, atol=0
            )
            torch.testing.assert_close(tickets, counter, rtol=0, atol=0)
            torch.testing.assert_close(out, torch.zeros_like(out), rtol=0, atol=0)
            torch.testing.assert_close(
                floating, torch.zeros_like(floating), rtol=0, atol=0
            )
        args = _local_fetch_args(17, "epochs", initial=3, value=0, device=DEVICE)
        _, out = code_and_output(
            _fragment_local_fetch_add,
            args,
            cute_fragment_skip_zero_atomics=True,
            cute_fragment_threads=32,
        )
        torch.testing.assert_close(out, torch.full_like(out, 3), rtol=0, atol=0)
        for actual, value in zip(args[1:4], (3, 3 ^ 85, 3), strict=True):
            torch.testing.assert_close(
                actual, torch.full_like(actual, value), rtol=0, atol=0
            )
