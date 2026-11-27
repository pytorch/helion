from __future__ import annotations

import unittest
from unittest.mock import patch

import pytest
import torch

import helion
from helion import _compat
from helion._testing import DEVICE
from helion._testing import HALF_DTYPE
from helion._testing import RefEagerTestBase
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import matchesBackends
from helion._testing import skipIfMetal
from helion._testing import skipIfTileIR
from helion._testing import skipIfXPU
from helion._testing import skipUnlessBackends
from helion._testing import skipUnlessTensorDescriptor
from helion._testing import xfailIfPallas
import helion.language as hl


@skipUnlessBackends(["triton"])
@skipIfTileIR("TileIR does not expose the static_ranges configuration slot")
def test_saved_static_range_keeps_its_single_write():
    @helion.kernel(
        backend="triton",
        static_shapes=True,
        index_dtype=torch.int32,
        config=helion.Config(block_sizes=[], static_ranges=[True]),
    )
    def write(out):
        for row in hl.grid(out.size(0)):
            for i in hl.grid(2**31 - 2, 2**31, 2):
                out[row, i & 3] = row + 1

    out = torch.full((2, 4), -73, dtype=torch.int32, device=DEVICE)
    write(out)
    assert out.tolist() == [[-73, -73, 1, -73], [-73, -73, 2, -73]]


@skipUnlessBackends(["cute"])
def test_ragged_dynamic_tile_end():
    @helion.kernel(
        backend="cute", static_shapes=False, config=helion.Config(block_sizes=[])
    )
    def write(out):
        for tile in hl.tile(out.size(0), block_size=4):
            out[tile] = tile.end

    out = torch.zeros(9, device=DEVICE, dtype=torch.int32)
    write(out)
    assert out.tolist() == [4] * 4 + [8] * 4 + [9]


@skipUnlessBackends(["cute"])
def test_flattened_tile_mask_across_int32():
    @helion.kernel(
        backend="cute",
        static_shapes=True,
        config=helion.Config(block_sizes=[4], num_threads=[2]),
    )
    def write(out):
        for tile in hl.tile(2**31 - 6, 2**31 - 1):
            out[(tile.index.to(torch.int64) - (2**31 - 6)) % 8] = 1

    out = torch.zeros(8, device=DEVICE, dtype=torch.int32)
    write(out)
    assert out.tolist() == [1] * 5 + [0] * 3


@skipUnlessBackends(["cute"])
def test_explicit_tile_masks_padded_int64_lanes_by_their_relative_offsets():
    @helion.kernel(
        backend="cute", static_shapes=True, config=helion.Config(block_sizes=[])
    )
    def write(out):
        for tile in hl.tile(2**63 - 6, 2**63 - 1, block_size=4):
            out[(tile.index.to(torch.int64) - (2**63 - 6)) % 8] = 1

    out = torch.zeros(8, device=DEVICE, dtype=torch.int32)
    write(out)
    assert out.tolist() == [1] * 5 + [0] * 3


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("literal_start", [False, True])
def test_shifted_dynamic_tile_masks_wrapped_int64_lanes(literal_start):
    @helion.kernel(
        backend="cute",
        static_shapes=False,
        config=helion.Config(block_sizes=[4], num_threads=[2]),
    )
    def write(out, start, stop, literal: hl.constexpr):
        origin = 2**63 - 6 if literal else start
        for tile in hl.tile(origin, stop):
            out[(tile.index.to(torch.int64) - origin) % 8] = 1

    out = torch.zeros(8, device=DEVICE, dtype=torch.int32)
    write(out, 2**63 - 6, 2**63 - 1, literal_start)
    assert out.tolist() == [1] * 5 + [0] * 3


@skipUnlessBackends(["triton", "cute"])
@pytest.mark.parametrize("bound_type", ["uint32", "uint64", "int64", "python"])
def test_shifted_tile_with_unsigned_endpoint(bound_type):
    @helion.kernel(
        static_shapes=False,
        config=helion.Config(block_sizes=[], pid_type="persistent_blocked"),
    )
    def fill(out, bound, wide: hl.constexpr):
        if wide:
            start, stop = bound, 2**63
        else:
            start, stop = -1, bound
        for tile in hl.tile(start, stop, block_size=4):
            out[tile.index & 7] = tile.end // 2

    wide = bound_type != "uint32"
    start, stop = (2**63 - 5, 2**63) if wide else (-1, 4)
    bound = start
    if bound_type != "python":
        bound = torch.tensor(
            start if wide else stop,
            dtype=getattr(torch, bound_type),
            device=DEVICE,
        )
    out = torch.full((8,), -73, dtype=torch.int64, device=DEVICE)
    if wide and matchesBackends(["tileir"]):
        # TileIR's grid coordinates must fit signed Int64 before a division.
        with pytest.raises(helion.exc.InvalidConfig, match="unbounded UInt64"):
            fill(out, bound, wide)
        return
    fill(out, bound, wide)
    expected = [-73] * 8
    for base in range(start, stop, 4):
        end = min(base + 4, stop)
        for i in range(base, end):
            expected[i % 8] = end // 2
    assert out.tolist() == expected


@skipUnlessBackends(["cute"])
def test_nd_shifted_dynamic_tile_masks_wrapped_int64_lanes():
    @helion.kernel(
        backend="cute",
        static_shapes=False,
        config=helion.Config(block_sizes=[4, 2], num_threads=[2, 1]),
    )
    def write(out, origin):
        for row, col in hl.tile([origin, 0], [origin + 5, 2]):
            out[(row.index.to(torch.int64) - origin) % 8, col] = 1

    out = torch.zeros(8, 2, device=DEVICE, dtype=torch.int32)
    write(out, 2**63 - 6)
    assert out.tolist() == [[1, 1]] * 5 + [[0, 0]] * 3


def grid_2d_pytorch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    bi, bj, m, k = x.size()
    k2, n = y.size()
    assert k == k2, f"size mismatch {k} != {k2}"
    out = torch.empty(
        bi,
        bj,
        m,
        n,
        dtype=torch.promote_types(x.dtype, y.dtype),
        device=x.device,
    )
    for i in range(bi):
        for j in range(bj):
            out[i, j] = torch.mm(x[i, j], y)
    return out


class TestGrid(RefEagerTestBase, TestCase):
    @skipUnlessTensorDescriptor("Tensor descriptor support is required")
    @patch.object(_compat, "_min_dot_size", lambda *args: (16, 16, 16))
    @skipIfXPU("Timeout on XPU")
    def test_grid_1d(self):
        @helion.kernel(static_shapes=True)
        def grid_1d(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            b, m, k = x.size()
            k2, n = y.size()
            assert k == k2, f"size mismatch {k} != {k2}"
            out = torch.empty(
                b, m, n, dtype=torch.promote_types(x.dtype, y.dtype), device=x.device
            )
            for i in hl.grid(b):
                for tile_m, tile_n in hl.tile([m, n]):
                    acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                    for tile_k in hl.tile(k):
                        acc = torch.addmm(acc, x[i, tile_m, tile_k], y[tile_k, tile_n])
                    out[i, tile_m, tile_n] = acc
            return out

        def grid_1d_pytorch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            b, m, k = x.size()
            k2, n = y.size()
            assert k == k2, f"size mismatch {k} != {k2}"
            out = torch.empty(
                b, m, n, dtype=torch.promote_types(x.dtype, y.dtype), device=x.device
            )
            for i in range(b):
                out[i] = torch.mm(x[i], y)
            return out

        args = (
            torch.randn([8, 16, 32], device=DEVICE, dtype=HALF_DTYPE),
            torch.randn([32, 4], device=DEVICE, dtype=HALF_DTYPE),
        )
        code, result = code_and_output(grid_1d, args)
        torch.testing.assert_close(result, grid_1d_pytorch(args[0], args[1]))

        # test again with block_ptr indexing
        code, result = code_and_output(
            grid_1d, args, block_sizes=[16, 16, 16], indexing="tensor_descriptor"
        )
        torch.testing.assert_close(result, grid_1d_pytorch(args[0], args[1]))

    @skipUnlessTensorDescriptor("Tensor descriptor support is required")
    @skipIfXPU("XPU tensor descriptor path has accuracy issue for this grid test")
    def test_grid_2d_idx_list(self):
        @helion.kernel(static_shapes=True)
        def grid_2d_idx_list(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            bi, bj, m, k = x.size()
            k2, n = y.size()
            assert k == k2, f"size mismatch {k} != {k2}"
            out = torch.empty(
                bi,
                bj,
                m,
                n,
                dtype=torch.promote_types(x.dtype, y.dtype),
                device=x.device,
            )
            for i, j in hl.grid([bi, bj]):
                for tile_m, tile_n in hl.tile([m, n]):
                    acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                    for tile_k in hl.tile(k):
                        acc = torch.addmm(
                            acc, x[i, j, tile_m, tile_k], y[tile_k, tile_n]
                        )
                    out[i, j, tile_m, tile_n] = acc
            return out

        args = (
            torch.randn([3, 4, 64, 32], device=DEVICE, dtype=HALF_DTYPE),
            torch.randn([32, 16], device=DEVICE, dtype=HALF_DTYPE),
        )

        code, result = code_and_output(grid_2d_idx_list, args)
        torch.testing.assert_close(result, grid_2d_pytorch(args[0], args[1]))

        code, result = code_and_output(
            grid_2d_idx_list,
            args,
            block_sizes=[64, 16, 16],
            indexing="tensor_descriptor",
        )
        torch.testing.assert_close(result, grid_2d_pytorch(args[0], args[1]))

    @skipIfMetal("aten.addmm not yet registered for Metal backend")
    def test_grid_2d_idx_nested(self):
        @helion.kernel(static_shapes=True)
        def grid_2d_idx_nested(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            bi, bj, m, k = x.size()
            k2, n = y.size()
            assert k == k2, f"size mismatch {k} != {k2}"
            out = torch.empty(
                bi,
                bj,
                m,
                n,
                dtype=torch.promote_types(x.dtype, y.dtype),
                device=x.device,
            )
            for i in hl.grid(bi):
                for j in hl.grid(bj):
                    for tile_m, tile_n in hl.tile([m, n]):
                        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                        for tile_k in hl.tile(k):
                            acc = torch.addmm(
                                acc, x[i, j, tile_m, tile_k], y[tile_k, tile_n]
                            )
                        out[i, j, tile_m, tile_n] = acc
            return out

        args = (
            torch.randn([3, 4, 64, 32], device=DEVICE, dtype=HALF_DTYPE),
            torch.randn([32, 16], device=DEVICE, dtype=HALF_DTYPE),
        )
        # Pin the small (mma.sync) config this test used before the H100 matmul seed
        # heuristic began promoting a [64, 16, 16] default. That default tiles M in a
        # single 64-wide block, which Triton lowers to a Hopper WGMMA (wgmma.mma_async);
        # Triton miscompiles that WGMMA inside this nested-hl.grid sequential loop on sm90
        # (every loop iteration returns iteration 0's result). Pinning keeps this test
        # checking nested-grid correctness, decoupled from the matmul heuristic default.
        code, result = code_and_output(
            grid_2d_idx_nested, args, block_sizes=[16, 16, 16]
        )
        torch.testing.assert_close(result, grid_2d_pytorch(args[0], args[1]))

    @skipIfMetal("BUG: hl.grid begin/end broken with PerThreadNDTileStrategy on Metal")
    def test_grid_begin_end(self):
        @helion.kernel(autotune_effort="none")
        def grid_begin_end(x: torch.Tensor) -> torch.Tensor:
            n = x.size(0)
            out = torch.zeros_like(x)
            for i in hl.grid(2, n - 2):  # grid(begin, end)
                out[i] = x[i] * 2
            return out

        def grid_begin_end_pytorch(x: torch.Tensor) -> torch.Tensor:
            n = x.size(0)
            out = torch.zeros_like(x)
            for i in range(2, n - 2):
                out[i] = x[i] * 2
            return out

        x = torch.randn([16], device=DEVICE, dtype=torch.float32)
        code, result = code_and_output(grid_begin_end, (x,))
        torch.testing.assert_close(result, grid_begin_end_pytorch(x))

    @skipIfMetal("BUG: hl.grid begin/end broken with PerThreadNDTileStrategy on Metal")
    def test_grid_begin_end_step(self):
        @helion.kernel(autotune_effort="none")
        def grid_begin_end_step(x: torch.Tensor) -> torch.Tensor:
            n = x.size(0)
            out = torch.zeros_like(x)
            for i in hl.grid(0, n, 2):  # grid(begin, end, step)
                out[i] = x[i] * 2
            return out

        def grid_begin_end_step_pytorch(x: torch.Tensor) -> torch.Tensor:
            n = x.size(0)
            out = torch.zeros_like(x)
            for i in range(0, n, 2):
                out[i] = x[i] * 2
            return out

        x = torch.randn([16], device=DEVICE, dtype=torch.float32)
        code, result = code_and_output(grid_begin_end_step, (x,))
        torch.testing.assert_close(result, grid_begin_end_step_pytorch(x))

    @skipIfMetal("BUG: hl.grid begin/end broken with PerThreadNDTileStrategy on Metal")
    def test_grid_end_step_kwarg(self):
        @helion.kernel(autotune_effort="none")
        def grid_end_step_kwarg(x: torch.Tensor) -> torch.Tensor:
            n = x.size(0)
            out = torch.zeros_like(x)
            for i in hl.grid(n, step=2):  # grid(end, step=step)
                out[i] = x[i] * 2
            return out

        def grid_end_step_kwarg_pytorch(x: torch.Tensor) -> torch.Tensor:
            n = x.size(0)
            out = torch.zeros_like(x)
            for i in range(0, n, 2):
                out[i] = x[i] * 2
            return out

        x = torch.randn([16], device=DEVICE, dtype=torch.float32)
        code, result = code_and_output(grid_end_step_kwarg, (x,))
        torch.testing.assert_close(result, grid_end_step_kwarg_pytorch(x))

    def test_grid_multidim_begin_end(self):
        @helion.kernel(autotune_effort="none")
        def grid_multidim_begin_end(x: torch.Tensor) -> torch.Tensor:
            m, n = x.size()
            out = torch.zeros_like(x)
            for i, j in hl.grid(
                [1, 1], [m - 1, n - 1]
            ):  # multidimensional grid(begin, end)
                out[i, j] = x[i, j] * 2
            return out

        def grid_multidim_begin_end_pytorch(x: torch.Tensor) -> torch.Tensor:
            m, n = x.size()
            out = torch.zeros_like(x)
            for i in range(1, m - 1):
                for j in range(1, n - 1):
                    out[i, j] = x[i, j] * 2
            return out

        x = torch.randn([8, 8], device=DEVICE, dtype=torch.float32)
        code, result = code_and_output(grid_multidim_begin_end, (x,))
        torch.testing.assert_close(result, grid_multidim_begin_end_pytorch(x))

    def test_grid_multidim_begin_end_step(self):
        @helion.kernel(autotune_effort="none")
        def grid_multidim_begin_end_step(x: torch.Tensor) -> torch.Tensor:
            m, n = x.size()
            out = torch.zeros_like(x)
            for i, j in hl.grid(
                [0, 0], [m, n], [2, 3]
            ):  # multidimensional grid(begin, end, step)
                out[i, j] = x[i, j] * 2
            return out

        def grid_multidim_begin_end_step_pytorch(x: torch.Tensor) -> torch.Tensor:
            m, n = x.size()
            out = torch.zeros_like(x)
            for i in range(0, m, 2):
                for j in range(0, n, 3):
                    out[i, j] = x[i, j] * 2
            return out

        x = torch.randn([8, 9], device=DEVICE, dtype=torch.float32)
        code, result = code_and_output(grid_multidim_begin_end_step, (x,))
        torch.testing.assert_close(result, grid_multidim_begin_end_step_pytorch(x))

    def test_tile_begin_end(self):
        @helion.kernel(autotune_effort="none")
        def tile_begin_end(x: torch.Tensor) -> torch.Tensor:
            out = torch.zeros_like(x)
            for tile in hl.tile(2, 10):  # tile(begin, end) - simple range [2, 10)
                out[tile] = x[tile] * 2
            return out

        def tile_begin_end_pytorch(x: torch.Tensor) -> torch.Tensor:
            out = torch.zeros_like(x)
            # Tile should process all indices in range [2, 10) in chunks
            for i in range(2, 10):
                out[i] = x[i] * 2
            return out

        x = torch.randn([15], device=DEVICE, dtype=torch.float32)
        code, result = code_and_output(tile_begin_end, (x,), block_size=4)
        torch.testing.assert_close(result, tile_begin_end_pytorch(x))

    def test_range_as_grid_basic(self):
        """Test that range() works as an alias for hl.grid() in device code."""

        @helion.kernel(autotune_effort="none")
        def range_kernel(x: torch.Tensor) -> torch.Tensor:
            batch = x.size(0)
            out = x.new_zeros(batch)
            for tile_batch in hl.tile(batch):
                for i in range(10):  # This should work now as alias for hl.grid(10)
                    out[tile_batch] += x[tile_batch] + i
            return out

        x = torch.randn(35, device=DEVICE)

        # Reference: sum over i of (x + i) = 10*x + sum(0..9) = 10*x + 45
        expected = 10 * x + 45

        code, result = code_and_output(range_kernel, (x,))
        torch.testing.assert_close(result, expected)

    def test_range_with_begin_end(self):
        """Test that range(begin, end) works as alias for hl.grid(begin, end)."""

        @helion.kernel(autotune_effort="none")
        def range_begin_end_kernel(x: torch.Tensor) -> torch.Tensor:
            batch = x.size(0)
            out = x.new_zeros(batch)
            for tile_batch in hl.tile(batch):
                for i in range(2, 7):  # range(begin, end)
                    out[tile_batch] += x[tile_batch] * i
            return out

        x = torch.randn(20, device=DEVICE)

        # Reference: x * sum(range(2, 7)) = x * sum(2,3,4,5,6) = x * 20
        expected = x * 20

        code, result = code_and_output(range_begin_end_kernel, (x,))
        torch.testing.assert_close(result, expected)

    @xfailIfPallas(
        "range(begin, end, step) lowers to _for_loop_step which has no "
        "emit_pipeline codegen"
    )
    def test_range_with_step(self):
        """Test that range(begin, end, step) works as alias for hl.grid(begin, end, step)."""

        @helion.kernel(autotune_effort="none")
        def range_step_kernel(x: torch.Tensor) -> torch.Tensor:
            batch = x.size(0)
            out = x.new_zeros(batch)
            for tile_batch in hl.tile(batch):
                for i in range(1, 10, 2):  # range(begin, end, step)
                    out[tile_batch] += x[tile_batch] / i
            return out

        x = torch.randn(6, device=DEVICE)

        # Reference: x * sum(1/i for i in range(1, 10, 2)) = x * sum(1/1, 1/3, 1/5, 1/7, 1/9)
        # = x * (1 + 1/3 + 1/5 + 1/7 + 1/9) = x * sum([1, 1/3, 1/5, 1/7, 1/9])
        reciprocal_sum = sum(1.0 / i for i in range(1, 10, 2))
        expected = x * reciprocal_sum

        code, result = code_and_output(range_step_kernel, (x,))
        torch.testing.assert_close(result, expected)

    def test_range_with_tensor_size(self):
        """Test that range(tensor.size(dim)) works with dynamic tensor dimensions."""

        @helion.kernel(autotune_effort="none")
        def range_tensor_size_kernel(x: torch.Tensor) -> torch.Tensor:
            batch = x.size(0)
            out = x.new_zeros(batch)
            for tile_batch in hl.tile(batch):
                for _ in range(x.size(1)):  # Use tensor dimension in range
                    out[tile_batch] += x[tile_batch, 0]  # Just use first column
            return out

        x = torch.randn(8, 5, device=DEVICE)  # 8 rows, 5 columns

        # Reference: Each row adds x[row, 0] for x.size(1) times = x[:, 0] * x.size(1)
        expected = x[:, 0] * x.size(1)

        code, result = code_and_output(range_tensor_size_kernel, (x,))
        torch.testing.assert_close(result, expected)


if __name__ == "__main__":
    unittest.main()
