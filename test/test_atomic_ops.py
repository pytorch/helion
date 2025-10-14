from __future__ import annotations

import unittest

import torch

import helion
from helion._testing import DEVICE
from helion._testing import RefEagerTestBase
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import skipIfRefEager
from helion._testing import skipIfRocm
import helion.language as hl


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


@helion.kernel()
def atomic_add_w_tile_attr(x: torch.Tensor) -> torch.Tensor:
    """Test atomic_add where the index is a symbolic int"""
    y = torch.zeros_like(x, device=x.device, dtype=torch.int32)
    for tile in hl.tile(x.size(0)):
        hl.atomic_add(y, [tile.begin], 1)
    return y


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
        self.assertExpectedJournal(code)

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
        self.assertExpectedJournal(code)

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
        self.assertExpectedJournal(code)

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
        self.assertExpectedJournal(code)

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
        self.assertExpectedJournal(code)

    def test_atomic_add_code_generation(self):
        """Test that the generated code contains atomic_add."""
        x = torch.zeros(10, device=DEVICE)
        y = torch.ones(10, device=DEVICE)
        args = (x, y)

        code, result = code_and_output(atomic_add_kernel, args)
        expected = torch.ones(10, device=DEVICE)
        torch.testing.assert_close(result, expected)
        self.assertIn("atomic_add", code)

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
        self.assertExpectedJournal(code)

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
        self.assertExpectedJournal(code)

    @skipIfRefEager("Error only raises in normal mode")
    def test_atomic_add_device_tensor_error(self):
        @helion.kernel(static_shapes=True, autotune_effort="none")
        def kernel(x: torch.Tensor) -> torch.Tensor:
            for tile in hl.tile(x.size(0), block_size=128):
                device_tensor = hl.zeros([tile], dtype=x.dtype)
                hl.atomic_add(device_tensor, [tile], x[tile])
            return x

        x = torch.ones(256, device=DEVICE, dtype=torch.float32)
        with self.assertRaisesRegex(
            helion.exc.AtomicOnDeviceTensor,
            r"hl\.atomic_add\(\)",
        ):
            kernel(x)

    # New tests for other atomics (correctness only; no journal asserts)
    def test_atomic_and(self):
        x0 = torch.full((8,), 0b1111, device=DEVICE, dtype=torch.int32)
        y = torch.tensor([0b1010] * 8, device=DEVICE, dtype=torch.int32)
        code, result = code_and_output(atomic_and_kernel, (x0.clone(), y))
        expected = torch.full((8,), 0b1111 & 0b1010, device=DEVICE, dtype=torch.int32)
        torch.testing.assert_close(result, expected)
        self.assertExpectedJournal(code)

    def test_atomic_or(self):
        x0 = torch.zeros(8, device=DEVICE, dtype=torch.int32)
        y = torch.tensor([0b1010] * 8, device=DEVICE, dtype=torch.int32)
        code, result = code_and_output(atomic_or_kernel, (x0.clone(), y))
        expected = torch.full((8,), 0b1010, device=DEVICE, dtype=torch.int32)
        torch.testing.assert_close(result, expected)
        self.assertExpectedJournal(code)

    def test_atomic_xor(self):
        x0 = torch.tensor([0b1010] * 8, device=DEVICE, dtype=torch.int32)
        y = torch.tensor([0b1100] * 8, device=DEVICE, dtype=torch.int32)
        code, result = code_and_output(atomic_xor_kernel, (x0.clone(), y))
        expected = torch.full((8,), 0b1010 ^ 0b1100, device=DEVICE, dtype=torch.int32)
        torch.testing.assert_close(result, expected)
        self.assertExpectedJournal(code)

    @skipIfRocm("ROCm backend currently lacks support for these atomics")
    def test_atomic_xchg(self):
        x0 = torch.zeros(8, device=DEVICE, dtype=torch.int32)
        y = torch.arange(8, device=DEVICE, dtype=torch.int32)
        code, result = code_and_output(atomic_xchg_kernel, (x0.clone(), y))
        torch.testing.assert_close(result, y)
        self.assertExpectedJournal(code)

    @skipIfRocm("ROCm backend currently lacks support for these atomics")
    def test_atomic_max(self):
        x = torch.tensor([1, 5, 3, 7], device=DEVICE, dtype=torch.int32)
        y = torch.tensor([4, 2, 9, 1], device=DEVICE, dtype=torch.int32)
        code, result = code_and_output(atomic_max_kernel, (x.clone(), y))
        expected = torch.tensor([4, 5, 9, 7], device=DEVICE, dtype=torch.int32)
        torch.testing.assert_close(result, expected)
        self.assertExpectedJournal(code)

    @skipIfRocm("ROCm backend currently lacks support for these atomics")
    def test_atomic_min(self):
        x = torch.tensor([1, 5, 3, 7], device=DEVICE, dtype=torch.int32)
        y = torch.tensor([4, 2, 9, 1], device=DEVICE, dtype=torch.int32)
        code, result = code_and_output(atomic_min_kernel, (x.clone(), y))
        expected = torch.tensor([1, 2, 3, 1], device=DEVICE, dtype=torch.int32)
        torch.testing.assert_close(result, expected)
        self.assertExpectedJournal(code)

    def test_atomic_cas(self):
        x = torch.tensor([1, 5, 3, 7], device=DEVICE, dtype=torch.int32)
        expect = torch.tensor([1, 6, 3, 0], device=DEVICE, dtype=torch.int32)
        y = torch.tensor([9, 9, 9, 9], device=DEVICE, dtype=torch.int32)
        code, result = code_and_output(atomic_cas_kernel, (x.clone(), y, expect))
        # Only positions where expect matches original x are replaced
        expected = torch.tensor([9, 5, 9, 7], device=DEVICE, dtype=torch.int32)
        torch.testing.assert_close(result, expected)
        self.assertExpectedJournal(code)


if __name__ == "__main__":
    unittest.main()
