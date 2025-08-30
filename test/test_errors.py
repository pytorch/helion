from __future__ import annotations

import unittest

import torch

import helion
from helion._testing import DEVICE
from helion._testing import RefEagerTestDisabled
from helion._testing import TestCase
from helion._testing import code_and_output
import helion.language as hl


class TestErrors(RefEagerTestDisabled, TestCase):
    def test_tile_unpacking(self):
        @helion.kernel()
        def sum_kernel(x: torch.Tensor) -> torch.Tensor:
            batch, seq_len, hidden = x.size()
            out = x.new_empty(batch, hidden)
            for tile_batch, tile_hidden in hl.tile(batch, hidden):
                out[tile_batch, tile_hidden] = x[tile_batch, :, tile_hidden].sum(1)
            return out

        with self.assertRaises(helion.exc.FailedToUnpackTile):
            code_and_output(sum_kernel, (torch.randn(2, 3, 4, device=DEVICE),))

    def test_tile_overpacking(self):
        @helion.kernel()
        def fn(x: torch.Tensor) -> torch.Tensor:
            batch = x.size(0)
            out = x.new_empty(batch)
            for tile_wrapped_in_tuple in hl.tile([batch]):
                out[tile_wrapped_in_tuple] = x[tile_wrapped_in_tuple, :].sum(1)
            return out

        with self.assertRaises(helion.exc.OverpackedTile):
            code_and_output(fn, (torch.randn(100, 100, device=DEVICE),))

    def test_invalid_config_insufficient_block_sizes(self):
        """Test that InvalidConfig shows helpful message for missing block sizes."""

        @helion.kernel(config=helion.Config(block_sizes=[32, 64]))
        def fn(x: torch.Tensor) -> torch.Tensor:
            batch, seq_len, hidden = x.size()
            out = torch.empty_like(x)
            for tile_batch, tile_seq, tile_hidden in hl.tile([batch, seq_len, hidden]):
                out[tile_batch, tile_seq, tile_hidden] = x[
                    tile_batch, tile_seq, tile_hidden
                ]
            return out

        with self.assertRaisesRegex(
            helion.exc.InvalidConfig,
            r"Not enough values for config.*expected 3 block sizes.*got 2.*"
            r"Did you forget to specify block sizes for all your hl\.tile\(\) dimensions\?",
        ):
            code_and_output(
                fn,
                (torch.randn(4, 8, 16, device=DEVICE),),
            )

    def test_rank_mismatch_indexing(self):
        """Test that RankMismatch shows tensor shapes in indexing errors."""

        @helion.kernel()
        def fn(x: torch.Tensor) -> torch.Tensor:
            batch = x.size(0)
            out = x.new_empty(batch)
            for tile_batch in hl.tile([batch]):
                scalar_val = x[tile_batch].sum()  # 1d index for 2d tensor
                out = scalar_val
            return out

        with self.assertRaisesRegex(
            helion.exc.RankMismatch,
            r"Expected ndim=2, but got ndim=1.*You have too few indices",
        ):
            code_and_output(fn, (torch.randn(4, 8, device=DEVICE),))

    def test_rank_mismatch_indexing_too_many(self):
        @helion.kernel()
        def fn(x: torch.Tensor) -> torch.Tensor:
            batch = x.size(0)
            fill = x.new_empty(batch, batch)
            for tile_batch in hl.tile(batch):
                fill = x[tile_batch, tile_batch]  # 2d index for 1d tensor
            return fill

        with self.assertRaisesRegex(
            helion.exc.RankMismatch,
            r"Expected ndim=1, but got ndim=2.*You have too many indices",
        ):
            code_and_output(fn, (torch.randn(8, device=DEVICE),))

    def test_invalid_device_for_loop(self):
        """Test that InvalidDeviceForLoop is raised for invalid for loops on device."""

        @helion.kernel()
        def fn(x: torch.Tensor) -> torch.Tensor:
            batch = x.size(0)
            out = x.new_empty(batch)
            for tile_batch in hl.tile(batch):
                for i in {1: None, 2: None, 3: None}:
                    out[tile_batch] = x[tile_batch] + i
            return out

        with self.assertRaises(helion.exc.InvalidDeviceForLoop):
            code_and_output(fn, (torch.randn(8, device=DEVICE),))

    def test_return_inside_grid_loop(self):
        """Test that return statement inside hl.grid loop raises proper error."""

        @helion.kernel()
        def fn(x: torch.Tensor) -> torch.Tensor:
            batch = x.size(0)
            out = x.new_empty(batch)
            for tile_batch in hl.grid(batch):
                if x[tile_batch] > 0:
                    return out  # This should not be allowed
                out[tile_batch] = x[tile_batch] * 2
            return out

        with self.assertRaises(helion.exc.NotAllowedOnDevice):
            code_and_output(fn, (torch.randn(8, device=DEVICE),))

    def test_assign_without_subscript1(self):
        """Test that modifying host variables inside device loops raises proper error."""

        @helion.kernel()
        def bad_fn(x: torch.Tensor) -> torch.Tensor:
            batch = x.size(0)
            result = torch.empty_like(x)
            for tile_batch in hl.tile(batch):
                # shouldn't be able to modify host variables on device
                result = x[tile_batch] * 2
            return result

        with self.assertRaises(helion.exc.CannotModifyHostVariableOnDevice):
            code_and_output(bad_fn, (torch.randn(8, device=DEVICE),))

    def test_assign_without_subscript2(self):
        """Test that reading device variables from host context raises proper error."""

        @helion.kernel()
        def bad_fn(x: torch.Tensor) -> torch.Tensor:
            batch = x.size(0)
            for tile_batch in hl.tile(batch):
                result = x[tile_batch] * 2
            return result  # shouldn't be able to read device variable here

        with self.assertRaises(helion.exc.CannotReadDeviceVariableOnHost):
            code_and_output(bad_fn, (torch.randn(8, device=DEVICE),))

    def test_device_tensor_subscript(self):
        @helion.kernel()
        def bad_fn(x: torch.Tensor) -> torch.Tensor:
            batch = x.size(0)
            result = torch.empty_like(x)
            for i in hl.tile(batch):
                tmp = x[i] * 2
                tmp[0] = 1  # This should not be allowed
                result[i] = tmp
            return result

        with self.assertRaises(helion.exc.DeviceTensorSubscriptAssignmentNotAllowed):
            code_and_output(bad_fn, (torch.randn(8, device=DEVICE),))

    def test_closure_fn(self):
        @helion.kernel()
        def bad_fn(x: torch.Tensor) -> torch.Tensor:
            def closure_fn():
                pass

            batch = x.size(0)
            result = torch.empty_like(x)
            for i in hl.tile(batch):
                result[i] = x[i] * 2
            return result

        with self.assertRaises(helion.exc.StatementNotSupported):
            code_and_output(bad_fn, (torch.randn(8, device=DEVICE),))

    def test_direct_scalar_tensor_in_device_context(self):
        """Test that direct scalar tensor usage gives clear error in device code."""

        @helion.kernel()
        def bad_fn(x: torch.Tensor, scalar_tensor: torch.Tensor) -> torch.Tensor:
            result = torch.empty_like(x)
            for tile in hl.tile(x.shape):
                result[tile] = x[tile] + scalar_tensor  # Error: direct scalar usage
            return result

        with self.assertRaises(helion.exc.HostTensorDirectUsage):
            code_and_output(
                bad_fn,
                (torch.randn(4, 4, device=DEVICE), torch.tensor(3.0, device=DEVICE)),
            )

    def test_control_flow_rank_mismatch_variable_name_and_hints(self):
        @helion.kernel()
        def fn(a: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(a)
            for ti in hl.tile(a.size(0)):
                if ti.index < 1:
                    x = hl.full([ti], 0.0, dtype=a.dtype)
                else:
                    x = hl.full([ti, ti], 0.0, dtype=a.dtype)
                out[ti] = x.sum()
            return a

        with self.assertRaises(
            helion.exc.ControlFlowTensorMismatch,
        ):
            code_and_output(fn, (torch.randn(4, device=DEVICE),))

    def test_too_many_args(self):
        @helion.kernel()
        def kernel(x: torch.Tensor) -> torch.Tensor:
            result = torch.zeros_like(x)
            for i in hl.tile(x.size()):
                result[i] = x[i]
            return result

        with self.assertRaisesRegex(
            TypeError, r"Too many arguments passed to the kernel, expected: 1 got: 2."
        ):
            a = torch.randn(8, device=DEVICE)
            code_and_output(kernel, (a, a))

    def test_kernel_without_device_loop(self):
        @helion.kernel()
        def bf16_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            # No hl.tile/hl.grid loops — should raise a friendly error
            return x + y

        with self.assertRaises(helion.exc.NoDeviceLoopsInKernel):
            x = torch.randn(4, 4, device=DEVICE)
            y = torch.randn(4, 4, device=DEVICE)
            code_and_output(bf16_add, (x, y))


if __name__ == "__main__":
    unittest.main()
