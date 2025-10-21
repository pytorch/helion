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
from helion._testing import skipIfRocm
from helion.autotuner import EnumFragment
from helion.autotuner import IntegerFragment
from helion.autotuner import PowerOfTwoFragment
import helion.language as hl
from helion.language import loops


class TestRegisterTunable(RefEagerTestBase, TestCase):
    maxDiff = 10000

    def test_power_of_two_fragment_basic(self):
        @helion.kernel(autotune_effort="none")
        def kernel_with_tunable(x: torch.Tensor) -> torch.Tensor:
            (n,) = x.size()
            out = torch.empty_like(x)

            # Register a tunable parameter for block size
            block_size = hl.register_tunable("foo", PowerOfTwoFragment(16, 256))

            for tile_n in hl.tile([n], block_size=[block_size * 2]):
                out[tile_n] = x[tile_n] * 2.0

            return out

        x = torch.randn(128, device=DEVICE, dtype=torch.float32)
        code, result = code_and_output(kernel_with_tunable, (x,))
        expected = x * 2.0
        torch.testing.assert_close(result, expected)
        self.assertIsInstance(
            self.getUserDefinedTunable(
                kernel_with_tunable.bind((x,)).config_spec.user_defined_tunables, "foo"
            ),
            PowerOfTwoFragment,
        )
        self.assertExpectedJournal(code)

    @patch.object(_compat, "_supports_tensor_descriptor", lambda: False)
    @patch.object(loops, "_supports_warp_specialize", lambda: False)
    def test_integer_fragment(self):
        @helion.kernel()
        def kernel_with_int_param(x: torch.Tensor) -> torch.Tensor:
            (n,) = x.size()
            out = torch.empty_like(x)
            # Register an integer tunable parameter
            multiplier = hl.register_tunable("multiplier", IntegerFragment(1, 10, 3))
            for tile_n in hl.tile([n]):
                out[tile_n] = x[tile_n] * multiplier
            return out

        x = torch.randn(128, device=DEVICE, dtype=torch.float32)
        code, result = code_and_output(
            kernel_with_int_param, (x,), block_size=64, multiplier=4
        )
        expected = x * 4
        torch.testing.assert_close(result, expected)
        self.assertExpectedJournal(
            repr(kernel_with_int_param.bind((x,)).config_spec.default_config())
        )
        self.assertExpectedJournal(code)

    def test_enum_fragment(self):
        @helion.kernel(config={"operation": 2})
        def kernel_with_enum(x: torch.Tensor) -> torch.Tensor:
            (n,) = x.size()
            out = torch.empty_like(x)

            # Register an enum tunable parameter
            operation = hl.register_tunable("operation", EnumFragment((1, 2, 4)))

            for tile_n in hl.tile([n], block_size=[64]):
                out[tile_n] = x[tile_n] * operation

            return out

        x = torch.randn(128, device=DEVICE, dtype=torch.float32)
        result = kernel_with_enum(x)
        expected = x * 2.0
        torch.testing.assert_close(result, expected)

    def test_tensor_allocated_with_block_size(self):
        @helion.kernel()
        def fn(x: torch.Tensor):
            m = x.size(0)
            block_m = hl.register_block_size(m)
            tiles_m = (m + block_m - 1) // block_m  # cdiv
            partial = torch.zeros(tiles_m, dtype=x.dtype, device=x.device)
            for tile in hl.tile(m, block_size=block_m):
                partial[tile.begin // block_m] = x[tile].sum()
            return partial.sum()

        x = torch.randn(1024, device=DEVICE, dtype=torch.float32)
        code, result = code_and_output(fn, (x,), block_size=64)
        self.assertExpectedJournal(code)
        torch.testing.assert_close(result, x.sum())

    @patch.object(_compat, "_supports_tensor_descriptor", lambda: False)
    @skipIfRocm("failure on rocm")
    def test_matmul_split_k(self):
        """Test matmul_split_k kernel with register_tunable"""

        @helion.kernel(
            config=helion.Config(
                block_sizes=[32, 64, 64],
                loop_orders=[[1, 2, 0]],
                num_warps=16,
                num_stages=8,
                indexing="block_ptr",
                split_k=64,
            )
        )
        def matmul_split_k(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, k = x.size()
            k2, n = y.size()
            assert k == k2, f"size mismatch {k} != {k2}"
            out = torch.zeros(
                [m, n], dtype=torch.promote_types(x.dtype, y.dtype), device=x.device
            )
            split_k = hl.register_tunable("split_k", PowerOfTwoFragment(1, 256))
            k_block = helion.next_power_of_2(helion.cdiv(k, split_k))
            for tile_m, tile_n, outer_k in hl.tile(
                [m, n, k], block_size=[None, None, k_block]
            ):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for inner_k in hl.tile(outer_k.begin, outer_k.end):
                    acc = torch.addmm(acc, x[tile_m, inner_k], y[inner_k, tile_n])
                hl.atomic_add(out, [tile_m, tile_n], acc)
            return out

        m, k, n = 64, 4096, 64
        x = torch.randn([m, k], device=DEVICE, dtype=torch.float16)
        y = torch.randn([k, n], device=DEVICE, dtype=torch.float16)

        code, result = code_and_output(matmul_split_k, (x, y))
        expected = x @ y
        torch.testing.assert_close(result, expected, rtol=1e-2, atol=1)
        self.assertIsInstance(
            self.getUserDefinedTunable(
                matmul_split_k.bind((x, y)).config_spec.user_defined_tunables, "split_k"
            ),
            PowerOfTwoFragment,
        )
        self.assertExpectedJournal(code)


if __name__ == "__main__":
    unittest.main()
