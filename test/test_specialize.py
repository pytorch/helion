from __future__ import annotations

import math
import unittest

import torch

import helion
from helion._testing import DEVICE
from helion._testing import RefEagerTestBase
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import skipIfCpu
from helion._testing import skipIfRefEager
from helion.exc import ShapeSpecializingAllocation
import helion.language as hl


@skipIfCpu("needs to be debugged")
class TestSpecialize(RefEagerTestBase, TestCase):
    maxDiff = 163842

    def test_sqrt_does_not_specialize(self):
        @helion.kernel()
        def fn(
            x: torch.Tensor,
        ) -> torch.Tensor:
            scale = 1.0 / math.sqrt(x.size(-1))
            out = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                out[tile] = x[tile] * scale
            return out

        x = torch.randn([512, 512], device=DEVICE)
        code, result = code_and_output(fn, (x,), block_sizes=[32, 1], flatten_loop=True)
        torch.testing.assert_close(result, x / math.sqrt(x.size(-1)))
        self.assertExpectedJournal(code)

    def test_specialize_host(self):
        @helion.kernel()
        def fn(
            x: torch.Tensor,
        ) -> torch.Tensor:
            hl.specialize(x.size(-1))
            scale = 1.0 / math.sqrt(x.size(-1))
            out = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                out[tile] = x[tile] * scale
            return out

        x = torch.randn([500, 500], device=DEVICE)
        code, result = code_and_output(fn, (x,), block_sizes=[32, 32])
        torch.testing.assert_close(result, x / math.sqrt(x.size(-1)))
        self.assertExpectedJournal(code)

    @skipIfRefEager("Ref eager mode won't raise ShapeSpecializingAllocation error")
    def test_dynamic_size_block_errors(self):
        @helion.kernel(static_shapes=False)
        def fn(
            x: torch.Tensor,
        ) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size(0)):
                acc = hl.zeros([tile, x.size(1)])
                acc += x[tile, :] + 1
                out[tile, :] = acc
            return out

        x = torch.randn([512, 512], device=DEVICE)
        with self.assertRaises(ShapeSpecializingAllocation):
            code_and_output(fn, (x,), block_size=16)

    def test_dynamic_size_block_specialize(self):
        @helion.kernel()
        def fn(
            x: torch.Tensor,
        ) -> torch.Tensor:
            hl.specialize(x.size(1))
            out = torch.empty_like(x)
            for tile in hl.tile(x.size(0)):
                acc = hl.zeros([tile, x.size(1)])
                acc += x[tile, :] + 1
                out[tile, :] = acc
            return out

        x = torch.randn([512, 512], device=DEVICE)
        code, result = code_and_output(fn, (x,), block_size=32)
        torch.testing.assert_close(result, x + 1)
        self.assertEqual(len(fn.bind((x,)).config_spec.reduction_loops), 0)
        self.assertExpectedJournal(code)

    def test_dynamic_size_block_non_power_of_two(self):
        @helion.kernel()
        def fn(
            x: torch.Tensor,
        ) -> torch.Tensor:
            hl.specialize(x.size(1))
            out = torch.empty_like(x)
            for tile in hl.tile(x.size(0)):
                acc = hl.zeros([tile, helion.next_power_of_2(x.size(1))])
                acc += x[tile, :] + 1
                out[tile, :] = acc
            return out

        x = torch.randn([500, 500], device=DEVICE)
        code, result = code_and_output(fn, (x,), block_size=32)
        torch.testing.assert_close(result, x + 1)
        self.assertTrueIfInNormalMode(
            len(fn.bind((x,)).config_spec.reduction_loops) == 0
        )
        self.assertTrueIfInNormalMode(fn.bind((x,)) is fn.bind((torch.zeros_like(x),)))
        self.assertTrueIfInNormalMode(
            fn.bind((x,)) is not fn.bind((torch.zeros_like(x[:, 1:]),))
        )
        self.assertExpectedJournal(code)

    def test_dynamic_size_block_non_power_of_two_outplace(self):
        @helion.kernel()
        def fn(
            x: torch.Tensor,
        ) -> torch.Tensor:
            hl.specialize(x.size(1))
            out = torch.empty_like(x)
            for tile in hl.tile(x.size(0)):
                acc = hl.zeros([tile, helion.next_power_of_2(x.size(1))])
                acc = acc + x[tile, :] + 1
                out[tile, :] = acc
            return out

        x = torch.randn([500, 500], device=DEVICE)
        code, result = code_and_output(fn, (x,), block_size=32)
        torch.testing.assert_close(result, x + 1)
        self.assertTrueIfInNormalMode(
            len(fn.bind((x,)).config_spec.reduction_loops) == 0
        )
        self.assertTrueIfInNormalMode(fn.bind((x,)) is fn.bind((torch.zeros_like(x),)))
        self.assertTrueIfInNormalMode(
            fn.bind((x,)) is not fn.bind((torch.zeros_like(x[:, 1:]),))
        )
        self.assertExpectedJournal(code)

    def test_dynamic_size_block_non_power_of_two_swap_order(self):
        @helion.kernel()
        def fn(
            x: torch.Tensor,
        ) -> torch.Tensor:
            hl.specialize(x.size(1))
            out = torch.empty_like(x)
            for tile in hl.tile(x.size(0)):
                acc = hl.zeros([tile, helion.next_power_of_2(x.size(1))])
                acc = x[tile, :] + acc + 1
                out[tile, :] = acc
            return out

        x = torch.randn([500, 500], device=DEVICE)
        code, result = code_and_output(fn, (x,), block_size=32)
        torch.testing.assert_close(result, x + 1)
        self.assertTrueIfInNormalMode(
            len(fn.bind((x,)).config_spec.reduction_loops) == 0
        )
        self.assertTrueIfInNormalMode(fn.bind((x,)) is fn.bind((torch.zeros_like(x),)))
        self.assertTrueIfInNormalMode(
            fn.bind((x,)) is not fn.bind((torch.zeros_like(x[:, 1:]),))
        )
        self.assertExpectedJournal(code)

    def test_dynamic_size_block_non_power_of_two_double_acc(self):
        @helion.kernel()
        def fn(
            x: torch.Tensor,
        ) -> torch.Tensor:
            hl.specialize(x.size(1))
            out = torch.empty_like(x)
            for tile in hl.tile(x.size(0)):
                acc = hl.zeros([tile, helion.next_power_of_2(x.size(1))])
                acc2 = hl.full([tile, helion.next_power_of_2(x.size(1))], 1.0)
                acc = acc + acc2
                acc = x[tile, :] + acc + 1
                out[tile, :] = acc
            return out

        x = torch.randn([500, 500], device=DEVICE)
        code, result = code_and_output(fn, (x,), block_size=32)
        torch.testing.assert_close(result, x + 2)
        self.assertTrueIfInNormalMode(
            len(fn.bind((x,)).config_spec.reduction_loops) == 0
        )
        self.assertTrueIfInNormalMode(fn.bind((x,)) is fn.bind((torch.zeros_like(x),)))
        self.assertTrueIfInNormalMode(
            fn.bind((x,)) is not fn.bind((torch.zeros_like(x[:, 1:]),))
        )
        self.assertExpectedJournal(code)

    def test_dynamic_size_block_non_power_of_two_matmul(self):
        @helion.kernel()
        def fn(
            x: torch.Tensor,
        ) -> torch.Tensor:
            hl.specialize(x.size(1))
            out = torch.empty_like(x)
            for tile in hl.tile(x.size(0)):
                acc = hl.full(
                    [tile, helion.next_power_of_2(x.size(1))],
                    1.0 / helion.next_power_of_2(x.size(1)),
                )
                acc2 = hl.full(
                    [
                        helion.next_power_of_2(x.size(1)),
                        helion.next_power_of_2(x.size(1)),
                    ],
                    1.0,
                )
                acc = torch.matmul(acc, acc2)
                acc = x[tile, :] + acc + 1
                out[tile, :] = acc
            return out

        x = torch.randn([500, 500], device=DEVICE)
        code, result = code_and_output(fn, (x,), block_size=32)
        torch.testing.assert_close(result, x + 2)
        self.assertTrueIfInNormalMode(
            len(fn.bind((x,)).config_spec.reduction_loops) == 0
        )
        self.assertTrueIfInNormalMode(fn.bind((x,)) is fn.bind((torch.zeros_like(x),)))
        self.assertTrueIfInNormalMode(
            fn.bind((x,)) is not fn.bind((torch.zeros_like(x[:, 1:]),))
        )
        self.assertExpectedJournal(code)

    def test_tensor_factory_specialize_non_power_of_2(self):
        def _test_with_factory(factory_fn, test_host=True):
            @helion.kernel()
            def reduce_kernel(
                x: torch.Tensor, tensor_factory_fn, test_host
            ) -> torch.Tensor:
                m_block = hl.register_block_size(x.size(0))
                grad_weight = x.new_empty(
                    [(x.size(0) + m_block - 1) // m_block, x.size(1)],
                    dtype=torch.float32,
                )
                weight_shape = hl.specialize(x.size(1))
                if test_host:
                    # Host-side tensor creation should NOT be padded
                    host_buffer = tensor_factory_fn(
                        x, weight_shape, dtype=torch.float32
                    )
                    # Verify host-side tensor has correct non-padded size
                    assert host_buffer.size(0) == 56
                for mb_cta in hl.tile(x.size(0), block_size=m_block):
                    # Device-side tensor creation should be padded to 64
                    grad_w_m = tensor_factory_fn(x, weight_shape, dtype=torch.float32)
                    # Set to 0 to normalize different factory functions
                    grad_w_m = grad_w_m * grad_w_m.new_zeros(weight_shape)
                    for mb in hl.tile(mb_cta.begin, mb_cta.end):
                        grad_w_m += x[mb, :].to(torch.float32).sum(0)
                    grad_weight[mb_cta.id, :] = grad_w_m
                return grad_weight.sum(0).to(x.dtype)

            x = torch.randn([128, 56], device=DEVICE, dtype=torch.float32)
            code, result = code_and_output(reduce_kernel, (x, factory_fn, test_host))
            reference = x.sum(0)
            torch.testing.assert_close(result, reference, rtol=1e-3, atol=1e-3)
            self.assertExpectedJournal(code)

        for name in ["zeros", "ones", "empty"]:
            _test_with_factory(
                lambda x, s, factory_name=name, **kw: getattr(torch, factory_name)(
                    s, device=x.device, **kw
                )
            )
        _test_with_factory(
            lambda x, s, **kw: torch.full([s], 1.0, device=x.device, **kw)
        )

        for name in ["zeros", "ones", "empty"]:
            _test_with_factory(
                lambda x, s, method_name=name, **kw: getattr(x, f"new_{method_name}")(
                    s, **kw
                ),
                test_host=True,
            )
        _test_with_factory(
            lambda x, s, **kw: x.new_full([s], 1.0, **kw), test_host=True
        )

        _test_with_factory(lambda x, s, **kw: hl.zeros([s], **kw), test_host=False)
        _test_with_factory(lambda x, s, **kw: hl.full([s], 1.0, **kw), test_host=False)

    def test_specialize_reduce(self):
        @helion.kernel()
        def fn(
            x: torch.Tensor,
        ) -> torch.Tensor:
            hl.specialize(x.size(1))
            out = x.new_empty([x.size(0)])
            for tile in hl.tile(x.size(0)):
                out[tile] = x[tile, :].sum(-1)
            return out

        x = torch.randn([500, 500], device=DEVICE)
        code, result = code_and_output(fn, (x,), block_size=32)
        torch.testing.assert_close(result, x.sum(-1))
        self.assertTrueIfInNormalMode(
            len(fn.bind((x,)).config_spec.reduction_loops) == 1
        )
        self.assertExpectedJournal(code)

    def test_specialize_tuple_element(self):
        """Test that hl.specialize works correctly with tuple elements."""

        @helion.kernel(config=helion.Config(block_sizes=[32]))
        def foo(x: torch.Tensor, bitshift: tuple[int, int]) -> torch.Tensor:
            out = x.new_empty(x.shape)
            val = hl.specialize(bitshift[0])
            for x_tile in hl.tile([x.shape[0]]):
                # compute_val equivalent: 1 << (32 - val)
                out[x_tile] = x[x_tile] + (1 << (32 - val))
            return out

        x = torch.ones(64, dtype=torch.int32, device=DEVICE)
        code, result = code_and_output(foo, (x, (16, 16)))
        # 1 << (32-16) = 1 << 16 = 65536
        expected = x + 65536
        torch.testing.assert_close(result, expected)
        # Verify that 65536 appears in the generated code as a constant
        self.assertIn("65536", code)
        self.assertExpectedJournal(code)


@skipIfCpu("needs to be debugged")
class TestSpecializeArgs(RefEagerTestBase, TestCase):
    """Tests for kernel.specialize_args() external specialization API."""

    maxDiff = 163842

    def test_specialize_args(self):
        """Test specialize_args: multiple tensors, multiple dims, negative indexing."""

        @helion.kernel(autotune_effort="none", static_shapes=False)
        def matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, k = x.size()
            k2, n = y.size()
            out = torch.empty([m, n], device=x.device, dtype=x.dtype)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc.to(x.dtype)
            return out

        m, k, n = 64, 128, 56
        x = torch.randn([m, k], device=DEVICE, dtype=torch.float16)
        y = torch.randn([k, n], device=DEVICE, dtype=torch.float16)

        # First, run WITHOUT specialize_args - dimensions should NOT be constants
        code_no_spec, result_no_spec = code_and_output(
            matmul,
            (x, y),
            block_sizes=[32, 32, 32],
        )
        torch.testing.assert_close(result_no_spec, x @ y, rtol=1e-2, atol=1e-2)
        self.assertNotIn("64", code_no_spec)  # x dim 0 = m should NOT be specialized
        self.assertNotIn("128", code_no_spec)  # x dim -1 = k should NOT be specialized
        self.assertNotIn("56", code_no_spec)  # y dim 1 = n should NOT be specialized

        # Now, run WITH specialize_args - dimensions SHOULD be constants
        code, result = code_and_output(
            matmul.specialize_args(x=[0, -1], y=[1]),
            (x, y),
            block_sizes=[32, 32, 32],
        )
        torch.testing.assert_close(result, x @ y, rtol=1e-2, atol=1e-2)
        self.assertIn("64", code)  # x dim 0 = m
        self.assertIn("128", code)  # x dim -1 = k
        self.assertIn("56", code)  # y dim 1 = n
        self.assertExpectedJournal(code)

        # Verify cache behavior: same specialized values hit cache
        specialized_kernel = matmul.specialize_args(x=[0, -1], y=[1])
        self.assertIs(specialized_kernel.bind((x, y)), specialized_kernel.bind((x, y)))
        # Verify cache behavior: different specialized values produce different bound kernels
        x2 = torch.randn([48, 96], device=DEVICE, dtype=torch.float16)
        y2 = torch.randn([96, 24], device=DEVICE, dtype=torch.float16)
        self.assertIsNot(
            specialized_kernel.bind((x, y)), specialized_kernel.bind((x2, y2))
        )

    def test_specialize_args_and_hl_specialize(self):
        """Test that external specialize_args and internal hl.specialize form a union."""

        @helion.kernel(autotune_effort="none", static_shapes=False)
        def dual_specialize(x: torch.Tensor) -> torch.Tensor:
            # Internal specialize on dim 0
            hl.specialize(x.size(0))
            out = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                out[tile] = x[tile] * 2
            return out

        x = torch.randn([320, 640], device=DEVICE)

        # First, run WITHOUT external specialize_args - only dim 0 should be specialized
        code_no_spec, result_no_spec = code_and_output(
            dual_specialize,
            (x,),
            block_sizes=[16, 16],
        )
        torch.testing.assert_close(result_no_spec, x * 2)
        self.assertIn("320", code_no_spec)  # dim 0 from internal specialize
        self.assertNotIn("640", code_no_spec)  # dim 1 should NOT be specialized

        # Now, run WITH external specialize_args on dim -1 (dim 1)
        # Result: both dim 0 AND dim 1 are specialized (union)
        code, result = code_and_output(
            dual_specialize.specialize_args(x=[-1]),
            (x,),
            block_sizes=[16, 16],
        )
        torch.testing.assert_close(result, x * 2)
        # Both dimensions should appear as constants
        self.assertIn("320", code)  # dim 0 from internal specialize
        self.assertIn("640", code)  # dim 1 from external specialize
        self.assertExpectedJournal(code)

        # Verify cache behavior: changing dim 1 (external) produces different bound kernel
        x2 = torch.randn([320, 128], device=DEVICE)  # same dim 0, different dim 1
        specialized_kernel = dual_specialize.specialize_args(x=[-1])
        self.assertIsNot(specialized_kernel.bind((x,)), specialized_kernel.bind((x2,)))

    @skipIfRefEager("Error checking not available in ref eager mode")
    def test_specialize_args_errors(self):
        """Test error handling for invalid specialize_args usage."""

        @helion.kernel(autotune_effort="none", static_shapes=False)
        def fn(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                out[tile] = x[tile]
            return out

        x = torch.randn([32, 64], device=DEVICE)  # 2D tensor

        # Error: dim out of range
        with self.assertRaises((IndexError, ValueError)):
            fn.specialize_args(x=[5])(x)

        # Error: unknown argument name
        with self.assertRaises(ValueError) as cm:
            fn.specialize_args(z=[-1])
        self.assertIn("Unknown argument", str(cm.exception))

    def test_specialize_args_chaining(self):
        """Test that chained specialize_args calls merge specializations."""

        @helion.kernel(autotune_effort="none", static_shapes=False)
        def fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, n = x.size()
            p = y.size(1)  # use y's dim 1 as a scalar
            out = x.new_empty([m, n])
            for tile_m, tile_n in hl.tile([m, n]):
                out[tile_m, tile_n] = x[tile_m, tile_n] * p
            return out

        x = torch.randn([37, 64], device=DEVICE)
        y = torch.randn([48, 127], device=DEVICE)

        # First, run WITHOUT specialize_args - dimensions should NOT be constants
        code_no_spec, result_no_spec = code_and_output(fn, (x, y), block_sizes=[16, 16])
        torch.testing.assert_close(result_no_spec, x * 127)
        self.assertNotIn("37", code_no_spec)  # x dim 0 should NOT be specialized
        self.assertNotIn("127", code_no_spec)  # y dim 1 should NOT be specialized

        # Now, chain two specialize_args calls - both should be preserved
        chained = fn.specialize_args(x=[0]).specialize_args(y=[1])

        code, result = code_and_output(chained, (x, y), block_sizes=[16, 16])
        torch.testing.assert_close(result, x * 127)
        # Both specializations should be present
        self.assertIn("37", code)  # x dim 0
        self.assertIn("127", code)  # y dim 1
        self.assertExpectedJournal(code)

        # Verify cache behavior: changing specialized values produces different bound kernels
        x2 = torch.randn([48, 64], device=DEVICE)  # different dim 0
        y2 = torch.randn([48, 256], device=DEVICE)  # different dim 1
        self.assertIsNot(chained.bind((x, y)), chained.bind((x2, y2)))


if __name__ == "__main__":
    unittest.main()
