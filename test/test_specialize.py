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

    def test_specialize_size_becomes_static(self):
        """Test that hl.specialize on a size makes it NOT passed to the triton kernel."""

        @helion.kernel(static_shapes=False)
        def fn(x: torch.Tensor) -> torch.Tensor:
            n = hl.specialize(x.size(0))
            out = torch.empty_like(x)
            for tile in hl.tile(n):
                out[tile] = x[tile] + 1
            return out

        x = torch.randn([137], device=DEVICE)  # Use prime to avoid alignment
        code, result = code_and_output(fn, (x,))
        torch.testing.assert_close(result, x + 1)
        # Verify x_size_0 is NOT passed as an argument (it should be static)
        self.assertNotIn("x_size_0", code)
        self.assertExpectedJournal(code)

    def test_specialize_stride_basic(self):
        """Test that hl.specialize works with tensor strides."""

        @helion.kernel(static_shapes=False, autotune_effort="none")
        def fn(x: torch.Tensor) -> torch.Tensor:
            stride = hl.specialize(x.stride(0))
            out = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                # Use stride in computation to verify it's a constant
                out[tile] = x[tile] + stride
            return out

        # Use empty_strided to create tensor with a unique stride value (137)
        # that won't be confused with shape values
        size = (64, 64)
        stride0 = 137  # Distinctive prime number for stride(0)
        stride1 = 1
        # Need storage size to fit: (size[0]-1)*stride0 + (size[1]-1)*stride1 + 1
        storage_size = (size[0] - 1) * stride0 + (size[1] - 1) * stride1 + 1
        storage = torch.randn(storage_size, device=DEVICE)
        x = torch.as_strided(storage, size, (stride0, stride1))

        code, result = code_and_output(fn, (x,))
        torch.testing.assert_close(result, x + x.stride(0))
        # Verify the unique stride value 137 is inlined as a constant
        self.assertIn("137", code)
        # Verify x_stride_0 is NOT passed as an argument (it should be inlined)
        self.assertNotIn("x_stride_0", code)
        self.assertExpectedJournal(code)

    def test_specialize_stride_creates_different_variants(self):
        """Test that different stride patterns create different kernel variants."""

        @helion.kernel(static_shapes=False, autotune_effort="none")
        def fn(x: torch.Tensor) -> torch.Tensor:
            stride = hl.specialize(x.stride(0))
            out = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                out[tile] = x[tile] + stride
            return out

        # Create two tensors with different unique stride values using empty_strided
        size = (64, 64)

        # First tensor with stride(0) = 173 (distinctive prime)
        stride0_a = 173
        storage_size_a = (size[0] - 1) * stride0_a + (size[1] - 1) * 1 + 1
        storage_a = torch.randn(storage_size_a, device=DEVICE)
        x_a = torch.as_strided(storage_a, size, (stride0_a, 1))

        # Second tensor with stride(0) = 257 (different distinctive prime)
        stride0_b = 257
        storage_size_b = (size[0] - 1) * stride0_b + (size[1] - 1) * 1 + 1
        storage_b = torch.randn(storage_size_b, device=DEVICE)
        x_b = torch.as_strided(storage_b, size, (stride0_b, 1))

        # These should create different bound kernels due to different strides
        bound1 = fn.bind((x_a,))
        bound2 = fn.bind((x_b,))

        # Verify different variants are used
        self.assertTrueIfInNormalMode(bound1 is not bound2)

        # Verify correctness
        result1 = fn(x_a)
        result2 = fn(x_b)
        torch.testing.assert_close(result1, x_a + stride0_a)
        torch.testing.assert_close(result2, x_b + stride0_b)

    def test_specialize_stride_tuple(self):
        """Test that hl.specialize works with tuple of strides."""

        @helion.kernel(static_shapes=False, autotune_effort="none")
        def fn(x: torch.Tensor) -> torch.Tensor:
            stride0, stride1 = hl.specialize((x.stride(0), x.stride(1)))
            out = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                out[tile] = x[tile] + stride0 + stride1
            return out

        # Create tensor with unique stride values using empty_strided
        # stride0 = 311, stride1 = 131 (distinctive primes unlikely to appear elsewhere)
        size = (64, 64)
        stride0 = 311
        stride1 = 131
        # Storage must fit the largest offset: (size[0]-1)*stride0 + (size[1]-1)*stride1 + 1
        storage_size = (size[0] - 1) * stride0 + (size[1] - 1) * stride1 + 1
        storage = torch.randn(storage_size, device=DEVICE)
        x = torch.as_strided(storage, size, (stride0, stride1))

        code, result = code_and_output(fn, (x,))
        expected = x + stride0 + stride1
        torch.testing.assert_close(result, expected)
        # Verify both unique stride values appear in the generated code
        self.assertIn("311", code)
        self.assertIn("131", code)
        # Verify both x_stride_0 and x_stride_1 are NOT passed as arguments (they should be inlined)
        self.assertNotIn("x_stride_0", code)
        self.assertNotIn("x_stride_1", code)
        self.assertExpectedJournal(code)


@skipIfCpu("needs to be debugged")
class TestMarkStatic(RefEagerTestBase, TestCase):
    """Tests for torch._dynamo.mark_static() external specialization API."""

    maxDiff = 163842

    def test_mark_static(self):
        """Test mark_static: multiple tensors, multiple dims, negative indexing."""

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

        m, k, n = 96, 128, 48

        # First, run WITHOUT mark_static - dimensions should NOT be constants
        x = torch.randn([m, k], device=DEVICE, dtype=torch.float16)
        y = torch.randn([k, n], device=DEVICE, dtype=torch.float16)
        code_no_spec, result_no_spec = code_and_output(
            matmul, (x, y), block_sizes=[32, 32, 32]
        )
        torch.testing.assert_close(result_no_spec, x @ y, rtol=1e-2, atol=1e-2)
        self.assertNotIn("96", code_no_spec)
        self.assertNotIn("128", code_no_spec)
        self.assertNotIn("48", code_no_spec)

        # Now, run WITH mark_static - dimensions SHOULD be constants
        x_static = torch.randn([m, k], device=DEVICE, dtype=torch.float16)
        y_static = torch.randn([k, n], device=DEVICE, dtype=torch.float16)
        torch._dynamo.mark_static(x_static, [0, -1])  # test list and negative index
        torch._dynamo.mark_static(y_static, 1)

        code, result = code_and_output(
            matmul, (x_static, y_static), block_sizes=[32, 32, 32]
        )
        torch.testing.assert_close(result, x_static @ y_static, rtol=1e-2, atol=1e-2)
        self.assertIn("96", code)
        self.assertIn("128", code)
        self.assertIn("48", code)
        self.assertExpectedJournal(code)

        # Cache hit: same tensors
        self.assertIs(
            matmul.bind((x_static, y_static)), matmul.bind((x_static, y_static))
        )
        # Cache miss: different specialized values
        x2 = torch.randn([48, 96], device=DEVICE, dtype=torch.float16)
        y2 = torch.randn([96, 24], device=DEVICE, dtype=torch.float16)
        torch._dynamo.mark_static(x2, [0, -1])
        torch._dynamo.mark_static(y2, 1)
        self.assertIsNot(matmul.bind((x_static, y_static)), matmul.bind((x2, y2)))

    def test_mark_static_and_hl_specialize(self):
        """Test that external mark_static and internal hl.specialize form a union."""

        @helion.kernel(autotune_effort="none", static_shapes=False)
        def fn(x: torch.Tensor) -> torch.Tensor:
            hl.specialize(x.size(0))  # internal specialize on dim 0
            out = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                out[tile] = x[tile] * 2
            return out

        # mark_static on dim 1 should combine with hl.specialize on dim 0
        x = torch.randn([320, 640], device=DEVICE)
        torch._dynamo.mark_static(x, -1)

        code, result = code_and_output(fn, (x,), block_sizes=[16, 16])
        torch.testing.assert_close(result, x * 2)
        self.assertIn("320", code)  # dim 0 from hl.specialize
        self.assertIn("640", code)  # dim 1 from mark_static
        self.assertExpectedJournal(code)

        # Cache miss: changing externally-specialized dim
        x2 = torch.randn([320, 128], device=DEVICE)
        torch._dynamo.mark_static(x2, -1)
        self.assertIsNot(fn.bind((x,)), fn.bind((x2,)))


if __name__ == "__main__":
    unittest.main()
