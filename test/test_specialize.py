from __future__ import annotations

import dataclasses
import math
from typing import Any
from typing import cast
import unittest

import torch

import helion
from helion._testing import DEVICE
from helion._testing import HALF_DTYPE
from helion._testing import AssertExpectedJournal
from helion._testing import RefEagerTestBase
from helion._testing import TestCase
from helion._testing import _get_backend
from helion._testing import code_and_output
from helion._testing import onlyBackends
from helion._testing import skipIfCute
from helion._testing import skipIfNotTriton
from helion._testing import skipIfRefEager
from helion._testing import skipUnlessBackends
from helion._testing import skipUnlessBlockPtr
from helion.autotuner.benchmark_provider import _searching_candidates
from helion.exc import ShapeSpecializingAllocation
import helion.language as hl


@dataclasses.dataclass
class TensorContainer:
    x: torch.Tensor


@onlyBackends(["triton", "cute"])
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

    def test_unspecialized_stride_is_runtime_value(self):
        """An unspecialized stride is not inlined from the example input."""

        @helion.kernel(static_shapes=False, autotune_effort="none")
        def fn(x: torch.Tensor) -> torch.Tensor:
            stride = x.stride(0)
            out = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                out[tile] = x[tile] + stride
            return out

        x = torch.randn([64, 64], device=DEVICE)
        code, result = code_and_output(fn, (x,))
        torch.testing.assert_close(result, x + x.stride(0))
        self.assertIn("stride = x.stride(0)", code)

        # CuTe guards the unit-stride axes and byte-alignment class needed by
        # vector copies. Exact stride values within that class stay dynamic.
        padded = torch.empty_strided(
            x.size(), (2 * x.stride(0), 1), dtype=x.dtype, device=x.device
        )
        padded.copy_(x)
        self.assertIs(fn.bind((x,)), fn.bind((padded,)))
        torch.testing.assert_close(fn(padded), padded + padded.stride(0))
        transposed = x.T
        if fn.settings.backend == "cute":
            self.assertIsNot(fn.bind((x,)), fn.bind((transposed,)))
        else:
            self.assertIs(fn.bind((x,)), fn.bind((transposed,)))
        torch.testing.assert_close(fn(transposed), transposed + transposed.stride(0))

    def test_unspecialized_stride_host_control_flow(self):
        """Host control flow reads an unspecialized stride at runtime."""

        @helion.kernel(static_shapes=False, autotune_effort="none")
        def fn(x: torch.Tensor) -> torch.Tensor:
            if x.stride(0) == 1:
                value = 1
            elif x.stride(0) == 64:
                value = 2
            else:
                value = 3
            out = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                out[tile] = x[tile] + value
            return out

        x = torch.randn([64, 64], device=DEVICE)
        transposed = x.T
        code, result = code_and_output(fn, (x,))
        self.assertIn("if x.stride(0) == 1:", code)
        self.assertIn("x.stride(0) == 64:", code)
        torch.testing.assert_close(result, x + 2)
        padded = torch.empty_strided(
            x.size(), (2 * x.stride(0), 1), dtype=x.dtype, device=x.device
        )
        padded.copy_(x)
        self.assertIs(fn.bind((x,)), fn.bind((padded,)))
        torch.testing.assert_close(fn(padded), padded + 3)
        if fn.settings.backend == "cute":
            self.assertIsNot(fn.bind((x,)), fn.bind((transposed,)))
        else:
            self.assertIs(fn.bind((x,)), fn.bind((transposed,)))
        torch.testing.assert_close(fn(transposed), transposed + 1)

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

    def test_specialize_concrete_stride(self):
        """Concrete contiguous strides are inlined and guarded in the cache key."""

        @helion.kernel(static_shapes=False, autotune_effort="none")
        def fn(x: torch.Tensor) -> torch.Tensor:
            stride = hl.specialize(x.stride(-1))
            out = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                out[tile] = x[tile] + stride
            return out

        x = torch.randn([64, 64], device=DEVICE)
        code, result = code_and_output(fn, (x,))
        torch.testing.assert_close(result, x + 1)
        self.assertNotIn("x_stride_1", code)

        transposed = x.T
        self.assertIsNot(fn.bind((x,)), fn.bind((transposed,)))
        torch.testing.assert_close(fn(transposed), transposed + 64)

    def test_specialize_concrete_stride_tuple(self):
        """A tuple of concrete strides retains each dimension's cache guard."""

        @helion.kernel(static_shapes=False, autotune_effort="none")
        def fn(x: torch.Tensor) -> torch.Tensor:
            stride0, stride1 = hl.specialize(x.stride())
            out = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                out[tile] = x[tile] + stride0 * 2 + stride1
            return out

        x = torch.randn([64, 64], device=DEVICE)
        code, result = code_and_output(fn, (x,))
        torch.testing.assert_close(result, x + 129)
        self.assertNotIn("x_stride_0", code)
        self.assertNotIn("x_stride_1", code)

        transposed = x.T
        self.assertIsNot(fn.bind((x,)), fn.bind((transposed,)))
        torch.testing.assert_close(fn(transposed), transposed + 66)

    def test_specialize_concrete_stride_keyword_dim(self):
        """The keyword spelling retains the tensor stride origin."""

        @helion.kernel(static_shapes=False, autotune_effort="none")
        def fn(x: torch.Tensor) -> torch.Tensor:
            stride = hl.specialize(x.stride(dim=-1))
            out = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                out[tile] = x[tile] + stride
            return out

        x = torch.randn([64, 64], device=DEVICE)
        transposed = x.T
        self.assertIsNot(fn.bind((x,)), fn.bind((transposed,)))
        torch.testing.assert_close(fn(x), x + 1)
        torch.testing.assert_close(fn(transposed), transposed + 64)

    def test_specialize_concrete_stride_expression(self):
        """Concrete stride dependencies survive host-side arithmetic."""

        @helion.kernel(static_shapes=False, autotune_effort="none")
        def fn(x: torch.Tensor) -> torch.Tensor:
            stride = hl.specialize(x.stride(-1) * 2)
            out = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                out[tile] = x[tile] + stride
            return out

        x = torch.randn([64, 64], device=DEVICE)
        transposed = x.T
        self.assertIsNot(fn.bind((x,)), fn.bind((transposed,)))
        torch.testing.assert_close(fn(x), x + 2)
        torch.testing.assert_close(fn(transposed), transposed + 128)

    def test_specialize_concrete_stride_container(self):
        """Tensor paths through list and tuple arguments are guarded exactly."""

        for container_type in (list, tuple):

            @helion.kernel(static_shapes=False, autotune_effort="none")
            def fn(xs: list[torch.Tensor]) -> torch.Tensor:
                x = xs[0]
                stride = hl.specialize(x.stride(-1))
                out = torch.empty_like(x)
                for tile in hl.tile(x.size()):
                    out[tile] = x[tile] + stride
                return out

            with self.subTest(container_type=container_type.__name__):
                x = torch.randn([64, 64], device=DEVICE)
                transposed = x.T
                contiguous_args = (container_type([x]),)
                transposed_args = (container_type([transposed]),)
                self.assertIsNot(
                    fn.bind(contiguous_args),
                    fn.bind(transposed_args),
                )
                code, result = code_and_output(fn, contiguous_args)
                self.assertNotIn("x_stride_1", code)
                torch.testing.assert_close(result, x + 1)
                torch.testing.assert_close(fn(*transposed_args), transposed + 64)

    def test_specialize_concrete_stride_dict(self):
        """Mapping paths are replayable, and static kernels need no extra guard."""

        for static_shapes in (False, True):

            @helion.kernel(
                static_shapes=static_shapes,
                autotune_effort="none",
            )
            def fn(xs: dict[str, torch.Tensor]) -> torch.Tensor:
                x = xs["x"]
                stride = hl.specialize(x.stride(-1))
                out = torch.empty_like(x)
                for tile in hl.tile(x.size()):
                    out[tile] = x[tile] + stride
                return out

            with self.subTest(static_shapes=static_shapes):
                x = torch.randn([64, 64], device=DEVICE)
                transposed = x.T
                contiguous_args = ({"x": x},)
                transposed_args = ({"x": transposed},)
                self.assertIsNot(
                    fn.bind(contiguous_args),
                    fn.bind(transposed_args),
                )
                torch.testing.assert_close(fn(*contiguous_args), x + 1)
                torch.testing.assert_close(fn(*transposed_args), transposed + 64)

    def test_specialize_concrete_stride_dataclass(self):
        """Attribute-like container fields retain their input source path."""

        @helion.kernel(static_shapes=False, autotune_effort="none")
        def fn(xs: TensorContainer, device: torch.device) -> torch.Tensor:
            x = xs.x
            stride = hl.specialize(x.stride(-1))
            out = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                out[tile] = x[tile] + stride
            return out

        x = torch.randn([64, 64], device=DEVICE)
        transposed = x.T
        contiguous_args = (TensorContainer(x), x.device)
        transposed_args = (TensorContainer(transposed), x.device)
        self.assertIsNot(
            fn.bind(contiguous_args),
            fn.bind(transposed_args),
        )
        torch.testing.assert_close(fn(*contiguous_args), x + 1)
        torch.testing.assert_close(fn(*transposed_args), transposed + 64)

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

    def test_dynamic_reduction_over_empty_rows(self):
        """A dynamic reduction extent of 0 still needs a length-1 block (the
        static path already uses one); Triton's next_power_of_2(0) is 0."""

        @helion.kernel(static_shapes=False, autotune_effort="none")
        def row_sum(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty([x.size(0)], device=x.device)
            for tile_m in hl.tile(x.size(0)):
                out[tile_m] = x[tile_m, :].sum(-1)
            return out

        for shape in ((64, 8), (64, 0)):
            with self.subTest(shape=shape):
                x = torch.randn(shape, device=DEVICE)
                torch.testing.assert_close(row_sum(x), x.sum(-1))

    @skipIfRefEager("bound kernels are not used in ref eager mode")
    def test_looped_reduction_reuses_bound_kernel_across_extents(self):
        """Only a config compiled to run keys a CuTe persistent reduction's
        extent bucket: a persistent candidate a search compiles must not
        re-bind a kernel whose looped winner works at every extent."""
        looped = helion.Config(block_sizes=[1], reduction_loops=[64])
        persistent = helion.Config(block_sizes=[1], reduction_loops=[None])

        @helion.kernel(static_shapes=False, configs=[looped, persistent])
        def row_sum(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty([x.size(0)], device=x.device)
            for tile_m in hl.tile(x.size(0)):
                out[tile_m] = x[tile_m, :].sum(-1)
            return out

        x = torch.randn(64, 100, device=DEVICE)
        bound = row_sum.bind((x,))
        with _searching_candidates():
            bound.compile_config(persistent)
        bound.set_config(looped)
        torch.testing.assert_close(row_sum(x), x.sum(-1), rtol=1e-4, atol=1e-4)
        self.assertFalse(
            row_sum._cute_reduction_extent_descriptors.get(bound._base_spec_key)
        )
        y = torch.randn(64, 300, device=DEVICE)
        torch.testing.assert_close(row_sum(y), y.sum(-1), rtol=1e-4, atol=1e-4)
        if "input_tensor_metadata" not in bound.env.compiler_fact_specialization_facts:
            # Otherwise the bound kernel is keyed on the exact input metadata.
            self.assertIs(row_sum.bind((y,)), bound)

    def test_looped_winner_autotunes_once_across_extent_buckets(self):
        """The autotuner's reference baseline at n=1024 is a persistent row
        reduction; compiling it must not key the extent bucket, so the looped
        winner serves every later extent from the one bound kernel (and the
        one search) of the first call.  A kernel keyed on its exact input
        metadata re-binds and searches per shape, and none of those searches
        may key an extent bucket either."""

        @helion.kernel(
            static_shapes=False,
            configs=[
                helion.Config(reduction_loops=[256]),
                helion.Config(reduction_loops=[512]),
            ],
        )
        def row_sum(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty([x.size(0)], device=x.device)
            for i in hl.grid(x.size(0)):
                out[i] = x[i, :].sum(-1)
            return out

        x = torch.randn(64, 1024, device=DEVICE)
        torch.testing.assert_close(row_sum(x), x.sum(-1), rtol=1e-4, atol=1e-3)
        bound = row_sum.bind((x,))
        self.assertFalse(
            row_sum._cute_reduction_extent_descriptors.get(bound._base_spec_key)
        )
        metadata_keyed = (
            "input_tensor_metadata" in bound.env.compiler_fact_specialization_facts
        )
        for n in (3000, 6000, 1500):
            with self.subTest(n=n):
                y = torch.randn(64, n, device=DEVICE)
                torch.testing.assert_close(row_sum(y), y.sum(-1), rtol=1e-4, atol=1e-3)
                rebound = row_sum.bind((y,))
                self.assertFalse(
                    row_sum._cute_reduction_extent_descriptors.get(
                        rebound._base_spec_key
                    )
                )
                if not metadata_keyed:
                    # Otherwise the bound kernel is keyed on the exact input
                    # metadata.
                    self.assertIs(rebound, bound)

    @skipIfRefEager("checks compile_config bound-kernel keying; runs no ref")
    def test_compile_config_keys_persistent_reduction_extent(self):
        """bound.compile_config() returns a callable compiled at the bound
        kernel's size hints, and a CuTe persistent reduction or full-slice
        load sizes its threads from the extent's hint: compiled at n=32, a
        later bind() at n=64 must get its own bound kernel, not reuse it."""
        persistent = helion.Config(block_sizes=[1], reduction_loops=[None])

        @helion.kernel(static_shapes=False, autotune_effort="none")
        def double(x: torch.Tensor) -> torch.Tensor:
            out = torch.zeros_like(x)
            for tile_m in hl.tile(x.size(0)):
                out[tile_m, :] = x[tile_m, :] * 2
            return out

        @helion.kernel(static_shapes=False, configs=[persistent])
        def row_sum(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty([x.size(0)], device=x.device)
            for tile_m in hl.tile(x.size(0)):
                out[tile_m] = x[tile_m, :].sum(-1)
            return out

        for n in (32, 64):
            with self.subTest(n=n):
                x = torch.randn(40, n, device=DEVICE)
                bound = double.bind((x,))
                compiled = bound.compile_config(bound.config_spec.default_config())
                torch.testing.assert_close(compiled(x), x * 2, rtol=0, atol=0)
                compiled = row_sum.bind((x,)).compile_config(persistent)
                torch.testing.assert_close(compiled(x), x.sum(-1), rtol=1e-4, atol=1e-4)

    def test_two_full_slice_dims_cover_each_extent_bucket(self):
        """A CuTe kernel over two ':' dims sizes its thread layout from both
        extents' power-of-2 buckets: calls at [48, 4, 16] and then
        [48, 4, 32] or [48, 8, 16] need their own bound kernels to cover the
        whole span, while [48, 4, 12] shares the first one."""

        @helion.kernel(static_shapes=False, autotune_effort="none")
        def double(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile_m in hl.tile(x.size(0)):
                out[tile_m, :, :] = x[tile_m, :, :] * 2
            return out

        @helion.kernel(static_shapes=False, autotune_effort="none")
        def row_sums(x: torch.Tensor) -> torch.Tensor:
            out = x.new_empty([x.size(0), x.size(1)])
            for tile_m in hl.tile(x.size(0)):
                out[tile_m, :] = x[tile_m, :, :].sum(-1)
            return out

        for shape in ((48, 4, 16), (48, 4, 32), (48, 8, 16), (48, 4, 12)):
            with self.subTest(shape=shape):
                x = torch.randn(shape, device=DEVICE)
                torch.testing.assert_close(double(x), x * 2, rtol=0, atol=0)
                torch.testing.assert_close(row_sums(x), x.sum(-1), rtol=1e-4, atol=1e-4)

    @skipUnlessBackends(["cute"])
    def test_compiled_config_refuses_a_larger_extent_bucket(self):
        """A config compiled through bound.compile_config() bakes the extent
        buckets of the bound kernel's sizes into the CuTe kernel; called
        directly at a larger bucket it raises rather than covering half of
        each row, and it still runs its own bucket and smaller ones, whose
        extents the layout's masks cover."""

        @helion.kernel(static_shapes=False, autotune_effort="none")
        def double(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile_m in hl.tile(x.size(0)):
                out[tile_m, :, :] = x[tile_m, :, :] * 2
            return out

        @helion.kernel(static_shapes=False, autotune_effort="none")
        def row_sum(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty([x.size(0)], device=x.device)
            for tile_m in hl.tile(x.size(0)):
                out[tile_m] = x[tile_m, :].sum(-1)
            return out

        x = torch.randn(48, 4, 16, device=DEVICE)
        bound = double.bind((x,))
        compiled = bound.compile_config(bound.config_spec.default_config())
        for shape in ((48, 4, 16), (48, 4, 12), (48, 4, 8), (48, 2, 16), (48, 1, 4)):
            with self.subTest(shape=shape):
                y = torch.randn(shape, device=DEVICE)
                torch.testing.assert_close(compiled(y), y * 2, rtol=0, atol=0)
        for shape in ((48, 4, 32), (48, 8, 16)):
            with (
                self.subTest(shape=shape),
                self.assertRaisesRegex(RuntimeError, "larger bucket"),
            ):
                compiled(torch.randn(shape, device=DEVICE))

        x = torch.randn(64, 1024, device=DEVICE)
        bound = row_sum.bind((x,))
        compiled = bound.compile_config(bound.config_spec.default_config())
        for n in (1024, 512, 96, 16, 0):
            with self.subTest(n=n):
                y = torch.randn(64, n, device=DEVICE)
                torch.testing.assert_close(compiled(y), y.sum(-1), rtol=1e-4, atol=1e-4)

    def test_size_one_first_call_does_not_poison_later_shapes(self):
        """A static_shapes=False kernel first bound at a size-1 dim compiles
        code that assumes it: a compile-time ``size != 1`` guards the symbol
        to 1, and the tile kernel drops that dim's index.  Later sizes, 0
        included, must get a kernel that does not assume it."""

        @helion.kernel(static_shapes=False, autotune_effort="none")
        def add_one(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                out[tile] = x[tile] + 1
            return out

        @helion.kernel(static_shapes=False, autotune_effort="none")
        def row_sum(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty([x.size(0)], device=x.device)
            for tile_m in hl.tile(x.size(0)):
                out[tile_m] = x[tile_m, :].sum(-1)
            return out

        for shapes in (((64, 1), (64, 8), (64, 0)), ((64, 1), (64, 8), (64, 32))):
            add_one.reset()
            row_sum.reset()
            for shape in shapes:
                with self.subTest(shapes=shapes, shape=shape):
                    x = torch.randn(shape, device=DEVICE)
                    torch.testing.assert_close(add_one(x), x + 1)
                    torch.testing.assert_close(row_sum(x), x.sum(-1))

    def test_empty_reduction_first_call_does_not_poison_later_shapes(self):
        """Bound at an empty reduction extent, the reduction block symbol must
        not be guarded 1-wide unkeyed: a later call with real rows has to run
        a kernel that reduces all of them."""

        # ``v - v.sum(-1, keepdim=True)`` broadcasts the reduced row against
        # the row, which is where the empty extent's block got guarded 1-wide.
        @helion.kernel(static_shapes=False, autotune_effort="none")
        def center(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile_m in hl.tile(x.size(0)):
                v = x[tile_m, :]
                out[tile_m, :] = v - v.sum(-1, keepdim=True)
            return out

        @helion.kernel(static_shapes=False, autotune_effort="none")
        def row_sum(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty([x.size(0)], device=x.device)
            for tile_m in hl.tile(x.size(0)):
                out[tile_m] = x[tile_m, :].sum(-1)
            return out

        cases = (
            (center, lambda x: x - x.sum(-1, keepdim=True)),
            (row_sum, lambda x: x.sum(-1)),
        )
        for kernel, reference in cases:
            for columns in ((0, 8), (0, 64), (1, 0, 8)):
                kernel.reset()
                for n in columns:
                    with self.subTest(kernel=kernel.name, columns=columns, n=n):
                        x = torch.randn(32, n, device=DEVICE)
                        torch.testing.assert_close(kernel(x), reference(x))

    @skipIfNotTriton("block_ptr indexing is Triton-only")
    @skipUnlessBlockPtr("block_ptr indexing is required")
    def test_codegen_size_one_guard_is_keyed(self):
        """Block-pointer codegen decides whether a dim broadcasts by comparing
        its size with 1, which records a guard after binding; the guard must
        key the bound kernel both ways (size-1 bias first, or full first)."""

        @helion.kernel(
            static_shapes=False,
            config=helion.Config(block_sizes=[8, 8, 8], indexing="block_ptr"),
        )
        def add_biases(
            x: torch.Tensor, bias1: torch.Tensor, bias2: torch.Tensor
        ) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                out[tile] = x[tile] + bias1[tile] + bias2[tile]
            return out

        broadcast = ((1, 24, 32), (16, 1, 32))
        full = ((16, 24, 32), (16, 24, 32))
        for shapes in ((broadcast, full), (full, broadcast)):
            add_biases.reset()
            for bias_shapes in shapes:
                with self.subTest(shapes=shapes, bias_shapes=bias_shapes):
                    x = torch.randn(16, 24, 32, device=DEVICE)
                    bias1 = torch.randn(bias_shapes[0], device=DEVICE)
                    bias2 = torch.randn(bias_shapes[1], device=DEVICE)
                    torch.testing.assert_close(
                        add_biases(x, bias1, bias2), x + bias1 + bias2
                    )

    def test_size_one_broadcast_dim_is_keyed(self):
        """Pointer and CuTe indexing broadcast a size-1 dim that a tile over a
        larger extent indexes.  A dynamic size is compared with 1 during code
        generation; that guard must key the bound kernel both ways (size-1
        first, or full first), for loaded operands and an ``extra_mask``."""

        @helion.kernel(static_shapes=False, autotune_effort="none")
        def add_biases(
            x: torch.Tensor, bias1: torch.Tensor, bias2: torch.Tensor
        ) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                out[tile] = x[tile] + bias1[tile] + bias2[tile]
            return out

        @helion.kernel(static_shapes=False, autotune_effort="none")
        def masked_copy(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            out = torch.zeros_like(x)
            for tile_m, tile_n, tile_k in hl.tile(x.size()):
                out[tile_m, tile_n, tile_k] = hl.load(
                    x,
                    [tile_m, tile_n, tile_k],
                    extra_mask=mask[tile_m, tile_n, tile_k],
                )
            return out

        broadcast = ((1, 8, 16), (4, 1, 16))
        full = ((4, 8, 16), (4, 8, 16))
        for shapes in ((broadcast, full), (full, broadcast)):
            add_biases.reset()
            for bias_shapes in shapes:
                with self.subTest(shapes=shapes, bias_shapes=bias_shapes):
                    x = torch.randn(4, 8, 16, device=DEVICE)
                    bias1 = torch.randn(bias_shapes[0], device=DEVICE)
                    bias2 = torch.randn(bias_shapes[1], device=DEVICE)
                    torch.testing.assert_close(
                        add_biases(x, bias1, bias2), x + bias1 + bias2
                    )

        for mask_shapes in (((1, 4, 1), (8, 4, 16)), ((8, 4, 16), (8, 1, 16))):
            masked_copy.reset()
            for mask_shape in mask_shapes:
                with self.subTest(mask_shapes=mask_shapes, mask_shape=mask_shape):
                    x = torch.randn(8, 4, 16, device=DEVICE)
                    mask = torch.rand(mask_shape, device=DEVICE) > 0.5
                    torch.testing.assert_close(
                        masked_copy(x, mask), torch.where(mask, x, 0.0)
                    )

    def test_own_extent_dims_record_no_broadcast_guard(self):
        """A dim indexed by a tile over its own size cannot broadcast, so code
        generation does not compare it with 1 and the kernel keeps a
        guard-free cache key."""

        @helion.kernel(static_shapes=False, autotune_effort="none")
        def add_one(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size(0)):
                out[tile] = x[tile] + 1
            return out

        @helion.kernel(static_shapes=False, autotune_effort="none")
        def row_sums(x: torch.Tensor) -> torch.Tensor:
            out = x.new_empty([x.size(0)])
            for tile in hl.tile(x.size(0)):
                out[tile] = x[tile, :].sum(-1)
            return out

        for shape in ((64, 32), (100, 24)):
            with self.subTest(shape=shape):
                x = torch.randn(shape, device=DEVICE)
                torch.testing.assert_close(add_one(x), x + 1)
                torch.testing.assert_close(row_sums(x), x.sum(-1))
                self.assertEqual(add_one.bind((x,))._guarded_input_facts(), [])
                self.assertEqual(row_sums.bind((x,))._guarded_input_facts(), [])

    @skipIfRefEager("compiles explicit configs")
    def test_block_size_derived_dims_record_no_broadcast_guard(self):
        """A buffer dim computed from a block size (``cdiv(n, block)``) is
        fixed by each config, not by the inputs.  Comparing it with 1 would
        evaluate the block size at its size hint and settle the dim for every
        config compiled after it."""

        @helion.kernel(static_shapes=True, autotune_effort="none")
        def tile_sums(
            x: torch.Tensor, double: hl.constexpr
        ) -> tuple[torch.Tensor, torch.Tensor]:
            rows, width = x.shape
            block = hl.register_block_size(width)
            parts = (width + block - 1) // block
            partial = torch.empty((rows, parts), device=x.device, dtype=x.dtype)
            out = torch.empty((rows,), device=x.device, dtype=x.dtype)
            for row, col in hl.tile([rows, width], block_size=[1, block]):
                partial[row, col.id] = x[row, col].sum(-1)
            hl.barrier()
            for row in hl.tile(rows):
                total = partial[row, :].sum(-1)
                out[row] = total * 2 if double else total
            return partial, out

        x = torch.arange(3 * 65, device=DEVICE, dtype=torch.float32).reshape(3, 65)
        for block, double in ((16, False), (16, True), (32, False), (128, False)):
            with self.subTest(block=block, double=double):
                bound = tile_sums.bind((x, double))
                config = bound.config_spec.default_config()
                config.config["pid_type"] = (
                    "flat" if _get_backend() == "cute" else "persistent_blocked"
                )
                config.config["block_sizes"] = [block, 4]
                config.config.pop("reduction_loops", None)
                partial, out = bound.compile_config(config)(x, double)
                expected = torch.stack(
                    [
                        x[:, start : start + block].sum(-1)
                        for start in range(0, 65, block)
                    ],
                    dim=-1,
                )
                torch.testing.assert_close(partial, expected)
                torch.testing.assert_close(out, expected.sum(-1) * (2 if double else 1))

    @skipIfCute("cute does not lower two reduction dims of different sizes")
    def test_equal_reduction_extents_at_bind_do_not_poison_later_shapes(self):
        """Bound with two equal reduction extents, a compile-time guard merges
        their reduction dims; a later call with different extents must not
        reuse that kernel."""

        @helion.kernel(static_shapes=False, autotune_effort="none")
        def two_row_sums(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.empty([x.size(0)], device=x.device)
            for tile_m in hl.tile(x.size(0)):
                out[tile_m] = x[tile_m, :].sum(-1) + y[tile_m, :].sum(-1)
            return out

        for x_cols, y_cols in ((8, 8), (8, 64), (64, 8)):
            with self.subTest(x_cols=x_cols, y_cols=y_cols):
                x = torch.randn(16, x_cols, device=DEVICE)
                y = torch.randn(16, y_cols, device=DEVICE)
                torch.testing.assert_close(two_row_sums(x, y), x.sum(-1) + y.sum(-1))

    def test_dynamic_reduction_extent_grows_across_calls(self):
        """One static_shapes=False kernel must stay correct when a later call
        grows a full-slice or reduction extent: CuTe sizes that axis's threads
        from the compile-time hint, so a larger power-of-2 bucket needs its own
        bound kernel."""

        @helion.kernel(static_shapes=False, autotune_effort="none")
        def add_one(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile_m in hl.tile(x.size(0)):
                out[tile_m, :] = x[tile_m, :] + 1
            return out

        @helion.kernel(static_shapes=False, autotune_effort="none")
        def row_softmax(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile_m in hl.tile(x.size(0)):
                out[tile_m, :] = torch.softmax(x[tile_m, :], dim=-1)
            return out

        for shape in ((64, 8), (64, 32), (100, 300), (64, 8)):
            with self.subTest(shape=shape):
                x = torch.randn(shape, device=DEVICE)
                torch.testing.assert_close(add_one(x), x + 1)
                torch.testing.assert_close(row_softmax(x), torch.softmax(x, dim=-1))


@onlyBackends(["triton", "cute"])
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

        # Keep this half-precision while staying below CuTe tcgen05 admission;
        # this test is about mark_static specialization, not MMA selection.
        m, k, n = 48, 96, 24

        # First, run WITHOUT mark_static - dimensions should NOT be constants
        x = torch.randn([m, k], device=DEVICE, dtype=HALF_DTYPE)
        y = torch.randn([k, n], device=DEVICE, dtype=HALF_DTYPE)
        code_no_spec, result_no_spec = code_and_output(
            matmul, (x, y), block_sizes=[32, 32, 32]
        )
        torch.testing.assert_close(result_no_spec, x @ y, rtol=1e-2, atol=1e-2)
        code_normalized = AssertExpectedJournal.normalize_source_comment_structure(
            code_no_spec
        )
        self.assertNotIn("48", code_normalized)
        self.assertNotIn("96", code_normalized)
        self.assertNotIn("24", code_normalized)

        # Now, run WITH mark_static - dimensions SHOULD be constants
        x_static = torch.randn([m, k], device=DEVICE, dtype=HALF_DTYPE)
        y_static = torch.randn([k, n], device=DEVICE, dtype=HALF_DTYPE)
        torch._dynamo.mark_static(x_static, [0, -1])  # test list and negative index
        torch._dynamo.mark_static(y_static, 1)

        code, result = code_and_output(
            matmul, (x_static, y_static), block_sizes=[32, 32, 32]
        )
        torch.testing.assert_close(result, x_static @ y_static, rtol=1e-2, atol=1e-2)
        self.assertIn("48", code)
        self.assertIn("96", code)
        self.assertIn("24", code)

        # Cache hit: same tensors
        self.assertIs(
            matmul.bind((x_static, y_static)), matmul.bind((x_static, y_static))
        )
        # Cache miss: different specialized values
        x2 = torch.randn([32, 80], device=DEVICE, dtype=HALF_DTYPE)
        y2 = torch.randn([80, 16], device=DEVICE, dtype=HALF_DTYPE)
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

        # Cache miss: changing externally-specialized dim
        x2 = torch.randn([320, 128], device=DEVICE)
        torch._dynamo.mark_static(x2, -1)
        self.assertIsNot(fn.bind((x,)), fn.bind((x2,)))

    @skipIfRefEager("specialization_key is not used in ref eager mode")
    def test_specialization_key_includes_hl_specialize(self):
        """Test that specialization_key() includes hl.specialize() extras after bind()."""

        @helion.kernel(static_shapes=False, autotune_effort="none")
        def fn(x: torch.Tensor) -> torch.Tensor:
            hl.specialize(x.size(-1))
            out = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                out[tile] = x[tile] * 2
            return out

        a = torch.randn([128, 64], device=DEVICE)
        b = torch.randn([128, 32], device=DEVICE)

        # Before bind: keys are equal (extras not yet known)
        key_a_before = fn.specialization_key((a,))
        key_b_before = fn.specialization_key((b,))
        self.assertEqual(key_a_before, key_b_before)

        # After bind: keys must differ because hl.specialize(x.size(-1))
        # makes the kernel depend on the last dimension
        fn.bind((a,))
        fn.bind((b,))

        key_a_after = fn.specialization_key((a,))
        key_b_after = fn.specialization_key((b,))
        self.assertNotEqual(key_a_after, key_b_after)

    @skipIfRefEager("Specialization-key schemas are not used in ref eager mode")
    def test_specialization_key_keeps_independent_extra_schemas(self) -> None:
        @helion.kernel(static_shapes=False, autotune_effort="none")
        def fn(x: torch.Tensor) -> torch.Tensor:
            return x

        a = torch.empty([8], device=DEVICE)
        b = torch.empty([16], device=DEVICE)
        signature = fn._base_specialization_key((a,))
        dynamic_fn = cast("Any", fn)

        dynamic_fn._specialize_extra[signature] = [
            lambda values: cast("torch.Tensor", values[0]).size(0)
        ]
        dynamic_fn._compiler_seed_specialize_extra.pop(signature, None)
        self.assertNotEqual(fn.specialization_key((a,)), fn.specialization_key((b,)))

        dynamic_fn._specialize_extra.pop(signature)
        dynamic_fn._compiler_seed_specialize_extra[signature] = (
            lambda values: (
                "config_num_sm",
                cast("torch.Tensor", values[0]).size(0),
            ),
        )
        self.assertNotEqual(fn.specialization_key((a,)), fn.specialization_key((b,)))


if __name__ == "__main__":
    unittest.main()
