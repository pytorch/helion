from __future__ import annotations

import math
import unittest

import torch

import helion
from helion._testing import DEVICE
from helion._testing import RefEagerTestBase
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import skipIfRefEager
from helion.exc import ShapeSpecializingAllocation
import helion.language as hl


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


if __name__ == "__main__":
    unittest.main()
