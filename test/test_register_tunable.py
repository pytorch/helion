from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

import helion
from helion import _compat
from helion._testing import DEVICE
from helion._testing import HALF_DTYPE
from helion._testing import RefEagerTestBase
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
from helion._testing import skipIfRefEager
from helion._testing import skipIfSharedMemoryLessThan
from helion._testing import xfailIfCute
from helion.autotuner import EnumFragment
from helion.autotuner import IntegerFragment
from helion.autotuner import PowerOfTwoFragment
import helion.language as hl
from helion.language import loops


@onlyBackends(["triton", "cute"])
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
        default_config = repr(
            kernel_with_int_param.bind((x,)).config_spec.default_config()
        )
        self.assertIn("multiplier=3", default_config)

    @skipIfRefEager("requires compiling a tunable reduction extent")
    def test_tunable_reduction_extent_expressions(self):
        @helion.kernel(autotune_effort="none")
        def reduction_width_plus_one(x: torch.Tensor) -> torch.Tensor:
            width = hl.register_tunable("width", IntegerFragment(2, 8, 4))
            tmp = torch.zeros([x.size(0), width + 1], dtype=x.dtype, device=x.device)
            out = torch.empty_like(x)
            for tile_m in hl.tile(x.size(0)):
                out[tile_m] = torch.sum(tmp[tile_m, :] + x[tile_m, None], dim=-1)
            return out

        @helion.kernel(autotune_effort="none")
        def reduction_twice_width(x: torch.Tensor) -> torch.Tensor:
            width = hl.register_tunable("width", IntegerFragment(2, 8, 4))
            tmp = torch.zeros([x.size(0), 2 * width], dtype=x.dtype, device=x.device)
            out = torch.empty_like(x)
            for tile_m in hl.tile(x.size(0)):
                out[tile_m] = torch.sum(tmp[tile_m, :] + x[tile_m, None], dim=-1)
            return out

        @helion.kernel(autotune_effort="none")
        def reduction_two_tunables(x: torch.Tensor) -> torch.Tensor:
            width = hl.register_tunable("width", IntegerFragment(2, 8, 4))
            extra = hl.register_tunable("extra", IntegerFragment(1, 4, 2))
            tmp = torch.zeros(
                [x.size(0), width + extra], dtype=x.dtype, device=x.device
            )
            out = torch.empty_like(x)
            for tile_m in hl.tile(x.size(0)):
                out[tile_m] = torch.sum(tmp[tile_m, :] + x[tile_m, None], dim=-1)
            return out

        x = torch.randn(16, device=DEVICE, dtype=torch.float32)
        selected_width = 6
        cases = (
            ("width_plus_one", reduction_width_plus_one, selected_width + 1),
            ("twice_width", reduction_twice_width, 2 * selected_width),
        )
        for name, kernel, scale in cases:
            with self.subTest(kernel=name):
                _code, result = code_and_output(
                    kernel, (x,), block_size=8, width=selected_width
                )
                torch.testing.assert_close(result, x * scale)

        _code, result = code_and_output(
            reduction_two_tunables,
            (x,),
            block_size=8,
            width=selected_width,
            extra=3,
        )
        torch.testing.assert_close(result, x * (selected_width + 3))

        reduction_specs = reduction_twice_width.bind((x,)).config_spec.reduction_loops
        self.assertEqual(len(reduction_specs), 1)
        self.assertEqual(reduction_specs[0].max_size_hint, 16)
        self.assertIsNotNone(reduction_specs[0].tunable_extent)

    @skipIfRefEager("requires compiling a tunable reduction extent")
    def test_tunable_reduction_zero_and_negative_selected_extents(self) -> None:
        @helion.kernel(autotune_effort="none")
        def reduction_width(x: torch.Tensor) -> torch.Tensor:
            width = hl.register_tunable("width", IntegerFragment(-2, 4, 2))
            tmp = torch.zeros([x.size(0), width], dtype=x.dtype, device=x.device)
            out = torch.empty_like(x)
            for tile_m in hl.tile(x.size(0)):
                out[tile_m] = torch.sum(tmp[tile_m, :], dim=-1)
            return out

        x = torch.randn(16, device=DEVICE, dtype=torch.float32)
        _code, result = code_and_output(
            reduction_width,
            (x,),
            block_size=8,
            width=0,
        )
        torch.testing.assert_close(result, torch.zeros_like(x))

        with self.assertRaisesRegex(
            helion.exc.InvalidConfig,
            "Tunable reduction extent.*evaluated to -1",
        ):
            code_and_output(
                reduction_width,
                (x,),
                block_size=8,
                width=-1,
            )

    @xfailIfCute("CuTe thread axis collision with differently-sized reduction blocks")
    def test_tunable_reduction_extent_does_not_alias_fixed_hint(self):
        @helion.kernel(autotune_effort="none")
        def mixed_reductions(
            x: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            width = hl.register_tunable("width", IntegerFragment(2, 8, 4))
            fixed = torch.zeros([x.size(0), 4], dtype=x.dtype, device=x.device)
            tuned = torch.zeros([x.size(0), width], dtype=x.dtype, device=x.device)
            fixed_out = torch.empty_like(x)
            tuned_out = torch.empty_like(x)
            for tile_m in hl.tile(x.size(0)):
                fixed_out[tile_m] = torch.sum(
                    fixed[tile_m, :] + x[tile_m, None], dim=-1
                )
                tuned_out[tile_m] = torch.sum(
                    tuned[tile_m, :] + x[tile_m, None], dim=-1
                )
            return fixed_out, tuned_out

        x = torch.randn(16, device=DEVICE, dtype=torch.float32)
        _code, (fixed_out, tuned_out) = code_and_output(
            mixed_reductions, (x,), block_size=8, width=6
        )
        torch.testing.assert_close(fixed_out, x * 4)
        torch.testing.assert_close(tuned_out, x * 6)

    @skipIfRefEager("requires compiling a tunable reduction extent")
    def test_tunable_reduction_extent_with_block_symbol(self):
        @helion.kernel(autotune_effort="none")
        def mixed_extent(x: torch.Tensor) -> torch.Tensor:
            block = hl.register_block_size(2, 16)
            width = hl.register_tunable("width", IntegerFragment(2, 8, 4))
            tmp = torch.zeros(
                [x.size(0), block + width], dtype=x.dtype, device=x.device
            )
            out = torch.empty_like(x)
            for tile_m in hl.tile(x.size(0)):
                out[tile_m] = torch.sum(tmp[tile_m, :] + x[tile_m, None], dim=-1)
            return out

        x = torch.randn(16, device=DEVICE, dtype=torch.float32)
        bound = mixed_extent.bind((x,))
        self.assertEqual(len(bound.config_spec.reduction_loops), 1)
        extent = bound.config_spec.reduction_loops[0].tunable_extent
        self.assertIsNotNone(extent)
        assert extent is not None
        self.assertEqual(len(extent.block_symbol_ids), 1)
        self.assertEqual(bound.config_spec.reduction_loops[0].max_size_hint, 24)
        bound.config_spec.default_config()

        @helion.kernel(autotune_effort="none")
        def fixed_block_extent(x: torch.Tensor) -> torch.Tensor:
            width = hl.register_tunable("width", IntegerFragment(2, 8, 4))
            out = torch.empty_like(x)
            for tile_m in hl.tile(x.size(0), block_size=8):
                tmp = hl.zeros([tile_m.block_size + width], dtype=x.dtype)
                out[tile_m] = x[tile_m] + torch.sum(tmp)
            return out

        fixed_bound = fixed_block_extent.bind((x,))
        fixed_spec = fixed_bound.config_spec.reduction_loops[0]
        fixed_extent = fixed_spec.tunable_extent
        self.assertIsNotNone(fixed_extent)
        assert fixed_extent is not None
        self.assertEqual(fixed_extent.block_symbol_ids, ())
        self.assertEqual(fixed_spec.max_size_hint, 16)
        fixed_bound.config_spec.default_config()

        @helion.kernel(autotune_effort="none")
        def inverse_block_extent(x: torch.Tensor) -> torch.Tensor:
            width = hl.register_tunable("width", IntegerFragment(2, 64, 4))
            out = torch.empty_like(x)
            for tile_m in hl.tile(x.size(0)):
                tmp = hl.zeros([2048 // tile_m.block_size + width], dtype=x.dtype)
                out[tile_m] = x[tile_m] + torch.sum(tmp)
            return out

        inverse_bound = inverse_block_extent.bind((x,))
        inverse_spec = inverse_bound.config_spec.reduction_loops[0]
        self.assertEqual(inverse_spec.max_size_hint, 2112)
        inverse_bound.config_spec.default_config()

    def test_tunable_reduction_extent_with_dynamic_input_symbol(self):
        @helion.kernel(
            backend="triton",
            static_shapes=False,
            autotune_effort="none",
        )
        def dynamic_extent(x: torch.Tensor) -> torch.Tensor:
            width = hl.register_tunable("width", IntegerFragment(2, 8, 4))
            tmp = torch.zeros(
                [x.size(0), x.size(1) + width], dtype=x.dtype, device=x.device
            )
            out = torch.empty([x.size(0)], dtype=x.dtype, device=x.device)
            for tile_m in hl.tile(x.size(0)):
                out[tile_m] = torch.sum(tmp[tile_m, :] + x[tile_m, 0, None], dim=-1)
            return out

        x = torch.randn(16, 7, device=DEVICE, dtype=torch.float32)
        _code, result = code_and_output(dynamic_extent, (x,), block_size=8, width=6)
        torch.testing.assert_close(result, x[:, 0] * 13)

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

    @skipIfRefEager("requires compiling a tunable reduction extent")
    def test_integer_enum_reduction_extent(self):
        @helion.kernel(autotune_effort="none")
        def enum_reduction(x: torch.Tensor) -> torch.Tensor:
            width = hl.register_tunable("width", EnumFragment((4, 128)))
            tmp = torch.zeros([x.size(0), width], dtype=x.dtype, device=x.device)
            out = torch.empty_like(x)
            for tile_m in hl.tile(x.size(0)):
                out[tile_m] = torch.sum(tmp[tile_m, :] + x[tile_m, None], dim=-1)
            return out

        x = torch.randn(16, device=DEVICE, dtype=torch.float32)
        _code, result = code_and_output(enum_reduction, (x,), block_size=8, width=128)
        torch.testing.assert_close(result, x * 128)
        spec = enum_reduction.bind((x,)).config_spec.reduction_loops[0]
        self.assertEqual(spec.max_size_hint, 128)

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
        torch.testing.assert_close(result, x.sum())

    @patch.object(_compat, "_supports_tensor_descriptor", lambda: False)
    @skipIfSharedMemoryLessThan(
        86016, reason="num_stages=8 requires 86016 bytes of shared memory"
    )
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
        x = torch.randn([m, k], device=DEVICE, dtype=HALF_DTYPE)
        y = torch.randn([k, n], device=DEVICE, dtype=HALF_DTYPE)

        code, result = code_and_output(matmul_split_k, (x, y))
        expected = (
            (x.cpu().float() @ y.cpu().float()).to(result.dtype).to(result.device)
        )
        torch.testing.assert_close(result, expected, rtol=1e-2, atol=1)
        self.assertIsInstance(
            self.getUserDefinedTunable(
                matmul_split_k.bind((x, y)).config_spec.user_defined_tunables, "split_k"
            ),
            PowerOfTwoFragment,
        )


if __name__ == "__main__":
    unittest.main()
