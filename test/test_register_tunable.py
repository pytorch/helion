from __future__ import annotations

import re
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
from helion._testing.selected_tunable import mixed_selected_nested_arithmetic
from helion._testing.selected_tunable import mixed_selected_scalar_uses
from helion._testing.selected_tunable import mixed_selected_tensor_forms
from helion._testing.selected_tunable import mixed_selected_tensor_product
from helion._testing.selected_tunable import nested_selected_float_results
from helion._testing.selected_tunable import selected_boolean_forms
from helion._testing.selected_tunable import selected_tensor_cast_chain
from helion._testing.selected_tunable import selected_tensor_unary
from helion.autotuner import BooleanFragment
from helion.autotuner import EnumFragment
from helion.autotuner import IntegerFragment
from helion.autotuner import PowerOfTwoFragment
from helion.autotuner.config_fragment import NumThreadsFragment
import helion.language as hl
from helion.language import loops

_DEVICE_FLAG = BooleanFragment()
_DEVICE_COUNT = NumThreadsFragment(128)
_DEVICE_FACTOR = EnumFragment((2, 16))


@onlyBackends(["triton", "cute"])
class TestRegisterTunable(RefEagerTestBase, TestCase):
    maxDiff = 10000

    @skipIfRefEager("compares selected scalar consumers through compiled device code")
    def test_mixed_selected_scalar_consumers(self):
        # Each consumer has distinct scalar lowering; both input dtypes are
        # needed to observe floating promotion and integer output conversion.
        for operation in ("clamp", "full_like", "device_arithmetic", "host_arithmetic"):
            for dtype in (torch.float32, torch.int32):
                x = torch.arange(-34, 34, device=DEVICE, dtype=dtype)[::2]
                original = x.clone()
                bound = mixed_selected_scalar_uses.bind((x, operation))
                compiled = {}
                for selected in (0.5, 2, 0.5):
                    with self.subTest(
                        operation=operation, dtype=dtype, selected=selected
                    ):
                        if selected not in compiled:
                            config = helion.Config(block_sizes=[32], selected=selected)
                            compiled[selected] = bound.compile_config(config)
                        actual = compiled[selected](x, operation)
                        expected = (
                            x.clamp(max=selected)
                            if operation == "clamp"
                            else torch.full_like(x, selected)
                            if operation == "full_like"
                            else x * (selected * 2)
                        )
                        self.assertEqual(actual.dtype, x.dtype)
                        torch.testing.assert_close(
                            actual, expected.to(x.dtype), rtol=0, atol=0
                        )
                torch.testing.assert_close(x, original, rtol=0, atol=0)

    @skipIfRefEager("compares selected cast chains through compiled device code")
    def test_selected_tensor_cast_chain(self):
        for mode, dtype in (
            ("chain", torch.float32),
            ("chain", torch.int64),
            ("alias_chain", torch.float32),
            ("alias_chain", torch.int64),
        ):
            x = torch.arange(-17, 17, device=DEVICE, dtype=dtype)[::2]
            bound = selected_tensor_cast_chain.bind((x, mode))
            for selected in (0.5, 2):
                with self.subTest(mode=mode, dtype=dtype, selected=selected):
                    config = helion.Config(block_sizes=[16], selected=selected)
                    actual = bound.compile_config(config)(x, mode)
                    cast_dtype = torch.float32 if mode == "alias_chain" else dtype
                    expected = (
                        (x * selected).to(torch.int32).to(cast_dtype)
                        + selected * selected
                    ).to(dtype)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    @skipIfRefEager("compares selected unary forms through compiled device code")
    def test_selected_tensor_unary(self):
        for operation, dtype in (
            ("abs", torch.float32),
            ("method_abs", torch.int32),
            ("negative", torch.float32),
            ("method_negative", torch.int32),
        ):
            x = torch.arange(-17, 17, device=DEVICE, dtype=dtype)[::2]
            original = x.clone()
            bound = selected_tensor_unary.bind((x, operation))
            compiled = {}
            for selected in (0.5, 2, 0.5):
                with self.subTest(operation=operation, dtype=dtype, selected=selected):
                    if selected not in compiled:
                        config = helion.Config(block_sizes=[16], selected=selected)
                        compiled[selected] = bound.compile_config(config)
                    actual = compiled[selected](x, operation)
                    value = x * selected
                    expected = (
                        value.abs() if "abs" in operation else -value
                    ) + selected
                    torch.testing.assert_close(
                        actual, expected.to(dtype), rtol=0, atol=0
                    )
            torch.testing.assert_close(x, original, rtol=0, atol=0)

    @skipIfRefEager("compares selected tensor forms through compiled device code")
    def test_compatible_selected_tensor_forms(self):
        # Float compatibility and integer promotion need separate actual dtypes;
        # where's erased integer-to-float scalar cast remains a CPU refusal.
        for operation, dtype in (
            ("augassign", torch.float32),
            ("where", torch.float32),
            ("method", torch.float32),
            ("augassign", torch.int32),
            ("method", torch.int32),
        ):
            x = torch.arange(-34, 34, device=DEVICE, dtype=dtype)[::2]
            original = x.clone()
            bound = mixed_selected_tensor_forms.bind((x, operation))
            for selected in (1, 0.5, 1):
                with self.subTest(operation=operation, dtype=dtype, selected=selected):
                    config = helion.Config(block_sizes=[32], selected=selected)
                    actual = bound.compile_config(config)(x, operation)
                    expected = (
                        torch.where(x > 0, x, selected)
                        if operation == "where"
                        else x * selected
                    )
                    torch.testing.assert_close(
                        actual, expected.to(x.dtype), rtol=0, atol=0
                    )
            torch.testing.assert_close(x, original, rtol=0, atol=0)

    @skipIfRefEager(
        "checks Boolean condition spelling without erasing integer promotion"
    )
    def test_boolean_integer_spelling(self):
        for operation in ("branch", "arithmetic"):
            x = (
                torch.tensor([-3.0, 5.0], device=DEVICE)
                if operation == "branch"
                else torch.tensor([True, False], device=DEVICE)
            )
            bound = selected_boolean_forms.bind((x, operation))
            for flag in (False, 0, True, 1, True):
                with self.subTest(
                    operation=operation, flag=flag, scalar_type=type(flag)
                ):
                    config = helion.Config(block_sizes=[32], flag=flag)
                    actual = bound.compile_config(config)(x, operation)
                    if operation == "branch":
                        expected = x + 2 if flag else x - 3
                    else:
                        expected = (x * flag + flag).to(torch.int64)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    @skipIfRefEager("compares configured scalar types through nested device code")
    def test_nested_boolean_selection_and_reverse_subtraction(self):
        x = torch.arange(-323, 323, device=DEVICE, dtype=torch.int32)
        x = x.reshape(17, 38)[:, ::2]
        original = x.clone()
        for choices, reverse in (((True, 2), False), ((2, 4.5), True)):
            bound = mixed_selected_nested_arithmetic.bind((x, choices, reverse))
            for selected in (*choices, choices[0]):
                with self.subTest(choices=choices, selected=selected):
                    config = helion.Config(block_sizes=[8, 16], selected=selected)
                    actual = bound.compile_config(config)(x, choices, reverse)
                    expected = selected - x if reverse else x * selected + 1
                    torch.testing.assert_close(
                        actual, expected.to(x.dtype), rtol=0, atol=0
                    )
        torch.testing.assert_close(x, original, rtol=0, atol=0)

    @skipIfRefEager("compares mixed selected scalar types through compiled device code")
    def test_mixed_selected_integer_tensor_product(self):
        x = torch.arange(-323, 323, device=DEVICE, dtype=torch.int32)
        x = x.reshape(17, 38)[:, ::2]
        original = x.clone()
        for nested in (False, True):
            bound = mixed_selected_tensor_product.bind((x, nested))
            for selected in (2, 4.5, 2):
                with self.subTest(nested=nested, selected=selected):
                    config = helion.Config(
                        block_sizes=[8, 16] if nested else [8], selected=selected
                    )
                    actual = bound.compile_config(config)(x, nested)
                    expected = (x * selected).to(x.dtype)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(x, original, rtol=0, atol=0)

    @skipIfRefEager("compares opposite selected zero signs and nested lowering")
    def test_nested_selected_float_signs(self):

        x = torch.arange(-323, 323, device=DEVICE, dtype=torch.float32)
        x = x.reshape(17, 38)[:, ::2]
        x[0, 0], x[0, 1] = -0.0, 0.0
        original = x.clone()
        bound = nested_selected_float_results.bind((x,))
        # Revisit both signs in the same cache, without recompiling each value.
        for selected in (-0.0, 0.0, -0.0, 1.0, 0.0):
            with self.subTest(value=selected.hex()):
                config = helion.Config(block_sizes=[8, 16], value=selected)
                outputs = bound.compile_config(config)(x)
                factors = (selected, selected / 2, -selected, selected * x.size(0))
                for actual, factor in zip(outputs, factors, strict=True):
                    expected = x * factor
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    torch.testing.assert_close(
                        torch.signbit(actual), torch.signbit(expected)
                    )
        torch.testing.assert_close(x, original, rtol=0, atol=0)
        torch.testing.assert_close(torch.signbit(x), torch.signbit(original))

    @skipIfRefEager("compares configured nested Boolean and tile-width values")
    def test_outer_tunable_negation_and_nested_tile_width(self):
        @helion.kernel(autotune_effort="none", static_shapes=False)
        def fn(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for row in hl.tile(x.size(0)):
                flag = hl.register_tunable("flag", BooleanFragment())
                width = hl.register_tunable("width", EnumFragment((8, 16)))
                enabled = not flag
                for column in hl.tile(x.size(1), block_size=width):
                    out[row, column] = x[row, column] * enabled + column.block_size
            return out

        x = torch.arange(17 * 19, device=DEVICE, dtype=torch.float32).reshape(17, 19)
        bound = fn.bind((x,))
        for flag, width in ((False, 8), (True, 16)):
            with self.subTest(flag=flag, width=width):
                config = helion.Config(block_sizes=[8], flag=flag, width=width)
                actual = bound.compile_config(config)(x)
                torch.testing.assert_close(actual, x * (not flag) + width)

    @skipIfRefEager("compares selected nested device expressions with eager arithmetic")
    def test_nested_derived_tunable_values(self):
        @helion.kernel(autotune_effort="none", static_shapes=False)
        def fn(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            out = torch.empty_like(x)
            mixed_out = torch.empty_like(x)
            for row in hl.tile(x.size(0)):
                integer = hl.register_tunable("integer", EnumFragment((2, 16)))
                fraction = hl.register_tunable("fraction", EnumFragment((2.5, 4.5)))
                mixed = hl.register_tunable("mixed", EnumFragment((2, 4.5)))
                value = integer + fraction / 2 + row.block_size + x.size(0)
                for column in hl.tile(x.size(1)):
                    out[row, column] = x[row, column] * value
                    mixed_out[row, column] = x[row, column] * mixed
            return out, mixed_out

        x = torch.arange(17 * 19, device=DEVICE, dtype=torch.float32).reshape(17, 19)
        bound = fn.bind((x,))
        for integer, fraction, mixed in ((2, 2.5, 2), (16, 4.5, 4.5)):
            with self.subTest(integer=integer, fraction=fraction):
                config = helion.Config(
                    block_sizes=[8, 16], integer=integer, fraction=fraction, mixed=mixed
                )
                actual, mixed_actual = bound.compile_config(config)(x)
                expected = x * (integer + fraction / 2 + 8 + x.size(0))
                torch.testing.assert_close(actual, expected)
                torch.testing.assert_close(mixed_actual, x * mixed, rtol=0, atol=0)

    @skipIfRefEager("compiles two selected configurations for one binding")
    def test_device_loop_selected_fragments(self):
        @helion.kernel(autotune_effort="none")
        def fn(x: torch.Tensor, inline: hl.constexpr) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.numel()):
                if inline:
                    flag = hl.register_tunable("flag", BooleanFragment())
                    count = hl.register_tunable("count", NumThreadsFragment(128))
                    factor = hl.register_tunable("factor", EnumFragment((2, 16)))
                else:
                    flag = hl.register_tunable("flag", _DEVICE_FLAG)
                    count = hl.register_tunable("count", _DEVICE_COUNT)
                    factor = hl.register_tunable("factor", _DEVICE_FACTOR)
                out[tile] = x[tile] + flag + count + factor
            return out

        x = torch.arange(37, device=DEVICE, dtype=torch.float32)
        # Exercise Boolean typing and nondefault integer values on real device
        # IR, with one partial tile. Inline identity-equality fragments exercise
        # repeated construction, while globals exercise snapshot reuse.
        for inline in (False, True):
            with self.subTest(inline=inline):
                bound = fn.bind((x, inline))
                for flag, count, factor in ((False, 0, 2), (True, 64, 16)):
                    config = helion.Config(
                        block_sizes=[32], flag=flag, count=count, factor=factor
                    )
                    actual = bound.compile_config(config)(x, inline)
                    torch.testing.assert_close(actual, x + flag + count + factor)

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

    def test_inline_host_helper_block_size(self):
        """A host helper over a tunable written inline in a block size has no
        host variable naming its result; the tile takes the value the helper
        gives for the config."""

        @helion.kernel(autotune_effort="none")
        def kernel_with_inline_helper(x: torch.Tensor) -> torch.Tensor:
            (n,) = x.size()
            out = torch.empty_like(x)
            foo = hl.register_tunable("foo", PowerOfTwoFragment(16, 256))
            for tile_n in hl.tile([n], block_size=[helion.next_power_of_2(foo)]):
                out[tile_n] = x[tile_n] * 2.0
            return out

        x = torch.randn(1000, device=DEVICE, dtype=torch.float32)
        code, result = code_and_output(kernel_with_inline_helper, (x,), foo=64)
        torch.testing.assert_close(result, x * 2.0)
        if not self._in_ref_eager_mode:
            # Masking keeps the output right at any block size: check the one
            # the config gives (the default foo is 16).
            self.assertRegex(code, r"_BLOCK_SIZE_0 = (tl\.constexpr\()?64\b")

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

    @skipIfRefEager("to_triton_code is not supported in ref eager mode")
    def test_tunable_device_arguments_are_constexpr(self):
        """Config-backed tunables must be compile-time device arguments."""

        @helion.kernel(static_shapes=True)
        def fn(x: torch.Tensor) -> torch.Tensor:
            (n,) = x.size()
            out = torch.empty_like(x)
            multiplier = hl.register_tunable("multiplier", IntegerFragment(1, 8, 2))
            use_add = hl.register_tunable("use_add", BooleanFragment())

            for tile in hl.tile(n):
                if use_add:
                    out[tile] = x[tile] + multiplier
                else:
                    out[tile] = x[tile] * multiplier
            return out

        x = torch.randn(128, device=DEVICE)
        bound = fn.bind((x,))

        for use_add in (False, True):
            config = helion.Config(block_sizes=[64], multiplier=3, use_add=use_add)
            code = bound.to_triton_code(config)
            signature = re.search(r"^def _helion_fn\((.*)\):$", code, re.MULTILINE)
            assert signature is not None
            constexpr_type = bound.env.backend.constexpr_type
            self.assertIn(f"multiplier: {constexpr_type}", signature.group(1))
            self.assertIn(f"use_add: {constexpr_type}", signature.group(1))

            result = bound.compile_config(config)(x)
            expected = x + 3 if use_add else x * 3
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
        # The blocked partial sums reorder a float32 reduction over values of
        # magnitude ~1; a near-zero total leaves nothing for rtol to scale, so
        # allow the reordering noise explicitly instead of the 1e-5 default.
        torch.testing.assert_close(result, x.sum(), rtol=1e-4, atol=1e-4)

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
