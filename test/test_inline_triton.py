from __future__ import annotations

import torch
import triton

import helion
from helion._testing import DEVICE
from helion._testing import RefEagerTestDisabled
from helion._testing import TestCase
from helion._testing import code_and_output
import helion.language as hl


class TestInlineTriton(RefEagerTestDisabled, TestCase):
    def test_inline_triton_simple(self) -> None:
        @helion.kernel(autotune_effort="none")
        def kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.shape):
                x_val = x[tile]
                y_val = y[tile]
                result = hl.inline_triton(
                    """
                    tmp = {lhs} + {rhs}
                    tmp
                    """,
                    args={"lhs": x_val, "rhs": y_val},
                    output_like=x_val,
                )
                out[tile] = result
            return out

        x = torch.randn(128, device=DEVICE, dtype=torch.float32)
        y = torch.randn_like(x)
        code, result = code_and_output(kernel, (x, y))
        self.assertExpectedJournal(code)
        torch.testing.assert_close(result, x + y)

    def test_inline_triton_multi_output(self) -> None:
        @helion.kernel(autotune_effort="none")
        def kernel(
            a: torch.Tensor, b: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            sum_out = torch.empty_like(a)
            diff_out = torch.empty_like(a)
            for tile in hl.tile(a.shape):
                a_val = a[tile]
                b_val = b[tile]
                sum_val, diff_val = hl.inline_triton(
                    """
                    sum_val = {0} + {1}
                    diff_val = {0} - {1}
                    sum_val, diff_val
                    """,
                    args=(a_val, b_val),
                    output_like=(a_val, a_val),
                )
                sum_out[tile] = sum_val
                diff_out[tile] = diff_val
            return sum_out, diff_out

        a = torch.randn(64, device=DEVICE, dtype=torch.float32)
        b = torch.randn_like(a)
        code, (sum_result, diff_result) = code_and_output(kernel, (a, b))
        self.assertExpectedJournal(code)
        torch.testing.assert_close(sum_result, a + b)
        torch.testing.assert_close(diff_result, a - b)

    def test_inline_triton_list_args_reuse(self) -> None:
        @helion.kernel(autotune_effort="none")
        def kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)

            for tile in hl.tile(x.shape):
                x_val = x[tile]
                y_val = y[tile]
                out[tile] = hl.inline_triton(
                    """
                    triple = {0} + {0} + {0}
                    triple + {1}
                    """,
                    args=[x_val, y_val],
                    output_like=x_val,
                )

            return out

        x = torch.randn(16, device=DEVICE, dtype=torch.float32)
        y = torch.randn_like(x)
        code, out = code_and_output(kernel, (x, y))
        self.assertExpectedJournal(code)
        torch.testing.assert_close(out, 3 * x + y)

    def test_inline_triton_invalid_output_like(self) -> None:
        @helion.kernel(autotune_effort="none")
        def kernel(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.shape):
                x_val = x[tile]
                out[tile] = hl.inline_triton(
                    "{0}\n",
                    args=(x_val,),
                    output_like="not a tensor",
                )
            return out

        x = torch.randn(8, device=DEVICE, dtype=torch.float32)
        with self.assertRaises(helion.exc.InvalidAPIUsage):
            code_and_output(kernel, (x,))

    def test_inline_triton_invalid_mapping_key(self) -> None:
        @helion.kernel(autotune_effort="none")
        def kernel(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.shape):
                x_val = x[tile]
                out[tile] = hl.inline_triton(
                    "{bad}\n",
                    args={0: x_val},
                    output_like=x_val,
                )
            return out

        x = torch.randn(8, device=DEVICE, dtype=torch.float32)
        with self.assertRaises(helion.exc.InvalidAPIUsage):
            code_and_output(kernel, (x,))

    def test_inline_triton_static_assert_mismatch(self) -> None:
        @helion.kernel(autotune_effort="none")
        def kernel(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.shape):
                x_val = x[tile]
                out_like = x_val
                out[tile] = hl.inline_triton(
                    """
                    reshaped = tl.reshape({0}, (1, {0}.shape[0]))
                    reshaped
                    """,
                    args=(x_val,),
                    output_like=out_like,
                )
            return out

        x = torch.randn(8, device=DEVICE, dtype=torch.float32)
        with self.assertRaises(
            (
                triton.compiler.errors.CompilationError,
                RuntimeError,
                helion.exc.InternalError,
            )
        ):
            code_and_output(kernel, (x,))

    def test_inline_triton_side_effect_only(self) -> None:
        @helion.kernel(autotune_effort="none")
        def kernel(x: torch.Tensor) -> torch.Tensor:
            flag = torch.zeros(1, device=x.device, dtype=x.dtype)
            for tile in hl.tile(x.shape):
                val = x[tile]
                _ = hl.inline_triton(
                    "tl.store({0}, {1}[0])",
                    args=(flag, val),
                    output_like=None,
                )
            return flag

        x = torch.randn(1, device=DEVICE, dtype=torch.float32)
        bound = kernel.bind((x,))
        code = bound.to_triton_code(bound.config_spec.default_config())
        self.assertIn("tl.store(", code)
