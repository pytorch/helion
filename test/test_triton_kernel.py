from __future__ import annotations

import torch
import triton
import triton.language as tl

import helion
from helion._testing import DEVICE
from helion._testing import RefEagerTestDisabled
from helion._testing import TestCase
from helion._testing import code_and_output
import helion.language as hl

# Global constant used by triton kernel
# Use tl.constexpr() for float constants to make them accessible in @triton.jit functions
GLOBAL_SCALE_FACTOR = tl.constexpr(2.0)
GLOBAL_EPSILON = tl.constexpr(1e-6)


@triton.jit
def add_pairs(a, b):
    return a + b


# Helper function used by nested helpers
@triton.jit
def _helper_add_one(x):
    return x + 1.0


# Triton kernel that uses multiple global variables
@triton.jit
def normalize_with_globals(a):
    return a * GLOBAL_SCALE_FACTOR + GLOBAL_EPSILON


# Triton kernel with nested helper calls
@triton.jit
def _helper_double(x):
    return x * 2.0


@triton.jit
def _helper_process(x):
    doubled = _helper_double(x)
    return _helper_add_one(doubled)


@triton.jit
def nested_helper_calls(a):
    return _helper_process(a)


@triton.jit
def pairwise_ops(a, b):
    sum_val = a + b
    prod_val = a * b
    return sum_val, prod_val


@triton.jit
def vector_mix_norm(a, b):
    mixed = tl.where(a > b, a - b, a + b)
    l2sq = tl.sum(mixed * mixed)
    inv = tl.rsqrt(tl.maximum(l2sq, 1e-12))
    return mixed * inv


@triton.jit
def side_effect_noop(x):
    # Use debug_barrier as a side-effect that doesn't require output
    tl.debug_barrier()


@helion.kernel(autotune_effort="none")
def triton_kernel_add_pairs(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.shape):
        x_val = x[tile]
        y_val = y[tile]
        # Pass by function name string to avoid closures in kernels
        result = hl.triton_kernel("add_pairs", args=(x_val, y_val), output_like=x_val)
        out[tile] = result
    return out


class TestTritonKernel(RefEagerTestDisabled, TestCase):
    def test_triton_kernel_simple_add(self) -> None:
        x = torch.randn(128, device=DEVICE, dtype=torch.float32)
        y = torch.randn_like(x)
        code, result = code_and_output(triton_kernel_add_pairs, (x, y))
        self.assertIn("@triton.jit", code)
        self.assertIn("add_pairs", code)
        torch.testing.assert_close(result, x + y)
        self.assertExpectedJournal(code)

    def test_triton_kernel_multi_output(self) -> None:
        @helion.kernel(autotune_effort="none")
        def k(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            sum_out = torch.empty_like(x)
            prod_out = torch.empty_like(x)
            for tile in hl.tile(x.shape):
                x_val = x[tile]
                y_val = y[tile]
                sum_val, prod_val = hl.triton_kernel(
                    "pairwise_ops",
                    args=(x_val, y_val),
                    output_like=(x_val, x_val),
                )
                sum_out[tile] = sum_val
                prod_out[tile] = prod_val
            return sum_out, prod_out

        x = torch.randn(64, device=DEVICE, dtype=torch.float32)
        y = torch.randn_like(x)
        code, (sum_result, prod_result) = code_and_output(k, (x, y))
        self.assertIn("pairwise_ops", code)
        torch.testing.assert_close(sum_result, x + y)
        torch.testing.assert_close(prod_result, x * y)
        self.assertExpectedJournal(code)

    def test_triton_kernel_tl_ops(self) -> None:
        @helion.kernel(autotune_effort="none")
        def k(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.shape):
                x_val = x[tile]
                y_val = y[tile]
                out[tile] = hl.triton_kernel(
                    "vector_mix_norm",
                    args=(x_val, y_val),
                    output_like=x_val,
                )
            return out

        x = torch.randn(96, device=DEVICE, dtype=torch.float32)
        y = torch.randn_like(x)
        code, result = code_and_output(k, (x, y))
        self.assertIn("vector_mix_norm", code)

        bs = 32
        expected = torch.empty_like(x)
        for i in range(0, x.numel(), bs):
            xa = x[i : i + bs]
            ya = y[i : i + bs]
            mixed = torch.where(xa > ya, xa - ya, xa + ya)
            l2sq = torch.sum(mixed * mixed)
            inv = torch.rsqrt(
                torch.maximum(l2sq, torch.tensor(1e-12, device=DEVICE, dtype=x.dtype))
            )
            expected[i : i + bs] = mixed * inv
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)
        self.assertExpectedJournal(code)

    def test_triton_kernel_with_multiple_globals(self) -> None:
        """Test that triton_kernel correctly copies global variables."""

        @helion.kernel(autotune_effort="none")
        def k(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.shape):
                x_val = x[tile]
                out[tile] = hl.triton_kernel(
                    "normalize_with_globals",
                    args=(x_val,),
                    output_like=x_val,
                )
            return out

        x = torch.randn(128, device=DEVICE, dtype=torch.float32)
        code, result = code_and_output(k, (x,))
        # Verify both global variables are copied
        self.assertIn("GLOBAL_SCALE_FACTOR", code)
        self.assertIn("GLOBAL_EPSILON", code)
        self.assertIn("normalize_with_globals", code)
        # Use .value to extract the raw float from tl.constexpr for comparison
        torch.testing.assert_close(
            result, x * GLOBAL_SCALE_FACTOR.value + GLOBAL_EPSILON.value
        )
        self.assertExpectedJournal(code)

    def test_triton_kernel_with_nested_helpers(self) -> None:
        """Test that triton_kernel correctly copies nested helper functions."""

        @helion.kernel(autotune_effort="none")
        def k(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.shape):
                x_val = x[tile]
                out[tile] = hl.triton_kernel(
                    "nested_helper_calls",
                    args=(x_val,),
                    output_like=x_val,
                )
            return out

        x = torch.randn(128, device=DEVICE, dtype=torch.float32)
        code, result = code_and_output(k, (x,))
        # Verify all nested helper functions are copied
        self.assertIn("nested_helper_calls", code)
        self.assertIn("_helper_process", code)
        self.assertIn("_helper_double", code)
        self.assertIn("_helper_add_one", code)
        # Expected: (x * 2.0) + 1.0
        torch.testing.assert_close(result, x * 2.0 + 1.0)
        self.assertExpectedJournal(code)

    def test_triton_kernel_output_like_none(self) -> None:
        """Test that triton_kernel with output_like=None emits the call."""

        @helion.kernel(autotune_effort="none")
        def k(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.shape):
                x_val = x[tile]
                # Call triton_kernel with output_like=None (side-effect only)
                hl.triton_kernel(
                    "side_effect_noop",
                    args=(x_val,),
                    output_like=None,
                )
                out[tile] = x_val
            return out

        x = torch.randn(128, device=DEVICE, dtype=torch.float32)
        code, result = code_and_output(k, (x,))
        # Verify the side-effect function is included in generated code
        self.assertIn("side_effect_noop", code)
        self.assertIn("tl.debug_barrier", code)
        # Output should be unchanged (side_effect_noop doesn't modify x_val)
        torch.testing.assert_close(result, x)
        self.assertExpectedJournal(code)

    def test_triton_kernel_function_object(self) -> None:
        """Test that triton_kernel accepts function objects directly (not just function string names)."""

        @helion.kernel(autotune_effort="none")
        def k(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.shape):
                x_val = x[tile]
                y_val = y[tile]
                # Pass function object directly instead of string name
                result = hl.triton_kernel(
                    add_pairs, args=(x_val, y_val), output_like=x_val
                )
                out[tile] = result
            return out

        x = torch.randn(128, device=DEVICE, dtype=torch.float32)
        y = torch.randn_like(x)
        code, result = code_and_output(k, (x, y))
        self.assertIn("@triton.jit", code)
        self.assertIn("add_pairs", code)
        torch.testing.assert_close(result, x + y)
        self.assertExpectedJournal(code)

    def test_triton_kernel_function_object_with_helpers(self) -> None:
        """Test that triton_kernel with function object correctly copies nested helpers."""

        @helion.kernel(autotune_effort="none")
        def k(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.shape):
                x_val = x[tile]
                # Pass function object directly
                out[tile] = hl.triton_kernel(
                    nested_helper_calls,
                    args=(x_val,),
                    output_like=x_val,
                )
            return out

        x = torch.randn(128, device=DEVICE, dtype=torch.float32)
        code, result = code_and_output(k, (x,))
        # Verify all nested helper functions are copied
        self.assertIn("nested_helper_calls", code)
        self.assertIn("_helper_process", code)
        self.assertIn("_helper_double", code)
        self.assertIn("_helper_add_one", code)
        # Expected: (x * 2.0) + 1.0
        torch.testing.assert_close(result, x * 2.0 + 1.0)
        self.assertExpectedJournal(code)


if __name__ == "__main__":
    import unittest

    unittest.main()
