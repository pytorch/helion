from __future__ import annotations

import contextlib
import functools
import io
import math
import os
import unittest

import torch

from . import test_dot
from . import test_examples
from . import test_grid
from . import test_loops
import helion
from helion._testing import TestCase
import helion.language as hl
from helion.language.ref_tile import RefTile
from helion.language.tile_proxy import Tile


@contextlib.contextmanager
def assert_helion_compilation(has_compilation: bool = True):
    """Context manager that asserts Helion compilation does happen / does not happen."""
    # Store original __init__ methods
    original_tile_init = Tile.__init__
    original_reftile_init = RefTile.__init__
    tile_init_called = []
    reftile_init_called = []

    def tracked_tile_init(self, *args, **kwargs):
        tile_init_called.append(True)
        return original_tile_init(self, *args, **kwargs)

    def tracked_reftile_init(self, *args, **kwargs):
        reftile_init_called.append(True)
        return original_reftile_init(self, *args, **kwargs)

    # Patch the __init__ methods
    Tile.__init__ = tracked_tile_init
    RefTile.__init__ = tracked_reftile_init

    try:
        yield

        # Check if init was called as expected
        if has_compilation:
            assert tile_init_called, "Tile.__init__ was not called in normal mode"
            assert not reftile_init_called, (
                "RefTile.__init__ was called when it should not have been in normal mode"
            )
        else:
            assert not tile_init_called, (
                "Tile.__init__ was called when it should not have been in ref eager mode"
            )
            assert reftile_init_called, (
                "RefTile.__init__ was not called when ref eager mode was expected"
            )
    finally:
        # Restore original __init__ methods
        Tile.__init__ = original_tile_init
        RefTile.__init__ = original_reftile_init


assert_no_helion_compilation = functools.partial(
    assert_helion_compilation, has_compilation=False
)


class RefEagerTestBase:
    """Base class for all ref eager mode tests with shared setup/teardown."""

    # Class-level tracking for assert_close counting
    _assert_close_count = 0
    _original_assert_close_func = None
    # Class-level tracking for assertRaises counting
    _assert_raises_count = 0
    _original_assert_raises_func = None
    # Class-level tracking for skipTest counting
    _skip_test_count = 0
    _original_skip_test_func = None

    def setUp(self):
        """Common setup for all ref eager tests."""
        super().setUp()

        # Set HELION_REF_EAGER environment variable
        self._old_env_value = os.environ.get("HELION_REF_EAGER")
        os.environ["HELION_REF_EAGER"] = "1"

        # Reset assert_close counter for this test
        RefEagerTestBase._assert_close_count = 0
        # Reset assertRaises counter for this test
        RefEagerTestBase._assert_raises_count = 0
        # Reset skipTest counter for this test
        RefEagerTestBase._skip_test_count = 0

        # Patch torch.testing.assert_close to count calls
        if RefEagerTestBase._original_assert_close_func is None:
            RefEagerTestBase._original_assert_close_func = torch.testing.assert_close

        def counting_assert_close(*args, **kwargs):
            RefEagerTestBase._assert_close_count += 1
            return RefEagerTestBase._original_assert_close_func(*args, **kwargs)

        torch.testing.assert_close = counting_assert_close

        # Patch self.assertRaises to count calls
        if RefEagerTestBase._original_assert_raises_func is None:
            RefEagerTestBase._original_assert_raises_func = self.assertRaises

        def counting_assert_raises(*args, **kwargs):
            RefEagerTestBase._assert_raises_count += 1
            return RefEagerTestBase._original_assert_raises_func(*args, **kwargs)

        self.assertRaises = counting_assert_raises

        # Patch self.skipTest to count calls
        if RefEagerTestBase._original_skip_test_func is None:
            RefEagerTestBase._original_skip_test_func = self.skipTest

        def counting_skip_test(*args, **kwargs):
            RefEagerTestBase._skip_test_count += 1
            return RefEagerTestBase._original_skip_test_func(*args, **kwargs)

        self.skipTest = counting_skip_test

    def tearDown(self):
        """Common teardown with assertion counting check."""
        try:
            # Check if the test was skipped
            test_method = getattr(self, self._testMethodName, None)
            is_skipped = (
                test_method is not None
                and hasattr(test_method, "__unittest_skip__")
                and test_method.__unittest_skip__
            )

            if not is_skipped:
                # Check that either assert_close, assertRaises, or skipTest was called
                total_assertions = (
                    RefEagerTestBase._assert_close_count
                    + RefEagerTestBase._assert_raises_count
                    + RefEagerTestBase._skip_test_count
                )
                self.assertGreater(
                    total_assertions,
                    0,
                    f"Test {self._testMethodName} did not call torch.testing.assert_close, assertRaises, or skipTest",
                )
        finally:
            # Restore the original assert_close function
            if RefEagerTestBase._original_assert_close_func is not None:
                torch.testing.assert_close = (
                    RefEagerTestBase._original_assert_close_func
                )

            # Restore the original assertRaises function
            if RefEagerTestBase._original_assert_raises_func is not None:
                self.assertRaises = RefEagerTestBase._original_assert_raises_func

            # Restore the original skipTest function
            if RefEagerTestBase._original_skip_test_func is not None:
                self.skipTest = RefEagerTestBase._original_skip_test_func

            # Restore HELION_REF_EAGER environment variable
            if self._old_env_value is None:
                os.environ.pop("HELION_REF_EAGER", None)
            else:
                os.environ["HELION_REF_EAGER"] = self._old_env_value

            super().tearDown()

    # NOTE: We no-op these methods because they commonly check behaviors that are not relevant in ref eager mode.
    # Instead, we solely rely on the unit test's `torch.testing.assert_close` and `assertRaises` checks to ensure ref eager mode's correctness.
    def assertExpectedJournal(self, value: str) -> None:
        pass

    def assertIn(self, member, container, msg=None):
        pass

    def assertTrue(self, expr, msg=None):
        pass

    def assertEqual(self, first, second, msg=None):
        pass

    def assertNotEqual(self, first, second, msg=None):
        pass

    def assertNotIn(self, member, container, msg=None):
        pass


class TestExamplesRefEager(RefEagerTestBase, test_examples.TestExamples):
    """Run all TestExamples tests in ref eager mode."""

    @unittest.skip("Test has skip_accuracy=True and doesn't call assert_close")
    def test_moe_matmul_ogs(self):
        pass

    def test_add(self):
        # Spot check to ensure that the Helion kernel in ref eager mode doesn't go through Helion compilation
        with assert_no_helion_compilation():
            super().test_add()

    @unittest.skip(
        "AssertionError: load must be decorated with @helion.ref() to be used in ref mode"
    )
    def test_concat(self):
        pass

    @unittest.skip(
        "AssertionError: load must be decorated with @helion.ref() to be used in ref mode"
    )
    def test_concat_block_ptr(self):
        pass

    @unittest.skip(
        "AssertionError: load must be decorated with @helion.ref() to be used in ref mode"
    )
    def test_cross_entropy(self):
        pass

    @unittest.skip(
        "AssertionError: load must be decorated with @helion.ref() to be used in ref mode"
    )
    def test_jagged_dense_add(self):
        pass

    @unittest.skip(
        "AssertionError: load must be decorated with @helion.ref() to be used in ref mode"
    )
    def test_jagged_mean(self):
        pass

    @unittest.skip(
        "AssertionError: register_reduction_dim must be decorated with @helion.ref() to be used in ref mode"
    )
    def test_matmul_layernorm_dynamic_shapes(self):
        pass

    @unittest.skip(
        "AssertionError: register_reduction_dim must be decorated with @helion.ref() to be used in ref mode"
    )
    def test_matmul_layernorm_static_shapes(self):
        pass

    @unittest.skip(
        "AssertionError: register_tunable must be decorated with @helion.ref() to be used in ref mode"
    )
    def test_matmul_split_k(self):
        pass

    @unittest.skip(
        "AssertionError: load must be decorated with @helion.ref() to be used in ref mode"
    )
    def test_segment_reduction(self):
        pass

    @unittest.skip(
        "AssertionError: register_block_size must be decorated with @helion.ref() to be used in ref mode"
    )
    def test_softmax_two_pass(self):
        pass

    @unittest.skip(
        "AssertionError: register_block_size must be decorated with @helion.ref() to be used in ref mode"
    )
    def test_softmax_two_pass_block_ptr(self):
        pass

    @unittest.skip(
        "RuntimeError: The size of tensor a (64) must match the size of tensor b (0) at non-singleton dimension 0"
    )
    def test_template_via_closure0(self):
        pass

    @unittest.skip(
        "RuntimeError: The size of tensor a (64) must match the size of tensor b (0) at non-singleton dimension 0"
    )
    def test_template_via_closure1(self):
        pass


class TestDotRefEager(RefEagerTestBase, test_dot.TestDot):
    """Run all TestDot tests in ref eager mode."""

    @unittest.skip("Matmul with float8_e5m2 dtype not supported in ref eager mode")
    def test_input_float8_e5m2_acc_None_dynamic_shape(self):
        pass

    @unittest.skip("Matmul with float8_e5m2 dtype not supported in ref eager mode")
    def test_input_float8_e5m2_acc_None_static_shape(self):
        pass

    @unittest.skip("Matmul with float8_e5m2 dtype not supported in ref eager mode")
    def test_input_float8_e5m2_acc_float16_dynamic_shape(self):
        pass

    @unittest.skip("Matmul with float8_e5m2 dtype not supported in ref eager mode")
    def test_input_float8_e5m2_acc_float16_static_shape(self):
        pass

    @unittest.skip("Matmul with float8_e5m2 dtype not supported in ref eager mode")
    def test_input_float8_e5m2_acc_float32_dynamic_shape(self):
        pass

    @unittest.skip("Matmul with float8_e5m2 dtype not supported in ref eager mode")
    def test_input_float8_e5m2_acc_float32_static_shape(self):
        pass

    @unittest.skip("int8 @ int8 -> int32 is not supported in ref eager mode")
    def test_input_int8_acc_None_dynamic_shape(self):
        pass

    @unittest.skip("int8 @ int8 -> int32 is not supported in ref eager mode")
    def test_input_int8_acc_None_static_shape(self):
        pass

    @unittest.skip("int8 @ int8 -> int32 is not supported in ref eager mode")
    def test_input_int8_acc_int32_dynamic_shape(self):
        pass

    @unittest.skip("int8 @ int8 -> int32 is not supported in ref eager mode")
    def test_input_int8_acc_int32_static_shape(self):
        pass


class TestGridRefEager(RefEagerTestBase, test_grid.TestGrid):
    """Run all TestGrid tests in ref eager mode."""


class TestLoopsRefEager(RefEagerTestBase, test_loops.TestLoops):
    """Run all TestLoops tests in ref eager mode."""

    @unittest.skip(
        "Static range test checks code generation, not relevant in ref eager mode"
    )
    def test_static_range_2d(self):
        pass

    @unittest.skip(
        "Static range test checks code generation, not relevant in ref eager mode"
    )
    def test_static_range_scalar(self):
        pass

    @unittest.skip(
        "AssertionError: register_block_size must be decorated with @helion.ref() to be used in ref mode"
    )
    def test_data_dependent_bounds1(self):
        pass

    @unittest.skip(
        "AssertionError: register_block_size must be decorated with @helion.ref() to be used in ref mode"
    )
    def test_data_dependent_bounds4(self):
        pass

    @unittest.skip(
        "AssertionError: register_block_size must be decorated with @helion.ref() to be used in ref mode"
    )
    def test_l2_grouping_with_register_block_size(self):
        pass

    @unittest.skip(
        "AssertionError: register_block_size must be decorated with @helion.ref() to be used in ref mode"
    )
    def test_register_block_size_minimum(self):
        pass

    @unittest.skip(
        "AssertionError: register_block_size must be decorated with @helion.ref() to be used in ref mode"
    )
    def test_reorder_with_register_block_size(self):
        pass

    def test_3d_device_loop0(self):
        super().test_3d_device_loop0(dim_size=8)

    def test_3d_device_loop1(self):
        super().test_3d_device_loop1(dim_size=8)

    def test_3d_device_loop2(self):
        super().test_3d_device_loop2(dim_size=8)

    def test_3d_device_loop3(self):
        super().test_3d_device_loop3(dim_size=8)


class TestRefEagerMisc(TestCase):
    def test_print_intermediate_tensor(self):
        @helion.kernel(ref_mode=helion.RefMode.EAGER)
        def print_intermediate_tensor_kernel(
            x: torch.Tensor, y: torch.Tensor
        ) -> torch.Tensor:
            out = torch.empty_like(x)
            m, n = x.shape
            for tile_m, tile_n in hl.tile([m, n]):
                x_val = x[tile_m, tile_n]
                y_val = y[tile_m, tile_n]
                sum_val = x_val + y_val
                print("x: ", x_val)
                print("y: ", y_val)
                print("sum: ", sum_val)
                out[tile_m, tile_n] = sum_val
            return out

        x = torch.ones([2, 2], device="cuda", dtype=torch.float32) * 10.0
        y = torch.ones([2, 2], device="cuda", dtype=torch.float32) * 5.0
        expected = x + y

        # Capture stdout to check print output
        captured_output = io.StringIO()
        with contextlib.redirect_stdout(captured_output):
            result = print_intermediate_tensor_kernel(x, y)

        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)

        # Check that the print statements produced output
        output = captured_output.getvalue()
        self.assertIn("x: ", output)
        self.assertIn("y: ", output)
        self.assertIn("sum: ", output)
        self.assertIn("[[10., 10.]", output)  # x values
        self.assertIn("[[5., 5.]", output)  # y values
        self.assertIn("[[15., 15.]", output)  # sum values

    def test_print_in_invalid_helion_kernel(self):
        """Test that print works even in invalid Helion kernels in ref eager mode."""

        @helion.kernel(ref_mode=helion.RefMode.EAGER)
        def incorrect_kernel(x: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            for tile_m, tile_n in hl.tile([m, n]):
                val = x[tile_m, tile_n]
                print("processing tile: ", val)
                # `pass` below causes this kernel to be invalid.
                # But we show that in ref-eager mode, the `print` statement above still works,
                # which is useful for debugging.
                pass  # noqa: PIE790
            return x

        x = torch.ones([2, 2], device="cuda", dtype=torch.float32) * math.pi

        # Capture stdout to check print output
        captured_output = io.StringIO()
        with contextlib.redirect_stdout(captured_output):
            _ = incorrect_kernel(x)

        # Check that the print statement produced output
        output = captured_output.getvalue()
        self.assertIn("processing tile: ", output)
        self.assertIn("[[3.14", output)  # The value printed

    def test_ref_eager_kernel_config(self):
        @helion.kernel(ref_mode=helion.RefMode.EAGER)
        def kernel(x: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            out = torch.empty_like(x)
            for tile_m, tile_n in hl.tile([m, n]):
                out[tile_m, tile_n] = x[tile_m, tile_n] * 2.0
            return out

        with assert_no_helion_compilation():
            x = torch.randn(128, 128, device="cuda")
            result = kernel(x)
            expected = x * 2.0
            torch.testing.assert_close(result, expected)


class TestRefModeNoContamination(TestCase):
    """Test that ref mode and normal mode does not contaminate each other."""

    def setUp(self):
        """Clear state before each test."""
        super().setUp()
        os.environ.pop("HELION_REF_EAGER", None)

    def tearDown(self):
        """Restore original state after each test."""
        os.environ.pop("HELION_REF_EAGER", None)
        super().tearDown()

    def test_normal_helion_compilation(self):
        # Create kernel in normal mode
        @helion.kernel
        def normal_kernel(x: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            out = torch.empty_like(x)
            for tile_m, tile_n in hl.tile([m, n]):
                out[tile_m, tile_n] = x[tile_m, tile_n] * 2.0
            return out

        # Execute and verify compilation happens
        x = torch.randn(64, 64, device="cuda")
        with assert_helion_compilation():
            result = normal_kernel(x)

        torch.testing.assert_close(result, x * 2.0)

    def test_ref_eager_mode_no_helion_compilation(self):
        # Set ref eager mode
        os.environ["HELION_REF_EAGER"] = "1"

        # Create kernel in ref eager mode
        @helion.kernel
        def ref_eager_kernel(x: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            out = torch.empty_like(x)
            for tile_m, tile_n in hl.tile([m, n]):
                out[tile_m, tile_n] = x[tile_m, tile_n] * 3.0
            return out

        # Execute without compilation
        x = torch.randn(64, 64, device="cuda")
        with assert_no_helion_compilation():
            result = ref_eager_kernel(x)

        torch.testing.assert_close(result, x * 3.0)

    def test_existing_kernel_respects_runtime_ref_eager_env_var_0(self):
        # Create kernel in normal mode
        @helion.kernel
        def shared_kernel(x: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            out = torch.empty_like(x)
            for tile_m, tile_n in hl.tile([m, n]):
                out[tile_m, tile_n] = x[tile_m, tile_n] * 5.0
            return out

        x = torch.randn(64, 64, device="cuda")

        # First execution in normal mode (should compile)
        with assert_helion_compilation():
            result1 = shared_kernel(x)
        torch.testing.assert_close(result1, x * 5.0)

        # Set ref eager mode
        os.environ["HELION_REF_EAGER"] = "1"

        # Same kernel should now run without compilation
        with assert_no_helion_compilation():
            result2 = shared_kernel(x)
        torch.testing.assert_close(result2, x * 5.0)

    def test_existing_kernel_respects_runtime_ref_eager_env_var_1(self):
        # Set ref eager mode and create kernel
        os.environ["HELION_REF_EAGER"] = "1"

        @helion.kernel
        def ref_eager_kernel(x: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            out = torch.empty_like(x)
            for tile_m, tile_n in hl.tile([m, n]):
                out[tile_m, tile_n] = x[tile_m, tile_n] * 4.0
            return out

        # Unset ref eager mode
        del os.environ["HELION_REF_EAGER"]

        # Kernel should run with compilation
        x = torch.randn(64, 64, device="cuda")
        with assert_helion_compilation():
            result = ref_eager_kernel(x)
        torch.testing.assert_close(result, x * 4.0)


if __name__ == "__main__":
    unittest.main()
