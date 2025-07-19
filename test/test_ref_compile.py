from __future__ import annotations

import os
import unittest

import pytest
import torch

from . import test_examples
import helion
from helion._testing import TestCase


class TestExamplesRefCompile(test_examples.TestExamples):
    """Run all TestExamples tests in reference torch.compile mode via HELION_REF_COMPILE=1."""

    # NOTE: All tests in TestExamples are run in ref torch.compile(fullgraph=True) mode by default in this test file.

    def setUp(self):
        """Set up test environment."""
        super().setUp()
        # Save original env var
        self._original_env_var_value = os.environ.get("HELION_REF_COMPILE")
        # Set ref compile mode
        os.environ["HELION_REF_COMPILE"] = "1"

    def tearDown(self):
        """Restore original environment."""
        super().tearDown()
        # Restore original env var
        if self._original_env_var_value is not None:
            os.environ["HELION_REF_COMPILE"] = self._original_env_var_value
        elif "HELION_REF_COMPILE" in os.environ:
            del os.environ["HELION_REF_COMPILE"]

    def test_add(self):
        """Override test_add to verify ref compile mode execution."""
        from torch._dynamo.utils import counters

        # Clear counters before running the test
        counters.clear()

        # Run the original test
        super().test_add()

        # In ref compile mode, torch.compile SHOULD be called
        # Check for torch.compile-related counters
        frames_total = counters.get("frames", {}).get("total", 0)
        aot_total = counters.get("aot_autograd", {}).get("total", 0)

        self.assertGreater(
            frames_total,
            0,
            f"torch.compile should be invoked in ref compile mode. frames.total={frames_total}",
        )
        self.assertGreater(
            aot_total,
            0,
            f"AOT autograd should be invoked in ref compile mode. aot_autograd.total={aot_total}",
        )
        stats_captured = counters.get("stats", {}).get("calls_captured", 0)
        self.assertGreater(
            stats_captured,
            0,
            f"Compilation should capture calls in ref compile mode. stats.calls_captured={stats_captured}",
        )

    @pytest.mark.skip(
        reason="torch.compile doesn't support data-dependent branching (if num_tokens != 0)"
    )
    def test_moe_matmul_ogs(self):
        super().test_moe_matmul_ogs()


class TestRefCompileKernelConfig(TestCase):
    """Test @helion.kernel(ref_compile=True) parameter functionality."""

    def setUp(self):
        """Set up test environment."""
        super().setUp()
        # Save original env var to ensure clean state
        self._original_env_var_value = os.environ.get("HELION_REF_COMPILE")
        # Remove env var to test parameter-only behavior
        if "HELION_REF_COMPILE" in os.environ:
            del os.environ["HELION_REF_COMPILE"]

    def tearDown(self):
        """Restore original environment."""
        super().tearDown()
        # Restore original env var
        if self._original_env_var_value is not None:
            os.environ["HELION_REF_COMPILE"] = self._original_env_var_value

    def test_ref_compile_kernel_config(self):
        """Test that ref_compile=True kernel config actually enables torch.compile."""
        from torch._dynamo.utils import counters

        counters.clear()

        @helion.kernel(ref_compile=True)
        def compile_test(x: torch.Tensor) -> torch.Tensor:
            return x + x * 2.0

        x = torch.randn(128, device="cuda")
        result = compile_test(x)
        expected = x + x * 2.0
        torch.testing.assert_close(result, expected)

        # In ref compile mode, torch.compile SHOULD be called
        frames_total = counters.get("frames", {}).get("total", 0)
        aot_total = counters.get("aot_autograd", {}).get("total", 0)
        self.assertGreater(
            frames_total,
            0,
            f"torch.compile should be invoked with ref_compile=True. frames.total={frames_total}",
        )
        self.assertGreater(
            aot_total,
            0,
            f"AOT autograd should be invoked with ref_compile=True. aot_autograd.total={aot_total}",
        )


if __name__ == "__main__":
    unittest.main()
