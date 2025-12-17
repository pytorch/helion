"""Tests for Helion's Pallas backend on TPU and GPU.

This test compiles and runs simple kernels using the Pallas backend.
Pallas supports both TPU (via JAX) and GPU (via Triton under the hood).
"""

from __future__ import annotations

import os
import unittest

import torch

# Set the backend to pallas before importing helion
os.environ["HELION_BACKEND"] = "pallas"

import helion
import helion.language as hl
from helion._testing import TestCase


def has_tpu() -> bool:
    """Check if TPU is available via JAX."""
    try:
        import jax

        devices = jax.devices("tpu")
        return len(devices) > 0
    except Exception:
        return False


def has_pallas_gpu() -> bool:
    """Check if GPU is available for Pallas via JAX."""
    try:
        import jax

        devices = jax.devices("gpu")
        return len(devices) > 0
    except Exception:
        return False


def has_pallas_device() -> bool:
    """Check if any Pallas-compatible device (TPU or GPU) is available."""
    return has_tpu() or has_pallas_gpu()


def get_pallas_device_type() -> str | None:
    """Get the type of available Pallas device.

    Returns:
        'tpu' if TPU is available (preferred)
        'gpu' if GPU is available
        None if no Pallas device is available
    """
    if has_tpu():
        return "tpu"
    if has_pallas_gpu():
        return "gpu"
    return None


# Skip all tests if no Pallas device is available
@unittest.skipUnless(has_pallas_device(), "No Pallas device (TPU or GPU) available")
class TestPallasDevice(TestCase):
    """Tests for the Pallas backend on TPU or GPU."""

    @classmethod
    def setUpClass(cls):
        """Set up test class with device info."""
        super().setUpClass()
        cls.device_type = get_pallas_device_type()
        print(f"\nRunning Pallas tests on: {cls.device_type}")

    def test_add_kernel(self):
        """Test a simple add kernel on Pallas device (TPU or GPU)."""

        @helion.kernel(
            config=helion.Config(block_sizes=[128]),
        )
        def add_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(out.size()):
                out[tile] = x[tile] + y[tile]
            return out

        # Use tensor size = block size (single block) for device compatibility
        # Both TPU and GPU have specific layout constraints for multi-block tiling
        x = torch.randn(128, device="cpu")
        y = torch.randn(128, device="cpu")

        # Get the generated code
        bound = add_kernel.bind((x, y))
        config = helion.Config(block_sizes=[128])
        code = bound.to_triton_code(config)

        # Verify generated code matches expected
        self.assertExpectedJournal(code)

        # Run the kernel
        result = add_kernel(x, y)

        # Verify correctness
        expected = x + y
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)

    def test_mul_kernel(self):
        """Test a multiplication kernel on Pallas device."""

        @helion.kernel(
            config=helion.Config(block_sizes=[128]),
        )
        def mul_kernel(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(out.size()):
                out[tile] = x[tile] * 2.0
            return out

        # Use tensor size = block size for device compatibility
        x = torch.randn(128, device="cpu")

        # Get the generated code
        bound = mul_kernel.bind((x,))
        config = helion.Config(block_sizes=[128])
        code = bound.to_triton_code(config)

        # Verify generated code matches expected
        self.assertExpectedJournal(code)

        # Run the kernel
        result = mul_kernel(x)

        # Verify correctness
        expected = x * 2.0
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)

    def test_sub_kernel(self):
        """Test a subtraction kernel on Pallas device."""

        @helion.kernel(
            config=helion.Config(block_sizes=[128]),
        )
        def sub_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(out.size()):
                out[tile] = x[tile] - y[tile]
            return out

        # Use tensor size = block size for device compatibility
        x = torch.randn(128, device="cpu")
        y = torch.randn(128, device="cpu")

        # Get the generated code
        bound = sub_kernel.bind((x, y))
        config = helion.Config(block_sizes=[128])
        code = bound.to_triton_code(config)

        # Verify generated code matches expected
        self.assertExpectedJournal(code)

        # Run the kernel
        result = sub_kernel(x, y)

        # Verify correctness
        expected = x - y
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)

    def test_fused_add_mul_kernel(self):
        """Test a fused add-multiply kernel on Pallas device."""

        @helion.kernel(
            config=helion.Config(block_sizes=[128]),
        )
        def fused_kernel(
            x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
        ) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(out.size()):
                out[tile] = (x[tile] + y[tile]) * z[tile]
            return out

        # Use tensor size = block size for device compatibility
        x = torch.randn(128, device="cpu")
        y = torch.randn(128, device="cpu")
        z = torch.randn(128, device="cpu")

        # Get the generated code
        bound = fused_kernel.bind((x, y, z))
        config = helion.Config(block_sizes=[128])
        code = bound.to_triton_code(config)

        # Verify generated code matches expected
        self.assertExpectedJournal(code)

        # Run the kernel
        result = fused_kernel(x, y, z)

        # Verify correctness
        expected = (x + y) * z
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)


class TestPallasCodeGeneration(TestCase):
    """Tests for Pallas code generation (no device required)."""

    def test_code_generation_patterns(self):
        """Test that generated code has correct Pallas patterns."""

        @helion.kernel(
            config=helion.Config(block_sizes=[16]),
        )
        def add_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(out.size()):
                out[tile] = x[tile] + y[tile]
            return out

        x = torch.randn(64, device="cpu")
        y = torch.randn(64, device="cpu")

        bound = add_kernel.bind((x, y))
        config = helion.Config(block_sizes=[16])
        code = bound.to_triton_code(config)

        # Verify generated code matches expected
        self.assertExpectedJournal(code)

    def test_no_triton_cdiv(self):
        """Test that grid computation doesn't use triton.cdiv."""

        @helion.kernel(
            config=helion.Config(block_sizes=[16]),
        )
        def add_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(out.size()):
                out[tile] = x[tile] + y[tile]
            return out

        x = torch.randn(100, device="cpu")  # Non-divisible size
        y = torch.randn(100, device="cpu")

        bound = add_kernel.bind((x, y))
        config = helion.Config(block_sizes=[16])
        code = bound.to_triton_code(config)

        # Verify generated code matches expected
        self.assertExpectedJournal(code)


if __name__ == "__main__":
    unittest.main()
