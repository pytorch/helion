from __future__ import annotations

import unittest

import torch

import helion
import helion.language as hl

DEVICE = "mps"

requires_mps = unittest.skipUnless(
    hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
    "requires MPS device",
)

_DEFAULT_CONFIG = [helion.Config(block_sizes=[256], num_warps=4)]


# ---------------------------------------------------------------------------
# Kernel definitions
# ---------------------------------------------------------------------------


@helion.kernel(backend="metal", configs=_DEFAULT_CONFIG)
def vector_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] + y[tile]
    return out


@helion.kernel(backend="metal", configs=_DEFAULT_CONFIG)
def vector_sub(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] - y[tile]
    return out


@helion.kernel(backend="metal", configs=_DEFAULT_CONFIG)
def vector_mul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] * y[tile]
    return out


@helion.kernel(backend="metal", configs=_DEFAULT_CONFIG)
def vector_div(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] / y[tile]
    return out


@helion.kernel(backend="metal", configs=_DEFAULT_CONFIG)
def vector_neg(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = -x[tile]
    return out


@helion.kernel(backend="metal", configs=_DEFAULT_CONFIG)
def saxpy(x: torch.Tensor, y: torch.Tensor, a: float, b: float) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = a * x[tile] + b * y[tile]
    return out


@helion.kernel(backend="metal", configs=_DEFAULT_CONFIG)
def relu(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = torch.where(x[tile] > 0, x[tile], 0.0)
    return out


@helion.kernel(backend="metal", configs=_DEFAULT_CONFIG)
def silu(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] * torch.sigmoid(x[tile])
    return out


@helion.kernel(backend="metal", configs=_DEFAULT_CONFIG)
def gelu_approx(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = (
            0.5
            * x[tile]
            * (1.0 + torch.tanh(0.7978845608 * (x[tile] + 0.044715 * x[tile] ** 3)))
        )
    return out


@helion.kernel(backend="metal", configs=_DEFAULT_CONFIG)
def exp_kernel(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = torch.exp(x[tile])
    return out


@helion.kernel(backend="metal", configs=_DEFAULT_CONFIG)
def log_kernel(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = torch.log(x[tile])
    return out


@helion.kernel(backend="metal", configs=_DEFAULT_CONFIG)
def sqrt_kernel(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = torch.sqrt(x[tile])
    return out


@helion.kernel(backend="metal", configs=_DEFAULT_CONFIG)
def abs_kernel(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = torch.abs(x[tile])
    return out


@helion.kernel(backend="metal", configs=_DEFAULT_CONFIG)
def sin_cos_add(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = torch.sin(x[tile]) + torch.cos(x[tile])
    return out


@helion.kernel(backend="metal", configs=_DEFAULT_CONFIG)
def clamp_kernel(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = torch.clamp(x[tile], -0.5, 0.5)
    return out


@helion.kernel(backend="metal", configs=_DEFAULT_CONFIG)
def add_into(x: torch.Tensor, y: torch.Tensor, out: torch.Tensor) -> None:
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] + y[tile]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

N = 1024


class TestMetalElementwise(unittest.TestCase):
    """Basic elementwise ops."""

    @requires_mps
    def test_vector_add(self) -> None:
        x = torch.randn(N, device=DEVICE)
        y = torch.randn(N, device=DEVICE)
        torch.testing.assert_close(vector_add(x, y), x + y)

    @requires_mps
    def test_vector_add_large(self) -> None:
        x = torch.randn(100000, device=DEVICE)
        y = torch.randn(100000, device=DEVICE)
        torch.testing.assert_close(vector_add(x, y), x + y)

    @requires_mps
    def test_vector_add_non_power_of_2(self) -> None:
        x = torch.randn(999, device=DEVICE)
        y = torch.randn(999, device=DEVICE)
        torch.testing.assert_close(vector_add(x, y), x + y)

    @requires_mps
    def test_vector_sub(self) -> None:
        x = torch.randn(N, device=DEVICE)
        y = torch.randn(N, device=DEVICE)
        torch.testing.assert_close(vector_sub(x, y), x - y)

    @requires_mps
    def test_vector_mul(self) -> None:
        x = torch.randn(N, device=DEVICE)
        y = torch.randn(N, device=DEVICE)
        torch.testing.assert_close(vector_mul(x, y), x * y)

    @requires_mps
    def test_vector_div(self) -> None:
        x = torch.randn(N, device=DEVICE)
        y = torch.randn(N, device=DEVICE).abs() + 0.1
        torch.testing.assert_close(vector_div(x, y), x / y)

    @requires_mps
    def test_vector_neg(self) -> None:
        x = torch.randn(N, device=DEVICE)
        torch.testing.assert_close(vector_neg(x), -x)


class TestMetalScalarArgs(unittest.TestCase):
    """Kernels with scalar (float) arguments passed as buffer params."""

    @requires_mps
    def test_saxpy(self) -> None:
        x = torch.randn(N, device=DEVICE)
        y = torch.randn(N, device=DEVICE)
        a, b = 2.5, 0.3
        torch.testing.assert_close(saxpy(x, y, a, b), a * x + b * y)


class TestMetalActivations(unittest.TestCase):
    """Activation functions exercising where/sigmoid/tanh/exp."""

    @requires_mps
    def test_relu(self) -> None:
        x = torch.randn(N, device=DEVICE)
        torch.testing.assert_close(relu(x), torch.relu(x))

    @requires_mps
    def test_silu(self) -> None:
        x = torch.randn(N, device=DEVICE)
        torch.testing.assert_close(silu(x), x * torch.sigmoid(x), atol=1e-5, rtol=1e-5)

    @requires_mps
    def test_gelu_approx(self) -> None:
        x = torch.randn(N, device=DEVICE)
        expected = 0.5 * x * (1.0 + torch.tanh(0.7978845608 * (x + 0.044715 * x**3)))
        torch.testing.assert_close(gelu_approx(x), expected, atol=1e-5, rtol=1e-5)


class TestMetalMathOps(unittest.TestCase):
    """Unary math ops: exp, log, sqrt, abs, sin, cos, clamp."""

    @requires_mps
    def test_exp(self) -> None:
        x = torch.randn(N, device=DEVICE)
        torch.testing.assert_close(exp_kernel(x), torch.exp(x), atol=1e-5, rtol=1e-5)

    @requires_mps
    def test_log(self) -> None:
        x = torch.rand(N, device=DEVICE) + 0.01
        torch.testing.assert_close(log_kernel(x), torch.log(x), atol=1e-5, rtol=1e-5)

    @requires_mps
    def test_sqrt(self) -> None:
        x = torch.rand(N, device=DEVICE) + 0.01
        torch.testing.assert_close(sqrt_kernel(x), torch.sqrt(x), atol=1e-5, rtol=1e-5)

    @requires_mps
    def test_abs(self) -> None:
        x = torch.randn(N, device=DEVICE)
        torch.testing.assert_close(abs_kernel(x), torch.abs(x))

    @requires_mps
    def test_sin_cos(self) -> None:
        x = torch.randn(N, device=DEVICE)
        torch.testing.assert_close(
            sin_cos_add(x), torch.sin(x) + torch.cos(x), atol=1e-5, rtol=1e-5
        )

    @requires_mps
    def test_clamp(self) -> None:
        x = torch.randn(N, device=DEVICE)
        torch.testing.assert_close(clamp_kernel(x), torch.clamp(x, -0.5, 0.5))


class TestMetalDtypes(unittest.TestCase):
    """Test elementwise ops across dtypes."""

    @requires_mps
    def test_vector_add_float16(self) -> None:
        x = torch.randn(N, device=DEVICE, dtype=torch.float16)
        y = torch.randn(N, device=DEVICE, dtype=torch.float16)
        torch.testing.assert_close(vector_add(x, y), x + y)

    @requires_mps
    def test_vector_add_bfloat16(self) -> None:
        x = torch.randn(N, device=DEVICE, dtype=torch.bfloat16)
        y = torch.randn(N, device=DEVICE, dtype=torch.bfloat16)
        torch.testing.assert_close(vector_add(x, y), x + y)

    @requires_mps
    def test_vector_add_int32(self) -> None:
        x = torch.randint(-100, 100, (N,), device=DEVICE, dtype=torch.int32)
        y = torch.randint(-100, 100, (N,), device=DEVICE, dtype=torch.int32)
        torch.testing.assert_close(vector_add(x, y), x + y)

    @requires_mps
    def test_neg_float16(self) -> None:
        x = torch.randn(N, device=DEVICE, dtype=torch.float16)
        torch.testing.assert_close(vector_neg(x), -x)

    @requires_mps
    def test_mul_bfloat16(self) -> None:
        x = torch.randn(N, device=DEVICE, dtype=torch.bfloat16)
        y = torch.randn(N, device=DEVICE, dtype=torch.bfloat16)
        torch.testing.assert_close(vector_mul(x, y), x * y)


class TestMetalBoundsMasking(unittest.TestCase):
    """Verify bounds masking prevents OOB accesses for non-aligned sizes."""

    @requires_mps
    def test_store_no_oob_write(self) -> None:
        """Use a sentinel buffer to detect OOB store writes.

        Allocates a contiguous buffer [N valid | PAD sentinel]. Passes the
        first N elements as the output view. If OOB threads write past
        element N-1, they corrupt the sentinel region.
        """
        n = 999
        pad = 256
        sentinel = 42.0
        buf = torch.full((n + pad,), sentinel, device=DEVICE)
        x = torch.randn(n, device=DEVICE)
        y = torch.randn(n, device=DEVICE)
        out = buf[:n]
        add_into(x, y, out)
        torch.mps.synchronize()
        torch.testing.assert_close(buf[:n], x + y)
        self.assertTrue(
            (buf[n:] == sentinel).all(),
            "OOB store detected: sentinel region was modified by padding threads",
        )

    @requires_mps
    def test_store_no_oob_write_size_1(self) -> None:
        """Extreme case: N=1 with block_size=256 → 255 OOB threads."""
        n = 1
        pad = 256
        sentinel = 42.0
        buf = torch.full((n + pad,), sentinel, device=DEVICE)
        x = torch.randn(n, device=DEVICE)
        y = torch.randn(n, device=DEVICE)
        out = buf[:n]
        add_into(x, y, out)
        torch.mps.synchronize()
        torch.testing.assert_close(buf[:n], x + y)
        self.assertTrue(
            (buf[n:] == sentinel).all(),
            "OOB store detected: sentinel region was modified by padding threads",
        )

    @requires_mps
    def test_codegen_has_mask_for_non_aligned(self) -> None:
        """Generated MSL must contain bounds checks for non-aligned sizes."""
        x = torch.randn(999, device=DEVICE)
        y = torch.randn(999, device=DEVICE)
        code = vector_add.bind((x, y)).to_triton_code()
        self.assertIn("mask_0", code, "mask variable not found in generated code")
        self.assertIn("if (mask_0)", code, "store guard not found in generated MSL")
        self.assertIn("?", code, "load ternary not found in generated MSL")

    @requires_mps
    def test_codegen_always_has_mask(self) -> None:
        """Metal always generates masks (force_tile_mask=True) because the
        launcher's threadgroup size can differ from the tile block_size."""
        x = torch.randn(1024, device=DEVICE)
        y = torch.randn(1024, device=DEVICE)
        code = vector_add.bind((x, y)).to_triton_code()
        self.assertIn("mask_0", code, "mask variable expected even for aligned size")


if __name__ == "__main__":
    unittest.main()
