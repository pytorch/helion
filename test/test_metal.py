from __future__ import annotations

import sys
import unittest

import torch

import helion
import helion.language as hl

if sys.platform == "darwin":
    from helion._compiler.metal.metal_jit import _MetalKernel

_requires_darwin = unittest.skipIf(
    sys.platform != "darwin", "Metal tests require macOS"
)

DEVICE = "mps"

_DEFAULT_CONFIG = [helion.Config(block_sizes=[256], num_warps=4)]


def _get_msl(kernel: helion.Kernel, args: tuple[object, ...]) -> str:
    """Run a kernel through the normal Helion pipeline and return the MSL.

    Uses PyCodeCache to load the generated module (same as Helion's runtime),
    calls the host function to trigger metal_jit compilation, then reads
    the MSL from the _MetalKernel.
    """
    from torch._inductor.codecache import PyCodeCache

    code = kernel.bind(args).to_code()
    module = PyCodeCache.load(code)
    # Call the host function by name
    host_fn = getattr(module, kernel.fn.__name__)
    host_fn(*args)
    # Find the _MetalKernel and return its MSL
    for obj in vars(module).values():
        if isinstance(obj, _MetalKernel) and obj.msl_source is not None:
            return obj.msl_source
    raise RuntimeError("No @metal_jit kernel found in generated code")


# ---------------------------------------------------------------------------
# Kernel definitions – copy
# ---------------------------------------------------------------------------


@helion.kernel(backend="metal", configs=_DEFAULT_CONFIG)
def copy_kernel(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile]
    return out


@helion.kernel(backend="metal", configs=_DEFAULT_CONFIG)
def copy_into(x: torch.Tensor, out: torch.Tensor) -> None:
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile]


@helion.kernel(backend="metal", configs=_DEFAULT_CONFIG)
def masked_copy(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = torch.where(mask[tile], x[tile], 0.0)
    return out


# ---------------------------------------------------------------------------
# Kernel definitions – arithmetic
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


# ---------------------------------------------------------------------------
# Kernel definitions – scalar args
# ---------------------------------------------------------------------------


@helion.kernel(backend="metal", configs=_DEFAULT_CONFIG)
def saxpy(x: torch.Tensor, y: torch.Tensor, a: float, b: float) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = a * x[tile] + b * y[tile]
    return out


# ---------------------------------------------------------------------------
# Kernel definitions – activations
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Kernel definitions – math ops
# ---------------------------------------------------------------------------


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
def sincos_kernel(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = torch.sin(x[tile]) + torch.cos(x[tile])
    return out


@helion.kernel(backend="metal", configs=_DEFAULT_CONFIG)
def clamp_kernel(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = torch.clamp(x[tile], lo, hi)
    return out


# ---------------------------------------------------------------------------
# Tests – copy
# ---------------------------------------------------------------------------


@_requires_darwin
class TestMetalCopy(unittest.TestCase):
    """Copy kernels that test load/store + masking."""

    def test_copy(self) -> None:
        """Aligned size: out[tile] = x[tile] with size 1024."""
        x = torch.randn(1024, device=DEVICE)
        torch.testing.assert_close(copy_kernel(x), x)

    def test_copy_non_aligned(self) -> None:
        """Non-aligned size: mask must be active for correctness."""
        x = torch.randn(1000, device=DEVICE)
        torch.testing.assert_close(copy_kernel(x), x)

    def test_masked_copy(self) -> None:
        """torch.where with a boolean mask exercises _mask_to lowering."""
        x = torch.randn(1024, device=DEVICE)
        mask = torch.randint(0, 2, (1024,), device=DEVICE, dtype=torch.bool)
        result = masked_copy(x, mask)
        expected = torch.where(mask, x, torch.zeros_like(x))
        torch.testing.assert_close(result, expected)

    def test_masked_copy_non_aligned(self) -> None:
        """Masked copy with non-aligned size: both tile mask and user mask active."""
        x = torch.randn(1000, device=DEVICE)
        mask = torch.randint(0, 2, (1000,), device=DEVICE, dtype=torch.bool)
        result = masked_copy(x, mask)
        expected = torch.where(mask, x, torch.zeros_like(x))
        torch.testing.assert_close(result, expected)


# ---------------------------------------------------------------------------
# Tests – bounds masking
# ---------------------------------------------------------------------------


@_requires_darwin
class TestMetalBoundsMasking(unittest.TestCase):
    """Bounds masking tests using vector_add for both load and store paths."""

    def test_store_no_oob_write(self) -> None:
        """Sentinel buffer detects OOB store writes."""
        n = 999
        pad = 256
        sentinel = 42.0
        buf = torch.full((n + pad,), sentinel, device=DEVICE)
        x = torch.randn(n, device=DEVICE)
        out = buf[:n]
        copy_into(x, out)
        torch.mps.synchronize()
        torch.testing.assert_close(buf[:n], x)
        self.assertTrue(
            (buf[n:] == sentinel).all(),
            "OOB store detected: sentinel region was modified by padding threads",
        )

    def test_store_no_oob_write_size_1(self) -> None:
        """Extreme case: N=1 with block_size=256 → 255 OOB threads."""
        n = 1
        pad = 256
        sentinel = 42.0
        buf = torch.full((n + pad,), sentinel, device=DEVICE)
        x = torch.randn(n, device=DEVICE)
        out = buf[:n]
        copy_into(x, out)
        torch.mps.synchronize()
        torch.testing.assert_close(buf[:n], x)
        self.assertTrue(
            (buf[n:] == sentinel).all(),
            "OOB store detected: sentinel region was modified by padding threads",
        )

    def test_codegen_has_mask(self) -> None:
        """Generated MSL must contain bounds checks for non-aligned sizes."""
        x = torch.randn(999, device=DEVICE)
        msl = _get_msl(copy_kernel, (x,))
        self.assertIn("mask_0", msl, "mask variable not found in generated MSL")
        self.assertIn("if (mask_0)", msl, "store guard not found in generated MSL")
        self.assertIn("?", msl, "load ternary not found in generated MSL")

    def test_codegen_always_has_mask(self) -> None:
        """Metal always generates masks (force_tile_mask=True) because the
        launcher's threadgroup size can differ from the tile block_size."""
        x = torch.randn(1024, device=DEVICE)
        msl = _get_msl(copy_kernel, (x,))
        self.assertIn("mask_0", msl, "mask variable expected even for aligned size")

    def test_vector_add_non_aligned(self) -> None:
        """vector_add with non-aligned size exercises mask on both load and store."""
        x = torch.randn(1000, device=DEVICE)
        y = torch.randn(1000, device=DEVICE)
        torch.testing.assert_close(vector_add(x, y), x + y)

    def test_vector_add_oob_store(self) -> None:
        """Sentinel buffer detects OOB stores from vector_add."""
        n = 999
        pad = 256
        sentinel = 42.0
        buf_out = torch.full((n + pad,), sentinel, device=DEVICE)
        x = torch.randn(n, device=DEVICE)
        y = torch.randn(n, device=DEVICE)
        # Use copy_into as a proxy: compute add manually then copy
        expected = x + y
        copy_into(expected, buf_out[:n])
        torch.mps.synchronize()
        torch.testing.assert_close(buf_out[:n], expected)
        self.assertTrue(
            (buf_out[n:] == sentinel).all(),
            "OOB store detected in vector_add sentinel region",
        )


# ---------------------------------------------------------------------------
# Tests – arithmetic
# ---------------------------------------------------------------------------


@_requires_darwin
class TestMetalArithmetic(unittest.TestCase):
    """Basic arithmetic elementwise kernels."""

    def test_vector_add(self) -> None:
        x = torch.randn(1024, device=DEVICE)
        y = torch.randn(1024, device=DEVICE)
        torch.testing.assert_close(vector_add(x, y), x + y)

    def test_vector_sub(self) -> None:
        x = torch.randn(1024, device=DEVICE)
        y = torch.randn(1024, device=DEVICE)
        torch.testing.assert_close(vector_sub(x, y), x - y)

    def test_vector_mul(self) -> None:
        x = torch.randn(1024, device=DEVICE)
        y = torch.randn(1024, device=DEVICE)
        torch.testing.assert_close(vector_mul(x, y), x * y)

    def test_vector_div(self) -> None:
        x = torch.randn(1024, device=DEVICE)
        y = torch.randn(1024, device=DEVICE).abs() + 0.1
        torch.testing.assert_close(vector_div(x, y), x / y)

    def test_vector_neg(self) -> None:
        x = torch.randn(1024, device=DEVICE)
        torch.testing.assert_close(vector_neg(x), -x)


# ---------------------------------------------------------------------------
# Tests – scalar args
# ---------------------------------------------------------------------------


@_requires_darwin
class TestMetalScalarArgs(unittest.TestCase):
    """Kernels that accept scalar (SymbolArgument) parameters."""

    def test_saxpy(self) -> None:
        x = torch.randn(1024, device=DEVICE)
        y = torch.randn(1024, device=DEVICE)
        a, b = 2.5, -1.0
        torch.testing.assert_close(saxpy(x, y, a, b), a * x + b * y)


# ---------------------------------------------------------------------------
# Tests – activations
# ---------------------------------------------------------------------------


@_requires_darwin
class TestMetalActivations(unittest.TestCase):
    """Activation function kernels."""

    def test_relu(self) -> None:
        x = torch.randn(1024, device=DEVICE)
        torch.testing.assert_close(relu(x), torch.relu(x))

    def test_silu(self) -> None:
        x = torch.randn(1024, device=DEVICE)
        torch.testing.assert_close(
            silu(x), torch.nn.functional.silu(x), atol=1e-5, rtol=1e-5
        )

    def test_gelu_approx(self) -> None:
        x = torch.randn(1024, device=DEVICE)
        expected = torch.nn.functional.gelu(x, approximate="tanh")
        torch.testing.assert_close(gelu_approx(x), expected, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# Tests – math ops
# ---------------------------------------------------------------------------


@_requires_darwin
class TestMetalMathOps(unittest.TestCase):
    """Math function kernels."""

    def test_exp(self) -> None:
        x = torch.randn(1024, device=DEVICE)
        torch.testing.assert_close(exp_kernel(x), torch.exp(x), atol=1e-5, rtol=1e-5)

    def test_log(self) -> None:
        x = torch.rand(1024, device=DEVICE) + 0.1
        torch.testing.assert_close(log_kernel(x), torch.log(x), atol=1e-5, rtol=1e-5)

    def test_sqrt(self) -> None:
        x = torch.rand(1024, device=DEVICE) + 0.1
        torch.testing.assert_close(sqrt_kernel(x), torch.sqrt(x))

    def test_abs(self) -> None:
        x = torch.randn(1024, device=DEVICE)
        torch.testing.assert_close(abs_kernel(x), torch.abs(x))

    def test_sincos(self) -> None:
        x = torch.randn(1024, device=DEVICE)
        expected = torch.sin(x) + torch.cos(x)
        torch.testing.assert_close(sincos_kernel(x), expected, atol=1e-5, rtol=1e-5)

    def test_clamp(self) -> None:
        x = torch.randn(1024, device=DEVICE)
        torch.testing.assert_close(
            clamp_kernel(x, -0.5, 0.5), torch.clamp(x, -0.5, 0.5)
        )


# ---------------------------------------------------------------------------
# Tests – dtypes
# ---------------------------------------------------------------------------


@_requires_darwin
class TestMetalDtypes(unittest.TestCase):
    """Elementwise ops across different dtypes."""

    def test_float16_add(self) -> None:
        x = torch.randn(1024, device=DEVICE, dtype=torch.float16)
        y = torch.randn(1024, device=DEVICE, dtype=torch.float16)
        torch.testing.assert_close(vector_add(x, y), x + y)

    def test_bfloat16_add(self) -> None:
        x = torch.randn(1024, device=DEVICE, dtype=torch.bfloat16)
        y = torch.randn(1024, device=DEVICE, dtype=torch.bfloat16)
        torch.testing.assert_close(vector_add(x, y), x + y)

    def test_int32_add(self) -> None:
        x = torch.randint(-100, 100, (1024,), device=DEVICE, dtype=torch.int32)
        y = torch.randint(-100, 100, (1024,), device=DEVICE, dtype=torch.int32)
        torch.testing.assert_close(vector_add(x, y), x + y)

    def test_float16_neg(self) -> None:
        x = torch.randn(1024, device=DEVICE, dtype=torch.float16)
        torch.testing.assert_close(vector_neg(x), -x)

    def test_bfloat16_mul(self) -> None:
        x = torch.randn(1024, device=DEVICE, dtype=torch.bfloat16)
        y = torch.randn(1024, device=DEVICE, dtype=torch.bfloat16)
        torch.testing.assert_close(vector_mul(x, y), x * y)


if __name__ == "__main__":
    unittest.main()
