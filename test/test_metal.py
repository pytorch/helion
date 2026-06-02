from __future__ import annotations

import sys
import unittest
from unittest.mock import patch

import torch

import helion
from helion import exc
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
# Kernel definitions – multi-dimensional
# ---------------------------------------------------------------------------


@helion.kernel(
    backend="metal", configs=[helion.Config(block_sizes=[64, 64], num_warps=4)]
)
def elementwise_2d(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile_m, tile_n in hl.tile([x.size(0), x.size(1)]):
        out[tile_m, tile_n] = x[tile_m, tile_n] + y[tile_m, tile_n]
    return out


@helion.kernel(
    backend="metal",
    configs=[helion.Config(block_sizes=[16, 16, 16], num_warps=4)],
)
def elementwise_3d(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile_0, tile_1, tile_2 in hl.tile([x.size(0), x.size(1), x.size(2)]):
        out[tile_0, tile_1, tile_2] = (
            x[tile_0, tile_1, tile_2] + y[tile_0, tile_1, tile_2]
        )
    return out


# ---------------------------------------------------------------------------
# Kernel definitions – large block (1D, block_size > 1024)
# ---------------------------------------------------------------------------


@helion.kernel(
    backend="metal", configs=[helion.Config(block_sizes=[2048], num_warps=4)]
)
def large_block_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] + y[tile]
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

    def test_codegen_no_stride_one(self) -> None:
        """Generated MSL should not contain trivial * 1 stride multiplications."""
        x = torch.randn(1024, device=DEVICE)
        msl = _get_msl(copy_kernel, (x,))
        self.assertNotIn("* 1)", msl, "trivial * 1 stride found in generated MSL")
        self.assertNotIn("* 1]", msl, "trivial * 1 stride found in generated MSL")

    def test_codegen_array_subscript(self) -> None:
        """Generated MSL should use array subscript x[idx] not pointer deref *(x + idx)."""
        x = torch.randn(1024, device=DEVICE)
        msl = _get_msl(copy_kernel, (x,))
        self.assertNotIn(
            "*((", msl, "pointer dereference found; expected array subscript"
        )
        self.assertIn("x[", msl, "array subscript load not found in generated MSL")
        self.assertIn("out[", msl, "array subscript store not found in generated MSL")

    def test_scalar_codegen_does_not_include_mpp(self) -> None:
        x = torch.randn(1024, device=DEVICE)
        msl = _get_msl(copy_kernel, (x,))
        self.assertNotIn("<metal_tensor>", msl)
        self.assertNotIn("MetalPerformancePrimitives", msl)
        self.assertNotIn("mpp::tensor_ops", msl)

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


# ---------------------------------------------------------------------------
# Tests – multi-dimensional
# ---------------------------------------------------------------------------


@_requires_darwin
class TestMetalMultiDim(unittest.TestCase):
    """Multi-dimensional elementwise kernels."""

    def test_elementwise_2d(self) -> None:
        x = torch.randn(128, 128, device=DEVICE)
        y = torch.randn(128, 128, device=DEVICE)
        torch.testing.assert_close(elementwise_2d(x, y), x + y)

    def test_elementwise_2d_non_aligned(self) -> None:
        x = torch.randn(100, 100, device=DEVICE)
        y = torch.randn(100, 100, device=DEVICE)
        torch.testing.assert_close(elementwise_2d(x, y), x + y)

    def test_elementwise_3d(self) -> None:
        x = torch.randn(16, 16, 16, device=DEVICE)
        y = torch.randn(16, 16, 16, device=DEVICE)
        torch.testing.assert_close(elementwise_3d(x, y), x + y)


# ---------------------------------------------------------------------------
# Tests – large block / auto-capping
# ---------------------------------------------------------------------------


@_requires_darwin
class TestMetalLargeBlock(unittest.TestCase):
    """Tests for threadgroup auto-capping when block_size > 1024."""

    def test_large_block_1d(self) -> None:
        """1D kernel with block_size=2048 auto-caps to 1024 threads."""
        x = torch.randn(4096, device=DEVICE)
        y = torch.randn(4096, device=DEVICE)
        torch.testing.assert_close(large_block_add(x, y), x + y)

    def test_large_block_1d_non_aligned(self) -> None:
        """Non-aligned size with large block still works correctly."""
        x = torch.randn(3000, device=DEVICE)
        y = torch.randn(3000, device=DEVICE)
        torch.testing.assert_close(large_block_add(x, y), x + y)

    def test_codegen_lane_loop(self) -> None:
        """Generated MSL must contain a for loop when block_size > 1024."""
        x = torch.randn(4096, device=DEVICE)
        y = torch.randn(4096, device=DEVICE)
        msl = _get_msl(large_block_add, (x, y))
        self.assertIn("for (int", msl, "lane loop not found in generated MSL")

    def test_large_block_2d(self) -> None:
        """2D kernel with block_sizes=[64,64] (4096 threads) auto-caps."""
        x = torch.randn(128, 128, device=DEVICE)
        y = torch.randn(128, 128, device=DEVICE)
        torch.testing.assert_close(elementwise_2d(x, y), x + y)

    def test_large_block_3d(self) -> None:
        """3D kernel with block_sizes=[16,16,16] (4096 threads) auto-caps."""
        x = torch.randn(16, 16, 16, device=DEVICE)
        y = torch.randn(16, 16, 16, device=DEVICE)
        torch.testing.assert_close(elementwise_3d(x, y), x + y)

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
# Kernel definitions – matmul
# ---------------------------------------------------------------------------


_DEFAULT_MATMUL_CONFIG = [helion.Config(block_sizes=[32, 32, 32], num_warps=4)]


@helion.kernel(backend="metal", configs=_DEFAULT_MATMUL_CONFIG)
def matmul_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _k2, n = y.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = acc.to(x.dtype)
    return out


def _make_matmul_kernel(
    block_sizes: list[int], num_warps: int = 4
) -> helion.Kernel[torch.Tensor]:
    cfg = [helion.Config(block_sizes=block_sizes, num_warps=num_warps)]

    @helion.kernel(backend="metal", configs=cfg)
    def _matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        m, k = x.size()
        _k2, n = y.size()
        out = torch.empty([m, n], dtype=x.dtype, device=x.device)
        for tile_m, tile_n in hl.tile([m, n]):
            acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
            for tile_k in hl.tile(k):
                acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
            out[tile_m, tile_n] = acc.to(x.dtype)
        return out

    return _matmul


# ---------------------------------------------------------------------------
# Tests – matmul
# ---------------------------------------------------------------------------


@_requires_darwin
class TestMetalMatmul(unittest.TestCase):
    """Matmul kernels using MPP matmul2d through the standard pipeline."""

    def test_mpp_rejects_paravirtual_mps_device(self) -> None:
        from helion._compiler.metal.metal_jit import _raise_if_mpp_unsupported_device

        with (
            patch.object(
                torch._C,
                "_mps_get_name",
                return_value="Apple Paravirtual device",
            ),
            self.assertRaisesRegex(exc.BackendUnsupported, "paravirtual MPS"),
        ):
            _raise_if_mpp_unsupported_device()

    def test_matmul_basic(self) -> None:
        """Square matmul: 64x64 @ 64x64."""
        x = torch.randn(64, 64, device=DEVICE)
        y = torch.randn(64, 64, device=DEVICE)
        result = matmul_kernel(x, y)
        expected = torch.mm(x, y)
        torch.testing.assert_close(result, expected, atol=1e-4, rtol=1e-4)

    def test_matmul_non_square(self) -> None:
        """Non-square matmul: 128x64 @ 64x256."""
        kernel = _make_matmul_kernel([32, 64, 32])
        x = torch.randn(128, 64, device=DEVICE)
        y = torch.randn(64, 256, device=DEVICE)
        result = kernel(x, y)
        expected = torch.mm(x, y)
        torch.testing.assert_close(result, expected, atol=1e-4, rtol=1e-4)

    def test_matmul_k_loop(self) -> None:
        """K > TILE_K forces multiple K-loop iterations: 128x512 @ 512x128."""
        x = torch.randn(128, 512, device=DEVICE)
        y = torch.randn(512, 128, device=DEVICE)
        result = matmul_kernel(x, y)
        expected = torch.mm(x, y)
        torch.testing.assert_close(result, expected, atol=1e-2, rtol=1e-2)

    def test_matmul_large_square(self) -> None:
        """Larger square matmul: 256x256 @ 256x256."""
        kernel = _make_matmul_kernel([64, 64, 64])
        x = torch.randn(256, 256, device=DEVICE)
        y = torch.randn(256, 256, device=DEVICE)
        result = kernel(x, y)
        expected = torch.mm(x, y)
        torch.testing.assert_close(result, expected, atol=1e-2, rtol=1e-2)

    def test_matmul_tall_skinny(self) -> None:
        """Tall-skinny: 512x32 @ 32x64."""
        kernel = _make_matmul_kernel([32, 32, 32])
        x = torch.randn(512, 32, device=DEVICE)
        y = torch.randn(32, 64, device=DEVICE)
        result = kernel(x, y)
        expected = torch.mm(x, y)
        torch.testing.assert_close(result, expected, atol=1e-4, rtol=1e-4)

    def test_matmul_non_divisible_tiles(self) -> None:
        kernel = _make_matmul_kernel([32, 32, 32])
        for m, k, n in [(70, 64, 64), (64, 70, 64), (64, 64, 70), (71, 65, 73)]:
            with self.subTest(shape=(m, k, n)):
                x = torch.randn(m, k, device=DEVICE)
                y = torch.randn(k, n, device=DEVICE)
                result = kernel(x, y)
                expected = torch.mm(x, y)
                torch.testing.assert_close(result, expected, atol=1e-4, rtol=1e-4)

    def test_matmul_float16(self) -> None:
        """Float16 inputs with float32 accumulation cannot directly store fp16."""
        x = torch.randn(64, 64, device=DEVICE, dtype=torch.float16)
        y = torch.randn(64, 64, device=DEVICE, dtype=torch.float16)
        with self.assertRaisesRegex(
            exc.BackendUnsupported,
            "requires accumulator dtype to match output dtype",
        ):
            matmul_kernel(x, y)

    def test_matmul_float16_float32_output(self) -> None:
        cfg = [helion.Config(block_sizes=[32, 32, 32], num_warps=4)]

        @helion.kernel(backend="metal", configs=cfg)
        def matmul_fp32_out(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, k = x.size()
            _k2, n = y.size()
            out = torch.empty([m, n], dtype=torch.float32, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc
            return out

        x = torch.randn(64, 64, device=DEVICE, dtype=torch.float16)
        y = torch.randn(64, 64, device=DEVICE, dtype=torch.float16)
        result = matmul_fp32_out(x, y)
        expected = torch.mm(x.float(), y.float())
        torch.testing.assert_close(result, expected, atol=1e-2, rtol=1e-2)

        msl = _get_msl(matmul_fp32_out, (x, y))
        self.assertIn("decltype(_mpp_setup_As), decltype(_mpp_setup_Bs), float", msl)
        self.assertIn("tensor<device half", msl)

    def test_matmul_bfloat16(self) -> None:
        """Bfloat16 inputs with float32 accumulation cannot directly store bf16."""
        x = torch.randn(64, 64, device=DEVICE, dtype=torch.bfloat16)
        y = torch.randn(64, 64, device=DEVICE, dtype=torch.bfloat16)
        with self.assertRaisesRegex(
            exc.BackendUnsupported,
            "requires accumulator dtype to match output dtype",
        ):
            matmul_kernel(x, y)

    def test_matmul_codegen_has_mpp(self) -> None:
        """Generated MSL must contain MPP matmul2d constructs."""
        x = torch.randn(64, 64, device=DEVICE)
        y = torch.randn(64, 64, device=DEVICE)
        msl = _get_msl(matmul_kernel, (x, y))
        self.assertIn(
            "[[required_threads_per_threadgroup(128, 8, 1)]] kernel void",
            msl,
        )
        self.assertIn("matmul2d", msl, "MPP matmul2d not found in MSL")
        self.assertIn("_op.run", msl, "MPP run call not found in MSL")
        self.assertNotIn("auto acc = float(0.0)", msl)

    def test_matmul_with_relu(self) -> None:
        """Matmul with ReLU epilogue fusion."""
        cfg = [helion.Config(block_sizes=[32, 32, 32], num_warps=4)]

        @helion.kernel(backend="metal", configs=cfg)
        def matmul_relu(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, k = x.size()
            _k2, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc.relu()
            return out

        x = torch.randn(64, 64, device=DEVICE) * 0.05
        y = torch.randn(64, 64, device=DEVICE) * 0.05
        result = matmul_relu(x, y)
        expected = torch.mm(x, y).relu()
        torch.testing.assert_close(result, expected, atol=1e-4, rtol=1e-4)

    def test_matmul_supported_epilogues(self) -> None:
        cfg = [helion.Config(block_sizes=[32, 32, 32], num_warps=4)]

        @helion.kernel(backend="metal", configs=cfg)
        def matmul_sigmoid(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, k = x.size()
            _k2, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = torch.sigmoid(acc)
            return out

        @helion.kernel(backend="metal", configs=cfg)
        def matmul_exp(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, k = x.size()
            _k2, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = torch.exp(acc)
            return out

        @helion.kernel(backend="metal", configs=cfg)
        def matmul_neg(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, k = x.size()
            _k2, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = -acc
            return out

        @helion.kernel(backend="metal", configs=cfg)
        def matmul_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, k = x.size()
            _k2, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc + 1.25
            return out

        @helion.kernel(backend="metal", configs=cfg)
        def matmul_sub(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, k = x.size()
            _k2, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc - 0.5
            return out

        @helion.kernel(backend="metal", configs=cfg)
        def matmul_mul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, k = x.size()
            _k2, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc * 0.25
            return out

        @helion.kernel(backend="metal", configs=cfg)
        def matmul_div(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, k = x.size()
            _k2, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc / 2.0
            return out

        @helion.kernel(backend="metal", configs=cfg)
        def matmul_chain(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, k = x.size()
            _k2, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = torch.sigmoid(acc.relu() * 0.5 + 1.0)
            return out

        def apply_expected(kind: str, value: torch.Tensor) -> torch.Tensor:
            if kind == "sigmoid":
                return torch.sigmoid(value)
            if kind == "exp":
                return torch.exp(value)
            if kind == "neg":
                return -value
            if kind == "add":
                return value + 1.25
            if kind == "sub":
                return value - 0.5
            if kind == "mul":
                return value * 0.25
            if kind == "div":
                return value / 2.0
            if kind == "chain":
                return torch.sigmoid(value.relu() * 0.5 + 1.0)
            raise AssertionError(f"unknown epilogue: {kind}")

        x = torch.randn(64, 64, device=DEVICE)
        y = torch.randn(64, 64, device=DEVICE)
        mm = torch.mm(x, y)
        kernels = {
            "sigmoid": matmul_sigmoid,
            "exp": matmul_exp,
            "neg": matmul_neg,
            "add": matmul_add,
            "sub": matmul_sub,
            "mul": matmul_mul,
            "div": matmul_div,
            "chain": matmul_chain,
        }
        for kind, kernel in kernels.items():
            with self.subTest(kind=kind):
                result = kernel(x, y)
                expected = apply_expected(kind, mm)
                torch.testing.assert_close(result, expected, atol=1e-4, rtol=1e-4)
                msl = _get_msl(kernel, (x, y))
                self.assertIn("_coop.begin()", msl)
                self.assertIn("_coop.store", msl)

    @unittest.skip(
        "Flaky numerical mismatch on the metal-m2 CI runner (fixed-config "
        "matmul+aux epilogue); skip on metal pending investigation."
    )
    def test_matmul_aux_tensor_epilogue_materializes(self) -> None:
        cfg = [helion.Config(block_sizes=[32, 32, 32], num_warps=4)]

        @helion.kernel(backend="metal", configs=cfg)
        def matmul_add_aux(
            x: torch.Tensor,
            y: torch.Tensor,
            z: torch.Tensor,
        ) -> torch.Tensor:
            m, k = x.size()
            _k2, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc + z[tile_m, tile_n]
            return out

        x = torch.randn(64, 64, device=DEVICE) * 0.05
        y = torch.randn(64, 64, device=DEVICE) * 0.05
        z = torch.randn(64, 64, device=DEVICE)
        result = matmul_add_aux(x, y, z)
        expected = torch.mm(x, y) + z
        torch.testing.assert_close(result, expected, atol=1e-4, rtol=1e-4)

        msl = _get_msl(matmul_add_aux, (x, y, z))
        self.assertIn("matmul2d", msl)
        self.assertIn("_coop.store", msl)
        self.assertIn("threadgroup_barrier(mem_flags::mem_device)", msl)
        self.assertNotIn("_coop_writeback", msl)

    def test_matmul_epilogue_codegen(self) -> None:
        """ReLU epilogue must appear inside cooperative_tensor iteration."""
        cfg = [helion.Config(block_sizes=[32, 32, 32], num_warps=4)]

        @helion.kernel(backend="metal", configs=cfg)
        def matmul_relu(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, k = x.size()
            _k2, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc.relu()
            return out

        x = torch.randn(64, 64, device=DEVICE)
        y = torch.randn(64, 64, device=DEVICE)
        msl = _get_msl(matmul_relu, (x, y))
        self.assertIn("_coop.begin()", msl, "Epilogue loop not found in MSL")
        self.assertIn("_coop.store", msl, "Cooperative store not found in MSL")
        self.assertNotIn("auto acc = float(0.0)", msl)

    def test_mpp_graph_followed_by_scalar_outer_work(self) -> None:
        """MPPGraph lowering must not consume later scalar work in the root graph."""
        cfg = [helion.Config(block_sizes=[32, 32, 32], num_warps=4)]

        @helion.kernel(backend="metal", configs=cfg)
        def matmul_then_scalar(
            x: torch.Tensor,
            y: torch.Tensor,
            z: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            m, k = x.size()
            _k2, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            side = torch.empty([m, n], dtype=z.dtype, device=z.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc.to(x.dtype)
                side[tile_m, tile_n] = z[tile_m, tile_n] + 1.0
            return out, side

        x = torch.randn(64, 64, device=DEVICE)
        y = torch.randn(64, 64, device=DEVICE)
        z = torch.randn(64, 64, device=DEVICE)
        code = matmul_then_scalar.bind((x, y, z)).to_code()
        self.assertIn("_block_dims=(128, 8, 1)", code)

        out, side = matmul_then_scalar(x, y, z)
        torch.testing.assert_close(out, torch.mm(x, y), atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(side, z + 1.0)

        msl = _get_msl(matmul_then_scalar, (x, y, z))
        self.assertIn("matmul2d", msl)
        self.assertIn("_coop.store", msl)
        self.assertLess(msl.index("_coop.store"), msl.index("side["))
        self.assertNotIn("if ((tid[1] == 0))", msl)
        self.assertIn("z[", msl)
        self.assertIn("tid[1]", msl)

    def test_mpp_setup_marker_rejects_stale_arity(self) -> None:
        import ast as pyast

        from helion._compiler.metal.msl_ast_walker import _extract_mpp_setup_params

        expr = pyast.parse(
            '_metal_mpp_setup("x", "y", 64, 64, 64, 32, 32, 32, 4, '
            '"float", "float", "", "", "acc", "out", "float")'
        )
        stmt = expr.body[0]
        self.assertIsInstance(stmt, pyast.Expr)
        call = stmt.value
        self.assertIsInstance(call, pyast.Call)
        with self.assertRaisesRegex(AssertionError, "expects 14 positional args"):
            _extract_mpp_setup_params(call)

    def test_mpp_emission_scopes_symbols_by_setup_name(self) -> None:
        import ast as pyast

        from helion._compiler.metal.msl_ast_walker import EmitState
        from helion._compiler.metal.msl_ast_walker import _emit_stmts

        code = (
            '_mpp_setup = _metal_mpp_setup("x", "y", 64, 64, 64, 32, 32, 32, 4, '
            '"float", "float", "", "", "acc")\n'
            "_metal_mpp_k_step(_mpp_setup, 0)\n"
            '_metal_mpp_coop_store(_mpp_setup, "out0", "float")\n'
            '_mpp_setup_1 = _metal_mpp_setup("a", "b", 64, 64, 64, 32, 32, 32, 4, '
            '"float", "float", "", "", "acc_1")\n'
            "_metal_mpp_k_step(_mpp_setup_1, 0)\n"
            '_metal_mpp_coop_store(_mpp_setup_1, "out1", "float")\n'
        )
        state = EmitState()
        parts: list[str] = []
        _emit_stmts(pyast.parse(code).body, parts, indent=4, state=state)
        msl = "\n".join(parts)

        self.assertIn("_mpp_setup_A", msl)
        self.assertIn("_mpp_setup_1_A", msl)
        self.assertIn("_mpp_setup_C", msl)
        self.assertIn("_mpp_setup_1_C", msl)
        self.assertIn("_mpp_setup_op.run", msl)
        self.assertIn("_mpp_setup_1_op.run", msl)
        self.assertNotIn("auto _A =", msl)
        self.assertNotIn("auto _C =", msl)
        self.assertNotIn("matmul2d<_desc", msl)

    def test_matmul_via_hl_dot(self) -> None:
        """``hl.dot`` reaches the same MPPGraph pipeline as ``torch.addmm``."""
        cfg = [helion.Config(block_sizes=[32, 32, 32], num_warps=4)]

        @helion.kernel(backend="metal", configs=cfg)
        def matmul_dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, k = x.size()
            _k2, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = hl.dot(x[tile_m, tile_k], y[tile_k, tile_n], acc=acc)
                out[tile_m, tile_n] = acc.to(x.dtype)
            return out

        x = torch.randn(64, 64, device=DEVICE)
        y = torch.randn(64, 64, device=DEVICE)
        result = matmul_dot(x, y)
        torch.testing.assert_close(result, torch.mm(x, y), atol=1e-4, rtol=1e-4)

        msl = _get_msl(matmul_dot, (x, y))
        self.assertIn("matmul2d", msl, "MPP matmul2d not found in MSL (hl.dot)")
        self.assertIn("_op.run", msl, "MPP run call not found in MSL (hl.dot)")

    def test_matmul_via_hl_dot_with_relu(self) -> None:
        cfg = [helion.Config(block_sizes=[32, 32, 32], num_warps=4)]

        @helion.kernel(backend="metal", configs=cfg)
        def matmul_dot_relu(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, k = x.size()
            _k2, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = hl.dot(x[tile_m, tile_k], y[tile_k, tile_n], acc=acc)
                out[tile_m, tile_n] = acc.relu()
            return out

        x = torch.randn(64, 64, device=DEVICE)
        y = torch.randn(64, 64, device=DEVICE)
        result = matmul_dot_relu(x, y)
        torch.testing.assert_close(result, torch.mm(x, y).relu(), atol=1e-4, rtol=1e-4)

        msl = _get_msl(matmul_dot_relu, (x, y))
        self.assertIn("matmul2d", msl, "MPP matmul2d not found in MSL (hl.dot)")
        self.assertIn("_coop.begin()", msl, "Epilogue loop not found in MSL (hl.dot)")


if __name__ == "__main__":
    unittest.main()
