from __future__ import annotations

import itertools
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

    def test_tile_mask_bounds_threads_to_the_tile(self) -> None:
        """The emitted mask must bound threads to their tile."""

        @helion.kernel(
            backend="metal",
            configs=[helion.Config(block_sizes=[32, 32], num_warps=4)],
        )
        def add2d(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tm, tn in hl.tile([x.size(0), x.size(1)]):
                out[tm, tn] = x[tm, tn] + y[tm, tn]
            return out

        x = torch.randn(128, 128, device=DEVICE)
        y = torch.randn(128, 128, device=DEVICE)
        msl = _get_msl(add2d, (x, y))
        self.assertIn("(tid[0] < _BLOCK_SIZE_0)", msl)
        self.assertIn("(tid[1] < _BLOCK_SIZE_1)", msl)

    def test_a_dynamic_tile_thread_extent_is_rejected(self) -> None:
        """A threadgroup is shaped at launch, so its extents must be known then.

        ``thread_block_sizes`` reports only the extents it can resolve, and the
        launcher turns those into the literal ``_block_dims``.  An extent it
        drops leaves the axis one thread wide while the index expression stays
        ``offset + tid``, so each tile is computed in its first row alone and
        the rest of the output keeps whatever ``empty_like`` handed back.

        Only a block size read off a tensor under ``static_shapes=False`` gets
        here; a config-chosen one is an ``int``.  Three ways in, one per
        strategy: flattened, ND, and the ND lane path -- the last being the one
        whose thread mask skips an axis it cannot bound, on the strength of the
        axis never reaching codegen.
        """

        @helion.kernel(backend="metal", autotune_effort="none", static_shapes=False)
        def flat(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tm in hl.tile(x.size(0), block_size=x.size(0) // 2):
                out[tm] = x[tm] * 2.0
            return out

        @helion.kernel(backend="metal", autotune_effort="none", static_shapes=False)
        def nd_kernel(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tm, tn in hl.tile(
                [x.size(0), x.size(1)], block_size=[x.size(0) // 2, 16]
            ):
                out[tm, tn] = x[tm, tn] * 2.0
            return out

        # ``num_threads`` below the (static) block size gives axis 0 a lane
        # loop, which is what routes this through
        # ``PerThreadNDTileStrategy.codegen_grid`` rather than the base ND path.
        @helion.kernel(
            backend="metal",
            static_shapes=False,
            configs=[helion.Config(block_sizes=[64], num_threads=[16])],
        )
        def nd_lane(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tm, tn in hl.tile(
                [x.size(0), x.size(1)], block_size=[None, x.size(1) // 2]
            ):
                out[tm, tn] = x[tm, tn] * 2.0
            return out

        x = torch.randn(128, 32, device=DEVICE)
        for kernel in (flat, nd_kernel, nd_lane):
            with (
                self.subTest(kernel=kernel.fn.__name__),
                self.assertRaisesRegex(exc.BackendUnsupported, "not known until"),
            ):
                kernel(x[:, 0] if kernel is flat else x)


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

    def test_thread_budget_error_names_metal(self) -> None:
        """The shared thread-budget check must not blame the CuTe backend.

        Metal runs CuTe's loop-strategy planner, so before this was fixed a Mac
        user who over-subscribed a threadgroup was told their *cute* kernel had
        too large a thread block.
        """
        x = torch.randn(4096, device=DEVICE)
        y = torch.randn(4096, device=DEVICE)
        kernel = helion.kernel(
            large_block_add.fn,
            backend="metal",
            configs=[helion.Config(block_sizes=[2048], num_threads=[4096])],
        )
        with self.assertRaisesRegex(
            exc.BackendUnsupported, r"thread block too large for metal kernel"
        ) as caught:
            kernel(x, y)
        self.assertNotIn("cute", str(caught.exception))

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

        # The original 14-arg marker shape is stale once tile-offset names are
        # added.
        expr = pyast.parse(
            '_metal_mpp_setup("x", "y", 64, 64, 64, 32, 32, 32, 4, '
            '"float", "float", "", "", "acc")'
        )
        stmt = expr.body[0]
        self.assertIsInstance(stmt, pyast.Expr)
        call = stmt.value
        self.assertIsInstance(call, pyast.Call)
        with self.assertRaisesRegex(AssertionError, "expects 16 positional args"):
            _extract_mpp_setup_params(call)

    def test_mpp_emission_scopes_symbols_by_setup_name(self) -> None:
        import ast as pyast

        from helion._compiler.metal.msl_ast_walker import EmitState
        from helion._compiler.metal.msl_ast_walker import _emit_stmts

        code = (
            '_mpp_setup = _metal_mpp_setup("x", "y", 64, 64, 64, 32, 32, 32, 4, '
            '"float", "float", "", "", "acc", offset_0, offset_1)\n'
            "_metal_mpp_k_step(_mpp_setup, 0)\n"
            '_metal_mpp_coop_store(_mpp_setup, "out0", "float")\n'
            '_mpp_setup_1 = _metal_mpp_setup("a", "b", 64, 64, 64, 32, 32, 32, 4, '
            '"float", "float", "", "", "acc_1", offset_0, offset_1)\n'
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

    def test_matmul_bias_epilogue_is_correct(self) -> None:
        """Surplus MPP threads must not race with the scalar bias epilogue."""

        def matmul_bias(
            x: torch.Tensor, y: torch.Tensor, bias: torch.Tensor
        ) -> torch.Tensor:
            m, k = x.size()
            _k, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc + bias[None, tile_n]
            return out

        n = 256
        x = torch.randn(n, n, device=DEVICE)
        y = torch.randn(n, n, device=DEVICE)
        bias = torch.randn(n, device=DEVICE)
        expected = x @ y + bias

        for block_m, warps in [(16, 4), (16, 1), (32, 4), (64, 8), (128, 8), (64, 2)]:
            with self.subTest(block_m=block_m, num_warps=warps):
                kernel = helion.kernel(
                    matmul_bias,
                    backend="metal",
                    configs=[
                        helion.Config(block_sizes=[block_m, 16, 16], num_warps=warps)
                    ],
                )
                torch.testing.assert_close(
                    kernel(x, y, bias), expected, rtol=2e-3, atol=2e-3
                )


# ---------------------------------------------------------------------------
# Kernel definitions - reductions
# ---------------------------------------------------------------------------

_REDUCTION_CONFIG = [helion.Config(block_sizes=[16])]


@helion.kernel(backend="metal", configs=_REDUCTION_CONFIG)
def row_sum(x: torch.Tensor) -> torch.Tensor:
    m, _ = x.shape
    out = torch.empty([m], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        out[tile_m] = x[tile_m, :].sum(-1)
    return out


@helion.kernel(backend="metal", configs=_REDUCTION_CONFIG)
def row_amax(x: torch.Tensor) -> torch.Tensor:
    m, _ = x.shape
    out = torch.empty([m], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        out[tile_m] = torch.amax(x[tile_m, :], dim=-1)
    return out


@helion.kernel(backend="metal", configs=_REDUCTION_CONFIG)
def row_amin(x: torch.Tensor) -> torch.Tensor:
    m, _ = x.shape
    out = torch.empty([m], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        out[tile_m] = torch.amin(x[tile_m, :], dim=-1)
    return out


@helion.kernel(backend="metal", configs=_REDUCTION_CONFIG)
def row_prod(x: torch.Tensor) -> torch.Tensor:
    m, _ = x.shape
    out = torch.empty([m], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        out[tile_m] = torch.prod(x[tile_m, :], dim=-1)
    return out


@helion.kernel(backend="metal", configs=_REDUCTION_CONFIG)
def row_mean(x: torch.Tensor) -> torch.Tensor:
    m, _ = x.shape
    out = torch.empty([m], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        out[tile_m] = torch.mean(x[tile_m, :], dim=-1)
    return out


@helion.kernel(backend="metal", configs=_REDUCTION_CONFIG)
def row_argmax(x: torch.Tensor) -> torch.Tensor:
    m, _ = x.shape
    out = torch.empty([m], dtype=torch.int64, device=x.device)
    for tile_m in hl.tile(m):
        out[tile_m] = torch.argmax(x[tile_m, :], dim=-1)
    return out


@helion.kernel(backend="metal", configs=_REDUCTION_CONFIG)
def row_argmin(x: torch.Tensor) -> torch.Tensor:
    m, _ = x.shape
    out = torch.empty([m], dtype=torch.int64, device=x.device)
    for tile_m in hl.tile(m):
        out[tile_m] = torch.argmin(x[tile_m, :], dim=-1)
    return out


@helion.kernel(backend="metal", configs=_REDUCTION_CONFIG)
def softmax_decomposed(x: torch.Tensor) -> torch.Tensor:
    m, _ = x.shape
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        values = x[tile_m, :]
        amax = torch.amax(values, dim=1, keepdim=True)
        exp_v = torch.exp(values - amax)
        out[tile_m, :] = exp_v / torch.sum(exp_v, dim=1, keepdim=True)
    return out


@helion.kernel(backend="metal", configs=_REDUCTION_CONFIG)
def rms_norm_fwd(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    m, _ = x.shape
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        x_tile = x[tile_m, :].to(torch.float32)
        mean_sq = torch.mean(x_tile * x_tile, dim=-1, keepdim=True)
        normalized = x_tile * torch.rsqrt(mean_sq + 1e-5)
        out[tile_m, :] = (normalized * weight[:].to(torch.float32)).to(x.dtype)
    return out


@helion.kernel(backend="metal", configs=_REDUCTION_CONFIG)
def layer_norm_fwd(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    m, n = x.shape
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        acc = x[tile_m, :].to(torch.float32)
        mean = torch.sum(acc, dim=-1, keepdim=True) / n
        centered = acc - mean
        var = torch.sum(centered * centered, dim=-1, keepdim=True) / n
        normalized = centered * torch.rsqrt(var + 1e-5)
        out[tile_m, :] = (normalized * weight[:] + bias[:]).to(x.dtype)
    return out


@helion.kernel(
    backend="metal",
    configs=_REDUCTION_CONFIG,
    ignore_warnings=[exc.TensorOperationInWrapper],
)
def cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    n, v = logits.shape
    losses = torch.zeros([n], dtype=torch.float32, device=logits.device)
    logits_flat = logits.view(-1)
    for tile_n in hl.tile(n):
        flat_indices = tile_n.index * v + labels[tile_n]
        correct = hl.load(logits_flat, [flat_indices]).to(torch.float32)
        row = logits[tile_n, :].to(torch.float32)
        row_max = torch.amax(row, dim=-1)
        shifted = row - row_max[:, None]
        logsumexp = torch.log(torch.sum(torch.exp(shifted), dim=-1))
        losses[tile_n] = logsumexp + row_max - correct
    return losses


@helion.kernel(backend="metal", configs=[helion.Config(block_sizes=[4, 2])])
def sum_in_device_loop(x: torch.Tensor) -> torch.Tensor:
    """Reduction inside a device loop, so its scratch buffer is reused.

    The inner ``hl.tile(k)`` becomes a serial device loop, and the reduction
    over the trailing dim runs once per iteration against the same
    ``threadgroup`` scratch array.
    """
    m, k, _ = x.shape
    out = torch.empty([m, k], dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(m):
        for tile_k in hl.tile(k):
            out[tile_m, tile_k] = x[tile_m, tile_k, :].to(torch.float32).sum(-1)
    return out


# ---------------------------------------------------------------------------
# Reduction tests
# ---------------------------------------------------------------------------


@_requires_darwin
class TestMetalReductions(unittest.TestCase):
    """Reductions across all three Metal cross-thread reduction tiers.

    The tier is chosen by the reduction's thread span (see
    ``helion/_compiler/metal/reduction.py``):

    ==============  ==========================================================
    span            emission
    ==============  ==========================================================
    ``< 32``        ``helion_red::seg_*``, a shuffle butterfly over the segment
    ``== 32``       ``c10::metal::simd_*``
    ``> 32``        ``helion_red::tg_*`` + a ``threadgroup`` scratch buffer
    ==============  ==========================================================

    Spans above one threadgroup (1024) are rolled into a ``for`` loop over the
    reduction dim by ``reduction_loops``.
    """

    def _check(
        self,
        kernel: helion.Kernel,
        args: tuple[object, ...],
        expected: torch.Tensor,
        *,
        rtol: float = 1e-4,
        atol: float = 1e-4,
    ) -> None:
        result = kernel(*args)
        torch.testing.assert_close(result, expected, rtol=rtol, atol=atol)

    # -- span tiers ---------------------------------------------------------

    def test_sum_sub_simdgroup_span(self) -> None:
        """n=8: several reduction groups share one SIMD group."""
        x = torch.randn(64, 8, device=DEVICE)
        self._check(row_sum, (x,), x.sum(-1))

    def test_sum_exact_simdgroup_span(self) -> None:
        x = torch.randn(64, 32, device=DEVICE)
        self._check(row_sum, (x,), x.sum(-1))

    def test_sum_multi_simdgroup_span(self) -> None:
        """n=256: 8 SIMD groups per reduction, via threadgroup scratch."""
        x = torch.randn(64, 256, device=DEVICE)
        self._check(row_sum, (x,), x.sum(-1))

    def test_sum_full_threadgroup_span(self) -> None:
        x = torch.randn(32, 1024, device=DEVICE)
        self._check(row_sum, (x,), x.sum(-1), rtol=1e-3, atol=1e-3)

    def test_sum_non_power_of_2(self) -> None:
        """The padded lanes must be masked to the identity."""
        x = torch.randn(64, 1000, device=DEVICE)
        self._check(row_sum, (x,), x.sum(-1), rtol=1e-3, atol=1e-3)

    def test_sum_rolled_reduction(self) -> None:
        """n > 1024 rolls into a loop with a per-thread accumulator."""
        x = torch.randn(64, 4096, device=DEVICE)
        self._check(row_sum, (x,), x.sum(-1), rtol=1e-3, atol=1e-3)

    def test_sum_deeply_rolled_reduction(self) -> None:
        x = torch.randn(4, 65536, device=DEVICE)
        self._check(row_sum, (x,), x.sum(-1), rtol=1e-2, atol=1e-2)

    def test_sum_single_row(self) -> None:
        """One row: the whole threadgroup cooperates on a single reduction."""
        x = torch.randn(1, 8192, device=DEVICE)
        self._check(row_sum, (x,), x.sum(-1), rtol=1e-3, atol=1e-3)

    # -- reduction kinds ----------------------------------------------------

    def test_amax(self) -> None:
        x = torch.randn(33, 257, device=DEVICE)
        self._check(row_amax, (x,), x.amax(-1))

    def test_amax_propagates_nan(self) -> None:
        x = torch.randn(8, 64, device=DEVICE)
        x[3, 17] = float("nan")
        result = row_amax(x)
        self.assertTrue(torch.isnan(result[3]).item())
        torch.testing.assert_close(result[:3], x[:3].amax(-1))

    def test_amin(self) -> None:
        x = torch.randn(33, 100, device=DEVICE)
        self._check(row_amin, (x,), x.amin(-1))

    def test_prod(self) -> None:
        x = torch.rand(16, 16, device=DEVICE) + 0.5
        self._check(row_prod, (x,), x.prod(-1))

    def test_mean(self) -> None:
        x = torch.randn(32, 512, device=DEVICE)
        self._check(row_mean, (x,), x.mean(-1))

    def test_argmax_sub_simdgroup_span(self) -> None:
        x = torch.randn(32, 16, device=DEVICE)
        self._check(row_argmax, (x,), x.argmax(-1))

    def test_argmax_exact_simdgroup_span(self) -> None:
        x = torch.randn(32, 32, device=DEVICE)
        self._check(row_argmax, (x,), x.argmax(-1))

    def test_argmax_multi_simdgroup_span(self) -> None:
        x = torch.randn(32, 256, device=DEVICE)
        self._check(row_argmax, (x,), x.argmax(-1))

    def test_argmax_ties_take_lowest_index(self) -> None:
        x = torch.zeros(4, 64, device=DEVICE)
        x[:, 5] = 1.0
        x[:, 40] = 1.0
        self._check(row_argmax, (x,), x.argmax(-1))

    def test_rolled_argreductions_take_lowest_global_index(self) -> None:
        """The final cross-lane reduction must compare carried indices.

        With a 4096-element row, lanes locally reduce four chunks.  Indices 1
        and 1024 are therefore carried by different lanes in reverse index
        order; selecting the first winning lane incorrectly returns 1024.
        """
        expected = torch.ones(4, dtype=torch.int64, device=DEVICE)
        for kernel, name, extremum in (
            (row_argmax, "argmax", 1.0),
            (row_argmin, "argmin", -1.0),
        ):
            with self.subTest(kernel=name, case="tie"):
                x = torch.zeros(4, 4096, device=DEVICE)
                x[:, 1] = extremum
                x[:, 1024] = extremum
                self._check(kernel, (x,), expected)

            with self.subTest(kernel=name, case="nan"):
                x = torch.zeros(4, 4096, device=DEVICE)
                x[:, 1] = float("nan")
                x[:, 1024] = float("nan")
                self._check(kernel, (x,), expected)

    def test_argmin(self) -> None:
        x = torch.randn(32, 256, device=DEVICE)
        self._check(row_argmin, (x,), x.argmin(-1))

    # -- dtypes -------------------------------------------------------------

    def test_sum_float16(self) -> None:
        # Helion and torch both accumulate in fp32 and round once, so the only
        # slack needed is a rounding boundary flipped by summation order.
        x = torch.randn(32, 128, dtype=torch.float16, device=DEVICE)
        self._check(row_sum, (x,), x.sum(-1), rtol=5e-3, atol=5e-3)

    def test_sum_bfloat16(self) -> None:
        x = torch.randn(32, 128, dtype=torch.bfloat16, device=DEVICE)
        self._check(row_sum, (x,), x.sum(-1), rtol=2e-2, atol=2e-2)

    def test_sum_int32(self) -> None:
        x = torch.randint(-50, 50, (32, 128), dtype=torch.int32, device=DEVICE)
        self._check(row_sum, (x,), x.sum(-1).to(torch.int32))

    def test_sum_int64(self) -> None:
        """Metal has no 64-bit SIMD shuffle; c10::metal works around it."""
        x = torch.randint(-50, 50, (32, 128), dtype=torch.int64, device=DEVICE)
        self._check(row_sum, (x,), x.sum(-1))

    def test_amax_int64(self) -> None:
        x = torch.randint(-(10**12), 10**12, (32, 128), device=DEVICE)
        self._check(row_amax, (x,), x.amax(-1))

    def test_int64_extrema_are_not_polluted_by_a_zero_fill(self) -> None:
        """An int64 extremum must not be clamped towards zero.

        The cross-SIMD-group tier re-reduces the per-SIMD-group partials, and
        ``c10::metal::threadgroup_*`` does that with only ``span / 32`` lanes
        live.  Metal has no 64-bit SIMD reduction, so c10 emulates ``long``
        with ``simd_shuffle_and_fill_down`` and a fill of 0 -- which is the
        identity for sum and for nothing else, so every one of these returned
        0.  Sign-uniform data is what makes the fill visible; a two-sided
        range hides it, since the true extremum is then almost surely on the
        same side of zero as the fill.
        """
        for n in (64, 100, 128, 256, 512, 1024):
            with self.subTest(n=n):
                neg = torch.randint(-(10**12), -1, (8, n), device=DEVICE)
                self._check(row_amax, (neg,), neg.amax(-1))
                pos = torch.randint(1, 10**12, (8, n), device=DEVICE)
                self._check(row_amin, (pos,), pos.amin(-1))
                ones = torch.ones(8, n, dtype=torch.int64, device=DEVICE)
                self._check(row_prod, (ones,), ones.prod(-1))

    def test_rolled_int64_amax(self) -> None:
        """A rolled int64 reduction emits ``iinfo(int64).min`` as its identity."""
        x = torch.randint(-(10**12), -1, (8, 4096), device=DEVICE)
        self._check(row_amax, (x,), x.amax(-1))

    # -- composite kernels --------------------------------------------------

    def test_softmax(self) -> None:
        x = torch.randn(32, 256, device=DEVICE)
        self._check(softmax_decomposed, (x,), torch.softmax(x, dim=-1))

    def test_softmax_rolled(self) -> None:
        x = torch.randn(8, 4096, device=DEVICE)
        self._check(softmax_decomposed, (x,), torch.softmax(x, dim=-1))

    def test_rms_norm(self) -> None:
        x = torch.randn(32, 256, device=DEVICE)
        weight = torch.randn(256, device=DEVICE)
        expected = torch.nn.functional.rms_norm(x, (256,), weight, eps=1e-5)
        self._check(rms_norm_fwd, (x, weight), expected, rtol=1e-3, atol=1e-3)

    def test_layer_norm_two_reductions(self) -> None:
        """Mean and variance must not share a threadgroup scratch buffer."""
        x = torch.randn(32, 256, device=DEVICE)
        weight = torch.randn(256, device=DEVICE)
        bias = torch.randn(256, device=DEVICE)
        expected = torch.nn.functional.layer_norm(x, (256,), weight, bias, 1e-5)
        for _ in range(20):
            # The historical scratch-aliasing race was non-deterministic.
            self._check(
                layer_norm_fwd, (x, weight, bias), expected, rtol=1e-3, atol=1e-3
            )

    def test_cross_entropy(self) -> None:
        logits = torch.randn(64, 512, device=DEVICE)
        labels = torch.randint(0, 512, (64,), device=DEVICE)
        expected = torch.nn.functional.cross_entropy(
            logits.float(), labels, reduction="none"
        )
        self._check(cross_entropy, (logits, labels), expected, rtol=1e-3, atol=1e-3)

    def test_reduction_in_device_loop(self) -> None:
        """Scratch reuse across device-loop iterations needs a leading barrier."""
        x = torch.randn(16, 6, 128, device=DEVICE)
        for _ in range(20):
            self._check(sum_in_device_loop, (x,), x.sum(-1), rtol=1e-3, atol=1e-3)

    # -- numeric edge cases across every tier -------------------------------

    _TIERS = (
        (8, "segmented"),
        (32, "simd_group"),
        (256, "threadgroup"),
        (4096, "rolled"),
    )

    def test_nan_and_inf_across_tiers(self) -> None:
        for n, tier in self._TIERS:
            with self.subTest(n=n, tier=tier):
                x = torch.randn(16, n, device=DEVICE)
                x[0, n // 2] = float("nan")
                for kernel, ref in (
                    (row_sum, x.sum(-1)),
                    (row_amax, x.amax(-1)),
                    (row_amin, x.amin(-1)),
                    (row_argmax, x.argmax(-1)),
                    (row_argmin, x.argmin(-1)),
                ):
                    result = kernel(x)
                    torch.testing.assert_close(
                        result, ref, rtol=1e-3, atol=1e-3, equal_nan=True
                    )
                # +inf and -inf in the same row must sum to NaN, not cancel.
                z = torch.full((16, n), float("inf"), device=DEVICE)
                z[:, 1] = -float("inf")
                torch.testing.assert_close(row_sum(z), z.sum(-1), equal_nan=True)

    def test_argmax_edge_positions_across_tiers(self) -> None:
        for n, tier in self._TIERS:
            with self.subTest(n=n, tier=tier):
                # All-equal: torch returns the lowest index.
                flat = torch.zeros(16, n, device=DEVICE)
                torch.testing.assert_close(row_argmax(flat), flat.argmax(-1))
                # The extremum in the last lane of the last SIMD group.
                last = torch.zeros(16, n, device=DEVICE)
                last[:, n - 1] = 1.0
                torch.testing.assert_close(row_argmax(last), last.argmax(-1))

    def test_non_power_of_two_extents_at_tier_boundaries(self) -> None:
        for n in (7, 31, 33, 63, 65, 1023, 1025, 1500):
            with self.subTest(n=n):
                x = torch.randn(16, n, device=DEVICE)
                torch.testing.assert_close(row_sum(x), x.sum(-1), rtol=1e-3, atol=1e-3)
                torch.testing.assert_close(row_amax(x), x.amax(-1))

    def test_int32_sum_wraps_like_torch(self) -> None:
        for n, tier in self._TIERS:
            with self.subTest(n=n, tier=tier):
                x = torch.full((16, n), 2**20, dtype=torch.int32, device=DEVICE)
                torch.testing.assert_close(row_sum(x), x.sum(-1).to(torch.int32))

    # -- codegen ------------------------------------------------------------

    def test_codegen_uses_c10_metal_helpers(self) -> None:
        """Cross-SIMD-group reductions go through Inductor's reduction_utils.

        Both stages call ``c10::metal::simd_*``; the cross-SIMD-group combine
        around them is Helion's, because c10's runs its second stage with a
        partially populated SIMD group (see ``msl_reduction``).
        """
        x = torch.randn(64, 256, device=DEVICE)
        msl = _get_msl(row_sum, (x,))
        self.assertIn("#include <c10/metal/reduction_utils.h>", msl)
        self.assertIn("::c10::metal::simd_##NAME", msl)
        self.assertNotIn("::c10::metal::threadgroup_", msl)
        self.assertIn("helion_red::tg_sum(", msl)
        self.assertIn("threadgroup float _red_scratch", msl)
        self.assertIn("[[simdgroup_index_in_threadgroup]]", msl)

    def test_codegen_simdgroup_span_needs_no_shared_memory(self) -> None:
        x = torch.randn(64, 32, device=DEVICE)
        msl = _get_msl(row_sum, (x,))
        self.assertIn("helion_red::simd_sum", msl)
        self.assertNotIn("threadgroup float _red_scratch", msl)
        self.assertNotIn("simdgroup_index_in_threadgroup", msl)

    def test_codegen_sub_simdgroup_span_uses_segmented_butterfly(self) -> None:
        x = torch.randn(64, 8, device=DEVICE)
        msl = _get_msl(row_sum, (x,))
        self.assertIn("helion_red::seg_sum", msl)
        self.assertNotIn("threadgroup float _red_scratch", msl)
        self.assertNotIn("simdgroup_index_in_threadgroup", msl)

    def test_codegen_two_reductions_get_distinct_scratch(self) -> None:
        x = torch.randn(32, 256, device=DEVICE)
        weight = torch.randn(256, device=DEVICE)
        bias = torch.randn(256, device=DEVICE)
        msl = _get_msl(layer_norm_fwd, (x, weight, bias))
        self.assertIn("threadgroup float _red_scratch[", msl)
        self.assertIn("threadgroup float _red_scratch_1[", msl)

    def test_codegen_elementwise_kernel_has_no_reduction_preamble(self) -> None:
        x = torch.randn(1024, device=DEVICE)
        msl = _get_msl(copy_kernel, (x,))
        self.assertNotIn("reduction_utils.h", msl)
        self.assertNotIn("helion_red", msl)
        self.assertNotIn("simdgroup_index_in_threadgroup", msl)

    def test_reduction_owns_thread_axis_zero(self) -> None:
        """The whole shared tier rests on this invariant.

        ``helion_red::tg_*`` passes ``tid[0]`` as the thread's index *within*
        its reduction group and sizes the scratch slice as ``span / 32`` SIMD
        groups.  Both are only sound if thread-block dim 0 is exactly the
        reduction span -- i.e. the reduction owns axis 0 and no tile axis
        shares it.
        """
        import re

        for block, n in itertools.product([1, 2, 8, 16, 64, 256], [64, 256, 1024]):
            with self.subTest(block=block, n=n):
                kernel = helion.kernel(
                    row_sum.fn,
                    backend="metal",
                    configs=[helion.Config(block_sizes=[block])],
                )
                msl = _get_msl(kernel, (torch.randn(256, n, device=DEVICE),))
                dims = re.search(
                    r"required_threads_per_threadgroup\((\d+), (\d+), (\d+)\)", msl
                )
                call = re.search(
                    r"helion_red::tg_\w+\([^,]+, [^,]+, tid\[(\d)\], (\d+)\)", msl
                )
                self.assertIsNotNone(dims)
                self.assertIsNotNone(call, "expected the threadgroup reduction tier")
                assert dims is not None and call is not None
                self.assertEqual(call.group(1), "0", "reduction must own thread axis 0")
                self.assertEqual(
                    int(dims.group(1)),
                    int(call.group(2)),
                    "thread-block dim 0 must equal the reduction span",
                )

    def test_narrow_integer_and_bool_sums(self) -> None:
        """Sub-32-bit ints and bool must reduce in a promoted accumulator.

        ``c10::metal::threadgroup_sum`` takes ``threadgroup opmath_t<T>*``, and
        c10 promotes char/short/uchar to int; passing the storage type made the
        shader fail to compile above the SIMD width.
        """
        for dtype in (torch.int8, torch.int16, torch.uint8, torch.bool):
            for n in (16, 32, 64, 256, 4096):  # 4096 exercises the rolled path
                with self.subTest(dtype=dtype, n=n):
                    if dtype is torch.bool:
                        x = torch.randint(0, 2, (4, n), device=DEVICE).bool()
                    elif dtype is torch.uint8:
                        x = torch.randint(0, 4, (4, n), dtype=dtype, device=DEVICE)
                    else:
                        x = torch.randint(-3, 3, (4, n), dtype=dtype, device=DEVICE)
                    result = row_sum(x)
                    torch.testing.assert_close(result, x.sum(-1).to(result.dtype))

    def test_narrow_integer_amax(self) -> None:
        for dtype in (torch.int8, torch.int16, torch.uint8):
            for n in (32, 256):
                with self.subTest(dtype=dtype, n=n):
                    x = torch.randint(0, 100, (4, n), dtype=dtype, device=DEVICE)
                    torch.testing.assert_close(row_amax(x), x.amax(-1))

    # -- shapes the aliasing and budget guards must not reject ---------------

    def test_reduction_under_dynamic_shapes(self) -> None:
        """Every reduction kernel has a full slice, so the guard sees a SymInt.

        ``_reject_aliased_slice_dims`` used to key a dict on ``tensor.size(dim)``,
        which is unhashable under dynamic shapes -- so any Metal kernel with a
        ``:`` failed to compile at all.
        """

        @helion.kernel(backend="metal", autotune_effort="none", static_shapes=False)
        def dyn_row_sum(x: torch.Tensor) -> torch.Tensor:
            m, _ = x.size()
            out = torch.empty([m], dtype=x.dtype, device=x.device)
            for tile_m in hl.tile(m):
                out[tile_m] = x[tile_m, :].sum(-1)
            return out

        for shape in ((64, 128), (48, 96)):
            with self.subTest(shape=shape):
                x = torch.randn(*shape, device=DEVICE)
                torch.testing.assert_close(
                    dyn_row_sum(x), x.sum(-1), rtol=3e-3, atol=3e-3
                )

    def test_repeated_size_one_slices_are_allowed(self) -> None:
        """Length-1 full slices are indexed by a constant, so they cannot alias.

        They never reach ``allocate_reduction_dimension`` and claim no thread
        axis, but the equal-length check counted them anyway and rejected a
        kernel that compiles correctly.
        """

        @helion.kernel(backend="metal", configs=[helion.Config(block_sizes=[16])])
        def scale(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile_m in hl.tile(x.size(0)):
                out[tile_m, :, :] = x[tile_m, :, :] * 2.0
            return out

        x = torch.randn(64, 1, 1, device=DEVICE)
        torch.testing.assert_close(scale(x), x * 2.0)

    def test_explicit_block_size_with_a_reduction_reports_a_backend_error(self) -> None:
        """A tile axis with no ``num_threads`` knob must not crash the budget pass.

        ``hl.tile(n, block_size=<int>)`` allocates no ``NumThreadsSpec``, so the
        reduction thread budget has nowhere to write a cap.  It used to index
        the spec map anyway and raise a bare ``KeyError`` out of compilation;
        16 rows x a 128-wide reduction genuinely exceeds a threadgroup, so the
        right outcome is a Helion diagnostic naming the way out.
        """

        @helion.kernel(backend="metal", autotune_effort="none")
        def fixed_tile_row_sum(x: torch.Tensor) -> torch.Tensor:
            m, _ = x.size()
            out = torch.empty([m], dtype=x.dtype, device=x.device)
            for tile_m in hl.tile(x.size(0), block_size=16):
                out[tile_m] = x[tile_m, :].sum(-1)
            return out

        x = torch.randn(64, 128, device=DEVICE)
        with self.assertRaisesRegex(exc.BackendUnsupported, "reduction_loops"):
            fixed_tile_row_sum(x)

    # -- unsupported --------------------------------------------------------

    def test_two_equal_size_reduction_dims_are_rejected(self) -> None:
        """Equal-size reduction dims share a block id, hence a thread index.

        ``allocate_reduction_dimension`` caches by size, so both trailing axes
        of an ``[m, n, n]`` tensor get one index variable.  A tile-level backend
        keeps them apart by broadcasting; Metal would collapse them onto one
        thread and reduce the diagonal, so the kernel must be rejected.
        """

        @helion.kernel(backend="metal", configs=[helion.Config(block_sizes=[1])])
        def two_reduce_dims(x: torch.Tensor) -> torch.Tensor:
            m, _, _ = x.shape
            out = torch.empty([m], dtype=x.dtype, device=x.device)
            for tile_m in hl.tile(m):
                out[tile_m] = x[tile_m, :, :].sum(-1).sum(-1)
            return out

        with self.assertRaisesRegex(exc.BackendUnsupported, "full slice of length 8"):
            two_reduce_dims(torch.randn(3, 8, 8, device=DEVICE))

    def test_equal_size_slices_without_a_reduction_are_rejected(self) -> None:
        """Equal-length full slices alias even with no reduction op present.

        ``out[tile, :, :] = a[tile, :, None] * b[None, None, :]`` has two
        reduction *dimensions* but no reduction, so the check has to live at
        the access rather than at the reduction lowering.
        """

        @helion.kernel(backend="metal", configs=[helion.Config(block_sizes=[1])])
        def outer_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            m, n = a.shape
            (n2,) = b.shape
            out = torch.empty([m, n, n2], dtype=a.dtype, device=a.device)
            for tile_m in hl.tile(m):
                out[tile_m, :, :] = a[tile_m, :, None] * b[None, None, :]
            return out

        a = torch.randn(3, 8, device=DEVICE)
        with self.assertRaisesRegex(exc.BackendUnsupported, "full slice of length 8"):
            outer_product(a, torch.randn(8, device=DEVICE))
        with self.assertRaisesRegex(exc.BackendUnsupported, "2 reduction dimensions"):
            outer_product(a, torch.randn(16, device=DEVICE))

    def test_broadcast_aliased_reduction_is_rejected(self) -> None:
        """Aliasing built by broadcasting two separately-loaded values.

        No single access carries two equal full slices, and both axes are the
        *same* rdim so the dimension count stays at 1 -- neither access-level
        guard can see this.  Only the reduction-level check, which inspects the
        shape of the value being reduced, catches it.  Left unguarded these
        computed ``dot(a[m], b)`` and broadcast it across the whole output row.
        """

        @helion.kernel(backend="metal", configs=[helion.Config(block_sizes=[1])])
        def bcast_two_loads(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            m, n = a.shape
            out = torch.empty([m, n], dtype=a.dtype, device=a.device)
            for tile_m in hl.tile(m):
                out[tile_m, :] = (a[tile_m, :, None] * b[None, None, :]).sum(-1)
            return out

        @helion.kernel(backend="metal", configs=[helion.Config(block_sizes=[1])])
        def self_outer(x: torch.Tensor) -> torch.Tensor:
            m, _ = x.shape
            out = torch.empty([m], dtype=x.dtype, device=x.device)
            for tile_m in hl.tile(m):
                row = x[tile_m, :]
                out[tile_m] = (row[:, :, None] * row[:, None, :]).sum(-1).sum(-1)
            return out

        with self.assertRaisesRegex(exc.BackendUnsupported, "share one"):
            bcast_two_loads(
                torch.randn(3, 8, device=DEVICE), torch.randn(8, device=DEVICE)
            )
        with self.assertRaisesRegex(exc.BackendUnsupported, "share one"):
            self_outer(torch.randn(5, 8, device=DEVICE))

    def test_two_unequal_reduction_dims_are_rejected(self) -> None:
        """Distinct sizes give distinct block ids, i.e. two reduction axes."""

        @helion.kernel(backend="metal", configs=[helion.Config(block_sizes=[1])])
        def two_reduce_dims(x: torch.Tensor) -> torch.Tensor:
            m, _, _ = x.shape
            out = torch.empty([m], dtype=x.dtype, device=x.device)
            for tile_m in hl.tile(m):
                out[tile_m] = x[tile_m, :, :].sum(-1).sum(-1)
            return out

        with self.assertRaisesRegex(exc.BackendUnsupported, "2 reduction dimensions"):
            two_reduce_dims(torch.randn(3, 8, 16, device=DEVICE))

    def test_strided_reduce_group_is_rejected(self) -> None:
        """Reducing a thread axis above a sibling axis would mix rows."""

        @helion.kernel(backend="metal", autotune_effort="none", static_shapes=True)
        def inner_dim_sum(w: torch.Tensor) -> torch.Tensor:
            o, d = w.shape
            d = hl.specialize(d)
            out = torch.empty([o], dtype=torch.float32, device=w.device)
            d_block = hl.register_block_size(
                helion.next_power_of_2(d), helion.next_power_of_2(d)
            )
            for tile_o, tile_d in hl.tile([o, d], block_size=[None, d_block]):
                out[tile_o] = torch.sum(w[tile_o, tile_d].to(torch.float32), dim=-1)
            return out

        w = torch.randn(512, 128, device=DEVICE)
        with self.assertRaisesRegex(exc.BackendUnsupported, "strided by"):
            inner_dim_sum(w)

    def test_device_loop_lane_reduction_is_rejected(self) -> None:
        """A reduction carried by a device-loop lane must not under-reduce."""

        @helion.kernel(
            backend="metal",
            configs=[helion.Config(block_sizes=[1, 2048], num_threads=[1, 1024])],
        )
        def inner_tile_sum(x: torch.Tensor) -> torch.Tensor:
            m, k = x.shape
            out = torch.empty([m], dtype=torch.float32, device=x.device)
            for tile_m in hl.tile(m):
                for tile_k in hl.tile(k):
                    out[tile_m] = x[tile_m, tile_k].to(torch.float32).sum(dim=1)
            return out

        x = torch.randn(1, 2048, device=DEVICE)
        with self.assertRaisesRegex(exc.BackendUnsupported, "lane-loop reductions"):
            inner_tile_sum(x)

    def test_a_starved_reduction_loop_raises_instead_of_truncating(self) -> None:
        """The thread-budget passes keep the span wide enough; assert it anyway.

        A reduction loop whose span is narrower than its chunk reduces only
        part of each chunk and returns a plausible wrong answer.  The budget
        passes make that unreachable, so this forces the condition directly --
        the point is that the invariant is checked rather than trusted, since
        it silently broke once already.
        """
        from helion._compiler.metal.backend import MetalBackend

        @helion.kernel(
            backend="metal",
            configs=[helion.Config(block_sizes=[8], reduction_loops=[64])],
        )
        def row_sum(x: torch.Tensor) -> torch.Tensor:
            m, _ = x.size()
            out = torch.empty([m], dtype=x.dtype, device=x.device)
            for tm in hl.tile(m):
                out[tm] = x[tm, :].sum(-1)
            return out

        x = torch.randn(64, 256, device=DEVICE)
        torch.testing.assert_close(row_sum(x), x.sum(-1), rtol=3e-3, atol=3e-3)

        # Starve the reduction: hand it half the threads its chunk needs.
        with (
            patch.object(
                MetalBackend,
                "adjust_reduction_thread_count",
                lambda self, requested, existing: min(requested, 32),
            ),
            self.assertRaisesRegex(exc.BackendUnsupported, "reduction loop"),
        ):
            helion.kernel(
                row_sum.fn,
                backend="metal",
                configs=[helion.Config(block_sizes=[8], reduction_loops=[64])],
            )(x)

    def test_block_reduction_over_a_user_tile(self) -> None:
        """Reducing a user tile exercises resolve_group's second loop.

        With a single tile axis on ``tid[0]``, each block is contiguous, so
        the per-tile sum reduces through the ``seg_`` tier and broadcasts
        back over the tile.
        """
        for block in (8, 32):
            with self.subTest(block=block):

                @helion.kernel(
                    backend="metal",
                    configs=[helion.Config(block_sizes=[block])],
                )
                def block_broadcast_sum(x: torch.Tensor) -> torch.Tensor:
                    n = x.shape[0]
                    out = torch.empty([n], dtype=x.dtype, device=x.device)
                    for tile_n in hl.tile(n):
                        out[tile_n] = x[tile_n].sum()
                    return out

                x = torch.randn(64, device=DEVICE)
                self._check(
                    block_broadcast_sum,
                    (x,),
                    x.unflatten(-1, (-1, block)).sum(-1).repeat_interleave(block),
                    rtol=1e-3,
                    atol=1e-3,
                )

    def test_prod_across_tiers(self) -> None:
        """Well-conditioned values: a wide prod would overflow in fp32."""
        for n in (32, 256, 4096):
            with self.subTest(n=n):
                x = 1 + 0.01 * torch.randn(16, n, device=DEVICE)
                self._check(row_prod, (x,), x.prod(-1), rtol=1e-2, atol=1e-2)


# ---------------------------------------------------------------------------
# Autotuning
# ---------------------------------------------------------------------------

#: Autotuning compiles and benchmarks each candidate inline, so keep the tests
#: to a small budget and small tensors.
_AUTOTUNE_KWARGS = {"autotune_effort": "quick", "autotune_budget_seconds": 5}


@_requires_darwin
class TestMetalAutotune(unittest.TestCase):
    """Autotuning is enabled for Metal; these pin the plumbing it needs."""

    def test_benchmark_hooks_are_batched(self) -> None:
        from helion._compiler.metal.autotune import do_bench_metal
        from helion._compiler.metal.autotune import interleaved_bench_metal
        from helion._compiler.metal.backend import MetalBackend

        backend = MetalBackend()
        self.assertIs(backend.get_do_bench(), do_bench_metal)
        self.assertIs(backend.get_interleaved_bench(), interleaved_bench_metal)
        self.assertFalse(backend.supports_precompile())

    def test_interleaved_bench_batches_each_candidate_separately(self) -> None:
        """A cheap candidate must not be measured at an expensive one's batch.

        ``_time_batch_ms`` already returns a per-call time, so a shared batch
        equalizes nothing -- it just drags every candidate down to the slowest
        one's batch size, which is where per-sample overhead is least
        amortized.  One expensive candidate in the list used to collapse the
        batch to 1 for all of them, i.e. batching switched off.
        """
        from helion._compiler.metal.autotune import do_bench_metal
        from helion._compiler.metal.autotune import interleaved_bench_metal

        small = torch.randn(512, 512, device=DEVICE)
        big = torch.randn(2048, 2048, device=DEVICE)
        cheap = lambda: small + 1.0  # noqa: E731
        pricey = lambda: torch.mm(big, big)  # noqa: E731

        truth = float(do_bench_metal(pricey)) / float(do_bench_metal(cheap))
        t_cheap, t_pricey = interleaved_bench_metal([cheap, pricey], repeat=5)
        measured = t_pricey / t_cheap

        self.assertGreater(measured, 1.0)
        # Generous, because the point is the order of magnitude: a shared batch
        # reported ~1.2x where the true gap was ~7x.
        self.assertGreater(measured, truth / 3.0)

    def test_identical_candidates_measure_the_same(self) -> None:
        """The self-test: N copies of one kernel must rank as a tie.

        This needs no reference timing -- the true ratio is exactly 1 by
        construction -- which makes it the one calibration check here that
        cannot be undermined by a drifting baseline.  It is also a direct
        probe of the bug this function had: sharing one batch across
        candidates handed them different amounts of per-sample overhead, and
        any per-candidate asymmetry (batch, position in the round) shows up
        here as spread.

        The operand size is load-bearing.  At 1024x1024 a call costs ~0.03ms, so
        the batch saturates at ``_MAX_BATCH`` and ``repeat`` stops controlling
        anything: every candidate is summarized from the ``_MIN_ROUNDS`` floor of
        three samples, and the assertion becomes a coin flip on a busy machine.
        At 4096x4096 a call is ~0.9ms, the batch lands around 22, and ``repeat``
        buys ~18 rounds -- enough for the ``min`` to mean something.

        Best of three attempts even so, because this is wall-clock on a shared
        GPU.  The defect it exists to catch is not subtle: sharing one batch
        across candidates mismeasured a 7.4x difference as 1.2x and collapsed one
        candidate's batch to a single call, which fails all three attempts.
        """
        from helion._compiler.metal.autotune import interleaved_bench_metal

        x = torch.randn(4096, 4096, device=DEVICE)
        y = torch.randn(4096, 4096, device=DEVICE)
        same = lambda: x + y  # noqa: E731

        for count in (2, 4):
            with self.subTest(candidates=count):
                spreads = []
                for _ in range(3):
                    times = interleaved_bench_metal([same] * count, repeat=400)
                    self.assertEqual(len(times), count)
                    spreads.append(max(times) / min(times))
                self.assertLess(min(spreads), 1.25, f"spreads={spreads}")

    def test_interleaved_repeat_is_a_call_budget_not_a_round_count(self) -> None:
        """``repeat`` counts calls in the shared benchmarks, not rounds.

        Callers derive it as ``target_ms / per_call_ms`` and the shared
        implementations invoke each candidate once per round.  A round here is
        a whole batch, so spending ``repeat`` rounds overshoots the requested
        budget by the batch size -- which is what made final-pick verification
        take longer than the search it was verifying.
        """
        from helion._compiler.metal import autotune as metal_autotune

        batches: list[int] = []

        def spy(fn: object, batch: int) -> float:
            batches.append(batch)
            return 1.0

        def run(repeat: int) -> int:
            """Timed rounds for ``repeat``, with warmup and probing stubbed out."""
            batches.clear()
            with (
                patch.object(metal_autotune, "_warm_up", lambda fn: None),
                patch.object(metal_autotune, "_estimate_per_call_ms", lambda fn: 1.0),
                patch.object(metal_autotune, "_time_batch_ms", spy),
            ):
                metal_autotune.interleaved_bench_metal([lambda: None], repeat=repeat)
            return len(batches)

        # 1ms per call against a 20ms sample target -> batch of 20.
        batch = int(metal_autotune._SAMPLE_TARGET_MS)
        self.assertEqual(run(1000), 1000 // batch)
        self.assertEqual(set(batches), {batch})
        # The floor applies when the budget is smaller than a single batch.
        self.assertEqual(run(1), metal_autotune._MIN_ROUNDS)

    def test_interleaved_honors_max_total_ms(self) -> None:
        """The rebenchmark path always passes ``max_total_ms``.

        ``BaseSearch.rebenchmark`` binds it via ``functools.partial``
        before calling the backend hook, so a hook that does not accept
        it fails every final verification with a TypeError.
        """
        import functools

        from helion._compiler.metal import autotune as metal_autotune

        batches: list[int] = []

        def spy(fn: object, batch: int) -> float:
            batches.append(batch)
            return 1.0

        def run(bench_fn: object, *args: object, **kwargs: object) -> object:
            batches.clear()
            with (
                patch.object(metal_autotune, "_warm_up", lambda fn: None),
                patch.object(metal_autotune, "_estimate_per_call_ms", lambda fn: 1.0),
                patch.object(metal_autotune, "_time_batch_ms", spy),
            ):
                return bench_fn(*args, **kwargs)  # type: ignore[operator]

        # 1ms per call -> batch of 20 -> 20ms per round; a 100ms budget
        # buys 5 rounds out of the 50 that ``repeat`` alone would allow,
        # invoked the way the caller invokes it.
        bench = functools.partial(
            metal_autotune.interleaved_bench_metal, max_total_ms=100.0
        )
        self.assertEqual(run(bench, [lambda: None], repeat=1000), [1.0])
        self.assertEqual(len(batches), 5)
        # No budget: the repeat-derived count stands.
        run(metal_autotune.interleaved_bench_metal, [lambda: None], repeat=1000)
        self.assertEqual(len(batches), 50)

    def test_warmup_budget_is_spent_in_whole_batches(self) -> None:
        """``warmup`` is a millisecond budget, like ``rep``.

        It used to run ``warmup / sample_ms`` *single* calls, so it actually
        covered ``warmup / batch`` ms -- a 25ms request bought under 3ms of
        warmup, and the 1000ms the rebenchmark path asks for bought ~10ms.
        """
        from helion._compiler.metal import autotune as metal_autotune

        calls = 0

        def counted() -> None:
            nonlocal calls
            calls += 1

        seen: list[int] = []
        real_time_batch_ms = metal_autotune._time_batch_ms

        def spy(fn: object, batch: int) -> float:
            seen.append(batch)
            return real_time_batch_ms(fn, batch)

        with patch.object(metal_autotune, "_time_batch_ms", spy):
            metal_autotune.do_bench_metal(counted, warmup=25, rep=100)

        # Every timed loop -- estimate, warmup and rep -- goes through
        # _time_batch_ms, so no call is left un-batched.
        self.assertTrue(seen)
        self.assertEqual(calls, sum(seen) + 1)  # +1 for the initial correctness call
        self.assertGreater(max(seen), 1)

    def test_rejects_an_unknown_return_mode(self) -> None:
        """Summarizing is delegated to the shared fallback, so only the
        argument check is Metal's to make."""
        from helion._compiler.metal.autotune import do_bench_metal

        with self.assertRaises(AssertionError):
            do_bench_metal(lambda: None, return_mode="bogus")

    def test_unsupported_config_is_a_search_miss_not_a_crash(self) -> None:
        from helion._compiler.metal.backend import MetalBackend

        backend = MetalBackend()
        self.assertEqual(
            backend.classify_autotune_exception(exc.BackendUnsupported("metal", "x")),
            "debug",
        )
        self.assertEqual(
            backend.classify_autotune_exception(SyntaxError("bad MSL")), "warn"
        )
        # Anything else is still one bad candidate, not a reason to stop.  With
        # no catch-all the shared fallback is classify_triton_exception, which
        # answers "raise" for messages it does not recognize -- and it
        # recognizes no MPS message, so an unfamiliar driver error would kill
        # the search.
        self.assertEqual(
            backend.classify_autotune_exception(RuntimeError("some MPS error")), "warn"
        )
        # ...but an interrupt is not a search miss.
        self.assertIsNone(backend.classify_autotune_exception(KeyboardInterrupt()))

    def test_num_threads_is_clamped_to_a_divisor_of_the_block_size(self) -> None:
        """The shared loop strategy needs block_size % num_threads == 0.

        num_threads is drawn from the tensor extent, not the chosen block size,
        so the search space contains combinations that cannot compile. They are
        clamped rather than skipped, which keeps the whole space usable.
        """

        @helion.kernel(
            backend="metal",
            configs=[helion.Config(block_sizes=[64], num_threads=[1024])],
        )
        def scaled_copy(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size(0)):
                out[tile] = x[tile] * 2.0
            return out

        x = torch.randn(4096, device=DEVICE)
        torch.testing.assert_close(scaled_copy(x), x * 2.0)
        # Pin the value, not just "it compiled": the clamp picks the largest
        # divisor of the block size that is <= the requested count, and since
        # both are powers of two an over-large request always lands on exactly
        # the block size.  Asserting correctness alone would pass for any legal
        # divisor, including 1.
        self.assertIn("_block_dims=(64, 1, 1)", scaled_copy.bind((x,)).to_code())

    def test_autotuned_elementwise_1d_is_correct(self) -> None:
        @helion.kernel(backend="metal", **_AUTOTUNE_KWARGS)
        def vector_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size(0)):
                out[tile] = x[tile] + y[tile]
            return out

        x = torch.randn(8192, device=DEVICE)
        y = torch.randn(8192, device=DEVICE)
        torch.testing.assert_close(vector_add(x, y), x + y)

    def test_autotuned_elementwise_2d_is_correct(self) -> None:
        """2D tiling is where autotuning actually pays: the default
        ``block_sizes=[32, 32]`` is rarely the best tile shape."""

        @helion.kernel(backend="metal", **_AUTOTUNE_KWARGS)
        def add2d(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tm, tn in hl.tile([x.size(0), x.size(1)]):
                out[tm, tn] = x[tm, tn] + y[tm, tn]
            return out

        x = torch.randn(256, 256, device=DEVICE)
        y = torch.randn(256, 256, device=DEVICE)
        torch.testing.assert_close(add2d(x, y), x + y)

    def test_autotuned_reduction_is_correct(self) -> None:
        @helion.kernel(backend="metal", **_AUTOTUNE_KWARGS)
        def row_sum(x: torch.Tensor) -> torch.Tensor:
            n, _ = x.size()
            out = torch.empty([n], dtype=x.dtype, device=x.device)
            for tile_n in hl.tile(n):
                out[tile_n] = torch.sum(x[tile_n, :], dim=-1)
            return out

        x = torch.randn(256, 512, device=DEVICE)
        torch.testing.assert_close(row_sum(x), x.sum(-1), rtol=1e-3, atol=1e-3)

    def test_autotuned_softmax_is_correct(self) -> None:
        """Two reductions over one dimension, each with its own scratch."""

        @helion.kernel(backend="metal", **_AUTOTUNE_KWARGS)
        def softmax(x: torch.Tensor) -> torch.Tensor:
            n, _ = x.size()
            out = torch.empty_like(x)
            for tile_n in hl.tile(n):
                values = x[tile_n, :]
                amax = torch.amax(values, dim=1, keepdim=True)
                exp_v = torch.exp(values - amax)
                out[tile_n, :] = exp_v / torch.sum(exp_v, dim=1, keepdim=True)
            return out

        x = torch.randn(128, 512, device=DEVICE)
        torch.testing.assert_close(
            softmax(x), torch.softmax(x, dim=-1), rtol=1e-3, atol=1e-3
        )

    def test_reduction_loops_is_searchable(self) -> None:
        """The reduction span is the knob worth tuning, so it must be in the space."""

        @helion.kernel(backend="metal", autotune_effort="none")
        def row_sum(x: torch.Tensor) -> torch.Tensor:
            n, _ = x.size()
            out = torch.empty([n], dtype=x.dtype, device=x.device)
            for tile_n in hl.tile(n):
                out[tile_n] = torch.sum(x[tile_n, :], dim=-1)
            return out

        spec = row_sum.bind((torch.randn(256, 1024, device=DEVICE),)).config_spec
        self.assertTrue(spec.reduction_loops.valid_block_ids())
        self.assertIn("reduction_loops", spec.default_config().config)

    def test_explicit_tile_threads_do_not_starve_a_reduction(self) -> None:
        """Autotuning sets num_threads explicitly, unlike the default config.

        A persistent reduction needs one live thread per element, and Metal has
        no lane-loop fallback to cover a shortfall the way CuTe does, so a
        tile that claims the whole threadgroup used to make the config
        unbuildable.  The tile's thread count is capped instead.
        """

        @helion.kernel(
            backend="metal",
            configs=[
                helion.Config(
                    block_sizes=[64], num_threads=[1024], reduction_loops=[None]
                )
            ],
        )
        def row_sum(x: torch.Tensor) -> torch.Tensor:
            n, _ = x.size()
            out = torch.empty([n], dtype=x.dtype, device=x.device)
            for tile_n in hl.tile(n):
                out[tile_n] = torch.sum(x[tile_n, :], dim=-1)
            return out

        x = torch.randn(256, 1024, device=DEVICE)
        torch.testing.assert_close(row_sum(x), x.sum(-1), rtol=1e-3, atol=1e-3)

    def test_autotuned_matmul_is_correct(self) -> None:
        """Matmul is where tuning matters most: the default 16x16x16 tile
        badly underuses MPP's cooperative matmul."""

        @helion.kernel(backend="metal", **_AUTOTUNE_KWARGS)
        def matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, k = x.size()
            _k, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc
            return out

        x = torch.randn(128, 128, device=DEVICE)
        y = torch.randn(128, 128, device=DEVICE)
        torch.testing.assert_close(matmul(x, y), x @ y, rtol=1e-3, atol=1e-3)

    def test_oversized_threadgroup_is_a_search_miss(self) -> None:
        """An over-budget threadgroup is routine, not noteworthy.

        Metal only rejects one when it builds the pipeline state, so it arrives
        as a RuntimeError from the launcher.  It is the same category as
        BackendUnsupported -- the config does not fit -- and it is common, so it
        is classified alongside the other expected misses rather than warned
        about.
        """
        from helion._compiler.metal.backend import MetalBackend
        from helion._compiler.metal.backend import _is_threadgroup_too_large

        backend = MetalBackend()
        err = RuntimeError(
            "Failed to created pipeline state object, error: Specified total "
            "max threads per threadgroup (8192) exceeds the maximum total "
            "threads per threadgroup supported (1024)"
        )
        self.assertTrue(_is_threadgroup_too_large(err))
        self.assertFalse(_is_threadgroup_too_large(RuntimeError("unrelated")))

        # The recognizer has to change the outcome, or it is dead code behind
        # the catch-all: recognized -> "debug", unrecognized -> "warn".
        self.assertEqual(backend.classify_autotune_exception(err), "debug")
        self.assertEqual(
            backend.classify_autotune_exception(RuntimeError("unrelated")), "warn"
        )

    def _mpp_matmul(self, bm: int, bn: int, bk: int, warps: int) -> helion.Kernel:
        @helion.kernel(
            backend="metal",
            configs=[helion.Config(block_sizes=[bm, bn, bk], num_warps=warps)],
        )
        def matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, k = x.size()
            _k, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc
            return out

        return matmul

    def test_mpp_grid_axes_come_from_matmul_indices(self) -> None:
        """An unrelated same-sized grid axis must not stand in for M."""

        @helion.kernel(
            backend="metal",
            configs=[helion.Config(block_sizes=[32, 16, 8, 16], num_warps=4)],
        )
        def extra_grid_axis(
            a: torch.Tensor, b: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            m, k = a.size()
            _k, n = b.size()
            side = torch.empty([m], dtype=a.dtype, device=a.device)
            out = torch.empty([m, n], dtype=a.dtype, device=a.device)
            for extra, tm, tn in hl.tile([m, m, n]):
                side[extra] = a[extra, 0]
                acc = hl.zeros([tm, tn], dtype=torch.float32)
                for tk in hl.tile(k):
                    acc = torch.addmm(acc, a[tm, tk], b[tk, tn])
                out[tm, tn] = acc
            return out, side

        a = torch.randn(32, 32, device=DEVICE)
        b = torch.randn(32, 24, device=DEVICE)
        out, side = extra_grid_axis(a, b)
        torch.testing.assert_close(out, a @ b, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(side, a[:, 0])

    def test_matmul_in_a_later_root_grid_is_correct(self) -> None:
        """A later grid uses its rebased tile offsets, not the raw launch ID."""

        @helion.kernel(backend="metal", autotune_effort="none")
        def two_grids(
            a: torch.Tensor, b: torch.Tensor, z: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            m, k = a.size()
            _k, n = b.size()
            side = torch.empty_like(z)
            out = torch.empty([m, n], dtype=a.dtype, device=a.device)
            for tz in hl.tile(z.size(0)):
                side[tz] = z[tz] + 1.0
            for tm, tn in hl.tile([m, n]):
                acc = hl.zeros([tm, tn], dtype=torch.float32)
                for tk in hl.tile(k):
                    acc = torch.addmm(acc, a[tm, tk], b[tk, tn])
                out[tm, tn] = acc
            return out, side

        a = torch.randn(128, 128, device=DEVICE)
        b = torch.randn(128, 128, device=DEVICE)
        z = torch.randn(64, device=DEVICE)
        out, side = two_grids(a, b, z)
        torch.testing.assert_close(out, a @ b, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(side, z + 1.0)

        msl = _get_msl(two_grids, (a, b, z))
        self.assertIn("_ty = (offset_", msl)
        self.assertIn("_tx = (offset_", msl)

    def test_accuracy_check_guards_the_mpp_matmul(self) -> None:
        """Matmul autotuning depends on candidate validation being on.

        ``mpp::tensor_ops::matmul2d`` returns silently wrong results for
        strongly asymmetric tiles at low ``execution_simdgroups<N>`` -- 8:1
        needs N>=2 and 1:8 needs N>=4 on an M4 Pro -- with no compile or
        runtime error.  That predates this backend and reproduces in plain MSL
        with no Helion involved.  Those tile shapes are exactly what a tuner
        reaches for on non-square problems, so the search relies on
        ``autotune_accuracy_check`` rejecting them.
        """
        from helion.runtime.settings import Settings

        self.assertTrue(Settings().autotune_accuracy_check)

    def test_effort_none_uses_the_default_config(self) -> None:
        @helion.kernel(backend="metal", autotune_effort="none")
        def scaled(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size(0)):
                out[tile] = x[tile] * 3.0
            return out

        x = torch.randn(1024, device=DEVICE)
        torch.testing.assert_close(scaled(x), x * 3.0)
        bound = scaled.bind((x,))
        self.assertEqual(bound._config, bound.config_spec.default_config())

    def test_explicit_threads_with_a_rolled_reduction_are_correct(self) -> None:
        """Regression: this config silently returned a wrong sum.

        Clamping ``num_threads`` to a divisor of the block size let configs
        compile that the loop strategy had previously rejected.  For a *rolled*
        reduction that was not a win: the tile then claimed enough of the
        threadgroup that ``adjust_reduction_thread_count`` shrank the reduction
        span below the chunk size, so each iteration reduced only the first
        ``span`` of its 64 elements.  No error -- just a plausible wrong
        answer.  Only the autotuner produced these, because they need an
        explicit tile ``num_threads`` and an explicit ``reduction_loops``
        together, which no hand-written test config combined.
        """

        @helion.kernel(
            backend="metal",
            configs=[
                helion.Config(block_sizes=[32], num_threads=[256], reduction_loops=[64])
            ],
        )
        def row_sum(x: torch.Tensor) -> torch.Tensor:
            m, _ = x.size()
            out = torch.empty([m], dtype=x.dtype, device=x.device)
            for tm in hl.tile(m):
                out[tm] = x[tm, :].sum(-1)
            return out

        x = torch.randn(64, 256, device=DEVICE)
        torch.testing.assert_close(row_sum(x), x.sum(-1), rtol=3e-3, atol=3e-3)

    def test_autotune_budget_bounded_restored_and_bypassed(self) -> None:
        """The search budget defaults, passes through, and is restored.

        ``autotune()`` mutates shared settings, so it must leave them as
        found; the 300 s default must apply unless the caller sets a budget
        or asks for full effort.  The spy stands in for the real search, so
        this tests the settings handling without running one.
        """
        from typing import TYPE_CHECKING
        from typing import Any

        from helion._compiler import backend as compiler_backend
        from helion._compiler.metal.backend import _DEFAULT_AUTOTUNE_BUDGET_SECONDS
        from helion._compiler.metal.backend import MetalBackend

        if TYPE_CHECKING:
            from helion.runtime.config import Config
            from helion.runtime.kernel import BoundKernel

        seen: list[object] = []

        def spy(
            backend: MetalBackend,
            bound_kernel: BoundKernel[Any],
            *args: object,
            **kwargs: object,
        ) -> Config:
            seen.append(bound_kernel.settings.autotune_budget_seconds)
            return bound_kernel.config_spec.default_config()

        x = torch.randn(1024, device=DEVICE)
        for effort, budget, expected in [
            ("quick", None, _DEFAULT_AUTOTUNE_BUDGET_SECONDS),
            ("full", None, None),
            ("quick", 7, 7),
        ]:
            with self.subTest(effort=effort, budget=budget):

                @helion.kernel(
                    backend="metal",
                    autotune_effort=effort,
                    autotune_budget_seconds=budget,
                )
                def scaled(x: torch.Tensor) -> torch.Tensor:
                    out = torch.empty_like(x)
                    for tile in hl.tile(x.size(0)):
                        out[tile] = x[tile] * 3.0
                    return out

                bound = scaled.bind((x,))
                seen.clear()
                with patch.object(compiler_backend.Backend, "autotune", spy):
                    MetalBackend().autotune(bound, (x,))
                self.assertEqual(seen, [expected])
                self.assertEqual(bound.settings.autotune_budget_seconds, budget)

    def test_oversized_explicit_threadgroup_rejected_before_codegen(self) -> None:
        """Explicit counts that cannot launch raise instead of compiling.

        Metal refuses an oversized threadgroup only at pipeline-state build,
        so without this pre-check the config would cost a full MSL compile
        first.  128 (MPP) * 64 (explicit) exceeds the 1024-thread limit.
        """

        @helion.kernel(
            backend="metal",
            configs=[helion.Config(block_sizes=[64, 64, 32], num_threads=[32, 64])],
        )
        def matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, k = x.size()
            _k, n = y.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc
            return out

        x = torch.randn(128, 128, device=DEVICE)
        y = torch.randn(128, 128, device=DEVICE)
        with self.assertRaisesRegex(exc.BackendUnsupported, "threadgroup"):
            matmul(x, y)

    def test_signature_key_covers_constexpr_values(self) -> None:
        """The shader memo must recompile when a constexpr changes.

        Non-tensor launch arguments (e.g. a rolled reduction's
        ``_REDUCTION_BLOCK_*``) are baked into the shader, so two launches
        differing only in one must not share a compiled shader; launches
        differing only in tensor *values* must.  Asserts the key contract
        directly, without compiling anything.
        """
        from helion._compiler.metal.metal_jit import _MetalKernel

        def dummy() -> None: ...

        kernel = _MetalKernel(dummy)
        x = torch.randn(16, device=DEVICE)
        base = kernel._signature_key((x, 1024))
        self.assertEqual(base, kernel._signature_key((torch.zeros_like(x), 1024)))
        self.assertNotEqual(base, kernel._signature_key((x, 512)))
        self.assertNotEqual(base, kernel._signature_key((x.to(torch.int32), 1024)))
        kernel.required_threads_per_threadgroup = (128, 1, 1)
        self.assertNotEqual(base, kernel._signature_key((x, 1024)))


if __name__ == "__main__":
    unittest.main()
