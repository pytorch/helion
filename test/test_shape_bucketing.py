from __future__ import annotations

import unittest

import torch

from helion._testing import DEVICE
from helion._testing import RefEagerTestBase
from helion._testing import TestCase
from helion._testing import skipIfCpu
from helion._testing import skipIfNotCUDA
from helion._testing import skipIfRefEager
import helion.language as hl
from helion.runtime.kernel import kernel
from helion.runtime.settings import Settings


# =============================================================================
# Kernel definitions
# =============================================================================


def pointwise_add_kernel(x: torch.Tensor, out: torch.Tensor) -> None:
    """Simple pointwise kernel: out = x + 1.0"""
    for tile in hl.tile(x.size()):
        out[tile] = x[tile] + 1.0


def reduction_sum_kernel(x: torch.Tensor) -> torch.Tensor:
    """Reduction kernel: sum along last dimension."""
    out = x.new_empty([x.size(0)])
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile, :].sum(-1)
    return out


# =============================================================================
# Test class
# =============================================================================


@skipIfCpu("needs to be debugged")
class TestShapeBucketing(RefEagerTestBase, TestCase):
    maxDiff = 16384

    # =========================================================================
    # static_shapes="none" tests - same code for size=1 and size>1
    # =========================================================================

    @skipIfRefEager("code generation not relevant in ref eager mode")
    @skipIfNotCUDA()
    def test_none_pointwise_2d_vary_dim0(self) -> None:
        """none mode, pointwise, 2D: vary dim0 (1->M). Same code for both."""
        k = kernel(pointwise_add_kernel, settings=Settings(static_shapes="none", autotune_effort="none"))

        # Compile with size=1 first
        x1 = torch.randn(1, 16, device=DEVICE, dtype=torch.float32)
        y1 = torch.empty_like(x1)
        k(x1, y1)
        torch.testing.assert_close(y1, x1 + 1.0, rtol=1e-4, atol=1e-4)

        # Then run with size>1
        x2 = torch.randn(4, 16, device=DEVICE, dtype=torch.float32)
        y2 = torch.empty_like(x2)
        k(x2, y2)
        torch.testing.assert_close(y2, x2 + 1.0, rtol=1e-4, atol=1e-4)

        # Verify same code is used (cache has single entry)
        bound = k.bind((x1, y1))
        self.assertEqual(len(bound._compile_cache), 1)

        # Journal the generated code (one entry - same code for both sizes)
        self.assertExpectedJournal(bound.to_triton_code())

    @skipIfRefEager("code generation not relevant in ref eager mode")
    @skipIfNotCUDA()
    def test_none_pointwise_2d_vary_dim1(self) -> None:
        """none mode, pointwise, 2D: vary dim1 (1->K). Same code for both."""
        k = kernel(pointwise_add_kernel, settings=Settings(static_shapes="none", autotune_effort="none"))

        # Compile with size=1 first
        x1 = torch.randn(16, 1, device=DEVICE, dtype=torch.float32)
        y1 = torch.empty_like(x1)
        k(x1, y1)
        torch.testing.assert_close(y1, x1 + 1.0, rtol=1e-4, atol=1e-4)

        # Then run with size>1
        x2 = torch.randn(16, 8, device=DEVICE, dtype=torch.float32)
        y2 = torch.empty_like(x2)
        k(x2, y2)
        torch.testing.assert_close(y2, x2 + 1.0, rtol=1e-4, atol=1e-4)

        # Verify same code is used
        bound = k.bind((x1, y1))
        self.assertEqual(len(bound._compile_cache), 1)

        # Journal the generated code
        self.assertExpectedJournal(bound.to_triton_code())

    @skipIfRefEager("code generation not relevant in ref eager mode")
    @skipIfNotCUDA()
    def test_none_pointwise_2d_vary_both_dims(self) -> None:
        """none mode, pointwise, 2D: both dims start at 1. Same code for both."""
        k = kernel(pointwise_add_kernel, settings=Settings(static_shapes="none", autotune_effort="none"))

        # Compile with (1, 1) first
        x1 = torch.randn(1, 1, device=DEVICE, dtype=torch.float32)
        y1 = torch.empty_like(x1)
        k(x1, y1)
        torch.testing.assert_close(y1, x1 + 1.0, rtol=1e-4, atol=1e-4)

        # Then run with larger sizes
        x2 = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        y2 = torch.empty_like(x2)
        k(x2, y2)
        torch.testing.assert_close(y2, x2 + 1.0, rtol=1e-4, atol=1e-4)

        # Verify same code is used
        bound = k.bind((x1, y1))
        self.assertEqual(len(bound._compile_cache), 1)

        # Journal the generated code
        self.assertExpectedJournal(bound.to_triton_code())

    @skipIfRefEager("code generation not relevant in ref eager mode")
    @skipIfNotCUDA()
    def test_none_reduction_2d(self) -> None:
        """none mode, reduction, 2D: vary dim0 (1->M). Same code for both."""
        k = kernel(reduction_sum_kernel, settings=Settings(static_shapes="none", autotune_effort="none"))

        # Compile with size=1 first
        x1 = torch.randn(1, 64, device=DEVICE, dtype=torch.float32)
        result1 = k(x1)
        torch.testing.assert_close(result1, x1.sum(-1), rtol=1e-4, atol=1e-4)

        # Then run with size>1
        x2 = torch.randn(4, 64, device=DEVICE, dtype=torch.float32)
        result2 = k(x2)
        torch.testing.assert_close(result2, x2.sum(-1), rtol=1e-4, atol=1e-4)

        # Verify same code is used
        bound = k.bind((x1,))
        self.assertEqual(len(bound._compile_cache), 1)

        # Journal the generated code
        self.assertExpectedJournal(bound.to_triton_code())

    # =========================================================================
    # static_shapes="ones" tests - different code for size=1 vs size>1
    # =========================================================================

    @skipIfRefEager("code generation not relevant in ref eager mode")
    @skipIfNotCUDA()
    def test_ones_pointwise_2d_dim0(self) -> None:
        """ones mode, pointwise, 2D: dim0 varies. Different code for size=1 vs >1."""
        k = kernel(pointwise_add_kernel, settings=Settings(static_shapes="ones", autotune_effort="none"))

        x1 = torch.randn(1, 16, device=DEVICE, dtype=torch.float32)
        x2 = torch.randn(4, 16, device=DEVICE, dtype=torch.float32)
        y1 = torch.empty_like(x1)
        y2 = torch.empty_like(x2)

        # Compile with size=1 first
        k(x1, y1)
        torch.testing.assert_close(y1, x1 + 1.0, rtol=1e-4, atol=1e-4)

        # Journal the size=1 specialized code
        bound1 = k.bind((x1, y1))
        self.assertExpectedJournal(bound1.to_triton_code())

        # Run with size>1
        k(x2, y2)
        torch.testing.assert_close(y2, x2 + 1.0, rtol=1e-4, atol=1e-4)

        # Verify different code is used
        bound2 = k.bind((x2, y2))
        self.assertIsNot(bound1, bound2)

        # Journal the size>1 code
        self.assertExpectedJournal(bound2.to_triton_code())

    @skipIfRefEager("code generation not relevant in ref eager mode")
    @skipIfNotCUDA()
    def test_ones_pointwise_2d_dim1(self) -> None:
        """ones mode, pointwise, 2D: dim1 varies. Different code for size=1 vs >1."""
        k = kernel(pointwise_add_kernel, settings=Settings(static_shapes="ones", autotune_effort="none"))

        x1 = torch.randn(16, 1, device=DEVICE, dtype=torch.float32)
        x2 = torch.randn(16, 8, device=DEVICE, dtype=torch.float32)
        y1 = torch.empty_like(x1)
        y2 = torch.empty_like(x2)

        # Compile with size=1 first
        k(x1, y1)
        torch.testing.assert_close(y1, x1 + 1.0, rtol=1e-4, atol=1e-4)

        # Journal the size=1 specialized code
        bound1 = k.bind((x1, y1))
        self.assertExpectedJournal(bound1.to_triton_code())

        # Run with size>1
        k(x2, y2)
        torch.testing.assert_close(y2, x2 + 1.0, rtol=1e-4, atol=1e-4)

        # Verify different code is used
        bound2 = k.bind((x2, y2))
        self.assertIsNot(bound1, bound2)

        # Journal the size>1 code
        self.assertExpectedJournal(bound2.to_triton_code())

    @skipIfRefEager("code generation not relevant in ref eager mode")
    @skipIfNotCUDA()
    def test_ones_reduction_2d(self) -> None:
        """ones mode, reduction, 2D: dim0 varies. Different code for size=1 vs >1."""
        k = kernel(reduction_sum_kernel, settings=Settings(static_shapes="ones", autotune_effort="none"))

        x1 = torch.randn(1, 64, device=DEVICE, dtype=torch.float32)
        x2 = torch.randn(4, 64, device=DEVICE, dtype=torch.float32)

        # Compile with size=1 first
        result1 = k(x1)
        torch.testing.assert_close(result1, x1.sum(-1), rtol=1e-4, atol=1e-4)

        # Journal the size=1 specialized code
        bound1 = k.bind((x1,))
        self.assertExpectedJournal(bound1.to_triton_code())

        # Run with size>1
        result2 = k(x2)
        torch.testing.assert_close(result2, x2.sum(-1), rtol=1e-4, atol=1e-4)

        # Verify different code is used
        bound2 = k.bind((x2,))
        self.assertIsNot(bound1, bound2)

        # Journal the size>1 code
        self.assertExpectedJournal(bound2.to_triton_code())

    # =========================================================================
    # static_shapes="all" tests - unique code for each exact size
    # =========================================================================

    @skipIfRefEager("code generation not relevant in ref eager mode")
    @skipIfNotCUDA()
    def test_all_pointwise_2d(self) -> None:
        """all mode, pointwise, 2D: unique code for each exact size."""
        k = kernel(pointwise_add_kernel, settings=Settings(static_shapes="all", autotune_effort="none"))

        x1 = torch.randn(1, 16, device=DEVICE, dtype=torch.float32)
        x2 = torch.randn(4, 16, device=DEVICE, dtype=torch.float32)
        y1 = torch.empty_like(x1)
        y2 = torch.empty_like(x2)

        # Compile with size=1 first
        k(x1, y1)
        torch.testing.assert_close(y1, x1 + 1.0, rtol=1e-4, atol=1e-4)

        # Journal the size=1 code
        bound1 = k.bind((x1, y1))
        self.assertExpectedJournal(bound1.to_triton_code())

        # Run with size>1
        k(x2, y2)
        torch.testing.assert_close(y2, x2 + 1.0, rtol=1e-4, atol=1e-4)

        # Verify different code is used
        bound2 = k.bind((x2, y2))
        self.assertIsNot(bound1, bound2)

        # Journal the size=4 code
        self.assertExpectedJournal(bound2.to_triton_code())

    @skipIfRefEager("code generation not relevant in ref eager mode")
    @skipIfNotCUDA()
    def test_all_reduction_2d(self) -> None:
        """all mode, reduction, 2D: unique code for each exact size."""
        k = kernel(reduction_sum_kernel, settings=Settings(static_shapes="all", autotune_effort="none"))

        x1 = torch.randn(1, 64, device=DEVICE, dtype=torch.float32)
        x2 = torch.randn(4, 64, device=DEVICE, dtype=torch.float32)

        # Compile with size=1 first
        result1 = k(x1)
        torch.testing.assert_close(result1, x1.sum(-1), rtol=1e-4, atol=1e-4)

        # Journal the size=1 code
        bound1 = k.bind((x1,))
        self.assertExpectedJournal(bound1.to_triton_code())

        # Run with size>1
        result2 = k(x2)
        torch.testing.assert_close(result2, x2.sum(-1), rtol=1e-4, atol=1e-4)

        # Verify different code is used
        bound2 = k.bind((x2,))
        self.assertIsNot(bound1, bound2)

        # Journal the size=4 code
        self.assertExpectedJournal(bound2.to_triton_code())

    # =========================================================================
    # Backward compatibility tests (True/False -> all/ones)
    # =========================================================================

    @skipIfRefEager("specialization keys not relevant in ref eager mode")
    def test_backward_compat_bool_to_string_modes(self) -> None:
        """Test backward compatibility: True maps to 'all', False maps to 'ones'."""
        t1 = torch.empty(1, 3)
        t2 = torch.empty(2, 3)
        t3 = torch.empty(3, 3)

        def dummy(x: torch.Tensor) -> torch.Tensor:
            return x

        # True -> 'all' mode: verify normalization and behavior
        settings_all = Settings(static_shapes=True)
        self.assertEqual(settings_all.static_shapes, "all")  # normalized to string
        k_all = kernel(dummy, settings=settings_all)
        key_all_2 = k_all.specialization_key([t2])
        key_all_3 = k_all.specialization_key([t3])
        self.assertNotEqual(key_all_2, key_all_3)  # each exact size is distinct

        # False -> 'ones' mode: verify normalization and behavior
        settings_ones = Settings(static_shapes=False)
        self.assertEqual(settings_ones.static_shapes, "ones")  # normalized to string
        k_ones = kernel(dummy, settings=settings_ones)
        key_ones_1 = k_ones.specialization_key([t1])
        key_ones_2 = k_ones.specialization_key([t2])
        key_ones_3 = k_ones.specialization_key([t3])
        self.assertNotEqual(key_ones_1, key_ones_2)  # 1 is distinct from >=2
        self.assertEqual(key_ones_2, key_ones_3)  # but 2 and 3 are same


# =============================================================================
# Test for reduction with size-1 dimension bug
# =============================================================================


def reduction_over_dim1_kernel(x: torch.Tensor) -> torch.Tensor:
    """Reduction kernel that sums along the last dimension."""
    out = x.new_empty([x.size(0)])
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile, :].sum(-1)
    return out


@skipIfCpu("needs to be debugged")
class TestReductionSize1Bug(RefEagerTestBase, TestCase):
    """Test that reduction over size-1 dimensions works correctly."""

    @skipIfRefEager("code generation not relevant in ref eager mode")
    @skipIfNotCUDA()
    def test_reduction_over_size1_dim(self) -> None:
        """Regression test: reduction over a size-1 dimension should work.

        When the reduction dimension is 1, the sum is a no-op but the shape
        should still be properly squeezed from [M, 1] to [M].
        """
        for mode in ["none", "ones", "all"]:
            with self.subTest(static_shapes=mode):
                k = kernel(
                    reduction_over_dim1_kernel,
                    settings=Settings(static_shapes=mode, autotune_effort="none"),
                )

                # Test with n=1 (reduction dimension is 1)
                x = torch.randn(32, 1, device=DEVICE, dtype=torch.float32)
                result = k(x)
                expected = x.sum(-1)

                torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)
                self.assertEqual(result.shape, expected.shape)


# =============================================================================
# Test for softmax with size-1 tile dimension bug
# =============================================================================


def softmax_two_pass_kernel(x: torch.Tensor) -> torch.Tensor:
    """Numerically optimized softmax in two passes - from examples/softmax.py.

    This kernel has nested hl.tile loops and reduces over the inner dimension.
    When n=1, the inner tile dimension gets eliminated, causing torch.amax(values, dim=1)
    to fail because values becomes 1D instead of 2D.
    """
    m, n = x.size()
    out = torch.empty_like(x)
    block_size_m = hl.register_block_size(m)
    block_size_n = hl.register_block_size(n)
    for tile_m in hl.tile(m, block_size=block_size_m):
        mi = hl.full([tile_m], float("-inf"), dtype=torch.float32)
        di = hl.zeros([tile_m], dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=block_size_n):
            values = x[tile_m, tile_n]
            local_amax = torch.amax(values, dim=1)
            mi_next = torch.maximum(mi, local_amax)
            di = di * torch.exp(mi - mi_next) + torch.exp(
                values - mi_next[:, None]
            ).sum(dim=1)
            mi = mi_next
        for tile_n in hl.tile(n, block_size=block_size_n):
            values = x[tile_m, tile_n]
            out[tile_m, tile_n] = torch.exp(values - mi[:, None]) / di[:, None]
    return out


@skipIfCpu("needs to be debugged")
class TestSoftmaxSize1TileBug(RefEagerTestBase, TestCase):
    """Test that softmax with nested tiles handles size-1 dimensions correctly.

    Regression test for the bug where, when the inner tile dimension was 1, the tile
    dimension got eliminated from the values tensor, causing subsequent operations
    like torch.amax(values, dim=1) to fail with "Dimension out of range".
    """

    @skipIfRefEager("code generation not relevant in ref eager mode")
    @skipIfNotCUDA()
    def test_softmax_two_pass_with_n1(self) -> None:
        """Regression test: softmax_two_pass should work when n=1.

        When n=1 (the inner reduction dimension), the nested tile loop should
        still produce 2D tensors for proper reduction operations.
        """
        for mode in ["none", "ones", "all"]:
            with self.subTest(static_shapes=mode):
                k = kernel(
                    softmax_two_pass_kernel,
                    settings=Settings(static_shapes=mode, autotune_effort="none"),
                )

                # Test with n=1 (inner tile dimension is 1)
                x = torch.randn(32, 1, device=DEVICE, dtype=torch.float32)
                result = k(x)
                expected = torch.nn.functional.softmax(x, dim=-1)

                torch.testing.assert_close(result, expected, rtol=1e-3, atol=1e-3)
                self.assertEqual(result.shape, expected.shape)


# =============================================================================
# Example configuration for static_shapes mode testing
# =============================================================================

# Shape variations to test 1-ness: (description, m, n)
ALL_SHAPES = [
    ("dim0=1", 1, 64),
    ("dim1=1", 32, 1),
    ("both=1", 1, 1),
    ("normal", 32, 64),
]

# Each example config: (example_name, fn_name, input_factory_with_shapes, reference_fn, shape_variations)
# input_factory_with_shapes: callable(m, n) -> tuple of args (m, n are dimensions)
# reference_fn: callable(*args) -> expected output
# shape_variations: list of (description, m, n) tuples to test
EXAMPLE_CONFIGS_WITH_SHAPES: list[tuple[str, str | None, object, object, list[tuple[str, int, int]]]] = [
    # Simple pointwise operations - support all shapes
    (
        "add",
        None,
        lambda m, n: (
            torch.randn(m, n, device=DEVICE, dtype=torch.float16),
            torch.randn(m, n, device=DEVICE, dtype=torch.float16),
        ),
        lambda x, y: torch.add(x, y),
        ALL_SHAPES,
    ),
    (
        "exp",
        "exp_fwd",
        lambda m, n: (torch.randn(m, n, device=DEVICE, dtype=torch.float16),),
        lambda x: torch.exp(x),
        ALL_SHAPES,
    ),
    # Reduction operations - test all shapes including n=1 edge case
    (
        "sum",
        "sum_kernel",
        lambda m, n: (torch.randn(m, n, device=DEVICE, dtype=torch.float32),),
        lambda x: x.sum(-1),
        ALL_SHAPES,
    ),
    (
        "softmax",
        "softmax_two_pass",
        lambda m, n: (torch.randn(m, n, device=DEVICE, dtype=torch.float32),),
        lambda x: torch.nn.functional.softmax(x, dim=-1),
        ALL_SHAPES,
    ),
    (
        "cross_entropy",
        "cross_entropy",
        lambda m, n: (
            torch.randn(m, n, device=DEVICE, dtype=torch.float32),
            torch.randint(0, n, (m,), device=DEVICE, dtype=torch.long),
        ),
        lambda logits, labels: torch.nn.functional.cross_entropy(logits, labels),
        ALL_SHAPES,
    ),
    # Normalization operations
    (
        "rms_norm",
        "rms_norm_fwd",
        lambda m, n: (
            torch.randn(m, n, device=DEVICE, dtype=torch.float16),
            torch.randn(n, device=DEVICE, dtype=torch.float16),
            1e-5,
        ),
        lambda x, w, eps: torch.nn.functional.rms_norm(x, (x.shape[-1],), w, eps),
        ALL_SHAPES,
    ),
    # Embedding lookup - support all shapes
    (
        "embedding",
        "embedding",
        lambda m, n: (
            torch.randint(0, 128, (m, n), device=DEVICE, dtype=torch.int32),
            torch.randn(128, 64, device=DEVICE, dtype=torch.float16),
        ),
        lambda x, w: torch.nn.functional.embedding(x.long(), w),
        ALL_SHAPES,
    ),
    # Concatenation - support all shapes
    (
        "concatenate",
        "concat2d_dim1",
        lambda m, n: (
            torch.randn(m, n, device=DEVICE, dtype=torch.float32),
            torch.randn(m, n + 8, device=DEVICE, dtype=torch.float32),
        ),
        lambda x, y: torch.cat([x, y], dim=1),
        ALL_SHAPES,
    ),
    # GEGLU activation
    (
        "geglu",
        "geglu",
        lambda m, n: (
            torch.randn(m, n, device=DEVICE, dtype=torch.float16),
            torch.randn(m, n, device=DEVICE, dtype=torch.float16),
        ),
        lambda x1, x2: torch.nn.functional.gelu(x1, approximate="tanh") * x2,
        ALL_SHAPES,
    ),
    # SwiGLU activation
    (
        "swiglu",
        "swiglu_fwd",
        lambda m, n: (
            torch.randn(m, n, device=DEVICE, dtype=torch.float16),
            torch.randn(m, n, device=DEVICE, dtype=torch.float16),
        ),
        lambda x1, x2: torch.nn.functional.silu(x1) * x2,
        ALL_SHAPES,
    ),
    # Additional softmax variants - test all shapes
    (
        "softmax",
        "softmax",  # Simple wrapper around torch.nn.functional.softmax
        lambda m, n: (torch.randn(m, n, device=DEVICE, dtype=torch.float32),),
        lambda x: torch.nn.functional.softmax(x, dim=-1),
        ALL_SHAPES,
    ),
    (
        "softmax",
        "softmax_decomposed",  # Decomposed softmax with explicit max, exp, normalize
        lambda m, n: (torch.randn(m, n, device=DEVICE, dtype=torch.float32),),
        lambda x: torch.nn.functional.softmax(x, dim=-1),
        ALL_SHAPES,
    ),
    # Long sum - reduction along last dimension
    (
        "long_sum",
        "longsum",
        lambda m, n: (torch.randn(m, n, device=DEVICE, dtype=torch.float32),),
        lambda x: x.sum(-1),
        ALL_SHAPES,
    ),
    # Welford layer norm - uses Welford's algorithm for mean/variance
    (
        "welford",
        "welford",
        lambda m, n: (
            torch.randn(n, device=DEVICE, dtype=torch.float32),  # weight
            torch.randn(n, device=DEVICE, dtype=torch.float32),  # bias
            torch.randn(m, n, device=DEVICE, dtype=torch.float32),  # x
            1e-5,  # eps
        ),
        lambda w, b, x, eps: torch.nn.functional.layer_norm(x, (x.shape[-1],), w, b, eps),
        ALL_SHAPES,
    ),
]


@skipIfCpu("needs to be debugged")
class TestExamplesStaticShapesModes(RefEagerTestBase, TestCase):
    """Test that various examples work correctly with all static_shapes modes."""

    @skipIfRefEager("code generation not relevant in ref eager mode")
    @skipIfNotCUDA()
    def test_examples_with_all_static_shapes_modes(self) -> None:
        """Test representative examples with all static_shapes modes and shape variations."""
        from helion._testing import EXAMPLES_DIR
        from helion._testing import import_path

        static_shapes_modes = ["none", "ones", "all"]

        for example_name, fn_name, input_factory, reference_fn, shapes in EXAMPLE_CONFIGS_WITH_SHAPES:
            for mode in static_shapes_modes:
                for shape_desc, m, n in shapes:
                    with self.subTest(example=example_name, static_shapes=mode, shape=shape_desc):
                        # Import the kernel fresh to avoid state pollution
                        mod = import_path(EXAMPLES_DIR / f"{example_name}.py")
                        kernel_fn = getattr(mod, fn_name or example_name)

                        # Clear any hardcoded configs and cached bound kernels from the kernel
                        # This allows the test to use dynamic configs based on input shapes
                        kernel_fn.configs = []
                        kernel_fn._bound_kernels = {}

                        # Set the static_shapes mode and disable autotuning for faster tests
                        kernel_fn.settings.static_shapes = mode
                        kernel_fn.settings.autotune_effort = "none"

                        # Create test inputs with the specified shapes and run
                        args = input_factory(m, n)
                        result = kernel_fn(*args)

                        # Compare with reference
                        expected = reference_fn(*args)

                        # Handle tuple results (some kernels return multiple values)
                        if isinstance(result, tuple) and not isinstance(expected, tuple):
                            result = result[0]

                        torch.testing.assert_close(
                            result.to(torch.float32),
                            expected.to(torch.float32),
                            rtol=1e-2,
                            atol=1e-1,
                        )


if __name__ == "__main__":
    unittest.main()
