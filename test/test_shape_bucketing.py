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
    # Helper methods for common test patterns
    # =========================================================================

    def _run_pointwise(
        self, k: object, shapes: tuple[int, ...]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run pointwise kernel and verify correctness."""
        x = torch.randn(*shapes, device=DEVICE, dtype=torch.float32)
        y = torch.empty_like(x)
        k(x, y)
        torch.testing.assert_close(y, x + 1.0, rtol=1e-4, atol=1e-4)
        return x, y

    def _run_reduction(self, k: object, shapes: tuple[int, ...]) -> torch.Tensor:
        """Run reduction kernel and verify correctness."""
        x = torch.randn(*shapes, device=DEVICE, dtype=torch.float32)
        result = k(x)
        torch.testing.assert_close(result, x.sum(-1), rtol=1e-4, atol=1e-4)
        return x

    # =========================================================================
    # Comprehensive tests (all mode/shape combinations with journals)
    # =========================================================================

    @skipIfRefEager("code generation not relevant in ref eager mode")
    @skipIfNotCUDA()
    def test_pointwise_all_modes_shapes(self) -> None:
        """Test pointwise kernel with all modes and shape variations, with journals."""
        # (desc, shapes_1, shapes_2, has_different_1ness)
        # has_different_1ness: whether shapes_1 and shapes_2 differ in whether any dim is 1
        shape_variations = [
            ("dim0=1", (1, 16), (4, 16), True),  # 1->4 changes 1-ness
            ("dim1=1", (16, 1), (16, 8), True),  # 1->8 changes 1-ness
            ("both=1", (1, 1), (4, 8), True),  # both 1s -> no 1s
        ]
        for mode in ["none", "ones", "all"]:
            for desc, shapes_1, shapes_2, has_different_1ness in shape_variations:
                with self.subTest(mode=mode, shapes=desc):
                    k = kernel(
                        pointwise_add_kernel,
                        settings=Settings(static_shapes=mode, autotune_effort="none"),
                    )
                    # Run with shapes_1
                    x1, y1 = self._run_pointwise(k, shapes_1)
                    bound1 = k.bind((x1, y1))

                    # Run with shapes_2
                    x2, y2 = self._run_pointwise(k, shapes_2)
                    bound2 = k.bind((x2, y2))

                    # Verify cache behavior and journal based on mode
                    if mode == "none":
                        # Same code for all sizes
                        self.assertEqual(len(bound1._compile_cache), 1)
                        self.assertExpectedJournal(bound1.to_triton_code())
                    elif mode == "ones":
                        # Only different code if 1-ness differs
                        if has_different_1ness:
                            self.assertIsNot(bound1, bound2)
                            self.assertExpectedJournal(bound1.to_triton_code())
                            self.assertExpectedJournal(bound2.to_triton_code())
                        else:
                            # Same code since 1-ness is the same
                            self.assertIs(bound1, bound2)
                            self.assertExpectedJournal(bound1.to_triton_code())
                    else:  # mode == "all"
                        # Different code for each exact size
                        self.assertIsNot(bound1, bound2)
                        self.assertExpectedJournal(bound1.to_triton_code())
                        self.assertExpectedJournal(bound2.to_triton_code())

    @skipIfRefEager("code generation not relevant in ref eager mode")
    @skipIfNotCUDA()
    def test_reduction_all_modes_shapes(self) -> None:
        """Test reduction kernel with all modes and shape variations, with journals.

        Note: We only vary the non-reduction dimension (dim0) because varying the
        reduction dimension would require different configs/code paths, which is
        not what we're testing here.
        """
        # (desc, shapes_1, shapes_2, has_different_1ness)
        # has_different_1ness: whether shapes_1 and shapes_2 differ in whether any dim is 1
        shape_variations = [
            ("dim0=1", (1, 64), (4, 64), True),  # 1->4 changes 1-ness
            ("dim0=2", (2, 64), (8, 64), False),  # 2->8, both non-1
        ]
        for mode in ["none", "ones", "all"]:
            for desc, shapes_1, shapes_2, has_different_1ness in shape_variations:
                with self.subTest(mode=mode, shapes=desc):
                    k = kernel(
                        reduction_sum_kernel,
                        settings=Settings(static_shapes=mode, autotune_effort="none"),
                    )
                    # Run with shapes_1
                    x1 = self._run_reduction(k, shapes_1)
                    bound1 = k.bind((x1,))

                    # Run with shapes_2
                    x2 = self._run_reduction(k, shapes_2)
                    bound2 = k.bind((x2,))

                    # Verify cache behavior and journal based on mode
                    if mode == "none":
                        # Same code for all sizes
                        self.assertEqual(len(bound1._compile_cache), 1)
                        self.assertExpectedJournal(bound1.to_triton_code())
                    elif mode == "ones":
                        # Only different code if 1-ness differs
                        if has_different_1ness:
                            self.assertIsNot(bound1, bound2)
                            self.assertExpectedJournal(bound1.to_triton_code())
                            self.assertExpectedJournal(bound2.to_triton_code())
                        else:
                            # Same code since 1-ness is the same
                            self.assertIs(bound1, bound2)
                            self.assertExpectedJournal(bound1.to_triton_code())
                    else:  # mode == "all"
                        # Different code for each exact size
                        self.assertIsNot(bound1, bound2)
                        self.assertExpectedJournal(bound1.to_triton_code())
                        self.assertExpectedJournal(bound2.to_triton_code())

    # =========================================================================
    # Regression tests for size-1 dimension bugs
    # =========================================================================

    @skipIfRefEager("code generation not relevant in ref eager mode")
    @skipIfNotCUDA()
    def test_reduction_over_size1_dim(self) -> None:
        """Regression test: reduction over a size-1 dimension should work.

        When the reduction dimension is 1, the sum is a no-op but the shape
        should still be properly squeezed from [M, 1] to [M].
        """

        def reduction_over_dim1_kernel(x: torch.Tensor) -> torch.Tensor:
            """Reduction kernel that sums along the last dimension."""
            out = x.new_empty([x.size(0)])
            for tile in hl.tile(x.size(0)):
                out[tile] = x[tile, :].sum(-1)
            return out

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

    @skipIfRefEager("code generation not relevant in ref eager mode")
    @skipIfNotCUDA()
    def test_softmax_two_pass_with_n1(self) -> None:
        """Regression test: softmax_two_pass should work when n=1.

        When n=1 (the inner reduction dimension), the nested tile loop should
        still produce 2D tensors for proper reduction operations.
        """

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

    @skipIfRefEager("code generation not relevant in ref eager mode")
    @skipIfNotCUDA()
    def test_view_flatten_symbolic_shapes_dim1_is_1(self) -> None:
        """Reproduce: .view(-1) fails with static_shapes='none' when n=1.

        This test reproduces the error from cross_entropy.py where
        `logits_flat = logits.view(-1)` fails because the tensor has
        symbolic shape [u0, u1] and symbolic strides (s3, 1), which
        prevents PyTorch from verifying the view is safe.

        The bug specifically occurs when the second dimension n=1.
        """
        from helion._testing import EXAMPLES_DIR
        from helion._testing import import_path

        # Import cross_entropy example
        mod = import_path(EXAMPLES_DIR / "cross_entropy.py")
        cross_entropy = mod.cross_entropy

        # Clear configs and set static_shapes='none'
        cross_entropy.configs = []
        cross_entropy._bound_kernels = {}
        cross_entropy.settings.static_shapes = "none"
        cross_entropy.settings.autotune_effort = "none"

        # Use shape (32, 1) - the n=1 is what triggers the bug
        m, n = 32, 1
        logits = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
        labels = torch.randint(0, n, (m,), device=DEVICE, dtype=torch.long)

        result = cross_entropy(logits, labels)
        expected = torch.nn.functional.cross_entropy(logits, labels)

        torch.testing.assert_close(result, expected, rtol=1e-2, atol=1e-2)

    @skipIfRefEager("code generation not relevant in ref eager mode")
    @skipIfNotCUDA()
    def test_nested_tile_with_python_int_size(self) -> None:
        """Nested tile loop should work when dimension size is a Python int.

        Tests with m=1 shape which triggers the bug when static_shapes='ones'
        causes the dimension size to be passed as a Python int to tl.range.
        """

        def nested_tile_kernel(x: torch.Tensor) -> torch.Tensor:
            """Nested tile kernel with registered block sizes.

            When static_shapes mode passes Python ints as dimension sizes,
            the generated code uses `n.to(tl.int32)` in tl.range. This fails
            because Python ints don't have a .to() method - tl.cast() is needed.

            Uses registered block sizes like softmax_two_pass to trigger the
            specific code path where tl.range receives a Python int bound.
            """
            m, n = x.size()
            out = torch.empty_like(x)
            block_size_m = hl.register_block_size(m)
            block_size_n = hl.register_block_size(n)
            for tile_m in hl.tile(m, block_size=block_size_m):
                for tile_n in hl.tile(n, block_size=block_size_n):
                    out[tile_m, tile_n] = x[tile_m, tile_n] + 1.0
            return out

        for mode in ["none", "ones", "all"]:
            # Test with m=1 which triggers block_size=1 specialization
            for m, n in [(1, 64), (32, 1), (1, 1), (32, 64)]:
                with self.subTest(static_shapes=mode, m=m, n=n):
                    k = kernel(
                        nested_tile_kernel,
                        settings=Settings(static_shapes=mode, autotune_effort="none"),
                    )

                    x = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
                    result = k(x)
                    expected = x + 1.0

                    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)

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
EXAMPLE_CONFIGS_WITH_SHAPES: list[
    tuple[str, str | None, object, object, list[tuple[str, int, int]]]
] = [
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
        lambda w, b, x, eps: torch.nn.functional.layer_norm(
            x, (x.shape[-1],), w, b, eps
        ),
        ALL_SHAPES,
    ),
    # Long sum variants - reduction along last dimension with different implementations
    (
        "long_sum",
        "longsum_w_red_loop",
        lambda m, n: (torch.randn(m, n, device=DEVICE, dtype=torch.float32),),
        lambda x: x.sum(-1),
        ALL_SHAPES,
    ),
    (
        "long_sum",
        "longsum_manual",
        lambda m, n: (torch.randn(m, n, device=DEVICE, dtype=torch.float32),),
        lambda x: x.sum(-1),
        ALL_SHAPES,
    ),
    # Layer norm forward - normalization over the last dimension
    (
        "layer_norm",
        "layer_norm_fwd",
        lambda m, n: (
            torch.randn(m, n, device=DEVICE, dtype=torch.float16),  # x
            [n],  # normalized_shape
            torch.randn(n, device=DEVICE, dtype=torch.float16),  # weight
            torch.randn(n, device=DEVICE, dtype=torch.float16),  # bias
            1e-5,  # eps
        ),
        lambda x, ns, w, b, eps: torch.nn.functional.layer_norm(x, ns, w, b, eps),
        ALL_SHAPES,
    ),
    # Matrix multiplication - with k dimension fixed
    (
        "matmul",
        "matmul",
        lambda m, n: (
            torch.randn(m, 64, device=DEVICE, dtype=torch.float16),  # x: [m, k]
            torch.randn(64, n, device=DEVICE, dtype=torch.float16),  # y: [k, n]
        ),
        lambda x, y: torch.matmul(x, y),
        ALL_SHAPES,
    ),
    # Matrix multiplication with split-K - higher parallelism for large K
    (
        "matmul_split_k",
        "matmul_split_k",
        lambda m, n: (
            torch.randn(m, 128, device=DEVICE, dtype=torch.float16),  # x: [m, k]
            torch.randn(128, n, device=DEVICE, dtype=torch.float16),  # y: [k, n]
        ),
        lambda x, y: torch.matmul(x, y),
        ALL_SHAPES,
    ),
    # Fused matmul + layer norm - with k dimension fixed
    (
        "matmul_layernorm",
        "matmul_layernorm",
        lambda m, n: (
            torch.randn(m, 64, device=DEVICE, dtype=torch.float16),  # x: [m, k]
            torch.randn(64, n, device=DEVICE, dtype=torch.float16),  # y: [k, n]
            torch.randn(n, device=DEVICE, dtype=torch.float16),  # weight: [n]
            torch.randn(n, device=DEVICE, dtype=torch.float16),  # bias: [n]
        ),
        lambda x, y, w, b: torch.nn.functional.layer_norm(
            torch.matmul(x, y).float(), [y.size(1)], w.float(), b.float()
        ).to(torch.float16),
        ALL_SHAPES,
    ),
    # Squeeze and excitation network forward - with k dimension fixed
    (
        "squeeze_and_excitation_net",
        "squeeze_and_excitation_net_fwd",
        lambda m, n: (
            torch.randn(m, n, device=DEVICE, dtype=torch.float16),  # x: [m, n]
            torch.randn(n, 32, device=DEVICE, dtype=torch.float16),  # a: [n, k]
            torch.randn(32, n, device=DEVICE, dtype=torch.float16),  # b: [k, n]
        ),
        lambda x, a, b: torch.mul(x, torch.sigmoid(torch.relu(x @ a) @ b)),
        ALL_SHAPES,
    ),
    # KL divergence forward - with specific settings
    (
        "kl_div",
        "kl_div_forward",
        lambda m, n: (
            torch.randn(m, n, device=DEVICE, dtype=torch.float32).log_softmax(
                dim=-1
            ),  # y_pred
            torch.randn(m, n, device=DEVICE, dtype=torch.float32).softmax(
                dim=-1
            ),  # y_true
            False,  # log_target
            "batchmean",  # reduction
            1e-10,  # eps
        ),
        lambda y_pred, y_true, log_target, reduction, eps: torch.nn.functional.kl_div(
            y_pred, y_true, reduction=reduction, log_target=log_target
        ),
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

        for (
            example_name,
            fn_name,
            input_factory,
            reference_fn,
            shapes,
        ) in EXAMPLE_CONFIGS_WITH_SHAPES:
            for mode in static_shapes_modes:
                for shape_desc, m, n in shapes:
                    with self.subTest(
                        example=example_name, static_shapes=mode, shape=shape_desc
                    ):
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
                        if isinstance(result, tuple) and not isinstance(
                            expected, tuple
                        ):
                            result = result[0]

                        torch.testing.assert_close(
                            result.to(torch.float32),
                            expected.to(torch.float32),
                            rtol=1e-2,
                            atol=1e-1,
                        )


# =============================================================================
# NOTE: Welford chunk.size(-1) issue
# =============================================================================
# The welford.py example was modified to use `tile.end - tile.begin` instead of
# `chunk.size(-1)` because chunk.size(-1) returns the block_size, not the actual
# number of valid elements in the tile. When n < block_size (e.g., n=1,
# block_size=4), dividing by block_size instead of n gives wrong results.
#
# Example:
# - n=1, block_size=4
# - chunk has 4 elements, but only 1 is valid (others are masked as 0)
# - sum_x = x[0] (correct, since masked elements are 0)
# - mean = sum_x / 4 (WRONG! should be sum_x / 1)
#
# Fix: Use `tile.end - tile.begin` to get actual tile size.
# =============================================================================


if __name__ == "__main__":
    unittest.main()
