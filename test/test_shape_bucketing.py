from __future__ import annotations

import unittest

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import RefEagerTestBase
from helion._testing import TestCase
from helion._testing import skipIfCpu
from helion._testing import skipIfNotCUDA
from helion._testing import skipIfRefEager
import helion.language as hl
from helion.runtime.kernel import kernel
from helion.runtime.settings import Settings


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


@skipIfCpu("needs to be debugged")
class TestShapeBucketing(RefEagerTestBase, TestCase):
    maxDiff = 16384

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
                        # Same BoundKernel object — proves no guard forced recompilation
                        self.assertIs(bound1, bound2)
                        code = bound1.to_triton_code()
                        self.assertExpectedJournal(code)
                        # Shapes should be symbolic, not hardcoded
                        self.assertIn("x_size_", code)
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
                        code1 = bound1.to_triton_code()
                        self.assertExpectedJournal(code1)
                        self.assertExpectedJournal(bound2.to_triton_code())
                        # Shapes should be hardcoded, not symbolic
                        self.assertNotIn("x_size_", code1)

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
                        # Same BoundKernel object — proves no guard forced recompilation
                        self.assertIs(bound1, bound2)
                        code = bound1.to_triton_code()
                        self.assertExpectedJournal(code)
                        # Shapes should be symbolic, not hardcoded
                        self.assertIn("x_size_", code)
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
                        code1 = bound1.to_triton_code()
                        self.assertExpectedJournal(code1)
                        self.assertExpectedJournal(bound2.to_triton_code())
                        # Shapes should be hardcoded, not symbolic
                        self.assertNotIn("x_size_", code1)

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

        @helion.kernel(ignore_warnings=[helion.exc.TensorOperationInWrapper])
        def cross_entropy(
            logits: torch.Tensor,
            labels: torch.Tensor,
        ) -> torch.Tensor:
            n, v = logits.shape
            losses = torch.zeros([n], dtype=logits.dtype, device=logits.device)
            logits_flat = logits.view(-1)
            for tile_n in hl.tile(n):
                labels_tile = labels[tile_n]
                base_indices_tile = tile_n.index * v
                flat_indices = base_indices_tile + labels_tile
                logits_at_target = hl.load(logits_flat, [flat_indices])
                logits_rows = logits[tile_n, :]
                max_logits = torch.amax(logits_rows, dim=-1, keepdim=True)
                shifted = logits_rows - max_logits
                exp_shifted = torch.exp(shifted)
                sum_exp = torch.sum(exp_shifted, dim=-1, keepdim=True)
                log_sum_exp = max_logits.squeeze(-1) + torch.log(sum_exp.squeeze(-1))
                losses[tile_n] = log_sum_exp - logits_at_target
            return losses.mean()

        k = kernel(
            cross_entropy,
            settings=Settings(static_shapes="none", autotune_effort="none"),
        )

        # Use shape (32, 1) - the n=1 is what triggers the bug
        m, n = 32, 1
        logits = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
        labels = torch.randint(0, n, (m,), device=DEVICE, dtype=torch.long)

        result = k(logits, labels)
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

    @skipIfRefEager("code generation not relevant in ref eager mode")
    @skipIfNotCUDA()
    def test_stride_variation_dynamic_modes(self) -> None:
        """Test kernels handle stride variations correctly in dynamic modes.

        In 'ones' and 'none' modes, strides are NOT in the cache key.
        Verify that a kernel compiled for a contiguous tensor produces
        correct results when reused for a non-contiguous (transposed) tensor.
        """
        for mode in ["ones", "none"]:
            with self.subTest(mode=mode, case="pointwise"):
                k = kernel(
                    pointwise_add_kernel,
                    settings=Settings(static_shapes=mode, autotune_effort="none"),
                )
                # Run with contiguous tensor first
                x1 = torch.randn(32, 64, device=DEVICE, dtype=torch.float32)
                y1 = torch.empty_like(x1)
                k(x1, y1)
                torch.testing.assert_close(y1, x1 + 1.0, rtol=1e-4, atol=1e-4)

                # Run with non-contiguous (transposed) tensor of same shape
                x2 = torch.randn(64, 32, device=DEVICE, dtype=torch.float32).T
                self.assertEqual(x2.shape, (32, 64))
                self.assertFalse(x2.is_contiguous())
                y2 = torch.empty(32, 64, device=DEVICE, dtype=torch.float32)
                k(x2, y2)
                torch.testing.assert_close(y2, x2 + 1.0, rtol=1e-4, atol=1e-4)

            with self.subTest(mode=mode, case="reduction"):
                k = kernel(
                    reduction_sum_kernel,
                    settings=Settings(static_shapes=mode, autotune_effort="none"),
                )
                # Run with contiguous tensor first
                x1 = torch.randn(8, 64, device=DEVICE, dtype=torch.float32)
                result1 = k(x1)
                torch.testing.assert_close(result1, x1.sum(-1), rtol=1e-4, atol=1e-4)

                # Run with non-contiguous tensor
                x2 = torch.randn(64, 8, device=DEVICE, dtype=torch.float32).T
                self.assertEqual(x2.shape, (8, 64))
                self.assertFalse(x2.is_contiguous())
                result2 = k(x2)
                torch.testing.assert_close(result2, x2.sum(-1), rtol=1e-4, atol=1e-4)

    @skipIfRefEager("code generation not relevant in ref eager mode")
    @skipIfNotCUDA()
    def test_zero_size_with_bucketing_modes(self) -> None:
        """Test zero-size dimensions work correctly with all bucketing modes.

        The _tensor_key has explicit zero-handling (0 if s == 0 else 2 in
        'none' mode) but these paths were untested with dynamic modes.
        """
        for mode in ["none", "ones", "all"]:
            with self.subTest(mode=mode, case="pointwise_zero_outer"):
                k = kernel(
                    pointwise_add_kernel,
                    settings=Settings(static_shapes=mode, autotune_effort="none"),
                )
                # Zero outer dim → empty result
                x = torch.randn(0, 32, device=DEVICE, dtype=torch.float32)
                y = torch.empty_like(x)
                k(x, y)
                self.assertEqual(y.shape, (0, 32))
                torch.testing.assert_close(y, x + 1.0, rtol=1e-4, atol=1e-4)

            # NOTE: reduction with zero-size reduction dim (e.g. [4, 0]) is not
            # tested because Triton's tl.arange(0, 0) is unsupported.

            with self.subTest(mode=mode, case="zero_distinct_from_nonzero"):
                k = kernel(
                    pointwise_add_kernel,
                    settings=Settings(static_shapes=mode, autotune_effort="none"),
                )
                # A kernel cached for [0, 32] should NOT be reused for [4, 32]
                x0 = torch.randn(0, 32, device=DEVICE, dtype=torch.float32)
                y0 = torch.empty_like(x0)
                k(x0, y0)
                bound0 = k.bind((x0, y0))

                x4 = torch.randn(4, 32, device=DEVICE, dtype=torch.float32)
                y4 = torch.empty_like(x4)
                k(x4, y4)
                bound4 = k.bind((x4, y4))
                self.assertIsNot(bound0, bound4)

    @skipIfRefEager("code generation not relevant in ref eager mode")
    @skipIfNotCUDA()
    def test_mark_static_with_all_modes(self) -> None:
        """Test mark_static works correctly with all static_shapes modes.

        Verifies that mark_static on a dimension causes its value to be
        specialized, that changing the marked dim causes a cache miss, and
        that changing a non-marked dim (same bucket) is a cache hit.
        """
        for mode in ["none", "ones", "all"]:
            with self.subTest(mode=mode):
                k = kernel(
                    pointwise_add_kernel,
                    settings=Settings(static_shapes=mode, autotune_effort="none"),
                )

                # Create tensor and mark dim 0 as static
                x1 = torch.randn(96, 32, device=DEVICE, dtype=torch.float32)
                torch._dynamo.mark_static(x1, 0)
                y1 = torch.empty_like(x1)
                k(x1, y1)
                torch.testing.assert_close(y1, x1 + 1.0, rtol=1e-4, atol=1e-4)
                bound1 = k.bind((x1, y1))

                # Verify marked dim value appears as constant in generated code
                code = bound1.to_triton_code()
                self.assertIn("96", code)

                # Changing marked dim value should cause cache miss
                x2 = torch.randn(128, 32, device=DEVICE, dtype=torch.float32)
                torch._dynamo.mark_static(x2, 0)
                y2 = torch.empty_like(x2)
                k(x2, y2)
                torch.testing.assert_close(y2, x2 + 1.0, rtol=1e-4, atol=1e-4)
                bound2 = k.bind((x2, y2))
                self.assertIsNot(bound1, bound2)  # different marked dim → cache miss

                # In dynamic modes, changing non-marked dim should be cache hit
                if mode in ("ones", "none"):
                    x3 = torch.randn(96, 48, device=DEVICE, dtype=torch.float32)
                    torch._dynamo.mark_static(x3, 0)
                    y3 = torch.empty_like(x3)
                    k(x3, y3)
                    torch.testing.assert_close(y3, x3 + 1.0, rtol=1e-4, atol=1e-4)
                    bound3 = k.bind((x3, y3))
                    self.assertIs(bound1, bound3)  # same bucket → cache hit

    @skipIfRefEager("code generation not relevant in ref eager mode")
    @skipIfNotCUDA()
    def test_none_mode_cross_size_reuse(self) -> None:
        """Test 'none' mode kernel compiled for one size works for other sizes.

        Verifies that generated code uses symbolic shapes (not hardcoded values)
        so a single compiled kernel works correctly across different sizes.
        """
        # Pointwise: compile with one shape, reuse for several others
        k = kernel(
            pointwise_add_kernel,
            settings=Settings(static_shapes="none", autotune_effort="none"),
        )

        shapes = [(1, 16), (5, 16), (1, 32), (7, 3)]
        for shape in shapes:
            with self.subTest(case="pointwise", shape=shape):
                x = torch.randn(*shape, device=DEVICE, dtype=torch.float32)
                y = torch.empty_like(x)
                k(x, y)
                torch.testing.assert_close(y, x + 1.0, rtol=1e-4, atol=1e-4)

        # All non-zero shapes share one BoundKernel in "none" mode
        x_check = torch.randn(7, 3, device=DEVICE, dtype=torch.float32)
        y_check = torch.empty_like(x_check)
        bound = k.bind((x_check, y_check))
        self.assertEqual(len(bound._compile_cache), 1)

        # Reduction: compile with one shape, reuse for another
        k_red = kernel(
            reduction_sum_kernel,
            settings=Settings(static_shapes="none", autotune_effort="none"),
        )

        for shape in [(1, 64), (8, 64)]:
            with self.subTest(case="reduction", shape=shape):
                x = torch.randn(*shape, device=DEVICE, dtype=torch.float32)
                result = k_red(x)
                torch.testing.assert_close(result, x.sum(-1), rtol=1e-4, atol=1e-4)

        x_red = torch.randn(8, 64, device=DEVICE, dtype=torch.float32)
        bound_red = k_red.bind((x_red,))
        self.assertEqual(len(bound_red._compile_cache), 1)

    @skipIfRefEager("code generation not relevant in ref eager mode")
    @skipIfNotCUDA()
    def test_zero_vs_nonzero_bucket_separation(self) -> None:
        """Test that zero-size and non-zero-size tensors use separate buckets.

        Verifies that _tensor_key distinguishes 0 from non-zero in all modes,
        preventing a kernel compiled for a zero-size dim from being reused
        for a non-zero size.
        """
        for mode in ["none", "ones", "all"]:
            with self.subTest(mode=mode):
                k = kernel(
                    pointwise_add_kernel,
                    settings=Settings(static_shapes=mode, autotune_effort="none"),
                )
                # Run with zero-size dim
                x0 = torch.randn(0, 16, device=DEVICE, dtype=torch.float32)
                y0 = torch.empty_like(x0)
                k(x0, y0)
                bound0 = k.bind((x0, y0))

                # Run with non-zero-size dim
                x4 = torch.randn(4, 16, device=DEVICE, dtype=torch.float32)
                y4 = torch.empty_like(x4)
                k(x4, y4)
                torch.testing.assert_close(y4, x4 + 1.0, rtol=1e-4, atol=1e-4)
                bound4 = k.bind((x4, y4))

                # Must be different bound kernels
                self.assertIsNot(bound0, bound4)

    @skipIfRefEager("code generation not relevant in ref eager mode")
    @skipIfNotCUDA()
    def test_nested_tile_none_mode_reuse(self) -> None:
        """Verify 'none' mode kernel with 2D tiling reuses across size-1 changes.

        This exercises the known_equal() fix in BlockedSubscriptIndexing
        (indexing_strategy.py) — if guards were added when checking size==1,
        the kernel compiled for m=1 would NOT be reusable for m=4.
        """

        def nested_tile_kernel(x: torch.Tensor) -> torch.Tensor:
            m, n = x.size()
            out = torch.empty_like(x)
            block_size_m = hl.register_block_size(m)
            block_size_n = hl.register_block_size(n)
            for tile_m in hl.tile(m, block_size=block_size_m):
                for tile_n in hl.tile(n, block_size=block_size_n):
                    out[tile_m, tile_n] = x[tile_m, tile_n] + 1.0
            return out

        k = kernel(
            nested_tile_kernel,
            settings=Settings(static_shapes="none", autotune_effort="none"),
        )

        # Compile with m=1 first (critical: size-1 triggers the guard risk)
        x1 = torch.randn(1, 64, device=DEVICE, dtype=torch.float32)
        result1 = k(x1)
        torch.testing.assert_close(result1, x1 + 1.0, rtol=1e-5, atol=1e-5)
        bound1 = k.bind((x1,))

        # Reuse for m=4 — must be same BoundKernel (proves no guard was added)
        x4 = torch.randn(4, 64, device=DEVICE, dtype=torch.float32)
        result4 = k(x4)
        torch.testing.assert_close(result4, x4 + 1.0, rtol=1e-5, atol=1e-5)
        bound4 = k.bind((x4,))

        self.assertIs(bound1, bound4)

    @skipIfRefEager("code generation not relevant in ref eager mode")
    @skipIfNotCUDA()
    def test_ones_vs_none_bucketing_contrast(self) -> None:
        """Directly contrast 'ones' vs 'none' mode for the same size-1 input.

        In 'ones': size=1 and size=3 are in DIFFERENT buckets, producing
        different compiled kernels with size-1-specific optimizations.

        In 'none': size=1 and size=3 are in the SAME bucket, sharing one
        compiled kernel with fully symbolic code.
        """
        shapes_1 = (1, 16)
        shapes_other = (3, 16)

        # --- "ones" mode: size=1 produces specialized code ---
        k_ones = kernel(
            pointwise_add_kernel,
            settings=Settings(static_shapes="ones", autotune_effort="none"),
        )
        x1_o, y1_o = self._run_pointwise(k_ones, shapes_1)
        x2_o, y2_o = self._run_pointwise(k_ones, shapes_other)
        bound_ones_1 = k_ones.bind((x1_o, y1_o))
        bound_ones_3 = k_ones.bind((x2_o, y2_o))

        # Different bound kernels: 1 and 3 are in different buckets
        self.assertIsNot(bound_ones_1, bound_ones_3)
        # Size-1 code has dim0 hardcoded out (no x_size_0 param)
        code_ones_1 = bound_ones_1.to_triton_code()
        self.assertNotIn("x_size_0", code_ones_1)

        # --- "none" mode: size=1 treated same as size=3 ---
        k_none = kernel(
            pointwise_add_kernel,
            settings=Settings(static_shapes="none", autotune_effort="none"),
        )
        x1_n, y1_n = self._run_pointwise(k_none, shapes_1)
        x2_n, y2_n = self._run_pointwise(k_none, shapes_other)
        bound_none_1 = k_none.bind((x1_n, y1_n))
        bound_none_3 = k_none.bind((x2_n, y2_n))

        # SAME bound kernel: 1 and 3 share a bucket
        self.assertIs(bound_none_1, bound_none_3)
        # Code has symbolic dim0 (x_size_0 is a parameter)
        code_none = bound_none_1.to_triton_code()
        self.assertIn("x_size_0", code_none)


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
            torch.randn(m, n, device=DEVICE, dtype=torch.float32),
            torch.randn(m, n, device=DEVICE, dtype=torch.float32),
        ),
        lambda x, y: torch.add(x, y),
        ALL_SHAPES,
    ),
    (
        "exp",
        "exp_fwd",
        lambda m, n: (torch.randn(m, n, device=DEVICE, dtype=torch.float32),),
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
            torch.randn(m, n, device=DEVICE, dtype=torch.float32),
            torch.randn(n, device=DEVICE, dtype=torch.float32),
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
            torch.randn(128, 64, device=DEVICE, dtype=torch.float32),
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
            torch.randn(m, n, device=DEVICE, dtype=torch.float32),
            torch.randn(m, n, device=DEVICE, dtype=torch.float32),
        ),
        lambda x1, x2: torch.nn.functional.gelu(x1, approximate="tanh") * x2,
        ALL_SHAPES,
    ),
    # SwiGLU activation
    (
        "swiglu",
        "swiglu_fwd",
        lambda m, n: (
            torch.randn(m, n, device=DEVICE, dtype=torch.float32),
            torch.randn(m, n, device=DEVICE, dtype=torch.float32),
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
            torch.randn(m, n, device=DEVICE, dtype=torch.float32),  # x
            [n],  # normalized_shape
            torch.randn(n, device=DEVICE, dtype=torch.float32),  # weight
            torch.randn(n, device=DEVICE, dtype=torch.float32),  # bias
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
            torch.randn(m, 64, device=DEVICE, dtype=torch.float32),  # x: [m, k]
            torch.randn(64, n, device=DEVICE, dtype=torch.float32),  # y: [k, n]
        ),
        lambda x, y: torch.matmul(x, y),
        ALL_SHAPES,
    ),
    # Matrix multiplication with split-K - higher parallelism for large K
    (
        "matmul_split_k",
        "matmul_split_k",
        lambda m, n: (
            torch.randn(m, 128, device=DEVICE, dtype=torch.float32),  # x: [m, k]
            torch.randn(128, n, device=DEVICE, dtype=torch.float32),  # y: [k, n]
        ),
        lambda x, y: torch.matmul(x, y),
        ALL_SHAPES,
    ),
    # Fused matmul + layer norm - with k dimension fixed
    (
        "matmul_layernorm",
        "matmul_layernorm",
        lambda m, n: (
            torch.randn(m, 64, device=DEVICE, dtype=torch.float32),  # x: [m, k]
            torch.randn(64, n, device=DEVICE, dtype=torch.float32),  # y: [k, n]
            torch.randn(n, device=DEVICE, dtype=torch.float32),  # weight: [n]
            torch.randn(n, device=DEVICE, dtype=torch.float32),  # bias: [n]
        ),
        lambda x, y, w, b: torch.nn.functional.layer_norm(
            torch.matmul(x, y), [y.size(1)], w, b
        ),
        ALL_SHAPES,
    ),
    # Squeeze and excitation network forward - with k dimension fixed
    (
        "squeeze_and_excitation_net",
        "squeeze_and_excitation_net_fwd",
        lambda m, n: (
            torch.randn(m, n, device=DEVICE, dtype=torch.float32),  # x: [m, n]
            torch.randn(n, 32, device=DEVICE, dtype=torch.float32),  # a: [n, k]
            torch.randn(32, n, device=DEVICE, dtype=torch.float32),  # b: [k, n]
        ),
        lambda x, a, b: torch.mul(x, torch.sigmoid(torch.relu(x @ a) @ b)),
        ALL_SHAPES,
    ),
    # Attention - complex indexing with batch/head dims
    (
        "attention",
        "attention",
        lambda m, n: (
            torch.randn(1, 1, m, 64, device=DEVICE, dtype=torch.float32),
            torch.randn(1, 1, n, 64, device=DEVICE, dtype=torch.float32),
            torch.randn(1, 1, n, 64, device=DEVICE, dtype=torch.float32),
        ),
        lambda q, k, v: torch.nn.functional.scaled_dot_product_attention(q, k, v),
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

STATIC_SHAPES_MODES = ["none", "ones", "all"]


def _generate_example_test_cases() -> tuple[
    list[tuple[str, str | None, object, object, str, str, int, int]], list[str]
]:
    """Generate test cases for parametrized test."""
    cases = []
    ids = []
    for (
        example_name,
        fn_name,
        input_factory,
        reference_fn,
        shapes,
    ) in EXAMPLE_CONFIGS_WITH_SHAPES:
        for mode in STATIC_SHAPES_MODES:
            for shape_desc, m, n in shapes:
                cases.append(
                    (
                        example_name,
                        fn_name,
                        input_factory,
                        reference_fn,
                        mode,
                        shape_desc,
                        m,
                        n,
                    )
                )
                ids.append(f"{example_name}-{mode}-{shape_desc}")
    return cases, ids


_EXAMPLE_TEST_CASES, _EXAMPLE_TEST_IDS = _generate_example_test_cases()


@skipIfCpu("needs to be debugged")
@skipIfRefEager("code generation not relevant in ref eager mode")
@skipIfNotCUDA()
@pytest.mark.parametrize(
    "example_name,fn_name,input_factory,reference_fn,mode,shape_desc,m,n",
    _EXAMPLE_TEST_CASES,
    ids=_EXAMPLE_TEST_IDS,
)
def test_example_static_shapes(
    example_name: str,
    fn_name: str | None,
    input_factory: object,
    reference_fn: object,
    mode: str,
    shape_desc: str,
    m: int,
    n: int,
) -> None:
    """Test example with specific static_shapes mode and shape."""
    from helion._testing import EXAMPLES_DIR
    from helion._testing import import_path

    # Import the kernel fresh to avoid state pollution
    mod = import_path(EXAMPLES_DIR / f"{example_name}.py")
    kernel_fn = getattr(mod, fn_name or example_name)

    # Clear any hardcoded configs and cached bound kernels from the kernel
    # This allows the test to use dynamic configs based on input shapes
    kernel_fn.configs = []
    kernel_fn._bound_kernels = {}

    # Set the static_shapes mode, disable autotuning for faster tests,
    # and use IEEE dot precision to avoid TF32 rounding in matmul
    kernel_fn.settings.static_shapes = mode
    kernel_fn.settings.autotune_effort = "none"
    kernel_fn.settings.dot_precision = "ieee"

    # Create test inputs with the specified shapes and run
    args = input_factory(m, n)
    result = kernel_fn(*args)

    # Compare with reference
    expected = reference_fn(*args)

    # Handle tuple results (some kernels return multiple values)
    if isinstance(result, tuple) and not isinstance(expected, tuple):
        result = result[0]

    torch.testing.assert_close(
        result,
        expected,
        rtol=1e-4,
        atol=1e-4,
    )

    # Verify shape-agnosticism in "none" mode: different non-zero shapes must
    # produce the same specialization key (tensor bucketing) and, when the
    # kernel has no extra specialized variables, reuse the same BoundKernel.
    if mode == "none":
        args2 = input_factory(m + 3, n + 5)
        # Check bucketing: specialization key must treat different non-zero
        # shapes as equivalent.
        key1 = kernel_fn.specialization_key(args)
        key2 = kernel_fn.specialization_key(args2)
        assert key1 == key2, (
            f"In 'none' mode, different non-zero shapes ({m},{n}) and "
            f"({m + 3},{n + 5}) produced different specialization keys for "
            f"{example_name}/{fn_name}:\n  key1={key1}\n  key2={key2}"
        )
        # Check kernel reuse: when the kernel has no extra specialization
        # (e.g. from hl.specialize or guards on normalized_shape), the same
        # compiled kernel must be reused for both shapes.
        bound1 = kernel_fn.bind(args)
        if not bound1.env.specialized_vars:
            bound2 = kernel_fn.bind(args2)
            assert bound1 is bound2, (
                f"In 'none' mode with no specialized vars, expected same "
                f"bound kernel for shapes ({m},{n}) and ({m + 3},{n + 5}) for "
                f"{example_name}/{fn_name}"
            )


if __name__ == "__main__":
    unittest.main()
