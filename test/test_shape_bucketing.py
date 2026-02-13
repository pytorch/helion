from __future__ import annotations

import re
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
from helion.runtime.config import Config
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

    def _make_kernel(self, fn, mode, **kwargs):
        """Create a kernel with the given static_shapes mode and autotune disabled."""
        return kernel(
            fn, settings=Settings(static_shapes=mode, autotune_effort="none"), **kwargs
        )

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

    def _assert_bucketing_behavior(
        self,
        k: object,
        bound1: object,
        bound2: object,
        mode: str,
        same_in_none: bool,
        same_in_ones: bool,
    ) -> None:
        """Verify cache/bucket behavior based on mode."""
        if mode == "none":
            if same_in_none:
                self.assertIs(bound1, bound2)
                # Both shapes mapped to one cache entry
                self.assertEqual(len(k._bound_kernels), 1)
            else:
                self.assertIsNot(bound1, bound2)
            self.assertIn("x_size_", bound1.to_triton_code())
        elif mode == "ones":
            if same_in_ones:
                self.assertIs(bound1, bound2)
                # Non-1 dims should use symbolic sizes when shapes are reused
                self.assertIn("x_size_", bound1.to_triton_code())
            else:
                self.assertIsNot(bound1, bound2)
                # bound2 has all dims ≥2, which are dynamic in "ones" mode
                # (assume_static_by_default=False), so must have symbolic sizes
                self.assertIn("x_size_", bound2.to_triton_code())
        else:  # mode == "all"
            self.assertIsNot(bound1, bound2)
            self.assertNotIn("x_size_", bound1.to_triton_code())

    def _run_bucketing_check(
        self, kernel_fn, run_and_bind, mode, shape1, shape2, same_in_none, same_in_ones
    ):
        """Create kernel, run two shapes, bind, and assert bucketing behavior."""
        k = self._make_kernel(kernel_fn, mode)
        bind_args1 = run_and_bind(k, shape1)
        bound1 = k.bind(bind_args1)
        bind_args2 = run_and_bind(k, shape2)
        bound2 = k.bind(bind_args2)
        self._assert_bucketing_behavior(
            k, bound1, bound2, mode, same_in_none, same_in_ones
        )
        return bound1, bound2

    def _check_kernel_correctness(
        self,
        kernel_fn,
        shapes,
        reference_fn,
        *,
        rtol=1e-4,
        atol=1e-4,
        modes=("none", "ones", "all"),
        **kernel_kwargs,
    ):
        """Run single-input kernel across modes and shapes, assert correctness."""
        for mode in modes:
            for shape in shapes:
                with self.subTest(static_shapes=mode, shape=shape):
                    k = self._make_kernel(kernel_fn, mode, **kernel_kwargs)
                    x = torch.randn(*shape, device=DEVICE, dtype=torch.float32)
                    result = k(x)
                    expected = reference_fn(x)
                    torch.testing.assert_close(result, expected, rtol=rtol, atol=atol)
                    self.assertEqual(result.shape, expected.shape)

    @skipIfRefEager("code generation not relevant in ref eager mode")
    @skipIfNotCUDA()
    def test_pointwise_all_modes_shapes(self) -> None:
        """Test pointwise kernel with all modes and shape variations, with journals."""
        # (desc, shapes_1, shapes_2, same_in_none, same_in_ones)
        shape_variations = [
            ("dim0=1", (1, 16), (4, 16), True, False),  # 1->4 changes 1-ness
            ("dim1=1", (16, 1), (16, 8), True, False),  # 1->8 changes 1-ness
            ("both=1", (1, 1), (4, 8), True, False),  # both 1s -> no 1s
        ]
        for mode in ["none", "ones", "all"]:
            for (
                desc,
                shapes_1,
                shapes_2,
                same_in_none,
                same_in_ones,
            ) in shape_variations:
                with self.subTest(mode=mode, shapes=desc):
                    bound1, bound2 = self._run_bucketing_check(
                        pointwise_add_kernel,
                        lambda k, s: self._run_pointwise(k, s),
                        mode,
                        shapes_1,
                        shapes_2,
                        same_in_none,
                        same_in_ones,
                    )
                    # In "ones" mode with dim0=1, size-1 dim is hardcoded away
                    if mode == "ones" and not same_in_ones and desc == "dim0=1":
                        self.assertNotIn("x_size_0", bound1.to_triton_code())
                    # Journal generated code for snapshot testing
                    self.assertExpectedJournal(bound1.to_triton_code())
                    if bound1 is not bound2:
                        self.assertExpectedJournal(bound2.to_triton_code())

    @skipIfRefEager("code generation not relevant in ref eager mode")
    @skipIfNotCUDA()
    def test_reduction_all_modes_shapes(self) -> None:
        """Test reduction kernel with all modes and shape variations, with journals.

        Note: We only vary the non-reduction dimension (dim0) because varying the
        reduction dimension would require different configs/code paths, which is
        not what we're testing here.
        """
        # (desc, shapes_1, shapes_2, same_in_none, same_in_ones)
        shape_variations = [
            ("dim0=1", (1, 64), (4, 64), True, False),  # 1->4 changes 1-ness
            ("dim0=2", (2, 64), (8, 64), True, True),  # 2->8, both non-1
        ]
        for mode in ["none", "ones", "all"]:
            for (
                desc,
                shapes_1,
                shapes_2,
                same_in_none,
                same_in_ones,
            ) in shape_variations:
                with self.subTest(mode=mode, shapes=desc):
                    bound1, bound2 = self._run_bucketing_check(
                        reduction_sum_kernel,
                        lambda k, s: (self._run_reduction(k, s),),
                        mode,
                        shapes_1,
                        shapes_2,
                        same_in_none,
                        same_in_ones,
                    )
                    self.assertExpectedJournal(bound1.to_triton_code())
                    if bound1 is not bound2:
                        self.assertExpectedJournal(bound2.to_triton_code())

    @skipIfRefEager("code generation not relevant in ref eager mode")
    @skipIfNotCUDA()
    def test_reduction_varying_reduction_dim(self) -> None:
        """Test reduction kernel bucketing when the reduction dimension (dim1) varies.

        Complements test_reduction_all_modes_shapes which only varies dim0.

        Sub-case 1 (both dims ≥ 2): The generated wrapper passes
        _RDIM_SIZE_1 = triton.next_power_of_2(x.size(1)) as a tl.constexpr,
        so Triton JIT transparently recompiles for different reduction sizes
        even when the same BoundKernel is reused.  We verify both bucket
        identity AND numerical correctness.

        Sub-case 2 (dim1=1 then dim1=64): Even when compiled with reduction
        dim=1, the compiler preserves the full reduction loop (using guard-free
        size comparisons).  In "none" mode, bucketing maps both shapes to
        the same bucket, and the reused code correctly handles larger dims
        via Triton JIT constexpr recompilation.
        """
        for mode in ["none", "ones", "all"]:
            # Sub-case 1: varying non-zero reduction dim (32,16) then (32,64)
            # Both dims ≥ 2, so the full reduction loop is present and reuse
            # works correctly via Triton JIT constexpr recompilation.
            with self.subTest(mode=mode, case="varying_nonzero_reduction"):
                k = self._make_kernel(reduction_sum_kernel, mode)
                x1 = self._run_reduction(k, (32, 16))
                bound1 = k.bind((x1,))

                # Verify correctness on reuse for all modes
                x2 = self._run_reduction(k, (32, 64))
                bound2 = k.bind((x2,))
                if mode == "none":
                    # Both non-zero → same bucket
                    self.assertIs(bound1, bound2)
                elif mode == "ones":
                    # 16 and 64 both bucket to 2 → same bucket
                    self.assertIs(bound1, bound2)
                else:  # mode == "all"
                    # Exact sizes differ → different BoundKernels
                    self.assertIsNot(bound1, bound2)

                self.assertExpectedJournal(bound1.to_triton_code())

            # Sub-case 2: reduction dim 1 vs 64: (32,1) then (32,64)
            # When compiled with dim1=1, guard-free comparisons preserve the
            # reduction loop.  In "none" mode, reuse works correctly via
            # Triton JIT constexpr recompilation.
            with self.subTest(mode=mode, case="reduction_dim1_vs_dim64"):
                k = self._make_kernel(reduction_sum_kernel, mode)
                x1 = self._run_reduction(k, (32, 1))
                bound1 = k.bind((x1,))

                if mode == "none":
                    # 1 and 64 both map to 2 in "none" mode → same bucket.
                    # Actually run the kernel to verify correctness of reuse.
                    x2 = self._run_reduction(k, (32, 64))
                    bound2 = k.bind((x2,))
                    self.assertIs(bound1, bound2)
                elif mode == "ones":
                    # 1-ness differs: min(1,2)=1 vs min(64,2)=2 → different
                    x2 = self._run_reduction(k, (32, 64))
                    bound2 = k.bind((x2,))
                    self.assertIsNot(bound1, bound2)
                else:  # mode == "all"
                    # Exact sizes differ → different BoundKernels
                    x2 = self._run_reduction(k, (32, 64))
                    bound2 = k.bind((x2,))
                    self.assertIsNot(bound1, bound2)

                self.assertExpectedJournal(bound1.to_triton_code())
                if bound1 is not bound2:
                    self.assertExpectedJournal(bound2.to_triton_code())

    @skipIfRefEager("code generation not relevant in ref eager mode")
    @skipIfNotCUDA()
    def test_reduction_over_size1_dim(self) -> None:
        """Regression test: reduction over a size-1 dimension should work.

        When the reduction dimension is 1, the sum is a no-op but the shape
        should still be properly squeezed from [M, 1] to [M].
        """
        self._check_kernel_correctness(
            reduction_sum_kernel, [(32, 1)], lambda x: x.sum(-1)
        )
        # Journal the generated code for size-1 reduction across modes
        for mode in ("none", "ones", "all"):
            k = self._make_kernel(reduction_sum_kernel, mode)
            x = torch.randn(32, 1, device=DEVICE, dtype=torch.float32)
            k(x)
            self.assertExpectedJournal(k.bind((x,)).to_triton_code())

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

        self._check_kernel_correctness(
            softmax_two_pass_kernel,
            [(32, 1)],
            lambda x: torch.nn.functional.softmax(x, dim=-1),
            rtol=1e-3,
            atol=1e-3,
        )

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
            settings=Settings(
                static_shapes="none",
                autotune_effort="none",
                ignore_warnings=[helion.exc.TensorOperationInWrapper],
            ),
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

        self._check_kernel_correctness(
            nested_tile_kernel,
            [(1, 64), (32, 1), (1, 1), (32, 64)],
            lambda x: x + 1.0,
            rtol=1e-5,
            atol=1e-5,
        )

    @skipIfRefEager("specialization keys not relevant in ref eager mode")
    def test_backward_compat_bool_to_string_modes(self) -> None:
        """Test backward compatibility: True maps to 'all', False maps to 'ones'."""
        t1 = torch.empty(1, 3)
        t2 = torch.empty(2, 3)
        t3 = torch.empty(3, 3)

        def dummy(x: torch.Tensor) -> torch.Tensor:
            return x

        # True -> 'all' mode: verify normalization and behavior
        settings_all = Settings(static_shapes=True, autotune_effort="none")
        self.assertEqual(settings_all.static_shapes, "all")  # normalized to string
        k_all = kernel(dummy, settings=settings_all)
        key_all_2 = k_all.specialization_key([t2])
        key_all_3 = k_all.specialization_key([t3])
        self.assertNotEqual(key_all_2, key_all_3)  # each exact size is distinct

        # False -> 'ones' mode: verify normalization and behavior
        settings_ones = Settings(static_shapes=False, autotune_effort="none")
        self.assertEqual(settings_ones.static_shapes, "ones")  # normalized to string
        k_ones = kernel(dummy, settings=settings_ones)
        key_ones_1 = k_ones.specialization_key([t1])
        key_ones_2 = k_ones.specialization_key([t2])
        key_ones_3 = k_ones.specialization_key([t3])
        self.assertNotEqual(key_ones_1, key_ones_2)  # 1 is distinct from >=2
        self.assertEqual(key_ones_2, key_ones_3)  # but 2 and 3 are same

    @skipIfRefEager("code generation not relevant in ref eager mode")
    @skipIfNotCUDA()
    def test_stride_handling(self) -> None:
        """Test stride handling across all bucketing modes.

        In 'ones'/'none' modes, strides are NOT in the cache key — a kernel
        compiled for contiguous input must still be correct for transposed input.
        In 'all' mode, strides ARE in the cache key — contiguous vs transposed
        same-shape tensors produce different specialization keys.
        """
        # Dynamic modes: correctness with different strides
        for mode in ["ones", "none"]:
            with self.subTest(mode=mode, case="pointwise"):
                k = self._make_kernel(pointwise_add_kernel, mode)
                x1 = torch.randn(32, 64, device=DEVICE, dtype=torch.float32)
                y1 = torch.empty_like(x1)
                k(x1, y1)
                torch.testing.assert_close(y1, x1 + 1.0, rtol=1e-4, atol=1e-4)

                x2 = torch.randn(64, 32, device=DEVICE, dtype=torch.float32).T
                self.assertEqual(x2.shape, (32, 64))
                self.assertFalse(x2.is_contiguous())
                y2 = torch.empty(32, 64, device=DEVICE, dtype=torch.float32)
                k(x2, y2)
                torch.testing.assert_close(y2, x2 + 1.0, rtol=1e-4, atol=1e-4)

            with self.subTest(mode=mode, case="reduction"):
                k = self._make_kernel(reduction_sum_kernel, mode)
                x1 = torch.randn(8, 64, device=DEVICE, dtype=torch.float32)
                result1 = k(x1)
                torch.testing.assert_close(result1, x1.sum(-1), rtol=1e-4, atol=1e-4)

                x2 = torch.randn(64, 8, device=DEVICE, dtype=torch.float32).T
                self.assertEqual(x2.shape, (8, 64))
                self.assertFalse(x2.is_contiguous())
                result2 = k(x2)
                torch.testing.assert_close(result2, x2.sum(-1), rtol=1e-4, atol=1e-4)

        # "all" mode: strides are in the specialization key
        with self.subTest(mode="all", case="key_includes_strides"):

            def dummy(x: torch.Tensor) -> torch.Tensor:
                return x

            k = kernel(
                dummy, settings=Settings(static_shapes="all", autotune_effort="none")
            )
            t_contig = torch.randn(32, 64, device=DEVICE, dtype=torch.float32)
            t_transposed = torch.randn(64, 32, device=DEVICE, dtype=torch.float32).T
            self.assertEqual(t_transposed.shape, (32, 64))

            key_contig = k.specialization_key([t_contig])
            key_transposed = k.specialization_key([t_transposed])
            self.assertNotEqual(key_contig, key_transposed)

            # Same shape + same strides → same key
            t_contig2 = torch.randn(32, 64, device=DEVICE, dtype=torch.float32)
            self.assertEqual(key_contig, k.specialization_key([t_contig2]))

    @skipIfRefEager("code generation not relevant in ref eager mode")
    @skipIfNotCUDA()
    def test_zero_size_dims(self) -> None:
        """Test zero-size dimensions work correctly with all bucketing modes.

        The _tensor_key has explicit zero-handling (0 if s == 0 else 2 in
        'none' mode) but these paths were untested with dynamic modes.
        """
        for mode in ["none", "ones", "all"]:
            # Correctness: zero-size dims produce empty but correct results
            for shape in [(0, 32), (0, 0)]:
                with self.subTest(mode=mode, case="correctness", shape=shape):
                    k = self._make_kernel(pointwise_add_kernel, mode)
                    x = torch.randn(*shape, device=DEVICE, dtype=torch.float32)
                    y = torch.empty_like(x)
                    k(x, y)
                    self.assertEqual(y.shape, shape)
                    torch.testing.assert_close(y, x + 1.0, rtol=1e-4, atol=1e-4)

            # NOTE: reduction with zero-size reduction dim (e.g. [4, 0]) is not
            # tested because Triton's tl.arange(0, 0) is unsupported.

            # Identity: zero-size shapes must not share BoundKernels
            for shape_a, shape_b in [((0, 32), (4, 32)), ((0, 0), (0, 32))]:
                with self.subTest(
                    mode=mode, case="identity", shapes=(shape_a, shape_b)
                ):
                    k = self._make_kernel(pointwise_add_kernel, mode)
                    xa = torch.randn(*shape_a, device=DEVICE, dtype=torch.float32)
                    ya = torch.empty_like(xa)
                    k(xa, ya)
                    bound_a = k.bind((xa, ya))

                    xb = torch.randn(*shape_b, device=DEVICE, dtype=torch.float32)
                    yb = torch.empty_like(xb)
                    k(xb, yb)
                    bound_b = k.bind((xb, yb))
                    self.assertIsNot(bound_a, bound_b)

    @skipIfRefEager("code generation not relevant in ref eager mode")
    @skipIfNotCUDA()
    def test_mark_static(self) -> None:
        """Test mark_static: single dim, multiple dims, and size-1 dims."""
        # Sub-case 1: Single dim mark_static across all modes
        for mode in ["none", "ones", "all"]:
            with self.subTest(mode=mode, case="single_dim"):
                k = self._make_kernel(pointwise_add_kernel, mode)
                x1 = torch.randn(96, 32, device=DEVICE, dtype=torch.float32)
                torch._dynamo.mark_static(x1, 0)
                y1 = torch.empty_like(x1)
                k(x1, y1)
                torch.testing.assert_close(y1, x1 + 1.0, rtol=1e-4, atol=1e-4)
                bound1 = k.bind((x1, y1))
                self.assertIn("96", bound1.to_triton_code())

                # Changing marked dim → cache miss
                x2 = torch.randn(128, 32, device=DEVICE, dtype=torch.float32)
                torch._dynamo.mark_static(x2, 0)
                y2 = torch.empty_like(x2)
                k(x2, y2)
                torch.testing.assert_close(y2, x2 + 1.0, rtol=1e-4, atol=1e-4)
                self.assertIsNot(bound1, k.bind((x2, y2)))

                # In dynamic modes, changing non-marked dim → cache hit
                if mode in ("ones", "none"):
                    x3 = torch.randn(96, 48, device=DEVICE, dtype=torch.float32)
                    torch._dynamo.mark_static(x3, 0)
                    y3 = torch.empty_like(x3)
                    k(x3, y3)
                    torch.testing.assert_close(y3, x3 + 1.0, rtol=1e-4, atol=1e-4)
                    self.assertIs(bound1, k.bind((x3, y3)))

        # Sub-case 2: Multiple dims marked static
        for mode in ["none", "ones"]:
            with self.subTest(mode=mode, case="multiple_dims"):
                k = self._make_kernel(pointwise_add_kernel, mode)
                x1 = torch.randn(48, 64, device=DEVICE, dtype=torch.float32)
                torch._dynamo.mark_static(x1, 0)
                torch._dynamo.mark_static(x1, 1)
                y1 = torch.empty_like(x1)
                k(x1, y1)
                torch.testing.assert_close(y1, x1 + 1.0, rtol=1e-4, atol=1e-4)
                bound1 = k.bind((x1, y1))
                code = bound1.to_triton_code()
                self.assertIn("48", code)
                self.assertIn("64", code)

                # Changing dim 0 → cache miss
                x2 = torch.randn(96, 64, device=DEVICE, dtype=torch.float32)
                torch._dynamo.mark_static(x2, 0)
                torch._dynamo.mark_static(x2, 1)
                y2 = torch.empty_like(x2)
                k(x2, y2)
                torch.testing.assert_close(y2, x2 + 1.0, rtol=1e-4, atol=1e-4)
                self.assertIsNot(bound1, k.bind((x2, y2)))

                # Changing dim 1 → cache miss
                x3 = torch.randn(48, 128, device=DEVICE, dtype=torch.float32)
                torch._dynamo.mark_static(x3, 0)
                torch._dynamo.mark_static(x3, 1)
                y3 = torch.empty_like(x3)
                k(x3, y3)
                torch.testing.assert_close(y3, x3 + 1.0, rtol=1e-4, atol=1e-4)
                self.assertIsNot(bound1, k.bind((x3, y3)))

        # Sub-case 3: Size-1 dim with mark_static
        for mode in ["none", "ones"]:
            with self.subTest(mode=mode, case="size1_dim"):
                k = self._make_kernel(pointwise_add_kernel, mode)
                x1 = torch.randn(1, 32, device=DEVICE, dtype=torch.float32)
                torch._dynamo.mark_static(x1, 0)
                y1 = torch.empty_like(x1)
                k(x1, y1)
                torch.testing.assert_close(y1, x1 + 1.0, rtol=1e-4, atol=1e-4)
                bound1 = k.bind((x1, y1))
                self.assertNotIn("x_size_0", bound1.to_triton_code())

                # Different marked value → different BoundKernel
                x2 = torch.randn(4, 32, device=DEVICE, dtype=torch.float32)
                torch._dynamo.mark_static(x2, 0)
                y2 = torch.empty_like(x2)
                k(x2, y2)
                torch.testing.assert_close(y2, x2 + 1.0, rtol=1e-4, atol=1e-4)
                self.assertIsNot(bound1, k.bind((x2, y2)))

    @skipIfRefEager("code generation not relevant in ref eager mode")
    @skipIfNotCUDA()
    def test_none_mode_cross_size_reuse(self) -> None:
        """Test 'none' mode kernel compiled for one size works for other sizes.

        Verifies that generated code uses symbolic shapes (not hardcoded values)
        so a single compiled kernel works correctly across different sizes.
        """
        # Pointwise: compile with one shape, reuse for several others
        k = self._make_kernel(pointwise_add_kernel, "none")

        shapes = [(1, 16), (5, 16), (1, 32), (7, 3)]
        for shape in shapes:
            with self.subTest(case="pointwise", shape=shape):
                x = torch.randn(*shape, device=DEVICE, dtype=torch.float32)
                y = torch.empty_like(x)
                k(x, y)
                torch.testing.assert_close(y, x + 1.0, rtol=1e-4, atol=1e-4)

        # All non-zero shapes share one BoundKernel in "none" mode
        self.assertEqual(len(k._bound_kernels), 1)

        # Reduction: compile with one shape, reuse for another
        k_red = self._make_kernel(reduction_sum_kernel, "none")

        for shape in [(1, 64), (8, 64)]:
            with self.subTest(case="reduction", shape=shape):
                x = torch.randn(*shape, device=DEVICE, dtype=torch.float32)
                result = k_red(x)
                torch.testing.assert_close(result, x.sum(-1), rtol=1e-4, atol=1e-4)

        self.assertEqual(len(k_red._bound_kernels), 1)

    @skipIfRefEager("code generation not relevant in ref eager mode")
    @skipIfNotCUDA()
    def test_nested_tile_none_mode_reuse(self) -> None:
        """Verify 'none' mode kernel with 2D tiling reuses across size-1 changes.

        This exercises the known_equal() fix in BlockedSubscriptIndexing
        (indexing_strategy.py) — if guards were added when checking size==1,
        the kernel compiled for m=1 would NOT be reusable for m=4.

        We compile with m=1 FIRST because that is the critical case: the
        indexing_strategy checks ``size == 1`` and known_equal() must return
        False (preventing a guard) even though the hint is 1.  An explicit
        config with block_size_m > 1 is provided so that the compiler does
        not eliminate the m-dimension loop (which would produce code that
        only processes one row regardless of runtime m).
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

        k = self._make_kernel(
            nested_tile_kernel, "none", config=Config(block_sizes=[32, 32])
        )

        # Compile with m=1 first (critical: size-1 triggers the guard risk
        # in known_equal())
        x1 = torch.randn(1, 64, device=DEVICE, dtype=torch.float32)
        result1 = k(x1)
        torch.testing.assert_close(result1, x1 + 1.0, rtol=1e-5, atol=1e-5)
        bound1 = k.bind((x1,))

        # Reuse for m=4 — must be same BoundKernel (proves no guard was
        # added on the size==1 check in indexing_strategy.py)
        x4 = torch.randn(4, 64, device=DEVICE, dtype=torch.float32)
        result4 = k(x4)
        torch.testing.assert_close(result4, x4 + 1.0, rtol=1e-5, atol=1e-5)
        bound4 = k.bind((x4,))

        self.assertIs(bound1, bound4)
        # Journal: verify the generated code uses symbolic sizes (shape-agnostic)
        self.assertExpectedJournal(bound1.to_triton_code())

    @skipIfRefEager("code generation not relevant in ref eager mode")
    @skipIfNotCUDA()
    def test_3d_tensor_mixed_size1_dims(self) -> None:
        """Test 3D tensors with mixed size-1 dimensions across all modes.

        All existing tests use 2D tensors. The bucketing logic in _tensor_key
        operates on obj.size() of any dimensionality, and indexing_strategy's
        known_equal() checks apply per-dimension. 3D tensors with mixed 1-sizes
        (e.g., [1, 32, 1]) exercise additional code paths.
        """

        def pointwise_3d_kernel(x: torch.Tensor, out: torch.Tensor) -> None:
            for tile in hl.tile(x.size()):
                out[tile] = x[tile] + 1.0

        # (desc, shape1, shape2, same_in_none, same_in_ones)
        shape_variations = [
            ("outer_inner_1", (1, 32, 1), (4, 32, 8), True, False),
            ("middle_1", (4, 1, 8), (4, 16, 8), True, False),
            ("all_1", (1, 1, 1), (4, 8, 16), True, False),
            ("no_1s", (4, 8, 16), (2, 3, 5), True, True),
        ]

        for mode in ["none", "ones", "all"]:
            for desc, shape1, shape2, same_in_none, same_in_ones in shape_variations:
                with self.subTest(mode=mode, shapes=desc):
                    self._run_bucketing_check(
                        pointwise_3d_kernel,
                        lambda k, s: self._run_pointwise(k, s),
                        mode,
                        shape1,
                        shape2,
                        same_in_none,
                        same_in_ones,
                    )


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
    # Force fresh module import to prevent state leakage between parametrized tests,
    # since import_path caches modules in sys.modules.
    import sys

    from helion._testing import EXAMPLES_DIR
    from helion._testing import import_path

    module_cache_key = f"helion._testing.{example_name}"
    sys.modules.pop(module_cache_key, None)
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

    # Kernels using tl.dot (matmul, attention, etc.) produce slightly different
    # rounding than cuBLAS/cuDNN, so use relaxed tolerances matching check_example().
    EXAMPLES_WITH_TL_DOT = {
        "matmul",
        "matmul_split_k",
        "matmul_layernorm",
        "squeeze_and_excitation_net",
        "attention",
    }
    has_tl_dot = example_name in EXAMPLES_WITH_TL_DOT
    torch.testing.assert_close(
        result,
        expected,
        rtol=1e-2 if has_tl_dot else 1e-4,
        atol=1e-1 if has_tl_dot else 1e-4,
    )

    # Code-generation verification: in "all" mode, no symbolic size vars
    # (like x_size_0, logits_size_1) should appear in the generated code.
    # The pattern _size_\d avoids false positives from block_size_n in comments.
    # This check runs even with specialized_vars because "all" mode sets
    # assume_static_by_default=True, so all dimensions should be hardcoded.
    bound_check = kernel_fn.bind(args)
    if mode == "all":
        code = bound_check.to_triton_code()
        assert not re.search(r"_size_\d", code), (
            f"'all' mode should not have symbolic size vars in code for "
            f"{example_name}, but found '_size_N'.\nCode: {code[:500]}"
        )

    # Code-generation verification for "none" mode: verify the generated code
    # is shape-agnostic by checking that tensor strides are passed dynamically
    # (via .stride() calls in the wrapper), not hardcoded as integer literals.
    # This catches the bug where ShapeEnv/FakeTensor guards could accidentally
    # specialize the code even when bucketing treats shapes as equivalent.
    if mode == "none" and not bound_check.env.specialized_vars:
        code = bound_check.to_triton_code()
        # Verify each input tensor has dynamic .stride() calls in the wrapper,
        # not hardcoded integer strides from ShapeEnv/FakeTensor specialization.
        stride_params = set(re.findall(r"(\w+)\.stride\(", code))
        input_tensor_count = sum(1 for a in args if isinstance(a, torch.Tensor))
        assert len(stride_params) >= input_tensor_count, (
            f"'none' mode should pass dynamic strides for all "
            f"{input_tensor_count} tensor parameters in "
            f"{example_name}/{fn_name}, but only found .stride() calls for "
            f"{len(stride_params)} parameters: {stride_params}.\n"
            f"Code: {code[:500]}"
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
        # Verify correctness of reuse: actually run the kernel with the
        # second set of shapes and check it produces correct results.
        # This catches the case where generated code is secretly specialized
        # for the first shape but bucketing maps both shapes to the same key.
        result2 = kernel_fn(*args2)
        expected2 = reference_fn(*args2)
        if isinstance(result2, tuple) and not isinstance(expected2, tuple):
            result2 = result2[0]
        torch.testing.assert_close(
            result2,
            expected2,
            rtol=1e-2 if has_tl_dot else 1e-4,
            atol=1e-1 if has_tl_dot else 1e-4,
        )
        # Check kernel reuse: different non-zero shapes should reuse the same
        # compiled kernel.  When bound kernels differ, it must be because
        # hl.specialize() variables took different concrete values — not
        # because an accidental guard was inserted on a non-specialized dim.
        bound1 = kernel_fn.bind(args)
        bound2 = kernel_fn.bind(args2)
        if bound1 is not bound2:
            assert bound1.env.specialized_vars, (
                f"In 'none' mode with no specialized vars, expected same "
                f"bound kernel for shapes ({m},{n}) and ({m + 3},{n + 5}) "
                f"for {example_name}/{fn_name}"
            )


if __name__ == "__main__":
    unittest.main()
