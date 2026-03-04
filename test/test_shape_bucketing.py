from __future__ import annotations

import re
import unittest
from unittest.mock import patch

import torch
from torch.testing._internal.common_utils import instantiate_parametrized_tests
from torch.testing._internal.common_utils import parametrize

import helion
from helion import _compat
from helion._testing import DEVICE
from helion._testing import RefEagerTestBase
from helion._testing import TestCase
from helion._testing import skipIfNotCUDA
from helion._testing import skipIfRefEager
from helion._testing import skipIfTileIR
import helion.language as hl
from helion.runtime.config import Config
from helion.runtime.kernel import kernel
from helion.runtime.settings import Settings


def pointwise_add_kernel(x: torch.Tensor) -> torch.Tensor:
    """Simple pointwise kernel: out = x + 1.0"""
    out = torch.empty_like(x)
    for tile in hl.tile(x.size()):
        out[tile] = x[tile] + 1.0
    return out


def reduction_sum_kernel(x: torch.Tensor) -> torch.Tensor:
    """Reduction kernel: sum along last dimension."""
    out = x.new_empty([x.size(0)])
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile, :].sum(-1)
    return out


def softmax_two_pass_kernel(x: torch.Tensor) -> torch.Tensor:
    """Numerically optimized softmax in two passes - from examples/softmax.py.

    This kernel has nested hl.tile loops and reduces over the inner dimension.
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


def nested_tile_kernel(x: torch.Tensor) -> torch.Tensor:
    """Nested tile kernel with registered block sizes."""
    m, n = x.size()
    out = torch.empty_like(x)
    block_size_m = hl.register_block_size(m)
    block_size_n = hl.register_block_size(n)
    for tile_m in hl.tile(m, block_size=block_size_m):
        for tile_n in hl.tile(n, block_size=block_size_n):
            out[tile_m, tile_n] = x[tile_m, tile_n] + 1.0
    return out


class TestShapeBucketing(RefEagerTestBase, TestCase):
    maxDiff = 16384

    def _make_kernel(self, fn, mode, **kwargs):
        """Create a kernel with the given static_shapes mode and autotune disabled."""
        return kernel(
            fn, settings=Settings(static_shapes=mode, autotune_effort="none"), **kwargs
        )

    def _check_bucketing(
        self,
        kernel_fn,
        ref_fn,
        shape1,
        shape2,
        same_in_ones,
        *,
        rtol=1e-4,
        atol=1e-4,
        modes=("none", "ones", "all"),
        **kernel_kwargs,
    ):
        """Check correctness, bucket identity, and cross-size reuse for two shapes."""
        for mode in modes:
            with self.subTest(mode=mode):
                k = self._make_kernel(kernel_fn, mode, **kernel_kwargs)
                x1 = torch.randn(*shape1, device=DEVICE, dtype=torch.float32)
                r1 = k(x1)
                torch.testing.assert_close(r1, ref_fn(x1), rtol=rtol, atol=atol)
                x2 = torch.randn(*shape2, device=DEVICE, dtype=torch.float32)
                r2 = k(x2)
                torch.testing.assert_close(r2, ref_fn(x2), rtol=rtol, atol=atol)
                bound1 = k.bind((x1,))
                bound2 = k.bind((x2,))
                if mode == "none":
                    self.assertIs(bound1, bound2)
                    self.assertEqual(len(k._bound_kernels), 1)
                elif mode == "ones":
                    if same_in_ones:
                        self.assertIs(bound1, bound2)
                        self.assertEqual(len(k._bound_kernels), 1)
                    else:
                        self.assertIsNot(bound1, bound2)
                        self.assertEqual(len(k._bound_kernels), 2)
                else:
                    assert mode == "all"
                    self.assertIsNot(bound1, bound2)

    @skipIfRefEager("code generation not relevant in ref eager mode")
    @skipIfNotCUDA()
    def test_pointwise_size1_bucketing_across_modes(self) -> None:
        """Test pointwise kernel with all modes and shape variations."""

        def ref(x):
            return x + 1.0

        # (desc, shape1, shape2, same_in_ones)
        shape_list = [
            ("dim0=1", (1, 16), (4, 16), False),  # 1->4 changes 1-ness
            ("dim1=1", (16, 1), (16, 8), False),  # 1->8 changes 1-ness
            ("both=1", (1, 1), (4, 8), False),  # both 1s -> no 1s
            ("no_1s", (2, 16), (8, 32), True),  # both ≥2 in all dims
            ("same_1_in_dim0", (1, 16), (1, 32), True),  # same 1-ness pattern
            ("dim0=2_dim1=1", (2, 1), (2, 8), False),  # boundary: size==bucket
            ("shape2_dim1_1", (4, 16), (4, 1), False),  # shape2 has 1-dim
            # 3D tensors with mixed size-1 dimensions
            ("3d_outer_inner_1", (1, 32, 1), (4, 32, 8), False),
            ("3d_middle_1", (4, 1, 8), (4, 16, 8), False),
            ("3d_all_1", (1, 1, 1), (4, 8, 16), False),
            ("3d_no_1s", (4, 8, 16), (2, 3, 5), True),
            ("3d_shape2_middle_1", (4, 16, 8), (4, 1, 8), False),
        ]
        for desc, shape1, shape2, same_in_ones in shape_list:
            with self.subTest(shapes=desc):
                self._check_bucketing(
                    pointwise_add_kernel, ref, shape1, shape2, same_in_ones
                )

        # Incremental bucket growth: same-1-ness shapes share one bucket,
        # then a different-1-ness shape creates a second bucket.
        with self.subTest(case="ones_mode_incremental_bucket_growth"):
            k = self._make_kernel(pointwise_add_kernel, "ones")
            # Three shapes with dim0=1 → all share one bucket
            for shape in [(1, 16), (1, 32), (1, 3)]:
                x = torch.randn(*shape, device=DEVICE, dtype=torch.float32)
                torch.testing.assert_close(k(x), ref(x), rtol=1e-4, atol=1e-4)
            self.assertEqual(len(k._bound_kernels), 1)
            # Different 1-ness pattern → second bucket
            x = torch.randn(4, 16, device=DEVICE, dtype=torch.float32)
            torch.testing.assert_close(k(x), ref(x), rtol=1e-4, atol=1e-4)
            self.assertEqual(len(k._bound_kernels), 2)

    @skipIfRefEager("code generation not relevant in ref eager mode")
    @skipIfNotCUDA()
    def test_reduction_size1_bucketing_across_modes(self) -> None:
        """Test reduction kernel with all modes and shape variations."""

        def ref(x):
            return x.sum(-1)

        # (desc, shape1, shape2, same_in_ones)
        shape_list = [
            ("dim0=1", (1, 64), (4, 64), False),  # 1->4 changes 1-ness
            ("dim0=2", (2, 64), (8, 64), True),  # 2->8, both non-1
            ("rdim1", (32, 1), (32, 64), False),  # was test_reduction_rdim1
            (
                "rdim_varies",
                (32, 16),
                (32, 64),
                True,
            ),  # was test_reduction_varying_nonzero_rdim
            ("both_1", (1, 1), (4, 32), False),  # was test_reduction_both_dims_1
        ]
        for desc, shape1, shape2, same_in_ones in shape_list:
            with self.subTest(shapes=desc):
                self._check_bucketing(
                    reduction_sum_kernel, ref, shape1, shape2, same_in_ones
                )

    @skipIfRefEager("code generation not relevant in ref eager mode")
    @skipIfNotCUDA()
    @parametrize("mode", ("none", "ones", "all"))
    def test_stride_specialization(self, mode: str) -> None:
        """Test stride handling across all bucketing modes.

        In 'ones'/'none' modes, strides are NOT in the cache key — a kernel
        compiled for contiguous input must still be correct for transposed input.
        In 'all' mode, strides ARE in the cache key — contiguous vs transposed
        same-shape tensors produce different specialization keys.
        """

        def ref(x):
            return x + 1.0

        if mode in ("ones", "none"):
            # Pointwise: correctness with different strides
            k = self._make_kernel(pointwise_add_kernel, mode)
            x1 = torch.randn(32, 64, device=DEVICE, dtype=torch.float32)
            result1 = k(x1)
            torch.testing.assert_close(result1, ref(x1), rtol=1e-4, atol=1e-4)

            x2 = torch.randn(64, 32, device=DEVICE, dtype=torch.float32).T
            self.assertEqual(x2.shape, (32, 64))
            self.assertFalse(x2.is_contiguous())
            result2 = k(x2)
            torch.testing.assert_close(result2, ref(x2), rtol=1e-4, atol=1e-4)

            # Strides are NOT in the cache key — same key for contig vs transposed
            key_contig = k.specialization_key([x1])
            key_transposed = k.specialization_key([x2])
            self.assertEqual(key_contig, key_transposed)

            # Reduction: correctness with different strides
            k = self._make_kernel(reduction_sum_kernel, mode)
            x1 = torch.randn(8, 64, device=DEVICE, dtype=torch.float32)
            result1 = k(x1)
            torch.testing.assert_close(result1, x1.sum(-1), rtol=1e-4, atol=1e-4)

            x2 = torch.randn(64, 8, device=DEVICE, dtype=torch.float32).T
            self.assertEqual(x2.shape, (8, 64))
            self.assertFalse(x2.is_contiguous())
            result2 = k(x2)
            torch.testing.assert_close(result2, x2.sum(-1), rtol=1e-4, atol=1e-4)

            key_contig = k.specialization_key([x1])
            key_transposed = k.specialization_key([x2])
            self.assertEqual(key_contig, key_transposed)
        else:
            assert mode == "all"

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
    @parametrize("mode", ("none", "ones", "all"))
    def test_mark_static_dims_force_recompilation_on_change(self, mode: str) -> None:
        """Test mark_static: single dim, multiple dims, and size-1 dims."""

        def ref(x):
            return x + 1.0

        # Single dim mark_static: changing marked dim → cache miss
        k = self._make_kernel(pointwise_add_kernel, mode)
        x1 = torch.randn(96, 32, device=DEVICE, dtype=torch.float32)
        torch._dynamo.mark_static(x1, 0)
        result1 = k(x1)
        torch.testing.assert_close(result1, ref(x1), rtol=1e-4, atol=1e-4)
        bound1 = k.bind((x1,))

        x2 = torch.randn(128, 32, device=DEVICE, dtype=torch.float32)
        torch._dynamo.mark_static(x2, 0)
        result2 = k(x2)
        torch.testing.assert_close(result2, ref(x2), rtol=1e-4, atol=1e-4)
        self.assertIsNot(bound1, k.bind((x2,)))

        # In dynamic modes, changing non-marked dim → cache hit
        if mode in ("ones", "none"):
            x3 = torch.randn(96, 48, device=DEVICE, dtype=torch.float32)
            torch._dynamo.mark_static(x3, 0)
            result3 = k(x3)
            torch.testing.assert_close(result3, ref(x3), rtol=1e-4, atol=1e-4)
            self.assertIs(bound1, k.bind((x3,)))

        # Remaining sub-cases only apply to dynamic modes
        if mode == "all":
            return

        # Multiple dims marked static: changing either dim → cache miss
        k = self._make_kernel(pointwise_add_kernel, mode)
        x1 = torch.randn(48, 64, device=DEVICE, dtype=torch.float32)
        torch._dynamo.mark_static(x1, 0)
        torch._dynamo.mark_static(x1, 1)
        result1 = k(x1)
        torch.testing.assert_close(result1, ref(x1), rtol=1e-4, atol=1e-4)
        bound1 = k.bind((x1,))

        x2 = torch.randn(96, 64, device=DEVICE, dtype=torch.float32)
        torch._dynamo.mark_static(x2, 0)
        torch._dynamo.mark_static(x2, 1)
        result2 = k(x2)
        torch.testing.assert_close(result2, ref(x2), rtol=1e-4, atol=1e-4)
        self.assertIsNot(bound1, k.bind((x2,)))

        x3 = torch.randn(48, 128, device=DEVICE, dtype=torch.float32)
        torch._dynamo.mark_static(x3, 0)
        torch._dynamo.mark_static(x3, 1)
        result3 = k(x3)
        torch.testing.assert_close(result3, ref(x3), rtol=1e-4, atol=1e-4)
        self.assertIsNot(bound1, k.bind((x3,)))

        # Size-1 dim with mark_static: different marked value → different BoundKernel
        k = self._make_kernel(pointwise_add_kernel, mode)
        x1 = torch.randn(1, 32, device=DEVICE, dtype=torch.float32)
        torch._dynamo.mark_static(x1, 0)
        result1 = k(x1)
        torch.testing.assert_close(result1, ref(x1), rtol=1e-4, atol=1e-4)
        bound1 = k.bind((x1,))

        x2 = torch.randn(4, 32, device=DEVICE, dtype=torch.float32)
        torch._dynamo.mark_static(x2, 0)
        result2 = k(x2)
        torch.testing.assert_close(result2, ref(x2), rtol=1e-4, atol=1e-4)
        self.assertIsNot(bound1, k.bind((x2,)))

        # static_indices changes the specialization key:
        # marked vs unmarked should produce different keys even with same shapes
        k = self._make_kernel(pointwise_add_kernel, mode)
        x_marked = torch.randn(32, 64, device=DEVICE, dtype=torch.float32)
        torch._dynamo.mark_static(x_marked, 0)
        x_unmarked = torch.randn(32, 64, device=DEVICE, dtype=torch.float32)
        key_marked = k.specialization_key([x_marked])
        key_unmarked = k.specialization_key([x_unmarked])
        self.assertNotEqual(key_marked, key_unmarked)

    @skipIfRefEager("code generation not relevant in ref eager mode")
    @skipIfNotCUDA()
    @parametrize("mode", ("none", "ones", "all"))
    def test_zero_size_dims(self, mode: str) -> None:
        """Test zero-size dimensions work correctly with all bucketing modes."""

        def ref(x):
            return x + 1.0

        # Correctness: zero-size dims produce empty but correct results
        # NOTE: reduction with zero-size reduction dim (e.g. [4, 0]) is not
        # tested because Triton's tl.arange(0, 0) is unsupported.
        for shape in [(0, 32), (0, 0)]:
            with self.subTest(case="correctness", shape=shape):
                k = self._make_kernel(pointwise_add_kernel, mode)
                x = torch.randn(*shape, device=DEVICE, dtype=torch.float32)
                result = k(x)
                self.assertEqual(result.shape, shape)
                torch.testing.assert_close(result, ref(x), rtol=1e-4, atol=1e-4)

        # Reuse: two zero-first-dim shapes with different second dims
        # should map to the same bucket in "none" and "ones" modes.
        if mode in ("none", "ones"):
            k = self._make_kernel(pointwise_add_kernel, mode)
            xa = torch.randn(0, 16, device=DEVICE, dtype=torch.float32)
            k(xa)
            xb = torch.randn(0, 32, device=DEVICE, dtype=torch.float32)
            k(xb)
            self.assertIs(k.bind((xa,)), k.bind((xb,)))

        # Specialization key: zero-size dim produces distinct key from non-zero dims
        k = self._make_kernel(pointwise_add_kernel, mode)
        x_zero = torch.randn(0, 32, device=DEVICE, dtype=torch.float32)
        x_nonzero = torch.randn(4, 32, device=DEVICE, dtype=torch.float32)
        key_zero = k.specialization_key([x_zero])
        key_nonzero = k.specialization_key([x_nonzero])
        self.assertNotEqual(key_zero, key_nonzero)

        # Identity: zero-size shapes must not share BoundKernels
        for shape_a, shape_b in [((0, 32), (4, 32)), ((0, 0), (0, 32))]:
            with self.subTest(case="identity", shapes=(shape_a, shape_b)):
                k = self._make_kernel(pointwise_add_kernel, mode)
                xa = torch.randn(*shape_a, device=DEVICE, dtype=torch.float32)
                k(xa)
                bound_a = k.bind((xa,))

                xb = torch.randn(*shape_b, device=DEVICE, dtype=torch.float32)
                k(xb)
                bound_b = k.bind((xb,))
                self.assertIsNot(bound_a, bound_b)

    @skipIfRefEager("code generation not relevant in ref eager mode")
    @skipIfNotCUDA()
    def test_softmax_two_pass_with_size1_inner_dim(self) -> None:
        """Test softmax two-pass kernel with size-1 dimensions."""

        def ref(x):
            return torch.softmax(x, dim=-1)

        shape_list = [
            ("n=1", (32, 1), (32, 64), False),
            ("both=1", (1, 1), (4, 6), False),
        ]
        for desc, shape1, shape2, same_in_ones in shape_list:
            with self.subTest(shapes=desc):
                self._check_bucketing(
                    softmax_two_pass_kernel,
                    ref,
                    shape1,
                    shape2,
                    same_in_ones,
                    rtol=1e-3,
                    atol=1e-3,
                )

    @skipIfRefEager("code generation not relevant in ref eager mode")
    @skipIfNotCUDA()
    def test_view_flatten_none_mode_with_unit_dim(self) -> None:
        """Test .view(-1) works with static_shapes='none' when n=1."""

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

        m, n = 32, 1
        logits = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
        labels = torch.randint(0, n, (m,), device=DEVICE, dtype=torch.long)

        result = k(logits, labels)
        expected = torch.nn.functional.cross_entropy(logits, labels)

        torch.testing.assert_close(result, expected, rtol=1e-2, atol=1e-2)

    @skipIfRefEager("code generation not relevant in ref eager mode")
    @skipIfNotCUDA()
    def test_nested_tile_with_size1_dims(self) -> None:
        """Test nested tile kernel with size-1 dimensions and none-mode reuse."""

        def ref(x):
            return x + 1.0

        shape_list = [
            ("m=1", (1, 64), (32, 64), False),
            ("n=1", (32, 1), (32, 64), False),
            ("both=1", (1, 1), (32, 64), False),
        ]
        for desc, shape1, shape2, same_in_ones in shape_list:
            with self.subTest(shapes=desc):
                self._check_bucketing(
                    nested_tile_kernel,
                    ref,
                    shape1,
                    shape2,
                    same_in_ones,
                    rtol=1e-5,
                    atol=1e-5,
                    config=Config(block_sizes=[32, 32]),
                )

    @skipIfRefEager("code generation not relevant in ref eager mode")
    @skipIfNotCUDA()
    @patch.object(_compat, "_supports_tensor_descriptor", lambda: False)
    @skipIfTileIR("TileIR does not support block_ptr indexing")
    def test_block_ptr_reduction_none_mode_size1(self) -> None:
        """Test block_ptr + reduction + none mode with size-1 reduction dim."""
        self._check_bucketing(
            reduction_sum_kernel,
            lambda x: x.sum(-1),
            (32, 1),
            (32, 64),
            False,
            modes=("none",),
            config=Config(block_sizes=[32], indexing="block_ptr"),
        )

    @skipIfRefEager("code generation not relevant in ref eager mode")
    @skipIfNotCUDA()
    def test_none_mode_size1_broadcast(self) -> None:
        """Test 'none' mode broadcast: bias is (1, n) but indexed with tiles from (m, n).

        In 'none' mode, size-1 dims are not specialized, so the compiler
        doesn't know at compile time that bias.size(0) == 1 and can't
        generate static broadcast code. The generated kernel must still
        produce correct results via runtime stride-based broadcasting.
        """

        def add_bias(x: torch.Tensor, bias: torch.Tensor, out: torch.Tensor) -> None:
            m, n = x.size()
            block_m = hl.register_block_size(m)
            block_n = hl.register_block_size(n)
            for tile_m in hl.tile(m, block_size=block_m):
                for tile_n in hl.tile(n, block_size=block_n):
                    out[tile_m, tile_n] = x[tile_m, tile_n] + bias[tile_m, tile_n]

        # First verify it works in "ones" mode (where broadcast is generated)
        k_ones = self._make_kernel(
            add_bias, "ones", config=Config(block_sizes=[32, 32])
        )
        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        bias = torch.randn(1, 8, device=DEVICE, dtype=torch.float32)
        out_ones = torch.empty_like(x)
        k_ones(x, bias, out_ones)
        expected = x + bias  # PyTorch broadcasts bias (1, 8) to (4, 8)
        torch.testing.assert_close(out_ones, expected, rtol=1e-4, atol=1e-4)

        # Now test "none" mode — the broadcast code path is NOT taken,
        # so the bias load might not correctly broadcast the size-1 dim.
        k_none = self._make_kernel(
            add_bias, "none", config=Config(block_sizes=[32, 32])
        )
        out_none = torch.empty_like(x)
        k_none(x, bias, out_none)
        torch.testing.assert_close(out_none, expected, rtol=1e-4, atol=1e-4)

        # In "none" mode, different non-zero shapes share the same specialization key.
        key_x = k_none.specialization_key([x, bias, out_none])
        bias_full = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        out_full = torch.empty_like(x)
        key_full = k_none.specialization_key([x, bias_full, out_full])
        self.assertEqual(key_x, key_full)

    @skipIfRefEager("code generation not relevant in ref eager mode")
    @skipIfNotCUDA()
    @patch.object(_compat, "_supports_tensor_descriptor", lambda: False)
    @skipIfTileIR("TileIR does not support block_ptr indexing")
    def test_block_ptr_broadcast_none_mode(self) -> None:
        """Test block_ptr indexing with broadcast in 'none' mode.

        In block_ptr mode, BlockedSubscriptIndexing uses tl.make_block_ptr which
        handles out-of-bounds via boundary_check (zero-padding). For a (1, n) tensor
        with block_shape (tile_m, tile_n), rows beyond the first are padded with 0
        instead of being broadcast from row 0. This tests whether the result is correct.

        Also tests reuse: compile with no broadcast first, reuse with broadcast.
        """

        def add_bias(x: torch.Tensor, bias: torch.Tensor, out: torch.Tensor) -> None:
            m, n = x.size()
            block_m = hl.register_block_size(m)
            block_n = hl.register_block_size(n)
            for tile_m in hl.tile(m, block_size=block_m):
                for tile_n in hl.tile(n, block_size=block_n):
                    out[tile_m, tile_n] = x[tile_m, tile_n] + bias[tile_m, tile_n]

        k = self._make_kernel(
            add_bias, "none", config=Config(block_sizes=[32, 32], indexing="block_ptr")
        )

        # First call: broadcast (bias has size-1 dim 0)
        with self.subTest(case="broadcast"):
            x = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
            bias = torch.randn(1, 8, device=DEVICE, dtype=torch.float32)
            out = torch.empty_like(x)
            k(x, bias, out)
            expected = x + bias  # PyTorch broadcasts bias (1, 8) to (4, 8)
            torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-4)

        # Second call: no broadcast then broadcast (reuse test)
        with self.subTest(case="reuse"):
            k2 = self._make_kernel(
                add_bias,
                "none",
                config=Config(block_sizes=[32, 32], indexing="block_ptr"),
            )
            # First call: no broadcast (bias same shape as x)
            x1 = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
            bias1 = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
            out1 = torch.empty_like(x1)
            k2(x1, bias1, out1)
            torch.testing.assert_close(out1, x1 + bias1, rtol=1e-4, atol=1e-4)

            # Second call: broadcast needed (bias has size-1 dim 0)
            x2 = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
            bias2 = torch.randn(1, 8, device=DEVICE, dtype=torch.float32)
            out2 = torch.empty_like(x2)
            k2(x2, bias2, out2)
            torch.testing.assert_close(out2, x2 + bias2, rtol=1e-4, atol=1e-4)

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


instantiate_parametrized_tests(TestShapeBucketing)


# Shape variations to test 1-ness: (description, m, n)
ALL_SHAPES = [
    ("dim0=1", 1, 64),
    ("dim1=1", 32, 1),
    ("both=1", 1, 1),
    ("normal", 32, 64),
]

# Each example config: (example_name, fn_name, input_fn, ref_fn, shape_list)
# input_fn: callable(m, n) -> tuple of args (m, n are dimensions)
# ref_fn: callable(*args) -> expected output
# shape_list: list of (description, m, n) tuples to test
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

# Examples where "none" mode reuse across shapes is known to fail.
# These kernels call hl.specialize() on shape-derived values (e.g.
# `hl.specialize(n)` in matmul_layernorm for the layer-norm normalized_shape,
# or head_dim in attention), which bakes compile-time constants into the
# generated Triton code.  When a second call arrives with different concrete
# shapes, the specialization key changes (because the specialized variable
# has a new value), so the kernel gets a new BoundKernel instead of reusing
# the first one — even though all *non-specialized* shape components match.
_NONE_MODE_REUSE_SKIP: set[tuple[str, str | None]] = {
    ("matmul_layernorm", "matmul_layernorm"),
    ("attention", "attention"),
}

STATIC_SHAPES_MODES = ["none", "ones", "all"]

# Examples that use tl.dot and need IEEE precision for tight tolerances.
_EXAMPLES_WITH_TL_DOT = {
    "matmul",
    "matmul_split_k",
    "matmul_layernorm",
    "squeeze_and_excitation_net",
    "attention",
}

# Build a lookup table keyed by unique example identifier (fn_name or
# example_name) so we can parametrize test methods by string key.
_EXAMPLE_CONFIGS_BY_KEY: dict[
    str, tuple[str, str | None, object, object, list[tuple[str, int, int]]]
] = {}
for _cfg in EXAMPLE_CONFIGS_WITH_SHAPES:
    _key = _cfg[1] or _cfg[0]
    assert _key not in _EXAMPLE_CONFIGS_BY_KEY, f"duplicate example key: {_key}"
    _EXAMPLE_CONFIGS_BY_KEY[_key] = _cfg

_EXAMPLE_KEYS = tuple(_EXAMPLE_CONFIGS_BY_KEY.keys())


class TestExampleStaticShapes(RefEagerTestBase, TestCase):
    maxDiff = 16384

    @skipIfRefEager("code generation not relevant in ref eager mode")
    @skipIfNotCUDA()
    @parametrize("mode", ("none", "ones", "all"))
    @parametrize("example_key", _EXAMPLE_KEYS)
    def test_example_kernel_across_shape_modes(
        self, mode: str, example_key: str
    ) -> None:
        """Test one example with one static_shapes mode across shape variations."""
        import sys

        from helion._testing import EXAMPLES_DIR
        from helion._testing import import_path

        example_name, fn_name, input_fn, ref_fn, shapes = _EXAMPLE_CONFIGS_BY_KEY[
            example_key
        ]
        for shape_desc, m, n in shapes:
            with self.subTest(shape=shape_desc):
                self._check_example(
                    example_name,
                    fn_name,
                    input_fn,
                    ref_fn,
                    mode,
                    m,
                    n,
                    import_path,
                    EXAMPLES_DIR,
                    sys,
                )

    def _check_example(
        self,
        example_name: str,
        fn_name: str | None,
        input_fn: object,
        ref_fn: object,
        mode: str,
        m: int,
        n: int,
        import_path: object,
        examples_dir: object,
        sys: object,
    ) -> None:
        # Force fresh module import to prevent state leakage between subtests,
        # since import_path caches modules in sys.modules.
        module_cache_key = f"helion._testing.{example_name}"
        sys.modules.pop(module_cache_key, None)
        mod = import_path(examples_dir / f"{example_name}.py")
        kernel_fn = getattr(mod, fn_name or example_name)

        # Clear any hardcoded configs and cached bound kernels from the kernel
        kernel_fn.configs = []
        kernel_fn._bound_kernels = {}

        # Set the static_shapes mode and disable autotuning for faster tests
        kernel_fn.settings.static_shapes = mode
        kernel_fn.settings.autotune_effort = "none"

        if example_name in _EXAMPLES_WITH_TL_DOT:
            kernel_fn.settings.dot_precision = "ieee"

        # Create test inputs with the specified shapes and run
        args = input_fn(m, n)
        result = kernel_fn(*args)

        # Compare with reference
        expected = ref_fn(*args)
        if isinstance(result, tuple) and not isinstance(expected, tuple):
            result = result[0]

        torch.testing.assert_close(
            result,
            expected,
            rtol=1e-4,
            atol=1e-4,
        )

        # Code-generation verification: in "all" mode, no symbolic size vars
        # (like x_size_0, logits_size_1) should appear in the generated code.
        # The pattern _size_\d avoids false positives from block_size_n.
        bound_check = kernel_fn.bind(args)
        if mode == "all":
            code = bound_check.to_triton_code()
            self.assertNotRegex(
                code,
                r"_size_\d",
                f"'all' mode should not have symbolic size vars for "
                f"{example_name}/{fn_name}",
            )
            # "all" mode: strides are hardcoded, no dynamic stride params
            self.assertNotRegex(
                code,
                r"\w+_stride_\d",
                f"'all' mode should not have symbolic stride vars for "
                f"{example_name}/{fn_name}",
            )

        # Code-generation verification for "ones" mode: strides should be
        # passed dynamically when all dimensions > 1 (just like "none" mode).
        if mode == "ones" and m > 1 and n > 1:
            code = bound_check.to_triton_code()
            stride_params = set(re.findall(r"(\w+)\.stride\(", code))
            multi_dim_tensor_count = sum(
                1 for a in args if isinstance(a, torch.Tensor) and a.dim() > 1
            )
            min_stride_params = max(1, multi_dim_tensor_count)
            self.assertGreaterEqual(
                len(stride_params),
                min_stride_params,
                f"'ones' mode should pass dynamic strides for at least "
                f"{min_stride_params} multi-dim tensor(s) in "
                f"{example_name}/{fn_name}, but only found .stride() calls "
                f"for {len(stride_params)} parameters: {stride_params}",
            )

        # Code-generation verification for "none" mode: verify the generated
        # code is shape-agnostic by checking that tensor strides are passed
        # dynamically (via .stride() calls in the wrapper).
        if mode == "none":
            code = bound_check.to_triton_code()
            stride_params = set(re.findall(r"(\w+)\.stride\(", code))
            # 1D contiguous tensors have trivially known stride (1,) and
            # may not require .stride() calls, so count only multi-dim tensors.
            multi_dim_tensor_count = sum(
                1 for a in args if isinstance(a, torch.Tensor) and a.dim() > 1
            )
            min_stride_params = max(1, multi_dim_tensor_count)
            self.assertGreaterEqual(
                len(stride_params),
                min_stride_params,
                f"'none' mode should pass dynamic strides for at least "
                f"{min_stride_params} multi-dim tensor(s) in "
                f"{example_name}/{fn_name}, but only found .stride() calls "
                f"for {len(stride_params)} parameters: {stride_params}",
            )

        # Verify shape-agnosticism in "none" mode: different non-zero shapes
        # must produce the same specialization key (tensor bucketing).
        if mode == "none":
            args2 = input_fn(m + 3, n + 5)
            key1 = kernel_fn.specialization_key(args)
            key2 = kernel_fn.specialization_key(args2)
            self.assertEqual(
                key1,
                key2,
                f"In 'none' mode, shapes ({m},{n}) and ({m + 3},{n + 5}) "
                f"produced different specialization keys for "
                f"{example_name}/{fn_name}",
            )
            if (example_name, fn_name) not in _NONE_MODE_REUSE_SKIP:
                # Verify correctness of reuse
                result2 = kernel_fn(*args2)
                expected2 = ref_fn(*args2)
                if isinstance(result2, tuple) and not isinstance(expected2, tuple):
                    result2 = result2[0]
                torch.testing.assert_close(
                    result2,
                    expected2,
                    rtol=1e-4,
                    atol=1e-4,
                )
                # When bound kernels differ, it must be because
                # hl.specialize() variables took different concrete values.
                bound1 = kernel_fn.bind(args)
                bound2 = kernel_fn.bind(args2)
                if bound1 is not bound2:
                    self.assertTrue(
                        bound1.env.specialized_vars,
                        f"In 'none' mode with no specialized vars, expected "
                        f"same bound kernel for shapes ({m},{n}) and "
                        f"({m + 3},{n + 5}) for {example_name}/{fn_name}",
                    )

        # Remove the module from sys.modules to prevent test contamination:
        # the modified settings (static_shapes, autotune_effort, etc.) would
        # leak into later tests that import the same example module.
        sys.modules.pop(module_cache_key, None)


instantiate_parametrized_tests(TestExampleStaticShapes)


if __name__ == "__main__":
    unittest.main()
