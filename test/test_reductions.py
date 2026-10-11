from __future__ import annotations

from typing import TYPE_CHECKING
import unittest
from unittest.mock import patch

import torch

import helion
from helion._compat import get_triton_version
from helion._testing import DEVICE
from helion._testing import HALF_DTYPE
from helion._testing import RefEagerTestBase
from helion._testing import TestCase
from helion._testing import _get_backend
from helion._testing import code_and_output
from helion._testing import onlyBackends
from helion._testing import skipIfMetal
from helion._testing import skipIfNotCUDA
from helion._testing import skipIfPallas
from helion._testing import skipIfRefEager
from helion._testing import skipIfRocm
from helion._testing import skipIfTileIR
from helion._testing import skipUnlessBackends
from helion._testing import skipUnlessCuteAvailable
from helion._testing import skipUnlessTensorDescriptor
from helion._testing import xfailIfPallasTpu
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Callable


@helion.kernel()
def sum_kernel(x: torch.Tensor) -> torch.Tensor:
    n, _m = x.size()
    out = torch.empty(
        [n],
        dtype=x.dtype,
        device=x.device,
    )
    for tile_n in hl.tile(n):
        out[tile_n] = x[tile_n, :].sum(-1)
    return out


@helion.kernel()
def sum_kernel_keepdims(x: torch.Tensor) -> torch.Tensor:
    _n, m = x.size()
    out = torch.empty(
        [1, m],
        dtype=x.dtype,
        device=x.device,
    )
    for tile_m in hl.tile(m):
        out[:, tile_m] = x[:, tile_m].sum(0, keepdim=True)
    return out


@helion.kernel(config={"block_sizes": [1]})
def reduce_kernel(
    x: torch.Tensor, fn: Callable[[torch.Tensor], torch.Tensor], out_dtype=torch.float32
) -> torch.Tensor:
    n, _m = x.size()
    out = torch.empty(
        [n],
        dtype=out_dtype,
        device=x.device,
    )
    for tile_n in hl.tile(n):
        out[tile_n] = fn(x[tile_n, :], dim=-1)
    return out


@helion.kernel(static_shapes=True)
def tile_row_sums(x: torch.Tensor, buf: torch.Tensor) -> torch.Tensor:
    """Stores the 64 row sums of ``buf[:, :]`` at the positions of a 64-wide
    tile, so the row dim is a full slice that a store places at the tile."""
    out = torch.empty([x.size(0)], dtype=torch.float32, device=x.device)
    for tile in hl.tile(x.size(0), block_size=64):
        out[tile] = buf[:, :].sum(-1)
    return out


@onlyBackends(["triton", "cute", "pallas", "metal"])
class TestReductions(RefEagerTestBase, TestCase):
    @skipUnlessBackends(["cute", "triton"])
    def test_factored_reductions_preserve_computed_coordinates(self):
        @helion.kernel(static_shapes=True)
        def factored(
            x: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            width = hl.specialize(x.size(1))
            assert width == 64
            maxima = torch.empty((x.size(0), 4), device=x.device, dtype=torch.int64)
            minima = torch.empty_like(maxima)
            selected = torch.empty_like(x)
            columns = torch.arange(16, device=x.device)
            for row in hl.tile(x.size(0)):
                values = x[row, :] * 3 + 1
                groups = values.reshape(values.size(0), 4, 16)
                high = groups.argmax(-1)
                low = groups.argmin(-1)
                retained = torch.where(
                    columns[None, None, :] == high[:, :, None],
                    groups,
                    torch.zeros_like(groups),
                ).reshape(values.size(0), 64)
                # Reuse the same producer with both flat and factored maps.
                selected[row, :] = retained + values - values.amax(-1, keepdim=True)
                maxima[row, :] = high
                minima[row, :] = low
            return maxima, minima, selected

        for dtype in (torch.float32, torch.float64, torch.int64):
            for block_rows in (1, 8):
                with self.subTest(dtype=dtype, block_rows=block_rows):
                    x = torch.randint(-100, 100, (17, 128), device=DEVICE).to(dtype)[
                        :, ::2
                    ]
                    if dtype == torch.int64:
                        x = x + 2**53
                    values = x * 3 + 1
                    groups = values.reshape(17, 4, 16)
                    high, low = groups.argmax(-1), groups.argmin(-1)
                    retained = torch.zeros_like(groups).scatter(
                        -1, high[:, :, None], groups.amax(-1, keepdim=True)
                    )
                    expected = (
                        retained.reshape(17, 64)
                        + values
                        - values.amax(-1, keepdim=True)
                    )
                    code, actual = code_and_output(
                        factored, (x,), block_sizes=[block_rows]
                    )
                    if _get_backend() == "cute":
                        self.assertIn("fragment_selected_index", code)
                    torch.testing.assert_close(
                        actual, (high, low, expected), rtol=0, atol=0
                    )

    @skipUnlessBackends(["cute", "triton"])
    def test_carried_argreduce_preserves_input_dtype_and_first_tie(self):
        @helion.kernel(static_shapes=True)
        def extrema(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            low_out = torch.empty((x.size(0),), device=x.device, dtype=torch.int64)
            high_out = torch.empty_like(low_out)
            for row in hl.tile(x.size(0)):
                values = x[row, :]
                low = hl.full([row], 0, dtype=torch.int64)
                high = hl.full([row], 0, dtype=torch.int64)
                for _step in range(2):
                    low = values.argmin(-1)
                    high = values.argmax(-1)
                    values = -values
                low_out[row] = low
                high_out[row] = high
            return low_out, high_out

        for dtype in (torch.float16, torch.float32, torch.float64, torch.int64):
            with self.subTest(dtype=dtype):
                if (
                    dtype == torch.int64
                    and _get_backend() == "tileir"
                    and get_triton_version().release == (3, 6, 0)
                ):
                    self.skipTest(
                        "TileIR 3.6.0 (0d9283bf) miscompiles carried int64 argreduce"
                    )
                x = torch.tensor(
                    [
                        [1, 4, 4, -2, -2, 0, 3],
                        [0, 0, 0, 0, 0, 0, 0],
                        [3, 1, 2, 3, 1, 2, 3],
                        [2, 3, 1, 3, 2, 1, 3],
                        [4, 4, -2, 4, -2, 4, -2],
                    ],
                    device=DEVICE,
                    dtype=dtype,
                )
                if dtype.is_floating_point:
                    x[0, 0] = float("nan")
                    x[2, 3] = float("nan")
                    x[3, -1] = float("nan")
                    x[4, 1] = x[4, 5] = float("nan")
                    x[1, 0] = -0.0
                else:
                    x = x + 2**53
                code, actual = code_and_output(extrema, (x,), block_sizes=[8])
                if _get_backend() == "cute":
                    self.assertIn("fragment_selected_index", code)
                torch.testing.assert_close(actual, ((-x).argmin(-1), (-x).argmax(-1)))

    @onlyBackends(["cute", "triton"])
    def test_integer_reductions_use_promoted_accumulators(self):
        @helion.kernel(static_shapes=True)
        def reduce_integer(x: torch.Tensor, product: hl.constexpr) -> torch.Tensor:
            out = torch.empty((x.size(0),), device=x.device, dtype=torch.int64)
            for row in hl.tile(x.size(0)):
                if product:
                    out[row] = x[row, :].prod(-1)
                else:
                    out[row] = x[row, :].sum(-1)
            return out

        cases = [
            (torch.bool, False, 97, 1),
            (torch.int8, False, 97, 3),
            (torch.int8, True, 9, 2),
        ]
        for dtype, product, width, value in cases:
            x = torch.full((17, width), value, device=DEVICE, dtype=dtype)
            expected = x.prod(-1) if product else x.sum(-1)
            for reduction_loop in (None, 4):
                with self.subTest(dtype=dtype, product=product, loop=reduction_loop):
                    _, output = code_and_output(
                        reduce_integer,
                        (x, product),
                        block_sizes=[1],
                        reduction_loops=[reduction_loop],
                    )
                    torch.testing.assert_close(output, expected)

    @onlyBackends(["cute", "triton"])
    def test_reduction_explicit_accumulator_dtype(self):
        @helion.kernel(static_shapes=True)
        def sum_as(x: torch.Tensor, dtype: hl.constexpr) -> torch.Tensor:
            out = torch.empty((x.size(0),), device=x.device, dtype=dtype)
            for row in hl.tile(x.size(0)):
                out[row] = x[row, :].sum(-1, dtype=dtype)
            return out

        integer_input = torch.full((17, 97), 3, device=DEVICE, dtype=torch.int8)
        # Exact in FP64, but FP32 accumulation loses the terms of magnitude 1.
        floating_input = (
            torch.tensor([1e8, 1, -1e8, 1] * 8, device=DEVICE, dtype=torch.float32)[
                None, :
            ]
            .expand(17, -1)
            .contiguous()
        )
        for x, dtype in [
            (integer_input, torch.int32),
            (floating_input, torch.float64),
        ]:
            for reduction_loop in (None, 4):
                with self.subTest(dtype=dtype, loop=reduction_loop):
                    _, output = code_and_output(
                        sum_as,
                        (x, dtype),
                        block_sizes=[1],
                        reduction_loops=[reduction_loop],
                    )
                    torch.testing.assert_close(
                        output, x.sum(-1, dtype=dtype), rtol=0, atol=0
                    )

    @onlyBackends(["cute", "triton"])
    def test_tiled_boolean_sum_uses_promoted_accumulator(self):
        @helion.kernel(static_shapes=True)
        def tile_count(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x, dtype=torch.int64)
            for row in hl.tile(x.size(0)):
                for col in hl.tile(x.size(1), block_size=64):
                    out[row, col] = x[row, col].sum(-1)[:, None]
            return out

        x = torch.ones((17, 128), device=DEVICE, dtype=torch.bool)
        for rows_per_block in (1, 16):
            with self.subTest(rows_per_block=rows_per_block):
                _, output = code_and_output(
                    tile_count, (x,), block_sizes=[rows_per_block]
                )
                torch.testing.assert_close(output, torch.full_like(output, 64))

    @onlyBackends(["cute"])
    def test_carried_min_max_preserve_nan(self):
        @helion.kernel(static_shapes=True)
        def extrema(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            low_out = torch.empty((x.size(0),), device=x.device, dtype=x.dtype)
            high_out = torch.empty_like(low_out)
            for row in hl.tile(x.size(0)):
                values = x[row, :]
                low = hl.full([row], 0.0, dtype=x.dtype)
                high = hl.full([row], 0.0, dtype=x.dtype)
                for _step in range(2):
                    low = values.amin(-1)
                    high = values.amax(-1)
                    values = values + 1
                low_out[row] = low
                high_out[row] = high
            return low_out, high_out

        x = torch.randn(5, 8, device=DEVICE)
        x[0, 0] = float("nan")
        x[1, 4] = float("nan")
        x[2, 7] = float("nan")
        x[3, 0] = -float("inf")
        x[3, 7] = float("inf")
        code, (low, high) = code_and_output(extrema, (x,), block_sizes=[8])
        self.assertIn("fragment_reduce_index", code)
        torch.testing.assert_close(low, (x + 1).amin(-1), equal_nan=True)
        torch.testing.assert_close(high, (x + 1).amax(-1), equal_nan=True)

    @onlyBackends(["cute"])
    def test_tile_reduction_after_full_slice_reduction(self):
        @helion.kernel(static_shapes=True)
        def filtered_max(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty((x.size(0),), device=x.device, dtype=x.dtype)
            for row in hl.tile(x.size(0)):
                threshold = x[row, :].amax(-1) * 0.3
                best = hl.full([row], -float("inf"), dtype=x.dtype)
                for col in hl.tile(x.size(1)):
                    values = x[row, col]
                    valid = torch.where(values >= threshold[:, None], values, 0.0)
                    best = torch.maximum(best, valid.amax(-1))
                out[row] = best
            return out

        for rows_per_block, reduction_size in [(16, 8), (32, 4)]:
            with self.subTest(rows_per_block=rows_per_block):
                x = torch.rand(65, 8, device=DEVICE)
                _, output = code_and_output(
                    filtered_max,
                    (x,),
                    block_sizes=[rows_per_block, 8],
                    reduction_loops=[reduction_size],
                )
                torch.testing.assert_close(output, x.amax(-1))

    @onlyBackends(["cute", "triton"])
    def test_tiled_argreduce_returns_local_indices(self):
        @helion.kernel(static_shapes=True)
        def tile_extrema(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            minima = torch.empty_like(x, dtype=torch.int64)
            maxima = torch.empty_like(x, dtype=torch.int64)
            for row in hl.tile(x.size(0)):
                for col in hl.tile(x.size(1), block_size=32):
                    values = x[row, col]
                    minima[row, col] = values.argmin(-1)[:, None]
                    maxima[row, col] = values.argmax(-1)[:, None]
            return minima, maxima

        x = (
            torch.arange(256, device=DEVICE, dtype=torch.float32)[None, :]
            .expand(3, -1)
            .contiguous()
        )
        _, (minima, maxima) = code_and_output(tile_extrema, (x,), block_sizes=[1])
        torch.testing.assert_close(minima, torch.zeros_like(minima))
        torch.testing.assert_close(maxima, torch.full_like(maxima, 31))

    @onlyBackends(["cute"])
    def test_runtime_loop_reductions_preserve_tensor_carries(self):
        @helion.kernel(static_shapes=True)
        def normalize_twice(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for row in hl.tile(x.size(0)):
                values = x[row, :]
                for _step in range(2):
                    low = values.amin(-1, keepdim=True)
                    high = values.amax(-1, keepdim=True)
                    scaled = (values - low) / (1 + high - low)
                    product = (1 + scaled * 0.001).prod(-1, keepdim=True)
                    values = scaled / product
                out[row, :] = values
            return out

        x = torch.randn(17, 64, device=DEVICE)
        expected = x
        for _step in range(2):
            low = expected.amin(-1, keepdim=True)
            high = expected.amax(-1, keepdim=True)
            scaled = (expected - low) / (1 + high - low)
            expected = scaled / (1 + scaled * 0.001).prod(-1, keepdim=True)
        code, output = code_and_output(normalize_twice, (x,), block_sizes=[32])
        self.assertIn("fragment_reduce_index", code)
        torch.testing.assert_close(output, expected, rtol=2e-5, atol=2e-6)

    @onlyBackends(["cute"])
    def test_reduction_with_independent_output_axis(self):
        @helion.kernel(static_shapes=True)
        def sum_outer(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.empty((x.size(0), y.numel()), device=x.device, dtype=x.dtype)
            for row in hl.tile(x.size(0)):
                total = x[row, :].sum(-1, keepdim=True)
                out[row, :] = total + y[:]
            return out

        for n, k, rows_per_block, reduction_size in [(128, 2, 32, 32), (256, 8, 1, 64)]:
            with self.subTest(n=n, k=k, rows_per_block=rows_per_block):
                x = torch.randn(17, n, device=DEVICE)
                y = torch.arange(k, device=DEVICE, dtype=x.dtype)
                code, output = code_and_output(
                    sum_outer,
                    (x, y),
                    block_sizes=[rows_per_block],
                    reduction_loops=[reduction_size],
                )
                torch.testing.assert_close(
                    output, x.sum(-1, keepdim=True) + y, rtol=1e-5, atol=1e-5
                )
                if reduction_size > 32:
                    self.assertIn("_cute_grouped_reduce_shared_two_stage", code)
                    self.assertNotIn("threads_in_group=64", code)

    @skipIfPallas("non-power-of-2 reduction dims not supported on Pallas")
    def test_strided_threaded_reduction_non_sum_ops(self):
        """Exercise strided threaded block reduction lowering for non-sum ops."""

        @helion.kernel(autotune_effort="none")
        def max_kernel(x: torch.Tensor) -> torch.Tensor:
            n, _m = x.size()
            out = torch.empty([n], dtype=x.dtype, device=x.device)
            for tile_n in hl.tile(n):
                out[tile_n] = torch.amax(x[tile_n, :], dim=-1)
            return out

        @helion.kernel(autotune_effort="none")
        def min_kernel(x: torch.Tensor) -> torch.Tensor:
            n, _m = x.size()
            out = torch.empty([n], dtype=x.dtype, device=x.device)
            for tile_n in hl.tile(n):
                out[tile_n] = torch.amin(x[tile_n, :], dim=-1)
            return out

        @helion.kernel(autotune_effort="none")
        def prod_kernel(x: torch.Tensor) -> torch.Tensor:
            n, _m = x.size()
            out = torch.empty([n], dtype=x.dtype, device=x.device)
            for tile_n in hl.tile(n):
                out[tile_n] = torch.prod(x[tile_n, :], dim=-1)
            return out

        x = torch.rand([32, 33], device=DEVICE, dtype=torch.float32) + 0.5
        cases = [
            (max_kernel, lambda t: torch.amax(t, dim=-1)),
            (min_kernel, lambda t: torch.amin(t, dim=-1)),
            (prod_kernel, lambda t: torch.prod(t, dim=-1)),
        ]
        for kernel, ref_fn in cases:
            with self.subTest(kernel=kernel.__name__):
                _code, out = code_and_output(kernel, (x,), block_size=8)
                torch.testing.assert_close(out, ref_fn(x), rtol=1e-4, atol=1e-4)

    @skipIfPallas("cross-warp shared-memory reduction not supported on Pallas")
    def test_cross_warp_reduction_non_sum_ops(self):
        """Exercise shared-memory (two-stage) strided reduction for non-sum ops.

        Using block_sizes=[1, 128] keeps the outer M-block at one row per
        CTA (so the warp-per-row layout's ``m_threads >= 2`` check fails)
        while the inner reduction spans 128 threads (4 warps cooperating
        on a single row), giving group_span=128 (>32 and %32==0) which
        triggers the shared two-stage reduction path on CuTe.
        """

        @helion.kernel(autotune_effort="none")
        def max_kernel(x: torch.Tensor) -> torch.Tensor:
            n, m = x.size()
            out = torch.empty([n], dtype=x.dtype, device=x.device)
            for tile_n in hl.tile(n):
                row_max = hl.full([tile_n], float("-inf"), dtype=x.dtype)
                for tile_m in hl.tile(m):
                    row_max = torch.maximum(
                        row_max, torch.amax(x[tile_n, tile_m], dim=1)
                    )
                out[tile_n] = row_max
            return out

        @helion.kernel(autotune_effort="none")
        def min_kernel(x: torch.Tensor) -> torch.Tensor:
            n, m = x.size()
            out = torch.empty([n], dtype=x.dtype, device=x.device)
            for tile_n in hl.tile(n):
                row_min = hl.full([tile_n], float("inf"), dtype=x.dtype)
                for tile_m in hl.tile(m):
                    row_min = torch.minimum(
                        row_min, torch.amin(x[tile_n, tile_m], dim=1)
                    )
                out[tile_n] = row_min
            return out

        @helion.kernel(autotune_effort="none")
        def prod_kernel(x: torch.Tensor) -> torch.Tensor:
            n, m = x.size()
            out = torch.empty([n], dtype=x.dtype, device=x.device)
            for tile_n in hl.tile(n):
                row_prod = hl.full([tile_n], 1.0, dtype=x.dtype)
                for tile_m in hl.tile(m):
                    row_prod = row_prod * torch.prod(x[tile_n, tile_m], dim=1)
                out[tile_n] = row_prod
            return out

        x = torch.rand([128, 128], device=DEVICE, dtype=torch.float32) + 0.5
        cases = [
            (max_kernel, lambda t: torch.amax(t, dim=-1)),
            (min_kernel, lambda t: torch.amin(t, dim=-1)),
            (prod_kernel, lambda t: torch.prod(t, dim=-1)),
        ]
        for kernel, ref_fn in cases:
            with self.subTest(kernel=kernel.__name__):
                code, out = code_and_output(kernel, (x,), block_sizes=[1, 128])
                torch.testing.assert_close(out, ref_fn(x), rtol=1e-4, atol=1e-4)
                if _get_backend() == "cute":
                    self.assertIn("_cute_grouped_reduce_shared_two_stage", code)

    @skipIfMetal(
        "Metal SIMD reductions need the reduced dim to be the fastest-varying thread axis"
    )
    def test_2d_tile_inner_dim_reduction_to_scalar(self):
        """Reduce the inner dim of a single 2D ``hl.tile([o, d])`` into a per-row scalar.

        Unlike ``test_cross_warp_reduction_non_sum_ops`` (which uses a separate
        inner ``hl.tile(m)`` loop), this tiles both dims together in ONE
        ``hl.tile([o, d])`` and reduces the inner dim (``dim=-1``) into a scalar
        ``out[tile_o]``. That routes to ``BlockReductionStrategy`` with a runtime
        lane loop over the block-resident inner dim. This form previously emitted
        a per-thread partial with NO cross-thread reduction (the stride-32 reduce
        group is spread across warps), so the threads owning a row raced to store
        their partial sums -> silently wrong output. The fix folds the lane loop
        into a per-thread partial and then combines across warps via
        ``_cute_grouped_reduce_shared_two_stage`` (group_span=128, >32 and %32==0).

        ``d_block`` is forced to the full power-of-2 extent so the reduction stays
        block-resident (the well-posed form). D=128 makes the reduce group span
        4 warps, exercising the cross-warp two-stage path on CuTe.
        """

        @helion.kernel(static_shapes=True, autotune_effort="none")
        def sum_kernel(w: torch.Tensor) -> torch.Tensor:
            o, d = w.shape
            d = hl.specialize(d)
            out = torch.empty([o], dtype=torch.float32, device=w.device)
            d_block = hl.register_block_size(
                helion.next_power_of_2(d), helion.next_power_of_2(d)
            )
            for tile_o, tile_d in hl.tile([o, d], block_size=[None, d_block]):
                out[tile_o] = torch.sum(w[tile_o, tile_d].to(torch.float32), dim=-1)
            return out

        @helion.kernel(static_shapes=True, autotune_effort="none")
        def amax_kernel(w: torch.Tensor) -> torch.Tensor:
            o, d = w.shape
            d = hl.specialize(d)
            out = torch.empty([o], dtype=torch.float32, device=w.device)
            d_block = hl.register_block_size(
                helion.next_power_of_2(d), helion.next_power_of_2(d)
            )
            for tile_o, tile_d in hl.tile([o, d], block_size=[None, d_block]):
                out[tile_o] = torch.amax(w[tile_o, tile_d].to(torch.float32), dim=-1)
            return out

        w = torch.randn([512, 128], device=DEVICE, dtype=torch.float32)
        cases = [
            (sum_kernel, lambda t: t.sum(-1)),
            (amax_kernel, lambda t: t.amax(-1)),
        ]
        for kernel, ref_fn in cases:
            with self.subTest(kernel=kernel.__name__):
                code, out = code_and_output(kernel, (w,))
                torch.testing.assert_close(out, ref_fn(w), rtol=1e-4, atol=1e-4)
                if _get_backend() == "cute":
                    self.assertIn("_cute_grouped_reduce_shared_two_stage", code)

    def test_sum_constant_inner_dim(self):
        """Sum over a known-constant inner dimension (e.g., 2) should work.

        This exercises constant reduction sizes in Inductor lowering.
        """

        @helion.kernel(static_shapes=True)
        def sum_const_inner(x: torch.Tensor) -> torch.Tensor:
            m, _n = x.size()
            out = torch.empty([m], dtype=x.dtype, device=x.device)
            for tile_m in hl.tile(m):
                out[tile_m] = x[tile_m, :].sum(-1)
            return out

        x = torch.randn([32, 2], device=DEVICE)
        code, out = code_and_output(sum_const_inner, (x,), block_size=16)
        torch.testing.assert_close(out, x.sum(-1), rtol=1e-4, atol=1e-4)

    @skipIfPallas("complex layernorm with fp16, not relevant to Pallas")
    @skipIfRefEager("Does not call assert_close")
    @skipIfMetal("hl.arange needs a Metal prims.iota lowering")
    def test_broken_layernorm(self):
        @helion.kernel(autotune_effort="none")
        def layer_norm_fwd(
            x: torch.Tensor,
            weight: torch.Tensor,
            bias: torch.Tensor,
            eps: float = 1e-5,
        ) -> torch.Tensor:
            m, n = x.size()
            out = torch.empty([m, n], dtype=torch.float16, device=x.device)
            hl.specialize(n)
            for tile_m in hl.tile(m):
                acc = x[tile_m, :].to(torch.float32)
                mean = hl.full([n], 0.0, acc.dtype)
                count = hl.arange(0, acc.shape[1], 1)
                delta = acc - mean
                mean = delta / count[None, :]
                delta2 = acc - mean.sum(-1)[:, None]
                m2 = delta * delta2
                var = m2 / n
                normalized = (acc - mean) * torch.rsqrt(var + eps)
                acc = normalized * (weight[:].to(torch.float32)) + (
                    bias[:].to(torch.float32)
                )
                out[tile_m, :] = acc
            return out

        args = (
            torch.ones(2, 2, device=DEVICE),
            torch.ones(2, device=DEVICE),
            torch.ones(2, device=DEVICE),
        )
        code_and_output(layer_norm_fwd, args)
        # results are nan due to division by zero, this kernel is broken

    def test_sum(self):
        args = (torch.randn([512, 512], device=DEVICE),)
        code, output = code_and_output(sum_kernel, args, block_size=1)
        torch.testing.assert_close(output, args[0].sum(-1), rtol=1e-04, atol=1e-04)

    def test_keepdim_scalar_reduction_broadcast(self):
        @helion.kernel(autotune_effort="none")
        def center_by_tile_mean(x: torch.Tensor) -> torch.Tensor:
            (n,) = x.size()
            out = torch.empty([n], dtype=torch.float32, device=x.device)
            for tile_n in hl.tile(n):
                vals = x[tile_n].to(torch.float32)
                mean = torch.mean(vals, dim=-1, keepdim=True)
                out[tile_n] = vals - mean
            return out

        x = ((torch.arange(128, device=DEVICE) % 2) * 2).to(HALF_DTYPE)
        _code, output = code_and_output(center_by_tile_mean, (x,), block_size=32)
        expected = x.float() - 1.0
        torch.testing.assert_close(output, expected, rtol=1e-3, atol=1e-3)

    @skipIfRefEager("block sizes are not applied in ref mode: no tile is padded")
    def test_mean_over_a_padded_tile_divides_by_the_tiles_elements(self):
        """A block wider than the dim, or the last tile of a dim the block does not divide, holds fewer elements than the block; the masked ones are out of the sum and must not count in the mean's divisor."""

        def tile_means_summed(x: torch.Tensor) -> torch.Tensor:
            b, m = x.size()
            out = torch.empty([b], dtype=x.dtype, device=x.device)
            for tile_b in hl.tile(b):
                acc = hl.zeros([tile_b], dtype=x.dtype)
                for tile_m in hl.tile(m):
                    acc = acc + x[tile_b, tile_m].mean(dim=1)
                out[tile_b] = acc
            return out

        x = torch.randn([4, 100], device=DEVICE)
        for static_shapes in (True, False):
            kernel = helion.kernel(
                tile_means_summed, autotune_effort="none", static_shapes=static_shapes
            )
            _code, output = code_and_output(kernel, (x,), block_sizes=[1, 128])
            torch.testing.assert_close(output, x.mean(dim=1), rtol=1e-4, atol=1e-4)
            # A backend may widen the block (Pallas tiles the lane dim by 128):
            # the expected per-tile means follow the block the kernel ran with.
            config = kernel.bind((x,))._normalized_config_copy(
                helion.Config(block_sizes=[1, 32])
            )
            width = config.block_sizes[1]
            tile_means = sum(
                x[:, start : start + width].mean(dim=1)
                for start in range(0, 100, width)
            )
            _code, output = code_and_output(kernel, (x,), block_sizes=[1, 32])
            torch.testing.assert_close(output, tile_means, rtol=1e-4, atol=1e-4)

    @skipUnlessTensorDescriptor("Tensor descriptor support is required")
    def test_sum_keepdims(self):
        args = (torch.randn([512, 512], device=DEVICE),)
        code, output = code_and_output(
            sum_kernel_keepdims, args, block_size=16, indexing="tensor_descriptor"
        )
        torch.testing.assert_close(
            output, args[0].sum(0, keepdim=True), rtol=1e-04, atol=1e-04
        )

    @skipUnlessTensorDescriptor("Tensor descriptor support is required")
    def test_argmin_argmax(self):
        for fn in (torch.argmin, torch.argmax):
            args = (torch.randn([512, 512], device=DEVICE), fn, torch.int64)
            code, output = code_and_output(
                reduce_kernel, args, block_size=16, indexing="tensor_descriptor"
            )
            torch.testing.assert_close(output, args[1](args[0], dim=-1))

    @skipIfPallas("Pallas TPU argreduce cannot write int64 keepdim outputs")
    def test_argmin_argmax_keepdim(self):
        @helion.kernel(autotune_effort="none")
        def argmax_keepdim_kernel(x: torch.Tensor) -> torch.Tensor:
            n, m = x.size()
            out = torch.empty([n, 1], dtype=torch.int64, device=x.device)
            for tile_n in hl.tile(n):
                out[tile_n, :] = torch.argmax(x[tile_n, :], dim=1, keepdim=True)
            return out

        @helion.kernel(autotune_effort="none")
        def argmin_keepdim_kernel(x: torch.Tensor) -> torch.Tensor:
            n, m = x.size()
            out = torch.empty([n, 1], dtype=torch.int64, device=x.device)
            for tile_n in hl.tile(n):
                out[tile_n, :] = torch.argmin(x[tile_n, :], dim=1, keepdim=True)
            return out

        x = torch.randn([32, 33], device=DEVICE)
        _, output = code_and_output(argmax_keepdim_kernel, (x,), block_size=8)
        torch.testing.assert_close(output, torch.argmax(x, dim=1, keepdim=True))
        _, output = code_and_output(argmin_keepdim_kernel, (x,), block_size=8)
        torch.testing.assert_close(output, torch.argmin(x, dim=1, keepdim=True))

    @skipIfPallas("Pallas TPU argreduce cannot write int64 scalar outputs")
    def test_argmin_argmax_dim_none(self):
        @helion.kernel(autotune_effort="none")
        def reduce_all_kernel(
            x: torch.Tensor, fn: Callable[[torch.Tensor], torch.Tensor]
        ) -> torch.Tensor:
            (n,) = x.size()
            out = torch.empty([1], dtype=torch.int64, device=x.device)
            for tile_n in hl.tile(n):
                out[0] = fn(x[tile_n])
            return out

        x = torch.randn([16], device=DEVICE)
        for fn in (torch.argmin, torch.argmax):
            with self.subTest(fn=f"{fn.__name__}_scalar"):
                _, output = code_and_output(reduce_all_kernel, (x, fn), block_size=16)
                torch.testing.assert_close(output, fn(x).reshape(1))

    @skipUnlessTensorDescriptor("Tensor descriptor support is required")
    def test_reduction_functions(self):
        for reduction_loop in (None, 16):
            for block_size in (1, 16):
                for indexing in ("tensor_descriptor", "pointer"):
                    for fn in (
                        torch.amax,
                        torch.amin,
                        torch.prod,
                        torch.sum,
                        torch.mean,
                    ):
                        args = (torch.randn([512, 512], device=DEVICE), fn)
                        _, output = code_and_output(
                            reduce_kernel,
                            args,
                            block_size=block_size,
                            indexing=indexing,
                            reduction_loop=reduction_loop,
                        )
                        torch.testing.assert_close(
                            output, fn(args[0], dim=-1), rtol=1e-3, atol=1e-3
                        )

    @skipUnlessTensorDescriptor("Tensor descriptor support is required")
    def test_mean(self):
        args = (torch.randn([512, 512], device=DEVICE), torch.mean, torch.float32)
        self.assertExpectedJournal(reduce_kernel.bind(args)._debug_str())
        code, output = code_and_output(
            reduce_kernel, args, block_size=8, indexing="tensor_descriptor"
        )
        torch.testing.assert_close(output, args[1](args[0], dim=-1))

    @skipIfMetal("Metal has no aten.view lowering")
    def test_reduction_dim_pinned_by_literal_reshape_stays_persistent(self):
        """A reshape to a literal size guards the reduction dim's symbol to it.

        ``x[tile, :].reshape(tile, 1, n)`` with ``n`` static turns the full
        slice's reduction dim into the constant ``n``, which the reduction
        roller cannot tell from other dims of that size: rolled, it moved only
        the store into the loop and left the load outside, whose ``:`` then
        resolved to the row tile when ``n`` equals the row extent.  Such a
        reduction dim must stay persistent.
        """

        @helion.kernel(static_shapes=True)
        def unit_reshape(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            m, n = x.shape
            out = torch.empty([m, 1, n], device=x.device, dtype=x.dtype)
            sums = torch.empty([m, 1], device=x.device, dtype=x.dtype)
            for tile in hl.tile(m):
                out[tile, :, :] = x[tile, :].reshape(tile, 1, n) * 2
                sums[tile, :] = x[tile, :].reshape(tile, 1, n).sum(-1)
            return out, sums

        x = torch.randn([64, 64], device=DEVICE)
        bound = unit_reshape.bind((x,))
        self.assertEqual(bound.env.config_spec.reduction_loops.valid_block_ids(), [])
        for block_size in (16, 32, 64):
            with self.subTest(block_size=block_size):
                _code, (out, sums) = code_and_output(
                    unit_reshape, (x,), block_size=block_size
                )
                torch.testing.assert_close(out, x[:, None, :] * 2)
                torch.testing.assert_close(sums, x.sum(-1, keepdim=True))

    def test_sum_looped(self):
        args = (torch.randn([512, 512], device=DEVICE),)
        code, output = code_and_output(
            sum_kernel, args, block_size=1, reduction_loop=64
        )
        torch.testing.assert_close(output, args[0].sum(-1), rtol=1e-04, atol=1e-04)

    @skipIfPallas("Pallas does not support broadcasting on store")
    def test_broadcast_store_looped_reduction(self):
        @helion.kernel(autotune_effort="none")
        def sum_and_broadcast(
            x: torch.Tensor, bias: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            m, n = x.size()
            s = torch.empty([m], dtype=x.dtype, device=x.device)
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m in hl.tile(m):
                s[tile_m] = torch.sum(x[tile_m, :], dim=-1)
                out[tile_m, :] = bias[tile_m].unsqueeze(-1)
            return s, out

        x = torch.randn(16, 64, device=DEVICE)
        bias = torch.arange(16, dtype=torch.float32, device=DEVICE)
        _, (s, out) = code_and_output(
            sum_and_broadcast, (x, bias), block_size=16, reduction_loop=16
        )
        torch.testing.assert_close(s, x.sum(-1), rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(out, bias[:, None].expand(16, 64))

    @skipUnlessTensorDescriptor("Tensor descriptor support is required")
    def test_argmin_argmax_looped(self):
        for fn in (torch.argmin, torch.argmax):
            args = (torch.randn([512, 512], device=DEVICE), fn, torch.int64)
            code, output = code_and_output(
                reduce_kernel,
                args,
                block_size=1,
                indexing="tensor_descriptor",
                reduction_loop=16,
            )
            torch.testing.assert_close(output, args[1](args[0], dim=-1))

    @skipIfPallas("Pallas cannot lower the strided slice x[tile, 1::2]")
    @skipIfMetal("Metal gives only one reduction dimension per kernel a thread axis")
    def test_rolled_and_persistent_reductions_of_two_slices(self):
        """A rolled reduction's thread axis sits below a later persistent one
        after its loop closes: the persistent reduction's lanes stay
        interleaved with it."""

        @helion.kernel(autotune_effort="none", static_shapes=True)
        def two_slices(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            m, n = x.shape
            hi = torch.zeros(m, device=x.device, dtype=torch.int64)
            lo = torch.zeros(m, device=x.device, dtype=x.dtype)
            for tile in hl.tile(m):
                hi[tile] = torch.argmax(x[tile, 1::2], dim=-1)
                lo[tile] = torch.amin(x[tile, 8:32], dim=-1)
            return hi, lo

        x = torch.randn([40, 64], device=DEVICE)
        _code, (hi, lo) = code_and_output(
            two_slices, (x,), block_sizes=[2], reduction_loops=[8]
        )
        torch.testing.assert_close(hi, torch.argmax(x[:, 1::2], dim=-1))
        torch.testing.assert_close(lo, x[:, 8:32].amin(dim=-1))

    @skipIfRocm("ROCm Triton worker crashes while compiling this reduction kernel")
    def test_reduction_loops_integer_values(self):
        """Test that reduction_loops with integer values works (issue #345 fix)."""

        @helion.kernel(autotune_effort="none")
        def layer_norm_reduction(
            x: torch.Tensor,
            weight: torch.Tensor,
            bias: torch.Tensor,
            eps: float = 1e-5,
        ) -> torch.Tensor:
            m, n = x.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)

            for tile_m in hl.tile(m):
                acc = x[tile_m, :].to(torch.float32)
                var, mean = torch.var_mean(acc, dim=-1, keepdim=True, correction=0)
                normalized = (acc - mean) * torch.rsqrt(var + eps)
                result = normalized * (weight[:].to(torch.float32)) + (
                    bias[:].to(torch.float32)
                )
                out[tile_m, :] = result
            return out

        x = torch.randn([32, 64], device=DEVICE, dtype=torch.bfloat16)
        weight = torch.randn([64], device=DEVICE, dtype=torch.bfloat16)
        bias = torch.randn([64], device=DEVICE, dtype=torch.bfloat16)
        eps = 1e-4

        args = (x, weight, bias, eps)

        # Test various reduction_loops configurations that previously failed
        for reduction_loop_value in [2, 4, 8]:
            with self.subTest(reduction_loop=reduction_loop_value):
                code, output = code_and_output(
                    layer_norm_reduction,
                    args,
                    block_size=32,
                    reduction_loop=reduction_loop_value,
                )

                # Compute expected result using PyTorch's layer_norm
                expected = torch.nn.functional.layer_norm(
                    x.float(), [64], weight.float(), bias.float(), eps
                ).bfloat16()

                torch.testing.assert_close(output, expected, rtol=1e-2, atol=1e-2)

        # Only check the generated code for one configuration to avoid redundant expected outputs
        code, _ = code_and_output(
            layer_norm_reduction, args, block_size=32, reduction_loop=4
        )

    @skipIfMetal("hl.arange needs a Metal prims.iota lowering")
    def test_reduction_over_arange_dim_stays_persistent(self):
        """Issue #2643: a reduction over an ``hl.arange()`` axis must not be
        registered as a rollable (looped) reduction.

        The arange axis is a concrete size with no block index var to re-bind
        per loop iteration, so the producing load cannot be sliced inside the
        reduction loop. Rolling it emitted a shape mismatch during codegen, so
        the autotuner must never offer a ``reduction_loop`` for it -- the
        reduction stays persistent.
        """

        @helion.kernel(static_shapes=True)
        def rms_over_arange(qkv: torch.Tensor) -> torch.Tensor:
            num_tokens, qk_heads, head_dim = qkv.shape
            out = torch.empty(
                [num_tokens, qk_heads], dtype=torch.float32, device=qkv.device
            )
            for tile_m, tile_gn in hl.tile(
                [num_tokens, qk_heads], block_size=[1, None]
            ):
                tile_n = hl.arange(head_dim)
                x_blk = qkv[tile_m, tile_gn, tile_n].to(dtype=torch.float32)
                out[tile_m, tile_gn] = x_blk.pow(2).sum(dim=-1)
            return out

        qkv = torch.randn([64, 8, 128], device=DEVICE, dtype=HALF_DTYPE)

        # The arange reduction axis must not be offered as a looped reduction,
        # otherwise the autotuner could pick a reduction_loop value that
        # produced a shape mismatch (issue #2643).
        bound = rms_over_arange.bind((qkv,))
        self.assertEqual(bound.env.config_spec.reduction_loops.valid_block_ids(), [])
        if _get_backend() == "cute":
            reduction_blocks = [
                block_id
                for block_id, block in enumerate(bound.env.block_sizes)
                if block.reduction
            ]
            self.assertEqual(len(reduction_blocks), 1)
            reduction_block = reduction_blocks[0]
            for spec in (
                bound.env.config_spec.num_threads,
                bound.env.config_spec.cute_vector_widths,
                bound.env.config_spec.cute_lane_layouts,
                bound.env.config_spec.cute_reduction_reloads,
            ):
                self.assertIn(reduction_block, spec.valid_block_ids())

        _code, output = code_and_output(rms_over_arange, (qkv,))
        expected = qkv.to(torch.float32).pow(2).sum(dim=-1)
        torch.testing.assert_close(output, expected, rtol=1e-2, atol=1e-2)

    @skipIfMetal("hl.arange needs a Metal prims.iota lowering")
    def test_reduction_over_arange_dim_size_coincides_with_slice(self):
        """Issue #2643 variant: an ``hl.arange()`` reduction whose size
        coincides with a slice reduction of the same size in the same loop.

        ``allocate_reduction_dimension`` unifies the two same-size reduction
        dimensions, so the arange load's reduced axis surfaces as the rdim
        symbol (not a concrete int) and an output-shape check would be fooled
        into thinking it is rollable. The arange axis is still indexed by an
        ``iota`` node that cannot be re-indexed inside a reduction loop, so the
        reduction must stay persistent.
        """

        @helion.kernel(static_shapes=True)
        def mixed(x: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            out = torch.empty([m], dtype=torch.float32, device=x.device)
            for tile_m in hl.tile(m):
                slice_sum = x[tile_m, :].to(torch.float32).sum(-1)
                tile_n = hl.arange(n)
                arange_sum = x[tile_m, tile_n].to(torch.float32).pow(2).sum(-1)
                out[tile_m] = slice_sum + arange_sum
            return out

        x = torch.randn([64, 128], device=DEVICE, dtype=HALF_DTYPE)

        # The unified rdim must not be offered as a looped reduction: rolling
        # it cannot re-index the iota-indexed arange load (issue #2643).
        bound = mixed.bind((x,))
        self.assertEqual(bound.env.config_spec.reduction_loops.valid_block_ids(), [])

        _code, output = code_and_output(mixed, (x,))
        expected = x.to(torch.float32).sum(-1) + x.to(torch.float32).pow(2).sum(-1)
        torch.testing.assert_close(output, expected, rtol=1e-2, atol=1e-2)

    @skipIfMetal("hl.arange needs a Metal prims.iota lowering")
    def test_reduction_of_arange_value_stays_persistent(self):
        """Issue #2643 variant: an ``hl.arange()`` over the reduction axis
        entering the reduction as a value, not as a load index.

        The iota is a fixed full-extent tensor either way: a reduction loop
        would leave it outside the loop (a shape mismatch on Triton, on CuTe
        a per-thread coordinate of another axis and a silently wrong sum),
        so the reduction must stay persistent.
        """

        @helion.kernel(static_shapes=True)
        def weighted_row_sum(x: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            out = torch.empty([m], dtype=torch.float32, device=x.device)
            for tile_m in hl.tile(m):
                row = x[tile_m, :].to(torch.float32)
                out[tile_m] = (row * hl.arange(n).to(torch.float32)[None, :]).sum(-1)
            return out

        x = torch.randn([64, 128], device=DEVICE, dtype=HALF_DTYPE)
        bound = weighted_row_sum.bind((x,))
        self.assertEqual(bound.env.config_spec.reduction_loops.valid_block_ids(), [])

        _code, output = code_and_output(weighted_row_sum, (x,), block_size=2)
        weights = torch.arange(128, device=DEVICE, dtype=torch.float32)
        expected = (x.to(torch.float32) * weights).sum(-1)
        torch.testing.assert_close(output, expected, rtol=1e-2, atol=1e-2)

    @skipIfRefEager("checks the offered reduction loops and a codegen refusal")
    @skipUnlessBackends(["cute", "triton"])
    def test_free_arange_reduction_rolls_only_through_fragments(self):
        """A reduction of values computed outside the loop from a free
        ``hl.arange`` (``part * chunk + hl.arange(chunk)``): only CuTe's
        computed-fragment lowering re-reads them per chunk, so only CuTe
        offers a reduction loop, and any other rolled lowering refuses it
        instead of folding the same full-extent value every chunk."""

        @helion.kernel(static_shapes=True, autotune_effort="none")
        def chunk_sums(x: torch.Tensor, chunk: hl.constexpr) -> torch.Tensor:
            parts = (x.size(1) + chunk - 1) // chunk
            out = torch.empty((x.size(0), parts), dtype=torch.int32, device=x.device)
            for row, part in hl.grid((x.size(0), parts)):
                columns = part * chunk + hl.arange(chunk)
                valid = columns < x.size(1)
                value = hl.load(x, [row, columns], extra_mask=valid)
                out[row, part] = torch.where(valid, value, 0).sum(dtype=torch.int32)
            return out

        x = torch.arange(3 * 131, dtype=torch.int32, device=DEVICE).reshape(3, 131)
        x = x % 13
        expected = torch.stack(
            [
                x[:, start : start + 64].sum(-1, dtype=torch.int32)
                for start in (0, 64, 128)
            ],
            dim=-1,
        )
        bound = chunk_sums.bind((x, 64))
        rollable = bound.config_spec.reduction_loops.valid_block_ids()
        config = bound.config_spec.default_config()
        torch.testing.assert_close(bound.compile_config(config)(x, 64), expected)
        if _get_backend() != "cute":
            self.assertEqual(rollable, [])
            return
        self.assertNotEqual(rollable, [])
        config.config["reduction_loops"] = [16]
        torch.testing.assert_close(bound.compile_config(config)(x, 64), expected)
        with (
            patch(
                "helion._compiler.cute.computed_fragment.codegen_computed_fragment_root",
                return_value=False,
            ),
            self.assertRaisesRegex(
                helion.exc.BackendUnsupported, "computed outside its loop"
            ),
        ):
            bound.to_code(config)

    @skipIfRefEager("checks the offered reduction loops")
    @skipUnlessBackends(["cute", "triton"])
    def test_free_arange_row_reduction_stays_persistent(self):
        """The computed-fragment lowering takes only complete rank-1 free
        ``hl.arange`` reductions, so a row-wise one over a tile of rows offers
        no reduction loop that every lowering would refuse."""

        @helion.kernel(static_shapes=True, autotune_effort="none")
        def chunk_row_sums(x: torch.Tensor, chunk: hl.constexpr) -> torch.Tensor:
            parts = (x.size(1) + chunk - 1) // chunk
            out = torch.empty((x.size(0), parts), dtype=torch.int32, device=x.device)
            for rows, part in hl.tile([x.size(0), parts], block_size=[None, 1]):
                columns = part.begin * chunk + hl.arange(chunk)
                valid = columns < x.size(1)
                value = hl.load(x, [rows, columns], extra_mask=valid[None, :])
                out[rows, part] = torch.where(valid[None, :], value, 0).sum(
                    -1, keepdim=True, dtype=torch.int32
                )
            return out

        x = torch.arange(8 * 131, dtype=torch.int32, device=DEVICE).reshape(8, 131)
        x = x % 13
        expected = torch.stack(
            [
                x[:, start : start + 64].sum(-1, dtype=torch.int32)
                for start in (0, 64, 128)
            ],
            dim=-1,
        )
        bound = chunk_row_sums.bind((x, 64))
        self.assertEqual(bound.config_spec.reduction_loops.valid_block_ids(), [])
        config = bound.config_spec.default_config()
        torch.testing.assert_close(bound.compile_config(config)(x, 64), expected)

    @skipIfMetal("hl.arange needs a Metal prims.iota lowering")
    def test_arange_reduction_with_synthetic_lanes(self):
        """A persistent ``hl.arange()`` reduction whose extent exceeds the live
        thread count must accumulate across synthetic lanes.

        On CuTe, when the reduction extent is wider than the threads available
        for it, the body runs inside a synthetic lane loop. The per-thread
        value must be carried across lanes so the final warp reduction sees
        every element; otherwise only the last lane's slice is reduced and the
        result is silently wrong (issue #2643). ``block_size=16`` splits the
        CuTe thread budget so the 128-wide reduction needs that lane loop.
        """

        @helion.kernel(static_shapes=True)
        def arange_reduce(x: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            out = torch.empty([m], dtype=torch.float32, device=x.device)
            for tile_m in hl.tile(m):
                tile_n = hl.arange(n)
                out[tile_m] = x[tile_m, tile_n].to(torch.float32).pow(2).sum(-1)
            return out

        x = torch.randn([64, 128], device=DEVICE, dtype=HALF_DTYPE)
        _code, output = code_and_output(arange_reduce, (x,), block_size=16)
        expected = x.to(torch.float32).pow(2).sum(-1)
        torch.testing.assert_close(output, expected, rtol=1e-2, atol=1e-2)

    def test_fp16_var_mean(self):
        @helion.kernel(static_shapes=True)
        def layer_norm_fwd_repro(
            x: torch.Tensor,
            weight: torch.Tensor,
            bias: torch.Tensor,
            eps: float = 1e-5,
        ) -> torch.Tensor:
            m, n = x.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            for tile_m in hl.tile(m):
                x_part = x[tile_m, :]
                var, mean = torch.var_mean(x_part, dim=-1, keepdim=True, correction=0)
                normalized = (x_part - mean) * torch.rsqrt(var.to(torch.float32) + eps)
                out[tile_m, :] = normalized * (weight[:].to(torch.float32)) + (
                    bias[:].to(torch.float32)
                )
            return out

        torch.manual_seed(0)
        batch_size = 32
        dim = 64
        x = torch.randn([batch_size, dim], device=DEVICE, dtype=torch.bfloat16)
        weight = torch.randn([dim], device=DEVICE, dtype=torch.bfloat16)
        bias = torch.randn([dim], device=DEVICE, dtype=torch.bfloat16)
        eps = 1e-4
        code1, result1 = code_and_output(
            layer_norm_fwd_repro,
            (x, weight, bias, eps),
            block_sizes=[32],
            reduction_loops=[None],
        )

        code2, result2 = code_and_output(
            layer_norm_fwd_repro,
            (x, weight, bias, eps),
            block_sizes=[32],
            reduction_loops=[8],
        )
        # var and mean come out in bf16, so the two reduction orders can round
        # a row's var or mean one bf16 ulp apart, which moves the bf16 outputs
        # by a few ulps (far beyond a 1e-3 tolerance; seen on TileIR).
        torch.testing.assert_close(result1, result2, rtol=2e-2, atol=3e-2)

    @xfailIfPallasTpu("fp16/bf16 1D tensors hit TPU Mosaic sublane alignment error")
    @skipIfTileIR("TileIR does not support log1p")
    def test_fp16_math_ops_fp32_fallback(self):
        """Test that mathematical ops with fp16/bfloat16 inputs now work via fp32 fallback."""

        @helion.kernel(autotune_effort="none")
        def rsqrt_fp16_kernel(x: torch.Tensor) -> torch.Tensor:
            result = torch.empty_like(x)
            for tile in hl.tile(x.size(0)):
                # This should now work via fp32 fallback
                result[tile] = torch.rsqrt(x[tile])
            return result

        @helion.kernel(autotune_effort="none")
        def multi_math_ops_fp16_kernel(x: torch.Tensor) -> torch.Tensor:
            result = torch.empty([x.size(0), 8], dtype=x.dtype, device=x.device)
            for tile in hl.tile(x.size(0)):
                # Test multiple operations that have confirmed fallbacks
                result[tile, 0] = torch.rsqrt(x[tile])
                result[tile, 1] = torch.sqrt(x[tile])
                result[tile, 2] = torch.sin(x[tile])
                result[tile, 3] = torch.cos(x[tile])
                result[tile, 4] = torch.log(x[tile])
                result[tile, 5] = torch.tanh(x[tile])
                result[tile, 6] = torch.log1p(x[tile])
                result[tile, 7] = torch.exp(x[tile])
            return result

        # Test with float16 - should now succeed
        x_fp16 = (
            torch.abs(torch.randn([32], device=DEVICE, dtype=torch.float16)) + 0.1
        )  # positive values for rsqrt

        code, result = code_and_output(rsqrt_fp16_kernel, (x_fp16,))

        # Verify result is correct compared to PyTorch's rsqrt
        expected = torch.rsqrt(x_fp16)
        torch.testing.assert_close(result, expected, rtol=1e-3, atol=1e-3)

        # Verify result maintains fp16 dtype
        self.assertEqual(result.dtype, torch.float16)

        # Test multiple math operations
        x_multi = torch.abs(torch.randn([16], device=DEVICE, dtype=torch.float16)) + 0.1
        code_multi, result_multi = code_and_output(
            multi_math_ops_fp16_kernel, (x_multi,)
        )

        # Verify each operation's correctness
        expected_rsqrt = torch.rsqrt(x_multi)
        expected_sqrt = torch.sqrt(x_multi)
        expected_sin = torch.sin(x_multi)
        expected_cos = torch.cos(x_multi)
        expected_log = torch.log(x_multi)
        expected_tanh = torch.tanh(x_multi)
        expected_log1p = torch.log1p(x_multi)
        expected_exp = torch.exp(x_multi)

        torch.testing.assert_close(
            result_multi[:, 0], expected_rsqrt, rtol=1e-3, atol=1e-3
        )
        torch.testing.assert_close(
            result_multi[:, 1], expected_sqrt, rtol=1e-3, atol=1e-3
        )
        torch.testing.assert_close(
            result_multi[:, 2], expected_sin, rtol=1e-3, atol=1e-3
        )
        torch.testing.assert_close(
            result_multi[:, 3], expected_cos, rtol=1e-3, atol=1e-3
        )
        torch.testing.assert_close(
            result_multi[:, 4], expected_log, rtol=1e-3, atol=1e-3
        )
        torch.testing.assert_close(
            result_multi[:, 5], expected_tanh, rtol=1e-3, atol=1e-3
        )
        torch.testing.assert_close(
            result_multi[:, 6], expected_log1p, rtol=1e-3, atol=1e-3
        )
        torch.testing.assert_close(
            result_multi[:, 7], expected_exp, rtol=1e-3, atol=1e-3
        )

        # Verify all results maintain fp16 dtype
        self.assertEqual(result_multi.dtype, torch.float16)

        # Test with bfloat16 if available
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            x_bf16 = (
                torch.abs(torch.randn([32], device=DEVICE, dtype=torch.bfloat16)) + 0.1
            )

            code_bf16, result_bf16 = code_and_output(rsqrt_fp16_kernel, (x_bf16,))

            # Verify bfloat16 result is correct
            expected_bf16 = torch.rsqrt(x_bf16)
            torch.testing.assert_close(result_bf16, expected_bf16, rtol=1e-2, atol=1e-2)

            # Verify result maintains bfloat16 dtype
            self.assertEqual(result_bf16.dtype, torch.bfloat16)

    @skipUnlessTensorDescriptor("Tensor descriptor support is required")
    def test_layer_norm_nonpow2_reduction(self):
        """Test layer norm with non-power-of-2 reduction dimension (1536)."""

        @helion.kernel(
            config=helion.Config(
                block_sizes=[2],
                indexing="tensor_descriptor",
                num_stages=4,
                num_warps=4,
                pid_type="flat",
            ),
            static_shapes=True,
        )
        def layer_norm_fwd_nonpow2(
            x: torch.Tensor,
            weight: torch.Tensor,
            bias: torch.Tensor,
            eps: float = 1e-5,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            m, n = x.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)
            mean = torch.empty([m], dtype=torch.float32, device=x.device)
            rstd = torch.empty([m], dtype=torch.float32, device=x.device)

            for tile_m in hl.tile(m):
                acc = x[tile_m, :].to(torch.float32)
                # Compute mean
                mean_val = torch.sum(acc, dim=-1) / n
                # Compute variance
                centered = acc - mean_val[:, None]
                var_val = torch.sum(centered * centered, dim=-1) / n
                # Compute reciprocal standard deviation
                rstd_val = torch.rsqrt(var_val + eps)
                # Normalize
                normalized = centered * rstd_val[:, None]
                # Apply affine transformation
                acc = normalized * (weight[:].to(torch.float32)) + (
                    bias[:].to(torch.float32)
                )
                out[tile_m, :] = acc.to(x.dtype)
                mean[tile_m] = mean_val
                rstd[tile_m] = rstd_val
            return out, mean, rstd

        batch_size = 4096
        dim = 1536  # Non-power-of-2 to trigger padding

        # Use tritonbench-style input distribution
        torch.manual_seed(42)
        x = -2.3 + 0.5 * torch.randn([batch_size, dim], device=DEVICE, dtype=HALF_DTYPE)
        weight = torch.randn([dim], device=DEVICE, dtype=HALF_DTYPE)
        bias = torch.randn([dim], device=DEVICE, dtype=HALF_DTYPE)
        eps = 1e-4

        code, (out, mean, rstd) = code_and_output(
            layer_norm_fwd_nonpow2,
            (x, weight, bias, eps),
        )

        # Compute expected result
        x_fp32 = x.to(torch.float32)
        mean_ref = x_fp32.mean(dim=1)
        var_ref = x_fp32.var(dim=1, unbiased=False)
        rstd_ref = torch.rsqrt(var_ref + eps)
        normalized_ref = (x_fp32 - mean_ref[:, None]) * rstd_ref[:, None]
        out_ref = (normalized_ref * weight.float() + bias.float()).half()

        # Check outputs
        torch.testing.assert_close(out, out_ref, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(mean, mean_ref, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(rstd, rstd_ref, rtol=1e-5, atol=1e-5)

    def test_size1_reduction_unsqueeze_sum(self):
        """Sum over a literal size-1 dim from unsqueeze should reduce rank (issue #1423).

        When unsqueeze creates a literal size-1 dimension and sum reduces over
        it, Inductor optimizes the reduction to a Pointwise op.  Without the
        fix, PointwiseLowering produces a result that keeps the size-1
        dimension, causing a rank mismatch at the store site.
        """

        @helion.kernel(
            config=helion.Config(block_sizes=[128], num_stages=1, num_warps=4),
            static_shapes=False,
        )
        def unsqueeze_sum(x: torch.Tensor) -> torch.Tensor:
            (D,) = x.shape
            out = torch.empty(D, dtype=torch.float32, device=x.device)
            for (tile_d,) in hl.tile([D]):
                val = x[tile_d].float()  # [D_tile]
                val2 = val.unsqueeze(0)  # [1, D_tile]
                reduced = val2.sum(0)  # should be [D_tile]
                hl.store(out, [tile_d.index], reduced)
            return out

        x = torch.randn(128, dtype=torch.bfloat16, device=DEVICE)
        code, out = code_and_output(unsqueeze_sum, (x,))
        torch.testing.assert_close(out, x.float(), rtol=1e-4, atol=1e-4)

    @skipIfMetal(
        "Metal passes symbolic sizes as float scalars; integer floor_divide fails"
    )
    def test_size1_reduction_keepdim_sum(self):
        """Second sum over a keepdim=True result should reduce rank (issue #1423).

        sum(0, keepdim=True) produces a [1, D_tile] tensor with a literal
        size-1 dimension.  A subsequent sum(0) over that literal-1 dim is
        converted to a Pointwise op by Inductor.  Without the fix the result
        retains the extra dimension, causing a rank mismatch at the store site.
        """

        @helion.kernel(
            config=helion.Config(block_sizes=[8, 128], num_stages=1, num_warps=4),
            static_shapes=False,
        )
        def keepdim_sum(x: torch.Tensor) -> torch.Tensor:
            T, D = x.shape
            out = torch.empty(D, dtype=torch.float32, device=x.device)
            for tile_t, tile_d in hl.tile([T, D]):
                val = x[tile_t, tile_d].float()  # [T_tile, D_tile]
                partial = val.sum(0, keepdim=True)  # [1, D_tile]
                result = partial.sum(0)  # should be [D_tile]
                hl.store(out, [tile_d.index], result)
            return out

        x = torch.randn(4, 128, dtype=torch.bfloat16, device=DEVICE)
        code, out = code_and_output(keepdim_sum, (x,))
        ref = x.float().sum(0)
        torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)

    @skipIfMetal("argreduce after matmul needs a Metal scalar aten.addmm lowering")
    def test_argmax_on_tile_after_matmul(self):
        """Test that argmax on a matmul tile returns the correct row indices."""

        @helion.kernel(autotune_effort="none")
        def matmul_argmax(
            x: torch.Tensor,
            y: torch.Tensor,
        ) -> torch.Tensor:
            m, k = x.size()
            k2, n = y.size()
            assert k == k2, f"size mismatch {k} != {k2}"
            out = torch.empty([m], dtype=torch.int32, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m] = acc.argmax(dim=1)
            return out

        # Use a full 16x16 tile so TPU/Pallas block-size promotion does not
        # turn this into a partial matmul tile with mismatched accumulator
        # and operand shapes.
        x = torch.eye(16, device=DEVICE)
        y = (
            torch.arange(16, device=DEVICE, dtype=x.dtype)[None, :]
            .expand(16, -1)
            .clone()
        )

        _, result = code_and_output(matmul_argmax, (x, y), block_sizes=[16, 16, 16])
        ref = (x @ y).argmax(dim=1).to(torch.int32)
        torch.testing.assert_close(result, ref)

    @skipIfPallas("Pallas TPU argreduce cannot write int64 keepdim outputs")
    @skipIfMetal("argreduce after matmul needs a Metal scalar aten.addmm lowering")
    def test_argmax_on_tile_after_matmul_keepdim(self):
        @helion.kernel(autotune_effort="none")
        def matmul_argmax_keepdim(
            x: torch.Tensor,
            y: torch.Tensor,
        ) -> torch.Tensor:
            m, k = x.size()
            k2, n = y.size()
            assert k == k2, f"size mismatch {k} != {k2}"
            out = torch.empty([m, 1], dtype=torch.int64, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, :] = acc.argmax(dim=1, keepdim=True)
            return out

        x = torch.eye(16, device=DEVICE)
        y = (
            torch.arange(16, device=DEVICE, dtype=x.dtype)[None, :]
            .expand(16, -1)
            .clone()
        )

        _, result = code_and_output(
            matmul_argmax_keepdim,
            (x, y),
            block_sizes=[16, 16, 16],
        )
        ref = (x @ y).argmax(dim=1, keepdim=True)
        torch.testing.assert_close(result, ref)

    @skipIfPallas("nested torch.matmul argreduce lowering is unsupported on Pallas")
    @skipIfMetal("argreduce after matmul needs a Metal scalar aten.addmm lowering")
    def test_argmax_on_tile_after_torch_matmul(self):
        @helion.kernel(autotune_effort="none")
        def torch_matmul_argmax(
            x: torch.Tensor,
            y: torch.Tensor,
        ) -> torch.Tensor:
            m, k = x.size()
            k2, n = y.size()
            assert k == k2, f"size mismatch {k} != {k2}"
            out = torch.empty([m], dtype=torch.int32, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                out[tile_m] = torch.matmul(x[tile_m, :], y[:, tile_n]).argmax(dim=1)
            return out

        x = torch.eye(8, device=DEVICE)
        y = torch.arange(8, device=DEVICE, dtype=x.dtype)[None, :].expand(8, -1).clone()

        _, result = code_and_output(
            torch_matmul_argmax,
            (x, y),
            block_sizes=[8, 8],
        )
        ref = (x @ y).argmax(dim=1).to(torch.int32)
        torch.testing.assert_close(result, ref)

    @skipIfPallas("barrier and persistent_blocked not supported on Pallas")
    @skipIfTileIR("TileIR does not support barrier operations")
    @skipIfMetal("hl.barrier() requires a persistent pid_type, unsupported on Metal")
    def test_reduction_loop_with_multiple_rdims(self):
        """Test that reduction_loops works when there are multiple reduction dimensions."""

        @helion.kernel(autotune_effort="none")
        def two_rdim_rms_norm(
            x: torch.Tensor,
            y: torch.Tensor,
            w1: torch.Tensor,
            w2: torch.Tensor,
            eps: float = 1e-5,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            big_dim = hl.specialize(x.size(1))
            small_count = hl.specialize(y.size(0))
            small_dim = hl.specialize(y.size(1))

            normed_x = torch.empty([1, big_dim], dtype=x.dtype, device=x.device)
            normed_y = torch.empty(
                [small_count, small_dim], dtype=x.dtype, device=x.device
            )

            # Phase 1: reduction over big_dim (creates rdim #1)
            for tile_m in hl.tile(1):
                x_tile = x[tile_m, :].to(torch.float32)
                mean_sq = torch.mean(x_tile * x_tile, dim=-1)
                inv_rms = torch.rsqrt(mean_sq + eps)
                normed_x[tile_m, :] = (
                    x_tile * inv_rms[:, None] * w1[:].to(torch.float32)
                ).to(x.dtype)

            hl.barrier()

            # Phase 2: reduction over small_dim (creates rdim #2)
            for tile_h in hl.tile(small_count):
                y_tile = y[tile_h, :].to(torch.float32)
                mean_sq = torch.mean(y_tile * y_tile, dim=-1)
                inv_rms = torch.rsqrt(mean_sq + eps)
                normed_y[tile_h, :] = (
                    y_tile * inv_rms[:, None] * w2[:].to(torch.float32)
                ).to(x.dtype)

            return normed_x, normed_y

        x = torch.randn([1, 256], device=DEVICE, dtype=HALF_DTYPE)
        y = torch.randn([8, 64], device=DEVICE, dtype=HALF_DTYPE)
        w1 = torch.randn([256], device=DEVICE, dtype=HALF_DTYPE)
        w2 = torch.randn([64], device=DEVICE, dtype=HALF_DTYPE)
        args = (x, y, w1, w2)

        code, (out_x, out_y) = code_and_output(
            two_rdim_rms_norm,
            args,
            block_sizes=[1, 1],
            reduction_loop=16,
            pid_type="persistent_blocked",
        )

        # Check Phase 1 result
        x_f = x.float()
        inv_rms_x = torch.rsqrt(torch.mean(x_f * x_f, dim=-1) + 1e-5)
        expected_x = (x_f * inv_rms_x[:, None] * w1.float()).half()
        torch.testing.assert_close(out_x, expected_x, rtol=1e-2, atol=1e-2)

        # Check Phase 2 result
        y_f = y.float()
        inv_rms_y = torch.rsqrt(torch.mean(y_f * y_f, dim=-1) + 1e-5)
        expected_y = (y_f * inv_rms_y[:, None] * w2.float()).half()
        torch.testing.assert_close(out_y, expected_y, rtol=1e-2, atol=1e-2)

    @skipUnlessBackends(["triton", "cute"])
    def test_bool_and_int_sums_accumulate_in_int64(self):
        """torch promotes bool and int32 sums to int64 on every reduction path."""

        @helion.kernel(autotune_effort="none", static_shapes=True)
        def counts(
            x: torch.Tensor, y: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            m, _ = x.shape
            positives = torch.empty([m], dtype=torch.int64, device=x.device)
            argmaxes = torch.empty([m], dtype=torch.int64, device=x.device)
            sums = torch.empty([m], dtype=torch.int64, device=x.device)
            for tile_m in hl.tile(m):
                row = x[tile_m, :]
                argmaxes[tile_m] = row.argmax(-1)
                positives[tile_m] = (row > 0).sum(-1)
                sums[tile_m] = y[tile_m, :].sum(-1)
            return positives, argmaxes, sums

        torch.manual_seed(0)
        # n=1000 combines across warps; reduction_loops rolls the reduction.
        for n, extra in ((64, {}), (1000, {}), (1000, {"reduction_loops": [32]})):
            x = torch.randn(8, n, device=DEVICE)
            # 2**30 per element overflows an int32 accumulator.
            y = torch.full((8, n), 2**30, device=DEVICE, dtype=torch.int32)
            _code, (positives, argmaxes, sums) = code_and_output(
                counts, (x, y), block_sizes=[2], **extra
            )
            torch.testing.assert_close(positives, (x > 0).sum(-1))
            torch.testing.assert_close(argmaxes, x.argmax(-1))
            torch.testing.assert_close(sums, y.sum(-1))

    @skipUnlessBackends(["triton", "cute"])
    def test_extremum_reductions_propagate_nan(self):
        """amax/amin return NaN and argmax/argmin the first NaN, as in torch."""

        @helion.kernel(autotune_effort="none", static_shapes=True)
        def extrema_2d(
            x: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            m, _ = x.shape
            maxes = torch.empty([m], dtype=x.dtype, device=x.device)
            mins = torch.empty([m], dtype=x.dtype, device=x.device)
            argmaxes = torch.empty([m], dtype=torch.int64, device=x.device)
            argmins = torch.empty([m], dtype=torch.int64, device=x.device)
            for tile_m in hl.tile(m):
                row = x[tile_m, :]
                maxes[tile_m] = row.amax(-1)
                mins[tile_m] = row.amin(-1)
                argmaxes[tile_m] = row.argmax(-1)
                argmins[tile_m] = row.argmin(-1)
            return maxes, mins, argmaxes, argmins

        @helion.kernel(autotune_effort="none", static_shapes=True)
        def extrema_3d(
            x: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            a, b, _ = x.shape
            maxes = torch.empty([a, b], dtype=x.dtype, device=x.device)
            mins = torch.empty([a, b], dtype=x.dtype, device=x.device)
            argmaxes = torch.empty([a, b], dtype=torch.int64, device=x.device)
            argmins = torch.empty([a, b], dtype=torch.int64, device=x.device)
            for tile_a in hl.tile(a):
                rows = x[tile_a, :, :]
                maxes[tile_a, :] = rows.amax(-1)
                mins[tile_a, :] = rows.amin(-1)
                argmaxes[tile_a, :] = rows.argmax(-1)
                argmins[tile_a, :] = rows.argmin(-1)
            return maxes, mins, argmaxes, argmins

        cases = [
            # One warp, across warps, rolled.
            (extrema_2d, (8, 32), {}),
            (extrema_2d, (8, 1000), {}),
            (extrema_2d, (8, 1000), {"reduction_loops": [32]}),
            # A reduced dim above a sibling thread axis; across synthetic lanes.
            (extrema_3d, (4, 8, 32), {}),
            (extrema_3d, (2, 2, 1024), {}),
        ]
        for kernel, shape, extra in cases:
            for dtype in (torch.float32, torch.float16, torch.bfloat16):
                torch.manual_seed(0)
                x = torch.randn(shape, device=DEVICE, dtype=dtype)
                rows = x.view(-1, shape[-1])
                rows[0::2, 3] = float("nan")
                rows[0::4, shape[-1] - 2] = float("nan")
                code, out = code_and_output(kernel, (x,), block_sizes=[2], **extra)
                if _get_backend() == "triton":
                    # One max.NaN/min.NaN per combine step, not the four or
                    # five compares and selects of Inductor's max2/min2.
                    self.assertIn("helion_triton_helpers.max_propagate_nan", code)
                    self.assertIn("helion_triton_helpers.min_propagate_nan", code)
                expected = (x.amax(-1), x.amin(-1), x.argmax(-1), x.argmin(-1))
                for actual, reference in zip(out, expected, strict=True):
                    torch.testing.assert_close(actual, reference, equal_nan=True)

    @skipUnlessBackends(["triton", "cute"])
    @skipIfRefEager(
        "tile positions depend on block_sizes, which ref mode does not apply"
    )
    def test_tile_argreduce_counts_from_the_tile_start(self):
        """argmax/argmin over ``x[tm, tn]`` return positions in the tile, as
        torch does on the tile tensor; ``+ tn.begin`` makes them global."""

        @helion.kernel(autotune_effort="none", static_shapes=True)
        def chunked_row_argmax(x: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            out = torch.empty([m], dtype=torch.int64, device=x.device)
            for tm in hl.tile(m):
                best = hl.full([tm], float("-inf"), dtype=torch.float32)
                idx = hl.zeros([tm], dtype=torch.int64)
                for tn in hl.tile(n):
                    v = x[tm, tn]
                    v_max = v.amax(1)
                    v_idx = torch.argmax(v, dim=1) + tn.begin
                    better = v_max > best
                    idx = torch.where(better, v_idx, idx)
                    best = torch.where(better, v_max, best)
                out[tm] = idx
            return out

        @helion.kernel(autotune_effort="none", static_shapes=True)
        def tile_argreduce(
            x: torch.Tensor, local: torch.Tensor, shifted: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            for tm, tn in hl.tile([x.size(0), x.size(1)]):
                local[tm, tn.id] = torch.argmax(x[tm, tn], dim=1)
                shifted[tm, tn.id] = torch.argmin(x[tm, tn], dim=1) + tn.begin
            return local, shifted

        @helion.kernel(autotune_effort="none", static_shapes=True)
        def row_tile_argmax(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
            for tm, tn in hl.tile([x.size(0), x.size(1)]):
                out[tm.id, tn] = torch.argmax(x[tm, tn], dim=0)
            return out

        def per_tile(x: torch.Tensor, block: int, fn: object) -> torch.Tensor:
            m, n = x.shape
            tiles = -(-n // block)
            pad = torch.full((m, tiles * block - n), float("nan"), device=x.device)
            grouped = torch.cat([x, pad], 1).view(m, tiles, block)
            # NaN padding never wins against the distinct finite values below.
            grouped = grouped.nan_to_num(nan=-1e30 if fn is torch.argmax else 1e30)
            return fn(grouped, dim=2)  # pyrefly: ignore [not-callable]

        configs = [{"block_sizes": [4, 32]}, {"block_sizes": [1, 64]}]
        if _get_backend() == "cute":
            # The column tile on 16 threads x 4 lanes, and one column per thread.
            configs += [
                {"block_sizes": [4, 64], "num_threads": [0, 16]},
                {"block_sizes": [4, 1]},
            ]
        x = torch.randperm(32 * 200, device=DEVICE).float().view(32, 200)
        for config in configs:
            with self.subTest(**config):
                bm, bn = config["block_sizes"]
                _, out = code_and_output(chunked_row_argmax, (x,), **config)
                torch.testing.assert_close(out, x.argmax(1), atol=0, rtol=0)

                tiles = -(-200 // bn)
                local = torch.zeros(32, tiles, dtype=torch.int64, device=DEVICE)
                shifted = torch.zeros_like(local)
                _, (local, shifted) = code_and_output(
                    tile_argreduce, (x, local, shifted), **config
                )
                begins = torch.arange(tiles, device=DEVICE)[None, :] * bn
                torch.testing.assert_close(
                    local, per_tile(x, bn, torch.argmax), atol=0, rtol=0
                )
                torch.testing.assert_close(
                    shifted, per_tile(x, bn, torch.argmin) + begins, atol=0, rtol=0
                )

                rows = torch.zeros(32 // bm, 200, dtype=torch.int64, device=DEVICE)
                _, rows = code_and_output(row_tile_argmax, (x, rows), **config)
                torch.testing.assert_close(
                    rows, x.view(32 // bm, bm, 200).argmax(1), atol=0, rtol=0
                )

    @skipUnlessBackends(["triton", "cute"])
    @skipIfRefEager("compiles specific reduction configs")
    def test_half_precision_sum_accumulates_in_float32(self):
        """An fp16/bf16 row sum is as accurate as torch's (which accumulates in
        fp32): summing in the half dtype is off by several output ulps."""

        @helion.kernel(static_shapes=True)
        def row_sum(x: torch.Tensor) -> torch.Tensor:
            m, _ = x.shape
            out = torch.empty([m], dtype=x.dtype, device=x.device)
            for tile in hl.tile(m):
                out[tile] = x[tile, :].sum(-1)
            return out

        torch.manual_seed(0)
        for dtype in (torch.float16, torch.bfloat16):
            x = torch.randn(64, 8192, device=DEVICE).to(dtype)
            expected = x.double().sum(-1)
            torch_error = (x.sum(-1).double() - expected).abs().max().item()
            bound = row_sum.bind((x,))
            for config in (
                helion.Config(block_sizes=[1]),
                helion.Config(block_sizes=[1], reduction_loops=[1024]),
            ):
                result = bound.compile_config(config)(x)
                error = (result.double() - expected).abs().max().item()
                self.assertLessEqual(error, 1.5 * torch_error, (dtype, config))

    @skipUnlessBackends(["triton", "cute"])
    @skipIfRefEager(
        "checks the rolled loop's fp32 accumulator; ref mode has no rolled loop and"
        " rounds the fp16 elementwise result as eager does"
    )
    def test_rolled_half_precision_prod(self):
        """A rolled fp16 prod keeps its fp32 loop-carried accumulator."""

        @helion.kernel(static_shapes=True)
        def row_prod(x: torch.Tensor) -> torch.Tensor:
            m, _ = x.shape
            out = torch.empty([m], dtype=x.dtype, device=x.device)
            for tile in hl.tile(m):
                out[tile] = (1.0 + x[tile, :] * 0.001).prod(-1)
            return out

        x = torch.randn(64, 1000, device=DEVICE).half()
        _, result = code_and_output(
            row_prod, (x,), block_sizes=[1], reduction_loops=[64]
        )
        expected = (1.0 + x.float() * 0.001).prod(-1).half()
        torch.testing.assert_close(result, expected, rtol=2e-3, atol=0)

    @skipIfMetal("Metal gives only one reduction dimension per kernel a thread axis")
    def test_double_sum_of_two_full_slices(self):
        """``x[i, :, :].sum(-1).sum(-1)``: both ``:`` dims spread over threads,
        so the second reduction combines lanes strided by the first's."""

        @helion.kernel(autotune_effort="none")
        def double_sum(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty([x.size(0)], dtype=x.dtype, device=x.device)
            for i in hl.grid(x.size(0)):
                out[i] = x[i, :, :].sum(-1).sum(-1)
            return out

        for shape in [(3, 4, 8), (3, 16, 32)]:
            x = torch.randn(shape, device=DEVICE)
            _, out = code_and_output(double_sum, (x,))
            torch.testing.assert_close(out, x.sum((1, 2)), rtol=1e-4, atol=1e-4)

    @skipIfPallas("Pallas TPU cannot write the int64 count and argmax outputs")
    @skipIfMetal("Metal gives only one reduction dimension per kernel a thread axis")
    def test_reduce_last_of_two_full_slices(self):
        """Reduce the last of two ``:`` dims; the middle one stays a live axis."""

        @helion.kernel(autotune_effort="none")
        def reduce_last(
            x: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            a, b, _c = x.shape
            sums = torch.empty([a, b], dtype=x.dtype, device=x.device)
            maxes = torch.empty([a, b], dtype=x.dtype, device=x.device)
            positives = torch.empty([a, b], dtype=torch.int64, device=x.device)
            argmaxes = torch.empty([a, b], dtype=torch.int64, device=x.device)
            for tile_a in hl.tile(a):
                tile = x[tile_a, :, :]
                sums[tile_a, :] = tile.sum(-1)
                maxes[tile_a, :] = tile.amax(-1)
                positives[tile_a, :] = (tile > 0).sum(-1)
                argmaxes[tile_a, :] = tile.argmax(-1)
            return sums, maxes, positives, argmaxes

        # (2, 2, 1024) splits the reduced dim across synthetic lanes.
        for shape in [(4, 8, 32), (8, 2, 16), (2, 64, 4), (2, 2, 1024)]:
            x = torch.randn(shape, device=DEVICE)
            _code, (sums, maxes, positives, argmaxes) = code_and_output(
                reduce_last, (x,), block_sizes=[2]
            )
            torch.testing.assert_close(sums, x.sum(-1), rtol=1e-4, atol=1e-4)
            torch.testing.assert_close(maxes, x.amax(-1))
            torch.testing.assert_close(positives, (x > 0).sum(-1))
            torch.testing.assert_close(argmaxes, x.argmax(-1))

    @skipIfRefEager("compiles one config for two input shapes")
    @skipIfPallas("Pallas does not pad tensor factories to a power of two")
    @skipIfMetal("not verified on Metal")
    def test_padded_factory_dim_keeps_dynamic_slice_dynamic(self) -> None:
        """A padded factory dim of 48 is not the dynamic full slice that is 48
        wide at bind time, whichever comes first: the slice must stay dynamic,
        so one compiled kernel serves a 40-wide input too."""

        @helion.kernel(static_shapes=False)
        def slice_plus_factory(x: torch.Tensor) -> torch.Tensor:
            m = x.size(0)
            out = torch.empty([m], dtype=torch.float32, device=x.device)
            for tile_m in hl.tile(m):
                out[tile_m] = x[tile_m, :].sum(-1) + hl.full(
                    [tile_m, 48], 1.0, dtype=torch.float32
                ).sum(-1)
            return out

        @helion.kernel(static_shapes=False)
        def factory_plus_slice(x: torch.Tensor) -> torch.Tensor:
            m = x.size(0)
            out = torch.empty([m], dtype=torch.float32, device=x.device)
            for tile_m in hl.tile(m):
                acc = hl.full([tile_m, 48], 1.0, dtype=torch.float32)
                row_sum = x[tile_m, :].sum(-1)
                out[tile_m] = row_sum + acc.sum(-1)
            return out

        x48 = torch.rand(32, 48, device=DEVICE)
        for kernel in (slice_plus_factory, factory_plus_slice):
            with self.subTest(kernel=kernel.name):
                compiled = kernel.bind((x48,)).compile_config(
                    helion.Config(block_sizes=[16])
                )
                for width in (48, 40):
                    x = torch.rand(32, width, device=DEVICE)
                    torch.testing.assert_close(
                        compiled(x), x.sum(-1) + 48, rtol=1e-4, atol=1e-4
                    )

    @skipIfPallas("Pallas does not pad tensor factories to a power of two")
    @skipIfMetal("not verified on Metal")
    def test_reduce_over_padded_factory_dim(self) -> None:
        """``hl.zeros([tile, 400])`` pads its 400 columns to 512; a reduction
        over that dim must see only the 400 columns of the full slice it is
        unified with, not the padding (whose centered value is -mean)."""

        @helion.kernel(static_shapes=True)
        def centered_square_sum(x: torch.Tensor) -> torch.Tensor:
            m = x.size(0)
            n = hl.specialize(x.size(1))
            out = torch.empty([m], dtype=torch.float32, device=x.device)
            for tile_m in hl.tile(m):
                acc = hl.zeros([tile_m, n], dtype=torch.float32)
                acc = acc + x[tile_m, :]
                centered = acc - acc.sum(-1, keepdim=True) / n
                out[tile_m] = (centered * centered).sum(-1)
            return out

        x = torch.rand(32, 400, device=DEVICE) + 1.0
        _, out = code_and_output(centered_square_sum, (x,), block_sizes=[16])
        expected = ((x - x.mean(-1, keepdim=True)) ** 2).sum(-1)
        torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-3)

    @skipIfRefEager("inspects the reduction_loops config surface")
    @skipIfPallas("Pallas lowers full slices without reduction loops")
    @skipIfMetal("not verified on Metal")
    def test_rows_stored_at_tile_positions_are_not_rolled(self) -> None:
        """``out[tile] = buf[:, :].sum(-1)`` stores the 64 slice rows at the
        tile's positions.  Rolling the row dim would hand each chunk of rows
        to a store that names all 64 tile positions (and zero only one chunk
        of a full-slice store), so only the reduced column dim is rollable."""

        x = torch.randn(64, device=DEVICE)
        buf = torch.randn(64, 128, device=DEVICE)
        bound = tile_row_sums.bind((x, buf))
        rollable = [
            bound.env.block_sizes[spec.block_ids[0]].size
            for spec in bound.config_spec.reduction_loops
        ]
        self.assertEqual(rollable, [128])
        if _get_backend() == "cute":
            # The rows still reach the tile's lanes through an exchange that
            # CuTe cannot place inside the lane loop this layout needs: it
            # must refuse, not return a partial result.
            with self.assertRaises(helion.exc.BackendUnsupported):
                code_and_output(tile_row_sums, (x, buf))
            return
        _, out = code_and_output(tile_row_sums, (x, buf), reduction_loops=[32])
        torch.testing.assert_close(out, buf.sum(-1), rtol=1e-4, atol=1e-4)

    @skipIfRefEager("reduction_loops only exists in compiled mode")
    @skipIfPallas("Pallas lowers full slices without reduction loops")
    @skipIfMetal("not verified on Metal")
    def test_rolled_column_sum_beside_a_live_row_axis(self) -> None:
        """``out[tile] = buf[:, :].sum(-1)`` with the column dim rolled: the
        row dim stays persistent and, allocated first, takes the lowest thread
        axis on CuTe, below the column chunk's lanes.  The chunk's finalize
        must combine the column lanes of each row, not consecutive lanes
        (which are rows); a plain 4-lane warp reduce summed 4 rows.  Covers
        the grouped warp reduce and the two-stage shared reduce."""

        @helion.kernel(static_shapes=True)
        def row_sums(x: torch.Tensor, buf: torch.Tensor) -> torch.Tensor:
            out = torch.empty([x.size(0)], dtype=torch.float32, device=x.device)
            for tile in hl.tile(x.size(0), block_size=x.size(0)):
                out[tile] = buf[:, :].sum(-1)
            return out

        for rows, cols, chunk in (
            (8, 16, 4),
            (8, 64, 4),
            (8, 64, 32),
            (8, 256, 16),
            (4, 512, 8),
            (4, 512, 64),
            (16, 128, 16),
        ):
            with self.subTest(rows=rows, cols=cols, chunk=chunk):
                x = torch.randn(rows, device=DEVICE)
                buf = torch.randn(rows, cols, device=DEVICE)
                _, out = code_and_output(row_sums, (x, buf), reduction_loops=[chunk])
                torch.testing.assert_close(out, buf.sum(-1), rtol=1e-4, atol=1e-4)

    @skipIfRefEager("reduction_loops only exists in compiled mode")
    @skipIfPallas("Pallas lowers full slices without reduction loops")
    @skipIfMetal("not verified on Metal")
    def test_reduction_chunk_lookup_skips_an_unregistered_earlier_dim(self) -> None:
        """``config.reduction_loops`` has a slot per *rollable* dim only.  The
        row dim of ``buf[:, :].sum(-1)`` stored at tile positions is not
        rollable, so the column dim allocated after it owns slot 0: a chunk
        lookup by allocation order handed the column chunk to the rows."""

        bound = tile_row_sums.bind(
            (torch.randn(64, device=DEVICE), torch.randn(64, 128, device=DEVICE))
        )
        config = helion.Config(block_sizes=[64], reduction_loops=[32])
        with bound.env:
            chunks = {
                info.size: info.from_config(config)
                for info in bound.env.block_sizes
                if info.reduction
            }
        self.assertEqual(chunks, {64: 64, 128: 32})

    @skipIfNotCUDA()
    @skipIfRefEager(
        "promoted-seed reduction_loops is only materialized in compiled mode"
    )
    @skipIfTileIR("TileIR reduction tiling differs")
    def test_mid_axis_reduce_wide_feature_default_config(self) -> None:
        """Regression: a reduction over a small middle axis co-resident with a wide
        feature ([M, R, N].sum(1), R small, N large) run with the promoted default (no
        explicit config) must produce a VALID reduction_loops and match the reference.
        The byte-budget reduction seed used to collapse the rolled chunk to
        ``reduction_loops=[1]``, which ``LoopedReductionStrategy`` rejects (block_size >
        1); the ``ReductionLoopSpec._normalize`` floor now repairs it. Shapes are kept
        small so the persistent tile fits a small GPU under parallel test execution."""

        @helion.kernel(autotune_effort="none")
        def mid_axis_reduce(x: torch.Tensor) -> torch.Tensor:
            m, _r, n = x.shape
            out = torch.empty([m, n], dtype=torch.float32, device=x.device)
            for tile_m in hl.tile(m):
                out[tile_m, :] = x[tile_m, :, :].to(torch.float32).sum(1)
            return out

        x = torch.randn([8, 4, 2048], device=DEVICE, dtype=torch.float32)
        expected = x.sum(1)
        _code, out = code_and_output(mid_axis_reduce, (x,))
        torch.testing.assert_close(out, expected, rtol=1e-3, atol=1e-3)


def _integer_loop_kernel():
    @helion.kernel(backend="cute", autotune_effort="none")
    def kernel(
        x: torch.Tensor,
        scratch: torch.Tensor,
        minimum: hl.constexpr,
        early: hl.constexpr,
        carry_input: hl.constexpr,
        side_effect: hl.constexpr,
        identity: hl.constexpr,
        begin: hl.constexpr,
        extra_mask: hl.constexpr,
        preserve_alias: hl.constexpr,
    ):
        rows, width = x.shape
        out = torch.empty((rows,), dtype=x.dtype, device=x.device)
        for row in hl.tile(rows):
            best = hl.full([row], identity, dtype=x.dtype)
            initial = best
            for col in hl.tile(begin, width):
                value = x[row, col]
                if extra_mask:
                    value = torch.where(col.index[None, :] % 3 == 1, identity, value)
                if carry_input:
                    value = torch.maximum(value, best[:, None])
                if minimum:
                    reduced = value.amin(-1)
                    best = torch.minimum(best, reduced)
                else:
                    reduced = value.amax(-1)
                    best = torch.maximum(best, reduced)
                if early:
                    scratch[row] = reduced
                if side_effect:
                    scratch[row] = best
            out[row] = best
            if preserve_alias:
                scratch[row] = initial
        return out

    return kernel


def _integer_loop_code(x, *, minimum=False, threads=128, enabled=True, **kw):
    from test._cute_binding import _cpu_bind
    from test._cute_binding import _forbid_native_compile
    from test._cute_binding import _mock_cuda_unavailable
    from test.cute_population_contracts import _target

    kernel = _integer_loop_kernel()
    from types import SimpleNamespace
    from unittest.mock import patch

    with (
        _mock_cuda_unavailable(),
        _target(),
        _forbid_native_compile(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CPU only")),
        patch("torch.cuda.current_device", return_value=0),
        patch(
            "torch.cuda.get_device_properties",
            return_value=SimpleNamespace(
                shared_memory_per_block_optin=232448, shared_memory_per_block=49152
            ),
        ),
    ):
        bound = _cpu_bind(
            kernel,
            (
                x,
                torch.zeros(x.shape[0], dtype=x.dtype),
                minimum,
                kw.get("early", False),
                kw.get("carry_input", False),
                kw.get("side_effect", False),
                (torch.iinfo(x.dtype).max if minimum else torch.iinfo(x.dtype).min)
                if not x.is_floating_point()
                else (float("inf") if minimum else -float("inf")),
                kw.get("begin", 0),
                kw.get("extra_mask", False),
                kw.get("preserve_alias", False),
            ),
        )
        config = {"block_sizes": [1, threads], "num_threads": [1, threads]}
        config.update(kw.get("config_override", {}))
        if enabled:
            config["cute_integer_loop_reduction"] = True
        return bound, bound.to_code(helion.Config(**config))


class TestIntegerLoopReductionCPU(unittest.TestCase):
    def test_integer_loop_collective_placement(self):
        import ast

        for dtype in (torch.int32, torch.int64):
            for minimum in (False, True):
                with self.subTest(dtype=dtype, minimum=minimum):
                    _, old = _integer_loop_code(
                        torch.zeros((3, 513), dtype=dtype),
                        minimum=minimum,
                        enabled=False,
                    )
                    _, new = _integer_loop_code(
                        torch.zeros((3, 513), dtype=dtype), minimum=minimum
                    )
                    self.assertNotEqual(old, new)

                    def collective_depth(code):
                        fn = next(
                            n
                            for n in ast.parse(code).body
                            if isinstance(n, ast.FunctionDef)
                            and n.name.startswith("_helion")
                        )
                        calls = []

                        def visit(n, depth):
                            if (
                                isinstance(n, ast.Call)
                                and isinstance(n.func, ast.Name)
                                and n.func.id == "_cute_grouped_reduce_shared_two_stage"
                            ):
                                calls.append(depth)
                            for c in ast.iter_child_nodes(n):
                                visit(c, depth + isinstance(n, ast.For))

                        visit(fn, 0)
                        return calls

                    self.assertEqual(collective_depth(old), [1])
                    self.assertEqual(collective_depth(new), [0])


def _integer_loop_model(source, x, *, return_scratch=False):
    """Execute the generated scalar statements over one complete physical CTA.

    Only the unchanged shared reducer helper is modeled as its exact typed
    collective. Load masks, loop/carry/phi statements and output stores execute.
    """
    import ast
    import operator
    from types import SimpleNamespace

    import numpy as np

    tree = ast.parse(source)
    fn = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name.startswith("_helion")
    )
    threads = next(
        k.value.elts[0].value
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        for k in n.keywords
        if k.arg == "block"
    )

    class Vectorize(ast.NodeTransformer):
        def visit_IfExp(self, n):
            self.generic_visit(n)
            return ast.Call(
                ast.Attribute(ast.Name("np", ast.Load()), "where", ast.Load()),
                [n.test, n.body, n.orelse],
                [],
            )

        def visit_BoolOp(self, n):
            self.generic_visit(n)
            result = n.values[0]
            for other in n.values[1:]:
                result = ast.Call(
                    ast.Attribute(
                        ast.Name("np", ast.Load()),
                        "logical_and" if isinstance(n.op, ast.And) else "logical_or",
                        ast.Load(),
                    ),
                    [result, other],
                    [],
                )
            return result

    class Pointer:
        def __init__(self, array, index=0):
            self.array = array
            self.index = index

        def __add__(self, index):
            return Pointer(self.array, self.index + index)

        def load(self):
            index = np.asarray(self.index)
            valid = (index >= 0) & (index < self.array.size)
            return np.where(
                valid, self.array.reshape(-1)[np.clip(index, 0, self.array.size - 1)], 0
            )

        def store(self, value):
            indices, values = np.broadcast_arrays(self.index, value)
            for index in np.unique(indices):
                selected = values[indices == index]
                assert np.all(selected == selected[0]), "divergent duplicate writers"
                self.array.reshape(-1)[index] = selected[0]

    def tensor(array):
        return SimpleNamespace(
            iterator=Pointer(array),
            layout=SimpleNamespace(
                stride=tuple(s // array.itemsize for s in array.strides)
            ),
        )

    events = []
    row = [0]

    def collective(value, op, identity, *args, **kwargs):
        assert kwargs == {"pre": 1, "group_span": threads, "group_count": 1}
        assert args[0].shape == (threads,)
        a = np.asarray(value)
        v = a.max() if op == "max" else a.min()
        events.append(op)
        return np.full(threads, v, dtype=a.dtype)

    ns = {
        "np": np,
        "operator": operator,
        "cutlass": SimpleNamespace(Int32=np.int32, Int64=np.int64),
        "cute": SimpleNamespace(
            arch=SimpleNamespace(
                thread_idx=lambda: (np.arange(threads, dtype=np.int32), 0, 0),
                block_idx=lambda: (row[0], 0, 0),
            ),
            math=SimpleNamespace(
                max=lambda a, b, **kw: np.maximum(a, b),
                min=lambda a, b, **kw: np.minimum(a, b),
            ),
        ),
        "_cute_grouped_reduce_shared_two_stage": collective,
        "_cute_python_mod": operator.mod,
        "_cute_join_cast": lambda value, like: np.asarray(
            value, dtype=np.asarray(like).dtype
        ),
    }
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and all(isinstance(t, ast.Name) for t in node.targets)
            and isinstance(node.value, ast.Constant)
        ):
            for target in node.targets:
                ns[target.id] = node.value.value
    fn.decorator_list = []
    fn.returns = None
    for arg in fn.args.args:
        arg.annotation = None
    module = ast.fix_missing_locations(
        Vectorize().visit(ast.Module(body=[fn], type_ignores=[]))
    )
    exec(compile(module, "<integer-loop-generated-model>", "exec"), ns)
    array = x.numpy()
    out = np.zeros(array.shape[0], dtype=array.dtype)
    scratch = np.zeros_like(out)
    arguments = {"x": tensor(array), "out": tensor(out), "scratch": tensor(scratch)}
    for i in range(array.shape[0]):
        row[0] = i
        ns[fn.name](*[arguments[a.arg] for a in fn.args.args])
    if return_scratch:
        return torch.from_numpy(out), torch.from_numpy(scratch), events
    return torch.from_numpy(out), events


class TestIntegerLoopReductionValuesCPU(unittest.TestCase):
    def test_exact_tails_extrema_and_duplicate_stores(self):
        for dtype in (torch.int32, torch.int64):
            limits = torch.iinfo(dtype)
            for threads, width in ((64, 129), (128, 513), (512, 1025), (1024, 2049)):
                x = torch.randint(-500, 500, (3, width), dtype=dtype)
                x[0, -1] = limits.max
                x[1, -1] = limits.min
                x[2].fill_(limits.min)
                for minimum in (False, True):
                    with self.subTest(dtype=dtype, threads=threads, minimum=minimum):
                        _, old = _integer_loop_code(
                            x, threads=threads, minimum=minimum, enabled=False
                        )
                        _, new = _integer_loop_code(x, threads=threads, minimum=minimum)
                        a, ae = _integer_loop_model(old, x)
                        b, be = _integer_loop_model(new, x)
                        expected = x.amin(-1) if minimum else x.amax(-1)
                        torch.testing.assert_close(a, expected, atol=0, rtol=0)
                        torch.testing.assert_close(b, expected, atol=0, rtol=0)
                        self.assertEqual(
                            len(ae), 3 * ((width + threads - 1) // threads)
                        )
                        self.assertEqual(len(be), 3)

    def test_early_read_carry_input_and_mutation_reject(self):
        from helion.exc import InvalidConfig

        x = torch.zeros((3, 513), dtype=torch.int64)
        for key in ("early", "carry_input", "side_effect"):
            with self.subTest(key=key), self.assertRaises(InvalidConfig):
                _integer_loop_code(x, **{key: True})

    def test_single_trip_and_lane_loop_fallback(self):
        for width, threads in ((17, 128), (128, 128)):
            with self.subTest(width=width):
                x = torch.zeros((3, width), dtype=torch.int64)
                _, old = _integer_loop_code(x, threads=threads, enabled=False)
                _, new = _integer_loop_code(x, threads=threads)
                self.assertEqual(old, new)


class TestIntegerLoopReductionProofCPU(unittest.TestCase):
    def test_physical_subgroups_and_serial_lanes_keep_old_schedule(self):
        x = torch.zeros((5, 1025), dtype=torch.int64)
        for blocks, threads in (
            ([4, 128], [4, 32]),
            ([1, 256], [1, 128]),
            ([1, 32], [1, 32]),
        ):
            with self.subTest(blocks=blocks, threads=threads):
                config = {"block_sizes": blocks, "num_threads": threads}
                _, old = _integer_loop_code(x, enabled=False, config_override=config)
                _, new = _integer_loop_code(x, config_override=config)
                self.assertEqual(old, new)

    def test_begin_extra_mask_and_initial_alias(self):
        for minimum in (False, True):
            for begin in (3, 513):
                with self.subTest(minimum=minimum, begin=begin):
                    x = torch.arange(3 * 513, dtype=torch.int64).reshape(3, 513) - 400
                    identity = (
                        torch.iinfo(x.dtype).max
                        if minimum
                        else torch.iinfo(x.dtype).min
                    )
                    _, old = _integer_loop_code(
                        x,
                        minimum=minimum,
                        begin=begin,
                        extra_mask=True,
                        preserve_alias=True,
                        enabled=False,
                    )
                    _, new = _integer_loop_code(
                        x,
                        minimum=minimum,
                        begin=begin,
                        extra_mask=True,
                        preserve_alias=True,
                    )
                    a, initial, _ = _integer_loop_model(old, x, return_scratch=True)
                    b, other, events = _integer_loop_model(new, x, return_scratch=True)
                    kept = x[:, begin:].clone()
                    kept[:, torch.arange(begin, 513) % 3 == 1] = identity
                    expected = (
                        (kept.amin(-1) if minimum else kept.amax(-1))
                        if kept.shape[1]
                        else torch.full((3,), identity, dtype=x.dtype)
                    )
                    self.assertTrue(
                        torch.equal(a, expected) and torch.equal(b, expected)
                    )
                    self.assertTrue(
                        torch.equal(initial, torch.full((3,), identity, dtype=x.dtype))
                    )
                    self.assertTrue(torch.equal(initial, other))
                    self.assertEqual(len(events), 3 if begin < 513 else 0)

    def test_float_reduction_rejects(self):
        with self.assertRaises(helion.exc.InvalidConfig):
            _integer_loop_code(torch.zeros((3, 513), dtype=torch.float32))

    def test_strict_boolean_and_deferred_population(self):
        import random
        from unittest.mock import patch

        from test._cute_binding import _forbid_native_compile
        from test._cute_binding import _mock_cuda_unavailable
        from test.cute_population_contracts import _target
        from test.cute_population_contracts import checked_initial_population

        from helion.autotuner.pattern_search import InitialPopulationStrategy
        from helion.autotuner.pattern_search import PatternSearch

        x = torch.zeros((3, 513), dtype=torch.int64)
        bound, _ = _integer_loop_code(x)
        args = (
            x,
            torch.zeros(3, dtype=x.dtype),
            False,
            False,
            False,
            False,
            torch.iinfo(x.dtype).min,
            0,
            False,
            False,
        )
        spec = bound.config_spec
        key = "cute_integer_loop_reduction"
        self.assertNotIn(key, spec.default_config().config)
        self.assertTrue(spec.compiler_coverage_groups[-1].deferred)
        self.assertEqual(spec.compiler_coverage_groups[-1].key, key)
        with bound.env, _mock_cuda_unavailable(), _target(), _forbid_native_compile():
            config = dict(spec.default_config().config)
            for bad in (1, "true", None):
                with self.subTest(bad=bad), self.assertRaises(helion.exc.InvalidConfig):
                    spec.create_config_generation().strict_config_pair(
                        helion.Config(**(config | {key: bad}))
                    )
            for seed in (0, 107):
                for strategy in (
                    InitialPopulationStrategy.FROM_RANDOM,
                    InitialPopulationStrategy.FROM_BEST_AVAILABLE,
                ):
                    random.seed(seed)
                    with (
                        patch.object(
                            spec, "cute_integer_loop_reduction_search_enabled", False
                        ),
                        patch.object(
                            spec,
                            "_compiler_coverage_groups",
                            spec.compiler_coverage_groups[:-1],
                        ),
                    ):
                        old = PatternSearch(
                            bound,
                            args,
                            initial_population=100,
                            initial_population_strategy=strategy,
                        )
                        a = [
                            dict(old.config_gen.canonicalize_flat(row)[1])
                            for row in checked_initial_population(old)
                        ]
                        rng = random.getstate()
                    random.seed(seed)
                    new = PatternSearch(
                        bound,
                        args,
                        initial_population=100,
                        initial_population_strategy=strategy,
                    )
                    b = [
                        dict(new.config_gen.canonicalize_flat(row)[1])
                        for row in checked_initial_population(new)
                    ]
                    self.assertEqual(a, b[: len(a)])
                    self.assertEqual(rng, random.getstate())
                    self.assertTrue(any(row.get(key) is True for row in b))
        _, witness = _integer_loop_code(x, config_override=b[-1])
        import ast

        for loop in ast.walk(ast.parse(witness)):
            if isinstance(loop, ast.For):
                self.assertFalse(
                    any(
                        isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Name)
                        and node.func.id == "_cute_grouped_reduce_shared_two_stage"
                        for node in ast.walk(loop)
                    )
                )

    @skipUnlessCuteAvailable("requires CuTe DSL")
    def test_actual_sdk_integer_minmax(self):
        import ast
        import importlib.util
        from pathlib import Path
        import sys
        import tempfile
        from unittest.mock import patch

        import cutlass
        from cutlass._mlir import ir
        from cutlass._mlir.dialects import func
        import cutlass.cute as cute

        for dtype, typ in ((torch.int32, cutlass.Int32), (torch.int64, cutlass.Int64)):
            for minimum in (False, True):
                with self.subTest(dtype=dtype, minimum=minimum):
                    _, code = _integer_loop_code(
                        torch.zeros((3, 513), dtype=dtype), minimum=minimum
                    )
                    tree = ast.parse(code)
                    fn = next(
                        n
                        for n in tree.body
                        if isinstance(n, ast.FunctionDef)
                        and n.name.startswith("_helion")
                    )
                    fn.decorator_list = [ast.parse("cute.jit", mode="eval").body]
                    tree.body = [
                        n
                        for n in tree.body
                        if isinstance(n, (ast.Import, ast.ImportFrom))
                        or isinstance(n, ast.Assign)
                        and all(isinstance(t, ast.Name) for t in n.targets)
                    ] + [fn]
                    with tempfile.TemporaryDirectory() as tmp:
                        path = Path(tmp) / "module.py"
                        path.write_text(ast.unparse(ast.fix_missing_locations(tree)))
                        spec = importlib.util.spec_from_file_location(
                            "integer_loop_sdk", path
                        )
                        mod = importlib.util.module_from_spec(spec)
                        sys.modules[spec.name] = mod
                        with patch(
                            "torch.cuda._lazy_init",
                            side_effect=AssertionError("CPU only"),
                        ):
                            spec.loader.exec_module(mod)
                            with ir.Context(), ir.Location.unknown():
                                emitted = ir.Module.create()
                                with ir.InsertionPoint(emitted.body):
                                    entry = func.FuncOp("entry", ([], []))
                                    with ir.InsertionPoint(entry.add_entry_block()):
                                        args = [
                                            cute.make_tensor(
                                                cute.make_ptr(
                                                    typ,
                                                    0,
                                                    cute.AddressSpace.gmem,
                                                    assumed_align=16,
                                                ),
                                                cute.make_layout(
                                                    (3, 513), stride=(513, 1)
                                                )
                                                if a.arg == "x"
                                                else cute.make_layout(
                                                    (3,), stride=(1,)
                                                ),
                                            )
                                            for a in fn.args.args
                                        ]
                                        getattr(mod, fn.name)(*args)
                                        func.ReturnOp([])
                                self.assertTrue(emitted.operation.verify())
                                self.assertIn("nvvm.barrier", str(emitted))


@onlyBackends("cute")
class TestIntegerLoopReductionNative(TestCase):
    def test_integer_minmax_tails(self):
        for dtype, threads in ((torch.int32, 64), (torch.int64, 1024)):
            x = torch.arange(3 * 2051, device=DEVICE, dtype=dtype).reshape(3, 2051)
            x[0, -1] = torch.iinfo(dtype).max
            x[1, -1] = torch.iinfo(dtype).min
            x[2].fill_(torch.iinfo(dtype).min)
            original = x.clone()
            for minimum in (False, True):
                for enabled in (False, True):
                    scratch = torch.zeros(3, device=DEVICE, dtype=dtype)
                    identity = (
                        torch.iinfo(dtype).max if minimum else torch.iinfo(dtype).min
                    )
                    args = (
                        x,
                        scratch,
                        minimum,
                        False,
                        False,
                        False,
                        identity,
                        0,
                        False,
                        False,
                    )
                    bound = _integer_loop_kernel().bind(args)
                    out = bound.compile_config(
                        helion.Config(
                            block_sizes=[1, threads],
                            num_threads=[1, threads],
                            cute_integer_loop_reduction=enabled,
                        )
                    )(*args)
                    expected = x.amin(-1) if minimum else x.amax(-1)
                    torch.testing.assert_close(out, expected, rtol=0, atol=0)
                    torch.testing.assert_close(x, original, rtol=0, atol=0)
                    self.assertEqual(scratch.count_nonzero().item(), 0)

    def test_masked_and_empty_ranges_preserve_initial_alias(self):
        x = (
            torch.arange(3 * 513, device=DEVICE, dtype=torch.int64).reshape(3, 513)
            - 800
        )
        original = x.clone()
        identity = torch.iinfo(x.dtype).min
        for begin in (0, 3, 513):
            for enabled in (False, True):
                scratch = torch.zeros(3, device=DEVICE, dtype=x.dtype)
                args = (
                    x,
                    scratch,
                    False,
                    False,
                    False,
                    False,
                    identity,
                    begin,
                    True,
                    True,
                )
                bound = _integer_loop_kernel().bind(args)
                out = bound.compile_config(
                    helion.Config(
                        block_sizes=[1, 128],
                        num_threads=[1, 128],
                        cute_integer_loop_reduction=enabled,
                    )
                )(*args)
                kept = x[:, begin:].clone()
                kept[:, torch.arange(begin, 513, device=DEVICE) % 3 == 1] = identity
                expected = (
                    kept.amax(-1) if begin < 513 else torch.full_like(scratch, identity)
                )
                torch.testing.assert_close(out, expected, rtol=0, atol=0)
                torch.testing.assert_close(
                    scratch, torch.full_like(scratch, identity), rtol=0, atol=0
                )
                torch.testing.assert_close(x, original, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
