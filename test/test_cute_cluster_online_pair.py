"""Tests for the cluster online-pair rewrite (``cluster_online_pair``).

For an online-softmax reduction pair split across a thread-block cluster,
the rewrite replaces the two DSM cluster exchanges (max, then sum) with a
CTA-local block reduce plus ONE packed ``(max, sum)`` exchange folded with
the online-softmax rescale, and reuses the sum sweep's cached exp values
in the write sweep.
"""

from __future__ import annotations

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
from helion._testing import skipIfRefEager
import helion.language as hl

cutlass = pytest.importorskip("cutlass")
cute = pytest.importorskip("cutlass.cute")


@helion.kernel(backend="cute")
def softmax_two_pass_kernel(x: torch.Tensor) -> torch.Tensor:
    """Mirrors ``examples/softmax.py::softmax_two_pass``."""
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


@helion.kernel(backend="cute", fast_math=True)
def softmax_fast_math_kernel(x: torch.Tensor) -> torch.Tensor:
    """Same as ``softmax_two_pass_kernel`` but with the fast_math setting."""
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


@helion.kernel(backend="cute", static_shapes=True)
def softmax_numerator_and_max(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """A three-pass softmax that stores its row max and the unnormalized
    ``exp(x - max)``, so neither hides behind the row sum."""
    m, n = x.size()
    out = torch.empty_like(x)
    row_max = torch.empty([m], dtype=torch.float32, device=x.device)
    block_size_m = hl.register_block_size(m)
    block_size_n = hl.register_block_size(n)
    for tile_m in hl.tile(m, block_size=block_size_m):
        mi = hl.full([tile_m], float("-inf"), dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=block_size_n):
            mi = torch.maximum(mi, torch.amax(x[tile_m, tile_n], dim=1))
        di = hl.zeros([tile_m], dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=block_size_n):
            di = di + torch.exp(x[tile_m, tile_n] - mi[:, None]).sum(dim=1)
        row_max[tile_m] = mi
        for tile_n in hl.tile(n, block_size=block_size_n):
            out[tile_m, tile_n] = torch.exp(x[tile_m, tile_n] - mi[:, None])
    return out, row_max


@helion.kernel(backend="cute", static_shapes=True)
def softmax_shifted_by_max_of_other(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """``sum(exp(y - max(x)))``: the exponent's minuend is not the value the
    max reduced, so a CTA's own max does not bound it."""
    m, n = x.size()
    out = torch.empty_like(x)
    sums = torch.empty([m], dtype=torch.float32, device=x.device)
    block_size_m = hl.register_block_size(m)
    block_size_n = hl.register_block_size(n)
    for tile_m in hl.tile(m, block_size=block_size_m):
        mi = hl.full([tile_m], float("-inf"), dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=block_size_n):
            mi = torch.maximum(mi, torch.amax(x[tile_m, tile_n], dim=1))
        di = hl.zeros([tile_m], dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=block_size_n):
            di = di + torch.exp(y[tile_m, tile_n] - mi[:, None]).sum(dim=1)
        sums[tile_m] = di
        for tile_n in hl.tile(n, block_size=block_size_n):
            out[tile_m, tile_n] = (
                torch.exp(x[tile_m, tile_n] - mi[:, None]) / di[:, None]
            )
    return out, sums


def _row_config(columns: int, dtype: torch.dtype) -> dict[str, object]:
    return {
        "block_sizes": [1, columns],
        "num_threads": [0, 256],
        "cute_vector_widths": [1, 4 if dtype == torch.float32 else 8],
        "cute_lane_layouts": ["blocked", "strided"],
    }


@helion.kernel(backend="cute")
def rowsum_kernel(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty([m], dtype=torch.float32, device=x.device)
    block_size_m = hl.register_block_size(m)
    block_size_n = hl.register_block_size(n)
    for tile_m in hl.tile(m, block_size=block_size_m):
        acc = hl.zeros([tile_m], dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=block_size_n):
            acc = acc + x[tile_m, tile_n].to(torch.float32).sum(dim=1)
        out[tile_m] = acc
    return out


@onlyBackends(["cute"])
class TestCuteClusterOnlinePair(TestCase):
    def test_pair_rewrite_fires_bf16_cl2(self) -> None:
        """Under ``fast_math`` (the pair matches its distributed scale, and
        its rescaled exponentials round differently), the softmax pattern
        with ``cute_cluster_n=2`` must compile to a single packed pair
        exchange: a CTA-local block reduce for the max,
        ``_cute_grouped_reduce_cluster_online_pair`` for the sum, an exp
        cache written by the sum sweep, and a rescale in the write sweep
        instead of an exp2 recompute."""
        x = torch.randn(64, 32768, device=DEVICE, dtype=torch.bfloat16)
        code, out = code_and_output(
            softmax_fast_math_kernel,
            (x,),
            block_sizes=[1, 32768],
            num_threads=[0, 256],
            cute_vector_widths=[1, 8],
            cute_lane_layouts=["blocked", "strided"],
            cute_cluster_n=2,
        )
        ref = torch.nn.functional.softmax(x, dim=1)
        torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-3)
        self.assertIn("_cute_grouped_reduce_cluster_online_pair(", code)
        self.assertIn("_cute_grouped_reduce_block(", code)
        # Both two-exchange call sites must be gone (exact-name match; the
        # pair helper's name extends it, so check the call spelling).
        self.assertEqual(code.count("_cute_grouped_reduce_cluster("), 0)
        self.assertIn("_pair_exp_cache_0", code)
        self.assertIn("_pair_rescale_0", code)
        # The pair exchange receives cluster_n Int64 (8-byte pair) slots.
        self.assertIn("cute.arch.alloc_smem(cutlass.Int64, 2)", code)

    def test_pair_rewrite_correct_fp16_cl4(self) -> None:
        x = torch.randn(32, 65536, device=DEVICE, dtype=torch.float16)
        code, out = code_and_output(
            softmax_fast_math_kernel,
            (x,),
            block_sizes=[1, 65536],
            num_threads=[0, 256],
            cute_vector_widths=[1, 8],
            cute_lane_layouts=["blocked", "strided"],
            cute_cluster_n=4,
            cute_min_blocks_per_mp=3,
        )
        ref = torch.nn.functional.softmax(x, dim=1)
        torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-3)
        self.assertIn("_cute_grouped_reduce_cluster_online_pair(", code)

    def test_exp2_fastmath_setting(self) -> None:
        """The ``fast_math`` SETTING (not a config — configs must not
        change numerics) puts ``fastmath=True`` on every emitted
        ``cute.math.exp2`` call; the default keeps the exact
        (denormal-preserving) lowering."""
        x = torch.randn(64, 8192, device=DEVICE, dtype=torch.bfloat16)
        code, out = code_and_output(
            softmax_fast_math_kernel,
            (x,),
            block_sizes=[1, 8192],
            num_threads=[0, 128],
            cute_vector_widths=[1, 8],
            cute_lane_layouts=["blocked", "strided"],
        )
        ref = torch.nn.functional.softmax(x, dim=1)
        torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-3)
        self.assertGreater(code.count("cute.math.exp2("), 0)
        self.assertEqual(code.count("cute.math.exp2("), code.count("fastmath=True"))

    def test_exp2_exact_by_default(self) -> None:
        """Without the setting, no config may introduce fastmath exp2; the
        exchanges pair without caching or rescaling exponentials."""
        x = torch.randn(64, 32768, device=DEVICE, dtype=torch.bfloat16)
        code, out = code_and_output(
            softmax_two_pass_kernel,
            (x,),
            block_sizes=[1, 32768],
            num_threads=[0, 256],
            cute_vector_widths=[1, 8],
            cute_lane_layouts=["blocked", "strided"],
            cute_cluster_n=2,
        )
        ref = torch.nn.functional.softmax(x, dim=1)
        torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-3)
        self.assertNotIn("fastmath=True", code)
        self.assertEqual(code.count("_cute_grouped_reduce_cluster_online_pair("), 1)
        self.assertEqual(code.count("_cute_grouped_reduce_cluster("), 0)
        self.assertNotIn("_pair_exp_cache_", code)
        self.assertNotIn("_pair_rescale_", code)

    @skipIfRefEager("checks the generated code")
    def test_default_pair_matches_the_unclustered_formula(self) -> None:
        """Without fast_math each output is ``exp2((x - max) * C) / sum`` as
        in the unclustered kernel; only the row sum is reassociated (CTA
        sums folded with the online rescale).  So per row the ratio to the
        unclustered output is one constant, bf16 outputs are at most one
        ulp apart, and -inf slices, -inf rows, NaN and +inf rows agree."""
        for dtype, columns, cluster in (
            (torch.float32, 32768, 2),
            (torch.float32, 65536, 4),
            (torch.bfloat16, 32768, 2),
            (torch.bfloat16, 65536, 4),
        ):
            torch.manual_seed(0)
            x = torch.randn(16, columns, device=DEVICE, dtype=dtype) * 3
            segment = columns // cluster
            x[1:5] = -torch.inf
            x[1, :segment] = -1000.0  # finite in the first CTA only
            x[2, segment : 2 * segment] = -1.0  # finite in the second only
            x[4, -1] = torch.inf
            x[5, 7] = torch.nan
            x[6, :segment] = -torch.inf  # one empty slice
            x[7, ::2] = -0.0
            config = _row_config(columns, dtype)
            with self.subTest(dtype=dtype, cluster=cluster):
                code, out = code_and_output(
                    softmax_two_pass_kernel, (x,), **config, cute_cluster_n=cluster
                )
                self.assertIn("_cute_grouped_reduce_cluster_online_pair(", code)
                self.assertNotIn("_pair_exp_cache_", code)
                _, expected = code_and_output(softmax_two_pass_kernel, (x,), **config)
                self.assertTrue(torch.equal(out.isnan(), expected.isnan()))
                self.assertTrue(torch.equal(out == 0, expected == 0))
                finite = (expected != 0) & ~expected.isnan()
                if dtype == torch.bfloat16:
                    ulps = (out.view(torch.int16) - expected.view(torch.int16)).abs()
                    self.assertLessEqual(int(ulps[finite].max()), 1)
                else:
                    # out / expected = (S' / S)(1 + d1) / (1 + d2) with the
                    # two divisions' roundings |d1|, |d2| < 2**-24, and S' / S
                    # within a few ulps of 1.
                    ratio = out.double() / expected.double()
                    high = torch.where(finite, ratio, -torch.inf).amax(1)
                    low = torch.where(finite, ratio, torch.inf).amin(1)
                    spread = (high - low)[finite.any(1)]
                    self.assertLessEqual(float(spread.max()), 4.01 * 2**-24)
                    torch.testing.assert_close(
                        out, expected, rtol=1e-5, atol=0, equal_nan=True
                    )

    @skipIfRefEager("checks the generated code")
    def test_paired_max_propagates_nan_like_the_unclustered_max(self) -> None:
        """The packed exchange's row max is NaN-propagating: a NaN in one
        CTA's slice makes the stored max and every unnormalized exponential
        of the row NaN, exactly as without clustering (and torch.amax)."""
        for dtype, columns, cluster in (
            (torch.float32, 32768, 2),
            (torch.float32, 65536, 4),
            (torch.bfloat16, 32768, 2),
            (torch.bfloat16, 65536, 4),
        ):
            torch.manual_seed(0)
            x = torch.randn(8, columns, device=DEVICE, dtype=dtype)
            segment = columns // cluster
            x[1, 3] = torch.nan  # in the first CTA's slice only
            x[2, -5] = torch.nan  # in the last CTA's slice only
            x[3, :segment] = -torch.inf  # one empty slice
            x[4] = -torch.inf
            x[5, segment] = torch.inf
            config = _row_config(columns, dtype)
            with self.subTest(dtype=dtype, cluster=cluster):
                code, (out, row_max) = code_and_output(
                    softmax_numerator_and_max, (x,), **config, cute_cluster_n=cluster
                )
                self.assertIn("_cute_grouped_reduce_cluster_online_pair(", code)
                _, (expected_out, expected_max) = code_and_output(
                    softmax_numerator_and_max, (x,), **config
                )
                torch.testing.assert_close(
                    row_max, expected_max, rtol=0, atol=0, equal_nan=True
                )
                self.assertTrue(row_max[1:3].isnan().all())
                torch.testing.assert_close(
                    out, expected_out, rtol=0, atol=0, equal_nan=True
                )

    @skipIfRefEager("checks the generated code")
    def test_minuend_other_than_the_reduced_value_keeps_two_exchanges(
        self,
    ) -> None:
        """For ``sum(exp(y - max(x)))`` a CTA's max of x does not bound y:
        an all--inf x slice with finite y, or a y far above that CTA's max,
        would lose or overflow terms in the CTA frame.  The exchanges stay
        unpaired and match the unclustered kernel."""
        for columns, cluster in ((32768, 2), (65536, 4)):
            torch.manual_seed(0)
            x = torch.randn(4, columns, device=DEVICE)
            y = torch.randn(4, columns, device=DEVICE)
            segment = columns // cluster
            x[1, :segment] = -torch.inf
            x[2, :segment] = 45.0
            x[2, segment:] = 90.0
            y[2, :segment] = 140.0
            config = _row_config(columns, torch.float32)
            with self.subTest(cluster=cluster):
                code, (out, sums) = code_and_output(
                    softmax_shifted_by_max_of_other,
                    (x, y),
                    **config,
                    cute_cluster_n=cluster,
                )
                self.assertNotIn("_cute_grouped_reduce_cluster_online_pair(", code)
                _, (expected_out, expected_sums) = code_and_output(
                    softmax_shifted_by_max_of_other, (x, y), **config
                )
                torch.testing.assert_close(sums, expected_sums, rtol=1e-5, atol=0)
                torch.testing.assert_close(
                    out, expected_out, rtol=1e-5, atol=0, equal_nan=True
                )

    def test_serial_block_reduce_routing(self) -> None:
        """Pin the serial-vs-two-stage dispatch contract: the cheap serial
        cross-warp combine only serves single whole-CTA groups of 2..8
        warps; everything else must keep the two-stage form."""
        from helion._compiler.cute.reduce_helpers import _use_serial_block_reduce

        self.assertTrue(_use_serial_block_reduce(1, 128, 1))
        self.assertTrue(_use_serial_block_reduce(1, 256, 1))
        # 16 warps: serial chain outgrows the two-stage shuffle cost.
        self.assertFalse(_use_serial_block_reduce(1, 512, 1))
        # single warp: the warp path handles it without shared memory.
        self.assertFalse(_use_serial_block_reduce(1, 32, 1))
        # pre-grouped or multi-group reductions keep the two-stage form.
        self.assertFalse(_use_serial_block_reduce(2, 128, 1))
        self.assertFalse(_use_serial_block_reduce(1, 128, 4))
        # non-warp-multiple spans keep the two-stage form.
        self.assertFalse(_use_serial_block_reduce(1, 48, 1))

    def test_single_site_keeps_two_exchange_form(self) -> None:
        """A cluster kernel without the (max, sum-of-exp) pair keeps the
        plain per-site cluster exchange."""
        x = torch.randn(64, 32768, device=DEVICE, dtype=torch.bfloat16)
        code, out = code_and_output(
            rowsum_kernel,
            (x,),
            block_sizes=[1, 32768],
            num_threads=[0, 256],
            cute_vector_widths=[1, 8],
            cute_lane_layouts=["blocked", "strided"],
            cute_cluster_n=2,
        )
        ref = x.to(torch.float32).sum(dim=1)
        torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-2)
        self.assertIn("_cute_grouped_reduce_cluster(", code)
        self.assertNotIn("_cute_grouped_reduce_cluster_online_pair(", code)


if __name__ == "__main__":
    import unittest

    unittest.main()
