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
        """The softmax pattern with ``cute_cluster_n=2`` must compile to a
        single packed pair exchange: a CTA-local block reduce for the max,
        ``_cute_grouped_reduce_cluster_online_pair`` for the sum, an exp
        cache written by the sum sweep, and a rescale in the write sweep
        instead of an exp2 recompute."""
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
            softmax_two_pass_kernel,
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
        """Without the setting, no config may introduce fastmath exp2 —
        including the cluster pair rewrite's helper fold."""
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
