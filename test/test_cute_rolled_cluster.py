"""Tests for the ``cute_cluster_n`` split of rolled (LoopedReductionStrategy)
row reductions.

For a rolled row reduction (rmsnorm/layernorm family: one grid tile over
rows, the full reduction dim rolled in a loop), ``cute_cluster_n > 1``
splits each row's roll range across the CTAs of a thread-block cluster;
one DSM exchange combines the partials (every CTA receives the full
result) and the consume sweep stores only the CTA's slice.
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
def rms_norm_rolled_kernel(
    x: torch.Tensor, weight: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mirrors ``examples/rms_norm.py::rms_norm_fwd``."""
    m, n = x.size()
    out = torch.empty_like(x)
    inv_rms = torch.empty([m], dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(m):
        x_tile = x[tile_m, :].to(torch.float32)
        mean_x_squared = torch.mean(x_tile * x_tile, dim=-1)
        inv_rms_tile = torch.rsqrt(mean_x_squared + 1e-6)
        out[tile_m, :] = (
            x_tile * inv_rms_tile[:, None] * weight[:].to(torch.float32)
        ).to(out.dtype)
        inv_rms[tile_m] = inv_rms_tile
    return out, inv_rms


@helion.kernel(backend="cute")
def row_amax_rolled_kernel(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty([m], dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(m):
        out[tile_m] = torch.amax(x[tile_m, :].to(torch.float32), dim=-1)
    return out


@helion.kernel(backend="cute")
def int_rowsum_rolled_kernel(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty([m], dtype=torch.int64, device=x.device)
    for tile_m in hl.tile(m):
        out[tile_m] = torch.sum(x[tile_m, :].to(torch.int64), dim=-1)
    return out


@helion.kernel(backend="cute")
def rowsum_atomic_total_kernel(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    total = torch.zeros([1], dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(m):
        s = torch.sum(x[tile_m, :].to(torch.float32), dim=-1)
        hl.atomic_add(total, [0], s.sum())
    return total


def _rms_ref(
    x: torch.Tensor, weight: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    xf = x.float()
    inv_rms = torch.rsqrt(xf.square().mean(dim=-1) + 1e-6)
    return (xf * inv_rms[:, None] * weight.float()).to(x.dtype), inv_rms


@onlyBackends(["cute"])
class TestCuteRolledCluster(TestCase):
    def test_rolled_cluster_fires_bf16_cl4(self) -> None:
        """bf16 rmsnorm with ``cute_cluster_n=4``: the roll range is sliced
        by the cluster rank, the finalize is one DSM cluster exchange, and
        the launch carries the cluster shape."""
        x = torch.randn(32, 8192, device=DEVICE, dtype=torch.bfloat16)
        weight = torch.randn(8192, device=DEVICE, dtype=torch.bfloat16)
        code, (out, inv_rms) = code_and_output(
            rms_norm_rolled_kernel,
            (x, weight),
            block_sizes=[1],
            num_threads=[1, 128],
            reduction_loops=[2048],
            cute_vector_widths=[8, 1],
            cute_cluster_n=4,
        )
        ref_out, ref_inv_rms = _rms_ref(x, weight)
        torch.testing.assert_close(out, ref_out, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(inv_rms, ref_inv_rms, rtol=1e-3, atol=1e-3)
        self.assertIn("_cute_grouped_reduce_cluster(", code)
        self.assertNotIn("_cute_grouped_reduce_shared_two_stage(", code)
        # Rank-sliced roll range (slice = 8192 / 4) on both sweeps.
        self.assertEqual(code.count("cute.arch.block_idx()[1]) * 2048"), 4)
        self.assertIn("_helion_cute_cluster_shape = (1, 4, 1)", code)

    def test_rolled_cluster_correct_fp32_cl2_multitrip(self) -> None:
        """fp32 (vec mode) with two roll trips per slice."""
        x = torch.randn(16, 8192, device=DEVICE, dtype=torch.float32)
        weight = torch.randn(8192, device=DEVICE, dtype=torch.float32)
        code, (out, inv_rms) = code_and_output(
            rms_norm_rolled_kernel,
            (x, weight),
            block_sizes=[1],
            num_threads=[1, 128],
            reduction_loops=[2048],
            cute_vector_widths=[4, 1],
            cute_cluster_n=2,
        )
        ref_out, ref_inv_rms = _rms_ref(x, weight)
        torch.testing.assert_close(out, ref_out, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(inv_rms, ref_inv_rms, rtol=1e-4, atol=1e-4)
        self.assertIn("_cute_grouped_reduce_cluster(", code)
        self.assertIn("_helion_cute_cluster_shape = (1, 2, 1)", code)

    def test_rolled_cluster_max_reduction(self) -> None:
        x = torch.randn(32, 8192, device=DEVICE, dtype=torch.bfloat16)
        code, out = code_and_output(
            row_amax_rolled_kernel,
            (x,),
            block_sizes=[1],
            num_threads=[1, 128],
            reduction_loops=[2048],
            cute_vector_widths=[8, 1],
            cute_cluster_n=2,
        )
        torch.testing.assert_close(out, x.float().amax(dim=-1))
        self.assertIn("_cute_grouped_reduce_cluster(", code)

    def test_rolled_cluster_skipped_indivisible(self) -> None:
        """When the extent does not split into whole per-CTA chunk multiples
        the knob is silently inert (single-CTA rolled codegen, still
        correct)."""
        x = torch.randn(32, 6144, device=DEVICE, dtype=torch.bfloat16)
        weight = torch.randn(6144, device=DEVICE, dtype=torch.bfloat16)
        code, (out, inv_rms) = code_and_output(
            rms_norm_rolled_kernel,
            (x, weight),
            block_sizes=[1],
            num_threads=[1, 128],
            reduction_loops=[2048],
            cute_vector_widths=[8, 1],
            cute_cluster_n=4,
        )
        ref_out, ref_inv_rms = _rms_ref(x, weight)
        torch.testing.assert_close(out, ref_out, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(inv_rms, ref_inv_rms, rtol=1e-3, atol=1e-3)
        self.assertNotIn("_cute_grouped_reduce_cluster(", code)
        self.assertNotIn("_helion_cute_cluster_shape", code)

    def test_rolled_cluster_skipped_multirow_cta(self) -> None:
        """Multiple rows per CTA (a sibling thread axis) would make the
        whole-CTA cluster reduce fold unrelated rows — the knob must stay
        inert."""
        x = torch.randn(32, 4096, device=DEVICE, dtype=torch.bfloat16)
        weight = torch.randn(4096, device=DEVICE, dtype=torch.bfloat16)
        code, (out, inv_rms) = code_and_output(
            rms_norm_rolled_kernel,
            (x, weight),
            block_sizes=[4],
            num_threads=[4, 64],
            reduction_loops=[1024],
            cute_vector_widths=[8, 1],
            cute_cluster_n=2,
        )
        ref_out, ref_inv_rms = _rms_ref(x, weight)
        torch.testing.assert_close(out, ref_out, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(inv_rms, ref_inv_rms, rtol=1e-3, atol=1e-3)
        self.assertNotIn("_cute_grouped_reduce_cluster(", code)
        self.assertNotIn("_helion_cute_cluster_shape", code)

    def test_rolled_cluster_rejects_non_fp32_acc(self) -> None:
        """The cluster exchange buffers partials as fp32; a wider
        accumulator must hard-fail instead of silently losing precision."""
        x = torch.randint(0, 4, (16, 4096), device=DEVICE, dtype=torch.int32)
        cfg = {
            "block_sizes": [1],
            "num_threads": [1, 128],
            "reduction_loops": [1024],
            "cute_vector_widths": [1, 1],
        }
        # Same config runs (exactly) without the cluster.
        code, out = code_and_output(
            int_rowsum_rolled_kernel, (x,), **cfg, cute_cluster_n=1
        )
        self.assertTrue(torch.equal(out, x.to(torch.int64).sum(dim=-1)))
        with self.assertRaisesRegex(
            helion.exc.BackendUnsupported, "fp32 reduction accumulator"
        ):
            code_and_output(int_rowsum_rolled_kernel, (x,), **cfg, cute_cluster_n=2)

    def test_rolled_cluster_skipped_with_atomics(self) -> None:
        """Statements outside the roll loop run once per cluster CTA, so a
        kernel using atomics (read-modify-write would replay cluster_n
        times) must keep the knob inert."""
        x = torch.randn(16, 4096, device=DEVICE, dtype=torch.bfloat16)
        code, out = code_and_output(
            rowsum_atomic_total_kernel,
            (x,),
            block_sizes=[1],
            num_threads=[1, 128],
            reduction_loops=[1024],
            cute_vector_widths=[1, 1],
            cute_cluster_n=2,
        )
        torch.testing.assert_close(out[0], x.float().sum(), rtol=1e-2, atol=1.0)
        self.assertNotIn("_cute_grouped_reduce_cluster(", code)
        self.assertNotIn("_helion_cute_cluster_shape", code)


if __name__ == "__main__":
    import unittest

    unittest.main()
