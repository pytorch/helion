"""Tests for the CuTe warp-per-row layout (multi-row CTAs with
one warp per row).

For 2D reduction kernels (outer M-grid tile + inner N-reduction tile)
where:

- 1 outer grid block (M), 1 inner non-reduction tile (N)
- N is a 32-multiple
- joint thread count <= 1024
- no MMA

``layout_propagation.py`` detects the shape and emits a
``CuTeGridExecutionPlan(block_axis_priority={N: 0, M: 1})``. The
thread-axis assignment then swaps so:

- N (inner reduction axis) lands on ``thread_idx[0]`` (32 contiguous
  lanes = one warp per row)
- M (outer grid row axis) lands on ``thread_idx[1]`` (warp index = row
  index)

The reduction dispatcher's ``group_span`` becomes the per-row size
(32 — one warp), NOT the per-CTA size (M_block * 32), so it falls
through to ``_cute_grouped_reduce_warp`` (single warp shuffle, no
shared memory, no sync_threads).

Lives in:
- ``helion/_compiler/cute/layout_propagation.py``
  (``_plan_warp_per_row_execution`` shape detector)
- ``helion/_compiler/tile_strategy.py`` (``_compute_thread_axis_offset``
  honors ``block_axis_priority``)
- ``helion/_compiler/tile_dispatch.py`` (``thread_axis_for_strategy``
  dedups sibling strategies with identical block_ids in the same branch)
- ``helion/_compiler/autotuner_heuristics/cute.py``
  (``CuteTileVecWarpPerRowHeuristic`` seeds the multi-row config)
"""

from __future__ import annotations

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import HALF_DTYPE
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
import helion.language as hl

cutlass = pytest.importorskip("cutlass")
cute = pytest.importorskip("cutlass.cute")


@pytest.fixture(autouse=True)
def _disable_online_to_3pass(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests in this file pin codegen details of the ORIGINAL online
    two-pass form.  The ``online_to_3pass`` rewrite would change them,
    so disable it here; the rewrite itself is covered in
    ``test_cute_online_to_3pass.py``.
    """
    monkeypatch.setenv("HELION_DISABLE_ONLINE_TO_3PASS", "1")


@helion.kernel(backend="cute")
def _reduction_kernel(x: torch.Tensor) -> torch.Tensor:
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


@onlyBackends(["cute"])
class TestCuteWarpPerRowLayout(TestCase):
    def test_multi_row_threaded_uses_warp_per_row(self) -> None:
        """``block_sizes=[8, 128], num_threads=[0, 32]`` — M is threaded.

        With the warp-per-row layout, the codegen swaps the thread-axis
        assignment so N (the reduction axis) lands on ``thread_idx[0]``
        (32 contiguous threads = one warp per row) and M lands on
        ``thread_idx[1]`` (warp index = row index).  Each warp's
        per-row reduction uses ``_cute_grouped_reduce_warp`` with
        ``pre=1, group_span=32`` (one warp shuffle per warp), NOT
        the cross-warp ``_cute_grouped_reduce_shared_two_stage`` path.

        The launch becomes ``block=(32, 8, 1)``: joint thread count
        is 8 × 32 = 256 (8 warps per CTA for higher occupancy on
        reduction-shaped kernels).
        """
        x = torch.randn(4096, 128, device=DEVICE, dtype=HALF_DTYPE)
        code, out = code_and_output(
            _reduction_kernel,
            (x,),
            block_sizes=[8, 128],
            num_threads=[0, 32],
            cute_vector_widths=[1, 4],
        )
        ref = torch.nn.functional.softmax(x, dim=1)
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
        # Warp-per-row launch: N (the reduction axis with 32 threads)
        # lands on axis 0, M (8 rows) lands on axis 1.
        self.assertIn("block=(32, 8, 1)", code)
        # Each warp reduces its own row via _cute_grouped_reduce_warp.
        self.assertIn("_cute_grouped_reduce_warp", code)
        self.assertIn("pre=1, group_span=32", code)
        # NO cross-warp SMEM reduce.
        self.assertNotIn("_cute_grouped_reduce_shared_two_stage", code)

    def test_warp_per_row_axis_swap_emits_warp_reduce(self) -> None:
        """The warp-per-row plan swaps the thread-axis assignment so:

        * N (inner reduction axis) lands on ``thread_idx[0]``
        * M (outer grid row axis) lands on ``thread_idx[1]``

        And the reduction dispatcher picks the per-warp
        ``_cute_grouped_reduce_warp`` path with ``pre=1, group_span=32``
        instead of the cross-warp SMEM path that would dominate without
        the axis swap.
        """
        x = torch.randn(4096, 12672, device=DEVICE, dtype=HALF_DTYPE)
        code, out = code_and_output(
            _reduction_kernel,
            (x,),
            block_sizes=[2, 128],
            num_threads=[0, 32],
            cute_vector_widths=[1, 4],
        )
        ref = torch.nn.functional.softmax(x, dim=1)
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
        # The launch is ``block=(32, 2, 1)`` — N on axis 0 (32 threads
        # per warp = one row), M on axis 1 (2 warps = 2 rows).
        self.assertIn("block=(32, 2, 1)", code)
        # M uses ``thread_idx[1]`` for the row index.
        self.assertIn(
            "indices_0 = tile_offset_0 + cutlass.Int32(cute.arch.thread_idx()[1])",
            code,
        )
        # N uses ``thread_idx[0]`` for the lane (32 contiguous lanes).
        self.assertIn("cute.arch.thread_idx()[0]) * 4", code)
        # Per-warp reduce.
        self.assertIn("_cute_grouped_reduce_warp", code)
        self.assertIn("pre=1, group_span=32", code)
        # NO shared-memory two-stage reduce.
        self.assertNotIn("_cute_grouped_reduce_shared_two_stage", code)

    def test_single_row_baseline_uses_warp_reduce(self) -> None:
        """Single-row baseline: ``block_sizes=[1, 128]`` with
        ``num_threads=[0, 32]`` produces one CTA per row, each with one
        warp of 32 threads. The reduce uses
        ``cute.arch.warp_reduction_max``, NOT the two-stage SMEM helper.
        Pin this so a future regression in heuristic seed coverage
        would be caught.
        """
        x = torch.randn(4096, 128, device=DEVICE, dtype=HALF_DTYPE)
        code, out = code_and_output(
            _reduction_kernel,
            (x,),
            block_sizes=[1, 128],
            num_threads=[0, 32],
            cute_vector_widths=[1, 4],
        )
        ref = torch.nn.functional.softmax(x, dim=1)
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
        # One warp per CTA.
        self.assertIn("block=(32, 1, 1)", code)
        # Warp reduce, no shared-mem reduction.
        self.assertIn("cute.arch.warp_reduction_max", code)
        self.assertNotIn("_cute_grouped_reduce_shared_two_stage", code)

    def test_serial_multi_row_does_not_swap_axes(self) -> None:
        """``block_sizes=[8, 128], num_threads=[1, 32]`` — M is SERIALIZED
        via a ``for lane_0 in range(8)`` loop, not threaded across warps.
        The warp-per-row axis swap MUST NOT fire here (it only applies
        when M is on the thread axis).  The single-warp ``block=(32, 1, 1)``
        layout stays; only the grid count drops (M / 8 CTAs).
        """
        x = torch.randn(4096, 128, device=DEVICE, dtype=HALF_DTYPE)
        code, out = code_and_output(
            _reduction_kernel,
            (x,),
            block_sizes=[8, 128],
            num_threads=[1, 32],
            cute_vector_widths=[1, 4],
        )
        ref = torch.nn.functional.softmax(x, dim=1)
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
        # M serialized — a ``for lane_0 in range(8):`` loop wraps the
        # per-row body.
        self.assertIn("for lane_0 in range(8):", code)
        # Single-warp launch (axis swap did NOT fire).
        self.assertIn("block=(32, 1, 1)", code)
        # Warp reduce still fires since group_span = 32.
        self.assertIn("cute.arch.warp_reduction_max", code)
