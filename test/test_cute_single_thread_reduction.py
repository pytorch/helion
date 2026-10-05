"""Regression coverage for CuTe reductions shrunk to a single live thread.

``CuteBackend.adjust_reduction_thread_count`` collapses a reduction's thread
count to 1 once the competing tile/reduction axes exhaust the CTA thread
budget. ``TileStrategy._compute_thread_axis_offset`` then lets a tile strategy
share that reduction's thread axis, so the reduction must index with a constant
0 rather than ``thread_idx()`` (which would alias the tile's thread id and
scatter the slice store across the wrong columns / out of bounds).  Both the
persistent (synthetic lane) and the rolled (``reduction_loops``) index forms
are covered.
"""

from __future__ import annotations

import torch

import helion
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
import helion.language as hl


@helion.kernel(backend="cute", static_shapes=True)
def _concat2d_dim1_slices(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.size(0) == y.size(0)
    out = torch.empty(
        [x.size(0), x.size(1) + y.size(1)], dtype=x.dtype, device=x.device
    )
    n1 = x.size(1)
    for tile_m in hl.tile(x.size(0)):
        out[tile_m, :n1] = x[tile_m, :]
        out[tile_m, n1:] = y[tile_m, :]
    return out


@helion.kernel(backend="cute", static_shapes=True)
def _row_sum(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty([x.size(0)], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(x.size(0)):
        out[tile_m] = x[tile_m, :].sum(-1)
    return out


def _synthetic_lane_index_lines(code: str) -> list[str]:
    """Return the index assignments driven by a synthetic reduction lane loop."""
    return [
        line
        for line in code.splitlines()
        if "indices_" in line
        and "synthetic_lane_" in line
        and "=" in line
        and not line.lstrip().startswith("for ")
    ]


def _rolled_lane_index_lines(code: str) -> list[str]:
    """Return the index assignments driven by a rolled reduction's lane loop."""
    return [
        line
        for line in code.splitlines()
        if "reduction_lane_" in line
        and "=" in line
        and not line.lstrip().startswith("for ")
    ]


@onlyBackends(["cute"])
class TestCuteSingleThreadReduction(TestCase):
    def _check(self, rows: int, x_cols: int, y_cols: int) -> None:
        torch.manual_seed(0)
        x = torch.randn(rows, x_cols, device=DEVICE)
        y = torch.randn(rows, y_cols, device=DEVICE)
        # 32 tile rows x 32 rolled-reduction threads fill the 1024-thread CTA
        # budget, so the full-slice ``y`` dim is shrunk to one live thread that
        # shares the tile's thread axis.
        code, out = code_and_output(
            _concat2d_dim1_slices,
            (x, y),
            block_sizes=[32],
            reduction_loops=[32],
        )
        torch.testing.assert_close(out, torch.cat((x, y), dim=1))
        lane_lines = _synthetic_lane_index_lines(code)
        self.assertTrue(lane_lines, code)
        for line in lane_lines:
            self.assertNotIn("thread_idx", line, code)

    def test_slice_store_single_thread_reduction_matches_cat(self) -> None:
        self._check(256, 128, 256)

    def test_slice_store_single_thread_reduction_ragged(self) -> None:
        self._check(64, 33, 257)

    def test_rolled_single_thread_reduction_indexes_with_constant(self) -> None:
        torch.manual_seed(0)
        x = torch.randn(2048, 192, device=DEVICE)
        # A 1024-thread tile exhausts the CTA budget, so the rolled column
        # reduction is shrunk to one live thread: its per-iteration index is
        # ``roffset + 0 + reduction_lane * 1``, never ``thread_idx()``.
        code, out = code_and_output(
            _row_sum, (x,), block_sizes=[1024], reduction_loops=[64]
        )
        torch.testing.assert_close(out, x.sum(-1), rtol=1e-4, atol=1e-4)
        lane_lines = _rolled_lane_index_lines(code)
        self.assertTrue(lane_lines, code)
        for line in lane_lines:
            self.assertNotIn("thread_idx", line, code)
