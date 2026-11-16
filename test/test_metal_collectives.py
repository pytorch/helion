from __future__ import annotations

import math
import sys
import unittest

import pytest
import torch

from helion._compiler.metal.msl_reduction import REDUCTION_PREAMBLE
from helion._compiler.metal.reduction import group_path
from helion._compiler.metal.reduction import group_scratch_slots

# Metal runs on macOS only: CI's Linux jobs skip the whole file.
pytestmark = pytest.mark.skipif(
    sys.platform != "darwin", reason="Metal tests require macOS"
)

DEVICE = "mps"


@unittest.skipUnless(torch.backends.mps.is_available(), "requires a Metal GPU")
class TestLaunchedAxisCollectives(unittest.TestCase):
    def _check(
        self, dims: tuple[int, int, int], axis: int, span: int, special: str = ""
    ) -> None:
        count = math.prod(dims)
        path = group_path(axis, dims)
        slots = max(1, group_scratch_slots(path, dims, axis))
        tx, ty, tz = dims
        shader = f"""
#include <metal_stdlib>
#include <c10/metal/reduction_utils.h>
using namespace metal;
{REDUCTION_PREAMBLE}
[[required_threads_per_threadgroup({tx}, {ty}, {tz})]]
kernel void collect(device const float* x, device float* out, device long* idx,
                    uint3 tid [[thread_position_in_threadgroup]]) {{
  const uint lane = tid.x + {tx} * (tid.y + {ty} * tid.z);
  const uint pos = tid[{axis}];
  const long input_index = pos < {span} ? long(100 + pos) : long(0);
  threadgroup float s0[{slots}], s1[{slots}], s2[{slots}], s3[{slots}];
  threadgroup float av0[{slots}], av1[{slots}];
  threadgroup long ai0[{slots}], ai1[{slots}];
  const float v = x[lane];
  const uint path = helion_red::path_{path};
  out[lane] = helion_red::group_sum(path, s0, v, tid, {axis}, {span}, {tx}, {ty}, {tz});
  out[{count} + lane] = helion_red::group_prod(path, s1, v, tid, {axis}, {span}, {tx}, {ty}, {tz});
  out[{2 * count} + lane] = helion_red::group_max(path, s2, v, tid, {axis}, {span}, {tx}, {ty}, {tz});
  out[{3 * count} + lane] = helion_red::group_min(path, s3, v, tid, {axis}, {span}, {tx}, {ty}, {tz});
  idx[lane] = helion_red::group_argmax(path, av0, ai0, v, input_index, tid, {axis}, {span}, {tx}, {ty}, {tz});
  idx[{count} + lane] = helion_red::group_argmin(path, av1, ai1, v, input_index, tid, {axis}, {span}, {tx}, {ty}, {tz});
}}
"""
        generator = torch.Generator().manual_seed(42)
        host = 1 + 0.005 * torch.randn(count, generator=generator)
        below = math.prod(dims[:axis])
        extent = dims[axis]
        positions = torch.arange(count) // below % extent
        if special == "negative_inf":
            host.fill_(float("-inf"))
        elif special == "positive_inf":
            host.fill_(float("inf"))
        elif special == "nan":
            host[positions == span // 2] = float("nan")
        # Nonmembers deliberately have non-neutral values and smaller indices
        # than all real inputs. Both the value and candidate masks are needed.
        host[positions >= span] = 1000
        expected = torch.empty(4, count)
        expected_idx = torch.empty(2, count, dtype=torch.int64)
        for lane in range(count):
            first = lane - int(positions[lane]) * below
            row = host[first : first + span * below : below]
            expected[:, lane] = torch.stack(
                (row.sum(), row.prod(), row.amax(), row.amin())
            )
            expected_idx[:, lane] = torch.stack((row.argmax(), row.argmin())) + 100
        out = torch.empty_like(expected, device=DEVICE)
        idx = torch.empty_like(expected_idx, device=DEVICE)
        library = torch.mps.compile_shader(shader)
        library.collect(host.to(DEVICE), out, idx, threads=dims, group_size=dims)
        torch.testing.assert_close(
            out.cpu(), expected, rtol=1e-4, atol=1e-4, equal_nan=True
        )
        torch.testing.assert_close(idx.cpu(), expected_idx)

    def test_full_launch_receives_each_rows_result(self) -> None:
        cases = (
            ((16, 2, 1), 0, 8),
            ((2, 16, 2), 1, 8),
            ((2, 64, 1), 1, 64),
            ((3, 2, 128), 2, 128),
            ((3, 12, 2), 1, 4),
            ((12, 2, 1), 0, 8),
            ((6, 64, 1), 0, 2),
            ((32, 16, 1), 1, 16),
        )
        for dims, axis, span in cases:
            with self.subTest(dims=dims, axis=axis, span=span):
                self._check(dims, axis, span)

    def test_power_of_two_strides_take_one_pass(self) -> None:
        """Rows a power of two apart reduce in one pass for all rows: within a
        SIMD group (size <= 32), over SIMD groups that each hold several rows
        (below < 32), or one row per lane (below >= 32), with spans narrower
        than the axis."""
        cases = (
            ((2, 4, 1), 1, 4),
            ((4, 8, 1), 1, 8),
            ((8, 16, 2), 1, 16),
            ((16, 8, 2), 1, 8),
            ((4, 4, 16), 2, 16),
            ((64, 4, 2), 1, 4),
            ((16, 4, 8), 2, 8),
            ((16, 4, 8), 2, 4),
            ((128, 4, 1), 1, 2),
        )
        for dims, axis, span in cases:
            with self.subTest(dims=dims, axis=axis, span=span):
                self.assertEqual(group_path(axis, dims), "strided")
                self._check(dims, axis, span)
        for dims, axis in (((16, 2, 1), 1), ((8, 16, 2), 1), ((16, 4, 8), 2)):
            for special in ("negative_inf", "positive_inf", "nan"):
                with self.subTest(dims=dims, axis=axis, special=special):
                    self._check(dims, axis, 2, special)

    def test_nonmembers_cannot_win_argreduction(self) -> None:
        for dims, axis in (((16, 2, 1), 0), ((2, 64, 1), 1), ((12, 2, 1), 0)):
            for special in ("negative_inf", "positive_inf", "nan"):
                with self.subTest(dims=dims, axis=axis, special=special):
                    self._check(dims, axis, 8, special)
