"""A CuTe reduction on a thread axis above another live axis folds only its own lanes.

Reductions take the leading thread axes in creation order, so in
``x[tile, :, :].sum(-1)`` the kept middle full slice claims axis 0 and the
reduced last dim axis 1, and the reduce must group its lanes by the extent of
the axes below it (``PersistentReductionStrategy._cute_sibling_axis_reduction_expr``).
test_reductions.py covers power-of-two extents at one block size; these cases
add masked (non power-of-two) reduced and kept extents, block sizes that move
the group across the warp boundary, a pointwise consumer of the reduction, and
the plain warp reduce kept for a reduction on the lowest axis.
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
def _sum_and_argmax_last(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    sums = torch.empty([x.size(0), x.size(1)], device=x.device)
    argmaxes = torch.empty([x.size(0), x.size(1)], device=x.device, dtype=torch.int64)
    for tile_m in hl.tile(x.size(0)):
        sums[tile_m, :] = x[tile_m, :, :].sum(-1)
        argmaxes[tile_m, :] = x[tile_m, :, :].argmax(-1)
    return sums, argmaxes


@helion.kernel(backend="cute", static_shapes=True)
def _amax_last_plus_scaled(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile_m in hl.tile(x.size(0)):
        out[tile_m, :, :] = x[tile_m, :, :] * 2 + x[tile_m, :, :].amax(-1, keepdim=True)
    return out


@helion.kernel(backend="cute", static_shapes=True)
def _sum_middle(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty([x.size(0), x.size(2)], device=x.device)
    for tile_m in hl.tile(x.size(0)):
        out[tile_m, :] = x[tile_m, :, :].sum(1)
    return out


# (shape, block size, grouped reduce helper): groups within one warp and
# groups spanning warps, with masked reduced (3, 5) and kept extents.
_CASES = (
    ((128, 32, 3), 8, "_cute_grouped_reduce_shared_two_stage("),
    ((64, 32, 16), 2, "_cute_grouped_reduce_shared_two_stage("),
    ((64, 8, 5), 8, "_cute_grouped_reduce_shared_two_stage("),
    ((32, 16, 64), 1, "_cute_grouped_reduce_shared_two_stage("),
    ((16, 4, 8), 4, "_cute_grouped_reduce_warp("),
    ((24, 3, 5), 2, "_cute_grouped_reduce_warp("),
)


@onlyBackends(["cute"])
class TestCuteReductionAboveSiblingAxis(TestCase):
    def test_sum_and_argmax_over_upper_axis(self) -> None:
        for shape, block_size, helper in _CASES:
            torch.manual_seed(0)
            x = torch.randn(*shape, device=DEVICE)
            with self.subTest(shape=shape, block_size=block_size):
                code, (sums, argmaxes) = code_and_output(
                    _sum_and_argmax_last, (x,), block_sizes=[block_size]
                )
                torch.testing.assert_close(sums, x.sum(-1), rtol=1e-4, atol=1e-4)
                torch.testing.assert_close(argmaxes, x.argmax(-1))
                self.assertIn(helper, code)

    def test_amax_over_upper_axis_beside_pointwise(self) -> None:
        for shape, block_size, _helper in _CASES:
            torch.manual_seed(0)
            x = torch.randn(*shape, device=DEVICE)
            with self.subTest(shape=shape, block_size=block_size):
                _code, out = code_and_output(
                    _amax_last_plus_scaled, (x,), block_sizes=[block_size]
                )
                torch.testing.assert_close(out, x * 2 + x.amax(-1, keepdim=True))

    def test_lowest_axis_reduction_keeps_the_plain_warp_reduce(self) -> None:
        torch.manual_seed(0)
        x = torch.randn(128, 32, 3, device=DEVICE)
        code, out = code_and_output(_sum_middle, (x,), block_sizes=[8])
        torch.testing.assert_close(out, x.sum(1), rtol=1e-4, atol=1e-4)
        self.assertIn("cute.arch.warp_reduction_sum(", code)
        self.assertNotIn("_cute_grouped_reduce", code)
