from __future__ import annotations

import re

import pytest
import torch

import helion
from helion._testing import skipUnlessBackends
from helion._testing.cute_codegen import forbid_cuda_work
from helion._testing.metal import render_msl
import helion.language as hl

pytestmark = [pytest.mark.cpu_only, skipUnlessBackends(["metal"])]


def test_negative_runtime_grid_origin_reaches_the_signed_branch() -> None:
    @helion.kernel(backend="metal", static_shapes=False)
    def mark(out):
        for r in hl.grid(-1, out.size(0) - 1):
            if r < 0:
                out[r + 1] = 31
            else:
                out[r + 1] = r

    with forbid_cuda_work():
        msl = render_msl(
            mark, (torch.empty(7, dtype=torch.int32),), helion.Config(block_sizes=[])
        )
    # Group zero must take the negative branch. Its physical ID and TID are uint.
    assert "static_cast<long>(pid_flat)" in msl
    match = re.search(r"auto (\w+) = \(\(-1\) \+ offsets_0\);", msl)
    assert match and f"({match[1]} < 0)" in msl


def test_shifted_tile_keeps_the_lane_signed_in_its_tail_mask() -> None:
    @helion.kernel(backend="metal", static_shapes=True)
    def increment(out):
        for r, c in hl.tile([-2, 0], [5, 3], block_size=[4, 2]):
            out[r + 2, c] = out[r + 2, c] + 1

    with forbid_cuda_work():
        msl = render_msl(
            increment,
            (torch.empty(7, 3, dtype=torch.int32),),
            helion.Config(block_sizes=[], flatten_loops=[False]),
        )
    # The first two lanes have coordinates -2 and -1. Converting either
    # to uint at the addition would make the upper-bound mask drop the store.
    assert "static_cast<long>(pid_0) * _BLOCK_SIZE_0" in msl
    assert "auto indices_0 = (offset_0 + tid[0]);" in msl
    assert "(indices_0 < 5)" in msl
