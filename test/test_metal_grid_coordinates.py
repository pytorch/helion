from __future__ import annotations

import re

import pytest
import torch

import helion
from helion._testing import skipIfRefEagerMark
from helion._testing import skipUnlessBackends
from helion._testing.cute_codegen import forbid_cuda_work
from helion._testing.metal import render_msl
import helion.language as hl

pytestmark = [
    pytest.mark.cpu_only,
    skipUnlessBackends(["metal"]),
    skipIfRefEagerMark("reads generated MSL"),
]


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


def test_static_partial_axis_divisor_cannot_overflow_msl_int() -> None:
    @helion.kernel(backend="metal", static_shapes=True)
    def write(out):
        for i, j in hl.tile([2**31 - 1, 2**31 - 1], block_size=[4, 4]):
            out[i.index % 8, j.index % 8] = 1

    with forbid_cuda_work():
        msl = render_msl(
            write,
            (torch.empty(8, 8, dtype=torch.int32),),
            helion.Config(block_sizes=[], flatten_loops=[False], pid_type="flat"),
        )
    # The PID decoder divides by ceil(INT32_MAX / 4). An addition to
    # INT32_MAX involving an MSL constexpr int would already overflow.
    assert "536870912" in msl


def test_scalar_argument_index_reads_its_buffer_after_an_alias_store() -> None:
    @helion.kernel(backend="metal", static_shapes=True)
    def gather(x, plane, alias, out):
        for i in hl.grid(1):
            p = plane
            alias[i] = 1
            out[i] = hl.load(x, [p])

    plane = torch.tensor(0)
    with forbid_cuda_work():
        msl = render_msl(
            gather,
            (
                torch.tensor([10, 20]),
                plane,
                plane.view(1),
                torch.empty(1, dtype=torch.int64),
            ),
            helion.Config(block_sizes=[]),
        )
    # At entry plane is zero. Reading it after the alias store picks x[1].
    after_store = msl.split("alias[", 1)[1].split(";", 1)[1]
    read = re.search(r"auto (\w+) = plane\[0\];", after_store)
    assert read and re.search(rf"x\[[^;]*\b{read[1]}\b", after_store[read.end() :]), msl
