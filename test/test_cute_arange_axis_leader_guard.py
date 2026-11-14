"""A tile-uniform atomic keeps its leader guard on a free ``hl.arange`` dim's
thread axis when its tile walks lanes.

The CuTe passes prove bounds from the strategies' static launch shape
(``_exact_thread_block_dims`` in ``helion/_compiler/device_function.py``),
which holds a one for an axis only a free ``hl.arange`` dim claims
(``Backend.synthetic_thread_axis_sizes``).  With 64 rows on 16 threads beside
a 128-wide ``hl.arange`` (launch (16, 64, 1)), the lane-tile bounds simplifier
proved the atomic's guard ``cute.arch.thread_idx()[1] == 0`` true and dropped
it, so every thread of the arange's axis added each element: 64 adds per
element (128 on 4 threads).  A tile without lanes kept its guard.
"""

from __future__ import annotations

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import skipIfNotCUDA
from helion._testing import skipUnlessBackends
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import mock_cuda_unavailable
import helion.language as hl

pytestmark = [pytest.mark.cpu_only, skipUnlessBackends(["cute"])]


def _arange_then_add(
    x: torch.Tensor, y: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    doubled = torch.empty_like(y)
    for tile in hl.tile(x.size(0)):
        cols = hl.arange(y.size(1))
        doubled[tile, cols] = y[tile, cols] * 2
        hl.atomic_add(out, [tile], x[tile])
    return doubled


# Lane-walking tiles: 64 rows on 16 and on 4 threads beside the 128 columns.
_LANE_TILES = {
    "16 threads": (helion.Config(block_sizes=[64], num_threads=[16]), (16, 64, 1)),
    "4 threads": (helion.Config(block_sizes=[64], num_threads=[4]), (4, 128, 1)),
}


@pytest.mark.parametrize("name", list(_LANE_TILES))
def test_an_atomic_beside_an_arange_axis_keeps_its_leader_guard(name: str) -> None:
    config, launch = _LANE_TILES[name]
    kernel = helion.kernel(_arange_then_add, backend="cute", static_shapes=True)
    args = (torch.randn(64), torch.randn(64, 128), torch.zeros(64))
    with mock_cuda_unavailable():
        code = cpu_bind(kernel, args).to_code(config)
    assert f"block={launch}" in code, code
    lines = [line.strip() for line in code.splitlines()]
    (atomic,) = [i for i, line in enumerate(lines) if "cute.arch.atomic_add(" in line]
    assert lines[atomic - 1] == "if cute.arch.thread_idx()[1] == 0:", code


@pytest.mark.cuda
@skipIfNotCUDA()
@pytest.mark.parametrize("name", list(_LANE_TILES))
def test_cuda_an_atomic_beside_an_arange_axis_adds_each_element_once(
    name: str,
) -> None:
    config, _launch = _LANE_TILES[name]
    kernel = helion.kernel(
        _arange_then_add, backend="cute", static_shapes=True, config=config
    )
    x = torch.randn(64, device=DEVICE)
    y = torch.randn(64, 128, device=DEVICE)
    out = torch.zeros(64, device=DEVICE)
    doubled = kernel(x, y, out)
    torch.testing.assert_close(out, x)
    torch.testing.assert_close(doubled, y * 2)
