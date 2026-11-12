"""CPU codegen tests: the loads-first split of a grid lane loop is bounded.

``cute/split_lane_loads.py`` rewrites a vectorized grid lane loop, which walks
``U = block_size / (num_threads * V)`` packets per thread, into a load phase
and a compute phase that are both unrolled at trace time, so every packet load
issues before the first store.  The trace grows with the ``U * V`` elements a
thread unrolls and every packet stays live until its lane computes, so the
split takes at most ``_MAX_UNROLLED_LANES`` packets (the bound of the
``cute_lane_unroll`` unroll) and a bounded number of elements per thread
(``_MAX_SPLIT_ELEMENTS``).  A longer loop stays rolled: a 2**21 block over two
threads gave ``range_constexpr(131072)``, which never finished compiling.
"""

from __future__ import annotations

import functools
import re
from typing import TYPE_CHECKING

import pytest
import torch

import helion
from helion._compiler.cute.split_lane_loads import _MAX_SPLIT_ELEMENTS
from helion._compiler.cute.unroll_lane_loads import _MAX_UNROLLED_LANES
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import mock_cuda_unavailable
import helion.language as hl

if TYPE_CHECKING:
    from helion.runtime.kernel import BoundKernel

pytestmark = pytest.mark.cpu_only


def _scale(x: torch.Tensor) -> torch.Tensor:
    (n,) = x.shape
    out = torch.empty_like(x)
    for tile in hl.tile(n):
        out[tile] = (x[tile].to(torch.float32) * 2.0).to(out.dtype)
    return out


@functools.cache
def _bound() -> BoundKernel:
    kernel = helion.kernel(
        _scale, backend="cute", static_shapes=True, autotune_effort="none"
    )
    with mock_cuda_unavailable():
        return cpu_bind(kernel, (torch.empty(1 << 21, dtype=torch.bfloat16),))


@pytest.mark.parametrize(
    ("packets", "num_threads", "width", "split"),
    [
        (2, 128, 8, True),
        # 64 packets of 8 elements: at both bounds.
        (64, 128, 8, True),
        (32, 128, 16, True),
        # 128 packets of 2 elements: past the packet bound only.
        (128, 128, 2, False),
        # 64 packets of 16 elements: past the element bound only.
        (64, 128, 16, False),
        # The whole 2**21-element input in one block of two threads.
        (131072, 2, 8, False),
    ],
)
def test_lane_loop_splits_within_the_unroll_bounds(
    packets: int, num_threads: int, width: int, split: bool
) -> None:
    # The cases straddle the two bounds.
    assert split == (
        packets <= _MAX_UNROLLED_LANES and packets * width <= _MAX_SPLIT_ELEMENTS
    )
    config = helion.Config(
        block_sizes=[packets * num_threads * width],
        num_threads=[num_threads],
        cute_vector_widths=[width],
    )
    with mock_cuda_unavailable():
        code = _bound().to_code(config)
    lane_loops = re.findall(r"for lane_0 in (.+):", code)
    if split:
        assert lane_loops == [f"cutlass.range_constexpr({packets})"] * 2
        assert "_lane_loads_" in code
    else:
        assert lane_loops == [f"range({packets})"]
        assert "_lane_loads_" not in code
