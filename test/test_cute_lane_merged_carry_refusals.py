"""Lane-merged loop carries the CuTe backend refuses, checked at codegen.

``acc = hl.zeros([tile_m, bn])`` updated in ``for tile_n in hl.tile(n,
block_size=bn)`` is one scalar per thread on CuTe, which folds the thread's
lanes of ``bn`` (see test_cute_lane_merged_carries.py).  A sum over ``bn``
after the loop that adds the carry's elements in another dtype than the
carry's would see the folded lanes, so the backend refuses it.
"""

from __future__ import annotations

import pytest
import torch

import helion
from helion import exc
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import mock_cuda_unavailable
import helion.language as hl

pytestmark = pytest.mark.cpu_only


def _narrowed_sum(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    bn = hl.register_block_size(n)
    out = torch.zeros([m], dtype=torch.float16, device=x.device)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, bn], dtype=x.dtype)
        for tile_n in hl.tile(n, block_size=bn):
            acc = acc + x[tile_m, tile_n]
        out[tile_m] = acc.sum(-1, dtype=torch.float16)
    return out


def _code(fn: object, args: tuple[torch.Tensor, ...], threads: int) -> str:
    kernel = helion.kernel(fn, backend="cute", static_shapes=True)
    config = helion.Config(block_sizes=[16, 1], num_threads=[threads, 1])
    with mock_cuda_unavailable():
        return cpu_bind(kernel, args).to_code(config)


def test_a_sum_that_narrows_the_carry_is_refused() -> None:
    """``acc.sum(-1, dtype=torch.float16)`` of a float32 carry casts each
    thread's folded total of four columns, not each column: totals of
    +-20,000 per column fit float16 and cancel in torch, but +-80,000 per
    thread overflow to +-inf.  With a thread per column nothing folds."""
    args = (torch.empty(64, 512),)
    with pytest.raises(exc.BackendUnsupported, match="own loop's block"):
        _code(_narrowed_sum, args, 4)
    assert "cutlass.Float16" in _code(_narrowed_sum, args, 16)
