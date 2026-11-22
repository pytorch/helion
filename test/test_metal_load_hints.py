"""Triton load hints in Metal kernels, rendered on CPU without a Metal device."""

from __future__ import annotations

import pytest
import torch

import helion
from helion._testing import skipUnlessBackends
from helion._testing.metal import render_msl
import helion.language as hl

pytestmark = [pytest.mark.cpu_only, skipUnlessBackends(["metal"])]


@helion.kernel(backend="metal", static_shapes=True)
def _cyclic_gather(table: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    t = table.size(0)
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] + table[tile.index % t]
    return out


def test_contiguity_hint_dropped_from_msl():
    """The shared indexing code marks a gather whose index repeats in runs of
    four (``tile.index % 4``) ``tl.max_contiguous`` for Triton; Metal reads
    the offset under it."""
    table = torch.arange(4, dtype=torch.int32)
    x = torch.zeros(64, dtype=torch.int32)
    config = helion.Config(block_sizes=[16])
    code = _cyclic_gather.bind((table, x)).to_code(config)
    assert "tl.max_contiguous" in code
    msl = render_msl(_cyclic_gather, (table, x), config)
    assert "max_contiguous" not in msl
    assert "table[" in msl
