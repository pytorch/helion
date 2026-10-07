"""Portable indexing tests."""

from __future__ import annotations

import torch

import helion
from helion._testing import DEVICE
from helion._testing import RefEagerTestBase
from helion._testing import TestCase
from helion._testing import output_only
import helion.language as hl


class TestIndexing(RefEagerTestBase, TestCase):
    def test_arange(self) -> None:
        @helion.kernel
        def arange(length: int, device: torch.device) -> torch.Tensor:
            out = torch.empty([length], dtype=torch.int32, device=device)
            for tile in hl.tile(length):
                out[tile] = tile.index
            return out

        result = output_only(arange, (100, DEVICE), block_size=32)
        torch.testing.assert_close(
            result, torch.arange(0, 100, device=DEVICE, dtype=torch.int32)
        )
