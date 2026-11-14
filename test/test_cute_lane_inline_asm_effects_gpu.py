"""Actual opaque writes must not change the producer's earlier input values."""

from __future__ import annotations

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import skipIfNotCUDA
from helion._testing import skipIfRocm
from helion._testing.cute_lane_inline_asm import snapshot_before_opaque_producer


@skipIfNotCUDA()
@skipIfRocm("inline PTX requires an NVIDIA target")
@pytest.mark.parametrize("shared", [False, True])
def test_lane_producer_keeps_input_snapshot_and_exact_effect_count(
    shared: bool,
) -> None:
    x = torch.empty((2, 256), device=DEVICE)
    counts = torch.zeros_like(x, dtype=torch.int32)
    index = torch.arange(x.numel(), device=DEVICE).view_as(x)
    # Each producer overwrites a future lane owned by another physical warp
    # in its CTA. The complete prefix must be retained before any write.
    destinations = (
        x.data_ptr() + ((index // 256) * 256 + (index % 256 + 33) % 256) * 4
    ).to(torch.uint64)
    counter_pointers = (counts.data_ptr() + index * 4).to(torch.uint64)
    bound = snapshot_before_opaque_producer._bind_isolated(
        (x, destinations, counter_pointers, shared)
    )
    config = helion.Config(
        block_sizes=[1], num_threads=[0, 64], cute_vector_widths=[1, 1]
    )
    compiled = bound.compile_config(config)
    for offset in (1, 20):
        original = index.to(torch.float32) + offset
        x.copy_(original)
        counts.zero_()
        actual = compiled(x, destinations, counter_pointers, shared)
        expected = (
            original - original.sum(-1, keepdim=True) if shared else original.sum(-1)
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert torch.equal(counts, torch.ones_like(counts))
        assert torch.equal(x, torch.zeros_like(x))
