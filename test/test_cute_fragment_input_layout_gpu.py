"""Native expansion, masking and changed-input layout checks."""

from __future__ import annotations

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import skipIfNotCUDA
from helion._testing.cute_fragment_layout import expanded_mask_counts


@skipIfNotCUDA()
@pytest.mark.parametrize("shape", [(8, 16), (11, 19)])
def test_expanded_mask_counts_replay(shape):
    value = torch.arange(shape[0] * shape[1], device=DEVICE, dtype=torch.float32)
    value = value.reshape(shape) * 0.125 - 3
    mask = torch.arange(shape[1], device=DEVICE) % 3 != 0
    bound = expanded_mask_counts.bind((value, mask))
    config = helion.Config(block_sizes=[8])
    code = bound.to_code(config)
    assert "fragment_sum" in code
    assert "fragment_reduce_index" in code
    assert "fragment_predicate" in code
    compiled = bound.compile_config(config)
    compiled(value, mask)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual, counts = compiled(value, mask)
    for replay in range(2):
        graph.replay()
        expected_counts = (mask.long() * 2).expand_as(value)
        torch.testing.assert_close(counts, expected_counts, rtol=0, atol=0)
        torch.testing.assert_close(actual, torch.where(mask, value, 0), rtol=0, atol=0)
        if replay == 0:
            mask.logical_not_()
            value.neg_().add_(0.75)
