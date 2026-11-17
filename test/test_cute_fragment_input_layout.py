"""The reduction inner function retains its pre-mask input layout."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

import helion
from helion._compiler.cute.computed_fragment import FragmentOps
from helion._testing import cpu_render_target
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_fragment_layout import expanded_mask_counts
from helion.language._tracing_ops import _mask_to

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize("shape", [(8, 16), (11, 19)])
def test_masked_expansion_uses_original_input_strides(shape):
    # The second shape has a partial root tile and a non-power-of-two column
    # extent, so both the write mask and the expanded input's domain remain live.
    observations = []
    original = FragmentOps.direct_coordinates

    def inspect(ops, index, strides, value_shape):
        masked = any(
            source.target is _mask_to for source, _value in ops.inputs.values()
        )
        observations.append((masked, strides, value_shape))
        return original(ops, index, strides, value_shape)

    values = (torch.randn(shape), torch.arange(shape[1]) % 3 != 0)
    with cpu_render_target(), patch.object(FragmentOps, "direct_coordinates", inspect):
        bound = cpu_bind(expanded_mask_counts, values)
        code = bound.to_code(helion.Config(block_sizes=[8]))
    masked_expansions = [
        strides
        for masked, strides, extents in observations
        if masked and extents[-1] == 2 and extents[0] >= shape[1]
    ]
    assert masked_expansions
    assert all(strides == (1, 0) for strides in masked_expansions)
    assert "fragment_sum" in code
    assert "fragment_reduce_index" in code
    assert "fragment_predicate" in code
