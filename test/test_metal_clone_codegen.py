"""Metal lane-store allocation proofs, rendered on CPU without a Metal device."""

from __future__ import annotations

import ast

import pytest
import torch

import helion
from helion._testing import skipUnlessBackends
from helion._testing.cute_codegen import forbid_cuda_work
import helion.language as hl

pytestmark = [pytest.mark.cpu_only, skipUnlessBackends(["metal"])]


@helion.kernel(backend="metal", static_shapes=True)
def _lane_output(x: torch.Tensor, allocation: hl.constexpr) -> torch.Tensor:
    if allocation == "method":
        out = x.clone()
    elif allocation == "function":
        out = torch.clone(x)
    elif allocation == "empty":
        out = torch.empty_like(x)
    else:
        out = x.view_as(x)
    for rows in hl.tile(x.size(0)):
        for cols in hl.tile(x.size(1)):
            value = x[rows, cols]
            out[rows, cols] = value - value.sum(dim=1)[:, None]
    return out


@pytest.mark.parametrize("allocation", ["method", "function", "empty"])
@forbid_cuda_work()
def test_metal_lane_stores_accept_proved_clone_allocation(allocation):
    value = torch.arange(256, dtype=torch.float32).reshape(16, 16)
    bound = _lane_output.bind((value, allocation))
    code = bound.to_code(helion.Config(block_sizes=[16, 16], num_threads=[16, 1]))
    lane_loops = [
        node
        for node in ast.walk(ast.parse(code))
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id.startswith("lane_")
    ]
    assert len(lane_loops) == 2
    out = bound.host_function.local_types["out"].fake_value
    assert bound.env.tensor_storage_is_compiler_allocated(out)
    assert not bound.env.tensor_storage_is_zero_filled_allocation(out)
    if allocation != "empty":
        # clone is fresh even when its memory-format/layout proof is unknown.
        assert (
            out.untyped_storage() not in bound.env._symbolically_exact_layout_storages
        )


@forbid_cuda_work()
def test_metal_lane_stores_still_reject_aliasing_view():
    value = torch.arange(256, dtype=torch.float32).reshape(16, 16)
    with pytest.raises(
        helion.exc.BackendUnsupported, match="may share memory|stored and loaded"
    ):
        _lane_output.bind((value, "view")).to_code(
            helion.Config(block_sizes=[16, 16], num_threads=[16, 1])
        )
