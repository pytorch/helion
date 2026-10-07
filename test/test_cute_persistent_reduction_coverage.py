from __future__ import annotations

import ast
from unittest.mock import patch

import pytest
import torch

import helion
from helion._testing import skipUnlessBackends

pytestmark = skipUnlessBackends(["cute"])


@pytest.mark.parametrize("name", ["rms_norm", "layer_norm"])
@pytest.mark.parametrize("width", [1024, 2048, 4096])
def test_graph_reduction_covers_extent_beyond_hardware_thread_cap(
    name: str, width: int
) -> None:
    from examples.layer_norm import layer_norm_bwd
    from examples.rms_norm import rms_norm_bwd

    x = torch.empty(64, width, dtype=torch.float16)
    weight = torch.empty(width, dtype=torch.float16)
    if name == "rms_norm":
        example = rms_norm_bwd
        inputs = (torch.empty_like(x), x, weight, torch.empty(64, 1))
    else:
        example = layer_norm_bwd
        inputs = (
            torch.empty_like(x),
            x,
            torch.empty(64),
            torch.empty(64),
            weight,
            True,
        )
    bound = helion.kernel(
        example.fn,
        backend="cute",
        autotune_effort="none",
        ignore_warnings=[helion.exc.TensorOperationInWrapper],
    ).bind(inputs)
    # Both row axes are scalar. The persistent feature reduction can initially
    # claim the entire 1024-thread block, but wider rows still require lanes.
    config = helion.Config(block_sizes=[32, 1])
    with patch(
        "helion._compiler.reduction_strategy._cute_shared_memory_budget_bytes",
        return_value=128 * 1024,
    ):
        source = ast.parse(bound.to_code(config))
    lane_extents = {
        ast.literal_eval(node.iter.args[0])
        for node in ast.walk(source)
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id.startswith("synthetic_lane_")
        and isinstance(node.iter, ast.Call)
    }
    block = next(
        ast.literal_eval(keyword.value)
        for node in ast.walk(source)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_launcher"
        for keyword in node.keywords
        if keyword.arg == "block"
    )
    assert len(lane_extents) <= 1
    per_thread = next(iter(lane_extents), 1)
    assert block[0] * per_thread == width
    assert block[1:] == (1, 1)
