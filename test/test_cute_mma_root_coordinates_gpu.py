from __future__ import annotations

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import skipIfCudaCapabilityLessThan
from helion._testing import skipIfNotCUDA
from helion._testing.cute_mma_root_coordinates import root_coordinate_matmul


@skipIfNotCUDA()
@skipIfCudaCapabilityLessThan((8, 0))
@pytest.mark.parametrize("mode", ["warp", "tcgen05"])
@pytest.mark.parametrize("kind", ["begin", "id", "mixed", "captured"])
def test_root_coordinates_and_changed_data_replay(mode: str, kind: str) -> None:
    if mode == "tcgen05" and torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("tcgen05 currently requires major architecture 10")
    values = (torch.arange(128 * 32).reshape(128, 32) % 13 - 6) / 8
    a = values.to(device=DEVICE, dtype=torch.bfloat16)
    b = torch.cat((torch.eye(32), -torch.eye(32)), dim=1).to(
        device=DEVICE, dtype=torch.bfloat16
    )
    args = (a, b, kind)
    bound = helion.kernel(
        root_coordinate_matmul, backend="cute", static_shapes=True
    ).bind(args)
    with bound.env:
        config = bound.config_spec.normalized_config(
            helion.Config(
                block_sizes=[64],
                cute_mma_dag=mode,
                cute_mma_dag_warps=4,
                cute_mma_dag_stages=2,
            )
        )
    compiled = bound.compile_config(config)
    compiled(*args)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = compiled(*args)
    row_begin = (torch.arange(128) // 64 * 64).float()[:, None]
    tile_id = (torch.arange(128) // 64).float()[:, None]
    for delta in (0.0, 0.125):
        changed = values + delta
        a.copy_(changed)
        dot = torch.cat((changed, -changed), dim=1)
        if kind == "mixed":
            expected = dot * (2 * row_begin + 3)
        elif kind == "captured":
            expected = dot * (row_begin + 1)
        else:
            expected = 2 * dot * (row_begin if kind == "begin" else tile_id)
        actual = compiled(*args)
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
        graph.replay()
        torch.testing.assert_close(captured.cpu(), expected, rtol=0, atol=0)
