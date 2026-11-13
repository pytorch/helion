"""Native ring wrap and changed-bound replay with wide memory indices."""

from __future__ import annotations

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import skipIfCudaCapabilityLessThan
from helion._testing import skipIfNotCUDA
from helion._testing.cute_mma_dag_ring import ring_matmul


@skipIfNotCUDA()
@skipIfCudaCapabilityLessThan((8, 0))
@pytest.mark.parametrize(
    ("mode", "loads", "stages", "dtype", "pass_warps"),
    [
        ("warp", "lane", 3, torch.float32, False),
        ("warp", "lane", 4, torch.float32, False),
        ("warp", "tma", 3, torch.float32, False),
        ("tcgen05", "tma", 3, torch.bfloat16, False),
        ("tcgen05", "tma", 3, torch.float32, True),
    ],
)
def test_ring_wrap_and_wide_bound_graph_replay(
    mode: str, loads: str, stages: int, dtype: torch.dtype, pass_warps: bool
) -> None:
    major = torch.cuda.get_device_capability()[0]
    if mode == "tcgen05" and major != 10:
        pytest.skip("tcgen05 MMA-DAG currently admits only major architecture 10")
    if loads == "tma" and major < 9:
        pytest.skip("TMA requires compute capability 9.0 or newer")
    # Nineteen is deliberately not a power of two: narrowing a large index
    # before modulo changes the referenced tile. Each tile also has distinct
    # data, so an incorrect ring wrap cannot reuse an equivalent operand.
    values = (torch.arange(19 * 64 * 64).reshape(19, 64, 64) % 31 - 15) / 16
    weights = ((torch.arange(19) % 5 + 1) / 8)[:, None, None]
    a = values.to(dtype=dtype, device=DEVICE)
    b = (weights * torch.eye(64)[None, :, :]).to(dtype=torch.bfloat16, device=DEVICE)
    bounds = torch.tensor([0, 26], dtype=torch.int64, device=DEVICE)
    args = (a, b, bounds)
    bound = helion.kernel(ring_matmul, backend="cute", static_shapes=True).bind(args)
    with bound.env:
        config = bound.config_spec.normalized_config(
            helion.Config(
                block_sizes=[64],
                cute_mma_dag=mode,
                cute_mma_dag_loads=loads,
                cute_mma_dag_stages=stages,
                cute_mma_dag_warps=8 if pass_warps else 4,
                cute_mma_dag_pass_warps="on" if pass_warps else "off",
            )
        )
    compiled = bound.compile_config(config)
    compiled(*args)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = compiled(*args)
    for replay, (start, stop) in enumerate(
        [
            (0, 26),
            (2**31 + 7, 2**31 + 24),
            (-(2**31) - 31, -(2**31) - 12),
            (2**31 + 7, 2**31 + 7),
            (2**31 + 9, 2**31 + 7),
        ]
    ):
        changed = (values + replay / 16).to(dtype).to(torch.bfloat16).float()
        a.copy_((values + replay / 16).to(dtype))
        bounds.copy_(torch.tensor([start, stop], dtype=torch.int64))
        indices = torch.tensor([i % 19 for i in range(start, stop)], dtype=torch.int64)
        expected = (changed * weights)[indices].sum(0)
        actual = compiled(*args)
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
        graph.replay()
        torch.testing.assert_close(captured.cpu(), expected, rtol=0, atol=0)
