"""Native typed grouped scale loads, including iteration offsets and replay."""

from __future__ import annotations

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import skipIfCudaCapabilityLessThan
from helion._testing import skipIfNotCUDA
from helion._testing.cute_mma_dag_bytes import require_byte_mma_mode
from helion._testing.cute_mma_dag_grouped import grouped_loop_matmul
from helion._testing.cute_mma_dag_grouped import grouped_matmul


@skipIfNotCUDA()
@skipIfCudaCapabilityLessThan((8, 0))
@pytest.mark.parametrize(
    "mode,dtype,row_group,group,wrap",
    [
        ("warp", torch.uint8, False, 32, False),
        ("tcgen05", torch.uint8, True, 3, False),
        ("warp", torch.int8, True, 5, True),
        ("tcgen05", torch.uint8, False, 3, True),
        ("warp", torch.float32, False, 7, False),
        ("tcgen05", torch.bfloat16, True, 17, False),
    ],
)
def test_grouped_scale_numerics(
    mode: str, dtype: torch.dtype, row_group: bool, group: int, wrap: bool
) -> None:
    require_byte_mma_mode(mode)
    m, k = 128, 64
    raw = torch.arange(m * k).reshape(m, k) * 7 + 53
    a = (raw % 256).to(torch.int8).to(DEVICE)
    b = torch.eye(k, dtype=torch.bfloat16, device=DEVICE)
    shape = (
        ((m + group - 1) // group, k) if row_group else (m, (k + group - 1) // group)
    )
    scale_values = torch.arange(shape[0] * shape[1]).reshape(shape)
    scale_values = (
        scale_values % 8 + 120 if dtype == torch.int8 else scale_values % 4 + 125
    )
    if wrap and dtype == torch.uint8:
        scale_values[::3] = 255
    scales = scale_values.to(dtype).to(DEVICE)
    inputs = (a, b, scales, group, row_group, wrap)
    bound = helion.kernel(grouped_matmul, backend="cute", static_shapes=True).bind(
        inputs
    )
    with bound.env:
        config = bound.config_spec.normalized_config(
            helion.Config(
                block_sizes=[64],
                cute_mma_dag=mode,
                cute_mma_dag_warps=4,
                cute_mma_dag_stages=1,
            )
        )
    compiled = bound.compile_config(config)
    compiled(*inputs)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = compiled(*inputs)
    for replay in range(2):
        if replay:
            a.copy_(((raw * 3 + 11) % 256).to(torch.int8))
            scales.copy_(scale_values.flip(0).to(dtype))
            b.copy_(torch.eye(k, dtype=torch.bfloat16).roll(3, 1))
        graph.replay()
        scale = scales.cpu()
        scale = scale + 1 if wrap else scale
        expanded = (
            scale[torch.arange(m) // group, :]
            if row_group
            else scale[:, torch.arange(k) // group]
        )
        value = (
            (a.cpu().float() * torch.exp2(expanded.float() - 127))
            .to(torch.bfloat16)
            .float()
        )
        expected = (value @ b.cpu().float()) * 2
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@skipIfNotCUDA()
@skipIfCudaCapabilityLessThan((8, 0))
@pytest.mark.parametrize("mode", ["warp", "tcgen05"])
def test_grouped_scale_crosses_iteration_boundary(mode: str) -> None:
    require_byte_mma_mode(mode)
    raw = torch.arange(64 * 128).reshape(64, 128) * 11 + 17
    a = (raw % 256).to(torch.int8).to(DEVICE)
    b = (
        torch.cat((torch.eye(64), torch.eye(64).roll(7, 1)))
        .to(torch.bfloat16)
        .to(DEVICE)
    )
    scale_values = (torch.arange(64 * 43).reshape(64, 43) % 4 + 125).to(torch.uint8)
    scales = scale_values.to(DEVICE)
    inputs = (a, b, scales, 3)
    bound = helion.kernel(grouped_loop_matmul, backend="cute", static_shapes=True).bind(
        inputs
    )
    with bound.env:
        config = bound.config_spec.normalized_config(
            helion.Config(
                block_sizes=[64, 64],
                cute_mma_dag=mode,
                cute_mma_dag_warps=4,
                cute_mma_dag_stages=2,
            )
        )
    compiled = bound.compile_config(config)
    compiled(*inputs)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = compiled(*inputs)
    for replay in range(2):
        if replay:
            a.copy_(((raw * 3 + 23) % 256).to(torch.int8))
            scales.copy_(scale_values.flip(1))
        graph.replay()
        scale = scales.cpu()[:, torch.arange(128) // 3]
        value = (
            (a.cpu().float() * torch.exp2(scale.float() - 127))
            .to(torch.bfloat16)
            .float()
        )
        expected = value @ b.cpu().float()
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
