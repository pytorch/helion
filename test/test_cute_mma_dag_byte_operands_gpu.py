"""Native correctness for packed byte conversions and their staged reuse."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import skipIfCudaCapabilityLessThan
from helion._testing import skipIfNotCUDA
from helion._testing.cute_mma_dag_bytes import byte_loop_matmul
from helion._testing.cute_mma_dag_bytes import byte_matmul
from helion._testing.cute_mma_dag_bytes import byte_rhs_tiled_matmul
from helion._testing.cute_mma_dag_bytes import require_byte_mma_mode


@skipIfNotCUDA()
@skipIfCudaCapabilityLessThan((8, 0))
@pytest.mark.parametrize(
    "mode,dtype,byte_rhs,transpose_byte,offset,output_dtype,wrap_byte",
    [
        ("warp", torch.int8, False, False, 0, torch.bfloat16, False),
        ("tcgen05", torch.uint8, False, True, 0, torch.float16, False),
        ("warp", torch.int8, False, True, 1, torch.bfloat16, False),
        ("tcgen05", torch.int8, True, False, 1, torch.bfloat16, False),
        ("warp", torch.int8, False, False, 0, torch.bfloat16, True),
        ("tcgen05", torch.uint8, False, False, 1, torch.bfloat16, True),
    ],
)
def test_byte_operand_numerics(
    mode: str,
    dtype: torch.dtype,
    byte_rhs: bool,
    transpose_byte: bool,
    offset: int,
    output_dtype: torch.dtype,
    wrap_byte: bool,
) -> None:
    require_byte_mma_mode(mode)
    storage = (torch.arange(64 * 80).reshape(64, 80) % 256).to(dtype).to(DEVICE)
    byte = storage[:, offset : offset + 64]
    other = torch.eye(64, dtype=output_dtype, device=DEVICE)
    args = (other, byte) if byte_rhs else (byte, other)
    inputs = (*args, byte_rhs, transpose_byte, wrap_byte)
    bound = helion.kernel(byte_matmul, backend="cute", static_shapes=True).bind(inputs)
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
    for replay in range(2):
        if replay:
            changed = (torch.arange(64 * 80).reshape(64, 80) * 7 + 91) % 256
            storage.copy_(changed.to(dtype))
            other.copy_(torch.eye(64, dtype=output_dtype).roll(3, 1))
        actual = compiled(*inputs)
        values = byte.cpu()
        if wrap_byte:
            values = values + 1
        values = (values.float() * 0.5).to(output_dtype).float()
        if transpose_byte:
            values = values.T
        reference = other.cpu().float()
        expected = (reference @ values if byte_rhs else values @ reference) * 2
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@skipIfNotCUDA()
@skipIfCudaCapabilityLessThan((8, 0))
@pytest.mark.parametrize("mode,dtype", [("warp", torch.int8), ("tcgen05", torch.uint8)])
def test_byte_operand_loop_graph(mode: str, dtype: torch.dtype) -> None:
    require_byte_mma_mode(mode)
    # Each iteration has different bytes; stale raw/converted ring slots must
    # not happen to reproduce the reference. Replay changes every input byte.
    values = torch.arange(3 * 64 * 64).reshape(3, 64, 64) * 11 + 71
    values = values + torch.arange(3)[:, None, None] * 17
    a = (values % 256).to(dtype).to(DEVICE)
    b = torch.eye(64, dtype=torch.bfloat16, device=DEVICE)
    bound = helion.kernel(byte_loop_matmul, backend="cute", static_shapes=True).bind(
        (a, b)
    )
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
    actual = compiled(a, b)
    expected = (a.cpu().float() * 0.5).to(torch.bfloat16).float().sum(0)
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = compiled(a, b)
    a.copy_(((values * 3 + 29) % 256).to(dtype))
    graph.replay()
    expected = (a.cpu().float() * 0.5).to(torch.bfloat16).float().sum(0)
    torch.testing.assert_close(captured.cpu(), expected, rtol=0, atol=0)


@skipIfNotCUDA()
@skipIfCudaCapabilityLessThan((8, 0))
@pytest.mark.parametrize("offset", [0, 1])
def test_byte_operand_ragged_rows(offset: int) -> None:
    storage = (torch.arange(61 * 80).reshape(61, 80) % 256).to(torch.int8).to(DEVICE)
    a = storage[:, offset : offset + 64]
    b = torch.eye(64, dtype=torch.bfloat16, device=DEVICE)
    inputs = (a, b, False, False, False)
    bound = helion.kernel(byte_matmul, backend="cute", static_shapes=True).bind(inputs)
    with bound.env:
        config = bound.config_spec.normalized_config(
            helion.Config(
                block_sizes=[64],
                cute_mma_dag="warp",
                cute_mma_dag_warps=4,
                cute_mma_dag_stages=1,
            )
        )
    actual = bound.compile_config(config)(*inputs)
    torch.testing.assert_close(actual.cpu(), a.cpu().float(), rtol=0, atol=0)


@skipIfNotCUDA()
@skipIfCudaCapabilityLessThan((8, 0))
@pytest.mark.parametrize("columns,dtype", [(8, torch.int8), (32, torch.uint8)])
def test_short_byte_tile_numerics(columns: int, dtype: torch.dtype) -> None:
    pitch = max(16, columns)
    values = torch.arange(64 * pitch).reshape(64, pitch)
    storage = (values % 256).to(dtype).to(DEVICE)
    byte = storage[:, :columns]
    other = torch.eye(64, dtype=torch.bfloat16, device=DEVICE)
    # Exercise the warp backend's legal N=8 atom while the common dot
    # frontend still imposes the cross-backend N>=16 policy.
    with patch("helion._compat._min_dot_size", return_value=(16, 8, 16)):
        bound = helion.kernel(
            byte_rhs_tiled_matmul, backend="cute", static_shapes=True
        ).bind((other, byte))
        compiled = bound.compile_config(
            helion.Config(
                block_sizes=[64, 8],
                cute_mma_dag="warp",
                cute_mma_dag_warps=4,
                cute_mma_dag_stages=1,
            )
        )
    for replay in range(2):
        if replay:
            storage.copy_(((values * 7 + 91) % 256).to(dtype))
            other.copy_(torch.eye(64, dtype=torch.bfloat16).roll(3, 1))
        actual = compiled(other, byte)
        expected = other.cpu().float() @ byte.cpu().float()
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
