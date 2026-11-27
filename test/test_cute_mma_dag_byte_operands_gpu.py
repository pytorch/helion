"""Native correctness for packed byte conversions and their staged reuse."""

from __future__ import annotations

import time
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


# The stress below queues its launches in batches until the clock or the
# batch count stops it.
_STRESS_BATCH = 128
_STRESS_SECONDS = 3.0
_STRESS_MAX_BATCHES = 64


@pytest.mark.slow
@skipIfNotCUDA()
@skipIfCudaCapabilityLessThan((8, 0))
def test_byte_operand_launches_queued_back_to_back() -> None:
    """The tcgen05 kernel of ``test_byte_operand_numerics`` whose bf16
    operand comes by TMA beside a byte operand the threads load themselves,
    launched back to back for a few seconds with other work queued between
    the launches, over four sets of inputs.

    That kernel once issued its first batch without waiting for the TMA
    load, and a launch then read the operand's tile before it had landed:
    whole output rows held one of the kernel's two accumulations or stale
    products. It took a device shared with other processes to see it (about
    one launch in 15,000 with 22 processes launching this kernel), which a
    lane's other workers provide and a run of this test alone mostly does
    not. The source test in ``test_cute_mma_dag_byte_operands.py`` pins the
    wait; this one runs the kernel where the race was."""
    require_byte_mma_mode("tcgen05")
    dtype, output_dtype, offset = torch.int8, torch.bfloat16, 1
    base = torch.arange(64 * 80).reshape(64, 80)
    cases = []
    for variant in range(4):
        values = base if variant == 0 else base * (6 * variant + 1) + 91 * variant
        storage = (values % 256).to(dtype).to(DEVICE)
        byte = storage[:, offset : offset + 64]
        other = torch.eye(64, dtype=output_dtype).roll(3 * variant, 1)
        halves = (byte.cpu().float() * 0.5).to(output_dtype).float()
        expected = (other.float() @ halves) * 2
        cases.append(
            ((other.to(DEVICE), byte, True, False, False), expected.to(DEVICE))
        )
    bound = helion.kernel(byte_matmul, backend="cute", static_shapes=True).bind(
        cases[0][0]
    )
    with bound.env:
        config = bound.config_spec.normalized_config(
            helion.Config(
                block_sizes=[64],
                cute_mma_dag="tcgen05",
                cute_mma_dag_warps=4,
                cute_mma_dag_stages=1,
            )
        )
    compiled = bound.compile_config(config)
    # Queued behind every launch, so the process's work on the device never
    # drains between two of them.
    filler = torch.zeros(16 << 20, dtype=torch.uint8, device=DEVICE)
    wrong: list[tuple[int, list[int]]] = []
    launches = 0
    deadline = time.monotonic() + _STRESS_SECONDS
    for batch in range(_STRESS_MAX_BATCHES):
        outputs = []
        for index in range(_STRESS_BATCH):
            outputs.append(compiled(*cases[(batch + index) % 4][0]))
            filler.add_(1)
        actual = torch.stack(outputs)
        expected = torch.stack(
            [cases[(batch + index) % 4][1] for index in range(_STRESS_BATCH)]
        )
        differs = (actual != expected).flatten(1).any(1)
        for index in differs.nonzero().flatten().tolist():
            rows = (actual[index] != expected[index]).any(1).nonzero().flatten()
            wrong.append((launches + index, rows.tolist()))
        launches += _STRESS_BATCH
        if time.monotonic() >= deadline:
            break
    assert not wrong, (
        f"{len(wrong)} of {launches} launches wrong; first (launch, rows): {wrong[:3]}"
    )


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
