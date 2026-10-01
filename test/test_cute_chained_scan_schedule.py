from __future__ import annotations

import ast

import pytest
import torch

from ._cute_aux import _cpu_codegen
from .test_cute_chained_loop_collectives import _postdot_chain
from .test_cute_chained_loop_collectives import _postdot_inputs
from .test_cute_chained_loop_collectives import _postdot_recurrence
from .test_cute_chained_loop_collectives import _postdot_source
from .test_cute_chained_loop_collectives import _prefix_coefficient_recurrence
from .test_cute_chained_loop_collectives import _prefix_reference
import helion
from helion._compiler.cute.chained_collectives import _warp_prefix
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends


def _config(schedule: str, warps: int = 4) -> helion.Config:
    return helion.Config(
        cute_chained_mma_schedule=schedule,
        cute_chained_scan_schedule="warp",
        num_warps=warps,
    )


@pytest.mark.parametrize("loop", [False, True])
@pytest.mark.parametrize(("rank", "axis"), [(1, 0), (2, 0), (2, 1)])
@pytest.mark.parametrize("schedule", ["coalesced", "tcgen05_tmem"])
def test_parallel_postdot_scan_codegen_cpu(loop, rank, axis, schedule) -> None:
    with _cpu_codegen():
        kernel = _postdot_recurrence if loop else _postdot_chain
        bound = kernel._bind_isolated(
            _postdot_inputs("cpu", loop=loop, rank=rank, axis=axis)
        )
        source = bound.to_code(_config(schedule))
        _postdot_source(source, loop=loop)
        assert "shuffle_sync_up" in source
        if not loop and axis == 0:
            assert "chain_collective_0_carry" in source


@pytest.mark.parametrize("extent", [1, 13, 16, 31, 32, 33, 48, 64, 129])
@pytest.mark.parametrize("vectors", [1, 3, 16, 33])
@pytest.mark.parametrize("threads", [32, 128, 512, 1024])
def test_parallel_scan_ownership_cpu(extent, vectors, threads) -> None:
    """Every coordinate is written once; invalid lanes still join shuffles."""
    warps = threads // 32
    owners = [
        (vector, part * 32 + lane)
        for step in range((vectors + warps - 1) // warps)
        for warp in range(warps)
        if (vector := warp + step * warps) < vectors
        for part in range((extent + 31) // 32)
        for lane in range(32)
        if part * 32 + lane < extent
    ]
    assert len(owners) == extent * vectors == len(set(owners))
    body = _warp_prefix(
        "prefix",
        extent,
        vectors,
        threads,
        ("prefix_vector", "prefix_position"),
        ["source = cutlass.Float32(1)"],
        "source",
    )
    tree = ast.parse("\n".join(body))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    assert (
        sum(ast.unparse(node.func) == "cute.arch.shuffle_sync_up" for node in calls)
        == 5
    )
    assert not any(ast.unparse(node.func) == "cute.arch.sync_threads" for node in calls)
    assert ("prefix_carry" in "\n".join(body)) == (extent > 32)


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("loop", [False, True])
@pytest.mark.parametrize(("rank", "axis"), [(1, 0), (2, 0), (2, 1)])
@pytest.mark.parametrize("schedule", ["coalesced", "tcgen05_tmem"])
def test_parallel_postdot_scan_gpu(loop, rank, axis, schedule) -> None:
    args = _postdot_inputs(DEVICE, loop=loop, rank=rank, axis=axis)
    original = [arg.clone() for arg in args if isinstance(arg, torch.Tensor)]
    kernel = _postdot_recurrence if loop else _postdot_chain
    compiled = kernel._bind_isolated(args).compile_config(_config(schedule))
    actual = compiled(*args)
    a, b, c = args[:3]
    if loop:
        history, final = actual
        expected = c.clone()
        for step in range(3):
            product = a[:, step].float() @ b[:, step].float()
            expected = 0.5 * expected + _prefix_reference(product, rank, axis)
            torch.testing.assert_close(history[:, step], expected, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(final, expected, rtol=1e-4, atol=1e-5)
    else:
        prefix = _prefix_reference(a.float() @ b.float(), rank, axis).to(c.dtype)
        expected = _prefix_reference(prefix.float() @ c.float(), rank, axis)
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
    repeated = compiled(*args)
    torch.testing.assert_close(repeated, actual, rtol=0, atol=0)
    for arg, saved in zip(args[:3], original, strict=True):
        torch.testing.assert_close(arg, saved, rtol=0, atol=0)


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize(
    ("axis", "steps", "warps"), [(0, 0, 4), (0, 3, 16), (1, 1, 4), (1, 3, 32)]
)
def test_parallel_prefix_partial_tiles_gpu(axis, steps, warps) -> None:
    torch.manual_seed(924)
    shapes = (
        (2, max(steps, 1), 13, 48),
        (2, max(steps, 1), 48, 32),
        (2, max(steps, 1), 13),
        (2, max(steps, 1), 13, 48),
        (2, 13, 32),
    )
    tensors = tuple(
        torch.randn(
            shape, device=DEVICE, dtype=torch.bfloat16 if index < 2 else torch.float32
        )
        * (0.125 if index < 2 or index == 4 else 0.015625)
        for index, shape in enumerate(shapes)
    )
    args = (*tensors, steps, axis)
    saved = tuple(tensor.clone() for tensor in tensors)
    compiled = _prefix_coefficient_recurrence._bind_isolated(args).compile_config(
        _config("tcgen05_tmem", warps)
    )
    history, final = compiled(*args)
    a, b, decay, delta, initial = tensors
    expected = initial.clone()
    for step in range(steps):
        row_prefix = decay[:, step].cumsum(1)
        matrix_prefix = delta[:, step].cumsum(axis + 1)
        energy = (matrix_prefix * matrix_prefix).sum(2)
        left = (a[:, step].float() * torch.exp(matrix_prefix)).to(a.dtype)
        expected = expected * torch.exp(row_prefix[:, :, None]) + (
            left.float() @ b[:, step].float()
        ) / (1.0 + energy[:, :, None])
        torch.testing.assert_close(history[:, step], expected, rtol=2e-4, atol=2e-5)
    torch.testing.assert_close(final, expected, rtol=2e-4, atol=2e-5)
    # The uninitialized history slot in a zero-trip loop is not an output contract.
    _, repeated = compiled(*args)
    torch.testing.assert_close(repeated, final, rtol=0, atol=0)
    torch.testing.assert_close(tensors, saved, rtol=0, atol=0)
