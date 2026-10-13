from __future__ import annotations

import ast

import pytest
import torch

from ._cute_aux import _cpu_codegen
import helion
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends
import helion.language as hl


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _root_cached(a, b, c, d):
    m, k = a.shape
    n = c.size(-1)
    output = torch.empty((m, n), dtype=torch.float32, device=a.device)
    for rows, cols in hl.tile([m, n], block_size=[128, 32]):
        kk = hl.arange(k)
        pp = hl.arange(b.size(-1))
        left = hl.load(a, [rows, kk], extra_mask=kk[None, :] % 3 != 0)
        first = hl.dot(left, b[kk, pp], out_dtype=torch.float32)
        prefix = hl.cumsum(first, dim=1)
        repeated = (torch.sigmoid(prefix) * 0.125).to(a.dtype)
        second = hl.dot(repeated, c[pp, cols], out_dtype=torch.float32)
        output[rows, cols] = hl.dot(
            repeated, d[pp, cols], acc=second, out_dtype=torch.float32
        )
    return output


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _loop_cached(a, b, c, d, initial, steps: int):
    steps = hl.specialize(steps)
    _, m, k = a.shape
    n = c.size(-1)
    history = torch.empty((max(steps, 1), m, n), dtype=torch.float32, device=a.device)
    output = torch.empty_like(initial)
    for rows, cols in hl.tile([m, n], block_size=[16, 32]):
        state = initial[rows, cols]
        for step in hl.tile(steps, block_size=1):
            kk = hl.arange(k)
            pp = hl.arange(b.size(-1))
            left = hl.load(a, [step.id, rows, kk], extra_mask=kk[None, :] % 3 != 0)
            first = hl.dot(left, b[step.id, kk, pp], out_dtype=torch.float32)
            prefix = hl.cumsum(first, dim=1)
            repeated = (torch.sigmoid(prefix) * 0.125).to(a.dtype)
            second = hl.dot(repeated, c[step.id, pp, cols], out_dtype=torch.float32)
            state = hl.dot(
                repeated,
                d[step.id, pp, cols],
                acc=state * 0.5 + second,
                out_dtype=torch.float32,
            )
            history[step.id, rows, cols] = state
        output[rows, cols] = state
    return history, output


def _inputs(device, loop, dtype, steps=3):
    torch.manual_seed(929)
    prefix = (max(steps, 1),) if loop else ()
    rows = 13 if loop else 128
    tensors = tuple(
        torch.randn((*prefix, m, n), device=device, dtype=dtype) * 0.125
        for m, n in ((rows, 32), (32, 32), (32, 32), (32, 32))
    )
    return (
        (*tensors, torch.randn((rows, 32), device=device) * 0.125, steps)
        if loop
        else tensors
    )


def _config(schedule, cached, loop):
    return helion.Config(
        cute_chained_mma_schedule=schedule,
        cute_chained_scan_schedule="warp",
        cute_chained_pointwise_cache_bytes=(4096 if loop else 16384) if cached else 0,
        num_warps=16 if loop and schedule == "tcgen05_tmem" else 4,
    )


@pytest.mark.parametrize("loop", [False, True])
@pytest.mark.parametrize("schedule", ["coalesced", "tcgen05_tmem"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_residency_postdot_codegen_cpu(loop, schedule, dtype) -> None:
    with _cpu_codegen():
        kernel = _loop_cached if loop else _root_cached
        bound = kernel._bind_isolated(_inputs("cpu", loop, dtype))
        source = bound.to_code(_config(schedule, True, loop))
    tree = ast.parse(source)
    assert "chain_pointwise_cache_0" in source
    assert "shuffle_sync_up" in source
    stores = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Subscript)
            and ast.unparse(target.value) == "chain_pointwise_cache_0"
            for target in node.targets
        )
    ]
    assert len(stores) == 1
    assert source.index("chain_0_acc") < source.index("chain_pointwise_cache_0_step")


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("loop,steps", [(False, 1), (True, 0), (True, 1), (True, 3)])
@pytest.mark.parametrize("schedule", ["coalesced", "tcgen05_tmem"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_residency_postdot_gpu(loop, steps, schedule, dtype) -> None:
    args = _inputs(DEVICE, loop, dtype, steps)
    saved = tuple(arg.clone() for arg in args if isinstance(arg, torch.Tensor))
    kernel = _loop_cached if loop else _root_cached
    ordinary = kernel._bind_isolated(args).compile_config(
        _config(schedule, False, loop)
    )
    cached = kernel._bind_isolated(args).compile_config(_config(schedule, True, loop))
    expected, actual = ordinary(*args), cached(*args)
    if loop and steps == 0:
        expected, actual = expected[1], actual[1]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    repeated = cached(*args)
    if loop and steps == 0:
        repeated = repeated[1]
    torch.testing.assert_close(repeated, actual, rtol=0, atol=0)
    torch.testing.assert_close(
        tuple(arg for arg in args if isinstance(arg, torch.Tensor)),
        saved,
        rtol=0,
        atol=0,
    )
