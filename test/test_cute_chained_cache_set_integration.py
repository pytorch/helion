from __future__ import annotations

import pytest
import torch

from ._cute_aux import _cpu_codegen
from .test_cute_chained_preparation_cut import _kda_fixture
from .test_cute_chained_preparation_cut import _runtime_values
from .test_cute_chained_preparation_prefill import _run_preparation_prefill
import helion
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends
import helion.language as hl


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _root_cache_set(a, b, c, d, e):
    m, k = a.shape
    n = c.size(-1)
    output = torch.empty((m, n), device=a.device, dtype=torch.float32)
    for rows, cols in hl.tile([m, n], block_size=[128, 32]):
        kk, pp = hl.arange(k), hl.arange(b.size(-1))
        first = hl.dot(a[rows, kk], b[kk, pp])
        other = hl.dot(e[rows, kk], b[kk, pp])
        left = (torch.sigmoid(first) * 0.125).to(a.dtype)
        right = (torch.tanh(other) * 0.125).to(a.dtype)
        one = hl.dot(left, c[pp, cols])
        two = hl.dot(right, c[pp, cols], acc=one)
        output[rows, cols] = hl.dot(left + right, d[pp, cols], acc=two)
    return output


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _loop_cache_set(a, b, c, d, e, initial, steps: int):
    steps = hl.specialize(steps)
    _, m, k = a.shape
    n = c.size(-1)
    history = torch.empty((steps, m, n), device=a.device, dtype=torch.float32)
    output = torch.empty_like(initial)
    for rows, cols in hl.tile([m, n], block_size=[16, 32]):
        state = initial[rows, cols]
        for step in hl.tile(steps, block_size=1):
            kk, pp = hl.arange(k), hl.arange(b.size(-1))
            first = hl.dot(a[step.id, rows, kk], b[step.id, kk, pp])
            other = hl.dot(e[step.id, rows, kk], b[step.id, kk, pp])
            left = (torch.sigmoid(first) * 0.125).to(a.dtype)
            right = (torch.tanh(other) * 0.125).to(a.dtype)
            one = hl.dot(left, c[step.id, pp, cols])
            two = hl.dot(right, c[step.id, pp, cols])
            state = hl.dot(
                left + right, d[step.id, pp, cols], acc=state * 0.5 + one + two
            )
            history[step.id, rows, cols] = state
        output[rows, cols] = state
    return history, output


def _args(device, loop, dtype, steps):
    torch.manual_seed(9861)
    prefix = (max(steps, 1),) if loop else ()
    rows = 13 if loop else 128
    tensors = tuple(
        torch.randn((*prefix, m, n), device=device, dtype=dtype) * 0.125
        for m, n in ((rows, 32), (32, 32), (32, 32), (32, 32), (rows, 32))
    )
    return (
        (*tensors, torch.randn((rows, 32), device=device) * 0.125, steps)
        if loop
        else tensors
    )


def _config(loop, count):
    return helion.Config(
        cute_chained_mma_schedule="tcgen05_tmem",
        cute_chained_pointwise_cache_bytes=4096 if loop else 16384,
        cute_chained_pointwise_cache_entries=count,
        cute_chained_warp_mma_rows=32 if loop else 0,
        cute_chained_preparation_pipeline=loop,
        num_warps=16 if loop else 4,
    )


@pytest.mark.parametrize("loop", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_independent_cache_set_root_and_loop_codegen_cpu(loop, dtype):
    kernel = _loop_cache_set if loop else _root_cache_set
    args = _args("cpu", loop, dtype, 3)
    with _cpu_codegen():
        bound = kernel._bind_isolated(args)
        with bound.env.use_runtime_arg_values(_runtime_values(kernel, args)):
            source = bound.to_code(_config(loop, 2))
    assert "chain_pointwise_cache_0" in source
    assert "chain_pointwise_cache_1" in source
    assert "chain_pointwise_cache_2" not in source
    assert ("chain_slot_bars" in source) == loop


@pytest.mark.parametrize("value_tile", [64, 128])
@pytest.mark.parametrize("rectangular_leaf", [False, True])
def test_cache_set_prefill_groups_and_optional_tma_codegen_cpu(
    value_tile, rectangular_leaf
):
    kernel, args = _kda_fixture()
    config = helion.Config(
        block_sizes=[value_tile],
        num_warps=16,
        cute_chained_mma_schedule="tcgen05_tmem",
        cute_chained_group_contractions=True,
        cute_chained_warp_mma_rows=32,
        cute_chained_pointwise_cache_bytes=4096,
        cute_chained_pointwise_cache_entries=4,
        cute_chained_scan_schedule="warp",
        cute_chained_scratch_layout="xor",
        cute_chained_pointwise_vectorize=True,
        cute_chained_pointwise_unroll=8,
        cute_chained_preparation_pipeline=True,
        cute_chained_leaf_pipeline="rectangular_tma" if rectangular_leaf else "legacy",
    )
    with _cpu_codegen():
        bound = kernel._bind_isolated(args)
        with bound.env.use_runtime_arg_values(_runtime_values(kernel, args)):
            source = bound.to_code(config)
    assert "chain_pointwise_cache_2" in source
    assert ("chained_rectangular_leaf_tma" in source) == rectangular_leaf


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("loop,steps", [(False, 1), (True, 0), (True, 1), (True, 5)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_independent_cache_set_gpu_preserves_values_replay_and_inputs(
    loop, steps, dtype
):
    kernel = _loop_cache_set if loop else _root_cache_set
    args = _args(DEVICE, loop, dtype, steps)
    saved = tuple(arg.clone() for arg in args if isinstance(arg, torch.Tensor))
    compiled = []
    for count in (1, 2):
        bound = kernel._bind_isolated(args)
        with bound.env.use_runtime_arg_values(_runtime_values(kernel, args)):
            compiled.append(bound.compile_config(_config(loop, count)))
    expected, actual = compiled[0](*args), compiled[1](*args)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(compiled[1](*args), actual, atol=0, rtol=0)
    torch.testing.assert_close(
        tuple(arg for arg in args if isinstance(arg, torch.Tensor)),
        saved,
        atol=0,
        rtol=0,
    )


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("bt32", [False, True])
@pytest.mark.parametrize("value_tile", [64, 128])
@pytest.mark.parametrize("consumer_warps", [4, 8])
def test_cache_set_prefill_gpu_preserves_ragged_source_and_reused_slot(
    bt32, value_tile, consumer_warps
):
    _run_preparation_prefill(bt32, value_tile, consumer_warps, cache_entries=4)


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("bt32", [False, True])
def test_cache_set_tma_prefill_gpu(bt32):
    _run_preparation_prefill(bt32, 128, 4, rectangular_leaf=True, cache_entries=4)
