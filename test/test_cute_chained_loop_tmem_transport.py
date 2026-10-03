from __future__ import annotations

import ast
from unittest.mock import patch

import pytest
import torch

from ._cute_aux import _cpu_codegen
from .test_cute_chained_preparation_cut import _kda_fixture
from .test_cute_chained_preparation_cut import _runtime_values
from .test_cute_chained_preparation_pipeline import _config
import helion
from helion._compiler.cute import chained_loop_tmem_transport as transport
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends
import helion.language as hl


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _packed_sequence(a, b, c, d, initial):
    steps, rows, reduction = a.shape
    width = initial.shape[-1]
    history = torch.empty((steps, rows, width), device=a.device)
    final = torch.empty_like(initial)
    for rr in hl.tile(rows, block_size=128):
        state = initial[rr, :]
        for step in hl.tile(steps, block_size=1):
            kk, jj = hl.arange(reduction), hl.arange(width)
            prepared = hl.dot(c[step.id, jj, kk], d[step.id, kk, jj]).to(a.dtype)
            source = hl.dot(a[step.id, rr, kk], b[step.id, kk, jj], acc=state)
            image = (source * 0.5 + state * 0.125).to(a.dtype)
            state = hl.dot(image, prepared, out_dtype=torch.float32)
            history[step.id, rr, jj] = state
        final[rr, :] = state
    return history, final


def _inputs(device, dtype, steps=3):
    torch.manual_seed(9573)
    return (
        *(
            torch.randn(shape, device=device, dtype=dtype) * 0.125
            for shape in (
                (steps, 128, 16),
                (steps, 16, 32),
                (steps, 32, 16),
                (steps, 16, 32),
            )
        ),
        torch.randn((128, 32), device=device) * 0.125,
    )


def _source(kernel, args, config):
    with _cpu_codegen():
        bound = kernel._bind_isolated(args)
        with bound.env.use_runtime_arg_values(_runtime_values(kernel, args)):
            return bound.to_code(config)


@pytest.mark.parametrize("consumer_warps", [4, 8])
def test_actual_kda_packs_two_disjoint_images_with_original_allocation_cpu(
    consumer_warps,
):
    kernel, args = _kda_fixture()
    config = _config(16, pipeline=True, consumer_warps=consumer_warps)
    config.config.update(
        block_sizes=[128],
        cute_chained_group_contractions=True,
        cute_chained_scratch_layout="xor",
        cute_chained_pointwise_vectorize=True,
        cute_chained_scan_schedule="warp",
        cute_chained_pointwise_cache_bytes=4096,
        cute_chained_pointwise_unroll=8,
    )
    with patch(
        "helion._compiler.cute.chained_loop_tmem_carry.plan_loop_tmem_carry",
        return_value=None,
    ):
        source = _source(kernel, args, config)
    assert "chain_allocator.allocate(256)" in source
    assert "NamedBarrier(barrier_id=4, num_threads=128)" in source
    for stage, destination, offset in ((10, 11, 160), (11, 13, 192)):
        assert (
            f"chain_{stage}_transport_target = cute.make_tensor(chain_tptr + {offset},"
            in source
        )
        assert f"chain_{stage}_result_row" not in source
        assert f"chain_{destination}_a_ptr" not in source
        assert f"chain_{destination}_weight_layout" in source
    assert "chain_13_seed_14_segment" in source
    assert "chain_13_seed_13_segment" not in source
    if consumer_warps == 8:
        for node in ast.walk(ast.parse(source)):
            if (
                isinstance(node, ast.If)
                and ast.unparse(node.test) == "chain_recurrence_thread < 128"
            ):
                assert "chain_recurrence_barrier.arrive_and_wait()" not in ast.unparse(
                    node
                )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_typed_loop_packs_original_expression_and_can_fall_back_cpu(dtype):
    args = _inputs("cpu", dtype)
    config = _config(16, pipeline=True, consumer_warps=8)
    source = _source(_packed_sequence, args, config)
    assert "chain_1_transport_values" in source
    assert "chain_1_result_row" not in source
    assert "chain_2_a_ptr" not in source
    assert "chain_2_weight_layout" in source
    with patch.object(transport, "plan_loop_tmem_bridges", return_value=()):
        ordinary = _source(_packed_sequence, args, config)
    assert "chain_1_result_row" in ordinary
    assert "chain_2_a_ptr" in ordinary
    assert "chain_tmem_barrier" not in ordinary


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("steps", [0, 1, 5])
@pytest.mark.parametrize("consumer_warps", [4, 8])
def test_loop_tmem_gpu_keeps_original_rounding_state_and_replay(
    dtype, steps, consumer_warps
):
    args = _inputs(DEVICE, dtype, steps)
    saved = tuple(value.clone() for value in args)
    config = _config(16, pipeline=True, consumer_warps=consumer_warps)
    bound = _packed_sequence._bind_isolated(args)
    with (
        bound.env.use_runtime_arg_values(_runtime_values(_packed_sequence, args)),
        patch.object(transport, "plan_loop_tmem_bridges", return_value=()),
    ):
        ordinary = bound.compile_config(config)
    packed_bound = _packed_sequence._bind_isolated(args)
    with packed_bound.env.use_runtime_arg_values(
        _runtime_values(_packed_sequence, args)
    ):
        source = packed_bound.to_code(config)
        assert "chain_1_transport_values" in source
        packed = packed_bound.compile_config(config)
    assert packed is not ordinary
    actual = packed(*args)
    torch.testing.assert_close(actual, ordinary(*args), atol=0, rtol=0)
    torch.testing.assert_close(packed(*args), actual, atol=0, rtol=0)
    torch.testing.assert_close(args, saved, atol=0, rtol=0)
