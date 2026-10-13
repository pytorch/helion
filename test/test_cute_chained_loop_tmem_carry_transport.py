from __future__ import annotations

import ast
from unittest.mock import patch

import pytest
import torch

from .test_cute_chained_loop_tmem_transport import _inputs
from .test_cute_chained_loop_tmem_transport import _kda_fixture
from .test_cute_chained_loop_tmem_transport import _source
from .test_cute_chained_preparation_cut import _runtime_values
from .test_cute_chained_preparation_pipeline import _config
import helion
from helion._compiler.cute import chained_loop
from helion._compiler.cute import chained_loop_tmem_transport as transport
from helion._compiler.cute.chained_execution import ChainedExecution
from helion._compiler.cute.chained_loop_tmem_carry_transport import (
    prepare_loop_tmem_carry,
)
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends
import helion.language as hl


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _resident_carry_sequence(a, b, c, d, initial):
    steps, rows, reduction = a.shape
    width = initial.shape[-1]
    history = torch.empty((steps, rows, width), device=a.device)
    final = torch.empty_like(initial)
    for rr in hl.tile(rows, block_size=128):
        state = initial[rr, :]
        for step in hl.tile(steps, block_size=1):
            kk, jj = hl.arange(reduction), hl.arange(width)
            prepared = hl.dot(c[step.id, jj, kk], d[step.id, kk, jj]).to(a.dtype)
            projected = hl.dot(state.to(a.dtype), prepared, out_dtype=torch.float32)
            state = hl.dot(a[step.id, rr, kk], b[step.id, kk, jj], acc=state * 0.5)
            history[step.id, rr, jj] = projected
        final[rr, :] = state
    return history, final


def test_actual_carry_separates_complete_arena_and_snapshot_cpu():
    original = transport.prepare_loop_tmem_transports
    records = []

    def observe(cg, plan, frontiers, groups):
        result = original(cg, plan, frontiers, groups)
        assert plan.preparation_pipeline is not None
        carry = prepare_loop_tmem_carry(
            cg, plan, frontiers, groups, plan.preparation_pipeline.residency, result[1]
        )
        assert carry is not None
        records.append(carry)
        return result

    kernel, args = _kda_fixture()
    config = _config(16, pipeline=True, consumer_warps=8)
    config.config.update(
        block_sizes=[128],
        cute_chained_group_contractions=True,
        cute_chained_scratch_layout="xor",
        cute_chained_pointwise_vectorize=True,
        cute_chained_scan_schedule="warp",
        cute_chained_pointwise_cache_bytes=4096,
        cute_chained_pointwise_unroll=8,
    )
    with patch.object(transport, "prepare_loop_tmem_transports", observe):
        source = _source(kernel, args, config)
    assert len(records) == 1
    carry = records[0]
    assert (carry.arena_offset, carry.snapshot_offset, carry.required_columns) == (
        224,
        384,
        448,
    )
    assert carry.accumulator(12) == carry.accumulator(13) == "(chain_tptr + 224)"
    assert carry.accumulator(10) is None
    for use in carry.candidate.snapshot_users:
        operand = carry.operand(use.group)
        assert operand is not None and operand.column_offset == 384
        assert operand.base == "chain_tptr"
    assert "chain_seed" not in "\n".join(carry.snapshot_lines)
    assert "chain_13_seed_14_values[chain_13_seed_14_index]" in "\n".join(
        carry.accumulator_lines
    )
    context = ChainedExecution(
        256,
        thread="consumer_thread",
        warp="consumer_warp",
        sync="consumer_barrier.arrive_and_wait()",
    )
    for lines in (
        carry.view(),
        carry.shared_transfer(upload=True),
        carry.shared_transfer(upload=False),
        carry.snapshot(context),
        carry.seed_values(context),
    ):
        ast.parse("\n".join(lines))
    assert "chain_tptr + 256" in "\n".join(carry.view())
    snapshot = "\n".join(carry.snapshot(context))
    assert "if consumer_thread < 128:" in snapshot
    assert snapshot.endswith("consumer_barrier.arrive_and_wait()")
    assert "chain_tptr + 384" in snapshot
    assert "chain_allocator.allocate(512)" in source
    assert "chain_12_acc = cute.make_tensor(chain_tptr + 224," in source
    assert "chain_13_acc = cute.make_tensor(chain_tptr + 224," in source
    assert "chain_10_a_ptr" not in source and "chain_12_a_ptr" not in source
    assert "chain_14_c[" not in source
    assert "chain_13_seed_14_load" in source
    assert "chain_loop_carry_2_next" not in source


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_unrelated_carry_loop_uses_separate_snapshot_and_fp32_seed_cpu(dtype):
    original = chained_loop.advance_carries
    transitions = []

    def advance(cg, plan, boundaries, scratch, *, execution, resident_carries):
        assert plan.loop is not None
        assert resident_carries == frozenset(
            carry.input_index for carry in plan.loop.region.carries
        )
        assert len(resident_carries) == 1
        lines = original(
            cg,
            plan,
            boundaries,
            scratch,
            execution=execution,
            resident_carries=resident_carries,
        )
        assert lines == [execution.sync]
        transitions.append(lines)
        return lines

    with patch.object(chained_loop, "advance_carries", advance):
        source = _source(
            _resident_carry_sequence,
            _inputs("cpu", dtype),
            _config(16, pipeline=True, consumer_warps=8),
        )
    assert len(transitions) == 1
    assert "chain_resident_carry_" in source
    assert "_snapshot_values" in source
    assert "chain_2_seed_2_load" in source
    assert "chain_2_c[" not in source
    assert "chain_1_a_ptr" not in source


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("steps", [0, 1, 5])
@pytest.mark.parametrize("consumer_warps", [4, 8])
def test_tmem_carry_gpu_preserves_authoritative_fp32_and_separate_snapshot(
    dtype, steps, consumer_warps
):
    args = _inputs(DEVICE, dtype, steps)
    saved = tuple(value.clone() for value in args)
    config = _config(16, pipeline=True, consumer_warps=consumer_warps)
    bound = _resident_carry_sequence._bind_isolated(args)
    with (
        bound.env.use_runtime_arg_values(
            _runtime_values(_resident_carry_sequence, args)
        ),
        patch(
            "helion._compiler.cute.chained_loop_tmem_carry.plan_loop_tmem_carry",
            return_value=None,
        ),
    ):
        original = bound.compile_config(config)
    resident_bound = _resident_carry_sequence._bind_isolated(args)
    with resident_bound.env.use_runtime_arg_values(
        _runtime_values(_resident_carry_sequence, args)
    ):
        assert "_snapshot_values" in resident_bound.to_code(config)
        resident = resident_bound.compile_config(config)
    assert resident is not original
    actual = resident(*args)
    torch.testing.assert_close(actual, original(*args), atol=0, rtol=0)
    torch.testing.assert_close(resident(*args), actual, atol=0, rtol=0)
    torch.testing.assert_close(args, saved, atol=0, rtol=0)
