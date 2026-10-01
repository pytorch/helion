from __future__ import annotations

import ast
from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from .test_cute_chained_loop_tmem_carry import _fixture
from .test_cute_chained_loop_tmem_transport import _kda_fixture
from .test_cute_chained_loop_tmem_transport import _source
from .test_cute_chained_operand_retention_integration import _kda_config
from .test_cute_chained_preparation_pipeline import _config
from .test_cute_chained_snapshot_compile import _compile_original_host
import helion
from helion._compiler.cute import chained_loop_tmem_carry_transport as transport
from helion._compiler.cute.chained_loop_tmem_carry import plan_loop_tmem_carry
from helion._compiler.cute.chained_tmem_accumulator import (
    plan_tmem_accumulator_residency,
)
import helion.language as hl


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _sequence(a, b, c, d, initial, log_decay):
    steps, rows, reduction = a.shape
    width = initial.shape[-1]
    history = torch.empty((steps, rows, width), device=a.device)
    final = torch.empty_like(initial)
    for rr in hl.tile(rows, block_size=128):
        state = initial[rr, :]
        for step in hl.tile(steps, block_size=1):
            kk, jj = hl.arange(reduction), hl.arange(width)
            prepared = hl.dot(c[step.id, jj, kk], d[step.id, kk, jj]).to(a.dtype)
            decay = log_decay[step.id, jj].exp()
            projected = hl.dot(state.to(a.dtype), prepared, out_dtype=torch.float32)
            state = hl.dot(
                a[step.id, rr, kk], b[step.id, kk, jj], acc=state * decay[None, :]
            )
            history[step.id, rr, jj] = projected
        final[rr, :] = state
    return history, final


def _args(dtype):
    return (
        *(
            torch.empty(shape, dtype=dtype)
            for shape in ((3, 128, 16), (3, 16, 64), (3, 64, 16), (3, 16, 64))
        ),
        torch.empty((128, 64)),
        torch.empty((3, 64)),
    )


def _capture_source(kernel, args, config):
    records = []
    original = transport.prepare_loop_tmem_carry

    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        records.append((result, kwargs.get("snapshot_cut")))
        return result

    with patch.object(transport, "prepare_loop_tmem_carry", capture):
        source = _source(kernel, args, config)
    assert len(records) == 1
    carry, cut = records[0]
    assert carry is not None
    return source, carry, cut


def _fanout_config(snapshot=32, seed=32):
    config = _config(16, pipeline=True, consumer_warps=8)
    config.config.update(
        cute_chained_warp_mma_rows=64,
        cute_chained_snapshot_tile_columns=snapshot,
        cute_chained_seed_tile_columns=seed,
    )
    return config


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
def test_decay_state_uses_ready_image_and_original_fp32_panel(dtype):
    source, carry, cut = _capture_source(_sequence, _args(dtype), _fanout_config())
    assert cut is not None and transport.snapshot_seed_available(carry.candidate, cut)
    assert carry.snapshot_seed is not None
    lines, _value = carry.snapshot_seed
    expression = "\n".join(lines)
    assert "chain_prepared_" in expression
    assert f"{carry.prefix}_values[{carry.prefix}_snapshot_index]" in expression
    assert f"{carry.seed}_load" not in source
    tree = ast.parse(source)
    panel = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == f"{carry.prefix}_snapshot_panel"
    )
    body = ast.unparse(panel)
    assert body.count("fence_view_async_tmem_load()") == 1
    assert body.count("chain_tmem_barrier.arrive_and_wait()") == 1
    transforms = [node for node in panel.body if isinstance(node, ast.For)]
    assert len(transforms) == 2
    stores = [
        index
        for index, node in enumerate(panel.body)
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and ast.unparse(node.value.func) == "cute.copy"
    ]
    assert len(stores) == 3  # Original FP32 read, packed snapshot, FP32 seed.
    assert stores[1] < panel.body.index(transforms[1]) < stores[2]
    assert f"{carry.prefix}_values[" in ast.unparse(transforms[1].body[-1].targets[0])
    assert "_update_writer.partition_D(" + carry.prefix + "_snapshot_segment)" in body
    assert "chain_recurrence_barrier.arrive_and_wait()" not in body
    assert f"{carry.prefix}_snapshot_column" in expression
    assert "_snapshot_column -" not in expression
    side = {
        node
        for node in carry.candidate.accumulator.all_input_nodes
        if node not in carry.candidate.accumulator_nodes
    }
    assert side
    # A value's presence in an arbitrary boundary dict does not establish READY.
    unpublished = replace(
        cut, images=tuple(i for i in cut.images if i.node not in side)
    )
    assert not transport.snapshot_seed_available(carry.candidate, unpublished)


@pytest.mark.parametrize("snapshot,seed", ((0, 0), (0, 32), (32, 0)))
def test_unselected_modes_keep_original_seed_read(snapshot, seed):
    with patch.object(transport, "snapshot_seed_available", side_effect=AssertionError):
        source, carry, cut = _capture_source(
            _sequence, _args(torch.bfloat16), _fanout_config(snapshot, seed)
        )
    assert cut is None and carry.snapshot_seed is None
    assert f"{carry.seed}_load" in source
    assert "_snapshot_update_copy" not in source


def test_late_product_and_other_carry_coefficients_do_not_gain_fanout():
    plan, groups = _fixture(width=16)
    residency = plan_tmem_accumulator_residency(plan, groups)
    assert residency is not None
    original = plan_loop_tmem_carry(plan, groups, residency)
    assert original is not None
    # This same-shape coefficient is produced from the old state by an earlier
    # recurrence dot. It must not be mistaken for a prepared coefficient.
    original.accumulator.args = (original.input, plan.dots[0])
    assert plan_loop_tmem_carry(plan, groups, residency) is None
    other_plan, other_groups = _fixture(mode="reads_other_carry")
    assert plan_loop_tmem_carry(other_plan, other_groups) is None


def test_actual_kda_fanout_preserves_member_origin_and_typed_raw_frontier():
    kernel, args = _kda_fixture()
    config = _kda_config()
    config.config.update(
        cute_chained_snapshot_tile_columns=32,
        cute_chained_compact_preparation=True,
    )
    source, carry, cut = _capture_source(kernel, args, config)
    assert cut is not None and carry.snapshot_seed is not None
    assert carry.candidate.member_offset == 32
    assert f"{carry.seed}_load" not in source
    expression = "\n".join(carry.snapshot_seed[0])
    assert f"{carry.prefix}_snapshot_column" in expression
    assert f"{carry.prefix}_snapshot_column - 32" not in expression
    assert "_raw" in source


def test_decay_fanout_compiles_original_host_without_cuda(tmp_path, monkeypatch):
    args = _args(torch.float16)
    source = _source(_sequence, args, _fanout_config())
    monkeypatch.chdir(tmp_path)
    with patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA forbidden")):
        ptx, launch = _compile_original_host(source, args, tmp_path)
    assert launch["block"] == (512, 1, 1)
    assert "tcgen05.ld.sync.aligned.32x32b.x32" in ptx
    assert "tcgen05.st.sync.aligned.32x32b.x32" in ptx
