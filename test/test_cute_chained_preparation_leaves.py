from __future__ import annotations

import ast
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
import sys
from types import ModuleType
from unittest.mock import patch

import pytest
import torch
from torch.fx import Node

from ._cute_aux import _cpu_codegen
from .test_cute_chained_preparation_cut import _kda_fixture
from .test_cute_chained_preparation_cut import _runtime_values
from .test_cute_chained_preparation_frame import _safe
from .test_cute_chained_preparation_pipeline import _config
from .test_cute_chained_rectangular_leaf import _environment
from .test_cute_chained_rectangular_leaf import _rectangle
import helion
from helion._compiler.cute import chained_preparation_leaves as leaves
from helion._compiler.cute import chained_prepared_values as prepared_values
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends
import helion.language as hl


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _raw_sequence(source, weights, common, initial, steps: int, origin: int, end: int):
    steps = hl.specialize(steps)
    history = torch.empty((steps, 16, 128), device=source.device, dtype=torch.float32)
    final = torch.empty_like(initial)
    for rr, cc in hl.tile([16, 128], block_size=[16, 128]):
        state = initial[rr, cc]
        for step in hl.tile(steps, block_size=1):
            kk, jj = hl.arange(64), hl.arange(64)
            row = rr.index + origin + step.id * 16
            raw = hl.load(source, [row, kk], extra_mask=(row < end)[:, None])
            first = hl.dot(raw.to(torch.bfloat16), weights[kk, jj].to(torch.bfloat16))
            state = hl.dot(
                first.to(torch.bfloat16),
                common[step.id, jj, cc].to(torch.bfloat16),
                acc=state,
            )
            history[step.id, rr, cc] = state
        final[rr, cc] = state
    return history, final


def _inputs(dtype=torch.bfloat16, steps=3, tail=0, *, device="cpu"):
    torch.manual_seed(9854)
    origin = 3
    return (
        torch.randn((origin + max(steps, 1) * 16, 64), device=device, dtype=dtype)
        * 0.125,
        torch.randn((64, 64), device=device, dtype=dtype) * 0.125,
        torch.randn((max(steps, 1), 64, 128), device=device, dtype=dtype) * 0.125,
        torch.randn((16, 128), device=device) * 0.125,
        steps,
        origin,
        origin + steps * 16 - tail,
    )


def _leaf_config(enabled=True):
    config = _config(16, pipeline=True)
    if enabled:
        config.config["cute_chained_leaf_pipeline"] = "rectangular_tma"
    return config


@contextmanager
def _capture():
    original_candidates = leaves.preparation_leaf_candidates
    original_insert = leaves.insert_preparation_leaf
    original_bind = prepared_values.bind_frame_buffers
    captured = {"candidates": [], "insertions": [], "bindings": []}

    def candidates(cg, plan, frame):
        result = original_candidates(cg, plan, frame)
        captured["candidates"].append((plan, frame, result))
        return result

    def insert(frame, leaf, **kwargs):
        result = original_insert(frame, leaf, **kwargs)
        captured["insertions"].append((frame, leaf, kwargs, result))
        return result

    def bind(frame, *args, **kwargs):
        captured["bindings"].append((frame, kwargs))
        return original_bind(frame, *args, **kwargs)

    with (
        patch.object(leaves, "preparation_leaf_candidates", candidates),
        patch.object(leaves, "insert_preparation_leaf", insert),
        patch.object(prepared_values, "bind_frame_buffers", bind),
    ):
        yield captured


def _source(kernel, args, config):
    with _cpu_codegen():
        bound = kernel._bind_isolated(args)
        with bound.env.use_runtime_arg_values(_runtime_values(kernel, args)):
            return bound.to_code(config)


def _assert_insertion(before, leaf, after, *, same_size=True):
    assert after is not None
    event = leaf.first_event
    assert after.cut is before.cut
    assert after.frontier_order is before.frontier_order
    assert after.buffers[:-1] == before.buffers
    assert after.buffers[-1].node is leaf.node
    assert after.buffers[-1].dtype == leaf.node.meta["val"].dtype
    assert after.buffers[-1].shape == leaf.proof.tile_shape
    assert after.actions[event].kind == "leaf"
    assert after.actions[event].writes == (leaf.name,)
    old_actions = (*after.actions[:event], *after.actions[event + 1 :])
    for old, new in zip(before.actions, old_actions, strict=True):
        assert new == replace(
            old,
            event=old.event + int(old.event >= event),
            reads=(*old.reads, leaf.name)
            if old.event in leaf.read_events
            else old.reads,
        )
    for old in before.layout.regions:
        new = after.layout.region(old.name)
        assert new == replace(
            old,
            live_from=old.live_from + int(old.live_from >= event),
            live_until=old.live_until + int(old.live_until > event),
        )
    region = after.layout.region(leaf.name)
    assert region.live_from == event and region.live_until == leaf.last_event + 2
    assert region.byte_size == leaf.byte_size
    assert region.byte_offset % 128 == 0
    if same_size:
        assert after.layout.allocated_bytes == before.layout.allocated_bytes
    assert after.peak_live_bytes == max(
        sum(
            region.byte_size
            for region in after.layout.regions
            if region.live_from <= action.event < region.live_until
        )
        for action in after.actions
    )
    for stage in after.stages:
        assert stage.a is after.layout.region(stage.a.name)
        assert stage.b is after.layout.region(stage.b.name)
    _safe(after)


@pytest.fixture(scope="module")
def raw_capture():
    args = _inputs()
    with _capture() as captured:
        source = _source(_raw_sequence, args, _leaf_config())
    return args, source, captured


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("steps,tail", [(0, 0), (1, 0), (5, 7)])
def test_public_raw_leaf_keeps_dtype_guards_and_publishes_before_read_cpu(
    dtype, steps, tail
):
    args = _inputs(dtype, steps, tail)
    with _capture() as captured:
        source = _source(_raw_sequence, args, _leaf_config())
    before, leaf, kwargs, after = captured["insertions"][-1]
    _assert_insertion(before, leaf, after)
    assert leaf.node.meta["val"].dtype == dtype
    assert leaf.proof.element_bytes == dtype.itemsize
    assert leaf.wrapper["kind"] == "chained_rectangular_leaf_tma"
    assert leaf.proof.mask is not None
    assert f"{leaf.name}_atom" in source and f"{leaf.name}_tensor" in source
    assert "cute.domain_offset((" in source
    assert "cute.nvgpu.cpasync.tma_partition(" in source
    assert (
        f"mbarrier_arrive_and_expect_tx(chain_slot_bars + 4, {leaf.byte_size})"
        in source
    )
    assert "cute.arch.mbarrier_wait(chain_slot_bars + 4, chain_iteration & 1)" in source
    assert "chain_slot_bars = cute.arch.alloc_smem(cutlass.Int64, 5" in source
    assert len(captured["bindings"]) == 1
    for frame, binding in captured["bindings"]:
        assert frame is after and binding["prepared_leaves"] == (leaf,)
        assert all(
            operand.region == frame.layout.region(operand.buffer.name)
            for operand in binding["prepared_operands"]
        )
    # Both the asynchronous branch and the unchanged masked scalar branch
    # publish exactly one barrier phase before consumers see the typed image.
    tree = ast.parse(source)
    guard = ast.dump(ast.parse(leaf.proof.guard, mode="eval").body)
    transaction = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If) and ast.dump(node.test) == guard
    )
    fallback = ast.unparse(ast.Module(body=transaction.orelse, type_ignores=[]))
    assert "chain_prep_barrier.arrive_and_wait()" in fallback
    assert "cute.arch.mbarrier_arrive(chain_slot_bars + 4)" in fallback
    assert ".load()" in fallback and "else" in fallback
    for step in range(max(steps, 1)):
        scalars = {
            "origin": args[5],
            "end": args[6],
            "chain_origin_0": 0,
            "chain_origin_1": 0,
            "chain_loop_index": step,
            "chain_loop_begin": 0,
            "chain_loop_end": steps,
        }
        enabled = steps > 0 and args[5] + step * 16 + 15 < args[6]
        assert _rectangle(leaf.proof, **scalars) == enabled
        if enabled:
            actual_origin = tuple(
                int(eval(text, _environment(**scalars))) for text in leaf.proof.origin
            )
            assert actual_origin == (args[5] + step * 16, 0)
            assert actual_origin[0] % 16 == 3


@pytest.mark.parametrize(
    "bad",
    [
        "negative",
        "after_ready",
        "empty_reads",
        "wrong_first",
        "wrong_last",
        "collision",
        "capacity",
    ],
)
def test_insertion_rejects_invalid_events_and_insufficient_frame_cpu(raw_capture, bad):
    _, _, captured = raw_capture
    before, leaf, kwargs, after = captured["insertions"][-1]
    if bad == "negative":
        leaf = replace(leaf, first_event=-1)
    elif bad == "after_ready":
        leaf = replace(leaf, last_event=before.actions[-1].event)
    elif bad == "empty_reads":
        leaf = replace(leaf, read_events=())
    elif bad == "wrong_first":
        leaf = replace(leaf, first_event=leaf.first_event + 1)
    elif bad == "wrong_last":
        leaf = replace(leaf, last_event=leaf.last_event + 1)
    elif bad == "collision":
        leaf = replace(leaf, name=before.buffers[0].name)
    else:
        kwargs = {**kwargs, "capacity": before.layout.allocated_bytes - 1}
    assert leaves.insert_preparation_leaf(before, leaf, **kwargs) is None


def test_original_frame_is_immutable_and_second_leaf_is_not_inserted_cpu(raw_capture):
    _, _, captured = raw_capture
    before, leaf, kwargs, after = captured["insertions"][-1]
    _assert_insertion(before, leaf, after)
    assert all(action.kind != "leaf" for action in before.actions)
    assert (
        leaves.insert_preparation_leaf(
            after, replace(leaf, name="another_leaf"), **kwargs
        )
        is None
    )


def test_actual_kda_discovers_raw_fx_leaves_and_inserts_only_fitting_gate_cpu():
    namespace = ModuleType("benchmarks")
    namespace.__path__ = [str(Path(__file__).resolve().parents[1] / "benchmarks")]
    with patch.dict(sys.modules, {"benchmarks": namespace}):
        kernel, args = _kda_fixture()
        config = _leaf_config()
        config.config.update(
            block_sizes=[128],
            num_stages=2,
            cute_chained_group_contractions=True,
            cute_chained_scratch_layout="xor",
            cute_chained_pointwise_vectorize=True,
            cute_chained_scan_schedule="warp",
            cute_chained_pointwise_cache_bytes=4096,
        )
        with _capture() as captured:
            source = _source(kernel, args, config)
    plan, frame, candidates = captured["candidates"][0]
    by_source = {}
    for leaf in candidates:
        host = leaf.node.args[0]
        assert isinstance(host, Node)
        name = host.args[0]
        assert isinstance(name, str)
        by_source[name.removesuffix("_rows")] = leaf
    assert set(by_source) == {"q", "k", "v", "gate"}
    successes = [record for record in captured["insertions"] if record[-1] is not None]
    assert len(successes) == 1
    before, leaf, kwargs, after = successes[0]
    assert leaf is by_source["gate"] and leaf.byte_size == 8192
    assert before is frame and before.layout.allocated_bytes == 49920
    assert kwargs["prepared_groups"]
    _assert_insertion(before, leaf, after)
    assert after.layout.allocated_bytes == 49920
    for name in ("q", "k", "v"):
        assert leaves.insert_preparation_leaf(before, by_source[name], **kwargs) is None
    # Independently exercise the exclusive-end rule at a later real read cut.
    # This enlarged test-only quota is not the admitted public pipeline quota.
    late = by_source["v"]
    assert any(
        region.live_until == late.first_event for region in before.layout.regions
    )
    expanded = leaves.insert_preparation_leaf(
        before,
        late,
        capacity=before.layout.allocated_bytes + late.byte_size,
        prepared_groups=kwargs["prepared_groups"],
    )
    assert expanded is not None
    _assert_insertion(before, late, expanded, same_size=False)
    assert expanded.layout.allocated_bytes > before.layout.allocated_bytes
    assert all(
        expanded.layout.region(region.name).live_until == region.live_until
        for region in before.layout.regions
        if region.live_until == late.first_event
    )
    assert f"chain_frames = cute.arch.alloc_smem(cutlass.Uint8, {2 * 49920}" in source
    for new_frame, binding in captured["bindings"]:
        assert new_frame is after
        for old, new in zip(
            kwargs["prepared_groups"], binding["prepared_groups"], strict=True
        ):
            assert old.byte_offset == new.byte_offset
            assert old.candidate.group == new.candidate.group
            assert new.candidate.live_from == old.candidate.live_from + int(
                old.candidate.live_from >= leaf.first_event
            )
            assert new.candidate.live_until == old.candidate.live_until + 1
        for operand in binding["prepared_operands"]:
            assert operand.region == after.layout.region(operand.buffer.name)


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("steps,tail", [(0, 0), (1, 0), (5, 7)])
def test_raw_leaf_gpu_preserves_non_aligned_origins_masks_replay_and_inputs(
    dtype, steps, tail
):
    args = _inputs(dtype, steps, tail, device=DEVICE)
    saved = tuple(value.clone() for value in args[:4])
    ordinary_bound = _raw_sequence._bind_isolated(args)
    with ordinary_bound.env.use_runtime_arg_values(
        _runtime_values(_raw_sequence, args)
    ):
        ordinary = ordinary_bound.compile_config(_leaf_config(False))
    native_bound = _raw_sequence._bind_isolated(args)
    with native_bound.env.use_runtime_arg_values(_runtime_values(_raw_sequence, args)):
        source = native_bound.to_code(_leaf_config())
        assert "cute.nvgpu.cpasync.tma_partition(" in source
        native = native_bound.compile_config(_leaf_config())
    expected, actual = ordinary(*args), native(*args)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(native(*args), actual, atol=0, rtol=0)
    # Cached TensorMaps describe the allocation, not this invocation's origin
    # or mask. Reuse both compiled callables with new owned scalar values.
    shifted = (*args[:5], args[5] - 1, args[6] - 1)
    torch.testing.assert_close(native(*shifted), ordinary(*shifted), atol=0, rtol=0)
    torch.testing.assert_close(args[:4], saved, atol=0, rtol=0)
