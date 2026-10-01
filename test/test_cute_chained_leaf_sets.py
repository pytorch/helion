from __future__ import annotations

from dataclasses import FrozenInstanceError
from dataclasses import replace
from pathlib import Path
import sys
from types import ModuleType
from unittest.mock import patch

import pytest
import torch

from .test_cute_chained_preparation_cut import _kda_fixture
from .test_cute_chained_preparation_frame import _safe
from .test_cute_chained_preparation_leaves import _leaf_config
from .test_cute_chained_preparation_leaves import _source
import helion
from helion._compiler.cute import chained_preparation_leaves as leaf_module
from helion._compiler.cute import chained_preparation_pipeline as pipeline_module
from helion._compiler.cute.chained_leaf_sets import LeafSetProtocol
from helion._compiler.cute.chained_leaf_sets import capture_leaf_set_witness
from helion._compiler.cute.chained_leaf_sets import extend_preparation_leaves
from helion._compiler.cute.chained_prepared_groups import prepared_group_candidates
from helion._compiler.cute.chained_prepared_operands import plan_prepared_operands
import helion.language as hl


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _reused_raw(a, b, weight, common, initial, steps: int, origin: int, end: int):
    steps = hl.specialize(steps)
    history = torch.empty((steps, 16, 128), device=a.device, dtype=torch.float32)
    final = torch.empty_like(initial)
    for rr, cc in hl.tile([16, 128], block_size=[16, 128]):
        state = initial[rr, cc]
        for step in hl.tile(steps, block_size=1):
            kk, jj = hl.arange(64), hl.arange(64)
            row = rr.index + origin + step.id * 16
            av = hl.load(a, [row, kk], extra_mask=(row < end)[:, None])
            bv = hl.load(b, [row, kk], extra_mask=(row < end)[:, None])
            first = hl.dot((av.float() + bv.float()).to(torch.bfloat16), weight[kk, jj])
            second = hl.dot(
                (av.float() - bv.float()).to(torch.bfloat16), weight[kk, jj]
            )
            state = hl.dot(
                (first + second).to(torch.bfloat16), common[step.id, jj, cc], acc=state
            )
            history[step.id, rr, cc] = state
        final[rr, cc] = state
    return history, final


def _inputs(dtype, steps):
    return (
        *(torch.empty((3 + max(steps, 1) * 16, 64), dtype=dtype) for _ in range(2)),
        torch.empty((64, 64), dtype=torch.bfloat16),
        torch.empty((max(steps, 1), 64, 128), dtype=torch.bfloat16),
        torch.empty((16, 128)),
        steps,
        3,
        3 + steps * 16 - 1,
    )


def _capture(kernel, args, config):
    original_emit = pipeline_module.emit_preparation_pipeline
    original_candidates = leaf_module.preparation_leaf_candidates
    original_insert = leaf_module.insert_preparation_leaf
    found = {}

    def emit(cg, plan, pipeline, *args, **kwargs):
        found["recurrence"] = pipeline.recurrence
        return original_emit(cg, plan, pipeline, *args, **kwargs)

    def candidates(cg, plan, frame):
        result = original_candidates(cg, plan, frame)
        found.update(plan=plan, discovery=frame, candidates=result)
        return result

    def insert(frame, leaf, **kwargs):
        result = original_insert(frame, leaf, **kwargs)
        if result is not None and "frame" not in found:
            plan, recurrence = found["plan"], found["recurrence"]
            witness = capture_leaf_set_witness(
                plan,
                found["discovery"],
                found["candidates"],
                prepared_groups=kwargs["prepared_groups"],
            )
            assert witness is not None
            groups = prepared_group_candidates(plan, result, recurrence)
            assert groups is not None
            by_group = {group.group: group for group in groups}
            found.update(
                witness=witness,
                frame=result,
                leaves=(leaf,),
                prepared_operands=plan_prepared_operands(plan, result, recurrence),
                prepared_groups=tuple(
                    replace(binding, candidate=by_group[binding.candidate.group])
                    for binding in kwargs["prepared_groups"]
                ),
            )
        return result

    with (
        patch.object(pipeline_module, "emit_preparation_pipeline", emit),
        patch.object(leaf_module, "preparation_leaf_candidates", candidates),
        patch.object(leaf_module, "insert_preparation_leaf", insert),
    ):
        found["source"] = _source(kernel, args, config)
    assert "frame" in found
    return found


@pytest.fixture(scope="module")
def captured_kda():
    namespace = ModuleType("benchmarks")
    namespace.__path__ = [str(Path(__file__).resolve().parents[1] / "benchmarks")]
    previous = sys.modules.get("benchmarks")
    sys.modules["benchmarks"] = namespace
    try:
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
        return _capture(kernel, args, config)
    finally:
        if previous is None:
            sys.modules.pop("benchmarks", None)
        else:
            sys.modules["benchmarks"] = previous


def _extend(captured, *, max_count=3, capacity=66304, **changes):
    arguments = {
        name: captured[name]
        for name in (
            "plan",
            "witness",
            "frame",
            "recurrence",
            "leaves",
            "prepared_operands",
            "prepared_groups",
        )
    }
    arguments.update(changes)
    return extend_preparation_leaves(
        **arguments, max_count=max_count, capacity=capacity
    )


def _assert_unchanged_offsets(before, after):
    for old in before.layout.regions:
        new = after.layout.region(old.name)
        assert (new.byte_offset, new.byte_size, new.alignment) == (
            old.byte_offset,
            old.byte_size,
            old.alignment,
        )
    assert after.buffers[: len(before.buffers)] == before.buffers
    assert after.cut is before.cut and after.frontier_order is before.frontier_order
    _safe(after)


def test_actual_raw_leaf_set_keeps_original_graph_and_all_native_bindings(captured_kda):
    result = _extend(captured_kda)
    assert result is not None and len(result.leaves) == 3
    assert captured_kda["frame"].layout.allocated_bytes == 49920
    assert result.frame.layout.allocated_bytes == 66304
    assert [
        result.frame.layout.region(leaf.name).byte_offset for leaf in result.leaves[1:]
    ] == [49920, 58112]
    _assert_unchanged_offsets(captured_kda["frame"], result.frame)
    assert result.prepared_groups and result.prepared_operands
    assert result.prepared_operands == plan_prepared_operands(
        captured_kda["plan"], result.frame, captured_kda["recurrence"]
    )
    groups = prepared_group_candidates(
        captured_kda["plan"], result.frame, captured_kda["recurrence"]
    )
    assert groups is not None
    for binding in result.prepared_groups:
        assert binding.candidate in groups
        for member in binding.candidate.members:
            assert (
                result.frame.layout.region(member.buffer.name).byte_offset
                == binding.byte_offset + member.byte_offset
            )
    original = captured_kda["discovery"]
    originals = {leaf.node: leaf for leaf in captured_kda["candidates"]}
    nonleaf = [action for action in result.frame.actions if action.kind != "leaf"]
    assert len(nonleaf) == len(original.actions)
    for leaf in result.leaves:
        old = originals[leaf.node]
        assert leaf.proof is old.proof and leaf.wrapper is old.wrapper
        assert leaf.read_events == tuple(
            nonleaf[event].event for event in old.read_events
        )
        assert (leaf.first_event, leaf.last_event) == (
            leaf.read_events[0],
            leaf.read_events[-1],
        )
        assert all(
            leaf.name in result.frame.actions[event].reads for event in leaf.read_events
        )
        region = result.frame.layout.region(leaf.name)
        assert region.live_from < leaf.first_event
        assert region.live_until == leaf.last_event + 1
    # Repeated extension is bookkeeping-idempotent, including final event fields.
    assert (
        _extend(
            captured_kda,
            frame=result.frame,
            leaves=result.leaves,
            prepared_operands=result.prepared_operands,
            prepared_groups=result.prepared_groups,
        )
        == result
    )
    with pytest.raises(FrozenInstanceError):
        result.frame = original  # pyrefly: ignore [read-only]


@pytest.mark.parametrize(
    "capacity,count", [(49920, 1), (58111, 1), (58112, 2), (66303, 2), (66304, 3)]
)
def test_quota_is_exact_and_failed_candidate_retains_original_load(
    captured_kda, capacity, count
):
    result = _extend(captured_kda, capacity=capacity)
    assert result is not None and len(result.leaves) == count
    assert result.frame.layout.allocated_bytes <= capacity
    _assert_unchanged_offsets(captured_kda["frame"], result.frame)


def test_one_leaf_reuses_original_frame_and_native_proof_objects(captured_kda):
    result = _extend(captured_kda, max_count=1, capacity=49920)
    assert result is not None and result.frame is captured_kda["frame"]
    assert result.prepared_operands is captured_kda["prepared_operands"]
    assert result.prepared_groups is captured_kda["prepared_groups"]


@pytest.mark.parametrize(
    "mode",
    ["duplicate_node", "duplicate_name", "empty", "unordered", "ambiguous_action"],
)
def test_discovery_witness_requires_unique_complete_original_actions(
    captured_kda, mode
):
    frame = captured_kda["discovery"]
    candidates = captured_kda["candidates"]
    first, *others = candidates
    if mode == "duplicate_node":
        candidates = (*candidates, first)
    elif mode == "duplicate_name":
        candidates = (first, replace(others[0], name=first.name), *others[1:])
    elif mode == "empty":
        candidates = (replace(first, read_events=()), *others)
    elif mode == "unordered":
        candidates = (replace(first, read_events=first.read_events[::-1]), *others)
    else:
        frame = replace(
            frame,
            actions=(
                frame.actions[0],
                replace(frame.actions[0], event=1),
                *frame.actions[2:],
            ),
        )
    assert capture_leaf_set_witness(captured_kda["plan"], frame, candidates) is None


def test_wrapper_mutation_invalidates_witness_without_changing_graph(captured_kda):
    wrapper = captured_kda["candidates"][0].wrapper
    previous = wrapper["tile"]
    try:
        wrapper["tile"] = (1, 1)
        assert _extend(captured_kda) is None
    finally:
        wrapper["tile"] = previous


@pytest.mark.parametrize("mutation", ["dtype", "stride"])
def test_tensor_metadata_mutation_invalidates_witness(captured_kda, mutation):
    leaf = captured_kda["candidates"][0]
    node = leaf.node if mutation == "dtype" else leaf.node.args[0]
    previous = node.meta["val"]
    try:
        node.meta["val"] = (
            torch.empty(leaf.proof.tile_shape, dtype=torch.float32)
            if mutation == "dtype"
            else torch.empty_strided(
                leaf.proof.shape,
                tuple(stride * 2 for stride in leaf.proof.strides),
                dtype=previous.dtype,
            )
        )
        assert _extend(captured_kda) is None
    finally:
        node.meta["val"] = previous


@pytest.mark.parametrize("bad", [0, -1, True, 1.0, "2"])
def test_strict_max_count(captured_kda, bad):
    assert _extend(captured_kda, max_count=bad) is None


@pytest.mark.parametrize("bad", [0, -1, True, 66304.0, 49919])
def test_strict_capacity(captured_kda, bad):
    assert _extend(captured_kda, capacity=bad) is None


@pytest.mark.parametrize(
    "mode",
    [
        "duplicate",
        "missing",
        "stale_events",
        "changed_reads",
        "moved_alias",
        "native",
        "group",
        "graph",
    ],
)
def test_missing_stale_or_nonunique_witness_fails_closed(captured_kda, mode):
    changes = {}
    witness, frame = captured_kda["witness"], captured_kda["frame"]
    if mode == "duplicate":
        changes["leaves"] = captured_kda["leaves"] * 2
    elif mode == "missing":
        changes["witness"] = replace(witness, candidates=witness.candidates[1:])
    elif mode == "stale_events":
        changes["witness"] = replace(
            witness,
            candidates=(
                replace(witness.candidates[0], read_events=(0,)),
                *witness.candidates[1:],
            ),
        )
    elif mode == "changed_reads":
        changes["frame"] = replace(
            frame,
            actions=(
                replace(frame.actions[0], reads=("invented",)),
                *frame.actions[1:],
            ),
        )
    elif mode == "moved_alias":
        regions = frame.layout.regions
        changes["frame"] = replace(
            frame,
            layout=replace(
                frame.layout,
                regions=(
                    replace(regions[0], byte_offset=regions[0].byte_offset + 128),
                    *regions[1:],
                ),
            ),
        )
    elif mode == "native":
        changes["prepared_operands"] = ()
    elif mode == "group":
        changes["prepared_groups"] = ()
    else:
        node = witness.candidates[0].node
        old = node.kwargs
        try:
            node.kwargs = {**old, "extra_mask": False}
            assert _extend(captured_kda) is None
        finally:
            node.kwargs = old
        return
    assert _extend(captured_kda, **changes) is None


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("steps", [0, 1, 5])
def test_generic_masked_typed_loops_keep_original_leaf_proofs(dtype, steps):
    initialized = torch.cuda.is_initialized()
    captured = _capture(_reused_raw, _inputs(dtype, steps), _leaf_config())
    result = _extend(captured, max_count=4, capacity=1 << 20)
    assert result is not None and len(result.leaves) >= 2
    _assert_unchanged_offsets(captured["frame"], result.frame)
    added = result.leaves[1:]
    assert all(len(leaf.read_events) >= 2 for leaf in added)
    assert any(leaf.node.meta["val"].dtype == dtype for leaf in added)
    assert all(
        leaf.proof.mask is not None
        for leaf in result.leaves
        if leaf.node.meta["val"].dtype == dtype
    )
    assert torch.cuda.is_initialized() == initialized


@pytest.mark.parametrize("slots", [2, 3, 7])
@pytest.mark.parametrize("count", [1, 2, 4])
@pytest.mark.parametrize("cohorts", [False, True])
def test_private_barriers_and_generation_on_zero_one_and_reused_slots(
    slots, count, cohorts
):
    protocol = LeafSetProtocol(slots, count, cohorts)
    indices = {
        protocol.leaf_index(leaf, slot)
        for leaf in range(count)
        for slot in range(slots)
    }
    assert indices == set(range(2 * slots, protocol.barrier_count))
    assert protocol.allocated_bytes >= protocol.barrier_bytes
    for iterations in (0, 1, 2, 5, 23):
        phases = dict.fromkeys(indices, 0)
        for iteration in range(iterations):
            slot, generation = iteration % slots, iteration // slots
            for leaf in range(count):
                index = protocol.leaf_index(leaf, slot)
                expected = (generation if cohorts else iteration) & 1
                assert phases[index] == expected
                phases[index] ^= 1  # Either TMA completion or scalar fallback arrives.
    assert protocol.phase == (
        "chain_generation & 1" if cohorts else "chain_iteration & 1"
    )


def test_default_pointer_strings_and_full_barrier_allocation_boundary():
    assert LeafSetProtocol(2, 1, False).leaf_pointer(0) == "chain_slot_bars + 4"
    assert (
        LeafSetProtocol(3, 1, True).leaf_pointer(0)
        == "chain_slot_bars + 6 + chain_slot"
    )
    assert (
        LeafSetProtocol(3, 3, True).leaf_pointer(2)
        == "chain_slot_bars + 12 + chain_slot"
    )
    exact, next_block = LeafSetProtocol(3, 14, True), LeafSetProtocol(7, 5, True)
    assert (exact.barrier_count, exact.barrier_bytes, exact.allocated_bytes) == (
        48,
        384,
        384,
    )
    assert (
        next_block.barrier_count,
        next_block.barrier_bytes,
        next_block.allocated_bytes,
    ) == (49, 392, 512)


@pytest.mark.parametrize(
    "slots,count,cohorts",
    [
        (0, 1, False),
        (2, 0, False),
        (True, 1, False),
        (2, True, False),
        (2, 1, 1),
        (2.0, 1, False),
        (2, -1, False),
    ],
)
def test_invalid_protocol_types_and_counts(slots, count, cohorts):
    with pytest.raises(ValueError):
        LeafSetProtocol(slots, count, cohorts)


@pytest.mark.parametrize(
    "ordinal,slot", [(-1, 0), (2, 0), (True, 0), (0, -1), (0, 3), (0, True)]
)
def test_invalid_protocol_coordinates(ordinal, slot):
    with pytest.raises(ValueError):
        LeafSetProtocol(3, 2, True).leaf_index(ordinal, slot)
