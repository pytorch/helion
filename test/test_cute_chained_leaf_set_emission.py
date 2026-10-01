from __future__ import annotations

import ast
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
import sys
from types import ModuleType
from unittest.mock import patch

import pytest
import torch

from ._cute_aux import _cpu_codegen
from .test_cute_chained_leaf_sets import _inputs as _raw_inputs
from .test_cute_chained_leaf_sets import _reused_raw
from .test_cute_chained_preparation_cut import _kda_fixture
from .test_cute_chained_preparation_cut import _runtime_values
from .test_cute_chained_preparation_frame import _safe
import helion
from helion import exc
from helion._compiler.cute import chained_leaf_sets as leaf_sets
from helion._compiler.cute import chained_pipeline_storage as storage_module
from helion._compiler.cute import chained_preparation_leaves as leaf_module
from helion._compiler.cute.chained_prepared_groups import prepared_group_candidates
from helion._compiler.cute.chained_prepared_operands import plan_prepared_operands


def _config(count=None, **overrides):
    values = {
        "block_sizes": [128],
        "num_warps": 16,
        "num_stages": 2,
        "cute_chained_mma_schedule": "tcgen05_tmem",
        "cute_chained_group_contractions": True,
        "cute_chained_warp_mma_rows": 32,
        "cute_chained_pointwise_cache_bytes": 4096,
        "cute_chained_scan_schedule": "warp",
        "cute_chained_scratch_layout": "xor",
        "cute_chained_pointwise_vectorize": True,
        "cute_chained_pointwise_unroll": 8,
        "cute_chained_vector_group": True,
        "cute_chained_preparation_pipeline": True,
        "cute_chained_leaf_pipeline": "rectangular_tma",
        "cute_chained_seed_tile_columns": 32,
        "cute_chained_pointwise_cache_layout": "xor",
        "cute_chained_preparation_cohorts": 3,
        "cute_chained_preparation_unroll": 1,
        "cute_chained_register_islands": True,
    }
    if count is not None:
        values["cute_chained_leaf_count"] = count
    values.update(overrides)
    return helion.Config.from_dict(values)


@pytest.fixture(scope="module")
def kda_arguments():
    namespace = ModuleType("benchmarks")
    namespace.__path__ = [str(Path(__file__).resolve().parents[1] / "benchmarks")]
    previous = sys.modules.get("benchmarks")
    sys.modules["benchmarks"] = namespace
    try:
        return _kda_fixture()
    finally:
        # Preserve any newly imported native CuTe modules; only this namespace
        # entry is temporary, not the process's entire import cache.
        if previous is None:
            sys.modules.pop("benchmarks", None)
        else:
            sys.modules["benchmarks"] = previous


def _code(arguments, config, *, final_quota=None):
    kernel, args = arguments
    original = storage_module.finalize_pipeline_storage
    records = []

    def finalize(plan, pipeline, transports, **kwargs):
        if final_quota is not None:
            kwargs["capacity_bytes"] = final_quota
        result = original(plan, pipeline, transports, **kwargs)
        records.append((plan, pipeline, transports, kwargs, result))
        return result

    initialized = torch.cuda.is_initialized()
    with (
        _cpu_codegen(),
        patch.object(storage_module, "finalize_pipeline_storage", finalize),
    ):
        bound = kernel._bind_isolated(args)
        with bound.env.use_runtime_arg_values(_runtime_values(kernel, args)):
            source = bound.to_code(config)
    assert torch.cuda.is_initialized() == initialized
    return source, records


def _calls(source, name):
    return [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call) and ast.unparse(node.func) == name
    ]


@pytest.mark.parametrize("cohorts", [1, 3])
def test_explicit_one_leaf_is_full_source_identity_and_never_calls_extension(
    kda_arguments, cohorts
):
    config = _config(cute_chained_preparation_cohorts=cohorts)
    explicit = _config(1, cute_chained_preparation_cohorts=cohorts)
    with patch.object(
        leaf_sets,
        "extend_preparation_leaves",
        side_effect=AssertionError("default invoked leaf extension"),
    ):
        before, _ = _code(kda_arguments, config)
        after, _ = _code(kda_arguments, explicit)
    assert before == after
    count = 5 if cohorts == 1 else 9
    assert (
        f"chain_slot_bars = cute.arch.alloc_smem(cutlass.Int64, {count}, alignment=128)"
        in before
    )


@pytest.mark.parametrize(
    "count,leaves,frame_bytes,charged,bars",
    [
        (1, 1, 49920, 166528, 9),
        (2, 2, 58112, 191104, 12),
        (4, 3, 66304, 215680, 15),
    ],
)
def test_actual_kda_leaf_count_uses_distinct_private_transactions_and_full_accounting(
    kda_arguments, count, leaves, frame_bytes, charged, bars
):
    source, records = _code(kda_arguments, _config(count))
    assert len(records) == 1
    plan, pipeline, transports, kwargs, storage = records[0]
    assert storage is not None
    assert len(pipeline.prepared_leaves) == leaves
    assert len({leaf.node for leaf in pipeline.prepared_leaves}) == leaves
    assert len({leaf.name for leaf in pipeline.prepared_leaves}) == leaves
    assert pipeline.frame.layout.allocated_bytes == frame_bytes
    assert pipeline.slot_barrier_count == bars
    assert storage.charged_bytes == charged
    assert storage.recurrence.a_bytes == storage.recurrence.b_bytes == 0
    assert storage.recurrence.layout.allocated_bytes == 16384
    assert storage.stages == transports
    assert kwargs["revision"].pipeline is pipeline
    assert all(view.pool == "frames" for view in storage.carry_views)
    assert (
        f"chain_frames = cute.arch.alloc_smem(cutlass.Uint8, {3 * frame_bytes}, alignment=128)"
        in source
    )
    assert (
        f"chain_slot_bars = cute.arch.alloc_smem(cutlass.Int64, {bars}, alignment=128)"
        in source
    )
    _safe(pipeline.frame)
    assert (
        len([action for action in pipeline.frame.actions if action.kind == "leaf"])
        == leaves
    )
    for ordinal, leaf in enumerate(pipeline.prepared_leaves):
        pointer = f"chain_slot_bars + {6 + 3 * ordinal} + chain_slot"
        assert any(
            ast.unparse(call.args[0]) == pointer
            for call in _calls(source, "cute.arch.mbarrier_arrive_and_expect_tx")
        )
        waits = [
            call
            for call in _calls(source, "cute.arch.mbarrier_wait")
            if ast.unparse(call.args[0]) == pointer
        ]
        assert (
            len(waits) == 1 and ast.unparse(waits[0].args[1]) == "chain_generation & 1"
        )
        assert leaf.wrapper["kind"] == "chained_rectangular_leaf_tma"
        assert all(name in source for name in leaf.wrapper["kernel_args"])
        region = pipeline.frame.layout.region(leaf.name)
        assert region.byte_size == leaf.byte_size
        action = next(
            action
            for action in pipeline.frame.actions
            if action.kind == "leaf" and action.nodes == (leaf.node,)
        )
        assert action.event == region.live_from
        if count > 1:
            assert all(
                region.live_from < event < region.live_until
                for event in leaf.read_events
            )
            assert all(
                leaf.name in pipeline.frame.actions[event].reads
                for event in leaf.read_events
            )
    # Late domain filtering may remove a singleton, never admit a stale handle.
    available = plan_prepared_operands(plan, pipeline.frame, pipeline.recurrence)
    assert available is not None
    assert all(operand in available for operand in pipeline.prepared_operands)
    groups = prepared_group_candidates(plan, pipeline.frame, pipeline.recurrence)
    assert groups is not None
    for binding in pipeline.prepared_groups:
        assert binding.candidate in groups
        for member in binding.candidate.members:
            assert (
                pipeline.frame.layout.region(member.buffer.name).byte_offset
                == binding.byte_offset + member.byte_offset
            )


@pytest.mark.parametrize("quota,accepted", [(215680, True), (215679, False)])
def test_complete_resource_quota_is_checked_after_actual_multi_leaf_selection(
    kda_arguments, quota, accepted
):
    if accepted:
        _, records = _code(kda_arguments, _config(4), final_quota=quota)
        storage = records[0][-1]
        assert storage is not None and storage.charged_bytes == quota
    else:
        with pytest.raises(
            exc.BackendUnsupported, match="complete post-transport allocation"
        ):
            _code(kda_arguments, _config(4), final_quota=quota)


@pytest.mark.parametrize(
    "failure",
    ["witness_missing", "witness_stale", "no_repeated_candidate", "append_capacity"],
)
def test_explicit_multi_leaf_request_cannot_silently_fall_back(kda_arguments, failure):
    original_capture = leaf_sets.capture_leaf_set_witness
    original_candidates = leaf_module.preparation_leaf_candidates
    original_extend = leaf_sets.extend_preparation_leaves

    def stale(*args, **kwargs):
        witness = original_capture(*args, **kwargs)
        assert witness is not None
        return replace(witness, facts=())

    def nonrepeated(*args, **kwargs):
        candidates = original_candidates(*args, **kwargs)
        return tuple(leaf for leaf in candidates if len(leaf.read_events) == 1)

    def no_append_capacity(
        plan, witness, frame, recurrence, leaves, operands, groups, **kwargs
    ):
        return original_extend(
            plan,
            witness,
            frame,
            recurrence,
            leaves,
            operands,
            groups,
            **{**kwargs, "capacity": frame.layout.allocated_bytes},
        )

    with ExitStack() as stack:
        if failure == "witness_missing":
            stack.enter_context(
                patch.object(leaf_sets, "capture_leaf_set_witness", return_value=None)
            )
        elif failure == "witness_stale":
            stack.enter_context(
                patch.object(leaf_sets, "capture_leaf_set_witness", stale)
            )
        elif failure == "no_repeated_candidate":
            stack.enter_context(
                patch.object(leaf_module, "preparation_leaf_candidates", nonrepeated)
            )
        else:
            stack.enter_context(
                patch.object(leaf_sets, "extend_preparation_leaves", no_append_capacity)
            )
        with pytest.raises(
            exc.BackendUnsupported, match="multiple preparation leaves require"
        ):
            _code(kda_arguments, _config(4))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("steps", [0, 3])
@pytest.mark.parametrize("cohorts", [1, 3])
@pytest.mark.parametrize("count", [2, 4])
def test_public_generic_masked_leaf_sets_keep_typed_images_and_protocol(
    dtype, steps, cohorts, count
):
    args = _raw_inputs(dtype, steps)
    config = helion.Config(
        num_warps=16,
        cute_chained_mma_schedule="tcgen05_tmem",
        cute_chained_warp_mma_rows=32,
        cute_chained_preparation_pipeline=True,
        cute_chained_leaf_pipeline="rectangular_tma",
        cute_chained_leaf_count=count,
        cute_chained_preparation_cohorts=cohorts,
        cute_chained_preparation_unroll=1,
    )
    source, records = _code((_reused_raw, args), config)
    assert len(records) == 1
    plan, pipeline, transports, kwargs, storage = records[0]
    assert storage is not None and storage.stages == transports
    leaves = pipeline.prepared_leaves
    assert 2 <= len(leaves) <= count
    assert len({leaf.node for leaf in leaves}) == len(leaves)
    assert any(leaf.node.meta["val"].dtype == dtype for leaf in leaves)
    masked = [leaf for leaf in leaves if leaf.proof.mask is not None]
    assert masked
    assert all(
        leaf.proof.element_bytes == leaf.node.meta["val"].dtype.itemsize
        for leaf in leaves
    )
    assert pipeline.slots == (2 if cohorts == 1 else cohorts)
    assert kwargs["cohorts"] is pipeline.cohorts
    protocol = leaf_sets.LeafSetProtocol(pipeline.slots, len(leaves), cohorts > 1)
    assert pipeline.slot_barrier_count == protocol.barrier_count
    assert dict(storage.allocations)["slot_barriers"] == protocol.barrier_bytes
    assert storage.charged_bytes <= kwargs["capacity_bytes"]
    _safe(pipeline.frame)
    waits = _calls(source, "cute.arch.mbarrier_wait")
    for ordinal, leaf in enumerate(leaves):
        pointer = protocol.leaf_pointer(ordinal)
        matched = [call for call in waits if ast.unparse(call.args[0]) == pointer]
        assert len(matched) == 1
        assert ast.unparse(matched[0].args[1]) == protocol.phase
        action = next(
            action
            for action in pipeline.frame.actions
            if action.kind == "leaf" and action.nodes == (leaf.node,)
        )
        region = pipeline.frame.layout.region(leaf.name)
        assert action.event == region.live_from
        assert region.byte_size == leaf.byte_size
        assert all(
            leaf.name in pipeline.frame.actions[event].reads
            for event in leaf.read_events
        )
        assert all(
            region.live_from < event < region.live_until for event in leaf.read_events
        )
    # Origins/end stay host inputs and every scalar fallback remains present,
    # including the final masked tile and an empty dynamic source interval.
    assert "origin" in source and "end" in source
    assert "mbarrier_arrive_and_expect_tx" in source
    assert "cute.arch.mbarrier_arrive(" in source
