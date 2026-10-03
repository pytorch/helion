from __future__ import annotations

import ast
from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from .test_cute_chained_collective_retention_search import (
    retention_bound as retention_bound,
)
from .test_cute_chained_loop_tmem_transport import _source
from .test_cute_chained_operand_retention_integration import _kda_config
from .test_cute_chained_preparation_cut import _kda_fixture
import helion
from helion import exc
from helion._compiler.cute import chained_preparation_pipeline as pipeline_module
from helion._compiler.cute import chained_scan_producer_emission as emission_module
from helion._compiler.cute.chained_collectives import _warp_prefix
from helion._compiler.cute.chained_collectives import warp_prefix_point
from helion._compiler.cute.chained_execution import ChainedExecution
from helion._compiler.cute.chained_matmul import _UnsupportedChain
from helion._compiler.cute.chained_pointwise_unroll import BoundedProducerUnroll
from helion._compiler.cute.chained_scan_producer import plan_scan_producer
from helion.autotuner.config_spec import CUTE_CHAINED_FRONTIER_STMATRIX_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_SCAN_PRODUCER_RETENTION_KEY as KEY
from helion.autotuner.config_spec import CUTE_CHAINED_SNAPSHOT_TILE_COLUMNS_KEY
import helion.language as hl


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _scan_sequence(x, y, z, initial, selected: hl.constexpr, valid_steps: hl.constexpr):
    _, size, width = x.shape
    steps: int = valid_steps  # pyrefly: ignore [bad-assignment]
    selected_value: int = selected  # pyrefly: ignore [bad-assignment]
    history = torch.empty((steps, size, width), dtype=torch.float32, device=x.device)
    final = torch.empty_like(initial)
    for _rows in hl.tile(size, block_size=32):
        initial_rows = hl.arange(32)
        initial_columns = hl.arange(width)
        state = initial[initial_rows, initial_columns]
        for step in hl.tile(steps, block_size=1):
            lanes = hl.arange(32)
            other = hl.arange(32)
            columns = hl.arange(width)
            raw = x[step.id, lanes, columns].float()
            prefix = hl.cumsum(raw * 0.125, dim=0)
            endpoint = torch.sum(
                torch.where((lanes == selected_value)[:, None], prefix, 0.0), dim=0
            )
            weights = y[step.id, lanes, columns].float()
            energy = torch.sum(weights * weights, dim=1)
            left = (torch.sigmoid(prefix) + energy[:, None] + raw * 0.0625).to(x.dtype)
            right = (energy[:, None] + prefix * 0.25).to(x.dtype)
            prepared = hl.dot(left, right.T, out_dtype=torch.float32)
            rhs = (prepared * 0.125 + z[step.id, lanes, other].float()).to(x.dtype)
            state = (
                hl.dot(rhs, state.to(x.dtype), acc=state, out_dtype=torch.float32)
                + endpoint[None, :]
            )
            history[step.id, lanes, columns] = state
        final[initial_rows, initial_columns] = state
    return history, final


def _sequence_args(dtype=torch.bfloat16, steps=1, selected=17):
    return (
        *(torch.empty((steps, 32, 64), dtype=dtype) for _ in range(2)),
        torch.empty((steps, 32, 32), dtype=dtype),
        torch.empty((32, 64), dtype=torch.float32),
        selected,
        steps,
    )


def _sequence_config(enabled=True):
    return helion.Config(
        num_warps=8,
        cute_chained_mma_schedule="tcgen05_tmem",
        cute_chained_preparation_pipeline=True,
        cute_chained_warp_mma_rows=32,
        cute_chained_scan_schedule="warp",
        cute_chained_pointwise_vectorize=True,
        cute_chained_leaf_pipeline="rectangular_tma",
        cute_chained_leaf_count=2,
        cute_chained_scan_producer_retention=enabled,
    )


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
def test_unrelated_nonlinear_sequence_uses_original_typed_scan(dtype):
    source = _source(_scan_sequence, _sequence_args(dtype), _sequence_config())
    assert "chain_scan_producer_" in source
    assert "shuffle_sync_up" in source
    assert "cutlass.Float32(0) +" in source


def _config(enabled=True):
    config = _kda_config()
    config.config[KEY] = enabled
    return config


def test_actual_kda_effective_leases_and_complete_source():
    kernel, args = _kda_fixture()
    captured = []
    original = pipeline_module._emit_scan_producer_stage

    def observe(cg, plan, pipeline, boundaries, execution, unroll, vector):
        candidate = pipeline.scan_producer
        assert candidate is not None
        assert candidate.revision.frame is pipeline.frame
        assert candidate.matches(
            candidate.revision.plan, pipeline, dict(candidate.revision.shapes)
        )
        assert candidate.residual_rows == (31,)
        assert len(candidate.deferred) == 1
        assert [action.kind for action in candidate.prelude] == [
            "leaf",
            "collective",
            "leaf",
            "collective",
        ]
        assert len(candidate.overrides) == 2
        for index, left in enumerate(candidate.regions):
            for right in candidate.regions[index + 1 :]:
                assert not (
                    left.overlaps_storage(right) and left.overlaps_lifetime(right)
                )
        assert all(
            region.byte_end <= pipeline.frame.layout.allocated_bytes
            for region in candidate.regions
        )
        for field, value in (
            ("residual_rows", (0,)),
            ("overrides", ()),
            ("regions", ()),
        ):
            assert not replace(candidate, **{field: value}).matches(
                candidate.revision.plan, pipeline, dict(candidate.revision.shapes)
            )
        changed_frame = replace(
            pipeline.frame,
            frontier_order=tuple(reversed(pipeline.frame.frontier_order)),
        )
        assert not candidate.matches(
            candidate.revision.plan,
            replace(pipeline, frame=changed_frame),
            dict(candidate.revision.shapes),
        )
        binding = pipeline.prepared_groups[0]
        reserved = replace(binding, candidate=replace(binding.candidate, live_from=0))
        earlier = replace(
            pipeline, prepared_groups=(reserved, *pipeline.prepared_groups[1:])
        )
        assert (
            plan_scan_producer(
                candidate.revision.plan,
                earlier,
                dict(candidate.revision.shapes),
                scratch_mode="xor",
            )
            is None
        )
        captured.append(candidate)
        return original(cg, plan, pipeline, boundaries, execution, unroll, vector)

    with patch.object(pipeline_module, "_emit_scan_producer_stage", observe):
        source = _source(kernel, args, _config())
    assert len(captured) == 1
    candidate = captured[0]
    assert f"chain_scan_producer_{candidate.first_event}_output_2" in source
    assert "cute.make_layout((32, 4), stride=(1, 32))" in source
    assert f"{candidate.buffer.name}_vector_step" not in source
    assert "chain_sync.arrive_mbarrier" in source
    assert "chain_retained_operand_0" in source
    ast.parse(source)


def test_failed_late_probe_keeps_boundaries_and_activation_unchanged():
    kernel, args = _kda_fixture()
    original = emission_module.emit_scan_producer
    seen = []

    def observe(
        cg, plan, candidate, boundaries, operands, *, execution, producer_unroll
    ):
        saved = dict(boundaries)
        tracker = BoundedProducerUnroll(4)
        with patch.object(
            emission_module,
            "emit_collectives_before",
            side_effect=_UnsupportedChain("late negative"),
        ):
            assert (
                original(
                    cg,
                    plan,
                    candidate,
                    boundaries,
                    operands,
                    execution=execution,
                    producer_unroll=tracker,
                )
                is None
            )
        assert boundaries == saved and not tracker.activated
        seen.append(True)
        return original(
            cg,
            plan,
            candidate,
            boundaries,
            operands,
            execution=execution,
            producer_unroll=producer_unroll,
        )

    with patch.object(emission_module, "emit_scan_producer", observe):
        _source(kernel, args, _config())
    assert seen == [True]


def test_output_snapshot_and_scan_use_one_final_storage_revision():
    from helion._compiler.cute import chained_pipeline_storage as storage

    kernel, args = _kda_fixture()
    config = _config()
    config.config["cute_chained_output_lease_snapshot"] = True
    original = storage.finalize_pipeline_storage
    seen = []

    def observe(plan, pipeline, stages, **kwargs):
        assert pipeline.scan_producer is not None
        assert kwargs["output_lease"] is not None
        assert kwargs["revision"].pipeline is pipeline
        result = original(plan, pipeline, stages, **kwargs)
        assert result is not None
        assert result.output_lease is kwargs["output_lease"]
        assert dict(result.allocations)["output_snapshot"] > 0
        seen.append(True)
        return result

    with patch.object(storage, "finalize_pipeline_storage", observe):
        source = _source(kernel, args, config)
    assert seen == [True]
    assert "chain_scan_producer_" in source and "chain_output_snapshot" in source


def test_default_false_never_discovers_a_scan_macro():
    kernel, args = _kda_fixture()
    config = _config(False)
    with patch(
        "helion._compiler.cute.chained_scan_producer.plan_scan_producer",
        side_effect=AssertionError("default discovery"),
    ):
        explicit = _source(kernel, args, config)
        config.config.pop(KEY)
        absent = _source(kernel, args, config)
    assert explicit == absent


@pytest.mark.parametrize("extent", (1, 16, 32, 64, 128))
def test_factored_prefix_preserves_original_order(extent):
    execution = ChainedExecution(128)
    point = warp_prefix_point(
        "prefix", extent, ["value = cutlass.Float32(7)"], "value", execution=execution
    )
    whole = _warp_prefix(
        "prefix",
        extent,
        17,
        128,
        ("prefix_position", "prefix_vector"),
        ["value = cutlass.Float32(7)"],
        "value",
        execution=execution,
    )
    assert point[1] == "prefix_acc = cutlass.Float32(0)"
    assert point[4] == "    prefix_acc = cutlass.Float32(0) + value"
    offsets = [
        int(line.split("offset=")[1].split(")")[0])
        for line in point
        if "shuffle_sync_up" in line
    ]
    assert offsets == [1, 2, 4, 8, 16]
    normalized = "\n".join(whole)
    for line in point:
        assert line.strip() in normalized
    assert ("prefix_carry" in "\n".join(point)) == (extent > 32)


@pytest.mark.parametrize("value", (0, 1, None, "true", (), []))
def test_strict_bool_schema(retention_bound, value):
    config = _config()
    config.config[KEY] = value
    with pytest.raises(exc.InvalidConfig, match="must be bool"):
        retention_bound.config_spec.normalized_config(config)


def test_schema_default_flat_and_prerequisites(retention_bound):
    spec = retention_bound.config_spec
    assert tuple(spec._flat_fields())[-13:] == (
        KEY,
        CUTE_CHAINED_FRONTIER_STMATRIX_KEY,
        CUTE_CHAINED_SNAPSHOT_TILE_COLUMNS_KEY,
        "cute_min_blocks_per_mp",
        "cute_chained_compact_preparation",
        "cute_chained_leaf_issue_batching",
        "cute_chained_broadcast_retention",
        "cute_chained_completed_member_store",
        "cute_native_matmul_metadata",
        "cute_chained_drain_tile_columns",
        "cute_chained_island_consumers",
        "cute_chained_async_vector_store",
        "cute_chained_fragment_epilogues",
    )
    assert spec.flatten_missing_field_default(KEY, {}) == (True, False)
    config = helion.Config(
        num_warps=8,
        cute_chained_mma_schedule="tcgen05_tmem",
        cute_chained_preparation_pipeline=True,
        cute_chained_warp_mma_rows=32,
        cute_chained_scan_schedule="warp",
        cute_chained_pointwise_vectorize=True,
        cute_chained_scan_producer_retention=True,
    )
    with pytest.raises(exc.InvalidConfig, match="eligible general contraction scan"):
        spec.normalized_config(config)
    # Isolate normalization/flat representation; actual discovered scan
    # eligibility is covered by the complete KDA source test above.
    with patch.object(spec, "cute_chained_scan_search_enabled", True):
        normalized = spec.normalized_config(config)
        generation = spec.create_config_generation()
        assert generation.unflatten(generation.flatten(normalized)) == normalized
        for key in (
            "cute_chained_preparation_pipeline",
            "cute_chained_pointwise_vectorize",
            "cute_chained_scan_schedule",
        ):
            changed = helion.Config.from_dict(dict(config.config))
            changed.config.pop(key)
            with pytest.raises(exc.InvalidConfig):
                spec.normalized_config(changed)
