from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError
from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from ._cute_aux import _cpu_codegen
from .test_cute_chained_cache_set_search import _flat_scalar_index
from .test_cute_chained_collective_retention_search import (
    retention_bound as retention_bound,
)
from .test_cute_chained_loop_tmem_transport import _source as _kernel_source
from .test_cute_chained_operand_retention_integration import _kda_config
from .test_cute_chained_preparation_cut import _kda_fixture
from .test_cute_chained_preparation_cut import _runtime_values
from .test_cute_chained_preparation_frame import _capture
import helion
from helion import exc
from helion._compiler.cute.chained_output_lease import plan_output_lease_snapshot
from helion._compiler.cute.chained_pipeline_storage import capture_storage_revision
from helion._compiler.cute.chained_pipeline_storage import finalize_pipeline_storage
from helion._compiler.cute.chained_pipeline_storage import select_stage_transports
from helion._compiler.cute.chained_preparation_pipeline import plan_preparation_pipeline
from helion.autotuner.config_spec import CUTE_CHAINED_FRONTIER_STMATRIX_KEY
from helion.autotuner.config_spec import CUTE_CHAINED_OUTPUT_LEASE_SNAPSHOT_KEY as KEY
from helion.autotuner.config_spec import CUTE_CHAINED_SCAN_PRODUCER_RETENTION_KEY
import helion.language as hl


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _output_metadata_loop(a, b, initial, rows_out, valid):
    steps, size, _ = a.shape
    history = torch.empty((steps, size, size), device=a.device, dtype=torch.float32)
    final = torch.empty_like(initial)
    for rows in hl.tile(size, block_size=16):
        state = initial[rows, :]
        for step in hl.tile(steps, block_size=1):
            columns = hl.arange(size)
            kk = hl.arange(size)
            prepared = hl.dot(
                a[step.id, kk, columns],
                b[step.id, kk, columns],
                out_dtype=torch.float32,
            )
            operand = (prepared * 0.125).to(a.dtype)
            state = hl.dot(state.to(a.dtype), operand, out_dtype=torch.float32)
            destination = rows_out[step.id, rows]
            mask = valid[step.id, rows]
            hl.store(
                history,
                [step.id, destination, columns],
                state,
                extra_mask=mask[:, None],
            )
        final[rows, :] = state
    return history, final


def _args(dtype=torch.bfloat16, steps=3):
    size = 16
    return (
        torch.empty((steps, size, size), dtype=dtype),
        torch.empty((steps, size, size), dtype=dtype),
        torch.empty((size, size), dtype=torch.float32),
        torch.arange(size, dtype=torch.int32).expand(steps, size).contiguous(),
        torch.ones((steps, size), dtype=torch.bool),
    )


def _config(**overrides):
    return helion.Config.from_dict(
        {
            "num_warps": 8,
            "cute_chained_mma_schedule": "tcgen05_tmem",
            "cute_chained_warp_mma_rows": 32,
            "cute_chained_preparation_pipeline": True,
        }
        | overrides
    )


def _case(dtype=torch.bfloat16, steps=3):
    args, config = _args(dtype, steps), _config()
    plan, cut, shapes = _capture(_output_metadata_loop, args, config)
    pipeline = plan_preparation_pipeline(plan, cut, shapes, 232448)
    assert pipeline is not None
    plan = replace(
        plan,
        loop_workspace=pipeline.recurrence.layout,
        warp_mma_stages=frozenset(
            stage.group.stages[0] for stage in pipeline.frame.stages
        ),
    )
    revision = capture_storage_revision(plan, pipeline, shapes)
    stages = select_stage_transports(pipeline, ())
    assert revision is not None and stages is not None
    return plan, pipeline, stages, revision


def _source(dtype=torch.bfloat16, steps=3, **overrides):
    args = _args(dtype, steps)
    with _cpu_codegen():
        bound = _output_metadata_loop._bind_isolated(args)
        with bound.env.use_runtime_arg_values(
            _runtime_values(_output_metadata_loop, args)
        ):
            return bound.to_code(_config(**overrides))


def _select(case):
    plan, pipeline, stages, revision = case
    return plan_output_lease_snapshot(plan, pipeline, stages, revision=revision)


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("steps", (0, 1, 3))
def test_original_store_index_and_mask_are_complete_typed_images(dtype, steps):
    case = _case(dtype, steps)
    selected = _select(case)
    assert selected is not None
    assert case[0].loop is not None and case[0].region is not None
    assert selected.matches(*case)
    assert {image.buffer.dtype for image in selected.images} == {
        torch.bool,
        torch.int32,
    }
    assert len(selected.images) == 3 and selected.layout.allocated_bytes == 384
    assert {image.buffer.shape for image in selected.images} == {(16,), (16, 1)}
    assert selected.roots[0] is case[0].loop.region.stores[0]
    assert selected.final_group is case[2][-1].group
    assert selected.result_reads
    for image in selected.images:
        assert image.buffer.node in case[0].region.nodes
        assert image.source == case[1].frame.layout.region(image.buffer.name)
        assert (
            selected.layout.region(image.name).byte_size
            >= image.buffer.dtype.itemsize * 16
        )
    with pytest.raises(FrozenInstanceError):
        selected.images = ()  # pyrefly: ignore [read-only]


@pytest.mark.parametrize(
    "field",
    ("images", "roots", "result_reads", "carry_reads", "resident_carries", "layout"),
)
def test_derived_fields_cannot_retain_authority_after_replacement(field):
    case = _case()
    selected = _select(case)
    assert selected is not None
    changes = {
        "images": replace(selected, images=()),
        "roots": replace(selected, roots=()),
        "result_reads": replace(selected, result_reads=()),
        "carry_reads": replace(selected, carry_reads=(case[0].dots[0],)),
        "resident_carries": replace(selected, resident_carries=frozenset((99,))),
        "layout": replace(
            selected, layout=replace(selected.layout, allocated_bytes=128)
        ),
    }
    assert not changes[field].matches(*case)


@pytest.mark.parametrize(
    "mutation", ("plan", "pipeline", "dtype", "kwargs", "mask", "final_group")
)
def test_stale_graph_frame_or_stage_selection_rejects(mutation):
    plan, pipeline, stages, revision = _case()
    selected = _select((plan, pipeline, stages, revision))
    assert selected is not None
    assert plan.loop is not None
    if mutation == "plan":
        plan = replace(plan)
    elif mutation == "pipeline":
        pipeline = replace(pipeline)
    elif mutation == "dtype":
        node = selected.images[0].buffer.node
        assert node is not None
        node.meta["val"] = node.meta["val"].to(torch.float32)
    elif mutation == "kwargs":
        plan.loop.region.stores[0].kwargs = {"unknown": True}
    elif mutation == "mask":
        store = plan.loop.region.stores[0]
        store.args = (*store.args[:3], None)
    else:
        stages = stages[:-1]
    assert not selected.matches(plan, pipeline, stages, revision)
    assert plan_output_lease_snapshot(plan, pipeline, stages, revision=revision) is None


def test_actual_kda_original_metadata_nodes_are_found_without_name_dispatch():
    kernel, args = _kda_fixture()
    config = helion.Config(
        block_sizes=[128],
        num_warps=16,
        cute_chained_mma_schedule="tcgen05_tmem",
        cute_chained_group_contractions=True,
        cute_chained_warp_mma_rows=32,
        cute_chained_scratch_layout="xor",
        cute_chained_pointwise_vectorize=True,
        cute_chained_scan_schedule="warp",
        cute_chained_pointwise_cache_bytes=4096,
        cute_chained_pointwise_unroll=8,
        cute_chained_preparation_pipeline=True,
    )
    plan, cut, shapes = _capture(kernel, args, config)
    pipeline = plan_preparation_pipeline(plan, cut, shapes, 232448)
    assert pipeline is not None
    plan = replace(
        plan,
        loop_workspace=pipeline.recurrence.layout,
        warp_mma_stages=frozenset(
            stage.group.stages[0] for stage in pipeline.frame.stages
        ),
    )
    revision = capture_storage_revision(plan, pipeline, shapes)
    stages = select_stage_transports(pipeline, ())
    assert revision is not None and stages is not None
    selected = plan_output_lease_snapshot(plan, pipeline, stages, revision=revision)
    assert selected is not None
    assert [(image.buffer.dtype, image.buffer.shape) for image in selected.images] == [
        (torch.int32, (32,)),
        (torch.bool, (32, 1)),
    ]
    assert selected.layout.allocated_bytes == 256
    assert all(image.buffer.node in cut.region.nodes for image in selected.images)


def test_snapshot_charge_is_independent_and_capacity_cannot_discount_frame():
    plan, pipeline, stages, revision = _case()
    proof = _select((plan, pipeline, stages, revision))
    assert proof is not None
    ordinary = finalize_pipeline_storage(
        plan, pipeline, stages, revision=revision, capacity_bytes=232448
    )
    selected = finalize_pipeline_storage(
        plan,
        pipeline,
        stages,
        revision=revision,
        capacity_bytes=232448,
        output_lease=proof,
    )
    assert ordinary is not None and selected is not None
    assert selected.output_lease is proof and ordinary.output_lease is None
    assert selected.recurrence == ordinary.recurrence
    assert selected.carry_views == ordinary.carry_views
    assert selected.allocations[:-1] == ordinary.allocations
    assert selected.allocations[-1] == ("output_snapshot", 384)
    assert selected.charged_bytes == ordinary.charged_bytes + 384
    assert (
        finalize_pipeline_storage(
            plan,
            pipeline,
            stages,
            revision=revision,
            capacity_bytes=selected.charged_bytes - 1,
            output_lease=proof,
        )
        is None
    )
    assert (
        finalize_pipeline_storage(
            plan,
            pipeline,
            stages,
            revision=revision,
            capacity_bytes=232448,
            output_lease=replace(proof, images=()),
        )
        is None
    )


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("steps", (0, 1, 3))
@pytest.mark.parametrize("cohorts", (1, 3))
def test_public_enabled_copies_after_completion_before_publication(
    dtype, steps, cohorts
):
    source = _source(
        dtype,
        steps,
        cute_chained_output_lease_snapshot=True,
        num_warps=16,
        cute_chained_preparation_cohorts=cohorts,
    )
    assert (
        "chain_output_snapshots = cute.arch.alloc_smem(cutlass.Uint8, 384, alignment=128)"
        in source
    )
    wait = source.index("cute.arch.mbarrier_wait(chain_bars + 1")
    copy = source.index("chain_output_snapshot_0_index =")
    slots = 2 if cohorts == 1 else cohorts
    release = f"chain_sync.arrive_mbarrier(chain_slot_bars + {slots} + chain_slot)"
    empty = source.index(release)
    extraction = source.index("chain_1_copy =")
    assert wait < copy < empty < extraction
    assert source.count(release) == 1
    assert source.index("chain_recurrence_barrier.arrive_and_wait()", copy) < empty
    assert source.count("cute.arch.sync_threads()") >= 2


def test_absent_false_does_not_discover_or_allocate():
    from helion._compiler.cute import chained_output_lease

    with patch.object(
        chained_output_lease,
        "plan_output_lease_snapshot",
        side_effect=AssertionError("inactive discovery"),
    ):
        ordinary = _source()
        disabled = _source(cute_chained_output_lease_snapshot=False)
    assert ordinary == disabled
    assert "chain_output_snapshot" not in ordinary


@pytest.mark.parametrize("value", (None, 0, 1, -1, 0.0, "true", [], {}))
@pytest.mark.parametrize("repair", (False, True))
def test_snapshot_schema_is_strict_bool(retention_bound, value, repair):
    with pytest.raises(exc.InvalidConfig, match=KEY):
        retention_bound.config_spec.normalize(
            _config(**{KEY: value}), _fix_invalid=repair
        )


def test_snapshot_schema_appends_exact_old_default_and_seed_prefix(retention_bound):
    spec = retention_bound.config_spec
    assert spec.supports_config_key(KEY)
    fields = spec._flat_fields()
    assert tuple(fields)[-14:] == (
        KEY,
        CUTE_CHAINED_SCAN_PRODUCER_RETENTION_KEY,
        CUTE_CHAINED_FRONTIER_STMATRIX_KEY,
        "cute_chained_snapshot_tile_columns",
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
    index = _flat_scalar_index(spec, KEY)
    generation = spec.create_config_generation()
    default = spec.default_config()
    assert KEY not in default.config
    absent = spec.normalized_config(_config())
    disabled = spec.normalized_config(_config(**{KEY: False}))
    enabled = spec.normalized_config(_config(**{KEY: True}))
    assert absent == disabled
    assert enabled.config == absent.config | {KEY: True}
    assert generation.unflatten(generation.flatten(enabled)) == enabled
    assert generation.flatten(enabled)[index] is True
    pairs = generation.seed_flat_config_pairs()
    with patch.object(
        spec,
        "_flat_fields",
        return_value={key: value for key, value in fields.items() if key != KEY},
    ):
        original = spec.create_config_generation()
        flat = generation.flatten(default)
        assert flat[:index] + flat[index + 1 :] == original.flatten(default)
        for (new_flat, new_config), (old_flat, old_config) in zip(
            pairs, original.seed_flat_config_pairs(), strict=True
        ):
            assert new_flat[:index] + new_flat[index + 1 :] == old_flat
            assert new_flat[index] is False
            assert new_config == old_config


@pytest.mark.parametrize(
    "key", ("cute_chained_preparation_pipeline", "cute_chained_mma_schedule")
)
@pytest.mark.parametrize("repair", (False, True))
def test_snapshot_does_not_invent_explicit_prerequisites(retention_bound, key, repair):
    config = _config(**{KEY: True})
    config.config.pop(key)
    with pytest.raises(exc.InvalidConfig):
        retention_bound.config_spec.normalize(config, _fix_invalid=repair)


@pytest.mark.parametrize("backend", ("triton", "pallas"))
def test_other_backends_strip_only_default(retention_bound, backend):
    spec = retention_bound.config_spec
    with patch.object(spec, "backend_name", backend):
        config = {KEY: False}
        spec.normalize(config)
        assert KEY not in config
        with pytest.raises(exc.InvalidConfig):
            spec.normalize(_config(**{KEY: True}))


def test_completion_ledger_rejects_missing_duplicate_and_invalid_transition():
    from helion._compiler.cute import chained_output_lease as lease
    from helion._compiler.cute.chained_matmul import _UnsupportedChain

    original = lease.bind_output_lease_snapshot
    observed = []

    def bind(plan, pipeline, storage, boundaries, execution):
        bound = original(plan, pipeline, storage, boundaries, execution)
        assert bound is not None
        assert bound._completion.receipt is None
        with pytest.raises(_UnsupportedChain):
            bound.epilogue_boundaries(plan, boundaries)
        with pytest.raises(_UnsupportedChain):
            bound.post_issue_lines(
                plan,
                replace(storage.stages[-1], tmem_accumulator="unproved"),
                execution,
                boundaries,
            )
        assert bound._completion.receipt is None
        bound._completion.receipt = object()
        assert not bound.matches(plan)
        with pytest.raises(_UnsupportedChain):
            bound.epilogue_boundaries(plan, boundaries)
        bound._completion.receipt = None
        observed.append((bound, boundaries))
        return bound

    with patch.object(lease, "bind_output_lease_snapshot", bind):
        _source(cute_chained_output_lease_snapshot=True)
    assert len(observed) == 1
    bound, boundaries = observed[0]
    assert bound.matches(bound.plan)
    assert bound._completion.receipt is bound._issued_receipt
    with pytest.raises(_UnsupportedChain):
        bound.post_issue_lines(
            bound.plan, bound.storage.stages[-1], bound.execution, boundaries
        )
    assert not replace(bound, copies=()).matches(bound.plan)
    epilogue = bound.epilogue_boundaries(bound.plan, boundaries)
    assert epilogue is not boundaries
    for node, name in boundaries.items():
        if node not in dict(bound.sources):
            assert epilogue[node] == name


@pytest.mark.parametrize("leaves", (1, 4))
def test_actual_kda_native_frame_complete_source_inverse(leaves):
    from helion._compiler.cute import chained_output_lease as lease

    kernel, args = _kda_fixture()
    config = _kda_config(leaves=leaves)
    before = _kernel_source(kernel, args, config)
    config.config[KEY] = True
    captured = []
    original = lease.bind_output_lease_snapshot

    def bind(*args, **kwargs):
        result = original(*args, **kwargs)
        assert result is not None
        captured.append(result)
        return result

    with patch.object(lease, "bind_output_lease_snapshot", bind):
        after = _kernel_source(kernel, args, config)
    assert len(captured) == 1
    bound = captured[0]
    assert [
        (image.buffer.dtype, image.buffer.shape) for image in bound.proof.images
    ] == [(torch.int32, (32,)), (torch.bool, (32, 1))]
    assert bound.proof.layout.allocated_bytes == 256
    names = {image.name: image.buffer.name for image in bound.proof.images}
    release = f"chain_sync.arrive_mbarrier(chain_slot_bars + {bound.pipeline.slots} + chain_slot)"

    class Restore(ast.NodeTransformer):
        def visit_Name(self, node):
            if node.id in names:
                node.id = names[node.id]
            return node

        def visit_Assign(self, node):
            if any(
                isinstance(target, ast.Name)
                and target.id.startswith("chain_output_snapshot")
                for target in node.targets
            ):
                return None
            return self.generic_visit(node)

        def visit_For(self, node):
            if isinstance(node.target, ast.Name) and node.target.id.startswith(
                "chain_output_snapshot"
            ):
                return None
            releases = [
                item
                for item in node.body
                if isinstance(item, ast.If)
                and any(
                    isinstance(stmt, ast.Expr) and ast.unparse(stmt.value) == release
                    for stmt in item.body
                )
            ]
            node = self.generic_visit(node)
            assert isinstance(node, ast.For)
            if releases:
                assert len(releases) == 1
                position = node.body.index(releases[0])
                assert ast.unparse(node.body[position - 1]) == bound.execution.sync
                node.body.pop(position - 1)
                node.body.remove(releases[0])
                node.body.append(releases[0])
            return node

    restored = Restore().visit(ast.parse(after))
    assert ast.dump(restored) == ast.dump(ast.parse(before))
