from __future__ import annotations

from dataclasses import fields
from itertools import starmap
from unittest.mock import patch

import pytest
import torch

from ._cute_aux import _cpu_codegen
from .test_cute_chained_loop_tmem_transport import _source as _loop_source
from .test_cute_chained_operand_retention_integration import _kda_config
from .test_cute_chained_preparation_cut import _kda_fixture
from .test_cute_chained_preparation_cut import _runtime_values
import helion
from helion._compiler.cute import chained_completed_store as completed
from helion._compiler.cute import chained_preparation_pipeline as pipeline
from helion._compiler.cute.chained_completed_members import plan_completed_member_map
from helion._compiler.cute.chained_execution import ChainedExecution
from helion._compiler.cute.chained_matmul import _UnsupportedChain
import helion.language as hl


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _completed_sequence(
    a, b, c, d, initial, indices, mask, output_dtype: hl.constexpr, masked: hl.constexpr
):
    steps = a.shape[0]
    dtype: torch.dtype = output_dtype  # pyrefly: ignore [bad-assignment]
    history = torch.empty((steps, 32, 128), device=a.device, dtype=dtype)
    final = torch.empty_like(initial)
    for rr in hl.tile(128, block_size=128):
        state = initial[rr, :]
        for step in hl.tile(steps, block_size=1):
            rows, kk, columns = hl.arange(32), hl.arange(16), hl.arange(16)
            prepared = hl.dot(
                (a[step.id, rows, kk] + c[step.id, rows, kk]).to(a.dtype),
                b[step.id, kk, columns],
                out_dtype=torch.float32,
            ).to(a.dtype)
            snapshot = state.to(a.dtype)
            result = hl.dot(prepared, snapshot.T, out_dtype=torch.float32)
            state = hl.dot(
                snapshot, d[step.id, kk, columns], acc=state, out_dtype=torch.float32
            )
            target = indices[step.id, rows]
            valid = mask[step.id, rows]
            if masked:
                hl.store(
                    history,
                    [step.id, target, rr],
                    torch.tanh(result + 0.125).to(dtype),
                    extra_mask=valid[:, None],
                )
            else:
                history[step.id, target, rr] = torch.tanh(result + 0.125).to(dtype)
        final[rr, :] = state
    return history, final


def _args(dtype, output_dtype, steps=3, masked=True):
    return (
        torch.empty((steps, 32, 16), dtype=dtype),
        torch.empty((steps, 16, 16), dtype=dtype),
        torch.empty((steps, 32, 16), dtype=dtype),
        torch.empty((steps, 16, 16), dtype=dtype),
        torch.empty((128, 16), dtype=torch.float32),
        torch.arange(32, dtype=torch.int32).expand(steps, 32).contiguous(),
        torch.ones((steps, 32), dtype=torch.bool),
        output_dtype,
        masked,
    )


def _source(
    dtype=torch.bfloat16,
    output_dtype=torch.bfloat16,
    *,
    enabled=False,
    steps=3,
    masked=True,
    observer=None,
    output_lease=False,
):
    args = _args(dtype, output_dtype, steps, masked)
    config = helion.Config(
        num_warps=8,
        cute_chained_mma_schedule="tcgen05_tmem",
        cute_chained_group_contractions=True,
        cute_chained_warp_mma_rows=32,
        cute_chained_preparation_pipeline=True,
    )
    if output_lease:
        config.config["cute_chained_output_lease_snapshot"] = True
    original = pipeline.emit_preparation_pipeline
    prepare = completed.prepare_completed_store
    observed = []

    def emit(*values, **kwargs):
        return original(*values, **kwargs, completed_member_store=enabled)

    def record(*values, **kwargs):
        result = prepare(*values, **kwargs)
        if result is not None:
            observed.append(result)
            if observer is not None:
                observer(result)
        return result

    with (
        _cpu_codegen(),
        patch.object(pipeline, "emit_preparation_pipeline", emit),
        patch.object(completed, "prepare_completed_store", record),
    ):
        bound = _completed_sequence._bind_isolated(args)
        with bound.env.use_runtime_arg_values(
            _runtime_values(_completed_sequence, args)
        ):
            source = bound.to_code(config)
    return source, observed


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("output_dtype", (torch.bfloat16, torch.float16, torch.float32))
def test_original_store_consumes_completed_member_once(dtype, output_dtype):
    source, actions = _source(dtype, output_dtype, enabled=True)
    assert len(actions) == 1
    action = actions[0]
    action.validate_consumed()
    assert f"for {action.prefix}_register" in source
    assert action.c_name not in dict(action.inputs).values()
    assert f"{action.c_name}[" not in source
    assert "fence_view_async_tmem_load" in source
    assert "chain_recurrence_barrier.arrive_and_wait()" in source


def test_disabled_path_never_discovers_completed_store():
    with patch.object(
        completed, "prepare_completed_store", side_effect=AssertionError("disabled")
    ):
        source, actions = _source()
    assert not actions
    assert "chain_store_0_register" not in source


def test_explicit_wrong_type_rejects_before_stage():
    with pytest.raises(
        Exception, match="completed member store must be an explicit boolean"
    ):
        _source(enabled=1)


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize(
    "full,offset,width", ((144, 16, 48), (160, 0, 32), (256, 128, 64))
)
def test_actual_tmem_a_c_map_equals_original_complete_load(dtype, full, offset, width):
    with patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA forbidden")):
        smem = plan_completed_member_map(
            (128, full), (width, 128), offset, width, True, dtype
        )
        tmem = plan_completed_member_map(
            (128, full), (width, 128), offset, width, True, dtype, operand_source="TMEM"
        )
    assert smem is not None and tmem is not None
    # Every full-group source word and ordered destination slot, not coverage only.
    assert smem == tmem
    assert tmem.same_store_order


@pytest.mark.parametrize("source", (False, 1, None, "tmem", "unknown"))
def test_unknown_operand_source_does_not_gain_layout_authority(source):
    assert (
        plan_completed_member_map(
            (128, 160), (32, 128), 0, 32, True, torch.bfloat16, operand_source=source
        )
        is None
    )


@pytest.mark.parametrize(
    "change",
    (
        "latch",
        "candidate",
        "stage",
        "region",
        "allocation",
        "native",
        "kwargs",
        "dtype",
        "alias",
    ),
)
def test_same_object_change_before_stage_rejects(change):
    actions = []

    def mutate(action):
        actions.append(action)
        if change == "latch":
            action._progress.token = action._tokens[4]
        elif change == "candidate":
            object.__setattr__(action.candidate, "descendants", ())
        elif change == "stage":
            object.__setattr__(
                action.storage.stages[-1], "tmem_accumulator", "different_tmem"
            )
        elif change == "region":
            contract = action.storage.completed_store
            region = (
                action.storage.recurrence.layout.region(action.c_name)
                if contract is None
                else contract.request
            )
            object.__setattr__(region, "byte_offset", region.byte_offset + 128)
        elif change == "allocation":
            object.__setattr__(action.storage, "allocations", ())
        elif change == "native":
            object.__setattr__(action.candidate.native, "selected_slots", ())
        elif change == "kwargs":
            action.candidate.store.kwargs = {"unsupported_effect": True}
        elif change == "alias":
            name = next(iter(action.plan.tensor_aliases))
            action.plan.tensor_aliases[name] += "_changed"
        else:
            action.candidate.node.meta["val"] = action.candidate.node.meta["val"].to(
                torch.float16
            )

    with pytest.raises(Exception, match="completed store stage binding changed"):
        _source(enabled=True, observer=mutate)
    assert len(actions) == 1
    assert actions[0]._progress.prepared is None


@pytest.mark.parametrize("field", [field.name for field in fields(ChainedExecution)])
@pytest.mark.parametrize("when", ("before_stage", "after_completion", "before_consume"))
def test_same_object_execution_mutation_rejects_at_every_boundary(field, when):
    complete = completed.CompletedStoreAction.complete_stage
    consume = completed.CompletedStoreAction.consume
    changed = []

    def mutate(action):
        assert action.matches(action.plan)
        value = 256 if field == "threads" else f"changed_{field}"
        assert getattr(action.execution, field) != value
        object.__setattr__(action.execution, field, value)
        assert not action.matches(action.plan)
        changed.append(action)
        if when == "before_stage":
            action.begin_stage(
                action.plan,
                action.storage.stages[-1],
                action.execution,
                dict(action.inputs),
                action.phase,
            )

    def complete_stage(action, plan, boundaries, lines):
        result = complete(action, plan, boundaries, lines)
        if when == "after_completion":
            mutate(action)
        return result

    def consume_store(action, *args):
        if when == "before_consume":
            mutate(action)
        return consume(action, *args)

    with (
        patch.object(completed.CompletedStoreAction, "complete_stage", complete_stage),
        patch.object(completed.CompletedStoreAction, "consume", consume_store),
        pytest.raises(Exception, match="completed store .* changed"),
    ):
        _source(enabled=True, observer=mutate if when == "before_stage" else None)
    assert len(changed) == 1


@pytest.mark.parametrize(
    "change", ("missing", "body", "duplicate", "early", "drop_point")
)
def test_missing_changed_or_duplicate_completion_cannot_consume(change):
    original = completed.CompletedStoreAction.consume
    begin = completed.CompletedStoreAction.begin_stage
    complete = completed.CompletedStoreAction.complete_stage

    def consume(action, plan, store, boundaries, execution, prefix, stage_lines):
        if change == "body":
            stage_lines = stage_lines[:-1]
        if change == "drop_point":
            action._progress.prepared = None
        result = original(
            action, plan, store, boundaries, execution, prefix, stage_lines
        )
        if change == "duplicate":
            return original(
                action, plan, store, boundaries, execution, prefix, stage_lines
            )
        return result

    def begin_stage(action, plan, selection, execution, boundaries, phase):
        if change == "early":
            original(
                action,
                plan,
                action.candidate.store,
                boundaries,
                execution,
                action.prefix,
                (),
            )
        return begin(action, plan, selection, execution, boundaries, phase)

    def complete_stage(action, plan, boundaries, lines):
        if change != "missing":
            return complete(action, plan, boundaries, lines)
        return None

    with (
        patch.object(completed.CompletedStoreAction, "consume", consume),
        patch.object(completed.CompletedStoreAction, "begin_stage", begin_stage),
        patch.object(completed.CompletedStoreAction, "complete_stage", complete_stage),
        pytest.raises(Exception, match="completed store consumption changed"),
    ):
        _source(enabled=True)


def test_failed_point_lowering_does_not_publish_or_complete():
    actions = []
    with (
        patch.object(
            completed,
            "lower_store_point",
            side_effect=_UnsupportedChain("probe failed"),
        ),
        pytest.raises(Exception, match="probe failed"),
    ):
        _source(enabled=True, observer=actions.append)
    # Point preparation now precedes charging: no stage action can even begin.
    assert not actions


@pytest.mark.parametrize("steps,masked", ((0, True), (1, False), (3, True)))
def test_zero_and_ragged_iteration_source_keeps_original_protocol(steps, masked):
    ordinary, _ = _source(steps=steps, masked=masked)
    source, actions = _source(steps=steps, masked=masked, enabled=True)
    assert len(actions) == 1
    # The completed store makes the carry resident in its output slot. Advancing
    # it has no staged stores, so only the existing final role join is needed.
    assert "chain_loop_carry_0_next" in ordinary
    assert "chain_loop_carry_0_next" not in source
    join = "chain_recurrence_barrier.arrive_and_wait()"
    assert source.count(join) + 1 == ordinary.count(join)
    for operation in (
        "cute.arch.mbarrier_wait(",
        "cute.arch.fence_view_async_tmem_load()",
        "chain_sync.arrive_mbarrier(",
        "cute.arch.sync_threads()",
    ):
        assert source.count(operation) == ordinary.count(operation)
    assert source.count("chain_final_store_0") == ordinary.count("chain_final_store_0")


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
def test_existing_detached_metadata_authority_is_preserved(dtype):
    ordinary, _ = _source(dtype, output_lease=True)
    source, actions = _source(dtype, enabled=True, output_lease=True)
    (action,) = actions
    assert action.output_lease is not None
    assert action._progress.prepared is not None
    assert all(
        name.startswith("chain_output_snapshot")
        for _, name in action._progress.prepared.reads
    )
    for operation in (
        "chain_sync.arrive_mbarrier(",
        "chain_recurrence_barrier.arrive_and_wait()",
    ):
        assert ordinary.count(operation) == source.count(operation)


@pytest.mark.parametrize("change", ("dtype", "shape", "alias", "completion"))
def test_detached_metadata_proof_mutation_rejects(change):
    def mutate(action):
        lease = action.output_lease
        assert lease is not None
        if change == "dtype":
            object.__setattr__(lease.proof.images[0].buffer, "dtype", torch.float64)
        elif change == "shape":
            object.__setattr__(lease.proof.images[0].buffer, "shape", (1,))
        elif change == "alias":
            name = next(iter(action.plan.tensor_aliases))
            object.__setattr__(lease.proof.revision, "aliases", ((name, "changed"),))
        else:
            lease._completion.receipt = object()

    with pytest.raises(Exception, match="completed store stage binding changed"):
        _source(enabled=True, output_lease=True, observer=mutate)


def test_actual_resident_metadata_endpoint_lifetime_and_mutations():
    kernel, args = _kda_fixture()
    emit = pipeline.emit_preparation_pipeline
    prepare = completed.CompletedStoreAction.prepare_point
    drain = completed.CompletedCarryEndpoints.drain
    checks = []

    def point(action, cg, plan, boundaries):
        prepare(action, cg, plan, boundaries)
        endpoint = action.carry_endpoints
        assert endpoint is not None
        (view,) = action.storage.carry_views
        assert endpoint.permits(plan, view)
        assert action._progress.prepared is not None
        metadata = action._progress.prepared.reads
        assert metadata
        for owner, field, value in (
            (endpoint, "entry", ()),
            (endpoint, "carry", None),
            (view, "byte_offset", view.byte_offset + 128),
            (view, "shape", (1,)),
            (view, "pool", "recurrence"),
            (action.storage.stages[-1], "tmem_carry", None),
            (action.pipeline.frame.layout.regions[0], "live_until", 1),
        ):
            previous = getattr(owner, field)
            object.__setattr__(owner, field, value)
            try:
                assert not endpoint.matches(plan)
                assert not action.matches(plan)
            finally:
                object.__setattr__(owner, field, previous)
            checks.append(field)
        with pytest.raises(_UnsupportedChain, match="drain is missing"):
            endpoint.validate_drained(plan, ())
        previous = endpoint._progress.token
        endpoint._progress.token = endpoint._tokens[1]
        assert not endpoint.matches(plan)
        endpoint._progress.token = previous
        original = action.carry_endpoints
        object.__setattr__(action, "carry_endpoints", None)
        assert not action.matches(plan)
        assert not all(starmap(action._metadata_read, metadata))
        object.__setattr__(action, "carry_endpoints", original)
        assert action.matches(plan)

    def end(endpoint, plan, carry):
        result = drain(endpoint, plan, carry)
        endpoint.validate_drained(plan, result)
        with pytest.raises(_UnsupportedChain, match="drain is missing"):
            endpoint.validate_drained(plan, ())
        with pytest.raises(_UnsupportedChain, match="endpoint proof changed"):
            drain(endpoint, plan, carry)
        checks.append("drain")
        return result

    with (
        patch.object(
            pipeline,
            "emit_preparation_pipeline",
            lambda *args, **kwargs: emit(*args, **kwargs, completed_member_store=True),
        ),
        patch.object(completed.CompletedStoreAction, "prepare_point", point),
        patch.object(completed.CompletedCarryEndpoints, "drain", end),
    ):
        source = _loop_source(kernel, args, _kda_config())
    assert len(checks) == 8
    assert "chain_store_0_register" in source
