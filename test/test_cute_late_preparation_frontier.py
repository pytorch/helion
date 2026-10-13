from __future__ import annotations

from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from .test_cute_chained_body_program import _source
from .test_cute_chained_frontier_materialization import completed as completed
from .test_cute_chained_island_publication import _capture
from helion import exc
from helion._compiler.cute import chained_island_publication as island
from helion._compiler.cute.chained_preparation_frame import PreparationBuffer
from helion._compiler.cute.chained_preparation_reads import CompletedPreparationStage
from helion._compiler.cute.chained_preparation_reads import PreparationReadFrontier
from helion._compiler.cute.chained_preparation_reads import resolved_preparation_reads
from helion._compiler.cute.chained_preparation_storage import (
    plan_accepted_preparation_storage,
)
from helion._compiler.cute.physical_use_frontier import InvalidPhysicalUse
from helion._compiler.cute.physical_use_frontier import PhysicalUseFrontier


def _real_read(completed):
    accepted = completed[2].physical.accepted
    action = next(
        action
        for action in accepted.actions
        if isinstance(action.proof, CompletedPreparationStage) and action.inputs
    )
    receipt = action.proof
    assert isinstance(receipt, CompletedPreparationStage)
    context = PreparationReadFrontier(accepted.pipeline, dict(accepted.revision.shapes))
    return accepted, action, receipt, context


def test_actual_late_reads_use_original_full_owners(completed):
    accepted, action, receipt, context = _real_read(completed)
    observed = []
    original = context.frontier.resolve

    def resolve(*args, **kwargs):
        result = original(*args, **kwargs)
        observed.extend(result)
        return result

    with patch.object(context.frontier, "resolve", resolve):
        reads = resolved_preparation_reads(
            accepted.pipeline,
            receipt.roots,
            dict(action.inputs),
            published=action.inputs,
            use_frontier=context,
        )
    assert reads == receipt.reads and observed
    assert all(isinstance(value.owner, PreparationBuffer) for value in observed)
    assert all(
        any(value.owner is owner for owner in accepted.pipeline.frame.buffers)
        for value in observed
    )
    # Replaying the same environment uses the same typed publications and cache.
    cache = dict(context.publications)
    assert (
        resolved_preparation_reads(
            accepted.pipeline,
            receipt.roots,
            dict(action.inputs),
            published=action.inputs,
            use_frontier=context,
        )
        == reads
    )
    assert all(context.publications[key] is value for key, value in cache.items())
    context.check()


@pytest.mark.parametrize("mutation", ["drop", "rename", "reorder", "foreign_pipeline"])
def test_completed_inventory_is_not_recaptured_from_current_boundaries(
    completed, mutation
):
    accepted, action, receipt, context = _real_read(completed)
    current = dict(action.inputs)
    pipeline = accepted.pipeline
    if mutation == "drop":
        current.pop(next(iter(current)))
    elif mutation == "rename":
        current[next(iter(current))] = "not_the_original_owner"
    elif mutation == "reorder":
        assert len(current) > 1
        current = dict(reversed(tuple(current.items())))
    else:
        pipeline = replace(pipeline)
    with pytest.raises(ValueError, match="completed publication inventory"):
        resolved_preparation_reads(
            pipeline,
            receipt.roots,
            current,
            published=action.inputs,
            use_frontier=context,
        )


@pytest.mark.parametrize("field", ["owner", "shape", "dtype", "node"])
def test_cached_publication_cannot_change_original_value_or_owner(completed, field):
    accepted, action, receipt, context = _real_read(completed)
    resolved_preparation_reads(
        accepted.pipeline,
        receipt.roots,
        dict(action.inputs),
        published=action.inputs,
        use_frontier=context,
    )
    key, value = next(iter(context.publications.items()))
    if field == "owner":
        altered = replace(value, owner=object())
    elif field == "shape":
        altered = replace(value, shape=(1,))
    elif field == "dtype":
        altered = replace(
            value,
            dtype=torch.float16 if value.dtype != torch.float16 else torch.float32,
        )
    else:
        altered = replace(
            value,
            node=next(node for node, _ in action.inputs if node is not value.node),
        )
    context.publications[key] = altered
    with pytest.raises(ValueError, match="cache lost original owner/view"):
        resolved_preparation_reads(
            accepted.pipeline,
            receipt.roots,
            dict(action.inputs),
            published=action.inputs,
            use_frontier=context,
        )


@pytest.mark.parametrize(
    "field", ["logical_shape", "full_shape", "row_offset", "logical_modes"]
)
def test_actual_retained_view_revision_cannot_change(completed, field):
    _, _, _, context = _real_read(completed)
    name, retained = next(iter(context.retained.items()))
    values = {
        "logical_shape": (1, 1),
        "full_shape": (1, 1),
        "row_offset": retained.row_offset + 16,
        "logical_modes": (1, 0),
    }
    context.retained[name] = replace(retained, **{field: values[field]})
    with pytest.raises(ValueError, match="publication environment changed"):
        context.check()


@pytest.mark.parametrize(
    "field", ["frame", "buffers", "retained_plan", "retained_owner"]
)
def test_current_pipeline_replacement_rejects_at_original_storage_consumer(
    completed, field
):
    original = completed[2].physical.accepted
    pipeline = replace(original.pipeline)
    context = PreparationReadFrontier(pipeline, dict(original.revision.shapes))
    accepted = replace(original, pipeline=pipeline, read_frontier=context)
    accepted = replace(accepted, _selection=accepted._fields())
    assert accepted.matches(
        original.revision.plan, pipeline, dict(original.revision.shapes)
    )
    if field == "frame":
        object.__setattr__(pipeline, "frame", replace(pipeline.frame))
    elif field == "buffers":
        frame = replace(
            pipeline.frame,
            buffers=tuple(replace(buffer) for buffer in pipeline.frame.buffers),
        )
        object.__setattr__(pipeline, "frame", frame)
    else:
        retained = pipeline.operand_retention
        assert retained is not None
        if field == "retained_owner":
            candidate = retained.candidates[0]
            replacement = replace(candidate, owner="foreign_full_owner")
            retained = replace(
                retained, candidates=(replacement, *retained.candidates[1:])
            )
        else:
            retained = replace(retained)
        object.__setattr__(pipeline, "operand_retention", retained)
    with pytest.raises(ValueError, match="publication environment changed"):
        context.check()
    assert (
        plan_accepted_preparation_storage(
            original.revision.plan,
            pipeline,
            accepted,
            capacity_bytes=completed[2].physical.capacity_bytes,
        )
        is None
    )


def test_real_fragment_binding_not_a_node_only_stop():
    original = island.resolved_preparation_reads
    observed = []

    def resolve(*args, **kwargs):
        bound = kwargs["fragments"]
        result = original(*args, **kwargs)
        assert bound.exports
        bad = replace(bound, published=())
        assert bound.published
        with pytest.raises(ValueError, match="original fragment binding"):
            original(*args, **{**kwargs, "fragments": bad})
        observed.append(bound)
        return result

    with patch.object(island, "resolved_preparation_reads", resolve):
        _, accepted, physical = _capture(True)
    assert observed and physical is not None and accepted.island_publications
    publication = accepted.island_publications[0]
    context = accepted.read_frontier
    expected = accepted.island_publications
    context.check(expected)
    # These are copies/drops of an actual successful token, not fabricated
    # publications. Value equality cannot replace the original attempt identity.
    for changed in ([], [replace(publication)], [publication, publication]):
        with (
            patch.object(context, "islands", changed),
            pytest.raises(ValueError, match="publication environment changed"),
        ):
            context.check(expected)


@pytest.mark.parametrize("member", ["buffers", "candidates"])
def test_same_current_container_replacement_rejects_before_repacking(completed, member):
    accepted = completed[2].physical.accepted
    pipeline = accepted.pipeline
    container = pipeline.frame if member == "buffers" else pipeline.operand_retention
    assert container is not None
    original = getattr(container, member)
    replacement = tuple(replace(item) for item in original)
    try:
        object.__setattr__(container, member, replacement)
        assert (
            plan_accepted_preparation_storage(
                accepted.revision.plan,
                pipeline,
                accepted,
                capacity_bytes=completed[2].physical.capacity_bytes,
            )
            is None
        )
    finally:
        object.__setattr__(container, member, original)
    accepted.read_frontier.check(accepted.island_publications)


@pytest.mark.parametrize("method", ["resolve", "read", "check"])
def test_materialized_root_cannot_bypass_common_frontier(method):
    error = InvalidPhysicalUse("unproven root read")
    with (
        patch.object(PhysicalUseFrontier, method, side_effect=error) as called,
        pytest.raises((InvalidPhysicalUse, exc.InternalError)) as rejected,
    ):
        _source()
    assert called.call_count > 0
    assert rejected.value is error or rejected.value.__cause__ is error
