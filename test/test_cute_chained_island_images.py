from __future__ import annotations

from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from .test_cute_chained_island_consumers import _config
from .test_cute_chained_loop_tmem_transport import _source
import helion
from helion import exc
from helion._compiler.cute import chained_preparation_actions as actions
from helion._compiler.cute import chained_preparation_storage as storage
from helion._compiler.cute import chained_register_emission as emission
import helion.language as hl


@helion.kernel(
    backend="cute", static_shapes=True, fast_math=True, autotune_effort="none"
)
def publication_outer_retained(x, y, initial):
    steps, rows, width = x.shape
    history = torch.empty((steps, rows, width), dtype=torch.float32, device=x.device)
    final = torch.empty_like(initial)
    for _ in hl.tile(rows, block_size=32):
        initial_rows, initial_columns = hl.arange(32), hl.arange(width)
        state = initial[initial_rows, initial_columns]
        for step in hl.tile(steps, block_size=1):
            i, j, columns = hl.arange(32), hl.arange(32), hl.arange(width)
            base = hl.dot(
                x[step.id, i, columns],
                y[step.id, j, columns].T,
                out_dtype=torch.float32,
            )
            diagonal = (i[:, None] // 16) == (j[None, :] // 16)
            factor = torch.where(diagonal, base, 0.0).to(x.dtype)
            square = hl.dot(factor, factor, out_dtype=torch.float32)
            cube = hl.dot(square.to(x.dtype), factor, out_dtype=torch.float32)
            coefficient = torch.where(
                diagonal, torch.exp((square + cube) * 0.0625), 0.0
            ).to(x.dtype)
            lower = torch.where((i[:, None] >= 16) & (j[None, :] < 16), base, 0.0).to(
                x.dtype
            )
            left = hl.dot(coefficient, lower, out_dtype=torch.float32)
            rounded = (-left).to(x.dtype)
            right = hl.dot(rounded, coefficient, out_dtype=torch.float32)
            frontier = (right + coefficient.float()).to(x.dtype)
            state = hl.dot(
                frontier, state.to(x.dtype), acc=state, out_dtype=torch.float32
            )
            history[step.id, i, columns] = state
        final[initial_rows, initial_columns] = state
    return history, final


def _capture(dtype, steps=1):
    args = (
        torch.zeros((steps, 32, 128), dtype=dtype),
        torch.zeros((steps, 32, 128), dtype=dtype),
        torch.zeros((32, 128)),
    )
    config = _config(True, generic=True)
    config.config["cute_chained_operand_retention"] = True
    bodies, physical = [], []
    original_build, original_bind = (
        actions.build_accepted_preparation,
        storage.bind_preparation_storage,
    )

    def build(*args, **kwargs):
        result = original_build(*args, **kwargs)
        bodies.append(result)
        return result

    def bind(*args, **kwargs):
        result = original_bind(*args, **kwargs)
        physical.append(result)
        return result

    with (
        patch.object(actions, "build_accepted_preparation", build),
        patch.object(storage, "bind_preparation_storage", bind),
    ):
        source = _source(publication_outer_retained, args, config)
    assert len(bodies) == len(physical) == 1 and physical[0] is not None
    assert len(bodies[0].island_publications) == 1
    publication = bodies[0].island_publications[0]
    assert publication.island_completion is not None
    assert len(bodies[0].island_reads) >= 2
    assert all(reader.matches(bodies[0]) for reader in bodies[0].island_reads)
    return source, bodies[0], physical[0]


@pytest.fixture(scope="module", params=(torch.bfloat16, torch.float16))
def completed(request):
    return _capture(request.param)


def test_actual_completed_outer_reads_both_images_and_retained_frontier(completed):
    source, accepted, physical = completed
    publication = accepted.island_publications[0]
    completion = publication.island_completion
    assert completion is not None and completion.current()
    action = next(item for item in accepted.actions if item.proof is completion.bound)
    assert completion.matches(accepted, action)
    assert publication.candidate.owner not in action.omitted
    assert len(completion.input_reads) == 2
    assert completion.input_reads[0][2] != completion.input_reads[1][2]
    assert len(completion.operand_uses) == 4
    assert accepted.island_reads[-1].action.first >= action.stop
    assert "chain_retained_operand" in source
    assert physical is not None


@pytest.mark.parametrize("steps", (0, 3))
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
def test_zero_and_multiple_loop_trip_source(dtype, steps):
    _capture(dtype, steps)


@pytest.mark.parametrize(
    "field",
    (
        "lines",
        "operand_uses",
        "input_reads",
        "alias_lines",
        "alias_outputs",
        "context",
        "dtype",
    ),
)
def test_completed_span_fields_cannot_change(completed, field):
    _, accepted, _ = completed
    publication = accepted.island_publications[0]
    completion = publication.island_completion
    assert completion is not None
    previous = getattr(completion, field)
    try:
        object.__setattr__(completion, field, "" if field == "dtype" else ())
        assert not completion.current()
        assert not publication.accepted(
            accepted,
            next(
                item
                for item in accepted.actions
                if item.island_publication is publication
            ),
        )
    finally:
        object.__setattr__(completion, field, previous)
    assert completion.current()


@pytest.mark.parametrize(
    "change",
    (
        "omitted_owner",
        "frontier_drop",
        "wrong_role",
        "read_drop",
        "alias_under_if",
        "join_drop",
    ),
)
def test_original_operations_and_final_read_are_required(completed, change):
    _, accepted, _ = completed
    publication = accepted.island_publications[0]
    completion = publication.island_completion
    assert completion is not None
    action = next(item for item in accepted.actions if item.proof is completion.bound)
    if change == "omitted_owner":
        assert not completion.matches(
            accepted,
            replace(action, omitted=(*action.omitted, publication.candidate.owner)),
        )
    elif change == "frontier_drop":
        changed = replace(accepted, island_reads=accepted.island_reads[:-1])
        producer = next(
            item for item in changed.actions if item.island_publication is publication
        )
        assert not publication.accepted(changed, producer)
    else:
        # Even a freshly captured field tuple is not enough: original typed
        # ports, actual full-owner reads and all-role stage scopes must match.
        if change == "wrong_role":
            uses = list(completion.operand_uses)
            stage, _, image = uses[0]
            uses[0] = (stage, "b", image)
            changed = replace(completion, operand_uses=tuple(uses))
        elif change == "read_drop":
            changed = replace(completion, input_reads=completion.input_reads[:-1])
        else:
            lines = list(completion.lines)
            if change == "alias_under_if":
                lines[completion.alias_first] = (
                    "if False:\n    " + lines[completion.alias_first]
                )
            else:
                lines.pop(completion.alias_first - 1)
            changed = replace(completion, lines=tuple(lines))
        changed = replace(changed, _selection=changed.fields())
        assert not changed.current()


@pytest.mark.parametrize("change", ("owner", "layout", "graph", "config"))
def test_same_object_late_original_facts_reject(completed, change):
    _, accepted, _ = completed
    publication = accepted.island_publications[0]
    completion = publication.island_completion
    assert completion is not None
    if change == "owner":
        target, field, value = publication.candidate, "target", "foreign_native"
    elif change == "layout":
        target = publication.candidate.stage.a
        field, value = "byte_offset", target.byte_offset + 128
    elif change == "graph":
        target = completion.bound.island.components[0].issues[1].node
        field, value = "args", tuple(reversed(target.args[:2])) + target.args[2:]
    else:
        config = publication.candidate.cg.device_function.config.config
        previous = config["cute_chained_operand_retention"]
        try:
            config["cute_chained_operand_retention"] = False
            assert not completion.current()
        finally:
            config["cute_chained_operand_retention"] = previous
        return
    previous = getattr(target, field)
    users = (
        tuple((node, tuple(node.users.items())) for node in target.graph.nodes)
        if change == "graph"
        else ()
    )
    try:
        object.__setattr__(target, field, value)
        assert not completion.current()
    finally:
        object.__setattr__(target, field, previous)
        for node, original in users:
            node.users.clear()
            node.users.update(original)
    assert completion.current()


@pytest.mark.parametrize("change", ("line", "return", "boundary", "duplicate"))
def test_actual_emitter_return_cannot_change_or_repeat(change):
    original = emission.emit_register_island
    hits = []

    def altered(cg, plan, bound, boundaries, **kwargs):
        result = original(cg, plan, bound, boundaries, **kwargs)
        if bound.island_input is not None:
            assert result is not None
            hits.append(bound)
            if change == "line":
                result.pop(-1)
            elif change == "return":
                result.insert(-1, "return")
            elif change == "boundary":
                boundaries[bound.island_input.candidate.operand] = "foreign_alias"
            else:
                completion = bound.island_input.island_completion
                assert completion is not None
                bound.island_input.record_island(completion, boundaries)
        return result

    with (
        patch.object(emission, "emit_register_island", altered),
        pytest.raises((exc.BackendUnsupported, exc.InternalError)),
    ):
        _capture(torch.bfloat16)
    assert hits
