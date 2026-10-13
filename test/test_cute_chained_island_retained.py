from __future__ import annotations

from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from .test_cute_chained_island_consumers import _config
from .test_cute_chained_island_consumers import _fixture
from .test_cute_chained_loop_tmem_transport import _source
from .test_cute_chained_preparation_cut import _kda_fixture
from helion import exc
from helion._compiler.cute import chained_body_program as body
from helion._compiler.cute import chained_island_publication as islands
from helion._compiler.cute import chained_preparation_actions as actions
from helion._compiler.cute import chained_preparation_storage as storage


def _capture_retained(family="kda", dtype=torch.bfloat16, enabled=True):
    generic = family != "kda"
    kernel, args = _fixture(dtype) if generic else _kda_fixture()
    config = _config(enabled, generic=generic)
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
        source = _source(kernel, args, config)
    assert len(bodies) == len(physical) == 1
    return source, bodies[0], physical[0]


@pytest.mark.parametrize(
    "family,dtype",
    (("kda", torch.bfloat16), ("generic", torch.bfloat16), ("generic", torch.float16)),
)
def test_original_retained_alias_uses_same_complete_owner(family, dtype):
    source, accepted, bound = _capture_retained(family, dtype)
    assert bound is not None
    assert len(accepted.island_publications) == 1
    publication = accepted.island_publications[0]
    assert publication.candidate.retained_input is not None
    alias = publication.retained_alias
    assert alias is not None
    assert publication.matches()
    if isinstance(alias.completion, islands.CompletedInputIsland):
        completion = alias.completion
        assert completion.current()
        action = next(
            item for item in accepted.actions if item.proof is completion.bound
        )
        assert completion.matches(accepted, action)
        assert len(completion.input_reads) == 2
        assert tuple(role for _, role, _, _ in completion.input_reads) == ("a", "b")
        assert completion.input_reads[0][2] != completion.input_reads[1][2]
        assert publication.candidate.owner not in action.omitted
        assert any(
            reader.action.first >= action.stop and reader.frontier is not None
            for reader in accepted.island_reads
        )
        alias_stop = completion.bound.first_event + 2
        assert publication.candidate.retained_input[1].publication_event == alias_stop
    else:
        assert alias.completion.original_warp_matches()
        assert len(accepted.island_reads) >= 3
        alias_stop = alias.completion.stop
    assert all(reader.matches(accepted) for reader in accepted.island_reads)
    assert "\n".join(alias.lines) in source
    assert (
        dict(alias.inputs)[publication.candidate.operand] == publication.candidate.owner
    )
    assert dict(alias.outputs)[
        publication.candidate.operand
    ] == publication.boundary_name(alias_stop)


def test_retained_default_and_explicit_false_source_exact():
    assert _capture_retained(enabled=None)[0] == _capture_retained(enabled=False)[0]


@pytest.fixture(scope="module")
def retained():
    return _capture_retained("generic")


@pytest.mark.parametrize(
    "field,value",
    (
        ("row_offset", 1),
        ("logical_modes", (1, 0)),
        ("owner", "wrong_owner"),
        ("publication_event", -1),
        ("dtype", torch.float32),
    ),
)
def test_same_object_original_retained_facts_do_not_recapture(retained, field, value):
    _, accepted, _ = retained
    publication = accepted.island_publications[0]
    selected = publication.candidate.retained_input
    assert selected is not None
    candidate = selected[1]
    original = getattr(candidate, field)
    try:
        object.__setattr__(candidate, field, value)
        assert not publication.matches()
    finally:
        object.__setattr__(candidate, field, original)
    assert publication.matches()


@pytest.mark.parametrize("change", ("copy", "drop", "line", "outputs", "inputs"))
def test_retained_transition_token_and_full_maps_are_latched(retained, change):
    _, accepted, _ = retained
    publication = accepted.island_publications[0]
    alias = publication.retained_alias
    assert alias is not None
    if change in ("copy", "drop"):
        original = publication.retained_alias
        try:
            publication.retained_alias = replace(alias) if change == "copy" else None
            assert not publication.matches()
        finally:
            publication.retained_alias = original
    else:
        field = "lines" if change == "line" else change
        original = getattr(alias, field)
        try:
            object.__setattr__(alias, field, ())
            assert not publication.matches()
        finally:
            object.__setattr__(alias, field, original)
    assert publication.matches()
    assert all(reader.matches(accepted) for reader in accepted.island_reads)


@pytest.mark.parametrize("change", ("line", "index", "boundary", "drop", "duplicate"))
def test_actual_after_return_alias_cannot_be_recertified(change):
    original = body._PreparationBody.publish_retained
    calls = []

    def altered(self, first, stop, recorder=None):
        lines = original(self, first, stop, recorder)
        if recorder is not None:
            publications = [
                item
                for item in recorder._publication_receipts
                if item.retained_alias is not None
                and item.retained_alias.completion.first == first
            ]
            if publications:
                assert len(publications) == 1
                item = publications[0]
                calls.append(item)
                if change == "line":
                    lines[0] += " # modified after original return"
                elif change == "index":
                    lines[0] = lines[0].replace(" = ", " = wrong_index + ")
                elif change == "boundary":
                    self.boundaries[item.candidate.operand] = "unpublished_native"
                elif change == "drop":
                    item.retained_alias = None
                else:
                    alias = item.retained_alias
                    assert alias is not None
                    item.record_retained(
                        alias.completion, alias.inputs, self.boundaries, alias.lines
                    )
        return lines

    with (
        patch.object(body._PreparationBody, "publish_retained", altered),
        pytest.raises(
            (exc.BackendUnsupported, exc.InternalError, islands._UnsupportedChain)
        ) as failure,
    ):
        _capture_retained("generic")
    if isinstance(failure.value, exc.InternalError):
        assert isinstance(failure.value.__cause__, islands._UnsupportedChain)
    assert len(calls) == 1


def test_missing_original_capture_fails_before_accepted_body():
    with (
        patch.object(
            islands.IslandConsumerPublication, "record_retained", return_value=None
        ),
        pytest.raises(
            (exc.BackendUnsupported, exc.InternalError, islands._UnsupportedChain)
        ) as failure,
    ):
        _capture_retained("generic")
    if isinstance(failure.value, exc.InternalError):
        assert isinstance(failure.value.__cause__, islands._UnsupportedChain)
