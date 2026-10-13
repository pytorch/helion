from __future__ import annotations

from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from .test_cute_chained_island_consumers import _config
from .test_cute_chained_island_consumers import _fixture
from .test_cute_chained_island_retained import _capture_retained
from .test_cute_chained_loop_tmem_transport import _source
from helion import exc
from helion._compiler.cute import chained_island_publication as islands
from helion._compiler.cute import chained_preparation_actions as actions
from helion._compiler.cute import chained_preparation_storage as storage


def _grouped(dtype, enabled=True):
    kernel, args = _fixture(dtype)
    config = _config(enabled, generic=True)
    config.config.update(
        cute_chained_operand_retention=True, cute_chained_vector_group=True
    )
    bodies, bindings = [], []
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
        bindings.append(result)
        return result

    with (
        patch.object(actions, "build_accepted_preparation", build),
        patch.object(storage, "bind_preparation_storage", bind),
    ):
        source = _source(kernel, args, config)
    assert len(bodies) == len(bindings) == 1 and bindings[0] is not None
    return source, bodies[0], bindings[0]


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("enabled", (False, True))
def test_unrelated_no_group_activation_is_not_repaired(dtype, enabled):
    with pytest.raises(
        exc.BackendUnsupported,
        match="vector grouping requires shared coordinate-owned operands",
    ):
        _grouped(dtype, enabled)


@pytest.mark.parametrize(
    "ordinal,field,value",
    (
        (0, "role", "b"),
        (1, "role", "a"),
        (0, "target", "wrong_owner"),
        (1, "target", "wrong_owner"),
        (0, "offset", 1),
        (1, "shape", (16, 32)),
    ),
)
def test_actual_operand_tuple_cannot_change_role_owner_or_geometry(
    ordinal, field, value
):
    original = islands.IslandConsumerPublication.grouped_operands
    seen = []

    def altered(self, operands):
        assert not self.consumed
        seen.append(self)
        entries = list(operands)
        entries[ordinal] = replace(entries[ordinal], **{field: value})
        return original(self, tuple(entries))

    with (
        patch.object(islands.IslandConsumerPublication, "grouped_operands", altered),
        pytest.raises(exc.InternalError) as failure,
    ):
        _grouped(torch.bfloat16)
    assert isinstance(failure.value.__cause__, islands._UnsupportedChain)
    assert len(seen) == 1 and not seen[0].consumed


def test_vector_mode_mutation_after_publication_is_not_new_selection():
    _, accepted, _ = _capture_retained("generic")
    (publication,) = accepted.island_publications
    vector = publication.candidate.vector
    original = vector.group_enabled
    try:
        vector.group_enabled = not original
        assert not publication.matches()
    finally:
        vector.group_enabled = original
    assert publication.matches()
