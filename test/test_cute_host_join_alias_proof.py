from __future__ import annotations

import ast

import pytest
import torch

import helion
from helion._compiler.cute.full_reduction_owner import requires_full_reduction_owner
from helion._compiler.runtime_tensor_guards import (
    tensor_pairs_with_proven_disjoint_storage,
)
from helion._testing import cpu_render_target
from helion._testing.cute_codegen import cpu_bind
from helion._testing.tunable_fragments import load_host_function
import helion.language as hl

pytestmark = pytest.mark.cpu_only


def _host_join_full_votes(
    x: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    select: int,
    kind: hl.constexpr,
    view: hl.constexpr,
) -> tuple[torch.Tensor, torch.Tensor]:
    first = torch.empty_like(left)
    second = torch.empty_like(right)
    if kind == "fresh":
        if select > 0:
            target = first
        else:
            target = second
    elif kind == "out":
        overwritten = torch.zeros(left.shape, out=left)
        if select > 0:
            target = overwritten
        else:
            target = right
    else:
        if select > 0:
            target = left
        else:
            target = right
    # A second join must retain the earlier join's full set of alternatives.
    if select < 0:
        selected = first
    else:
        selected = target
    if view:
        selected = selected[:, :]
    for row in hl.tile(x.size(0), block_size=1):
        values = x[row.id, :, :]
        vote = torch.any(values > 2)
        left[row.id, :] = torch.amax(values, dim=1)
        selected[row.id, :] = torch.where(vote, torch.amin(values, dim=1), 0.0)
    return left, selected


@pytest.mark.parametrize(
    "kind,select,view",
    [
        ("inputs", 1, False),
        ("inputs", 0, False),
        ("inputs", 1, True),
        ("inputs", 0, True),
        ("out", 1, False),
        ("out", 0, True),
    ],
)
def test_host_join_cannot_prove_fragment_destinations_disjoint(kind, select, view):
    kernel = helion.kernel(_host_join_full_votes, backend="cute", static_shapes=True)
    args = (torch.empty(3, 32, 32), torch.empty(3, 32), torch.empty(3, 32))
    with (
        cpu_render_target(fake_device_sm_count=148),
        pytest.raises(helion.exc.BackendUnsupported, match="require disjoint storage"),
    ):
        # Even the currently disjoint branch may alias left on the next call.
        # out= writes an existing input and must not grant fresh provenance.
        cpu_bind(kernel, (*args, select, kind, view))


@pytest.mark.parametrize("view", [False, True])
def test_fresh_host_join_preserves_safe_proofs_and_member_aliases(view):
    kernel = helion.kernel(_host_join_full_votes, backend="cute", static_shapes=True)
    args = (torch.empty(3, 32, 32), torch.empty(3, 32), torch.empty(3, 32))
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(kernel, (*args, 1, "fresh", view))
        host = bound.host_function
        assert requires_full_reduction_owner(host.device_ir.graphs)
        with bound.env, host:
            tensors = {
                name: host.local_types[name].fake_value
                for name in ("left", "right", "first", "second", "selected")
            }
            pairs = tensor_pairs_with_proven_disjoint_storage(bound.env, tensors)
        assert frozenset(("first", "second")) in pairs
        assert frozenset(("left", "selected")) in pairs
        assert frozenset(("right", "selected")) in pairs
        assert frozenset(("first", "selected")) not in pairs
        assert frozenset(("second", "selected")) not in pairs
        config = bound.config_spec.default_config()
        ast.parse(bound.to_code(config))
        entry = load_host_function(bound, config, lambda *args, **kwargs: None)
        # Exercise actual wrapper branch/alias behavior on the same binding;
        # device arithmetic is deliberately not executed by this CPU test.
        for select in (1, 0, -1, 1):
            output, selected = entry(*args, select, "fresh", view)
            assert output is args[1]
            assert selected.shape == args[1].shape
            assert selected.untyped_storage() != args[1].untyped_storage()
            assert selected.untyped_storage() != args[2].untyped_storage()


def _transient_storage_view(first, selected, left, use_data):
    original = first.data
    if use_data:
        first.data = left
    else:
        first.set_(left)
    view = selected[:, :]
    if use_data:
        first.data = original
    else:
        first.set_(original)
    return view


def _mutated_host_join_full_votes(
    x: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    select: int,
    mode: hl.constexpr,
) -> tuple[torch.Tensor, torch.Tensor]:
    first = torch.empty_like(left)
    second = torch.empty_like(right)
    if mode == "shared_storage":
        second = first[:, :]
    if select > 0:
        inner = first
    else:
        inner = second
    if mode == "nested":
        if select < 0:
            target = first
        else:
            target = inner
        # The intermediate joined Tensor itself must be watched, not just
        # the leaf owners obtained by flattening its storage alternatives.
        inner.set_(left)
    elif mode == "opaque_set_restore" or mode == "opaque_data_restore":
        target = _transient_storage_view(
            first, inner, left, mode == "opaque_data_restore"
        )
    else:
        target = inner
        if mode == "view_before":
            target = target[:, :]
        original = first.data
        if mode == "shared_storage":
            # first and second originally share storage but are distinct
            # Tensor objects; changing second must not be hidden by dedup.
            second.set_(left)
        else:
            first.set_(left)
        if mode == "view_after" or mode == "restore":
            target = target[:, :]
        if mode == "restore":
            # The view may retain the temporary storage after restoration.
            first.set_(original)
    for row in hl.tile(x.size(0), block_size=1):
        values = x[row.id, :, :]
        vote = torch.any(values > 2)
        left[row.id, :] = torch.amax(values, dim=1)
        target[row.id, :] = torch.where(vote, torch.amin(values, dim=1), 0.0)
    return left, target


@pytest.mark.parametrize(
    "mode,select",
    [
        ("plain", 1),
        ("view_after", 1),
        ("view_before", 1),
        ("restore", 1),
        ("shared_storage", 0),
        ("nested", 1),
        ("opaque_set_restore", 1),
        ("opaque_data_restore", 1),
    ],
)
def test_mutating_host_join_owners_invalidates_fragment_alias_proof(mode, select):
    kernel = helion.kernel(
        _mutated_host_join_full_votes, backend="cute", static_shapes=True
    )
    args = (torch.empty(3, 32, 32), torch.empty(3, 32), torch.empty(3, 32))
    with (
        cpu_render_target(fake_device_sm_count=148),
        pytest.raises(helion.exc.BackendUnsupported, match="require disjoint storage"),
    ):
        # Even view_before, whose eager storage stays safe, deliberately loses
        # this optimization after its possible owner mutates. Never substitute
        # the owner's current storage for a previously captured view's storage.
        cpu_bind(kernel, (*args, select, mode))


def test_inference_tensor_joins_do_not_require_an_unavailable_version_counter():
    kernel = helion.kernel(_host_join_full_votes, backend="cute", static_shapes=True)
    with torch.inference_mode(), cpu_render_target(fake_device_sm_count=148):
        args = (torch.empty(3, 32, 32), torch.empty(3, 32), torch.empty(3, 32))
        bound = cpu_bind(kernel, (*args, 1, "fresh", False))
        with bound.env, bound.host_function:
            selected = bound.host_function.local_types["selected"].fake_value
            assert bound.env.tensor_storage_provenance(selected) is not None
        ast.parse(bound.to_code(bound.config_spec.default_config()))
