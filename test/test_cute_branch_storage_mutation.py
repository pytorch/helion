from __future__ import annotations

import ast

import pytest
import torch

import helion
from helion._testing import cpu_render_target
from helion._testing.cute_codegen import cpu_bind
import helion.language as hl

pytestmark = pytest.mark.cpu_only


def _branch_mutating_votes(
    x: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    select: int,
    source: hl.constexpr,
    control: hl.constexpr,
    joined: hl.constexpr,
) -> tuple[torch.Tensor, torch.Tensor]:
    fresh = torch.empty_like(left)
    second = torch.empty_like(right)
    if source == "input":
        first = left
        replacement = fresh
    else:
        first = fresh
        replacement = left
    alias = first[:, :]
    if control == "if":
        if select > 0:
            target = first
        else:
            first.set_(replacement)
            target = second
    else:
        target = first
        while select > 0:
            first.set_(replacement)
            target = second
            select -= 1
    if not joined:
        target = first
    for row in hl.tile(x.size(0), block_size=1):
        values = x[row.id, :, :]
        vote = torch.any(values > 2)
        alias[row.id, :] = torch.amax(values, dim=1)
        target[row.id, :] = torch.where(vote, torch.amin(values, dim=1), 0.0)
    return alias, target


@pytest.mark.parametrize("source", ["input", "fresh"])
@pytest.mark.parametrize("control", ["if", "loop"])
@pytest.mark.parametrize("joined", [False, True])
def test_sibling_or_zero_trip_mutation_cannot_prove_destinations_disjoint(
    source, control, joined
):
    # Each axis crosses a distinct proof boundary: input/factory provenance,
    # sibling/zero-trip tracing, and synthetic-join/ordinary tensor pairs.
    kernel = helion.kernel(_branch_mutating_votes, backend="cute", static_shapes=True)
    args = (torch.empty(3, 32, 32), torch.empty(3, 32), torch.empty(3, 32))
    select = 1 if control == "if" else 0
    with (
        cpu_render_target(fake_device_sm_count=148),
        pytest.raises(helion.exc.BackendUnsupported, match="require disjoint storage"),
    ):
        # The selected runtime path preserves alias == target storage. Tracing
        # a different arm or a loop body must not manufacture disjointness.
        cpu_bind(kernel, (*args, select, source, control, joined))


def _safe_value_write_votes(
    x: torch.Tensor, left: torch.Tensor, right: torch.Tensor, select: int
) -> tuple[torch.Tensor, torch.Tensor]:
    first = torch.zeros_like(left)
    second = torch.empty_like(right)
    if select > 0:
        target = first
    else:
        target = second
    first.fill_(0.0)
    for row in hl.tile(x.size(0), block_size=1):
        values = x[row.id, :, :]
        vote = torch.any(values > 2)
        left[row.id, :] = torch.amax(values, dim=1)
        target[row.id, :] = torch.where(vote, torch.amin(values, dim=1), 0.0)
    return left, target


def test_value_writes_preserve_storage_proof_and_opaque_calls_invalidate_captures():
    kernel = helion.kernel(_safe_value_write_votes, backend="cute", static_shapes=True)
    args = (torch.empty(3, 32, 32), torch.empty(3, 32), torch.empty(3, 32))
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(kernel, (*args, 1))
        ast.parse(bound.to_code(bound.config_spec.default_config()))
        env = bound.env
        host = bound.host_function
        first = host.local_types["first"].fake_value
        target = host.local_types["target"].fake_value
        left = host.local_types["left"].fake_value
        assert env.tensor_storage_provenance(target) is not None
        assert env.tensor_storage_is_compiler_allocated(first)
        assert env.tensor_storage_is_zero_filled_allocation(first)
        # A no-argument Python helper can reach any of these through captures.
        # Invalidating only the explicit arguments or existing joins is unsafe.
        with env.track_runtime_layout_mutations((), opaque=True):
            pass
        for value in (first, target, left):
            assert env.tensor_storage_provenance(value) is None
            assert not env.tensor_storage_is_compiler_allocated(value)
            assert not env.tensor_storage_is_zero_filled_allocation(value)
            assert not env.tensor_layout_is_symbolically_exact(value)
        # Independent allocations after that helper remain useful; merely
        # returning an old tensor must not restore its invalidated provenance.
        with env, env.fake_mode:
            new = torch.empty((3, 32))
            env.register_tensor_factory_layout(torch.empty, ((3, 32),), {}, new)
            joined = torch.empty_like(new)
            env.register_tensor_storage_join(joined, (first, new))
        assert env.tensor_storage_provenance(joined) is None
        assert env.tensor_storage_is_compiler_allocated(new)
        assert env.tensor_layout_is_symbolically_exact(new)
