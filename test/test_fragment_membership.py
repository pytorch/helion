from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

import helion
from helion._testing import cpu_render_target
from helion._testing import onlyBackends
from helion._testing.binding_observation import observe_compiled_binding
from helion._testing.binding_observation import scalar_additions
from helion.autotuner import BooleanFragment
import helion.language as hl
from helion.runtime.kernel import BoundKernel

pytestmark = pytest.mark.cpu_only
ANCHOR = BooleanFragment()
ANCHORS = (ANCHOR,)


@helion.kernel(autotune_effort="none")
def _fragment_membership(x, value: hl.constexpr, mode: hl.constexpr):
    out = torch.empty_like(x)
    if mode == "host_in":
        offset = 2.0 if value in ANCHORS else 5.0
    elif mode == "host_not_in":
        offset = 5.0 if value not in ANCHORS else 2.0
    else:
        offset = 0.0
    for tile in hl.tile(x.numel()):
        if mode == "device_in":
            result = x[tile] + (2.0 if value in ANCHORS else 5.0)
        elif mode == "device_not_in":
            result = x[tile] + (5.0 if value not in ANCHORS else 2.0)
        else:
            result = x[tile] + offset
        out[tile] = result
    return out


@helion.kernel(autotune_effort="none")
def _sequence_membership(x, values: hl.constexpr, constructed: hl.constexpr):
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        if constructed:
            matches = ANCHOR in (values[0],)  # noqa: FURB171 - exercise membership
        else:
            matches = ANCHOR in values
        out[tile] = x[tile] + (2.0 if matches else 5.0)
    return out


def _check_routes(fn, first_args, changed_args):
    with (
        cpu_render_target(),
        patch.object(BoundKernel, "__call__", observe_compiled_binding),
    ):
        kernel = helion.kernel(fn, autotune_effort="none")
        first = kernel(*first_args)
        second = kernel(*changed_args)
        fresh = helion.kernel(fn, autotune_effort="none").bind(changed_args)
        fresh.to_code()
        assert scalar_additions(first) == [2.0]
        assert scalar_additions(second) == scalar_additions(fresh) == [5.0]
        assert first is not second
        assert second.to_code() == fresh.to_code()
        with patch.object(kernel, "bind", side_effect=AssertionError("cache miss")):
            assert kernel(*first_args) is first
            assert kernel(*changed_args) is second


@onlyBackends(["triton", "cute"])
@pytest.mark.parametrize(
    "mode", ["host_in", "host_not_in", "device_in", "device_not_in"]
)
def test_fragment_membership_routes_cached_calls(mode):
    tensor = torch.empty(17)
    _check_routes(
        _fragment_membership.fn,
        (tensor, ANCHOR, mode),
        (tensor, BooleanFragment(), mode),
    )


@onlyBackends(["triton", "cute"])
@pytest.mark.parametrize("constructed", [False, True])
def test_sequence_membership_routes_cached_calls(constructed):
    tensor = torch.empty(17)
    _check_routes(
        _sequence_membership.fn,
        (tensor, (ANCHOR,), constructed),
        (tensor, (BooleanFragment(),), constructed),
    )


@helion.kernel(autotune_effort="none")
def _scalar_membership(x, value: hl.constexpr):
    if value in (4, 8):
        offset = 2.0
    else:
        offset = 5.0
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        out[tile] = x[tile] + offset
    return out


@onlyBackends(["triton", "cute"])
@pytest.mark.parametrize("value,expected", [(4, 2.0), (7, 5.0)])
def test_scalar_membership_uses_collection_values(value, expected):
    with cpu_render_target():
        bound = _scalar_membership.bind((torch.empty(17), value))
        bound.to_code()
    assert scalar_additions(bound) == [expected]
    assert not any(
        name.startswith("fragment_call_input:")
        for name in bound._env.runtime_input_specializations
    )


@helion.kernel(autotune_effort="none")
def _nested_membership(x, values: hl.constexpr):
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        out[tile] = x[tile] + (2.0 if values in (ANCHORS,) else 5.0)  # noqa: FURB171 - exercise nested membership
    return out


@onlyBackends(["triton", "cute"])
def test_nested_membership_guards_input_leaves():
    tensor = torch.empty(17)
    _check_routes(
        _nested_membership.fn,
        (tensor, (ANCHOR,)),
        (tensor, (BooleanFragment(),)),
    )


@helion.kernel(autotune_effort="none")
def _iterated_membership(x, values: hl.constexpr):
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        result = x[tile]
        for value in values:
            result = result + (2.0 if value in ANCHORS else 5.0)
        out[tile] = result
    return out


@onlyBackends(["triton", "cute"])
def test_membership_guards_all_static_iterations():
    tensor = torch.empty(17)
    first_values = (BooleanFragment(), BooleanFragment())
    changed_values = (first_values[0], ANCHOR)
    with (
        cpu_render_target(),
        patch.object(BoundKernel, "__call__", observe_compiled_binding),
    ):
        kernel = helion.kernel(_iterated_membership.fn, autotune_effort="none")
        first = kernel(tensor, first_values)
        second = kernel(tensor, changed_values)
        fresh = helion.kernel(_iterated_membership.fn, autotune_effort="none").bind(
            (tensor, changed_values)
        )
        fresh.to_code()
        assert scalar_additions(first) == [5.0, 5.0]
        assert scalar_additions(second) == scalar_additions(fresh) == [5.0, 2.0]
        assert first is not second
        with patch.object(kernel, "bind", side_effect=AssertionError("cache miss")):
            assert kernel(tensor, first_values) is first
            assert kernel(tensor, changed_values) is second


@helion.kernel(autotune_effort="none")
def _mapping_membership(x, key: hl.constexpr):
    values = {4: x, 8: x}
    offset = 2.0 if key in values else 5.0
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        out[tile] = x[tile] + offset
    return out


@onlyBackends(["triton", "cute"])
@pytest.mark.parametrize("key,expected", [(4, 2.0), (7, 5.0)])
def test_mapping_membership_ignores_nonliteral_values(key, expected):
    with cpu_render_target():
        bound = _mapping_membership.bind((torch.empty(17), key))
        bound.to_code()
    assert scalar_additions(bound) == [expected]
