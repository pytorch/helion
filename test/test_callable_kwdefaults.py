from __future__ import annotations

import ast
from unittest.mock import patch

import pytest
import torch

import helion
from helion._compiler.lift_closures import lift_closures
from helion._compiler.variable_origin import ArgumentOrigin
from helion._testing import cpu_render_target
from helion._testing import onlyBackends
import helion.language as hl
from helion.runtime.kernel import BoundKernel

pytestmark = pytest.mark.cpu_only


@helion.kernel(autotune_effort="none")
def _apply(x, fn, override: hl.constexpr = False):
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        if override:
            out[tile] = fn(x[tile], bias=7.0)
        else:
            out[tile] = fn(x[tile])
    return out


def _with_scalar(bias):
    def apply(value, *, bias=bias):
        return value + bias

    return apply


def _with_mutable(bias):
    def apply(value, *, bias=[bias]):  # noqa: B006 -- exercise a shared mutable default
        return value + bias[0]

    return apply


def _required(value, *, bias):
    return value + bias


def _offsets(bound):
    result = [
        node.args[1]
        for graph in bound.host_function.device_ir.graphs
        for node in graph.graph.nodes
        if node.target is torch.ops.aten.add.Tensor
    ]
    ast.parse(bound.to_code())
    return result


def _observe(bound, *args):
    bound.to_code()
    bound._run = lambda *unused: bound
    return bound


@onlyBackends(["triton", "cute"])
@pytest.mark.parametrize("mutable", [False, True])
def test_keyword_defaults_keep_cached_and_fresh_programs_consistent(mutable):
    tensor = torch.empty(17)
    factory = _with_mutable if mutable else _with_scalar
    fn = factory(2.5)
    with cpu_render_target(), patch.object(BoundKernel, "__call__", _observe):
        kernel = helion.kernel(_apply.fn, autotune_effort="none")
        first = kernel(tensor, fn)
        assert _offsets(first) == [2.5]
        prepared = kernel._prepared_call
        assert prepared is not None
        if mutable:
            fn.__kwdefaults__["bias"][0] = 5.5
        else:
            fn.__kwdefaults__["bias"] = 5.5
        assert not prepared.matches(kernel, (tensor, fn))
        second = kernel(tensor, fn)
        assert second is not first
        assert _offsets(second) == [5.5]
        fresh = helion.kernel(_apply.fn, autotune_effort="none").bind(
            (tensor, factory(5.5))
        )
        assert second.to_code() == fresh.to_code()
        with patch.object(kernel, "bind", side_effect=AssertionError("cache miss")):
            assert kernel(tensor, factory(5.5)) is second
            assert kernel(tensor, factory(2.5)) is first
        assert _offsets(first) == [2.5]
    # Normal Python invocation agrees with the selected keyword default.
    assert fn(1.0) == 6.5
    assert _with_scalar(2.5)(1.0) == 3.5


@onlyBackends(["triton", "cute"])
def test_explicit_keywords_override_defaults_and_required_keywords_stay_required():
    tensor = torch.empty(17)
    with cpu_render_target():
        for fn in (_with_scalar(2.5), _required):
            bound = helion.kernel(_apply.fn, autotune_effort="none").bind(
                (tensor, fn, True)
            )
            assert _offsets(bound) == [7.0]
        with pytest.raises(
            helion.exc.TorchOpTracingError, match="required keyword-only argument"
        ):
            helion.kernel(_apply.fn, autotune_effort="none").bind(
                (tensor, _required, False)
            )
    assert _required(1.0, bias=7.0) == 8.0
    assert _with_scalar(2.5)(1.0, bias=7.0) == 8.0


@onlyBackends(["triton", "cute"])
def test_lifting_preserves_keyword_dictionary_and_mutable_aliases():
    shared = []

    def defaults(positional=shared, *, left=shared, right=shared):
        return positional, left, right

    tensor = torch.empty(17)
    with cpu_render_target():
        # Use a real frontend context; only these pure Python defaults are called.
        bound = helion.kernel(_apply.fn, autotune_effort="none").bind(
            (tensor, _required, True)
        )
        with bound.env, bound.host_function:
            lifted = lift_closures(defaults, ArgumentOrigin("fn"))
            first = lifted()
            assert all(value is shared for value in first)
            shared.append(4)
            assert lifted()[1] is defaults()[1] is shared
            assert lifted()[1] == [4]
            replacement = [8]
            defaults.__kwdefaults__["left"] = replacement
            actual = lifted()
            normal = defaults()
            assert all(a is b for a, b in zip(actual, normal, strict=True))
            assert actual[0] is actual[2] is shared
            assert actual[1] is replacement
            # Replacing the mapping belongs to a fresh lifted capture, just like
            # replacing positional __defaults__ after an earlier capture.
            defaults.__kwdefaults__ = {"left": replacement, "right": replacement}
            fresh = lift_closures(defaults, ArgumentOrigin("fn"))
            actual = fresh()
            assert actual[0] is shared
            assert actual[1] is actual[2] is replacement
