from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

import helion
from helion._compiler.tuple_constants import TupleConstants
from helion._testing import cpu_render_target
from helion._testing import onlyBackends
from helion._testing.binding_observation import assert_scalar_addition
from helion._testing.binding_observation import observe_compiled_binding
import helion.language as hl
from helion.runtime.kernel import BoundKernel
from helion.runtime.ref_mode import RefMode

pytestmark = pytest.mark.cpu_only


def _distinct_tuple(value):
    elements = list(value)
    result = tuple(elements)
    assert result == value and result is not value
    return result


def literal_identity(x, values):
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        local = (4, 8)
        if local is values:
            out[tile] = x[tile] + 2.0
        else:
            out[tile] = x[tile] + 5.0
    return out


@onlyBackends(["triton", "cute"])
@pytest.mark.parametrize("alias", [True, False])
def test_nonempty_tuple_literal_identity_matches_eager(alias):
    constant = next(
        value
        for value in literal_identity.__code__.co_consts
        if type(value) is tuple and value == (4, 8)
    )
    values = constant if alias else tuple([4, 8])
    assert (values is constant) is alias
    x = torch.zeros(17)
    expected = 2.0 if alias else 5.0
    with cpu_render_target():
        eager = helion.kernel(
            literal_identity, autotune_effort="none", ref_mode=RefMode.EAGER
        )
        result = eager(x, values)
        torch.testing.assert_close(result, x + expected, rtol=0, atol=0)
        compiled = helion.kernel(literal_identity, autotune_effort="none")
        bound = compiled.bind((x, values))
        assert_scalar_addition(bound, expected)


def _check_cached_routes(fn, aliased, distinct, *, expected=2.0):
    x = torch.zeros(17)
    with cpu_render_target():
        eager = helion.kernel(fn, autotune_effort="none", ref_mode=RefMode.EAGER)
        torch.testing.assert_close(eager(x, aliased), x + expected, rtol=0, atol=0)
        torch.testing.assert_close(eager(x, distinct), x + 5.0, rtol=0, atol=0)
        with patch.object(
            BoundKernel, "ensure_config_exists", observe_compiled_binding
        ):
            kernel = helion.kernel(fn, autotune_effort="none")
            first = kernel(x, aliased)
            second = kernel(x, distinct)
            assert_scalar_addition(first, expected)
            assert_scalar_addition(second, 5.0)
            assert not first.env.requires_fresh_binding
            assert not second.env.requires_fresh_binding
            prepared = kernel._prepared_call
            assert prepared is not None and prepared.matches(kernel, (x, distinct))
            assert not prepared.matches(kernel, (x, aliased))
            assert_scalar_addition(first(x, distinct), 5.0)
            assert_scalar_addition(second(x, aliased), expected)
            with patch.object(
                BoundKernel, "__init__", side_effect=AssertionError("cache miss")
            ):
                assert kernel(x, aliased) is first
                assert kernel(x, distinct) is second


@onlyBackends(["triton", "cute"])
def test_tuple_constant_anchor_survives_alias_then_distinct_calls():
    constant = next(
        value for value in literal_identity.__code__.co_consts if type(value) is tuple
    )
    _check_cached_routes(literal_identity, constant, _distinct_tuple(constant))


def nested_literal_identity(x, values):
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        local = ((4, 8), (-0.0, -2))
        if local[1] is values:
            out[tile] = x[tile] + 2.0
        else:
            out[tile] = x[tile] + 5.0
    return out


@onlyBackends(["triton", "cute"])
def test_nested_tuple_constant_preserves_subobject_identity():
    outer = next(
        value
        for value in nested_literal_identity.__code__.co_consts
        if type(value) is tuple
    )
    constant = outer[1]
    assert constant[0].hex() == "-0x0.0p+0" and constant[1] == -2
    _check_cached_routes(nested_literal_identity, constant, _distinct_tuple(constant))


def signed_zero_literal_identity(x, values):
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        negative = (-0.0, -2)
        positive = (+0.0, -2)
        # Observe both anchors in the first binding. The separate uncovered-
        # schema controls cover short-circuit paths that require fresh tracing.
        is_negative = negative is values
        is_positive = positive is values
        if is_negative:
            out[tile] = x[tile] + 2.0
        elif is_positive:
            out[tile] = x[tile] + 3.0
        else:
            out[tile] = x[tile] + 5.0
    return out


@onlyBackends(["triton", "cute"])
@pytest.mark.parametrize("positions", [True, False])
def test_signed_zero_tuple_constants_remain_distinct(positions):
    original_init = TupleConstants.__init__

    def init(resolver, code):
        original_init(resolver, code)
        if not positions:
            resolver.has_positions = False

    constants = [
        value
        for value in signed_zero_literal_identity.__code__.co_consts
        if type(value) is tuple
    ]
    negative = next(value for value in constants if value[0].hex() == "-0x0.0p+0")
    positive = next(value for value in constants if value[0].hex() == "0x0.0p+0")
    assert negative == positive and negative is not positive
    with patch.object(TupleConstants, "__init__", init):
        _check_cached_routes(
            signed_zero_literal_identity, negative, _distinct_tuple(negative)
        )
        _check_cached_routes(
            signed_zero_literal_identity,
            positive,
            _distinct_tuple(positive),
            expected=3.0,
        )


def dynamic_tuple_identity(x, values: hl.constexpr):
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        constant = (4, 8)
        local = (values[0], values[1])
        if local is values or local is constant:
            out[tile] = x[tile] + 2.0
        else:
            out[tile] = x[tile] + 5.0
    return out


@onlyBackends(["triton", "cute"])
def test_equal_runtime_tuple_is_not_replaced_by_code_constant():
    constant = next(
        value
        for value in dynamic_tuple_identity.__code__.co_consts
        if type(value) is tuple
    )
    x = torch.zeros(17)
    with cpu_render_target():
        eager = helion.kernel(
            dynamic_tuple_identity, autotune_effort="none", ref_mode=RefMode.EAGER
        )
        compiled = helion.kernel(dynamic_tuple_identity, autotune_effort="none")
        for value in (constant, _distinct_tuple(constant)):
            torch.testing.assert_close(eager(x, value), x + 5.0, rtol=0, atol=0)
            bound = compiled.bind((x, value))
            assert_scalar_addition(bound, 5.0)
            assert not bound.env.requires_fresh_binding
