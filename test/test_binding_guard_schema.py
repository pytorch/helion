from __future__ import annotations

import ast
from dataclasses import dataclass
import gc
from typing import NamedTuple
from unittest.mock import patch
import weakref

import pytest
import torch

import helion
from helion._compiler.compile_environment import RuntimeInputSpecialization
from helion._testing import cpu_render_target
from helion._testing import onlyBackends
from helion._testing.binding_observation import (
    observe_compiled_binding as _compile_observer,
)
from helion._testing.binding_observation import scalar_additions as _offset
from helion.autotuner import BooleanFragment
import helion.language as hl
from helion.runtime.kernel import BoundKernel

pytestmark = pytest.mark.cpu_only
FIRST = BooleanFragment()
SECOND = BooleanFragment()
SELECTOR = BooleanFragment()
FIRST_CONTAINER = (FIRST,)
SECOND_CONTAINER = (SECOND,)


@helion.kernel(autotune_effort="none")
def _nested_fragment(x, a: hl.constexpr, b: hl.constexpr):
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        if a == FIRST:
            if b == SECOND:
                out[tile] = x[tile] + 2.0
            else:
                out[tile] = x[tile] + 3.0
        else:
            out[tile] = x[tile] + 5.0
    return out


@helion.kernel(autotune_effort="none")
def _nested_container(x, a: hl.constexpr, b: hl.constexpr):
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        if a is FIRST_CONTAINER:
            if b is SECOND_CONTAINER:
                out[tile] = x[tile] + 2.0
            else:
                out[tile] = x[tile] + 3.0
        else:
            out[tile] = x[tile] + 5.0
    return out


@onlyBackends(["triton", "cute"])
@pytest.mark.parametrize("container", [False, True])
def test_later_branch_preserves_new_guard_in_eager_and_direct_calls(container):
    if container:
        fn = _nested_container.fn
        first, second = FIRST_CONTAINER, SECOND_CONTAINER
        other_first, other_second = tuple([FIRST]), tuple([SECOND])
    else:
        fn = _nested_fragment.fn
        first, second = FIRST, SECOND
        other_first, other_second = BooleanFragment(), BooleanFragment()
    kernel = helion.kernel(fn, autotune_effort="none")
    x = torch.empty(17)
    shallow_args = (x, other_first, second)
    deep_args = (x, first, second)
    changed_args = (x, first, other_second)
    with (
        cpu_render_target(),
        patch.object(BoundKernel, "ensure_config_exists", _compile_observer),
    ):
        shallow = kernel(*shallow_args)
        assert _offset(shallow) == [5.0]
        prepared = kernel._prepared_call
        assert prepared is not None and prepared.matches(kernel, shallow_args)
        assert not prepared.matches(kernel, deep_args)
        deep = kernel(*deep_args)
        assert _offset(deep) == [2.0]
        changed = kernel(*changed_args)
        fresh = helion.kernel(fn, autotune_effort="none").bind(changed_args)
        assert _offset(changed) == _offset(fresh) == [3.0]
        assert _offset(deep(*changed_args)) == [3.0]
        assert _offset(shallow(*deep_args)) == [2.0]
        rebound = kernel(*shallow_args)
        assert _offset(rebound) == [5.0]
        assert kernel(*shallow_args) is rebound


@helion.kernel(autotune_effort="none")
def _selected_anchor(x, selector: hl.constexpr, value: hl.constexpr):
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        if selector == SELECTOR:
            anchors = (FIRST,)
        else:
            anchors = (SECOND,)
        for anchor in anchors:
            if value == anchor:
                out[tile] = x[tile] + 2.0
            else:
                out[tile] = x[tile] + 5.0
    return out


@onlyBackends(["triton", "cute"])
@pytest.mark.parametrize("container", [False, True])
def test_shallow_first_call_does_not_force_repeated_deep_traces(container):
    fn = _nested_container.fn if container else _nested_fragment.fn
    first, second = (
        (FIRST_CONTAINER, SECOND_CONTAINER) if container else (FIRST, SECOND)
    )
    other = tuple([FIRST]) if container else BooleanFragment()
    kernel = helion.kernel(fn, autotune_effort="none")
    x = torch.empty(17)
    with cpu_render_target():
        shallow = kernel.bind((x, other, second))
        assert _offset(shallow) == [5.0]
        deep = kernel.bind((x, first, second))
        assert _offset(deep) == [2.0]
        for _ in range(4):
            assert kernel.bind((x, first, second)) is deep
        assert not deep.env.requires_fresh_binding


@onlyBackends(["triton", "cute"])
def test_same_site_guard_with_different_anchor_is_not_reused():
    kernel = helion.kernel(_selected_anchor.fn, autotune_effort="none")
    x = torch.empty(17)
    other_selector = BooleanFragment()
    other_value = BooleanFragment()
    with (
        cpu_render_target(),
        patch.object(BoundKernel, "ensure_config_exists", _compile_observer),
    ):
        first = kernel(x, other_selector, FIRST)
        assert _offset(first) == [5.0]
        selected = kernel(x, SELECTOR, FIRST)
        assert _offset(selected) == [2.0]
        args = (x, SELECTOR, other_value)
        fresh = helion.kernel(_selected_anchor.fn, autotune_effort="none").bind(args)
        assert _offset(kernel(*args)) == _offset(fresh) == [5.0]
        assert _offset(selected(*args)) == [5.0]
        assert kernel(x, other_selector, FIRST) is first


@dataclass(frozen=True)
class _DataclassPair:
    a: object
    b: object


class _NamedPair(NamedTuple):
    a: object
    b: object


@helion.kernel(autotune_effort="none")
def _field_fragment_equality(x, holder):
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        if holder.a == holder.b:
            out[tile] = x[tile] + 2.0
        else:
            out[tile] = x[tile] + 5.0
    return out


@onlyBackends(["triton", "cute"])
@pytest.mark.parametrize("holder_type", [_DataclassPair, _NamedPair])
def test_fragment_field_aliases_preserve_cached_and_direct_routes(holder_type):
    kernel = helion.kernel(_field_fragment_equality.fn, autotune_effort="none")
    x = torch.empty(17)
    shared = holder_type(hl.constexpr(FIRST), hl.constexpr(FIRST))
    distinct = holder_type(hl.constexpr(FIRST), hl.constexpr(BooleanFragment()))
    with (
        cpu_render_target(),
        patch.object(BoundKernel, "ensure_config_exists", _compile_observer),
    ):
        first = kernel(x, shared)
        assert _offset(first) == [2.0]
        prepared = kernel._prepared_call
        if prepared is not None:
            assert prepared.matches(kernel, (x, shared))
            assert not prepared.matches(kernel, (x, distinct))
        second = kernel(x, distinct)
        fresh = helion.kernel(_field_fragment_equality.fn, autotune_effort="none")
        assert _offset(second) == _offset(fresh.bind((x, distinct))) == [5.0]
        assert _offset(first(x, distinct)) == [5.0]
        assert _offset(second(x, shared)) == [2.0]
        assert kernel(x, shared) is first
        assert kernel(x, distinct) is second


@helion.kernel(autotune_effort="none")
def _captured_tuple_constructor(x, value: hl.constexpr, host: hl.constexpr):
    if host:
        captured = tuple(FIRST_CONTAINER)
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        if not host:
            captured = tuple(FIRST_CONTAINER)
        if value is captured:
            out[tile] = x[tile] + 2.0
        else:
            out[tile] = x[tile] + 5.0
    return out


@onlyBackends(["triton", "cute"])
@pytest.mark.parametrize("host", [False, True])
def test_identity_preserving_tuple_constructor_keeps_capture_origin(host):
    kernel = helion.kernel(_captured_tuple_constructor.fn, autotune_effort="none")
    x = torch.empty(17)
    same = (x, FIRST_CONTAINER, host)
    other = (x, tuple([FIRST]), host)
    with (
        cpu_render_target(),
        patch.object(BoundKernel, "ensure_config_exists", _compile_observer),
    ):
        first = kernel(*same)
        assert _offset(first) == [2.0]
        second = kernel(*other)
        fresh = helion.kernel(_captured_tuple_constructor.fn, autotune_effort="none")
        assert _offset(second) == _offset(fresh.bind(other)) == [5.0]
        assert _offset(first(*other)) == [5.0]
        assert kernel(*same) is first
        assert kernel(*other) is second


@helion.kernel(autotune_effort="none")
def _ordinary(x):
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        out[tile] = x[tile] + 2.0
    return out


@onlyBackends(["triton", "cute"])
def test_ordinary_direct_call_does_not_add_a_bind():
    kernel = helion.kernel(_ordinary.fn, autotune_effort="none")
    x = torch.empty(17)
    with (
        cpu_render_target(),
        patch.object(BoundKernel, "ensure_config_exists", _compile_observer),
    ):
        bound = kernel(x)
        with patch.object(kernel, "bind", side_effect=AssertionError("extra bind")):
            assert bound(x) is bound


@helion.kernel(autotune_effort="none")
def _selected_input_container(x, values: hl.constexpr, index: hl.constexpr):
    selected = values[index]
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        if selected is values[0]:
            out[tile] = x[tile] + 2.0
        else:
            out[tile] = x[tile] + 5.0
    return out


@onlyBackends(["triton", "cute"])
def test_complete_partition_keeps_specialized_source_selection():
    kernel = helion.kernel(_selected_input_container.fn, autotune_effort="none")
    x = torch.empty(17)
    left, right = tuple([4, 8]), tuple([4, 8])
    calls = [
        ((x, (left, right), 0), 2.0),
        ((x, (left, right), 1), 5.0),
        ((x, (left, left), 1), 2.0),
    ]
    with (
        cpu_render_target(),
        patch.object(BoundKernel, "ensure_config_exists", _compile_observer),
    ):
        bounds = []
        for args, expected in calls:
            bound = kernel(*args)
            assert _offset(bound) == [expected]
            bounds.append(bound)
        for (args, expected), bound in zip(calls, bounds, strict=True):
            assert kernel(*args) is bound
            assert _offset(bounds[0](*args)) == [expected]


def test_binding_anchor_witness_retains_identity_without_repr_or_equality():
    class Anchor:
        def __eq__(self, other):
            raise AssertionError("anchor equality is not a semantic identity")

        def __repr__(self):
            raise AssertionError("anchor is not persistent metadata")

    anchor = Anchor()
    reference = weakref.ref(anchor)
    specialization = RuntimeInputSpecialization(
        sources=(),
        classifier_identity="anchor",
        classifier=lambda values: True,
        binding_anchors=(anchor,),
    )
    same = RuntimeInputSpecialization(
        sources=(),
        classifier_identity="anchor",
        classifier=lambda values: True,
        binding_anchors=(anchor,),
    )
    different = RuntimeInputSpecialization(
        sources=(),
        classifier_identity="anchor",
        classifier=lambda values: True,
        binding_anchors=(Anchor(),),
    )
    assert specialization.same_binding_semantics(same)
    assert not specialization.same_binding_semantics(different)
    assert repr(specialization) == repr(different)
    del anchor, same
    gc.collect()
    assert reference() is not None
    del specialization
    gc.collect()
    assert reference() is None


@onlyBackends(["triton", "cute"])
def test_multi_shape_keeps_selection_after_later_guard_requires_fresh_binding():
    tensor = torch.empty(17)
    shallow_args = (tensor, BooleanFragment(), FIRST)
    deep_args = (tensor, SELECTOR, FIRST)

    def compile_observer(bound, config, *, allow_print=True):
        ast.parse(bound.to_code(config))
        return lambda *args: bound

    with (
        cpu_render_target(),
        patch.object(BoundKernel, "compile_config", compile_observer),
        patch(
            "helion.runtime.kernel._canonicalize_multi_shape_device",
            return_value=torch.device("cuda", 0),
        ),
        patch("helion.runtime.kernel._current_device_index", return_value=0),
    ):
        kernel = helion.kernel(_selected_anchor.fn)
        initial = kernel.bind(shallow_args)
        probe = kernel.bind(deep_args)
        assert not initial.env.requires_fresh_binding
        assert probe.env.requires_fresh_binding
        assert (
            kernel._get_bound_kernel_cache_key(
                deep_args, kernel._base_specialization_key(deep_args)
            )
            is not None
        )
        with probe.env:
            selected = probe.config_spec.default_config()
        selected.config["block_sizes"] = [64]
        with patch.object(
            type(probe.env.backend), "autotune", return_value=selected
        ) as tune:
            assert (
                kernel.autotune_multi([deep_args], cache_tag="later-guard") == selected
            )
            assert len(kernel._fresh_selections) == 1
            assert kernel(*deep_args)._config == selected
            assert kernel(*deep_args)._config == selected
            assert tune.call_count == 1
