from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

import helion
from helion._testing import cpu_render_target
from helion._testing import onlyBackends
from helion._testing.binding_observation import observe_compiled_binding
from helion.autotuner import BooleanFragment
from helion.autotuner import EnumFragment
import helion.language as hl
from helion.runtime.kernel import BoundKernel

pytestmark = pytest.mark.cpu_only
ANCHOR = BooleanFragment()
SAVED = (ANCHOR,)
NESTED = ((ANCHOR,),)


@helion.kernel(autotune_effort="none")
def _compare_container(x, values: hl.constexpr, mode: hl.constexpr):
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        if mode == "tuple_eq":
            matches = values == SAVED
        elif mode == "nested_eq":
            matches = values == NESTED
        else:
            matches = values == SAVED
        if matches:
            out[tile] = x[tile] + 2.0
        else:
            out[tile] = x[tile] + 5.0
    return out


@onlyBackends(["triton", "cute"])
@pytest.mark.parametrize("mode", ["tuple_eq", "nested_eq", "reverse_eq"])
def test_container_comparison_routes_cached_calls(mode):
    kernel = helion.kernel(
        _compare_container.fn,
        autotune_effort="none",
        key=(lambda *args: 0) if mode == "nested_eq" else None,
    )
    tensor = torch.empty(17)

    def args(shared):
        fragment = ANCHOR if shared else BooleanFragment()
        if mode == "nested_eq":
            value = NESTED if shared else ((fragment,),)
        else:
            value = SAVED if shared else (fragment,)
        return tensor, hl.constexpr(value), mode

    with (
        cpu_render_target(),
        patch.object(BoundKernel, "__call__", observe_compiled_binding),
    ):
        first = kernel(*args(True))
        second = kernel(*args(False))
        fresh = helion.kernel(_compare_container.fn, autotune_effort="none")
        assert second.to_code() == fresh.bind(args(False)).to_code()
        assert first is not second
        assert first.to_code() != second.to_code()
        with patch.object(kernel, "bind", side_effect=AssertionError("cache miss")):
            assert kernel(*args(True)) is first
            assert kernel(*args(False)) is second


@helion.kernel(autotune_effort="none")
def _compare_iterated_containers(x, groups: hl.constexpr):
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        result = x[tile]
        for group in groups:
            if group == SAVED:
                result = result + 2.0
            else:
                result = result + 5.0
        out[tile] = result
    return out


@onlyBackends(["triton", "cute"])
def test_iterated_container_equality_retains_actual_input_paths():
    tensor = torch.empty(17)
    kernel = helion.kernel(_compare_iterated_containers.fn, autotune_effort="none")

    def args(shared):
        first = (BooleanFragment(),)
        second = SAVED if shared else (BooleanFragment(),)
        return tensor, hl.constexpr((first, second))

    with (
        cpu_render_target(),
        patch.object(BoundKernel, "__call__", observe_compiled_binding),
    ):
        first = kernel(*args(True))
        other = args(False)
        second = kernel(*other)
        fresh = helion.kernel(_compare_iterated_containers.fn, autotune_effort="none")
        assert second.to_code() == fresh.bind(other).to_code()
        assert first is not second
        assert first.to_code() != second.to_code()
        with patch.object(kernel, "bind", side_effect=AssertionError("cache miss")):
            assert kernel(*args(True)) is first
            assert kernel(*args(False)) is second


def _mapping_fragment(mapping):  # noqa: FURB118 - exercise an opaque Python helper
    return mapping["key"]


@helion.kernel(autotune_effort="none")
def _constructed_mapping(x, values: hl.constexpr, mode: hl.constexpr):
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        result = x[tile]
        for fragment in values:
            if mode == "integer":
                holder = {0: fragment}
                alias = holder
                chosen = alias[0]
            elif mode == "nested":
                holder = {"outer": {"key": fragment}}
                alias = holder
                chosen = alias["outer"]["key"]
            elif mode == "comprehension":
                # Exercise DictComp origin propagation rather than a constructor call.
                holder = {key: fragment for key in ("key",)}  # noqa: C420
                alias = holder
                chosen = alias["key"]
            elif mode == "helper":
                holder = {"key": fragment}
                alias = holder
                chosen = _mapping_fragment(alias)
            else:
                holder = {"key": fragment}
                alias = holder
                chosen = alias["key"]
            if chosen == ANCHOR:
                result = result + 2.0
            else:
                result = result + 5.0
        out[tile] = result
    return out


@onlyBackends(["triton", "cute"])
@pytest.mark.parametrize(
    "mode", ["literal", "integer", "nested", "comprehension", "helper"]
)
def test_constructed_mapping_preserves_each_iterated_input_origin(mode):
    tensor = torch.empty(17)
    first_values = (BooleanFragment(), BooleanFragment())
    changed_values = (first_values[0], ANCHOR)
    inputs = [(tensor, values, mode) for values in (first_values, changed_values)]

    def additions(bound):
        return [
            node.args[1]
            for graph in bound.host_function.device_ir.graphs
            for node in graph.graph.nodes
            if node.target is torch.ops.aten.add.Tensor
        ]

    with (
        cpu_render_target(),
        patch.object(BoundKernel, "__call__", observe_compiled_binding),
    ):
        kernel = helion.kernel(_constructed_mapping.fn, autotune_effort="none")
        first = kernel(*inputs[0])
        fresh = helion.kernel(_constructed_mapping.fn, autotune_effort="none").bind(
            inputs[1]
        )
        second = kernel(*inputs[1])
        assert additions(first) == [5.0, 5.0]
        assert additions(fresh) == [5.0, 2.0]
        assert additions(second) == [5.0, 2.0]
        assert first is not second
        prepared = kernel._prepared_call
        assert prepared is not None
        assert not prepared.matches(kernel, inputs[0])
        with patch.object(kernel, "bind", side_effect=AssertionError("cache miss")):
            assert kernel(*inputs[0]) is first
            assert kernel(*inputs[1]) is second


@helion.kernel(autotune_effort="none", static_shapes=False)
def _inline_fragment_comparison(x, fragment: hl.constexpr):
    size = hl.specialize(x.size(0))
    out = torch.empty_like(x)
    for tile in hl.tile(size):
        matches = fragment == EnumFragment((1, 2))
        out[tile] = x[tile] + (2.0 if matches else 5.0)
    return out


@onlyBackends(["triton", "cute"])
def test_inline_fragment_value_comparison_reuses_each_specialized_size():
    kernel = helion.kernel(
        _inline_fragment_comparison.fn, autotune_effort="none", static_shapes=False
    )
    fragment = EnumFragment((1, 2))
    tensors = (torch.empty(17), torch.empty(33))
    with cpu_render_target():
        first, second = [kernel.bind((tensor, fragment)) for tensor in tensors]
        assert first is not second
        assert first._base_spec_key == second._base_spec_key
        assert not first.env.requires_fresh_binding
        assert not second.env.requires_fresh_binding
        for tensor, expected in zip(tensors, (first, second), strict=True):
            assert kernel.bind((tensor, EnumFragment((1, 2)))) is expected
        other = kernel.bind((tensors[0], EnumFragment((2, 3))))
        assert other.to_code() != first.to_code()


VALUE_ANCHOR = EnumFragment((1, 2))


@helion.kernel(autotune_effort="none")
def _captured_value_comparison(x, fragment: hl.constexpr):
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        out[tile] = x[tile] + (2.0 if fragment == VALUE_ANCHOR else 5.0)
    return out


@onlyBackends(["triton", "cute"])
def test_value_comparison_guard_still_observes_captured_anchor_mutation(monkeypatch):
    kernel = helion.kernel(_captured_value_comparison.fn, autotune_effort="none")
    args = (torch.empty(17), EnumFragment((1, 2)))
    with cpu_render_target():
        first = kernel.bind(args)
        assert kernel.bind(args) is first
        monkeypatch.setattr(VALUE_ANCHOR, "choices", (2, 3))
        changed = kernel.bind(args)
        assert changed is not first
        assert changed.to_code() != first.to_code()
        fresh = helion.kernel(_captured_value_comparison.fn, autotune_effort="none")
        assert changed.to_code() == fresh.bind(args).to_code()
