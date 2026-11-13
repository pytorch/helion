from __future__ import annotations

import ast
from dataclasses import dataclass
from types import ModuleType
from unittest.mock import patch

import pytest
import torch

import helion
from helion._testing import cpu_render_target
from helion._testing import onlyBackends
from helion.autotuner import BooleanFragment
from helion.autotuner.config_fragment import ConfigSpecFragment
import helion.language as hl
from helion.runtime.kernel import BoundKernel

pytestmark = pytest.mark.cpu_only

GLOBAL_FRAGMENT = BooleanFragment()
ANCHORS = ModuleType("fragment_anchors")
ANCHORS.fragment = BooleanFragment()
REVISITED_ANCHORS = (BooleanFragment(), BooleanFragment())


@helion.kernel(autotune_effort="none")
def _compare_updated_literal(x, form: hl.constexpr, op: hl.constexpr):
    def compare(value):
        if op == "eq":
            take = value == 1
        else:
            take = value < 2
        return take

    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        value = 0
        value += 1
        if form == "tuple":
            values = (value, 3)
            value = values[0]
        if form == "helper":
            take = compare(value)
        elif op == "eq":
            take = value == 1
        else:
            take = value < 2
        if take:
            out[tile] = x[tile] + 2.0
        else:
            out[tile] = x[tile] + 5.0
    return out


@onlyBackends(["triton", "cute"])
@pytest.mark.parametrize("form", ["direct", "tuple", "helper"])
@pytest.mark.parametrize("op", ["eq", "lt"])
def test_updated_literal_comparison_has_no_fragment_guard(form, op):
    kernel = helion.kernel(_compare_updated_literal.fn, autotune_effort="none")
    with cpu_render_target():
        bound = kernel.bind((torch.empty(17), form, op))
        assert not any(
            name.startswith("fragment_comparison:")
            for name in bound.env.runtime_input_specializations
        )
        additions = [
            node.args[1]
            for graph in bound.host_function.device_ir.graphs
            for node in graph.graph.nodes
            if node.target is torch.ops.aten.add.Tensor
        ]
        assert additions == [2.0]
        ast.parse(bound.to_code())


@helion.kernel(autotune_effort="none")
def _compare_anchor(x, value: hl.constexpr, nested: hl.constexpr, op: hl.constexpr):
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        if op == 0:
            if nested:
                equal = value[0][0] == GLOBAL_FRAGMENT
            else:
                equal = value == GLOBAL_FRAGMENT
        elif op == 1:
            if nested:
                equal = value[0][0] != GLOBAL_FRAGMENT
            else:
                equal = value != GLOBAL_FRAGMENT
        elif op == 2:
            if nested:
                equal = value[0][0] is ANCHORS.fragment
            else:
                equal = value is ANCHORS.fragment
        else:
            if nested:
                equal = ANCHORS.fragment is not value[0][0]
            else:
                equal = ANCHORS.fragment is not value
        if equal:
            out[tile] = x[tile] + 2.0
        else:
            out[tile] = x[tile] + 5.0
    return out


@onlyBackends(["triton", "cute"])
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("op", [0, 1, 2, 3])
def test_argument_anchor_comparison_routes_cached_and_prepared_calls(nested, op):
    # Eight cases cover all equality/identity operators and direct/nested inputs.
    # Reverse operands and module globals are represented in the source above.
    kernel = helion.kernel(
        _compare_anchor.fn,
        autotune_effort="none",
        key=(lambda *args: 0) if op % 2 else None,
    )
    tensor = torch.empty(37)
    anchor = GLOBAL_FRAGMENT if op < 2 else ANCHORS.fragment

    def args(fragment):
        value = hl.constexpr(((fragment,),)) if nested else fragment
        return tensor, value, nested, op

    def observe_bound(bound, *unused_args):
        bound._run = lambda *run_args: bound
        return bound._run(*unused_args)

    with (
        cpu_render_target(),
        patch.object(BoundKernel, "__call__", observe_bound),
    ):
        first = kernel(*args(anchor))
        distinct = args(BooleanFragment())
        second = kernel(*distinct)
        fresh = helion.kernel(_compare_anchor.fn, autotune_effort="none")
        assert second.to_code() == fresh.bind(distinct).to_code()
        assert first is not second
        assert first.to_code() != second.to_code()
        with patch.object(kernel, "bind", side_effect=AssertionError("cache miss")):
            assert kernel(*args(BooleanFragment())) is second
            assert kernel(*args(anchor)) is first


@dataclass(frozen=True)
class _ValueFragment(ConfigSpecFragment):
    value: int = 3

    def default(self):
        return self.value


CUSTOM_FRAGMENT = _ValueFragment()


@helion.kernel(autotune_effort="none")
def _compare_custom_anchor(x, fragment: hl.constexpr):
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        if fragment == CUSTOM_FRAGMENT:
            out[tile] = x[tile] + 2.0
        else:
            out[tile] = x[tile] + 5.0
    return out


@onlyBackends(["triton", "cute"])
def test_custom_fragment_equality_keeps_value_semantics():
    kernel = helion.kernel(_compare_custom_anchor.fn, autotune_effort="none")
    tensor = torch.empty(37)
    with cpu_render_target():
        first = kernel.bind((tensor, CUSTOM_FRAGMENT))
        assert kernel.bind((tensor, _ValueFragment())) is first
        different = kernel.bind((tensor, _ValueFragment(7)))
        assert different.to_code() != first.to_code()


@onlyBackends(["triton", "cute"])
def test_indexed_comparison_unwraps_nested_constexpr():
    kernel = helion.kernel(_compare_anchor.fn, autotune_effort="none")
    tensor = torch.empty(37)

    def args(fragment):
        return tensor, hl.constexpr((hl.constexpr((fragment,)),)), True, 0

    with cpu_render_target():
        shared = kernel.bind(args(GLOBAL_FRAGMENT))
        distinct = args(BooleanFragment())
        other = kernel.bind(distinct)
        assert shared.to_code() != other.to_code()
        assert kernel.bind(distinct) is other


@helion.kernel(autotune_effort="none")
def _compare_dictionary_anchor(x, values):
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        if values["fragment"] == GLOBAL_FRAGMENT:
            out[tile] = x[tile] + 2.0
        else:
            out[tile] = x[tile] + 5.0
    return out


@onlyBackends(["triton", "cute"])
def test_dictionary_index_retains_fragment_anchor_guard():
    kernel = helion.kernel(_compare_dictionary_anchor.fn, autotune_effort="none")
    tensor = torch.empty(37)
    with cpu_render_target():
        shared = kernel.bind((tensor, {"fragment": hl.constexpr(GLOBAL_FRAGMENT)}))
        distinct = (tensor, {"fragment": hl.constexpr(BooleanFragment())})
        other = kernel.bind(distinct)
        fresh = helion.kernel(_compare_dictionary_anchor.fn, autotune_effort="none")
        assert other.to_code() == fresh.bind(distinct).to_code()
        assert shared.to_code() != other.to_code()


@helion.kernel(autotune_effort="none")
def _compare_revisited_anchors(x, value: hl.constexpr):
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        result = x[tile]
        for anchor in REVISITED_ANCHORS:
            if value == anchor:
                result = result + 2.0
            else:
                result = result + 5.0
        out[tile] = result
    return out


@onlyBackends(["triton", "cute"])
def test_revisited_comparison_guards_every_captured_anchor():
    kernel = helion.kernel(_compare_revisited_anchors.fn, autotune_effort="none")
    tensor = torch.empty(37)

    def comparison_guards(bound):
        return {
            key: guard
            for key, guard in bound.env.runtime_input_specializations.items()
            if key.startswith("fragment_comparison:")
        }

    def observe_bound(bound, *args):
        guards = comparison_guards(bound)
        assert len(guards) == 2
        bound.to_code()
        assert comparison_guards(bound) == guards
        bound._run = lambda *run_args: bound
        return bound._run(*args)

    with (
        cpu_render_target(),
        patch.object(BoundKernel, "__call__", observe_bound),
    ):
        fragments = (BooleanFragment(), *REVISITED_ANCHORS)
        bounds = [kernel(tensor, fragment) for fragment in fragments]
        assert len({id(bound) for bound in bounds}) == 3
        for fragment, bound in zip(fragments, bounds, strict=True):
            guards = comparison_guards(bound)
            assert len(guards) == 2
            classifiers = tuple(guard.classifier for guard in guards.values())
            fresh = helion.kernel(_compare_revisited_anchors.fn, autotune_effort="none")
            assert bound.to_code() == fresh.bind((tensor, fragment)).to_code()
            assert comparison_guards(bound) == guards
            assert all(
                guard.classifier is classifier
                for guard, classifier in zip(
                    comparison_guards(bound).values(),
                    classifiers,
                    strict=True,
                )
            )
        with patch.object(kernel, "bind", side_effect=AssertionError("cache miss")):
            for fragment, bound in zip(fragments, bounds, strict=True):
                assert kernel(tensor, fragment) is bound
            assert kernel(tensor, BooleanFragment()) is bounds[0]


@helion.kernel(autotune_effort="none")
def _compare_iterated_arguments(x, values: hl.constexpr, mode: hl.constexpr):
    def matches(fragment):
        return fragment == GLOBAL_FRAGMENT

    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        result = x[tile]
        if mode == "indexed":
            for index in hl.static_range(2):
                if values[index] == GLOBAL_FRAGMENT:
                    result = result + 2.0
                else:
                    result = result + 5.0
        elif mode == "nested":
            for group in values:
                for fragment in group:
                    alias = fragment
                    if alias == GLOBAL_FRAGMENT:
                        result = result + 2.0
                    else:
                        result = result + 5.0
        elif mode == "call":
            for fragment in values:
                if matches(values[0]):
                    result = result + 2.0
                else:
                    result = result + 5.0
                if fragment == GLOBAL_FRAGMENT:
                    result = result + 2.0
                else:
                    result = result + 5.0
        elif mode == "comprehension":
            for fragment in values:
                flags = [fragment == GLOBAL_FRAGMENT for fragment in values]
                if flags[0]:
                    result = result + 2.0
                else:
                    result = result + 5.0
                if fragment == GLOBAL_FRAGMENT:
                    result = result + 2.0
                else:
                    result = result + 5.0
        elif mode == "constructed":
            sequence = (values[0], values[1])
            alias = sequence
            for fragment in alias:
                if fragment == GLOBAL_FRAGMENT:
                    result = result + 2.0
                else:
                    result = result + 5.0
        elif mode == "list_alias":
            sequence = [values[0], values[0]]
            alias = sequence
            alias[1] = values[1]
            for fragment in sequence:
                if fragment == GLOBAL_FRAGMENT:
                    result = result + 2.0
                else:
                    result = result + 5.0
        elif mode == "comprehension_values":
            # Exercise comprehension tracing, rather than the list constructor.
            sequence = [fragment for fragment in values]  # noqa: C416
            for fragment in sequence:
                if fragment == GLOBAL_FRAGMENT:
                    result = result + 2.0
                else:
                    result = result + 5.0
        else:
            for fragment in values:
                if fragment == GLOBAL_FRAGMENT:
                    result = result + 2.0
                else:
                    result = result + 5.0
        out[tile] = result
    return out


@onlyBackends(["triton", "cute"])
@pytest.mark.parametrize(
    "mode",
    [
        "direct",
        "indexed",
        "nested",
        "call",
        "comprehension",
        "constructed",
        "list_alias",
        "comprehension_values",
    ],
)
def test_iterated_arguments_retain_actual_comparison_paths(mode):
    kernel = helion.kernel(_compare_iterated_arguments.fn, autotune_effort="none")
    tensor = torch.empty(17)

    def args(anchor_index):
        fragments = tuple(
            GLOBAL_FRAGMENT if index == anchor_index else BooleanFragment()
            for index in range(2)
        )
        if mode == "nested":
            fragments = tuple((fragment,) for fragment in fragments)
        return tensor, hl.constexpr(fragments), mode

    def guards(bound):
        return tuple(
            guard.classifier
            for key, guard in bound.env.runtime_input_specializations.items()
            if key.startswith("fragment_comparison:")
        )

    def observe(bound, *values):
        before = guards(bound)
        bound.to_code()
        assert all(a is b for a, b in zip(before, guards(bound), strict=True))
        bound._run = lambda *unused: bound
        return bound._run(*values)

    with cpu_render_target(), patch.object(BoundKernel, "__call__", observe):
        # All inputs have distinct elements: only their relationship to the
        # captured anchor changes, not the across-argument alias partition.
        bounds = [kernel(*args(index)) for index in (1, 0, -1)]
        assert len({id(bound) for bound in bounds}) == 3
        for index, bound in zip((1, 0, -1), bounds, strict=True):
            fresh = helion.kernel(
                _compare_iterated_arguments.fn, autotune_effort="none"
            )
            assert bound.to_code() == fresh.bind(args(index)).to_code()
        with patch.object(kernel, "bind", side_effect=AssertionError("cache miss")):
            for index, bound in zip((1, 0, -1), bounds, strict=True):
                assert kernel(*args(index)) is bound


@helion.kernel(autotune_effort="none")
def _compare_fragment_helper(x, values: hl.constexpr):
    def identity(fragment):
        return fragment

    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        result = x[tile]
        for fragment in values:
            if identity(fragment) is GLOBAL_FRAGMENT:
                result = result + 2.0
            else:
                result = result + 5.0
        out[tile] = result
    return out


@onlyBackends(["triton", "cute"])
def test_fragment_returning_helper_preserves_each_value_and_input_path():
    kernel = helion.kernel(_compare_fragment_helper.fn, autotune_effort="none")
    tensor = torch.empty(17)

    def args(anchor_index):
        return tensor, hl.constexpr(
            tuple(
                GLOBAL_FRAGMENT if index == anchor_index else BooleanFragment()
                for index in range(2)
            )
        )

    def observe(bound, *values):
        before = tuple(bound.env.runtime_input_specializations.values())
        bound.to_code()
        assert tuple(bound.env.runtime_input_specializations.values()) == before
        bound._run = lambda *unused: bound
        return bound._run(*values)

    with cpu_render_target(), patch.object(BoundKernel, "__call__", observe):
        bounds = [kernel(*args(index)) for index in (0, 1, -1)]
        assert len({id(bound) for bound in bounds}) == 3
        for index, bound in zip((0, 1, -1), bounds, strict=True):
            additions = [
                node.args[1]
                for graph in bound.host_function.device_ir.graphs
                for node in graph.graph.nodes
                if node.op == "call_function"
                and node.target is torch.ops.aten.add.Tensor
            ]
            assert additions == (
                [2.0, 5.0] if index == 0 else [5.0, 2.0] if index == 1 else [5.0, 5.0]
            )
            fresh = helion.kernel(_compare_fragment_helper.fn, autotune_effort="none")
            assert bound.to_code() == fresh.bind(args(index)).to_code()
        with patch.object(kernel, "bind", side_effect=AssertionError("cache miss")):
            for index, bound in zip((0, 1, -1), bounds, strict=True):
                assert kernel(*args(index)) is bound
