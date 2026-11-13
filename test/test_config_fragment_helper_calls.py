from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace
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
from helion.autotuner import BooleanFragment
from helion.autotuner import IntegerFragment
from helion.autotuner.config_fragment import ConfigSpecFragment
import helion.language as hl
from helion.runtime.kernel import BoundKernel
from helion.runtime.kernel import _RuntimeInputSpecializationExtractor

pytestmark = pytest.mark.cpu_only

ANCHOR = BooleanFragment()
HELPER_CALLS = []


class _PickedFragment(IntegerFragment):
    def __init__(self, fragment):
        super().__init__(low=2, high=5, default_val=2 if fragment is ANCHOR else 5)


@helion.kernel(autotune_effort="none")
def _custom_constructor(x, fragment: hl.constexpr, mode: hl.constexpr):
    if mode == "host":
        factor = hl.register_tunable("factor", _PickedFragment(fragment))
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        if mode == "device":
            factor = hl.register_tunable("factor", _PickedFragment(fragment))
        out[tile] = x[tile] + factor
    return out


@onlyBackends(["triton", "cute"])
@pytest.mark.parametrize("eager", [False, True])
def test_opaque_fragment_bindings_release_evicted_owners(eager):
    kernel = helion.kernel(_custom_constructor.fn, autotune_effort="none")
    tensor = torch.empty(17)
    references = []

    def observe(bound, *args):
        bound.to_code()
        bound._run = lambda *unused: bound
        return bound

    with (
        cpu_render_target(),
        patch.object(BoundKernel, "__call__", observe),
        patch("helion.runtime.kernel._OPAQUE_BINDING_CACHE_SIZE", 2),
    ):
        for _ in range(4):
            fragment = BooleanFragment()
            references.append(weakref.ref(fragment))
            args = (tensor, fragment, "host")
            bound = kernel(*args) if eager else kernel.bind(args)
            assert bound.config_spec.default_config()["factor"] == 5
            assert (kernel(*args) if eager else kernel.bind(args)) is bound
            assert len(kernel._bound_kernels) <= 2
            del bound, args, fragment
        gc.collect()
        assert all(reference() is None for reference in references[:2])
        assert all(reference() is not None for reference in references[2:])
        kernel.reset()
        gc.collect()
        assert all(reference() is None for reference in references)


@helion.kernel(autotune_effort="none")
def _conditional_opaque_helper(x, selector: hl.constexpr, fragment: hl.constexpr):
    if selector == ANCHOR:
        factor = _PickedFragment(fragment).default_val
    else:
        factor = 5
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        out[tile] = x[tile] + factor
    return out


@onlyBackends(["triton", "cute"])
def test_inherited_opaque_guard_also_bounds_owner_retention():
    kernel = helion.kernel(_conditional_opaque_helper.fn, autotune_effort="none")
    tensor = torch.empty(17)
    selector = BooleanFragment()
    references = []
    with (
        cpu_render_target(),
        patch("helion.runtime.kernel._OPAQUE_BINDING_CACHE_SIZE", 1),
    ):
        first = kernel.bind((tensor, ANCHOR, BooleanFragment()))
        signature = first._base_spec_key
        del first
        for _ in range(3):
            fragment = BooleanFragment()
            references.append(weakref.ref(fragment))
            bound = kernel.bind((tensor, selector, fragment))
            assert bound._base_spec_key == signature
            assert bound.env.has_opaque_fragment_inputs
            assert len(kernel._bound_kernels) == 1
            del bound, fragment
    gc.collect()
    assert all(reference() is None for reference in references[:-1])
    assert references[-1]() is not None


@onlyBackends(["triton", "cute"])
def test_opaque_lifetime_upgrade_reaches_later_shallow_bindings():
    kernel = helion.kernel(_conditional_opaque_helper.fn, autotune_effort="none")
    tensor = torch.empty(17)
    selector = BooleanFragment()
    references = []

    def without_lifetime_fact(**kwargs):
        # Model a compatible projection published before the compiler learned
        # that its result can retain an input owner. Its classifier is exact.
        return replace(
            RuntimeInputSpecialization(**kwargs), retains_input_identity=False
        )

    with (
        cpu_render_target(),
        patch("helion.runtime.kernel._OPAQUE_BINDING_CACHE_SIZE", 1),
    ):
        with patch(
            "helion._compiler.fragment_call.RuntimeInputSpecialization",
            side_effect=without_lifetime_fact,
        ):
            first = kernel.bind((tensor, ANCHOR, BooleanFragment()))
        signature = first._base_spec_key
        old_schema = kernel._specialize_extra[signature]
        key = "fragment_call_input:fragment"
        old_fact = first.env.runtime_input_specializations[key]
        first.env.register_runtime_input_specialization(
            key, replace(old_fact, retains_input_identity=True)
        )
        first.env.register_runtime_input_specialization(key, old_fact)
        assert first.env.runtime_input_specializations[key].retains_input_identity
        del first
        upgraded = kernel.bind((tensor, ANCHOR, BooleanFragment()))
        assert not upgraded.env.requires_fresh_binding
        schema = kernel._specialize_extra[signature]
        assert schema is not old_schema and len(schema) == len(old_schema)
        for previous, current in zip(old_schema, schema, strict=True):
            if isinstance(previous, _RuntimeInputSpecializationExtractor):
                assert isinstance(current, _RuntimeInputSpecializationExtractor)
                assert current.classifier is previous.classifier
                assert current.source_extractors is previous.source_extractors
        assert any(
            isinstance(extractor, _RuntimeInputSpecializationExtractor)
            and extractor.specialization is not None
            and extractor.specialization.retains_input_identity
            for extractor in schema
        )
        del upgraded, schema, old_schema
        for _ in range(3):
            fragment = BooleanFragment()
            references.append(weakref.ref(fragment))
            bound = kernel.bind((tensor, selector, fragment))
            assert bound.env.has_opaque_fragment_inputs
            assert len(kernel._bound_kernels) == 1
            del bound, fragment
        gc.collect()
        assert all(reference() is None for reference in references[:-1])
        kernel.reset()
        gc.collect()
        assert all(reference() is None for reference in references)


@onlyBackends(["triton", "cute"])
@pytest.mark.parametrize("mode", ["host", "device"])
def test_custom_fragment_constructor_guards_identity_inputs(mode):
    tensor = torch.empty(17)
    first_args = (tensor, ANCHOR, mode)
    changed_args = (tensor, BooleanFragment(), mode)

    def observe(bound, *args):
        bound.to_code()
        bound._run = lambda *unused: bound
        return bound

    with cpu_render_target(), patch.object(BoundKernel, "__call__", observe):
        kernel = helion.kernel(_custom_constructor.fn, autotune_effort="none")
        first = kernel(*first_args)
        fresh = helion.kernel(_custom_constructor.fn, autotune_effort="none").bind(
            changed_args
        )
        second = kernel(*changed_args)
        assert first.config_spec.default_config()["factor"] == 2
        assert fresh.config_spec.default_config()["factor"] == 5
        assert second.config_spec.default_config()["factor"] == 5
        assert first is not second
        assert first.to_code() != second.to_code()
        assert fresh.to_code() == second.to_code()
        with patch.object(kernel, "bind", side_effect=AssertionError("cache miss")):
            assert kernel(*first_args) is first
            assert kernel(*changed_args) is second


@helion.kernel(autotune_effort="none")
def _iterated_custom_constructor(x, fragments: hl.constexpr):
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        result = x[tile]
        for fragment in fragments:
            chosen = _PickedFragment(fragment)
            result = result + chosen.default_val
        out[tile] = result
    return out


@onlyBackends(["triton", "cute"])
def test_custom_fragment_constructor_uses_each_unrolled_input():
    with cpu_render_target():
        bound = _iterated_custom_constructor.bind(
            (torch.empty(17), (ANCHOR, BooleanFragment()))
        )
        additions = [
            node.args[1]
            for graph in bound.host_function.device_ir.graphs
            for node in graph.graph.nodes
            if node.target is torch.ops.aten.add.Tensor
        ]
        assert additions == [2, 5]
        bound.to_code()


def _identity(fragment):
    HELPER_CALLS.append(fragment)
    return fragment


@helion.kernel()
def _selected_helper_fragment(x, fragment: hl.constexpr):
    factor = hl.register_tunable("factor", _identity(fragment))
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        out[tile] = x[tile] + factor
    return out


@onlyBackends(["triton", "cute"])
@pytest.mark.parametrize("decorator_config", [False, True])
def test_eviction_does_not_publish_bound_explicit_config(decorator_config):
    configured = helion.Config(block_sizes=[32], factor=False)
    kernel = helion.kernel(
        _selected_helper_fragment.fn,
        autotune_effort="none",
        **({"config": configured} if decorator_config else {}),
    )
    tensor = torch.empty(17)
    fragment = BooleanFragment()

    def compile_observer(bound, config, *, allow_print=True):
        bound.to_code(config)
        return lambda *args: bound

    with (
        cpu_render_target(),
        patch.object(BoundKernel, "compile_config", compile_observer),
        patch("helion.runtime.kernel._OPAQUE_BINDING_CACHE_SIZE", 1),
    ):
        original = kernel.bind((tensor, fragment))
        with original.env:
            expected = (
                configured
                if decorator_config
                else original.config_spec.default_config()
            )
            selected = original.config_spec.default_config()
        selected.config.update(factor=True, block_sizes=[64])
        original.set_config(selected)
        kernel.bind((tensor, BooleanFragment()))
        assert original not in kernel._opaque_bindings
        assert not kernel._fresh_selections
        rebound = kernel.bind((tensor, fragment))
        assert rebound is not original
        with patch.object(
            type(rebound.env.backend), "autotune", side_effect=AssertionError("retune")
        ):
            assert kernel(tensor, fragment) is rebound
        assert rebound._config == rebound._normalize_config(expected)
        assert original._config == selected
        kernel.reset()
        assert not kernel._fresh_selections


@onlyBackends(["triton", "cute"])
def test_eviction_keeps_autotune_selection_before_bound_override():
    kernel = helion.kernel(
        _selected_helper_fragment.fn,
        config=helion.Config(block_sizes=[32], factor=False),
    )
    tensor, fragment = torch.empty(17), BooleanFragment()

    def compile_observer(bound, config, *, allow_print=True):
        bound.to_code(config)
        return lambda *args: bound

    with (
        cpu_render_target(),
        patch.object(BoundKernel, "compile_config", compile_observer),
        patch("helion.runtime.kernel._OPAQUE_BINDING_CACHE_SIZE", 1),
    ):
        original = kernel.bind((tensor, fragment))
        with original.env:
            tuned = original.config_spec.default_config()
        tuned.config.update(factor=True, block_sizes=[64])
        with patch.object(type(original.env.backend), "autotune", return_value=tuned):
            assert original.autotune((tensor, fragment)) == tuned
        explicit = helion.Config(block_sizes=[128], factor=False)
        original.set_config(explicit)
        kernel.bind((tensor, BooleanFragment()))
        assert original not in kernel._opaque_bindings
        assert len(kernel._fresh_selections) == 1
        rebound = kernel.bind((tensor, fragment))
        with patch.object(
            type(rebound.env.backend), "autotune", side_effect=AssertionError("retune")
        ):
            assert kernel(tensor, fragment) is rebound
        assert rebound._config == tuned
        assert original._config == original._normalize_config(explicit)


def _first(values):
    HELPER_CALLS.append(values)
    return values[0]


def _global_return(fragment):
    HELPER_CALLS.append(fragment)
    return ANCHOR


@helion.kernel(autotune_effort="none")
def _helper_comparison(x, values: hl.constexpr, mode: hl.constexpr):
    if mode == "host":
        first = _identity(values[0])
        second = _identity(values[1])
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        result = x[tile]
        for index in hl.static_range(2):
            if mode == "host":
                fragment = first if index == 0 else second
            elif mode == "keyword":
                fragment = _identity(fragment=values[index])
            elif mode == "nested":
                fragment = _first((values[index],))
            elif mode == "starred":
                fragment = _identity(*(values[index],))
            elif mode == "mapping":
                mapping = {"fragment": values[index]}
                fragment = _identity(**mapping)
            else:
                fragment = _identity(values[index])
            if fragment is ANCHOR:
                result = result + 2.0
            else:
                result = result + 5.0
        out[tile] = result
    return out


@onlyBackends(["triton", "cute"])
@pytest.mark.parametrize(
    "mode", ["direct", "keyword", "nested", "host", "starred", "mapping"]
)
def test_ordinary_helper_preserves_values_and_argument_paths(mode):
    kernel = helion.kernel(_helper_comparison.fn, autotune_effort="none")
    tensor = torch.empty(17)

    def args(anchor_index):
        return (
            tensor,
            hl.constexpr(
                tuple(
                    ANCHOR if index == anchor_index else BooleanFragment()
                    for index in range(2)
                )
            ),
            mode,
        )

    def additions(bound):
        return [
            node.args[1]
            for graph in bound.host_function.device_ir.graphs
            for node in graph.graph.nodes
            if node.op == "call_function" and node.target is torch.ops.aten.add.Tensor
        ]

    def observe(bound, *values):
        before = tuple(bound.env.runtime_input_specializations.values())
        bound.to_code()
        assert tuple(bound.env.runtime_input_specializations.values()) == before
        bound._run = lambda *unused: bound
        return bound._run(*values)

    with cpu_render_target(), patch.object(BoundKernel, "__call__", observe):
        expected = ([2.0, 5.0], [5.0, 2.0], [5.0, 5.0])
        inputs = [args(index) for index in (0, 1, -1)]
        bounds = []
        for values, offsets in zip(inputs, expected, strict=True):
            bound = kernel(*values)
            assert additions(bound) == offsets
            fresh = helion.kernel(_helper_comparison.fn, autotune_effort="none")
            assert bound.to_code() == fresh.bind(values).to_code()
            bounds.append(bound)
        assert len({id(bound) for bound in bounds}) == 3
        # Fresh declarations conservatively rebind; opaque helper internals
        # need not prove that a returned input really originated there.
        fresh = kernel(*args(-1))
        assert fresh not in bounds
        assert additions(fresh) == expected[-1]
        old_key = kernel.specialization_key(inputs[-1])
        new_key = kernel.specialization_key(args(-1))
        assert old_key != new_key
        assert repr(old_key) == repr(new_key)
        calls = len(HELPER_CALLS)
        with patch.object(kernel, "bind", side_effect=AssertionError("cache miss")):
            for values, bound in zip(inputs, bounds, strict=True):
                assert kernel(*values) is bound
                assert kernel(*values) is bound
        assert len(HELPER_CALLS) == calls


@helion.kernel(autotune_effort="none")
def _compare_global_return(x, fragment: hl.constexpr):
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        if _global_return(fragment) is fragment:
            out[tile] = x[tile] + 2.0
        else:
            out[tile] = x[tile] + 5.0
    return out


@onlyBackends(["triton", "cute"])
def test_global_return_is_not_misclassified_as_input_alias():
    kernel = helion.kernel(_compare_global_return.fn, autotune_effort="none")
    tensor = torch.empty(17)
    with cpu_render_target():
        shared = kernel.bind((tensor, ANCHOR))
        distinct = (tensor, BooleanFragment())
        other = kernel.bind(distinct)
        fresh = helion.kernel(_compare_global_return.fn, autotune_effort="none")
        assert other.to_code() == fresh.bind(distinct).to_code()
        assert other.to_code() != shared.to_code()
        calls = len(HELPER_CALLS)
        assert kernel.bind(distinct) is other
        assert kernel.bind((tensor, ANCHOR)) is shared
        assert len(HELPER_CALLS) == calls


@dataclass(frozen=True)
class _ValueFragment(ConfigSpecFragment):
    value: int

    def default(self):
        return self.value


CUSTOM_ANCHOR = _ValueFragment(3)


@helion.kernel(autotune_effort="none")
def _compare_custom_return(x, fragment: hl.constexpr):
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        if _identity(fragment) == CUSTOM_ANCHOR:
            out[tile] = x[tile] + 2.0
        else:
            out[tile] = x[tile] + 5.0
    return out


@onlyBackends(["triton", "cute"])
def test_opaque_helper_preserves_custom_fragment_equality():
    kernel = helion.kernel(_compare_custom_return.fn, autotune_effort="none")
    tensor = torch.empty(17)
    with cpu_render_target():
        shared = kernel.bind((tensor, CUSTOM_ANCHOR))
        equal = kernel.bind((tensor, _ValueFragment(3)))
        other = kernel.bind((tensor, _ValueFragment(7)))
        assert equal.to_code() == shared.to_code()
        assert other.to_code() != shared.to_code()


@dataclass
class _FragmentHolder:
    fragment: object


class _NamedHolder(NamedTuple):
    fragment: object


@dataclass(frozen=True)
class _FragmentDomainHolder(ConfigSpecFragment):
    fragment: object = field(compare=False)

    def default(self):
        return True


@helion.kernel(autotune_effort="none")
def _compare_attribute_return(x, holder):
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        if _identity(holder.fragment) is ANCHOR:
            out[tile] = x[tile] + 2.0
        else:
            out[tile] = x[tile] + 5.0
    return out


@onlyBackends(["triton", "cute"])
@pytest.mark.parametrize("holder_type", [_FragmentHolder, _NamedHolder])
def test_opaque_helper_guards_attribute_fragment_argument(holder_type):
    kernel = helion.kernel(_compare_attribute_return.fn, autotune_effort="none")
    tensor = torch.empty(17)
    with cpu_render_target():
        args = (tensor, holder_type(hl.constexpr(ANCHOR)))
        shared = kernel.bind(args)
        other_args = (tensor, holder_type(hl.constexpr(BooleanFragment())))
        other = kernel.bind(other_args)
        fresh = helion.kernel(_compare_attribute_return.fn, autotune_effort="none")
        assert other.to_code() == fresh.bind(other_args).to_code()
        assert other.to_code() != shared.to_code()
        assert kernel.bind(args) is shared
        assert kernel.bind(other_args) is other


@onlyBackends(["triton", "cute"])
def test_opaque_helper_guards_fragment_attribute_origin():
    kernel = helion.kernel(_compare_attribute_return.fn, autotune_effort="none")
    tensor = torch.empty(17)
    with cpu_render_target():
        first = (tensor, hl.constexpr(_FragmentDomainHolder(ANCHOR)))
        second = (tensor, hl.constexpr(_FragmentDomainHolder(BooleanFragment())))
        assert first[1].value == second[1].value
        shared = kernel.bind(first)
        other = kernel.bind(second)
        fresh = helion.kernel(_compare_attribute_return.fn, autotune_effort="none")
        assert other.to_code() == fresh.bind(second).to_code()
        assert other.to_code() != shared.to_code()
        assert kernel.bind(first) is shared
        assert kernel.bind(second) is other


def _pack_fragment(fragment, kind):
    HELPER_CALLS.append(fragment)
    if kind == "tuple":
        return (fragment,)
    if kind == "list":
        return [fragment]
    if kind == "dict":
        return {"fragment": fragment}
    if kind == "dataclass":
        return _FragmentHolder(fragment)
    return _NamedHolder(fragment)


@helion.kernel(autotune_effort="none")
def _compare_container_return(
    x, fragment: hl.constexpr, kind: hl.constexpr, on_host: hl.constexpr
):
    if on_host:
        packed = _pack_fragment(fragment, kind)
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        if not on_host:
            packed = _pack_fragment(fragment, kind)
        if kind == "tuple" or kind == "list":
            value = packed[0]
        elif kind == "dict":
            value = packed["fragment"]
        else:
            value = packed.fragment
        if value is ANCHOR:
            out[tile] = x[tile] + 2.0
        else:
            out[tile] = x[tile] + 5.0
    return out


@onlyBackends(["triton", "cute"])
@pytest.mark.parametrize(
    "kind,on_host",
    [
        ("tuple", False),
        ("list", False),
        ("dict", False),
        ("dataclass", False),
        ("namedtuple", False),
        ("tuple", True),
        ("dataclass", True),
    ],
)
def test_opaque_helper_guards_fragment_containing_return(kind, on_host):
    kernel = helion.kernel(_compare_container_return.fn, autotune_effort="none")
    tensor = torch.empty(17)

    def observe(bound, *values):
        before = tuple(bound.env.runtime_input_specializations.values())
        bound.to_code()
        assert tuple(bound.env.runtime_input_specializations.values()) == before
        bound._run = lambda *unused: bound
        return bound._run(*values)

    with cpu_render_target(), patch.object(BoundKernel, "__call__", observe):
        inputs = [
            (tensor, value, kind, on_host)
            for value in (ANCHOR, BooleanFragment(), BooleanFragment())
        ]
        bounds = []
        for args, expected in zip(inputs, (2.0, 5.0, 5.0), strict=True):
            bound = kernel(*args)
            additions = [
                node.args[1]
                for graph in bound.host_function.device_ir.graphs
                for node in graph.graph.nodes
                if node.op == "call_function"
                and node.target is torch.ops.aten.add.Tensor
            ]
            assert additions == [expected]
            fresh = helion.kernel(_compare_container_return.fn, autotune_effort="none")
            assert bound.to_code() == fresh.bind(args).to_code()
            bounds.append(bound)
        assert len({id(bound) for bound in bounds}) == 3
        calls = len(HELPER_CALLS)
        with patch.object(kernel, "bind", side_effect=AssertionError("cache miss")):
            for args, bound in zip(inputs, bounds, strict=True):
                assert kernel(*args) is bound
                assert kernel(*args) is bound
        assert len(HELPER_CALLS) == calls


@helion.kernel(autotune_effort="none")
def _ordinary_callable_argument(x, fn):
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        out[tile] = fn(x[tile])
    return out


@helion.kernel(autotune_effort="none")
def _constexpr_callable_argument(x, fn: hl.constexpr):
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        out[tile] = fn(x[tile])
    return out


@onlyBackends(["triton", "cute"])
@pytest.mark.parametrize("constexpr", [False, True])
def test_fragment_free_callable_does_not_add_identity_classifier(constexpr):
    function = (
        _constexpr_callable_argument.fn if constexpr else _ordinary_callable_argument.fn
    )
    kernel = helion.kernel(function, autotune_effort="none")
    tensor = torch.empty(17)
    fn = (lambda value: value + 1) if constexpr else torch.relu
    with cpu_render_target():
        bound = kernel.bind((tensor, fn))
        bound.to_code()
        assert not any(
            key.startswith("fragment_call_input:")
            for key in bound.env.runtime_input_specializations
        )
        assert kernel.bind((tensor, fn)) is bound


@helion.kernel(autotune_effort="none")
def _ordinary_callable_container(x, callbacks):
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        out[tile] = callbacks[0](x[tile])
    return out


@onlyBackends(["triton", "cute"])
@pytest.mark.parametrize("container", [tuple, list, dict])
def test_keyed_callable_container_elides_classifier_and_tracks_default(container):
    def callback(value, offset=1):
        return value + offset

    callbacks = {0: callback} if container is dict else container([callback])
    tensor = torch.empty(17)
    args = (tensor, callbacks)
    kernel = helion.kernel(_ordinary_callable_container.fn, autotune_effort="none")
    with cpu_render_target():
        bound = kernel.bind(args)
        source = bound.to_code()
        assert not any(
            key.startswith("fragment_call_input:")
            for key in bound.env.runtime_input_specializations
        )
        assert kernel.bind(args) is bound
        # Codegen is real; only device compilation/execution is modeled.
        with patch.object(
            BoundKernel, "compile_config", return_value=lambda *args: bound
        ):
            bound.set_config(bound.config_spec.default_config())
            assert kernel(*args) is bound  # Populate the normal dispatch entry.
            with patch.object(kernel, "bind", side_effect=AssertionError("rebind")):
                assert bound(*args) is bound
                assert kernel(*args) is bound
        callback.__defaults__ = (2,)
        updated = kernel.bind(args)
        assert updated is not bound
        assert updated.to_code() != source
