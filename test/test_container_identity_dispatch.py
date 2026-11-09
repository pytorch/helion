from __future__ import annotations

from dataclasses import dataclass
import gc
from typing import NamedTuple
from unittest.mock import patch
import weakref

import pytest
import torch
from torch.utils import _pytree as pytree

import helion
from helion._compiler._dynamo.variables import infer_output_spec
from helion._compiler.compile_environment import CompileEnvironment
from helion._testing import cpu_render_target
from helion._testing import onlyBackends
from helion._testing.binding_observation import assert_scalar_addition
from helion._testing.binding_observation import observe_compiled_binding
from helion.autotuner import BooleanFragment
import helion.language as hl
from helion.runtime.kernel import BoundKernel

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize("kind", ["tuple", "list", "dict"])
def test_host_created_output_collection_can_be_projected_after_binding(kind):
    @helion.kernel(autotune_effort="none")
    def fn(x, kind: hl.constexpr):
        out = torch.empty_like(x)
        for tile in hl.tile(x.numel()):
            out[tile] = x[tile] + 1
        if kind == "tuple":
            outputs = tuple([out, x])
        elif kind == "list":
            outputs = list((out, x))  # noqa: C410 -- test host constructor proxies
        else:
            outputs = dict(output=out, input=x)  # noqa: C408 -- test host constructor proxies
        return outputs

    x = torch.empty(7)
    with cpu_render_target():
        assert not CompileEnvironment.has_current()
        spec = infer_output_spec(fn, (x, kind))
        assert not CompileEnvironment.has_current()
    assert [leaf["shape"] for leaf in spec["leaf_specs"]] == [[7], [7]]
    assert [leaf["dtype"] for leaf in spec["leaf_specs"]] == [x.dtype, x.dtype]
    out = torch.empty_like(x)
    expected = {
        "tuple": (out, x),
        "list": [out, x],
        "dict": {"output": out, "input": x},
    }[kind]
    assert pytree.treespec_loads(spec["tree_spec_str"]) == pytree.tree_structure(
        expected
    )
    assert spec["direct_aliases"] == {1: "x"}


@helion.kernel(autotune_effort="none")
def _local_container(x, left, right, kind: hl.constexpr):
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        if kind == "tuple":
            local = (4, 8)
        elif kind == "list":
            local = [4, 8]
        else:
            local = {"a": 4, "b": 8}
        same_input = left is right
        same_local = local is left
        if same_input:
            offset = 2.0
        elif same_local:
            offset = 3.0
        else:
            offset = 5.0
        out[tile] = x[tile] + offset
    return out


def _assert_reused_routes(fn, shared, distinct, *, require_prepared=True):
    with (
        cpu_render_target(),
        patch.object(BoundKernel, "ensure_config_exists", observe_compiled_binding),
    ):
        kernel = helion.kernel(fn, autotune_effort="none")
        first = kernel(*shared)
        assert_scalar_addition(first, 2.0)
        second = kernel(*distinct)
        fresh = helion.kernel(fn, autotune_effort="none").bind(distinct)
        assert_scalar_addition(second, 5.0)
        assert_scalar_addition(fresh, 5.0)
        assert not first.env.requires_fresh_binding
        assert not second.env.requires_fresh_binding
        prepared = kernel._prepared_call
        if require_prepared:
            assert prepared is not None and prepared.matches(kernel, distinct)
            assert not prepared.matches(kernel, shared)
        assert_scalar_addition(first(*distinct), 5.0)
        assert_scalar_addition(second(*shared), 2.0)
        with patch.object(
            BoundKernel, "__init__", side_effect=AssertionError("cache miss")
        ):
            assert kernel(*shared) is first
            assert kernel(*distinct) is second
        return kernel, first, second


@onlyBackends(["triton", "cute"])
@pytest.mark.parametrize("kind", ["tuple", "list", "dict"])
def test_constructed_local_container_does_not_become_a_captured_anchor(kind):
    factory = {
        "tuple": lambda: tuple([4, 8]),
        "list": lambda: [4, 8],
        "dict": lambda: {"a": 4, "b": 8},
    }[kind]
    left, right = factory(), factory()
    assert left == right and left is not right
    x = torch.empty(17)
    _, first, second = _assert_reused_routes(
        _local_container.fn, (x, left, left, kind), (x, left, right, kind)
    )
    anchors = [
        descriptor.binding_anchors
        for bound in (first, second)
        for descriptor in bound.env.runtime_input_specializations.values()
        if descriptor.binding_anchors
    ]
    # CPython folds the tuple literal into a stable code constant. Lists and
    # dictionaries are new allocations; neither becomes a captured anchor.
    assert bool(anchors) is (kind == "tuple")


@dataclass
class _DataclassHolder:
    value: int


class _NamedHolder(NamedTuple):
    value: int


@helion.kernel(autotune_effort="none")
def _holder_identity(x, left, right):
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        if left is right:
            out[tile] = x[tile] + 2.0
        else:
            out[tile] = x[tile] + 5.0
    return out


@onlyBackends(["triton", "cute"])
@pytest.mark.parametrize("holder", [_DataclassHolder, _NamedHolder])
def test_holder_identity_preserves_aliases_in_fresh_cached_and_direct_calls(holder):
    left, right = holder(4), holder(4)
    x = torch.empty(17)
    _assert_reused_routes(
        _holder_identity.fn, (x, left, left), (x, left, right), require_prepared=False
    )


@onlyBackends(["triton", "cute"])
def test_holder_identity_does_not_retain_runtime_input():
    holder = _DataclassHolder(4)
    reference = weakref.ref(holder)
    with cpu_render_target():
        kernel = helion.kernel(_holder_identity.fn, autotune_effort="none")
        bound = kernel.bind((torch.empty(17), holder, holder))
        assert_scalar_addition(bound, 2.0)
        del holder
        gc.collect()
        assert reference() is None
        assert not bound.env._container_identities
        assert_scalar_addition(bound, 2.0)


@dataclass
class _Receiver:
    fragment: object

    def same(self, other):
        return self.fragment is other


@helion.kernel(autotune_effort="none")
def _bound_method_identity(x, callback: hl.constexpr, value: hl.constexpr):
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        if callback(value):
            out[tile] = x[tile] + 2.0
        else:
            out[tile] = x[tile] + 5.0
    return out


@onlyBackends(["triton", "cute"])
def test_bound_method_receiver_fragment_alias_is_guarded():
    left, right = BooleanFragment(), BooleanFragment()
    receiver = _Receiver(left)
    callback = receiver.same
    x = torch.empty(17)
    _assert_reused_routes(
        _bound_method_identity.fn,
        (x, callback, left),
        (x, callback, right),
        require_prepared=False,
    )
    with (
        cpu_render_target(),
        patch.object(BoundKernel, "ensure_config_exists", observe_compiled_binding),
    ):
        kernel = helion.kernel(_bound_method_identity.fn, autotune_effort="none")
        first = kernel(x, callback, left)
        assert_scalar_addition(first, 2.0)
        receiver.fragment = right
        assert_scalar_addition(kernel(x, callback, left), 5.0)
        assert_scalar_addition(first(x, callback, left), 5.0)
        receiver.fragment = left
        assert kernel(x, callback, left) is first


@helion.kernel(autotune_effort="none")
def _tuple_constructor_identity(x, left, right):
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        copied = tuple(left)
        if copied is right:
            out[tile] = x[tile] + 2.0
        else:
            out[tile] = x[tile] + 5.0
    return out


@onlyBackends(["triton", "cute"])
def test_identity_preserving_tuple_constructor_keeps_input_guards():
    left, right = tuple([4, 8]), tuple([4, 8])
    x = torch.empty(17)
    _assert_reused_routes(
        _tuple_constructor_identity.fn, (x, left, left), (x, left, right)
    )


EMPTY = ()
FIRST = tuple([4, 8])
SECOND = tuple([4, 8])


@helion.kernel(autotune_effort="none")
def _empty_identity(x, values):
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        local = ()
        if local is values and EMPTY is values:
            out[tile] = x[tile] + 2.0
        else:
            out[tile] = x[tile] + 5.0
    return out


@onlyBackends(["triton", "cute"])
def test_empty_tuple_occurrence_does_not_reclassify_captured_or_input_identity():
    x = torch.empty(17)
    with (
        cpu_render_target(),
        patch.object(BoundKernel, "ensure_config_exists", observe_compiled_binding),
    ):
        kernel = helion.kernel(_empty_identity.fn, autotune_effort="none")
        first = kernel(x, EMPTY)
        second = kernel(x, FIRST)
        assert_scalar_addition(first, 2.0)
        assert_scalar_addition(second, 5.0)
        assert kernel(x, EMPTY) is first
        assert kernel(x, FIRST) is second


@helion.kernel(autotune_effort="none")
def _globals_only_identity(x, unused_container):
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        if FIRST is SECOND:
            out[tile] = x[tile] + 2.0
        else:
            out[tile] = x[tile] + 5.0
    return out


@onlyBackends(["triton", "cute"])
def test_global_only_comparison_does_not_publish_an_input_partition():
    with cpu_render_target():
        bound = _globals_only_identity.bind((torch.empty(17), FIRST))
        assert_scalar_addition(bound, 5.0)
        assert not any(
            key.startswith("container_identity:")
            for key in bound.env.runtime_input_specializations
        )


@helion.kernel(autotune_effort="none")
def _later_anchor(x, left: hl.constexpr, right: hl.constexpr):
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        if left is FIRST:
            anchors = (SECOND,)
        else:
            anchors = (FIRST,)
        for anchor in anchors:
            if right is anchor:
                offset = 2.0
            else:
                offset = 5.0
        out[tile] = x[tile] + offset
    return out


@onlyBackends(["triton", "cute"])
def test_changed_anchor_warns_once_and_keeps_fresh_correctness(caplog):
    x = torch.empty(17)
    with (
        cpu_render_target(),
        patch.object(BoundKernel, "ensure_config_exists", observe_compiled_binding),
    ):
        kernel = helion.kernel(_later_anchor.fn, autotune_effort="none")
        assert_scalar_addition(kernel(x, tuple([4, 8]), SECOND), 5.0)
        for _ in range(3):
            deep = kernel(x, FIRST, SECOND)
            assert_scalar_addition(deep, 2.0)
            assert deep.env.requires_fresh_binding
        messages = [
            record.message
            for record in caplog.records
            if "uncovered binding specialization" in record.message
        ]
        assert len(messages) == 1
        assert "container_identity:" in messages[0]
