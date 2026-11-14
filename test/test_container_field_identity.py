from __future__ import annotations

from dataclasses import InitVar
from dataclasses import dataclass
from dataclasses import field
import gc
from typing import ClassVar
from typing import NamedTuple
from unittest.mock import patch
import weakref

import pytest
import torch

import helion
from helion._testing import cpu_render_target
from helion._testing import onlyBackends
from helion._testing.binding_observation import assert_scalar_addition
from helion._testing.binding_observation import observe_compiled_binding
from helion.autotuner import BooleanFragment
from helion.autotuner.config_fragment import ConfigSpecFragment
import helion.language as hl
from helion.runtime.kernel import BoundKernel

pytestmark = pytest.mark.cpu_only
FRAGMENT = BooleanFragment()
SAVED = (FRAGMENT,)


@dataclass(unsafe_hash=True)
class DataclassHolder:
    values: object


class NamedHolder(NamedTuple):
    values: object


@dataclass(frozen=True)
class FragmentHolder(ConfigSpecFragment):
    values: object

    def default(self):
        return True


@helion.kernel(autotune_effort="none")
def _field_identity(x, holder: hl.constexpr):
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        if holder.values is SAVED:
            out[tile] = x[tile] + 2.0
        else:
            out[tile] = x[tile] + 5.0
    return out


def _check_routes(fn, first_args, second_args):
    kernel = helion.kernel(fn, autotune_effort="none")
    with (
        cpu_render_target(),
        patch.object(BoundKernel, "__call__", observe_compiled_binding),
    ):
        first = kernel(*first_args)
        assert_scalar_addition(first, 2.0)
        assert_scalar_addition(
            helion.kernel(fn, autotune_effort="none").bind(second_args), 5.0
        )
        prepared = kernel._prepared_call
        if prepared is not None:
            assert prepared.matches(kernel, first_args)
            assert not prepared.matches(kernel, second_args)
        second = kernel(*second_args)
        assert second is not first
        assert_scalar_addition(second, 5.0)
        assert kernel(*first_args) is first
        assert kernel(*second_args) is second
        assert_scalar_addition(first, 2.0)
        return kernel, first, second


@onlyBackends(["triton", "cute"])
@pytest.mark.parametrize("holder_type", [DataclassHolder, NamedHolder, FragmentHolder])
def test_field_identity_preserves_fresh_and_cached_branches(holder_type):
    x = torch.empty(17)
    _check_routes(
        _field_identity.fn,
        (x, holder_type(SAVED)),
        (x, holder_type(tuple([FRAGMENT]))),
    )


@onlyBackends(["triton", "cute"])
def test_mutated_field_invalidates_prepared_identity_guard():
    x = torch.empty(17)
    holder = DataclassHolder(SAVED)
    with (
        cpu_render_target(),
        patch.object(BoundKernel, "__call__", observe_compiled_binding),
    ):
        kernel = helion.kernel(_field_identity.fn, autotune_effort="none")
        first = kernel(x, holder)
        assert_scalar_addition(first, 2.0)
        prepared = kernel._prepared_call
        holder.values = tuple([FRAGMENT])
        if prepared is not None:
            assert not prepared.matches(kernel, (x, holder))
        fresh = helion.kernel(_field_identity.fn, autotune_effort="none").bind(
            (x, holder)
        )
        assert_scalar_addition(fresh, 5.0)
        second = kernel(x, holder)
        assert second is not first
        assert_scalar_addition(second, 5.0)
        holder.values = SAVED
        assert kernel(x, holder) is first
        assert_scalar_addition(first, 2.0)


@helion.kernel(autotune_effort="none")
def _nested_fields(x, left: hl.constexpr, right: hl.constexpr):
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        fields = (left.values, right.values)
        if fields[0][0].values is fields[1][0].values:
            out[tile] = x[tile] + 2.0
        else:
            out[tile] = x[tile] + 5.0
    return out


@onlyBackends(["triton", "cute"])
def test_nested_attribute_and_index_paths_preserve_argument_aliases():
    x = torch.empty(17)
    left = NamedHolder((DataclassHolder(SAVED),))
    right = NamedHolder((DataclassHolder(tuple([FRAGMENT])),))
    _check_routes(_nested_fields.fn, (x, left, left), (x, left, right))


@helion.kernel(autotune_effort="none")
def _fragment_field(x, holder: hl.constexpr):
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        if holder.values is FRAGMENT:
            out[tile] = x[tile] + 2.0
        else:
            out[tile] = x[tile] + 5.0
    return out


@onlyBackends(["triton", "cute"])
def test_fragment_field_comparison_uses_attribute_provenance():
    x = torch.empty(17)
    _check_routes(
        _fragment_field.fn,
        (x, DataclassHolder(FRAGMENT)),
        (x, DataclassHolder(BooleanFragment())),
    )


@helion.kernel(autotune_effort="none")
def _tensor_field(holder, other):
    out = torch.empty_like(holder.values[0])
    for tile in hl.tile(holder.values[0].numel()):
        if holder.values is other:
            out[tile] = holder.values[0][tile] + 2.0
        else:
            out[tile] = holder.values[0][tile] + 5.0
    return out


@onlyBackends(["triton", "cute"])
def test_field_identity_guards_do_not_retain_tensor_inputs():
    tensor = torch.empty(17)
    values = (tensor,)
    holder = NamedHolder(values)
    reference = weakref.ref(tensor)
    with cpu_render_target():
        kernel = helion.kernel(_tensor_field.fn, autotune_effort="none")
        bound = kernel.bind((holder, values))
        assert_scalar_addition(bound, 2.0)
        del tensor, values, holder
        gc.collect()
        assert reference() is None
        first = bound.to_code(helion.Config(block_sizes=[32]))
        assert bound.to_code(helion.Config(block_sizes=[64])) != first
        assert bound.to_code(helion.Config(block_sizes=[32])) == first
        assert not bound.env._container_identities
        assert reference() is None


@helion.kernel(autotune_effort="none")
def _dict_identity(holder, other):
    out = torch.empty_like(holder["items"][0])
    for tile in hl.tile(holder["items"][0].numel()):
        if holder["items"] is other:
            out[tile] = holder["items"][0][tile] + 2.0
        else:
            out[tile] = holder["items"][0][tile] + 5.0
    return out


@onlyBackends(["triton", "cute"])
def test_dictionary_string_paths_remain_indexing_not_attributes():
    tensor = torch.empty(17)
    values = (tensor,)
    other = tuple([tensor])
    kernel, _, _ = _check_routes(
        _dict_identity.fn, ({"items": values}, values), ({"items": values}, other)
    )
    assert kernel._prepared_call is not None


@onlyBackends(["triton", "cute"])
def test_tensor_field_cached_call_checks_alias_partition():
    tensor = torch.empty(17)
    values = (tensor,)
    other = tuple([tensor])
    kernel, first, second = _check_routes(
        _tensor_field.fn,
        (NamedHolder(values), values),
        (NamedHolder(values), other),
    )
    # Namedtuple roots do not use the existing fast/prepared argument path.
    assert kernel._fast_dispatch_key((NamedHolder(values), values)) is None
    assert kernel._prepared_call is None
    assert first is not second


@dataclass
class InitVarHolder:
    initial: InitVar[object]
    values: object = field(init=False)

    def __post_init__(self, initial):
        self.values = initial


@dataclass
class ClassVarHolder:
    values: object
    nonstored: ClassVar[object]


@onlyBackends(["triton", "cute"])
@pytest.mark.parametrize("holder_type", [InitVarHolder, ClassVarHolder])
def test_dataclass_pseudo_fields_do_not_enter_cached_identity_guards(holder_type):
    tensor = torch.empty(17)
    values = (tensor,)
    other = tuple([tensor])
    holder = holder_type(values)
    kernel, first, second = _check_routes(
        _tensor_field.fn, (holder, values), (holder, other)
    )
    # init=False is a stored field: changing its alias must still choose the
    # same branch as a fresh binding, with the tensor field key unchanged.
    initial_key = kernel._base_specialization_key((holder, values))
    holder.values = other
    assert kernel._base_specialization_key((holder, values)) == initial_key
    with cpu_render_target():
        changed = kernel.bind((holder, values))
        fresh = helion.kernel(_tensor_field.fn, autotune_effort="none").bind(
            (holder, values)
        )
        assert_scalar_addition(changed, 5.0)
        assert changed.to_code() == fresh.to_code()
        assert changed is not first
    assert first is not second


@helion.kernel(autotune_effort="none", static_shapes=False)
def _specialized_field(holder):
    stride = hl.specialize(holder.values.stride(0))
    out = torch.empty_like(holder.values)
    for tile in hl.tile(holder.values.numel()):
        out[tile] = holder.values[tile] + stride
    return out


@onlyBackends(["triton", "cute"])
def test_tensor_field_specialization_replays_attribute_source():
    with cpu_render_target():
        kernel = helion.kernel(
            _specialized_field.fn, autotune_effort="none", static_shapes=False
        )
        first_args = (NamedHolder(torch.empty(34)[::2]),)
        second_args = (NamedHolder(torch.empty(85)[::5]),)
        first = kernel.bind(first_args)
        second = kernel.bind(second_args)
        assert first is not second
        assert_scalar_addition(first, 2)
        assert_scalar_addition(second, 5)
        assert kernel.bind(first_args) is first
        assert kernel.bind(second_args) is second
