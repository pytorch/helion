from __future__ import annotations

import ast
import random

import pytest
import torch

import helion
from helion._testing import cpu_render_target
from helion._testing import onlyBackends
from helion._testing.cute_codegen import cpu_bind
from helion.autotuner.config_fragment import ConfigSpecFragment
import helion.language as hl

pytestmark = pytest.mark.cpu_only


class _DirectAllocator(ConfigSpecFragment):
    def __new__(cls):
        return object.__new__(cls)

    def __init__(self):
        self.value = 3
        self.rng = random

    def default(self):
        return self.value


class _SingletonAllocator(_DirectAllocator):
    def __new__(cls):
        return SHARED_SINGLETON

    def __init__(self):
        pass


SHARED_SINGLETON = object.__new__(_SingletonAllocator)
SHARED_SINGLETON.value = 3
SHARED_SINGLETON.rng = random
FACTORY = _DirectAllocator


def _make_fragment():
    return FACTORY()


@helion.kernel(autotune_effort="none")
def _direct(x):
    factor = hl.register_tunable("factor", FACTORY())
    output = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        output[tile] = x[tile] * factor
    return output


@helion.kernel(autotune_effort="none")
def _helper(x):
    factor = hl.register_tunable("factor", _make_fragment())
    output = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        output[tile] = x[tile] * factor
    return output


@pytest.mark.parametrize("helper", [False, True])
@onlyBackends(["triton", "cute"])
def test_custom_allocator_retains_resources(helper, monkeypatch):
    monkeypatch.setitem(globals(), "FACTORY", _DirectAllocator)
    with cpu_render_target():
        bound = cpu_bind(_helper if helper else _direct, (torch.empty(37),))
        captured = bound.config_spec.user_defined_tunables["factor"]
        assert type(captured) is _DirectAllocator
        assert captured.rng is random and captured.default() == 3
        ast.parse(bound.to_code(bound.config_spec.default_config()))


@pytest.mark.parametrize("helper", [False, True])
@onlyBackends(["triton", "cute"])
def test_custom_allocator_returning_singleton_does_not_reallocate_or_mutate_it(
    helper, monkeypatch
):
    monkeypatch.setitem(globals(), "FACTORY", _SingletonAllocator)
    with cpu_render_target():
        bound = cpu_bind(_helper if helper else _direct, (torch.empty(37),))
        captured = bound.config_spec.user_defined_tunables["factor"]
        assert type(captured) is _SingletonAllocator and captured is SHARED_SINGLETON
        assert captured.rng is random
    assert SHARED_SINGLETON.value == 3 and SHARED_SINGLETON.rng is random


class _SearchState:
    def __init__(self):
        self.calls = []


class _PrivateStateFragment(ConfigSpecFragment):
    def __init__(self):
        self.state = _SearchState()
        self.domain = [3, 5]

    def default(self):
        return self.domain[0]


@helion.kernel(autotune_effort="none")
def _shared(x, declaration: hl.constexpr):
    factor = hl.register_tunable("factor", declaration)
    output = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        output[tile] = x[tile] * factor
    return output


@onlyBackends(["triton", "cute"])
def test_custom_fragments_retain_objects_across_bindings():
    declaration = _PrivateStateFragment()
    declaration.domain.append(declaration.domain)
    with cpu_render_target():
        first = cpu_bind(_shared, (torch.empty(37), declaration))
        second = cpu_bind(_shared, (torch.empty(37), declaration))
        assert first.config_spec.user_defined_tunables["factor"] is declaration
        assert second.config_spec.user_defined_tunables["factor"] is declaration
        assert declaration.domain[-1] is declaration.domain
        declaration.state.calls.append("shared-search-state")
        assert first.config_spec.user_defined_tunables["factor"].state.calls == [
            "shared-search-state"
        ]
        ast.parse(first.to_code(first.config_spec.default_config()))


class _RefusesCopyFragment(_PrivateStateFragment):
    def __deepcopy__(self, memo):
        raise AssertionError("registration must not copy custom fragments")


@onlyBackends(["triton", "cute"])
def test_registration_does_not_invoke_custom_copy_hooks():
    declaration = _RefusesCopyFragment()
    with cpu_render_target():
        bound = cpu_bind(_shared, (torch.empty(37), declaration))
        assert bound.config_spec.user_defined_tunables["factor"] is declaration
        ast.parse(bound.to_code(bound.config_spec.default_config()))
