from __future__ import annotations

import ast
from dataclasses import dataclass
from unittest.mock import patch

import pytest
import torch

import helion
from helion._testing import cpu_render_target
from helion._testing import onlyBackends
from helion._testing.cute_codegen import cpu_bind
from helion.autotuner.config_fragment import ConfigSpecFragment
from helion.autotuner.config_fragment import EnumFragment
from helion.autotuner.config_fragment import NumThreadsFragment
import helion.language as hl

pytestmark = pytest.mark.cpu_only
HIGH = 8
CREATED = []
COPY_CALLS = []
SHARED_THREADS = NumThreadsFragment(128)


@dataclass
class _EqualWrapper(ConfigSpecFragment):
    inner: ConfigSpecFragment

    def default(self):
        return self.inner.default()


def _equal_wrapper():
    return _EqualWrapper(SHARED_THREADS)


@helion.kernel(autotune_effort="none")
def _equal_declarations(x):
    first = hl.register_tunable("factor", _equal_wrapper())
    second = hl.register_tunable("factor", _equal_wrapper())
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        out[tile] = x[tile] + first + second
    return out


@onlyBackends(["triton", "cute"])
def test_equal_custom_declarations_retain_shared_nested_identity():
    assert _equal_wrapper() == _equal_wrapper()
    with cpu_render_target():
        bound = cpu_bind(_equal_declarations, (torch.empty(37),))
        captured = bound.config_spec.user_defined_tunables["factor"]
        assert captured.inner is SHARED_THREADS
        ast.parse(bound.to_code(bound.config_spec.default_config()))


class _IdentityCopyHook(ConfigSpecFragment):
    def __init__(self, return_self):
        self.return_self = return_self
        self.history = []

    def default(self):
        return 3

    def __deepcopy__(self, memo):
        COPY_CALLS.append(self)
        if self.return_self:
            return self
        result = type(self)(False)
        result.history = self.history
        return result


@helion.kernel(autotune_effort="none")
def _custom_argument(x, fragment: hl.constexpr):
    factor = hl.register_tunable("factor", fragment)
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        out[tile] = x[tile] + factor
    return out


@onlyBackends(["triton", "cute"])
def test_custom_registration_preserves_object_without_copy_hook():
    fragment = _IdentityCopyHook(True)
    calls = []
    with (
        patch.dict(_custom_argument.fn.__globals__, COPY_CALLS=calls),
        cpu_render_target(),
    ):
        bound = cpu_bind(_custom_argument, (torch.empty(37), fragment))
        captured = bound.config_spec.user_defined_tunables["factor"]
        assert captured is fragment
        assert captured.history is fragment.history
        assert not calls
        ast.parse(bound.to_code(bound.config_spec.default_config()))


def _builtin_domain(high):
    choices = (2, high)
    result = EnumFragment(choices, search_choices=choices)
    CREATED.append(result)
    return result


@helion.kernel(autotune_effort="none")
def _builtin_literal_domain(x):
    factor = hl.register_tunable("factor", _builtin_domain(HIGH))
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        out[tile] = x[tile] + factor
    return out


@onlyBackends(["triton", "cute"])
def test_builtin_literal_snapshot_preserves_shared_choice_fields():
    created = []
    with (
        patch.dict(_builtin_literal_domain.fn.__globals__, CREATED=created),
        cpu_render_target(),
    ):
        bound = cpu_bind(_builtin_literal_domain, (torch.empty(37),))
        captured = bound.config_spec.user_defined_tunables["factor"]
        assert captured is not created[0]
        assert captured.choices is captured.search_choices
        assert captured.choices == (2, 8)
        assert all(type(value) is int for value in captured.choices)
        fingerprint = bound.config_spec.cache_fingerprint_hash()
        created[0].choices = (3, 9)
        assert captured.choices == (2, 8)
        assert bound.config_spec.cache_fingerprint_hash() == fingerprint
        ast.parse(bound.to_code(bound.config_spec.default_config()))
