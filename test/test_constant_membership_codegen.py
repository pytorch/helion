"""Default tuple membership uses proved positional values."""

from __future__ import annotations

import ast

import pytest
import torch

import helion
from helion._compiler.type_info import LiteralType
from helion._compiler.type_info import SymBoolType
from helion._testing import cpu_render_target
from helion._testing.binding_observation import scalar_additions
from helion._testing.constant_membership import record_membership
from helion._testing.cute_codegen import cpu_bind

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize("form", ["literal", "custom", "symbolic"])
def test_namedtuple_membership_uses_proved_values(form):
    with cpu_render_target():
        bound = cpu_bind(
            helion.kernel(record_membership, static_shapes=False),
            (torch.ones(7), form),
        )
        for name, expected in (
            ("member", True),
            ("nonmember", False),
            ("field_member", False),
        ):
            value = bound.host_function.local_types[name]
            if form == "literal":
                assert isinstance(value, LiteralType) and value.value is expected
            else:
                assert isinstance(value, SymBoolType)
        if form == "literal":
            assert scalar_additions(bound) == [3, 13, 19]
        ast.parse(bound.to_code(bound.config_spec.default_config()))
