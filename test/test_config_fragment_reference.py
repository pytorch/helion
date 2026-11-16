from __future__ import annotations

import enum
from unittest.mock import patch

import numpy as np
import pytest
import torch

import helion
from helion._testing import cpu_render_target
from helion._testing import onlyBackends
from helion.autotuner import BooleanFragment
from helion.autotuner import EnumFragment
from helion.autotuner import IntegerFragment
import helion.language as hl

pytestmark = pytest.mark.cpu_only


@helion.kernel(ref_mode=helion.RefMode.EAGER, autotune_effort="none")
def reference_scalar(x, kind: hl.constexpr):
    if kind == "float":
        fragment = EnumFragment((3.5,))
    elif kind == "zero":
        fragment = EnumFragment((-0.0,))
    elif kind == "bool":
        fragment = BooleanFragment()
    else:
        fragment = IntegerFragment(1, 8, 2)
    factor = hl.register_tunable("factor", fragment)
    return x * factor, factor


@pytest.mark.parametrize(
    "kind,selected", [("float", 3.5), ("zero", -0.0), ("bool", False), ("int", 2)]
)
@onlyBackends(["triton", "cute"])
def test_reference_tunable_preserves_default_scalar_type(kind, selected):
    x = torch.arange(4, dtype=torch.float32)
    with cpu_render_target():
        result, actual = reference_scalar(x, kind)
    torch.testing.assert_close(result, x * selected)
    assert type(actual) is type(selected)
    assert actual == selected
    if isinstance(selected, float):
        assert actual.hex() == selected.hex()


@helion.kernel(ref_mode=helion.RefMode.EAGER, autotune_effort="none")
def unsupported_reference_scalar(x):
    return hl.register_tunable("factor", EnumFragment(("not-a-scalar",)))


@onlyBackends(["triton", "cute"])
def test_reference_unsupported_scalar_uses_public_diagnostic():
    with (
        cpu_render_target(),
        pytest.raises(helion.exc.TunableTypeNotSupported, match="str"),
    ):
        unsupported_reference_scalar(torch.empty(1))


class _IntegerEnum(enum.IntEnum):
    VALUE = 2


class _FloatSubclass(float):
    pass


@helion.kernel(autotune_effort="none")
def fragment_scalar(x):
    return hl.register_tunable("factor", FRAGMENT)


FRAGMENT = EnumFragment((2,))


@pytest.mark.parametrize(
    "value", [_IntegerEnum.VALUE, _FloatSubclass(2.5), np.float64(2.5)]
)
@onlyBackends(["triton", "cute"])
def test_reference_and_compiler_reject_scalar_subclass_defaults(value):
    fragment = EnumFragment((value,))
    with (
        patch.dict(fragment_scalar.fn.__globals__, FRAGMENT=fragment),
        cpu_render_target(),
    ):
        compiled = helion.kernel(fragment_scalar.fn, autotune_effort="none")
        with pytest.raises(helion.exc.TunableTypeNotSupported):
            compiled.bind((torch.empty(1),))
        reference = helion.kernel(
            fragment_scalar.fn, ref_mode=helion.RefMode.EAGER, autotune_effort="none"
        )
        with pytest.raises(helion.exc.TunableTypeNotSupported):
            reference(torch.empty(1))


@onlyBackends(["triton", "cute"])
def test_reference_rejects_integer_subclass_before_clamping():
    fragment = IntegerFragment(1, 8, _IntegerEnum.VALUE)
    with (
        patch.dict(fragment_scalar.fn.__globals__, FRAGMENT=fragment),
        cpu_render_target(),
    ):
        compiled = helion.kernel(fragment_scalar.fn, autotune_effort="none")
        with pytest.raises(helion.exc.TunableTypeNotSupported):
            compiled.bind((torch.empty(1),))
        reference = helion.kernel(
            fragment_scalar.fn, ref_mode=helion.RefMode.EAGER, autotune_effort="none"
        )
        with pytest.raises(helion.exc.TunableTypeNotSupported):
            reference(torch.empty(1))
