from __future__ import annotations

import ast
from unittest.mock import patch

import pytest
import torch

import helion
from helion._testing import cpu_render_target
from helion._testing import onlyBackends
from helion._testing.config_fragment_registration import _ValueFragment
from helion._testing.config_fragment_registration import added_state_declarations
from helion._testing.config_fragment_registration import check_shared_fragment_revisit
from helion._testing.config_fragment_registration import container_fragment_revisit
from helion._testing.config_fragment_registration import global_thread_fragment
from helion._testing.config_fragment_registration import grid_fragment
from helion._testing.config_fragment_registration import helper_returned_fragment
from helion._testing.config_fragment_registration import shared_host_fragment
from helion._testing.cute_codegen import cpu_bind
from helion._testing.tunable_fragments import IdentityFragment
from helion._testing.tunable_fragments import check_host_selected_value
from helion._testing.tunable_fragments import host_assignment_values
from helion.autotuner import BooleanFragment
from helion.autotuner import EnumFragment
from helion.autotuner.config_fragment import ConfigSpecFragment
from helion.autotuner.config_fragment import NumThreadsFragment
import helion.language as hl

pytestmark = pytest.mark.cpu_only


def _same_name_builtin(kind: object) -> BooleanFragment | NumThreadsFragment:
    return BooleanFragment() if kind == "bool" else NumThreadsFragment(128)


@helion.kernel(autotune_effort="none")
def same_name_host_builtins(
    x: torch.Tensor, kind: hl.constexpr, helper: hl.constexpr
) -> torch.Tensor:
    if helper:
        first = hl.register_tunable("value", _same_name_builtin(kind))
        second = hl.register_tunable("value", _same_name_builtin(kind))
    elif kind == "bool":
        first = hl.register_tunable("value", BooleanFragment())
        second = hl.register_tunable("value", BooleanFragment())
    else:
        first = hl.register_tunable("value", NumThreadsFragment(128))
        second = hl.register_tunable("value", NumThreadsFragment(128))
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        out[tile] = x[tile] + first + second
    return out


@helion.kernel(autotune_effort="none")
def same_name_supplied_fragments(
    x: torch.Tensor, first: hl.constexpr, second: hl.constexpr
) -> torch.Tensor:
    # pyrefly: ignore [no-matching-overload]
    a = hl.register_tunable("value", first)
    # pyrefly: ignore [no-matching-overload]
    b = hl.register_tunable("value", second)
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        out[tile] = x[tile] + a + b
    return out


@helion.kernel(autotune_effort="none")
def changing_constructor_argument(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        high = 32
        result = x[tile]
        for _index in range(3):
            value = hl.register_tunable("value", NumThreadsFragment(high))
            result = result + value
            high += 32
        out[tile] = result
    return out


@helion.kernel(autotune_effort="none")
def mixed_scalar_declarations(x: torch.Tensor, kind: hl.constexpr) -> torch.Tensor:
    if kind == "float_int":
        first = hl.register_tunable("value", EnumFragment((2.0, 4.0)))
        second = hl.register_tunable("value", EnumFragment((2, 4)))
    elif kind == "int_float":
        first = hl.register_tunable("value", EnumFragment((2, 4)))
        second = hl.register_tunable("value", EnumFragment((2.0, 4.0)))
    elif kind == "bool_int":
        first = hl.register_tunable("value", EnumFragment((False, True)))
        second = hl.register_tunable("value", EnumFragment((0, 1)))
    else:
        first = hl.register_tunable("value", EnumFragment((0, 1)))
        second = hl.register_tunable("value", EnumFragment((False, True)))
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        if kind == "float_int" or kind == "bool_int":
            # pyrefly: ignore [unsupported-operation]
            out[tile] = x[tile] + (second & 1) + first
        else:
            out[tile] = x[tile] + second + first
    return out


class _BooleanSubclass(BooleanFragment):
    pass


class _IntSubclass(int):
    pass


class _FloatSubclass(float):
    pass


class _IndependentInequalityFragment(ConfigSpecFragment):
    def default(self) -> int:
        return 2

    def __eq__(self, other: object) -> bool:
        return True

    def __ne__(self, other: object) -> bool:
        return True


@onlyBackends(["triton", "cute"])
def test_custom_fragment_retains_independent_inequality():
    left, right = _IndependentInequalityFragment(), _IndependentInequalityFragment()
    assert left == right and left != right
    with cpu_render_target(), pytest.raises(helion.exc.TunableNameConflict):
        cpu_bind(same_name_supplied_fragments, (torch.empty(37), left, right))


class _SubclassDefaultFragment(ConfigSpecFragment):
    def __init__(self, scalar_type: type, subclass_at: int) -> None:
        self.scalar_type = scalar_type
        self.subclass_at = subclass_at
        self.calls = 0

    def default(self) -> int | float:
        self.calls += 1
        if self.calls >= self.subclass_at:
            return self.scalar_type(2)
        # pyrefly: ignore [not-callable]
        return self.scalar_type.__base__(2)


@pytest.mark.parametrize("kind", ["bool", "threads"])
@pytest.mark.parametrize("helper", [False, True])
@onlyBackends(["triton", "cute"])
def test_same_name_host_builtins_share_selected_value(kind, helper):
    left, right = _same_name_builtin(kind), _same_name_builtin(kind)
    # Registration equivalence does not change the public Python equality.
    assert left is not right and left != right
    with cpu_render_target():
        args = (torch.empty(37), kind, helper)
        bound = cpu_bind(same_name_host_builtins, args)
        assert list(bound.config_spec.user_defined_tunables) == ["value"]
        captured = bound.config_spec.user_defined_tunables["value"]
        assert type(captured) is type(left)
        check_host_selected_value(
            bound, args, (False, True) if kind == "bool" else (0, 64)
        )


@pytest.mark.parametrize(
    "kind", ["high", "extra_bool", "extra_threads", "subclass", "custom"]
)
@onlyBackends(["triton", "cute"])
def test_same_name_registration_retains_unproved_domain_refusal(kind):
    if kind == "high":
        left, right = NumThreadsFragment(128), NumThreadsFragment(256)
    elif kind in ("extra_bool", "extra_threads"):
        left, right = (
            _same_name_builtin("bool" if kind == "extra_bool" else "threads"),
            _same_name_builtin("bool" if kind == "extra_bool" else "threads"),
        )
        left.extra = right.extra = 3
    elif kind == "subclass":
        left, right = _BooleanSubclass(), _BooleanSubclass()
    else:
        left, right = IdentityFragment(), IdentityFragment()
    with cpu_render_target(), pytest.raises(helion.exc.TunableNameConflict):
        cpu_bind(same_name_supplied_fragments, (torch.empty(37), left, right))


@pytest.mark.parametrize("identity", [False, True])
@onlyBackends(["triton", "cute"])
def test_same_name_registration_preserves_custom_equality(identity):
    left = IdentityFragment() if identity else _ValueFragment()
    right = left if identity else _ValueFragment()
    with cpu_render_target():
        args = (torch.empty(37), left, right)
        bound = cpu_bind(same_name_supplied_fragments, args)
        assert type(bound.config_spec.user_defined_tunables["value"]) is type(left)
        check_host_selected_value(bound, args, (3, 7))


@onlyBackends(["triton", "cute"])
def test_loop_carried_constructor_argument_is_not_cached_as_a_constant():
    with (
        cpu_render_target(),
        pytest.warns(FutureWarning, match="Symbolic config fragment domains"),
        pytest.raises(helion.exc.InvalidConfig, match="Expected integer power of two"),
    ):
        cpu_bind(changing_constructor_argument, (torch.empty(37),))


@onlyBackends(["triton", "cute"])
def test_fresh_identity_only_custom_instances_remain_refused():
    with (
        cpu_render_target(),
        pytest.raises(helion.exc.TypeInferenceError, match="control flow"),
    ):
        cpu_bind(grid_fragment, (torch.empty(37), "custom"))


@pytest.mark.parametrize("kind", ["helper", "changing_custom"])
@onlyBackends(["triton", "cute"])
def test_external_factory_changes_are_observed_on_revisit(kind):
    with (
        patch.dict(grid_fragment.fn.__globals__, _EXTERNAL_STATE=[32, 64]),
        cpu_render_target(),
    ):
        with pytest.raises(helion.exc.TypeInferenceError, match="control flow"):
            cpu_bind(grid_fragment, (torch.empty(37), kind))
        assert not grid_fragment.fn.__globals__["_EXTERNAL_STATE"]


@onlyBackends(["triton", "cute"])
def test_mutation_of_first_constructed_domain_prevents_revisit_reuse():
    with (
        patch.dict(grid_fragment.fn.__globals__, _EXTERNAL_STATE=[]),
        cpu_render_target(),
    ):
        with pytest.raises(
            helion.exc.TypeInferenceError,
            match="Config fragment changed during compilation; keep registered domains immutable",
        ):
            cpu_bind(grid_fragment, (torch.empty(37), "mutating_helper"))
        assert grid_fragment.fn.__globals__["_EXTERNAL_STATE"][0].high == 64


@onlyBackends(["triton", "cute"])
def test_revisit_domain_does_not_use_symbolic_hints():
    with (
        cpu_render_target(),
        pytest.warns(FutureWarning, match="Symbolic config fragment domains"),
        pytest.raises(helion.exc.TypeInferenceError, match="control flow"),
    ):
        cpu_bind(global_thread_fragment, (torch.empty(37), "dynamic", 64))


@pytest.mark.parametrize("kind", ["float_int", "int_float", "bool_int", "int_bool"])
@onlyBackends(["triton", "cute"])
def test_same_name_cannot_reuse_a_different_scalar_type(kind):
    with cpu_render_target(), pytest.raises(helion.exc.TunableNameConflict):
        cpu_bind(mixed_scalar_declarations, (torch.empty(37), kind))


@pytest.mark.parametrize(
    "field,left_values,right_values",
    [
        pytest.param("choices", (2, 4), (2, 4.0), id="choices_type"),
        pytest.param("choices", (2.0, 0.0), (2.0, -0.0), id="choices_zero"),
        pytest.param("search_choices", (2, 4), (2, 4.0), id="search_type"),
        pytest.param("coverage_choices", (0.0,), (-0.0,), id="coverage_zero"),
    ],
)
@pytest.mark.parametrize("reverse", [False, True])
@onlyBackends(["triton", "cute"])
def test_same_name_checks_nondefault_scalar_domains(
    field, left_values, right_values, reverse
):
    choices = (2.0, 0.0) if field == "coverage_choices" else (2, 4)
    left = EnumFragment(**({"choices": choices} | {field: left_values}))
    right = EnumFragment(**({"choices": choices} | {field: right_values}))
    # Public dataclass equality deliberately remains unchanged. Registration
    # additionally checks the complete typed domain, even if defaults agree.
    assert left == right
    assert type(left.default()) is type(right.default())
    if reverse:
        left, right = right, left
    with cpu_render_target(), pytest.raises(helion.exc.TunableNameConflict):
        cpu_bind(same_name_supplied_fragments, (torch.empty(37), left, right))


@pytest.mark.parametrize(
    "fragment,field,values",
    [
        (NumThreadsFragment(32), "high", (32, 64)),
        (EnumFragment((2, 4)), "choices", ((2, 4), (4, 8))),
    ],
)
@onlyBackends(["triton", "cute"])
def test_shared_host_registration_checks_fresh_metadata_on_memo_hit(
    fragment, field, values
):
    remaining = list(values)
    with (
        patch.dict(
            shared_host_fragment.fn.__globals__,
            _EXTERNAL_STATE=[fragment, field, remaining],
        ),
        cpu_render_target(),
        pytest.raises(
            helion.exc.TypeInferenceError,
            match="Config fragment changed during compilation; keep registered domains immutable",
        ),
    ):
        cpu_bind(shared_host_fragment, (torch.empty(37),))
    assert not remaining


@pytest.mark.parametrize("scalar_type", [_IntSubclass, _FloatSubclass])
@pytest.mark.parametrize("subclass_at", [1, 2])
@onlyBackends(["triton", "cute"])
def test_numeric_subclass_default_reports_unsupported_type(scalar_type, subclass_at):
    fragment = _SubclassDefaultFragment(scalar_type, subclass_at)
    with (
        patch.dict(
            shared_host_fragment.fn.__globals__,
            _EXTERNAL_STATE=[fragment, "calls", []],
        ),
        cpu_render_target(),
        pytest.raises(helion.exc.TunableTypeNotSupported),
    ):
        cpu_bind(shared_host_fragment, (torch.empty(37),))
    assert fragment.calls >= subclass_at


@pytest.mark.parametrize("unpack", [False, True])
@onlyBackends(["triton", "cute"])
def test_container_helper_revisit_refuses_changed_domain(unpack):
    with (
        patch.dict(
            container_fragment_revisit.fn.__globals__, _EXTERNAL_STATE=[128, 256]
        ),
        cpu_render_target(),
        pytest.raises(helion.exc.TypeInferenceError, match="control flow"),
    ):
        cpu_bind(container_fragment_revisit, (torch.empty(37), "threads", unpack))


@pytest.mark.parametrize("unpack", [False, True])
@onlyBackends(["triton", "cute"])
def test_container_helper_keeps_custom_identity_equality(unpack):
    with (
        cpu_render_target(),
        pytest.raises(helion.exc.TypeInferenceError, match="control flow"),
    ):
        cpu_bind(container_fragment_revisit, (torch.empty(37), "custom", unpack))


@pytest.mark.parametrize("kind", ["threads", "enum"])
@onlyBackends(["triton", "cute"])
def test_registered_builtin_cannot_bypass_mutation_check_by_adding_state(kind):
    fragment = NumThreadsFragment(32) if kind == "threads" else EnumFragment((2, 4))
    changes = [False, True]
    with (
        patch.dict(
            added_state_declarations.fn.__globals__, _EXTERNAL_STATE=[fragment, changes]
        ),
        cpu_render_target(),
        pytest.raises(
            helion.exc.TypeInferenceError,
            match="Config fragment changed during compilation",
        ),
    ):
        cpu_bind(added_state_declarations, (torch.empty(37),))
    assert not changes


@pytest.mark.parametrize("kind", ["threads", "enum"])
@onlyBackends(["triton", "cute"])
def test_shared_fragment_revisit_refuses_metadata_mutation(kind):
    check_shared_fragment_revisit(kind, True)


@onlyBackends(["triton", "cute"])
def test_module_helper_returning_fragment_keeps_host_selected_value():
    with cpu_render_target():
        bound = cpu_bind(helper_returned_fragment, (torch.empty(37), False))
        config = bound.config_spec.default_config()
        config.config["helper_factor"] = 16
        selected = host_assignment_values(bound, config, "host_value")
        assert len(selected) == 1
        assert isinstance(selected[0], ast.Constant) and selected[0].value == 16
