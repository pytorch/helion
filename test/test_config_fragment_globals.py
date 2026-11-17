from __future__ import annotations

import ast
from dataclasses import dataclass
import linecache
import math
import random
import struct
import threading
from types import ModuleType
from unittest.mock import patch
import warnings

import pytest
import torch
from torch.utils._pytree import tree_leaves

import helion
from helion._testing import cpu_render_target
from helion._testing import onlyBackends
from helion._testing.cute_codegen import cpu_bind
from helion._testing.tunable_fragments import IdentityFragment
from helion._testing.tunable_fragments import host_assignment_values
from helion._testing.tunable_fragments import load_host_function
from helion._testing.tunable_fragments import render_tunable_config
from helion.autotuner import BooleanFragment
from helion.autotuner import EnumFragment
from helion.autotuner import IntegerFragment
from helion.autotuner import LocalAutotuneCache
from helion.autotuner import PowerOfTwoFragment
from helion.autotuner.base_search import BaseSearch
from helion.autotuner.config_fragment import BlockSizeFragment
from helion.autotuner.config_fragment import ConfigSpecFragment
from helion.autotuner.config_fragment import NumThreadsFragment
from helion.autotuner.config_fragment import NumWarpsFragment
import helion.language as hl
from helion.language.tunable_ops import _get_tunable
from helion.runtime.kernel import _make_prepared_arg_guard

pytestmark = pytest.mark.cpu_only

CHOICES = EnumFragment((2, 16))
tuning = ModuleType("tuning")
tuning.CHOICES = CHOICES
REPEATED = BooleanFragment()
tuning.REPEATED = REPEATED
SCALAR = 3
tuning.SCALAR = SCALAR
CHOICES_LIST = [2, 16]


@onlyBackends(["triton", "cute"])
@pytest.mark.parametrize(
    "fragment,selected,expected",
    [
        (EnumFragment((2.5, 4.5)), None, 2.5),
        (EnumFragment((2.5, 4.5)), 4.5, 4.5),
        (BooleanFragment(), None, False),
        (BooleanFragment(), False, False),
        (BooleanFragment(), True, True),
        (IntegerFragment(2, 6, 4), None, 4),
        (IntegerFragment(2, 6, 4), 0, 2),
        (IntegerFragment(2, 6, 4), 9, 6),
    ],
)
def test_reference_tunable_preserves_scalar_type_and_integer_clamping(
    fragment, selected, expected
):
    config = {} if selected is None else {"value": selected}

    @helion.kernel(ref_mode=helion.RefMode.EAGER, config=config)
    def kernel(x: torch.Tensor, domain: hl.constexpr):
        value = hl.register_tunable("value", domain)
        out = torch.empty_like(x)
        for tile in hl.tile(x.numel()):
            out[tile] = x[tile] * value
        return out, value

    x = torch.arange(1, 6, dtype=torch.float32)
    actual, value = kernel(x, fragment)
    torch.testing.assert_close(actual, x * expected, rtol=0, atol=0)
    assert type(value) is type(expected)
    assert value == expected


@helion.kernel(autotune_effort="none")
def global_fragment(x: torch.Tensor) -> torch.Tensor:
    factor = hl.register_tunable("factor", CHOICES)
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        out[tile] = x[tile] * factor
    return out


@helion.kernel(autotune_effort="none")
def inline_fragment(x: torch.Tensor) -> torch.Tensor:
    factor = hl.register_tunable("factor", EnumFragment((2, 16)))
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        out[tile] = x[tile] * factor
    return out


@helion.kernel(autotune_effort="none")
def module_fragment(x: torch.Tensor) -> torch.Tensor:
    factor = hl.register_tunable("factor", tuning.CHOICES)
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        out[tile] = x[tile] * factor
    return out


@helion.kernel(autotune_effort="none")
def global_scalar_fragment(x):
    factor = hl.register_tunable("factor", EnumFragment((SCALAR,)))
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        out[tile] = x[tile] * factor
    return out


@helion.kernel(autotune_effort="none")
def module_scalar_fragment(x):
    factor = hl.register_tunable("factor", EnumFragment((tuning.SCALAR,)))
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        out[tile] = x[tile] * factor
    return out


@helion.kernel(autotune_effort="none")
def derived_global_fragment(x, operation: hl.constexpr):
    if operation == "not":
        choices = (False, not SCALAR)
    elif operation == "negate":
        choices = (-SCALAR,)
    elif operation == "negate_divide":
        choices = ((-SCALAR) / 2,)
    elif operation == "divide":
        choices = (0.0, (SCALAR + 1) / 2)
    else:
        choices = (0, SCALAR + 1)
    factor = hl.register_tunable("factor", EnumFragment(choices))
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        out[tile] = x[tile] * factor
    return out


@helion.kernel(autotune_effort="none")
def converted_global_fragment(x):
    factor = hl.register_tunable("factor", EnumFragment(tuple(CHOICES_LIST)))
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        out[tile] = x[tile] * factor
    return out


@helion.kernel(autotune_effort="none", static_shapes=False)
def converted_symbolic_fragment(x, derived: hl.constexpr):
    if derived:
        choices = [SCALAR + 1, SCALAR + 2]
    else:
        choices = [x.size(0), 16]
    factor = hl.register_tunable("factor", EnumFragment(tuple(choices)))
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        out[tile] = x[tile] * factor
    return out


@helion.kernel(autotune_effort="none")
def repeated_global_fragment(x: torch.Tensor) -> torch.Tensor:
    first = hl.register_tunable("factor", REPEATED)
    second = hl.register_tunable("factor", REPEATED)
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        out[tile] = x[tile] + first + second
    return out


@helion.kernel(autotune_effort="none")
def repeated_module_fragment(x: torch.Tensor) -> torch.Tensor:
    first = hl.register_tunable("factor", tuning.REPEATED)
    second = hl.register_tunable("factor", tuning.REPEATED)
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        out[tile] = x[tile] + first + second
    return out


@helion.kernel(autotune_effort="none")
def device_fragment(x: torch.Tensor) -> torch.Tensor:
    host = hl.register_tunable("factor", tuning.REPEATED)
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        first = hl.register_tunable("factor", tuning.REPEATED)
        second = hl.register_tunable("factor", tuning.REPEATED)
        out[tile] = x[tile] + host + first + second
    return out


@helion.kernel(autotune_effort="none")
def device_only_fragment(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        factor = hl.register_tunable("factor", tuning.REPEATED)
        out[tile] = x[tile] * factor
    return out


class _InlineFragment(IdentityFragment):
    def __init__(self):
        super().__init__()
        # Modules cannot be deep-copied and remain shared opaque search state.
        self.rng = random


class _CopyableFragment(_InlineFragment):
    def __deepcopy__(self, memo):
        result = type(self)()
        result.value = self.value
        memo[id(self)] = result
        return result


@dataclass(frozen=True)
class _HashableFragment(ConfigSpecFragment):
    value: int = 3

    def default(self):
        return self.value


@helion.kernel(autotune_effort="none")
def constexpr_fragment(x, fragment: hl.constexpr):
    factor = hl.register_tunable("factor", fragment)
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        out[tile] = x[tile] * factor
    return out


def _identity_fragment(fragment):
    return fragment


def _make_enum_fragment(value, zero):
    return EnumFragment((zero, value))


def _make_single_fragment(value):
    return EnumFragment((value,))


def _make_inline_fragment():
    return _InlineFragment()


def _get_shared_fragment():
    return CHOICES


def _publish_fragment():
    fragment = EnumFragment((2, 16))
    tuning.CREATED = fragment
    return fragment


tuning.get = _get_shared_fragment


@helion.kernel(autotune_effort="none")
def helper_shared_fragment(x):
    factor = hl.register_tunable("factor", tuning.get())
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        out[tile] = x[tile] * factor
    return out


@helion.kernel(autotune_effort="none")
def helper_published_fragment(x):
    factor = hl.register_tunable("factor", _publish_fragment())
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        out[tile] = x[tile] * factor
    return out


@helion.kernel(autotune_effort="none")
def helper_inline_fragment(x):
    fragment = _make_inline_fragment()
    first = hl.register_tunable("factor", fragment)
    second = hl.register_tunable("factor", _identity_fragment(fragment))
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        out[tile] = x[tile] * first + second
    return out


@helion.kernel(autotune_effort="none")
def helper_constexpr_fragment(x, fragment: hl.constexpr):
    factor = hl.register_tunable("factor", _identity_fragment(fragment))
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        out[tile] = x[tile] * factor
    return out


@helion.kernel(autotune_effort="none")
def helper_numeric_fragment(x, value, zero: hl.constexpr):
    factor = hl.register_tunable("factor", _make_enum_fragment(value, zero))
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        out[tile] = x[tile] * factor
    return out


@helion.kernel(autotune_effort="none")
def helper_concrete_fragment(x, value: hl.constexpr, use_global: hl.constexpr):
    if use_global:
        fragment = _make_single_fragment(SCALAR)
    else:
        fragment = _make_single_fragment(value)
    factor = hl.register_tunable("factor", fragment)
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        out[tile] = x[tile] * factor
    return out


@helion.kernel(autotune_effort="none")
def inline_custom_fragment(x):
    fragment = _InlineFragment()
    first = hl.register_tunable("factor", fragment)
    second = hl.register_tunable("factor", fragment)
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        out[tile] = x[tile] * first + second
    return out


@helion.kernel(autotune_effort="none")
def numeric_fragment(x, value, zero: hl.constexpr):
    factor = hl.register_tunable("factor", EnumFragment((zero, value)))
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        out[tile] = x[tile] * factor
    return out


@helion.kernel(autotune_effort="none")
def constexpr_numeric_fragment(x, value: hl.constexpr):
    factor = hl.register_tunable("factor", EnumFragment((value,)))
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        out[tile] = x[tile] * factor
    return out


@helion.kernel(autotune_effort="none")
def ordinary_minimum(x, limit):
    factor = min(limit, 4)
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        out[tile] = x[tile] * factor
    return out


@dataclass
class _FragmentBox:
    choices: EnumFragment


@helion.kernel(autotune_effort="none")
def argument_fragment(x, fragment, container: hl.constexpr):
    if container == "nested":
        tunable = fragment[0]["choices"]
    elif container == "dataclass":
        tunable = fragment.choices
    else:
        tunable = fragment
    factor = hl.register_tunable("factor", tunable)
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        out[tile] = x[tile] * factor
    return out


@helion.kernel(autotune_effort="none", static_shapes=False)
def symbolic_compatibility_fragment(x, n, form: hl.constexpr):
    if form == "enum":
        fragment = EnumFragment((1, n))
    else:
        fragment = IntegerFragment(1, x.size(0))
    factor = hl.register_tunable("factor", fragment)
    again = hl.register_tunable("factor", fragment)
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        out[tile] = x[tile] * factor + again
    return out


@helion.kernel(autotune_effort="none", static_shapes=False)
def dynamic_fragment(x, form: hl.constexpr):
    size = x.size(0)
    if form == "positional":
        fragment = PowerOfTwoFragment(16, size)
    elif form == "keyword":
        fragment = PowerOfTwoFragment(low=16, high=size)
    elif form == "nested":
        fragment = EnumFragment((2, size))
    else:
        fragment = PowerOfTwoFragment(16, hl.specialize(size))
    factor = hl.register_tunable("factor", fragment)
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        out[tile] = x[tile] * factor
    return out


@helion.kernel(autotune_effort="none")
def conflicting_fragments(x):
    first = hl.register_tunable("factor", EnumFragment((2, 16)))
    second = hl.register_tunable("factor", EnumFragment((4, 8)))
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        out[tile] = x[tile] + first + second
    return out


def closure_fragment():
    choices = EnumFragment((2, 16))

    @helion.kernel(autotune_effort="none")
    def kernel(x: torch.Tensor) -> torch.Tensor:
        factor = hl.register_tunable("factor", choices)
        out = torch.empty_like(x)
        for tile in hl.tile(x.numel()):
            out[tile] = x[tile] * factor
        return out

    return kernel


@pytest.mark.parametrize("kind", ["global", "inline", "module"])
@onlyBackends(["triton", "cute"])
def test_preconstructed_fragment_preserves_choices_and_codegen(kind):
    kernel = {
        "global": global_fragment,
        "inline": inline_fragment,
        "module": module_fragment,
    }[kind]
    with cpu_render_target():
        bound = cpu_bind(kernel, (torch.empty(37),))
        fragment = bound.config_spec.user_defined_tunables["factor"]
        assert isinstance(fragment, EnumFragment)
        assert fragment.choices == (2, 16)
        assert all(type(value) is int for value in fragment.choices)
        config = bound.config_spec.default_config()
        sources = []
        for value in (2, 16):
            config.config["factor"] = value
            source = bound.to_code(config)
            ast.parse(source)
            sources.append(source)
        assert sources[0] != sources[1]
    assert CHOICES.choices == (2, 16)
    assert all(type(value) is int for value in CHOICES.choices)


def test_kernel_closure_remains_unsupported():
    with pytest.raises(helion.exc.ClosuresNotSupported):
        closure_fragment()


@pytest.mark.parametrize("helper", [False, True])
@onlyBackends(["triton", "cute"])
def test_inline_custom_fragment_preserves_uncopyable_search_state(helper):
    kernel = helper_inline_fragment if helper else inline_custom_fragment
    with cpu_render_target():
        bound = cpu_bind(kernel, (torch.empty(37),))
        captured = bound.config_spec.user_defined_tunables["factor"]
        assert isinstance(captured, _InlineFragment)
        assert captured.rng is random and captured.default() == 3
        ast.parse(bound.to_code(bound.config_spec.default_config()))


@onlyBackends(["triton", "cute"])
def test_shared_custom_fragment_retains_original_object():
    fragment = _CopyableFragment()
    with cpu_render_target():
        bound = cpu_bind(helper_constexpr_fragment, (torch.empty(37), fragment))
        captured = bound.config_spec.user_defined_tunables["factor"]
        assert captured is fragment and captured.rng is random
        assert captured.default() == 3
        ast.parse(bound.to_code(bound.config_spec.default_config()))


@pytest.mark.parametrize("state_kind", ["module", "lock"])
@pytest.mark.parametrize("form", ["annotated", "wrapped"])
@onlyBackends(["triton", "cute"])
def test_uncopyable_shared_fragment_retains_opaque_state(state_kind, form):
    fragment = IdentityFragment()
    fragment.search_state = random if state_kind == "module" else threading.Lock()
    if form == "annotated":
        kernel, args = helper_constexpr_fragment, (torch.empty(37), fragment)
    else:
        kernel, args = (
            argument_fragment,
            (
                torch.empty(37),
                hl.constexpr(fragment),
                "direct",
            ),
        )
    with cpu_render_target():
        bound = cpu_bind(kernel, args)
        captured = bound.config_spec.user_defined_tunables["factor"]
        assert captured is fragment
        assert captured.search_state is fragment.search_state
    assert fragment.default() == 3


@pytest.mark.parametrize("value,zero", [(3, 0), (3.5, 0.0), (True, False)])
@onlyBackends(["triton", "cute"])
def test_symbolic_numeric_fragment_arguments_warn_and_render(value, zero):
    with (
        cpu_render_target(),
        pytest.warns(FutureWarning, match="Symbolic config fragment domains"),
    ):
        bound = cpu_bind(numeric_fragment, (torch.empty(37), value, zero))
        captured = bound.config_spec.user_defined_tunables["factor"]
        assert isinstance(
            captured.choices[1], (torch.SymInt, torch.SymFloat, torch.SymBool)
        )
        assert not bound.env.tunable_fragment_snapshots
        config = bound.config_spec.default_config()
        config.config["factor"] = value
        ast.parse(bound.to_code(config))


@pytest.mark.parametrize("value", [3, 3.5, True])
@onlyBackends(["triton", "cute"])
def test_constexpr_numeric_fragment_arguments_remain_supported(value):
    with cpu_render_target():
        bound = cpu_bind(constexpr_numeric_fragment, (torch.empty(37), value))
        captured = bound.config_spec.user_defined_tunables["factor"]
        assert captured.default() == value
        assert type(captured.default()) is type(value)
        ast.parse(bound.to_code(bound.config_spec.default_config()))


@pytest.mark.parametrize("value,zero", [(3, 0), (3.5, 0.0), (True, False)])
@onlyBackends(["triton", "cute"])
def test_helper_fragment_retains_symbolic_metadata(value, zero):
    with (
        cpu_render_target(),
        pytest.warns(FutureWarning, match="Symbolic config fragment domains"),
    ):
        bound = cpu_bind(helper_numeric_fragment, (torch.empty(37), value, zero))
        captured = bound.config_spec.user_defined_tunables["factor"]
        assert isinstance(
            captured.choices[1], (torch.SymInt, torch.SymFloat, torch.SymBool)
        )
        assert not bound.env.tunable_fragment_snapshots
        config = bound.config_spec.default_config()
        config.config["factor"] = value
        ast.parse(bound.to_code(config))


@pytest.mark.parametrize("value", [3, 3.5, True])
@pytest.mark.parametrize("use_global", [False, True])
@onlyBackends(["triton", "cute"])
def test_helper_fragment_keeps_concrete_metadata(value, use_global):
    with (
        patch.dict(helper_concrete_fragment.fn.__globals__, SCALAR=value),
        cpu_render_target(),
    ):
        bound = cpu_bind(helper_concrete_fragment, (torch.empty(37), value, use_global))
        captured = bound.config_spec.user_defined_tunables["factor"]
        assert captured.default() == value and type(captured.default()) is type(value)
        ast.parse(bound.to_code(bound.config_spec.default_config()))


@onlyBackends(["triton", "cute"])
def test_helper_constexpr_fragment_keeps_snapshot_after_caller_mutation():
    kernel = helion.kernel(helper_constexpr_fragment.fn, autotune_effort="none")
    tensor = torch.empty(37)
    with cpu_render_target():
        fragment = EnumFragment((2, 16))
        bound = kernel.bind((tensor, fragment))
        captured = bound.config_spec.user_defined_tunables["factor"]
        original_key = _persistent_key(bound, (tensor, fragment))
        ast.parse(bound.to_code(bound.config_spec.default_config()))
        fragment.choices = (4, 8)
        assert captured.choices == (2, 16)
        fresh_args = (tensor, EnumFragment((2, 16)))
        fresh = kernel.bind(fresh_args)
        # Opaque helper calls rebind fresh fragment identities, while their
        # captured domain and persistent autotune identity remain unchanged.
        assert fresh is not bound
        assert fresh.config_spec.user_defined_tunables["factor"].choices == (2, 16)
        assert (
            _persistent_key(fresh, fresh_args).stable_hash()
            == original_key.stable_hash()
        )
        changed = kernel.bind((tensor, fragment))
        assert changed is not bound
        assert changed.config_spec.user_defined_tunables["factor"].choices == (4, 8)
        assert (
            _persistent_key(changed, (tensor, fragment)).stable_hash()
            != original_key.stable_hash()
        )


@onlyBackends(["triton", "cute"])
def test_helper_shared_fragment_snapshots_its_domain():
    fragment = EnumFragment((2, 16))
    with (
        patch.dict(helper_shared_fragment.fn.__globals__, CHOICES=fragment),
        cpu_render_target(),
    ):
        bound = cpu_bind(helper_shared_fragment, (torch.empty(37),))
        captured = bound.config_spec.user_defined_tunables["factor"]
        fingerprint = bound.config_spec.cache_fingerprint_hash()
        fragment.choices = (4, 8)
        assert captured.choices == (2, 16)
        assert bound.config_spec.cache_fingerprint_hash() == fingerprint
        ast.parse(bound.to_code(bound.config_spec.default_config()))


@onlyBackends(["triton", "cute"])
def test_helper_published_allocation_still_requires_a_snapshot():
    with patch.dict(vars(tuning), CREATED=None), cpu_render_target():
        bound = cpu_bind(helper_published_fragment, (torch.empty(37),))
        captured = bound.config_spec.user_defined_tunables["factor"]
        fingerprint = bound.config_spec.cache_fingerprint_hash()
        tuning.CREATED.choices = (4, 8)
        assert captured.choices == (2, 16)
        assert bound.config_spec.cache_fingerprint_hash() == fingerprint
        ast.parse(bound.to_code(bound.config_spec.default_config()))


@onlyBackends(["triton", "cute"])
def test_symbolic_nonfragment_callable_remains_dynamic():
    kernel = helion.kernel(ordinary_minimum.fn, autotune_effort="none")
    with cpu_render_target():
        tensor = torch.empty(37)
        bound = kernel.bind((tensor, 3))
        assert isinstance(bound.fake_args[1], torch.SymInt)
        assert kernel.bind((tensor, 7)) is bound
        module = ast.parse(bound.to_code(bound.config_spec.default_config()))
        assert any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "min"
            for node in ast.walk(module)
        )


@pytest.mark.parametrize("kind", ["global", "module"])
@pytest.mark.parametrize("old,new", [(3, 5), (3.5, 5.5), (True, False), (-0.0, 0.0)])
@onlyBackends(["triton", "cute"])
def test_literal_global_numeric_metadata_is_snapshotted(kind, old, new):
    original = global_scalar_fragment if kind == "global" else module_scalar_fragment
    kernel = helion.kernel(original.fn, autotune_effort="none")
    namespace = original.fn.__globals__ if kind == "global" else vars(tuning)
    with patch.dict(namespace, SCALAR=old), cpu_render_target():
        args = (torch.empty(37),)
        bound = kernel.bind(args)
        captured = bound.config_spec.user_defined_tunables["factor"]
        assert type(captured.default()) is type(old) and captured.default() == old
        if isinstance(old, float):
            assert math.copysign(1, captured.default()) == math.copysign(1, old)
        ast.parse(bound.to_code(bound.config_spec.default_config()))
        namespace["SCALAR"] = new
        assert kernel.bind(args) is bound
        assert captured.default() == old
        fresh = cpu_bind(kernel, args)
        assert fresh.config_spec.user_defined_tunables["factor"].default() == new
        ast.parse(fresh.to_code(fresh.config_spec.default_config()))


@pytest.mark.parametrize(
    "operation,old,new,expected_old,expected_new",
    [
        ("add", 3, 5, (0, 4), (0, 6)),
        ("add", 3.5, 5.5, (0, 4.5), (0, 6.5)),
        ("divide", 3, 5, (0.0, 2.0), (0.0, 3.0)),
        ("not", True, False, (False, False), (False, True)),
        ("negate", 0.0, -0.0, (-0.0,), (0.0,)),
        ("negate_divide", 0.0, -0.0, (-0.0,), (0.0,)),
    ],
)
@onlyBackends(["triton", "cute"])
def test_derived_global_expression_keeps_typed_literal_snapshot(
    operation, old, new, expected_old, expected_new
):
    kernel = helion.kernel(derived_global_fragment.fn, autotune_effort="none")
    with patch.dict(kernel.fn.__globals__, SCALAR=old), cpu_render_target():
        args = (torch.empty(37), operation)
        bound = kernel.bind(args)
        kernel.fn.__globals__["SCALAR"] = new
        assert kernel.bind(args) is bound
        fresh = cpu_bind(kernel, args)
        for current, expected in ((bound, expected_old), (fresh, expected_new)):
            captured = current.config_spec.user_defined_tunables["factor"]
            assert captured.choices == expected
            assert tuple(map(type, captured.choices)) == tuple(map(type, expected))
            for actual, wanted in zip(captured.choices, expected, strict=True):
                if isinstance(wanted, float):
                    assert actual.hex() == wanted.hex()
            config = current.config_spec.default_config()
            config.config["factor"] = expected[-1]
            assignments = host_assignment_values(current, config, "factor")
            assert len(assignments) == 1
            emitted = ast.literal_eval(assignments[0])
            assert type(emitted) is type(expected[-1]) and emitted == expected[-1]
            if isinstance(emitted, float):
                assert emitted.hex() == expected[-1].hex()


@pytest.mark.parametrize(
    "old,new",
    [((2, 16), (4, 8)), ((2.5, 4.5), (3.5, 5.5)), ((False, True), (True, False))],
)
@onlyBackends(["triton", "cute"])
def test_converted_global_literal_sequence_preserves_snapshot(old, new):
    kernel = helion.kernel(converted_global_fragment.fn, autotune_effort="none")
    choices = list(old)
    with (
        patch.dict(kernel.fn.__globals__, CHOICES_LIST=choices),
        cpu_render_target(),
    ):
        args = (torch.empty(37),)
        bound = kernel.bind(args)
        captured = bound.config_spec.user_defined_tunables["factor"]
        assert captured.choices == old
        assert tuple(map(type, captured.choices)) == tuple(map(type, old))
        choices[:] = new
        assert kernel.bind(args) is bound
        assert captured.choices == old
        fresh = cpu_bind(kernel, args)
        assert fresh.config_spec.user_defined_tunables["factor"].choices == new
        for current, selected in ((bound, old[-1]), (fresh, new[-1])):
            config = current.config_spec.default_config()
            config.config["factor"] = selected
            assignments = host_assignment_values(current, config, "factor")
            assert len(assignments) == 1
            value = assignments[0]
            assert isinstance(value, ast.Constant)
            assert type(value.value) is type(selected) and value.value == selected


@onlyBackends(["triton", "cute"])
def test_converted_nonliteral_sequence_rejects_symbolic_default():
    with (
        cpu_render_target(),
        pytest.warns(FutureWarning, match="Symbolic config fragment domains"),
        pytest.raises(helion.exc.TunableTypeNotSupported),
    ):
        cpu_bind(converted_symbolic_fragment, (torch.empty(37), False))


@onlyBackends(["triton", "cute"])
def test_converted_derived_global_sequence_has_concrete_domain():
    with cpu_render_target():
        bound = cpu_bind(converted_symbolic_fragment, (torch.empty(37), True))
        assert bound.config_spec.user_defined_tunables["factor"].choices == (4, 5)
        ast.parse(bound.to_code(bound.config_spec.default_config()))


def _persistent_key(bound, args):
    # Exercise actual cache construction; only the hardware query is modelled.
    # cpu_render_target's lazy-init guard continues to forbid native CUDA work.
    with (
        patch(
            "helion.autotuner.local_cache.extract_device",
            return_value=torch.device("cuda:0"),
        ),
        patch(
            "helion.autotuner.local_cache.get_device_name",
            return_value="cpu-test-target",
        ),
        patch("torch.cuda.is_available", return_value=True),
    ):
        return LocalAutotuneCache(BaseSearch(bound, args)).key


@pytest.mark.parametrize("form", ["annotation", "wrapper"])
@pytest.mark.parametrize(
    "fragment_type", [EnumFragment, IntegerFragment, PowerOfTwoFragment]
)
@onlyBackends(["triton", "cute"])
def test_constexpr_fragment_normal_bind_and_persistent_key(form, fragment_type):
    def make_fragment():
        return (
            EnumFragment((2, 8))
            if fragment_type is EnumFragment
            else fragment_type(2, 8, 2)
        )

    original = constexpr_fragment if form == "annotation" else argument_fragment
    kernel = helion.kernel(original.fn, autotune_effort="none")
    tensor = torch.empty(37)

    def arguments(fragment):
        return (
            (tensor, fragment)
            if form == "annotation"
            else (tensor, hl.constexpr(fragment), "direct")
        )

    with cpu_render_target():
        fragment = make_fragment()
        args = arguments(fragment)
        bound = kernel.bind(args)
        assert kernel.bind(args) is bound
        assert kernel.bind(arguments(make_fragment())) is bound
        captured = bound.config_spec.user_defined_tunables["factor"]
        assert captured is not fragment and captured.default() == 2
        ast.parse(bound.to_code(bound.config_spec.default_config()))
        key = _persistent_key(bound, args)
        original_repr = repr(key)
        assert hex(id(fragment)) not in original_repr
        if isinstance(fragment, EnumFragment):
            fragment.choices = (4, 8)
        else:
            fragment.default_val = 4
        changed = kernel.bind(args)
        assert changed is not bound and captured.default() == 2
        assert changed.config_spec.user_defined_tunables["factor"].default() == 4
        assert repr(key) == original_repr
        changed_key = _persistent_key(changed, args)
        assert changed_key.stable_hash() != key.stable_hash()
        ast.parse(changed.to_code(changed.config_spec.default_config()))
        # A fresh bind with the same declaration has the same persistent key;
        # no retained object identity or process address enters serialization.
        fresh_kernel = helion.kernel(original.fn, autotune_effort="none")
        fresh_fragment = make_fragment()
        if isinstance(fresh_fragment, EnumFragment):
            fresh_fragment.choices = (4, 8)
        else:
            fresh_fragment.default_val = 4
        fresh_args = arguments(fresh_fragment)
        fresh = fresh_kernel.bind(fresh_args)
        assert (
            _persistent_key(fresh, fresh_args).stable_hash()
            == changed_key.stable_hash()
        )


@onlyBackends(["triton", "cute"])
def test_hashable_constexpr_fragment_equality_is_unchanged():
    kernel = helion.kernel(constexpr_fragment.fn, autotune_effort="none")
    with cpu_render_target():
        tensor = torch.empty(37)
        first = kernel.bind((tensor, _HashableFragment(3)))
        assert kernel.bind((tensor, _HashableFragment(3))) is first
        assert kernel.bind((tensor, _HashableFragment(5))) is not first
        ast.parse(first.to_code(first.config_spec.default_config()))


@pytest.mark.parametrize("kind", ["boolean", "threads", "block_size", "num_warps"])
@onlyBackends(["triton", "cute"])
def test_other_builtin_constexpr_fragments_have_stable_value_keys(kind):
    def make():
        if kind == "boolean":
            return BooleanFragment()
        if kind == "threads":
            return NumThreadsFragment(128)
        if kind == "block_size":
            return BlockSizeFragment(2, 8, 2)
        return NumWarpsFragment(2, 8, 2)

    kernel = helion.kernel(constexpr_fragment.fn, autotune_effort="none")
    tensor = torch.empty(37)
    with cpu_render_target():
        fragment = make()
        bound = kernel.bind((tensor, fragment))
        assert kernel.bind((tensor, make())) is bound
        key = _persistent_key(bound, (tensor, fragment))
        assert hex(id(fragment)) not in repr(key)
        fresh_kernel = helion.kernel(constexpr_fragment.fn, autotune_effort="none")
        fresh_fragment = make()
        fresh = fresh_kernel.bind((tensor, fresh_fragment))
        assert (
            _persistent_key(fresh, (tensor, fresh_fragment)).stable_hash()
            == key.stable_hash()
        )
        ast.parse(bound.to_code(bound.config_spec.default_config()))
        if kind != "boolean":
            original = bound.config_spec.cache_fingerprint_hash()
            projected = bound.config_spec.projected_cache_fingerprint_hash()
            fragment.high *= 2
            changed = kernel.bind((tensor, fragment))
            assert changed is not bound
            assert changed.config_spec.cache_fingerprint_hash() != original
            assert changed.config_spec.projected_cache_fingerprint_hash() != projected


@onlyBackends(["triton", "cute"])
def test_constexpr_enum_key_preserves_signed_zero():
    kernel = helion.kernel(constexpr_fragment.fn, autotune_effort="none")
    with cpu_render_target():
        tensor = torch.empty(37)
        positive_args = (tensor, EnumFragment((0.0,)))
        negative_args = (tensor, EnumFragment((-0.0,)))
        positive = kernel.bind(positive_args)
        negative = kernel.bind(negative_args)
        assert positive is not negative
        assert (
            positive.config_spec.user_defined_tunables["factor"].default().hex()
            == (0.0).hex()
        )
        assert (
            negative.config_spec.user_defined_tunables["factor"].default().hex()
            == (-0.0).hex()
        )
        assert (
            _persistent_key(positive, positive_args).stable_hash()
            != _persistent_key(negative, negative_args).stable_hash()
        )


@pytest.mark.parametrize("form", ["annotation", "wrapper"])
@onlyBackends(["triton", "cute"])
def test_constexpr_enum_nan_bits_preserve_binding_and_cache_identity(form):
    original = constexpr_fragment if form == "annotation" else argument_fragment
    kernel = helion.kernel(original.fn, autotune_effort="none")
    tensor = torch.empty(37)
    # Isolate payload and sign changes; a fresh NaN object with the same bits
    # must still reuse the binding instead of falling back to object identity.
    payloads = ("7ff8000000000001", "7ff8000000000002", "fff8000000000001")

    def arguments(payload):
        value = struct.unpack("!d", bytes.fromhex(payload))[0]
        fragment = EnumFragment((value,))
        return (
            (tensor, fragment)
            if form == "annotation"
            else (tensor, hl.constexpr(fragment), "direct")
        )

    def captured_bits(bound):
        value = bound.config_spec.user_defined_tunables["factor"].default()
        return struct.pack("!d", value).hex()

    with cpu_render_target():
        bounds = [kernel.bind(arguments(payload)) for payload in payloads]
        assert len({id(bound) for bound in bounds}) == len(payloads)
        persistent = []
        for payload, bound in zip(payloads, bounds, strict=True):
            args = arguments(payload)
            assert kernel.bind(args) is bound
            assert captured_bits(bound) == payload
            fresh = helion.kernel(original.fn, autotune_effort="none").bind(args)
            assert captured_bits(fresh) == payload
            key = _persistent_key(bound, args).stable_hash()
            assert _persistent_key(fresh, args).stable_hash() == key
            persistent.append(key)
            guard = _make_prepared_arg_guard(kernel, args)
            for other in payloads:
                assert guard(arguments(other)) is (other == payload)
        assert len(set(persistent)) == len(payloads)
        assert len({kernel._fast_dispatch_key(arguments(p)) for p in payloads}) == 3


@pytest.mark.parametrize("form", ["object", "bound_method", "constexpr_object"])
@onlyBackends(["triton", "cute"])
def test_constexpr_enum_with_opaque_choices_keeps_prior_refusal(form):
    choice = {
        "object": object(),
        "bound_method": {}.get,
        "constexpr_object": hl.constexpr(object()),
    }[form]
    kernel = helion.kernel(constexpr_fragment.fn, autotune_effort="none")
    with (
        cpu_render_target(),
        pytest.raises(TypeError, match="unhashable.*EnumFragment"),
    ):
        kernel.bind((torch.empty(37), EnumFragment((2, choice))))


@onlyBackends(["triton", "cute"])
@pytest.mark.parametrize("kind", ["global", "module"])
def test_global_fragment_snapshot_preserves_bound_cache_identity(kind):
    fragment = EnumFragment((2, 16))
    original = global_fragment if kind == "global" else module_fragment
    kernel = helion.kernel(original.fn, autotune_effort="none")
    namespace = global_fragment.fn.__globals__ if kind == "global" else vars(tuning)
    with (
        patch.dict(namespace, CHOICES=fragment),
        cpu_render_target(),
    ):
        args = (torch.empty(37),)
        first = kernel.bind(args)
        captured = first.config_spec.user_defined_tunables["factor"]
        assert isinstance(captured, EnumFragment)
        fingerprint = first.config_spec.cache_fingerprint_hash()
        fragment.choices = (4, 8)
        assert captured.choices == (2, 16)
        assert first.config_spec.cache_fingerprint_hash() == fingerprint
        assert captured is not fragment
        assert kernel.bind(args) is first
        # A new isolated binding reads the new global metadata. Existing
        # cached kernels still follow the ordinary specialize-once policy.
        second = cpu_bind(kernel, (torch.empty(37),))
        recaptured = second.config_spec.user_defined_tunables["factor"]
        assert isinstance(recaptured, EnumFragment)
        assert recaptured.choices == (4, 8)
        assert second.config_spec.cache_fingerprint_hash() != fingerprint


@pytest.mark.parametrize("kind", ["global", "module"])
@pytest.mark.parametrize(
    "fragment", [BooleanFragment(), NumThreadsFragment(128), IdentityFragment()]
)
@onlyBackends(["triton", "cute"])
def test_repeated_registration_preserves_builtin_snapshot_or_custom_identity(
    kind, fragment
):
    kernel = repeated_global_fragment if kind == "global" else repeated_module_fragment
    namespace = kernel.fn.__globals__ if kind == "global" else vars(tuning)
    with patch.dict(namespace, REPEATED=fragment), cpu_render_target():
        bound = cpu_bind(kernel, (torch.empty(37),))
        captured = bound.config_spec.user_defined_tunables["factor"]
        assert (captured is fragment) is isinstance(fragment, IdentityFragment)
        assert type(captured) is type(fragment)
        assert captured.default() == fragment.default()
        ast.parse(bound.to_code(bound.config_spec.default_config()))


@pytest.mark.parametrize(
    "fragment,values,symbolic_type",
    [
        (BooleanFragment(), (False, True), torch.SymBool),
        (NumThreadsFragment(128), (0, 64), torch.SymInt),
        (EnumFragment((2, 16)), (2, 16), torch.SymInt),
    ],
)
@pytest.mark.parametrize("kernel", [device_fragment, device_only_fragment])
@onlyBackends(["triton", "cute"])
def test_device_registration_carries_selected_typed_value(
    fragment, values, symbolic_type, kernel
):
    with patch.object(tuning, "REPEATED", fragment), cpu_render_target():
        x = torch.arange(37, dtype=torch.float32)
        bound = cpu_bind(kernel, (x,))
        captured = bound.config_spec.user_defined_tunables["factor"]
        assert captured is not fragment and type(captured) is type(fragment)
        graphs = bound.host_function.device_ir.graphs
        nodes = [node for graph in graphs for node in graph.graph.nodes]
        assert not any(
            isinstance(value, ConfigSpecFragment)
            for node in nodes
            for value in tree_leaves((node.args, node.kwargs))
        )
        reads = [node for node in nodes if node.target is _get_tunable]
        # Separate declaration sites retain separate symbols, even when they
        # share a config key. Repeated visits to one site still reuse its read.
        assert len(reads) == (2 if kernel is device_fragment else 1)
        assert all(node.args == ("factor",) and not node.kwargs for node in reads)
        assert all(isinstance(node.meta["val"], symbolic_type) for node in reads)
        sources = [render_tunable_config(bound, "factor", value) for value in values]
        assert sources[0] != sources[1]
        for value in values:
            config = bound.config_spec.default_config()
            config.config["factor"] = value
            launches = []

            def observe_launch(
                device, grid, input_tensor, out, *selected, records=launches, **kwargs
            ):
                assert input_tensor is x
                records.append(selected)

            # Execute the real host wrapper, replacing only the GPU launch.
            # The host declaration and device constants must select the same
            # Python scalar type; the device-only case has no scalar argument.
            load_host_function(bound, config, observe_launch)(x)
            expected_args = (value,) if kernel is device_fragment else ()
            assert launches == [expected_args]
            assert all(type(item) is type(value) for item in launches[0])
            reference = helion.kernel(
                kernel.fn, config=config, ref_mode=helion.RefMode.EAGER
            )
            expected = (
                x + value + value + value if kernel is device_fragment else x * value
            )
            torch.testing.assert_close(reference(x), expected, rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["global", "module"])
@pytest.mark.parametrize("fragment_type", [IntegerFragment, PowerOfTwoFragment])
@onlyBackends(["triton", "cute"])
def test_global_integer_fragment_domains_have_distinct_persistent_identity(
    kind, fragment_type
):
    fragment = fragment_type(2, 16, 2)
    kernel = global_fragment if kind == "global" else module_fragment
    namespace = kernel.fn.__globals__ if kind == "global" else vars(tuning)
    with (
        patch.dict(namespace, CHOICES=fragment),
        cpu_render_target(),
    ):
        bound = cpu_bind(kernel, (torch.empty(37),))
        captured = bound.config_spec.user_defined_tunables["factor"]
        assert type(captured) is type(fragment)
        assert type(captured.low) is int and type(captured.high) is int
        ast.parse(bound.to_code(bound.config_spec.default_config()))
        layout = bound.config_spec.structural_fingerprint()
        identity = bound.config_spec.cache_fingerprint_hash()
        projected = bound.config_spec.projected_cache_fingerprint_hash()
        fragment.low, fragment.high, fragment.default_val = 4, 8, 4
        assert (captured.low, captured.high, captured.default_val) == (2, 16, 2)
        assert bound.config_spec.cache_fingerprint_hash() == identity
        assert bound.config_spec.projected_cache_fingerprint_hash() == projected
        changed = cpu_bind(kernel, (torch.empty(37),))
        assert changed.config_spec.structural_fingerprint() == layout
        changed_identity = changed.config_spec.cache_fingerprint_hash()
        changed_projected = changed.config_spec.projected_cache_fingerprint_hash()
        assert changed_identity != identity
        assert changed_projected != projected
        fragment.default_val = 8
        new_default = cpu_bind(kernel, (torch.empty(37),))
        assert new_default.config_spec.structural_fingerprint() == layout
        assert new_default.config_spec.cache_fingerprint_hash() != changed_identity
        assert (
            new_default.config_spec.projected_cache_fingerprint_hash()
            != changed_projected
        )


@pytest.mark.parametrize("container", ["direct", "nested", "dataclass"])
@onlyBackends(["triton", "cute"])
def test_fragment_kernel_arguments_cannot_define_tunable_domains(container):
    with cpu_render_target():
        for choices in ((2, 16), (4, 8)):
            value = EnumFragment(choices)
            if container == "nested":
                value = [{"choices": value}]
            elif container == "dataclass":
                value = _FragmentBox(value)
            with (
                pytest.warns(FutureWarning, match="Symbolic config fragment domains"),
                pytest.raises(helion.exc.TunableTypeNotSupported, match="SymInt"),
            ):
                argument_fragment.bind((torch.empty(37), value, container))


@pytest.mark.parametrize("form", ["positional", "keyword", "nested"])
@onlyBackends(["triton", "cute"])
def test_dynamic_fragment_arguments_warn_without_snapshotting(form):
    with (
        cpu_render_target(),
        pytest.warns(FutureWarning, match="Symbolic config fragment domains"),
    ):
        bound = cpu_bind(dynamic_fragment, (torch.empty(64), form))
        captured = bound.config_spec.user_defined_tunables["factor"]
        symbolic = captured.choices[1] if form == "nested" else captured.high
        assert isinstance(symbolic, torch.SymInt)
        assert symbolic._sympy_().free_symbols
        assert not bound.env.tunable_fragment_snapshots
        ast.parse(bound.to_code(bound.config_spec.default_config()))


@pytest.mark.parametrize("form", ["enum", "integer"])
@onlyBackends(["triton", "cute"])
def test_symbolic_fragment_fixed_config_compatibility(form):
    with (
        cpu_render_target(),
        pytest.warns(
            FutureWarning, match="Symbolic config fragment domains"
        ) as warnings,
    ):
        bound = cpu_bind(symbolic_compatibility_fragment, (torch.empty(37), 64, form))
        fragment = bound.config_spec.user_defined_tunables["factor"]
        symbolic = fragment.choices[1] if form == "enum" else fragment.high
        assert isinstance(symbolic, torch.SymInt)
        assert symbolic._sympy_().free_symbols
        assert not bound.env.tunable_fragment_snapshots
        config = bound.config_spec.default_config()
        config.config["factor"] = 3
        selected = host_assignment_values(bound, config, "factor", "again")
        # Each emitted declaration uses the selected value, independently of
        # whether code generation combines their assignments.
        assert selected
        assert all(
            isinstance(node, ast.Constant) and node.value == 3 for node in selected
        )
    assert len(warnings) == 1


@onlyBackends(["triton", "cute"])
def test_symbolic_fragment_warning_uses_kernel_source_per_binding():
    with cpu_render_target(), warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("default")
        for size in (37, 41):
            cpu_bind(
                symbolic_compatibility_fragment, (torch.empty(size), 64, "integer")
            )
    domain_warnings = [
        item
        for item in recorded
        if str(item.message).startswith("Symbolic config fragment domains")
    ]
    assert len(domain_warnings) == 2
    for item in domain_warnings:
        assert item.category is FutureWarning
        assert item.filename == symbolic_compatibility_fragment.fn.__code__.co_filename
        assert "IntegerFragment(1, x.size(0))" in linecache.getline(
            item.filename, item.lineno
        )


@onlyBackends(["triton", "cute"])
def test_specialized_fragment_bound_remains_supported():
    with cpu_render_target():
        bound = cpu_bind(dynamic_fragment, (torch.empty(64), "specialized"))
        ast.parse(bound.to_code(bound.config_spec.default_config()))


@onlyBackends(["triton", "cute"])
def test_different_fragment_definitions_still_conflict():
    with cpu_render_target(), pytest.raises(helion.exc.TunableNameConflict):
        cpu_bind(conflicting_fragments, (torch.empty(37),))


@onlyBackends(["triton", "cute"])
def test_inline_fragment_guidance_works_with_real_bind():
    with cpu_render_target():
        bound = inline_fragment.bind((torch.empty(37),))
        assert bound.config_spec.user_defined_tunables["factor"].choices == (2, 16)
        ast.parse(bound.to_code(bound.config_spec.default_config()))


@pytest.mark.parametrize("captured", [False, True])
@onlyBackends(["triton", "cute"])
def test_fragment_fields_remain_runtime_data(captured):
    # These are ordinary dataclass field reads, with no tunable declaration.
    if captured:

        @helion.kernel(autotune_effort="none")
        def kernel(x):
            scalar = CHOICES.choices[0]
            out = torch.empty_like(x)
            for tile in hl.tile(x.numel()):
                out[tile] = x[tile] + scalar
            return out
    else:

        @helion.kernel(autotune_effort="none")
        def kernel(x, fragment):
            scalar = fragment.choices[0]
            out = torch.empty_like(x)
            for tile in hl.tile(x.numel()):
                out[tile] = x[tile] + scalar
            return out

    x = torch.empty(17)
    fragment = EnumFragment((2, 16))
    args = (x,) if captured else (x, fragment)
    values = []

    def observe(device, grid, source, output, scalar, **kwargs):
        assert source is x
        values.append(scalar)

    with cpu_render_target(), patch.dict(kernel.fn.__globals__, CHOICES=fragment):
        bound = cpu_bind(kernel, args)
        assert not bound.config_spec.user_defined_tunables
        config = bound.config_spec.default_config()
        source = ast.parse(bound.to_code(config))
        host = next(
            node
            for node in source.body
            if isinstance(node, ast.FunctionDef) and node.name == kernel.fn.__name__
        )
        launch = next(
            node
            for node in ast.walk(host)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_launcher"
        )
        device = next(
            node
            for node in source.body
            if isinstance(node, ast.FunctionDef) and node.name == launch.args[0].id
        )
        assert len(device.args.args) == 3
        scalar_name = device.args.args[2].arg
        assert any(
            isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id == scalar_name
            for node in ast.walk(device)
        )
        run = load_host_function(bound, config, observe)
        run(*args)
        fragment.choices = (5, 16)
        run(*args)
        if not captured:
            run(x, EnumFragment((7, 16)))
        assert values == ([2, 5] if captured else [2, 5, 7])
