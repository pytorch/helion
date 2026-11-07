"""CPU-only source and host-capture tests for heterogeneous ordered phases."""

from __future__ import annotations

import dataclasses
from types import ModuleType

import pytest
import torch

import helion
from helion._testing.constant_membership import ordered_membership_phases
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import cpu_target
from helion._testing.cute_codegen import load_generated_module
from helion._testing.cute_codegen import stage_sources
from helion._testing.cute_ordered_metadata import computed_metadata_phases
from helion.autotuner.config_fragment import PowerOfTwoFragment
import helion.language as hl

pytestmark = pytest.mark.cpu_only

_HOST_EFFECTS = []


class _MetadataProvider(ModuleType):
    def __add__(self, value):
        _HOST_EFFECTS.append(value)
        return 1


_metadata_provider = _MetadataProvider("_ordered_metadata")


def _opaque_predicate(n):
    _HOST_EFFECTS.append(n)
    return True


@helion.kernel(backend="cute", static_shapes=True)
def _opaque_assertion_phases(x: torch.Tensor) -> torch.Tensor:
    n = x.numel()
    assert _opaque_predicate(n)
    temporary = torch.empty_like(x)
    output = torch.empty_like(x)
    for i in hl.tile(n):
        temporary[i] = x[i] + 1
    hl.barrier()
    for i in hl.tile(n):
        output[i] = temporary[i] * 2
    return output


@helion.kernel(backend="cute", static_shapes=True)
def _opaque_condition_phases(x: torch.Tensor) -> torch.Tensor:
    n = x.numel()
    if _opaque_predicate(n):
        bias = 1
    else:
        bias = 2
    temporary = torch.empty_like(x)
    output = torch.empty_like(x)
    for i in hl.tile(n):
        temporary[i] = x[i] + bias
    hl.barrier()
    for i in hl.tile(n):
        output[i] = temporary[i] * 2
    return output


@helion.kernel(backend="cute", static_shapes=True)
def _overloaded_condition_phases(x: torch.Tensor) -> torch.Tensor:
    n = x.numel()
    if (_metadata_provider + 0) == 1:
        bias = 1
    else:
        bias = 2
    temporary = torch.empty_like(x)
    output = torch.empty_like(x)
    for i in hl.tile(n):
        temporary[i] = x[i] + bias
    hl.barrier()
    for i in hl.tile(n):
        output[i] = temporary[i] * 2
    return output


def _choose_abs():
    _HOST_EFFECTS.append("callee")
    return abs


@helion.kernel(backend="cute", static_shapes=True)
def _overloaded_metadata_argument_phases(x: torch.Tensor) -> torch.Tensor:
    n = x.numel()
    if x.size((_metadata_provider + 0) - 1) > 0:
        bias = 1
    else:
        bias = 2
    temporary = torch.empty_like(x)
    output = torch.empty_like(x)
    for i in hl.tile(n):
        temporary[i] = x[i] + bias
    hl.barrier()
    for i in hl.tile(n):
        output[i] = temporary[i] * 2
    return output


@helion.kernel(backend="cute", static_shapes=True)
def _effectful_callee_phases(x: torch.Tensor) -> torch.Tensor:
    n = x.numel()
    if _choose_abs()(1) == 1:
        bias = 1
    else:
        bias = 2
    temporary = torch.empty_like(x)
    output = torch.empty_like(x)
    for i in hl.tile(n):
        temporary[i] = x[i] + bias
    hl.barrier()
    for i in hl.tile(n):
        output[i] = temporary[i] * 2
    return output


class _EffectfulFragment(PowerOfTwoFragment):
    def __bool__(self):
        _HOST_EFFECTS.append("fragment truth test")
        return False


@helion.kernel(backend="cute", static_shapes=True)
def _fragment_truth_condition_phases(x: torch.Tensor) -> torch.Tensor:
    n = x.numel()
    fragment = _EffectfulFragment(1, 4, 2)
    if not fragment:
        bias = 1
    else:
        bias = 2
    temporary = torch.empty_like(x)
    output = torch.empty_like(x)
    for i in hl.tile(n):
        temporary[i] = x[i] + bias
    hl.barrier()
    for i in hl.tile(n):
        output[i] = temporary[i] * 2
    return output


@dataclasses.dataclass
class _ClassGuard:
    value: int = 1

    def __bool__(self):
        _HOST_EFFECTS.append("class truth test")
        return True


_class_guard = _ClassGuard()


@helion.kernel(backend="cute", static_shapes=True)
def _class_truth_condition_phases(x: torch.Tensor) -> torch.Tensor:
    n = x.numel()
    if not _class_guard:
        bias = 1
    else:
        bias = 2
    temporary = torch.empty_like(x)
    output = torch.empty_like(x)
    for i in hl.tile(n):
        temporary[i] = x[i] + bias
    hl.barrier()
    for i in hl.tile(n):
        output[i] = temporary[i] * 2
    return output


class _EffectfulFragmentConstructor(PowerOfTwoFragment):
    def __new__(cls, *args):
        _HOST_EFFECTS.append("fragment constructor")
        return True


@helion.kernel(backend="cute", static_shapes=True)
def _fragment_constructor_condition_phases(x: torch.Tensor) -> torch.Tensor:
    n = x.numel()
    if _EffectfulFragmentConstructor(1, 4, 2):
        bias = 1
    else:
        bias = 2
    temporary = torch.empty_like(x)
    output = torch.empty_like(x)
    for i in hl.tile(n):
        temporary[i] = x[i] + bias
    hl.barrier()
    for i in hl.tile(n):
        output[i] = temporary[i] * 2
    return output


@helion.kernel(backend="cute", static_shapes=True)
def _registered_metadata_phases(x: torch.Tensor) -> torch.Tensor:
    parts = hl.register_tunable("parts", PowerOfTwoFragment(1, 4, 2))
    n = x.numel() + parts - parts
    temporary = torch.empty_like(x)
    output = torch.empty_like(x)
    for i in hl.tile(n):
        temporary[i] = x[i] + 1
    hl.barrier()
    for i in hl.tile(n):
        output[i] = temporary[i] * 2
    return output


@helion.kernel(backend="cute", static_shapes=True)
def _bound_registered_metadata_phases(x: torch.Tensor) -> torch.Tensor:
    fragment = PowerOfTwoFragment(1, 4, 2)
    parts = hl.register_tunable("parts", fragment)
    n = x.numel() + parts - parts
    temporary = torch.empty_like(x)
    output = torch.empty_like(x)
    for i in hl.tile(n):
        temporary[i] = x[i] + 1
    hl.barrier()
    for i in hl.tile(n):
        output[i] = temporary[i] * 2
    return output


class _EffectfulFragmentInit(PowerOfTwoFragment):
    def __init__(self, low, high, initial):
        _HOST_EFFECTS.append("fragment init")
        super().__init__(low, high, initial)


@helion.kernel(backend="cute", static_shapes=True)
def _effectful_fragment_binding_phases(x: torch.Tensor) -> torch.Tensor:
    fragment = _EffectfulFragmentInit(1, 4, 2)
    parts = hl.register_tunable("parts", fragment)
    temporary = torch.empty_like(x)
    output = torch.empty_like(x)
    for i in hl.tile(x.numel()):
        temporary[i] = x[i] + parts
    hl.barrier()
    for i in hl.tile(x.numel()):
        output[i] = temporary[i] * 2
    return output


def test_effectful_fragment_binding_is_not_replayed():
    with cpu_target():
        bound = cpu_bind(_effectful_fragment_binding_phases, (torch.ones(8),))
        calls = len(_HOST_EFFECTS)
        config = dict(bound.config_spec.default_config()) | {"pid_type": "flat"}
        with pytest.raises(helion.exc.BackendUnsupported, match="shape-only"):
            bound.to_code(helion.Config.from_dict(config))
        assert len(_HOST_EFFECTS) == calls


@helion.kernel(backend="cute", static_shapes=True)
def _builtin_guard_phases(
    x: torch.Tensor, mode: hl.constexpr, shift: hl.constexpr
) -> torch.Tensor:
    temporary = torch.empty_like(x)
    output = torch.empty_like(x)
    assert temporary.shape == x.shape
    if temporary.numel() > 0 and x.dtype == torch.float32 and mode == "fast":
        factor = 2
    else:
        factor = 1
    if shift:
        bias = shift
    else:
        bias = 0.0
    for i in hl.tile(x.numel()):
        temporary[i] = x[i] * factor
    hl.barrier()
    for i in hl.tile(x.numel()):
        output[i] = temporary[i] + bias
    return output


@pytest.mark.parametrize("mode", ["fast", "slow"])
@pytest.mark.parametrize("shift", [None, 2.0])
def test_builtin_comparisons_and_prior_allocation_guards(mode, shift):
    with cpu_target():
        bound = cpu_bind(_builtin_guard_phases, (torch.ones(8), mode, shift))
        config = dict(bound.config_spec.default_config()) | {"pid_type": "flat"}
        source = bound.to_code(helion.Config.from_dict(config))
        assert source.count("PyCodeCache.load(") == 2


@pytest.mark.parametrize(
    "kernel", [_registered_metadata_phases, _bound_registered_metadata_phases]
)
def test_registered_fragment_metadata_remains_supported(kernel):
    with cpu_target():
        bound = cpu_bind(kernel, (torch.ones(8),))
        config = dict(bound.config_spec.default_config()) | {"pid_type": "flat"}
        code = bound.to_code(helion.Config.from_dict(config))
        assert code.count("PyCodeCache.load(") == 2


@pytest.mark.parametrize(
    "kernel",
    [
        _opaque_assertion_phases,
        _opaque_condition_phases,
        _overloaded_condition_phases,
        _overloaded_metadata_argument_phases,
        _effectful_callee_phases,
        _fragment_truth_condition_phases,
        _class_truth_condition_phases,
        _fragment_constructor_condition_phases,
    ],
)
def test_opaque_host_predicates_are_not_replayed(kernel):
    with cpu_target():
        bound = cpu_bind(kernel, (torch.ones(8),))
        calls = len(_HOST_EFFECTS)
        config = dict(bound.config_spec.default_config()) | {"pid_type": "flat"}
        with pytest.raises(helion.exc.BackendUnsupported, match="pure"):
            bound.to_code(helion.Config.from_dict(config))
        assert len(_HOST_EFFECTS) == calls


@pytest.mark.parametrize("shift", [None, 2.0])
def test_ordered_metadata_and_exact_storage_capture(shift, tmp_path):
    x = torch.arange(2 * 128 + 3, dtype=torch.bfloat16)[3:].as_strided(
        (2, 7, 8), (128, 12, 1)
    )
    y = torch.ones(2, 8, 9, dtype=torch.bfloat16)
    with cpu_target():
        bound = cpu_bind(computed_metadata_phases, (x, y, shift))
        assert "flat" in bound.config_spec.allowed_pid_types
        config = dict(bound.config_spec.default_config()) | {
            "pid_type": "flat",
            "cute_fragment_half_mma": True,
        }
        code = bound.to_code(helion.Config.from_dict(config))
    bodies = stage_sources(code)
    assert len(bodies) == 2
    assert "_helion_fragment_mma.mma_bf16(" in bodies[0]
    assert "_helion_fragment_mma.mma_bf16(" not in bodies[1]
    # Execute actual generated hosts with a fake launcher, preserving the
    # exact as_strided object, its offset and alias owner across both stages.
    path = tmp_path / "metadata_generated.py"
    path.write_text(code)
    module = load_generated_module(path)
    calls = []

    def launch(kernel, grid, *args, **kwargs):
        calls.append((kernel, grid, args, kwargs))

    original = x.clone()
    output = module.computed_metadata_phases(x, y, shift, _launcher=launch)
    assert len(calls) == 2
    assert output.shape == (2, 7, 9)
    torch.testing.assert_close(x, original, rtol=0, atol=0)
    aliases = [
        arg
        for _, _, args, _ in calls
        for arg in args
        if isinstance(arg, torch.Tensor) and torch._C._is_alias_of(arg, x)
    ]
    storage_views = [arg for arg in aliases if arg.ndim == 1]
    assert storage_views and all(arg.storage_offset() == 3 for arg in storage_views)
    assert all(arg is storage_views[0] for arg in storage_views)


_module_options = ModuleType("_ordered_options")
_module_options.__dict__["validate"] = False


@helion.kernel(backend="cute", static_shapes=True)
def _module_flag_phases(x: torch.Tensor, alias: hl.constexpr) -> torch.Tensor:
    if alias:
        enabled = _module_options.validate
        if enabled:
            assert x.numel() > 0
    else:
        if _module_options.validate:
            assert x.numel() > 0
    temporary = torch.empty_like(x)
    output = torch.empty_like(x)
    for i in hl.tile(x.numel()):
        temporary[i] = x[i] + 1
    hl.barrier()
    for i in hl.tile(x.numel()):
        output[i] = temporary[i] * 2
    return output


@pytest.mark.parametrize("alias", [False, True])
def test_mutable_module_flags_are_not_folded(alias, monkeypatch):
    # A literal type is insufficient: dispatch does not guard this flag, so
    # cached ordered launches must not freeze an assertion's enabled state.
    monkeypatch.setattr(_module_options, "validate", False)
    with cpu_target():
        bound = cpu_bind(_module_flag_phases, (torch.ones(8), alias))
        config = dict(bound.config_spec.default_config()) | {"pid_type": "flat"}
        with pytest.raises(helion.exc.BackendUnsupported, match="pure|shape-only"):
            bound.to_code(helion.Config.from_dict(config))


@pytest.mark.parametrize("mode,bias", [("fast", 3), ("missing", 7)])
def test_ordered_membership_selects_the_actual_branch(mode, bias):
    with cpu_target():
        bound = cpu_bind(
            helion.kernel(
                ordered_membership_phases, backend="cute", static_shapes=True
            ),
            (torch.ones(7), mode),
        )
        config = dict(bound.config_spec.default_config()) | {"pid_type": "flat"}
        source = bound.to_code(helion.Config.from_dict(config))
    assert f"bias = {bias}" in source
    assert f"bias = {7 if bias == 3 else 3}" not in source
    assert len(stage_sources(source)) == 2
