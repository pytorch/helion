from __future__ import annotations

import ast
import gc
from unittest.mock import patch
import weakref

import pytest
import torch

import helion
from helion._testing import cpu_render_target
from helion._testing.tunable_fragments import host_assignment_values
from helion.autotuner import EnumFragment
import helion.language as hl

pytestmark = pytest.mark.cpu_only

SCALAR = 1e16
OTHER = 2.0


@helion.kernel(autotune_effort="none")
def guarded_global_arithmetic(x, operation: hl.constexpr):
    if operation == "division":
        if SCALAR != 0.0:
            scale = 1.0 / SCALAR
        else:
            scale = 1.0
    elif operation == "floor":
        if SCALAR > 0:
            scale = 64 // SCALAR
        else:
            scale = 1
    else:
        if SCALAR < 1:
            scale = SCALAR**400
        else:
            scale = 1.0
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        out[tile] = x[tile] * scale
    return out


@pytest.mark.parametrize(
    "operation,scalar",
    [
        ("division", 0.0),
        ("division", 2.0),
        ("floor", 0),
        ("floor", 2),
        ("power", 10.0),
        ("power", 0.5),
    ],
)
def test_guarded_global_arithmetic_keeps_symbolic_tracing(operation, scalar):
    kernel = helion.kernel(guarded_global_arithmetic.fn, autotune_effort="none")
    with patch.dict(kernel.fn.__globals__, SCALAR=scalar), cpu_render_target():
        bound = kernel.bind((torch.empty(17), operation))
        assert not bound.config_spec.user_defined_tunables
        ast.parse(bound.to_code(bound.config_spec.default_config()))


@helion.kernel(autotune_effort="none")
def global_arithmetic(x, operation: hl.constexpr):
    if operation == "cancellation":
        value = (SCALAR + 1) - SCALAR + OTHER
    elif operation == "add_zero":
        value = SCALAR + 0.0
    elif operation == "constant":
        value = (SCALAR + 1) - SCALAR
    elif operation == "constant_tail":
        value = SCALAR - SCALAR + OTHER + 1
    else:
        value = (-SCALAR) / 2
    factor = hl.register_tunable("factor", EnumFragment((value,)))
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        out[tile] = x[tile] * factor
    return out


@pytest.mark.parametrize(
    "operation,scalar,expected,changed_expected",
    [
        ("cancellation", 1e16, 2.0, 3.0),
        ("add_zero", -0.0, 0.0, 1.0),
        ("negate_divide", 0.0, -0.0, -0.5),
        ("constant", 1e16, 0.0, 1.0),
        ("constant_tail", 1e16, 3.0, 3.0),
    ],
)
def test_global_float_operations_preserve_python_order(
    operation, scalar, expected, changed_expected
):
    kernel = helion.kernel(global_arithmetic.fn, autotune_effort="none")
    with patch.dict(kernel.fn.__globals__, SCALAR=scalar), cpu_render_target():
        args = (torch.empty(17), operation)
        bound = kernel.bind(args)
        value = bound.config_spec.user_defined_tunables["factor"].default()
        assert type(value) is float and value.hex() == expected.hex()
        kernel.fn.__globals__["SCALAR"] = 1.0
        assert kernel.bind(args) is bound
        fresh = helion.kernel(kernel.fn, autotune_effort="none").bind(args)
        for current, wanted in ((bound, expected), (fresh, changed_expected)):
            actual = current.config_spec.user_defined_tunables["factor"].default()
            assert type(actual) is float and actual.hex() == wanted.hex()
            assert _emitted_factor(current).hex() == wanted.hex()


def _emitted_factor(bound):
    values = host_assignment_values(bound, bound.config_spec.default_config(), "factor")
    assert len(values) == 1
    emitted = ast.literal_eval(values[0])
    assert type(emitted) is float
    return emitted


@helion.kernel(autotune_effort="none")
def global_zero_pair(x):
    # The second operation may have the same SymPy expression as the first.
    value = +SCALAR
    adjusted = value + 0.0
    factor = hl.register_tunable("factor", EnumFragment((value, adjusted)))
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        out[tile] = x[tile] * factor
    return out


def test_equal_symbol_expressions_keep_separate_occurrences_and_tensor_lifetime():
    kernel = helion.kernel(global_zero_pair.fn, autotune_effort="none")
    with patch.dict(kernel.fn.__globals__, SCALAR=-0.0), cpu_render_target():
        tensor = torch.empty(17)
        reference = weakref.ref(tensor)
        bound = kernel.bind((tensor,))
        choices = bound.config_spec.user_defined_tunables["factor"].choices
        assert tuple(value.hex() for value in choices) == ("-0x0.0p+0", "0x0.0p+0")
        del tensor
        gc.collect()
        assert reference() is None
        assert _emitted_factor(bound).hex() == "-0x0.0p+0"


@helion.kernel(autotune_effort="none")
def global_comparison(x):
    factor = hl.register_tunable(
        "factor", EnumFragment((((SCALAR + 1) - SCALAR) == 0,))
    )
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        out[tile] = x[tile] * factor
    return out


def test_simplified_comparison_keeps_python_boolean():
    kernel = helion.kernel(global_comparison.fn, autotune_effort="none")
    with patch.dict(kernel.fn.__globals__, SCALAR=1e16), cpu_render_target():
        bound = kernel.bind((torch.empty(17),))
        assert bound.config_spec.user_defined_tunables["factor"].default() is True
        source = ast.parse(bound.to_code(bound.config_spec.default_config()))
        assert any(
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Constant)
            and node.value.value is True
            and any(
                isinstance(target, ast.Name) and target.id == "factor"
                for target in node.targets
            )
            for node in ast.walk(source)
        )


def _opaque_arithmetic(value):
    return (value + 1) - value


@helion.kernel(autotune_effort="none", static_shapes=False)
def unproved_arithmetic(x, runtime, mode: hl.constexpr):
    if mode == "helper":
        value = _opaque_arithmetic(SCALAR)
    elif mode == "runtime":
        value = (runtime + 1) - runtime
    elif mode == "branch":
        if runtime > 0:
            value = SCALAR
        else:
            value = SCALAR + 0.0
    elif mode == "literal_branch":
        if runtime > 0:
            value = 1.0
        else:
            value = (SCALAR + 1) - SCALAR
    else:
        value = (x.size(0) + 0.0 + 1) - x.size(0)
    factor = hl.register_tunable("factor", EnumFragment((0.0, value)))
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        out[tile] = x[tile] * factor
    return out


@pytest.mark.parametrize(
    "mode", ["helper", "runtime", "shape", "branch", "literal_branch"]
)
def test_unproved_operations_do_not_acquire_literal_provenance(mode):
    kernel = helion.kernel(
        unproved_arithmetic.fn, autotune_effort="none", static_shapes=False
    )
    scalar = -0.0 if mode == "branch" else 1e16
    with (
        patch.dict(kernel.fn.__globals__, SCALAR=scalar),
        cpu_render_target(),
        pytest.warns(FutureWarning, match="Symbolic config fragment domains"),
    ):
        bound = kernel.bind((torch.empty(17), 1e16, mode))
        choices = bound.config_spec.user_defined_tunables["factor"].choices
        assert isinstance(choices[1], torch.SymFloat)
        assert bound.env.resolve_fragment_global(choices[1]) is None
        config = bound.config_spec.default_config()
        config.config["factor"] = 2.0
        ast.parse(bound.to_code(config))
