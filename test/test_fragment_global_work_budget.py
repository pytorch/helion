from __future__ import annotations

import ast
from unittest.mock import patch

import pytest
import torch

import helion
from helion._compiler import type_info
from helion._testing import cpu_render_target
from helion.autotuner import EnumFragment
import helion.language as hl

pytestmark = pytest.mark.cpu_only

ENABLED = False
BASE = 2


@helion.kernel(autotune_effort="none")
def guarded_growth(x, operation: hl.constexpr):
    if ENABLED:
        if operation == "power":
            value = BASE**100000
        elif operation == "multiply":
            value = (BASE**2048) * (BASE**2048)
        else:
            value = BASE << 100000
    else:
        value = 1
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        out[tile] = x[tile] * value
    return out


@pytest.mark.parametrize("operation", ["power", "shift", "multiply"])
def test_untaken_global_growth_does_not_evaluate_large_python_integer(operation):
    original = type_info._eval_binary_impl
    symbolic_calls = []

    def bounded_evaluator(op, left, right):
        if (
            isinstance(op, ast.Mult)
            and type(left) is int
            and type(right) is int
            and left.bit_length() + right.bit_length() > 4096
        ):
            raise MemoryError("unexpected concrete compile-time multiplication")
        if isinstance(op, (ast.Pow, ast.LShift)) and right == 100000:
            # Protect the test process before the optional recorder performs
            # a concrete large allocation. The authored branch is not taken.
            if type(left) is int and type(right) is int:
                raise MemoryError("unexpected concrete compile-time growth")
            symbolic_calls.append(type(op))
        elif isinstance(op, ast.Mult) and isinstance(left, torch.SymInt):
            symbolic_calls.append(type(op))
        return original(op, left, right)

    kernel = helion.kernel(guarded_growth.fn, autotune_effort="none")
    with (
        patch.object(type_info, "_eval_binary_impl", bounded_evaluator),
        cpu_render_target(),
    ):
        bound = kernel.bind((torch.empty(17), operation))
        ast.parse(bound.to_code(bound.config_spec.default_config()))
    assert symbolic_calls


@helion.kernel(autotune_effort="none")
def small_global_domain(x, operation: hl.constexpr):
    if operation == "power":
        high = BASE**8
    else:
        high = BASE << 8
    value = hl.register_tunable("factor", EnumFragment((high,)))
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        out[tile] = x[tile] * value
    return out


@pytest.mark.parametrize("operation,expected", [("power", 256), ("shift", 512)])
def test_bounded_global_growth_keeps_exact_domain(operation, expected):
    kernel = helion.kernel(small_global_domain.fn, autotune_effort="none")
    with cpu_render_target():
        bound = kernel.bind((torch.empty(17), operation))
    assert bound.config_spec.user_defined_tunables["factor"].choices == (expected,)
