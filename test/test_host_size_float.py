"""Host shape arithmetic must retain its runtime size dependencies."""

from __future__ import annotations

import pytest
import torch

import helion
from helion._compiler.type_info import SymFloatType
from helion._compiler.type_info import SymIntType
from helion._testing import cpu_render_target
from helion._testing.tunable_fragments import load_host_function
import helion.language as hl

pytestmark = pytest.mark.cpu_only


def _fractional_size(x, form: hl.constexpr):
    if form == "direct":
        n = int(x.size(0) / 2)
    else:
        fraction = x.size(0) / 2
        n = int(fraction)
    out = torch.empty((n,), dtype=x.dtype, device=x.device)
    for tile in hl.tile(n):
        out[tile] = x[tile]
    return out, n


@pytest.mark.parametrize("form", ["direct", "local"])
def test_size_float_conversion_reuses_symbolic_binding(form):
    initialized = torch.cuda.is_initialized()
    kernel = helion.kernel(_fractional_size, static_shapes=False)
    bindings = []
    with cpu_render_target():
        for size in (9, 13, 7, 9):
            x = torch.empty(size)
            bound = kernel.bind((x, form))
            bindings.append(bound)
            n_type = bound.host_function.local_types["n"]
            assert isinstance(n_type, SymIntType)
            assert n_type.to_sympy().free_symbols
            entry = load_host_function(
                bound, bound.config_spec.default_config(), lambda *args, **kwargs: None
            )
            out, n = entry(x, form)
            expected = int(size / 2)
            assert n == expected
            assert out.shape == (expected,)
    # CuTe can retain distinct shape-class bindings. Both backends must reuse
    # the original binding when its original shape returns, without a baked n.
    assert bindings[-1] is bindings[0]
    if bindings[0].env.backend.name == "triton":
        assert all(bound is bindings[0] for bound in bindings)
    assert torch.cuda.is_initialized() == initialized


def _ordered_size_float(x, y):
    first = x.size(0) / 1.0 + 1.0
    second = y.size(0) / 1.0 - x.size(0) / 1.0
    value = first + second
    out = torch.empty_like(y)
    for tile in hl.tile(y.numel()):
        out[tile] = value
    return out, value


def test_size_float_arithmetic_retains_ordered_ieee_values():
    initialized = torch.cuda.is_initialized()
    # The empty input needs no huge allocation. Above 2**53, algebraically
    # cancelling its size loses Python's intermediate rounding.
    x, y = torch.empty((2**54, 0)), torch.empty(2)
    expected = (x.size(0) / 1.0 + 1.0) + (y.size(0) / 1.0 - x.size(0) / 1.0)
    assert expected == 2.0
    with cpu_render_target():
        bound = helion.kernel(_ordered_size_float, static_shapes=False).bind((x, y))
        value_type = bound.host_function.local_types["value"]
        assert isinstance(value_type, SymFloatType)
        assert value_type.to_sympy().is_Symbol
        entry = load_host_function(
            bound, bound.config_spec.default_config(), lambda *args, **kwargs: None
        )
        out, value = entry(x, y)
    assert value == expected
    assert out.shape == y.shape
    assert torch.cuda.is_initialized() == initialized
