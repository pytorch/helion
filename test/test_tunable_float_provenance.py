from __future__ import annotations

import ast
from unittest.mock import patch

import pytest
import torch

import helion
from helion._testing import cpu_render_target
from helion._testing import onlyBackends
from helion._testing.cute_codegen import cpu_bind
from helion._testing.tunable_fragments import load_host_function
from helion.autotuner import EnumFragment
import helion.language as hl

pytestmark = pytest.mark.cpu_only


@helion.kernel(autotune_effort="none")
def derived_float_tunable(x, runtime, mode: hl.constexpr):
    factor = hl.register_tunable("factor", EnumFragment((2.0, 1e16)))
    if mode == "assigned_product":
        value = factor * 2.0
    elif mode == "assigned_division":
        value = factor / 2.0
    elif mode == "assigned_order":
        value = (factor + 1.0) - factor
    elif mode == "mixed_assigned":
        value = factor / runtime
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        if mode == "inline_product":
            out[tile] = x[tile] * (factor * 2.0)
        elif mode == "inline_unary":
            out[tile] = x[tile] * -(+factor)
        elif mode == "mixed_inline":
            out[tile] = x[tile] * (factor * runtime)
        elif mode == "mixed_order":
            out[tile] = x[tile] * ((runtime + 1.0) - runtime + factor)
        else:
            out[tile] = x[tile] * value
    return out


@pytest.mark.parametrize(
    "mode",
    [
        "assigned_product",
        "assigned_division",
        "assigned_order",
        "inline_product",
        "inline_unary",
        "mixed_assigned",
        "mixed_inline",
        "mixed_order",
    ],
)
@onlyBackends(["triton", "cute"])
def test_derived_tunable_keeps_constexpr_only_without_runtime_inputs(mode):
    args = (torch.empty(17), 2.0, mode)
    with cpu_render_target():
        bound = cpu_bind(derived_float_tunable, args)
        for factor in (2.0, 1e16):
            config = bound.config_spec.default_config()
            config.config["factor"] = factor
            module = ast.parse(bound.to_code(config))
            functions = [n for n in module.body if isinstance(n, ast.FunctionDef)]
            host = next(n for n in functions if n.name == bound.kernel.fn.__name__)
            launch = next(
                n
                for n in ast.walk(host)
                if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name)
                and n.func.id == "_launcher"
            )
            device = next(n for n in functions if n.name == launch.args[0].id)
            assert len(device.args.args) == 3
            scalar = device.args.args[2]
            constexpr = (
                scalar.annotation is not None
                and ast.unparse(scalar.annotation) == bound.env.backend.constexpr_type
            )
            assert constexpr == (not mode.startswith("mixed")), ast.unparse(device)
            records = []

            def observe(_device, _grid, _x, _out, value, records=records, **kwargs):
                records.append(value)

            with patch.object(bound, "to_code", wraps=bound.to_code) as render:
                run = load_host_function(bound, config, observe)
            render.assert_called_once_with(config)
            for runtime in (2.0, 1e16):
                run(args[0], runtime, mode)
                if mode in ("assigned_product", "inline_product"):
                    expected = factor * 2.0
                elif mode == "assigned_division":
                    expected = factor / 2.0
                elif mode == "assigned_order":
                    expected = (factor + 1.0) - factor
                elif mode == "inline_unary":
                    expected = -(+factor)
                elif mode == "mixed_assigned":
                    expected = factor / runtime
                elif mode == "mixed_inline":
                    expected = factor * runtime
                else:
                    expected = (runtime + 1.0) - runtime + factor
                assert type(records[-1]) is float
                assert records[-1].hex() == expected.hex()
            assert len(records) == 2


@helion.kernel(autotune_effort="none")
def scalar_observation_tensor_inputs(x, scalar_tensor, value: int):
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        out[tile] = x[tile] + scalar_tensor[()] + value
    return out


def test_scalar_observation_preserves_real_scalar_and_singleton_tensors():
    x = torch.ones(1)
    scalar = torch.tensor(7.0)
    with cpu_render_target():
        bound = cpu_bind(scalar_observation_tensor_inputs, (x, scalar, 2))
        launches = []
        host = load_host_function(
            bound,
            bound.config_spec.default_config(),
            lambda kernel, grid, *args, **kwargs: launches.append(args),
            scalar_values=True,
        )
        for value in (2, 5):
            host(x, scalar, value)
            args = launches[-1]
            assert [arg for arg in args if type(arg) is int] == [value]
            tensors = [arg for arg in args if isinstance(arg, torch.Tensor)]
            for original in (x, scalar):
                assert any(
                    t.untyped_storage() is original.untyped_storage() for t in tensors
                )
