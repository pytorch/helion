from __future__ import annotations

import ast
import struct
from unittest.mock import patch

import pytest
import torch

import helion
from helion._compiler.type_info import _preserve_host_float_operation
from helion._testing import cpu_render_target
from helion._testing import onlyBackends
from helion._testing.cute_codegen import cpu_bind
from helion._testing.host_float_control_flow import aliased_host_float
from helion._testing.host_float_control_flow import rebound_host_float
from helion._testing.tunable_fragments import host_functions
from helion._testing.tunable_fragments import load_host_function
import helion.language as hl

pytestmark = pytest.mark.cpu_only
SCALAR = 1e16


@helion.kernel(autotune_effort="none")
def scalar_device_use(x, runtime, mode: hl.constexpr):
    if mode == "direct":
        value = (SCALAR + 1) - SCALAR
    elif mode == "branch":
        if runtime > 0:
            value = 1.0
        else:
            value = (SCALAR + 1) - SCALAR
    elif mode == "reverse":
        if runtime > 0:
            value = (SCALAR + 1) - SCALAR
        else:
            value = 1.0
    elif mode == "runtime":
        value = (runtime + 1) - runtime
    else:
        value = SCALAR * runtime
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        out[tile] = x[tile] * value
    return out


def _scalar_host(kernel, args, expected_name=None, scalar_count=1):
    bound = cpu_bind(kernel, args)
    config = bound.config_spec.default_config()
    source = bound.to_code(config)
    host, functions = host_functions(bound, config)
    launch = next(
        node
        for node in ast.walk(host)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_launcher"
    )
    assert isinstance(launch.args[0], ast.Name)
    device = next(node for node in functions if node.name == launch.args[0].id)
    assert len(device.args.args) == 2 + scalar_count, source
    for parameter in device.args.args[2:]:
        assert any(
            isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id == parameter.arg
            for node in ast.walk(device)
        ), source
    if expected_name is not None:
        assert isinstance(launch.args[4], ast.Name)
        assert launch.args[4].id == expected_name
    records = []

    def observe_launch(device, grid, source_tensor, output, *scalars, **kwargs):
        assert source_tensor is args[0]
        assert len(scalars) == scalar_count
        records.append(scalars)

    host_fn = load_host_function(bound, config, observe_launch)

    def run(*args):
        records.clear()
        host_fn(*args)
        assert len(records) == 1
        assert all(type(value) is float for value in records[0])
        return records[0][0] if scalar_count == 1 else records[0]

    return run, source


@pytest.mark.parametrize("mode", ["direct", "branch", "reverse", "runtime", "product"])
@onlyBackends(["triton", "cute"])
def test_host_float_value_reaches_device(mode):
    x = torch.empty(17)
    kernel = helion.kernel(scalar_device_use.fn, autotune_effort="none")
    with cpu_render_target(), patch.dict(kernel.fn.__globals__, SCALAR=1e16):
        run, _source = _scalar_host(kernel, (x, 1e16, mode), "value")
        for scalar, runtime in [(1e16, -1e16), (1.0, 2.0), (1e16, 1e16)]:
            with patch.dict(kernel.fn.__globals__, SCALAR=scalar):
                if mode == "direct":
                    expected = (scalar + 1) - scalar
                elif mode == "branch":
                    expected = 1.0 if runtime > 0 else (scalar + 1) - scalar
                elif mode == "reverse":
                    expected = (scalar + 1) - scalar if runtime > 0 else 1.0
                elif mode == "runtime":
                    expected = (runtime + 1) - runtime
                else:
                    expected = scalar * runtime
                assert run(x, runtime, mode).hex() == expected.hex()


@helion.kernel(autotune_effort="none")
def inline_float_use(x, runtime, mode: hl.constexpr):
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        if mode == "global":
            out[tile] = x[tile] * ((SCALAR + 1) - SCALAR)
        elif mode == "runtime":
            out[tile] = x[tile] * ((runtime + 1) - runtime)
        elif mode == "sign":
            out[tile] = x[tile] * -(+SCALAR)
        else:
            out[tile] = x[tile] * ((SCALAR + 1) * runtime - SCALAR)
    return out


@pytest.mark.parametrize("mode", ["global", "runtime", "sign", "order"])
@onlyBackends(["triton", "cute"])
def test_inline_host_float_expression_stays_dynamic(mode):
    x = torch.empty(17)
    kernel = helion.kernel(inline_float_use.fn, autotune_effort="none")
    with cpu_render_target(), patch.dict(kernel.fn.__globals__, SCALAR=1e16):
        run, _source = _scalar_host(kernel, (x, 1e16, mode))
        for scalar, runtime in [
            (1e16, 1.0),
            (1.0, 1e16),
            (-0.0, -0.0),
            (float("inf"), 1.0),
        ]:
            with patch.dict(kernel.fn.__globals__, SCALAR=scalar):
                if mode == "global":
                    expected = (scalar + 1) - scalar
                elif mode == "runtime":
                    expected = (runtime + 1) - runtime
                elif mode == "sign":
                    expected = -(+scalar)
                else:
                    expected = (scalar + 1) * runtime - scalar
                assert run(x, runtime, mode).hex() == expected.hex()


@helion.kernel(autotune_effort="none")
def guarded_division(x, runtime):
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        if x[tile].sum() > 0:
            out[tile] = x[tile] * (SCALAR / runtime)
        else:
            out[tile] = x[tile] * SCALAR
    return out


@onlyBackends(["triton", "cute"])
def test_inline_division_is_not_hoisted_out_of_device_branch():
    x = -torch.ones(17)
    with cpu_render_target():
        kernel = helion.kernel(guarded_division.fn, autotune_effort="none")
        run, source = _scalar_host(kernel, (x, 0.0), scalar_count=2)
        assert sorted(run(x, 0.0)) == [0.0, SCALAR]
        host = next(
            node
            for node in ast.parse(source).body
            if isinstance(node, ast.FunctionDef) and node.name == kernel.fn.__name__
        )
        assert not any(isinstance(node, ast.Div) for node in ast.walk(host))


@onlyBackends(["triton", "cute"])
def test_inline_integer_conversion_requires_a_finite_literal():
    with cpu_render_target():
        kernel = helion.kernel(scalar_device_use.fn, autotune_effort="none")
        bound = cpu_bind(kernel, (torch.empty(17), 1e16, "direct"))
        with bound.env, bound.host_function:
            runtime = bound.host_function.local_types["runtime"].proxy()
            symbolic_int = bound.env.create_unbacked_symint()
            marker = object()
            origins = dict(bound.host_function.expr_to_origin)
            for integer in (10**400, symbolic_int):
                assert (
                    _preserve_host_float_operation(
                        marker, ast.Add(), (runtime, integer)
                    )
                    is marker
                )
            assert bound.host_function.expr_to_origin == origins


@helion.kernel(autotune_effort="none")
def mixed_device_float(x):
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        out[tile] = x[tile] * (SCALAR + tile.begin) + ((x[tile] + 1.0) - x[tile])
    return out


@onlyBackends(["triton", "cute"])
def test_device_tensor_and_block_values_keep_device_arithmetic():
    x = torch.empty(17)
    with cpu_render_target():
        kernel = helion.kernel(mixed_device_float.fn, autotune_effort="none")
        run, source = _scalar_host(kernel, (x,))
        assert run(x) == SCALAR
        assert "host_float" not in source


@helion.kernel(autotune_effort="none")
def inline_nonfinite_literal(x, runtime, mode: hl.constexpr):
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        if mode == "positive":
            out[tile] = x[tile] * (runtime + 1e309)
        elif mode == "negative":
            out[tile] = x[tile] * (runtime + -1e309)
        else:
            out[tile] = x[tile] * (runtime + (1e309 - 1e309))
    return out


@pytest.mark.parametrize("mode", ["positive", "negative", "nan"])
@onlyBackends(["triton", "cute"])
def test_inline_nonfinite_literals_need_no_global_names(mode):
    x = torch.empty(17)
    with cpu_render_target():
        kernel = helion.kernel(inline_nonfinite_literal.fn, autotune_effort="none")
        run, _source = _scalar_host(kernel, (x, 2.0, mode))
        expected = {
            "positive": float("inf"),
            "negative": -float("inf"),
            "nan": float("nan"),
        }[mode]
        assert run(x, 2.0, mode).hex() == expected.hex()


@pytest.mark.parametrize("branch", [False, True, "alias"])
@onlyBackends(["triton", "cute"])
def test_rebound_host_float_does_not_overwrite_device_argument(branch):
    with cpu_render_target():
        bound = (
            cpu_bind(aliased_host_float, (torch.ones(1, 4), 1.0))
            if branch == "alias"
            else cpu_bind(rebound_host_float, (torch.ones(1, 4, 4), 1.0, branch))
        )
        config = bound.config_spec.default_config()
        config.config["block_sizes"] = [1, 1, 4] if branch else [1, 1, 4, 4]
        host, functions = host_functions(bound, config)
        launch = next(
            node
            for node in ast.walk(host)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_launcher"
        )
        device = next(node for node in functions if node.name == launch.args[0].id)
        arguments = {arg.arg for arg in device.args.args}
        assigned = {
            node.id
            for node in ast.walk(device)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
        }
        assert not arguments & assigned, bound.to_code(config)


@helion.kernel(autotune_effort="none")
def nonfinite_inline_sign(x, scale: float, value: hl.constexpr):
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        out[tile] = torch.copysign(x[tile], scale * value)
    return out


@pytest.mark.parametrize(
    "bits", ["7ff8000000000042", "fff8000000000037", "7ff0000000000001"]
)
@onlyBackends(["triton", "cute"])
def test_inline_nonfinite_float_preserves_operand_sign_and_payload(bits):
    value = struct.unpack("!d", bytes.fromhex(bits))[0]
    x = torch.ones(17)
    with cpu_render_target():
        run, source = _scalar_host(nonfinite_inline_sign, (x, 2.0, value))
        ast.parse(source)
        for scale in (2.0, -2.0):
            actual = run(x, scale, value)
            expected = scale * value
            assert struct.pack("!d", actual) == struct.pack("!d", expected)
            torch.testing.assert_close(
                torch.copysign(x, actual), torch.copysign(x, expected), rtol=0, atol=0
            )
