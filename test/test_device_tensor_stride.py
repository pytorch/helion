from __future__ import annotations

import ast
from dataclasses import dataclass

import pytest
import torch
from torch._inductor.codecache import PyCodeCache

import helion
from helion._compiler.variable_origin import TensorStrideOrigin
from helion._testing import cpu_render_target
import helion.language as hl
from helion.runtime.kernel import _make_prepared_arg_guard

pytestmark = pytest.mark.cpu_only


def test_repeated_host_created_stride_uses_one_symbol_during_both_passes():
    @helion.kernel(static_shapes=False, autotune_effort="none")
    def fn(x):
        created = torch.empty_like(x)
        out = torch.empty_like(x)
        for row, column in hl.tile(x.shape):
            first = created.stride(0)
            second = created.stride(0)
            out[row, column] = x[row, column] + first + second
        return out

    with cpu_render_target():
        bound = fn.bind((torch.empty(3, 4),))
        calls = [
            node
            for statement in bound.host_function.body
            for node in ast.walk(statement)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "stride"
        ]
        propagated = {call._type_info.value._sympy_() for call in calls}
        # Output-layout metadata may have another symbol with the same host
        # origin. Compare the explicit calls against their actual scalar uses.
        arithmetic_symbols = {
            node.args[1].args[0]
            for graph in bound.host_function.device_ir.graphs
            for node in graph.graph.nodes
            if node.target is torch.ops.aten.add.Tensor
            and isinstance(node.args[1], torch.fx.Node)
        }
        lowered = {
            expr
            for expr, origin in bound.host_function.expr_to_origin.items()
            if str(expr) in arithmetic_symbols
            and isinstance(origin.origin, TensorStrideOrigin)
            and origin.origin.host_str() == "created.stride(0)"
        }
        assert len(propagated) == len(arithmetic_symbols) == 1
        assert propagated == lowered
        ast.parse(bound.to_code())


def _device_stride(x, form: hl.constexpr):
    saved_stride = x.stride
    out = torch.empty_like(x)
    for row, column in hl.tile(x.shape):
        if form == "positional":
            stride = x.stride(0)
        elif form == "keyword":
            stride = x.stride(dim=0)
        elif form == "negative":
            stride = x.stride(-2)
        elif form == "tuple":
            stride = x.stride()[0]
        else:
            stride = saved_stride(0)
        out[row, column] = x[row, column] + stride
    return out


def _host_stride(x):
    stride = x.stride(0)
    out = torch.empty_like(x)
    for row, column in hl.tile(x.shape):
        out[row, column] = x[row, column] + stride
    return out


def _capture_launch(code, name, args):
    # The real source must be inspectable by Triton's jit decorator. Loading
    # it also rejects duplicate kernel parameters before any launch.
    namespace = vars(PyCodeCache.load(code))
    calls = []

    def launch(kernel, grid, *values, **options):
        calls.append(values)

    namespace[name](*args, _launcher=launch)
    assert len(calls) == 1
    module = ast.parse(code)
    wrapper = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    call = next(
        node
        for node in ast.walk(wrapper)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_launcher"
    )
    assert isinstance(call.args[0], ast.Name)
    device = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == call.args[0].id
    )
    # Pallas passes scalar arguments as one-element tensors; inspect the same
    # scalar value for every backend without dropping those arguments.
    values = (
        value.item()
        if isinstance(value, torch.Tensor) and value.numel() == 1
        else value
        for value in calls[0]
    )
    return dict(zip((arg.arg for arg in device.args.args), values, strict=True))


@pytest.mark.parametrize(
    "form", ["positional", "keyword", "negative", "tuple", "saved"]
)
def test_device_stride_uses_current_runtime_layout(form):
    kernel = helion.kernel(_device_stride, static_shapes=False, autotune_effort="none")
    x = torch.empty(3, 4)
    padded = torch.empty_strided(x.shape, (20, 1))
    with cpu_render_target():
        bound = kernel.bind((x, form))
        code = bound.to_code()
        for tensor in (x, padded):
            values = _capture_launch(code, kernel.name, (tensor, form))
            assert values["x_stride_0"] == tensor.stride(0)
        # The stride used in scalar arithmetic must not be the size symbol
        # that FakeTensor uses for the original contiguous layout.
        arithmetic = [
            node.args[1]
            for graph in bound.host_function.device_ir.graphs
            for node in graph.graph.nodes
            if node.target is torch.ops.aten.add.Tensor
        ]
        assert len(arithmetic) == 1
        assert isinstance(arithmetic[0], torch.fx.Node)
        symbol_name = arithmetic[0].args[0]
        origins = {
            str(expr): value.origin.host_str()
            for expr, value in bound.host_function.expr_to_origin.items()
            if str(expr) == symbol_name
        }
        assert origins.get(symbol_name) == "x.stride(0)", (symbol_name, origins)


def test_noncontiguous_device_stride_and_host_local_control():
    tensor = torch.empty_strided((3, 4), (20, 1))
    with cpu_render_target():
        for fn, args in (
            (_device_stride, (tensor, "positional")),
            (_host_stride, (tensor,)),
        ):
            kernel = helion.kernel(fn, static_shapes=False, autotune_effort="none")
            code = kernel.bind(args).to_code()
            values = _capture_launch(code, kernel.name, args)
            assert 20 in [value for value in values.values() if isinstance(value, int)]


def _unrolled_strides(values):
    out = torch.empty_like(values[0])
    for row, column in hl.tile(out.shape):
        result = values[0][row, column]
        for tensor in values:
            for stride in tensor.stride():
                result = result + stride
        out[row, column] = result
    return out


def test_unrolled_stride_calls_use_each_receiver_and_dimension():
    tensors = [
        torch.empty_strided((3, 4), (20, 2)),
        torch.empty_strided((3, 4), (40, 3)),
    ]
    with cpu_render_target():
        kernel = helion.kernel(
            _unrolled_strides, static_shapes=False, autotune_effort="none"
        )
        bound = kernel.bind((tensors,))
        code = bound.to_code()
        values = _capture_launch(code, kernel.name, (tensors,))
        assert {2, 3, 20, 40}.issubset(
            value for value in values.values() if isinstance(value, int)
        )
        host_strides = {
            value.origin.host_str()
            for value in bound.host_function.expr_to_origin.values()
            if value.fake_value is not None and ".stride(" in value.origin.host_str()
        }
        assert host_strides == {
            f"values[{index}].stride({axis})" for index in range(2) for axis in range(2)
        } | {"out.stride(0)", "out.stride(1)"}


def _traced_stride_increment(bound, arguments):
    # Read the scalar operands from the actual traced arithmetic, then resolve
    # their emitted host origins on these runtime inputs. A raw fake stride
    # incorrectly resolves to out.size(1) or a captured constant instead.
    origins = {
        str(expr): value.origin
        for expr, value in bound.host_function.expr_to_origin.items()
    }
    total = 0
    for graph in bound.host_function.device_ir.graphs:
        for node in graph.graph.nodes:
            if node.target is torch.ops.aten.add.Tensor:
                value = node.args[1]
                if isinstance(value, torch.fx.Node):
                    value = eval(origins[value.args[0]].host_str(), {}, arguments)
                total += value
    return total


@pytest.mark.parametrize("container", [list, tuple])
def test_aliased_unrolled_strides_preserve_runtime_layout_and_alias_guards(container):
    # Both layouts stay outside the canonical-layout specialization.
    x = torch.empty_strided((3, 4), (12, 2))
    padded = torch.empty_strided(x.shape, (20, 1))
    other = torch.empty_strided(x.shape, (40, 3))
    initial = (container((x, x)),)
    changed_layout = (container((padded, padded)),)
    separate = (container((padded, other)),)
    with cpu_render_target():
        kernel = helion.kernel(
            _unrolled_strides, static_shapes=False, autotune_effort="none"
        )
        bound = kernel.bind(initial)
        code = bound.to_code()
        assert kernel.bind(changed_layout) is bound
        for args in (initial, changed_layout):
            _capture_launch(code, kernel.name, args)
            assert _traced_stride_increment(
                bound, {"values": args[0], "out": torch.empty_like(args[0][0])}
            ) == sum(sum(tensor.stride()) for tensor in args[0])
        # The chosen input origin can serve both receiver occurrences only
        # while their alias relationship is unchanged. Exercise the existing
        # full, fast and prepared dispatch protections with identical metadata.
        nonalias_same_layout = (
            container((padded, torch.empty_strided(x.shape, (20, 1)))),
        )
        assert kernel.specialization_key(changed_layout) != kernel.specialization_key(
            nonalias_same_layout
        )
        assert kernel._fast_dispatch_key(changed_layout) != kernel._fast_dispatch_key(
            nonalias_same_layout
        )
        guard = _make_prepared_arg_guard(kernel, changed_layout)
        assert guard(changed_layout)
        assert not guard(nonalias_same_layout)
        reverse_guard = _make_prepared_arg_guard(kernel, nonalias_same_layout)
        assert reverse_guard(nonalias_same_layout)
        assert not reverse_guard(changed_layout)
        distinct_bound = kernel.bind(separate)
        assert distinct_bound is not bound
        distinct_code = distinct_bound.to_code()
        _capture_launch(distinct_code, kernel.name, separate)
        assert (
            _traced_stride_increment(
                distinct_bound, {"values": separate[0], "out": torch.empty_like(padded)}
            )
            == 64
        )
        assert kernel.bind(changed_layout) is bound


@dataclass
class _StrideField:
    tensor: torch.Tensor


def _nested_alias_stride(values):
    saved_stride = values["field"].tensor.stride
    out = torch.empty_like(values["rows"][0])
    for row, column in hl.tile(out.shape):
        out[row, column] = (
            values["rows"][0][row, column] + saved_stride(0) + saved_stride(-1)
        )
    return out


def test_nested_aliased_field_saved_stride_uses_current_receiver():
    # Both layouts stay outside the canonical-layout specialization.
    x = torch.empty_strided((3, 4), (12, 2))
    padded = torch.empty_strided(x.shape, (20, 1))
    separate_tensor = torch.empty_strided(x.shape, (40, 3))

    def arguments(rows, field):
        return ({"rows": [rows], "field": _StrideField(field)},)

    initial = arguments(x, x)
    changed = arguments(padded, padded)
    separate = arguments(padded, separate_tensor)
    with cpu_render_target():
        kernel = helion.kernel(
            _nested_alias_stride, static_shapes=False, autotune_effort="none"
        )
        bound = kernel.bind(initial)
        code = bound.to_code()
        assert kernel.bind(changed) is bound
        launch = _capture_launch(code, kernel.name, changed)
        assert 20 in [value for value in launch.values() if isinstance(value, int)]
        assert (
            _traced_stride_increment(
                bound, {"values": changed[0], "out": torch.empty_like(padded)}
            )
            == 21
        )
        assert kernel.specialization_key(changed) != kernel.specialization_key(separate)
        distinct_bound = kernel.bind(separate)
        assert distinct_bound is not bound
        _capture_launch(distinct_bound.to_code(), kernel.name, separate)
        assert (
            _traced_stride_increment(
                distinct_bound, {"values": separate[0], "out": torch.empty_like(padded)}
            )
            == 43
        )
        assert kernel.bind(changed) is bound


@pytest.mark.parametrize(
    "form", ["host_scalar", "host_tuple", "device_scalar", "device_tuple"]
)
def test_host_created_stride_preserves_runtime_shape(form):
    @helion.kernel(static_shapes=False, autotune_effort="none")
    def fn(x, form: hl.constexpr):
        created = torch.empty_like(x)
        if form == "host_scalar":
            saved = created.stride(0)
        elif form == "host_tuple":
            saved = created.stride()[0]
        else:
            saved = 0
        out = torch.empty_like(x)
        for row, column in hl.tile(x.shape):
            if form == "device_scalar":
                stride = created.stride(0)
            elif form == "device_tuple":
                stride = created.stride()[0]
            else:
                stride = saved
            out[row, column] = x[row, column] + stride
        return out

    with cpu_render_target():
        initial = torch.empty(3, 4)
        bound = fn.bind((initial, form))
        code = bound.to_code()
        for x in (torch.empty(3, 4), torch.empty(5, 7), torch.empty(4, 3).t()):
            current = fn.bind((x, form))
            if bound.settings.backend == "triton":
                if x.is_contiguous():
                    assert current is bound
                else:
                    assert current is not bound
            # Contiguous shapes reuse the canonical-layout proof; a transpose
            # changes that guard. Stride arithmetic must follow the selected
            # binding in both cases, independently of alignment specialization.
            current_code = code if current is bound else current.to_code()
            _capture_launch(current_code, fn.name, (x, form))
            created = torch.empty_like(x)
            assert _traced_stride_increment(
                current,
                {
                    "x": x,
                    "created": created,
                    "out": torch.empty_like(x),
                    "saved": created.stride(0),
                },
            ) == created.stride(0)
