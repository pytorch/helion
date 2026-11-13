from __future__ import annotations

import ast
import itertools
from types import SimpleNamespace
from typing import Callable
from unittest.mock import patch

import pytest
import torch
from torch.fx.experimental.symbolic_shapes import ShapeEnv

import helion
from helion._compiler.cute import cute_epilogue
from helion._testing.cute_codegen import cpu_bind
import helion.language as hl
from helion.language import memory_ops

pytestmark = pytest.mark.cpu_only


@helion.kernel(backend="cute", static_shapes=True)
def _matmul_with_scalar_store(
    x: torch.Tensor, y: torch.Tensor, scale: float
) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.empty_like(x)
    scalar_out = torch.empty_like(x)
    for row, col in hl.tile(x.shape):
        acc = hl.zeros([row, col], dtype=torch.float32)
        for inner in hl.tile(x.size(1)):
            acc = hl.dot(x[row, inner], y[inner, col], acc=acc)
        out[row, col] = acc
        scalar = scale
        scalar_out[row, col] = scalar
    return out, scalar_out


def test_scalar_store_beside_matmul_keeps_scalar_codegen() -> None:
    x = torch.empty((128, 128), dtype=torch.bfloat16)
    bound = cpu_bind(_matmul_with_scalar_store, (x, x.clone(), 0.25))
    source = bound.to_code(bound.config_spec.autotune_reference_config())
    ast.parse(source)
    assert "scalar_out" in source
    assert bound.host_function is not None
    stores = [
        node
        for graph in bound.host_function.device_ir.graphs
        for node in graph.graph.nodes
        if node.target is memory_ops.store and isinstance(node.args[2], torch.fx.Node)
    ]
    assert len(stores) == 2
    values = [node.args[2].meta.get("val") for node in stores]
    assert sum(isinstance(value, torch.SymFloat) for value in values) == 1
    assert sum(isinstance(value, torch.Tensor) for value in values) == 1


def test_carrier_shape_requires_tensor_metadata() -> None:
    env = ShapeEnv()
    graph = torch.fx.Graph()
    scalar = graph.placeholder("scalar")
    anchor = graph.placeholder("unrelated_matmul")
    scalar.meta["val"] = env.create_unbacked_symfloat()
    assert (
        cute_epilogue.analyze_tcgen05_unary_epilogue_chain(
            None, scalar, target_fx_nodes={anchor}, inner_outputs_by_graph_id={}
        )
        is None
    )
    for value in (
        None,
        env.create_unbacked_symint(),
        env.create_unbacked_symbool(),
        1,
        1.5,
        True,
        SimpleNamespace(shape=(4, 16)),
    ):
        scalar.meta["val"] = value
        assert cute_epilogue._carrier_tile_shape(scalar) is None
    for shape in ((), (4, 16)):
        scalar.meta["val"] = torch.empty(shape)
        assert cute_epilogue._carrier_tile_shape(scalar) == shape


def _call(
    graph: torch.fx.Graph,
    target: Callable[..., object],
    args: tuple[object, ...],
    dtype: torch.dtype,
) -> torch.fx.Node:
    result = graph.call_function(target, args=args)
    result.meta["val"] = torch.empty((4, 16), dtype=dtype)
    return result


def _render(
    value: torch.fx.Node, anchor: torch.fx.Node, values: torch.Tensor
) -> torch.Tensor:
    analysis = cute_epilogue.analyze_tcgen05_unary_epilogue_chain(
        None, value, target_fx_nodes={anchor}, inner_outputs_by_graph_id={}
    )
    assert analysis is not None and analysis[1] is anchor
    names = itertools.count()
    source, expression = analysis[0].render_prelude_and_expr(
        "values", lambda prefix: f"{prefix}_{next(names)}", ""
    )
    namespace = {
        "values": values,
        "cutlass": SimpleNamespace(
            Float16=torch.float16, BFloat16=torch.bfloat16, Float32=torch.float32
        ),
    }
    exec(source + f"result = {expression}\n", namespace)
    return namespace["result"]


@pytest.mark.parametrize("intermediate", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("output", [torch.float16, torch.bfloat16, torch.float32])
def test_intermediate_cast_and_each_arithmetic_step_preserve_rounding(
    intermediate: torch.dtype, output: torch.dtype
) -> None:
    graph = torch.fx.Graph()
    anchor = graph.placeholder("mma")
    anchor.meta["val"] = torch.empty((4, 16), dtype=torch.float32)
    scale = _call(graph, torch.ops.aten.mul.Tensor, (anchor, 1.03125), torch.float32)
    rounded = _call(
        graph,
        torch.ops.prims.convert_element_type.default,
        (scale, intermediate),
        intermediate,
    )
    bias = _call(graph, torch.ops.aten.add.Tensor, (rounded, 0.00390625), intermediate)
    result = _call(graph, torch.ops.aten.mul.Tensor, (bias, 1.0625), intermediate)
    final = _call(
        graph, torch.ops.prims.convert_element_type.default, (result, output), output
    )
    values = torch.linspace(-3.125, 3.125, 64).reshape(4, 16)
    actual = _render(final, anchor, values).to(output)
    expected = (((values * 1.03125).to(intermediate) + 0.00390625) * 1.0625).to(output)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    dropped_rounding = ((values * 1.03125 + 0.00390625) * 1.0625).to(output)
    assert not torch.equal(expected, dropped_rounding)


@pytest.mark.parametrize("intermediate", [torch.int32, torch.float64, torch.bool])
def test_casts_outside_fp16_bf16_fp32_remain_rejected(
    intermediate: torch.dtype,
) -> None:
    graph = torch.fx.Graph()
    anchor = graph.placeholder("mma")
    anchor.meta["val"] = torch.empty((4, 16), dtype=torch.float32)
    rounded = _call(
        graph,
        torch.ops.prims.convert_element_type.default,
        (anchor, intermediate),
        intermediate,
    )
    final = _call(
        graph,
        torch.ops.prims.convert_element_type.default,
        (rounded, torch.float32),
        torch.float32,
    )
    assert (
        cute_epilogue.analyze_tcgen05_unary_epilogue_chain(
            None, final, target_fx_nodes={anchor}, inner_outputs_by_graph_id={}
        )
        is None
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_silu_decomposition_does_not_drop_intermediate_rounding(
    dtype: torch.dtype,
) -> None:
    graph = torch.fx.Graph()
    carrier = graph.placeholder("carrier")
    carrier.meta["val"] = torch.empty((4, 16), dtype=dtype)
    sigmoid = _call(graph, torch.ops.aten.sigmoid.default, (carrier,), dtype)
    product = _call(graph, torch.ops.aten.mul.Tensor, (carrier, sigmoid), dtype)
    assert cute_epilogue._classify_silu(product) is None


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_auxiliary_arithmetic_rounds_before_a_later_expression(
    dtype: torch.dtype,
) -> None:
    graph = torch.fx.Graph()
    load = graph.placeholder("load")
    load.meta["val"] = torch.empty((4, 16), dtype=dtype)
    bias = _call(graph, torch.ops.aten.add.Tensor, (load, 0.03125), dtype)
    product = _call(graph, torch.ops.aten.mul.Tensor, (bias, 1.0625), dtype)
    with patch.object(
        cute_epilogue,
        "aux_tensor_load_kind",
        side_effect=lambda node, **kwargs: ("exact", None) if node is load else None,
    ):
        expression = cute_epilogue._classify_auxiliary_tensor_expr(
            product,
            carrier_tile_shape=(4, 16),
            carrier_tile_index_nodes=None,
            carrier_global_shape=(4, 16),
        )
    assert expression is not None
    operands = cute_epilogue._auxiliary_tensor_expr_operands(expression)
    assert len(operands) == 1
    names = itertools.count()
    source, result = cute_epilogue._render_auxiliary_tensor_expr(
        expression,
        "unused",
        {operands[0]: "values"},
        lambda prefix: f"{prefix}_{next(names)}",
        "",
    )
    values = torch.linspace(-3.125, 3.125, 64).reshape(4, 16).to(dtype)
    namespace = {
        "values": values,
        "cutlass": SimpleNamespace(
            Float16=torch.float16, BFloat16=torch.bfloat16, Float32=torch.float32
        ),
    }
    exec(source + f"result = {result}\n", namespace)
    torch.testing.assert_close(
        namespace["result"].to(dtype), (values + 0.03125) * 1.0625, rtol=0, atol=0
    )


@pytest.mark.parametrize(
    ("dtype", "literal"),
    [
        (torch.float16, "0.300048828125"),
        (torch.bfloat16, "0.30078125"),
        (torch.float32, "0.30000001192092896"),
    ],
)
@pytest.mark.parametrize(
    "target",
    [torch.ops.aten.gt.Scalar, torch.ops.aten.eq.Scalar, torch.ops.aten.le.Scalar],
)
def test_comparison_scalar_is_rounded_to_the_operand_dtype(
    dtype: torch.dtype, literal: str, target: object
) -> None:
    # torch compares ``fp16_tensor > 0.3`` against fp16(0.3), as do the SIMT
    # and Triton lowerings; the chain renders that rounded literal.
    graph = torch.fx.Graph()
    load = graph.placeholder("load")
    load.meta["val"] = torch.empty((4, 16), dtype=dtype)
    compare = _call(graph, target, (load, 0.3), torch.bool)  # type: ignore[arg-type]
    with patch.object(
        cute_epilogue,
        "aux_tensor_load_kind",
        side_effect=lambda node, **kwargs: ("exact", None) if node is load else None,
    ):
        expression = cute_epilogue._classify_auxiliary_tensor_expr(
            compare,
            carrier_tile_shape=(4, 16),
            carrier_tile_index_nodes=None,
            carrier_global_shape=(4, 16),
        )
    assert expression is not None
    operands = cute_epilogue._auxiliary_tensor_expr_operands(expression)
    assert len(operands) == 1
    names = itertools.count()
    source, result = cute_epilogue._render_auxiliary_tensor_expr(
        expression,
        "unused",
        {operands[0]: "values"},
        lambda prefix: f"{prefix}_{next(names)}",
        "",
    )
    assert f"({literal})" in source + result
    assert "(0.3)" not in source + result
    # The planted value fp16(0.3) itself, its neighbours and a far value.
    planted = torch.tensor(0.3, dtype=dtype)
    values = (
        torch.stack(
            [
                planted,
                torch.nextafter(planted, torch.tensor(0.0, dtype=dtype)),
                torch.nextafter(planted, torch.tensor(1.0, dtype=dtype)),
                torch.tensor(-2.0, dtype=dtype),
            ]
        )
        .repeat(16, 1)
        .T.contiguous()
    )
    namespace = {"values": values.float()}
    exec(source + f"result = {result}\n", namespace)
    assert isinstance(target, torch._ops.OpOverload)
    torch.testing.assert_close(namespace["result"], target(values, 0.3))


def test_comparison_scalar_outside_the_operand_dtype_declines() -> None:
    # fp16(1e5) is inf; torch compares against inf, which the chain does not model.
    graph = torch.fx.Graph()
    load = graph.placeholder("load")
    load.meta["val"] = torch.empty((4, 16), dtype=torch.float16)
    compare = _call(graph, torch.ops.aten.gt.Scalar, (load, 1e5), torch.bool)
    with patch.object(
        cute_epilogue,
        "aux_tensor_load_kind",
        side_effect=lambda node, **kwargs: ("exact", None) if node is load else None,
    ):
        assert (
            cute_epilogue._classify_auxiliary_tensor_expr(
                compare,
                carrier_tile_shape=(4, 16),
                carrier_tile_index_nodes=None,
                carrier_global_shape=(4, 16),
            )
            is None
        )
