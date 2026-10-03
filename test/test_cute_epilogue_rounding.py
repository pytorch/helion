from __future__ import annotations

import itertools
from types import SimpleNamespace
from typing import Callable
from unittest.mock import patch

import pytest
import torch

from helion._compiler.cute import cute_epilogue


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
