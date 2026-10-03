"""Original BT16 gate point expressions and ordered serial prefix actions."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

import torch

from ..inductor_lowering import PointwiseCodegenPolicy
from . import chained_matmul as chain
from .chained_fragment_epilogue import bind_register_fragment_expression
from .chunk_prefill_prepared_bt16 import node_arg
from .chunk_prefill_prepared_factor_inputs import _source
from .producer_phase import PointStore
from .producer_phase import PointValue
from .producer_phase import emit_point_actions
from .producer_phase import serial_prefix_actions

if TYPE_CHECKING:
    from torch.fx import Node

    from ..generate_ast import GenerateAST
    from .chained_fragment_epilogue import FragmentExpression
    from .chunk_prefill_prepared_bt16 import BT16Bindings


def bind_bt16_gates(
    cg: GenerateAST, owner: BT16Bindings
) -> tuple[tuple[str, str], ...]:
    owner.check()
    prefix = owner.region.step.gate_scan
    masked = node_arg(prefix, 1)
    increment = node_arg(masked, 1)
    activation = node_arg(increment, 0)
    tanh = node_arg(node_arg(activation, 0), 0)
    rate_product = node_arg(node_arg(tanh, 0), 0)
    rate, biased = node_arg(rate_product, 0), node_arg(rate_product, 1)
    raw, bias = node_arg(biased, 0), node_arg(node_arg(biased, 1), 0)
    scale = node_arg(increment, 1)
    if masked.target is not torch.ops.aten.where.self or prefix.args[2:] != (
        0,
        False,
        False,
    ):
        raise chain._UnsupportedChain("changed original BT16 serial prefix")
    coords = ("row", "column")

    def expression(
        source: Node,
        target: Node,
        fragment: str,
        inputs: dict[Node, tuple[tuple[str, ...], str]],
        coordinates: tuple[str, ...] = coords,
    ) -> FragmentExpression:
        result = bind_register_fragment_expression(
            cg,
            owner.graph,
            source,
            target,
            coordinates,
            coordinates,
            fragment,
            "cutlass.Float32",
            inputs=inputs,
            policy=PointwiseCodegenPolicy(sigmoid_tanh=True),
        )
        result.check()
        return result

    gate_lines: list[str] = []
    values = []
    for index in range(2):
        point = expression(
            raw,
            increment,
            f"raw{index}",
            {
                bias: (("column",), "bias"),
                rate: ((), "rate"),
                scale: ((), "gate_scale"),
            },
        )
        gate_lines.extend(point.expression.lines)
        values.append(point.value)
    gate_lines.append("return " + ", ".join(values))
    gate = _source(
        "_helion_bt16_gate_pair",
        "raw0, raw1, rate, bias, gate_scale",
        ast.parse("\n".join(gate_lines)).body,
    )
    points = tuple(
        PointValue(
            (),
            ast.parse(f"value{index}", mode="eval").body,
            PointStore(ast.parse(f"result[{index}]", mode="eval").body),
        )
        for index in range(2)
    )
    serial = emit_point_actions(
        serial_prefix_actions("carry", points, round_each_add=True)
    )
    assert serial is not None
    scan = _source(
        "_helion_bt16_prefix_pair",
        "carry, value0, value1",
        [
            *ast.parse("result = [cutlass.Float32(0.0), cutlass.Float32(0.0)]").body,
            *serial,
            *ast.parse("return result[0], result[1]").body,
        ],
    )
    scan = (
        scan[0],
        "from helion._compiler.cute.affine_recurrence_primitives import add_fp32_rn as _helion_add_fp32_rn\n"
        + scan[1],
    )
    exp_gate = node_arg(node_arg(owner.projection.lhs, 0), 1)
    exponent = expression(prefix, exp_gate, "prefix", {})
    exp = _source(
        "_helion_bt16_gate_exp",
        "prefix",
        ast.parse(
            "\n".join((*exponent.expression.lines, f"return {exponent.value}"))
        ).body,
    )
    rate_argument = node_arg(rate, 0)
    a_log, log2_e = node_arg(rate_argument, 0), node_arg(rate_argument, 1)
    coefficient = expression(a_log, rate, "value", {log2_e: ((), "log2_e")}, ())
    rate_source = _source(
        "_helion_bt16_gate_rate",
        "value, log2_e",
        ast.parse(
            "\n".join((*coefficient.expression.lines, f"return {coefficient.value}"))
        ).body,
    )
    lower_mask = node_arg(node_arg(owner.lower_value, 0), 0)
    beta = node_arg(node_arg(node_arg(lower_mask, 1), 1), 0)
    beta_source = node_arg(beta, 0)
    beta_point = expression(beta_source, beta, "value", {}, ("row",))
    beta_helper = _source(
        "_helion_bt16_beta",
        "value",
        ast.parse(
            "\n".join((*beta_point.expression.lines, f"return {beta_point.value}"))
        ).body,
    )
    owner.check()
    return gate, scan, exp, rate_source, beta_helper
