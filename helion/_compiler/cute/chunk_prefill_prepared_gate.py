"""Original gate increment and beta pointwise lowering at the native scan cut."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

import torch
from torch.fx import Node

from ..compile_environment import CompileEnvironment
from ..inductor_lowering import PointwiseCodegenPolicy
from .chained_fragment_epilogue import bind_register_fragment_expression
from .factor_affine_reductions import pack_fp32_constexpr_loops
from .producer_phase import PointStore
from .producer_phase import PointValue
from .producer_phase import emit_point_actions
from .producer_phase import serial_prefix_actions

if TYPE_CHECKING:
    from ..generate_ast import GenerateAST
    from .chunk_prefill_prepared_issue import FastRecurrence


def _node(value: object) -> Node:
    assert isinstance(value, Node)
    return value


def _source(name: str, parameters: str, body: list[ast.stmt]) -> tuple[str, str]:
    return name, "\n".join(
        (
            "@cute.jit",
            f"def {name}({parameters}):",
            *(
                "    " + line
                for line in ast.unparse(
                    ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
                ).splitlines()
            ),
            "",
        )
    )


def bind_fast_gate_publications(
    cg: GenerateAST, owner: FastRecurrence
) -> tuple[tuple[str, str], ...]:
    owner.check()
    masked = _node(owner.region.step.gate_scan.args[1])
    assert masked.target is torch.ops.aten.where.self
    increment = _node(masked.args[1])
    activation = _node(increment.args[0])
    half_tanh = _node(activation.args[0])
    tanh = _node(half_tanh.args[0])
    half_input = _node(tanh.args[0])
    rate_product = _node(half_input.args[0])
    rate, biased = map(_node, rate_product.args)
    raw, bias_view = map(_node, biased.args)
    bias = _node(bias_view.args[0])
    scale = _node(increment.args[1])
    coordinates = ("row", "key")
    full_points = []
    tail_points = []
    tanh_body = []
    tanh_values = []
    for index in range(4):
        expression = bind_register_fragment_expression(
            cg,
            owner.graph,
            raw,
            increment,
            coordinates,
            coordinates,
            f"raw[{index}]",
            "cutlass.Float32",
            inputs={
                bias: (("key",), "bias"),
                rate: ((), "gate_rate"),
                scale: ((), "gate_scale_log2"),
            },
        )
        expression.check()
        statements = ast.parse("\n".join(expression.expression.lines)).body
        store = PointStore(ast.parse(f"result[{index}]", mode="eval").body)
        tanh_expression = bind_register_fragment_expression(
            cg,
            owner.graph,
            raw,
            tanh,
            coordinates,
            coordinates,
            f"raw[{index}]",
            "cutlass.Float32",
            inputs={bias: (("key",), "bias"), rate: ((), "gate_rate")},
        )
        tanh_body.extend(ast.parse("\n".join(tanh_expression.expression.lines)).body)
        tanh_values.append(tanh_expression.value)
        full_expression = bind_register_fragment_expression(
            cg,
            owner.graph,
            activation,
            increment,
            coordinates,
            coordinates,
            f"activated[{index}]",
            "cutlass.Float32",
            inputs={scale: ((), "gate_scale_log2")},
        )
        full_points.append(
            PointValue(
                tuple(ast.parse("\n".join(full_expression.expression.lines)).body),
                ast.parse(full_expression.value, mode="eval").body,
                store,
            )
        )
        tail = ast.parse(f"increment_{index} = cutlass.Float32(0.0)")
        tail.body.append(
            ast.If(
                test=ast.parse(
                    f"chunk * 32 + row0 + {index} < seqlen", mode="eval"
                ).body,
                body=[
                    *statements,
                    *ast.parse(f"increment_{index} = {expression.value}").body,
                ],
                orelse=[],
            )
        )
        tail_points.append(
            PointValue(
                tuple(tail.body),
                ast.parse(f"increment_{index}", mode="eval").body,
                store,
            )
        )
    activation_expression = bind_register_fragment_expression(
        cg,
        owner.graph,
        tanh,
        activation,
        coordinates,
        coordinates,
        "cutlass.Float32(tanh_values[element])",
        "cutlass.Float32",
    )
    activation_body = ast.parse(
        "\n".join(
            (
                "activated = []",
                "for element in cutlass.range_constexpr(4):",
                *("    " + line for line in activation_expression.expression.lines),
                f"    activated.append({activation_expression.value})",
            )
        )
    ).body
    env = CompileEnvironment.current()
    activation_body = pack_fp32_constexpr_loops(
        activation_body,
        fast_math=env.settings.fast_math,
        target_device_capability=env.config_spec.target_device_capability,
        float_scalar_names=set(),
    )
    tanh_body.extend(ast.parse("tanh_values = (" + ", ".join(tanh_values) + ")").body)
    full = emit_point_actions(
        serial_prefix_actions("prefix", tuple(full_points), evaluate_first=True)
    )
    tail = emit_point_actions(
        serial_prefix_actions("prefix", tuple(tail_points), evaluate_first=True)
    )
    assert full is not None and tail is not None
    body = ast.parse(
        "result = cutlass.Array(cutlass.Float32, 4, space=cutlass.AddressSpace.rmem)"
    ).body
    body.append(
        ast.If(
            test=ast.parse("cutlass.const_expr(MASK_TAIL)", mode="eval").body,
            body=list(tail),
            orelse=[*tanh_body, *activation_body, *full],
        )
    )
    body.extend(ast.parse("return result[0], result[1], result[2], result[3]").body)
    gate = _source(
        "_helion_gate_group",
        "raw, bias, gate_rate, gate_scale_log2, prefix, row0, chunk, seqlen, MASK_TAIL: cutlass.Constexpr",
        body,
    )
    rhs = _node(owner.inverse_product.rhs.args[0])
    beta_view = _node(rhs.args[0])
    beta = _node(beta_view.args[0])
    assert beta.target is torch.ops.aten.sigmoid.default
    beta_raw = _node(beta.args[0])
    expression = bind_register_fragment_expression(
        cg,
        owner.graph,
        beta_raw,
        beta,
        ("row",),
        ("row",),
        "value",
        "cutlass.Float32",
        policy=PointwiseCodegenPolicy(sigmoid_tanh=True),
    )
    expression.check()
    beta_helper = _source(
        "_helion_beta",
        "value",
        ast.parse(
            "\n".join((*expression.expression.lines, f"return {expression.value}"))
        ).body,
    )
    owner.check()
    return gate, beta_helper
