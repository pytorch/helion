"""Ordinary pointwise lowering for the fast state's original register images."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

from torch.fx import Node

from ..compile_environment import CompileEnvironment
from .chained_fragment_epilogue import bind_register_fragment_expression
from .factor_affine_reductions import pack_fp32_constexpr_loops

if TYPE_CHECKING:
    from ..generate_ast import GenerateAST
    from .chunk_prefill_prepared_issue import FastRecurrence


def _pair_source(
    cg: GenerateAST,
    owner: FastRecurrence,
    name: str,
    parameters: str,
    source: Node,
    target: Node,
    coordinates: tuple[str, str],
    fragment: str,
    inputs: dict[Node, tuple[tuple[str, ...], str]],
) -> tuple[str, str]:
    expression = bind_register_fragment_expression(
        cg,
        owner.graph,
        source,
        target,
        coordinates,
        coordinates,
        fragment,
        "cutlass.Float32",
        inputs=inputs,
    )
    expression.check()
    lines = [
        "result = []",
        "for element in cutlass.range_constexpr(2):",
        *("    " + line for line in expression.expression.lines),
        f"    result.append({expression.value})",
        "return result[0], result[1]",
    ]
    env = CompileEnvironment.current()
    body = pack_fp32_constexpr_loops(
        ast.parse("\n".join(lines)).body,
        fast_math=env.settings.fast_math,
        target_device_capability=env.config_spec.target_device_capability,
        float_scalar_names={"center_scale"},
    )
    source_text = (
        "\n".join(
            (
                "@cute.jit",
                f"def {name}({parameters}):",
                *(
                    "    " + line
                    for line in ast.unparse(
                        ast.Module(body=body, type_ignores=[])
                    ).splitlines()
                ),
            )
        )
        + "\n"
    )
    return name, source_text


def _point_source(
    cg: GenerateAST,
    owner: FastRecurrence,
    name: str,
    source: Node,
    target: Node,
    coordinates: tuple[str, ...],
    target_coordinates: tuple[str, ...],
) -> tuple[str, str]:
    expression = bind_register_fragment_expression(
        cg,
        owner.graph,
        source,
        target,
        coordinates,
        target_coordinates,
        "value",
        "cutlass.Float32",
    )
    expression.check()
    return name, "\n".join(
        (
            "@cute.jit",
            f"def {name}(value):",
            *("    " + line for line in expression.expression.lines),
            f"    return {expression.value}",
            "",
        )
    )


def bind_fast_state_publications(
    cg: GenerateAST,
    owner: FastRecurrence,
) -> tuple[tuple[str, str], ...]:
    owner.check()
    gamma_view = owner.scaled_state.args[1]
    assert isinstance(gamma_view, Node)
    gamma = gamma_view.args[0]
    assert isinstance(gamma, Node)
    rhs = owner.inverse_product.rhs.args[0]
    assert isinstance(rhs, Node)
    beta_view, difference = rhs.args
    assert isinstance(beta_view, Node) and isinstance(difference, Node)
    beta = beta_view.args[0]
    raw, projected = difference.args
    assert (
        isinstance(beta, Node) and isinstance(raw, Node) and isinstance(projected, Node)
    )
    projection, center = projected.args
    assert projection is owner.projection.node and isinstance(center, Node)
    center_image = center.args[0]
    assert isinstance(center_image, Node)
    center_value = center_image.args[1]
    assert isinstance(center_value, Node)
    gate_scale = center_value.args[0]
    assert isinstance(gate_scale, Node)
    output_value = owner.region.step.output_store.args[2]
    assert isinstance(output_value, Node)
    return (
        _pair_source(
            cg,
            owner,
            "_helion_state_decay_pair",
            "values, coefficients",
            owner.state_input,
            owner.scaled_state,
            ("value", "key"),
            "cutlass.Float32(values[element])",
            {gamma: (("key",), "cutlass.Float32(coefficients[element])")},
        ),
        _pair_source(
            cg,
            owner,
            "_helion_state_rhs_pair",
            "prediction, raw, beta, center_scale",
            owner.projection.node,
            rhs,
            ("token", "value"),
            "cutlass.Float32(prediction[element])",
            {
                raw: (("token", "value"), "cutlass.Float32(raw[element])"),
                beta: (("token",), "cutlass.Float32(beta[element])"),
                center: ((), "center_scale"),
            },
        ),
        _point_source(
            cg,
            owner,
            "_helion_state_snapshot_cast",
            owner.state_input,
            owner.packed_state_nodes[0],
            ("value", "key"),
            ("key", "value"),
        ),
        _point_source(
            cg,
            owner,
            "_helion_state_rhs_cast",
            rhs,
            owner.inverse_product.rhs,
            ("token", "value"),
            ("token", "value"),
        ),
        _point_source(
            cg,
            owner,
            "_helion_state_update_cast",
            owner.inverse_product.node,
            owner.output_update.rhs,
            ("token", "value"),
            ("token", "value"),
        ),
        _point_source(
            cg,
            owner,
            "_helion_state_output_cast",
            owner.output_update.node,
            output_value,
            ("token", "value"),
            ("token", "value"),
        ),
        _point_source(
            cg,
            owner,
            "_helion_state_center_coefficient",
            gate_scale,
            center,
            (),
            (),
        ),
    )
