"""Original register point and row-collective lowering for factor input images."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

import torch
from torch.fx import Node

from ..compile_environment import CompileEnvironment
from ..inductor_lowering import PointwiseCodegenPolicy
from .chained_fragment_epilogue import bind_register_fragment_expression
from .factor_affine_reductions import pack_fp32_constexpr_loops
from .producer_phase import RowSumPoint
from .producer_phase import RowSumTopology
from .producer_phase import emit_point_actions
from .producer_phase import row_sum_actions

if TYPE_CHECKING:
    from ..generate_ast import GenerateAST
    from .chained_fragment_epilogue import FragmentExpression
    from .chunk_prefill_prepared_issue import FastRecurrence


def _arg(node: Node, index: int) -> Node:
    result = node.args[index]
    assert isinstance(result, Node)
    return result


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


def _pair_source(
    name: str,
    parameters: str,
    expressions: tuple[tuple[str, FragmentExpression], ...],
    scalar_names: set[str],
) -> tuple[str, str]:
    lines: list[str] = []
    for result, expression in expressions:
        expression.check()
        lines.extend((f"{result} = []", "for element in cutlass.range_constexpr(2):"))
        lines.extend("    " + line for line in expression.expression.lines)
        lines.append(f"    {result}.append({expression.value})")
    lines.append(
        "return "
        + ", ".join(f"{name}[0], {name}[1]" for name, expression in expressions)
    )
    env = CompileEnvironment.current()
    body = pack_fp32_constexpr_loops(
        ast.parse("\n".join(lines)).body,
        fast_math=env.settings.fast_math,
        target_device_capability=env.config_spec.target_device_capability,
        float_scalar_names=scalar_names,
    )
    return _source(name="_helion_" + name, parameters=parameters, body=body)


def _scalar_source(
    cg: GenerateAST,
    owner: FastRecurrence,
    name: str,
    parameters: str,
    source: Node,
    target: Node,
    coordinates: tuple[str, ...],
    fragment: str,
    inputs: dict[Node, tuple[tuple[str, ...], str]],
) -> tuple[str, str]:
    value = bind_register_fragment_expression(
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
    value.check()
    lines = [*value.expression.lines, f"return {value.value}"]
    return _source(name, parameters, ast.parse("\n".join(lines)).body)


def bind_fast_factor_inputs(
    cg: GenerateAST, owner: FastRecurrence
) -> tuple[tuple[str, str], ...]:
    """Bind actual rounded restore images and original row-reduction values."""
    owner.check()
    coords = ("row", "column")
    qr = owner.query.lhs
    kr = owner.state_update.rhs
    q_product = _arg(qr, 0)
    k_product = _arg(kr, 0)
    qd = _arg(_arg(q_product, 0), 0)
    ki = _arg(_arg(k_product, 0), 0)
    center_scale = _arg(q_product, 1)
    restore_factor = _arg(_arg(k_product, 1), 0)
    kd = owner.projection.lhs
    qscaled, decay = _arg(_arg(qd, 0), 0), _arg(_arg(qd, 0), 1)
    qnorm, scale = _arg(qscaled, 0), _arg(qscaled, 1)
    knorm = _arg(_arg(kd, 0), 0)
    qraw, kraw = _arg(qnorm, 0), _arg(knorm, 0)
    qinv, kinv = _arg(_arg(qnorm, 1), 0), _arg(_arg(knorm, 1), 0)
    qsum, ksum = _arg(_arg(qinv, 0), 0), _arg(_arg(kinv, 0), 0)
    assert _arg(_arg(kd, 0), 1) is decay
    assert _arg(_arg(ki, 0), 0) is knorm
    assert qsum.target is torch.ops.aten.sum.dim_IntList and qsum.args[1] == [-1]
    assert ksum.target is torch.ops.aten.sum.dim_IntList and ksum.args[1] == [-1]

    def expression(
        source: Node,
        target: Node,
        fragment: str,
        inputs: dict[Node, tuple[tuple[str, ...], str]],
    ) -> FragmentExpression:
        return bind_register_fragment_expression(
            cg,
            owner.graph,
            source,
            target,
            coords,
            coords,
            fragment,
            "cutlass.BFloat16",
            inputs=inputs,
            policy=PointwiseCodegenPolicy(reciprocal_ftz="FAST_RCP"),
        )

    restore_q = expression(
        qd, qr, "cutlass.Float32(qv[element])", {center_scale: ((), "center_scale")}
    )
    restore_k = expression(
        ki,
        kr,
        "cutlass.Float32(kv[element])",
        {restore_factor: (("column",), "cutlass.Float32(restore_factors[element])")},
    )
    restore = _pair_source(
        "factor_restore_pair",
        "qv, kv, restore_factors, center_scale",
        (("query", restore_q), ("key", restore_k)),
        {"center_scale"},
    )
    row_points: list[RowSumPoint] = []
    for name, raw, summed in (("qsum", qraw, qsum), ("ksum", kraw, ksum)):
        square = _arg(summed, 0)
        assert square.target is torch.ops.aten.mul.Tensor and square.args == (raw, raw)
        point = expression(
            raw,
            square,
            f"cutlass.Float32({'qv' if name == 'qsum' else 'kv'}[element])",
            {},
        )
        point.check()
        row_points.append(RowSumPoint(name, tuple(point.expression.lines), point.value))
    env = CompileEnvironment.current()
    row_actions = row_sum_actions(
        tuple(row_points),
        RowSumTopology(16, 8, True, 2, "xor"),
        element="element",
        position="position",
        lane="lane",
        extent=128,
        fast_math=env.settings.fast_math,
        target_device_capability=env.config_spec.target_device_capability,
    )
    row_body = emit_point_actions(row_actions)
    assert row_body is not None
    q_normalize = bind_register_fragment_expression(
        cg, owner.graph, qsum, qinv, ("row",), ("row",), "qsum_acc", "cutlass.Float32"
    )
    k_normalize = bind_register_fragment_expression(
        cg, owner.graph, ksum, kinv, ("row",), ("row",), "ksum_acc", "cutlass.Float32"
    )
    q_normalize.check()
    k_normalize.check()
    normalize_lines = [
        *q_normalize.expression.lines,
        *k_normalize.expression.lines,
        f"return {q_normalize.value}, {k_normalize.value}",
    ]
    rows = _source(
        "_helion_factor_row_normalization",
        "qv, kv, lane",
        [*row_body, *ast.parse("\n".join(normalize_lines)).body],
    )
    prefix, center = _arg(_arg(decay, 0), 0), _arg(_arg(decay, 0), 1)
    assert prefix is owner.region.step.gate_scan
    coefficient_inputs: dict[Node, tuple[tuple[str, ...], str]] = {
        qinv: (("row",), "qi"),
        kinv: (("row",), "ki"),
        prefix: (coords, "cutlass.Float32(prefix[element])"),
        center: ((), "center"),
        scale: ((), "scale"),
    }
    centered = _pair_source(
        "factor_centered_pair",
        "qv, kv, prefix, qi, ki, scale, center, FAST_RCP: cutlass.Constexpr",
        (
            (
                "query",
                expression(
                    qraw, qd, "cutlass.Float32(qv[element])", coefficient_inputs
                ),
            ),
            (
                "key",
                expression(
                    kraw, kd, "cutlass.Float32(kv[element])", coefficient_inputs
                ),
            ),
            (
                "inverse_key",
                expression(
                    kraw, ki, "cutlass.Float32(kv[element])", coefficient_inputs
                ),
            ),
        ),
        {"qi", "ki", "scale", "center"},
    )
    gamma = _arg(_arg(owner.scaled_state, 1), 0)
    last_prefix = _arg(gamma, 0)
    assert _arg(_arg(restore_factor, 0), 0) is last_prefix
    assert last_prefix.target is torch.ops.aten.sum.dim_IntList and last_prefix.args[
        1
    ] == [0]
    selected_prefix = _arg(last_prefix, 0)
    assert selected_prefix.target is torch.ops.aten.where.self
    assert _arg(selected_prefix, 1) is prefix
    selected_row = _arg(_arg(selected_prefix, 0), 0)
    assert selected_row.target is torch.ops.aten.eq.Scalar
    assert selected_row.args[1] == owner.region.chunk_size - 1
    gate_scale = _arg(center, 0)
    restored_coefficient = _scalar_source(
        cg,
        owner,
        "_helion_factor_restore_coefficient",
        "prefix, gate_scale_log2",
        last_prefix,
        restore_factor,
        ("column",),
        "prefix",
        {gate_scale: ((), "gate_scale_log2")},
    )
    center_coefficient = _scalar_source(
        cg,
        owner,
        "_helion_factor_center_coefficient",
        "gate_scale_log2",
        gate_scale,
        center_scale,
        (),
        "gate_scale_log2",
        {},
    )
    center_value = _scalar_source(
        cg,
        owner,
        "_helion_factor_center_value",
        "gate_scale_log2",
        gate_scale,
        center,
        (),
        "gate_scale_log2",
        {},
    )
    gamma_coefficient = _scalar_source(
        cg,
        owner,
        "_helion_factor_gamma_coefficient",
        "prefix",
        last_prefix,
        gamma,
        ("column",),
        "prefix",
        {},
    )
    masked = _arg(owner.region.step.gate_scan, 1)
    increment = _arg(masked, 1)
    activation = _arg(increment, 0)
    half_tanh = _arg(activation, 0)
    tanh = _arg(half_tanh, 0)
    half_input = _arg(tanh, 0)
    rate_product = _arg(half_input, 0)
    rate = _arg(rate_product, 0)
    rate_argument = _arg(rate, 0)
    a_log, log2_e = _arg(rate_argument, 0), _arg(rate_argument, 1)
    rate_coefficient = _scalar_source(
        cg,
        owner,
        "_helion_factor_gate_rate",
        "value, log2_e",
        a_log,
        rate,
        (),
        "value",
        {log2_e: ((), "log2_e")},
    )
    return (
        restore,
        rows,
        centered,
        restored_coefficient,
        center_coefficient,
        gamma_coefficient,
        rate_coefficient,
        center_value,
    )
