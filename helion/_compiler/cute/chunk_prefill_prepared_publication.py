"""Ordinary pointwise source for the fast factor's original typed store cuts."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch.fx import Node

from .chained_fragment_epilogue import bind_register_fragment_expression

if TYPE_CHECKING:
    from ..generate_ast import GenerateAST
    from .chunk_prefill_prepared_inverse import FastInverse
    from .chunk_prefill_prepared_pairwise import FastPairwise


def bind_fast_factor_publications(
    cg: GenerateAST, pairwise: FastPairwise, inverse: FastInverse
) -> tuple[tuple[str, str], ...]:
    """Small JIT scalar functions; the existing native loops own their calls."""
    pairwise.check()
    inverse.check()
    graph = pairwise.owner.graph
    coordinates = ("row", "column")
    lower_bf16 = pairwise.lower_value.args[0]
    assert isinstance(lower_bf16, Node)
    lower_mask = lower_bf16.args[0]
    assert isinstance(lower_mask, Node)
    multiplied = lower_mask.args[1]
    assert isinstance(multiplied, Node)
    beta_rows = multiplied.args[1]
    assert isinstance(beta_rows, Node)
    beta = beta_rows.args[0]
    assert isinstance(beta, Node)
    inverse16_source = inverse.inverse16_bf16.args[0]
    assert isinstance(inverse16_source, Node)
    cuts: tuple[
        tuple[str, Node, Node, str, dict[Node, tuple[tuple[str, ...], str]], str], ...
    ] = (
        (
            "_helion_factor_lower",
            pairwise.kk.node,
            pairwise.lower_value,
            "cutlass.Float32",
            {beta: (("row",), "beta")},
            "value, beta, row, column",
        ),
        (
            "_helion_factor_qk",
            pairwise.qk.node,
            pairwise.qk_value,
            "cutlass.BFloat16",
            {},
            "value, row, column",
        ),
        (
            "_helion_factor_inverse16",
            inverse16_source,
            inverse.inverse16_bf16,
            "cutlass.BFloat16",
            {},
            "value, row, column",
        ),
        (
            "_helion_factor_cross16",
            pairwise.lower_value,
            inverse.products[6].spec.rhs,
            "cutlass.BFloat16",
            {},
            "value, row, column",
        ),
        (
            "_helion_factor_final",
            inverse.products[7].spec.node,
            inverse.final_bf16,
            "cutlass.BFloat16",
            {},
            "value, row, column",
        ),
    )
    result = []
    for name, source, target, dtype, inputs, parameters in cuts:
        cast_node = target if dtype == "cutlass.BFloat16" else target.args[0]
        assert isinstance(cast_node, Node)
        before_cast = cast_node.args[0]
        assert isinstance(before_cast, Node)
        expression = bind_register_fragment_expression(
            cg,
            graph,
            source,
            before_cast,
            coordinates,
            coordinates,
            "value",
            "cutlass.Float32",
            inputs=inputs,
        )
        expression.check()
        converted = bind_register_fragment_expression(
            cg,
            graph,
            before_cast,
            target,
            coordinates,
            coordinates,
            expression.value
            if before_cast is source
            else f"_helion_materialize_fp32({expression.value})",
            dtype,
        )
        converted.check()
        lines = (
            "from helion._compiler.cute.affine_recurrence_primitives import materialize_fp32 as _helion_materialize_fp32",
            "@cute.jit",
            f"def {name}({parameters}):",
            *("    " + line for line in expression.expression.lines),
            *("    " + line for line in converted.expression.lines),
            f"    return {converted.value}",
        )
        result.append((name, "\n".join(lines) + "\n"))
    result.extend(_inverse_points(cg, inverse))
    return tuple(result)


def _inverse_points(
    cg: GenerateAST, inverse: FastInverse
) -> tuple[tuple[str, str], ...]:
    """The original additive diagonal states and negative typed coupling edges."""
    n1 = inverse.coupling_snapshot.args[0]
    assert isinstance(n1, Node)
    n0 = n1.args[0]
    assert isinstance(n0, Node)
    diagonal = inverse.products[0].spec.lhs.args[0]
    assert isinstance(diagonal, Node)
    selected16 = inverse.inverse16_bf16.args[0]
    assert isinstance(selected16, Node)
    n2 = selected16.args[2]
    assert isinstance(n2, Node)
    coordinates = ("row", "column")
    cuts: tuple[
        tuple[str, Node, Node, str, dict[Node, tuple[tuple[str, ...], str]], str], ...
    ] = (
        (
            "_helion_inverse_n0",
            diagonal,
            n0,
            "cutlass.Float32",
            {},
            "value, row, column",
        ),
        (
            "_helion_inverse_n1",
            n0,
            n1,
            "cutlass.Float32",
            {inverse.products[1].spec.node: (coordinates, "delta")},
            "value, delta",
        ),
        (
            "_helion_inverse_n2",
            n1,
            n2,
            "cutlass.Float32",
            {inverse.products[3].spec.node: (coordinates, "delta")},
            "value, delta",
        ),
        (
            "_helion_inverse_inner_negative",
            inverse.products[4].spec.node,
            inverse.products[5].spec.lhs,
            "cutlass.Float16",
            {},
            "value, row, column",
        ),
        (
            "_helion_inverse_outer_negative",
            inverse.products[6].spec.node,
            inverse.products[7].spec.lhs,
            "cutlass.BFloat16",
            {},
            "value, row, column",
        ),
    )
    result = []
    for name, source, target, dtype, inputs, parameters in cuts:
        before_cast = target if dtype == "cutlass.Float32" else target.args[0]
        assert isinstance(before_cast, Node)
        expression = bind_register_fragment_expression(
            cg,
            inverse.owner.graph,
            source,
            before_cast,
            coordinates,
            coordinates,
            "value",
            "cutlass.Float32",
            inputs=inputs,
        )
        expression.check()
        body = list(expression.expression.lines)
        value = expression.value
        if before_cast is not target:
            converted = bind_register_fragment_expression(
                cg,
                inverse.owner.graph,
                before_cast,
                target,
                coordinates,
                coordinates,
                f"_helion_materialize_fp32({value})",
                dtype,
            )
            converted.check()
            body.extend(converted.expression.lines)
            value = converted.value
        result.append(
            (
                name,
                "\n".join(
                    (
                        "from helion._compiler.cute.affine_recurrence_primitives import materialize_fp32 as _helion_materialize_fp32",
                        "@cute.jit",
                        f"def {name}({parameters}):",
                        *("    " + line for line in body),
                        f"    return {value}",
                        "",
                    )
                ),
            )
        )
    inverse.check()
    return tuple(result)
