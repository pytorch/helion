"""Original BT16 pointwise work at its existing packed publication cuts."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .chained_fragment_epilogue import bind_register_fragment_expression
from .chunk_prefill_prepared_bt16 import node_arg

if TYPE_CHECKING:
    from torch.fx import Node

    from ..generate_ast import GenerateAST
    from .chunk_prefill_prepared_bt16 import BT16Bindings


def bind_bt16_factor_publications(
    cg: GenerateAST, owner: BT16Bindings
) -> tuple[tuple[str, str], ...]:
    """The shared packed cast leaf consumes each original typed-edge image."""
    owner.check()
    coordinates = ("row", "column")
    lower_cast = node_arg(owner.lower_value, 0)
    lower_mask = node_arg(lower_cast, 0)
    beta = node_arg(node_arg(node_arg(lower_mask, 1), 1), 0)
    qk_mask = node_arg(owner.output_update.lhs, 0)
    final_selection = node_arg(owner.region.step.inverse, 0)
    negative = node_arg(final_selection, 1)
    cuts: tuple[
        tuple[str, Node, Node, dict[Node, tuple[tuple[str, ...], str]], str], ...
    ] = (
        (
            "_helion_bt16_lower",
            owner.kk.node,
            lower_mask,
            {beta: (("row",), "beta")},
            "value, beta, row, column",
        ),
        ("_helion_bt16_qk", owner.qk.node, qk_mask, {}, "value, row, column"),
        ("_helion_bt16_n0", owner.diagonal, owner.n0, {}, "value, row, column"),
        (
            "_helion_bt16_n1",
            owner.n0,
            owner.n1,
            {owner.products[1].spec.node: (coordinates, "delta")},
            "value, delta",
        ),
        (
            "_helion_bt16_n2",
            owner.n1,
            owner.n2,
            {owner.products[3].spec.node: (coordinates, "delta")},
            "value, delta",
        ),
        ("_helion_bt16_negative", owner.products[5].spec.node, negative, {}, "value"),
    )
    result = []
    for name, source, target, inputs, parameters in cuts:
        expression = bind_register_fragment_expression(
            cg,
            owner.graph,
            source,
            target,
            coordinates,
            coordinates,
            "value",
            "cutlass.Float32",
            inputs=inputs,
        )
        expression.check()
        lines = ["@cute.jit", f"def {name}({parameters}):"]
        lines.extend("    " + line for line in expression.expression.lines)
        lines.append(f"    return {expression.value}")
        result.append((name, "\n".join(lines) + "\n"))
    owner.check()
    return tuple(result)
