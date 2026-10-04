"""Original BT16 normalization and decay images on the existing lane geometry."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

import torch

from ..compile_environment import CompileEnvironment
from ..inductor_lowering import PointwiseCodegenPolicy
from . import chained_matmul as chain
from .chained_fragment_epilogue import bind_register_fragment_expression
from .chunk_prefill_prepared_bt16 import node_arg
from .chunk_prefill_prepared_factor_inputs import _pair_source
from .chunk_prefill_prepared_factor_inputs import _source
from .producer_phase import RowSumPoint
from .producer_phase import RowSumTopology
from .producer_phase import emit_point_actions
from .producer_phase import row_sum_actions

if TYPE_CHECKING:
    from torch.fx import Node

    from ..generate_ast import GenerateAST
    from .chained_fragment_epilogue import FragmentExpression
    from .chunk_prefill_prepared_bt16 import BT16Bindings


def bind_bt16_inputs(
    cg: GenerateAST, owner: BT16Bindings
) -> tuple[tuple[str, str], ...]:
    owner.check()
    coords = ("row", "column")
    qd, kd, kr = owner.query.lhs, owner.projection.lhs, owner.state_update.rhs
    ki = node_arg(owner.kk.rhs, 0)
    if any(
        node.target is not torch.ops.prims.convert_element_type.default
        or node.args[1] is not torch.bfloat16
        for node in (qd, kd, ki, kr)
    ):
        raise chain._UnsupportedChain("changed original BT16 input publication dtype")
    q_product, k_product = node_arg(qd, 0), node_arg(kd, 0)
    qnorm, knorm = node_arg(q_product, 0), node_arg(k_product, 0)
    decay = node_arg(k_product, 1)
    qraw, kraw = node_arg(qnorm, 0), node_arg(knorm, 0)
    qinv, kinv = node_arg(node_arg(qnorm, 1), 0), node_arg(node_arg(knorm, 1), 0)
    qsum, ksum = node_arg(node_arg(qinv, 0), 0), node_arg(node_arg(kinv, 0), 0)
    inverse_product = node_arg(ki, 0)
    reciprocal = node_arg(inverse_product, 1)
    restore_product = node_arg(kr, 0)
    gamma = node_arg(node_arg(restore_product, 1), 0)
    if (
        qsum.target is not torch.ops.aten.sum.dim_IntList
        or qsum.args[1] != [-1]
        or ksum.target is not torch.ops.aten.sum.dim_IntList
        or ksum.args[1] != [-1]
        or node_arg(q_product, 1) is not decay
        or node_arg(restore_product, 0) is not inverse_product
    ):
        raise chain._UnsupportedChain(
            "changed original BT16 normalization or restore cuts"
        )

    def expression(
        source: Node,
        target: Node,
        fragment: str,
        inputs: dict[Node, tuple[tuple[str, ...], str]],
        coordinates: tuple[str, ...] = coords,
    ) -> FragmentExpression:
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
            policy=PointwiseCodegenPolicy(
                reciprocal_ftz="True", minmax_ftz=True, minmax_propagate_nan=False
            ),
        )
        value.check()
        return value

    qpoint = expression(qraw, node_arg(qsum, 0), "cutlass.Float32(qv[element])", {})
    kpoint = expression(kraw, node_arg(ksum, 0), "cutlass.Float32(kv[element])", {})
    env = CompileEnvironment.current()
    actions = row_sum_actions(
        (
            RowSumPoint("qsum", tuple(qpoint.expression.lines), qpoint.value),
            RowSumPoint("ksum", tuple(kpoint.expression.lines), kpoint.value),
        ),
        RowSumTopology(lanes=8, elements_per_lane=16, lane_block=8, shuffle="xor"),
        element="element",
        position="column",
        lane="lane",
        extent=128,
        fast_math=env.settings.fast_math,
        target_device_capability=env.config_spec.target_device_capability,
    )
    body = emit_point_actions(actions)
    assert body is not None
    qnormalize = expression(qsum, qinv, "qsum_acc", {}, ("row",))
    knormalize = expression(ksum, kinv, "ksum_acc", {}, ("row",))
    tail = [
        *qnormalize.expression.lines,
        *knormalize.expression.lines,
        f"return {qnormalize.value}, {knormalize.value}",
    ]
    normalize = _source(
        "_helion_bt16_normalize",
        "qv, kv, lane",
        [*body, *ast.parse("\n".join(tail)).body],
    )
    inverse = expression(decay, reciprocal, "value", {})
    inverse_source = _source(
        "_helion_bt16_reciprocal",
        "value",
        ast.parse(
            "\n".join((*inverse.expression.lines, f"return {inverse.value}"))
        ).body,
    )
    inputs: dict[Node, tuple[tuple[str, ...], str]] = {
        kinv: (("row",), "ki"),
        decay: (coords, "cutlass.Float32(decay[element])"),
        reciprocal: (coords, "cutlass.Float32(inverse_decay[element])"),
        gamma: (("column",), "cutlass.Float32(last_decay[element])"),
    }
    key = _pair_source(
        "bt16_key_pair",
        "kv, decay, inverse_decay, last_decay, ki",
        (
            (
                "decayed",
                expression(kraw, k_product, "cutlass.Float32(kv[element])", inputs),
            ),
            (
                "inverse",
                expression(
                    kraw, inverse_product, "cutlass.Float32(kv[element])", inputs
                ),
            ),
            (
                "restored",
                expression(
                    kraw, restore_product, "cutlass.Float32(kv[element])", inputs
                ),
            ),
        ),
        {"ki"},
    )
    query = _pair_source(
        "bt16_query_pair",
        "qv, decay, qi",
        (
            (
                "decayed",
                expression(
                    qraw,
                    q_product,
                    "cutlass.Float32(qv[element])",
                    {
                        qinv: (("row",), "qi"),
                        decay: (coords, "cutlass.Float32(decay[element])"),
                    },
                ),
            ),
        ),
        {"qi"},
    )
    owner.check()
    return normalize, inverse_source, key, query
