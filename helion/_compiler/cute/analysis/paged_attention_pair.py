"""Prove that two adjacent leading items share the attention iteration range."""

from __future__ import annotations

from typing import TYPE_CHECKING

import sympy
import torch
from torch.utils._sympy.functions import FloorDiv

from ...host_function import HostFunction
from ...variable_origin import TileBeginOrigin
from ...variable_origin import TileIdOrigin

if TYPE_CHECKING:
    from torch.fx import Node

    from ...compile_environment import CompileEnvironment
    from ..cute_mma_dag import Access
    from ..cute_mma_dag import Axis
    from ..paged_attention_transport import PagedAttentionTransport


def paired_attention_item_axis(
    env: CompileEnvironment, transport: PagedAttentionTransport
) -> Axis | None:
    """An even head pair with identical typed inputs to both loop endpoints.

    The transport proof already establishes request=item//heads and the
    corresponding page-table, length and physical-head accesses. This proof
    additionally excludes per-head bounds. It never reconstructs arithmetic:
    equal inputs to the admitted uniform DAG imply equal results, including
    each source INT32 wrap before promotion to an INT64 endpoint.
    """
    semantic = transport.semantic
    program = semantic.program
    if (
        not semantic.denominator_per_key
        or transport.heads % 2
        or transport.items % 2
        or len(program.lead_axes) != 1
        or env.config_spec.cute_mma_dag_phase_index is not None
    ):
        return None
    lead = program.lead_axes[0]
    item = transport.item
    info = HostFunction.current().expr_to_origin.get(item)
    if (
        lead.extent != transport.items
        or info is None
        or not isinstance(info.origin, (TileBeginOrigin, TileIdOrigin))
        or env.canonical_block_id(info.origin.block_id) != lead.block_id
    ):
        return None

    loads = {program.resolve(load.node): load for load in program.scalar_loads}
    uniform = {program.resolve(node) for node in program.uniform_values}
    memo: dict[Node, bool] = {}

    def access_invariant(access: Access) -> bool:
        return not access.index_map and all(
            subscript.kind == "scalar"
            and subscript.offset is None
            and subscript.window is None
            and expression_invariant(subscript.expr)
            for subscript in access.subscripts
        )

    def expression_invariant(expression: object) -> bool:
        if isinstance(expression, (int, sympy.Integer)):
            return True
        if isinstance(expression, torch.SymInt):
            expression = expression._sympy_()
        if isinstance(expression, sympy.Symbol):
            load = program.index_loads.get(expression)
            return load is not None and access_invariant(load)
        if not isinstance(expression, sympy.Expr):
            return False
        if isinstance(expression, FloorDiv) and expression.args[0] == item:
            divisor = expression.args[1]
            # (2*p)//d == (2*p+1)//d for every positive even d.
            return (
                isinstance(divisor, sympy.Integer) and divisor > 0 and divisor % 2 == 0
            )
        return bool(expression.args) and all(
            expression_invariant(argument) for argument in expression.args
        )

    def value_invariant(value: object) -> bool:
        if not isinstance(value, torch.fx.Node):
            return expression_invariant(value)
        node = program.resolve(value)
        if node in memo:
            return memo[node]
        memo[node] = False  # Any cycle requires a separate recurrence proof.
        if node in loads:
            result = access_invariant(loads[node])
        elif isinstance(node.meta.get("val"), (torch.SymInt, int)):
            result = expression_invariant(node.meta["val"])
        else:
            result = node in uniform and all(
                value_invariant(argument) for argument in node.all_input_nodes
            )
        memo[node] = result
        return result

    if not all(
        value_invariant(bound) for bound in (semantic.loop.begin, semantic.loop.end)
    ):
        return None
    return lead
