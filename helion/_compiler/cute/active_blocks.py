"""Logical axes still referenced by a set of lowered CuTe device graphs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import sympy
import torch

from ...language.matmul_ops import MATMUL_FACT_ID_META
from ..device_ir import ForLoopGraphInfo
from ..inductor_lowering import ReductionLowering

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Sequence

    from ..compile_environment import CompileEnvironment
    from ..device_ir import GraphInfo


def active_block_ids(
    graphs: Sequence[GraphInfo], root_blocks: Iterable[int], env: CompileEnvironment
) -> frozenset[int]:
    """Keep shape/index symbols, explicit loops and actual reduction lowerings."""
    result = set(root_blocks)

    def visit(value: object) -> None:
        if isinstance(value, torch.Tensor):
            visit(tuple(value.shape))
            visit(value.stride())
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)
        elif isinstance(value, torch.SymInt):
            visit(value._sympy_())
        elif isinstance(value, sympy.Expr):
            for symbol in value.free_symbols:
                block_id = env.get_block_id(symbol)
                if block_id is not None:
                    result.add(block_id)

    for graph in graphs:
        if isinstance(graph, ForLoopGraphInfo):
            result.update(graph.block_ids)
        for node in graph.graph.nodes:
            visit(node.meta.get("val"))
            lowering = node.meta.get("lowering")
            if isinstance(lowering, ReductionLowering):
                result.add(lowering.block_index)
            fact_id = node.meta.get(MATMUL_FACT_ID_META)
            if isinstance(fact_id, int):
                fact = env.config_spec.matmul_facts[fact_id]
                result.update(
                    block_id
                    for block_id in (fact.m_block_id, fact.n_block_id, fact.k_block_id)
                    if block_id is not None
                )
    return frozenset(result)
