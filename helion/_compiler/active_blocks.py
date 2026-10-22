"""Logical axes still referenced by a set of lowered device graphs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import sympy
import torch

from ..language import _MEMORY_OPS
from ..language.matmul_ops import MATMUL_FACT_ID_META
from ..language.memory_ops import load
from .compile_environment import CompileEnvironment
from .device_ir import ForLoopGraphInfo
from .indexing_strategy import SubscriptIndexing
from .inductor_lowering import ReductionLowering

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Sequence

    from .device_ir import GraphInfo


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


def unused_block_ids(graphs: Sequence[GraphInfo], block_ids: set[int]) -> set[int]:
    """The ``block_ids`` nothing in ``graphs`` uses."""
    env = CompileEnvironment.current()
    unused = block_ids - active_block_ids(graphs, (), env)
    for graph in graphs:
        for node in graph.graph.nodes:
            # stores and atomics use rdims through their subscripts (a load's
            # val has them), which were shaped when traced, so this
            # allocates nothing
            if (
                node.target in _MEMORY_OPS
                and node.target is not load
                and isinstance(node.args[1], (list, tuple))
            ):
                target = node.args[0]
                if isinstance(target, (list, tuple)):
                    target = target[0]  # StackTensor's shape/stride template
                assert isinstance(target, torch.fx.Node)
                tensor = target.meta["val"]
                assert isinstance(tensor, torch.Tensor)
                index = torch.fx.map_arg(node.args[1], lambda n: n.meta["val"])
                shape = SubscriptIndexing.compute_shape(tensor, [*index])
                unused.difference_update(map(env.get_block_id, shape))
    return unused
