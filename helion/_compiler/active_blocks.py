"""Logical axes still referenced by a set of lowered device graphs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import sympy
import torch

from ..language.matmul_ops import MATMUL_FACT_ID_META
from .compile_environment import CompileEnvironment
from .device_ir import ForLoopGraphInfo
from .inductor_lowering import ReductionLowering
from .loop_dependency_checker import memory_write_target
from .utils import compute_slice_size

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


def unused_block_ids(
    graphs: Sequence[GraphInfo],
    block_ids: set[int],
    *,
    used_block_ids: frozenset[int] | None = None,
) -> set[int]:
    """The ``block_ids`` nothing in ``graphs`` uses."""
    env = CompileEnvironment.current()
    if used_block_ids is None:
        used_block_ids = active_block_ids(graphs, (), env)
    unused = block_ids - used_block_ids
    if not unused:
        return unused
    for graph in graphs:
        for node in graph.graph.nodes:
            # Scalar stores have no tensor result carrying full-slice axes.
            # Match their logical extents directly: compute_shape can allocate
            # new reduction dimensions, which this liveness analysis must not do.
            if (target := memory_write_target(node)) is not None and isinstance(
                node.args[1], (list, tuple)
            ):
                tensor = torch.fx.map_arg(target, lambda n: n.meta["val"])
                if isinstance(tensor, tuple):
                    tensor = tensor[0]
                assert isinstance(tensor, torch.Tensor)
                indices = torch.fx.map_arg(node.args[1], lambda n: n.meta["val"])
                indices = [index for index in indices if index is not None]
                for size, index in zip(tensor.shape, indices, strict=True):
                    if not isinstance(index, slice):
                        continue
                    extent = compute_slice_size(index, size)
                    # ShapeEnv can know equal extents whose expression keys differ.
                    used = set()
                    for block_id in unused:
                        logical = env.block_sizes[block_id].size
                        if isinstance(logical, (int, torch.SymInt)) and env.known_equal(
                            logical, extent
                        ):
                            used.add(block_id)
                    unused.difference_update(used)
                    if not unused:
                        return unused
    return unused
