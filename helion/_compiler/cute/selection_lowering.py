"""Lower tensor selection programs through the register-fragment compiler."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch.fx import Graph
from torch.fx import GraphModule
from torch.fx import Node

from ..source_location import SourceLocation
from .row_fragment import RowFragment
from .row_fragment import RowFragmentLayout

if TYPE_CHECKING:
    from types import CodeType

    import torch

    from ..generate_ast import GenerateAST


def _clone_program(traced: GraphModule, code: CodeType) -> GraphModule:
    """Keep cached traces free of configuration-dependent lowering metadata."""
    graph = Graph()
    copied: dict[Node, Node] = {}
    location = SourceLocation(
        code.co_firstlineno,
        0,
        code.co_firstlineno,
        0,
        code.co_name,
        code.co_filename,
    )
    for node in traced.graph.nodes:
        copied[node] = graph.node_copy(node, copied.__getitem__)
        copied[node].meta["location"] = location
    program = GraphModule(traced, graph)
    for name, child in tuple(program.named_children()):
        if isinstance(child, GraphModule):
            program.add_module(name, _clone_program(child, code))
    return program


def _emit_program(
    cg: GenerateAST,
    program: GraphModule,
    keys: str,
    dtype: torch.dtype,
    groups: int,
    registers: int,
    lane: str,
) -> RowFragment:
    from .register_program import emit_register_program

    layout = RowFragmentLayout(groups, registers, lane)
    inputs = (RowFragment(keys, dtype, groups * registers, layout),)
    (selected,) = emit_register_program(
        cg, program, inputs, lanes=groups, lane_expr=lane
    )
    return selected


def emit_selection_network(
    cg: GenerateAST,
    *,
    keys: str,
    dtype: torch.dtype,
    groups: int,
    registers: int,
    k: int,
    lane: str,
    mode: str = "distributed",
    sort_network: str = "batcher",
    merge_schedule: str = "sequential",
) -> RowFragment:
    """Bind logical candidate groups to the current subgroup's registers.

    The tensor program describes independent candidate groups, then merges
    them. Its group/register coordinates are independent of the input memory
    layout: callers have already encoded the original column in each key.
    The register compiler is responsible for every inter-thread exchange.
    """
    from .selection_network import selection_network
    from .selection_network import trace_selection_network

    traced = trace_selection_network(
        groups, registers, k, dtype, sort_network, merge_schedule, mode
    )
    return _emit_program(
        cg,
        _clone_program(traced, selection_network.__code__),
        keys,
        dtype,
        groups,
        registers,
        lane,
    )


def emit_coarse_selection_network(
    cg: GenerateAST,
    *,
    keys: str,
    dtype: torch.dtype,
    registers: int,
    k: int,
    warp_lane: str,
    groups_per_result: int,
    index_bits: int,
    vector_width: int,
    sort_network: str,
    merge_schedule: str,
    recovery: str,
    payload_only: bool,
) -> RowFragment:
    """Lower guarded selection with one convergent predicate per warp.

    The tensor algorithm's first axis covers all 32 logical candidate groups.
    Its subgroups select independent rows, while its scalar condition reduces
    all row guards so both branch programs have full-warp participation.
    """
    from .selection_coarse import coarse_rank_selection
    from .selection_coarse import trace_coarse_rank_selection

    traced = trace_coarse_rank_selection(
        32,
        registers,
        k,
        dtype,
        sort_network,
        merge_schedule,
        index_bits,
        vector_width,
        groups_per_result,
        recovery,
        payload_only,
    )
    return _emit_program(
        cg,
        _clone_program(traced, coarse_rank_selection.__code__),
        keys,
        dtype,
        32,
        registers,
        warp_lane,
    )
