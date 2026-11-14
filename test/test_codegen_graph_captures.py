"""Codegen graph copies retain captures from their own graph collection."""

from __future__ import annotations

import operator

import pytest
from torch.fx import Graph

from helion import Config
from helion._compiler.device_ir import DeviceIR
from helion._compiler.device_ir import ElseGraphInfo
from helion._compiler.device_ir import ForLoopGraphInfo
from helion._compiler.device_ir import IfGraphInfo
from helion._compiler.device_ir import NodeArgsGraphInfo
from helion._compiler.device_ir import RootGraphInfo
from helion._compiler.device_ir import WhileConditionGraphInfo
from helion._compiler.device_ir import WhileLoopGraphInfo
from helion._compiler.device_ir import control_flow_parent_entries
from helion.language import _tracing_ops

pytestmark = pytest.mark.cpu_only


def _nested_graphs():
    root, loop, yes, no, condition, body = (Graph() for _ in range(6))
    argument = root.placeholder("argument")
    initial = root.call_function(operator.add, (argument, 3))
    loop_input = loop.placeholder("loop_input")
    branch_value = loop.call_function(operator.mul, (loop_input, 2))
    yes_input = yes.placeholder("yes_input")
    no_input = no.placeholder("no_input")
    condition_input = condition.placeholder("condition_input")
    body_input = body.placeholder("body_input")
    condition.output((condition.call_function(operator.gt, (condition_input, 0)),))
    body.output((body.call_function(operator.sub, (body_input, 1)),))
    yes.output((yes.call_function(_tracing_ops._while_loop, (4, 5, [yes_input])),))
    no.output((no_input,))
    loop.output(
        (
            loop.call_function(
                _tracing_ops._if, (True, 2, 3, [branch_value], [branch_value])
            ),
        )
    )
    root.output((root.call_function(_tracing_ops._for_loop, (1, [0], [2], [initial])),))
    other = ElseGraphInfo(3, no, [branch_value])
    ir = DeviceIR()
    ir.graphs = [
        RootGraphInfo(0, root),
        ForLoopGraphInfo(1, loop, [initial], [0]),
        IfGraphInfo(2, yes, [branch_value], else_branch=3),
        other,
        WhileConditionGraphInfo(4, condition, [yes_input]),
        WhileLoopGraphInfo(5, body, [yes_input], 4),
    ]
    ir.root_ids = [0]
    return ir


@pytest.mark.parametrize("aliases_first", [False, True])
def test_codegen_captures_ignore_unreachable_analysis_parents(aliases_first):
    ir = _nested_graphs()
    aliases = [ir.graphs[index].copy() for index in (1, 2)]
    for graph_id, info in enumerate(aliases, len(ir.graphs)):
        info.graph_id = graph_id
    ir.graphs = aliases + ir.graphs if aliases_first else ir.graphs + aliases
    original_args = {
        info.graph_id: tuple(info.node_args)
        for info in ir.graphs
        if isinstance(info, NodeArgsGraphInfo)
    }
    copied = {
        info.graph_id: info
        for info in ir.build_codegen_graphs(Config(), roll_reductions=False)
    }
    # Analysis copies still call the same children, but only the original
    # parents are reachable from root 0. Cover both branch and while slots.
    for child, parent in ((2, 1), (3, 1), (4, 2), (5, 2)):
        assert copied[child].node_args[0].graph is copied[parent].graph
    for info in ir.graphs:
        if isinstance(info, NodeArgsGraphInfo):
            assert tuple(info.node_args) == original_args[info.graph_id]


@pytest.mark.parametrize("children_first", [False, True])
@pytest.mark.parametrize("removed_capture", [False, True])
def test_codegen_captures_and_branch_links_belong_to_the_copy(
    children_first, removed_capture
):
    ir = _nested_graphs()
    if removed_capture:
        root = ir.graphs[0].graph
        stale = ir.graphs[1].node_args[0]
        with root.inserting_before(stale):
            current = root.call_function(operator.add, (stale.args[0], 7))
        stale.replace_all_uses_with(current)
        root.erase_node(stale)
        assert ir.graphs[1].node_args[0] is stale
    if children_first:
        ir.graphs.reverse()
    original_text = [str(info.graph) for info in ir.graphs]
    original_args = {
        info.graph_id: tuple(info.node_args)
        for info in ir.graphs
        if isinstance(info, NodeArgsGraphInfo)
    }
    original_nodes = {node for info in ir.graphs for node in info.graph.nodes}
    actual_args = {
        graph_id: tuple(parent.args[slot])
        for graph_id, (parent, slot) in control_flow_parent_entries(ir.graphs).items()
    }
    # The origin-analysis helper excludes mutable while state. Its two
    # graphs nevertheless start from these call arguments, not stale nodes.
    while_call = next(
        node
        for info in ir.graphs
        for node in info.graph.nodes
        if node.target is _tracing_ops._while_loop
    )
    actual_args[4] = actual_args[5] = tuple(while_call.args[2])
    first = ir.build_codegen_graphs(Config(), roll_reductions=False)
    second = ir.build_codegen_graphs(Config(), roll_reductions=False)
    for copied in (first, second):
        node_map = {
            old: new
            for source, destination in zip(ir.graphs, copied, strict=True)
            for old, new in zip(
                source.graph.nodes, destination.graph.nodes, strict=True
            )
        }
        assert original_nodes.isdisjoint(node_map.values())
        copied_by_id = {info.graph_id: info for info in copied}
        for info in copied:
            if isinstance(info, NodeArgsGraphInfo):
                assert info.node_args == [
                    node_map[node] for node in actual_args[info.graph_id]
                ]
                for placeholder, source in zip(
                    info.graph.find_nodes(op="placeholder"), info.node_args, strict=True
                ):
                    assert info.placeholder_to_outer_arg(placeholder) is source
            if isinstance(info, IfGraphInfo):
                assert info.else_branch == 3
        # Modifying one copied producer is observed through its copied capture.
        producer = copied_by_id[1].node_args[0]
        producer.args = (producer.args[0], 19)
        assert (
            copied_by_id[1]
            .placeholder_to_outer_arg(next(iter(copied_by_id[1].graph.nodes)))
            .args[1]
            == 19
        )
    first_nodes = {node for info in first for node in info.graph.nodes}
    assert first_nodes.isdisjoint(node for info in second for node in info.graph.nodes)
    assert [str(info.graph) for info in ir.graphs] == original_text
    for info in ir.graphs:
        if isinstance(info, NodeArgsGraphInfo):
            assert tuple(info.node_args) == original_args[info.graph_id]
        if isinstance(info, IfGraphInfo):
            assert info.else_branch == 3
