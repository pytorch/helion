from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch
from torch.fx import Graph

from helion._compiler.cute.cute_mma import _decode_cute_mma_target
from helion._compiler.cute.cute_mma import _is_zero_init_acc_node
from helion._compiler.device_ir import DeviceIR
from helion._compiler.device_ir import ForLoopGraphInfo
from helion._compiler.device_ir import RootGraphInfo
from helion.language import _tracing_ops
from helion.language import creation_ops

if TYPE_CHECKING:
    from torch.fx import Node

    from helion._compiler.device_ir import GraphInfo


def _loop_graph(seed: Node, graph_id: int) -> ForLoopGraphInfo:
    body = Graph()
    acc = body.placeholder("acc")
    acc.meta["val"] = torch.empty(4, 4)
    carried = body.call_function(_tracing_ops._new_var, args=(acc,))
    carried.meta["val"] = torch.empty(4, 4)
    body.output((carried,))
    return ForLoopGraphInfo(graph_id, body, node_args=[seed], block_ids=[graph_id])


def _graphs() -> list[GraphInfo]:
    root = Graph()
    zero = root.call_function(creation_ops.full, args=([4, 4], 0, torch.float32))
    zero.meta["val"] = torch.empty(4, 4)
    bias = root.placeholder("bias")
    bias.meta["val"] = torch.empty(4, 4)
    root.output((zero, bias))
    return [_loop_graph(zero, 0), _loop_graph(bias, 1), RootGraphInfo(2, root)]


def _acc(graph: GraphInfo) -> Node:
    return next(iter(graph.graph.find_nodes(op="placeholder")))


@pytest.mark.parametrize("index, expected", [(0, True), (1, False)])
def test_copied_accumulator_uses_exact_active_graph(index: int, expected: bool) -> None:
    ir = DeviceIR()
    ir.graphs = _graphs()
    copies = [graph.copy() for graph in ir.graphs]
    acc = _acc(copies[index])
    assert not _is_zero_init_acc_node(acc, device_ir=ir)
    assert _is_zero_init_acc_node(acc, graphs=copies) is expected


@pytest.mark.parametrize("included", [[], [0], [0, 1]])
def test_explicit_graphs_never_substitute_a_structural_match(
    included: list[int],
) -> None:
    graphs = _graphs()
    copied_zero = _acc(graphs[0].copy())
    assert not _is_zero_init_acc_node(
        copied_zero, graphs=[graphs[index] for index in included]
    )


def test_explicit_graphs_preserve_extra_loop_carry_rejection() -> None:
    graphs = _graphs()
    graph = graphs[0]
    assert isinstance(graph, ForLoopGraphInfo)
    acc = _acc(graph)
    output = next(iter(graph.graph.find_nodes(op="output")))
    with graph.graph.inserting_before(output):
        other = graph.graph.call_function(_tracing_ops._new_var, args=(acc,))
    other.meta["val"] = torch.empty(4, 4)
    output.args = ((*output.args[0], other),)
    assert not _is_zero_init_acc_node(acc, graphs=graphs)


@pytest.mark.parametrize("index, expected_seed", [(0, False), (1, True)])
def test_native_target_decoding_uses_exact_active_graph(
    index: int, expected_seed: bool
) -> None:
    graphs = [graph.copy() for graph in _graphs()]
    loop = graphs[index]
    assert isinstance(loop, ForLoopGraphInfo)
    loop.node_args.extend([loop.node_args[0], loop.node_args[0]])
    graph = graphs[index].graph
    acc = _acc(graphs[index])
    output = next(iter(graph.find_nodes(op="output")))
    with graph.inserting_before(output):
        lhs = graph.placeholder("lhs")
        rhs = graph.placeholder("rhs")
        mma = graph.call_function(torch.ops.aten.addmm.default, args=(acc, lhs, rhs))
    mma.meta["val"] = torch.empty(4, 4)
    output.args = ((mma,),)
    decoded = _decode_cute_mma_target(mma, graphs=graphs)
    assert decoded is not None
    assert decoded.requires_accumulator_seed is expected_seed
