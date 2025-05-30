from __future__ import annotations

import operator
from typing import TYPE_CHECKING

import torch
from torch.fx import map_arg

from helion._compiler.inductor_lowering import APIFuncLowering
from helion._compiler.inductor_lowering import ReductionLowering
from helion._compiler.inductor_lowering import aten_lowering_dispatch
from helion._compiler.tile_strategy import TileStrategy
from helion.language._decorators import is_api_func
from helion.language._tracing_ops import _for_loop
from helion.language._tracing_ops import _get_symnode
from helion.language._tracing_ops import _host_tensor
from helion.language._tracing_ops import _if
from helion.language.memory_ops import store

if TYPE_CHECKING:
    from helion._compiler.compile_environment import BlockSizeInfo
    from helion._compiler.device_ir import DeviceIR
    from helion._compiler.device_ir import RolledReductionInfo

_duplicate_ops: tuple[object, ...] = (
    _host_tensor,
    _get_symnode,
    torch.ops.aten.sym_size.int,
)


class ReductionRoller:
    """This does the opposite of unrolling, it takes persistent reductions and turns them into looped reductions."""

    def __init__(
        self,
        device_ir: DeviceIR,
        rdim: BlockSizeInfo,
        graph_id_to_info: dict[int, RolledReductionInfo],
    ) -> None:
        self.device_ir = device_ir
        self.rdim = rdim
        self.graph_id_to_info = graph_id_to_info
        # inner graph contains ops on the reduction dimension
        self.inner_args: list[torch.fx.Node] = []
        self.inner_graph: torch.fx.Graph = torch.fx.Graph()
        self.inner_nodes: dict[torch.fx.Node, torch.fx.Node] = {}
        self.inner_count: int = 0
        self.inner_available: set[torch.fx.Node] = set()
        # outer graph contains ops that are not on the reduction dimension
        self.outer_graph: torch.fx.Graph = torch.fx.Graph()
        self.outer_nodes: dict[torch.fx.Node, torch.fx.Node] = {}
        self.outer_count: int = 0
        self.seen: set[torch.fx.Node] = set()
        self.available: set[torch.fx.Node] = set()
        self.graphs_added: list[int] = []
        self._size_node: torch.fx.Node | None = None

    def is_reduction(self, node: torch.fx.Node) -> bool:
        """Check if a node is a reduction"""
        return (
            node.op == "call_function"
            and isinstance(lowering := node.meta["lowering"], ReductionLowering)
            and lowering.block_index == self.rdim.block_size_idx
        )

    def should_go_in_inner_graph(self, node: torch.fx.Node) -> bool:
        """Nodes go in the inner graph if they use the reduction dimension"""
        if node.op in {"placeholder", "output"}:
            return False
        assert node.op == "call_function", f"Unsupported node type {node.op}"

        if node.target in (_for_loop, _if):
            if node.target is _for_loop:
                graph_id, *_ = node.args
            else:
                _, graph_id, _ = node.args
            assert isinstance(graph_id, int)
            info = self.graph_id_to_info[graph_id]
            if info.used_rdim:
                if not info.can_be_rolled_by_caller:
                    raise NotImplementedError("for loop with mixed reduction dim usage")
                return True
            return False

        if node.target in _duplicate_ops:
            return False

        if self.is_reduction(node):
            return True

        if node.target is store:
            _, _, stored_node, _ = node.args
            assert isinstance(stored_node, torch.fx.Node)
            val = stored_node.meta["val"]
        else:
            val = node.meta["val"]

        num_rdims = 0
        if isinstance(val, torch.Tensor):
            for size in val.size():
                block_idx = TileStrategy.get_block_index(size)
                num_rdims += block_idx == self.rdim.block_size_idx
            if num_rdims > 1:
                raise NotImplementedError(
                    "multiple reduction dims of same size not supported"
                )
        else:
            raise NotImplementedError(
                f"Unsupported value type {type(val)} from {node.target}"
            )

        return num_rdims > 0

    def size_node(self, meta: dict[str, object]) -> torch.fx.Node:
        """Create a node that represents the size of the reduction dimension"""
        if self._size_node is not None:
            return self._size_node
        self._size_node = node = self.outer_graph.call_function(
            _get_symnode,
            (f"rdim{self.rdim.block_size_idx}",),
            {},
        )
        node.meta.update(meta)
        node.meta["val"] = self.rdim.size
        # pyre-ignore[6]
        node.meta["lowering"] = APIFuncLowering(_get_symnode)
        return node

    def start_new_graph(self) -> None:
        if self.inner_count == 0:
            return

        inner_nodes: dict[torch.fx.Node, torch.fx.Node] = self.inner_nodes
        outputs = {}
        for orig_node, inner_node in inner_nodes.items():
            if self.is_reduction(orig_node) and orig_node not in self.outer_nodes:
                outputs[orig_node] = inner_node
            self.available.add(orig_node)
        graph = self.inner_graph
        graph.output([*outputs.values()])
        gm = torch.fx.GraphModule({}, graph)
        graph_id = self.device_ir.add_reduction_loop_graph(
            gm,
            block_index=self.rdim.block_size_idx,
        )
        self.graphs_added.append(graph_id)

        location_meta = {
            "location": next(iter(inner_nodes)).meta["location"],
            "stack_trace": next(iter(inner_nodes)).meta["stack_trace"],
        }
        output_node = self.outer_graph.call_function(
            _for_loop,
            (graph_id, [0], [self.size_node(location_meta)], self.inner_args),
            {},
        )
        output_node.meta.update(location_meta)
        output_node.meta["val"] = [n.meta["val"] for n in outputs]
        assert is_api_func(_for_loop)
        output_node.meta["lowering"] = APIFuncLowering(_for_loop)
        for i, orig_node in enumerate(outputs):
            self.outer_nodes[orig_node] = n = self.outer_graph.call_function(
                operator.getitem,
                (output_node, i),
                {},
            )
            n.meta.update(location_meta)
            n.meta["val"] = orig_node.meta["val"]
            n.meta["lowering"] = aten_lowering_dispatch[n.target](n)

        self.inner_args = []
        self.inner_graph = torch.fx.Graph()
        self.inner_nodes = {}
        self.inner_count = 0
        self.inner_available = set()

        def readd(node: torch.fx.Node) -> None:
            if (
                node not in inner_nodes
                or node in self.inner_nodes
                or self.is_reduction(node)
            ):
                return
            for n in node.all_input_nodes:
                readd(n)
            new_node = self.inner_graph.create_node(
                node.op,
                node.target,
                *map_arg((node.args, node.kwargs), self.get_inner_arg),
                name=node.name,
            )
            new_node.meta.update(node.meta)
            self.inner_nodes[node] = new_node

        # re-add any nodes that still have pending users
        for node in inner_nodes:
            if {*node.users} - self.seen:
                readd(node)

    def get_inner_arg(self, node: torch.fx.Node) -> torch.fx.Node:
        """Get the input node for the inner graph"""
        if node in self.inner_nodes:
            return self.inner_nodes[node]
        if node.target is _get_symnode:
            # this is a fake node we can duplicate in both graphs
            self.inner_nodes[node] = new_node = self.inner_graph.create_node(
                node.op,
                node.target,
                node.args,
                node.kwargs,
                name=node.name,
            )
            new_node.meta.update(node.meta)
            return new_node
        # need to create a new placeholder arg in the inner graph
        outer_node = self.outer_nodes[node]
        if outer_node.target in _duplicate_ops:
            # These fake nodes can be duplicated
            self.inner_nodes[node] = new_node = self.inner_graph.create_node(
                node.op,
                node.target,
                *map_arg((node.args, node.kwargs), self.get_inner_arg),
                name=node.name,
            )
            new_node.meta.update(node.meta)
            return new_node
        placeholders = self.inner_graph.find_nodes(op="placeholder")
        with self.inner_graph.inserting_after(
            placeholders[-1] if placeholders else self.inner_graph._root
        ):
            self.inner_nodes[node] = placeholder = self.inner_graph.placeholder(
                outer_node.name
            )
            placeholder.meta.update(node.meta)
        self.inner_args.append(outer_node)
        return placeholder

    def process(self, graph: torch.fx.Graph) -> torch.fx.Graph:
        for node in graph.nodes:
            if self.should_go_in_inner_graph(node):
                if not all(
                    (n in self.available or n in self.inner_available)
                    for n in node.all_input_nodes
                ):
                    self.start_new_graph()
                new_node = self.inner_graph.create_node(
                    node.op,
                    node.target,
                    *map_arg((node.args, node.kwargs), self.get_inner_arg),
                    name=node.name,
                )
                new_node.meta.update(node.meta)
                self.inner_nodes[node] = new_node
                self.inner_count += self.is_nontrivial(node)
                if not self.is_reduction(node):
                    self.inner_available.add(node)
            else:
                if (
                    not all((n in self.available) for n in node.all_input_nodes)
                    or node.op == "output"
                ):
                    self.start_new_graph()
                new_node = self.outer_graph.create_node(
                    node.op,
                    node.target,
                    *map_arg((node.args, node.kwargs), self.outer_nodes.__getitem__),
                    name=node.name,
                )
                new_node.meta.update(node.meta)
                self.outer_nodes[node] = new_node
                self.outer_count += self.is_nontrivial(node)
                self.available.add(node)
            self.seen.add(node)
        return self.outer_graph

    def is_nontrivial(self, node: torch.fx.Node) -> bool:
        """Check if a node should be counting in (outer|inner)_count"""
        return node.op == "call_function" and node.target not in _duplicate_ops
