"""Graph transformation that converts fully-folded grid dimensions into device loops.

When a tile dimension has ``grid_foldings = -1``, the dimension is removed from
the GPU launch grid and wrapped in an inner ``tl.range`` device loop instead.
This is implemented as an IR graph transformation (similar to
:mod:`roll_reduction`) so that the standard ``codegen_device_loop`` path handles
the loop — no duplicated offset/index/mask codegen is needed.
"""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING

import torch
from torch.fx import map_arg

from ..language._tracing_ops import _for_loop
from ..language._tracing_ops import _get_symnode
from .compile_environment import CompileEnvironment
from .device_ir import ForLoopGraphInfo
from .inductor_lowering import APIFuncLowering
from .roll_reduction import _duplicate_ops

if TYPE_CHECKING:
    from .device_ir import DeviceIR


class GridFoldingTransformer:
    """Wrap a fully-folded grid dimension into a ForLoopGraphInfo device loop.

    For a grid tile ``hl.tile([M, N])`` with ``grid_foldings = [[0, -1]]``, this
    transformer wraps the root graph body in a ``ForLoopGraphInfo`` for block_id N.
    The standard ``codegen_device_loop`` path then generates the loop code.

    All body nodes move into the inner graph; the outer graph just contains
    placeholders, a ``_for_loop`` call, and an output node.
    """

    def __init__(
        self,
        device_ir: DeviceIR,
        block_id: int,
        numel: object,
    ) -> None:
        self.device_ir = device_ir
        self.block_id = block_id
        self.numel = numel
        self.inner_graph: torch.fx.Graph = torch.fx.Graph()
        self.inner_args: list[torch.fx.Node] = []
        self.inner_nodes: dict[torch.fx.Node, torch.fx.Node] = {}
        self.outer_graph: torch.fx.Graph = torch.fx.Graph()
        self.outer_nodes: dict[torch.fx.Node, torch.fx.Node] = {}
        self._numel_node: torch.fx.Node | None = None

    def _get_inner_node(self, node: torch.fx.Node) -> torch.fx.Node:
        """Map an original graph node to its inner-graph counterpart.

        If the node is already in the inner graph, return it.  For
        ``_get_symnode`` and other duplicate-safe ops, duplicate them.
        Otherwise create a placeholder in the inner graph and record
        the corresponding outer node in ``inner_args``.
        """
        if node in self.inner_nodes:
            return self.inner_nodes[node]
        # Nodes that can safely be duplicated (e.g. _get_symnode)
        if node.target in _duplicate_ops:
            self.inner_nodes[node] = new_node = self.inner_graph.create_node(
                node.op,
                node.target,
                *map_arg((node.args, node.kwargs), self._get_inner_node),
                name=node.name,
            )
            new_node.meta.update(node.meta)
            return new_node
        # Otherwise: promote to a placeholder input of the inner graph
        outer_node = self.outer_nodes[node]
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

    def _numel_node_for(self, meta: dict[str, object]) -> torch.fx.Node:
        """Create an outer-graph node representing the folded dimension's size."""
        if self._numel_node is not None:
            return self._numel_node
        self._numel_node = node = self.outer_graph.call_function(
            _get_symnode,
            (f"grid_folding_dim{self.block_id}",),
            {},
        )
        node.meta.update(meta)
        node.meta["val"] = CompileEnvironment.current().block_sizes[self.block_id].size
        node.meta["lowering"] = APIFuncLowering(_get_symnode)
        return node

    def process(self, root_graph: torch.fx.Graph) -> torch.fx.Graph:
        """Transform *root_graph* by wrapping all body nodes in a device loop.

        Returns a new outer graph that replaces the root graph.  The inner
        graph is registered as a :class:`ForLoopGraphInfo` on the device IR.
        """
        from .aten_lowering import aten_lowering_dispatch

        location_meta: dict[str, object] | None = None

        for node in root_graph.nodes:
            if node.op == "placeholder":
                # Create in both inner and outer graphs
                outer_node = self.outer_graph.placeholder(node.name)
                outer_node.meta.update(node.meta)
                self.outer_nodes[node] = outer_node

                inner_node = self.inner_graph.placeholder(node.name)
                inner_node.meta.update(node.meta)
                self.inner_nodes[node] = inner_node
                self.inner_args.append(outer_node)

            elif node.op == "output":
                # Finalize the inner graph and create _for_loop in outer.
                # The output node's args[0] may be None (void body with
                # only side effects), a list/tuple, or a single Node.
                raw_out = node.args[0] if node.args else None
                if raw_out is None:
                    orig_outputs: list[torch.fx.Node] = []
                elif isinstance(raw_out, (list, tuple)):
                    orig_outputs = [n for n in raw_out if isinstance(n, torch.fx.Node)]
                elif isinstance(raw_out, torch.fx.Node):
                    orig_outputs = [raw_out]
                else:
                    orig_outputs = []
                inner_outputs = [self.inner_nodes[n] for n in orig_outputs]
                self.inner_graph.output(inner_outputs)

                # Register the inner graph as a ForLoopGraphInfo
                graph_id = self.device_ir.add_graph(
                    self.inner_graph,
                    ForLoopGraphInfo,
                    node_args=self.inner_args,
                    block_ids=[self.block_id],
                )

                # Create _for_loop call in the outer graph
                assert location_meta is not None
                numel_node = self._numel_node_for(location_meta)
                for_loop_node = self.outer_graph.call_function(
                    _for_loop,
                    (graph_id, [0], [numel_node], self.inner_args),
                    {},
                )
                for_loop_node.meta.update(location_meta)
                for_loop_node.meta["val"] = [n.meta["val"] for n in inner_outputs]
                for_loop_node.meta["lowering"] = APIFuncLowering(_for_loop)

                # Extract individual outputs via getitem
                outer_outputs: list[torch.fx.Node] = []
                for i, orig_out in enumerate(orig_outputs):
                    getitem_node = self.outer_graph.call_function(
                        operator.getitem,
                        (for_loop_node, i),
                        {},
                    )
                    getitem_node.meta.update(location_meta)
                    getitem_node.meta["val"] = orig_out.meta["val"]
                    getitem_node.meta["lowering"] = aten_lowering_dispatch[
                        getitem_node.target
                    ](getitem_node)
                    self.outer_nodes[orig_out] = getitem_node
                    outer_outputs.append(getitem_node)

                self.outer_graph.output(outer_outputs)

            else:
                # Body node — goes to inner graph only
                if location_meta is None:
                    location_meta = {
                        "location": node.meta.get("location", ""),
                        "stack_trace": node.meta.get("stack_trace", ""),
                    }
                new_node = self.inner_graph.create_node(
                    node.op,
                    node.target,
                    *map_arg((node.args, node.kwargs), self._get_inner_node),
                    name=node.name,
                )
                new_node.meta.update(node.meta)
                self.inner_nodes[node] = new_node

        return self.outer_graph
