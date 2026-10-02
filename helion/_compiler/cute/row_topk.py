"""Prove row-wise register selection surrounded by ordinary graph operations."""

from __future__ import annotations

from dataclasses import dataclass
import operator
from typing import TYPE_CHECKING

import torch
from torch.fx import Node

from ...language._tracing_ops import _for_loop
from ...language._tracing_ops import _get_symnode
from ...language._tracing_ops import _host_tensor
from ...language.memory_ops import load
from ...language.memory_ops import store
from ..compile_environment import CompileEnvironment
from ..device_ir import ReductionLoopGraphInfo
from ..device_ir import RootGraphInfo
from .ordered_selection import is_ordered_selection
from .ordered_selection import selection_args
from .row_fragment import row_fragment_logical_extent
from .row_fragment import supports_row_fragment_node

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..device_ir import GraphInfo


@dataclass(frozen=True)
class RowTopKGraph:
    root_graph: torch.fx.Graph
    selection: Node
    source: Node
    row_block_id: int
    x: torch.Tensor
    n: int
    k: int
    largest: bool
    loads: tuple[Node, ...]
    stores: tuple[Node, ...]
    aliases: dict[Node, Node]
    read_tensors: tuple[torch.Tensor, ...]
    write_tensors: tuple[torch.Tensor, ...]
    stable_ties: bool = False

    @property
    def tensors(self) -> tuple[torch.Tensor, ...]:
        return (*self.read_tensors, *self.write_tensors)


def _tensor(node: object) -> torch.Tensor | None:
    if not isinstance(node, Node):
        return None
    value = node.meta.get("val")
    return value if isinstance(value, torch.Tensor) else None


def _host_tensor_value(node: object) -> torch.Tensor | None:
    return (
        _tensor(node)
        if isinstance(node, Node) and node.target is _host_tensor
        else None
    )


def _row_block(node: object) -> int | None:
    if not isinstance(node, Node) or node.target is not _get_symnode:
        return None
    value = node.meta.get("val")
    if not isinstance(value, (int, torch.SymInt)):
        return None
    return CompileEnvironment.current().get_block_id(value)


def _extent(value: object) -> int | None:
    if isinstance(value, Node):
        value = value.meta.get("val")
    if not isinstance(value, (int, torch.SymInt)):
        return None
    return row_fragment_logical_extent(value)


def _static_view_shape_query(node: Node) -> bool:
    """Admit static tensor sizes only as direct view-shape operands."""
    if (
        node.target is not torch.ops.aten.sym_size.int
        or len(node.args) != 2
        or node.kwargs
    ):
        return False
    source, dim = node.args
    tensor = _tensor(source)
    size = node.meta.get("val")
    if (
        tensor is None
        or type(dim) is not int
        or not -tensor.ndim <= dim < tensor.ndim
        or not isinstance(size, (int, torch.SymInt))
        or _extent(size) is None
        or not CompileEnvironment.current().known_equal(size, tensor.size(dim))
        or not node.users
    ):
        return False
    for user in node.users:
        if (
            user.target
            not in (
                torch.ops.aten.expand.default,
                torch.ops.aten.view.default,
                torch.ops.aten.reshape.default,
                torch.ops.aten._unsafe_view.default,
            )
            or len(user.args) != 2
            or not isinstance(user.args[1], (tuple, list))
            or node not in user.args[1]
            or user.args[0] is node
            or any(isinstance(value, Node) for value in user.kwargs.values())
        ):
            return False
    return True


def _unique_tensors(nodes: Sequence[Node]) -> tuple[torch.Tensor, ...]:
    tensors: dict[int, torch.Tensor] = {}
    for node in nodes:
        tensor = _host_tensor_value(node.args[0])
        assert tensor is not None
        tensors[id(tensor)] = tensor
    return tuple(tensors.values())


def match_row_topk(
    graphs: Sequence[GraphInfo], *, noncanonical_block_ids: set[int]
) -> RowTopKGraph | None:
    """Accept one complete-row selection and a pure DAG with row-local stores.

    Full-extent reduction carriers are aliases for their child graph outputs;
    they do not introduce another execution domain. Input reductions remain
    replicated row statistics and can be reused when selected values need
    exact recovery at their original index.
    Runtime storage disjointness remains a separate cache-specialized proof.
    """
    roots = {id(info.graph): info for info in graphs if isinstance(info, RootGraphInfo)}
    if len(roots) != 1:
        return None
    root = next(iter(roots.values()))
    root_nodes = list(root.graph.nodes)
    if (
        not root_nodes
        or root_nodes[-1].op != "output"
        or root_nodes[-1].args != (None,)
    ):
        return None
    # Unrelated heuristic facts may be registered without an active environment.
    if not any(
        is_ordered_selection(node) for info in graphs for node in info.graph.nodes
    ):
        return None
    env = CompileEnvironment.current()
    infos = {info.graph_id: info for info in graphs}
    aliases: dict[Node, Node] = {}
    nodes: list[Node] = []
    carrier_widths: list[int] = []
    active_graphs: set[int] = set()

    def collect(graph: torch.fx.Graph) -> bool:
        if id(graph) in active_graphs:
            return False
        active_graphs.add(id(graph))
        for node in graph.nodes:
            if node.op == "output" or node in aliases:
                continue
            if node.target is not _for_loop:
                nodes.append(node)
                continue
            if len(node.args) != 4 or node.kwargs:
                return False
            graph_id, begin, end, arguments = node.args
            if not isinstance(graph_id, int):
                return False
            child = infos.get(graph_id)
            if (
                not isinstance(child, ReductionLoopGraphInfo)
                or len(child.block_ids) != 1
                or begin != [0]
                or not isinstance(end, (tuple, list))
                or len(end) != 1
                or not isinstance(arguments, (tuple, list))
            ):
                return False
            block = env.block_sizes[child.block_ids[0]]
            width = _extent(end[0])
            if not block.reduction or width is None or width != _extent(block.size):
                return False
            carrier_widths.append(width)
            placeholders = list(child.graph.find_nodes(op="placeholder"))
            if len(placeholders) != len(arguments) or not all(
                isinstance(argument, Node) for argument in arguments
            ):
                return False
            for placeholder, argument in zip(placeholders, arguments, strict=True):
                assert isinstance(argument, Node)
                aliases[placeholder] = argument
            output = next(reversed(child.graph.nodes))
            if (
                output.op != "output"
                or len(output.args) != 1
                or not isinstance(output.args[0], (list, tuple))
            ):
                return False
            returned = output.args[0]
            for user in node.users:
                if (
                    user.target is not operator.getitem
                    or user.kwargs
                    or len(user.args) != 2
                    or user.args[0] is not node
                    or type(user.args[1]) is not int
                    or not 0 <= user.args[1] < len(returned)
                    or not isinstance(returned[user.args[1]], Node)
                ):
                    return False
                aliases[user] = returned[user.args[1]]
            if not collect(child.graph):
                return False
        active_graphs.remove(id(graph))
        return True

    if not collect(root.graph):
        return None

    def resolve(node: Node) -> Node:
        while node in aliases:
            node = aliases[node]
        return node

    selections = [node for node in nodes if is_ordered_selection(node)]
    stores = tuple(node for node in nodes if node.target is store)
    loads = tuple(node for node in nodes if node.target is load)
    if len(selections) != 1 or not stores or not loads:
        return None
    selection = selections[0]
    arguments = selection_args(selection)
    if arguments is None:
        return None
    source, k, largest, stable_ties = arguments
    source = resolve(source)
    source_tensor = _tensor(source)
    if (
        source_tensor is None
        or source_tensor.ndim != 2
        or source_tensor.dtype not in (torch.float16, torch.bfloat16)
    ):
        return None
    n = _extent(source_tensor.size(1))
    if k is None:
        k = n
    row_block_id = env.get_block_id(source_tensor.size(0))
    if (
        n is None
        or k is None
        or not 0 < k <= n <= 32768
        or row_block_id is None
        or row_block_id in noncanonical_block_ids
        or env.block_sizes[row_block_id].reduction
        or any(width != n for width in carrier_widths)
    ):
        return None
    m = _extent(env.block_sizes[row_block_id].size)
    if m is None or m <= 0:
        return None
    widths = {1, n, k}
    anchor: torch.Tensor | None = None
    for node in loads:
        if len(node.args) != 4 or node.args[2:] != (None, None) or node.kwargs:
            return None
        tensor = _host_tensor_value(node.args[0])
        subscript = node.args[1]
        if (
            tensor is None
            or not isinstance(subscript, (tuple, list))
            or any(not isinstance(size, int) for size in tensor.shape)
            or any(
                not isinstance(stride, int) or stride < 0 for stride in tensor.stride()
            )
        ):
            return None
        if tensor.ndim == 2:
            if (
                len(subscript) != 2
                or _row_block(subscript[0]) != row_block_id
                or subscript[1] != slice(None)
                or tensor.size(0) != m
                or tensor.size(1) not in widths
            ):
                return None
            if tensor.size(1) == n and tensor.dtype == source_tensor.dtype:
                anchor = tensor if anchor is None else anchor
        elif tensor.ndim == 1:
            column_load = (
                subscript == [slice(None)]
                or subscript == (slice(None),)
                or subscript == [None, slice(None)]
                or subscript == (None, slice(None))
            )
            row_load = len(subscript) == 1 and _row_block(subscript[0]) == row_block_id
            if not (
                (column_load and tensor.size(0) in widths)
                or (row_load and tensor.size(0) == m)
            ):
                return None
        else:
            return None
    if anchor is None:
        return None

    for node in stores:
        if len(node.args) != 4 or node.args[3] is not None or node.kwargs:
            return None
        tensor = _host_tensor_value(node.args[0])
        subscript, value = node.args[1:3]
        if (
            tensor is None
            or tensor.ndim not in (1, 2)
            or any(not isinstance(size, int) for size in tensor.shape)
            or not isinstance(subscript, (tuple, list))
            or len(subscript) != tensor.ndim
            or _row_block(subscript[0]) != row_block_id
            or tensor.size(0) != m
            or not isinstance(value, Node)
            or _tensor(resolve(value)) is None
            or any(not isinstance(stride, int) for stride in tensor.stride())
        ):
            return None
        value_tensor = _tensor(resolve(value))
        assert value_tensor is not None
        destination_columns = tensor.size(1) if tensor.ndim == 2 else 1
        if value_tensor.ndim == 0:
            value_columns = 1
        elif value_tensor.ndim == 1:
            if env.get_block_id(value_tensor.size(0)) == row_block_id:
                # Rank-one row values target rank-one row stores. Broadcasting
                # them to [rows, columns] would right-align rows with columns.
                if tensor.ndim != 1:
                    return None
                value_columns = 1
            else:
                if tensor.ndim != 2:
                    return None
                value_columns = _extent(value_tensor.size(0))
        elif value_tensor.ndim == 2 and tensor.ndim == 2:
            value_columns = _extent(value_tensor.size(1))
        else:
            return None
        if value_columns not in (1, destination_columns):
            return None
        if tensor.ndim == 1:
            if tensor.stride(0) <= 0:
                return None
        elif (
            subscript[1] != slice(None)
            or tensor.size(1) not in widths
            or tensor.stride(1) < 0
            or (tensor.size(1) > 1 and tensor.stride(1) == 0)
            or tensor.stride(0) < (tensor.size(1) - 1) * tensor.stride(1) + 1
        ):
            # Conservative span proof: separate rows cannot write the same
            # address, including outputs with padded or sliced column strides.
            return None

    def valid_fragment_shape(tensor: torch.Tensor) -> bool:
        if tensor.ndim == 0:
            return True
        if tensor.ndim == 1:
            return (
                env.get_block_id(tensor.size(0)) == row_block_id
                or _extent(tensor.size(0)) in widths
            )
        if tensor.ndim != 2:
            return False
        # Unit-axis views may add/remove the row axis, but must not reinterpret
        # per-row scalars as a vector of columns shared between different rows.
        return (
            env.get_block_id(tensor.size(1)) != row_block_id
            and _extent(tensor.size(1)) in widths
            and (
                env.get_block_id(tensor.size(0)) == row_block_id
                or _extent(tensor.size(0)) == 1
            )
        )

    for node in nodes:
        if (
            node.target in (load, store, _host_tensor, _get_symnode)
            or node is selection
            or _static_view_shape_query(node)
        ):
            continue
        if node.target is operator.getitem:
            if (
                node.kwargs
                or len(node.args) != 2
                or node.args[0] is not selection
                or node.args[1] not in (0, 1)
            ):
                return None
            continue
        tensor = _tensor(node)
        if (
            not supports_row_fragment_node(node)
            or tensor is None
            or not valid_fragment_shape(tensor)
            or (
                isinstance(node.target, torch._ops.OpOverload)
                and (
                    node.target._schema.is_mutable
                    or torch.Tag.nondeterministic_seeded in node.target.tags
                )
            )
        ):
            return None
        if (
            tensor.ndim == 2
            and isinstance(node.target, torch._ops.OpOverload)
            and (torch.Tag.pointwise in node.target.tags)
        ):
            for argument in node.all_input_nodes:
                operand = _tensor(resolve(argument))
                if (
                    operand is not None
                    and operand.ndim == 1
                    and env.get_block_id(operand.size(0)) == row_block_id
                ):
                    # A row scalar must carry its trailing unit dimension;
                    # otherwise ordinary tensor broadcasting uses column axes.
                    return None

    prologue: set[Node] = set()

    def recoverable_source(node: Node) -> bool:
        node = resolve(node)
        if node in prologue:
            return True
        prologue.add(node)
        if node.target is load:
            return True
        return supports_row_fragment_node(node) and all(
            recoverable_source(argument)
            for argument in node.all_input_nodes
            if _tensor(resolve(argument)) is not None
        )

    if not recoverable_source(source):
        return None
    # No output may alias an input or another output even through distinct
    # views. Separate StorageImpls still need the final runtime span proof.
    read_tensors = _unique_tensors(loads)
    write_tensors = _unique_tensors(stores)
    if len(write_tensors) != len(stores):
        return None
    for index, output in enumerate(write_tensors):
        if any(
            output.untyped_storage()._cdata == other.untyped_storage()._cdata
            for other in (*read_tensors, *write_tensors[:index])
        ):
            return None
    return RowTopKGraph(
        root.graph,
        selection,
        source,
        row_block_id,
        anchor,
        n,
        k,
        largest,
        loads,
        stores,
        aliases,
        read_tensors,
        write_tensors,
        stable_ties,
    )
