"""Layout-aware lowering of tile values carried by a serial loop.

Ownership is proved on FX graphs. After thread assignment, a typed plan binds
the loop interface to thread-local register tiles. The emitter opens element
scopes directly; it never matches, moves, or rewrites scalar AST statements.
"""

from __future__ import annotations

import ast
import contextlib
import dataclasses
import operator
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

import torch
from torch.fx import Node
from torch.fx.node import map_arg

from ...language import _tracing_ops
from ...language import creation_ops
from ...language import memory_ops
from ...language import tile_ops
from ..ast_extension import expr_from_string
from ..ast_extension import statement_from_string
from ..compile_environment import CompileEnvironment
from ..compile_environment import FixedBlockSizeSource
from ..inductor_lowering import CodegenState
from ..inductor_lowering import GraphInterpreter
from .fragment import FragmentEmitter
from .fragment import FragmentType
from .fragment import RegisterBuffer
from .fragment import SSAFragment

if TYPE_CHECKING:
    from collections.abc import Iterator

    from ..device_ir import DeviceIR
    from ..device_ir import ForLoopGraphInfo
    from ..device_ir import RootGraphInfo
    from ..generate_ast import GenerateAST
    from ..tile_strategy import DeviceGridState
    from ..tile_strategy import ThreadTileLayout
    from .loop_schedule import LoopMemoryPlan


def _index_axis(index: object) -> int | None:
    if not isinstance(index, Node) or index.target not in {
        _tracing_ops._get_symnode,
        _tracing_ops._new_var,
        tile_ops.tile_begin,
        torch.ops.aten.sym_size.int,
    }:
        return None
    env = CompileEnvironment.current()
    value = index.meta.get("val")
    if index.target is tile_ops.tile_begin:
        arg = index.args[0]
        if isinstance(arg, Node):
            value = arg.meta.get("val")
    if not isinstance(value, (int, torch.SymInt)):
        return None
    block = env.get_block_id(value)
    return None if block is None else env.canonical_block_id(block)


def _independent_memory(info: ForLoopGraphInfo, ir: DeviceIR) -> bool:
    """Prove element/iteration ownership using compiler-owned allocations.

    No fact here depends on sample pointers or absence of an input Source.
    External output arguments retain the ordinary lowering, as do loops which
    read a buffer they write.  The latter restriction also protects future
    loop pipelining from memory-carried dependencies.
    """
    env = CompileEnvironment.current()
    if len(ir.grid_block_ids) != 1:
        return False
    from ..device_ir import RootGraphInfo

    roots = [graph for graph in ir.graphs if isinstance(graph, RootGraphInfo)]
    if len(roots) != 1 or len(ir.graphs) != 2:
        return False
    axes = set(ir.grid_block_ids[0])
    allowed_axes = axes | set(info.block_ids)
    writes = set()
    reads = set()
    ownership = {}
    # Prefix/final stores are crossed by the interchange too. In particular,
    # inspecting only the loop body misses a final store through an input alias.
    for node in (*roots[0].graph.nodes, *info.graph.nodes):
        if node.op in ("placeholder", "output"):
            continue
        if node.target in (memory_ops.load, memory_ops.store):
            source = node.args[0]
            if not isinstance(source, Node):
                return False
            tensor = source.meta.get("val")
            if not isinstance(tensor, torch.Tensor):
                return False
            storage = tensor.untyped_storage()
            if node.target is memory_ops.load:
                reads.add(storage)
                continue
            if storage not in env.fresh_allocation_storages:
                return False
            indices = node.args[1]
            if not isinstance(indices, (tuple, list)):
                return False
            mapped = [_index_axis(index) for index in indices]
            for index, axis in zip(indices, mapped, strict=True):
                if (
                    isinstance(index, Node)
                    and index.target is tile_ops.tile_begin
                    and axis is not None
                    and axis in axes
                ):
                    block_source = env.block_sizes[axis].block_size_source
                    if (
                        not isinstance(block_source, FixedBlockSizeSource)
                        or block_source.value != 1
                    ):
                        return False
            mapped_axes = [axis for axis in mapped if axis is not None]
            if not axes <= set(mapped_axes) <= allowed_axes:
                return False
            if len(set(mapped_axes)) != len(mapped_axes):
                return False
            if any(
                axis is None and index != 0
                for axis, index in zip(mapped, indices, strict=True)
            ):
                return False
            owner = (
                tuple(axis if axis in axes else None for axis in mapped),
                tuple(tensor.stride()),
                tensor.storage_offset(),
            )
            if storage in ownership and ownership[storage] != owner:
                return False
            ownership[storage] = owner
            covered = 1
            for stride, size in sorted(zip(tensor.stride(), tensor.shape, strict=True)):
                if type(stride) is not int or type(size) is not int or stride < covered:
                    return False
                covered += (size - 1) * stride
            writes.add(storage)
        elif (
            node.target
            in {
                _tracing_ops._get_symnode,
                _tracing_ops._host_tensor,
                _tracing_ops._new_var,
                _tracing_ops._phi,
                tile_ops.tile_begin,
                tile_ops.tile_index,
                torch.ops.aten.sym_size.int,
            }
            or node.graph is roots[0].graph
            and node.target
            in {
                creation_ops.full,
                _tracing_ops._for_loop,
                operator.getitem,
            }
        ):
            continue
        elif not (
            isinstance(node.target, torch._ops.OpOverload)
            and torch.Tag.pointwise in node.target.tags
        ):
            return False
    return bool(writes) and not writes.intersection(reads)


def find_loop(ir: DeviceIR) -> ForLoopGraphInfo | None:
    from ..device_ir import ForLoopGraphInfo

    return next(
        (
            info
            for info in ir.graphs
            if isinstance(info, ForLoopGraphInfo) and _independent_memory(info, ir)
        ),
        None,
    )


@dataclasses.dataclass(frozen=True)
class TileLoopPlan:
    root: RootGraphInfo
    loop: ForLoopGraphInfo
    call: Node
    layout: ThreadTileLayout
    inputs: tuple[Node, ...]
    output_args: tuple[int, ...]
    prefix: tuple[Node, ...]
    suffix: tuple[Node, ...]
    preserved: tuple[Node, ...]
    memory: LoopMemoryPlan | None


def plan_root(
    cg: GenerateAST, root: RootGraphInfo, grid: DeviceGridState
) -> TileLoopPlan | None:
    """Commit the graph/interface/layout schedule before scalar body emission."""
    info = find_loop(cg.host_function.device_ir)
    if (
        info is None
        or info.loop_interface is None
        or not grid.lane_loops
        or len(info.block_ids) != 1
    ):
        return None
    layout = grid.thread_tile_layouts.get(grid.lane_loops[-1][0])
    env = CompileEnvironment.current()
    if (
        layout is None
        or env.block_sizes[info.block_ids[0]].from_config(cg.device_function.config)
        != 1
    ):
        return None
    nodes = tuple(root.graph.nodes)
    calls = [
        node
        for node in nodes
        if node.target is _tracing_ops._for_loop and node.args[0] == info.graph_id
    ]
    if len(calls) != 1:
        return None
    call = calls[0]
    # A per-element bound cannot be lifted outside the element scope.
    for bound in (*call.args[1], *call.args[2]):
        if isinstance(bound, Node):
            if (
                bound.target is not _tracing_ops._get_symnode
                or _index_axis(bound) is not None
            ):
                return None
        elif type(bound) is not int:
            return None
    inputs = tuple(call.args[3])
    if not inputs or any(
        not isinstance(node, Node) or not isinstance(node.meta.get("val"), torch.Tensor)
        for node in inputs
    ):
        return None
    interface = info.loop_interface
    assert interface.input_count == len(inputs)
    assert sorted(carry.output_index for carry in interface.carries) == list(
        range(len(info.graph.find_nodes(op="output")[0].args[0]))
    )
    position = nodes.index(call)
    prefix, suffix = nodes[:position], nodes[position + 1 :]
    suffix_nodes = set(suffix)
    # Original prefix values used after the loop are distinct from updated
    # carries (e.g. final + initial). Preserve them rather than overwrite them.
    preserved = tuple(
        node
        for node in prefix
        if isinstance(node.meta.get("val"), torch.Tensor)
        and node.target is not _tracing_ops._host_tensor
        and any(
            user in suffix_nodes and user.target is not _tracing_ops._phi
            for user in node.users
        )
    )
    from .loop_schedule import plan_memory

    return TileLoopPlan(
        root,
        info,
        call,
        layout,
        inputs,
        tuple(
            carry.input_index
            for carry in sorted(interface.carries, key=lambda carry: carry.output_index)
        ),
        prefix,
        suffix,
        preserved,
        plan_memory(info, call, layout, cg.device_function.config),
    )


class TileGraphEmitter(GraphInterpreter):
    """Reuse ordinary pointwise/index/memory lowering in an explicit scope."""

    # An interpreter's results include ASTs and nested tuples/lists, unlike
    # torch.fx.Argument, which describes inputs to an FX graph node.
    env: dict[Node, Any]

    def emit(self, node: Node) -> object:
        if node in self.env:
            return self.env[node]
        # The planned loop's phi is a value selection, not a request to merge
        # global scalar names. Its result already denotes the final register tile.
        if node.target is _tracing_ops._phi:
            rhs = node.args[1]
            assert isinstance(rhs, Node)
            result = self.emit(rhs)
        else:
            for dependency in node.all_input_nodes:
                self.emit(dependency)
            result = self.run_node(node)
        self.env[node] = result
        return result

    def api_state(self, node: Node) -> CodegenState:
        from ..generate_ast import GenerateAST

        cg = self.cg
        assert isinstance(cg, GenerateAST)
        for dependency in node.all_input_nodes:
            self.emit(dependency)
        ast_args = map_arg(node.args, lambda arg: self.env[arg])
        proxy_args = map_arg(node.args, lambda arg: arg.meta["val"])
        return CodegenState(cg, node, self.env, list(proxy_args), list(ast_args))


class TileLoopEmitter:
    def __init__(self, cg: GenerateAST, plan: TileLoopPlan) -> None:
        self.cg = cg
        self.plan = plan
        self.layout = plan.layout
        self.df = cg.device_function
        self.fragments = FragmentEmitter(cg)
        self.arguments: list[SSAFragment] = []
        self.initials: dict[Node, SSAFragment] = {}
        self.next_buffers: list[RegisterBuffer] = []

    def allocate(
        self, prefix: str, dtype: torch.dtype, elements: int | None = None
    ) -> RegisterBuffer:
        return self.fragments.allocate(
            prefix,
            FragmentType(
                dtype, self.layout.fragment_elements if elements is None else elements
            ),
        )

    @contextlib.contextmanager
    def groups(self) -> Iterator[None]:
        loop = statement_from_string(
            f"for {self.layout.group_var} in cutlass.range_constexpr({self.layout.group_extent}):\n    pass"
        )
        assert isinstance(loop, ast.For)
        body: list[ast.AST] = list(self.layout.group_setup)
        loop.body = cast("list[ast.stmt]", body)
        self.cg.add_statement(loop)
        with self.cg.set_statements(body):
            yield

    @contextlib.contextmanager
    def vector_elements(self) -> Iterator[None]:
        loop = statement_from_string(
            f"for {self.layout.element_var} in cutlass.range_constexpr({self.layout.elements}):\n    pass"
        )
        assert isinstance(loop, ast.For)
        body: list[ast.AST] = list(self.layout.element_setup)
        loop.body = cast("list[ast.stmt]", body)
        self.cg.add_statement(loop)
        with self.cg.set_statements(body):
            yield

    @contextlib.contextmanager
    def elements(self) -> Iterator[None]:
        with self.groups(), self.vector_elements():
            yield

    def graph(self, info: ForLoopGraphInfo | RootGraphInfo) -> TileGraphEmitter:
        return TileGraphEmitter(info.graph, self.cg)

    def write(self, buffer: RegisterBuffer, result: object) -> None:
        self.fragments.store(buffer, self.layout.fragment_index, result)

    def initialize(self) -> None:
        buffers = {
            node: self.allocate("tile_initial_buffer", node.meta["val"].dtype)
            for node in dict.fromkeys((*self.plan.inputs, *self.plan.preserved))
        }
        with self.elements():
            graph = self.graph(self.plan.root)
            for node in self.plan.prefix:
                graph.emit(node)
            for node, buffer in buffers.items():
                self.write(buffer, graph.emit(node))
        self.initials = {
            node: self.fragments.pack(buffer, "tile_loop_initial")
            for node, buffer in buffers.items()
        }
        carried = set(self.plan.output_args)
        self.arguments = [
            self.fragments.bind(self.initials[node], "tile_loop_value")
            if i in carried
            else self.initials[node]
            for i, node in enumerate(self.plan.inputs)
        ]
        self.next_buffers = [
            self.fragments.allocate("tile_next_buffer", self.arguments[i].type)
            for i in self.plan.output_args
        ]

    def body_graph(self) -> TileGraphEmitter:
        graph = self.graph(self.plan.loop)
        for node, value in zip(
            self.plan.loop.graph.find_nodes(op="placeholder"),
            self.arguments,
            strict=True,
        ):
            graph.env[node] = value.element(self.layout.fragment_index)
        return graph

    def update(
        self,
        loads: dict[Node, RegisterBuffer] | None = None,
        stores: dict[Node, RegisterBuffer] | None = None,
    ) -> None:
        with self.elements():
            graph = self.body_graph()
            for node, value in (loads or {}).items():
                graph.env[node] = value.element(
                    "0" if value.type.elements == 1 else self.layout.fragment_index
                )
            for node in self.plan.loop.graph.nodes:
                if node.op in ("placeholder", "output"):
                    continue
                if stores is not None and node in stores:
                    source = node.args[2]
                    self.write(
                        stores[node],
                        graph.emit(source) if isinstance(source, Node) else source,
                    )
                    graph.env[node] = None
                else:
                    graph.emit(node)
            outputs = self.plan.loop.graph.find_nodes(op="output")[0].args[0]
            for buffer, node in zip(self.next_buffers, outputs, strict=True):
                self.write(buffer, graph.emit(node))
        # Form all immutable next values before rebinding any loop argument.
        results = [
            self.fragments.pack(buffer, "tile_next_value")
            for buffer in self.next_buffers
        ]
        for arg, value in zip(self.plan.output_args, results, strict=True):
            self.fragments.assign(self.arguments[arg], value)

    def finish(self) -> None:
        with self.elements():
            graph = self.graph(self.plan.root)
            graph.env.update(
                {
                    node: value.element(self.layout.fragment_index)
                    for node, value in self.initials.items()
                }
            )
            graph.env[self.plan.call] = [
                self.arguments[arg].element(self.layout.fragment_index)
                for arg in self.plan.output_args
            ]
            for node in self.plan.suffix:
                graph.emit(node)

    def emit(self) -> None:
        self.initialize()
        # Only uniform bound metadata is evaluated here, never initial tile
        # arithmetic outside its element scope.
        graph = self.graph(self.plan.root)
        for node, value in zip(self.plan.inputs, self.arguments, strict=True):
            graph.env[node] = expr_from_string(value.name)
        state = graph.api_state(self.plan.call)
        loop = self.df.tile_strategy.codegen_device_loop(
            state, self.plan.loop.block_ids
        )
        if self.plan.memory is None:
            with self.cg.add_device_loop(loop):
                self.update()
        else:
            from .loop_schedule import emit_memory_loop

            emit_memory_loop(self, loop, self.plan.memory)
        self.finish()


def codegen_root(cg: GenerateAST, root: RootGraphInfo, grid: DeviceGridState) -> bool:
    plan = plan_root(cg, root, grid)
    if plan is None:
        return False
    layout = plan.layout
    # Materialize the surrounding layout scopes normally, but open the selected
    # element scope ourselves for initialization/update/consumption. This is
    # planned before any user scalar AST exists, not a loop interchange.
    element_setup_ids = {id(stmt) for stmt in layout.element_setup}
    outer = dataclasses.replace(
        grid,
        lane_loops=[
            (name, extent)
            for name, extent in grid.lane_loops
            if name != layout.group_var
        ],
        lane_setup_statements=[
            stmt
            for stmt in grid.lane_setup_statements
            if id(stmt) not in element_setup_ids
        ],
        vec_lane_wrappers={
            name: wrapper
            for name, wrapper in grid.vec_lane_wrappers.items()
            if name != layout.group_var
        },
    )
    body: list[ast.AST] = []
    cg.device_function.cute_state.emitting_tile_loop = True
    try:
        with cg.set_statements(body):
            TileLoopEmitter(cg, plan).emit()
    finally:
        cg.device_function.cute_state.emitting_tile_loop = False
    cg.statements_stack[-1].extend(grid.outer_prefix)
    cg.statements_stack[-1].extend(outer.wrap_body(body))
    cg.statements_stack[-1].extend(grid.outer_suffix)
    return True
