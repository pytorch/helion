"""Lower pure, complete tensor rows through static subgroup register routing.

Admission depends on memory ownership and ordinary tensor operations. It does
not recognize sorting, selection, or any other algorithm expressed by the DAG.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import math
import operator
from typing import TYPE_CHECKING
from typing import cast

import torch
from torch._subclasses.fake_tensor import unset_fake_temporarily
from torch.fx import Graph
from torch.fx import GraphModule
from torch.fx import Node

from ... import exc
from ...language._tracing_ops import _get_symnode
from ...language._tracing_ops import _host_tensor
from ...language._tracing_ops import _if
from ...language._tracing_ops import _mask_to
from ...language._tracing_ops import _new_var
from ...language._tracing_ops import _phi
from ...language.memory_ops import load
from ...language.memory_ops import store
from ...language.view_ops import subscript
from ..ast_extension import expr_from_string
from ..compile_environment import CompileEnvironment
from ..device_ir import ElseGraphInfo
from ..device_ir import IfGraphInfo
from ..device_ir import RootGraphInfo
from ..host_function import HostFunction
from ..program_id import XYZProgramIDs
from ..tile_strategy import DeviceGridState
from .bounded_gather import integer_bounds
from .memory_ops import runtime_tensors_are_proven_disjoint
from .register_program import emit_register_program
from .register_program import register_program_supported
from .row_fragment import RowFragment
from .row_fragment import RowFragmentLayout
from .row_fragment import row_fragment_logical_extent

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..device_ir import GraphInfo
    from ..generate_ast import GenerateAST
    from ..tile_dispatch import TileStrategyDispatch


@dataclass(frozen=True)
class RegisterRegionPlan:
    root_graph_id: int
    root_graph: Graph
    row_block_id: int
    rows: int
    groups: int
    threads: int
    rows_per_block: int
    loads: tuple[Node, ...]
    stores: tuple[Node, ...]
    module: GraphModule


def _memory_tensor(node: Node) -> torch.Tensor | None:
    source = node.args[0]
    if not isinstance(source, Node) or source.target is not _host_tensor:
        return None
    value = source.meta.get("val")
    if not isinstance(value, torch.Tensor) or value.ndim != 3:
        return None
    if any(type(size) is not int or size <= 0 for size in value.shape):
        return None
    return value


def _nonoverlapping(tensor: torch.Tensor) -> bool:
    span = 1
    dimensions = sorted(zip(tensor.stride(), tensor.shape, strict=True))
    for stride, size in dimensions:
        if type(stride) is not int or stride < span:
            return False
        span += (size - 1) * stride
    return True


def _static_shape(tensor: torch.Tensor) -> tuple[int, ...] | None:
    dimensions = [row_fragment_logical_extent(size) for size in tensor.shape]
    if any(size is None or size <= 0 for size in dimensions):
        return None
    return tuple(cast("list[int]", dimensions))


class _TensorGraphBuilder:
    """Normalize pure Helion tensor/control-flow IR without algorithm matching."""

    def __init__(self, graphs: Sequence[GraphInfo], groups: int) -> None:
        self.graphs = {info.graph_id: info for info in graphs}
        self.groups = groups
        self.graph = Graph()
        self.attributes: dict[str, GraphModule] = {}
        self.replacements: dict[Node, Node] = {}
        self.static_nodes: set[Node] = set()
        self.bounds: dict[Node, tuple[int, int]] = {}

    def _record_bounds(self, node: Node) -> None:
        # Masking an arbitrary signed integer with a nonnegative constant gives
        # an input-independent bound, including when the input itself is negative.
        if node.target in (
            torch.ops.aten.bitwise_and.Tensor,
            torch.ops.aten.bitwise_and.Scalar,
        ):
            value = node.meta["val"]
            if value.dtype in (
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.uint8,
            ):
                for mask in node.args:
                    if type(mask) is int and 0 <= mask <= torch.iinfo(value.dtype).max:
                        self.bounds[node] = (0, mask)
                        return

    def _logical_shape(self, node: Node) -> tuple[int, ...] | None:
        value = node.meta["val"]
        shape = _static_shape(value)
        if shape is not None:
            return shape
        if node.target in (torch.ops.aten.view.default, torch.ops.aten.reshape.default):
            source, sizes = node.args
            if (
                isinstance(source, Node)
                and source in self.replacements
                and isinstance(sizes, (list, tuple))
                and all(
                    type(size) is int and (size > 0 or size == -1) for size in sizes
                )
                and sizes.count(-1) <= 1
            ):
                sizes = cast("Sequence[int]", sizes)
                extent = self.replacements[source].meta["val"].numel()
                fixed = math.prod(size for size in sizes if size != -1)
                if extent % fixed == 0 and (-1 in sizes or extent == fixed):
                    return tuple(
                        extent // fixed if size == -1 else size for size in sizes
                    )
        return None

    def _gather_is_bounded(self, node: Node) -> bool:
        source, dim, index = node.args[:3]
        if not isinstance(source, Node) or not isinstance(index, Node):
            return False
        shape = source.meta["val"].shape
        if type(dim) is not int or not -len(shape) <= dim < len(shape):
            return False
        bounds = integer_bounds(index, captured_bounds=self.bounds)
        return bounds is not None and 0 <= bounds[0] <= bounds[1] < shape[dim]

    def _conditional(self, node: Node) -> bool:
        if self.groups != 32 or len(node.args) != 5 or node.kwargs:
            return False
        predicate, true_id, false_id, true_args, false_args = node.args
        if (
            not isinstance(predicate, Node)
            or predicate not in self.replacements
            or type(true_id) is not int
            or type(false_id) is not int
            or not isinstance(true_args, (list, tuple))
            or not isinstance(false_args, (list, tuple))
        ):
            return False
        condition = self.replacements[predicate]
        value = condition.meta.get("val")
        true_info, false_info = self.graphs.get(true_id), self.graphs.get(false_id)
        if (
            not isinstance(value, torch.Tensor)
            or value.numel() != 1
            or not isinstance(true_info, IfGraphInfo)
            or not isinstance(false_info, ElseGraphInfo)
            or true_info.else_branch != false_id
            or not true_info.branches_outputs
        ):
            return False
        operands: list[Node] = []
        for source in (*true_args, *false_args):
            if not isinstance(source, Node) or source not in self.replacements:
                return False
            if source not in operands:
                operands.append(source)
        names = {}
        for arg_names, arguments in (
            (true_info.if_arg_names, true_args),
            (true_info.else_arg_names, false_args),
        ):
            if arg_names is None or len(arg_names) != len(arguments):
                return False
            names.update(zip(arg_names, arguments, strict=True))
        children = []
        outputs = []
        for branch_index, (info, arguments) in enumerate(
            ((true_info, true_args), (false_info, false_args))
        ):
            child = _TensorGraphBuilder(tuple(self.graphs.values()), self.groups)
            captures = {}
            for operand in operands:
                captured = child.graph.placeholder(operand.name)
                original = self.replacements[operand]
                captured.meta = dict(original.meta)
                captures[operand] = captured
                if original in self.static_nodes:
                    child.static_nodes.add(captured)
                bounds = integer_bounds(original, captured_bounds=self.bounds)
                if bounds is not None:
                    child.bounds[captured] = bounds
            placeholders = list(info.graph.find_nodes(op="placeholder"))
            if len(placeholders) != len(arguments):
                return False
            child.replacements.update(
                (placeholder, captures[cast("Node", argument)])
                for placeholder, argument in zip(placeholders, arguments, strict=True)
            )
            if not child.copy(info):
                return False
            returned = list(info.graph.nodes)[-1].args[0]
            if not isinstance(returned, (tuple, list)):
                return False
            branch_outputs = []
            for pair in true_info.branches_outputs:
                if len(pair) != 2:
                    return False
                selected = pair[branch_index]
                if type(selected) is int and 0 <= selected < len(returned):
                    source = returned[selected]
                    if not isinstance(source, Node) or source not in child.replacements:
                        return False
                    result = child.replacements[source]
                elif isinstance(selected, str) and selected in names:
                    result = captures[cast("Node", names[selected])]
                else:
                    return False
                result_value = result.meta.get("val")
                if (
                    not isinstance(result_value, torch.Tensor)
                    or result_value.ndim != 2
                    or result_value.shape[0] != self.groups
                ):
                    return False
                branch_outputs.append(result)
            module = child.finish(branch_outputs)
            if module is None:
                return False
            children.append(module)
            outputs.append([result.meta["val"] for result in branch_outputs])
        if any(
            a.shape != b.shape or a.dtype != b.dtype
            for a, b in zip(*outputs, strict=True)
        ):
            return False
        attributes = []
        for label, child in zip(("true", "false"), children, strict=True):
            name = f"{node.name}_{label}"
            self.attributes[name] = child
            attributes.append(self.graph.get_attr(name))
        result = self.graph.call_function(
            torch.ops.higher_order.cond,
            (
                condition,
                *attributes,
                tuple(self.replacements[operand] for operand in operands),
            ),
        )
        result.meta["val"] = tuple(outputs[0])
        self.replacements[node] = result
        return True

    def copy(self, info: GraphInfo, loads: tuple[Node, ...] = ()) -> bool:
        for node in info.graph.nodes:
            if node.op == "output" or (
                node.op == "placeholder" and node in self.replacements
            ):
                continue
            if node.target in (_host_tensor, _get_symnode, store):
                if not isinstance(info, RootGraphInfo):
                    return False
                continue
            if node.target is _if:
                if not self._conditional(node):
                    return False
                continue
            if node.target is operator.getitem:
                source, index = node.args
                if (
                    not isinstance(source, Node)
                    or source.target is not _if
                    or source not in self.replacements
                    or type(index) is not int
                    or any(user.target is not _phi for user in node.users)
                ):
                    return False
                conditional = self.replacements[source]
                count = len(conditional.meta["val"])
                if not 0 <= index < 2 * count:
                    return False
                # Helion traces both branch values before merging with _phi.
                # The pure cond already returns the merged value for each pair.
                key = (conditional, index % count)
                merged = next(
                    (
                        user
                        for user in conditional.users
                        if user.target is operator.getitem and user.args == key
                    ),
                    None,
                )
                if merged is None:
                    merged = self.graph.call_function(operator.getitem, key)
                    merged.meta["val"] = conditional.meta["val"][index % count]
                self.replacements[node] = merged
                continue
            if node.target in (_new_var, _mask_to, _phi):
                source = node.args[0]
                if not isinstance(source, Node) or source not in self.replacements:
                    return False
                replacement = self.replacements[source]
                if node.target is _phi and (
                    not isinstance(node.args[1], Node)
                    or self.replacements.get(node.args[1]) is not replacement
                ):
                    return False
                value = node.meta.get("val")
                if (
                    not isinstance(value, torch.Tensor)
                    or tuple(value.shape) != tuple(source.meta["val"].shape)
                    or value.dtype != replacement.meta["val"].dtype
                ):
                    return False
                # Complete logical slices have no padded register elements;
                # reduction mask markers therefore preserve every element.
                self.replacements[node] = replacement
                continue
            value = node.meta.get("val")
            if not isinstance(value, torch.Tensor):
                return False
            shape = self._logical_shape(node)
            if node in loads:
                tensor = _memory_tensor(node)
                assert tensor is not None
                shape = tuple(tensor.shape[1:])
                new = self.graph.placeholder(node.name)
            elif node.target is subscript:
                source, indices = node.args
                if (
                    not isinstance(source, Node)
                    or source not in self.replacements
                    or not isinstance(indices, (list, tuple))
                    or any(
                        index is not None and index != slice(None) for index in indices
                    )
                    or shape is None
                ):
                    return False
                new = self.graph.call_function(
                    torch.ops.aten.view.default,
                    (self.replacements[source], list(shape)),
                )
            else:
                target = node.target
                if (
                    node.op != "call_function"
                    or not isinstance(target, torch._ops.OpOverload)
                    or target._schema.is_mutable
                    or torch.Tag.nondeterministic_seeded in target.tags
                    or any(
                        source not in self.replacements
                        for source in node.all_input_nodes
                    )
                    or shape is None
                ):
                    return False
                new = self.graph.node_copy(node, self.replacements.__getitem__)
            assert shape is not None
            # Logical extents are proved complete slices, never padding hints.
            with unset_fake_temporarily():
                new.meta["val"] = torch.empty(shape, dtype=value.dtype, device="meta")
            self.replacements[node] = new
            if node not in loads and all(
                self.replacements[source] in self.static_nodes
                for source in node.all_input_nodes
            ):
                self.static_nodes.add(new)
            self._record_bounds(new)
            if (
                new.target
                in (
                    torch.ops.aten.gather.default,
                    torch.ops.aten.index_select.default,
                )
                and new.args[2] not in self.static_nodes
            ):
                if (
                    new.target is torch.ops.aten.index_select.default
                    or not self._gather_is_bounded(new)
                ):
                    return False
        return True

    def finish(self, outputs: Sequence[Node]) -> GraphModule | None:
        self.graph.output(tuple(outputs))
        self.graph.eliminate_dead_code()
        module = GraphModule(self.attributes, self.graph)
        if not register_program_supported(
            module, list(self.graph.find_nodes(op="placeholder")), lanes=self.groups
        ):
            return None
        return module


def _tensor_graph(
    root: RootGraphInfo,
    loads: tuple[Node, ...],
    stores: tuple[Node, ...],
    groups: int,
    graphs: Sequence[GraphInfo],
) -> GraphModule | None:
    builder = _TensorGraphBuilder(graphs, groups)
    if not builder.copy(root, loads):
        return None
    outputs = []
    for node in stores:
        source = node.args[2]
        if not isinstance(source, Node) or source not in builder.replacements:
            return None
        tensor = _memory_tensor(node)
        assert tensor is not None
        result = builder.replacements[source]
        if tuple(result.meta["val"].shape) != tuple(tensor.shape[1:]):
            return None
        outputs.append(result)
    return builder.finish(outputs)


def plan_register_region(
    graphs: Sequence[GraphInfo], tile_strategy: TileStrategyDispatch
) -> RegisterRegionPlan | None:
    """Admit one scalar-row grid, disjoint full-slice IO and pure tensor regions."""
    host = HostFunction.current()
    roots = [graph for graph in graphs if isinstance(graph, RootGraphInfo)]
    if len(host.device_ir.root_ids) != 1 or len(roots) != 1:
        return None
    # The IR may retain unused rolled alternatives; only reachable pure branch
    # graphs are admitted along with the root's complete memory operations.
    root = roots[0]
    nodes = list(root.graph.nodes)
    if not nodes or nodes[-1].op != "output" or nodes[-1].args != (None,):
        return None
    loads = tuple(node for node in nodes if node.target is load)
    stores = tuple(node for node in nodes if node.target is store)
    if not loads or not stores:
        return None
    first = _memory_tensor(loads[0])
    if first is None:
        return None
    rows, groups, _registers = first.shape
    if groups > 32 or groups & (groups - 1):
        return None
    subscript = loads[0].args[1]
    if not isinstance(subscript, (list, tuple)) or len(subscript) != 3:
        return None
    row = subscript[0]
    if not isinstance(row, Node) or row.target is not _get_symnode:
        return None
    env = CompileEnvironment.current()
    row_value = row.meta.get("val")
    if not isinstance(row_value, torch.SymInt):
        return None
    row_block_id = env.get_block_id(row_value)
    if (
        row_block_id is None
        or row_block_id not in env.config_spec.grid_block_ids
        or row_block_id in host.device_ir.noncanonical_task_origin_block_ids
        or env.block_sizes[row_block_id].size != rows
        or any(user.target not in (load, store) for user in row.users)
    ):
        return None
    tensors = []
    for node in (*loads, *stores):
        tensor = _memory_tensor(node)
        if (
            tensor is None
            or tensor.shape[:2] != first.shape[:2]
            or node.kwargs
            or len(node.args) != 4
            or node.args[1] != [row, slice(None), slice(None)]
            or node.args[3] is not None
            or (node.target is load and node.args[2] is not None)
            or (node.target is store and not _nonoverlapping(tensor))
        ):
            return None
        tensors.append(tensor)
    # Fresh wrapper allocations have private storage. Runtime arguments need
    # the cache-specialized span proof, including distinct DLPack wrappers.
    for index in range(len(loads), len(tensors)):
        output = tensors[index]
        for other in tensors[:index]:
            if output.untyped_storage()._cdata == other.untyped_storage()._cdata:
                return None
            if (
                output.untyped_storage() not in env._symbolically_exact_layout_storages
                and other.untyped_storage()
                not in env._symbolically_exact_layout_storages
                and not runtime_tensors_are_proven_disjoint(env, output, other)
            ):
                return None
    module = _tensor_graph(root, loads, stores, groups, graphs)
    if module is None:
        return None
    config = tile_strategy.strategies[0].fn.config
    if any(
        env.config_spec.num_threads.config_get(config.num_threads, block_id, 0)
        for block_id in env.config_spec.num_threads.valid_block_ids()
        if block_id != row_block_id
    ):
        return None
    requested = env.config_spec.num_threads.config_get(
        config.num_threads, row_block_id, 0
    )
    threads = requested or 128
    if (
        type(threads) is not int
        or threads < 32
        or threads > 1024
        or threads & (threads - 1)
    ):
        raise exc.InvalidConfig(
            f"register region requires a power-of-two thread count in [32, 1024], got {threads}"
        )
    return RegisterRegionPlan(
        root.graph_id,
        root.graph,
        row_block_id,
        rows,
        groups,
        threads,
        threads // groups,
        loads,
        stores,
        module,
    )


def codegen_register_region(cg: GenerateAST, plan: RegisterRegionPlan) -> bool:
    """Load full subgroup fragments, inline the tensor DAG, and store results."""
    fn = cg.device_function
    backend = CompileEnvironment.current().backend
    root = cg.current_root_graph_info
    grid = cg.current_grid_state
    if (
        root is None
        or root.graph_id != plan.root_graph_id
        or root.graph is not plan.root_graph
        or fn.pid is None
        or not isinstance(grid, DeviceGridState)
        or plan.row_block_id not in grid.block_id_to_info
    ):
        return False
    row_info = grid.block_id_to_info[plan.row_block_id]
    pid_info = fn.pid.pid_info
    if (
        row_info.begin_expr != 0
        or row_info.end_expr != plan.rows
        or len(pid_info) != 1
        or pid_info[0].block_id != plan.row_block_id
    ):
        return False
    fn.pid = XYZProgramIDs(
        pid_info=[pid_info[0]._replace(block_size_var=str(plan.rows_per_block))]
    )

    def emit(source: str) -> None:
        for statement in ast.parse(source).body:
            cg.add_statement(statement)

    thread = fn.new_var("register_thread")
    lane = fn.new_var("register_lane")
    row = fn.new_var("register_row")
    valid = fn.new_var("register_valid_row")
    emit(f"{thread} = cutlass.Int32(cute.arch.thread_idx()[0])")
    emit(f"{lane} = {thread} % {plan.groups}")
    emit(
        f"{row} = cutlass.Int64(cute.arch.block_idx()[0]) * {plan.rows_per_block} + cutlass.Int64({thread} // {plan.groups})"
    )
    emit(f"{valid} = {row} < {plan.rows}")

    def tensor_name(node: Node) -> str:
        tensor = _memory_tensor(node)
        assert tensor is not None
        name = fn.tensor_arg(tensor).name
        fn.placeholder_args.add(name)
        return name

    fragments = []
    for node in plan.loads:
        tensor = _memory_tensor(node)
        assert tensor is not None
        registers = tensor.size(2)
        dtype = backend.dtype_str(tensor.dtype)
        name = fn.new_var("register_input")
        item = fn.new_var("register_load_index")
        emit(f"{name} = cute.make_rmem_tensor({registers}, {dtype})")
        emit(f"{name}.fill({dtype}(0))")
        memory = tensor_name(node)
        emit(
            f"for {item} in cutlass.range_constexpr({registers}):\n    if {valid}:\n        {name}[{item}] = {memory}[{row}, {lane}, {item}]"
        )
        fragments.append(
            RowFragment(
                name,
                tensor.dtype,
                plan.groups * registers,
                RowFragmentLayout(plan.groups, registers, lane),
            )
        )
    outputs = emit_register_program(
        cg, plan.module, fragments, lanes=plan.groups, lane_expr=lane
    )
    for node, fragment in zip(plan.stores, outputs, strict=True):
        tensor = _memory_tensor(node)
        assert tensor is not None
        name = tensor_name(node)
        item = fn.new_var("register_store_index")
        value = ast.unparse(
            backend.cast_ast(expr_from_string(f"{fragment.name}[{item}]"), tensor.dtype)
        )
        emit(
            f"for {item} in cutlass.range_constexpr({tensor.size(2)}):\n    if {valid}:\n        {name}[{row}, {lane}, {item}] = {value}"
        )
    return True
