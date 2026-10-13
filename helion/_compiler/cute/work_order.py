from __future__ import annotations

import ast
import dataclasses
import math
import operator
from typing import TYPE_CHECKING

import sympy
import torch
from torch.fx import Graph
from torch.fx import Node

from ... import exc
from ...language import _tracing_ops
from ...language import memory_ops
from ...language import tile_ops
from ..ast_extension import expr_from_string
from ..compile_environment import CompileEnvironment
from ..device_function import VarInfo
from ..device_ir import ForLoopGraphInfo
from ..device_ir import HelperFunctionGraphInfo
from ..device_ir import RootGraphInfo
from ..host_function import HostFunction
from ..inductor_lowering import GraphInterpreter
from ..loop_dependency_checker import INTRA_LOOP_RAW_BARRIER_META
from ..program_id import FlatProgramIDs
from ..tile_dependency import CoordinateDomain
from ..tile_dependency import _analyze_integer_expression
from ..tile_dependency import coordinate_axis_symbol
from ..tile_strategy import DeviceGridState
from ..tile_strategy import _BaseNDTileStrategy
from ..variable_origin import GridOrigin
from ..variable_origin import TileIdOrigin
from .memory_ops import register_cute_tensor_alias_specializations
from .memory_ops import runtime_tensor_sources_are_proven_disjoint

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ..device_ir import DeviceIR
    from ..generate_ast import GenerateAST


class UnsupportedWorkOrder(ValueError):
    pass


_INTEGER_DTYPES = frozenset((torch.int32, torch.int64, torch.bool))
_POINTWISE = frozenset(
    (
        operator.add,
        operator.sub,
        operator.mul,
        operator.neg,
        operator.lt,
        operator.le,
        operator.eq,
        operator.gt,
        operator.ge,
        torch.ops.aten.add.Tensor,
        torch.ops.aten.sub.Tensor,
        torch.ops.aten.mul.Tensor,
        torch.ops.aten.neg.default,
        torch.ops.aten.lt.Tensor,
        torch.ops.aten.le.Tensor,
        torch.ops.aten.eq.Tensor,
        torch.ops.aten.gt.Tensor,
        torch.ops.aten.ge.Tensor,
        torch.ops.prims.convert_element_type.default,
    )
)


def _arguments(value: object) -> object:
    if isinstance(value, Node):
        return (Node, id(value))
    if isinstance(value, (tuple, list)):
        return (type(value), tuple(_arguments(item) for item in value))
    if isinstance(value, dict):
        return (dict, tuple((key, _arguments(item)) for key, item in value.items()))
    return value


def _node_fact(node: Node) -> tuple[object, ...]:
    return (
        node.op,
        node.target,
        _arguments(node.args),
        _arguments(node.kwargs),
        id(node.meta.get("val")),
        id(node.meta.get("lowering")),
        node.meta.get(INTRA_LOOP_RAW_BARRIER_META),
    )


def _is_integer_scalar(node: Node) -> bool:
    value = node.meta.get("val")
    if isinstance(value, (int, torch.SymInt, torch.SymBool)):
        return True
    return (
        isinstance(value, torch.Tensor)
        and value.ndim == 0
        and value.dtype in _INTEGER_DTYPES
    )


@dataclasses.dataclass(frozen=True)
class ScalarSlice:
    """An original pure scalar SSA slice, not permission to speculate its reads.

    The eventual work-order admission must separately prove the candidate domain,
    read/write disjointness, range arithmetic and complete task independence.
    This projection only reuses the original lowering with explicit coordinate
    leaves; it does not contain another expression renderer.
    """

    graph: Graph
    outputs: tuple[Node | int, ...]
    leaves: tuple[Node, ...]
    nodes: tuple[Node, ...]
    facts: tuple[tuple[object, ...], ...]
    # Keep referenced values/lowerings alive; IDs alone are not ownership.
    owners: tuple[tuple[object, object], ...] = dataclasses.field(repr=False)
    membership: tuple[Node, ...] = dataclasses.field(repr=False)

    def check(self) -> None:
        if tuple(self.graph.nodes) != self.membership:
            raise UnsupportedWorkOrder("scalar root membership or order changed")
        if any(node.graph is not self.graph for node in (*self.nodes, *self.leaves)):
            raise UnsupportedWorkOrder("foreign scalar graph")
        if tuple(_node_fact(node) for node in self.nodes) != self.facts:
            raise UnsupportedWorkOrder("scalar SSA changed after discovery")


def scalar_slice(
    graph: Graph, outputs: tuple[Node | int, ...], leaves: tuple[Node, ...]
) -> ScalarSlice:
    if len(set(leaves)) != len(leaves) or not outputs:
        raise UnsupportedWorkOrder("scalar leaves and outputs must be unambiguous")
    wanted: set[Node] = set()

    def visit(node: Node) -> None:
        if node in wanted:
            return
        if node.graph is not graph or node.op != "call_function":
            raise UnsupportedWorkOrder("scalar slice escapes its original root")
        if node.meta.get(INTRA_LOOP_RAW_BARRIER_META):
            raise UnsupportedWorkOrder("work key crosses a memory effect")
        wanted.add(node)
        if node in leaves:
            if not _is_integer_scalar(node) or node.target not in (
                _tracing_ops._get_symnode,
                tile_ops.tile_id,
            ):
                raise UnsupportedWorkOrder(
                    "only original integer coordinate leaves bind"
                )
            return
        if node.target is _tracing_ops._host_tensor:
            value = node.meta["val"]
            if (
                not isinstance(value, torch.Tensor)
                or value.dtype not in _INTEGER_DTYPES
            ):
                raise UnsupportedWorkOrder("work metadata must be integer")
            return
        if not _is_integer_scalar(node):
            raise UnsupportedWorkOrder("work key needs original integer scalar values")
        if node.target is memory_ops.load:
            source = node.args[0]
            if (
                not isinstance(source, Node)
                or source.target is not _tracing_ops._host_tensor
            ):
                raise UnsupportedWorkOrder(
                    "work metadata must come from an original host tensor"
                )
        elif (
            node.target not in _POINTWISE
            and node.target is not _tracing_ops._get_symnode
        ):
            raise UnsupportedWorkOrder("unsupported scalar key operation")
        for arg in node.all_input_nodes:
            visit(arg)

    for output in outputs:
        if isinstance(output, Node):
            visit(output)
        elif type(output) is not int:
            raise UnsupportedWorkOrder("work bounds must be integer scalars")
    if not set(leaves) <= wanted:
        raise UnsupportedWorkOrder("coordinate binding does not belong to the key")
    nodes = tuple(node for node in graph.nodes if node in wanted)
    return ScalarSlice(
        graph,
        outputs,
        leaves,
        nodes,
        tuple(_node_fact(node) for node in nodes),
        tuple((node.meta.get("val"), node.meta.get("lowering")) for node in nodes),
        tuple(graph.nodes),
    )


@dataclasses.dataclass(frozen=True)
class ScalarProjection:
    statements: tuple[ast.stmt, ...]
    values: tuple[ast.expr, ...]


def project_scalar_slice(
    cg: GenerateAST, spec: ScalarSlice, bindings: Mapping[Node, ast.AST]
) -> ScalarProjection:
    """Lower a detached clone only after the original body bound its arguments."""
    spec.check()
    if tuple(bindings) != spec.leaves:
        raise UnsupportedWorkOrder(
            "coordinate bindings differ from the discovered leaves"
        )
    if not all(isinstance(value, ast.expr) for value in bindings.values()):
        raise UnsupportedWorkOrder("coordinate binding must be an expression")
    df = cg.device_function
    env = CompileEnvironment.current()
    if env.backend_name != "cute" or df.config.load_eviction_policies:
        raise UnsupportedWorkOrder(
            "scalar projection requires CuTe default load-site policy"
        )
    grid = cg.current_grid_state
    if not isinstance(grid, DeviceGridState):
        raise UnsupportedWorkOrder(
            "scalar projection requires the original active grid"
        )
    axes = []
    for leaf in spec.leaves:
        value = leaf.meta["val"]
        if not isinstance(value, torch.SymInt):
            raise UnsupportedWorkOrder(
                "coordinate leaf must retain its original grid origin"
            )
        origin = HostFunction.current().expr_to_origin.get(value._sympy_())
        if origin is None or type(origin.origin) not in (GridOrigin, TileIdOrigin):
            raise UnsupportedWorkOrder("coordinate leaf is not a scalar grid index")
        assert isinstance(origin.origin, GridOrigin)
        axis = env.canonical_block_id(origin.origin.block_id)
        if (
            axis not in grid.block_ids
            or cg.active_device_loops.get(axis) != [grid]
            or df.resolved_block_size(axis) != 1
            or grid.strategy.mask_var(axis) is not None
            or axis in grid.block_thread_axes
        ):
            raise UnsupportedWorkOrder(
                "coordinate needs an independent full scalar grid axis"
            )
        axes.append(axis)
    if len(set(axes)) != len(axes):
        raise UnsupportedWorkOrder("multiple leaves ambiguously bind one grid axis")
    arguments = tuple(df.arguments)
    tensors = tuple(df._tensor_args.items())
    for node in spec.nodes:
        if (
            node.target is _tracing_ops._host_tensor
            and node.meta["val"] not in df._tensor_args
        ):
            raise UnsupportedWorkOrder(
                "original body has not bound the metadata argument"
            )
    graph = Graph()
    copied: dict[Node, Node] = {}
    for leaf in spec.leaves:
        copied[leaf] = graph.placeholder(leaf.name)
        copied[leaf].meta = dict(leaf.meta)
    for node in spec.nodes:
        if node not in copied:
            copied[node] = graph.node_copy(node, copied.__getitem__)
            copied[node].meta = {
                key: value for key, value in node.meta.items() if key != "codegen"
            }
    graph.output(
        tuple(
            copied[value] if isinstance(value, Node) else value
            for value in spec.outputs
        )
    )
    statements: list[ast.AST] = []
    tracked = cg._track_statement_owners
    indices = dict(grid.strategy.index_vars)
    offsets = dict(grid.strategy.offset_vars)
    sizes = dict(df.block_size_var_cache)
    load_index = df.device_load_index
    try:
        cg._track_statement_owners = False
        with cg.set_statements(statements):
            # An existing result of the same symbolic expression belongs to the
            # original task, not the new candidate coordinate.
            for node in spec.nodes:
                value = node.meta.get("val")
                if isinstance(value, torch.SymInt):
                    expression = value._sympy_()
                    assert isinstance(expression, sympy.Expr)
                    df.expr_to_var_info.pop(expression, None)
            args = []
            for leaf, axis in zip(spec.leaves, axes, strict=True):
                value = cg.lift(bindings[leaf])
                args.append(value)
                # The original memory lowering consults the physical grid maps,
                # not just its AST arguments. These are full scalar axes (no
                # lane or mask), and the original maps are restored below.
                grid.strategy.index_vars[axis] = value.id
                grid.strategy.offset_vars[axis] = value.id
                df.block_size_var_cache[(axis,)] = "1"
                symbolic = leaf.meta["val"]
                if isinstance(symbolic, torch.SymInt):
                    expression = symbolic._sympy_()
                    assert isinstance(expression, sympy.Expr)
                    df.expr_to_var_info[expression] = VarInfo(value.id, copied[leaf])
            values = GraphInterpreter(graph, cg).run(*args)
            expressions: list[ast.expr] = []
            for value in values:
                if type(value) is int:
                    value = expr_from_string(str(value))
                if not isinstance(value, ast.expr):
                    raise UnsupportedWorkOrder(
                        "scalar lowering did not return scalar expressions"
                    )
                expressions.append(value)
    finally:
        cg._track_statement_owners = tracked
        grid.strategy.index_vars.clear()
        grid.strategy.index_vars.update(indices)
        grid.strategy.offset_vars.clear()
        grid.strategy.offset_vars.update(offsets)
        df.block_size_var_cache.clear()
        df.block_size_var_cache.update(sizes)
        df.device_load_index = load_index
        if len(df.arguments) != len(arguments) or any(
            left is not right
            for left, right in zip(df.arguments, arguments, strict=True)
        ):
            raise UnsupportedWorkOrder(
                "scalar projection changed original host arguments"
            )
        if len(df._tensor_args) != len(tensors) or any(
            df._tensor_args.get(key) is not value for key, value in tensors
        ):
            raise UnsupportedWorkOrder(
                "scalar projection changed original tensor bindings"
            )
        spec.check()
    body: list[ast.stmt] = []
    for statement in statements:
        if not isinstance(statement, ast.stmt):
            raise UnsupportedWorkOrder("scalar lowering emitted a non-statement")
        body.append(statement)
    return ScalarProjection(tuple(body), tuple(expressions))


def _static(value: object) -> int:
    env = CompileEnvironment.current()
    if isinstance(value, torch.SymInt):
        value = value._sympy_()
    if isinstance(value, sympy.Expr):
        value = env.specialize_expr(value)
    if type(value) is int or isinstance(value, sympy.Integer):
        return int(value)
    raise UnsupportedWorkOrder("work ordering requires a static task domain")


def _grid_axis(node: Node) -> int | None:
    value = node.meta.get("val")
    if not isinstance(value, torch.SymInt):
        return None
    origin = HostFunction.current().expr_to_origin.get(value._sympy_())
    if origin is None or type(origin.origin) not in (GridOrigin, TileIdOrigin):
        return None
    assert isinstance(origin.origin, GridOrigin)
    return CompileEnvironment.current().canonical_block_id(origin.origin.block_id)


def register_work_order_aliases(ir: DeviceIR) -> None:
    """Register existing storage-only facts before the original bound-key cut.

    This is a conservative opportunity filter, not config/geometry admission.
    No tensor contents are examined and no bound snapshot is refreshed later.
    """
    if len(ir.root_ids) != 1 or len(ir.task_families) != 1:
        return
    config = CompileEnvironment.current().config_spec
    config.cute_work_order_axes = ir.task_families[0].logical_axis_order
    root = ir.graphs[ir.root_ids[0]].graph
    loops = [
        node for node in root.nodes if _tracing_ops.is_for_loop_target(node.target)
    ]
    if len(loops) != 1:
        return
    begin, end = loops[0].args[1:3]
    if (
        not isinstance(begin, (tuple, list))
        or not isinstance(end, (tuple, list))
        or len(begin) != 1
        or len(end) != 1
    ):
        return
    try:
        outputs = (begin[0], end[0])
        key = scalar_slice(root, outputs, _coordinate_leaves(root, outputs))
        axes = {_grid_axis(node) for node in key.nodes}
        eligible = tuple(
            item.block_id
            for item in ir.task_families[0].axes
            if item.canonical_origin
            and item.block_id in axes
            and 2 <= _static(item.extent) <= 32
        )
    except UnsupportedWorkOrder:
        return
    if eligible and any(node.target is memory_ops.load for node in key.nodes):
        config.cute_work_order_candidates = eligible
        register_cute_tensor_alias_specializations(CompileEnvironment.current())


def _coordinate_leaves(
    graph: Graph, outputs: tuple[Node | int, ...]
) -> tuple[Node, ...]:
    needed: set[Node] = set()
    pending = [value for value in outputs if isinstance(value, Node)]
    while pending:
        node = pending.pop()
        if node not in needed:
            needed.add(node)
            pending.extend(node.all_input_nodes)
    return tuple(
        node for node in graph.nodes if node in needed and _grid_axis(node) is not None
    )


def _metadata_domain(spec: ScalarSlice, domain: CoordinateDomain) -> None:
    """Prove the original scalar load addresses for every candidate, not its data.

    Only original direct, unmasked integer metadata loads are admitted initially.
    Their index grammar is bounded by the existing coordinate relation analyzer.
    No mask is removed, no tensor content becomes a host specialization.
    """
    axes = domain.axis_count_expressions
    substitutions = {}
    for node in spec.nodes:
        if (axis := _grid_axis(node)) is not None:
            if axis not in axes or domain.block_sizes[axis] != 1:
                raise UnsupportedWorkOrder("metadata index is not a scalar task axis")
            substitutions[node.meta["val"]._sympy_()] = coordinate_axis_symbol(axis)
    bounds = tuple((axis, 0, _static(count), 1) for axis, count in axes.items())
    for node in spec.nodes:
        if node.target is not memory_ops.load:
            continue
        if len(node.args) != 4 or node.args[2:] != (None, None) or node.kwargs:
            raise UnsupportedWorkOrder(
                "work metadata needs an original full scalar load"
            )
        source, indices = node.args[:2]
        assert isinstance(source, Node)
        tensor = source.meta["val"]
        assert isinstance(tensor, torch.Tensor)
        if not isinstance(indices, (list, tuple)) or len(indices) != tensor.ndim:
            raise UnsupportedWorkOrder(
                "work metadata needs complete direct coordinates"
            )
        largest_offset = 0
        for index, extent, stride in zip(
            indices, tensor.shape, tensor.stride(), strict=True
        ):
            value = index.meta.get("val") if isinstance(index, Node) else index
            if isinstance(value, torch.SymInt):
                expression = value._sympy_()
            elif type(value) is int:
                expression = sympy.Integer(value)
            else:
                raise UnsupportedWorkOrder(
                    "indirect metadata coordinates are unsupported"
                )
            expression = expression.xreplace(substitutions)
            assert isinstance(expression, sympy.Expr)
            _rewritten, interval = _analyze_integer_expression(
                expression, domain=domain, source_bounds=bounds
            )
            if interval is None:
                raise UnsupportedWorkOrder(
                    "metadata coordinate has no complete domain proof"
                )
            low, high = map(_static, interval)
            size, spacing = _static(extent), _static(stride)
            if not 0 <= low <= high < size or spacing < 0:
                raise UnsupportedWorkOrder(
                    "metadata coordinate escapes its full tensor"
                )
            largest_offset += high * spacing
        # The unchanged scalar memory renderer uses Int32 pointer arithmetic.
        if largest_offset > 2**31 - 1:
            raise UnsupportedWorkOrder(
                "metadata address exceeds original index arithmetic"
            )


@dataclasses.dataclass(frozen=True)
class WorkOrderPlan:
    """A permutation of one original parallel task axis, before any body effect.

    This does not define an inter-CTA schedule. The original single parallel
    TaskFamily remains unordered; dependent roots, persistent/worklist routes,
    atomics and extra control-flow regions are not admitted. The new speculative
    metadata reads additionally require full-domain and original runtime alias
    proofs. The launch must still pass the final full-warp check before use.
    """

    axis: int
    count: int
    domain: CoordinateDomain
    scalar: ScalarSlice
    leaf_axes: tuple[int, ...]
    loop: Node
    step: int
    # Ordinary CuTe range_str explicitly narrows both bounds to Int32. The
    # accepted common loop passes its original scalar bounds to cutlass.range.
    # Retain this physical distinction, not a raw endpoint distance.
    range32: bool
    context: tuple[object, ...] = dataclasses.field(repr=False)

    def check(self, cg: GenerateAST) -> None:
        self.scalar.check()
        current = discover_work_order(cg, self.axis)
        if self != current:
            raise UnsupportedWorkOrder("work-order domain, range or effects changed")


def discover_work_order(cg: GenerateAST, axis: int) -> WorkOrderPlan:
    env = CompileEnvironment.current()
    df = cg.device_function
    ir = HostFunction.current().device_ir
    root = cg.current_root_graph_info
    if (
        env.backend_name != "cute"
        or type(df.pid) is not FlatProgramIDs
        or env.compact_worklist_plan is not None
        or len(ir.root_ids) != 1
        or len(ir.task_families) != 1
        or root is None
        or root.graph_id != ir.root_ids[0]
        or ir.implicit_dependency_starts
        or (ir.tile_dependency_graph is not None and ir.tile_dependency_graph.edges)
        or any(
            df.config.get(key, 1) != 1
            for key in ("tcgen05_cluster_m", "tcgen05_cluster_n")
        )
    ):
        raise UnsupportedWorkOrder(
            "work ordering requires one original flat independent root"
        )
    family = ir.task_families[0]
    extents = tuple(_static(item.extent) for item in family.axes)
    blocks = tuple(
        _static(df.resolved_block_size(item.block_id)) for item in family.axes
    )
    if (
        axis not in family.logical_axis_order
        or not all(item.canonical_origin for item in family.axes)
        or any(type(block) is not int or block <= 0 for block in blocks)
        or any(size <= 0 for size in extents)
    ):
        raise UnsupportedWorkOrder(
            "work ordering needs complete canonical task geometry"
        )
    counts = tuple(
        (size + block - 1) // block for size, block in zip(extents, blocks, strict=True)
    )
    position = family.logical_axis_order.index(axis)
    if (
        blocks[position] != 1
        or not 2 <= counts[position] <= 32
        or math.prod(counts) >= 2**31
    ):
        raise UnsupportedWorkOrder(
            "work ordering needs a bounded scalar axis of 2..32 tasks"
        )
    domain = CoordinateDomain(
        family.logical_axis_order,
        tuple(zip(family.logical_axis_order, counts, strict=True)),
        tuple(zip(family.logical_axis_order, blocks, strict=True)),
        kind="task_order",
    )
    loops = [
        node
        for node in root.graph.nodes
        if _tracing_ops.is_for_loop_target(node.target)
    ]
    bodies = [info for info in cg.codegen_graphs if isinstance(info, ForLoopGraphInfo)]
    if (
        len(loops) != 1
        or len(bodies) != 1
        or any(
            not isinstance(
                info, (RootGraphInfo, ForLoopGraphInfo, HelperFunctionGraphInfo)
            )
            for info in cg.codegen_graphs
        )
    ):
        raise UnsupportedWorkOrder("work ordering requires one original counted loop")
    loop, body = loops[0], bodies[0]
    if loop.args[0] != body.graph_id or len(body.block_ids) != 1:
        raise UnsupportedWorkOrder("work-order loop binding changed")
    begins, ends = loop.args[1:3]
    if (
        not isinstance(begins, (tuple, list))
        or not isinstance(ends, (tuple, list))
        or len(begins) != 1
        or len(ends) != 1
    ):
        raise UnsupportedWorkOrder("work ordering requires one scalar range")
    outputs = (begins[0], ends[0])
    if any(not isinstance(value, Node) and type(value) is not int for value in outputs):
        raise UnsupportedWorkOrder("work-order bounds must retain original scalar SSA")
    # Discover all coordinate leaves actually reached by these original bounds.
    leaves = _coordinate_leaves(root.graph, outputs)
    leaf_axes = tuple(item for node in leaves if (item := _grid_axis(node)) is not None)
    if axis not in leaf_axes or len(leaf_axes) != len(leaves):
        raise UnsupportedWorkOrder(
            "work range does not depend on the selected task axis"
        )
    scalar = scalar_slice(root.graph, outputs, leaves)
    _metadata_domain(scalar, domain)
    chained = df.cute_state.chained_matmul_plan
    if chained is not None:
        if (
            chained.loop is None
            or chained.loop.call is not loop
            or chained.loop.body is not body
        ):
            raise UnsupportedWorkOrder(
                "work order is not bound to the accepted common loop"
            )
        step = _static(df.resolved_block_size(body.block_ids[0]))
    else:
        strategies = [
            strategy
            for strategy in df.tile_strategy.strategies
            if body.block_ids[0] in strategy.block_ids
        ]
        if len(strategies) != 1 or not isinstance(strategies[0], _BaseNDTileStrategy):
            raise UnsupportedWorkOrder(
                "work order needs the original ND counted-range lowering"
            )
        if len(loop.args) == 5:
            explicit = loop.args[4]
            if not isinstance(explicit, (tuple, list)) or len(explicit) != 1:
                raise UnsupportedWorkOrder("work order requires one original step")
            step = explicit[0]
            if step in (None, 1):
                step = _static(df.resolved_block_size(body.block_ids[0]))
        else:
            step = _static(df.resolved_block_size(body.block_ids[0]))
    if type(step) is not int or not 0 < step < 2**31:
        raise UnsupportedWorkOrder(
            "work ordering requires a positive original static step"
        )
    nodes = tuple(node for info in cg.codegen_graphs for node in info.graph.nodes)
    stores = tuple(node for node in nodes if node.target is memory_ops.store)
    if not stores or any(
        node.is_impure()
        and node.op not in ("placeholder", "output")
        and node.target not in (memory_ops.store, _tracing_ops._phi)
        and not _tracing_ops.is_for_loop_target(node.target)
        for node in nodes
    ):
        raise UnsupportedWorkOrder("work ordering cannot reorder an unbound effect")
    metadata = tuple(
        node.meta["val"]
        for node in scalar.nodes
        if node.target is _tracing_ops._host_tensor
    )
    writer_facts = []
    for store in stores:
        source = store.args[0]
        if (
            not isinstance(source, Node)
            or source.target is not _tracing_ops._host_tensor
        ):
            raise UnsupportedWorkOrder(
                "work ordering requires original explicit write owners"
            )
        writer = source.meta["val"]
        assert isinstance(writer, torch.Tensor)
        for tensor in metadata:
            if writer.untyped_storage() in env.fresh_allocation_storages:
                if writer.untyped_storage()._cdata == tensor.untyped_storage()._cdata:
                    raise UnsupportedWorkOrder(
                        "work metadata aliases a fresh write owner"
                    )
            else:
                # Match the original loop storage resolver: wrapper reshapes
                # may have no direct Source, but retain the exact allocation.
                # Check every visible input Source of that allocation; names
                # and distinct FakeTensor objects are never disjointness facts.
                readers = tuple(
                    argument
                    for value, argument in env.input_sources.items()
                    if value.untyped_storage()._cdata == tensor.untyped_storage()._cdata
                )
                writers = tuple(
                    argument
                    for value, argument in env.input_sources.items()
                    if value.untyped_storage()._cdata == writer.untyped_storage()._cdata
                )
                if (
                    not readers
                    or not writers
                    or any(
                        not runtime_tensor_sources_are_proven_disjoint(env, left, right)
                        for left in readers
                        for right in writers
                    )
                ):
                    raise UnsupportedWorkOrder(
                        "work metadata lacks the original runtime disjointness proof"
                    )
        writer_facts.append(
            (
                source,
                id(writer),
                tuple(writer.shape),
                tuple(writer.stride()),
                writer.storage_offset(),
            )
        )
    context = (
        root,
        family,
        tuple(ir.root_ids),
        tuple(cg.codegen_graphs),
        tuple((node, _node_fact(node)) for node in nodes),
        tuple(writer_facts),
        tuple(
            (
                id(tensor),
                tensor.dtype,
                tuple(tensor.shape),
                tuple(tensor.stride()),
                tensor.storage_offset(),
            )
            for tensor in metadata
        ),
        tuple((id(arg), arg, arg.name, arg.host_str()) for arg in df.arguments),
        tuple(
            (id(tensor), tensor, id(arg), arg, arg.name, arg._host_str)
            for tensor, arg in df._tensor_args.items()
        ),
        ir.tile_dependency_graph,
        chained,
        step,
    )
    return WorkOrderPlan(
        axis,
        counts[position],
        domain,
        scalar,
        leaf_axes,
        loop,
        step,
        chained is None,
        context,
    )


@dataclasses.dataclass(frozen=True)
class WorkPermutation:
    statements: tuple[ast.stmt, ...]
    selected: ast.Name


def emit_work_permutation(
    cg: GenerateAST, plan: WorkOrderPlan, coordinates: Mapping[int, ast.expr]
) -> WorkPermutation:
    """Use one shared warp algorithm over the original projected range SSA.

    The caller must insert this before every coordinate-dependent body effect,
    and final launch admission must establish full active warps. No original
    tensor load/mask/cast or range body is rendered here.
    """
    plan.check(cg)
    if set(coordinates) != set(plan.leaf_axes) or not all(
        isinstance(value, ast.expr) for value in coordinates.values()
    ):
        raise UnsupportedWorkOrder("work-order physical coordinate bindings changed")
    df = cg.device_function
    candidate = df.new_var("work_candidate")
    projected = project_scalar_slice(
        cg,
        plan.scalar,
        {
            leaf: expr_from_string(candidate)
            if axis == plan.axis
            else coordinates[axis]
            for leaf, axis in zip(plan.scalar.leaves, plan.leaf_axes, strict=True)
        },
    )
    begin, end = map(ast.unparse, projected.values)
    if plan.range32:
        # Exactly the ordinary CuTe range_str contract, including truncation.
        begin, end = f"cutlass.Int32({begin})", f"cutlass.Int32({end})"
    start = df.new_var("work_begin")
    stop = df.new_var("work_end")
    distance = df.new_var("work_distance")
    key = df.new_var("work_trips")
    peer = df.new_var("work_peer")
    peer_key = df.new_var("work_peer_trips")
    rank = df.new_var("work_rank")
    selected = df.new_var("work_selected")
    original = ast.unparse(coordinates[plan.axis])
    prefix = ast.parse(
        f"{candidate} = cutlass.Int32(cute.arch.lane_idx()) % {plan.count}"
    ).body
    ranking = ast.parse(f"""
{start} = cutlass.Int64({begin})
{stop} = cutlass.Int64({end})
{key} = cutlass.Uint64(0)
if {stop} > {start}:
    {distance} = cutlass.Uint64({stop}) - cutlass.Uint64({start})
    {key} = {distance} // cutlass.Uint64({plan.step}) + cutlass.Uint64({distance} % cutlass.Uint64({plan.step}) != 0)
{rank} = cutlass.Int32(0)
for {peer} in cutlass.range_constexpr({plan.count}):
    {peer_key} = cute.arch.shuffle_sync({key}, {peer})
    {rank} += cutlass.Int32(({peer_key} > {key}) | (({peer_key} == {key}) & ({peer} < {candidate})))
{selected} = cute.arch.warp_reduction_max({candidate} if {rank} == {original} else cutlass.Int32(-1))
""").body
    # Unsigned subtraction computes the exact positive distance even across
    # signed zero, without signed overflow or the overflowing d+step-1 idiom.
    # Comparisons use the full 64-bit count; ties use the original candidate ID.
    return WorkPermutation(
        (*prefix, *projected.statements, *ranking),
        ast.Name(id=selected, ctx=ast.Load()),
    )


def requested_work_axis(cg: GenerateAST) -> int | None:
    policies = cg.device_function.config.get("cute_grid_work_order")
    if policies is None:
        return None
    if not isinstance(policies, (tuple, list)):
        raise UnsupportedWorkOrder("invalid normalized work-order policies")
    axes = CompileEnvironment.current().config_spec.cute_work_order_axes
    selected = [
        axis
        for axis, policy in zip(axes, policies, strict=True)
        if policy == "longest_first"
    ]
    if len(selected) != 1:
        raise UnsupportedWorkOrder("invalid normalized work-order request")
    return selected[0]


def install_ordinary_work_order(cg: GenerateAST) -> tuple[ast.stmt, ...]:
    """Late ordinary-body insertion, after its real arguments and grid exist."""
    try:
        axis = requested_work_axis(cg)
        if axis is None:
            return ()
        plan = discover_work_order(cg, axis)
        grid = cg.current_grid_state
        assert isinstance(grid, DeviceGridState)
        origins = {
            item: ast.Name(id=grid.strategy.index_var(item), ctx=ast.Load())
            for item in plan.leaf_axes
        }
        permutation = emit_work_permutation(cg, plan, origins)
        names = tuple(
            dict.fromkeys(
                (grid.strategy.index_var(axis), grid.strategy.offset_var(axis))
            )
        )
        assignments = tuple(
            ast.fix_missing_locations(
                ast.Assign(
                    targets=[ast.Name(id=name, ctx=ast.Store())],
                    value=permutation.selected,
                )
            )
            for name in names
        )
        if cg.device_function.cute_state.work_order_plan is not None:
            raise UnsupportedWorkOrder("multiple work-order body installations")
        cg.device_function.cute_state.work_order_plan = plan
        return (*permutation.statements, *assignments)
    except UnsupportedWorkOrder as error:
        raise exc.BackendUnsupported("cute", str(error)) from error


def validate_work_order_launch(cg: GenerateAST, block_arg: str) -> None:
    """Use the unchanged launcher's actual dimensions, never config num_warps."""
    plan = cg.device_function.cute_state.work_order_plan
    requested = requested_work_axis(cg)
    if requested is None:
        if plan is not None:
            raise exc.BackendUnsupported(
                "cute", "work-order request changed after emission"
            )
        return
    if (
        plan is None
        or plan.axis != requested
        or cg.device_function.cute_state.direct_affine_plan is not None
    ):
        raise exc.BackendUnsupported(
            "cute", "work ordering was not installed by the original scalar/common root"
        )
    plan.scalar.check()
    expression = ast.parse(block_arg).body[0]
    if not isinstance(expression, ast.Assign) or not isinstance(
        expression.value, ast.Tuple
    ):
        raise exc.BackendUnsupported(
            "cute", "work ordering requires a static full-warp launch"
        )
    values = expression.value.elts
    dimensions = [
        value.value
        for value in values
        if isinstance(value, ast.Constant) and type(value.value) is int
    ]
    if len(values) != 3 or len(dimensions) != 3:
        raise exc.BackendUnsupported(
            "cute", "work ordering requires a static full-warp launch"
        )
    x, y, z = dimensions
    if x < 32 or x % 32 or y != 1 or z != 1:
        raise exc.BackendUnsupported(
            "cute", "work ordering requires original full active warps"
        )
