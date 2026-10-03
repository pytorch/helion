"""Distribute row-separable materialized regions into ordered CuTe launches.

The opt-in analysis uses a temporary compilation environment. The accepted AST
is then compiled normally in the owning BoundKernel, giving each root independent
axis/config identities while preserving the original input guards and allocations.
"""

from __future__ import annotations

import ast
import copy
import dataclasses
import itertools
import operator
from typing import TYPE_CHECKING
from typing import cast

import torch

from ... import language as hl
from ...language import _tracing_ops
from ...language import creation_ops
from ...language import memory_ops
from ...language import tile_ops
from ...language.constexpr import ConstExpr
from ..ast_extension import ExtendedAST
from ..ast_extension import LoopType
from ..ast_extension import create
from ..compile_environment import CompileEnvironment
from ..device_ir import ForLoopGraphInfo
from ..indexing_strategy import subscript_tile_info
from ..kernel_compiler import KernelCompiler
from ..type_info import IterType
from ..type_info import TensorType
from ..type_info import TileIndexType
from ..variable_origin import ArgumentOrigin
from ..variable_origin import GridOrigin
from .promote_output_axis import _FRESH_FACTORIES
from .promote_output_axis import _access_tensor
from .promote_output_axis import _canonical_iterator
from .promote_output_axis import _empty_output
from .promote_output_axis import _fresh_tensors
from .promote_output_axis import _indices
from .promote_output_axis import _ordinary_operation

if TYPE_CHECKING:
    from typing import Any

    import sympy

    from ...runtime.kernel import Kernel
    from ..device_ir import DeviceIR
    from ..host_function import HostFunction


@dataclasses.dataclass(frozen=True)
class MaterializedFissionPlan:
    root_index: int
    source_root_key: str
    materialized_names: tuple[str, ...]
    region_count: int
    # A backend operand-materialization proof can supply complete ordered
    # roots while reusing the allocation/launch/cache bundle below.
    source_root_index: int | None = None
    replacement_body: tuple[ast.stmt, ...] | None = None
    # These complete, separately launched roots are proven pointwise by the
    # extractor. Their SIMT layouts remain tunable beside a native MMA region.
    pointwise_region_indices: tuple[int, ...] = ()
    # Populated only by the original typed operand-materialization proof.
    # Optional fused schedules must still recheck the owning producer/consumer.
    operand_interleave_factor: int | None = None


class _Ineligible(Exception):
    pass


_ANALYSIS_CALLS = (
    *_FRESH_FACTORIES,
    hl.tile,
    hl.load,
    hl.store,
    hl.zeros,
    hl.full,
    hl.arange,
    hl.dot,
    torch.mm,
    torch.matmul,
    torch.addmm,
    torch.add,
    torch.sub,
    torch.mul,
    torch.div,
    torch.abs,
    torch.neg,
    torch.relu,
    torch.sigmoid,
    torch.tanh,
    torch.exp,
    torch.log,
    torch.sqrt,
    torch.rsqrt,
    torch.where,
    torch.minimum,
    torch.maximum,
    torch.clamp,
    torch.flip,
)
_ANALYSIS_METHODS = frozenset(
    (
        "size",
        "stride",
        "dim",
        "ndimension",
        "to",
        "float",
        "half",
        "bfloat16",
        "reshape",
        "view",
        "transpose",
        "permute",
        "contiguous",
        "sum",
        "mean",
        "amax",
        "amin",
        "max",
        "min",
        "relu",
        "sigmoid",
        "tanh",
        "exp",
        "log",
    )
)


def _safe_analysis_source(host: HostFunction) -> bool:
    """Only retrace syntax whose evaluation cannot run opaque Python effects.

    The FX proof below remains authoritative for memory and row separability.
    This earlier gate prevents even a rejected analysis from executing a user's
    Python helper or allocation callback an extra time.
    """
    roots = [
        (index, stmt)
        for index, stmt in enumerate(host.body)
        if isinstance(stmt, ast.For)
    ]
    if len(roots) != 1:
        return False
    root_index, root = roots[0]
    if len(root.body) < 2 or not all(isinstance(stmt, ast.For) for stmt in root.body):
        return False
    locals_ = set(host.params.arguments)
    locals_.update(
        node.id
        for stmt in host.body
        for node in ast.walk(stmt)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    )

    def global_value(node: ast.AST) -> object:
        if isinstance(node, ast.Name) and node.id not in locals_:
            return host.fn.__globals__.get(node.id)
        if isinstance(node, ast.Attribute):
            parent = global_value(node.value)
            if parent is torch or parent is hl:
                return vars(parent).get(node.attr)
        return None

    def local_receiver(node: ast.AST) -> bool:
        if isinstance(node, ast.Name):
            return node.id in locals_
        if isinstance(node, ast.Subscript):
            return local_receiver(node.value)
        if isinstance(node, ast.Attribute):
            return node.attr in ("T", "mT", "real", "imag", "data") and local_receiver(
                node.value
            )
        if isinstance(node, ast.Call):
            return any(
                global_value(node.func) is known for known in _ANALYSIS_CALLS
            ) or (
                isinstance(node.func, ast.Attribute)
                and node.func.attr in _ANALYSIS_METHODS
                and local_receiver(node.func.value)
            )
        return False

    def known_global(value: object) -> bool:
        return (
            value is torch
            or value is hl
            or type(value)
            in (
                int,
                float,
                bool,
                str,
                type(None),
                torch.dtype,
                torch.device,
                torch.layout,
                torch.memory_format,
            )
            or any(value is known for known in _ANALYSIS_CALLS)
        )

    tensor_names = {
        name
        for name, value in host.params.arguments.items()
        if isinstance(value, torch.Tensor)
    }
    for stmt in host.body[:root_index]:
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
            continue
        if not isinstance(stmt, ast.Assign):
            return False
        if not all(
            isinstance(target, ast.Name)
            or isinstance(target, (ast.Tuple, ast.List))
            and all(isinstance(item, ast.Name) for item in target.elts)
            for target in stmt.targets
        ):
            return False
        if isinstance(stmt.value, ast.Call) and any(
            global_value(stmt.value.func) is factory for factory in _FRESH_FACTORIES
        ):
            if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
                return False
            tensor_names.add(stmt.targets[0].id)
        elif not _shape_only(stmt.value, tensor_names):
            return False
    for stmt in host.body:
        for node in ast.walk(stmt):
            if isinstance(node, ast.Call):
                callee = global_value(node.func)
                if any(callee is known for known in _ANALYSIS_CALLS):
                    continue
                if (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr in _ANALYSIS_METHODS
                    and local_receiver(node.func.value)
                ):
                    continue
                return False
            if (
                isinstance(node, ast.Name)
                and isinstance(node.ctx, ast.Load)
                and node.id not in locals_
            ):
                value = global_value(node)
                if known_global(value):
                    continue
                return False
            if isinstance(node, ast.Attribute) and not local_receiver(node.value):
                value = global_value(node)
                if value is None or not known_global(value):
                    return False
            if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
                targets = (
                    node.targets if isinstance(node, ast.Assign) else [node.target]
                )
                if any(isinstance(target, ast.Attribute) for target in targets):
                    return False
    return True


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise _Ineligible(reason)


def _value_has_row(value: object, symbol: sympy.Symbol) -> bool:
    if isinstance(value, torch.Tensor):
        return any(
            isinstance(size, torch.SymInt) and symbol in size._sympy_().free_symbols
            for size in value.shape
        )
    if isinstance(value, (tuple, list)):
        return any(_value_has_row(item, symbol) for item in value)
    return False


def _canonical_row_shape(value: object, symbol: sympy.Symbol) -> bool:
    if isinstance(value, torch.Tensor):
        row_dims = [
            index
            for index, size in enumerate(value.shape)
            if isinstance(size, torch.SymInt) and symbol in size._sympy_().free_symbols
        ]
        return not row_dims or (
            row_dims == [0]
            and isinstance(value.shape[0], torch.SymInt)
            and value.shape[0]._sympy_() == symbol
        )
    if isinstance(value, (tuple, list)):
        return all(_canonical_row_shape(item, symbol) for item in value)
    return True


def _contains_node(value: object, node: torch.fx.Node) -> bool:
    if isinstance(value, (tuple, list)):
        return any(_contains_node(item, node) for item in value)
    return value is node


def _row_separable(
    host: HostFunction,
    graph_ids: tuple[int, ...],
    row_block_id: int,
    env: CompileEnvironment,
) -> bool:
    symbol = env.block_sizes[row_block_id].symbol()
    for graph_id in graph_ids:
        for node in host.device_ir.graphs[graph_id].graph.nodes:
            if node.target in (
                tile_ops.tile_begin,
                tile_ops.tile_end,
                tile_ops.tile_id,
            ):
                return False
            value = node.meta.get("val")
            if not _canonical_row_shape(value, symbol):
                return False
            if isinstance(value, torch.SymInt):
                for variable in value._sympy_().free_symbols:
                    origin = host.expr_to_origin.get(cast("sympy.Basic", variable))
                    if origin is not None and isinstance(origin.origin, GridOrigin):
                        return False
                if symbol in value._sympy_().free_symbols:
                    for user in node.users:
                        in_memory_indices = user.target in (
                            memory_ops.load,
                            memory_ops.store,
                        ) and _contains_node(user.args[1], node)
                        in_shape = user.target is creation_ops.full and _contains_node(
                            user.args[0], node
                        )
                        if not (in_memory_indices or in_shape):
                            return False
            if not any(
                _value_has_row(arg.meta.get("val"), symbol)
                for arg in node.all_input_nodes
            ):
                continue
            target = node.target
            if target in (
                memory_ops.load,
                memory_ops.store,
                _tracing_ops._new_var,
                _tracing_ops._phi,
                _tracing_ops._for_loop,
            ):
                continue
            if node.op == "output":
                continue
            if target is operator.getitem:
                # Access a loop-result tuple, never index across a tensor's rows.
                if not isinstance(node.args[0], torch.fx.Node) or isinstance(
                    node.args[0].meta.get("val"), torch.Tensor
                ):
                    return False
                continue
            if target is torch.ops.aten.sym_size.int:
                continue
            if (
                isinstance(target, torch._ops.OpOverload)
                and torch.Tag.pointwise in target.tags
            ):
                continue
            if target in (torch.ops.aten.mm.default, torch.ops.aten.addmm.default):
                positions = (0, 1) if target is torch.ops.aten.mm.default else (1, 2)
                lhs, rhs = (node.args[index] for index in positions)
                if not isinstance(lhs, torch.fx.Node) or not isinstance(
                    rhs, torch.fx.Node
                ):
                    return False
                lhs_value, rhs_value = lhs.meta.get("val"), rhs.meta.get("val")
                if (
                    not isinstance(lhs_value, torch.Tensor)
                    or not isinstance(rhs_value, torch.Tensor)
                    or not isinstance(value, torch.Tensor)
                ):
                    return False
                if (
                    lhs_value.ndim != 2
                    or value.ndim != 2
                    or env.resolve_block_id(lhs_value.shape[0]) != row_block_id
                    or env.resolve_block_id(value.shape[0]) != row_block_id
                    or _value_has_row(rhs_value, symbol)
                ):
                    return False
                continue
            return False
    return True


def _shape_only(node: ast.AST, tensor_names: set[str]) -> bool:
    if isinstance(node, (ast.Constant, ast.Name, ast.Load)):
        return True
    if isinstance(node, (ast.Tuple, ast.List)):
        return all(_shape_only(item, tensor_names) for item in node.elts)
    if isinstance(node, ast.Attribute):
        return (
            isinstance(node.value, ast.Name)
            and node.value.id in tensor_names
            and node.attr in ("shape", "dtype", "device", "ndim")
        )
    if isinstance(node, ast.Subscript):
        return _shape_only(node.value, tensor_names) and _shape_only(
            node.slice, tensor_names
        )
    if isinstance(node, ast.Call):
        return (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in tensor_names
            and node.func.attr in ("size", "stride", "dim", "ndimension")
            and not node.keywords
            and all(_shape_only(arg, tensor_names) for arg in node.args)
        )
    if isinstance(node, ast.BinOp):
        return (
            isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod))
            and _shape_only(node.left, tensor_names)
            and _shape_only(node.right, tensor_names)
        )
    return False


def _is_contiguous_2d(tensor: torch.Tensor, env: CompileEnvironment) -> bool:
    return (
        tensor.ndim == 2
        and env.known_equal(tensor.stride(1), 1)
        and env.known_equal(tensor.stride(0), tensor.size(1))
    )


def _exact_tile(index: object, block_id: int, env: CompileEnvironment) -> bool:
    info = subscript_tile_info(env, index)
    return (
        info is not None
        and info.block_id == block_id
        and env.known_equal(info.offset, 0)
        and info.block_size is None
    )


def _covered_column(
    index: object, tensor: torch.Tensor, env: CompileEnvironment
) -> bool:
    if isinstance(index, slice):
        return index.start is None and index.stop is None and index.step is None
    info = subscript_tile_info(env, index)
    if info is None:
        return False
    extent = env.block_sizes[info.block_id].size
    return (
        env.known_equal(info.offset, 0)
        and info.block_size is None
        and isinstance(extent, (int, torch.SymInt))
        and env.known_equal(extent, tensor.size(1))
    )


def _descendants(host: HostFunction, graph_id: int) -> tuple[int, ...]:
    pending = [graph_id]
    result: list[int] = []
    while pending:
        current = pending.pop()
        _require(current not in result, "shared/cyclic nested graphs")
        result.append(current)
        for node in host.device_ir.graphs[current].graph.nodes:
            _require(_ordinary_operation(node), "opaque or impure region operation")
            _require(node.target is not _tracing_ops._if, "conditional region")
            if _tracing_ops.is_for_loop_target(node.target):
                _require(node.target is _tracing_ops._for_loop, "strided nested loop")
                _require(
                    all(
                        isinstance(value, int) and value == 0 for value in node.args[1]
                    ),
                    "nonzero nested loop origin",
                )
                child = node.args[0]
                assert isinstance(child, int)
                pending.append(child)
    return tuple(result)


def _single_tile_binding(loop: ast.For) -> bool:
    # Assignment targets are not type-annotated. Inspect the iterator's value
    # instead, so a name receiving hl.tile([extent]) is not flattened to a tile.
    return (
        isinstance(loop.target, ast.Name)
        and isinstance(loop.iter, ExtendedAST)
        and isinstance(info := loop.iter._type_info, IterType)
        and isinstance(info.inner, TileIndexType)
    )


def _prove_materialized_fission(
    host: HostFunction, env: CompileEnvironment, source_roots: dict[int, str]
) -> MaterializedFissionPlan:
    _require(env.backend.name == "cute", "CuTe-only extraction")
    ir = host.device_ir
    _require(len(ir.root_ids) == 1, "requires one original root")
    roots = [
        stmt
        for stmt in host.body
        if isinstance(stmt, ast.For)
        and isinstance(stmt, ExtendedAST)
        and stmt._loop_type is LoopType.GRID
    ]
    _require(len(roots) == 1, "requires one top-level source root")
    root = roots[0]
    root_index = host.body.index(root)
    indices = _indices(root)
    bounds = _canonical_iterator(root)
    _require(
        indices is not None
        and len(indices) == 1
        and isinstance(indices[0], TileIndexType)
        and bounds is not None
        and len(bounds) == 1
        and _single_tile_binding(root)
        and not root.orelse,
        "requires one canonical row tile",
    )
    assert indices is not None and bounds is not None
    row_block_id = indices[0].block_id
    row_size = env.block_sizes[row_block_id].size
    _require(
        isinstance(env.block_sizes[row_block_id].size, (int, torch.SymInt))
        and row_block_id not in ir.noncanonical_task_origin_block_ids,
        "noncanonical row ownership",
    )
    assert isinstance(row_size, (int, torch.SymInt))
    siblings = root.body
    _require(
        len(siblings) >= 2 and all(isinstance(stmt, ast.For) for stmt in siblings),
        "root must contain only sibling tile loops",
    )
    tail = host.body[root_index + 1 :]
    _require(
        len(tail) == 1 and isinstance(tail[0], ast.Return), "post-device host effects"
    )
    _require(
        not host.args.posonlyargs
        and not host.args.kwonlyargs
        and host.args.vararg is None
        and host.args.kwarg is None
        and not host.constexpr_args,
        "fission requires ordinary positional inputs",
    )
    argument_names = tuple(host.params.arguments)
    tensor_names = {
        name
        for name, value in host.params.arguments.items()
        if isinstance(value, torch.Tensor)
    }
    fresh = _fresh_tensors(host)
    capture_names: list[str] = []
    capture_values: list[torch.Tensor] = []
    for stmt in host.body[:root_index]:
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
            continue
        _require(isinstance(stmt, ast.Assign), "unsupported host prelude")
        assert isinstance(stmt, ast.Assign)
        _require(
            not any(
                isinstance(node, ast.Name)
                and isinstance(node.ctx, ast.Store)
                and node.id in (*argument_names, *capture_names)
                for target in stmt.targets
                for node in ast.walk(target)
            ),
            "host prelude rebinds an input or materialized allocation",
        )
        value_type = cast("ExtendedAST", stmt.value)._type_info
        if isinstance(value_type, TensorType):
            _require(
                isinstance(stmt.value, ast.Call) and value_type.proxy() in fresh,
                "aliased or unsupported host tensor allocation",
            )
            _require(
                len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name),
                "allocation must bind one name",
            )
            target = cast("ast.Name", stmt.targets[0]).id
            _require(
                target not in argument_names + tuple(capture_names), "rebound capture"
            )
            capture_names.append(target)
            capture_values.append(value_type.proxy())
            tensor_names.add(target)
        else:
            _require(
                all(
                    isinstance(target, ast.Name)
                    or isinstance(target, (ast.Tuple, ast.List))
                    and all(isinstance(item, ast.Name) for item in target.elts)
                    for target in stmt.targets
                ),
                "host prelude must assign local metadata names",
            )
            _require(_shape_only(stmt.value, tensor_names), "effectful host prelude")
    _require(bool(capture_names), "requires fresh materialized outputs")
    captured_by_storage = {
        value.untyped_storage()._cdata: (name, value)
        for name, value in zip(capture_names, capture_values, strict=True)
    }
    _require(len(captured_by_storage) == len(capture_values), "aliased captures")
    root_graph = ir.graphs[ir.root_ids[0]].graph
    top_calls = [
        node
        for node in root_graph.nodes
        if _tracing_ops.is_for_loop_target(node.target)
    ]
    _require(len(top_calls) == len(siblings), "source/IR region ownership mismatch")
    for node in root_graph.nodes:
        _require(_ordinary_operation(node), "effect outside extracted regions")
        _require(
            node.target not in (memory_ops.load, memory_ops.store, _tracing_ops._if),
            "memory or control outside extracted regions",
        )

    stores: dict[int, tuple[int, torch.fx.Node, torch.Tensor]] = {}
    accesses: list[tuple[int, torch.fx.Node, torch.Tensor]] = []
    column_ids: list[int] = []
    for region_index, (stmt, call) in enumerate(zip(siblings, top_calls, strict=True)):
        assert isinstance(stmt, ast.For)
        axes = _indices(stmt)
        ends = _canonical_iterator(stmt)
        _require(
            axes is not None
            and len(axes) == 1
            and isinstance(axes[0], TileIndexType)
            and ends is not None
            and len(ends) == 1
            and _single_tile_binding(stmt)
            and not stmt.orelse,
            "requires canonical sibling output columns",
        )
        assert axes is not None and ends is not None
        column_id = axes[0].block_id
        column_size = env.block_sizes[column_id].size
        _require(
            isinstance(env.block_sizes[column_id].size, (int, torch.SymInt))
            and column_id not in ir.noncanonical_task_origin_block_ids,
            "device-dependent or noncanonical columns",
        )
        assert isinstance(column_size, (int, torch.SymInt))
        graph_id = call.args[0]
        assert isinstance(graph_id, int)
        info = ir.graphs[graph_id]
        _require(
            isinstance(info, ForLoopGraphInfo)
            and info.block_ids == [column_id]
            and _empty_output(info.graph),
            "register result escapes the region",
        )
        assert isinstance(info, ForLoopGraphInfo)
        _require(not info.node_args, "region captures device registers")
        graph_ids = _descendants(host, graph_id)
        _require(
            _row_separable(host, graph_ids, row_block_id, env),
            "cross-row or tile-boundary computation",
        )
        column_ids.append(column_id)
        for child_id in graph_ids:
            for node in ir.graphs[child_id].graph.nodes:
                if node.target not in (memory_ops.load, memory_ops.store):
                    continue
                tensor = _access_tensor(node)
                _require(tensor is not None, "unknown memory allocation")
                assert tensor is not None
                accesses.append((region_index, node, tensor))
                if node.target is not memory_ops.store:
                    continue
                storage_id = tensor.untyped_storage()._cdata
                indices = node.args[1]
                _require(
                    tensor in fresh
                    and _is_contiguous_2d(tensor, env)
                    and env.known_equal(tensor.size(0), row_size)
                    and env.known_equal(tensor.size(1), column_size)
                    and isinstance(indices, (tuple, list))
                    and len(indices) == 2
                    and _exact_tile(indices[0], row_block_id, env)
                    and _exact_tile(indices[1], column_id, env)
                    and child_id == graph_id
                    and len(node.args) == 4
                    and node.args[3] is None
                    and storage_id not in stores,
                    "write is not one unconditional complete fresh row/column region",
                )
                stores[storage_id] = (region_index, node, tensor)
    _require(bool(stores), "no materialized stores")

    predecessors: list[set[int]] = [set() for stmt in siblings]
    for region_index, node, tensor in accesses:
        producer = stores.get(tensor.untyped_storage()._cdata)
        if producer is None:
            continue  # Original inputs are read-only, so runtime input aliasing is harmless.
        writer, store, stored_tensor = producer
        indices = node.args[1]
        _require(
            tensor is stored_tensor
            and isinstance(indices, (list, tuple))
            and len(indices) == 2
            and _exact_tile(indices[0], row_block_id, env),
            "alias or cross-row dependency",
        )
        assert isinstance(indices, (list, tuple))
        _require(writer <= region_index, "read-before-producer dependency")
        if writer < region_index:
            _require(
                _covered_column(indices[1], tensor, env),
                "consumer columns are not covered by the producer",
            )
            predecessors[region_index].add(writer)
        elif node.target is memory_ops.load:
            _require(
                _exact_tile(indices[1], column_ids[region_index], env)
                and node.graph is store.graph
                and list(node.graph.nodes).index(store)
                < list(node.graph.nodes).index(node),
                "same-region read is not dominated by the matching pointwise store",
            )
    _require(any(predecessors), "no materialized inter-region dependency")

    _require(root_index in source_roots, "root changed during tracing")
    return MaterializedFissionPlan(
        root_index=root_index,
        source_root_key=source_roots[root_index],
        materialized_names=tuple(capture_names),
        region_count=len(siblings),
    )


def plan_materialized_fission(
    kernel: Kernel[Any],
    args: tuple[object, ...],
    owning_env: CompileEnvironment,
) -> MaterializedFissionPlan | None:
    """Analyze in isolation before the owning compilation creates any block IDs."""
    if (
        not owning_env.settings.cute_region_fission
        or owning_env.backend_name != "cute"
        or owning_env._is_distributed
    ):
        return None
    if any(
        type(value)
        not in (torch.Tensor, torch.nn.Parameter, int, float, bool, type(None))
        or isinstance(value, ConstExpr)
        or annotation is ConstExpr
        for value, annotation in zip(args, kernel._annotations, strict=True)
    ):
        return None
    analysis_env = CompileEnvironment(
        owning_env.device,
        owning_env.settings,
        index_dtype=owning_env.index_dtype,
        is_distributed=False,
    )
    with analysis_env:
        fake_args = [
            analysis_env.to_fake(value, ArgumentOrigin(name))
            for name, value in zip(kernel.signature.parameters, args, strict=True)
        ]
        compiler = KernelCompiler(analysis_env)
        host = compiler.parse(kernel.fn, fake_args, {})
        with host, compiler._compilation_context():
            if not _safe_analysis_source(host):
                return None
            compiler.unroll(host)
            compiler.customize_ast(host)
            source_roots = {
                index: ast.dump(stmt)
                for index, stmt in enumerate(host.body)
                if isinstance(stmt, ast.For)
            }
            # Avoid tracing kernels without the necessary sibling-loop shape.
            if not any(
                isinstance(stmt, ast.For)
                and len(stmt.body) >= 2
                and all(isinstance(child, ast.For) for child in stmt.body)
                for stmt in host.body
            ):
                return None
            compiler.propagate_types(host)
            compiler.finalize_config()
            compiler.lower(host)
            try:
                return _prove_materialized_fission(host, analysis_env, source_roots)
            except _Ineligible:
                return None


def apply_materialized_fission(
    host: HostFunction, plan: MaterializedFissionPlan
) -> None:
    """Distribute the proven raw AST before normal type propagation and lowering."""
    source_index = (
        plan.root_index if plan.source_root_index is None else plan.source_root_index
    )
    root = host.body[source_index]
    assert isinstance(root, ast.For) and isinstance(root, ExtendedAST)
    assert ast.dump(root) == plan.source_root_key
    if plan.replacement_body is not None:
        host.body[:] = [
            cast("ast.stmt", _clone_untyped(statement))
            for statement in plan.replacement_body
        ]
        return
    assert len(root.body) == plan.region_count
    assert isinstance(root.iter, ast.Call)
    new_roots: list[ast.stmt] = []
    used_names = {
        node.id
        for stmt in host.body
        for node in ast.walk(stmt)
        if isinstance(node, ast.Name)
    }
    for region_index, sibling in enumerate(root.body):
        assert isinstance(sibling, ast.For) and isinstance(sibling, ExtendedAST)
        assert isinstance(sibling.iter, ast.Call) and isinstance(
            sibling.iter, ExtendedAST
        )
        with sibling:
            target = create(
                ast.Tuple, elts=[root.target, sibling.target], ctx=ast.Store()
            )
            bounds = create(
                ast.List, elts=[root.iter.args[0], sibling.iter.args[0]], ctx=ast.Load()
            )
            iterator = sibling.iter.copy(args=[bounds])
            # Clone root target/bounds too: type propagation assigns distinct axis
            # metadata for every resulting launch, including the shared row extent.
            distributed = cast("ast.For", sibling.copy(target=target, iter=iterator))
            distributed = cast("ast.For", _clone_untyped(distributed))
            # A promoted target becomes a host grid binding. Give each root
            # private names so a later region's serial loop may still reuse the
            # original source name (e.g. one GEMM's N is another GEMM's K).
            renames: dict[str, str] = {}
            assert isinstance(root.target, ast.Name)
            assert isinstance(sibling.target, ast.Name)
            for name in (root.target.id, sibling.target.id):
                renamed = f"_helion_region_{region_index}_{name}"
                while renamed in used_names:
                    renamed += "_"
                used_names.add(renamed)
                renames[name] = renamed
            for node in ast.walk(distributed):
                if isinstance(node, ast.Name) and node.id in renames:
                    node.id = renames[node.id]
            new_roots.append(distributed)
    host.body[plan.root_index : plan.root_index + 1] = new_roots


def forward_materialized_self_loads(host: HostFunction, device_ir: DeviceIR) -> None:
    """Expose same-tile dependencies while retaining their materialized stores.

    The fission proof establishes fresh, unaliased, complete output stores.
    Restrict forwarding to one graph, identical canonical indices, an unmasked
    load, and a store value already converted to the memory dtype. Only
    shape-preserving pointwise consumers may observe the forwarded value:
    out-of-bounds lanes lose the reload's zero padding, which reductions or
    matrix products could otherwise propagate into valid lanes. This keeps
    rounding observable and lets the existing MMA epilogue see every store.
    """
    env = CompileEnvironment.current()
    assert env.cute_fission_plan is not None
    fresh = _fresh_tensors(host)

    def key(node: torch.fx.Node) -> tuple[torch.Tensor, int, int] | None:
        tensor = _access_tensor(node)
        if tensor is None or tensor not in fresh or node.kwargs or len(node.args) != 4:
            return None
        indices = node.args[1]
        if not isinstance(indices, (list, tuple)) or len(indices) != 2:
            return None
        infos = [subscript_tile_info(env, index) for index in indices]
        if any(
            info is None or not _exact_tile(index, info.block_id, env)
            for index, info in zip(indices, infos, strict=True)
        ):
            return None
        assert infos[0] is not None and infos[1] is not None
        return tensor, infos[0].block_id, infos[1].block_id

    def same_shape(value: object, reference: torch.Tensor) -> bool:
        return (
            isinstance(value, torch.Tensor)
            and value.ndim == reference.ndim
            and all(
                itertools.starmap(
                    env.known_equal, zip(value.shape, reference.shape, strict=True)
                )
            )
        )

    def pointwise_uses(
        node: torch.fx.Node, access: tuple[torch.Tensor, int, int]
    ) -> bool:
        loaded = node.meta.get("val")
        if not isinstance(loaded, torch.Tensor):
            return False
        pending = [(node, user) for user in node.users]
        visited: set[torch.fx.Node] = set()
        while pending:
            operand, user = pending.pop()
            if user.target is memory_ops.store:
                destination = key(user)
                if (
                    destination is None
                    or destination[1:] != access[1:]
                    or user.args[2] is not operand
                    or user.args[3] is not None
                ):
                    return False
                continue
            if user in visited:
                continue
            if (
                user.op != "call_function"
                or not isinstance(user.target, torch._ops.OpOverload)
                or torch.Tag.pointwise not in user.target.tags
                or user.target._schema.is_mutable
                or not same_shape(user.meta.get("val"), loaded)
            ):
                return False
            visited.add(user)
            pending.extend((user, consumer) for consumer in user.users)
        return True

    for info in device_ir.graphs:
        stored: dict[tuple[torch.Tensor, int, int], torch.fx.Node] = {}
        for node in list(info.graph.nodes):
            if _tracing_ops.is_for_loop_target(node.target) or not _ordinary_operation(
                node
            ):
                stored.clear()
                continue
            if node.target not in (memory_ops.store, memory_ops.load):
                continue
            access = key(node)
            if access is None:
                if node.target is memory_ops.store:
                    stored.clear()
                continue
            if node.target is memory_ops.store:
                value = node.args[2]
                value_fake = (
                    value.meta.get("val") if isinstance(value, torch.fx.Node) else None
                )
                if (
                    node.args[3] is None
                    and isinstance(value, torch.fx.Node)
                    and isinstance(value_fake, torch.Tensor)
                    and value_fake.dtype is access[0].dtype
                ):
                    stored[access] = value
                else:
                    stored.pop(access, None)
            elif (
                node.args[2] is None
                and node.args[3] is None
                and (value := stored.get(access)) is not None
            ):
                loaded_fake = node.meta.get("val")
                stored_fake = value.meta.get("val")
                if not (
                    isinstance(loaded_fake, torch.Tensor)
                    and same_shape(stored_fake, loaded_fake)
                    and pointwise_uses(node, access)
                ):
                    continue
                node.replace_all_uses_with(value)
                info.graph.erase_node(node)


def _clone_untyped(node: ast.AST) -> ast.AST:
    fields: dict[str, object] = {}
    for key, value in ast.iter_fields(node):
        if isinstance(value, ast.AST):
            fields[key] = _clone_untyped(value)
        elif isinstance(value, list):
            fields[key] = [
                _clone_untyped(item) if isinstance(item, ast.AST) else item
                for item in value
            ]
        else:
            fields[key] = value
    if not isinstance(node, ExtendedAST):
        result = copy.copy(node)
        for key, value in fields.items():
            setattr(result, key, value)
        return result
    result = node.new(fields)
    result._type_info = None
    result._loop_type = LoopType.UNSET
    result._root_id = None
    return cast("ast.AST", result)
