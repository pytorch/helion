"""Opt-in scheduling of independent lanes around one typed FP32 recurrence.

Discovery is not admission. Typed FX coordinates, actual grid ownership, late
lowered expressions and current-call host guards must all agree. The original
device body is preserved for calls whose current storage ranges overlap.
"""

from __future__ import annotations

import ast
import copy
import dataclasses
import inspect
import math
import operator
from typing import TYPE_CHECKING
from typing import NoReturn
from typing import TypeVar
from typing import cast

import torch
from torch.fx import Node

from ... import exc
from ... import language
from ...language import _tracing_ops
from ...language import creation_ops
from ...language import memory_ops
from ...language import tile_ops
from ..ast_extension import ExtendedAST
from ..ast_extension import statement_from_string
from ..ast_read_writes import ast_rename
from ..compile_environment import CompileEnvironment
from ..host_function import HostFunction

if TYPE_CHECKING:
    from collections.abc import Callable

    import sympy

    from ..device_function import DeviceFunction
    from ..device_ir import DeviceIR
    from ..device_ir import GraphInfo
    from ..tile_strategy import DeviceGridState
    from .serial_lane_coarsen import CoarsenPlan


KEY = "cute_serial_lane_schedule"
LOAD_KEY = "cute_serial_lane_load_schedule"
TAIL_KEY = "cute_serial_lane_tail_schedule"
LOAD_SCHEDULES = ("group2", "prefetch2", "group4", "prefetch4")
_FLOATS = (torch.float32, torch.float16, torch.bfloat16)
_POINTWISE = {
    torch.ops.aten.add.Tensor,
    torch.ops.aten.sub.Tensor,
    torch.ops.aten.mul.Tensor,
    torch.ops.aten.div.Tensor,
    torch.ops.aten.exp.default,
    torch.ops.aten.neg.default,
    torch.ops.aten.abs.default,
    torch.ops.aten.clamp.default,
    torch.ops.aten.maximum.default,
    torch.ops.aten.minimum.default,
    torch.ops.prims.convert_element_type.default,
}
_METADATA = {
    _tracing_ops._host_tensor,
    _tracing_ops._get_symnode,
    _tracing_ops._new_var,
    tile_ops.tile_begin,
    torch.ops.aten.sym_size.int,
}

# Canonical compiler API exports, not arbitrary values under a trusted module.
# Admission requires both the original attribute name and resolved identity.
_ORIGINAL_TORCH_EXPORTS: dict[str, object] = {
    "Tensor": torch.Tensor,
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "empty": torch.empty,
    "empty_like": torch.empty_like,
    "zeros": torch.zeros,
    "zeros_like": torch.zeros_like,
    "full": torch.full,
    "full_like": torch.full_like,
    "exp": torch.exp,
    "add": torch.add,
    "sub": torch.sub,
    "mul": torch.mul,
    "div": torch.div,
    "neg": torch.neg,
    "abs": torch.abs,
    "clamp": torch.clamp,
    "maximum": torch.maximum,
    "minimum": torch.minimum,
}
_ORIGINAL_LANGUAGE_EXPORTS: dict[str, object] = {
    "specialize": language.specialize,
    "tile": language.tile,
    "grid": language.grid,
    "full": language.full,
    "zeros": language.zeros,
}
_ORIGINAL_TENSOR_DESCRIPTORS: dict[str, object] = {
    "size": torch.Tensor.size,
    "stride": torch.Tensor.stride,
    "numel": torch.Tensor.numel,
    "dim": torch.Tensor.dim,
    "ndimension": torch.Tensor.ndimension,
    "element_size": torch.Tensor.element_size,
}


class _Unsupported(Exception):
    pass


_T = TypeVar("_T")


def _typed(value: object, kind: type[_T]) -> _T:
    if not isinstance(value, kind):
        raise _Unsupported
    return value


def _require(value: object) -> None:
    if not value:
        raise _Unsupported


def _check_original_host(host: HostFunction) -> None:
    """Reject dependencies whose original value is lost by fake specialization.

    A folded FX constant is not proof of a literal initializer. Inspect the
    original typed source before using it; current runtime metadata checks do
    not cover storage offsets, foreign globals, or specialized device indices.
    """
    from ..type_info import CallableType
    from ..type_info import LiteralType
    from ..type_info import PythonModuleType
    from ..type_info import TensorAttributeType
    from ..type_info import TensorType

    def reject(reason: str) -> NoReturn:
        raise exc.BackendUnsupported(
            "cute", f"serial lane original host metadata: {reason}"
        )

    tree = ast.Module(body=host.body, type_ignores=[])
    nested_scopes = (
        ast.GeneratorExp,
        ast.ListComp,
        ast.SetComp,
        ast.DictComp,
        ast.Lambda,
        ast.FunctionDef,
        ast.AsyncFunctionDef,
    )
    if any(isinstance(node, nested_scopes) for node in ast.walk(tree)):
        reject("nested Python expression/function scope")
    closure = inspect.getclosurevars(host.definition.fn)
    if closure.nonlocals or any(
        provider is not torch and provider is not language
        for provider in closure.globals.values()
    ):
        reject("untrusted original global provider")

    metadata = {
        "shape",
        "size",
        "stride",
        "dtype",
        "ndim",
        "numel",
        "dim",
        "ndimension",
        "element_size",
    }
    views = {"T", "reshape", "view", "transpose", "permute", "contiguous"}
    device_methods = {
        "to",
        "float",
        "half",
        "bfloat16",
        "exp",
        "abs",
        "neg",
        "clamp",
        "clamp_min",
        "clamp_max",
    }
    allocations = {
        torch.empty,
        torch.empty_like,
        torch.zeros,
        torch.zeros_like,
        torch.full,
        torch.full_like,
    }
    pure_calls = allocations | {
        language.specialize,
        language.tile,
        language.grid,
        language.full,
        language.zeros,
        torch.Tensor.size,
        torch.Tensor.stride,
        torch.Tensor.numel,
        torch.Tensor.dim,
        torch.Tensor.ndimension,
        torch.Tensor.element_size,
        torch.exp,
        torch.add,
        torch.sub,
        torch.mul,
        torch.div,
        torch.neg,
        torch.abs,
        torch.clamp,
        torch.maximum,
        torch.minimum,
    }
    scalars = {int, float, bool, abs, min, max, len, tuple, list, range}
    parents = {
        child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)
    }
    for node in ast.walk(tree):
        if not isinstance(node, ExtendedAST):
            continue
        info = node._type_info
        if (
            isinstance(info, TensorAttributeType)
            and info.attr() not in metadata | views | device_methods
        ):
            reject(f"unmodeled tensor attribute: {info.attr()}")
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ExtendedAST)
            and isinstance(node.value._type_info, TensorType)
        ):
            if node.attr == "device":
                parent = parents[node]
                call = parents.get(parent)
                if not (
                    isinstance(parent, ast.keyword)
                    and parent.arg == "device"
                    and isinstance(call, ast.Call)
                    and isinstance(call.func, ExtendedAST)
                    and isinstance(call.func._type_info, CallableType)
                    and call.func._type_info.value in allocations
                ):
                    reject("device only as current allocation device")
            elif node.attr not in metadata | views | device_methods:
                reject(f"unmodeled tensor attribute: {node.attr}")
        if not isinstance(node, ast.Call) or info is None:
            continue
        if not isinstance(node.func, ExtendedAST):
            reject("missing original call provenance")
        function = cast("ExtendedAST", node.func)._type_info
        if isinstance(function, TensorAttributeType):
            allowed = metadata | views
            if not info.origin.is_host() and isinstance(info, TensorType):
                allowed |= device_methods
            if function.attr() not in allowed:
                reject("unmodeled host tensor call")
        elif isinstance(function, CallableType):
            target = function.value
            if target in pure_calls:
                continue
            if target in scalars:
                if node.keywords:
                    reject("unmodeled scalar constructor keywords")
                for arg in node.args:
                    if (
                        isinstance(arg, ExtendedAST)
                        and arg._type_info is not None
                        and arg._type_info.contains_tensor()
                        and not (
                            target is len and isinstance(arg._type_info, TensorType)
                        )
                    ):
                        reject("scalar constructor reading tensor data")
            else:
                reject("unmodeled host callable")
        else:
            reject("unresolved original host call")

    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        if not isinstance(node, ExtendedAST) or not isinstance(node.value, ExtendedAST):
            reject("attribute provenance: missing original typing")
        base = cast("ExtendedAST", node.value)._type_info
        leaf = cast("ExtendedAST", node)._type_info
        if isinstance(base, PythonModuleType):
            if base.value is torch:
                exports = _ORIGINAL_TORCH_EXPORTS
            elif base.value is language:
                exports = _ORIGINAL_LANGUAGE_EXPORTS
            else:
                reject("attribute provenance: noncanonical module namespace")
            if (
                node.attr not in exports
                or not isinstance(leaf, LiteralType)
                or leaf.value is not exports[node.attr]
            ):
                reject("attribute provenance: noncanonical module export")
        elif isinstance(base, CallableType):
            if (
                base.value is not torch.Tensor
                or node.attr not in _ORIGINAL_TENSOR_DESCRIPTORS
                or not isinstance(leaf, CallableType)
                or leaf.value is not _ORIGINAL_TENSOR_DESCRIPTORS[node.attr]
            ):
                reject("attribute provenance: noncanonical callable attribute")
        elif isinstance(base, LiteralType) and isinstance(base.value, torch.dtype):
            reject("attribute provenance: dtype export is not a property namespace")


def _value(node: object) -> torch.Tensor:
    value = _typed(node, Node).meta.get("val")
    _require(isinstance(value, torch.Tensor))
    return cast("torch.Tensor", value)


def _axis(index: object) -> int | None:
    env = CompileEnvironment.current()
    if type(index) is int:
        _require(index == 0)
        return None
    _require(isinstance(index, Node))
    node = cast("Node", index)
    _require(node.target in _METADATA)
    value = node.meta.get("val")
    _require(isinstance(value, (int, torch.SymInt)))
    block = env.get_block_id(cast("int | torch.SymInt", value))
    if block is None and node.target is tile_ops.tile_begin:
        arg = node.args[0]
        _require(isinstance(arg, Node))
        block = env.get_block_id(cast("Node", arg).meta["val"])
    _require(block is not None)
    return env.canonical_block_id(cast("int", block))


@dataclasses.dataclass(frozen=True)
class Recurrence:
    loop_id: int
    serial_axis: int
    steps: int
    init: Node
    store: Node
    loads: tuple[Node, ...]
    axes: tuple[int | None, ...]
    state: Node
    out_dtype: torch.dtype


def discover(ir: DeviceIR, graphs: list[GraphInfo] | None = None) -> Recurrence | None:
    """Conservative typed family discovery, with no sampled alias assumption."""
    from ..device_ir import ForLoopGraphInfo

    env = CompileEnvironment.current()
    graphs = ir.graphs if graphs is None else graphs
    try:
        _require(len(ir.root_ids) == 1 and len(graphs) == 2)
        root = graphs[ir.root_ids[0]].graph
        loops = [n for n in root.nodes if n.target is _tracing_ops._for_loop]
        _require(len(loops) == 1)
        loop = loops[0]
        graph_id, begin, end, args = loop.args
        _require(type(graph_id) is int and begin == [0])
        _require(isinstance(end, list) and len(end) == 1)
        steps = end[0]
        _require(type(steps) is int and 0 < steps < 2**31)
        _require(isinstance(args, list) and len(args) == 1)
        init = args[0]
        _require(isinstance(init, Node) and init.target is creation_ops.full)
        _require(_value(init).dtype is torch.float32)
        _require(type(init.args[1]) in (int, float) and math.isfinite(init.args[1]))
        for n in root.nodes:
            _require(
                n.op == "output"
                or n.target
                in {
                    creation_ops.full,
                    _tracing_ops._for_loop,
                    _tracing_ops._get_symnode,
                    _tracing_ops._phi,
                    operator.getitem,
                }
            )
            if n.target is _tracing_ops._phi:
                _require(not n.users)
        info = graphs[graph_id]
        _require(isinstance(info, ForLoopGraphInfo) and len(info.block_ids) == 1)
        serial = info.block_ids[0]
        placeholders = list(info.graph.find_nodes(op="placeholder"))
        _require(
            len(placeholders) == 1 and _value(placeholders[0]).dtype is torch.float32
        )
        stores = [n for n in info.graph.nodes if n.target is memory_ops.store]
        loads = tuple(n for n in info.graph.nodes if n.target is memory_ops.load)
        _require(len(stores) == 1 and 1 <= len(loads) <= 2)
        store = stores[0]
        _require(store.args[3] is None)
        out = store.args[0]
        _require(isinstance(out, Node) and out.target is _tracing_ops._host_tensor)
        dtype = _value(out).dtype
        _require(dtype in _FLOATS)
        prior = store.args[2]
        while isinstance(prior, Node) and prior.target in {
            _tracing_ops._new_var,
            torch.ops.prims.convert_element_type.default,
        }:
            prior = prior.args[0]
        _require(prior is placeholders[0])
        axes = tuple(_axis(index) for index in store.args[1])
        _require(axes.count(serial) == 1)
        _require(
            len({a for a in axes if a is not None}) == sum(a is not None for a in axes)
        )
        tensor = _value(out)
        _require(len(axes) == tensor.ndim)
        for dim, axis in enumerate(axes):
            extent = 1 if axis is None else env.block_sizes[axis].size_hint()
            _require(type(tensor.shape[dim]) is int and tensor.shape[dim] == extent)
        # Positive-stride non-overlapping layout, including dense permutations.
        covered = 1
        for stride, size in sorted(zip(tensor.stride(), tensor.shape, strict=True)):
            _require(type(stride) is int and stride > 0 and stride >= covered)
            covered += (size - 1) * stride
        _require(covered < 2**31)
        state = None
        for load in loads:
            _require(load.args[2:] == (None, None))
            source = load.args[0]
            _require(
                isinstance(source, Node) and source.target is _tracing_ops._host_tensor
            )
            load_axes = tuple(_axis(index) for index in load.args[1])
            source_value = _value(source)
            _require(source_value.dtype in _FLOATS)
            _require(all(type(s) is int and s >= 0 for s in source_value.stride()))
            _require(
                sum(
                    (size - 1) * stride
                    for size, stride in zip(
                        source_value.shape, source_value.stride(), strict=True
                    )
                )
                < 2**31
            )
            _require(len(load_axes) == source_value.ndim)
            for dim, axis in enumerate(load_axes):
                _require(axis is None or axis in axes)
                extent = 1 if axis is None else env.block_sizes[axis].size_hint()
                _require(
                    type(source_value.shape[dim]) is int
                    and source_value.shape[dim] >= extent
                )
            if tuple(_value(load).shape) == tuple(_value(placeholders[0]).shape):
                _require(state is None and source_value.dtype is torch.float32)
                _require(load_axes == axes)
                state = load
            else:
                _require(_value(load).numel() == 1)
        _require(state is not None)
        for n in info.graph.nodes:
            _require(
                n.op in ("placeholder", "output")
                or n.target
                in _METADATA | _POINTWISE | {memory_ops.store, memory_ops.load}
            )
            if n.target in _POINTWISE:
                _require(_value(n).dtype in _FLOATS)
                if _value(n).dtype is not torch.float32:
                    _require(
                        n.target is torch.ops.prims.convert_element_type.default
                        and set(n.users) == {store}
                    )
        outputs = next(n for n in info.graph.nodes if n.op == "output").args[0]
        _require(
            isinstance(outputs, list)
            and len(outputs) == 1
            and isinstance(outputs[0], Node)
        )
        _require(_value(outputs[0]).dtype is torch.float32)
        return Recurrence(
            graph_id,
            serial,
            steps,
            init,
            store,
            loads,
            axes,
            cast("Node", state),
            dtype,
        )
    except _Unsupported:
        return None


@dataclasses.dataclass(frozen=True)
class SerialLanePlan:
    recurrence: Recurrence
    lane: str
    base: str
    index: str
    carry: str
    vector_axis: int
    width: int
    pointer_signatures: tuple[str, ...]
    output_name: str
    state_name: str
    guard_alias: str
    coarsen: CoarsenPlan | None = None


def _dump(node: ast.AST, df: DeviceFunction) -> str:
    plain = (
        ast.parse(ast.unparse(node)).body[0]
        if isinstance(node, ast.stmt)
        else ast.parse(ast.unparse(node), mode="eval").body
    )
    # Identity-copy elimination may rename the store value after capture.
    # Match the entire address here; variant separately proves the exact
    # previous-carry narrowing chain in the final store expression.
    if (
        isinstance(plain, ast.Call)
        and isinstance(plain.func, ast.Attribute)
        and plain.func.attr == "store"
    ):
        plain.args = []
    return ast.dump(
        ast_rename(plain, {k: v[0] for k, v in df._variable_renames.items()})
    )


def _memory_calls(node: ast.AST) -> list[ast.Call]:
    return [
        n
        for n in ast.walk(node)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr in ("load", "store")
    ]


def capture(df: DeviceFunction, grid: DeviceGridState) -> None:
    from ..tile_strategy import PerThreadNDTileStrategy

    if KEY not in df.config.config:
        return
    host = HostFunction.current()
    _check_original_host(host)
    try:
        env = CompileEnvironment.current()
        rec = discover(host.device_ir, df.codegen.codegen_graphs)
        if rec is None:
            raise _Unsupported
        _require(len(host.device_ir.grid_block_ids) == 1)
        strategy = df.tile_strategy.block_id_to_strategy[
            tuple(host.device_ir.grid_block_ids[0])
        ]
        strategy = _typed(strategy, PerThreadNDTileStrategy)
        _require(not strategy.mma_mode)
        _require(
            all(df.resolved_block_size(bid) == 1 for bid in strategy.inactive_block_ids)
        )
        _require(not df.tile_strategy.has_surplus_threads_for_strategy(strategy))
        _require(df.config.config.get("pid_type", "flat") == "flat")
        _require(df.config.config.get("cute_cluster_n", 1) == 1)
        _require(len(grid.vec_lane_wrappers) == 1 and not grid.deferred_vector_ops)
        wrapper = next(iter(grid.vec_lane_wrappers.values()))
        _require(not wrapper.elide_outer_loop and len(wrapper.outer_for.body) == 2)
        iterator = _typed(wrapper.vloop.iter, ast.Call)
        _require(len(iterator.args) == 1)
        width = _typed(iterator.args[0], ast.Constant).value
        _require(type(width) is int and width in (2, 4))
        width = cast("int", width)
        vector_axes = [
            bid
            for bid, name in strategy._cute_vec_lane_var_by_block.items()
            if name == wrapper.vec_lane_var
        ]
        _require(len(vector_axes) == 1)
        vector_axis = vector_axes[0]
        _require(
            strategy._cute_lane_layout_by_block.get(vector_axis, "blocked") == "blocked"
        )
        _require(vector_axis in rec.axes and vector_axis != rec.serial_axis)
        root_axes = set(strategy.block_ids)
        _require(
            root_axes == {a for a in rec.axes if a is not None} - {rec.serial_axis}
        )
        for bid in root_axes:
            block = df.resolved_block_size(bid)
            _require(
                type(block) is int and env.block_sizes[bid].size_hint() % block == 0
            )
            _require(bid not in host.device_ir.noncanonical_task_origin_block_ids)
        signatures = []
        for node in (rec.store, *rec.loads):
            entries = df.codegen._statements_by_owner_node_id.get(id(node), [])
            calls = [
                call for _, statement in entries for call in _memory_calls(statement)
            ]
            _require(len(calls) == 1)
            signatures.append(_dump(calls[0], df))
        init_entries = df.codegen._statements_by_owner_node_id.get(id(rec.init), [])
        initializers = [
            statement
            for _, statement in init_entries
            if isinstance(statement, ast.Assign)
        ]
        _require(
            len(initializers) == 1 and isinstance(initializers[0].targets[0], ast.Name)
        )
        names = {}
        for node in (rec.store, rec.state):
            calls = [
                call
                for _, statement in df.codegen._statements_by_owner_node_id[id(node)]
                for call in _memory_calls(statement)
            ]
            roots = [
                x.value.id
                for x in ast.walk(_typed(calls[0].func, ast.Attribute).value)
                if isinstance(x, ast.Attribute)
                and x.attr == "iterator"
                and isinstance(x.value, ast.Name)
            ]
            _require(len(roots) == 1)
            names[node] = roots[0]
        public_tensors = [
            value
            for value in host.params.arguments.values()
            if isinstance(value, torch.Tensor)
        ]
        current_tensors = [
            value
            for value in env.runtime_arg_values_by_name.values()
            if isinstance(value, torch.Tensor)
        ]
        # A sample-bound same-object argument may have been coalesced into one
        # device parameter. A later raw call must not lose an original input.
        _require(len({id(value) for value in public_tensors}) == len(public_tensors))
        _require(len({id(value) for value in current_tensors}) == len(current_tensors))
        _require(names[rec.store] != names[rec.state])
        for load in rec.loads:
            load_calls = [
                call
                for _, statement in df.codegen._statements_by_owner_node_id[id(load)]
                for call in _memory_calls(statement)
            ]
            _require(
                not any(
                    isinstance(n, ast.Attribute)
                    and n.attr == "iterator"
                    and isinstance(n.value, ast.Name)
                    and n.value.id == names[rec.store]
                    for n in ast.walk(load_calls[0])
                )
            )
        host_names = set(host.params.arguments)
        host_names.update(
            n.id
            for statement in host.body
            for n in ast.walk(statement)
            if isinstance(n, ast.Name)
        )
        guard_alias = df.new_var("serial_lane_guard", dce=False)
        while guard_alias in host_names:
            guard_alias = df.new_var("serial_lane_guard", dce=False)
        df.cute_state.serial_lane_plan = SerialLanePlan(
            rec,
            wrapper.vec_lane_var,
            wrapper.base_index_var,
            strategy.index_var(vector_axis),
            _typed(initializers[0].targets[0], ast.Name).id,
            vector_axis,
            width,
            tuple(signatures),
            names[rec.store],
            names[rec.state],
            guard_alias,
        )
        if df.config.config.get("cute_serial_lane_coarsen", 1) == 2:
            from .serial_lane_coarsen import capture_pair

            df.cute_state.serial_lane_plan = dataclasses.replace(
                df.cute_state.serial_lane_plan,
                coarsen=capture_pair(df, strategy, rec, vector_axis, width),
            )
    except _Unsupported:
        raise exc.BackendUnsupported(
            "cute", "serial lane schedule ownership or typed recurrence proof"
        ) from None


def _names(node: ast.AST) -> set[str]:
    return {
        n.id
        for n in ast.walk(node)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
    }


def _assign_name(node: ast.AST) -> str | None:
    if (
        isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
    ):
        return node.targets[0].id
    return None


def _statements(text: str) -> list[ast.stmt]:
    return ast.parse(text).body


def _loop(lane: str, width: int, body: list[ast.stmt]) -> ast.For:
    node = cast(
        "ast.For",
        ast.parse(f"for {lane} in cutlass.range_constexpr({width}):\n    pass").body[0],
    )
    node.body = body
    return node


def load_schedule_bytes(rec: Recurrence, width: int, mode: str) -> int:
    """Nominal private payload, not a ptxas register or occupancy prediction."""
    size = width * (8 + _value(rec.store.args[0]).element_size())
    if mode.startswith("prefetch"):
        scalar_bytes = sum(
            _value(load.args[0]).element_size()
            for load in rec.loads
            if load is not rec.state
        )
        size += int(mode[-1]) * (4 * width + scalar_bytes)
    return size


@dataclasses.dataclass(frozen=True)
class _VectorPhases:
    prefix: list[ast.stmt]
    store: list[ast.stmt]
    invariant: list[ast.stmt]
    read: list[ast.stmt]
    update: list[ast.stmt]
    state_values: str
    lane: str


def _schedule_loads(
    df: DeviceFunction,
    plan: SerialLanePlan,
    serial: ast.For,
    phases: _VectorPhases,
    bank_names: set[str] | None = None,
) -> list[ast.stmt]:
    """Private raw-load ring. Consumers retain the original typed AST/order."""

    # Newly generated phases may contain ExtendedAST source locations. They
    # have no semantic role here and are not deepcopy-compatible.
    def plain(nodes: list[ast.stmt]) -> list[ast.stmt]:
        return _statements(ast.unparse(ast.Module(body=nodes, type_ignores=[])))

    phases = dataclasses.replace(
        phases,
        prefix=plain(phases.prefix),
        store=plain(phases.store),
        invariant=plain(phases.invariant),
        read=plain(phases.read),
        update=plain(phases.update),
    )
    serial = _typed(plain([serial])[0], ast.For)
    mode = _typed(df.config.config[LOAD_KEY], str)
    _require(mode in LOAD_SCHEDULES)
    depth, prefetch = int(mode[-1]), mode.startswith("prefetch")
    _require(
        load_schedule_bytes(plan.recurrence, plan.width, mode)
        <= (128 if prefetch else 64)
    )
    steps = plan.recurrence.steps
    peel = df.config.config.get(TAIL_KEY, "guarded") == "peel_final_group"
    # discover() already proves the final loop-carried phi has no users and
    # the update consists solely of typed, side-effect-free FP32 operations.
    _require(not peel or (prefetch and steps >= depth and steps % depth == 0))
    serial_name = _typed(serial.target, ast.Name).id
    group = df.new_var("serial_group", dce=False)
    base = df.new_var("serial_group_base", dce=False)
    groups = (steps + depth - 1) // depth
    outer = _typed(
        _statements(
            f"for {group} in range(cutlass.Int32(0), cutlass.Int32({groups - int(peel)})):\n    {base} = {group} * cutlass.Int32({depth})"
        )[0],
        ast.For,
    )
    declarations: list[ast.stmt] = []
    scalar = [node for node in phases.prefix + phases.invariant if _memory_calls(node)]
    _require(len(scalar) == len(plan.recurrence.loads) - 1)
    scalar_load = _typed(scalar[0], ast.Assign) if scalar else None
    if scalar_load is not None:
        _require(_assign_name(scalar_load) is not None)
        _require(
            isinstance(scalar_load.value, ast.Call)
            and len(_memory_calls(scalar_load)) == 1
        )
        _require(scalar_load.value is _memory_calls(scalar_load)[0])

    def cloned(nodes: list[ast.stmt]) -> list[ast.stmt]:
        return copy.deepcopy(nodes)

    slots: list[str] = []
    scalar_slots: str | None = None
    scalar_name = _assign_name(scalar_load) if scalar_load is not None else None
    fetch: Callable[[int, str], list[ast.stmt]] | None = None
    if prefetch:
        # The original typed load indices are affine tile axes/constants. Close
        # the late address SSA slice too: no moved address may depend on carry,
        # a raw load result or a consumer-local arithmetic value.
        definitions = {
            _assign_name(node): node
            for node in phases.prefix + phases.invariant
            if _assign_name(node) is not None
        }
        local_names = {
            node.id
            for node in ast.walk(serial)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
        }
        aliases = {k: v[0] for k, v in df._variable_renames.items()}
        carry = aliases.get(plan.carry, plan.carry)
        selected: list[ast.stmt] = []
        visiting: set[str] = set()
        done: set[str] = set()

        def dependency(value: str) -> None:
            _require(value != carry and value not in visiting)
            if value in done or value == serial_name:
                return
            if value not in definitions:
                _require(value not in local_names)
                return
            node = definitions[value]
            _require(not _memory_calls(node))
            visiting.add(value)
            for used in sorted(_names(node)):
                dependency(used)
            visiting.remove(value)
            done.add(value)
            selected.append(node)

        pointer = _typed(phases.read[0], ast.Assign).value
        required = _names(pointer)
        if scalar_load is not None:
            required |= _names(
                _typed(_typed(scalar_load.value, ast.Call).func, ast.Attribute).value
            )
        for used in sorted(required):
            dependency(used)
        future = df.new_var("serial_future", dce=False)
        fetch_body = [*selected, *scalar, *phases.read]
        # Fetch-local names cannot overwrite current consumer coordinates or
        # values. The state register destination is substituted per literal slot.
        fetch_names = sorted(
            {
                node.id
                for statement in fetch_body
                for node in ast.walk(statement)
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
            }
        )
        mapping = {
            n: df.new_var("serial_prefetch_" + n, dce=False) for n in fetch_names
        }
        mapping[serial_name] = future
        scalar_slots = (
            df.new_var("serial_raw_coefficients", dce=False) if scalar else None
        )
        if scalar_slots is not None:
            raw = next(
                load
                for load in plan.recurrence.loads
                if load is not plan.recurrence.state
            )
            ty = CompileEnvironment.current().backend.dtype_str(
                _value(raw.args[0]).dtype
            )
            declarations += _statements(
                f"{scalar_slots} = cute.make_rmem_tensor(({depth},), {ty})"
            )
        slots = [df.new_var("serial_prefetch_states", dce=False) for _ in range(depth)]
        if bank_names is not None:
            bank_names.update(slots)
        for slot in slots:
            declarations += _statements(
                f"{slot} = cute.make_rmem_tensor(({plan.width},), cutlass.Float32)"
            )

        def fetch_slot(slot: int, expression: str) -> list[ast.stmt]:
            result = _statements(f"{future} = {expression}")
            result += [
                ast_rename(
                    copy.deepcopy(node), mapping | {phases.state_values: slots[slot]}
                )
                for node in fetch_body
            ]
            if scalar_slots is not None:
                assert scalar_name is not None
                result += _statements(
                    f"{scalar_slots}[{slot}] = {mapping[scalar_name]}"
                )
            return result

        fetch = fetch_slot
        for slot in range(min(depth, steps)):
            declarations += fetch(slot, f"cutlass.Int32({slot})")

    def consume(slot: int, *, final: bool = False) -> list[ast.stmt]:
        expression = (
            f"cutlass.Int32({steps - depth + slot})"
            if final
            else f"{base} + cutlass.Int32({slot})"
        )
        body = _statements(f"{serial_name} = {expression}")
        if prefetch:
            assert fetch is not None and len(slots) == depth
            body.append(
                _loop(
                    phases.lane,
                    plan.width,
                    _statements(
                        f"{phases.state_values}[{phases.lane}] = {slots[slot]}[{phases.lane}]"
                    ),
                )
            )
            current_scalar = (
                df.new_var("serial_current_coefficient", dce=False) if scalar else None
            )
            if current_scalar is not None:
                body += _statements(f"{current_scalar} = {scalar_slots}[{slot}]")
            if not final:
                refill = fetch(slot, f"{serial_name} + cutlass.Int32({depth})")
                if peel:
                    # Earlier groups end at steps-depth-1. All future indices
                    # therefore lie in [depth, steps-1], before pointer formation.
                    body.extend(refill)
                else:
                    # Test subtraction before computing a future pointer/index.
                    body.append(
                        ast.If(
                            test=ast.parse(
                                f"{serial_name} < cutlass.Int32({steps - depth})",
                                mode="eval",
                            ).body,
                            body=refill,
                            orelse=[],
                        )
                    )
            update = [] if final and slot == depth - 1 else phases.update
            for node in phases.prefix + phases.store + phases.invariant + update:
                body += (
                    _statements(f"{scalar_name} = {current_scalar}")
                    if node is scalar_load
                    else [copy.deepcopy(node)]
                )
        else:
            body += cloned(serial.body)
        return body

    for slot in range(depth):
        body = consume(slot)
        # Guard a partial serial group before forming its current index. Full
        # root tiles remain required independently by the original plan.
        if steps % depth:
            outer.body.append(
                ast.If(
                    test=ast.parse(
                        f"{base} < cutlass.Int32({steps - slot})", mode="eval"
                    ).body,
                    body=body,
                    orelse=[],
                )
            )
        else:
            outer.body.extend(body)
    if peel:
        final_group = [
            node for slot in range(depth) for node in consume(slot, final=True)
        ]
        return [*declarations, *([outer] if groups > 1 else []), *final_group]
    return [*declarations, outer]


def variant(df: DeviceFunction, original: ast.FunctionDef) -> ast.FunctionDef:
    """Rewrite only a typed, rematched late region; leave original untouched."""
    plan = df.cute_state.serial_lane_plan
    try:
        if plan is None:
            raise _Unsupported
        aliases = {k: v[0] for k, v in df._variable_renames.items()}

        def name(value: str) -> str:
            return aliases.get(value, value)

        lane, base, index, carry = map(
            name, (plan.lane, plan.base, plan.index, plan.carry)
        )
        candidate = cast("ast.FunctionDef", ast.parse(ast.unparse(original)).body[0])
        candidate.name = original.name + "_serial_lane"
        loops = [
            n
            for n in ast.walk(candidate)
            if isinstance(n, ast.For)
            and isinstance(n.target, ast.Name)
            and n.target.id == lane
        ]
        _require(len(loops) == 1)
        vloop = loops[0]
        _require(len(vloop.body) == 3)
        index_setup, initialize, serial = vloop.body
        _require(
            _assign_name(index_setup) == index and _assign_name(initialize) == carry
        )
        serial = _typed(serial, ast.For)
        _require(not serial.orelse)
        _require(all(isinstance(n, (ast.Assign, ast.Expr)) for n in serial.body))
        calls = _memory_calls(serial)
        _require(sorted(_dump(c, df) for c in calls) == sorted(plan.pointer_signatures))
        store_positions = [
            i
            for i, n in enumerate(serial.body)
            if isinstance(n, ast.Expr)
            and isinstance(n.value, ast.Call)
            and isinstance(n.value.func, ast.Attribute)
            and n.value.func.attr == "store"
        ]
        _require(len(store_positions) == 1)
        store_pos = store_positions[0]
        first = next(i for i, n in enumerate(serial.body) if carry in _names(n))
        _require(first <= store_pos)
        prefix = serial.body[:first]
        store_group = serial.body[first : store_pos + 1]
        store_expr = _typed(cast("ast.Expr", store_group[-1]).value, ast.Call)
        _require(len(store_expr.args) == 1)
        definitions = {
            _assign_name(n): n.value
            for n in store_group[:-1]
            if isinstance(n, ast.Assign)
        }
        prior = store_expr.args[0]
        visited = set()
        dtype = CompileEnvironment.current().backend.dtype_str(
            plan.recurrence.out_dtype
        )
        while True:
            if isinstance(prior, ast.Name) and prior.id in definitions:
                _require(prior.id not in visited)
                visited.add(prior.id)
                prior = definitions[prior.id]
            elif isinstance(prior, ast.Call):
                _require(
                    ast.unparse(prior.func) in {dtype, "cutlass.Float32"}
                    and len(prior.args) == 1
                    and not prior.keywords
                )
                prior = prior.args[0]
            else:
                break
        _require(isinstance(prior, ast.Name) and prior.id == carry)
        suffix_start = next(
            i
            for i in range(store_pos + 1, len(serial.body))
            if _names(serial.body[i]) & {carry, index, lane}
        )
        invariant = serial.body[store_pos + 1 : suffix_start]
        update = serial.body[suffix_start:]
        _require(_assign_name(update[-1]) == carry)
        # Nothing lane-varying may survive between these two phases except
        # the one typed carry. Preserve invariant definitions in original order.
        store_defs = {_assign_name(n) for n in store_group if _assign_name(n)}
        _require(
            not (set().union(*(_names(n) for n in invariant + update)) & store_defs)
        )
        _require(
            all(not (_names(n) & {carry, index, lane}) for n in prefix + invariant)
        )
        _require(not _memory_calls(ast.Module(body=store_group[:-1], type_ignores=[])))
        _require(all(isinstance(n, ast.Assign) for n in update))
        state_loads = [n for n in update if _memory_calls(n)]
        _require(len(state_loads) == 1 and len(_memory_calls(state_loads[0])) == 1)
        state_load = cast("ast.Assign", state_loads[0])
        _require(
            isinstance(state_load.value, ast.Call)
            and isinstance(state_load.value.func, ast.Attribute)
            and state_load.value.func.attr == "load"
        )
        _require(_assign_name(state_load) is not None)
        # All scalar/index side effects are already excluded by typed FX;
        # later lowering must not introduce barriers, callbacks or hidden calls.
        pure_calls = {
            "cutlass.Float32",
            "cutlass.BFloat16",
            "cutlass.Float16",
            "cutlass.Int32",
            "cutlass.Int64",
            "cute.math.exp2",
            "cute.math.min",
            "cute.math.max",
            "min",
            "max",
            "abs",
            "cute.math.fma",
        }
        for node in ast.walk(serial):
            if (
                isinstance(node, ast.Call)
                and node not in calls
                and node is not serial.iter
            ):
                _require(ast.unparse(node.func) in pure_calls)
        width = plan.width
        carry_values = df.new_var("serial_carry_values", dce=False)
        init_loop = _loop(
            lane,
            width,
            [
                copy.deepcopy(index_setup),
                copy.deepcopy(initialize),
                statement_from_string(f"{carry_values}[{lane}] = {carry}"),
            ],
        )
        replacement = [
            *_statements(
                f"{carry_values} = cute.make_rmem_tensor(({width},), cutlass.Float32)"
            ),
            init_loop,
        ]
        carry_read = _statements(f"{carry} = {carry_values}[{lane}]")[0]
        carry_write = _statements(f"{carry_values}[{lane}] = {carry}")[0]
        vector_phases: _VectorPhases | None = None
        bank_names = {carry_values} if plan.coarsen is not None else None
        if df.config.config[KEY] == "step_major":
            serial.body = [
                *copy.deepcopy(prefix),
                _loop(
                    lane,
                    width,
                    [
                        copy.deepcopy(index_setup),
                        copy.deepcopy(carry_read),
                        *copy.deepcopy(store_group),
                    ],
                ),
                *copy.deepcopy(invariant),
                _loop(
                    lane,
                    width,
                    [
                        copy.deepcopy(index_setup),
                        copy.deepcopy(carry_read),
                        *copy.deepcopy(update),
                        carry_write,
                    ],
                ),
            ]
        else:
            out_values = df.new_var("serial_out_values", dce=False)
            state_values = df.new_var("serial_state_values", dce=False)
            if bank_names is not None:
                bank_names.update((out_values, state_values))
            dtype = CompileEnvironment.current().backend.dtype_str(
                plan.recurrence.out_dtype
            )
            _require(
                width * (8 + _value(plan.recurrence.store.args[0]).element_size()) <= 64
            )
            replacement += _statements(
                f"{out_values} = cute.make_rmem_tensor(({width},), {dtype})\n{state_values} = cute.make_rmem_tensor(({width},), cutlass.Float32)"
            )
            store = _typed(cast("ast.Expr", store_group[-1]).value, ast.Call)
            _require(len(store.args) == 1)
            out_phase = _loop(
                lane,
                width,
                [
                    copy.deepcopy(index_setup),
                    copy.deepcopy(carry_read),
                    *copy.deepcopy(store_group[:-1]),
                    statement_from_string(
                        f"{out_values}[{lane}] = {{value}}",
                        value=copy.deepcopy(store.args[0]),
                    ),
                ],
            )

            def transfer(call: ast.Call, values: str, is_store: bool) -> list[ast.stmt]:
                pointer = copy.deepcopy(_typed(call.func, ast.Attribute).value)

                class Base(ast.NodeTransformer):
                    def visit_Name(self, node: ast.Name) -> ast.AST:
                        return (
                            ast.copy_location(ast.Name(id=base, ctx=node.ctx), node)
                            if node.id == index
                            else node
                        )

                pointer = Base().visit(pointer)
                source_name = name(plan.output_name if is_store else plan.state_name)
                tensor = _value(
                    plan.recurrence.store.args[0]
                    if is_store
                    else plan.recurrence.state.args[0]
                )
                dim = plan.recurrence.axes.index(plan.vector_axis)
                elem = tensor.element_size()
                ptr_name = df.new_var("serial_pointer", dce=False)
                vec_name = df.new_var("serial_vector", dce=False)
                atom = df.new_var("serial_copy", dce=False)
                ty = CompileEnvironment.current().backend.dtype_str(tensor.dtype)
                setup = _statements(
                    f"{ptr_name} = {ast.unparse(pointer)}\n{atom} = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), {ty}, num_bits_per_copy={8 * width * elem})"
                )
                guard = f"({base} >= 0) & ({base} + {width} <= {tensor.shape[dim]}) & ({source_name}.layout.stride[{dim}] == 1) & ({ptr_name}.toint() % {width * elem} == 0)"
                fast = _statements(
                    f"{vec_name} = cute.make_tensor({ptr_name}.align({width * elem}), cute.make_layout(({width},), stride=(1,)))\ncute.copy({atom}, {values if is_store else vec_name}, {vec_name if is_store else values})"
                )
                if is_store:
                    scalar_call = copy.deepcopy(call)
                    scalar_call.args = [
                        ast.parse(f"{values}[{lane}]", mode="eval").body
                    ]
                    scalar = ast.Expr(value=scalar_call)
                else:
                    scalar = statement_from_string(
                        f"{values}[{lane}] = {{value}}", value=copy.deepcopy(call)
                    )
                fallback = _loop(lane, width, [copy.deepcopy(index_setup), scalar])
                return [
                    *setup,
                    ast.If(
                        test=ast.parse(guard, mode="eval").body,
                        body=fast,
                        orelse=[fallback],
                    ),
                ]

            read_phase = transfer(
                _typed(state_load.value, ast.Call), state_values, False
            )
            updated = copy.deepcopy(update)
            for i, statement in enumerate(update):
                if statement is state_load:
                    updated[i] = statement_from_string(
                        f"{_assign_name(state_load)} = {state_values}[{lane}]"
                    )
            store_phase = [out_phase, *transfer(store, out_values, True)]
            update_phase: list[ast.stmt] = [
                _loop(
                    lane,
                    width,
                    [
                        copy.deepcopy(index_setup),
                        copy.deepcopy(carry_read),
                        *updated,
                        carry_write,
                    ],
                ),
            ]
            serial.body = [
                *copy.deepcopy(prefix),
                *store_phase,
                *copy.deepcopy(invariant),
                *read_phase,
                *update_phase,
            ]
            vector_phases = _VectorPhases(
                prefix,
                store_phase,
                invariant,
                read_phase,
                update_phase,
                state_values,
                lane,
            )
        if LOAD_KEY in df.config.config:
            _require(df.config.config[KEY] == "step_major_vector")
            assert vector_phases is not None
            replacement += _schedule_loads(
                df, plan, serial, vector_phases, bank_names=bank_names
            )
        else:
            replacement.append(serial)

        if plan.coarsen is not None:
            from .serial_lane_coarsen import expand_banks
            from .serial_lane_coarsen import remap_pid

            assert bank_names is not None and vector_phases is not None
            pair_index = name(plan.coarsen.index)
            _require(
                not any(
                    pair_index in _names(node)
                    for node in vector_phases.prefix + vector_phases.invariant
                )
            )
            _require(
                pair_index
                in _names(ast.Module(body=vector_phases.read, type_ignores=[]))
            )
            replacement = expand_banks(
                df,
                replacement,
                pair_index,
                plan.coarsen.block,
                bank_names,
                split_tail=TAIL_KEY in df.config.config,
            )
            candidate = remap_pid(df, candidate, plan.coarsen)

        class Replace(ast.NodeTransformer):
            def visit_For(self, node: ast.For) -> ast.AST | list[ast.stmt]:
                if node is vloop:
                    return replacement
                return self.generic_visit(node)

        return cast("ast.FunctionDef", Replace().visit(candidate))
    except (_Unsupported, StopIteration):
        raise exc.BackendUnsupported(
            "cute", "serial lane schedule late expression/phase proof"
        ) from None


def _contract(value: object) -> tuple[object, ...]:
    if isinstance(value, torch.SymInt):
        expression = CompileEnvironment.current().specialize_expr(
            cast("sympy.Expr", value.node.expr)
        )
        if expression.free_symbols or not expression.is_Integer:
            raise exc.BackendUnsupported(
                "cute", "serial lane schedule unavailable scalar specialization"
            )
        value = int(expression)
    if isinstance(value, torch.Tensor):
        if not all(type(n) is int and n > 0 for n in value.shape) or not all(
            type(s) is int and s >= 0 for s in value.stride()
        ):
            raise exc.BackendUnsupported(
                "cute", "serial lane schedule requires static metadata"
            )
        return "tensor", tuple(value.shape), tuple(value.stride()), str(value.dtype), 16
    if type(value) in (int, float, bool):
        return (
            "scalar",
            type(value).__name__,
            cast("float", value).hex() if isinstance(value, float) else value,
        )
    raise exc.BackendUnsupported(
        "cute", "serial lane schedule unavailable public dependency"
    )


def guarded_call(df: DeviceFunction, call: ast.AST) -> ast.AST:
    from ..device_function import ConstExprArg
    from ..device_function import TensorArg
    from ..device_function import _is_literal_constexpr

    if KEY not in df.config.config:
        return call
    plan = df.cute_state.serial_lane_plan
    if plan is None:
        raise exc.BackendUnsupported(
            "cute", "serial lane schedule requires an active vector lane"
        )
    params = [
        arg
        for arg in df.sorted_args()
        if not (isinstance(arg, ConstExprArg) and _is_literal_constexpr(arg))
    ]
    if not all(isinstance(arg, TensorArg) for arg in params):
        raise exc.BackendUnsupported(
            "cute", "serial lane schedule only supports tensor launch arguments"
        )
    hosts = [arg.host_str() for arg in params]
    if not all(
        isinstance(ast.parse(name, mode="eval").body, ast.Name) for name in hosts
    ):
        raise exc.BackendUnsupported(
            "cute", "serial lane schedule unresolved current host origin"
        )
    contracts = tuple(_contract(cast("TensorArg", arg).fake_value) for arg in params)
    writes = tuple(i for i, arg in enumerate(params) if arg.name == plan.output_name)
    if len(writes) != 1:
        raise exc.BackendUnsupported(
            "cute", "serial lane schedule output origin mismatch"
        )
    guard = plan.guard_alias
    launch = next(
        n
        for n in ast.walk(call)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "_launcher"
    )
    launch.args[0] = ast.parse(
        f"{guard}.select_kernel({df.name}, {df.name}_serial_lane, ({', '.join(hosts)},), {contracts!r}, {writes!r})",
        mode="eval",
    ).body
    if plan.coarsen is not None:
        # Evaluate the unchanged current-call guard exactly once; aliases retain
        # the entire original kernel and its original launch grid.
        host = HostFunction.current()
        used = set(host.params.arguments)
        used.update(
            n.id
            for statement in host.body
            for n in ast.walk(statement)
            if isinstance(n, ast.Name)
        )
        selected = df.new_var("serial_selected_kernel", dce=False)
        while selected in used:
            selected = df.new_var("serial_selected_kernel", dce=False)
        launch.args[0] = ast.NamedExpr(
            target=ast.Name(id=selected, ctx=ast.Store()), value=launch.args[0]
        )
        launch.args[1] = ast.IfExp(
            test=ast.Compare(
                left=ast.Name(id=selected, ctx=ast.Load()),
                ops=[ast.Is()],
                comparators=[ast.Name(id=df.name + "_serial_lane", ctx=ast.Load())],
            ),
            body=ast.Tuple(elts=[ast.Constant(plan.coarsen.grid // 2)], ctx=ast.Load()),
            orelse=launch.args[1],
        )
    return call


def entry_statements(df: DeviceFunction, removed: set[str]) -> list[ast.stmt]:
    if KEY not in df.config.config:
        return []
    if removed:
        raise exc.BackendUnsupported(
            "cute", "serial lane schedule cannot check erased public arguments"
        )
    host = HostFunction.current()
    names = list(host.params.arguments)
    contracts = tuple(_contract(value) for value in host.params.arguments.values())
    plan = df.cute_state.serial_lane_plan
    if plan is None:
        raise exc.BackendUnsupported(
            "cute", "serial lane schedule missing entry contract"
        )
    guard = plan.guard_alias
    return _statements(
        f"from helion.runtime.cute import serial_lane_guard as {guard}\n{guard}.validate_trace()\n{guard}.validate_entry(({', '.join(names)},), {contracts!r})"
    )
