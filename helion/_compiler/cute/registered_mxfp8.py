"""Prove a registered task grid and complete packed MXFP8 K reduction."""

from __future__ import annotations

import ast
import math
import operator
from typing import TYPE_CHECKING
from typing import cast

import sympy
import torch
from torch.fx import Node
from torch.utils._sympy.functions import FloorDiv

from ...autotuner import PowerOfTwoFragment
from ...language import _tracing_ops
from ...language import creation_ops
from ...language import loops
from ...language import memory_ops
from ...language import tile_ops
from ...language import tunable_ops
from ...language import view_ops
from ...language.matmul_ops import dot_scaled
from ..ast_extension import ExtendedAST
from ..ast_extension import LoopType
from ..compile_environment import CompileEnvironment
from ..compile_environment import FixedBlockSizeSource
from ..device_ir import ElseGraphInfo
from ..device_ir import ForLoopGraphInfo
from ..device_ir import IfGraphInfo
from ..device_ir import RootGraphInfo
from ..host_function import HostFunction
from ..type_info import CallableType
from ..type_info import IterType
from ..type_info import LiteralType
from ..type_info import SequenceType
from ..type_info import TensorAttributeType
from ..type_info import TensorType
from ..variable_origin import GridOrigin
from .block_scaled_mma import _metadata_resolver
from .promote_output_axis import _FRESH_FACTORIES
from .promote_output_axis import _FRESH_STRIDED_FACTORIES
from .promote_output_axis import _FRESH_TENSOR_METHODS
from .promote_output_axis import _READ_ONLY_HOST_CALLS
from .promote_output_axis import _READ_ONLY_TENSOR_METHODS
from .promote_output_axis import _read_only_host_dsl

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch.fx.node import Target

    from ..device_ir import DeviceIR
    from ..device_ir import GraphInfo
    from .mxfp8_mma import Mxfp8Match


def _sym(value: object, env: CompileEnvironment) -> sympy.Expr | None:
    if isinstance(value, Node):
        value = value.meta.get("val")
    if isinstance(value, (torch.SymInt, torch.SymBool)):
        value = value._sympy_()
    if type(value) is int:
        return sympy.Integer(value)
    if isinstance(value, sympy.Basic):
        return env.shape_env.replace(value)
    return None


def _ast_value(node: ast.AST) -> object:
    if not isinstance(node, ExtendedAST) or node._type_info is None:
        raise ValueError("untyped host expression")
    return node._type_info.proxy()


def _literal_bool(node: ast.AST) -> bool | None:
    if isinstance(node, ExtendedAST) and isinstance(node._type_info, LiteralType):
        value = node._type_info.value
        if type(value) is bool:
            return value
    return None


def registered_host_freshness(
    host: HostFunction, partition_symbol: sympy.Expr
) -> tuple[set[torch.Tensor], list[ast.Return]] | None:
    """Validate both symbolic branches, retaining original host statements.

    A known literal branch was not traced and cannot mutate retained metadata.
    Both branches of a config condition are checked before factories establish
    storage ownership. This prevents an unreachable metadata mutation from
    rebinding a retained factory tensor to input storage.
    """
    env = CompileEnvironment.current()
    factories: set[torch.Tensor] = set()
    selected: list[ast.stmt] = []

    def calls_safe(expressions: list[ast.AST]) -> bool:
        for expression in expressions:
            if (
                isinstance(expression, (ast.Attribute, ast.Subscript))
                and isinstance(expression.ctx, (ast.Store, ast.Del))
                or isinstance(expression, (ast.NamedExpr, ast.Lambda))
            ):
                return False
            if not isinstance(expression, ast.Call):
                continue
            if (
                not isinstance(expression.func, ExtendedAST)
                or expression.func._type_info is None
                or any(keyword.arg in (None, "out") for keyword in expression.keywords)
            ):
                return False
            callee = expression.func._type_info
            safe = False
            is_factory = False
            if isinstance(callee, CallableType):
                safe = (
                    any(callee.value is function for function in _READ_ONLY_HOST_CALLS)
                    or _read_only_host_dsl(expression, callee)
                    or callee.value is loops.grid
                    or callee.value is tunable_ops.register_tunable
                    or callee.value is PowerOfTwoFragment
                    and all(type(_ast_value(arg)) is int for arg in expression.args)
                    and not expression.keywords
                )
                is_factory = any(
                    callee.value is function
                    for function in (*_FRESH_FACTORIES, *_FRESH_STRIDED_FACTORIES)
                )
            elif isinstance(callee, TensorAttributeType):
                safe = callee.attr() in _READ_ONLY_TENSOR_METHODS | {"sum", "to"}
                is_factory = callee.attr() in _FRESH_TENSOR_METHODS
            if not safe:
                return False
            if is_factory:
                if not isinstance(expression, ExtendedAST) or not isinstance(
                    expression._type_info, TensorType
                ):
                    return False
                factories.add(expression._type_info.proxy())
        return True

    def validate(statements: list[ast.stmt], select: bool) -> bool:
        for statement in statements:
            if isinstance(statement, ast.For):
                if (
                    not isinstance(statement, ExtendedAST)
                    or statement._loop_type is not LoopType.GRID
                    or not isinstance(statement.iter, ast.Call)
                    or not calls_safe(list(ast.walk(statement.iter)))
                ):
                    return False
                if select:
                    selected.append(statement)
                continue
            if isinstance(statement, ast.If):
                if not calls_safe(list(ast.walk(statement.test))):
                    return False
                literal = _literal_bool(statement.test)
                if literal is not None:
                    if not validate(
                        statement.body if literal else statement.orelse, select
                    ):
                        return False
                    continue
                condition = _sym(_ast_value(statement.test), env)
                if condition is None or not condition.free_symbols <= {
                    partition_symbol
                }:
                    return False
                value = sympy.simplify(
                    condition.xreplace({partition_symbol: sympy.Integer(1)})
                )
                if value not in (sympy.true, sympy.false):
                    return False
                if not validate(statement.body, select and value is sympy.true):
                    return False
                if not validate(statement.orelse, select and value is sympy.false):
                    return False
                continue
            if not isinstance(
                statement, (ast.Assign, ast.AnnAssign, ast.Expr, ast.Assert, ast.Return)
            ):
                return False
            if not calls_safe(list(ast.walk(statement))):
                return False
            if select:
                selected.append(statement)
        return True

    try:
        if not validate(host.body, True):
            return None
    except (ValueError, AttributeError, TypeError, NotImplementedError):
        return None
    returns = [statement for statement in selected if isinstance(statement, ast.Return)]
    return factories, returns


def _combine(name: str, *arguments: object) -> object:
    if name in ("add", "mul", "and"):
        flattened = []
        for argument in arguments:
            if isinstance(argument, tuple) and argument and argument[0] == name:
                flattened.extend(argument[1:])
            else:
                flattened.append(argument)
        # Symbolic scalar metadata has already removed multiplication by one
        # and addition of zero. These exact integer identities do not cancel
        # any intermediate overflow or change an index tensor's shape.
        identity = 1 if name == "mul" else 0 if name == "add" else None
        if identity is not None:
            flattened = [
                item
                for item in flattened
                if not (type(item) is int and item == identity)
            ]
            if len(flattened) == 1:
                return flattened[0]
        return (name, *sorted(flattened, key=repr))
    return (name, *arguments)


def _kwargs_valid(node: Node) -> bool:
    if node.target is torch.ops.prims.iota.default:
        start, step = node.kwargs.get("start"), node.kwargs.get("step")
        if not isinstance(start, (int, float)) or not isinstance(step, (int, float)):
            return False
        return (
            set(node.kwargs) <= {"start", "step", "dtype", "device", "requires_grad"}
            and start == 0
            and step == 1
            and node.kwargs.get("dtype") is torch.int32
            and node.kwargs.get("requires_grad", False) is False
        )
    if node.target is torch.ops.aten.add.Tensor:
        alpha = node.kwargs.get("alpha", 1)
        return (
            set(node.kwargs) <= {"alpha"}
            and isinstance(alpha, (int, float))
            and type(alpha) in (int, float)
            and alpha == 1
        )
    if node.target is torch.ops.aten.div.Tensor_mode:
        return node.kwargs == {"rounding_mode": "floor"}
    return not node.kwargs


class _Expressions:
    def __init__(
        self,
        symbols: dict[sympy.Expr, str],
        bindings: dict[Node, str],
        group_block: int,
    ) -> None:
        self.symbols = symbols
        self.bindings = bindings
        self.group_block = group_block
        self.env = CompileEnvironment.current()

    def __call__(self, value: object) -> object:
        if isinstance(value, (tuple, list)):
            return tuple(self(item) for item in value)
        if not isinstance(value, Node):
            return value
        if value in self.bindings:
            return self.bindings[value]
        if not _kwargs_valid(value):
            return ("unsupported",)
        target = value.target
        if target in (_tracing_ops._new_var, torch.ops.aten.alias.default):
            return self(value.args[0]) if len(value.args) == 1 else ("unsupported",)
        if target is _tracing_ops._get_symnode:
            if len(value.args) != 1:
                return ("unsupported_symbol",)
            symbol = _sym(value, self.env)
            return (
                self.symbols.get(symbol, ("unknown_symbol",))
                if symbol is not None
                else ("unknown_symbol",)
            )
        if target in (tile_ops.tile_begin, tile_ops.tile_index):
            if len(value.args) != 1 or self(value.args[0]) != "group_block":
                return ("unsupported_tile",)
            return "group_begin" if target is tile_ops.tile_begin else "groups"
        if target is torch.ops.prims.iota.default:
            return ("iota", self(value.args[0]))
        if target is view_ops.subscript and len(value.args) == 2:
            dimensions = value.args[1]
            if dimensions == [slice(None), None]:
                return ("column_vector", self(value.args[0]))
            if dimensions == [None, slice(None)]:
                return ("row_vector", self(value.args[0]))
            return ("unsupported_view",)
        if (
            target is torch.ops.prims.convert_element_type.default
            and len(value.args) == 2
        ):
            return ("cast", value.args[1], self(value.args[0]))
        names: dict[Target, str] = {
            operator.add: "add",
            torch.ops.aten.add.Tensor: "add",
            operator.mul: "mul",
            torch.ops.aten.mul.Tensor: "mul",
            operator.eq: "eq",
            torch.ops.aten.eq.Scalar: "eq",
            torch.ops.aten.lt.Scalar: "lt",
            torch.ops.aten.bitwise_and.Tensor: "and",
            torch.ops.aten.bitwise_and.Scalar: "and",
            torch.ops.aten.div.Tensor_mode: "floor_divide",
            torch.ops.aten.remainder.Scalar: "remainder",
            torch.ops.aten.__rshift__.Tensor: "shift_right",
            torch.ops.aten.bitwise_right_shift.Tensor: "shift_right",
        }
        name = names.get(target)
        return (
            _combine(name, *(self(argument) for argument in value.args))
            if name
            else ("unknown", str(target))
        )


def _closed(graph: torch.fx.Graph, sinks: list[Node]) -> bool:
    allowed = set(sinks)
    pending = sinks.copy()
    while pending:
        for node in pending.pop().all_input_nodes:
            if node not in allowed:
                allowed.add(node)
                pending.append(node)
    return all(
        node.op in ("placeholder", "output") or node in allowed for node in graph.nodes
    )


def _host_tensor(node: object) -> torch.Tensor | None:
    return (
        node.meta.get("val")
        if isinstance(node, Node)
        and node.target is _tracing_ops._host_tensor
        and len(node.args) == 1
        and not node.kwargs
        and isinstance(node.meta.get("val"), torch.Tensor)
        else None
    )


def int32_registered_domain(
    m: int, n: int, k: int, row_width: int, column_width: int, group_width: int
) -> bool:
    """Bound rounding numerators and every lane before signed Int32 casts.

    The largest padded coordinate is below extent + maximum tile width.
    Include the rounding numerator too rather than relying on host/device
    widening of an intermediate expression or the current default width.
    """
    if any(
        type(value) is not int or value <= 0
        for value in (m, n, k, row_width, column_width, group_width)
    ):
        return False
    limit = 2**31
    return (
        m * n < limit
        and m + row_width - 1 < limit
        and n + column_width - 1 < limit
        and k + group_width * 32 - 1 < limit
    )


def match_registered_mxfp8(
    graphs: Sequence[GraphInfo], ir: DeviceIR
) -> Mxfp8Match | None:
    """Recognize semantics and origins, independently of workload names."""
    from .mxfp8_mma import Mxfp8Match

    env, host = CompileEnvironment.current(), HostFunction.current()
    try:
        if (
            len(ir.root_ids) != 1
            or len(ir.phases) != 1
            or len(ir.grid_block_ids[0]) != 3
        ):
            return None
        root = graphs[ir.root_ids[0]]
        if not isinstance(root, RootGraphInfo):
            return None
        reductions = list(
            root.graph.find_nodes(op="call_function", target=_tracing_ops._for_loop)
        )
        choices = list(
            root.graph.find_nodes(op="call_function", target=_tracing_ops._if)
        )
        if len(reductions) != 1 or len(choices) != 1:
            return None
        reduction, choice = reductions[0], choices[0]
        if (
            len(reduction.args) != 4
            or len(choice.args) != 5
            or reduction.kwargs
            or choice.kwargs
        ):
            return None
        body, final, partial = (
            graphs[reduction.args[0]],
            graphs[choice.args[1]],
            graphs[choice.args[2]],
        )
        if (
            not isinstance(body, ForLoopGraphInfo)
            or not isinstance(final, IfGraphInfo)
            or not isinstance(partial, ElseGraphInfo)
        ):
            return None
        if {graph.graph_id for graph in graphs} != {
            root.graph_id,
            body.graph_id,
            final.graph_id,
            partial.graph_id,
        }:
            return None
        if (
            len(body.block_ids) != 1
            or not isinstance(reduction.args[3], (list, tuple))
            or len(reduction.args[3]) != 3
        ):
            return None
        rows, columns, initial = reduction.args[3]
        if not isinstance(rows, Node) or not isinstance(columns, Node):
            return None
        group_block = body.block_ids[0]
        row_grid, column_grid, partition_grid = ir.grid_block_ids[0]
        partition_symbol = _sym(env.block_sizes[partition_grid].size, env)
        if partition_symbol is None:
            return None
        config_expression = env.config_value_expressions.get(partition_symbol)
        if (
            partition_symbol not in env.tunable_symbols
            or config_expression is None
            or config_expression.operation != "config"
            or len(config_expression.arguments) != 1
            or not isinstance(config_expression.arguments[0], str)
        ):
            return None
        partition_key = cast("str", config_expression.arguments[0])
        fragment = env.config_spec.user_defined_tunables.get(partition_key)
        if (
            not isinstance(fragment, PowerOfTwoFragment)
            or type(fragment.default()) is not int
            or not fragment.low <= 1 <= fragment.high
        ):
            return None
        origins: dict[int, sympy.Expr] = {}
        for expression, origin in host.expr_to_origin.items():
            if type(origin.origin) is GridOrigin and origin.origin.block_id in (
                row_grid,
                column_grid,
                partition_grid,
            ):
                if origin.origin.block_id in origins:
                    return None
                origins[origin.origin.block_id] = env.shape_env.replace(expression)
        if set(origins) != {row_grid, column_grid, partition_grid}:
            return None
        symbols: dict[sympy.Expr, str] = {
            origins[row_grid]: "row_task",
            origins[column_grid]: "column_task",
            origins[partition_grid]: "partition",
            partition_symbol: "partitions",
            env.block_sizes[group_block].symbol(): "group_block",
        }
        # Registered row/column owners are recovered from the actual iotas.
        row_iotas = [
            node
            for node in rows.all_input_nodes
            if node.target is torch.ops.prims.iota.default
        ]
        column_iotas = [
            node
            for node in columns.all_input_nodes
            if node.target is torch.ops.prims.iota.default
        ]
        if len(row_iotas) != 1 or len(column_iotas) != 1:
            return None
        owners = []
        for iota, token in (
            (row_iotas[0], "row_block"),
            (column_iotas[0], "column_block"),
        ):
            if len(iota.args) != 1 or not isinstance(iota.args[0], Node):
                return None
            owner = env.get_block_id(iota.args[0].meta["val"])
            if owner is None or owner in ir.grid_block_ids[0] or owner == group_block:
                return None
            owners.append(owner)
            symbols[env.block_sizes[owner].symbol()] = token
        if len(set(owners)) != 2:
            return None
        expression = _Expressions(symbols, {}, group_block)
        C = _combine
        if expression(rows) != C(
            "add", ("iota", "row_block"), C("mul", "row_task", "row_block")
        ) or expression(columns) != C(
            "add", ("iota", "column_block"), C("mul", "column_task", "column_block")
        ):
            return None
        if (
            not isinstance(initial, Node)
            or initial.target is not creation_ops.full
            or initial.kwargs
            or initial.args[0] != [row_iotas[0].args[0], column_iotas[0].args[0]]
            or type(initial.args[1]) not in (int, float)
            or initial.args[1] != 0
            or math.copysign(1, initial.args[1]) < 0
            or initial.args[2:] != (torch.float32, None)
        ):
            return None
        dots = list(body.graph.find_nodes(op="call_function", target=dot_scaled))
        stores = [
            list(graph.graph.find_nodes(op="call_function", target=memory_ops.store))
            for graph in (final, partial)
        ]
        if len(dots) != 1 or any(len(values) != 1 for values in stores):
            return None
        dot, final_store, partial_store = dots[0], stores[0][0], stores[1][0]
        if (
            len(dot.args) != 8
            or dot.args[2] != "e4m3"
            or dot.args[5] != "e4m3"
            or dot.args[7] not in (None, torch.float32)
            or dot.kwargs
            or dot.meta["val"].dtype != torch.float32
        ):
            return None
        body_placeholders = list(body.graph.find_nodes(op="placeholder"))
        # GraphInfo.copy retains diagnostic node_args from the source graph.
        # The actual call operands bind placeholder positions in the copied
        # graph, and are the semantic bridge checked by this proof.
        if len(body_placeholders) != 3:
            return None
        body_expression = _Expressions(
            symbols,
            dict(zip(body_placeholders, ("rows", "columns", "acc"), strict=True)),
            group_block,
        )
        if body_expression(dot.args[6]) != "acc":
            return None
        lhs = dot.args[0]
        rhs_transpose = dot.args[3]
        if (
            not isinstance(rhs_transpose, Node)
            or rhs_transpose.target is not torch.ops.aten.permute.default
            or rhs_transpose.args[1:] != ([1, 0],)
            or rhs_transpose.kwargs
        ):
            return None
        rhs = rhs_transpose.args[0]
        if not isinstance(lhs, Node) or not isinstance(rhs, Node):
            return None
        if any(
            value.target is not memory_ops.load
            or len(value.args) != 4
            or value.args[3] is not None
            or value.kwargs
            for value in (lhs, rhs)
        ):
            return None
        a, b = _host_tensor(lhs.args[0]), _host_tensor(rhs.args[0])
        if (
            a is None
            or b is None
            or a.dtype != torch.float8_e4m3fn
            or b.dtype != torch.float8_e4m3fn
            or a.ndim != 2
            or b.ndim != 2
        ):
            return None
        resolve = _metadata_resolver(env)
        m, k = map(resolve, a.shape)
        n, rhs_k = map(resolve, b.shape)
        if m is None or n is None or k is None:
            return None
        if (
            any(value <= 0 or value >= 2**31 for value in (m, n, k))
            or k != rhs_k
            or k % 32
            or m * n >= 2**31
        ):
            return None
        maximum_widths = []
        for block_id in (*owners, group_block):
            item = env.config_spec.block_sizes.block_id_lookup(block_id)
            maximum_widths.append(max(item.max_size, item.declared_max_size or 0))
        if not int32_registered_domain(m, n, k, *maximum_widths):
            return None
        groups = k // 32
        span = FloorDiv(sympy.Add(partition_symbol, groups, -1), partition_symbol)
        group_start = sympy.Mul(origins[partition_grid], span)
        group_end = sympy.Min(
            groups, sympy.Mul(sympy.Add(origins[partition_grid], 1), span)
        )
        symbols[group_end] = "group_end"
        if (
            len(reduction.args[1]) != 1
            or len(reduction.args[2]) != 1
            or _sym(reduction.args[1][0], env) != group_start
            or _sym(reduction.args[2][0], env) != group_end
        ):
            return None
        if _sym(env.block_sizes[group_block].size, env) != sympy.Add(
            group_end, sympy.Mul(-1, group_start)
        ):
            return None
        for grid, owner, extent in (
            (row_grid, owners[0], m),
            (column_grid, owners[1], n),
        ):
            if _sym(env.block_sizes[grid].size, env) != FloorDiv(
                sympy.Add(env.block_sizes[owner].symbol(), extent, -1),
                env.block_sizes[owner].symbol(),
            ):
                return None
        for grid in ir.grid_block_ids[0]:
            source = env.block_sizes[grid].block_size_source
            if not isinstance(source, FixedBlockSizeSource) or source.value != 1:
                return None
        phase = ir.phases[0]
        if len(phase.root_nodes) != 1:
            return None
        iterator = phase.root_nodes[0].iter
        if (
            not isinstance(iterator, ast.Call)
            or not isinstance(iterator, ExtendedAST)
            or len(iterator.args) != 1
            or iterator.keywords
            or not isinstance(iterator.func, ExtendedAST)
            or not isinstance(iterator.func._type_info, CallableType)
            or iterator.func._type_info.value is not loops.grid
        ):
            return None
        iterator_type = iterator._type_info
        if (
            not isinstance(iterator_type, IterType)
            or not isinstance(iterator_type.inner, SequenceType)
            or len(iterator_type.inner.unpack()) != 3
        ):
            return None
        k_columns = C(
            "add", C("mul", "group_begin", 32), ("iota", C("mul", "group_block", 32))
        )
        row_mask = ("lt", ("column_vector", "rows"), m)
        column_mask = ("lt", ("column_vector", "columns"), n)
        k_mask = ("lt", ("row_vector", k_columns), C("mul", "group_end", 32))
        if (
            body_expression(lhs.args[1]) != ("rows", k_columns)
            or body_expression(rhs.args[1]) != ("columns", k_columns)
            or body_expression(lhs.args[2]) != C("and", row_mask, k_mask)
            or body_expression(rhs.args[2]) != C("and", column_mask, k_mask)
        ):
            return None
        scales = []
        for value, row_token, mask in (
            (dot.args[1], "rows", row_mask),
            (dot.args[4], "columns", column_mask),
        ):
            if (
                not isinstance(value, Node)
                or value.target is not torch.ops.prims.convert_element_type.default
                or value.args[1:] != (torch.uint8,)
                or value.kwargs
            ):
                return None
            masked = value.args[0]
            if (
                not isinstance(masked, Node)
                or masked.target is not torch.ops.aten.bitwise_and.Scalar
                or masked.args[1:] != (255,)
                or masked.kwargs
            ):
                return None
            shift = masked.args[0]
            if (
                not isinstance(shift, Node)
                or shift.target
                not in (
                    torch.ops.aten.__rshift__.Tensor,
                    torch.ops.aten.bitwise_right_shift.Tensor,
                )
                or shift.kwargs
            ):
                return None
            words = shift.args[0]
            if (
                not isinstance(words, Node)
                or words.target is not memory_ops.load
                or len(words.args) != 4
                or words.args[3] is not None
                or words.kwargs
            ):
                return None
            scale = _host_tensor(words.args[0])
            if scale is None or scale.dtype != torch.int32 or scale.ndim != 2:
                return None
            if (
                body_expression(words.args[1])
                != (row_token, ("floor_divide", "groups", 4))
                or body_expression(shift.args[1])
                != ("row_vector", C("mul", ("remainder", "groups", 4), 8))
                or body_expression(words.args[2])
                != C("and", mask, ("lt", ("row_vector", "groups"), "group_end"))
            ):
                return None
            scales.append(scale)
        for tensor, extent in ((scales[0], m), (scales[1], n)):
            if tuple(map(resolve, tensor.shape)) != (extent, (k + 127) // 128):
                return None
            row_stride, column_stride = map(resolve, tensor.stride())
            if (
                row_stride != 1
                or column_stride is None
                or column_stride < extent
                or column_stride % 4
            ):
                return None
        for tensor in (a, b):
            row_stride, column_stride = map(resolve, tensor.stride())
            if (
                column_stride != 1
                or row_stride is None
                or row_stride < k
                or row_stride % 16
            ):
                return None
        if any(env.tensor_input_source(tensor) is None for tensor in (a, b, *scales)):
            return None
        loop_output = next(iter(body.graph.find_nodes(op="output")))
        if loop_output.args != ([dot],):
            return None
        phis = list(root.graph.find_nodes(op="call_function", target=_tracing_ops._phi))
        if len(phis) != 1 or phis[0].args[0] is not initial:
            return None
        carried = phis[0].args[1]
        if (
            not isinstance(carried, Node)
            or carried.target is not operator.getitem
            or carried.args != (reduction, 0)
            or carried.kwargs
        ):
            return None
        if (
            expression(choice.args[0]) != ("eq", "partitions", 1)
            or choice.args[3] != [rows, columns, phis[0]]
            or choice.args[4] != choice.args[3]
        ):
            return None
        output_flat = _host_tensor(final_store.args[0])
        partial_flat = _host_tensor(partial_store.args[0])
        if (
            output_flat is None
            or partial_flat is None
            or output_flat.dtype != torch.bfloat16
            or partial_flat.dtype != torch.float32
            or output_flat.ndim != 1
            or partial_flat.ndim != 1
        ):
            return None
        if (
            not output_flat.is_contiguous()
            or not partial_flat.is_contiguous()
            or output_flat.storage_offset() != 0
            or partial_flat.storage_offset() != 0
            or resolve(output_flat.numel()) != m * n
            or _sym(partial_flat.numel(), env) != sympy.Mul(partition_symbol, m, n)
        ):
            return None
        for info, store, direct in (
            (final, final_store, True),
            (partial, partial_store, False),
        ):
            placeholders = list(info.graph.find_nodes(op="placeholder"))
            if len(placeholders) != 3 or len(store.args) != 4 or store.kwargs:
                return None
            expr = _Expressions(
                symbols,
                dict(zip(placeholders, ("rows", "columns", "acc"), strict=True)),
                group_block,
            )
            linear = C(
                "add", C("mul", ("column_vector", "rows"), n), ("row_vector", "columns")
            )
            if not direct:
                linear = C(
                    "add",
                    C(
                        "mul",
                        C("add", C("mul", "partition", m), ("column_vector", "rows")),
                        n,
                    ),
                    ("row_vector", "columns"),
                )
            output_mask = C("and", row_mask, ("lt", ("row_vector", "columns"), n))
            if (
                expr(store.args[1]) != (linear,)
                or expr(store.args[2])
                != (("cast", torch.bfloat16, "acc") if direct else "acc")
                or expr(store.args[3]) != output_mask
            ):
                return None
        freshness = registered_host_freshness(host, partition_symbol)
        if freshness is None:
            return None
        factories, returns = freshness
        output = output_flat._base
        scratch = partial_flat._base
        if (
            output is None
            or scratch is None
            or output not in factories
            or scratch not in factories
            or tuple(map(resolve, output.shape)) != (m, n)
            or output.dtype != torch.bfloat16
            or tuple(_sym(value, env) for value in scratch.shape)
            != (partition_symbol, sympy.Integer(m), sympy.Integer(n))
        ):
            return None
        if (
            not output.is_contiguous()
            or output.storage_offset() != 0
            or len(returns) != 1
        ):
            return None
        return_value = returns[0].value
        if return_value is None or _ast_value(return_value) is not output:
            return None
        if any(
            not _kwargs_valid(node)
            for graph in (root, body, final, partial)
            for node in graph.graph.nodes
            if node.op == "call_function"
        ):
            return None
        if (
            not _closed(root.graph, [reduction, choice])
            or not _closed(body.graph, [dot])
            or not _closed(final.graph, [final_store])
            or not _closed(partial.graph, [partial_store])
        ):
            return None
        return Mxfp8Match(
            root.graph_id,
            a,
            b,
            scales[0],
            scales[1],
            output,
            (owners[0], owners[1]),
            (torch.bfloat16,),
            False,
            m,
            n,
            k,
            partition_key,
        )
    except (
        AttributeError,
        IndexError,
        KeyError,
        TypeError,
        ValueError,
        AssertionError,
    ):
        return None
