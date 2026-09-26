"""Flatten rectangular, jointly reduced jagged/feature domains on CuTe.

The accepted reductions accumulate in FP32 at both levels. Pointwise recipes
and fresh-output stores retain their order relative to the reductions. The
rewrite changes only the reduction tree and the enumeration of independent
elements; it never moves a memory effect across a pass boundary.
"""

from __future__ import annotations

import ast
import dataclasses
from typing import TYPE_CHECKING
from typing import cast

import sympy
import torch
from torch.fx.experimental.symbolic_shapes import statically_known_true

from ... import language as hl
from ..ast_extension import ExtendedAST
from ..ast_extension import create
from ..compile_environment import CompileEnvironment
from .full_slice_matmul import _bound_names
from .full_slice_matmul import _full_slice
from .full_slice_matmul import _global_value
from .full_slice_matmul import _metadata_expression
from .online_to_3pass import _ext_copy

if TYPE_CHECKING:
    from ..host_function import HostFunction


_Axes = tuple[str | None, ...]
_INTEGER_DTYPES = (torch.int32, torch.int64)
_POINTWISE = (
    torch.abs,
    torch.exp,
    torch.log,
    torch.relu,
    torch.rsqrt,
    torch.sigmoid,
    torch.sqrt,
    torch.tanh,
)


class _Decline(Exception):
    """A proof obligation is outside the supported envelope."""


@dataclasses.dataclass(frozen=True)
class _Memory:
    dtype: torch.dtype
    rank: int
    fresh: bool


@dataclasses.dataclass(frozen=True)
class _Value:
    node: ast.expr
    axes: _Axes
    dtype: torch.dtype | None
    integer: ast.expr | None = None
    positive_zero: bool = False


def _copy(node: ast.expr) -> ast.expr:
    return cast("ast.expr", _ext_copy(node))


def _name(value: str) -> ast.Name:
    return create(ast.Name, id=value, ctx=ast.Load())


def _same(first: ast.AST, second: ast.AST) -> bool:
    return ast.dump(first, include_attributes=False) == ast.dump(
        second, include_attributes=False
    )


def _mapped(axes: _Axes) -> _Axes:
    if len(axes) == 3:
        if not (
            axes[0] in (None, "B") and axes[1] in (None, "K") and axes[2] in (None, "M")
        ):
            raise _Decline
        return (axes[0], "F" if "K" in axes or "M" in axes else None)
    if len(axes) > 3:
        raise _Decline
    if axes == ("K", "M"):
        return ("F",)
    result = tuple("F" if axis in ("K", "M") else axis for axis in axes)
    if result.count("F") > 1:
        raise _Decline
    return result


def _broadcast_axes(first: _Axes, second: _Axes) -> _Axes:
    rank = max(len(first), len(second))
    first = (None,) * (rank - len(first)) + first
    second = (None,) * (rank - len(second)) + second
    result = []
    for lhs, rhs in zip(first, second, strict=True):
        if lhs is not None and rhs is not None and lhs != rhs:
            raise _Decline
        result.append(lhs or rhs)
    return tuple(result)


def _broadcast(node: ast.expr, source: _Axes, target: _Axes) -> ast.expr:
    """Rebuild a broadcast after identifying K and M with the flat axis."""
    source, target = _mapped(source), _mapped(target)
    if source == target:
        return node
    cursor = 0
    indices: list[ast.expr | ast.Slice] = []
    for axis in target:
        if cursor < len(source) and source[cursor] == axis:
            indices.append(create(ast.Slice, lower=None, upper=None, step=None))
            cursor += 1
        elif axis is None:
            indices.append(create(ast.Constant, value=None))
        else:
            raise _Decline
    if cursor != len(source):
        raise _Decline
    return create(
        ast.Subscript,
        value=node,
        slice=create(ast.Tuple, elts=indices, ctx=ast.Load()),
        ctx=ast.Load(),
    )


def _dtype(
    node: ast.expr, host: HostFunction, memory: dict[str, _Memory]
) -> torch.dtype:
    if (
        isinstance(node, ast.Attribute)
        and node.attr == "dtype"
        and isinstance(node.value, ast.Name)
        and node.value.id in memory
    ):
        return memory[node.value.id].dtype
    value = _global_value(node, host, _bound_names(host))
    if isinstance(value, torch.dtype):
        return value
    raise _Decline


def _metadata_value(node: ast.expr, values: dict[str, object]) -> object:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return values.get(node.id)
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        value = values.get(node.value.id)
        if isinstance(value, torch.Tensor) and node.attr == "shape":
            return value.shape
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.attr == "size"
        and not node.keywords
    ):
        value = values.get(node.func.value.id)
        if isinstance(value, torch.Tensor):
            if not node.args:
                return value.size()
            if len(node.args) == 1 and isinstance(node.args[0], ast.Constant):
                dimension = node.args[0].value
                if type(dimension) is int:
                    return value.size(dimension)
    return None


def _prelude(
    host: HostFunction,
) -> tuple[dict[str, _Memory], set[str], dict[str, object]]:
    memory = {
        name: _Memory(value.dtype, value.ndim, False)
        for name, value in host.params.arguments.items()
        if isinstance(value, torch.Tensor)
    }
    values = dict(host.params.arguments)
    metadata = set(values) - set(memory)
    bound = _bound_names(host)
    for statement in host.body:
        if isinstance(statement, ast.For):
            return memory, metadata, values
        if isinstance(statement, ast.Expr) and isinstance(
            statement.value, ast.Constant
        ):
            continue
        if isinstance(statement, ast.Assert) and _metadata_expression(
            statement.test, set(memory), metadata, host, bound
        ):
            continue
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            raise _Decline
        target, value = statement.targets[0], statement.value
        names = (
            [target.id]
            if isinstance(target, ast.Name)
            else [item.id for item in target.elts if isinstance(item, ast.Name)]
            if isinstance(target, (ast.List, ast.Tuple))
            else []
        )
        if not names or set(names) & (set(values) | set(memory)):
            raise _Decline
        if isinstance(target, (ast.List, ast.Tuple)) and len(names) != len(target.elts):
            raise _Decline
        if (
            isinstance(value, ast.Call)
            and _global_value(value.func, host, bound) is torch.empty_like
            and len(value.args) == 1
            and isinstance(value.args[0], ast.Name)
            and value.args[0].id in memory
            and not value.keywords
            and len(names) == 1
        ):
            source = memory[value.args[0].id]
            memory[names[0]] = dataclasses.replace(source, fresh=True)
            continue
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Attribute)
            and isinstance(value.func.value, ast.Name)
            and value.func.value.id in memory
            and value.func.attr in ("view", "reshape")
            and len(names) == 1
            and len(value.args) == 1
            and not value.keywords
            and isinstance(value.args[0], ast.UnaryOp)
            and isinstance(value.args[0].op, ast.USub)
            and isinstance(value.args[0].operand, ast.Constant)
            and value.args[0].operand.value == 1
        ):
            memory[names[0]] = dataclasses.replace(memory[value.func.value.id], rank=1)
            continue
        if not _metadata_expression(value, set(memory), metadata, host, bound):
            raise _Decline
        resolved = _metadata_value(value, values)
        if len(names) > 1:
            if not isinstance(resolved, (list, tuple, torch.Size)) or len(
                resolved
            ) != len(names):
                raise _Decline
            values.update(zip(names, resolved, strict=True))
        else:
            values[names[0]] = resolved
        metadata.update(names)
    raise _Decline


def _loop(node: ast.stmt, function: object, host: HostFunction) -> tuple[str, ast.expr]:
    if not (
        isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and isinstance(node.iter, ast.Call)
        and _global_value(node.iter.func, host, _bound_names(host)) is function
        and len(node.iter.args) == 1
        and not node.iter.keywords
        and not node.orelse
    ):
        raise _Decline
    return node.target.id, node.iter.args[0]


def _sum(node: ast.expr) -> ast.expr:
    if not (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "sum"
        and not node.args
        and len(node.keywords) == 1
        and node.keywords[0].arg == "dim"
        and isinstance(node.keywords[0].value, ast.Constant)
        and node.keywords[0].value.value == 1
    ):
        raise _Decline
    return node.func.value


def _accumulate(node: ast.stmt) -> tuple[str, ast.expr]:
    if not (
        isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and isinstance(node.value, ast.BinOp)
        and isinstance(node.value.op, ast.Add)
        and isinstance(node.value.left, ast.Name)
        and node.value.left.id == node.targets[0].id
    ):
        raise _Decline
    return node.targets[0].id, _sum(node.value.right)


def _escapes(statements: list[ast.stmt], names: set[str]) -> bool:
    """Reject reads of removed/retyped temporaries before their next definition."""

    def reads(node: ast.AST) -> bool:
        return any(
            isinstance(item, ast.Name)
            and isinstance(item.ctx, ast.Load)
            and item.id in names
            for item in ast.walk(node)
        )

    names = set(names)
    for statement in statements:
        if isinstance(statement, ast.Assign):
            if reads(statement.value):
                return True
            for target in statement.targets:
                if not isinstance(target, ast.Name):
                    if reads(target):
                        return True
                else:
                    names.discard(target.id)
        elif isinstance(statement, ast.For):
            if reads(statement.iter):
                return True
            inner = names - {
                item.id
                for item in ast.walk(statement.target)
                if isinstance(item, ast.Name)
            }
            if _escapes(statement.body, inner) or _escapes(statement.orelse, names):
                return True
        elif reads(statement):
            return True
    return False


class _Recipe:
    def __init__(
        self,
        host: HostFunction,
        memory: dict[str, _Memory],
        metadata: set[str],
        row: str,
    ) -> None:
        self.host = host
        self.memory = memory
        self.metadata = metadata
        self.row = row
        self.locals = {name: _Value(_name(name), (), None) for name in metadata}
        self.tiles = {row: "B"}
        self.feature: ast.expr | None = None
        self.flat: str | None = None

    def scalar(self, node: ast.expr) -> bool:
        if isinstance(node, ast.Constant):
            return True
        if isinstance(node, ast.Name):
            return node.id in self.metadata
        if isinstance(node, ast.BinOp):
            return self.scalar(node.left) and self.scalar(node.right)
        if isinstance(node, ast.UnaryOp):
            return self.scalar(node.operand)
        return False

    def integer_broadcast(
        self, node: ast.expr, source: _Axes, target: _Axes
    ) -> ast.expr:
        if self.scalar(node):
            return node
        if isinstance(node, ast.BinOp):
            return create(
                ast.BinOp,
                left=self.integer_broadcast(node.left, source, target),
                op=node.op,
                right=self.integer_broadcast(node.right, source, target),
            )
        return _broadcast(node, source, target)

    def simplify_index(self, node: ast.expr) -> ast.expr:
        """Cancel the exact quotient/remainder introduced by flattening."""
        if self.flat is None or not any(
            isinstance(item, ast.Name) and item.id == self.flat
            for item in ast.walk(node)
        ):
            return node
        atoms: dict[str, sympy.Symbol] = {}
        originals: dict[sympy.Symbol, ast.expr] = {}

        def symbolic(value: ast.expr) -> sympy.Expr:
            if isinstance(value, ast.Constant) and type(value.value) is int:
                return sympy.Integer(value.value)
            if isinstance(value, ast.BinOp):
                left, right = symbolic(value.left), symbolic(value.right)
                if isinstance(value.op, ast.Add):
                    return sympy.Add(left, right)
                if isinstance(value.op, ast.Sub):
                    return sympy.Add(left, sympy.Mul(-1, right))
                if isinstance(value.op, ast.Mult):
                    return sympy.Mul(left, right)
                if isinstance(value.op, ast.FloorDiv):
                    return sympy.floor(sympy.Mul(left, sympy.Pow(right, -1)))
                if isinstance(value.op, ast.Mod):
                    return sympy.Add(
                        left,
                        sympy.Mul(
                            -1,
                            right,
                            sympy.floor(sympy.Mul(left, sympy.Pow(right, -1))),
                        ),
                    )
            key = ast.dump(value, include_attributes=False)
            if key not in atoms:
                symbol = sympy.Symbol(f"index_{len(atoms)}", integer=True)
                atoms[key] = symbol
                originals[symbol] = value
            return atoms[key]

        def expression(value: sympy.Expr) -> ast.expr:
            if value in originals:
                return _copy(originals[cast("sympy.Symbol", value)])
            if isinstance(value, sympy.Integer):
                return create(ast.Constant, value=int(value))
            if isinstance(value, (sympy.Add, sympy.Mul)):
                nodes = [expression(item) for item in value.args]
                result = nodes[0]
                for item in nodes[1:]:
                    result = create(
                        ast.BinOp,
                        left=result,
                        op=ast.Add() if isinstance(value, sympy.Add) else ast.Mult(),
                        right=item,
                    )
                return result
            raise _Decline

        simplified = sympy.expand(symbolic(node))
        try:
            return expression(simplified)
        except _Decline:
            return node

    def value(self, node: ast.expr) -> _Value:
        if isinstance(node, ast.Name):
            if node.id in self.locals:
                return self.locals[node.id]
            if node.id in self.tiles:
                return _Value(_copy(node), (self.tiles[node.id],), torch.int64)
            raise _Decline
        if isinstance(node, ast.Constant) and type(node.value) in (int, float, bool):
            return _Value(
                _copy(node), (), None, _copy(node) if type(node.value) is int else None
            )
        if (
            isinstance(node, ast.Attribute)
            and node.attr == "index"
            and isinstance(node.value, ast.Name)
            and node.value.id in self.tiles
        ):
            axis = self.tiles[node.value.id]
            result = _copy(node)
            if axis in ("K", "M"):
                assert self.flat is not None and self.feature is not None
                result = create(
                    ast.BinOp,
                    left=create(
                        ast.Attribute,
                        value=_name(self.flat),
                        attr="index",
                        ctx=ast.Load(),
                    ),
                    op=ast.FloorDiv() if axis == "K" else ast.Mod(),
                    right=_copy(self.feature),
                )
            return _Value(result, (axis,), torch.int64, result)
        if isinstance(node, ast.BinOp) and isinstance(
            node.op,
            (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow),
        ):
            left, right = self.value(node.left), self.value(node.right)
            axes = _broadcast_axes(left.axes, right.axes)
            dtype = (
                torch.promote_types(left.dtype, right.dtype)
                if left.dtype is not None and right.dtype is not None
                else left.dtype or right.dtype
            )
            result = create(
                ast.BinOp, left=_copy(left.node), op=node.op, right=_copy(right.node)
            )
            integer = None
            if dtype in _INTEGER_DTYPES and isinstance(
                node.op, (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod)
            ):
                integer = create(
                    ast.BinOp,
                    left=_copy(left.integer or left.node),
                    op=node.op,
                    right=_copy(right.integer or right.node),
                )
            return _Value(result, axes, dtype, integer)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = self.value(node.operand)
            return _Value(
                create(ast.UnaryOp, op=node.op, operand=_copy(value.node)),
                value.axes,
                value.dtype,
                positive_zero=value.positive_zero and isinstance(node.op, ast.UAdd),
            )
        if isinstance(node, ast.Subscript):
            if isinstance(node.value, ast.Name) and node.value.id in self.memory:
                memory = self.memory[node.value.id]
                if memory.rank != 1 or memory.fresh:
                    raise _Decline
                index = self.value(cast("ast.expr", node.slice))
                return _Value(
                    create(
                        ast.Subscript,
                        value=_copy(node.value),
                        slice=_copy(index.node),
                        ctx=ast.Load(),
                    ),
                    index.axes,
                    memory.dtype,
                )
            value = self.value(node.value)
            indices = (
                node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
            )
            index_axes: list[str | None] = []
            position = 0
            for index in indices:
                if isinstance(index, ast.Constant) and index.value is None:
                    index_axes.append(None)
                elif _full_slice(index) and position < len(value.axes):
                    index_axes.append(value.axes[position])
                    position += 1
                else:
                    raise _Decline
            if position != len(value.axes):
                raise _Decline
            target = tuple(index_axes)
            result = _broadcast(_copy(value.node), value.axes, target)
            integer = (
                self.integer_broadcast(_copy(value.integer), value.axes, target)
                if value.integer is not None
                else None
            )
            return _Value(result, target, value.dtype, integer, value.positive_zero)
        if isinstance(node, ast.Call):
            function = _global_value(node.func, self.host, _bound_names(self.host))
            if function is hl.load:
                if not (
                    len(node.args) == 2
                    and not node.keywords
                    and isinstance(node.args[0], ast.Name)
                    and node.args[0].id in self.memory
                    and isinstance(node.args[1], (ast.List, ast.Tuple))
                    and len(node.args[1].elts) == 1
                ):
                    raise _Decline
                memory = self.memory[node.args[0].id]
                if memory.fresh or memory.rank != 1:
                    raise _Decline
                index = self.value(node.args[1].elts[0])
                address = self.simplify_index(_copy(index.integer or index.node))
                result = create(
                    ast.Call,
                    func=_copy(node.func),
                    args=[
                        _copy(node.args[0]),
                        create(ast.List, elts=[address], ctx=ast.Load()),
                    ],
                    keywords=[],
                )
                return _Value(result, index.axes, memory.dtype)
            if function is hl.zeros:
                if not (
                    len(node.args) == 1
                    and isinstance(node.args[0], (ast.List, ast.Tuple))
                    and len(node.keywords) == 1
                    and node.keywords[0].arg == "dtype"
                ):
                    raise _Decline
                axes = tuple(
                    self.tiles[item.id]
                    for item in node.args[0].elts
                    if isinstance(item, ast.Name) and item.id in self.tiles
                )
                if len(axes) != len(node.args[0].elts):
                    raise _Decline
                return _Value(
                    _copy(node),
                    axes,
                    _dtype(node.keywords[0].value, self.host, self.memory),
                    positive_zero=True,
                )
            if (
                any(function is item for item in _POINTWISE)
                and len(node.args) == 1
                and not node.keywords
            ):
                value = self.value(node.args[0])
                if value.dtype != torch.float32:
                    raise _Decline
                return _Value(
                    create(
                        ast.Call,
                        func=_copy(node.func),
                        args=[_copy(value.node)],
                        keywords=[],
                    ),
                    value.axes,
                    value.dtype,
                )
            if isinstance(node.func, ast.Attribute) and node.func.attr in (
                "to",
                "float",
            ):
                value = self.value(node.func.value)
                if node.keywords:
                    raise _Decline
                if node.func.attr == "float" and not node.args:
                    dtype = torch.float32
                elif node.func.attr == "to" and len(node.args) == 1:
                    dtype = _dtype(node.args[0], self.host, self.memory)
                else:
                    raise _Decline
                return _Value(
                    create(
                        ast.Call,
                        func=create(
                            ast.Attribute,
                            value=_copy(value.node),
                            attr=node.func.attr,
                            ctx=ast.Load(),
                        ),
                        args=[_copy(item) for item in node.args],
                        keywords=[],
                    ),
                    value.axes,
                    dtype,
                )
        raise _Decline

    def assignment(self, statement: ast.stmt) -> ast.stmt:
        if not (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
        ):
            raise _Decline
        name = statement.targets[0].id
        if name in self.memory or name in self.metadata or name in self.tiles:
            raise _Decline
        value = self.value(statement.value)
        self.locals[name] = _Value(
            _name(name),
            value.axes,
            value.dtype,
            value.integer or (_name(name) if value.dtype in _INTEGER_DTYPES else None),
            value.positive_zero,
        )
        result = cast("ast.Assign", _ext_copy(statement))
        result.value = value.node
        return result


def _rewrite_nest(
    outer: ast.For,
    recipe: _Recipe,
    metadata_values: dict[str, object],
    following: list[ast.stmt],
    flat: str,
    bound: str,
) -> tuple[list[ast.stmt], bool]:
    feature, extent = _loop(outer, hl.tile, recipe.host)
    width = _metadata_value(extent, metadata_values)
    if not isinstance(width, (int, torch.SymInt)) or not statically_known_true(
        width > 0
    ):
        raise _Decline
    reduction = len(outer.body) == 3
    if not reduction and len(outer.body) != 1:
        raise _Decline
    inner = outer.body[1] if reduction else outer.body[0]
    jagged, length = _loop(inner, hl.jagged_tile, recipe.host)
    assert isinstance(inner, ast.For)
    length_value = recipe.value(length)
    if length_value.axes != ("B",) or length_value.dtype != torch.int64:
        raise _Decline
    nested = _Recipe(recipe.host, recipe.memory, recipe.metadata, recipe.row)
    nested.locals = dict(recipe.locals)
    nested.tiles.update({feature: "M", jagged: "K"})
    nested.feature, nested.flat = extent, flat
    temporary_names = {
        item.id
        for item in ast.walk(outer)
        if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Store)
    }
    carry = None
    partial = None
    expression = None
    if reduction:
        initial = outer.body[0]
        nested.assignment(initial)
        assert isinstance(initial, ast.Assign) and isinstance(
            initial.targets[0], ast.Name
        )
        partial = initial.targets[0].id
        if (
            not isinstance(initial.value, ast.Call)
            or _global_value(initial.value.func, recipe.host, _bound_names(recipe.host))
            is not hl.zeros
        ):
            raise _Decline
        if (
            nested.locals[partial].axes != ("B", "M")
            or nested.locals[partial].dtype != torch.float32
        ):
            raise _Decline
        carry, finish = _accumulate(outer.body[-1])
        if (
            not isinstance(finish, ast.Name)
            or finish.id != partial
            or carry not in recipe.locals
        ):
            raise _Decline
        if (
            recipe.locals[carry].axes != ("B",)
            or recipe.locals[carry].dtype != torch.float32
            or not recipe.locals[carry].positive_zero
        ):
            raise _Decline
        destination, expression = _accumulate(inner.body[-1])
        if destination != partial:
            raise _Decline
        temporary_names.discard(carry)
    if _escapes(following, temporary_names):
        raise _Decline
    if any(
        isinstance(item, ast.Name)
        and isinstance(item.ctx, ast.Store)
        and item.id in recipe.locals
        for statement in inner.body[:-1]
        for item in ast.walk(statement)
    ):
        raise _Decline
    body = [nested.assignment(statement) for statement in inner.body[:-1]]
    if reduction:
        assert carry is not None and partial is not None and expression is not None
        value = nested.value(expression)
        if value.axes != ("B", "K", "M") or value.dtype != torch.float32:
            raise _Decline
        # Partial/carry values cannot feed the pointwise expression itself.
        if any(
            isinstance(item, ast.Name) and item.id in (partial, carry)
            for statement in [*inner.body[:-1], expression]
            for item in ast.walk(statement)
        ):
            raise _Decline
        result = cast("ast.Assign", _ext_copy(inner.body[-1]))
        result.targets = [create(ast.Name, id=carry, ctx=ast.Store())]
        result.value = create(
            ast.BinOp,
            left=_name(carry),
            op=ast.Add(),
            right=create(
                ast.Call,
                func=create(
                    ast.Attribute, value=_copy(value.node), attr="sum", ctx=ast.Load()
                ),
                args=[],
                keywords=[
                    create(ast.keyword, arg="dim", value=create(ast.Constant, value=1))
                ],
            ),
        )
        body.append(result)
        # Another reduction reusing this carry must not inherit its original
        # +0 fact. Empty jagged domains skip the flattened loop entirely,
        # whereas the source feature loop still adds its zero partial sums.
        recipe.locals[carry] = dataclasses.replace(
            recipe.locals[carry], positive_zero=False
        )
    else:
        store = inner.body[-1]
        if not (isinstance(store, ast.Expr) and isinstance(store.value, ast.Call)):
            raise _Decline
        call = store.value
        if not (
            _global_value(call.func, recipe.host, _bound_names(recipe.host)) is hl.store
            and len(call.args) == 3
            and not call.keywords
            and isinstance(call.args[0], ast.Name)
            and call.args[0].id in recipe.memory
            and isinstance(call.args[1], (ast.List, ast.Tuple))
            and len(call.args[1].elts) == 1
        ):
            raise _Decline
        memory = recipe.memory[call.args[0].id]
        if not memory.fresh or memory.rank != 1:
            raise _Decline
        index, value = nested.value(call.args[1].elts[0]), nested.value(call.args[2])
        if index.axes != ("B", "K", "M") or value.axes != index.axes:
            raise _Decline
        address = nested.simplify_index(_copy(index.integer or index.node))
        flat_uses = [
            item
            for item in ast.walk(address)
            if isinstance(item, ast.Name) and item.id == flat
        ]
        # A unit-stride flat term plus a row-only base is injective within every
        # transformed rectangle. Arbitrary gathers/scatters retain their loops.
        if (
            len(flat_uses) != 1
            or not isinstance(address, ast.BinOp)
            or not isinstance(address.op, ast.Add)
        ):
            raise _Decline
        flat_term = (
            address.left
            if any(
                isinstance(item, ast.Name) and item.id == flat
                for item in ast.walk(address.left)
            )
            else address.right
        )
        if not (
            isinstance(flat_term, ast.Subscript)
            and isinstance(flat_term.value, ast.Attribute)
            and isinstance(flat_term.value.value, ast.Name)
            and flat_term.value.value.id == flat
            and flat_term.value.attr == "index"
        ):
            raise _Decline
        body.append(
            create(
                ast.Expr,
                value=create(
                    ast.Call,
                    func=_copy(call.func),
                    args=[
                        _copy(call.args[0]),
                        create(ast.List, elts=[address], ctx=ast.Load()),
                        _copy(value.node),
                    ],
                    keywords=[],
                ),
            )
        )
    result = cast("ast.For", _ext_copy(inner))
    result.target = create(ast.Name, id=flat, ctx=ast.Store())
    assert isinstance(result.iter, ast.Call)
    length_assignment = create(
        ast.Assign,
        targets=[create(ast.Name, id=bound, ctx=ast.Store())],
        value=create(ast.BinOp, left=_copy(length), op=ast.Mult(), right=_copy(extent)),
    )
    result.iter.args = [_name(bound)]
    result.body = body
    return [length_assignment, result], reduction


def flatten_nested_reductions(host: HostFunction) -> int:
    env = CompileEnvironment.current()
    if env.backend_name != "cute" or not env.settings.cute_flatten_nested_reductions:
        return 0
    # The prelude records allocation identity and input metadata. A host loop
    # can change either before a later device grid, including redirecting a
    # fresh output view to overlapping input storage with Tensor.set_. Only
    # known device loops may separate the prelude from a transformed grid.
    bound_names = _bound_names(host)
    if any(
        isinstance(statement, ast.For)
        and not (
            isinstance(statement.iter, ast.Call)
            and any(
                _global_value(statement.iter.func, host, bound_names) is function
                for function in (hl.tile, hl.grid, hl.jagged_tile)
            )
        )
        for statement in host.body
    ):
        return 0
    try:
        memory, metadata, values = _prelude(host)
    except _Decline:
        return 0
    used = _bound_names(host) | set(host.fn.__globals__)
    changed = 0
    for root in host.body:
        try:
            row, extent = _loop(root, hl.tile, host)
            if isinstance(extent, (ast.List, ast.Tuple)):
                continue
            assert isinstance(root, ast.For) and isinstance(root, ExtendedAST)
            with root:
                recipe = _Recipe(host, memory, metadata, row)
                body: list[ast.stmt] = []
                reductions = 0
                for position, statement in enumerate(root.body):
                    if isinstance(statement, ast.For):
                        flat = "_helion_flat_reduction"
                        while flat in used:
                            flat += "_"
                        used.add(flat)
                        bound = flat + "_length"
                        while bound in used:
                            bound += "_"
                        used.add(bound)
                        replacement, reduction = _rewrite_nest(
                            statement,
                            recipe,
                            values,
                            root.body[position + 1 :],
                            flat,
                            bound,
                        )
                        body.extend(replacement)
                        reductions += reduction
                    else:
                        body.append(recipe.assignment(statement))
                if reductions:
                    root.body = body
                    changed += reductions
        except _Decline:
            continue
    return changed
