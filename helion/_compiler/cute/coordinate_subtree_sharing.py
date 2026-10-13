"""One bounded, ordered scalar-coordinate sharing step.

This module does not assume that arbitrary Python operators or calls are pure.
Only proved Int32/constexpr arithmetic participates, and a shared Numeric wrapper
must pass through fresh arithmetic before escaping its original expression.
"""

from __future__ import annotations

import ast
from collections import Counter
from dataclasses import dataclass
import struct
from typing import TYPE_CHECKING
from typing import TypeAlias

from ..ast_deepcopy import deepcopy_ast
from .simplify_proven_bounds import _axis_index
from .simplify_proven_bounds import _call_path

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence

    from .coordinate_sdk_effects import CoordinateSDKEffects

_LIMIT = (1 << 31) - 1
_OPERATORS = (ast.Add, ast.Mult, ast.FloorDiv, ast.Mod)
_STATEMENTS = (
    ast.Assign,
    ast.AugAssign,
    ast.For,
    ast.If,
    ast.With,
    ast.Expr,
    ast.Pass,
    ast.ImportFrom,
)
_FORBIDDEN = (
    ast.Delete,
    ast.NamedExpr,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.Lambda,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
    ast.Global,
    ast.Nonlocal,
    ast.Try,
    ast.While,
    ast.AsyncFor,
    ast.AsyncWith,
    ast.Await,
    ast.Yield,
    ast.YieldFrom,
    ast.Import,
)
_Path: TypeAlias = tuple[tuple[int, str], ...]


class _Unsupported(Exception):
    pass


def _literal(value: object) -> tuple[str, object]:
    if value is None or type(value) in (str, int, bool, bytes):
        return type(value).__name__, value
    if type(value) is float:
        return "float64-bits", struct.pack("!d", value)
    if value is Ellipsis:
        return "ellipsis", None
    raise _Unsupported("noncanonical AST scalar")


@dataclass(frozen=True)
class _Image:
    nodes: tuple[tuple[object, ...], ...]
    roots: tuple[int, ...]


class _Index:
    """Collision-free bottom-up structural IDs; no recursive subtree dumps."""

    def __init__(
        self,
        body: Sequence[ast.AST],
        rewrite: Callable[[ast.AST], ast.AST | None] | None = None,
    ) -> None:
        self.keys: dict[int, int] = {}
        self.pool: list[tuple[object, ...]] = []
        self.lookup: dict[tuple[object, ...], int] = {}
        self.prototypes: dict[int, ast.AST] = {}
        self.sizes: list[int] = []
        self.occupied: set[str] = set()
        self.active: set[int] = set()
        self.rewrite = rewrite
        roots = []
        for node in body:
            key = self.visit(node)
            if key is not None:
                roots.append(key)
        self.image = _Image(tuple(self.pool), tuple(roots))

    def visit(self, original: ast.AST) -> int | None:
        node = self.rewrite(original) if self.rewrite is not None else original
        if node is None:
            return None
        if type(node).__module__ != "ast" or node._fields != type(node)._fields:
            raise _Unsupported("noncanonical AST node")
        identity = id(node)
        if identity in self.active:
            raise _Unsupported("cyclic AST")
        if identity in self.keys:
            return self.keys[identity]
        self.active.add(identity)
        fields = []
        size = 1
        for name in node._fields:
            try:
                value = getattr(node, name)
            except AttributeError:
                raise _Unsupported("missing AST field") from None
            if isinstance(value, ast.AST):
                key = self.visit(value)
                if key is None:
                    raise _Unsupported("removed non-list AST field")
                encoded: object = ("node", key)
                size += self.sizes[key]
            elif type(value) is list:
                values: list[tuple[str, object]] = []
                for item in value:
                    if isinstance(item, ast.AST):
                        key = self.visit(item)
                        if key is not None:
                            values.append(("node", key))
                            size += self.sizes[key]
                    else:
                        values.append(("literal", _literal(item)))
                encoded = ("list", tuple(values))
            else:
                encoded = ("literal", _literal(value))
            fields.append((name, encoded))
        entry = (type(node).__name__, tuple(fields))
        key = self.lookup.get(entry)
        if key is None:
            key = len(self.pool)
            self.lookup[entry] = key
            self.pool.append(entry)
            self.sizes.append(size)
            self.prototypes[key] = node
        self.keys[identity] = key
        self.active.remove(identity)
        if isinstance(node, ast.Name):
            self.occupied.add(node.id)
        if isinstance(node, ast.arg):
            self.occupied.add(node.arg)
        if isinstance(node, ast.alias):
            self.occupied.add(node.asname or node.name.split(".")[0])
        return key


@dataclass(frozen=True)
class _Scalar:
    low: int
    high: int
    runtime_i32: bool
    numeric_roots: frozenset[str] = frozenset()


def _interval(
    low: int, high: int, runtime: bool, roots: frozenset[str]
) -> _Scalar | None:
    if 0 <= low <= high <= _LIMIT:
        return _Scalar(low, high, runtime, roots)
    return None


@dataclass(frozen=True)
class _Choice:
    key: int
    scope: _Path
    anchor: int
    count: int
    savings: int
    scalar: _Scalar


@dataclass(frozen=True)
class SubtreeSharingProof:
    before: _Image
    after: _Image
    name: str
    expression: str
    scope: _Path
    anchor: int
    occurrences: int
    nodes_saved: int
    interval: tuple[int, int]
    numeric_roots: frozenset[str]

    def check_before(self, body: Sequence[ast.AST]) -> None:
        if _Index(body).image != self.before:
            raise ValueError("coordinate sharing original body changed")

    def check_after(self, body: Sequence[ast.AST]) -> None:
        if _Index(body).image != self.after:
            raise ValueError("coordinate sharing installed body changed")

    def check_inverse(self, body: Sequence[ast.AST]) -> None:
        self.check_after(body)
        expression = ast.parse(self.expression, mode="eval").body
        assignments = uses = 0

        def inverse(node: ast.AST) -> ast.AST | None:
            nonlocal assignments, uses
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == self.name
                for target in node.targets
            ):
                if (
                    len(node.targets) != 1
                    or _Index([node.value]).image != _Index([expression]).image
                ):
                    raise ValueError("coordinate sharing binding changed")
                assignments += 1
                return None
            if isinstance(node, ast.Name) and node.id == self.name:
                if type(node.ctx) is not ast.Load:
                    raise ValueError("coordinate sharing has an unexpected write")
                uses += 1
                return expression
            return node

        recovered = _Index(body, inverse).image
        if assignments != 1 or uses != self.occurrences or recovered != self.before:
            raise ValueError("coordinate sharing complete inverse failed")


class _Analysis:
    def __init__(
        self,
        body: list[ast.stmt],
        threads: tuple[int, int, int],
        grid: tuple[int, int, int],
        excluded: frozenset[str],
        readonly_uses: frozenset[int],
    ) -> None:
        self.index = _Index(body)
        self.threads = threads
        self.grid = grid
        self.excluded = excluded
        self.readonly_uses = readonly_uses
        self.writes: Counter[str] = Counter()
        self.escaped: set[str] = set()
        self.parents: dict[int, ast.AST] = {}
        self.names: dict[int, frozenset[str] | None] = {}
        self.memo: dict[tuple[object, ...], _Scalar | None] = {}
        self.choices: list[_Choice] = []
        seen: set[int] = set()
        stack: list[ast.AST] = list(body)
        while stack:
            node = stack.pop()
            if isinstance(node, ast.stmt) and type(node) not in _STATEMENTS:
                raise _Unsupported("statement is not admitted")
            if isinstance(node, _FORBIDDEN):
                raise _Unsupported("binding/control construct is not admitted")
            if isinstance(node, (ast.stmt, ast.expr)):
                if id(node) in seen:
                    raise _Unsupported("shared expression/statement AST object")
                seen.add(id(node))
            if isinstance(node, ast.Name) and type(node.ctx) in (ast.Store, ast.Del):
                self.writes[node.id] += 1
            if isinstance(node, ast.ImportFrom):
                if (
                    node.level != 0
                    or not node.module
                    or any(alias.name == "*" for alias in node.names)
                ):
                    raise _Unsupported("unbounded import binding")
                for alias in node.names:
                    self.writes[alias.asname or alias.name] += 1
            if isinstance(node, (ast.For, ast.With)):
                targets = (
                    [node.target]
                    if isinstance(node, ast.For)
                    else [
                        item.optional_vars
                        for item in node.items
                        if item.optional_vars is not None
                    ]
                )
                if any(
                    not (
                        type(target) is ast.Name
                        or (
                            type(target) is ast.Tuple
                            and all(type(item) is ast.Name for item in target.elts)
                        )
                    )
                    for target in targets
                ):
                    raise _Unsupported("effectful loop/context binding target")
            if isinstance(node, ast.Assign):
                if len(node.targets) != 1 or not isinstance(
                    node.targets[0], (ast.Name, ast.Tuple, ast.Subscript)
                ):
                    raise _Unsupported("assignment target is not admitted")
                if isinstance(node.targets[0], ast.Tuple) and not all(
                    type(item) is ast.Name for item in node.targets[0].elts
                ):
                    raise _Unsupported("tuple assignment target is not admitted")
                if isinstance(node.value, ast.Name):
                    self.escaped.add(node.value.id)
                    if isinstance(node.targets[0], ast.Name):
                        self.escaped.add(node.targets[0].id)
            if isinstance(node, ast.Call):
                if _call_path(node.func) != ("cutlass", "Int32"):
                    for value in (*node.args, *(kw.value for kw in node.keywords)):
                        if (
                            isinstance(value, ast.Name)
                            and id(value) not in readonly_uses
                        ):
                            self.escaped.add(value.id)
            if isinstance(node, ast.Attribute) and type(node.ctx) is not ast.Load:
                raise _Unsupported("attribute mutation is not admitted")
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                self.escaped.add(node.value.id)
            for child in ast.iter_child_nodes(node):
                self.parents[id(child)] = node
                stack.append(child)
        if self.writes.keys() & {"cutlass", "cute", "range"}:
            raise _Unsupported("SDK/global binding changed")

    def expression_names(self, key: int) -> frozenset[str] | None:
        if key in self.names:
            return self.names[key]
        node = self.index.prototypes[key]
        result: frozenset[str] | None = None
        if isinstance(node, ast.Constant) and type(node.value) is int:
            result = frozenset()
        elif isinstance(node, ast.Name) and type(node.ctx) is ast.Load:
            result = frozenset((node.id,))
        elif isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
            left = self.expression_names(self.index.keys[id(node.left)])
            right = self.expression_names(self.index.keys[id(node.right)])
            if left is not None and right is not None:
                result = left | right
        self.names[key] = result
        return result

    def value(self, node: ast.expr, env: dict[str, _Scalar]) -> _Scalar | None:
        key = self.index.keys[id(node)]
        names = self.expression_names(key)
        memo_key = (
            (key, tuple((name, env.get(name)) for name in sorted(names)))
            if names is not None
            else None
        )
        if memo_key is not None and memo_key in self.memo:
            return self.memo[memo_key]
        result = None
        if isinstance(node, ast.Constant) and type(node.value) is int:
            result = _interval(node.value, node.value, False, frozenset())
        elif isinstance(node, ast.Name):
            if self.writes[node.id] == 1 and node.id not in self.excluded:
                result = env.get(node.id)
        elif isinstance(node, ast.Call):
            if (
                _call_path(node.func) == ("cutlass", "Int32")
                and len(node.args) == 1
                and not node.keywords
            ):
                for kind, dimensions in (
                    ("thread_idx", self.threads),
                    ("block_idx", self.grid),
                ):
                    axis = _axis_index(node.args[0], kind)
                    if axis is not None:
                        result = _Scalar(0, dimensions[axis] - 1, True)
                        break
                else:
                    value = self.value(node.args[0], env)
                    if value is not None:
                        result = _Scalar(
                            value.low, value.high, True, value.numeric_roots
                        )
        elif isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
            left, right = self.value(node.left, env), self.value(node.right, env)
            if left is not None and right is not None:
                runtime = left.runtime_i32 or right.runtime_i32
                roots = left.numeric_roots | right.numeric_roots
                if isinstance(node.op, ast.Add):
                    result = _interval(
                        left.low + right.low, left.high + right.high, runtime, roots
                    )
                elif isinstance(node.op, ast.Mult):
                    result = _interval(
                        left.low * right.low, left.high * right.high, runtime, roots
                    )
                elif (
                    isinstance(node.right, ast.Constant)
                    and type(node.right.value) is int
                    and node.right.value > 0
                ):
                    divisor = node.right.value
                    if isinstance(node.op, ast.FloorDiv):
                        result = _interval(
                            left.low // divisor, left.high // divisor, runtime, roots
                        )
                    else:
                        result = _interval(
                            0, min(left.high, divisor - 1), runtime, roots
                        )
        if memo_key is not None:
            self.memo[memo_key] = result
        return result

    @staticmethod
    def written(statement: ast.AST) -> set[str]:
        return {
            node.id
            for node in ast.walk(statement)
            if isinstance(node, ast.Name) and type(node.ctx) in (ast.Store, ast.Del)
        }

    def fresh_parent(self, use: ast.BinOp, scalar: _Scalar) -> bool:
        parent = self.parents[id(use)]
        if not isinstance(parent, ast.BinOp) or type(parent.op) not in _OPERATORS:
            return False
        other = parent.right if parent.left is use else parent.left
        if not isinstance(other, ast.Constant) or type(other.value) is not int:
            return False
        constant = other.value
        if not 0 <= constant <= _LIMIT:
            return False
        if isinstance(parent.op, ast.Add):
            return scalar.high + constant <= _LIMIT
        if isinstance(parent.op, ast.Mult):
            return scalar.high * constant <= _LIMIT
        return parent.left is use and constant > 0

    def observe_escapes(self, expression: ast.expr, env: dict[str, _Scalar]) -> None:
        for node in ast.walk(expression):
            if not isinstance(node, ast.Name) or type(node.ctx) is not ast.Load:
                continue
            if id(node) in self.readonly_uses:
                continue
            scalar = env.get(node.id)
            if scalar is None or not scalar.runtime_i32:
                continue
            parent = self.parents[id(node)]
            if isinstance(parent, ast.BinOp) and self.value(parent, env) is not None:
                continue
            if (
                isinstance(parent, ast.Call)
                and _call_path(parent.func) == ("cutlass", "Int32")
                and parent.args == [node]
                and not parent.keywords
            ):
                continue
            if (
                isinstance(parent, ast.Compare)
                and all(
                    type(op) in (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE)
                    for op in parent.ops
                )
                and all(
                    self.value(value, env) is not None
                    for value in (parent.left, *parent.comparators)
                )
            ):
                continue
            self.escaped.add(node.id)

    def block(self, body: list[ast.stmt], env: dict[str, _Scalar], path: _Path) -> None:
        occurrences: dict[int, list[tuple[int, ast.BinOp]]] = {}
        for position, statement in enumerate(body):
            for node in ast.walk(statement):
                if (
                    isinstance(node, ast.BinOp)
                    and self.expression_names(self.index.keys[id(node)]) is not None
                ):
                    occurrences.setdefault(self.index.keys[id(node)], []).append(
                        (position, node)
                    )
        for position, statement in enumerate(body):
            if isinstance(statement, ast.Assign):
                self.observe_escapes(statement.value, env)
                for target in statement.targets:
                    if isinstance(target, ast.Subscript):
                        self.observe_escapes(target, env)
                value = self.value(statement.value, env)
                if isinstance(statement.targets[0], ast.Name) and value is not None:
                    for node in ast.walk(statement.value):
                        if not isinstance(node, ast.BinOp):
                            continue
                        key = self.index.keys[id(node)]
                        uses = occurrences.get(key, [])
                        if len(uses) < 2 or uses[0][0] != position:
                            continue
                        scalar = self.value(node, env)
                        if (
                            scalar is None
                            or not scalar.runtime_i32
                            or scalar.numeric_roots & self.escaped
                        ):
                            continue
                        # The shared Numeric object itself must never become an
                        # alias/call argument: each use creates a fresh result.
                        if any(not self.fresh_parent(use, scalar) for _, use in uses):
                            continue
                        savings = (
                            len(uses) * (self.index.sizes[key] - 2)
                            - self.index.sizes[key]
                            - 3
                        )
                        if savings > 0:
                            self.choices.append(
                                _Choice(key, path, position, len(uses), savings, scalar)
                            )
                    target = statement.targets[0].id
                    roots = value.numeric_roots | (
                        {target} if value.runtime_i32 else set()
                    )
                    env[target] = _Scalar(
                        value.low, value.high, value.runtime_i32, frozenset(roots)
                    )
                else:
                    for name in self.written(statement):
                        env.pop(name, None)
            elif isinstance(statement, ast.For):
                self.observe_escapes(statement.iter, env)
                local = dict(env)
                for name in self.written(statement):
                    local.pop(name, None)
                iterator = statement.iter
                if (
                    isinstance(statement.target, ast.Name)
                    and isinstance(iterator, ast.Call)
                    and _call_path(iterator.func) == ("cutlass", "range_constexpr")
                    and not iterator.keywords
                    and 1 <= len(iterator.args) <= 3
                ):
                    values = [self.value(argument, env) for argument in iterator.args]
                    if all(
                        value is not None
                        and not value.runtime_i32
                        and value.low == value.high
                        for value in values
                    ):
                        integers = [value.low for value in values if value is not None]
                        start, stop, step = (
                            (0, integers[0], 1)
                            if len(integers) == 1
                            else (*integers, 1)
                            if len(integers) == 2
                            else tuple(integers)
                        )
                        if 0 <= start < stop <= _LIMIT and step > 0:
                            local[statement.target.id] = _Scalar(start, stop - 1, False)
                self.block(statement.body, local, (*path, (position, "body")))
                self.block(statement.orelse, dict(env), (*path, (position, "orelse")))
                for name in self.written(statement):
                    env.pop(name, None)
            elif isinstance(statement, (ast.If, ast.With)):
                if isinstance(statement, ast.If):
                    self.observe_escapes(statement.test, env)
                else:
                    for item in statement.items:
                        self.observe_escapes(item.context_expr, env)
                local = dict(env)
                if isinstance(statement, ast.With):
                    for item in statement.items:
                        if item.optional_vars is not None:
                            for name in self.written(item.optional_vars):
                                local.pop(name, None)
                self.block(statement.body, local, (*path, (position, "body")))
                if isinstance(statement, ast.If):
                    self.block(
                        statement.orelse, dict(env), (*path, (position, "orelse"))
                    )
                for name in self.written(statement):
                    env.pop(name, None)
            elif isinstance(statement, ast.AugAssign):
                self.observe_escapes(statement.value, env)
                if isinstance(statement.target, ast.Subscript):
                    self.observe_escapes(statement.target, env)
                for name in self.written(statement):
                    env.pop(name, None)
            elif isinstance(statement, ast.Expr):
                self.observe_escapes(statement.value, env)


def share_coordinate_subtree(
    body: list[ast.stmt],
    *,
    thread_block_dims: tuple[int, int, int],
    block_grid_dims: tuple[int, int, int],
    excluded_names: frozenset[str],
    protected_names: frozenset[str],
    fresh_name: Callable[[str], str],
    sdk_effects: CoordinateSDKEffects | None = None,
) -> tuple[list[ast.stmt], SubtreeSharingProof | None]:
    if (
        type(body) is not list
        or len(thread_block_dims) != 3
        or len(block_grid_dims) != 3
    ):
        return body, None
    if any(
        type(x) is not int or not 0 < x <= _LIMIT
        for x in (*thread_block_dims, *block_grid_dims)
    ):
        return body, None
    if thread_block_dims[0] * thread_block_dims[1] * thread_block_dims[2] > 1024:
        return body, None
    readonly_uses: frozenset[int] = frozenset()
    if sdk_effects is not None:
        from .coordinate_sdk_effects import CoordinateSDKEffects
        from .coordinate_sdk_effects import readonly_slice_uses

        if type(sdk_effects) is not CoordinateSDKEffects:
            raise ValueError("untyped coordinate SDK effect contract")
        sdk_effects.check()
        readonly_uses = readonly_slice_uses(body)
    try:
        analysis = _Analysis(
            body, thread_block_dims, block_grid_dims, excluded_names, readonly_uses
        )
        analysis.block(body, {}, ())
    except _Unsupported:
        return body, None
    choices = [
        choice
        for choice in analysis.choices
        if not choice.scalar.numeric_roots & analysis.escaped
    ]
    if not choices:
        return body, None
    choice = max(choices, key=lambda item: item.savings)
    name = fresh_name("_helion_coordinate_shared")
    if (
        not name.isidentifier()
        or name in analysis.index.occupied | protected_names | excluded_names
    ):
        raise ValueError("coordinate sharing fresh-name collision")
    expression = analysis.index.prototypes[choice.key]
    assert isinstance(expression, ast.expr)
    expression_text = ast.unparse(expression)
    result = deepcopy_ast(body)
    copied = _Index(result)
    replacements = 0

    class Rewrite(ast.NodeTransformer):
        def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
            nonlocal replacements
            if copied.pool[copied.keys[id(node)]] == analysis.index.pool[choice.key]:
                replacements += 1
                return ast.copy_location(ast.Name(id=name, ctx=ast.Load()), node)
            return self.generic_visit(node)

    target = result
    for index, field in choice.scope:
        target = getattr(target[index], field)
    rewrite = Rewrite()
    for index in range(choice.anchor, len(target)):
        target[index] = rewrite.visit(target[index])
    if replacements != choice.count:
        raise ValueError("coordinate sharing occurrence set changed")
    target.insert(
        choice.anchor,
        ast.Assign(
            targets=[ast.Name(id=name, ctx=ast.Store())],
            value=ast.parse(expression_text, mode="eval").body,
        ),
    )
    ast.fix_missing_locations(ast.Module(body=result, type_ignores=[]))
    proof = SubtreeSharingProof(
        analysis.index.image,
        _Index(result).image,
        name,
        expression_text,
        choice.scope,
        choice.anchor,
        choice.count,
        choice.savings,
        (choice.scalar.low, choice.scalar.high),
        choice.scalar.numeric_roots,
    )
    proof.check_inverse(result)
    return result, proof
