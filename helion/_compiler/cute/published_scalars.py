"""Retain typed scalar recipes rooted in immutable, published shared slots."""

from __future__ import annotations

import ast
from copy import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import cast

import torch

from ..ast_extension import ExtendedAST

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence

    from ..device_ir import GraphInfo


def _text(node: ast.AST) -> str:
    return ast.unparse(node)


_T = TypeVar("_T", bound=ast.AST)


def _clone(node: _T) -> _T:
    # Generated expressions may carry source-location wrappers whose constructor
    # cannot be called by copy.deepcopy. Preserve those wrappers recursively.
    if isinstance(node, ExtendedAST):
        result = node.copy()
    else:
        result = copy(node)
    for field, value in ast.iter_fields(node):
        if isinstance(value, ast.AST):
            setattr(result, field, _clone(value))
        elif isinstance(value, list):
            setattr(
                result,
                field,
                [_clone(v) if isinstance(v, ast.AST) else v for v in value],
            )
    return cast("_T", result)


def _constant(node: ast.AST, value: int) -> bool:
    return (
        isinstance(node, ast.Constant)
        and type(node.value) is int
        and node.value == value
    )


def _integer_slot(node: ast.AST) -> int | None:
    """Evaluate only closed Python integer coordinates, never SDK arithmetic.

    Both operands must be known even for multiplication by zero. Names,
    calls, booleans, floating values and typed casts cannot establish a slot.
    Recognition leaves the original expression and its evaluation unchanged.
    """
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return node.value
    if isinstance(node, ast.UnaryOp):
        value = _integer_slot(node.operand)
        if value is not None:
            if isinstance(node.op, ast.UAdd):
                return value
            if isinstance(node.op, ast.USub):
                return -value
    if isinstance(node, ast.BinOp):
        left, right = _integer_slot(node.left), _integer_slot(node.right)
        if left is not None and right is not None:
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.FloorDiv) and right != 0:
                return left // right
            if isinstance(node.op, ast.Mod) and right != 0:
                return left % right
    return None


def _sync(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and _text(node.value.func) == "cute.arch.sync_threads"
        and not node.value.args
        and not node.value.keywords
    )


def _owner_loop(node: ast.AST, thread: str, threads: int) -> bool:
    if not isinstance(node, ast.For) or node.orelse:
        return False
    call = node.iter
    if not isinstance(call, ast.Call) or _text(call.func) != "range":
        return False
    if call.keywords or len(call.args) != 3 or not _constant(call.args[1], 1):
        return False
    start, _, step = call.args
    return (_text(start) == thread and _constant(step, threads)) or (
        _text(start) == f"{thread} // 32" and _constant(step, threads // 32)
    )


def immutable_publications(
    body: list[ast.AST], buffers: set[str], thread: str, threads: int
) -> dict[str, int]:
    """Final scalar writer in an owner loop, followed by CTA publication.

    Earlier scratch uses remain untouched. Reject every pointer/alias escape,
    ambiguous final writer, or non-scalar use in the final immutable epoch.
    Pooled allocation capacity is irrelevant to slot zero.
    """
    parents = {
        child: parent
        for root in body
        for parent in ast.walk(root)
        for child in ast.iter_child_nodes(parent)
    }
    roots = {
        child: index for index, root in enumerate(body) for child in ast.walk(root)
    }
    result = {}
    for name in buffers:
        references = [
            node
            for root in body
            for node in ast.walk(root)
            if isinstance(node, ast.Name) and node.id == name
        ]
        writes = []
        valid = bool(references)
        for node in references:
            subscript = parents.get(node)
            if not isinstance(subscript, ast.Subscript) or subscript.value is not node:
                valid = False
                break
            if isinstance(subscript.ctx, ast.Store):
                assignment = parents.get(subscript)
                if not isinstance(assignment, ast.Assign) or assignment.targets != [
                    subscript
                ]:
                    valid = False
                    break
                writes.append(assignment)
            elif not isinstance(subscript.ctx, ast.Load):
                valid = False
                break
        if not valid or not writes:
            continue
        # Traversal visits top-level roots in execution order. Require the
        # entire last writing root to contain exactly this one slot-zero
        # store; do not infer ordering between nested or conditional stores.
        writer = writes[-1]
        index = roots[writer]
        if any(
            roots[node] >= index
            and (
                _integer_slot(cast("ast.Subscript", parents[node]).slice) != 0
                or (roots[node] == index and parents.get(parents[node]) is not writer)
                or (
                    roots[node] > index
                    and not isinstance(
                        cast("ast.Subscript", parents[node]).ctx, ast.Load
                    )
                )
            )
            for node in references
        ):
            continue
        owner = body[index]
        if (
            not _owner_loop(owner, thread, threads)
            or index + 1 >= len(body)
            or not _sync(body[index + 1])
        ):
            continue
        # Only the existing warp leader guard may enclose the store.
        parent = parents[writer]
        guarded = isinstance(parent, ast.If)
        if isinstance(parent, ast.If):
            if _text(parent.test) != f"{thread} % 32 == 0" or any(
                not isinstance(n, ast.Pass) for n in parent.orelse
            ):
                continue
            parent = parents[parent]
        if parent is not owner:
            continue
        assert isinstance(owner, ast.For)
        assert isinstance(owner.iter, ast.Call)
        if _text(owner.iter.args[0]) != thread and not guarded:
            continue
        if any(
            isinstance(n, (ast.Break, ast.Continue, ast.Return, ast.Raise))
            for n in ast.walk(owner)
        ):
            continue
        result[name] = index + 1
    return result


_TYPES = frozenset(
    f"cutlass.{name}"
    for name in (
        "Boolean",
        "Int8",
        "Int16",
        "Int32",
        "Int64",
        "Uint8",
        "Uint16",
        "Uint32",
        "Uint64",
        "Float16",
        "BFloat16",
        "Float32",
        "Float64",
    )
)
_CALLS = _TYPES | frozenset(
    (
        "float",
        "cute.math.min",
        "cute.math.max",
        "cute.math.log2",
        "operator.add",
        "operator.sub",
        "operator.mul",
        "operator.truediv",
        "operator.floordiv",
        "operator.mod",
        "operator.and_",
        "operator.or_",
        "operator.xor",
        "operator.lshift",
        "operator.rshift",
        "operator.neg",
        "operator.eq",
        "operator.ne",
        "operator.lt",
        "operator.le",
        "operator.gt",
        "operator.ge",
    )
)


@dataclass(frozen=True)
class _Value:
    expression: ast.expr
    slots: frozenset[str]


def _recipe(
    node: ast.AST, values: dict[str, _Value], published: set[str]
) -> _Value | None:
    if isinstance(node, ast.Constant):
        return _Value(_clone(node), frozenset())
    if isinstance(node, ast.Name):
        return values.get(node.id)
    if isinstance(node, ast.Subscript):
        if (
            isinstance(node.value, ast.Name)
            and node.value.id in published
            and _integer_slot(node.slice) == 0
        ):
            return _Value(_clone(node), frozenset((node.value.id,)))
        return None
    if isinstance(node, ast.Call):
        func = _text(node.func)
        if func in _CALLS:
            receiver = None
        elif isinstance(node.func, ast.Attribute) and node.func.attr == "bitcast":
            receiver = _recipe(node.func.value, values, published)
            if receiver is None:
                return None
        else:
            return None
        args = []
        slots = set() if receiver is None else set(receiver.slots)
        for arg in node.args:
            # A bitcast's dtype is a literal SDK type, not a scalar load.
            if func.endswith(".bitcast") and _text(arg) in _TYPES:
                args.append(_clone(arg))
                continue
            item = _recipe(arg, values, published)
            if item is None:
                return None
            args.append(_clone(item.expression))
            slots.update(item.slots)
        keywords = []
        for keyword in node.keywords:
            item = _recipe(keyword.value, values, published)
            if keyword.arg is None or item is None:
                return None
            keywords.append(ast.keyword(arg=keyword.arg, value=_clone(item.expression)))
            slots.update(item.slots)
        function = (
            _clone(node.func)
            if receiver is None
            else ast.Attribute(
                value=_clone(receiver.expression), attr="bitcast", ctx=ast.Load()
            )
        )
        return _Value(
            ast.Call(func=function, args=args, keywords=keywords), frozenset(slots)
        )
    if isinstance(node, (ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.IfExp)):
        result = _clone(node)
        slots = set()
        for field, value in ast.iter_fields(node):
            if isinstance(value, ast.expr):
                item = _recipe(value, values, published)
                if item is None:
                    return None
                setattr(result, field, _clone(item.expression))
                slots.update(item.slots)
            elif isinstance(value, list):
                output = []
                for child in value:
                    if isinstance(child, ast.expr):
                        item = _recipe(child, values, published)
                        if item is None:
                            return None
                        output.append(_clone(item.expression))
                        slots.update(item.slots)
                    else:
                        output.append(_clone(child))
                setattr(result, field, output)
        return _Value(result, frozenset(slots))
    return None


def _full_loop(node: ast.AST, thread: str, threads: int) -> bool:
    """All CTA threads execute the first body iteration; no tail or zero-trip."""
    if not isinstance(node, ast.For) or node.orelse:
        return False
    call = node.iter
    if (
        not isinstance(call, ast.Call)
        or _text(call.func) != "range"
        or call.keywords
        or len(call.args) != 3
        or _text(call.args[0]) != thread
        or not isinstance(call.args[1], ast.Constant)
        or type(call.args[1].value) is not int
        or call.args[1].value < threads
        or not _constant(call.args[2], threads)
    ):
        return False
    return not any(
        isinstance(n, (ast.Break, ast.Continue, ast.Return, ast.Raise))
        for n in ast.walk(node)
    )


def _full_first_slot(node: ast.AST, thread: str, threads: int) -> list[ast.stmt] | None:
    """Unconditionally executed body of a canonical register loop's slot zero.

    Later slots may be partial. Only this exact physical-domain guard can be
    crossed; arbitrary inner masks remain guarded and cannot seed retention.
    """
    if not isinstance(node, ast.For) or node.orelse:
        return None
    call = node.iter
    if (
        not isinstance(node.target, ast.Name)
        or node.target.id == thread
        or not isinstance(call, ast.Call)
        or _text(call.func) != "cutlass.range_constexpr"
        or call.keywords
        or len(call.args) != 1
        or not isinstance(call.args[0], ast.Constant)
        or type(call.args[0].value) is not int
        or call.args[0].value <= 0
        or len(node.body) != 2
    ):
        return None
    assignment, guard = node.body
    if (
        not isinstance(assignment, ast.Assign)
        or len(assignment.targets) != 1
        or not isinstance(assignment.targets[0], ast.Name)
        or assignment.targets[0].id in {thread, node.target.id}
        or _text(assignment.value) != f"{thread} + {node.target.id} * {threads}"
        or not isinstance(guard, ast.If)
        or any(not isinstance(n, ast.Pass) for n in guard.orelse)
        or not isinstance(guard.test, ast.Compare)
        or _text(guard.test.left) != assignment.targets[0].id
        or len(guard.test.ops) != 1
        or not isinstance(guard.test.ops[0], ast.Lt)
        or len(guard.test.comparators) != 1
        or not isinstance(guard.test.comparators[0], ast.Constant)
        or type(guard.test.comparators[0].value) is not int
        or guard.test.comparators[0].value < threads
    ):
        return None
    protected = {thread, node.target.id, assignment.targets[0].id}
    if any(
        isinstance(n, (ast.Break, ast.Continue, ast.Return, ast.Raise))
        or (
            isinstance(n, ast.Name)
            and isinstance(n.ctx, ast.Store)
            and n.id in protected
        )
        for n in ast.walk(guard)
    ):
        return None
    return guard.body


def reuse_published_scalars(
    body: list[ast.AST],
    buffers: set[str],
    thread: str,
    threads: int,
    new_var: Callable[[str], str],
) -> int:
    """Hoist only already-unconditionally-executed typed scalar assignments.

    First uses under unproved masks/branches or a partial or absent first
    iteration cannot seed a hoist. Later guarded uses can reuse a dominating
    previously evaluated value.
    Every key is the exact typed AST with current SSA definitions; no arithmetic
    reassociation, literal coercion or per-value reciprocal substitution.
    """
    if any(
        isinstance(n, ast.Name)
        and isinstance(n.ctx, ast.Store)
        and n.id in {"cutlass", "cute", "operator", "float"}
        for s in body
        for n in ast.walk(s)
    ):
        return 0
    publications = immutable_publications(body, buffers, thread, threads)
    if not publications:
        return 0
    cache: dict[str, str] = {}
    values: dict[str, _Value] = {}
    changed = 0

    def block(
        statements: Sequence[ast.AST],
        env: dict[str, _Value],
        published: set[str],
        hoists: list[ast.AST],
        can_hoist: bool,
    ) -> None:
        nonlocal changed
        for statement in statements:
            if (
                isinstance(statement, ast.Assign)
                and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
            ):
                name = statement.targets[0].id
                recipe = _recipe(statement.value, env, published)
                env.pop(name, None)
                if recipe is None:
                    continue
                if not recipe.slots:
                    env[name] = recipe
                    continue
                # ast.dump distinguishes Bool/Int/Float constants and signed
                # zero. SDK casts/bitcasts, keyword flags and order are retained.
                key = ast.dump(recipe.expression, include_attributes=False)
                if key not in cache and can_hoist:
                    cache[key] = new_var("fragment_published_scalar")
                    hoists.append(
                        ast.fix_missing_locations(
                            ast.copy_location(
                                ast.Assign(
                                    targets=[ast.Name(id=cache[key], ctx=ast.Store())],
                                    value=_clone(recipe.expression),
                                ),
                                statement,
                            )
                        )
                    )
                if key in cache:
                    statement.value = ast.Name(id=cache[key], ctx=ast.Load())
                    env[name] = _Value(_clone(statement.value), recipe.slots)
                    changed += 1
                else:
                    # Keep guarded current definitions local. They cannot
                    # escape this block or seed speculative arithmetic.
                    env[name] = recipe
            elif isinstance(statement, (ast.If, ast.For, ast.While)):
                assigned = {
                    n.id
                    for n in ast.walk(statement)
                    if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
                }
                nested = {k: v for k, v in env.items() if k not in assigned}
                block(statement.body, dict(nested), published, hoists, False)
                block(statement.orelse, dict(nested), published, hoists, False)
                for name in assigned:
                    env.pop(name, None)
            else:
                for node in ast.walk(statement):
                    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                        env.pop(node.id, None)

    result = []
    for index, statement in enumerate(body):
        published = {name for name, after in publications.items() if index > after}
        hoists: list[ast.AST] = []
        guaranteed = _full_first_slot(statement, thread, threads)
        if _full_loop(statement, thread, threads):
            assert isinstance(statement, ast.For)
            guaranteed = statement.body
        if guaranteed is not None:
            assert isinstance(statement, ast.For)
            assigned = {
                n.id
                for n in ast.walk(statement)
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
            }
            local = {k: v for k, v in values.items() if k not in assigned}
            block(guaranteed, local, published, hoists, True)
            for name in assigned:
                values.pop(name, None)
        else:
            block([statement], values, published, hoists, False)
        result.extend(hoists)
        result.append(statement)
    body[:] = result
    return changed


def scalar_source_candidates(graphs: list[GraphInfo]) -> bool:
    """Cheap capability fact; final emitted storage/epoch proof stays mandatory."""
    from ...language import memory_ops
    from ..inductor_lowering import ReductionLowering

    return any(
        isinstance(node.meta.get("val"), torch.Tensor)
        and node.meta["val"].ndim == 0
        and (
            isinstance(node.meta.get("lowering"), ReductionLowering)
            or node.target is memory_ops.load
        )
        for graph in graphs
        for node in graph.graph.nodes
    )


@dataclass(frozen=True)
class PublishedScalarRequest:
    """An emitted fragment owner, re-proved after arithmetic lowering.

    Delaying retention avoids making a new denominator invariant for the older
    reciprocal/FMA passes. Names alone are not a proof: the final body must
    still contain one exact thread binding and no storage references outside
    that owner's scope except its original shared allocation.
    """

    buffers: frozenset[str]
    thread: str
    threads: int

    def lower(
        self,
        body: list[ast.stmt],
        renames: dict[str, str],
        new_var: Callable[[str], str],
    ) -> int:
        thread = renames.get(self.thread, self.thread)
        buffers = {renames.get(name, name) for name in self.buffers}
        scopes: list[tuple[list[ast.stmt], int]] = []

        def visit(statements: list[ast.stmt]) -> None:
            for index, statement in enumerate(statements):
                if (
                    isinstance(statement, ast.Assign)
                    and len(statement.targets) == 1
                    and isinstance(statement.targets[0], ast.Name)
                    and statement.targets[0].id == thread
                    and _text(statement.value)
                    == "cutlass.Int32(cute.arch.thread_idx()[0])"
                ):
                    scopes.append((statements, index))
                if isinstance(statement, (ast.For, ast.If, ast.While)):
                    visit(statement.body)
                    visit(statement.orelse)

        visit(body)
        stores = [
            n
            for statement in body
            for n in ast.walk(statement)
            if isinstance(n, ast.Name)
            and isinstance(n.ctx, ast.Store)
            and n.id == thread
        ]
        if len(scopes) != 1 or len(stores) != 1:
            return 0
        scope, index = scopes[0]
        tail: list[ast.AST] = list(scope[index:])
        owned = {n for statement in tail for n in ast.walk(statement)}
        allocations: set[ast.AST] = set()
        for statement in body:
            for n in ast.walk(statement):
                if (
                    isinstance(n, ast.Assign)
                    and len(n.targets) == 1
                    and isinstance(n.targets[0], ast.Name)
                    and n.targets[0].id in buffers
                    and isinstance(n.value, ast.Call)
                    and isinstance(n.value.func, ast.Attribute)
                    and n.value.func.attr == "allocate_tensor"
                ):
                    allocations.add(n.targets[0])
        if any(
            isinstance(n, ast.Name)
            and n.id in buffers
            and n not in owned
            and n not in allocations
            for statement in body
            for n in ast.walk(statement)
        ):
            return 0
        changed = reuse_published_scalars(tail, buffers, thread, self.threads, new_var)
        scope[index:] = cast("list[ast.stmt]", tail)
        return changed
