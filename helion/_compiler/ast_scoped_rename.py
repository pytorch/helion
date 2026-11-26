"""Preserve the local names of Python expressions during compiler variable coalescing."""

from __future__ import annotations

import ast
from collections import defaultdict
import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import cast

if TYPE_CHECKING:
    from collections.abc import Iterator

_Node = TypeVar("_Node", bound=ast.AST)

_Comprehension = ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp
EXPRESSION_SCOPES = (
    ast.Lambda,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
)


def _lambda_locals(fn: ast.Lambda) -> set[str]:
    names = {a.arg for a in ast.iter_child_nodes(fn.args) if isinstance(a, ast.arg)}
    # Walrus assignments in comprehensions belong to the containing function.
    # A nested lambda owns its body, but its defaults execute in this scope.
    pending: list[ast.AST] = [fn.body]
    while pending:
        node = pending.pop()
        if isinstance(node, ast.NamedExpr) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        if isinstance(node, ast.Lambda):
            pending.append(node.args)
        else:
            pending.extend(ast.iter_child_nodes(node))
    return names


def _target_names(target: ast.expr) -> set[str]:
    names: set[str] = set()
    pending = [target]
    while pending:
        item = pending.pop()
        if isinstance(item, ast.Name):
            names.add(item.id)
        elif isinstance(item, ast.Starred):
            pending.append(item.value)
        elif isinstance(item, (ast.Tuple, ast.List)):
            pending.extend(item.elts)
    return names


def _lambda(at: ast.AST, args: list[str], body: ast.expr) -> ast.Lambda:
    return ast.copy_location(
        ast.Lambda(
            args=ast.arguments(
                posonlyargs=[],
                args=[ast.copy_location(ast.arg(name), at) for name in args],
                kwonlyargs=[],
                kw_defaults=[],
                defaults=[],
            ),
            body=body,
        ),
        at,
    )


@dataclass
class _Scope:
    names: set[str]
    is_lambda: bool
    refs: dict[str, list[ast.Name]] = dataclasses.field(
        default_factory=lambda: defaultdict(list)
    )
    aliases: dict[str, str] = dataclasses.field(default_factory=dict)


def _shell(node: _Node) -> _Node:
    result = cast("_Node", ast.AST.__new__(type(node)))
    vars(result).update(vars(node))
    return result


def _copy_occurrence(root: _Node) -> _Node:
    # Templates splice the same object at multiple positions. Detach every
    # occurrence *inside this expression*; deepcopy preserves that sharing.
    # Keep ExtendedAST locations, types and other compiler annotations shallow.
    result = _shell(root)
    work: list[tuple[ast.AST, ast.AST]] = [(root, result)]
    while work:
        node, owned = work.pop()
        for field, value in ast.iter_fields(node):
            if isinstance(value, ast.AST):
                child = _shell(value)
                setattr(owned, field, child)
                work.append((value, child))
            elif isinstance(value, list):
                items = list(value)  # Never modify FX's immutable input list.
                for i, item in enumerate(items):
                    if isinstance(item, ast.AST):
                        child = _shell(item)
                        items[i] = child
                        work.append((item, child))
                setattr(
                    owned, field, items if type(value) is list else type(value)(items)
                )
    return result


@dataclass(frozen=True)
class _Slot:
    parent: ast.AST
    field: str
    index: int | None = None

    def put(self, node: ast.AST) -> None:
        if self.index is None:
            setattr(self.parent, self.field, node)
        else:
            values = getattr(self.parent, self.field)
            if values[self.index] is node:
                return
            if type(values) is list:
                values[self.index] = node
            else:
                items = list(values)
                items[self.index] = node
                setattr(self.parent, self.field, type(values)(items))


def _children(node: ast.AST) -> Iterator[tuple[ast.AST, _Slot]]:
    for field, value in ast.iter_fields(node):
        if isinstance(value, ast.AST):
            yield value, _Slot(node, field)
        elif isinstance(value, list):
            for i, item in enumerate(value):
                if isinstance(item, ast.AST):
                    yield item, _Slot(node, field, i)


@dataclass
class _Finish:
    node: ast.Lambda | _Comprehension
    scope: _Scope


class _ScopedRename:
    def __init__(self, root: ast.AST, renames: dict[str, str]) -> None:
        self.root = root
        self.renames = renames
        self.used: set[str] | None = None
        self.sequence = 0

    def fresh(self) -> str:
        if self.used is None:
            self.used = set(self.renames) | set(self.renames.values())
            for node in ast.walk(self.root):
                if isinstance(node, ast.Name):
                    self.used.add(node.id)
                elif isinstance(node, ast.arg):
                    self.used.add(node.arg)
        while True:
            name = f"_helion_scoped_{self.sequence}"
            self.sequence += 1
            if name not in self.used:
                self.used.add(name)
                return name

    def name(self, node: ast.Name, scopes: tuple[_Scope, ...]) -> ast.expr:
        for scope in reversed(scopes):
            if node.id in scope.names:
                if not scope.is_lambda:
                    scope.refs[node.id].append(node)
                return node
        target = self.renames.get(node.id, node.id)
        if target != node.id:
            for scope in scopes:
                if target in scope.names:
                    if target not in scope.aliases:
                        scope.aliases[target] = self.fresh()
                    if scope.is_lambda:
                        # Keep the actual callback's positional and keyword ABI.
                        # Resolve the outer name at invocation, not at creation.
                        get = ast.copy_location(
                            ast.Name(id=scope.aliases[target], ctx=ast.Load()), node
                        )
                        return ast.copy_location(ast.Call(get, [], []), node)
            node.id = target
        return node

    def wrap(self, node: ast.Lambda, scope: _Scope) -> ast.expr:
        if not scope.aliases:
            return node
        params = list(scope.aliases.values())
        values: list[ast.expr] = [
            _lambda(
                node, [], ast.copy_location(ast.Name(id=name, ctx=ast.Load()), node)
            )
            for name in scope.aliases
        ]
        # Defaults execute once, in their original scope, including walrus writes.
        for key in ("defaults", "kw_defaults"):
            defaults = getattr(node.args, key)
            updated = list(defaults)
            for index, default in enumerate(defaults):
                if default is not None:
                    name = self.fresh()
                    params.append(name)
                    values.append(default)
                    updated[index] = ast.copy_location(
                        ast.Name(id=name, ctx=ast.Load()), default
                    )
            setattr(node.args, key, type(defaults)(updated))
        return ast.copy_location(
            ast.Call(_lambda(node, params, node), values, []), node
        )

    def rewrite(self, original: ast.expr) -> ast.expr:
        if not any(
            isinstance(n, ast.Name) and self.renames.get(n.id, n.id) != n.id
            for n in ast.walk(original)
        ):
            return original
        root = _copy_occurrence(original)
        holder = ast.Expression(body=root)
        work: list[tuple[ast.AST | _Finish, _Slot, tuple[_Scope, ...]]] = [
            (root, _Slot(holder, "body"), ())
        ]
        while work:
            node, slot, scopes = work.pop()
            if isinstance(node, _Finish):
                if isinstance(node.node, ast.Lambda):
                    slot.put(self.wrap(node.node, node.scope))
                else:
                    for name, fresh in node.scope.aliases.items():
                        for ref in node.scope.refs[name]:
                            ref.id = fresh
            elif isinstance(node, ast.Name):
                slot.put(self.name(node, scopes))
            elif isinstance(node, ast.Lambda):
                scope = _Scope(_lambda_locals(node), is_lambda=True)
                work.extend(
                    (
                        (_Finish(node, scope), slot, scopes),
                        (node.body, _Slot(node, "body"), (*scopes, scope)),
                        (node.args, _Slot(node, "args"), scopes),
                    )
                )
            elif isinstance(node, _Comprehension):
                first = node.generators[0]
                scope = _Scope(
                    {
                        name
                        for gen in node.generators
                        for name in _target_names(gen.target)
                    },
                    is_lambda=False,
                )
                inner = (*scopes, scope)
                work.append((_Finish(node, scope), slot, scopes))
                for child, child_slot in _children(node):
                    if isinstance(child, ast.comprehension):
                        for value, elem in _children(child):
                            context = (
                                scopes
                                if child is first and elem.field == "iter"
                                else inner
                            )
                            work.append((value, elem, context))
                    else:
                        work.append((child, child_slot, inner))
            else:
                work.extend(
                    (child, child_slot, scopes) for child, child_slot in _children(node)
                )
        return holder.body


def rename(root: ast.AST, renames: dict[str, str]) -> ast.AST:
    if isinstance(root, EXPRESSION_SCOPES):
        rewritten = _ScopedRename(root, renames).rewrite(root)
        if type(rewritten) is type(root) and rewritten is not root:
            for key, value in ast.iter_fields(rewritten):
                setattr(root, key, value)
            return root
        return rewritten
    # Inspect and detach scoped occurrences before applying any flat writes:
    # the template builder may share those exact nodes with the outer body.
    scoped = None
    writes: dict[ast.Name, str] = {}
    replacements: dict[ast.AST, ast.expr] = {}
    parents: set[ast.AST] = set()
    pending = [root]
    while pending:
        node = pending.pop()
        if isinstance(node, ast.Name):
            if (target := renames.get(node.id, node.id)) != node.id:
                writes[node] = target
        else:
            for child in ast.iter_child_nodes(node):
                if isinstance(child, EXPRESSION_SCOPES):
                    if scoped is None:
                        scoped = _ScopedRename(root, renames)
                    if child not in replacements:
                        replacements[child] = scoped.rewrite(child)
                    if replacements[child] is not child:
                        parents.add(node)
                else:
                    pending.append(child)
    for parent in parents:
        for field, value in ast.iter_fields(parent):
            if isinstance(value, ast.AST) and value in replacements:
                setattr(parent, field, replacements[value])
            elif isinstance(value, list) and any(
                isinstance(item, ast.AST) and replacements.get(item, item) is not item
                for item in value
            ):
                # Rebuild just this field; its owner may be an immutable FX list.
                setattr(
                    parent,
                    field,
                    type(value)(
                        [
                            replacements.get(item, item)
                            if isinstance(item, ast.AST)
                            else item
                            for item in value
                        ]
                    ),
                )
    for name, target in writes.items():
        name.id = target
    return root
