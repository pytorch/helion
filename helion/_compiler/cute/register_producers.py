"""Cache repeated typed producers in their proved per-thread register slots."""

from __future__ import annotations

import ast
from dataclasses import dataclass
import operator
from typing import TYPE_CHECKING
from typing import cast

from ... import exc
from .published_scalars import _TYPES
from .published_scalars import _clone
from .published_scalars import _recipe
from .published_scalars import _text
from .published_scalars import _Value
from .published_scalars import immutable_publications
from .register_snapshots import MAX_SLOTS
from .register_snapshots import SnapshotOwner

if TYPE_CHECKING:
    from collections.abc import Callable


def _assignment(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
    )


def _stores(node: ast.AST) -> set[str]:
    return {
        n.id
        for n in ast.walk(node)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
    }


def _dtype(node: ast.AST) -> str | None:
    if isinstance(node, ast.Call) and _text(node.func) in _TYPES:
        return _text(node.func)
    if isinstance(node, ast.IfExp):
        left, right = _dtype(node.body), _dtype(node.orelse)
        return left if left == right else None
    return None


@dataclass
class _Owner:
    loop: ast.For
    statements: list[ast.stmt]
    slot: str
    size: int
    slots: int
    root: int
    seed: bool
    active_size: int


def _contribution_domain(
    statements: list[ast.stmt], index: str, slot: str, size: int
) -> tuple[list[ast.stmt], int] | None:
    """Recognize a static prefix of the already-proved owner coordinate.

    Atomic index evaluation remains inside its logical-domain guard. Only
    coordinate bounds can seed a cache here; value-dependent masks still fail
    the ordinary masked-first-use rule.
    """
    if not statements or not isinstance(statements[-1], ast.If):
        return None
    guard = statements[-1]
    if any(not isinstance(node, ast.Pass) for node in guard.orelse):
        return None
    domains: dict[str, int] = {}
    owner = SnapshotOwner(index, slot, size)

    def bound(node: ast.expr) -> int | None:
        if isinstance(node, ast.Name):
            return domains.get(node.id)
        if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And):
            bounds = [bound(value) for value in node.values]
            if all(value is not None for value in bounds):
                return min(cast("list[int]", bounds))
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            right = node.comparators[0]
            coordinate: ast.expr
            limit: int
            if (
                isinstance(node.ops[0], ast.Lt)
                and isinstance(right, ast.Constant)
                and type(right.value) is int
                and right.value > 0
            ):
                coordinate, limit = node.left, min(size, right.value)
            elif (
                isinstance(node.ops[0], ast.LtE)
                and isinstance(node.left, ast.Constant)
                and type(node.left.value) is int
                and node.left.value == 0
            ):
                coordinate, limit = right, size
            else:
                return None
            try:
                owner.prove(_text(coordinate))
            except exc.InvalidConfig:
                return None
            return limit
        return None

    for statement in statements[:-1]:
        if not _assignment(statement):
            return None
        statement = cast("ast.Assign", statement)
        limit = bound(statement.value)
        if limit is None:
            return None
        domains[cast("ast.Name", statement.targets[0]).id] = limit
    limit = bound(guard.test)
    return (guard.body, limit) if limit is not None else None


def _owner(
    node: ast.AST, thread: str, threads: int, root: int, seed: bool
) -> _Owner | None:
    if (
        not isinstance(node, ast.For)
        or node.orelse
        or not isinstance(node.target, ast.Name)
        or not isinstance(node.iter, ast.Call)
        or _text(node.iter.func) != "cutlass.range_constexpr"
        or node.iter.keywords
        or len(node.iter.args) != 1
        or not isinstance(node.iter.args[0], ast.Constant)
        or type(node.iter.args[0].value) is not int
        or not 0 < node.iter.args[0].value <= MAX_SLOTS
        or len(node.body) != 2
    ):
        return None
    index, guard = node.body
    slot = node.target.id
    if (
        not _assignment(index)
        or not isinstance(guard, ast.If)
        or any(not isinstance(n, ast.Pass) for n in guard.orelse)
    ):
        return None
    index = cast("ast.Assign", index)
    name = cast("ast.Name", index.targets[0]).id
    if (
        len({thread, slot, name}) != 3
        or _text(index.value) != f"{thread} + {slot} * {threads}"
        or not isinstance(guard.test, ast.Compare)
        or _text(guard.test.left) != name
        or len(guard.test.ops) != 1
        or not isinstance(guard.test.ops[0], ast.Lt)
        or len(guard.test.comparators) != 1
        or not isinstance(guard.test.comparators[0], ast.Constant)
        or type(guard.test.comparators[0].value) is not int
    ):
        return None
    size = guard.test.comparators[0].value
    slots = node.iter.args[0].value
    if (
        size <= 0
        or (size + threads - 1) // threads != slots
        or _stores(guard) & {thread, slot, name}
        or any(
            isinstance(n, (ast.Break, ast.Continue, ast.Return, ast.Raise))
            for n in ast.walk(guard)
        )
    ):
        return None
    domain = _contribution_domain(guard.body, name, slot, size)
    statements, active_size = domain if domain is not None else (guard.body, size)
    return _Owner(node, statements, slot, size, slots, root, seed, active_size)


@dataclass
class _Use:
    statement: ast.Assign
    owner: _Owner
    recipe: _Value
    dtype: str
    closure: frozenset[ast.Assign]


def cache_register_producers(
    body: list[ast.stmt],
    buffers: set[str],
    snapshots: dict[str, int],
    thread: str,
    threads: int,
    new_var: Callable[[str], str],
) -> int:
    """One bounded cache, with first-use execution and physical epoch proofs.

    Snapshot identities come from the existing readonly-load/coordinate proof.
    Scalar recipes may read only final, collectively published shared slots.
    Inspect final physical references as well: pointer aliases, later writes,
    remapped slots and unknown calls never become cache dependencies.
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
    if any(
        _stores(root) & {"cutlass", "cute", "operator", "float"} for root in body
    ) or any(
        isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef))
        for root in body
        for n in ast.walk(root)
    ):
        return 0
    publications = immutable_publications(
        cast("list[ast.AST]", body), buffers, thread, threads
    )
    final_writes: dict[str, int] = {}
    for name in snapshots:
        writes = []
        valid = True
        for root in body:
            for node in ast.walk(root):
                if not isinstance(node, ast.Name) or node.id != name:
                    continue
                parent = parents[node]
                if isinstance(parent, ast.Subscript) and parent.value is node:
                    if isinstance(parent.ctx, ast.Store):
                        writes.append(roots[node])
                    elif not isinstance(parent.ctx, ast.Load):
                        valid = False
                elif isinstance(parent, ast.Assign) and parent.targets == [node]:
                    valid &= (
                        isinstance(parent.value, ast.Call)
                        and _text(parent.value.func) == "cute.make_rmem_tensor"
                    )
                    writes.append(roots[node])
                elif isinstance(parent, ast.Attribute) and parent.attr == "fill":
                    call = parents[parent]
                    valid &= isinstance(call, ast.Call) and call.func is parent
                    writes.append(roots[node])
                else:
                    valid = False
        if valid and writes:
            final_writes[name] = max(writes)

    groups: dict[str, list[_Use]] = {}
    removable: list[tuple[list[ast.stmt], ast.Assign]] = []
    global_values: dict[str, _Value] = {}
    slot_symbol = new_var("fragment_producer_coordinate")

    def recipe(
        expression: ast.expr,
        env: dict[str, _Value],
        published: set[str],
        owner: _Owner | None,
    ) -> _Value | None:
        values = dict(env)

        class Reads(ast.NodeTransformer):
            def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
                if (
                    owner is not None
                    and isinstance(node.value, ast.Name)
                    and node.value.id in final_writes
                    and final_writes[node.value.id] < owner.root
                    and snapshots[node.value.id] == owner.size
                    and _text(node.slice) == owner.slot
                    and isinstance(node.ctx, ast.Load)
                ):
                    name = new_var("fragment_producer_input")
                    values[name] = _Value(
                        ast.Subscript(
                            value=_clone(node.value),
                            slice=ast.Name(id=slot_symbol, ctx=ast.Load()),
                            ctx=ast.Load(),
                        ),
                        frozenset((node.value.id,)),
                    )
                    return ast.Name(id=name, ctx=ast.Load())
                return node

        return _recipe(
            cast("ast.expr", Reads().visit(_clone(expression))), values, published
        )

    def scan(
        statements: list[ast.stmt],
        env: dict[str, _Value],
        published: set[str],
        owner: _Owner | None,
    ) -> None:
        closures: dict[str, frozenset[ast.Assign]] = {}
        for node in statements:
            if _assignment(node):
                node = cast("ast.Assign", node)
                name = cast("ast.Name", node.targets[0]).id
                value = recipe(node.value, env, published, owner)
                closure = frozenset().union(
                    *(
                        closures.get(item.id, frozenset())
                        for item in ast.walk(node.value)
                        if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Load)
                    )
                )
                env.pop(name, None)
                closures.pop(name, None)
                if value is not None:
                    env[name] = value
                    closures[name] = closure | {node}
                    if owner is not None:
                        removable.append((statements, node))
                        dtype = _dtype(value.expression)
                        inputs = value.slots & final_writes.keys()
                        if dtype is not None and inputs:
                            key = ast.dump(value.expression, include_attributes=False)
                            groups.setdefault(key, []).append(
                                _Use(node, owner, value, dtype, closure)
                            )
            elif isinstance(node, ast.If):
                assigned = _stores(node)
                nested = {k: v for k, v in env.items() if k not in assigned}
                # A masked first use may not seed storage. A later use can only
                # reuse a cache filled by an earlier unconditional owner loop.
                for arm in (node.body, node.orelse):
                    if owner is None:
                        for child in arm:
                            physical = _owner(
                                child, thread, threads, roots[child], False
                            )
                            if physical is not None:
                                scan(
                                    physical.statements,
                                    dict(nested),
                                    published,
                                    physical,
                                )
                    else:
                        scan(arm, dict(nested), published, None)
                for name in assigned:
                    env.pop(name, None)
                    closures.pop(name, None)
            else:
                for name in _stores(node):
                    env.pop(name, None)
                    closures.pop(name, None)

    for index, node in enumerate(body):
        published = {name for name, after in publications.items() if index > after}
        owner = _owner(node, thread, threads, index, True)
        if owner is not None:
            env = {k: v for k, v in global_values.items() if k not in _stores(node)}
            scan(owner.statements, env, published, owner)
            for name in _stores(node):
                global_values.pop(name, None)
        else:
            scan([node], global_values, published, None)

    scored = []
    for uses in groups.values():
        first = uses[0]
        # A later wider consumer retains its original recipe; its presence
        # need not disable reuse between the proved narrower consumers.
        uses = [use for use in uses if use.owner.active_size <= first.owner.active_size]
        cost = sum(
            isinstance(n, (ast.BinOp, ast.UnaryOp, ast.Call, ast.Compare, ast.IfExp))
            for n in ast.walk(first.recipe.expression)
        )
        if (
            first.owner.seed
            and len({id(use.owner.loop) for use in uses}) > 1
            and cost >= 8
            and all(use.owner.size == first.owner.size for use in uses)
            and all(use.owner.root >= first.owner.root for use in uses)
        ):
            scored.append(((len(uses) - 1) * cost, uses))
    if not scored:
        return 0
    _, uses = max(scored, key=operator.itemgetter(0))
    first = uses[0]
    name = new_var("fragment_register_producer")
    allocation = ast.parse(
        f"{name} = cute.make_rmem_tensor(({first.owner.slots},), {first.dtype})\n"
        f"{name}.fill({first.dtype}(0))"
    ).body
    position = body.index(first.owner.loop)
    body[position:position] = allocation
    first.owner.statements.insert(
        first.owner.statements.index(first.statement) + 1,
        ast.parse(
            f"{name}[{first.owner.slot}] = {first.dtype}({_text(first.statement.targets[0])})"
        ).body[0],
    )
    for use in uses[1:]:
        use.statement.value = ast.parse(f"{name}[{use.owner.slot}]", mode="eval").body
    protected = {id(use.statement) for use in uses}
    replaced_dependencies = set().union(*(use.closure for use in uses[1:]))
    while True:
        loaded = {
            n.id
            for root in body
            for n in ast.walk(root)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
        }
        dead = [
            (statements, node)
            for statements, node in removable
            if id(node) not in protected
            and node in replaced_dependencies
            and cast("ast.Name", node.targets[0]).id not in loaded
        ]
        if not dead:
            break
        for statements, node in dead:
            statements.remove(node)
            removable.remove((statements, node))
    return len(uses) - 1


@dataclass(frozen=True)
class RegisterProducerRequest:
    buffers: frozenset[str]
    snapshots: tuple[tuple[str, int], ...]
    thread: str
    threads: int

    @staticmethod
    def independent_allocations(prefix: list[ast.stmt], buffers: set[str]) -> bool:
        """Distinct names must be distinct monotonic shared allocations.

        Fragment lifetime reuse retains the same physical buffer name. Verify
        the final allocation calls too, so a later alias/view or allocator
        reset cannot hide a write behind another name.
        """
        allocations: dict[str, tuple[str, ast.Assign]] = {}
        for node in prefix:
            if not _assignment(node):
                continue
            node = cast("ast.Assign", node)
            name = cast("ast.Name", node.targets[0]).id
            if name not in buffers:
                continue
            value = node.value
            if (
                name in allocations
                or not isinstance(value, ast.Call)
                or not isinstance(value.func, ast.Attribute)
                or value.func.attr != "allocate_tensor"
                or not isinstance(value.func.value, ast.Name)
            ):
                return False
            allocations[name] = (value.func.value.id, node)
        if allocations.keys() != buffers:
            return False
        allocators = {name for name, _ in allocations.values()}
        for allocator in allocators:
            definitions = [
                node
                for node in prefix
                if _assignment(node)
                and _text(cast("ast.Assign", node).targets[0]) == allocator
                and _text(cast("ast.Assign", node).value)
                == "cutlass.utils.SmemAllocator()"
            ]
            if len(definitions) != 1:
                return False
            allowed = {id(n) for n in ast.walk(definitions[0])}
            for owner, allocation in allocations.values():
                if owner == allocator:
                    allowed.update(id(n) for n in ast.walk(allocation))
                    if prefix.index(allocation) <= prefix.index(definitions[0]):
                        return False
            if any(
                isinstance(n, ast.Name) and n.id == allocator and id(n) not in allowed
                for root in prefix
                for n in ast.walk(root)
            ):
                return False
        return True

    def lower(
        self,
        body: list[ast.stmt],
        renames: dict[str, str],
        new_var: Callable[[str], str],
    ) -> int:
        thread = renames.get(self.thread, self.thread)
        buffers = {renames.get(name, name) for name in self.buffers}
        snapshots = {renames.get(name, name): size for name, size in self.snapshots}
        matches = [
            index
            for index, node in enumerate(body)
            if _assignment(node)
            and _text(cast("ast.Assign", node).targets[0]) == thread
            and _text(cast("ast.Assign", node).value)
            == "cutlass.Int32(cute.arch.thread_idx()[0])"
        ]
        if len(matches) != 1 or sum(thread in _stores(node) for node in body) != 1:
            return 0
        index = matches[0]
        if not self.independent_allocations(body[:index], buffers):
            return 0
        # Initial scope is a complete root body. Do not hoist storage outside
        # nested roots or accept a dependency escaping that root's lifetime.
        tail = body[index:]
        allocators = {
            cast(
                "ast.Name",
                cast(
                    "ast.Attribute",
                    cast("ast.Call", cast("ast.Assign", node).value).func,
                ).value,
            ).id
            for node in body[:index]
            if _assignment(node)
            and _text(cast("ast.Assign", node).targets[0]) in buffers
        }
        if any(
            isinstance(n, ast.Name) and n.id in allocators
            for node in tail
            for n in ast.walk(node)
        ):
            return 0
        for node in body[:index]:
            if any(
                isinstance(n, ast.Name)
                and isinstance(n.ctx, ast.Load)
                and n.id in buffers | snapshots.keys()
                for n in ast.walk(node)
            ):
                return 0
        changed = cache_register_producers(
            tail, buffers, snapshots, thread, self.threads, new_var
        )
        if changed:
            body[index:] = tail
        return changed
