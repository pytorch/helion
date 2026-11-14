"""Conservative byte packing for ordinary, synchronous fragment storage."""

from __future__ import annotations

import ast
import dataclasses
from itertools import combinations
from typing import TYPE_CHECKING

from ... import exc
from .fragment_storage import aligned_shared_bytes
from .scalar_ast import SCALAR_CALLS
from .scalar_ast import SCALAR_CAST_CALLS
from .scalar_ast import is_flat_scalar_access

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Sequence

    import torch


class _Refused(Exception):
    pass


@dataclasses.dataclass(frozen=True)
class ArenaView:
    name: str
    dtype: torch.dtype
    count: int
    offset: int
    byte_size: int


@dataclasses.dataclass(frozen=True)
class SharedArenaPlan:
    views: tuple[ArenaView, ...]
    byte_size: int
    # Ordinary Python AST text strips compiler-owned tracing metadata.
    body_source: str


def _is_barrier(node: ast.stmt) -> bool:
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and ast.unparse(node.value) == "cute.arch.sync_threads()"
    )


def _references(node: ast.AST, names: frozenset[str]) -> set[str]:
    return {
        child.id
        for child in ast.walk(node)
        if isinstance(child, ast.Name) and child.id in names
    }


def _global_pointer(node: ast.AST) -> bool:
    if (
        isinstance(node, ast.Attribute)
        and node.attr == "iterator"
        and isinstance(node.value, ast.Name)
    ):
        return True
    return (
        isinstance(node, ast.BinOp)
        and isinstance(node.op, (ast.Add, ast.Sub))
        and _global_pointer(node.left)
    )


_SYNCHRONOUS_CALLS = frozenset(
    {
        "range",
        "float",
        "int",
        "bool",
        "abs",
        "min",
        "max",
        "_cute_python_mod",
        "_cute_numeric_extremum",
        "cute.arch.thread_idx",
        "cute.arch.block_idx",
        "cute.arch.sync_threads",
        "cute.arch.shuffle_sync",
        "cute.arch.fmin",
        "cute.arch.fmax",
    }
    | SCALAR_CAST_CALLS
    | SCALAR_CALLS
    | {"operator.truediv", "operator.not_"}
    | {
        f"cute.math.{operation}"
        for operation in (
            "min",
            "max",
            "exp",
            "exp2",
            "log",
            "log2",
            "div",
            "sqrt",
            "rsqrt",
            "tanh",
            "abs",
            "fma",
        )
    }
)


class _RenameNames(ast.NodeTransformer):
    def __init__(self, renames: Mapping[str, str]) -> None:
        self.renames = renames

    def visit_Name(self, node: ast.Name) -> ast.AST:
        return ast.copy_location(
            ast.Name(id=self.renames.get(node.id, node.id), ctx=node.ctx), node
        )


def _synchronous_call(node: ast.Call) -> bool:
    """Only known scalar intrinsics and direct synchronous pointer accesses."""
    name = ast.unparse(node.func)
    if name == "_cute_inline_asm_elementwise":
        return any(
            keyword.arg == "is_pure"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is True
            for keyword in node.keywords
        )
    if name in _SYNCHRONOUS_CALLS:
        return True
    return isinstance(node.func, ast.Attribute) and (
        node.func.attr in ("load", "store")
        and _global_pointer(node.func.value)
        or node.func.attr == "bitcast"
        and isinstance(node.func.value, ast.Call)
        and _synchronous_call(node.func.value)
    )


def _check_direct_accesses(body: Sequence[ast.stmt], names: frozenset[str]) -> None:
    for statement in body:
        for parent in ast.walk(statement):
            for child in ast.iter_child_nodes(parent):
                if isinstance(child, ast.Name) and child.id in names:
                    if not is_flat_scalar_access(parent, child):
                        # No pointers, aliases, helper arguments or rebinding.
                        raise _Refused


def _check_synchronous_calls(body: Sequence[ast.stmt]) -> None:
    for statement in body:
        for node in ast.walk(statement):
            if isinstance(node, ast.Call) and not _synchronous_call(node):
                raise _Refused


def _access_region(
    body: Sequence[ast.stmt], names: frozenset[str]
) -> Sequence[ast.stmt]:
    """The top-level span containing every use, including its enclosing control.

    Whole-kernel escape checks precede this projection. Anything that refers
    to an arena view outside the original root therefore either extends this
    span (and changes its skeleton) or refuses as a pointer/view escape.
    """
    positions = [i for i, statement in enumerate(body) if _references(statement, names)]
    return body[positions[0] : positions[-1] + 1] if positions else []


class _Epochs:
    def __init__(self, names: frozenset[str], edges: set[frozenset[str]]) -> None:
        self.names = names
        self.edges = edges

    def touch(self, node: ast.AST, pending: set[str]) -> set[str]:
        active = pending | _references(node, self.names)
        self.edges.update(frozenset(pair) for pair in combinations(sorted(active), 2))
        return active

    def walk(self, body: Sequence[ast.stmt], pending: set[str]) -> set[str]:
        for statement in body:
            if _is_barrier(statement):
                pending = set()
            elif isinstance(statement, ast.If):
                entry = self.touch(statement.test, pending)
                # The guard is not proved CTA-uniform. Distinct threads can
                # execute different arms, including their shared accesses.
                left = _references(
                    ast.Module(body=statement.body, type_ignores=[]), self.names
                )
                right = _references(
                    ast.Module(body=statement.orelse, type_ignores=[]), self.names
                )
                self.edges.update(
                    frozenset((a, b)) for a in left for b in right if a != b
                )
                pending = self.walk(statement.body, entry) | self.walk(
                    statement.orelse, entry
                )
            elif isinstance(statement, (ast.For, ast.While)):
                control = (
                    statement.iter if isinstance(statement, ast.For) else statement.test
                )
                entry = self.touch(control, pending)
                # The domain is a finite powerset; include zero iterations,
                # backedges and condition reads before each subsequent body.
                for _ in range(len(self.names) + 1):
                    tail = self.walk(statement.body, entry)
                    expanded = self.touch(control, entry | tail)
                    if expanded == entry:
                        break
                    entry = expanded
                else:
                    raise _Refused
                pending = self.walk(statement.orelse, entry | tail)
            elif isinstance(
                statement,
                (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Expr, ast.Pass),
            ):
                pending = self.touch(statement, pending)
            else:
                # In particular, no break/continue/return/try paths.
                raise _Refused
        return pending


def shared_access_skeleton(
    body: Sequence[ast.stmt], names: frozenset[str]
) -> tuple[object, ...]:
    """Keep shared expressions, controls and barriers; omit pure scalar work."""
    result: list[object] = []
    for statement in body:
        if _is_barrier(statement):
            result.append(("barrier",))
        elif isinstance(statement, (ast.If, ast.For, ast.While)):
            control = (
                statement.iter if isinstance(statement, ast.For) else statement.test
            )
            inner = shared_access_skeleton(statement.body, names)
            otherwise = shared_access_skeleton(statement.orelse, names)
            if inner or otherwise or _references(control, names):
                result.append(
                    (
                        type(statement).__name__,
                        ast.dump(control),
                        ast.dump(statement.target)
                        if isinstance(statement, ast.For)
                        else None,
                        inner,
                        otherwise,
                    )
                )
        elif _references(statement, names):
            result.append(ast.dump(statement))
    return tuple(result)


def plan_shared_arena(
    buffers: Sequence[tuple[str, torch.dtype, int]],
    live_edges: set[frozenset[str]],
    body: Sequence[ast.stmt],
) -> SharedArenaPlan | None:
    """Pack only noninterfering complete slots; never infer async retirement."""
    names = frozenset(name for name, _dtype, _count in buffers)
    assert len(names) == len(buffers), "duplicate shared buffer names"
    if not names:
        return None
    sizes = {name: aligned_shared_bytes(count, dtype) for name, dtype, count in buffers}
    if any(size <= 0 for size in sizes.values()):
        return None
    assert all(len(edge) == 2 and edge <= names for edge in live_edges)
    edges = set(live_edges)
    try:
        _check_direct_accesses(body, names)
        _check_synchronous_calls(body)
        _Epochs(names, edges).walk(body, set())
    except _Refused:
        return None
    neighbors = {name: set() for name in names}
    for edge in edges:
        left, right = sorted(edge)
        neighbors[left].add(right)
        neighbors[right].add(left)
    offsets: dict[str, int] = {}
    # Place the most constrained equal-size slots first. Name-only ordering
    # can waste a whole slot even for a path-shaped interference graph.
    for name in sorted(
        names, key=lambda name: (-sizes[name], -len(neighbors[name]), name)
    ):
        occupied = sorted(
            (offsets[other], offsets[other] + sizes[other])
            for other in neighbors[name]
            if other in offsets
        )
        begin = 0
        for low, high in occupied:
            if begin + sizes[name] <= low:
                break
            begin = max(begin, high)
        offsets[name] = begin
    byte_size = max(offsets[name] + sizes[name] for name in names)
    if byte_size >= sum(sizes.values()):
        return None
    views = tuple(
        ArenaView(name, dtype, count, offsets[name], sizes[name])
        for name, dtype, count in buffers
    )
    assert all(
        view.offset % 16 == 0
        and view.count * view.dtype.itemsize <= view.byte_size
        and view.offset + view.byte_size <= byte_size
        for view in views
    )
    assert all(
        offsets[left] + sizes[left] <= offsets[right]
        or offsets[right] + sizes[right] <= offsets[left]
        for edge in edges
        for left, right in [sorted(edge)]
    )
    return SharedArenaPlan(
        views,
        byte_size,
        ast.unparse(ast.Module(body=list(body), type_ignores=[])),
    )


def shared_schedule_is_unchanged(
    plan: SharedArenaPlan,
    body: Sequence[ast.stmt],
    renames: Mapping[str, str],
) -> bool:
    """Recheck after late passes; no original liveness fact survives a rewrite."""

    original_names = frozenset(view.name for view in plan.views)
    final_names = frozenset(renames.get(name, name) for name in original_names)
    if len(final_names) != len(original_names):
        return False
    expected = _RenameNames(renames).visit(ast.parse(plan.body_source))
    assert isinstance(expected, ast.Module)
    try:
        # Escape, call and control-transfer checks remain whole-kernel. An
        # opaque call can access shared memory without naming these views.
        # Only the shared-access skeleton excludes unrelated root barriers.
        _check_direct_accesses(body, final_names)
        _check_synchronous_calls(body)
        _Epochs(final_names, set()).walk(body, set())
        region = _access_region(body, final_names)
    except _Refused:
        return False
    return shared_access_skeleton(region, final_names) == shared_access_skeleton(
        _access_region(expected.body, final_names), final_names
    )


@dataclasses.dataclass(frozen=True)
class SharedArenaRequest:
    plan: SharedArenaPlan
    arena_name: str
    allocation_source: str

    def validate(self, body: Sequence[ast.stmt], renames: Mapping[str, str]) -> None:
        expected = _RenameNames(renames).visit(ast.parse(self.allocation_source))
        assert isinstance(expected, ast.Module)
        declarations: dict[str, str] = {}
        for statement in expected.body:
            assert isinstance(statement, ast.Assign) and len(statement.targets) == 1
            target = statement.targets[0]
            assert isinstance(target, ast.Name)
            declarations[target.id] = ast.dump(statement)
        seen: set[str] = set()
        declaration_order = list(declarations)
        remaining: list[ast.stmt] = []
        for statement in body:
            if (
                isinstance(statement, ast.Assign)
                and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
                and statement.targets[0].id in declarations
            ):
                name = statement.targets[0].id
                if (
                    name in seen
                    or name != declaration_order[len(seen)]
                    or ast.dump(statement) != declarations[name]
                ):
                    raise exc.InvalidConfig(
                        "shared arena allocation changed after lowering"
                    )
                seen.add(name)
            else:
                if _references(statement, frozenset(declarations) - seen):
                    raise exc.InvalidConfig(
                        "shared arena allocation moved after its use"
                    )
                remaining.append(statement)
        arena = renames.get(self.arena_name, self.arena_name)
        if seen != set(declarations) or any(
            isinstance(node, ast.Name) and node.id == arena
            for statement in remaining
            for node in ast.walk(statement)
        ):
            raise exc.InvalidConfig("shared arena allocation escaped its typed views")
        if not shared_schedule_is_unchanged(self.plan, remaining, renames):
            raise exc.InvalidConfig(
                "shared arena access schedule changed after lowering"
            )
