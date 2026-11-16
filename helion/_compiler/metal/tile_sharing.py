"""Share threadgroup storage between MPP tiles that are never live together.

Every threadgroup tile is declared at kernel scope, so their bytes add up even
when their values never coexist: attention's score materialization
(``_mpp_mat``, read back into registers by the softmax) and its P operand tile
(``_mpp_tile``, staged after the softmax and read by the second op) are both
written and read within one iteration of the K loop, one after the other.
:func:`share_tiles` gives such tiles one storage, with a barrier between
consecutive users and, inside a loop, after the last one, so no thread writes
a tile's bytes while another can still read the previous tile from them.
Fewer threadgroup bytes let more threadgroups share a GPU core, and let larger
blocks fit the threadgroup memory at all.

A tile may share when its every access is one of the MPP markers' (a staging
write, a materialization, a subscript read, an op's operand, an accumulator's
seed), the first in program order is a write, none sits under an ``if``, a
``while`` or a loop's ``else``, and the statement list holding them all (its
home) runs on every thread of the threadgroup alike.  Each writer covers the
whole block the readers read (``_helion_mpp_stage`` writes every in-block element,
masked ones as zero, and ``_helion_mpp_materialize`` every valid one), so a tile
never reads a value from before its first write in the same run of its home,
and the storage's other users can't be observed.
"""

from __future__ import annotations

import ast
import dataclasses

from .mpp_matmul import ACC
from .mpp_matmul import ACC_SEED
from .mpp_matmul import BARRIER
from .mpp_matmul import MAT
from .mpp_matmul import RUN
from .mpp_matmul import STAGE

_WRITE, _READ, _OTHER = "write", "read", "other"

_Path = list[tuple[list[ast.stmt], int]]


@dataclasses.dataclass
class _Live:
    """Where a tile is live: statements ``first..last`` of the list ``home``."""

    home: list[ast.stmt]
    first: int
    last: int


def share_tiles(
    body: list[ast.stmt],
    buffers: list[tuple[str, str, int | str]],
) -> tuple[list[ast.stmt], list[tuple[str, str, int | str]]]:
    """Merge every pair of tiles that can share storage, largest saving
    first.  ``body`` is edited in place, merged tiles renamed to their
    storage and barriers inserted; return it and the remaining ``(name,
    dtype, slots)`` declarations.  Two tiles can share when both may share,
    have one home and one element type, and their ranges in the home don't
    overlap."""
    decl = {n: (d, s) for n, d, s in buffers if isinstance(s, int)}
    live = _live_ranges(body, set(decl))
    groups = [[n] for n in sorted(live)]

    # Slots measure the saving: only tiles of one element type merge.
    def size(group: list[str]) -> int:
        return max(int(decl[n][1]) for n in group)

    def disjoint(a: list[str], b: list[str]) -> bool:
        return (
            live[a[0]].home is live[b[0]].home
            and decl[a[0]][0] == decl[b[0]][0]
            and all(
                live[m].last < live[n].first or live[n].last < live[m].first
                for m in a
                for n in b
            )
        )

    while merges := [
        (min(size(a), size(b)), -i, -j)
        for i, a in enumerate(groups)
        for j, b in enumerate(groups)
        if i < j and disjoint(a, b)
    ]:
        _, neg_i, neg_j = max(merges)
        merged = groups[-neg_i] + groups.pop(-neg_j)
        groups[-neg_i] = sorted(merged, key=lambda n: live[n].first)

    rename: dict[str, str] = {}
    slots: dict[str, int] = {}
    fences: dict[int, tuple[list[ast.stmt], set[int]]] = {}
    for group in groups:
        if len(group) == 1:
            continue
        storage, home = group[0], live[group[0]].home
        slots[storage] = size(group)
        _, at = fences.setdefault(id(home), (home, set()))
        for name in group[1:]:
            rename[name] = storage
            at.add(live[name].first)
        if _in_loop(body, home):
            at.add(live[group[-1]].last + 1)
    for home, at in fences.values():
        for index in sorted(at, reverse=True):
            home.insert(
                index, ast.Expr(ast.Call(ast.Name(BARRIER, ast.Load()), [], []))
            )
    for stmt in body:
        for node in ast.walk(stmt):
            if isinstance(node, ast.Name) and node.id in rename:
                node.id = rename[node.id]
            elif isinstance(node, ast.Constant) and node.value in rename:
                node.value = rename[str(node.value)]
    return body, [(n, d, slots.get(n, s)) for n, d, s in buffers if n not in rename]


def _live_ranges(body: list[ast.stmt], names: set[str]) -> dict[str, _Live]:
    """The live range of every tile in ``names`` that may share storage."""
    events: dict[str, list[tuple[str, _Path]]] = {}

    def visit(stmts: list[ast.stmt], path: _Path) -> None:
        for index, stmt in enumerate(stmts):
            here = [*path, (stmts, index)]
            if isinstance(stmt, (ast.For, ast.While, ast.If)):
                own: ast.AST = stmt.iter if isinstance(stmt, ast.For) else stmt.test
            else:
                own = stmt
            kinds: dict[str, set[str]] = {}
            for name, kind in _accesses(own, names):
                kinds.setdefault(name, set()).add(kind)
            for name, found in kinds.items():
                # A statement's reads are evaluated before its write.
                for kind in (_OTHER, _READ, _WRITE):
                    if kind in found:
                        events.setdefault(name, []).append((kind, here))
            if isinstance(stmt, (ast.For, ast.While, ast.If)):
                visit(stmt.body, here)
                visit(stmt.orelse, here)

    visit(body, [])
    out: dict[str, _Live] = {}
    for name, evs in events.items():
        if evs[0][0] != _WRITE or any(kind == _OTHER for kind, _ in evs):
            continue
        paths = [p for _, p in evs]
        depth = 0
        while all(
            len(p) > depth + 1 and p[depth + 1][0] is paths[0][depth + 1][0]
            for p in paths
        ):
            depth += 1
        home = paths[0][depth][0]
        if any(_guarded_below(p, depth) for p in paths):
            continue
        if home is body or _fenced(home):
            indices = [p[depth][1] for p in paths]
            out[name] = _Live(home, min(indices), max(indices))
    return out


def _guarded_below(path: _Path, depth: int) -> bool:
    """Whether an access at ``path`` sits under an ``if`` or ``while`` (or a
    loop's ``else``) nested inside the list at ``depth``.

    A ``for`` body is entered although it may run no iteration: a tile read
    after a loop that wrote none of it would read unwritten memory without
    sharing as well, and the lane loops that do write tiles cover the block
    between the threads."""
    for (stmts, index), (child, _) in zip(
        path[depth:], path[depth + 1 :], strict=False
    ):
        stmt = stmts[index]
        if not isinstance(stmt, ast.For) or child is not stmt.body:
            return True
    return False


def _accesses(expr: ast.AST, names: set[str]) -> list[tuple[str, str]]:
    """Each access to a tile in ``names`` within ``expr``, with its kind.

    Every subscript counts as a read, an assignment's target too: the
    markers write tiles, and a tile whose first access is not a marker's
    write never shares."""
    out: list[tuple[str, str]] = []
    for node in ast.walk(expr):
        for child in ast.iter_child_nodes(node):
            name = _tile_name(child)
            if name is None or name not in names:
                continue
            kind = _OTHER
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                pos = next((i for i, a in enumerate(node.args) if a is child), -1)
                callee = node.func.id
                if isinstance(child, ast.Constant):
                    # ``_helion_mpp_acc`` names its staged seed tile as a string.
                    kind = _READ if callee == ACC and pos == ACC_SEED else _OTHER
                elif (callee == STAGE and pos == 0) or (callee == MAT and pos == 1):
                    kind = _WRITE
                elif callee == RUN:
                    kind = _READ
            elif isinstance(node, ast.Subscript) and node.value is child:
                kind = _READ
            out.append((name, kind))
    return out


def _tile_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _fenced(stmts: list[ast.stmt]) -> bool:
    """Whether the walker already puts a threadgroup barrier in ``stmts``
    itself, around a materialization or an op with a staged operand, so a
    barrier there is reached by every thread alike."""
    for stmt in stmts:
        value = stmt.value if isinstance(stmt, (ast.Expr, ast.Assign)) else None
        if not (isinstance(value, ast.Call) and isinstance(value.func, ast.Name)):
            continue
        if value.func.id == MAT:
            return True
        if value.func.id == RUN:
            spaces = next(k.value for k in value.keywords if k.arg == "spaces")
            if any(ast.literal_eval(spaces)):
                return True
    return False


def _in_loop(body: list[ast.stmt], home: list[ast.stmt]) -> bool:
    """Whether ``home`` runs inside a loop of ``body``."""

    def search(stmts: list[ast.stmt], looped: bool) -> bool | None:
        if stmts is home:
            return looped
        for stmt in stmts:
            if isinstance(stmt, (ast.For, ast.While, ast.If)):
                inner = looped or not isinstance(stmt, ast.If)
                for child in (stmt.body, stmt.orelse):
                    if (found := search(child, inner)) is not None:
                        return found
        return None

    found = search(body, False)
    assert found is not None
    return found
